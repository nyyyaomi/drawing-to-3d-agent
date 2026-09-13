from __future__ import annotations

import json
import os
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps


@dataclass(frozen=True)
class EditOperation:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EditPlan:
    image_ops: list[EditOperation] = field(default_factory=list)
    mask_ops: list[EditOperation] = field(default_factory=list)
    add_base: bool = False
    base_height: float | None = None
    notes: list[str] = field(default_factory=list)

    def add_image_op(self, name: str, **params: Any) -> None:
        self.image_ops.append(EditOperation(name, params))

    def add_mask_op(self, name: str, **params: Any) -> None:
        self.mask_ops.append(EditOperation(name, params))

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_ops": [op.__dict__ for op in self.image_ops],
            "mask_ops": [op.__dict__ for op in self.mask_ops],
            "add_base": self.add_base,
            "base_height": self.base_height,
            "notes": self.notes,
        }


class RuleBasedPlanner:
    """Turns plain-language editing requests into deterministic image operations."""

    def plan(self, prompt: str) -> EditPlan:
        text = prompt.lower().strip()
        plan = EditPlan()
        if not text:
            plan.notes.append("No prompt supplied; using the drawing as-is.")
            return plan

        if any(word in text for word in ("mirror", "flip horizontal", "reflect")):
            plan.add_image_op("mirror")
        if "flip vertical" in text or "upside down" in text:
            plan.add_image_op("flip_vertical")
        if "invert" in text or "negative" in text:
            plan.add_image_op("invert")

        rotate_match = re.search(r"rotate\s*(-?\d+)", text)
        if rotate_match:
            plan.add_image_op("rotate", degrees=float(rotate_match.group(1)))
        elif "rotate left" in text:
            plan.add_image_op("rotate", degrees=90.0)
        elif "rotate right" in text:
            plan.add_image_op("rotate", degrees=-90.0)

        if any(word in text for word in ("thicker", "thicken", "bold", "fatter")):
            plan.add_mask_op("dilate", radius=_word_radius(text, default=2))
        if any(word in text for word in ("thinner", "thin", "shrink lines")):
            plan.add_mask_op("erode", radius=_word_radius(text, default=1))
        if any(word in text for word in ("smooth", "round", "soften")):
            plan.add_mask_op("smooth", radius=1.25)
        if any(phrase in text for phrase in ("fill holes", "fill hole", "solid", "filled")):
            plan.add_mask_op("fill_holes")
        if any(phrase in text for phrase in ("clean", "remove specks", "remove noise", "despeckle")):
            plan.add_mask_op("remove_small_components", min_area=8)
        if any(phrase in text for phrase in ("add a base", "add base", "stand", "plaque")):
            plan.add_base = True
            plan.base_height = 1.0

        if not plan.image_ops and not plan.mask_ops and not plan.add_base:
            plan.notes.append(
                "The prompt did not match a built-in edit; exporting the drawing as-is."
            )
        return plan


class OpenAIPlanner:
    """Optional LLM planner that maps free-form instructions into safe local edits."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def plan(self, prompt: str) -> EditPlan:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI planner requires `pip install openai`."
            ) from exc

        client = OpenAI()
        allowed_ops = {
            "image_ops": ["mirror", "flip_vertical", "invert", "rotate"],
            "mask_ops": [
                "dilate",
                "erode",
                "smooth",
                "fill_holes",
                "remove_small_components",
            ],
        }
        instructions = (
            "Return only JSON. Convert the user's drawing edit request into these "
            f"allowed operations: {json.dumps(allowed_ops)}. "
            "Use image_ops for geometry/intensity edits and mask_ops for binary "
            "shape edits. Supported params: rotate.degrees, dilate.radius, "
            "erode.radius, smooth.radius, remove_small_components.min_area, "
            "add_base boolean, base_height number, notes array."
        )
        response = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        return _plan_from_json(json.loads(content))


def load_grayscale(path: Path) -> Image.Image:
    image = Image.open(path)
    if "A" in image.getbands():
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        image = background
    return image.convert("L")


def apply_image_ops(image: Image.Image, ops: list[EditOperation]) -> Image.Image:
    current = image
    for op in ops:
        if op.name == "mirror":
            current = ImageOps.mirror(current)
        elif op.name == "flip_vertical":
            current = ImageOps.flip(current)
        elif op.name == "invert":
            current = ImageOps.invert(current)
        elif op.name == "rotate":
            degrees = float(op.params.get("degrees", 0.0))
            current = current.rotate(degrees, expand=True, fillcolor=255)
        else:
            raise ValueError(f"Unsupported image operation: {op.name}")
    return current


def image_to_mask(image: Image.Image, threshold: int = 180, ink: str = "auto") -> np.ndarray:
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if ink == "auto":
        border = _border_pixels(gray)
        use_dark_ink = float(border.mean()) >= 128.0
    elif ink == "dark":
        use_dark_ink = True
    elif ink == "light":
        use_dark_ink = False
    else:
        raise ValueError("ink must be one of: auto, dark, light")

    if use_dark_ink:
        return gray < threshold
    return gray > threshold


def apply_mask_ops(mask: np.ndarray, ops: list[EditOperation]) -> np.ndarray:
    current = mask.astype(bool)
    for op in ops:
        if op.name == "dilate":
            current = dilate(current, int(op.params.get("radius", 2)))
        elif op.name == "erode":
            current = erode(current, int(op.params.get("radius", 1)))
        elif op.name == "smooth":
            current = smooth_mask(current, float(op.params.get("radius", 1.25)))
        elif op.name == "fill_holes":
            current = fill_holes(current)
        elif op.name == "remove_small_components":
            current = remove_small_components(
                current, int(op.params.get("min_area", 8))
            )
        else:
            raise ValueError(f"Unsupported mask operation: {op.name}")
    return current


def crop_mask(mask: np.ndarray, padding: int = 2) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No ink was detected in the drawing.")

    height, width = mask.shape
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding + 1, width)
    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + padding + 1, height)
    return mask[y0:y1, x0:x1]


def resize_mask(mask: np.ndarray, max_size: int) -> np.ndarray:
    if max_size <= 0:
        raise ValueError("max_size must be greater than zero")
    height, width = mask.shape
    largest = max(height, width)
    if largest <= max_size:
        return mask

    scale = max_size / float(largest)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    resized = image.resize(new_size, resample=Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8) > 127


def resize_mask_exact(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be greater than zero")
    image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    resized = image.resize((width, height), resample=Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8) > 127


def resize_mask_to_axis(
    mask: np.ndarray,
    *,
    target_axis_length: int,
    max_cross_axis: int,
) -> np.ndarray:
    """Resize a side-view mask so its horizontal axis lines up with the top view."""
    if target_axis_length <= 0:
        raise ValueError("target_axis_length must be greater than zero")
    if max_cross_axis <= 0:
        raise ValueError("max_cross_axis must be greater than zero")

    height, width = mask.shape
    scaled_height = max(1, round(height * (target_axis_length / float(width))))
    scaled_height = min(scaled_height, max_cross_axis)
    return resize_mask_exact(mask, target_axis_length, scaled_height)


def save_mask_preview(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), mode="L")
    preview.save(path)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, radius)
    if radius == 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            if (dy - radius) ** 2 + (dx - radius) ** 2 <= radius**2:
                out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, radius)
    if radius == 0:
        return mask.copy()
    return ~dilate(~mask, radius)


def smooth_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=max(radius, 0.1)))
    return np.asarray(blurred, dtype=np.uint8) > 127


def fill_holes(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    background = ~mask
    seen = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def push(y: int, x: int) -> None:
        if 0 <= y < height and 0 <= x < width and background[y, x] and not seen[y, x]:
            seen[y, x] = True
            queue.append((y, x))

    for x in range(width):
        push(0, x)
        push(height - 1, x)
    for y in range(height):
        push(y, 0)
        push(y, width - 1)

    while queue:
        y, x = queue.popleft()
        push(y - 1, x)
        push(y + 1, x)
        push(y, x - 1)
        push(y, x + 1)

    holes = background & ~seen
    return mask | holes


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    min_area = max(1, min_area)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    keep = np.zeros_like(mask, dtype=bool)

    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or seen[start_y, start_x]:
                continue
            component: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
            seen[start_y, start_x] = True
            while queue:
                y, x = queue.popleft()
                component.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            if len(component) >= min_area:
                for y, x in component:
                    keep[y, x] = True
    return keep


def _border_pixels(gray: np.ndarray) -> np.ndarray:
    if gray.shape[0] == 1 or gray.shape[1] == 1:
        return gray.ravel()
    return np.concatenate(
        [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]]
    )


def _word_radius(text: str, default: int) -> int:
    if any(word in text for word in ("slightly", "little", "small")):
        return 1
    if any(word in text for word in ("very", "much", "big", "heavy", "lots")):
        return max(default + 2, 3)
    return default


def _plan_from_json(data: dict[str, Any]) -> EditPlan:
    plan = EditPlan()
    for item in data.get("image_ops", []):
        if isinstance(item, str):
            plan.add_image_op(item)
        else:
            plan.add_image_op(str(item.get("name")), **dict(item.get("params", {})))
    for item in data.get("mask_ops", []):
        if isinstance(item, str):
            plan.add_mask_op(item)
        else:
            plan.add_mask_op(str(item.get("name")), **dict(item.get("params", {})))
    plan.add_base = bool(data.get("add_base", False))
    if data.get("base_height") is not None:
        plan.base_height = float(data["base_height"])
    notes = data.get("notes", [])
    if isinstance(notes, list):
        plan.notes = [str(note) for note in notes]
    return plan
