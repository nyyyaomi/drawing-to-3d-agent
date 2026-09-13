from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .image_editing import (
    EditOperation,
    OpenAIPlanner,
    RuleBasedPlanner,
    apply_image_ops,
    apply_mask_ops,
    crop_mask,
    image_to_mask,
    load_grayscale,
    resize_mask,
    resize_mask_to_axis,
    save_mask_preview,
)
from .mesh import (
    low_poly_hull_triangles,
    mask_to_triangles,
    smooth_triangles,
    visual_hull_voxels,
    voxels_to_triangles,
    write_model,
)


PlannerName = Literal["rules", "openai"]
InkMode = Literal["auto", "dark", "light"]
SideAxis = Literal["x", "y"]


@dataclass
class AgentConfig:
    threshold: int = 180
    ink: InkMode = "auto"
    max_size: int = 160
    height: float = 4.0
    pixel_size: float = 0.4
    base_height: float = 0.0
    base_margin: int = 4
    fill_holes: bool = False
    side_axis: SideAxis = "x"
    top_mirror: bool = False
    top_flip_vertical: bool = False
    side_mirror: bool = False
    side_flip_vertical: bool = False
    low_poly: bool = False
    low_poly_angle: float = 10.0
    low_poly_tolerance: float = 3.0
    smooth_iterations: int = 0
    smooth_strength: float = 0.35
    planner: PlannerName = "rules"
    openai_model: str | None = None
    crop_padding: int = 2


@dataclass
class AgentResult:
    output_path: Path
    preview_path: Path | None
    triangle_count: int
    filled_pixels: int
    mask_width: int
    mask_height: int
    base_height: float
    plan: dict
    generation_mode: str = "single"
    side_preview_path: Path | None = None
    side_mask_width: int | None = None
    side_mask_height: int | None = None
    voxel_count: int | None = None


class DrawingTo3DAgent:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()

    def generate(
        self,
        input_image: Path | str,
        output_model: Path | str,
        *,
        side_image: Path | str | None = None,
        prompt: str = "",
        preview_path: Path | str | None = None,
        side_preview_path: Path | str | None = None,
        solid_name: str = "drawing_model",
    ) -> AgentResult:
        input_path = Path(input_image)
        output_path = Path(output_model)
        preview = Path(preview_path) if preview_path else None
        side_preview = Path(side_preview_path) if side_preview_path else None

        if output_path.suffix.lower() not in {".stl", ".obj"}:
            raise ValueError("Use an output path ending in .stl or .obj.")

        planner = self._planner()
        plan = planner.plan(prompt)

        if self.config.fill_holes and not _has_mask_op(plan.mask_ops, "fill_holes"):
            plan.add_mask_op("fill_holes")

        top_image_ops = [
            *plan.image_ops,
            *self._view_image_ops(
                mirror=self.config.top_mirror,
                flip_vertical=self.config.top_flip_vertical,
            ),
        ]
        mask = self._trace_image(input_path, top_image_ops, plan.mask_ops)
        mask = resize_mask(mask, self.config.max_size)

        if preview is not None:
            save_mask_preview(mask, preview)

        base_height = self.config.base_height
        if plan.add_base:
            base_height = max(base_height, plan.base_height or 1.0)

        if side_image is not None:
            side_image_ops = [
                *plan.image_ops,
                *self._view_image_ops(
                    mirror=self.config.side_mirror,
                    flip_vertical=self.config.side_flip_vertical,
                ),
            ]
            side_mask = self._trace_image(Path(side_image), side_image_ops, plan.mask_ops)
            axis_length = mask.shape[1] if self.config.side_axis == "x" else mask.shape[0]
            side_mask = resize_mask_to_axis(
                side_mask,
                target_axis_length=axis_length,
                max_cross_axis=self.config.max_size,
            )

            if side_preview is not None:
                save_mask_preview(side_mask, side_preview)

            voxel_count: int | None = None
            if self.config.low_poly:
                triangles = low_poly_hull_triangles(
                    mask,
                    side_mask,
                    side_axis=self.config.side_axis,
                    pixel_size=self.config.pixel_size,
                    base_height=base_height,
                    base_margin=self.config.base_margin,
                    simplify_angle_degrees=self.config.low_poly_angle,
                    simplify_tolerance_pixels=self.config.low_poly_tolerance,
                )
                generation_mode = "two_view_low_poly"
            else:
                voxels = visual_hull_voxels(
                    mask,
                    side_mask,
                    side_axis=self.config.side_axis,
                )
                if not voxels.any():
                    raise ValueError("The top and side silhouettes did not overlap into a model.")

                triangles = voxels_to_triangles(
                    voxels,
                    pixel_size=self.config.pixel_size,
                    base_height=base_height,
                    base_margin=self.config.base_margin,
                )
                voxel_count = int(voxels.sum())
                generation_mode = "two_view"

            triangles = self._smooth_mesh(triangles)
            triangle_count = write_model(output_path, triangles, model_name=solid_name)

            return AgentResult(
                output_path=output_path,
                preview_path=preview,
                side_preview_path=side_preview,
                triangle_count=triangle_count,
                filled_pixels=int(mask.sum()),
                mask_width=int(mask.shape[1]),
                mask_height=int(mask.shape[0]),
                side_mask_width=int(side_mask.shape[1]),
                side_mask_height=int(side_mask.shape[0]),
                voxel_count=voxel_count,
                base_height=base_height,
                plan=plan.to_dict(),
                generation_mode=generation_mode,
            )

        triangles = mask_to_triangles(
            mask,
            pixel_size=self.config.pixel_size,
            height=self.config.height,
            base_height=base_height,
            base_margin=self.config.base_margin,
        )
        triangles = self._smooth_mesh(triangles)
        triangle_count = write_model(output_path, triangles, model_name=solid_name)

        return AgentResult(
            output_path=output_path,
            preview_path=preview,
            triangle_count=triangle_count,
            filled_pixels=int(mask.sum()),
            mask_width=int(mask.shape[1]),
            mask_height=int(mask.shape[0]),
            base_height=base_height,
            plan=plan.to_dict(),
            generation_mode="single",
        )

    def _planner(self) -> RuleBasedPlanner | OpenAIPlanner:
        if self.config.planner == "rules":
            return RuleBasedPlanner()
        if self.config.planner == "openai":
            return OpenAIPlanner(model=self.config.openai_model)
        raise ValueError(f"Unknown planner: {self.config.planner}")

    def _trace_image(
        self,
        path: Path,
        image_ops: list[EditOperation],
        mask_ops: list[EditOperation],
    ):
        image = load_grayscale(path)
        image = apply_image_ops(image, image_ops)
        mask = image_to_mask(
            image,
            threshold=self.config.threshold,
            ink=self.config.ink,
        )
        mask = apply_mask_ops(mask, mask_ops)
        return crop_mask(mask, padding=self.config.crop_padding)

    def _smooth_mesh(self, triangles):
        return smooth_triangles(
            triangles,
            iterations=self.config.smooth_iterations,
            strength=self.config.smooth_strength,
        )

    def _view_image_ops(
        self,
        *,
        mirror: bool,
        flip_vertical: bool,
    ) -> list[EditOperation]:
        ops: list[EditOperation] = []
        if mirror:
            ops.append(EditOperation("mirror"))
        if flip_vertical:
            ops.append(EditOperation("flip_vertical"))
        return ops


def _has_mask_op(ops: list[EditOperation], name: str) -> bool:
    return any(op.name == name for op in ops)
