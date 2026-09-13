"""Render the actual sample STL into the README illustration with Pillow."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from drawing_to_3d_agent import AgentConfig, DrawingTo3DAgent


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = root / "docs"
    result = DrawingTo3DAgent(AgentConfig(
        low_poly=True, fill_holes=True, max_size=160,
    )).generate(
        root / "examples/sample_top_view.png", docs / "demo.stl",
        side_image=root / "examples/sample_side_view.png",
    )
    points = [
        [float(value) for value in line.split()[1:]]
        for line in result.output_path.read_text(encoding="ascii").splitlines()
        if line.strip().startswith("vertex ")
    ]
    mesh = np.asarray(points).reshape(-1, 3, 3)
    canvas = Image.new("RGB", (1440, 760), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)

    def font(size):
        for name in ("DejaVuSans.ttf", "Arial.ttf", "C:/Windows/Fonts/arial.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
        return ImageFont.load_default()

    draw.text((54, 36), "DRAWING TO 3D", fill="#15212b", font=font(34))
    draw.text((54, 88), "Two silhouettes. One faceted mesh.", fill="#51616f", font=font(22))
    draw.line((510, 160, 510, 650), fill="#d2dae0", width=2)
    for filename, label, box in (
        ("sample_top_view.png", "01  TOP VIEW", (54, 206, 448, 396)),
        ("sample_side_view.png", "02  SIDE VIEW", (54, 474, 448, 654)),
    ):
        draw.text((54, box[1] - 36), label, fill="#435463", font=font(19))
        with Image.open(root / "examples" / filename) as source:
            tile = ImageOps.contain(source.convert("RGB"), (box[2]-box[0], box[3]-box[1]))
        canvas.paste(tile, (box[0] + (box[2]-box[0]-tile.width)//2, box[1]))

    # Orthographic projection of the exported triangles, with back-face culling.
    camera = np.array([-1.5, -2.0, 1.5])
    camera /= np.linalg.norm(camera)
    right = np.cross([0.0, 0.0, 1.0], camera)
    right /= np.linalg.norm(right)
    up = np.cross(camera, right)
    center = (mesh.reshape(-1, 3).min(axis=0) + mesh.reshape(-1, 3).max(axis=0)) / 2
    centered = mesh - center
    projected = np.stack((centered @ right, -(centered @ up)), axis=-1)
    bounds = projected.reshape(-1, 2)
    span = np.ptp(bounds, axis=0)
    scale = min(760 / span[0], 410 / span[1])
    screen = (projected - (bounds.min(axis=0) + bounds.max(axis=0))/2) * scale + [965, 405]
    normals = np.cross(mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    depths = centered.mean(axis=1) @ camera
    light = np.array([-0.6, -0.8, 1.8])
    light /= np.linalg.norm(light)
    for index in np.argsort(depths):
        if normals[index] @ camera <= 0:
            continue
        brightness = 0.55 + 0.45 * max(0, float(normals[index] @ light))
        color = tuple(int(channel * brightness) for channel in (58, 178, 157))
        draw.polygon([tuple(point) for point in screen[index]], fill=color, outline="#276b61", width=1)
    draw.text((572, 169), "03  GENERATED STL", fill="#435463", font=font(19))
    draw.text((572, 643), f"{result.triangle_count} triangles  |  Low-poly + fill holes", fill="#435463", font=font(20))
    draw.text((54, 706), "Python / NumPy / Pillow     -     Rendered from the exported mesh", fill="#667682", font=font(18))
    canvas.save(docs / "demo.png")
    print(f"Generated demo.stl and demo.png ({result.triangle_count} triangles)")


if __name__ == "__main__":
    main()
