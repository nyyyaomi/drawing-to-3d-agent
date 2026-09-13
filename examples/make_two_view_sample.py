from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    folder = Path(__file__).parent
    top_path = folder / "sample_top_view.png"
    side_path = folder / "sample_side_view.png"

    top = Image.new("L", (240, 170), 255)
    top_draw = ImageDraw.Draw(top)
    top_draw.rounded_rectangle((38, 34, 202, 136), radius=36, fill=0)
    top_draw.ellipse((82, 58, 158, 112), fill=255)
    top_draw.rectangle((116, 34, 202, 136), fill=0)
    top.save(top_path)

    side = Image.new("L", (240, 130), 255)
    side_draw = ImageDraw.Draw(side)
    side_draw.polygon(
        [(38, 104), (38, 82), (80, 52), (124, 34), (202, 34), (202, 104)],
        fill=0,
    )
    side_draw.rectangle((38, 98, 202, 112), fill=0)
    side.save(side_path)

    print(f"Wrote {top_path}")
    print(f"Wrote {side_path}")


if __name__ == "__main__":
    main()
