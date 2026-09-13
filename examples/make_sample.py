from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    out = Path(__file__).with_name("sample_drawing.png")
    image = Image.new("L", (260, 180), 255)
    draw = ImageDraw.Draw(image)

    bolt = [(118, 18), (62, 100), (105, 100), (82, 162), (178, 72), (128, 72)]
    draw.polygon(bolt, fill=0)
    draw.line([(32, 154), (220, 154)], fill=0, width=8)
    draw.ellipse((188, 30, 232, 74), outline=0, width=8)

    image.save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
