from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import AgentConfig, DrawingTo3DAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="draw3d-agent",
        description="Edit drawing views and export an STL model.",
    )
    parser.add_argument(
        "input_image",
        help="Source drawing image, or the top view when --side-image is used.",
    )
    parser.add_argument("output_model", help="Output model path ending in .stl or .obj.")
    parser.add_argument(
        "--side-image",
        help="Optional side-view drawing. When supplied, the model is built from top and side silhouettes.",
    )
    parser.add_argument(
        "--side-axis",
        choices=["x", "y"],
        default="x",
        help="Which top-view axis the side image's horizontal axis lines up with.",
    )
    parser.add_argument(
        "--top-mirror",
        action="store_true",
        help="Mirror only the top/input view left-to-right before tracing.",
    )
    parser.add_argument(
        "--top-flip-vertical",
        action="store_true",
        help="Flip only the top/input view upside down before tracing.",
    )
    parser.add_argument(
        "--side-mirror",
        action="store_true",
        help="Mirror only the side view left-to-right before tracing.",
    )
    parser.add_argument(
        "--side-flip-vertical",
        action="store_true",
        help="Flip only the side view upside down before tracing.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Natural edit request, e.g. 'make it thicker, fill holes, add a base'.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=4.0,
        help="Extrusion height for one-image mode. In two-view mode, the side image defines height.",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.4,
        help="Model units represented by one processed image pixel.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=180,
        help="Ink threshold from 0 to 255.",
    )
    parser.add_argument(
        "--ink",
        choices=["auto", "dark", "light"],
        default="auto",
        help="Use dark ink, light ink, or auto-detect from the image border.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=160,
        help="Downsample largest processed side to this many pixels.",
    )
    parser.add_argument(
        "--low-poly",
        action="store_true",
        help="Simplify silhouette outlines into straight faceted segments for two-view models.",
    )
    parser.add_argument(
        "--low-poly-angle",
        type=float,
        default=10.0,
        help="Keep low-poly vertices when silhouette direction changes by this many degrees.",
    )
    parser.add_argument(
        "--low-poly-tolerance",
        type=float,
        default=3.0,
        help="Simplify low-poly outline lines within this many traced pixels.",
    )
    parser.add_argument(
        "--smooth-iterations",
        type=int,
        default=0,
        help="Number of Laplacian mesh smoothing passes to apply before writing STL.",
    )
    parser.add_argument(
        "--smooth-strength",
        type=float,
        default=0.35,
        help="Smoothing strength from 0 to 1 for each mesh smoothing pass.",
    )
    parser.add_argument(
        "--fill-holes",
        action="store_true",
        help="Fill enclosed holes before extrusion.",
    )
    parser.add_argument(
        "--base-height",
        type=float,
        default=0.0,
        help="Add a rectangular base with this height.",
    )
    parser.add_argument(
        "--base-margin",
        type=int,
        default=4,
        help="Base margin in processed image pixels.",
    )
    parser.add_argument(
        "--preview",
        help="Optional path for the edited top-view preview PNG.",
    )
    parser.add_argument(
        "--side-preview",
        help="Optional path for the edited side-view preview PNG.",
    )
    parser.add_argument(
        "--planner",
        choices=["rules", "openai"],
        default="rules",
        help="Use the local rule planner or an optional OpenAI JSON planner.",
    )
    parser.add_argument(
        "--openai-model",
        default=None,
        help="Model name for --planner openai. Defaults to OPENAI_MODEL or a package default.",
    )
    parser.add_argument(
        "--solid-name",
        default="drawing_model",
        help="Name written inside the STL file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a short summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = AgentConfig(
        threshold=args.threshold,
        ink=args.ink,
        max_size=args.max_size,
        height=args.height,
        pixel_size=args.pixel_size,
        base_height=args.base_height,
        base_margin=args.base_margin,
        fill_holes=args.fill_holes,
        side_axis=args.side_axis,
        top_mirror=args.top_mirror,
        top_flip_vertical=args.top_flip_vertical,
        side_mirror=args.side_mirror,
        side_flip_vertical=args.side_flip_vertical,
        low_poly=args.low_poly,
        low_poly_angle=args.low_poly_angle,
        low_poly_tolerance=args.low_poly_tolerance,
        smooth_iterations=args.smooth_iterations,
        smooth_strength=args.smooth_strength,
        planner=args.planner,
        openai_model=args.openai_model,
    )
    agent = DrawingTo3DAgent(config)
    result = agent.generate(
        Path(args.input_image),
        Path(args.output_model),
        side_image=Path(args.side_image) if args.side_image else None,
        prompt=args.prompt,
        preview_path=Path(args.preview) if args.preview else None,
        side_preview_path=Path(args.side_preview) if args.side_preview else None,
        solid_name=args.solid_name,
    )

    payload = {
        "output": str(result.output_path),
        "preview": str(result.preview_path) if result.preview_path else None,
        "side_preview": str(result.side_preview_path) if result.side_preview_path else None,
        "mode": result.generation_mode,
        "triangles": result.triangle_count,
        "filled_pixels": result.filled_pixels,
        "voxels": result.voxel_count,
        "mask_size": [result.mask_width, result.mask_height],
        "side_mask_size": (
            [result.side_mask_width, result.side_mask_height]
            if result.side_mask_width is not None and result.side_mask_height is not None
            else None
        ),
        "base_height": result.base_height,
        "plan": result.plan,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Wrote {payload['output']}")
        if payload["preview"]:
            print(f"Wrote preview {payload['preview']}")
        if payload["side_preview"]:
            print(f"Wrote side preview {payload['side_preview']}")
        print(
            f"Mesh: {payload['triangles']} triangles, "
            f"{payload['filled_pixels']} filled pixels, "
            f"mask {payload['mask_size'][0]}x{payload['mask_size'][1]}"
        )
        if payload["side_mask_size"] is not None:
            if result.voxel_count is None:
                print(
                    f"Two-view low-poly hull: "
                    f"side mask {payload['side_mask_size'][0]}x{payload['side_mask_size'][1]}"
                )
            else:
                print(
                    f"Two-view hull: {result.voxel_count} voxels, "
                    f"side mask {payload['side_mask_size'][0]}x{payload['side_mask_size'][1]}"
                )
        if result.plan["notes"]:
            print("Notes: " + "; ".join(result.plan["notes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
