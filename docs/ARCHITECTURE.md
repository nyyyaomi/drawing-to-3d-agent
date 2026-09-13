# Architecture

## Responsibilities

| Module | Responsibility |
| --- | --- |
| `agent.py` | Configuration, planning, processing, view alignment, and export orchestration. |
| `image_editing.py` | Local and optional LLM planners; thresholding, morphology, cropping, resizing, previews. |
| `mesh.py` | Extrusion, voxel hulls, profile simplification, triangulation, smoothing, STL/OBJ writers. |
| `cli.py` | Command-line options and human-readable or JSON results. |

Only NumPy and Pillow are required for the local pipeline. The optional OpenAI
SDK is imported only when its planner is selected. There is no trained vision
model in this repository; the reconstruction is geometric.

## Two-View Geometry

In voxel mode, a 3D location is occupied only if its projection falls inside both
binary silhouettes. This is a two-view visual hull. It preserves visible holes
in the masks, but its surface inherits the sampling grid.

Low-poly mode samples the minimum and maximum bounds of each silhouette along
their shared axis. Those four boundary functions define a rectangle at each
position. The algorithm simplifies each function with Ramer-Douglas-Peucker,
then removes small angular turns. Neighboring rectangles are connected with
triangles, and both ends are capped.

This model is deliberately approximate: cross-sections are rectangles, the ends
are located at sample centers, and internal gaps between extrema are filled.
Blank longitudinal sections may also be bridged. It is not a general surface
reconstruction or a Boolean CAD operation.

## Why Interpolation Matters

The top view may need a corner at a position where the side view is straight.
The mesh needs the union of both views' retained sample positions. Reusing the
original raster height at those extra positions would reintroduce tiny ridges.
Instead, each boundary is linearly interpolated from its own retained vertices.
This keeps the side edge on a straight segment while allowing the top to change
width. A regression test covers this case with a quantized ramp.

## Coordinate Conventions

The mesh uses x and y in the ground plane and z for height. Image rows increase
downward; model z increases upward. With `side_axis="x"`, image columns map to
model x. With `side_axis="y"`, top image rows align with side image columns; the
top of the top image maps to larger model y. Independent flips and mirrors let
the user correct drawing orientation before tracing.

Images are cropped and resized independently to align their shared length.
This assumes corresponding endpoints, not calibrated cameras. Side-view height
is capped by `max_size`, which can alter proportions on very tall inputs.
`pixel_size` sets model units per processed pixel; dimensions are not inferred
from labels in the drawing.

## Validation and Tradeoffs

Tests use synthetic silhouettes with known properties and exercise real STL/OBJ
exports. They check closed, consistently oriented sample meshes, a straight ramp
with extra top-view vertices, view flips, invalid inputs, and the CLI contract.
They do not guarantee valid topology or dimensional accuracy for every input.

Increasing resolution increases voxel memory and face counts. Profile-based
generation avoids allocating a dense 3D volume, but sacrifices cross-section
detail and holes. Raising the simplification angle can exceed the initial
distance tolerance, and smoothing can shrink features. The optional base is a
separate shell; this project does not implement mesh Boolean union.

The OpenAI planner accepts JSON operations but is not a hardened untrusted-input
service. Its output is dispatched through known image operations, which reject
unknown names. Production use would need stricter schema and parameter limits.
