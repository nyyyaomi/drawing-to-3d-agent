# Portfolio Notes

## Project Summary

Drawing to 3D Agent is a Python tool that reconstructs approximate 3D meshes from
top and side drawing silhouettes. Its low-poly path simplifies raster boundaries
into straight segments and exports STL or OBJ files for inspection in a slicer
or mesh editor.

## Resume Wording

- Developed a Python drawing-to-3D tool with NumPy and Pillow that combines top
  and side silhouettes into STL/OBJ meshes, with configurable view alignment and
  polygon simplification.
- Added profile interpolation to preserve straight edges across shared mesh
  vertices, plus automated geometry and CLI regression tests and a GitHub Actions
  workflow for Windows and Linux.

Use claims that reflect the work you understand and can explain. Development
used AI coding assistance. Do not claim a trained AI model, production users,
printing accuracy, or performance improvements that have not been measured.

## Demo Sequence

1. Show the two included input silhouettes and the generated STL in the README.
2. Run the Quick Start command and open the output in a mesh viewer.
3. Change the simplification tolerance and explain how shape detail changes.
4. Show a view flip, then discuss why two silhouettes cannot reveal hidden shape.
5. Walk through the straight-edge regression test and the CI workflow.

## Interview Topics

- How grayscale thresholding and morphology prepare a silhouette.
- Why a visual hull is ambiguous with only two projections.
- How Ramer-Douglas-Peucker and angular pruning reduce raster noise.
- Why each profile must be interpolated at vertices introduced by another view.
- The tradeoff between voxel fidelity and rectangular low-poly cross-sections.
- How triangle winding and shared edges affect a closed mesh.
- How an optional language planner differs from the geometric modeling engine.
