from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, TypeAlias

import numpy as np

Vector: TypeAlias = tuple[float, float, float]
Triangle: TypeAlias = tuple[Vector, Vector, Vector]
LowPolySample: TypeAlias = tuple[float, float, float, float, float]


def mask_to_triangles(
    mask: np.ndarray,
    *,
    pixel_size: float = 0.4,
    height: float = 4.0,
    base_height: float = 0.0,
    base_margin: int = 4,
) -> list[Triangle]:
    if pixel_size <= 0:
        raise ValueError("pixel_size must be greater than zero")
    if height <= 0:
        raise ValueError("height must be greater than zero")
    if base_height < 0:
        raise ValueError("base_height cannot be negative")

    mask = mask.astype(bool)
    rows, cols = mask.shape
    triangles: list[Triangle] = []

    if base_height > 0:
        margin = max(0, base_margin) * pixel_size
        triangles.extend(
            cuboid_triangles(
                -margin,
                -margin,
                0.0,
                cols * pixel_size + margin,
                rows * pixel_size + margin,
                base_height,
            )
        )

    z0 = base_height
    z1 = base_height + height
    for row in range(rows):
        for col in range(cols):
            if not mask[row, col]:
                continue
            x0 = col * pixel_size
            x1 = (col + 1) * pixel_size
            y0 = (rows - row - 1) * pixel_size
            y1 = (rows - row) * pixel_size

            _add_quad(
                triangles,
                (x0, y0, z1),
                (x1, y0, z1),
                (x1, y1, z1),
                (x0, y1, z1),
            )
            _add_quad(
                triangles,
                (x0, y1, z0),
                (x1, y1, z0),
                (x1, y0, z0),
                (x0, y0, z0),
            )

            if col == 0 or not mask[row, col - 1]:
                _add_quad(
                    triangles,
                    (x0, y0, z0),
                    (x0, y0, z1),
                    (x0, y1, z1),
                    (x0, y1, z0),
                )
            if col == cols - 1 or not mask[row, col + 1]:
                _add_quad(
                    triangles,
                    (x1, y1, z0),
                    (x1, y1, z1),
                    (x1, y0, z1),
                    (x1, y0, z0),
                )
            if row == rows - 1 or not mask[row + 1, col]:
                _add_quad(
                    triangles,
                    (x1, y0, z0),
                    (x1, y0, z1),
                    (x0, y0, z1),
                    (x0, y0, z0),
                )
            if row == 0 or not mask[row - 1, col]:
                _add_quad(
                    triangles,
                    (x0, y1, z0),
                    (x0, y1, z1),
                    (x1, y1, z1),
                    (x1, y1, z0),
                )
    return triangles


def visual_hull_voxels(
    top_mask: np.ndarray,
    side_mask: np.ndarray,
    *,
    side_axis: str = "x",
) -> np.ndarray:
    """Combine top and side silhouettes into a simple two-view visual hull.

    Returns a boolean array in z, y, x order. The side view's vertical image axis
    is flipped so the bottom of the drawing becomes z=0.
    """
    top_mask = top_mask.astype(bool)
    side_mask = np.flipud(side_mask.astype(bool))

    if side_axis == "x":
        if side_mask.shape[1] != top_mask.shape[1]:
            raise ValueError("side view width must match top view width for side_axis='x'")
        return side_mask[:, np.newaxis, :] & top_mask[np.newaxis, :, :]

    if side_axis == "y":
        if side_mask.shape[1] != top_mask.shape[0]:
            raise ValueError("side view width must match top view height for side_axis='y'")
        return side_mask[:, :, np.newaxis] & top_mask[np.newaxis, :, :]

    raise ValueError("side_axis must be one of: x, y")


def voxels_to_triangles(
    voxels: np.ndarray,
    *,
    pixel_size: float = 0.4,
    base_height: float = 0.0,
    base_margin: int = 4,
) -> list[Triangle]:
    if pixel_size <= 0:
        raise ValueError("pixel_size must be greater than zero")
    if base_height < 0:
        raise ValueError("base_height cannot be negative")

    voxels = voxels.astype(bool)
    z_count, rows, cols = voxels.shape
    triangles: list[Triangle] = []

    if base_height > 0:
        margin = max(0, base_margin) * pixel_size
        triangles.extend(
            cuboid_triangles(
                -margin,
                -margin,
                0.0,
                cols * pixel_size + margin,
                rows * pixel_size + margin,
                base_height,
            )
        )

    for z in range(z_count):
        for row in range(rows):
            for col in range(cols):
                if not voxels[z, row, col]:
                    continue

                x0 = col * pixel_size
                x1 = (col + 1) * pixel_size
                y0 = (rows - row - 1) * pixel_size
                y1 = (rows - row) * pixel_size
                z0 = base_height + z * pixel_size
                z1 = base_height + (z + 1) * pixel_size

                if z == z_count - 1 or not voxels[z + 1, row, col]:
                    _add_quad(
                        triangles,
                        (x0, y0, z1),
                        (x1, y0, z1),
                        (x1, y1, z1),
                        (x0, y1, z1),
                    )
                if z == 0 or not voxels[z - 1, row, col]:
                    _add_quad(
                        triangles,
                        (x0, y1, z0),
                        (x1, y1, z0),
                        (x1, y0, z0),
                        (x0, y0, z0),
                    )
                if col == 0 or not voxels[z, row, col - 1]:
                    _add_quad(
                        triangles,
                        (x0, y0, z0),
                        (x0, y0, z1),
                        (x0, y1, z1),
                        (x0, y1, z0),
                    )
                if col == cols - 1 or not voxels[z, row, col + 1]:
                    _add_quad(
                        triangles,
                        (x1, y1, z0),
                        (x1, y1, z1),
                        (x1, y0, z1),
                        (x1, y0, z0),
                    )
                if row == rows - 1 or not voxels[z, row + 1, col]:
                    _add_quad(
                        triangles,
                        (x1, y0, z0),
                        (x1, y0, z1),
                        (x0, y0, z1),
                        (x0, y0, z0),
                    )
                if row == 0 or not voxels[z, row - 1, col]:
                    _add_quad(
                        triangles,
                        (x0, y1, z0),
                        (x0, y1, z1),
                        (x1, y1, z1),
                        (x1, y1, z0),
                    )
    return triangles


def low_poly_hull_triangles(
    top_mask: np.ndarray,
    side_mask: np.ndarray,
    *,
    side_axis: str = "y",
    pixel_size: float = 0.4,
    base_height: float = 0.0,
    base_margin: int = 4,
    simplify_angle_degrees: float = 10.0,
    simplify_tolerance_pixels: float = 3.0,
) -> list[Triangle]:
    """Build a faceted two-view hull from simplified silhouette outlines.

    This intentionally removes tiny pixel-step ridges. It keeps vertices at
    corners and meaningful bends, then connects them with straight facets.
    """
    if pixel_size <= 0:
        raise ValueError("pixel_size must be greater than zero")
    if base_height < 0:
        raise ValueError("base_height cannot be negative")

    top_mask = top_mask.astype(bool)
    side_mask = side_mask.astype(bool)
    triangles: list[Triangle] = []

    rows, cols = top_mask.shape
    if base_height > 0:
        margin = max(0, base_margin) * pixel_size
        triangles.extend(
            cuboid_triangles(
                -margin,
                -margin,
                0.0,
                cols * pixel_size + margin,
                rows * pixel_size + margin,
                base_height,
            )
        )

    if side_axis == "y":
        samples = _low_poly_samples_y(
            top_mask,
            side_mask,
            pixel_size=pixel_size,
            base_height=base_height,
        )
        samples = _simplify_low_poly_samples(
            samples,
            angle_degrees=simplify_angle_degrees,
            tolerance=simplify_tolerance_pixels * pixel_size,
        )
        triangles.extend(_low_poly_samples_y_to_triangles(samples))
        return triangles

    if side_axis == "x":
        samples = _low_poly_samples_x(
            top_mask,
            side_mask,
            pixel_size=pixel_size,
            base_height=base_height,
        )
        samples = _simplify_low_poly_samples(
            samples,
            angle_degrees=simplify_angle_degrees,
            tolerance=simplify_tolerance_pixels * pixel_size,
        )
        triangles.extend(_low_poly_samples_x_to_triangles(samples))
        return triangles

    raise ValueError("side_axis must be one of: x, y")


def cuboid_triangles(
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
) -> list[Triangle]:
    triangles: list[Triangle] = []
    _add_quad(triangles, (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))
    _add_quad(triangles, (x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0))
    _add_quad(triangles, (x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0))
    _add_quad(triangles, (x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (x1, y0, z0))
    _add_quad(triangles, (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (x0, y0, z0))
    _add_quad(triangles, (x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0))
    return triangles


def write_ascii_stl(path: Path, triangles: Iterable[Triangle], solid_name: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_solid_name(solid_name)
    count = 0
    with path.open("w", encoding="ascii", newline="\n") as stl:
        stl.write(f"solid {safe_name}\n")
        for triangle in triangles:
            normal = _normal(triangle)
            stl.write(
                "  facet normal "
                f"{normal[0]:.8g} {normal[1]:.8g} {normal[2]:.8g}\n"
            )
            stl.write("    outer loop\n")
            for vertex in triangle:
                stl.write(
                    f"      vertex {vertex[0]:.8g} {vertex[1]:.8g} {vertex[2]:.8g}\n"
                )
            stl.write("    endloop\n")
            stl.write("  endfacet\n")
            count += 1
        stl.write(f"endsolid {safe_name}\n")
    return count


def write_obj(path: Path, triangles: Iterable[Triangle], object_name: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_solid_name(object_name)
    vertices, faces = _index_triangles(triangles)

    with path.open("w", encoding="ascii", newline="\n") as obj:
        obj.write(f"# Generated by drawing_to_3d_agent\n")
        obj.write(f"o {safe_name}\n")
        for vertex in vertices:
            obj.write(f"v {vertex[0]:.8g} {vertex[1]:.8g} {vertex[2]:.8g}\n")
        for a, b, c in faces:
            obj.write(f"f {a + 1} {b + 1} {c + 1}\n")
    return len(faces)


def write_model(path: Path, triangles: Iterable[Triangle], model_name: str) -> int:
    suffix = path.suffix.lower()
    if suffix == ".stl":
        return write_ascii_stl(path, triangles, solid_name=model_name)
    if suffix == ".obj":
        return write_obj(path, triangles, object_name=model_name)
    raise ValueError("Supported output formats are .stl and .obj.")


def smooth_triangles(
    triangles: Iterable[Triangle],
    *,
    iterations: int = 4,
    strength: float = 0.35,
    preserve_bottom: bool = True,
) -> list[Triangle]:
    """Laplacian-smooth a closed triangle mesh while keeping topology intact."""
    if iterations <= 0:
        return list(triangles)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("smooth strength must be between 0 and 1")

    vertices, faces = _index_triangles(triangles)
    if not vertices:
        return []

    adjacency: list[set[int]] = [set() for _ in vertices]
    for a, b, c in faces:
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))

    locked: set[int] = set()
    if preserve_bottom:
        min_z = min(vertex[2] for vertex in vertices)
        locked = {
            index
            for index, vertex in enumerate(vertices)
            if abs(vertex[2] - min_z) <= 1e-9
        }

    current = [list(vertex) for vertex in vertices]
    for _ in range(iterations):
        next_vertices = [vertex[:] for vertex in current]
        for index, neighbors in enumerate(adjacency):
            if index in locked or not neighbors:
                continue
            avg_x = sum(current[neighbor][0] for neighbor in neighbors) / len(neighbors)
            avg_y = sum(current[neighbor][1] for neighbor in neighbors) / len(neighbors)
            avg_z = sum(current[neighbor][2] for neighbor in neighbors) / len(neighbors)
            next_vertices[index][0] = current[index][0] * (1.0 - strength) + avg_x * strength
            next_vertices[index][1] = current[index][1] * (1.0 - strength) + avg_y * strength
            next_vertices[index][2] = current[index][2] * (1.0 - strength) + avg_z * strength
        current = next_vertices

    return [
        (
            _as_vector(current[a]),
            _as_vector(current[b]),
            _as_vector(current[c]),
        )
        for a, b, c in faces
    ]


def _index_triangles(triangles: Iterable[Triangle]) -> tuple[list[Vector], list[tuple[int, int, int]]]:
    vertex_to_index: dict[Vector, int] = {}
    vertices: list[Vector] = []
    faces: list[tuple[int, int, int]] = []

    for triangle in triangles:
        face: list[int] = []
        for vertex in triangle:
            index = vertex_to_index.get(vertex)
            if index is None:
                index = len(vertices)
                vertex_to_index[vertex] = index
                vertices.append(vertex)
            face.append(index)
        faces.append((face[0], face[1], face[2]))
    return vertices, faces


def _as_vector(vertex: list[float]) -> Vector:
    return (float(vertex[0]), float(vertex[1]), float(vertex[2]))


def _low_poly_samples_y(
    top_mask: np.ndarray,
    side_mask: np.ndarray,
    *,
    pixel_size: float,
    base_height: float,
) -> list[LowPolySample]:
    rows, _ = top_mask.shape
    side_rows, side_cols = side_mask.shape
    if side_cols != rows:
        raise ValueError("side view width must match top view height for side_axis='y'")

    raw: list[LowPolySample] = []
    for row in range(rows):
        x_bounds = _bounds_from_line(top_mask[row, :])
        z_bounds = _bounds_from_line(side_mask[:, row])
        if x_bounds is None or z_bounds is None:
            continue

        x0, x1 = x_bounds
        z_row0, z_row1 = z_bounds
        y = (rows - row - 0.5) * pixel_size
        z0 = base_height + (side_rows - z_row1) * pixel_size
        z1 = base_height + (side_rows - z_row0) * pixel_size
        raw.append((y, x0 * pixel_size, x1 * pixel_size, z0, z1))

    return raw


def _low_poly_samples_x(
    top_mask: np.ndarray,
    side_mask: np.ndarray,
    *,
    pixel_size: float,
    base_height: float,
) -> list[LowPolySample]:
    rows, cols = top_mask.shape
    side_rows, side_cols = side_mask.shape
    if side_cols != cols:
        raise ValueError("side view width must match top view width for side_axis='x'")

    raw: list[LowPolySample] = []
    for col in range(cols):
        y_bounds = _bounds_from_line(top_mask[:, col])
        z_bounds = _bounds_from_line(side_mask[:, col])
        if y_bounds is None or z_bounds is None:
            continue

        y_row0, y_row1 = y_bounds
        z_row0, z_row1 = z_bounds
        x = (col + 0.5) * pixel_size
        y0 = (rows - y_row1) * pixel_size
        y1 = (rows - y_row0) * pixel_size
        z0 = base_height + (side_rows - z_row1) * pixel_size
        z1 = base_height + (side_rows - z_row0) * pixel_size
        raw.append((x, y0, y1, z0, z1))

    return raw


def _bounds_from_line(line: np.ndarray) -> tuple[int, int] | None:
    filled = np.where(line)[0]
    if len(filled) == 0:
        return None
    return int(filled.min()), int(filled.max()) + 1


def _simplify_low_poly_samples(
    samples: list[LowPolySample],
    *,
    angle_degrees: float,
    tolerance: float,
) -> list[LowPolySample]:
    if len(samples) < 2:
        raise ValueError("Not enough overlapping top/side silhouette samples.")

    simplified_by_value: dict[int, set[int]] = {}
    keep: set[int] = {0, len(samples) - 1}
    for value_index in range(1, 5):
        curve = [(sample[0], sample[value_index]) for sample in samples]
        simplified = _rdp_indices(curve, tolerance=tolerance)
        simplified = _drop_small_turn_indices(
            curve,
            indices=simplified,
            angle_degrees=angle_degrees,
        )
        simplified_by_value[value_index] = simplified
        keep.update(simplified)

    simplified_samples: list[LowPolySample] = []
    for index in sorted(keep):
        axis = samples[index][0]
        simplified_samples.append(
            (
                axis,
                _interpolated_curve_value(samples, simplified_by_value[1], index, 1),
                _interpolated_curve_value(samples, simplified_by_value[2], index, 2),
                _interpolated_curve_value(samples, simplified_by_value[3], index, 3),
                _interpolated_curve_value(samples, simplified_by_value[4], index, 4),
            )
        )
    return simplified_samples


def _interpolated_curve_value(
    samples: list[LowPolySample],
    key_indices: set[int],
    index: int,
    value_index: int,
) -> float:
    ordered = sorted(key_indices)
    if index in key_indices:
        return samples[index][value_index]

    previous_index = ordered[0]
    next_index = ordered[-1]
    for candidate in ordered:
        if candidate < index:
            previous_index = candidate
        elif candidate > index:
            next_index = candidate
            break

    if previous_index == next_index:
        return samples[previous_index][value_index]

    axis = samples[index][0]
    previous_axis = samples[previous_index][0]
    next_axis = samples[next_index][0]
    if next_axis == previous_axis:
        return samples[previous_index][value_index]

    ratio = (axis - previous_axis) / (next_axis - previous_axis)
    previous_value = samples[previous_index][value_index]
    next_value = samples[next_index][value_index]
    return previous_value + ratio * (next_value - previous_value)


def _rdp_indices(
    points: list[tuple[float, float]],
    *,
    tolerance: float,
) -> set[int]:
    if len(points) <= 2 or tolerance <= 0:
        return set(range(len(points)))

    keep = {0, len(points) - 1}

    def simplify(start: int, end: int) -> None:
        if end <= start + 1:
            return
        max_distance = -1.0
        max_index = start
        a = points[start]
        b = points[end]
        for index in range(start + 1, end):
            distance = _point_line_distance(points[index], a, b)
            if distance > max_distance:
                max_distance = distance
                max_index = index
        if max_distance > tolerance:
            keep.add(max_index)
            simplify(start, max_index)
            simplify(max_index, end)

    simplify(0, len(points) - 1)
    return keep


def _drop_small_turn_indices(
    points: list[tuple[float, float]],
    *,
    indices: set[int],
    angle_degrees: float,
) -> set[int]:
    ordered = sorted(indices)
    if len(ordered) <= 2 or angle_degrees <= 0:
        return set(ordered)

    changed = True
    while changed and len(ordered) > 2:
        changed = False
        next_ordered = [ordered[0]]
        for previous_index, current_index, next_index in zip(
            ordered,
            ordered[1:],
            ordered[2:],
            strict=False,
        ):
            turn = _turn_angle_degrees(
                points[previous_index],
                points[current_index],
                points[next_index],
            )
            if turn >= angle_degrees:
                next_ordered.append(current_index)
            else:
                changed = True
        next_ordered.append(ordered[-1])
        ordered = next_ordered
    return set(ordered)


def _point_line_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    nearest_x = ax + t * dx
    nearest_y = ay + t * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def _turn_angle_degrees(
    previous: tuple[float, float],
    current: tuple[float, float],
    following: tuple[float, float],
) -> float:
    ax = current[0] - previous[0]
    ay = current[1] - previous[1]
    bx = following[0] - current[0]
    by = following[1] - current[1]
    len_a = math.hypot(ax, ay)
    len_b = math.hypot(bx, by)
    if len_a == 0 or len_b == 0:
        return 0.0
    dot = max(-1.0, min(1.0, (ax * bx + ay * by) / (len_a * len_b)))
    return math.degrees(math.acos(dot))


def _low_poly_samples_y_to_triangles(samples: list[LowPolySample]) -> list[Triangle]:
    triangles: list[Triangle] = []
    rects = [_rect_from_y_sample(sample) for sample in samples]

    for current, next_rect in zip(rects, rects[1:], strict=False):
        a0, b0, c0, d0 = current
        a1, b1, c1, d1 = next_rect
        _add_quad(triangles, a0, b0, b1, a1)
        _add_quad(triangles, b0, c0, c1, b1)
        _add_quad(triangles, c0, d0, d1, c1)
        _add_quad(triangles, d0, a0, a1, d1)

    a0, b0, c0, d0 = rects[0]
    a1, b1, c1, d1 = rects[-1]
    _add_quad(triangles, a0, d0, c0, b0)
    _add_quad(triangles, a1, b1, c1, d1)
    return triangles


def _low_poly_samples_x_to_triangles(samples: list[LowPolySample]) -> list[Triangle]:
    triangles: list[Triangle] = []
    rects = [_rect_from_x_sample(sample) for sample in samples]

    for current, next_rect in zip(rects, rects[1:], strict=False):
        a0, b0, c0, d0 = current
        a1, b1, c1, d1 = next_rect
        _add_quad(triangles, a0, b0, b1, a1)
        _add_quad(triangles, b0, c0, c1, b1)
        _add_quad(triangles, c0, d0, d1, c1)
        _add_quad(triangles, d0, a0, a1, d1)

    a0, b0, c0, d0 = rects[0]
    a1, b1, c1, d1 = rects[-1]
    _add_quad(triangles, a0, d0, c0, b0)
    _add_quad(triangles, a1, b1, c1, d1)
    return triangles


def _rect_from_y_sample(sample: LowPolySample) -> tuple[Vector, Vector, Vector, Vector]:
    y, x0, x1, z0, z1 = sample
    return (
        (x0, y, z0),
        (x1, y, z0),
        (x1, y, z1),
        (x0, y, z1),
    )


def _rect_from_x_sample(sample: LowPolySample) -> tuple[Vector, Vector, Vector, Vector]:
    x, y0, y1, z0, z1 = sample
    return (
        (x, y0, z0),
        (x, y1, z0),
        (x, y1, z1),
        (x, y0, z1),
    )


def _add_quad(
    triangles: list[Triangle],
    v0: Vector,
    v1: Vector,
    v2: Vector,
    v3: Vector,
) -> None:
    triangles.append((v0, v1, v2))
    triangles.append((v0, v2, v3))


def _normal(triangle: Triangle) -> Vector:
    a, b, c = triangle
    ux, uy, uz = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    vx, vy, vz = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length == 0:
        return (0.0, 0.0, 0.0)
    return (nx / length, ny / length, nz / length)


def _safe_solid_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())
    return cleaned or "drawing_model"
