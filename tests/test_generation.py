from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

from drawing_to_3d_agent import AgentConfig, DrawingTo3DAgent
from drawing_to_3d_agent.mesh import low_poly_hull_triangles, visual_hull_voxels


ROOT = Path(__file__).resolve().parents[1]


def read_stl(path: Path) -> np.ndarray:
    vertices = [
        [float(value) for value in line.split()[1:]]
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip().startswith("vertex ")
    ]
    return np.asarray(vertices).reshape(-1, 3, 3)


class MeshAssertions:
    def assert_closed_outward_mesh(self, triangles):
        mesh = np.asarray(triangles, dtype=float)
        self.assertTrue(np.isfinite(mesh).all())
        areas = np.linalg.norm(
            np.cross(mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0]), axis=1
        )
        self.assertTrue((areas > 1e-9).all(), "Degenerate triangle")
        edges = Counter()
        for triangle in triangles:
            for a, b in zip(triangle, [*triangle[1:], triangle[0]]):
                edges[(tuple(a), tuple(b))] += 1
        for (a, b), count in edges.items():
            self.assertEqual(count, 1, "Duplicate directed edge")
            self.assertEqual(edges[(b, a)], 1, "Open or inconsistently wound edge")
        volume = np.einsum(
            "ij,ij->i", mesh[:, 0], np.cross(mesh[:, 1], mesh[:, 2])
        ).sum() / 6
        self.assertGreater(volume, 0, "Mesh must have outward orientation")


class GeometryTests(MeshAssertions, unittest.TestCase):
    def test_low_poly_is_closed_for_both_axes(self):
        top = np.ones((18, 32), dtype=bool)
        for axis, length in (("x", 32), ("y", 18)):
            with self.subTest(axis=axis):
                side = np.ones((12, length), dtype=bool)
                triangles = low_poly_hull_triangles(top, side, side_axis=axis)
                self.assertEqual(len(triangles), 12)
                self.assert_closed_outward_mesh(triangles)

    def test_extra_top_vertices_do_not_reintroduce_side_ridges(self):
        length = 81
        top_x = np.zeros((30, length), dtype=bool)
        side = np.zeros((40, length), dtype=bool)
        for column in range(length):
            width = round(np.interp(column, [0, 17, 41, 63, 80], [10, 26, 12, 24, 10]))
            left = (30 - width) // 2
            top_x[left:left + width, column] = True
            height = 12 + round(column * 0.2)
            side[-height:, column] = True
        for axis, top in (("x", top_x), ("y", top_x.T)):
            with self.subTest(axis=axis):
                mesh = np.asarray(low_poly_hull_triangles(
                    top, side, side_axis=axis, pixel_size=1,
                    simplify_tolerance_pixels=1, simplify_angle_degrees=0,
                ))
                vertices = mesh.reshape(-1, 3)
                position = vertices[:, 0] if axis == "x" else length - vertices[:, 1]
                stations = np.unique(position)
                self.assertGreater(len(stations), 2, "Top view must introduce extra vertices")
                heights = np.array([vertices[position == x, 2].max() for x in stations])
                expected = np.interp(stations, [stations[0], stations[-1]], [12, 28])
                np.testing.assert_allclose(heights, expected, atol=1e-9)
                self.assert_closed_outward_mesh(mesh)

    def test_voxel_projection_matches_both_silhouettes(self):
        top_x = np.ones((4, 6), dtype=bool)
        top_x[1, 2:4] = False
        side = np.ones((3, 6), dtype=bool)
        side[0, 2] = False
        for axis, top in (("x", top_x), ("y", top_x.T)):
            with self.subTest(axis=axis):
                volume = visual_hull_voxels(top, side, side_axis=axis)
                np.testing.assert_array_equal(volume.any(axis=0), top)
                projection = volume.any(axis=1 if axis == "x" else 2)
                np.testing.assert_array_equal(projection, np.flipud(side))


class AgentTests(MeshAssertions, unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_end_to_end_low_poly_stl(self):
        result = DrawingTo3DAgent(AgentConfig(
            max_size=96, low_poly=True, fill_holes=True,
        )).generate(
            ROOT / "examples/sample_top_view.png", self.folder / "model.stl",
            side_image=ROOT / "examples/sample_side_view.png",
            preview_path=self.folder / "top.png", side_preview_path=self.folder / "side.png",
        )
        mesh = read_stl(result.output_path)
        self.assertEqual(result.generation_mode, "two_view_low_poly")
        self.assertEqual(len(mesh), result.triangle_count)
        self.assertLess(len(mesh), 200)
        self.assert_closed_outward_mesh(mesh)
        with Image.open(result.preview_path) as preview:
            self.assertEqual(preview.size, (result.mask_width, result.mask_height))
        self.assertTrue(result.side_preview_path.exists())

    def test_single_image_obj_has_valid_faces_and_height(self):
        source = self.folder / "rectangle.png"
        drawing = Image.new("L", (16, 16), 255)
        ImageDraw.Draw(drawing).rectangle((4, 4, 11, 11), fill=0)
        drawing.save(source)
        result = DrawingTo3DAgent(AgentConfig(height=7.5)).generate(
            source, self.folder / "single.obj",
        )
        lines = result.output_path.read_text(encoding="ascii").splitlines()
        vertices = [list(map(float, line.split()[1:])) for line in lines if line.startswith("v ")]
        faces = [list(map(int, line.split()[1:])) for line in lines if line.startswith("f ")]
        self.assertEqual(len(faces), result.triangle_count)
        self.assertTrue(all(1 <= index <= len(vertices) for face in faces for index in face))
        self.assertEqual(max(v[2] for v in vertices) - min(v[2] for v in vertices), 7.5)
        self.assertEqual(result.generation_mode, "single")

    def test_top_flip_reverses_preview_without_flipping_side(self):
        drawing = Image.new("L", (32, 32), 255)
        ImageDraw.Draw(drawing).polygon([(4, 4), (12, 4), (27, 27), (4, 27)], fill=0)
        source = self.folder / "asymmetric.png"
        drawing.save(source)
        previews = []
        side_previews = []
        for flip in (False, True):
            result = DrawingTo3DAgent(AgentConfig(
                top_flip_vertical=flip, low_poly=True,
            )).generate(
                source, self.folder / f"flip_{flip}.stl",
                side_image=ROOT / "examples/sample_side_view.png",
                preview_path=self.folder / f"top_{flip}.png",
                side_preview_path=self.folder / f"side_{flip}.png",
            )
            with Image.open(result.preview_path) as preview:
                previews.append(np.array(preview))
            with Image.open(result.side_preview_path) as preview:
                side_previews.append(np.array(preview))
        np.testing.assert_array_equal(previews[1], np.flipud(previews[0]))
        np.testing.assert_array_equal(side_previews[1], side_previews[0])

    def test_blank_input_and_unsupported_export_fail(self):
        blank = self.folder / "blank.png"
        Image.new("L", (16, 16), 255).save(blank)
        agent = DrawingTo3DAgent()
        with self.assertRaisesRegex(ValueError, "No ink"):
            agent.generate(blank, self.folder / "blank.stl")
        with self.assertRaisesRegex(ValueError, "ending in"):
            agent.generate(blank, self.folder / "invalid.fbx")

    def test_cli_json_and_rules_plan(self):
        output = self.folder / "cli.stl"
        run = subprocess.run([
            sys.executable, "-m", "drawing_to_3d_agent",
            str(ROOT / "examples/sample_top_view.png"), str(output),
            "--side-image", str(ROOT / "examples/sample_side_view.png"),
            "--low-poly", "--max-size", "64", "--prompt", "fill holes", "--json",
        ], cwd=ROOT, capture_output=True, text=True, check=True)
        payload = json.loads(run.stdout)
        self.assertEqual(payload["mode"], "two_view_low_poly")
        self.assertEqual(payload["triangles"], len(read_stl(output)))
        self.assertIn("fill_holes", [op["name"] for op in payload["plan"]["mask_ops"]])


if __name__ == "__main__":
    unittest.main()
