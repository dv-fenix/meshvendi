"""Determinism and shape checks for FPS sampling and normalization.

Verifies that:
  * FPS on the same mesh with the same seed produces byte-identical points.
  * FPS on the same mesh with *different* seeds produces different points.
  * Normalization centers at origin and respects the unit-sphere constraint.
  * Surface sampler returns the requested count with shape ``(count, 3)``.
"""

from __future__ import annotations

import unittest

import numpy as np
import trimesh

from diversity.sampling import (_farthest_point_sampling, _sample_surface,
                                fps_sample, normalize_to_unit_sphere)


def _unit_box_mesh() -> trimesh.Trimesh:
    """A simple watertight cube; trivially samplable."""
    return trimesh.creation.box(extents=(1.0, 1.0, 1.0))


class TestSurfaceSampling(unittest.TestCase):

    def test_returns_correct_shape(self):
        mesh = _unit_box_mesh()
        rng = np.random.default_rng(0)
        pts = _sample_surface(mesh, count=200, rng=rng)
        self.assertEqual(pts.shape, (200, 3))

    def test_points_lie_on_or_in_box(self):
        mesh = _unit_box_mesh()
        rng = np.random.default_rng(0)
        pts = _sample_surface(mesh, count=500, rng=rng)
        # Cube extends [-0.5, 0.5] on each axis. Surface points must be within.
        self.assertLessEqual(float(np.max(np.abs(pts))), 0.5 + 1e-9)


class TestFPSDeterminism(unittest.TestCase):

    def test_same_seed_yields_identical_clouds(self):
        mesh = _unit_box_mesh()
        a = fps_sample(mesh, n=64, rng=np.random.default_rng(123))
        b = fps_sample(mesh, n=64, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_yield_different_clouds(self):
        mesh = _unit_box_mesh()
        a = fps_sample(mesh, n=64, rng=np.random.default_rng(1))
        b = fps_sample(mesh, n=64, rng=np.random.default_rng(2))
        self.assertFalse(np.array_equal(a, b))

    def test_fps_picks_correct_count(self):
        mesh = _unit_box_mesh()
        pts = fps_sample(mesh, n=128, rng=np.random.default_rng(0))
        self.assertEqual(pts.shape, (128, 3))


class TestFPSAlgorithm(unittest.TestCase):

    def test_fps_picks_extreme_points_on_line(self):
        # Candidates at integer positions along x-axis. FPS with n=2 should
        # pick endpoints (or one endpoint and the seed); at minimum, the second
        # point must be one of the two extremes since they maximize distance
        # from any single seed.
        candidates = np.zeros((11, 3))
        candidates[:, 0] = np.arange(11)
        rng = np.random.default_rng(0)
        picked = _farthest_point_sampling(candidates, n=2, rng=rng)
        x_picked = sorted(picked[:, 0].tolist())
        self.assertEqual(len(x_picked), 2)
        # The second-picked point should be one of the extremes.
        self.assertTrue(0.0 in x_picked or 10.0 in x_picked)


class TestNormalization(unittest.TestCase):

    def test_centered_at_origin(self):
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(100, 3)) + np.array([5.0, -2.0, 1.0])
        norm = normalize_to_unit_sphere(pts)
        np.testing.assert_allclose(norm.mean(axis=0), 0.0, atol=1e-12)

    def test_max_radius_is_one(self):
        rng = np.random.default_rng(1)
        pts = rng.normal(size=(100, 3)) * 17.5
        norm = normalize_to_unit_sphere(pts)
        radii = np.linalg.norm(norm, axis=1)
        self.assertAlmostEqual(float(radii.max()), 1.0, places=10)

    def test_all_points_inside_unit_sphere(self):
        rng = np.random.default_rng(2)
        pts = rng.normal(size=(200, 3))
        norm = normalize_to_unit_sphere(pts)
        radii = np.linalg.norm(norm, axis=1)
        self.assertLessEqual(float(radii.max()), 1.0 + 1e-12)


if __name__ == "__main__":
    unittest.main()
