"""PSD and structural checks for the kernel matrix.

Synthesizes a small set of point clouds (random points on spheres of varying
radii) and verifies:

  * ``K[i, i] == 1`` after normalization
  * ``K`` is symmetric
  * ``K`` is PSD up to numerical precision
  * The squared-distance matrix returns the same values as a naive double loop
  * The median-heuristic sigma is positive and inside the actual distance range
"""

from __future__ import annotations

import unittest
from typing import List

import numpy as np

from diversity.kernel import (build_kernel_matrices, median_heuristic_sigma,
                              squared_distance_matrix)


def _random_sphere_points(
    n: int, radius: float, rng: np.random.Generator
) -> np.ndarray:
    """Uniform points on the surface of a sphere of given radius."""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * radius


def _make_synthetic_clouds(N: int, n: int, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    radii = np.linspace(0.5, 1.0, N)
    return [_random_sphere_points(n, float(r), rng) for r in radii]


class TestSquaredDistanceMatrix(unittest.TestCase):

    def test_matches_naive_loop(self):
        rng = np.random.default_rng(0)
        P = rng.normal(size=(20, 3))
        Q = rng.normal(size=(15, 3))
        d2_fast = squared_distance_matrix(P, Q)
        d2_naive = np.zeros((20, 15))
        for i in range(20):
            for j in range(15):
                d2_naive[i, j] = float(np.sum((P[i] - Q[j]) ** 2))
        np.testing.assert_allclose(d2_fast, d2_naive, atol=1e-10)

    def test_self_pair_diagonal_is_zero(self):
        rng = np.random.default_rng(1)
        P = rng.normal(size=(10, 3))
        d2 = squared_distance_matrix(P, P)
        np.testing.assert_allclose(np.diag(d2), 0.0, atol=1e-10)

    def test_no_negative_entries(self):
        rng = np.random.default_rng(2)
        P = rng.normal(size=(50, 3))
        d2 = squared_distance_matrix(P, P)
        self.assertGreaterEqual(d2.min(), 0.0)


class TestKernelMatrix(unittest.TestCase):

    def test_diagonal_is_one(self):
        clouds = _make_synthetic_clouds(N=6, n=64, seed=42)
        sigma = median_heuristic_sigma(clouds, np.random.default_rng(0))
        _, K = build_kernel_matrices(clouds, sigma)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_symmetric(self):
        clouds = _make_synthetic_clouds(N=5, n=64, seed=43)
        sigma = median_heuristic_sigma(clouds, np.random.default_rng(0))
        _, K = build_kernel_matrices(clouds, sigma)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_psd_up_to_numerical_precision(self):
        clouds = _make_synthetic_clouds(N=8, n=64, seed=44)
        sigma = median_heuristic_sigma(clouds, np.random.default_rng(0))
        _, K = build_kernel_matrices(clouds, sigma)
        eigvals = np.linalg.eigvalsh(0.5 * (K + K.T))
        # Allow a small numerical slack; in practice we expect ~ -1e-15.
        self.assertGreaterEqual(float(eigvals.min()), -1e-10)


def _detect_torch_accelerator() -> str:
    """Return an available torch GPU device, or '' if none."""
    try:
        import torch
    except ImportError:
        return ""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return ""


class TestTorchBackendMatchesNumpy(unittest.TestCase):
    """The torch GPU path must agree with the numpy CPU path numerically.

    Skipped when no torch GPU is available. The torch backend runs in float32
    on device, so we compare K matrices at single-precision tolerance.
    """

    def setUp(self) -> None:
        device = _detect_torch_accelerator()
        if not device:
            self.skipTest("No torch GPU (cuda/mps) available")
        self.device = device

    def test_K_matches_numpy_within_float32_tolerance(self):
        # Small N and modest n keep this test fast and well below any GPU
        # memory threshold even on Apple unified memory.
        clouds = _make_synthetic_clouds(N=5, n=128, seed=12345)
        sigma = median_heuristic_sigma(clouds, np.random.default_rng(0))

        _, K_cpu = build_kernel_matrices(clouds, sigma, device="cpu")
        _, K_gpu = build_kernel_matrices(
            clouds, sigma, device=self.device, chunk_size=2
        )

        # Diagonals must be exactly 1 for both.
        np.testing.assert_allclose(np.diag(K_gpu), 1.0, atol=1e-10)

        # Off-diagonals: float32 kernel mean has ~6 decimal digits of
        # precision; tolerance picked accordingly.
        np.testing.assert_allclose(K_gpu, K_cpu, atol=1e-5, rtol=1e-5)


class TestMedianHeuristic(unittest.TestCase):

    def test_sigma_is_positive(self):
        clouds = _make_synthetic_clouds(N=6, n=64, seed=45)
        sigma = median_heuristic_sigma(clouds, np.random.default_rng(0))
        self.assertGreater(sigma, 0.0)

    def test_sigma_within_distance_range(self):
        clouds = _make_synthetic_clouds(N=4, n=32, seed=46)
        # Build the full distribution of pairwise distances; sigma must lie in it.
        all_d = []
        for i in range(len(clouds)):
            for j in range(len(clouds)):
                d2 = squared_distance_matrix(clouds[i], clouds[j])
                all_d.append(np.sqrt(d2).ravel())
        all_d = np.concatenate(all_d)
        sigma = median_heuristic_sigma(clouds, np.random.default_rng(0))
        self.assertGreaterEqual(sigma, float(all_d.min()))
        self.assertLessEqual(sigma, float(all_d.max()))


if __name__ == "__main__":
    unittest.main()
