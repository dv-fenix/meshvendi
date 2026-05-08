"""Diagnostic checks for the diversity metric.

Three diagnostics that the metric writeup recommends running every time:

  1. Self-consistency: replicate one mesh K times with *different* FPS seeds,
     run the full pipeline, and confirm VS_1 is close to 1. This catches FPS
     stochasticity leaking into the metric (either the seed isn't being
     respected or n is too small to dampen the noise).

  2. Off-diagonal range: report [min, median, max] of the strict upper triangle
     of the normalized K. A healthy range is roughly [0.3, 0.95]; if every
     entry exceeds 0.95 the kernel is in the "dilution regime" where it can no
     longer discriminate between meshes and the headline number loses meaning.

  3. Eigenvalue spectrum: top 5 and bottom 5 eigenvalues of K/N, plus the most
     negative raw eigenvalue. Eigenvalues below ``-1e-6`` would indicate a PSD
     violation (should not happen for this kernel, but worth checking).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from diversity.kernel import (DEFAULT_TORCH_CHUNK_SIZE, build_kernel_matrices,
                              median_heuristic_sigma)
from diversity.sampling import fps_sample, load_mesh, normalize_to_unit_sphere
from diversity.vendi import compute_vendi_scores

HIGH_OFF_DIAG_THRESHOLD = 0.95


@dataclass(frozen=True)
class OffDiagonalStats:
    """Distribution of strict-upper-triangle entries of K."""

    min: float
    median: float
    max: float
    high_dilution_warning: bool  # True if median > HIGH_OFF_DIAG_THRESHOLD


@dataclass(frozen=True)
class EigenvalueStats:
    """Top and bottom eigenvalues of K / N."""

    top_5: List[float]
    bottom_5: List[float]
    most_negative_raw: float


def off_diagonal_stats(K: np.ndarray) -> OffDiagonalStats:
    """Compute [min, median, max] over the strict upper triangle of K."""
    N = K.shape[0]
    if N < 2:
        # Edge case: degenerate input, no off-diagonal entries to report.
        return OffDiagonalStats(
            min=1.0, median=1.0, max=1.0, high_dilution_warning=False
        )
    iu = np.triu_indices(N, k=1)
    vals = K[iu]
    med = float(np.median(vals))
    return OffDiagonalStats(
        min=float(vals.min()),
        median=med,
        max=float(vals.max()),
        high_dilution_warning=med > HIGH_OFF_DIAG_THRESHOLD,
    )


def eigenvalue_stats(
    eigvals_desc: np.ndarray, most_negative_raw: float
) -> EigenvalueStats:
    """Top-5 and bottom-5 eigenvalues from the descending-sorted clipped array."""
    top_5 = eigvals_desc[:5].tolist()
    bottom_5 = eigvals_desc[-5:][::-1].tolist()  # smallest first for readability
    return EigenvalueStats(
        top_5=[float(x) for x in top_5],
        bottom_5=[float(x) for x in bottom_5],
        most_negative_raw=float(most_negative_raw),
    )


def self_consistency_score(
    mesh_path: Path,
    n: int,
    seed: int,
    num_duplicates: int,
    rng_for_sigma: Optional[np.random.Generator] = None,
    sigma: Optional[float] = None,
    device: str = "cpu",
    chunk_size: int = DEFAULT_TORCH_CHUNK_SIZE,
) -> float:
    """Run the full pipeline on ``num_duplicates`` re-samplings of one mesh.

    Each duplicate is FPS-sampled with a *different* seed (seed + k for
    k = 0..K-1). This is intentional: identical seeds would yield byte-identical
    point clouds, making VS_1 trivially equal to 1 and the check vacuous. The
    point of the check is to verify that the kernel still says "same shape"
    *despite* FPS stochasticity. Do not "fix" this by reusing one seed.

    Args:
        mesh_path: Mesh to duplicate.
        n: Points per FPS sample.
        seed: Base seed; duplicate k uses ``seed + k``.
        num_duplicates: How many resamples to build the K from.
        rng_for_sigma: RNG used by the median heuristic. Required iff
            ``sigma`` is None.
        sigma: If provided, used as the kernel bandwidth and the median
            heuristic is skipped. Lets callers diagnose self-consistency at
            the same sigma the main pipeline used.
        device: Hardware backend for the kernel-matrix step
            (``"cpu"`` / ``"mps"`` / ``"cuda"``).
        chunk_size: GPU pair-batch size; ignored on CPU.

    Returns:
        The resulting VS_1. Should be close to 1 (within ~0.05) when sigma
        is at the scale of typical within-cloud distances.
    """
    if sigma is None and rng_for_sigma is None:
        raise ValueError(
            "self_consistency_score: provide either sigma or rng_for_sigma"
        )

    mesh = load_mesh(mesh_path)
    point_clouds: List[np.ndarray] = []
    for k in range(num_duplicates):
        # Distinct seed per duplicate -- see docstring.
        rng_k = np.random.default_rng(seed + k)
        pts = fps_sample(mesh, n, rng_k)
        point_clouds.append(normalize_to_unit_sphere(pts))

    sigma_value = (
        float(sigma)
        if sigma is not None
        else median_heuristic_sigma(point_clouds, rng_for_sigma)
    )
    _, K = build_kernel_matrices(
        point_clouds, sigma_value, device=device, chunk_size=chunk_size
    )
    result = compute_vendi_scores(K, qs=[1.0])
    return float(result.scores[1.0])


def select_self_consistency_mesh(mesh_paths: List[Path]) -> Optional[Path]:
    """Pick a mesh for the self-consistency check.

    Sort by name for determinism; pick the first.
    """
    if not mesh_paths:
        return None
    return sorted(mesh_paths)[0]
