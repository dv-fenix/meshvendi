"""Mesh surface sampling and point-cloud normalization.

Pipeline per mesh:
    1. Area-weighted random surface sampling to produce ``oversample_factor * n``
       candidate points (so FPS has a dense pool to pick from).
    2. Farthest-Point Sampling to pick ``n`` well-spread points.
    3. Center at centroid and rescale so the maximum centroid-to-point distance
       is 1 (unit bounding sphere).

All randomness flows through an explicit ``numpy.random.Generator`` so calls are
reproducible across processes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import trimesh

PathLike = Union[str, Path]


def load_mesh(path: PathLike) -> trimesh.Trimesh:
    """Load a mesh file as a single :class:`trimesh.Trimesh`.

    Concatenates multi-geometry scenes so callers always get a single watertight
    triangle soup. Supported formats follow trimesh: ``.obj``, ``.ply``, ``.stl``,
    ``.off``, etc.
    """
    obj = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if isinstance(obj, trimesh.Scene):
        merged = trimesh.util.concatenate(obj.dump())
        if isinstance(merged, trimesh.Trimesh):
            return merged
    raise ValueError(
        f"Could not load {path} as a triangle mesh (got {type(obj).__name__})"
    )


def _sample_surface(
    mesh: trimesh.Trimesh, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Area-weighted uniform random sampling of ``count`` points on the surface.

    Returns float64 array of shape ``(count, 3)``. We roll our own (rather than
    using :func:`trimesh.sample.sample_surface`) so the sampling is fully driven
    by the supplied ``rng`` and reproducible without touching numpy's globals.
    """
    triangles = mesh.triangles  # (F, 3, 3)
    areas = mesh.area_faces.astype(np.float64)
    if areas.sum() <= 0:
        raise ValueError("Mesh has zero surface area")
    probs = areas / areas.sum()

    face_idx = rng.choice(len(triangles), size=count, replace=True, p=probs)
    chosen = triangles[face_idx]  # (count, 3, 3)

    u = rng.random(count)
    v = rng.random(count)
    flip = u + v > 1.0
    u = np.where(flip, 1.0 - u, u)
    v = np.where(flip, 1.0 - v, v)
    w = 1.0 - u - v

    pts = (
        chosen[:, 0] * u[:, None]
        + chosen[:, 1] * v[:, None]
        + chosen[:, 2] * w[:, None]
    )
    return pts.astype(np.float64)


def _farthest_point_sampling(
    candidates: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Greedy farthest-point sampling: pick ``n`` points from ``candidates``.

    Standard greedy FPS: start from a random seed point, then iteratively pick
    whichever candidate maximizes its distance to the current selected set.
    Operates on squared distances (monotonic in true distance, no sqrt needed).
    """
    M = candidates.shape[0]
    if M < n:
        raise ValueError(f"FPS needs at least n={n} candidates, got {M}")

    selected = np.empty(n, dtype=np.int64)
    selected[0] = int(rng.integers(0, M))
    min_sq_dist = np.full(M, np.inf, dtype=np.float64)
    last = candidates[selected[0]]

    for i in range(1, n):
        diff = candidates - last
        sq = np.einsum("ij,ij->i", diff, diff)
        np.minimum(min_sq_dist, sq, out=min_sq_dist)
        nxt = int(np.argmax(min_sq_dist))
        selected[i] = nxt
        last = candidates[nxt]

    return candidates[selected]


def fps_sample(
    mesh: trimesh.Trimesh,
    n: int,
    rng: np.random.Generator,
    oversample_factor: int = 8,
) -> np.ndarray:
    """Sample ``n`` points on the mesh surface via FPS over a dense candidate pool.

    Returns ``(n, 3)`` float64 array. The candidate pool is ``oversample_factor *
    n`` area-weighted random surface points; FPS then picks ``n`` of them.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    n_candidates = max(n * oversample_factor, n)
    candidates = _sample_surface(mesh, n_candidates, rng)
    return _farthest_point_sampling(candidates, n, rng)


def normalize_to_unit_sphere(points: np.ndarray) -> np.ndarray:
    """Center at centroid and scale so the max centroid distance is 1.

    Standardizes scale and translation so the kernel is comparing shape, not
    overall size or position. Returns a new ``(n, 3)`` array.
    """
    centered = points - points.mean(axis=0, keepdims=True)
    max_radius = float(np.linalg.norm(centered, axis=1).max())
    if max_radius <= 0:
        raise ValueError("Point cloud has zero radius after centering")
    return centered / max_radius


def sample_and_normalize(
    mesh_path: PathLike,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Convenience wrapper: load, FPS, normalize. Returns ``(n, 3)`` float64."""
    mesh = load_mesh(mesh_path)
    pts = fps_sample(mesh, n, rng)
    return normalize_to_unit_sphere(pts)
