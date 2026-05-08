"""Geometric Mesh Diversity Metric.

Computes Vendi Score-based diversity over a directory of mesh files using a
Gaussian kernel mean embedding on FPS-sampled, scale-normalized point clouds.

Public entry point: ``compute_diversity``.
"""

from diversity.cli import compute_diversity

__all__ = ["compute_diversity"]
