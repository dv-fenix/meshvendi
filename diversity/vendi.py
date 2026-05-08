"""Vendi Score computation from a normalized similarity matrix.

Given a symmetric ``N x N`` similarity matrix ``K`` with ``K[i, i] == 1``, the
Vendi Score of order ``q`` is defined via the eigenvalues ``lambda_i`` of
``K / N``:

    VS_q = (sum_i lambda_i^q) ^ (1 / (1 - q))      for q != 1
    VS_1 = exp(-sum_i lambda_i * log(lambda_i))    (the q -> 1 limit)

These are exponentiated Renyi entropies of the eigenvalue distribution. They
range from 1 (every item is identical under the kernel) to N (every item is
mutually orthogonal).

Interpretation:
    - VS_0.5 weights small eigenvalues more: rare/long-tail modes count more.
    - VS_1 is the "balanced" version (Shannon entropy in the exponent).
    - VS_2 weights large eigenvalues more: dominated by the top modes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

NEG_EIGVAL_TOL = 1e-6


@dataclass(frozen=True)
class VendiResult:
    """Bundle of outputs from a single Vendi computation.

    Attributes:
        scores: ``{q: VS_q}`` for each requested order.
        eigenvalues: All eigenvalues of ``K / N`` (clipped to >= 0), descending.
        most_negative_eigenvalue: The most negative raw eigenvalue (pre-clipping),
            useful as a diagnostic of PSD violations.
    """

    scores: dict
    eigenvalues: np.ndarray
    most_negative_eigenvalue: float


def _vs_q(eigvals: np.ndarray, q: float) -> float:
    """Single VS_q from a clipped, non-negative eigenvalue array."""
    if abs(q - 1.0) < 1e-12:
        nz = eigvals[eigvals > 0]
        ent = -float(np.sum(nz * np.log(nz)))
        return float(np.exp(ent))
    s = float(np.sum(eigvals**q))
    if s <= 0:
        return 0.0
    return float(s ** (1.0 / (1.0 - q)))


def compute_vendi_scores(K: np.ndarray, qs: Iterable[float]) -> VendiResult:
    """Compute VS_q for each q from a normalized similarity matrix K.

    Args:
        K: ``(N, N)`` symmetric similarity matrix with ``K[i, i] == 1``.
        qs: Iterable of orders. ``1`` is handled via the entropy limit;
            ``q != 1`` uses the closed-form Renyi expression.

    Returns:
        :class:`VendiResult` containing the scores keyed by q, the clipped
        descending eigenvalue array, and the most negative raw eigenvalue (a
        diagnostic for numerical PSD violations).
    """
    N = K.shape[0]
    if K.shape != (N, N):
        raise ValueError(f"K must be square, got shape {K.shape}")

    # Symmetrize defensively to suppress floating-point asymmetry from kernel build.
    K_sym = 0.5 * (K + K.T)
    raw = np.linalg.eigvalsh(K_sym / N)
    most_neg = float(raw.min())

    if most_neg < -NEG_EIGVAL_TOL:
        warnings.warn(
            f"Largest negative eigenvalue is {most_neg:.3e} (tol {-NEG_EIGVAL_TOL}); "
            "the kernel matrix is not PSD beyond numerical noise.",
            RuntimeWarning,
            stacklevel=2,
        )

    eigvals = np.clip(raw, 0.0, None)
    eigvals_desc = np.sort(eigvals)[::-1]

    scores = {q: _vs_q(eigvals_desc, q) for q in qs}
    return VendiResult(
        scores=scores,
        eigenvalues=eigvals_desc,
        most_negative_eigenvalue=most_neg,
    )
