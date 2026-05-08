"""Closed-form sanity checks for the Vendi Score formulas.

These exercise the math in :mod:`diversity.vendi` against three standard
limiting cases where the value of VS_q is known by hand:

  * ``K = I_N``     -> VS_q = N for every q (perfect diversity)
  * ``K = 1_{N,N}`` -> VS_q = 1 for every q (no diversity)
  * Block-diagonal of k all-ones blocks -> VS_1 = k
"""

from __future__ import annotations

import unittest

import numpy as np

from diversity.vendi import compute_vendi_scores

QS = (0.5, 1.0, 2.0)


class TestVendiClosedForms(unittest.TestCase):

    def test_identity_matrix_yields_N(self):
        N = 7
        K = np.eye(N)
        result = compute_vendi_scores(K, QS)
        for q in QS:
            with self.subTest(q=q):
                self.assertAlmostEqual(result.scores[q], float(N), places=8)

    def test_all_ones_matrix_yields_one(self):
        N = 9
        K = np.ones((N, N))
        result = compute_vendi_scores(K, QS)
        for q in QS:
            with self.subTest(q=q):
                # Tolerance loosened for q=0.5: eigvalsh produces N-1 eigenvalues
                # at O(eps), and VS_{q<1} amplifies them via lambda^q (~ sqrt(eps)
                # for q=0.5). 6 places still cleanly separates 1.0 from any real
                # diversity above ~1.000001.
                self.assertAlmostEqual(result.scores[q], 1.0, places=6)

    def test_block_diagonal_yields_block_count(self):
        # Build K with k all-ones blocks of size m on the diagonal.
        k = 4
        m = 3
        N = k * m
        K = np.zeros((N, N))
        for b in range(k):
            K[b * m : (b + 1) * m, b * m : (b + 1) * m] = 1.0

        result = compute_vendi_scores(K, [1.0])
        self.assertAlmostEqual(result.scores[1.0], float(k), places=8)

    def test_eigenvalues_sum_to_one(self):
        # K[i, i] == 1 means trace(K) == N, so eigenvalues of K/N sum to 1.
        rng = np.random.default_rng(0)
        N = 12
        A = rng.normal(size=(N, N))
        K = A @ A.T  # PSD
        K = K / np.sqrt(np.outer(np.diag(K), np.diag(K)))  # diag := 1
        result = compute_vendi_scores(K, QS)
        self.assertAlmostEqual(float(result.eigenvalues.sum()), 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
