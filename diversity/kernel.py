"""Gaussian kernel mean embedding and normalized similarity matrix.

For two point clouds ``P`` and ``Q`` with ``n`` points each:

    K_tilde(P, Q) = (1 / n^2) * sum_{a, b} exp(-||x_a - y_b||^2 / (2 * sigma^2))

This is the (unnormalized) inner product of the empirical kernel mean
embeddings under a Gaussian base kernel. We then normalize:

    K(i, j) = K_tilde(i, j) / sqrt(K_tilde(i, i) * K_tilde(j, j))

so K(i, i) = 1 by construction (the diagonal condition required by the
Vendi Score).

This module exposes a CPU implementation (numpy) and a GPU implementation
(PyTorch on MPS or CUDA). The CPU version is always built; the GPU version
is gated by torch availability and is reached via :func:`build_kernel_matrices`
when ``device != 'cpu'``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# Default number of pair distance matrices held on device at once.
# At n=2048 each pair is 16 MB float32, so the default keeps peak GPU
# residency around 128 MB -- conservative enough for Apple's unified memory.
DEFAULT_TORCH_CHUNK_SIZE = 8


def squared_distance_matrix(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances between rows of P and Q.

    Uses the identity ``||p - q||^2 = ||p||^2 + ||q||^2 - 2 p . q`` so the
    full ``(n, m)`` matrix is built with one matmul + two outer sums. Negative
    values from floating-point cancellation are clamped to 0.

    Args:
        P: ``(n, d)`` array.
        Q: ``(m, d)`` array.

    Returns:
        ``(n, m)`` float64 array of squared distances.
    """
    P_norm = np.einsum("ij,ij->i", P, P)[:, None]  # (n, 1)
    Q_norm = np.einsum("ij,ij->i", Q, Q)[None, :]  # (1, m)
    d2 = P_norm + Q_norm - 2.0 * (P @ Q.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def kernel_mean_embedding_inner_product(
    P: np.ndarray, Q: np.ndarray, sigma: float
) -> float:
    """Compute the unnormalized kernel mean embedding inner product K_tilde(P, Q).

    This is the (1 / n^2) sum of the Gaussian kernel evaluated at every cross-pair
    of points. Self-pairs (P == Q) are valid and produce K_tilde(P, P) >= 1/n
    (since the diagonal terms of the Gram matrix are exactly 1).
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    d2 = squared_distance_matrix(P, Q)
    gram = np.exp(-d2 / (2.0 * sigma * sigma))
    return float(gram.mean())


def median_heuristic_sigma(
    point_clouds: List[np.ndarray],
    rng: np.random.Generator,
    num_pairs: int = 50,
    dists_per_pair: int = 1000,
) -> float:
    """Pick sigma as the median pairwise distance over a random subsample of pairs.

    Standard median heuristic: sigma should sit in the middle of the typical
    distance scale so the kernel neither saturates at 1 (sigma too large) nor
    collapses to 0 (sigma too small).

    Implementation:
      - Sample ``num_pairs`` ordered mesh-pair indices ``(i, j)`` uniformly from
        all ``N^2`` pairs (diagonal included; this matches standard practice and
        avoids biasing sigma upward by excluding within-cloud distances).
      - For each sampled pair, compute the full point-by-point distance matrix
        and randomly subsample ``dists_per_pair`` distances from it (so memory
        stays bounded regardless of n).
      - Return the median of the concatenated samples.

    Args:
        point_clouds: List of ``(n, 3)`` arrays.
        rng: Numpy generator controlling pair selection and per-pair subsampling.
        num_pairs: Number of (i, j) pairs to draw. If ``N^2`` <= num_pairs, all
            pairs are used.
        dists_per_pair: Distance values to sample per pair (caps memory).

    Returns:
        Median pairwise distance, used as the Gaussian bandwidth sigma.
    """
    N = len(point_clouds)
    if N == 0:
        raise ValueError("Need at least one point cloud")

    total_pairs = N * N
    if total_pairs <= num_pairs:
        pair_idx = [(i, j) for i in range(N) for j in range(N)]
    else:
        flat = rng.choice(total_pairs, size=num_pairs, replace=False)
        pair_idx = [(int(idx // N), int(idx % N)) for idx in flat]

    samples: List[np.ndarray] = []
    for i, j in pair_idx:
        d2 = squared_distance_matrix(point_clouds[i], point_clouds[j])
        d = np.sqrt(d2).ravel()
        if d.size > dists_per_pair:
            sel = rng.choice(d.size, size=dists_per_pair, replace=False)
            d = d[sel]
        samples.append(d)

    all_samples = np.concatenate(samples)
    sigma = float(np.median(all_samples))
    if sigma <= 0:
        raise ValueError(
            "Median pairwise distance is 0; point clouds may be degenerate."
        )
    return sigma


def _build_kernel_matrices_numpy(
    point_clouds: List[np.ndarray], sigma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """CPU/numpy implementation of :func:`build_kernel_matrices`.

    The upper triangle (with diagonal) is computed pair by pair and mirrored.
    Each pairwise call materializes one ``n x n`` distance matrix at a time, so
    peak memory is bounded by a single pair regardless of N.
    """
    N = len(point_clouds)
    K_tilde = np.zeros((N, N), dtype=np.float64)
    inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)

    for i in range(N):
        for j in range(i, N):
            d2 = squared_distance_matrix(point_clouds[i], point_clouds[j])
            val = float(np.exp(-d2 * inv_two_sigma_sq).mean())
            K_tilde[i, j] = val
            K_tilde[j, i] = val

    diag = np.diag(K_tilde)
    if np.any(diag <= 0):
        raise ValueError("K_tilde diagonal contains non-positive entries")
    K = K_tilde / np.sqrt(np.outer(diag, diag))
    return K_tilde, K


def _build_kernel_matrices_torch(
    point_clouds: List[np.ndarray],
    sigma: float,
    device: str,
    chunk_size: int = DEFAULT_TORCH_CHUNK_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """GPU implementation using PyTorch on MPS or CUDA.

    Strategy:
      - Stack all point clouds onto the device once, in float32.
      - For each row i, compute K_tilde[i, j] for j >= i in chunks of size
        ``chunk_size``. Each chunk holds a ``(B, n, n)`` distance matrix on
        device, so peak GPU residency is bounded by a single chunk regardless
        of N.
      - Cast each scalar K_tilde value to float64 on the host. Mean of bounded
        non-negative values is well-conditioned in float32, so the kernel mean
        is accurate to ~1e-6 relative; final eigvalsh runs on float64.

    MPS does not support float64; we therefore force float32 on the device
    and cast on the way out. CUDA could use float64 but float32 is fine for
    the same reason and roughly doubles throughput.

    Args:
        point_clouds: ``N`` arrays each of shape ``(n, 3)``.
        sigma: Gaussian bandwidth.
        device: ``"mps"`` or ``"cuda"``.
        chunk_size: Pairs processed at once. Default 8 keeps peak GPU
            residency around 128 MB at n=2048. Tune up for CUDA cards with
            spare VRAM.

    Returns:
        ``(K_tilde, K)`` float64 numpy arrays of shape ``(N, N)``.
    """
    import torch  # local import: only required when this path is taken

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    N = len(point_clouds)
    if N == 0:
        raise ValueError("Need at least one point cloud")

    # All clouds onto the device, float32. (N, n, 3) is small even for N=1000.
    stacked_np = np.stack(point_clouds).astype(np.float32, copy=False)
    clouds = torch.from_numpy(stacked_np).to(device)
    norms = (clouds * clouds).sum(dim=-1)  # (N, n)

    inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)
    K_tilde = np.zeros((N, N), dtype=np.float64)

    with torch.no_grad():
        for i in range(N):
            Pi = clouds[i : i + 1]  # (1, n, 3)
            Pi_norm = norms[i : i + 1].unsqueeze(-1)  # (1, n, 1)

            for jstart in range(i, N, chunk_size):
                jend = min(jstart + chunk_size, N)
                B = jend - jstart

                Qj = clouds[jstart:jend]  # (B, n, 3)
                Qj_norm = norms[jstart:jend].unsqueeze(1)  # (B, 1, n)

                Pe = Pi.expand(B, -1, -1)
                cross = torch.bmm(Pe, Qj.transpose(-1, -2))  # (B, n, n)
                d2 = (Pi_norm + Qj_norm - 2.0 * cross).clamp_min_(0.0)
                kij = torch.exp(-d2 * inv_two_sigma_sq).mean(dim=(1, 2))  # (B,)

                kij_host = kij.detach().to("cpu").numpy().astype(np.float64)
                K_tilde[i, jstart:jend] = kij_host
                # Mirror the upper-triangle entry into the lower triangle.
                K_tilde[jstart:jend, i] = kij_host

                # Help the allocator promptly release each chunk.
                del Qj, Qj_norm, Pe, cross, d2, kij

    # Free device tensors before returning.
    del clouds, norms
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    diag = np.diag(K_tilde)
    if np.any(diag <= 0):
        raise ValueError("K_tilde diagonal contains non-positive entries")
    K = K_tilde / np.sqrt(np.outer(diag, diag))
    return K_tilde, K


def build_kernel_matrices(
    point_clouds: List[np.ndarray],
    sigma: float,
    device: str = "cpu",
    chunk_size: int = DEFAULT_TORCH_CHUNK_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build both K_tilde (unnormalized) and K (normalized) for a set of clouds.

    Dispatches to a numpy CPU implementation or a PyTorch GPU implementation
    based on ``device``.

    Args:
        point_clouds: List of ``N`` arrays each of shape ``(n, 3)``.
        sigma: Gaussian bandwidth.
        device: ``"cpu"`` (numpy), ``"mps"`` (Apple GPU), or ``"cuda"``
            (NVIDIA GPU). Default ``"cpu"``.
        chunk_size: Pairs processed simultaneously on device. Ignored for
            ``"cpu"``. Default 8 keeps peak GPU residency at ~128 MB for
            n=2048.

    Returns:
        ``(K_tilde, K)`` where both are ``(N, N)`` float64. ``K[i, i] == 1``.
    """
    if device == "cpu":
        return _build_kernel_matrices_numpy(point_clouds, sigma)
    return _build_kernel_matrices_torch(
        point_clouds, sigma, device=device, chunk_size=chunk_size
    )
