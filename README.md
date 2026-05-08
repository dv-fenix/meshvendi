# MeshVendi

Computes Vendi-Score-based diversity over a directory of mesh files. Each mesh
is reduced to a fixed-size point cloud via farthest-point sampling on its
surface, normalized to a unit bounding sphere, and compared to every other
mesh through a Gaussian kernel mean embedding. The eigenvalues of the
resulting normalized similarity matrix yield three Vendi Scores
(`VS_0.5`, `VS_1`, `VS_2`).

## Install

The module needs `numpy`, `scipy`, and `trimesh`. Optional `pytorch3d` is
detected at runtime but the current implementation uses the numpy/scipy path.

```
pip install numpy scipy trimesh
```

Python 3.10+.

## Run

```
python -m diversity --mesh-dir /path/to/meshes [--n 2048] [--seed 42] \
                     [--sigma 0.1] [--device auto] [--chunk-size 8] \
                     [--output results.json] \
                     [--no-diagnostics] [--quiet]
```

`--sigma` is optional. If omitted, sigma is selected by the median heuristic
over a random subsample of mesh pairs (default behavior). If supplied, the
median heuristic is skipped and the given value is used both for the main
kernel matrix and for the self-consistency diagnostic.

`--device` controls the backend for the kernel-matrix step (the dominant
cost). Options: `auto` (default, prefers CUDA → MPS → CPU), `cpu` (numpy),
`mps` (Apple Silicon GPU), `cuda` (NVIDIA GPU).

`--chunk-size` is the number of pair distance matrices held on the GPU at
once. The default of 8 keeps peak GPU residency at ~128 MB at n=2048,
chosen conservatively for Apple's unified memory. Raise it on a dedicated
CUDA card with spare VRAM.

Or import:

```python
from diversity import compute_diversity

# Default: median heuristic for sigma.
results = compute_diversity("/path/to/meshes", n=2048, seed=42)

# Or pin sigma explicitly:
results = compute_diversity("/path/to/meshes", n=2048, seed=42, sigma=0.1)
# {'VS_0.5': ..., 'VS_1': ..., 'VS_2': ..., 'sigma': 0.1,
#  'sigma_source': 'user', 'N': ..., 'off_diag_stats': {...},
#  'eigenvalue_stats': {...}, 'self_consistency_VS_1': ...}
```

## Method

For a set of `N` meshes:

1. **Sample.** Each mesh surface is presampled with area-weighted random
   sampling (`8 * n` candidates), then reduced to `n = 2048` points by greedy
   farthest-point sampling. Randomness is driven by an explicit
   `numpy.random.Generator`.
2. **Normalize.** Each point cloud is centered at its centroid and rescaled
   so the maximum centroid-to-point distance is 1 (unit bounding sphere).
3. **Kernel mean embedding.** For each mesh pair `(i, j)`,
   ```
   K_tilde(i, j) = (1 / n^2) sum_{a, b} exp(-||x_a - y_b||^2 / (2 sigma^2))
   ```
4. **Bandwidth.** `sigma` is the median pairwise distance over a random
   subsample of mesh pairs (50 ordered pairs, 1000 distances each, including
   diagonal pairs to avoid biasing `sigma` upward).
5. **Normalize matrix.** `K(i, j) = K_tilde(i, j) / sqrt(K_tilde(i, i) *
   K_tilde(j, j))`, so `K(i, i) = 1`.
6. **Vendi Score.** With eigenvalues `lambda_i` of `K / N`:
   - `VS_q = (sum lambda_i^q)^(1/(1-q))` for `q != 1`
   - `VS_1 = exp(-sum lambda_i log lambda_i)` (the limit as `q -> 1`)

## Interpreting `VS_q` at different `q`

All three are exponentiated Renyi entropies of the eigenvalue distribution.
They share the bounds `1 <= VS_q <= N`:

- **`VS_1`** is the balanced "effective number of distinct items." Most
  natural single number. Equivalent to `exp(Shannon entropy)`.
- **`VS_0.5`** weights small eigenvalues more than `VS_1`. Picks up rare /
  long-tail modes; tends to be larger than `VS_1` when there is a long tail
  of small modes.
- **`VS_2`** weights large eigenvalues more. Dominated by the top modes;
  tends to be smaller than `VS_1` when one or two modes dominate.

If `VS_0.5 ~= VS_1 ~= VS_2`, the eigenvalue distribution is flat and the set
is evenly diverse. A wide spread (`VS_0.5 >> VS_2`) indicates a small number
of dominant modes plus a long tail of minor variation.

## Diagnostics

`--no-diagnostics` suppresses these. By default they always run.

1. **Self-consistency.** One mesh is replicated `min(N, 50)` times, each
   replicate FPS-sampled with a *different* seed (`seed + k`). The resulting
   `VS_1` should be close to `1.0` (within ~0.05). If it is not, FPS noise is
   leaking into the metric: either the FPS seed is not being respected or `n`
   is too small for the mesh's geometric complexity.
2. **Off-diagonal range.** `[min, median, max]` of strict-upper-triangle
   entries of `K`. Healthy range is roughly `[0.3, 0.95]`. Median above
   `0.95` is the **dilution regime** — the kernel is failing to separate
   meshes and the headline number is unreliable; consider switching to a
   displacement-field kernel.
3. **Eigenvalue spectrum.** Top 5 and bottom 5 eigenvalues of `K/N`, plus
   the most negative raw eigenvalue. Anything below `-1e-6` is a PSD
   violation and warrants investigation.

## Layout

```
diversity/
    __init__.py        compute_diversity export
    __main__.py        python -m diversity entry
    sampling.py        load + FPS + normalize
    kernel.py          K_tilde, median heuristic, normalized K (cpu + torch)
    vendi.py           eigendecomp + VS_q
    diagnostics.py     three sanity checks
    device.py          cpu / mps / cuda selection
    cli.py             argparse + orchestrator
    README.md
tests/
    test_sampling.py
    test_kernel.py     includes torch-vs-numpy correctness check
    test_vendi.py
```
