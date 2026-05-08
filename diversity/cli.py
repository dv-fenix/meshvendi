"""CLI and top-level orchestrator for the diversity metric.

Public function :func:`compute_diversity` runs the full pipeline and returns
a results dictionary. :func:`main` is the argparse entry point used by
``python -m diversity``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from diversity.device import select_device
from diversity.diagnostics import (eigenvalue_stats, off_diagonal_stats,
                                   select_self_consistency_mesh,
                                   self_consistency_score)
from diversity.kernel import (DEFAULT_TORCH_CHUNK_SIZE, build_kernel_matrices,
                              median_heuristic_sigma)
from diversity.sampling import fps_sample, load_mesh, normalize_to_unit_sphere
from diversity.vendi import compute_vendi_scores

SUPPORTED_EXTENSIONS = {".obj", ".ply", ".stl", ".off"}
DEFAULT_QS = (0.5, 1.0, 2.0)
SELF_CONSISTENCY_MAX_DUPLICATES = 50


def discover_meshes(mesh_dir: Path) -> List[Path]:
    """Return a sorted list of mesh files in ``mesh_dir`` (non-recursive)."""
    if not mesh_dir.is_dir():
        raise NotADirectoryError(f"{mesh_dir} is not a directory")
    paths = [
        p
        for p in mesh_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(paths)


def _format_q_key(q: float) -> str:
    """Map 0.5 -> 'VS_0.5', 1.0 -> 'VS_1', 2.0 -> 'VS_2'."""
    if abs(q - round(q)) < 1e-12:
        return f"VS_{int(round(q))}"
    return f"VS_{q}"


def compute_diversity(
    mesh_dir: str | Path,
    n: int = 2048,
    seed: int = 42,
    run_diagnostics: bool = True,
    qs: tuple = DEFAULT_QS,
    verbose: bool = False,
    sigma: Optional[float] = None,
    device: str = "auto",
    chunk_size: int = DEFAULT_TORCH_CHUNK_SIZE,
) -> Dict[str, Any]:
    """Run the full pipeline on a directory of meshes and return results.

    Args:
        mesh_dir: Directory containing ``.obj``/``.ply``/``.stl``/``.off`` files.
        n: Points per mesh after FPS (default 2048).
        seed: Master seed for all randomness (default 42).
        run_diagnostics: If True, run the three sanity-check diagnostics
            (adds the cost of one extra small pipeline pass on a duplicated mesh).
        qs: Vendi orders to compute (default ``(0.5, 1.0, 2.0)``).
        verbose: Print per-step progress to stderr.
        sigma: If provided, use this Gaussian bandwidth and skip the median
            heuristic. The same value is used for the self-consistency check
            so the diagnostic reports on the kernel actually being used. If
            ``None`` (default), sigma is selected via the median heuristic.
        device: Hardware backend for the kernel matrix step. One of
            ``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda"``. ``"auto"`` prefers
            CUDA, then MPS, then CPU.
        chunk_size: GPU pair-batch size; ignored on CPU. Default 8 keeps peak
            device residency at ~128 MB for n=2048 -- safe for Apple unified
            memory. Tune up on dedicated CUDA cards.

    Returns:
        Dict with keys: ``VS_0.5``, ``VS_1``, ``VS_2``, ``sigma``,
        ``sigma_source`` (``"user"`` or ``"median_heuristic"``), ``N``,
        ``n``, ``seed``, ``device``, plus diagnostic keys ``off_diag_stats``,
        ``eigenvalue_stats``, ``self_consistency_VS_1`` when
        ``run_diagnostics=True``.
    """
    if sigma is not None and sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    resolved_device = select_device(device)
    if verbose and resolved_device != "cpu":
        print(
            f"[device] kernel-matrix backend = {resolved_device} "
            f"(chunk_size={chunk_size})",
            file=sys.stderr,
        )
    mesh_dir = Path(mesh_dir)
    mesh_paths = discover_meshes(mesh_dir)
    N = len(mesh_paths)
    if N < 2:
        raise ValueError(f"Need at least 2 meshes in {mesh_dir}; found {N}.")

    # Spawn independent RNG streams from the master seed so different stages
    # (per-mesh sampling, sigma sampling, self-consistency) cannot collide.
    master_ss = np.random.SeedSequence(seed)
    sampling_ss, sigma_ss, sc_ss = master_ss.spawn(3)

    # 1) Sample + normalize each mesh.
    if verbose:
        print(f"[1/4] Sampling {N} meshes (n={n} per mesh)...", file=sys.stderr)
    t0 = time.time()
    mesh_seeds = sampling_ss.spawn(N)
    point_clouds: List[np.ndarray] = []
    for idx, (path, ss) in enumerate(zip(mesh_paths, mesh_seeds)):
        rng = np.random.default_rng(ss)
        mesh = load_mesh(path)
        pts = fps_sample(mesh, n, rng)
        pts = normalize_to_unit_sphere(pts)
        point_clouds.append(pts)
        if verbose:
            print(
                f"      [{idx + 1}/{N}] {path.name} ({time.time() - t0:.1f}s elapsed)",
                file=sys.stderr,
            )

    # 2) Bandwidth: user-specified or median heuristic.
    if sigma is None:
        if verbose:
            print("[2/4] Median-heuristic bandwidth selection...", file=sys.stderr)
        sigma_rng = np.random.default_rng(sigma_ss)
        sigma_value = median_heuristic_sigma(point_clouds, sigma_rng)
        sigma_source = "median_heuristic"
    else:
        if verbose:
            print(
                f"[2/4] Using user-supplied sigma (skipping median heuristic)...",
                file=sys.stderr,
            )
        sigma_value = float(sigma)
        sigma_source = "user"
    if verbose:
        print(f"      sigma = {sigma_value:.6f} ({sigma_source})", file=sys.stderr)

    # 3) Build kernel matrix (this is the expensive step).
    if verbose:
        n_pairs = N * (N + 1) // 2
        print(
            f"[3/4] Building {N}x{N} kernel matrix ({n_pairs} pair evaluations)...",
            file=sys.stderr,
        )
    t1 = time.time()
    _, K = build_kernel_matrices(
        point_clouds, sigma_value, device=resolved_device, chunk_size=chunk_size
    )
    if verbose:
        print(f"      done in {time.time() - t1:.1f}s", file=sys.stderr)

    # 4) Vendi scores.
    if verbose:
        print("[4/4] Eigendecomposition + Vendi scores...", file=sys.stderr)
    vendi = compute_vendi_scores(K, qs)

    results: Dict[str, Any] = {_format_q_key(q): float(vendi.scores[q]) for q in qs}
    results["sigma"] = float(sigma_value)
    results["sigma_source"] = sigma_source
    results["N"] = N
    results["n"] = n
    results["seed"] = seed
    results["device"] = resolved_device

    if run_diagnostics:
        if verbose:
            print("[diag] Off-diagonal range...", file=sys.stderr)
        od = off_diagonal_stats(K)
        results["off_diag_stats"] = asdict(od)

        if verbose:
            print("[diag] Eigenvalue spectrum...", file=sys.stderr)
        es = eigenvalue_stats(vendi.eigenvalues, vendi.most_negative_eigenvalue)
        results["eigenvalue_stats"] = asdict(es)

        if verbose:
            print(
                "[diag] Self-consistency (this re-runs sampling+kernel on duplicates)...",
                file=sys.stderr,
            )
        sc_mesh = select_self_consistency_mesh(mesh_paths)
        num_dup = min(N, SELF_CONSISTENCY_MAX_DUPLICATES)
        # Use a high seed offset that won't collide with per-mesh seeds.
        sc_seed = int(np.random.default_rng(sc_ss).integers(0, 2**31 - 1))
        # When the user fixed sigma, propagate it to the self-consistency check
        # so the diagnostic reports on the kernel actually being used. Otherwise
        # let self-consistency run its own median heuristic on the duplicates.
        sc_sigma_override = sigma_value if sigma_source == "user" else None
        sc_sigma_rng = (
            np.random.default_rng(sc_ss.spawn(1)[0])
            if sc_sigma_override is None
            else None
        )
        results["self_consistency"] = {
            "mesh": sc_mesh.name if sc_mesh is not None else None,
            "num_duplicates": num_dup,
            "VS_1": (
                self_consistency_score(
                    sc_mesh,
                    n,
                    sc_seed,
                    num_dup,
                    rng_for_sigma=sc_sigma_rng,
                    sigma=sc_sigma_override,
                    device=resolved_device,
                    chunk_size=chunk_size,
                )
                if sc_mesh is not None
                else None
            ),
        }
        # Spec requested top-level key:
        results["self_consistency_VS_1"] = results["self_consistency"]["VS_1"]

    return results


def _format_human(results: Dict[str, Any]) -> str:
    """Pretty-print the results dict for stdout."""
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("Geometric Mesh Diversity Metric")
    lines.append("=" * 60)
    lines.append(f"N (mesh count)        : {results['N']}")
    lines.append(f"n (points per mesh)   : {results['n']}")
    lines.append(f"seed                  : {results['seed']}")
    if "device" in results:
        lines.append(f"device                : {results['device']}")
    sigma_label = {
        "median_heuristic": "median heur.",
        "user": "user-supplied",
    }.get(results.get("sigma_source", "median_heuristic"), "median heur.")
    lines.append(f"sigma ({sigma_label:13}): {results['sigma']:.6f}")
    lines.append("-" * 60)
    for key in ("VS_0.5", "VS_1", "VS_2"):
        if key in results:
            lines.append(f"{key:22}: {results[key]:.6f}")

    if "off_diag_stats" in results:
        lines.append("-" * 60)
        lines.append("Diagnostics")
        lines.append("-" * 60)
        od = results["off_diag_stats"]
        lines.append(
            f"K off-diag [min, median, max]: "
            f"[{od['min']:.4f}, {od['median']:.4f}, {od['max']:.4f}]"
        )
        if od["high_dilution_warning"]:
            lines.append(
                "  WARNING: median off-diagonal > 0.95 -- kernel is in the "
                "dilution regime; the metric is losing discriminative power. "
                "Consider switching to a displacement-field kernel."
            )

        es = results["eigenvalue_stats"]
        lines.append(
            f"Top 5 eigenvalues of K/N    : " f"{[f'{x:.4f}' for x in es['top_5']]}"
        )
        lines.append(
            f"Bottom 5 eigenvalues of K/N : " f"{[f'{x:.4f}' for x in es['bottom_5']]}"
        )
        lines.append(f"Most negative raw eigenvalue: {es['most_negative_raw']:.3e}")
        if es["most_negative_raw"] < -1e-6:
            lines.append(
                "  WARNING: most negative eigenvalue exceeds -1e-6; "
                "PSD violation beyond numerical noise."
            )

        sc = results.get("self_consistency")
        if sc is not None and sc.get("VS_1") is not None:
            lines.append(
                f"Self-consistency VS_1       : {sc['VS_1']:.6f} "
                f"(target ~1.0 +- 0.05; {sc['num_duplicates']} duplicates of {sc['mesh']!r})"
            )
            if abs(sc["VS_1"] - 1.0) > 0.05:
                lines.append(
                    "  WARNING: self-consistency check deviates from 1.0 by > 0.05; "
                    "FPS stochasticity may be leaking. Check seed handling and n."
                )
    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """argparse entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m diversity",
        description=(
            "Compute Vendi-Score diversity (VS_0.5, VS_1, VS_2) over a "
            "directory of meshes via Gaussian kernel mean embeddings on "
            "FPS-sampled point clouds."
        ),
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        required=True,
        help="Directory containing .obj/.ply/.stl/.off files.",
    )
    parser.add_argument(
        "--n", type=int, default=2048, help="Points per mesh after FPS (default 2048)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Master random seed (default 42)."
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Gaussian bandwidth. If omitted, sigma is selected "
        "by the median heuristic over a random subsample of "
        "mesh pairs (default behavior).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for a JSON dump of the results dict.",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Skip self-consistency, off-diag, and eigenvalue diagnostics.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-step progress on stderr."
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Hardware backend for the kernel-matrix step. "
        "'auto' picks CUDA > MPS > CPU. (default: auto)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_TORCH_CHUNK_SIZE,
        help=f"GPU pair-batch size (ignored on CPU). "
        f"Default {DEFAULT_TORCH_CHUNK_SIZE} keeps peak "
        f"GPU residency at ~128 MB for n=2048; raise on "
        f"dedicated CUDA cards with spare VRAM.",
    )
    args = parser.parse_args(argv)

    results = compute_diversity(
        mesh_dir=args.mesh_dir,
        n=args.n,
        seed=args.seed,
        run_diagnostics=not args.no_diagnostics,
        verbose=not args.quiet,
        sigma=args.sigma,
        device=args.device,
        chunk_size=args.chunk_size,
    )

    print(_format_human(results))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(results, f, indent=2)
        if not args.quiet:
            print(f"Wrote {args.output}", file=sys.stderr)

    return 0
