"""Device selection for the kernel-matrix backend.

The kernel-matrix step is the bottleneck of the pipeline. We have three
backends:

* ``"cpu"``   -- numpy + scipy. The default fallback, always available.
* ``"mps"``   -- PyTorch on Apple Silicon's Metal Performance Shaders.
* ``"cuda"``  -- PyTorch on an NVIDIA GPU.

Device selection is intentionally conservative on Apple Silicon because the
GPU shares unified memory with the CPU; oversized batches can trigger
system-wide memory pressure or kernel crashes. The chunk-size knob in
:mod:`diversity.kernel` controls how many pair distance matrices live on the
device at once.

Note that Apple's MPS backend does not support float64. The torch backend
therefore runs the hot path in float32 and casts back to float64 on the host
once each pair's scalar value is ready -- mean-of-non-negative-bounded values
is well-conditioned enough that float32 is plenty here.
"""

from __future__ import annotations

from typing import Literal

DeviceStr = Literal["auto", "cpu", "mps", "cuda"]


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def select_device(requested: DeviceStr = "auto") -> str:
    """Resolve a device request into a concrete backend string.

    Args:
        requested: One of ``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda"``.

    Returns:
        ``"cpu"``, ``"mps"``, or ``"cuda"``. Auto-selection prefers CUDA over
        MPS over CPU.

    Raises:
        RuntimeError: If a non-auto request cannot be honored (e.g. user asked
            for ``"cuda"`` on a machine without CUDA).
    """
    if requested == "cpu":
        return "cpu"

    if requested == "auto":
        if not _torch_available():
            return "cpu"
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # Explicit request for a GPU backend: validate.
    if not _torch_available():
        raise RuntimeError(
            f"device={requested!r} requested but PyTorch is not installed; "
            "install torch or pass --device cpu."
        )
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available")
        return "cuda"
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("device='mps' requested but MPS is not available")
        return "mps"

    raise ValueError(f"Unknown device {requested!r}")


def synchronize(device: str) -> None:
    """Block until pending ops on ``device`` finish.

    Used in timing code; safe no-op for cpu.
    """
    if device == "cpu":
        return
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
