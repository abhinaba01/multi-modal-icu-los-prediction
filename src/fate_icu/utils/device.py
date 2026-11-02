"""Runtime device choice: ``cuda`` when available, else ``cpu`` (CLAUDE.md A5)."""

from __future__ import annotations


def cuda_available() -> bool:
    """True when PyTorch is installed and sees a CUDA device."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def get_device(preference: str = "auto") -> str:
    """Resolve the compute device.

    Args:
        preference: ``auto`` (cuda if available, else cpu), ``cpu`` or ``cuda``.

    Returns:
        ``"cuda"`` or ``"cpu"``.

    Raises:
        ValueError: Unknown preference.
        RuntimeError: ``cuda`` requested but unavailable.
    """
    if preference not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unknown device preference: {preference!r}")
    if preference == "cpu":
        return "cpu"
    has_cuda = cuda_available()
    if preference == "cuda" and not has_cuda:
        raise RuntimeError("FATE_DEVICE=cuda but no CUDA device is available")
    return "cuda" if has_cuda else "cpu"
