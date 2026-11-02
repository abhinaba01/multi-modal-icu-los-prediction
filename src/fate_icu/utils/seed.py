"""Seeding for reproducible runs (CLAUDE.md A4: 5 seeds for neural models)."""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_N_SEEDS = 5


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and, when installed, PyTorch.

    Args:
        seed: The seed.
        deterministic: Also request deterministic cuDNN / PyTorch kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    # Only affects subprocesses started after this call (hash randomisation is fixed
    # at interpreter start-up).
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def run_seeds(base_seed: int, n: int = DEFAULT_N_SEEDS) -> list[int]:
    """The ``n`` consecutive seeds a multi-seed run uses: ``base_seed .. base_seed + n - 1``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return [base_seed + i for i in range(n)]
