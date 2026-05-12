"""Global random seed for reproducibility.

Every module that needs randomness must call seed_everything() once at startup,
or import the pre-seeded RNG from this module.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

_SEED: int = 42
_SEEDED: bool = False


def seed_everything(seed: int = _SEED, *, force: bool = False) -> int:
    """Set global random seed across Python, NumPy, and environment.

    Idempotent by default — only seeds once per process unless `force=True`.
    Returns the seed used.
    """
    global _SEEDED, _SEED
    if _SEEDED and not force:
        return _SEED

    _SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    _SEEDED = True
    return seed


def get_seed() -> int:
    """Return the current seed value."""
    return _SEED


def get_rng() -> np.random.Generator:
    """Return a pre-seeded NumPy random generator."""
    return np.random.default_rng(_SEED)


def child_seed(offset: int = 0) -> int:
    """Derive a deterministic child seed from the master seed."""
    return _SEED + offset + 1
