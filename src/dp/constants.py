"""Shared constants and helpers for reproducibility."""

from __future__ import annotations

import numpy as np

RANDOM_STATE = 42


def get_rng(random_state: int | None = None) -> np.random.Generator:
    """Return a NumPy random generator seeded from the central RANDOM_STATE."""
    seed = RANDOM_STATE if random_state is None else random_state
    return np.random.default_rng(seed)
