from __future__ import annotations

import numpy as np

# Default configuration values for the differential privacy pipeline
EPSILON: float = 0.1
DELTA: float = 1e-5
SENSITIVITY: float = 1.0
TRUTH_P: float = 0.7
K: int = 5


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy random number generator initialised with ``seed``.

    Parameters
    ----------
    seed: int | None
        Seed for reproducibility. ``None`` yields an unpredictable generator.
    """
    return np.random.default_rng(seed)
