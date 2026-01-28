"""Differential privacy noise mechanisms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import get_rng


def add_laplace_noise(
    data: pd.DataFrame,
    epsilon: float = 0.1,
    sensitivity: float = 1.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Add Laplace noise to numeric columns.

    Args:
        data: Data to perturb.
        epsilon: Privacy budget controlling the noise magnitude.
        sensitivity: Query sensitivity.
        random_state: Seed for the random number generator.

    Returns:
        DataFrame with Laplace noise added to numeric columns.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    scale = sensitivity / epsilon
    rng = get_rng(random_state)
    noisy = data.copy()
    num_cols = data.select_dtypes(include="number").columns
    if len(num_cols):
        noise = rng.laplace(0, scale, size=(len(data), len(num_cols)))
        noisy[num_cols] = data[num_cols] + noise
    return noisy


def add_gaussian_noise(
    data: pd.DataFrame,
    epsilon: float = 0.1,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Add Gaussian noise using the analytic Gaussian mechanism.

    Args:
        data: Data to perturb.
        epsilon: Privacy budget controlling the noise magnitude.
        delta: Probability of privacy breach in the Gaussian mechanism.
        sensitivity: Query sensitivity.
        random_state: Seed for the random number generator.

    Returns:
        DataFrame with Gaussian noise added to numeric columns.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be between 0 and 1")
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    rng = get_rng(random_state)
    noisy = data.copy()
    num_cols = data.select_dtypes(include="number").columns
    if len(num_cols):
        noise = rng.normal(0, sigma, size=(len(data), len(num_cols)))
        noisy[num_cols] = data[num_cols] + noise
    return noisy
