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


def randomized_response(
    series: pd.Series,
    epsilon: float = 1.0,
    random_state: int | None = None,
) -> pd.Series:
    """Apply randomized response to a binary categorical Series.

    Args:
        series: Binary series to privatize.
        epsilon: Privacy budget controlling flip probability.
        random_state: Seed for the random number generator.

    Returns:
        Privatized series with randomized response applied.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    values = series.dropna().unique()
    if len(values) != 2:
        raise ValueError("randomized_response expects a binary series")
    value_a, value_b = values
    rng = get_rng(random_state)
    prob_keep = np.exp(epsilon) / (np.exp(epsilon) + 1)
    keep_mask = rng.random(len(series)) < prob_keep
    flipped = series.copy()
    flip_values = np.where(flipped == value_a, value_b, value_a)
    flipped = flipped.where(keep_mask, flip_values)
    return flipped


def apply_randomized_response(
    df: pd.DataFrame,
    columns: list[str],
    epsilon: float = 1.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Apply randomized response to selected binary columns."""
    if not columns:
        return df.copy()
    rng = get_rng(random_state)
    noisy = df.copy()
    for column in columns:
        seed = None if random_state is None else int(rng.integers(0, 1_000_000))
        noisy[column] = randomized_response(noisy[column], epsilon=epsilon, random_state=seed)
    return noisy
