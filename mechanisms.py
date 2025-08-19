"""Differential privacy mechanisms used by the CLI."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def add_laplace_noise(
    data: pd.DataFrame,
    epsilon: float,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Apply Laplace noise to numeric data."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / epsilon
    noise = rng.laplace(0.0, scale, size=data.shape)
    return data + noise


def add_gaussian_noise(
    data: pd.DataFrame,
    epsilon: float,
    delta: float,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Apply Gaussian noise using the analytic mechanism."""
    sigma = np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=data.shape)
    return data + noise


def add_exponential_noise(
    data: pd.DataFrame,
    epsilon: float,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Apply exponential noise to numeric data."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / epsilon
    noise = rng.exponential(scale, size=data.shape)
    return data + noise


def add_geometric_noise(
    data: pd.DataFrame,
    epsilon: float,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Apply geometric noise for integer data."""
    p = 1 - np.exp(-epsilon)
    rng = np.random.default_rng(seed)
    noise = rng.geometric(p, size=data.shape) - 1
    return data + noise


def randomised_response(
    series: pd.Series,
    truth_p: float,
    seed: Optional[int] = None,
) -> pd.Series:
    """Randomised response for categorical data."""
    rng = np.random.default_rng(seed)
    categories = series.unique()
    report_truth = rng.random(len(series)) < truth_p
    res = series.copy()
    res[~report_truth] = rng.choice(categories, size=(~report_truth).sum())
    return res
