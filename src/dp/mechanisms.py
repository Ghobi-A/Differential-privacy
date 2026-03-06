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


def exponential_mechanism(
    candidates: list,
    utility_scores: np.ndarray,
    epsilon: float,
    sensitivity: float = 1.0,
    random_state: int | None = None,
):
    """Select a candidate using the exponential mechanism (ε-DP).

    The exponential mechanism samples from *candidates* with probability
    proportional to ``exp(epsilon * score / (2 * sensitivity))``, providing
    ε-differential privacy when *sensitivity* is the global L1 sensitivity of
    the utility function.

    Args:
        candidates: Ordered collection of outputs to select from.
        utility_scores: Array of real-valued utility scores, one per candidate.
            Higher scores indicate more preferred outputs.
        epsilon: Privacy budget.  Larger values favour high-utility candidates
            more strongly at the cost of weaker privacy.
        sensitivity: Global sensitivity of the utility function — the maximum
            change in any single candidate's score when one record in the
            database changes.  Defaults to 1.0.
        random_state: Seed for the random number generator.

    Returns:
        The selected element from *candidates*.

    Raises:
        ValueError: If *epsilon* or *sensitivity* are not positive, or if
            *candidates* and *utility_scores* have different lengths.

    Examples:
        >>> import numpy as np
        >>> from dp.mechanisms import exponential_mechanism
        >>> candidates = ["low", "medium", "high"]
        >>> scores = np.array([1.0, 5.0, 3.0])
        >>> exponential_mechanism(candidates, scores, epsilon=1.0, random_state=0)
        'medium'
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    scores = np.asarray(utility_scores, dtype=float)
    if len(candidates) != len(scores):
        raise ValueError(
            f"candidates and utility_scores must have the same length "
            f"(got {len(candidates)} and {len(scores)})"
        )
    if len(candidates) == 0:
        raise ValueError("candidates must not be empty")

    # Numerically stable: subtract max before exponentiation.
    log_weights = epsilon * scores / (2.0 * sensitivity)
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    probabilities = weights / weights.sum()

    rng = get_rng(random_state)
    index = rng.choice(len(candidates), p=probabilities)
    return candidates[index]
