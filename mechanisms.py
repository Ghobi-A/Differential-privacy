import pandas as pd
import numpy as np


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
    scale = sensitivity / epsilon
    rng = np.random.default_rng(random_state)
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
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    rng = np.random.default_rng(random_state)
    noisy = data.copy()
    num_cols = data.select_dtypes(include="number").columns
    if len(num_cols):
        noise = rng.normal(0, sigma, size=(len(data), len(num_cols)))
        noisy[num_cols] = data[num_cols] + noise
    return noisy


def add_exponential_noise(
    data: pd.DataFrame,
    epsilon: float = 0.1,
    sensitivity: float = 1.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Add exponential noise to numeric columns.

    Args:
        data: Data to perturb.
        epsilon: Privacy budget controlling the noise magnitude.
        sensitivity: Query sensitivity.
        random_state: Seed for the random number generator.

    Returns:
        DataFrame with exponential noise added to numeric columns.
    """
    scale = sensitivity / epsilon
    rng = np.random.default_rng(random_state)
    noisy = data.copy()
    num_cols = data.select_dtypes(include="number").columns
    if len(num_cols):
        noise = rng.exponential(scale, size=(len(data), len(num_cols))) - scale
        noisy[num_cols] = data[num_cols] + noise
    return noisy


def add_geometric_noise(
    data: pd.DataFrame, epsilon: float = 0.1, random_state: int | None = None
) -> pd.DataFrame:
    """Add geometric noise for integer‑valued data.

    Args:
        data: Data to perturb.
        epsilon: Privacy budget controlling the noise magnitude.
        random_state: Seed for the random number generator.

    Returns:
        DataFrame with geometric noise added to integer‑valued columns.
    """
    p = 1 - np.exp(-epsilon)
    rng = np.random.default_rng(random_state)
    noisy = data.copy()
    num_cols = data.select_dtypes(include="number").columns
    if len(num_cols):
        noise = rng.geometric(p, size=(len(data), len(num_cols))) - rng.geometric(
            p, size=(len(data), len(num_cols))
        )
        noisy[num_cols] = data[num_cols] + noise

    # Preserve integer dtypes by rounding and casting back to the original type.
    int_cols = data.select_dtypes(include="integer").columns
    for col in int_cols:
        noisy[col] = noisy[col].round().astype(data[col].dtype)

    return noisy


def randomised_response(series: pd.Series, p: float = 0.7, random_state: int | None = None) -> pd.Series:
    """Apply randomised response to a categorical variable."""
    values = series.unique()
    rng = np.random.default_rng(random_state)
    rand = rng.random(len(series))
    random_response = rng.choice(values, size=len(series))
    return pd.Series(np.where(rand < p, series, random_response), index=series.index)
