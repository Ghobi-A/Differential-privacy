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
    """Apply randomised response to a categorical variable.

    NaN values are left untouched and excluded from the random-choice pool.
    """
    rng = np.random.default_rng(random_state)

    # Operate only on non-NaN entries so that NaNs remain untouched and
    # are not part of the random response pool.
    not_nan = series.notna()
    result = series.copy()
    if not not_nan.any():
        return result

    non_nan_series = series[not_nan]
    values = pd.unique(non_nan_series)
    if len(values) <= 1:
        return result

    rand = rng.random(len(non_nan_series))
    alternatives = {value: values[values != value] for value in values}
    random_response = np.array(
        [rng.choice(alternatives[value]) for value in non_nan_series.to_numpy()],
        dtype=values.dtype,
    )

    result[not_nan] = np.where(rand < p, non_nan_series, random_response)
    return result
