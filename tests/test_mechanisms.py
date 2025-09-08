import numpy as np
import pandas as pd
import pandas.testing as pdt

from mechanisms import (
    add_laplace_noise,
    add_gaussian_noise,
    add_exponential_noise,
    add_geometric_noise,
    randomised_response,
)


def _check_noise(func):
    data = pd.DataFrame(np.zeros((10, 5)))
    out1 = func(data, random_state=0)
    out2 = func(data, random_state=0)
    assert out1.shape == data.shape
    pdt.assert_frame_equal(out1, out2)
    out3 = func(data, random_state=1)
    assert not out1.equals(out3)


def _check_noise_mixed(func, dtype=float, check_dtype=False):
    data = pd.DataFrame({"num": np.zeros(10, dtype=dtype), "cat": ["x"] * 10})
    out1 = func(data, random_state=0)
    # Non-numeric columns should remain unchanged
    pdt.assert_series_equal(out1["cat"], data["cat"])
    out2 = func(data, random_state=0)
    pdt.assert_frame_equal(out1, out2)
    out3 = func(data, random_state=1)
    assert not out1["num"].equals(out3["num"])
    if check_dtype:
        assert out1["num"].dtype == data["num"].dtype


def test_laplace_noise():
    _check_noise(add_laplace_noise)


def test_gaussian_noise():
    _check_noise(add_gaussian_noise)


def test_exponential_noise():
    _check_noise(add_exponential_noise)


def test_exponential_noise_centered():
    data = pd.DataFrame(np.zeros((10000, 1)))
    noisy = add_exponential_noise(data, epsilon=1.0, random_state=0)
    noise = (noisy - data).to_numpy().ravel()
    assert abs(noise.mean()) < 0.05


def test_geometric_noise():
    _check_noise(add_geometric_noise)


def test_geometric_noise_centered():
    data = pd.DataFrame(np.zeros((10000, 1), dtype=int))
    noisy = add_geometric_noise(data, epsilon=1.0, random_state=0)
    noise = (noisy - data).to_numpy().ravel()
    assert abs(noise.mean()) < 0.05
    # Integer dtypes should be preserved after adding noise
    assert noisy.dtypes.equals(data.dtypes)


def test_laplace_noise_mixed_types():
    _check_noise_mixed(add_laplace_noise)


def test_gaussian_noise_mixed_types():
    _check_noise_mixed(add_gaussian_noise)


def test_exponential_noise_mixed_types():
    _check_noise_mixed(add_exponential_noise)


def test_geometric_noise_mixed_types():
    _check_noise_mixed(add_geometric_noise, dtype=int, check_dtype=True)


def test_randomised_response():
    series = pd.Series(['a', 'b', 'c', 'a'])
    out1 = randomised_response(series, random_state=0)
    out2 = randomised_response(series, random_state=0)
    assert out1.shape == series.shape
    pdt.assert_series_equal(out1, out2)
    out3 = randomised_response(series, random_state=1)
    assert not out1.equals(out3)


def test_randomised_response_nan_untouched():
    series = pd.Series(["a", np.nan, "b", np.nan])
    out = randomised_response(series, p=0.0, random_state=0)
    # NaN positions should remain NaN
    assert out.isna().tolist() == series.isna().tolist()
    # Randomisation should only draw from non-NaN values
    assert out.dropna().isin(["a", "b"]).all()
