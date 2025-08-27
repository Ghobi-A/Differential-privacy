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


def test_laplace_noise():
    _check_noise(add_laplace_noise)


def test_gaussian_noise():
    _check_noise(add_gaussian_noise)


def test_exponential_noise():
    _check_noise(add_exponential_noise)


def test_geometric_noise():
    _check_noise(add_geometric_noise)


def test_geometric_noise_centered():
    data = pd.DataFrame(np.zeros((10000, 1), dtype=int))
    noisy = add_geometric_noise(data, epsilon=1.0, random_state=0)
    noise = (noisy - data).to_numpy().ravel()
    assert abs(noise.mean()) < 0.05
    # Integer dtypes should be preserved after adding noise
    assert noisy.dtypes.equals(data.dtypes)


def test_randomised_response():
    series = pd.Series(['a', 'b', 'c', 'a'])
    out1 = randomised_response(series, random_state=0)
    out2 = randomised_response(series, random_state=0)
    assert out1.shape == series.shape
    pdt.assert_series_equal(out1, out2)
    out3 = randomised_response(series, random_state=1)
    assert not out1.equals(out3)
