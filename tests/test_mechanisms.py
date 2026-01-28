import numpy as np
import pandas as pd
import pandas.testing as pdt

from dp.mechanisms import add_gaussian_noise, add_laplace_noise


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


def test_laplace_noise_mixed_types():
    _check_noise_mixed(add_laplace_noise)


def test_gaussian_noise_mixed_types():
    _check_noise_mixed(add_gaussian_noise)
