import matplotlib
import numpy as np
import pandas as pd

from dp.evaluation import plot_roc_curves, privacy_utility_sweep
from dp.models import build_model_registry

matplotlib.use("Agg")


def _sample_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    size = 50
    return pd.DataFrame(
        {
            "age": rng.integers(18, 65, size=size),
            "bmi": rng.normal(30, 5, size=size),
            "sex": rng.choice(["male", "female"], size=size),
            "smoker": rng.choice(["yes", "no"], size=size),
        }
    )


def test_privacy_utility_sweep_runs():
    df = _sample_df()
    sweep = privacy_utility_sweep(
        df,
        target="smoker",
        epsilons=[0.5, 1.0],
        mechanism="laplace",
        random_state=0,
        models=build_model_registry(),
    )
    assert not sweep.results.empty
    assert {"model", "epsilon", "roc_auc"}.issubset(sweep.results.columns)
    assert set(sweep.roc_curves.keys()) == {"0.5", "1.0"}


def test_plot_roc_curves():
    df = _sample_df(seed=1)
    sweep = privacy_utility_sweep(
        df,
        target="smoker",
        epsilons=[0.5],
        mechanism="laplace",
        random_state=0,
        models=build_model_registry(),
    )
    fig = plot_roc_curves(sweep.roc_curves["0.5"], title="ROC")
    assert fig.axes
