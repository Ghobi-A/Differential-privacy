"""Evaluation utilities for privacy-utility analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .mechanisms import add_gaussian_noise, add_laplace_noise
from .models import build_model_registry
from .pipeline import DatasetSplit, build_preprocessor, split_dataset


@dataclass(frozen=True)
class SweepResult:
    results: pd.DataFrame
    roc_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]


def _encode_labels(y: pd.Series) -> np.ndarray:
    if y.dtype.kind in {"i", "u", "b", "f"}:
        return y.to_numpy()
    classes = sorted(y.unique())
    mapping = {label: idx for idx, label in enumerate(classes)}
    return y.map(mapping).to_numpy()


def _apply_noise(
    df: pd.DataFrame,
    mechanism: str,
    epsilon: float,
    delta: float,
    sensitivity: float,
    random_state: int | None,
) -> pd.DataFrame:
    if mechanism == "laplace":
        return add_laplace_noise(df, epsilon=epsilon, sensitivity=sensitivity, random_state=random_state)
    if mechanism == "gaussian":
        return add_gaussian_noise(
            df,
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            random_state=random_state,
        )
    raise ValueError("Only 'laplace' and 'gaussian' mechanisms are supported.")


def evaluate_models(
    split: DatasetSplit,
    mechanism: str,
    epsilon: float,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    random_state: int | None = None,
    models: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Train models on noised training data and return ROC-AUC scores."""
    models = models or build_model_registry()
    noisy_train = _apply_noise(
        split.X_train,
        mechanism=mechanism,
        epsilon=epsilon,
        delta=delta,
        sensitivity=sensitivity,
        random_state=random_state,
    )
    preprocessor = build_preprocessor(noisy_train)
    X_train = preprocessor.fit_transform(noisy_train)
    X_test = preprocessor.transform(split.X_test)
    y_train = _encode_labels(split.y_train)
    y_test = _encode_labels(split.y_test)

    rows: list[dict[str, float | str]] = []
    roc_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_scores = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_scores)
        rows.append({"model": name, "epsilon": epsilon, "roc_auc": roc_auc})
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_data[name] = (fpr, tpr)
    return pd.DataFrame(rows), roc_data


def privacy_utility_sweep(
    df: pd.DataFrame,
    target: str,
    epsilons: Iterable[float],
    mechanism: str = "laplace",
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    random_state: int | None = None,
    models: dict[str, object] | None = None,
) -> SweepResult:
    """Run a privacy-utility sweep across epsilon values."""
    split = split_dataset(df, target=target)
    all_rows: list[pd.DataFrame] = []
    roc_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for epsilon in epsilons:
        scores, roc_data = evaluate_models(
            split,
            mechanism=mechanism,
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            random_state=random_state,
            models=models,
        )
        all_rows.append(scores)
        roc_curves[str(epsilon)] = roc_data
    results = pd.concat(all_rows, ignore_index=True)
    return SweepResult(results=results, roc_curves=roc_curves)


def plot_privacy_utility(results: pd.DataFrame) -> plt.Figure:
    """Plot ROC-AUC versus epsilon for each model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name in results["model"].unique():
        subset = results[results["model"] == model_name]
        ax.plot(subset["epsilon"], subset["roc_auc"], marker="o", label=model_name)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Privacy–Utility Trade-off")
    ax.legend()
    ax.grid(True)
    return fig


def plot_roc_curves(
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "ROC Curves",
) -> plt.Figure:
    """Plot ROC curves for a single epsilon value."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, (fpr, tpr) in roc_curves.items():
        ax.plot(fpr, tpr, label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig
