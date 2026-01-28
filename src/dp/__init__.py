"""Differential privacy tooling for the insurance dataset."""

from .constants import RANDOM_STATE
from .evaluation import plot_privacy_utility, plot_roc_curves, privacy_utility_sweep
from .mechanisms import (
    add_gaussian_noise,
    add_laplace_noise,
    apply_randomized_response,
    randomized_response,
)
from .models import build_decision_tree_model, build_model_registry, build_svm_model
from .pipeline import load_dataset, preprocess_split, split_dataset

__all__ = [
    "RANDOM_STATE",
    "add_gaussian_noise",
    "add_laplace_noise",
    "apply_randomized_response",
    "build_decision_tree_model",
    "build_model_registry",
    "build_svm_model",
    "load_dataset",
    "plot_privacy_utility",
    "plot_roc_curves",
    "preprocess_split",
    "privacy_utility_sweep",
    "randomized_response",
    "split_dataset",
]
