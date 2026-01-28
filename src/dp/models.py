"""Model factories for the insurance dataset experiments."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .constants import RANDOM_STATE


def build_baseline_models() -> dict[str, object]:
    """Return a dictionary of baseline sklearn models."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "svm": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    }
