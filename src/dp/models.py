"""Model factories for the insurance dataset experiments."""

from __future__ import annotations

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .constants import RANDOM_STATE


def build_svm_model() -> SVC:
    """Build an RBF-kernel SVM with probability estimates enabled."""
    return SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)


def build_decision_tree_model() -> DecisionTreeClassifier:
    """Build a decision tree classifier."""
    return DecisionTreeClassifier(random_state=RANDOM_STATE)


def build_model_registry() -> dict[str, object]:
    """Return a dictionary of supported models for evaluation."""
    return {
        "svm": build_svm_model(),
        "decision_tree": build_decision_tree_model(),
    }
