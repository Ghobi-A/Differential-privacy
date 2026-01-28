"""Training pipeline with leakage-safe preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import RANDOM_STATE
from .mechanisms import add_gaussian_noise, add_laplace_noise


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the insurance dataset from a repo-root-relative path."""
    return pd.read_csv(Path(path))


def split_dataset(
    df: pd.DataFrame,
    target: str = "smoker",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> DatasetSplit:
    """Split the dataset into train/test partitions."""
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return DatasetSplit(X_train, X_test, y_train, y_test)


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing transformer for numeric/categorical columns."""
    numeric_features = df.select_dtypes(include="number").columns
    categorical_features = df.select_dtypes(exclude="number").columns
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )


def apply_feature_noise(
    df: pd.DataFrame,
    mechanism: str = "laplace",
    epsilon: float = 0.1,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Apply feature-level noise to numeric columns only."""
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")

    if mechanism == "laplace":
        noisy_num = add_laplace_noise(
            numeric,
            epsilon=epsilon,
            sensitivity=sensitivity,
            random_state=random_state,
        )
    elif mechanism == "gaussian":
        noisy_num = add_gaussian_noise(
            numeric,
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            random_state=random_state,
        )
    else:
        raise ValueError("Only 'laplace' and 'gaussian' mechanisms are supported.")

    return pd.concat([noisy_num, categorical], axis=1)[df.columns]


def preprocess_split(split: DatasetSplit) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Fit preprocessing on train only, then transform train/test."""
    preprocessor = build_preprocessor(split.X_train)
    X_train = preprocessor.fit_transform(split.X_train)
    X_test = preprocessor.transform(split.X_test)
    return X_train, X_test, preprocessor
