"""
Differential Privacy Machine‑Learning Pipeline
---------------------------------------------

This module restructures the original exploratory notebook into a reusable, modular
pipeline.  Functions are provided to load and prepare the data, apply a variety of
differential‑privacy noise mechanisms, anonymise quasi‑identifiers, train several
classifiers and evaluate both predictive and fairness metrics.  The goal is to
illustrate best practices for organising code in a project while keeping the
scientific intent of the original work.

Usage:
    python dp_refactored.py --data insurance.csv --random-state 42

Dependencies are listed in requirements.txt.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)
import tensorflow as tf


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_numpy_seed(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy Generator seeded for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.default_rng(seed)


def infer_schema(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Infer categorical and numeric columns using pandas type checks."""
    cat_cols: list[str] = []
    num_cols: list[str] = []
    for col in df.columns:
        series = df[col]
        if is_numeric_dtype(series) and not is_bool_dtype(series):
            num_cols.append(col)
        elif is_categorical_dtype(series) or is_object_dtype(series) or is_bool_dtype(series):
            cat_cols.append(col)
    return cat_cols, num_cols


def validate_columns(
    df: pd.DataFrame,
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    strict: bool = True,
) -> Tuple[list[str], list[str]]:
    """Validate columns for NaNs and identifier-like patterns.

    Returns the categorical and numeric column lists used for validation.
    """
    if cat_cols is None or num_cols is None:
        inferred_cat, inferred_num = infer_schema(df)
        if cat_cols is None:
            cat_cols = inferred_cat
        if num_cols is None:
            num_cols = inferred_num

    issues = []
    if df[cat_cols + num_cols].isnull().any().any():
        issues.append("NaNs detected in data")

    suspicious = []
    for col in df.columns:
        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
        if unique_ratio > 0.95 or "id" in col.lower():
            suspicious.append(col)
    if suspicious:
        issues.append(f"Potential identifier columns: {suspicious}")

    if issues:
        msg = "; ".join(issues)
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    return cat_cols, num_cols


def read_csv_safe(path: str, strict: bool = True) -> pd.DataFrame:
    """Read a CSV file with validation and logging."""
    logger.info("Reading CSV from %s", path)
    df = pd.read_csv(path)
    validate_columns(df, strict=strict)
    return df


def write_csv_safe(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to CSV with logging."""
    logger.info("Writing CSV to %s", path)
    df.to_csv(path, index=False)

# --------------------------------------------------------------------------------------
# Noise mechanisms
# --------------------------------------------------------------------------------------

def add_laplace_noise(
    data: pd.DataFrame,
    epsilon: float = 0.1,
    sensitivity: float = 1.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Add Laplace noise to numeric columns.

    Args:
        data: DataFrame of numeric values.
        epsilon: Privacy budget; smaller values add more noise.
        sensitivity: Sensitivity of the query; defaults to 1.
        random_state: Seed for the random number generator.

    Returns:
        Noised DataFrame.
    """
    delta = sensitivity / epsilon
    rng = set_numpy_seed(random_state)
    noise = rng.laplace(0, delta, data.shape)
    return data + noise


def add_gaussian_noise(
    data: pd.DataFrame,
    epsilon: float = 0.1,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Add Gaussian noise using the analytic Gaussian mechanism.

    Args:
        random_state: Seed for the random number generator.
    """
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    rng = set_numpy_seed(random_state)
    noise = rng.normal(0, sigma, data.shape)
    return data + noise


def add_exponential_noise(
    data: pd.DataFrame,
    scale: float = 1.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Add exponential noise (Laplacian in L1 space).

    Args:
        random_state: Seed for the random number generator.
    """
    rng = set_numpy_seed(random_state)
    noise = rng.exponential(scale, data.shape)
    return data + noise


def add_geometric_noise(
    data: pd.DataFrame,
    epsilon: float = 0.1,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Add geometric noise for integer‑valued data.

    Args:
        random_state: Seed for the random number generator.
    """
    p = 1 - np.exp(-epsilon)
    rng = set_numpy_seed(random_state)
    noise = rng.geometric(p, size=data.shape) - 1
    return data + noise


def randomised_response(
    series: pd.Series,
    p: float = 0.7,
    random_state: int | None = None,
) -> pd.Series:
    """Apply randomised response to a categorical variable.

    Each value is reported truthfully with probability p; otherwise a random
    category is selected.

    Args:
        random_state: Seed for the random number generator.
    """
    values = series.unique()
    rng = set_numpy_seed(random_state)
    rand = rng.random(len(series))
    random_response = rng.choice(values, size=len(series))
    return pd.Series(np.where(rand < p, series, random_response), index=series.index)


# --------------------------------------------------------------------------------------
# Anonymisation
# --------------------------------------------------------------------------------------

def anonymise_dataset(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Generalise quasi‑identifiers to satisfy approximate k‑anonymity."""
    anon = df.copy()
    # Age: bucket into reasonable groups and collapse small bins
    max_age = anon['age'].max()
    age_bins = [0, 18, 25, 35, 45, 55, 65, max_age + 2]
    age_labels = [f'{age_bins[i]}-{age_bins[i+1]-1}' for i in range(len(age_bins)-1)]
    anon['age'] = pd.cut(anon['age'], bins=age_bins, right=False, labels=age_labels)
    anon['age'] = anon['age'].cat.add_categories('Other')
    counts = anon['age'].value_counts()
    anon.loc[anon['age'].isin(counts[counts < k].index), 'age'] = 'Other'
    # BMI: quantile binning into quartiles
    anon['bmi'] = pd.qcut(anon['bmi'], q=4, duplicates='drop')
    # Children: binary indicator
    anon['children'] = anon['children'].apply(lambda n: 'Has Children' if n > 0 else 'No Children')
    return anon


# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------

def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    """Prepare features and target for modelling.

    Returns design matrix X, target vector y and fitted ColumnTransformer.
    """
    X = df.drop('smoker', axis=1)
    y = df['smoker'].apply(lambda x: 1 if str(x).lower().strip() == 'yes' else 0)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        # Ignore unseen categories at transform time to keep feature space stable
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ])
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y.to_numpy(), preprocessor


# --------------------------------------------------------------------------------------
# Modelling and evaluation
# --------------------------------------------------------------------------------------

@dataclass
class ModelResult:
    name: str
    accuracy: float
    dpd: float  # Demographic parity difference
    eod: float  # Equalised odds difference


def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int | None = None,
) -> SVC:
    """Train an SVM classifier with reasonable defaults and basic hyper‑parameter search.

    Args:
        random_state: Seed for SMOTE and model initialisation.
    """
    param_grid = {
        'C': [1, 5, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    grid = GridSearchCV(SVC(random_state=random_state), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_res, y_res)
    return grid.best_estimator_


def train_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int | None = None,
) -> DecisionTreeClassifier:
    """Train a decision tree classifier.

    Args:
        random_state: Seed for SMOTE and model initialisation.
    """
    tree = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    tree.fit(X_res, y_res)
    return tree


def train_neural_network(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int | None = None,
) -> Sequential:
    """Train a simple feed‑forward neural network.

    Args:
        random_state: Seed for SMOTE and weight initialisation.
    """
    if random_state is not None:
        tf.random.set_seed(random_state)
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # For demonstration we keep epochs small; adjust as needed
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    model.fit(X_res, y_res, epochs=10, batch_size=32, verbose=0)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, sensitive: Iterable[int]) -> Tuple[float, float, float]:
    """Compute accuracy and fairness metrics on the test set."""
    if hasattr(model, 'predict_proba'):
        preds = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    else:
        preds = model.predict(X_test)
        preds = np.asarray(preds).ravel()
        # Convert non-binary outputs (e.g., probabilities) to class labels
        unique_vals = np.unique(preds)
        if not set(unique_vals).issubset({0, 1}):
            preds = (preds >= 0.5).astype(int)
        else:
            preds = preds.astype(int)
    accuracy = accuracy_score(y_test, preds)
    # Fairness: compute parity differences with respect to sensitive attribute indices
    dpd = demographic_parity_difference(y_test, preds, sensitive_features=sensitive)
    eod = equalized_odds_difference(y_test, preds, sensitive_features=sensitive)
    return accuracy, dpd, eod


def run_pipeline(df: pd.DataFrame, random_state: int | None = 42) -> Dict[str, ModelResult]:
    """Run noise mechanisms, train models and return metrics.

    Args:
        random_state: Seed controlling randomness throughout the pipeline.
    """
    set_numpy_seed(random_state)
    results = {}
    datasets: Dict[str, pd.DataFrame] = {
        'Original': df.copy(),
        'Laplace': df.copy(),
        'Gaussian': df.copy(),
        'Exponential': df.copy(),
        'Geometric': df.copy(),
        'RR': df.copy(),
        'Anonymised': anonymise_dataset(df.copy())
    }
    # Apply noise
    numeric = df.select_dtypes(include=[np.number]).columns
    datasets['Laplace'][numeric] = add_laplace_noise(
        datasets['Laplace'][numeric], random_state=random_state
    )
    datasets['Gaussian'][numeric] = add_gaussian_noise(
        datasets['Gaussian'][numeric], random_state=random_state
    )
    datasets['Exponential'][numeric] = add_exponential_noise(
        datasets['Exponential'][numeric], random_state=random_state
    )
    datasets['Geometric'][numeric] = add_geometric_noise(
        datasets['Geometric'][numeric], random_state=random_state
    )
    # Randomised response for categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col == 'smoker':
            continue
        datasets['RR'][col] = randomised_response(
            datasets['RR'][col], random_state=random_state
        )
    # Process each dataset
    for name, data in datasets.items():
        X, y, preproc = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=random_state
        )
        # Use sex column as sensitive feature for fairness evaluation.
        # Determine the index dynamically from the feature names to remain
        # robust to changes in the OneHotEncoder's layout.
        sensitive_col_index = None
        feature_names = preproc.get_feature_names_out()
        sex_indices = [i for i, name in enumerate(feature_names) if name.startswith('cat__sex_')]
        if sex_indices:
            sensitive_col_index = sex_indices[0]
        # Train models
        models = {
            'SVM': train_svm(X_train, y_train, random_state=random_state),
            'DecisionTree': train_decision_tree(X_train, y_train, random_state=random_state),
            'NeuralNetwork': train_neural_network(X_train, y_train, random_state=random_state),
        }
        sensitive_feature = (
            X_test[:, sensitive_col_index]
            if sensitive_col_index is not None
            else np.zeros(len(y_test))
        )
        for mname, model in models.items():
            acc, dpd, eod = evaluate_model(model, X_test, y_test, sensitive_feature)
            results[f'{name}_{mname}'] = ModelResult(
                name=f'{name}_{mname}', accuracy=acc, dpd=dpd, eod=eod
            )
    return results


def main():
    parser = argparse.ArgumentParser(description='Run differential privacy ML pipeline.')
    parser.add_argument('--data', type=str, default='insurance.csv', help='Path to the insurance CSV dataset.')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()
    df = read_csv_safe(args.data, strict=True)
    res = run_pipeline(df, random_state=args.random_state)
    for key in sorted(res.keys()):
        r = res[key]
        print(f"{r.name}: acc={r.accuracy:.3f}, dpd={r.dpd:.3f}, eod={r.eod:.3f}")


if __name__ == '__main__':
    main()
