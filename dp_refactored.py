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
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import tensorflow as tf

# --------------------------------------------------------------------------------------
# Local differential privacy mechanisms
# --------------------------------------------------------------------------------------

def laplace(
    column: pd.Series | np.ndarray,
    epsilon: float = 1.0,
    sensitivity: float = 1.0,
    random_state: int | None = None,
) -> pd.Series | np.ndarray:
    """Apply Laplace mechanism to a numeric column.

    Args:
        column: 1-D numeric data.
        epsilon: Privacy budget controlling the scale of the noise.
        sensitivity: Query sensitivity.
        random_state: Seed for the random number generator.

    Returns:
        Noised column with the same type as the input.
    """
    scale = sensitivity / epsilon
    rng = np.random.default_rng(random_state)
    noise = rng.laplace(0.0, scale, size=len(column))
    noised = column.to_numpy() if isinstance(column, pd.Series) else np.asarray(column)
    noised = noised + noise
    return pd.Series(noised, index=column.index) if isinstance(column, pd.Series) else noised


def gaussian(
    column: pd.Series | np.ndarray,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    random_state: int | None = None,
) -> pd.Series | np.ndarray:
    """Apply Gaussian mechanism to a numeric column."""
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0.0, sigma, size=len(column))
    noised = column.to_numpy() if isinstance(column, pd.Series) else np.asarray(column)
    noised = noised + noise
    return pd.Series(noised, index=column.index) if isinstance(column, pd.Series) else noised


def exponential(
    column: pd.Series | np.ndarray,
    epsilon: float = 1.0,
    sensitivity: float = 1.0,
    random_state: int | None = None,
) -> pd.Series | np.ndarray:
    """Apply centred exponential noise to a numeric column."""
    scale = sensitivity / epsilon
    rng = np.random.default_rng(random_state)
    noise = rng.exponential(scale, size=len(column)) - scale
    noised = column.to_numpy() if isinstance(column, pd.Series) else np.asarray(column)
    noised = noised + noise
    return pd.Series(noised, index=column.index) if isinstance(column, pd.Series) else noised


def geometric(
    column: pd.Series | np.ndarray,
    epsilon: float = 1.0,
    sensitivity: float = 1.0,
    random_state: int | None = None,
) -> pd.Series | np.ndarray:
    """Apply two‑sided geometric noise to integer data."""
    p = 1 - np.exp(-epsilon / sensitivity)
    rng = np.random.default_rng(random_state)
    magnitudes = rng.geometric(p, size=len(column)) - 1
    signs = rng.choice([-1, 1], size=len(column))
    noise = signs * magnitudes
    noised = column.to_numpy() if isinstance(column, pd.Series) else np.asarray(column)
    noised = noised + noise
    return pd.Series(noised, index=column.index) if isinstance(column, pd.Series) else noised


def ldp_numeric(
    column: pd.Series | np.ndarray,
    mechanism: str = "laplace",
    random_state: int | None = None,
    **kwargs,
) -> pd.Series | np.ndarray:
    """Apply a numeric local DP mechanism.

    Args:
        column: Numeric data to perturb.
        mechanism: One of ``'laplace'``, ``'gaussian'``, ``'exponential'`` or ``'geometric'``.
        random_state: Seed for the RNG.
        **kwargs: Additional parameters passed to the mechanism.
    """
    mech = mechanism.lower()
    mechanisms = {
        "laplace": laplace,
        "gaussian": gaussian,
        "exponential": exponential,
        "geometric": geometric,
    }
    if mech not in mechanisms:
        raise ValueError(f"Unknown mechanism '{mechanism}'")
    return mechanisms[mech](column, random_state=random_state, **kwargs)


def randomised_response(
    series: pd.Series,
    truth_p: float = 0.7,
    random_state: int | None = None,
) -> pd.Series:
    """Randomised response for categorical data.

    Each value is reported truthfully with probability ``truth_p``; otherwise a
    random category from the observed domain is returned.
    """
    rng = np.random.default_rng(random_state)
    values = series.unique()
    rand = rng.random(len(series))
    random_response = rng.choice(values, size=len(series))
    result = np.where(rand < truth_p, series.to_numpy(), random_response)
    return pd.Series(result, index=series.index)


def apply_ldp(
    df: pd.DataFrame,
    numeric_cols: Iterable[str] | None = None,
    categorical_cols: Iterable[str] | None = None,
    numeric_mechanism: str = "laplace",
    truth_p: float = 0.7,
    random_state: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Apply local DP mechanisms to selected columns of ``df``.

    Numeric columns are split into floating and integer types.  Integer columns are
    detected with ``select_dtypes(include='integer')``.  When the numeric mechanism
    is ``'geometric'`` it is applied directly to these integer columns; otherwise
    the noised values are rounded and cast back to the original integer dtype.
    Floating columns are perturbed as-is using the specified numeric mechanism.

    Args:
        df: DataFrame to perturb.
        numeric_cols: Columns to apply numeric mechanism to.  If ``None``,
            numeric columns are automatically selected.
        categorical_cols: Columns to apply randomised response to.  If ``None``,
            categorical columns are automatically selected.
        numeric_mechanism: Mechanism name passed to :func:`ldp_numeric`.
        truth_p: Probability of reporting the true value for categorical columns.
        random_state: Seed for the RNG.
        **kwargs: Additional parameters forwarded to the numeric mechanism.

    Returns:
        A new ``DataFrame`` with perturbed columns.
    """
    result = df.copy()
    rng = np.random.default_rng(random_state)

    if numeric_cols is None:
        float_cols = result.select_dtypes(include="floating").columns
        int_cols = result.select_dtypes(include="integer").columns
    else:
        numeric_cols = list(numeric_cols)
        float_cols = [c for c in numeric_cols if pd.api.types.is_float_dtype(result[c])]
        int_cols = [c for c in numeric_cols if pd.api.types.is_integer_dtype(result[c])]

    if categorical_cols is None:
        categorical_cols = result.select_dtypes(include=["object", "category"]).columns

    for col in float_cols:
        col_seed = rng.integers(0, 2**32 - 1)
        result[col] = ldp_numeric(result[col], mechanism=numeric_mechanism, random_state=col_seed, **kwargs)

    for col in int_cols:
        col_seed = rng.integers(0, 2**32 - 1)
        noised = ldp_numeric(result[col], mechanism=numeric_mechanism, random_state=col_seed, **kwargs)
        if numeric_mechanism != "geometric":
            noised = noised.round().astype(result[col].dtype)
        else:
            noised = noised.astype(result[col].dtype)
        result[col] = noised

    for col in categorical_cols:
        col_seed = rng.integers(0, 2**32 - 1)
        result[col] = randomised_response(result[col], truth_p=truth_p, random_state=col_seed)
    return result


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
    # Apply noise to numeric columns
    numeric = df.select_dtypes(include=[np.number]).columns
    datasets['Laplace'] = apply_ldp(
        datasets['Laplace'], numeric_cols=numeric, categorical_cols=[],
        numeric_mechanism='laplace', random_state=random_state
    )
    datasets['Gaussian'] = apply_ldp(
        datasets['Gaussian'], numeric_cols=numeric, categorical_cols=[],
        numeric_mechanism='gaussian', random_state=random_state
    )
    datasets['Exponential'] = apply_ldp(
        datasets['Exponential'], numeric_cols=numeric, categorical_cols=[],
        numeric_mechanism='exponential', random_state=random_state
    )
    datasets['Geometric'] = apply_ldp(
        datasets['Geometric'], numeric_cols=numeric, categorical_cols=[],
        numeric_mechanism='geometric', random_state=random_state
    )
    # Randomised response for categorical columns (excluding target)
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c != 'smoker']
    datasets['RR'] = apply_ldp(
        datasets['RR'], numeric_cols=[], categorical_cols=cat_cols,
        truth_p=0.7, random_state=random_state
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
    df = pd.read_csv(args.data)
    df.dropna(inplace=True)
    res = run_pipeline(df, random_state=args.random_state)
    for key in sorted(res.keys()):
        r = res[key]
        print(f"{r.name}: acc={r.accuracy:.3f}, dpd={r.dpd:.3f}, eod={r.eod:.3f}")


if __name__ == '__main__':
    main()
