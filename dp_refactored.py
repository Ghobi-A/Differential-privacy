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
    python dp_refactored.py --data insurance.csv

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

# --------------------------------------------------------------------------------------
# Noise mechanisms
# --------------------------------------------------------------------------------------

def add_laplace_noise(data: pd.DataFrame, epsilon: float = 0.1, sensitivity: float = 1.0) -> pd.DataFrame:
    """Add Laplace noise to numeric columns.

    Args:
        data: DataFrame of numeric values.
        epsilon: Privacy budget; smaller values add more noise.
        sensitivity: Sensitivity of the query; defaults to 1.

    Returns:
        Noised DataFrame.
    """
    delta = sensitivity / epsilon
    noise = np.random.laplace(0, delta, data.shape)
    return data + noise


def add_gaussian_noise(data: pd.DataFrame, epsilon: float = 0.1, delta: float = 1e-5, sensitivity: float = 1.0) -> pd.DataFrame:
    """Add Gaussian noise using the analytic Gaussian mechanism."""
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise


def add_exponential_noise(data: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """Add exponential noise (Laplacian in L1 space)."""
    noise = np.random.exponential(scale, data.shape)
    return data + noise


def add_geometric_noise(data: pd.DataFrame, epsilon: float = 0.1) -> pd.DataFrame:
    """Add geometric noise for integer‑valued data."""
    p = 1 - np.exp(-epsilon)
    noise = np.random.geometric(p, size=data.shape) - 1
    return data + noise


def randomised_response(series: pd.Series, p: float = 0.7) -> pd.Series:
    """Apply randomised response to a categorical variable.

    Each value is reported truthfully with probability p; otherwise a random
    category is selected.
    """
    values = series.unique()
    rand = np.random.rand(len(series))
    random_response = np.random.choice(values, size=len(series))
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
        ('cat', OneHotEncoder(sparse_output=False), categorical_cols)
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


def train_svm(X: np.ndarray, y: np.ndarray) -> SVC:
    """Train an SVM classifier with reasonable defaults and basic hyper‑parameter search."""
    param_grid = {
        'C': [1, 5, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    grid = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_res, y_res)
    return grid.best_estimator_


def train_decision_tree(X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    tree.fit(X_res, y_res)
    return tree


def train_neural_network(X: np.ndarray, y: np.ndarray) -> Sequential:
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # For demonstration we keep epochs small; adjust as needed
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    model.fit(X_res, y_res, epochs=10, batch_size=32, verbose=0)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, sensitive: Iterable[int]) -> Tuple[float, float, float]:
    """Compute accuracy and fairness metrics on the test set."""
    if hasattr(model, 'predict_proba'):
        preds = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    else:
        preds = (model.predict(X_test) >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, preds)
    # Fairness: compute parity differences with respect to sensitive attribute indices
    dpd = demographic_parity_difference(y_test, preds, sensitive)
    eod = equalized_odds_difference(y_test, preds, sensitive)
    return accuracy, dpd, eod


def run_pipeline(df: pd.DataFrame) -> Dict[str, ModelResult]:
    """Run noise mechanisms, train models and return metrics."""
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
    datasets['Laplace'][numeric] = add_laplace_noise(datasets['Laplace'][numeric])
    datasets['Gaussian'][numeric] = add_gaussian_noise(datasets['Gaussian'][numeric])
    datasets['Exponential'][numeric] = add_exponential_noise(datasets['Exponential'][numeric])
    datasets['Geometric'][numeric] = add_geometric_noise(datasets['Geometric'][numeric])
    # Randomised response for categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        datasets['RR'][col] = randomised_response(datasets['RR'][col])
    # Process each dataset
    for name, data in datasets.items():
        X, y, preproc = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        # Use sex column as sensitive feature for fairness evaluation
        # Determine sensitive feature index after preprocessing
        sensitive_col_index = None
        # Preprocessor order: numeric_cols then categorical columns in alphabetical order
        # Sensitive attribute 'sex' will be encoded by OneHotEncoder; we take the first dummy as indicator
        # Build mask: number of numeric features + index of 'sex' dummy
        num_num = len(preproc.transformers_[0][2])
        cat_names = list(preproc.transformers_[1][1].get_feature_names_out(preproc.transformers_[1][2]))
        sex_indices = [i for i, cname in enumerate(cat_names) if cname.startswith('sex_')]
        if sex_indices:
            sensitive_col_index = num_num + sex_indices[0]
        # Train models
        models = {
            'SVM': train_svm(X_train, y_train),
            'DecisionTree': train_decision_tree(X_train, y_train),
            'NeuralNetwork': train_neural_network(X_train, y_train)
        }
        for mname, model in models.items():
            acc, dpd, eod = evaluate_model(model, X_test, y_test, X_test[:, sensitive_col_index])
            results[f'{name}_{mname}'] = ModelResult(name=f'{name}_{mname}', accuracy=acc, dpd=dpd, eod=eod)
    return results


def main():
    parser = argparse.ArgumentParser(description='Run differential privacy ML pipeline.')
    parser.add_argument('--data', type=str, default='insurance.csv', help='Path to the insurance CSV dataset.')
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    df.dropna(inplace=True)
    res = run_pipeline(df)
    for key in sorted(res.keys()):
        r = res[key]
        print(f"{r.name}: acc={r.accuracy:.3f}, dpd={r.dpd:.3f}, eod={r.eod:.3f}")


if __name__ == '__main__':
    main()
