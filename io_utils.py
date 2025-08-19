"""Utility helpers for I/O and column type inference."""
from __future__ import annotations

from typing import List, Tuple, Optional

import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(path)


def infer_columns(
    df: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
    num_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Infer categorical and numeric columns if not provided."""
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if num_cols is None:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return cat_cols, num_cols
