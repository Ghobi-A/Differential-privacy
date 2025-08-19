"""Anonymisation helpers such as k-anonymity checks."""
from __future__ import annotations

from typing import List

import pandas as pd


def check_k_anonymity(df: pd.DataFrame, cat_cols: List[str], k: int) -> bool:
    """Return True if every group defined by ``cat_cols`` has at least ``k`` rows."""
    if not cat_cols or k <= 1:
        return True
    group_sizes = df.groupby(cat_cols).size()
    return (group_sizes >= k).all()


def enforce_k_anonymity(df: pd.DataFrame, cat_cols: List[str], k: int) -> pd.DataFrame:
    """Drop records that do not satisfy k-anonymity."""
    if not cat_cols or k <= 1:
        return df
    sizes = df.groupby(cat_cols).transform("size")
    return df[sizes >= k].reset_index(drop=True)
