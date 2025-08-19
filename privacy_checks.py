import pandas as pd
import numpy as np


def check_k_anonymity(df: pd.DataFrame, quasi_columns: list[str], k: int) -> bool:
    """Return True if each group defined by quasi_columns has at least k rows."""
    group_sizes = df.groupby(quasi_columns).size()
    return bool((group_sizes >= k).all())


def check_l_diversity(df: pd.DataFrame, quasi_columns: list[str], sensitive_column: str, l: int) -> bool:
    """Return True if each quasi-identifier group has at least l distinct sensitive values."""
    diversity = df.groupby(quasi_columns)[sensitive_column].nunique()
    return bool((diversity >= l).all())


def check_t_closeness(df: pd.DataFrame, quasi_columns: list[str], sensitive_column: str, t: float) -> bool:
    """Return True if distribution of sensitive column in each group is within t of overall distribution.

    The distance metric used is total variation distance (half the L1 difference).
    """
    overall_dist = df[sensitive_column].value_counts(normalize=True)
    for _, group in df.groupby(quasi_columns):
        group_dist = group[sensitive_column].value_counts(normalize=True)
        # align indices
        aligned = overall_dist.to_frame('overall').join(group_dist.to_frame('group'), how='outer').fillna(0)
        tvd = 0.5 * np.abs(aligned['overall'] - aligned['group']).sum()
        if tvd > t:
            return False
    return True
