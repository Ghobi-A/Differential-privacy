import pandas as pd

from privacy_checks import (
    check_k_anonymity,
    check_l_diversity,
    check_t_closeness,
)


def test_k_anonymity():
    df = pd.DataFrame(
        {
            'q1': [1, 1, 2, 2],
            'q2': [1, 1, 2, 2],
            's': ['a', 'b', 'a', 'b'],
        }
    )
    assert check_k_anonymity(df, ['q1', 'q2'], k=2)
    df_fail = df.copy()
    df_fail.loc[0, 'q1'] = 3
    assert not check_k_anonymity(df_fail, ['q1', 'q2'], k=2)


def test_l_diversity():
    df = pd.DataFrame(
        {
            'q': [1, 1, 2, 2],
            's': ['a', 'b', 'a', 'b'],
        }
    )
    assert check_l_diversity(df, ['q'], 's', l=2)
    df_fail = pd.DataFrame(
        {
            'q': [1, 1, 2, 2],
            's': ['a', 'a', 'a', 'b'],
        }
    )
    assert not check_l_diversity(df_fail, ['q'], 's', l=2)


def test_t_closeness():
    df = pd.DataFrame(
        {
            'q': [1, 1, 2, 2],
            's': ['a', 'b', 'a', 'b'],
        }
    )
    assert check_t_closeness(df, ['q'], 's', t=0.1)
    df_fail = pd.DataFrame(
        {
            'q': [1, 1, 2, 2],
            's': ['a', 'a', 'a', 'b'],
        }
    )
    assert not check_t_closeness(df_fail, ['q'], 's', t=0.1)
