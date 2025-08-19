"""Simple command line interface for noise mechanisms.

Reads a CSV file, applies a selected noise mechanism to numeric columns and
writes the result to a new CSV file.
"""

import argparse
import pandas as pd
from pathlib import Path

from mechanisms import (
    add_laplace_noise,
    add_gaussian_noise,
    add_exponential_noise,
    add_geometric_noise,
)

MECHANISMS = {
    'laplace': add_laplace_noise,
    'gaussian': add_gaussian_noise,
    'exponential': add_exponential_noise,
    'geometric': add_geometric_noise,
}

def main():
    parser = argparse.ArgumentParser(description="Apply DP noise mechanism to CSV data")
    parser.add_argument('--input', type=Path, required=True, help='Input CSV file')
    parser.add_argument('--output', type=Path, required=True, help='Output CSV file')
    parser.add_argument('--mechanism', choices=MECHANISMS.keys(), default='laplace')
    parser.add_argument('--random-state', type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    numeric = df.select_dtypes(include=['number']).columns
    func = MECHANISMS[args.mechanism]
    df.loc[:, numeric] = func(df[numeric], random_state=args.random_state)
    df.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()
