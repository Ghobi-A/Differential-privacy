"""Simple command line interface for noise mechanisms.

Reads a CSV file, applies a selected noise mechanism to numeric columns or
randomised response to categorical columns, and writes the result to a new
CSV file.
"""

import argparse
import pandas as pd
from pathlib import Path

from mechanisms import (
    add_laplace_noise,
    add_gaussian_noise,
    add_exponential_noise,
    add_geometric_noise,
    randomised_response,
)

MECHANISMS = {
    'laplace': add_laplace_noise,
    'gaussian': add_gaussian_noise,
    'exponential': add_exponential_noise,
    'geometric': add_geometric_noise,
    'randomised-response': randomised_response,
}

def main():
    parser = argparse.ArgumentParser(
        description="Apply DP mechanism to CSV data, including randomised response for categorical values"
    )
    parser.add_argument('--input', type=Path, required=True, help='Input CSV file')
    parser.add_argument('--output', type=Path, required=True, help='Output CSV file')
    parser.add_argument('--mechanism', choices=MECHANISMS.keys(), default='laplace')
    parser.add_argument('--random-state', type=int, default=None)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--sensitivity', type=float, default=1.0)
    parser.add_argument(
        '--probability',
        type=float,
        default=0.7,
        help='Truthful response probability for the randomised-response mechanism',
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    numeric = df.select_dtypes(include=['number']).columns
    categorical = df.select_dtypes(exclude=['number']).columns
    func = MECHANISMS[args.mechanism]

    if args.mechanism == 'randomised-response':
        for col in categorical:
            df[col] = func(
                df[col], p=args.probability, random_state=args.random_state
            )
    else:
        kwargs = {'random_state': args.random_state}
        if args.mechanism in {'laplace', 'gaussian', 'exponential', 'geometric'}:
            kwargs['epsilon'] = args.epsilon
        if args.mechanism in {'laplace', 'gaussian', 'exponential'}:
            kwargs['sensitivity'] = args.sensitivity
        df.loc[:, numeric] = func(df[numeric], **kwargs)
    df.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()
