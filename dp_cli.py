"""Command-line interface for applying differential privacy mechanisms."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

import io_utils
import mechanisms
import privacy_checks as pc


def _parse_cols(arg: str | None) -> List[str] | None:
    if arg is None:
        return None
    return [c.strip() for c in arg.split(',') if c.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply differential privacy mechanisms")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Directory for output CSVs")
    parser.add_argument("--methods", required=True, help="Comma separated list of methods")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Privacy epsilon")
    parser.add_argument("--delta", type=float, default=1e-5, help="Delta for Gaussian noise")
    parser.add_argument("--k", type=int, default=0, help="k for k-anonymity")
    parser.add_argument("--truth-p", type=float, default=0.75, dest="truth_p",
                        help="Truth probability for randomised response")
    parser.add_argument("--cat-cols", dest="cat_cols",
                        help="Comma separated categorical column names")
    parser.add_argument("--num-cols", dest="num_cols",
                        help="Comma separated numeric column names")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if k-anonymity check fails")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = io_utils.read_csv(args.input)
    cat_cols = _parse_cols(args.cat_cols)
    num_cols = _parse_cols(args.num_cols)
    cat_cols, num_cols = io_utils.infer_columns(df, cat_cols, num_cols)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip().lower() for m in args.methods.split(',') if m.strip()]

    summary = []
    for method in methods:
        current = df.copy()
        if method == "laplace":
            current[num_cols] = mechanisms.add_laplace_noise(
                current[num_cols], epsilon=args.epsilon, seed=args.seed
            )
        elif method == "gaussian":
            current[num_cols] = mechanisms.add_gaussian_noise(
                current[num_cols], epsilon=args.epsilon, delta=args.delta, seed=args.seed
            )
        elif method == "exponential":
            current[num_cols] = mechanisms.add_exponential_noise(
                current[num_cols], epsilon=args.epsilon, seed=args.seed
            )
        elif method == "geometric":
            current[num_cols] = mechanisms.add_geometric_noise(
                current[num_cols], epsilon=args.epsilon, seed=args.seed
            )
        elif method in {"rr", "randomised_response", "randomized_response"}:
            for col in cat_cols:
                current[col] = mechanisms.randomised_response(
                    current[col], truth_p=args.truth_p, seed=args.seed
                )
        elif method in {"k", "k-anon", "k_anonymity", "kanonymity"}:
            current = pc.enforce_k_anonymity(current, cat_cols, args.k)
        else:
            raise ValueError(f"Unknown method: {method}")

        k_pass = True
        if args.k > 0:
            k_pass = pc.check_k_anonymity(current, cat_cols, args.k)
            if not k_pass and args.strict:
                raise SystemExit(f"k-anonymity check failed for {method}")

        out_file = out_dir / f"{Path(args.input).stem}_{method}.csv"
        current.to_csv(out_file, index=False)
        summary.append({"method": method, "rows": len(current), "k_pass": k_pass})

    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
