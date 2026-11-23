"""Command-line entry point for Task B early outcome baselines."""

import argparse
import json
from pathlib import Path

from .outcome.models import DEFAULT_C_VALUES
from .outcome.pipeline import run_task_b
from .source import DATA_DIR, OUTPUT_DIR


DEFAULT_DATA = DATA_DIR / "election_model.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "task_b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--protocol",
        choices=("percentage", "fixed", "all"),
        default="all",
    )
    parser.add_argument("--max-features", type=int, default=40000)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=DEFAULT_C_VALUES,
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        help="Use only the first N elections from each chronological split",
    )
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Missing election dataset: {args.data}")
    results, metadata = run_task_b(
        args.data,
        args.output,
        args.protocol,
        tuple(args.c_values),
        args.max_features,
        args.bootstrap,
        args.seed,
        args.smoke_limit,
    )
    print(results.to_string(index=False))
    print(json.dumps(metadata, indent=2))
    print(f"Report: {(args.output / 'report.html').resolve()}")


if __name__ == "__main__":
    main()
