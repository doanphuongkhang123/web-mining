"""Run the full VoteLens data pipeline."""

import argparse
from pathlib import Path

from .eda import run_analysis
from .preprocess import DEFAULT_EVIDENCE, prepare
from .source import ROOT, ensure_dataset
from .vectorize import vectorize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Local SNAP gzip file. Downloads SNAP when omitted and absent.")
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE, help="Archive evidence verified against SNAP")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    data = args.data or ensure_dataset()
    processed = args.output / "processed"

    print("Running vote-level EDA", flush=True)
    run_analysis(data, args.output / "eda")
    print("Preparing election cohorts and features", flush=True)
    prepare(data, args.evidence, processed)
    print("Fitting train-only TF-IDF transforms", flush=True)
    print(vectorize(processed, args.output / "tfidf").to_string(index=False))
    print(f"Outputs saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
