"""Run the RfA grouping, comment preprocessing, and four toxicity EDA charts."""

import argparse
import json
from pathlib import Path

from .eda import run_eda
from .preprocess import preprocess
from .source import OUTPUT_DIR, scored_path, snap_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Local SNAP text or gzip file")
    parser.add_argument("--scored", type=Path, help="Local ZIP or CSV containing toxicity scores")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    scores = scored_path(args.scored)
    print("Preprocessing SNAP votes with the seven-day rule", flush=True)
    prepared = preprocess(snap_path(args.data), args.output / "preprocessed")
    print("Running EDA with local toxicity scores", flush=True)
    analysis = run_eda(scores, args.output / "eda")
    print(json.dumps({"preprocessing": prepared, "eda": analysis}, indent=2))
    print(f"Report: {(args.output / 'eda' / 'report.html').resolve()}")


if __name__ == "__main__":
    main()
