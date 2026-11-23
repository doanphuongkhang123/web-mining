"""Colab-oriented CLI for fine-tuning RoBERTa on Task B."""

import argparse
import json
from pathlib import Path

from .outcome.transformer import DEFAULT_MODEL_NAME, run_roberta
from .source import DATA_DIR, OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "election_model.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "task_b_roberta",
    )
    parser.add_argument(
        "--protocol", choices=("percentage", "fixed", "all"), default="all"
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        help="Optional subset such as: --windows 5votes 10votes or '5 votes'",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun completed windows instead of resuming from partial CSVs",
    )
    args = parser.parse_args()
    windows = args.windows
    if windows:
        windows = [
            " ".join(value.replace("_", " ").replace("votes", " votes").split())
            for value in windows
        ]
    if not args.data.is_file():
        raise FileNotFoundError(f"Missing election dataset: {args.data}")
    results, predictions, metadata = run_roberta(
        data_path=args.data,
        output=args.output,
        protocol=args.protocol,
        model_name=args.model_name,
        windows=windows,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        gradient_accumulation=args.gradient_accumulation,
        warmup_ratio=args.warmup_ratio,
        bootstrap=args.bootstrap,
        seed=args.seed,
        smoke_limit=args.smoke_limit,
        save_checkpoints=args.save_checkpoints,
        prepare_only=args.prepare_only,
        overwrite=args.overwrite,
    )
    if not results.empty:
        print(results.to_string(index=False))
        print(f"Predictions: {len(predictions)}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
