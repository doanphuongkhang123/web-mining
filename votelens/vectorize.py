"""Fit separate train-only TF-IDF transforms for outcome and stance."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import DEFAULT_OUTPUT as PROCESSED_DIR, SPLITS
from .source import ROOT

DEFAULT_OUTPUT = ROOT / "outputs" / "tfidf"


def vectorize(data_dir, output_dir):
    targets = pd.read_csv(data_dir / "outcome_targets.csv", usecols=["election_id", "split"])
    features = pd.read_csv(
        data_dir / "outcome_features.csv.gz",
        usecols=["election_id", "comment_text"], dtype={"comment_text": str}, keep_default_na=False,
    )
    outcome = targets.merge(features, on="election_id", how="left", validate="one_to_one")
    if outcome.comment_text.isna().any():
        raise ValueError("Some outcome targets have no matching feature row")
    stance = pd.read_csv(
        data_dir / "stance_comments.csv.gz",
        usecols=["record_id", "split", "comment_clean"], dtype={"comment_clean": str}, keep_default_na=False,
    )
    tasks = (
        ("outcome", outcome, "election_id", "comment_text"),
        ("stance", stance, "record_id", "comment_clean"),
    )

    sizes = []
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".tfidf-", dir=output_dir) as directory:
        staged = Path(directory)
        for task, frame, key, text in tasks:
            if not frame[key].is_unique or set(frame.split) != set(SPLITS):
                raise ValueError(f"{task} needs unique IDs and all three chronological splits")
            folder = staged / task
            folder.mkdir()
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
            vectorizer.fit(frame.loc[frame.split.eq("train"), text])
            joblib.dump(vectorizer, folder / "vectorizer.joblib")

            for split in SPLITS:
                rows = frame.loc[frame.split.eq(split)]
                matrix = vectorizer.transform(rows[text])
                save_npz(folder / f"{split}.npz", matrix)
                rows[[key]].to_csv(folder / f"{split}_ids.csv", index=False)
                sizes.append({
                    "task": task, "split": split, "rows": matrix.shape[0],
                    "features": matrix.shape[1], "nonzero": matrix.nnz,
                })

        summary = pd.DataFrame(sizes)
        summary.to_csv(staged / "summary.csv", index=False)
        for task, *_ in tasks:
            destination = output_dir / task
            destination.mkdir(exist_ok=True)
            for item in (staged / task).iterdir():
                item.replace(destination / item.name)
        (staged / "summary.csv").replace(output_dir / "summary.csv")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR, help="Directory with preprocessed tables")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(vectorize(args.data, args.output).to_string(index=False))
