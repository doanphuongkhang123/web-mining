"""Group SNAP RfAs by calendar-day gaps and clean comments."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .source import OUTPUT_DIR, read_snap, snap_path
from .text import CleanedComment, TEXT_STATUSES, TEXT_VERSION, clean_comment

DEFAULT_OUTPUT = OUTPUT_DIR / "preprocessed"
ELECTION_COLUMNS = ["tgt", "yea", "res", "vote_start"]
GROUP_COLUMNS = ELECTION_COLUMNS[:-1]
VOTE_GAP_DAYS = 7


def group_votes(votes):
    votes = votes.copy()
    for column in ("yea", "vot", "res"):
        votes[column] = pd.to_numeric(votes[column], errors="raise")
    if (not votes.vot.isin([-1, 0, 1]).all()
            or not votes.res.isin([-1, 1]).all()
            or votes.tgt.str.strip().eq("").any()):
        raise ValueError("Invalid vote, outcome, or candidate")
    votes["textid"] = range(1, len(votes) + 1)

    # Use calendar dates for the gap rule and sort within each day by the time string.
    parts = votes.dat.str.partition(",")
    votes["time"] = parts[0].str.strip().replace("", pd.NA)
    dates = parts[2].str.strip()
    full_month = pd.to_datetime(dates, format="%d %B %Y", errors="coerce")
    short_month = pd.to_datetime(dates, format="%d %b %Y", errors="coerce")
    votes["date"] = full_month.fillna(short_month)
    votes = votes.sort_values([*GROUP_COLUMNS, "date", "time", "textid"])

    gaps = votes.groupby(GROUP_COLUMNS, sort=False).date.diff().dt.days
    votes["vote_start"] = votes.date.where(gaps.isna() | gaps.gt(VOTE_GAP_DAYS))
    votes["vote_start"] = votes.groupby(GROUP_COLUMNS, sort=False).vote_start.ffill()
    return votes[["textid", *ELECTION_COLUMNS, "time", "date", "src", "vot", "txt"]]


def clean_comments(votes):
    comments = votes.loc[votes.txt.notna() & votes.txt.ne(""), ["textid", "vot", "src", "txt"]]
    cleaned = [
        clean_comment(raw, vote, voter)
        for raw, vote, voter in comments[["txt", "vot", "src"]].itertuples(index=False, name=None)
    ]
    text = pd.DataFrame(cleaned, index=comments.index, columns=CleanedComment._fields)
    return comments[["textid", "vot"]].join(text)


def preprocess(data_path, output):
    votes = group_votes(read_snap(data_path))
    comments = clean_comments(votes)
    by_vote = comments.groupby(["vot", "text_status"]).size().unstack(fill_value=0)
    by_vote = by_vote.reindex(index=[-1, 0, 1], columns=TEXT_STATUSES, fill_value=0)
    by_status = by_vote.sum()
    summary = {
        "vote_records": len(votes),
        "inferred_rfa_keys": len(votes[ELECTION_COLUMNS].drop_duplicates()),
        "missing_dates": int(votes.date.isna().sum()),
        "missing_vote_start": int(votes.vote_start.isna().sum()),
        "nonempty_raw_comments": len(comments),
        "comments_empty_after_cleaning": int(comments.txt_clean.str.strip().eq("").sum()),
        "text_preprocessing_version": TEXT_VERSION,
        "text_status_counts": {status: int(count) for status, count in by_status.items()},
        "text_status_by_vote": {
            str(vote): {status: int(count) for status, count in row.items()}
            for vote, row in by_vote.iterrows()
        },
        "gap_threshold_days": VOTE_GAP_DAYS,
    }

    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".preprocess-", dir=output) as temp:
        stage = Path(temp)
        votes.to_csv(stage / "votes.csv.gz", index=False, date_format="%Y-%m-%d")
        comments.drop(columns="vot").to_csv(stage / "comments_for_scoring.csv.gz", index=False)
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for name in ("votes.csv.gz", "comments_for_scoring.csv.gz", "summary.json"):
            (stage / name).replace(output / name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Path to a local SNAP text or gzip file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(preprocess(snap_path(args.data), args.output), indent=2))


if __name__ == "__main__":
    main()
