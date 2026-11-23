"""Build leakage-safe election prefixes and numerical features."""

from collections import defaultdict

import numpy as np
import pandas as pd

from .protocol import Election, LABELS, Sample, split_for_year


GRAPH_FEATURE_NAMES = (
    "candidate_past_support",
    "candidate_past_oppose",
    "candidate_past_neutral",
    "candidate_past_total",
    "candidate_past_signed_support_share",
    "candidate_past_unique_voters",
    "observed_voter_past_total_mean",
    "observed_voter_past_total_median",
    "observed_voter_past_total_max",
    "observed_voter_past_signed_support_share_mean",
    "observed_voter_known_fraction",
    "observed_pair_past_total_mean",
    "observed_pair_past_signed_support_share_mean",
    "current_vote_historical_alignment",
    "current_vote_historical_alignment_coverage",
)
VOTE_FEATURE_NAMES = (
    "observed_votes",
    "support_votes",
    "oppose_votes",
    "neutral_votes",
    "signed_votes",
    "support_share_signed",
    "oppose_share_signed",
    "support_share_all",
    "oppose_share_all",
    "neutral_share_all",
    "unique_voters",
    "elapsed_hours",
    "vote_rate_per_hour",
    "support_rate_per_hour",
    "oppose_rate_per_hour",
    "first_half_support_share",
    "second_half_support_share",
    "support_share_trend",
)


def safe_ratio(numerator, denominator, default=0.0):
    return float(numerator / denominator) if denominator else float(default)


def stance_statistics(values):
    support = int(values.get("support", 0))
    oppose = int(values.get("oppose", 0))
    neutral = int(values.get("neutral", 0))
    signed = support + oppose
    return {
        "support": support,
        "oppose": oppose,
        "neutral": neutral,
        "total": support + oppose + neutral,
        "support_share": safe_ratio(support, signed, 0.5),
        "signed": signed,
    }


def add_history_event(state, row):
    sign = int(row.stance)
    key = "support" if sign == 1 else "oppose" if sign == -1 else "neutral"
    candidate = str(row.candidate)
    state["candidates"][candidate][key] += 1
    if pd.isna(row.voter):
        return
    voter = str(row.voter)
    state["voters"][voter][key] += 1
    state["pairs"][(voter, candidate)][key] += 1
    state["voter_candidates"][voter].add(candidate)
    state["candidate_voters"][candidate].add(voter)


def snapshot_history(state, candidate, voters):
    candidate_counts = stance_statistics(state["candidates"][candidate])
    voter_snapshots = {}
    pair_snapshots = {}
    for voter in voters:
        voter_snapshots[voter] = stance_statistics(state["voters"][voter])
        pair_snapshots[voter] = stance_statistics(state["pairs"][(voter, candidate)])
    return {
        "candidate": {
            **candidate_counts,
            "unique_voters": len(state["candidate_voters"][candidate]),
        },
        "voters": voter_snapshots,
        "pairs": pair_snapshots,
    }


def validate_data(frame):
    required = {
        "model_election_id",
        "timestamp",
        "vote_order",
        "voter",
        "candidate",
        "stance",
        "outcome",
        "txt_clean",
        "text_status",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for column in ("stance", "outcome"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not frame.stance.isin(LABELS + (0,)).all():
        raise ValueError("Invalid stance values")
    if not frame.outcome.isin(LABELS).all():
        raise ValueError("Invalid outcome values")
    timestamps = pd.to_datetime(frame.timestamp, errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError("election_model.csv contains rows with invalid timestamps")
    frame["timestamp"] = timestamps


def build_elections(frame):
    validate_data(frame)
    state = {
        "voters": defaultdict(lambda: defaultdict(int)),
        "candidates": defaultdict(lambda: defaultdict(int)),
        "pairs": defaultdict(lambda: defaultdict(int)),
        "voter_candidates": defaultdict(set),
        "candidate_voters": defaultdict(set),
    }
    ordered_events = list(
        frame.sort_values(["timestamp", "textid"]).itertuples(index=False)
    )
    event_position = 0

    groups = [
        (election_id, election.sort_values(["timestamp", "textid"]).copy())
        for election_id, election in frame.groupby(
            "model_election_id", sort=False, dropna=False
        )
    ]
    groups.sort(key=lambda item: (item[1].timestamp.iloc[0], str(item[0])))
    elections = []
    for election_id, rows in groups:
        start = rows.timestamp.iloc[0]
        while (
            event_position < len(ordered_events)
            and ordered_events[event_position].timestamp < start
        ):
            add_history_event(state, ordered_events[event_position])
            event_position += 1

        if rows.outcome.nunique() != 1:
            raise ValueError(f"Election {election_id} has inconsistent outcomes")
        if rows.candidate.nunique() != 1:
            raise ValueError(f"Election {election_id} has inconsistent candidates")
        split = split_for_year(int(start.year))
        if split == "excluded":
            continue
        candidate = str(rows.candidate.iloc[0])
        voters = {str(value) for value in rows.voter.dropna()}
        elections.append(
            Election(
                election_id=str(election_id),
                start=start,
                outcome=int(rows.outcome.iloc[0]),
                rows=rows,
                history=snapshot_history(state, candidate, voters),
                split=split,
            )
        )
    return elections
def prefix_length(window, total_votes):
    if window.kind == "percentage":
        return max(1, int(window.value * total_votes))
    return int(window.value)


def vote_features(rows):
    stances = rows.stance.astype(int).to_numpy()
    support = int(np.count_nonzero(stances == 1))
    oppose = int(np.count_nonzero(stances == -1))
    neutral = int(np.count_nonzero(stances == 0))
    signed = support + oppose
    total = len(rows)
    elapsed = max(
        float((rows.timestamp.iloc[-1] - rows.timestamp.iloc[0]).total_seconds() / 3600),
        0.0,
    )
    midpoint = max(1, total // 2)
    first = stances[:midpoint]
    second = stances[midpoint:]
    first_support_share = safe_ratio(
        int(np.count_nonzero(first == 1)),
        int(np.count_nonzero((first == 1) | (first == -1))),
        0.5,
    )
    second_support_share = safe_ratio(
        int(np.count_nonzero(second == 1)),
        int(np.count_nonzero((second == 1) | (second == -1))),
        first_support_share,
    )
    support_share_signed = safe_ratio(support, signed, 0.5)
    return np.asarray(
        (
            total,
            support,
            oppose,
            neutral,
            signed,
            support_share_signed,
            safe_ratio(oppose, signed, 0.5),
            safe_ratio(support, total),
            safe_ratio(oppose, total),
            safe_ratio(neutral, total),
            rows.voter.nunique(),
            elapsed,
            total / (elapsed + 1.0),
            support / (elapsed + 1.0),
            oppose / (elapsed + 1.0),
            first_support_share,
            second_support_share,
            second_support_share - first_support_share,
        ),
        dtype=np.float64,
    )


def graph_features(rows, history):
    candidate = history["candidate"]
    voter_values = [history["voters"][str(voter)] for voter in rows.voter.dropna()]
    pair_values = [history["pairs"][str(voter)] for voter in rows.voter.dropna()]
    known_voters = [value for value in voter_values if value["total"] > 0]
    known_pairs = [value for value in pair_values if value["total"] > 0]
    voter_totals = [value["total"] for value in voter_values]
    voter_shares = [value["support_share"] for value in known_voters]
    pair_totals = [value["total"] for value in pair_values]
    pair_shares = [value["support_share"] for value in known_pairs]

    aligned = []
    for row in rows.itertuples(index=False):
        stance = int(row.stance)
        if stance == 0 or pd.isna(row.voter):
            continue
        prior = history["voters"].get(
            str(row.voter), {"signed": 0, "support_share": 0.5}
        )
        if prior["signed"] > 0:
            prior_sign = 1 if prior["support_share"] >= 0.5 else -1
            aligned.append(float(stance == prior_sign))

    return np.asarray(
        (
            candidate["support"],
            candidate["oppose"],
            candidate["neutral"],
            candidate["total"],
            candidate["support_share"],
            candidate["unique_voters"],
            float(np.mean(voter_totals)) if voter_totals else 0.0,
            float(np.median(voter_totals)) if voter_totals else 0.0,
            float(np.max(voter_totals)) if voter_totals else 0.0,
            float(np.mean(voter_shares)) if voter_shares else 0.5,
            safe_ratio(len(known_voters), len(voter_values)),
            float(np.mean(pair_totals)) if pair_totals else 0.0,
            float(np.mean(pair_shares)) if pair_shares else 0.5,
            float(np.mean(aligned)) if aligned else 0.5,
            safe_ratio(len(aligned), len(rows)),
        ),
        dtype=np.float64,
    )


def build_samples(elections, windows):
    samples = []
    for window in windows:
        minimum = 10 if window.kind == "percentage" else 20
        for election in elections:
            total = len(election.rows)
            if total < minimum:
                continue
            count = prefix_length(window, total)
            prefix = election.rows.iloc[:count]
            text_rows = prefix.loc[
                prefix.text_status.eq("ready") & prefix.txt_clean.notna()
            ]
            text = " ".join(
                str(value).strip() for value in text_rows.txt_clean if str(value).strip()
            )
            samples.append(
                Sample(
                    window=window.label,
                    election_id=election.election_id,
                    split=election.split,
                    outcome=election.outcome,
                    start=election.start,
                    total_votes=total,
                    observed_votes=count,
                    text=text,
                    vote_features=vote_features(prefix),
                    graph_features=graph_features(prefix, election.history),
                )
            )
    return samples
