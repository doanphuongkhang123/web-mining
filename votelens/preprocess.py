"""Prepare chronological Wiki-RfA datasets from SNAP and archived election evidence."""

import argparse
from collections import Counter, defaultdict
import gzip
import html
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata

import numpy as np
import pandas as pd

from .source import FIELDS, ROOT, TOKEN_RE, clean_comment, ensure_dataset, parse_records, sha256_file

DEFAULT_EVIDENCE = Path(__file__).resolve().parent / "resources" / "archive_evidence.json.gz"
DEFAULT_OUTPUT = ROOT / "outputs" / "processed"

MONTHS = (
    "January February March April May June July August September October November December"
).split()
MONTH_LOOKUP = {name.lower(): n for n, name in enumerate(MONTHS, 1)}
MONTH_LOOKUP.update({name[:3].lower(): n for n, name in enumerate(MONTHS, 1)})
MONTH_LOOKUP["sept"] = 9
DATE_PATTERN = r"(?P<hour>\d{1,2}):(?P<minute>\d{2}),?\s+(?P<day>\d{1,2})\s+(?P<month>[a-zA-Z]+)\s+(?P<year>\d{4})"
DATE_RE = re.compile(DATE_PATTERN)
STAMP_FORMAT = "%Y-%m-%dT%H:%MZ"
HORIZONS = (10, 20, 30)
VAL_START = pd.Timestamp("2008-01-01", tz="UTC")
TEST_START = pd.Timestamp("2010-01-01", tz="UTC")
SPLITS = ("train", "validation", "test")
GRAPH_SIGNS = (-1, 1)


def normalized_name(text):
    return " ".join(unicodedata.normalize("NFC", text).replace("_", " ").split()).casefold()


def parse_snap_dates(values):
    parts = values.str.strip().where(values.str.strip().str.fullmatch(DATE_RE)).str.extract(DATE_RE)
    parts["month"] = parts["month"].str.lower().map(MONTH_LOOKUP)
    parts = parts.astype(float)
    clock_is_valid = parts.hour.between(0, 23) & parts.minute.between(0, 59)
    return pd.to_datetime(parts.where(clock_is_valid), errors="coerce", utc=True)


def load_snap(path):
    votes, issues = parse_records(path)
    if votes.empty or issues or votes.isna().any().any() or votes.TGT.str.strip().eq("").any():
        raise ValueError(f"Unexpected SNAP record schema: {issues[:5]}")
    for name, values in {"VOT": {"-1", "0", "1"}, "RES": {"-1", "1"}}.items():
        if not set(votes[name]).issubset(values):
            raise ValueError(f"Unexpected {name} values")
        votes[name] = votes[name].astype(int)
    votes["YEA"] = votes.YEA.astype(int)
    votes["record_id"] = np.arange(1, len(votes) + 1)
    # File-order runs are hypotheses. Archive evidence establishes their identities.
    votes["block_id"] = votes.TGT.ne(votes.TGT.shift()).cumsum()
    votes["timestamp"] = parse_snap_dates(votes.DAT)
    votes["exact_duplicate"] = votes.duplicated(FIELDS)
    votes["drop_duplicate"] = (
        votes.duplicated(["block_id", *FIELDS])
        & votes.SRC.str.strip().ne("") & votes.timestamp.notna()
    )
    votes["voter_id"] = votes.SRC.map(normalized_name)
    votes["candidate_id"] = votes.TGT.map(normalized_name)
    votes["self_vote"] = votes.voter_id.eq(votes.candidate_id)
    return votes


MIN_MATCH_RATIO = 0.80
MAX_VOTE_AGE = pd.Timedelta(days=21)
LIST_DATE_TOLERANCE = pd.Timedelta(days=2)
LABEL_BUFFER = pd.Timedelta(days=7)
SIGNATURE_LINK = re.compile(r"\[\[\s*(?:user(?: talk)?|special:contributions)\s*[:/][^\]]+\]\]", re.I)
FILE_LINK = re.compile(r"\[\[\s*(?:file|image):[^\]]*\]\]", re.I)
VOTE_PREFIX = re.compile(
    r"^\s*[#*:;—–-]*\s*(?:i\s+)?(?:(?:strongly|weakly|mildly|very|conditional|"
    r"strong|weak|mild|qualified|qualifiedly|snow)\s+){0,3}"
    r"(?:support(?:ing)?|oppose|opposition|neutral(?:ity)?|abstain|object)\b"
    r"(?:\s*,?\s*(?:strongly|weakly|mildly)\b)?\s*[,.:;!?—–-]*\s*", re.I,
)
OUTCOME_TEXT = re.compile(
    r"\b(?:rfa|nomination|request|election)\s+(?:has been\s+|was\s+|is\s+)?"
    r"(?:closed as|successful|unsuccessful|failed|withdrawn)\b"
    r"|\b(?:closed as (?:successful|unsuccessful)|(?:has been|was) promoted"
    r"|final (?:tally|result)|(?:candidate|nominee) (?:has )?withdrawn)\b", re.I,
)
REFERENCE = re.compile(r"<ref\b[^>]*?/>|<ref\b[^>]*>.*?</ref\s*>", re.I | re.S)
STRIKE = re.compile(r"<(?:s|del|strike)\b[^>]*>.*?</(?:s|del|strike)\s*>", re.I | re.S)
EXTERNAL_LINK = re.compile(r"\[https?://[^\s\]]+(?:\s+([^\]]*))?\]", re.I)
VOTE_ONLY = re.compile(r"^(?:per\s+(?:the\s+)?noms?\.?|as\s+(?:co-?)?nom(?:inator)?\.?)$", re.I)


def preprocess_comment(raw):
    text = html.unescape(raw)
    text = STRIKE.sub(" ", REFERENCE.sub(" ", text))
    text = FILE_LINK.sub(" ", SIGNATURE_LINK.sub(" ", text))
    text = EXTERNAL_LINK.sub(lambda match: match[1] or " ", text)
    text = clean_comment(text)
    text = re.sub(DATE_PATTERN + r"\s*\(UTC\)", " ", text, flags=re.I)
    text = re.sub(r"\b(?:user(?: talk)?):\s*[^\[\]\s]+", " ", text, flags=re.I)
    text = VOTE_PREFIX.sub("", text)
    text = " ".join(text.split()).strip(" #*:-–—")
    if not TOKEN_RE.search(text) or VOTE_ONLY.fullmatch(text) or OUTCOME_TEXT.search(text):
        return ""
    return unicodedata.normalize("NFC", text)


def read_evidence(path, data_sha):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        evidence = json.load(source)
    if evidence.get("version") != 1 or evidence.get("source_sha256") != data_sha:
        raise ValueError("Archive evidence was collected against a different SNAP file")
    return evidence


def match_archives(votes, evidence):
    distinct_dates = votes.groupby("block_id").timestamp.nunique().to_dict()
    timestamps_by_record = votes.set_index("record_id").timestamp.dropna().to_dict()
    plausible = defaultdict(dict)
    dates_by_page = defaultdict(dict)
    for page in evidence["pages"]:
        listed = pd.Timestamp(page["archive_date"], tz="UTC")
        lower = listed - MAX_VOTE_AGE
        upper = listed + LIST_DATE_TOLERANCE
        for match in page["matches"]:
            block_id = match["block_id"]
            count = distinct_dates[block_id]
            matching_dates = {
                timestamps_by_record[i] for i in match["record_ids"]
                if i in timestamps_by_record and lower <= timestamps_by_record[i] < upper
            }
            dates_by_page[block_id].setdefault(page["page_id"], set()).update(matching_dates)
            if not count or len(matching_dates) < min(5, count) or len(matching_dates) / count < MIN_MATCH_RATIO:
                continue
            plausible[block_id][page["page_id"]] = (page, match, listed)
    unique_use = Counter(
        page_id for options in plausible.values() for page_id in options
    )
    matches = {}
    for block_id, options in plausible.items():
        if len(options) != 1:
            continue
        page_id, (page, match, listed) = next(iter(options.items()))
        chosen_dates = dates_by_page[block_id][page_id]
        has_competing_dates = any(
            other_page_id != page_id and len(dates - chosen_dates) >= 2
            for other_page_id, dates in dates_by_page[block_id].items()
        )
        if unique_use[page_id] == 1 and not has_competing_dates:
            matches[block_id] = (page, match, listed)
    return matches, set(plausible) - matches.keys()


def assign_split(first_vote, label_by):
    if first_vote < VAL_START:
        return "train" if label_by < VAL_START else "purged"
    if first_vote < TEST_START:
        return "validation" if label_by < TEST_START else "purged"
    return "test"


def election_audit(votes, matches, ambiguous):
    checked = set()
    elections = []
    for block_id, group in votes.groupby("block_id", sort=False):
        kept = group.loc[~group.drop_duplicate]
        dated = kept.loc[kept.timestamp.notna()].sort_values(["timestamp", "record_id"])
        row = {
            "block_id": block_id, "election_id": "", "candidate_id": group.candidate_id.iat[0],
            "candidate_name": group.TGT.iat[0], "snap_year": group.YEA.iat[0],
            "snap_records": len(group), "undated_records": len(kept) - len(dated),
            "exact_duplicates_flagged": int(group.exact_duplicate.sum()),
            "duplicates_removed": int(group.drop_duplicate.sum()),
            "snap_outcome": group.RES.iat[0] if group.RES.nunique() == 1 else None,
            "archive_date": None, "archive_page": "", "archive_revision_id": None,
            "archive_outcome": None, "signature_matches": 0, "usable_votes": 0,
            "first_observed_vote_at": pd.NaT, "last_observed_vote_at": pd.NaT,
            "closed_at": pd.NaT, "label_available_by": pd.NaT, "split": "",
            "status": "ambiguous_archive" if block_id in ambiguous else "no_archive_match",
        }
        for k in HORIZONS:
            row[f"eligible_k{k}"] = False
            row[f"k{k}_status"] = row["status"]
        elections.append(row)
        if block_id not in matches:
            continue

        page, match, listed = matches[block_id]
        confirmed = dated.loc[dated.record_id.isin(match["record_ids"])]
        candidates = [pd.Timestamp(stamp) for stamp in page["close_candidates"]]
        closings = [
            stamp for stamp in candidates
            if listed - LIST_DATE_TOLERANCE <= stamp < listed + LIST_DATE_TOLERANCE
            and stamp >= confirmed.timestamp.max()
        ]
        closed_at = closings[0] if len(closings) == 1 else pd.NaT
        lower_bound = listed - MAX_VOTE_AGE
        vote_bound = closed_at if pd.notna(closed_at) else listed + LIST_DATE_TOLERANCE
        usable = confirmed.loc[confirmed.timestamp.ge(lower_bound) & confirmed.timestamp.lt(vote_bound)]
        by = closed_at if pd.notna(closed_at) else listed + LABEL_BUFFER
        first = usable.timestamp.min()
        split = assign_split(first, by) if pd.notna(first) else ""
        row.update(
            election_id=f"enwiki-{page['page_id']}", archive_date=page["archive_date"],
            archive_page=page["title"], archive_revision_id=page["revision_id"],
            archive_outcome=page["archive_outcome"], signature_matches=len(confirmed),
            usable_votes=len(usable), first_observed_vote_at=first,
            last_observed_vote_at=usable.timestamp.max(), closed_at=closed_at, label_available_by=by,
            split=split, status="archive_matched",
        )
        checked.update(usable.record_id)
        reason = None
        if group.RES.nunique() != 1 or row["snap_outcome"] != row["archive_outcome"]:
            reason = "outcome_disagreement"
        elif row["undated_records"]:
            reason = "contains_undated_records"
        elif split == "purged":
            reason = "label_crosses_split"
        for k in HORIZONS:
            k_reason = reason
            if k_reason is None:
                if len(dated) < k:
                    k_reason = "fewer_than_k_votes"
                elif not dated.record_id.iloc[:k].isin(usable.record_id).all():
                    k_reason = "prefix_timestamp_unconfirmed"
                elif len(dated) > k and dated.timestamp.iat[k - 1] == dated.timestamp.iat[k]:
                    k_reason = "tie_at_cutoff"
                else:
                    k_reason = "eligible"
            row[f"k{k}_status"] = k_reason
            row[f"eligible_k{k}"] = k_reason == "eligible"

    audit = pd.DataFrame(elections)
    for column in ("archive_revision_id", "archive_outcome", "snap_outcome"):
        audit[column] = audit[column].astype("Int64")
    return audit, checked


def past_graph_features(prefix, edges):
    by_voter = defaultdict(Counter)
    by_candidate = defaultdict(Counter)
    edge_iter = iter(edges[["timestamp", "voter_id", "candidate_id", "vote"]].itertuples(index=False, name=None))
    edge = next(edge_iter, None)
    rows = []
    for election_id, group in prefix.groupby("election_id", sort=False):
        start = group.timestamp.iat[0]
        while edge is not None and edge[0] < start:
            _, voter_id, candidate_id, vote = edge
            by_voter[voter_id][vote] += 1
            by_candidate[candidate_id][vote] += 1
            edge = next(edge_iter, None)
        candidate_history = by_candidate.get(group.candidate_id.iat[0], Counter())
        voters = group.loc[group.voter_id.ne(""), "voter_id"].unique()
        histories = [by_voter[voter] for voter in voters if voter in by_voter]
        rows.append({
            "election_id": election_id,
            "graph_candidate_prior_support": candidate_history[1],
            "graph_candidate_prior_oppose": candidate_history[-1],
            "graph_candidate_has_history": int(bool(candidate_history)),
            "graph_known_voter_count": len(voters),
            "graph_voter_history_fraction": len(histories) / len(voters) if len(voters) else 0.0,
            "graph_voter_mean_prior_votes": (
                sum(sum(history.values()) for history in histories) / len(voters) if len(voters) else 0.0
            ),
            "graph_voter_mean_prior_support_ratio": (
                sum(history[1] / sum(history.values()) for history in histories) / len(histories)
                if histories else 0.0
            ),
        })
    return pd.DataFrame(rows)


def export_csv(frame, path):
    options = {"compression": {"method": "gzip", "mtime": 0}} if path.suffix == ".gz" else {}
    frame.to_csv(path, index=False, **options)


def validate(audit, votes, edges, prefix, features, targets, stance):
    checked = audit.loc[audit.status.eq("archive_matched")]
    assert checked.election_id.is_unique and votes.record_id.is_unique
    assert votes.timestamp_verified.sum() == checked.usable_votes.sum()
    assert set(edges.vote.unique()).issubset(GRAPH_SIGNS)
    assert edges.timestamp.is_monotonic_increasing
    assert edges.timestamp_verified.all() and edges.voter_id.ne("").all()
    assert edges.voter_id.ne(edges.candidate_id).all()
    assert prefix.timestamp_verified.all() and prefix.record_id.is_unique
    assert prefix.groupby("election_id").size().eq(10).all()
    assert targets.election_id.is_unique and features.election_id.is_unique
    assert set(prefix.election_id) == set(targets.election_id) == set(features.election_id)
    assert features[["vote_support", "vote_oppose", "vote_neutral"]].sum(axis=1).eq(10).all()
    assert set(targets.groupby("split").label.nunique()) == {2}
    assert set(targets.split) == set(SPLITS)
    assert targets.loc[targets.split.eq("train"), "label_available_by"].max() < VAL_START
    assert targets.loc[targets.split.eq("validation"), "label_available_by"].max() < TEST_START
    assert stance.groupby("election_id").split.nunique().eq(1).all()
    assert stance.timestamp_verified.all() and stance.comment_clean.ne("").all()
    overlapping_splits = stance.merge(targets[["election_id", "split"]], on="election_id")
    assert overlapping_splits.split_x.eq(overlapping_splits.split_y).all()
    assert not {"label", "RES", "closed_at", "archive_outcome", "last_observed_vote_at"}.intersection(features)


def make_summaries(raw, audit, features, targets, stance, edges, source_sha, evidence):
    annual = audit.groupby("snap_year").agg(
        snap_records=("snap_records", "sum"), snap_blocks=("block_id", "size"),
        archive_matched=("election_id", lambda col: col.ne("").sum()),
        eligible_k10=("eligible_k10", "sum"), eligible_k20=("eligible_k20", "sum"),
        eligible_k30=("eligible_k30", "sum"),
    ).reset_index()
    split_summary = targets.groupby("split", sort=False).agg(
        elections=("label", "size"), accepted=("label", lambda col: col.eq(1).sum()),
        rejected=("label", lambda col: col.eq(-1).sum()),
        first_start=("first_observed_vote_at", "min"), last_start=("first_observed_vote_at", "max"),
    ).reindex(SPLITS).reset_index()
    split_summary["acceptance_rate"] = split_summary.accepted / split_summary.elections
    stance_summary = stance.groupby(["split", "stance_label"]).size().rename("comments").reset_index()
    retention = audit.groupby("snap_outcome").agg(
        snap_runs=("block_id", "size"), archive_matched=("election_id", lambda col: col.ne("").sum()),
        eligible_k10=("eligible_k10", "sum"),
    ).reset_index()
    retention["eligibility_rate"] = retention.eligible_k10 / retention.snap_runs
    horizons = []
    for k in HORIZONS:
        selected = audit.loc[audit[f"eligible_k{k}"]]
        horizons.append({
            "k": k, "elections": len(selected), "accepted": int(selected.snap_outcome.eq(1).sum()),
            "rejected": int(selected.snap_outcome.eq(-1).sum()),
        })
    shared = audit[[f"eligible_k{k}" for k in HORIZONS]].all(axis=1)
    summary = {
        "source_sha256": source_sha, "archive_titles_requested": evidence["titles_requested"],
        "raw_vote_records": len(raw), "parseable_vote_dates": int(raw.timestamp.notna().sum()),
        "missing_vote_dates": int(raw.timestamp_raw.str.strip().eq("").sum()),
        "invalid_vote_dates": int((raw.timestamp.isna() & raw.timestamp_raw.str.strip().ne("")).sum()),
        "missing_voter_ids": int(raw.voter_id.eq("").sum()),
        "blank_comments_raw": int(raw.comment_raw.str.strip().eq("").sum()),
        "blank_comments_clean": int(raw.comment_clean.eq("").sum()),
        "exact_duplicates_flagged": int(raw.exact_duplicate.sum()),
        "duplicates_removed": int(raw.drop_duplicate.sum()),
        "raw_candidate_runs": len(audit),
        "raw_candidate_years": int(audit[["candidate_id", "snap_year"]].drop_duplicates().shape[0]),
        "candidate_years_with_multiple_archive_matches": int(
            audit.loc[audit.election_id.ne("")].groupby(["candidate_id", "snap_year"]).size().gt(1).sum()
        ),
        "archive_matched": int(audit.election_id.ne("").sum()),
        "archive_matched_with_exact_close": int(audit.closed_at.notna().sum()),
        "archive_matched_outcome_disagreements": int(
            (audit.election_id.ne("") & audit.snap_outcome.ne(audit.archive_outcome)).sum()
        ),
        "verified_timestamp_votes": int(raw.timestamp_verified.sum()),
        "signed_graph_edges": len(edges), "stance_comments": len(stance),
        "outcome_elections": len(features), "outcome_with_no_clean_text": int(features.comment_text.eq("").sum()),
        "k10_counts_by_status": {key: int(count) for key, count in audit.k10_status.value_counts().items()},
        "horizons": horizons, "elections_eligible_at_all_k": int(shared.sum()),
        "validation_start_utc": str(VAL_START), "test_start_utc": str(TEST_START),
    }
    return summary, annual, split_summary, stance_summary, retention


def plot_eda(annual, split_summary, horizons, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    ink, blue, green, red = "#23354e", "#82a3ce", "#2a8c75", "#ca6651"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    fig.suptitle("VoteLens · From raw SNAP records to chronological cohorts",
                 x=0.03, ha="left", fontsize=16, weight="bold", color=ink)
    left, middle, right = axes
    x = np.arange(len(annual))
    left.bar(x, annual.snap_blocks, color="#e4eaf2", label="SNAP candidate runs")
    left.bar(x, annual.archive_matched, color=blue, label="Matched to archive")
    left.bar(x, annual.eligible_k10, color=green, label="Eligible at 10 votes")
    left.set_xticks(x, annual.snap_year.astype(str).str[-2:])
    left.set(xlabel="SNAP year (20xx)", ylabel="Runs / matched elections", title="Coverage by year")
    left.legend(frameon=False, loc="upper right", fontsize=8)

    cohort = pd.DataFrame(horizons)
    for ax, table, names, title in (
        (middle, cohort, cohort.k.map(lambda k: f"{k} votes"), "Eligibility by vote cutoff"),
        (right, split_summary, ["Train", "Validation", "Test"], "Class balance at 10 votes"),
    ):
        x = np.arange(len(table))
        ax.bar(x, table.accepted, color=green, label="Accepted")
        ax.bar(x, table.rejected, bottom=table.accepted, color=red, label="Rejected")
        ax.bar_label(ax.containers[0], label_type="center", color="white", fontsize=9)
        ax.bar_label(ax.containers[1], label_type="center", color="white", fontsize=9)
        ax.set_xticks(x, names)
        ax.set(title=title, ylabel="Elections")
        ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.grid(axis="y", color="#e4eaf2")
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def prepare(data_path, evidence_path, output):
    source_sha = sha256_file(data_path)
    evidence = read_evidence(evidence_path, source_sha)
    raw = load_snap(data_path)
    matches, ambiguous = match_archives(raw, evidence)
    audit, checked = election_audit(raw, matches, ambiguous)
    raw["timestamp_verified"] = raw.record_id.isin(checked)
    raw["election_id"] = raw.block_id.map(audit.set_index("block_id").election_id)
    raw["comment_clean"] = raw.TXT.map(preprocess_comment)
    raw = raw.rename(columns={"VOT": "vote", "YEA": "snap_year", "DAT": "timestamp_raw",
                              "SRC": "voter_name", "TXT": "comment_raw"})
    votes = raw[[
        "record_id", "block_id", "election_id", "candidate_id", "voter_id", "voter_name", "vote",
        "snap_year", "timestamp", "timestamp_raw", "timestamp_verified", "comment_raw",
        "comment_clean", "self_vote", "exact_duplicate", "drop_duplicate",
    ]]
    events = votes.loc[votes.timestamp_verified].sort_values(["timestamp", "record_id"])
    edges = events.loc[events.voter_id.ne("") & ~events.self_vote & events.vote.isin(GRAPH_SIGNS),
                       ["record_id", "election_id", "timestamp", "timestamp_verified", "voter_id", "candidate_id", "vote"]]
    selected = audit.loc[audit.eligible_k10].sort_values(["first_observed_vote_at", "election_id"])
    prefix = events.loc[events.election_id.isin(selected.election_id)].groupby("election_id", sort=False).head(10).copy()
    prefix["vote_index"] = prefix.groupby("election_id", sort=False).cumcount() + 1
    prefix = prefix[["election_id", "vote_index", "record_id", "timestamp", "timestamp_verified", "candidate_id",
                     "voter_id", "vote", "comment_clean"]]
    groups = prefix.groupby("election_id", sort=False)
    features = groups.agg(
        prediction_at=("timestamp", "max"),
        vote_support=("vote", lambda col: col.eq(1).sum()),
        vote_oppose=("vote", lambda col: col.eq(-1).sum()),
        vote_neutral=("vote", lambda col: col.eq(0).sum()),
        comment_text=("comment_clean", lambda col: "\n".join(value for value in col if value)),
    ).reset_index()
    features = features.merge(past_graph_features(prefix, edges), on="election_id", validate="one_to_one")
    targets = selected[["election_id", "split", "snap_outcome", "first_observed_vote_at",
                        "label_available_by"]].rename(columns={"snap_outcome": "label"})
    stance_splits = audit.loc[
        audit.split.isin(SPLITS) & audit.snap_outcome.eq(audit.archive_outcome), ["election_id", "split"]
    ]
    stance = events.loc[events.comment_clean.ne(""), [
        "record_id", "election_id", "timestamp", "timestamp_verified", "vote", "comment_clean",
    ]].merge(stance_splits, on="election_id", validate="many_to_one")
    stance = stance.rename(columns={"vote": "stance_label"})
    validate(audit, votes, edges, prefix, features, targets, stance)
    summary, annual, split_summary, stance_summary, retention = make_summaries(
        raw, audit, features, targets, stance, edges, source_sha, evidence,
    )
    matched = audit.loc[audit.election_id.ne("")]
    repeat = matched.loc[matched.duplicated(["candidate_id", "snap_year"], keep=False)]
    train = features.merge(
        targets.loc[targets.split.eq("train"), ["election_id", "label"]],
        on="election_id", validate="one_to_one",
    )
    train_support = pd.crosstab(train.vote_support, train.label).rename(columns={-1: "rejected", 1: "accepted"})
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "votes.csv.gz": votes,
        "elections.csv": matched.drop(columns="status"),
        "candidate_run_audit.csv": audit,
        "prefix_votes.csv.gz": prefix,
        "outcome_features.csv.gz": features,
        "outcome_targets.csv": targets,
        "stance_comments.csv.gz": stance,
        "signed_edges.csv.gz": edges,
        "eda_by_year.csv": annual,
        "eda_by_split.csv": split_summary,
        "eda_stance_counts.csv": stance_summary,
        "eda_retention_by_outcome.csv": retention,
        "eda_by_cutoff.csv": pd.DataFrame(summary["horizons"]),
        "eda_same_year_rfas.csv": repeat[[
            "candidate_name", "snap_year", "election_id", "archive_page", "signature_matches",
        ]].sort_values(["candidate_name", "snap_year", "election_id"]),
        "eda_train_support_by_outcome.csv": train_support.reset_index(),
    }
    with TemporaryDirectory(prefix=".prepare-", dir=output) as directory:
        staged = Path(directory)
        for name, frame in tables.items():
            export_csv(frame, staged / name)
        plot_eda(annual, split_summary, summary["horizons"], staged / "cohort_eda.png")
        (staged / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for name in (*tables, "cohort_eda.png", "summary.json"):
            (staged / name).replace(output / name)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Local SNAP gzip file")
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.data or ensure_dataset(), args.evidence, args.output)
