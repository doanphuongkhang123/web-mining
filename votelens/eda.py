"""Explore SNAP Wiki-RfA at the vote level with tables, charts, and reports."""

from pathlib import Path
from collections import Counter
import argparse
import base64
import html
import json
import math
import platform
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import networkx as nx
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .source import (
    DEFAULT_DATA, FIELDS, ROOT, SOURCE_URL, TOKEN_RE, clean_comment, ensure_dataset, parse_records, sha256_file,
)

DEFAULT_OUTPUT = ROOT / "outputs" / "eda"
VOTE_NAMES = {-1: "Oppose", 0: "Neutral", 1: "Support"}
OUTCOME_NAMES = {-1: "Rejected", 1: "Accepted"}
COLORS = {-1: "#d65d50", 0: "#8d99ae", 1: "#208b82"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | {
    "support", "supports", "supported", "supporting", "oppose", "opposes",
    "opposed", "opposing", "neutral", "neutrality", "strong", "weak", "per",
    "nom", "noms", "nominator", "nominators", "wikipedia", "wiki", "wp",
    "user", "users", "talk", "contribs", "contributions", "http", "https",
    "www", "com", "org", "diff", "unsigned", "utc", "comment", "comments",
}


def gini(values):
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values) or not values.sum():
        return 0.0
    ranks = np.arange(1, len(values) + 1)
    return float(2 * np.dot(ranks, values) / (len(values) * values.sum()) - (len(values) + 1) / len(values))


def write_table(frame, path, index=False):
    frame.to_csv(path, index=index)


def percentage(n, total):
    return 100 * n / total if total else 0.0


def ccdf(ax, values, label, color):
    values = np.asarray(values)
    x, counts = np.unique(values[values > 0], return_counts=True)
    y = np.cumsum(counts[::-1])[::-1] / counts.sum()
    ax.step(x, y, where="post", label=label, color=color, linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")


def build_analysis(data_path, output_dir):
    data_path, output_dir = Path(data_path), Path(output_dir)
    figures, tables = output_dir / "figures", output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    raw, parse_issues = parse_records(data_path)
    if raw.empty or raw.isna().any().any() or parse_issues:
        raise ValueError("Unexpected record schema; inspect input before proceeding: %r" % parse_issues[:5])
    df = raw.copy()
    for column, allowed in [("VOT", {-1, 0, 1}), ("RES", {-1, 1})]:
        df[column] = pd.to_numeric(df[column], errors="raise").astype(int)
        if not set(df[column]).issubset(allowed):
            raise ValueError("Unexpected values in " + column)
    df["YEA"] = pd.to_numeric(df["YEA"], errors="raise").astype(int)
    long_month = pd.to_datetime(df.DAT, format="%H:%M, %d %B %Y", errors="coerce")
    short_month = pd.to_datetime(df.DAT, format="%H:%M, %d %b %Y", errors="coerce")
    df["timestamp"] = long_month.fillna(short_month)
    df["missing_source"] = df.SRC.eq("")
    df["missing_date"] = df.DAT.eq("")
    df["invalid_date"] = df.DAT.ne("") & df.timestamp.isna()
    df["missing_text"] = df.TXT.str.strip().eq("")
    df["exact_duplicate"] = raw.duplicated(FIELDS)
    df["self_vote"] = df.SRC.eq(df.TGT) & df.SRC.ne("")
    df["year_mismatch"] = df.timestamp.notna() & df.timestamp.dt.year.ne(df.YEA)
    df["raw_whitespace_tokens"] = df.TXT.str.split().str.len()
    tokens = df.TXT.map(clean_comment).str.lower().map(TOKEN_RE.findall)
    df["cleaned_word_count"] = tokens.str.len()
    n = len(df)
    named = df.loc[~df.missing_source]
    users = sorted((set(df.SRC) | set(df.TGT)) - {""})
    vote_counts = df.VOT.value_counts().reindex([-1, 0, 1], fill_value=0)
    outcome_counts = df.RES.value_counts().reindex([-1, 1], fill_value=0)
    vote_table = pd.DataFrame({"vote": vote_counts.index, "label": [VOTE_NAMES[x] for x in vote_counts.index], "records": vote_counts.values, "percent": 100 * vote_counts.values / n})
    outcome_table = pd.crosstab(df.RES, df.VOT).reindex(index=[-1, 1], columns=[-1, 0, 1], fill_value=0)
    outcome_table.index = [OUTCOME_NAMES[x] for x in outcome_table.index]
    outcome_table.index.name = "Election outcome"
    outcome_table.columns = [VOTE_NAMES[x] for x in outcome_table.columns]
    outcome_table["Total"] = outcome_table.sum(axis=1)
    outcome_pct = outcome_table.iloc[:, :3].div(outcome_table.Total, axis=0) * 100

    annual = df.groupby("YEA").agg(
        records=("VOT", "size"), active_voters=("SRC", lambda s: s[s.ne("")].nunique()),
        distinct_candidates=("TGT", "nunique"), support=("VOT", lambda s: s.eq(1).sum()),
        oppose=("VOT", lambda s: s.eq(-1).sum()), neutral=("VOT", lambda s: s.eq(0).sum()),
        accepted_outcome_records=("RES", lambda s: s.eq(1).sum()),
    )
    for label in ["support", "oppose", "neutral"]:
        annual[label + "_percent"] = annual[label] / annual.records * 100
    annual["accepted_outcome_record_percent"] = annual.accepted_outcome_records / annual.records * 100
    quality_cols = ["missing_source", "missing_date", "invalid_date", "missing_text", "exact_duplicate", "self_vote", "year_mismatch"]
    quality = pd.DataFrame({"issue": quality_cols, "records": [int(df[c].sum()) for c in quality_cols]})
    quality["percent"] = quality.records / n * 100
    quality_annual = df.groupby("YEA")[quality_cols].mean() * 100
    unresolved = df.loc[df.invalid_date, FIELDS]
    deltas = (df.loc[df.year_mismatch, "timestamp"].dt.year - df.loc[df.year_mismatch, "YEA"]).value_counts().sort_index().rename_axis("timestamp_year_minus_YEA").to_frame("records")

    voters = named.groupby("SRC").agg(
        votes_cast=("VOT", "size"), distinct_candidates=("TGT", "nunique"),
        support=("VOT", lambda s: s.eq(1).sum()), oppose=("VOT", lambda s: s.eq(-1).sum()),
        neutral=("VOT", lambda s: s.eq(0).sum()), active_years=("YEA", "nunique"),
    ).sort_values("votes_cast", ascending=False)
    voters["support_percent"] = voters.support / voters.votes_cast * 100
    candidates = df.groupby("TGT").agg(
        votes_received=("VOT", "size"), known_voters=("SRC", lambda s: s[s.ne("")].nunique()),
        active_years=("YEA", "nunique"), observed_outcomes=("RES", "nunique"),
        support=("VOT", lambda s: s.eq(1).sum()), oppose=("VOT", lambda s: s.eq(-1).sum()),
        neutral=("VOT", lambda s: s.eq(0).sum()),
    ).sort_values("votes_received", ascending=False)
    candidates["support_percent"] = candidates.support / candidates.votes_received * 100
    top_count = int(math.ceil(len(voters) * 0.01))
    candidate_year = df.groupby(["TGT", "YEA"]).agg(records=("VOT", "size"), observed_outcomes=("RES", "nunique"))

    # Graph: all named users as nodes; named, non-self vote pairs as edges.
    # Neutral votes are retained and repeated pairs collapse only for topology.
    graph_rows = named.loc[~named.self_vote]
    graph = nx.DiGraph()
    graph.add_nodes_from(users)
    graph.add_edges_from(graph_rows[["SRC", "TGT"]].itertuples(index=False, name=None))
    components = sorted([len(c) for c in nx.weakly_connected_components(graph)], reverse=True)
    strong_components = sorted([len(c) for c in nx.strongly_connected_components(graph)], reverse=True)
    pair_history = graph_rows.groupby(["SRC", "TGT"]).VOT.agg(lambda x: tuple(sorted(set(x))))
    history_counts = pair_history.value_counts()
    pairs_with_opposite_votes = int(sum(count for history, count in history_counts.items() if -1 in history and 1 in history))
    pair_table = history_counts.rename_axis("vote_values_observed").reset_index(name="pairs")
    pair_table["vote_values_observed"] = pair_table.vote_values_observed.map(lambda values: ", ".join(VOTE_NAMES[x] for x in values))
    pair_table["percent"] = pair_table.pairs / len(pair_history) * 100
    degree_table = pd.DataFrame({"user": users, "in_degree": [graph.in_degree(x) for x in users], "out_degree": [graph.out_degree(x) for x in users]})

    comment_stats = df.groupby("VOT").agg(
        records=("TXT", "size"), empty_comments=("missing_text", "sum"),
        median_words_all=("cleaned_word_count", "median"), mean_words_all=("cleaned_word_count", "mean"),
        p90_words_all=("cleaned_word_count", lambda s: s.quantile(.9)),
        median_raw_whitespace_tokens=("raw_whitespace_tokens", "median"),
    ).rename(index=VOTE_NAMES)
    comment_stats["empty_comment_percent"] = comment_stats.empty_comments / comment_stats.records * 100
    nonempty_stats = df.loc[~df.missing_text].groupby("VOT").cleaned_word_count.agg(["median", "mean"])
    nonempty_stats.index = nonempty_stats.index.map(VOTE_NAMES)
    comment_stats["median_words_nonempty"] = nonempty_stats["median"]
    comment_stats["mean_words_nonempty"] = nonempty_stats["mean"]
    term_documents = {v: Counter() for v in [-1, 0, 1]}
    for vote, words in zip(df.VOT, tokens):
        term_documents[vote].update({w for w in words if len(w) > 2 and w not in STOP_WORDS})
    vocabulary = set(term_documents[-1]) | set(term_documents[1])
    term_rows = []
    for term in vocabulary:
        a, b = term_documents[-1][term], term_documents[1][term]
        if a + b >= 200:
            pa, pb = (a + .5) / (vote_counts[-1] + 1), (b + .5) / (vote_counts[1] + 1)
            term_rows.append({"term": term, "oppose_documents": a, "support_documents": b,
                              "oppose_document_percent": a / vote_counts[-1] * 100,
                              "support_document_percent": b / vote_counts[1] * 100,
                              "log2_oppose_to_support_prevalence": math.log2(pa / pb)})
    terms = pd.DataFrame(term_rows).sort_values("log2_oppose_to_support_prevalence", ascending=False)
    top_terms = pd.DataFrame([
        {"vote": VOTE_NAMES[v], "term": term, "documents": count, "document_percent": count / vote_counts[v] * 100}
        for v in [-1, 0, 1] for term, count in term_documents[v].most_common(25)
    ])
    duplicate_free = df.loc[~df.exact_duplicate]
    sensitivity = pd.DataFrame([
        {"sample": "All vote records (primary)", "records": n, "support_percent": df.VOT.eq(1).mean() * 100},
        {"sample": "Exclude exact duplicate rows (sensitivity only)", "records": len(duplicate_free), "support_percent": duplicate_free.VOT.eq(1).mean() * 100},
        {"sample": "Known source identities only", "records": len(named), "support_percent": named.VOT.eq(1).mean() * 100},
    ])

    summary = {
        "input": str(data_path.resolve()), "sha256": sha256_file(data_path),
        "records": n, "fields": FIELDS, "parser_issues": len(parse_issues),
        "known_users": len(users), "known_voters": len(voters), "distinct_candidates": len(candidates),
        "distinct_pairs_including_blank_source": len(df[["SRC", "TGT"]].drop_duplicates()),
        "distinct_named_pairs_including_self_votes": len(named[["SRC", "TGT"]].drop_duplicates()),
        "votes": {VOTE_NAMES[k]: int(v) for k, v in vote_counts.items()},
        "vote_percent": {VOTE_NAMES[k]: float(v / n * 100) for k, v in vote_counts.items()},
        "outcome_record_counts": {OUTCOME_NAMES[k]: int(v) for k, v in outcome_counts.items()},
        "quality": {c: int(df[c].sum()) for c in quality_cols},
        "usable_timestamps": int(df.timestamp.notna().sum()),
        "date_min": str(df.timestamp.min()), "date_max": str(df.timestamp.max()),
        "election_start_year_min": int(df.YEA.min()), "election_start_year_max": int(df.YEA.max()),
        "candidate_year_groups": len(candidate_year),
        "candidate_year_groups_with_multiple_outcomes": int(candidate_year.observed_outcomes.gt(1).sum()),
        "candidates_observed_in_multiple_years": int(candidates.active_years.gt(1).sum()),
        "top_one_percent_voters_count": top_count,
        "top_one_percent_voters_share": float(voters.votes_cast.iloc[:top_count].sum() / len(named)),
        "voter_gini_among_active_voters": gini(voters.votes_cast),
        "median_votes_per_voter": float(voters.votes_cast.median()),
        "median_votes_per_candidate": float(candidates.votes_received.median()),
        "graph": {
            "definition": "Named users; directed non-self distinct voter-candidate pairs, including neutral votes",
            "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
            "vote_records": len(graph_rows), "density": nx.density(graph),
            "reciprocity": nx.reciprocity(graph), "weak_components": len(components),
            "largest_weak_component": components[0], "strong_components": len(strong_components),
            "largest_strong_component": strong_components[0], "isolates": nx.number_of_isolates(graph),
            "pairs_with_support_and_oppose": pairs_with_opposite_votes,
            "pairs_with_multiple_records": int(graph_rows.groupby(["SRC", "TGT"]).size().gt(1).sum()),
        },
        "raw_whitespace_tokens_mean": float(df.raw_whitespace_tokens.mean()),
        "raw_whitespace_tokens_median": float(df.raw_whitespace_tokens.median()),
        "cleaned_words_mean": float(df.cleaned_word_count.mean()),
        "cleaned_words_median": float(df.cleaned_word_count.median()),
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__, "networkx": nx.__version__, "scikit-learn": sklearn.__version__},
        "source_documentation": SOURCE_URL,
    }
    export = {
        "vote_distribution": (vote_table, False), "vote_by_outcome_counts": (outcome_table, True),
        "vote_by_outcome_percent": (outcome_pct, True), "annual_summary": (annual, True),
        "data_quality": (quality, False), "data_quality_by_year_percent": (quality_annual, True),
        "invalid_dates": (unresolved, False), "year_mismatch_deltas": (deltas, True),
        "voter_activity": (voters, True), "candidate_activity": (candidates, True),
        "candidate_year_aggregates_NOT_elections": (candidate_year, True),
        "pair_vote_histories": (pair_table, False), "network_degrees": (degree_table, False),
        "comment_length_by_vote": (comment_stats, True), "distinctive_terms": (terms, False),
        "frequent_terms": (top_terms, False), "duplicate_sensitivity": (sensitivity, False),
        "exact_duplicate_examples": (raw.loc[df.exact_duplicate].head(20), False),
        "year_mismatch_examples": (df.loc[df.year_mismatch, FIELDS].head(30), False),
    }
    for name, (frame, index) in export.items():
        write_table(frame, tables / (name + ".csv"), index=index)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"df": df, "summary": summary, "vote_table": vote_table, "outcome_counts": outcome_table,
            "outcome_pct": outcome_pct, "annual": annual, "quality": quality,
            "quality_annual": quality_annual, "voters": voters, "candidates": candidates,
            "degrees": degree_table, "pair_table": pair_table, "comment_stats": comment_stats,
            "terms": terms, "sensitivity": sensitivity, "output_dir": output_dir}


def make_figures(a):
    output = a["output_dir"] / "figures"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "#ffffff",
                         "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff"})
    paths = []

    def save(fig, name):
        fig.savefig(output / (name + ".png"), dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(name + ".png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    order = [1, -1, 0]
    counts = a["df"].VOT.value_counts()
    bars = axes[0].bar([VOTE_NAMES[v] for v in order], [counts[v] for v in order], color=[COLORS[v] for v in order], width=.65)
    axes[0].bar_label(bars, labels=[f"{counts[v]:,}\n({counts[v]/len(a['df']):.1%})" for v in order], padding=5)
    axes[0].set_ylim(0, counts.max() * 1.2)
    axes[0].set_ylabel("Vote records")
    axes[0].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[0].set_title("Support dominates the vote records")
    left = np.zeros(2)
    for v in [1, 0, -1]:
        values = a["outcome_pct"][VOTE_NAMES[v]].values
        axes[1].barh(a["outcome_pct"].index, values, left=left, color=COLORS[v], label=VOTE_NAMES[v])
        for i, value in enumerate(values):
            if value > 5:
                axes[1].text(left[i] + value / 2, i, f"{value:.1f}%", ha="center", va="center", color="white", weight="bold")
        left += values
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Share of votes within the observed outcome")
    axes[1].xaxis.set_major_formatter(PercentFormatter())
    axes[1].set_title("Vote composition by election outcome")
    axes[1].legend(loc="lower center", bbox_to_anchor=(.5, -.32), ncol=3, frameon=False)
    save(fig, "01_votes_and_outcomes")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, constrained_layout=True)
    annual = a["annual"]
    colors = ["#375e8b" if y not in [2003, 2013] else "#9fb6cd" for y in annual.index]
    axes[0].bar(annual.index, annual.records, color=colors)
    axes[0].set_ylabel("Vote records")
    axes[0].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[0].set_title("Recorded voting activity peaks in 2006")
    axes[0].text(.99, .95, "2003 and 2013 are boundary years\n2013 is incomplete", transform=axes[0].transAxes, ha="right", va="top", color="#526171")
    for v in [1, -1, 0]:
        axes[1].plot(annual.index, annual[VOTE_NAMES[v].lower() + "_percent"], marker="o", color=COLORS[v], label=VOTE_NAMES[v])
    axes[1].set_ylabel("Share of records")
    axes[1].yaxis.set_major_formatter(PercentFormatter())
    axes[1].set_xlabel("Election-start year (YEA), as recorded")
    axes[1].set_xticks(annual.index)
    axes[1].legend(ncol=3, frameon=False)
    axes[1].grid(axis="y", alpha=.15)
    save(fig, "02_annual_activity")

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    quality_columns = ["missing_source", "missing_date", "invalid_date", "missing_text", "exact_duplicate", "year_mismatch"]
    matrix = a["quality_annual"][quality_columns].T
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(1, matrix.values.max()))
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), ["Missing voter", "Blank timestamp", "Malformed timestamp", "Empty comment", "Exact duplicate row", "Timestamp year differs"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            label = "<0.1" if 0 < value < .1 else f"{value:.1f}"
            ax.text(j, i, label, ha="center", va="center", color="white" if value > matrix.values.max() * .55 else "#222222", fontsize=9)
    ax.set_title("Data quality varies substantially by recorded election-start year")
    ax.set_xlabel("YEA · cells show percent of that year's records; issues can overlap")
    fig.colorbar(im, ax=ax, label="Percent of yearly records", shrink=.85)
    save(fig, "03_data_quality_by_year")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ccdf(axes[0], a["voters"].votes_cast, "Votes cast per active voter", "#375e8b")
    ccdf(axes[0], a["candidates"].votes_received, "Votes received per candidate", "#208b82")
    axes[0].set_xlabel("Recorded votes (log scale)")
    axes[0].set_ylabel("Fraction with at least this many votes (log)")
    axes[0].set_title("Participation has a long tail")
    axes[0].legend(frameon=False, fontsize=9)
    values = np.sort(a["voters"].votes_cast.values)
    x = np.r_[0, np.arange(1, len(values) + 1) / len(values)]
    y = np.r_[0, np.cumsum(values) / values.sum()]
    axes[1].plot(x, y, color="#375e8b", linewidth=2, label="Observed concentration")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#a4adb5", label="Equal participation")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_xlabel("Cumulative share of active voters, least active first")
    axes[1].set_ylabel("Cumulative share of known-source vote records")
    axes[1].set_title(f"Active-voter Gini = {a['summary']['voter_gini_among_active_voters']:.3f}")
    axes[1].legend(frameon=False, fontsize=9)
    save(fig, "04_participation_concentration")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ccdf(axes[0], a["degrees"].out_degree, "Out-degree: distinct candidates voted on", "#375e8b")
    ccdf(axes[0], a["degrees"].in_degree, "In-degree: distinct voters received", "#208b82")
    axes[0].set_xlabel("Distinct neighbors (log scale)")
    axes[0].set_ylabel("Fraction of positive-degree users at or above (log)")
    axes[0].set_title("Directed network degree distributions")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    pair = a["pair_table"].sort_values("pairs")
    bars = axes[1].barh(pair.vote_values_observed, pair.pairs, color="#697e97")
    axes[1].set_xscale("log")
    axes[1].bar_label(bars, labels=[f"{v:,}" for v in pair.pairs], padding=4, fontsize=9)
    axes[1].set_xlim(1, pair.pairs.max() * 5)
    axes[1].set_xlabel("Distinct named, non-self pairs (log scale)")
    axes[1].set_title("Some pairs have multiple vote labels over time")
    save(fig, "05_network_and_pair_histories")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    df = a["df"]
    comment_order = [1, 0, -1]
    box = axes[0].boxplot(
        [df.loc[df.VOT.eq(v), "cleaned_word_count"] + 1 for v in comment_order],
        tick_labels=[VOTE_NAMES[v] for v in comment_order],
        patch_artist=True, showfliers=False, medianprops={"color": "#222222"},
    )
    for patch, v in zip(box["boxes"], [1, 0, -1]):
        patch.set_facecolor(COLORS[v]); patch.set_alpha(.7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Cleaned word count + 1 (log scale)")
    axes[0].set_title("Comment lengths by vote\nBoxes: IQR; whiskers: 1.5 × IQR; outliers hidden")
    stats = a["comment_stats"].reindex(["Support", "Neutral", "Oppose"])
    bars = axes[1].bar(stats.index, stats.median_words_nonempty, color=[COLORS[v] for v in [1, 0, -1]])
    axes[1].bar_label(bars, labels=[f"{v:.0f}" for v in stats.median_words_nonempty], padding=4)
    axes[1].set_ylim(0, stats.median_words_nonempty.max() * 1.2)
    axes[1].set_ylabel("Median cleaned words, nonempty raw comments")
    axes[1].set_title("Typical length among nonempty comments")
    save(fig, "06_comment_lengths")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    terms = a["terms"]
    for ax, chosen, title in [
        (axes[0], terms.head(12).iloc[::-1], "More prevalent in opposing comments"),
        (axes[1], terms.tail(12), "More prevalent in supporting comments"),
    ]:
        y = np.arange(len(chosen))
        ax.barh(y + .18, chosen.oppose_document_percent, height=.34, color=COLORS[-1], label="Oppose")
        ax.barh(y - .18, chosen.support_document_percent, height=.34, color=COLORS[1], label="Support")
        ax.set_yticks(y, chosen.term)
        ax.set_xlabel("Percent of vote-class comments containing term")
        ax.xaxis.set_major_formatter(PercentFormatter())
        ax.set_title(title)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Exploratory vocabulary contrasts · ≥200 combined support/oppose documents per term\nVote labels and common wiki boilerplate removed; terms ranked by smoothed prevalence ratio", fontsize=11)
    save(fig, "07_comment_vocabulary")
    return paths


def markdown_table(frame, include_index=True, max_rows=None):
    # Avoid adding a tabulate dependency for the portable Markdown report.
    data = frame.head(max_rows).copy() if max_rows else frame.copy()
    if include_index:
        data = data.reset_index()
    if "YEA" in data:
        data["YEA"] = data["YEA"].astype(str)
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            if 0 < abs(value) < .01:
                return f"{value:.4f}"
            return f"{value:,.2f}"
        if isinstance(value, (int, np.integer)):
            return f"{value:,}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    headers = list(map(str, data.columns))
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(fmt(v) for v in row) + " |" for row in data.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_reports(a, figure_names):
    s, df, out = a["summary"], a["df"], a["output_dir"]
    g, quality = s["graph"], s["quality"]
    annual = a["annual"]
    peak = annual.records.idxmax()
    missing_dates_early = a["quality_annual"].loc[2004, "missing_date"]
    raw_duplicate_diff = float(a["sensitivity"].iloc[1].support_percent - a["sensitivity"].iloc[0].support_percent)
    sections = []
    html_sections = []

    def embedded_figure(name, alt):
        encoded = base64.b64encode((out / "figures" / name).read_bytes()).decode("ascii")
        return f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}">'

    def add(title, paragraphs, table=None, figure=None):
        body = "\n\n".join(p.replace(SOURCE_URL, f"[Stanford SNAP dataset documentation]({SOURCE_URL})") for p in paragraphs)
        sections.append("## " + title + "\n\n" + body)
        markup = "<section><h2>" + html.escape(title) + "</h2>" + "".join("<p>" + html.escape(p).replace(SOURCE_URL, f'<a href="{SOURCE_URL}">Stanford SNAP dataset documentation</a>') + "</p>" for p in paragraphs)
        if table is not None:
            sections.append(markdown_table(table))
            markup += table.to_html(border=0, float_format=lambda x: f"{x:,.2f}", classes="data")
        if figure:
            sections.append(f"![{title}](figures/{figure})")
            markup += embedded_figure(figure, title)
        html_sections.append(markup + "</section>")

    add("Scope and main findings", [
        f"The local file contains {s['records']:,} vote records from election-start years 2003–2013, with {s['known_voters']:,} identified voters and {s['distinct_candidates']:,} distinct candidates. There are {s['known_users']:,} distinct nonempty user names across both roles. Each record describes a vote and optional comment; it is not an election-level observation.",
        f"Support accounts for {s['vote_percent']['Support']:.2f}% of records, opposition for {s['vote_percent']['Oppose']:.2f}%, and neutral votes for {s['vote_percent']['Neutral']:.2f}%. Recorded volume peaks in {peak} ({annual.loc[peak, 'records']:,} records). The last year is incomplete.",
        f"Participation is concentrated: the top {s['top_one_percent_voters_count']:,} active voters (the top 1%, rounded up) contribute {s['top_one_percent_voters_share']:.1%} of records with known sources. The median active voter casts {s['median_votes_per_voter']:g} recorded votes; the median candidate receives {s['median_votes_per_candidate']:g} across all observed candidacies.",
        f"Missing voter identities ({quality['missing_source']:,}), missing timestamps ({quality['missing_date']:,}), empty comments ({quality['missing_text']:,}), and exact duplicate records ({quality['exact_duplicate']:,}) require explicit handling. These issues overlap. The file provides no election ID, so election counts and election-level success rates are not directly identifiable.",
    ])
    dictionary = pd.DataFrame({"Field": FIELDS, "Meaning": [
        "Voter user name; blank in some rows", "Candidate user name", "-1 oppose; 0 neutral; 1 support",
        "Election outcome: -1 rejected; 1 accepted", "Election-start year, as documented",
        "Vote timestamp; abbreviated or full English month; timezone unspecified",
        "Vote comment in wiki markup; can be empty",
    ]})
    add("Schema and provenance", [
        "Parsed all records using blank lines as separators and only the first colon as the field separator. All seven fields are present in every record, all categorical codes are valid, and there are no parser anomalies. User names are preserved exactly, including case and punctuation; no identity normalization is inferred.",
        "SNAP's documentation describes 198,275 votes and 11,381 users, but the local file has 11,380 nonempty names; counting the empty string as a user gives 11,381. The local file has 189,003 distinct raw source–target pairs including blank sources, one fewer than the 189,004 documented. These are documentation/local-count differences, not evidence that a particular record should be changed.",
        "Source documentation: " + SOURCE_URL,
        "Input SHA-256: " + s["sha256"],
    ], table=dictionary.set_index("Field"))
    add("Data quality and handling", [
        "All vote records are retained for descriptive vote statistics. Blank voters are excluded from identity-based voter summaries and graph edges; they are never merged into an artificial shared user. Self-votes are reported separately and excluded from graph edges. Repeated voter–candidate pairs remain in vote-level analysis because they can refer to different candidacies.",
        f"There are {quality['exact_duplicate']:,} exact duplicates beyond the first identical record. Removing them as a sensitivity analysis changes the support share by {raw_duplicate_diff:+.4f} percentage points. No records are silently deleted. Exact matching does not establish why duplicates exist.",
        f"Timestamps are parsed with two explicit formats (full and abbreviated English month). {quality['invalid_date']} nonempty strings are malformed and remain unparsed: '14:47, 1 Julu 2012', '20:44, 3 Janry 2010', '31:29, 24 May 2008', and '17:44, 26 Mya 2007'. No dates are guessed or repaired. There are {s['usable_timestamps']:,} usable timestamps spanning {s['date_min']} to {s['date_max']}.",
        f"The parsed timestamp year differs from YEA for {quality['year_mismatch']:,} records. Some differences may reflect year-boundary elections or extraction problems; the file alone does not resolve the cause. YEA is used for annual plots, not overwritten with DAT. Blank timestamps affect {missing_dates_early:.1f}% of 2004 records, so timestamp-only analyses underrepresent early years.",
        "The observed last timestamp is in June 2013, whereas the source page describes coverage through May 2013. Treat the end boundary as partial and retain this discrepancy in provenance notes.",
    ], table=a["quality"].set_index("issue"), figure="03_data_quality_by_year.png")
    add("Vote balance and observed outcomes", [
        f"Among records attached to accepted outcomes, {a['outcome_pct'].loc['Accepted', 'Support']:.2f}% are supportive; among records attached to rejected outcomes, {a['outcome_pct'].loc['Rejected', 'Support']:.2f}% are supportive. This association is descriptive and expected: community votes contribute to the eventual outcome.",
        f"{s['outcome_record_counts']['Accepted']:,} records are associated with accepted outcomes and {s['outcome_record_counts']['Rejected']:,} with rejected outcomes. Their relative frequency weights elections by recorded votes and must not be interpreted as the percentage of elections that succeeded.",
        f"There are {s['candidate_year_groups']:,} candidate–year aggregates, of which {s['candidate_year_groups_with_multiple_outcomes']:,} contain both outcomes. Even aggregates with one outcome may combine multiple candidacies. Candidate names, candidate–year pairs, and contiguous record blocks are not validated election IDs.",
    ], table=a["outcome_counts"], figure="01_votes_and_outcomes.png")
    annual_view = annual[["records", "active_voters", "distinct_candidates", "support_percent", "oppose_percent", "neutral_percent"]]
    add("Temporal patterns", [
        f"Vote volume rises to a maximum in {peak}, then falls in subsequent complete years. The 2006 count is {annual.loc[2006, 'records']:,}; the 2012 count is {annual.loc[2012, 'records']:,}. This describes recorded vote activity, not an election count or the overall size of Wikipedia.",
        "The first and final years are boundary years. In particular, 2013 must not be compared to complete years as a full-year total. Annual percentages are based on YEA and can reflect changes in participants, candidacies, extraction quality, and vote composition. No causal explanation is established by these trends.",
    ], table=annual_view, figure="02_annual_activity.png")
    add("Voter and candidate participation", [
        f"Voting activity is unevenly distributed, with an active-voter Gini coefficient of {s['voter_gini_among_active_voters']:.3f}. The most active identified voter, {a['voters'].index[0]}, has {a['voters'].iloc[0].votes_cast:,.0f} records. Candidate totals combine every candidacy observed for that user.",
        f"{s['candidates_observed_in_multiple_years']:,} candidates appear in more than one recorded election-start year. Repeated source–target pairs and mixed vote histories should therefore be treated as longitudinal observations rather than automatic duplicates. Long-tailed distributions alone do not demonstrate a power law.",
    ], table=a["voters"][["votes_cast", "distinct_candidates", "active_years", "support_percent"]].head(10), figure="04_participation_concentration.png")
    network_table = pd.DataFrame({"Metric": ["Known user nodes", "Distinct directed non-self pairs", "Underlying vote records", "Density", "Reciprocity", "Weak components", "Largest weak component", "Strong components", "Largest strong component", "Isolated named users", "Pairs with both support and opposition"],
        "Value": [f"{g['nodes']:,}", f"{g['edges']:,}", f"{g['vote_records']:,}", f"{g['density']:.6f}", f"{g['reciprocity']:.2%}", str(g['weak_components']), f"{g['largest_weak_component']:,}", str(g['strong_components']), f"{g['largest_strong_component']:,}", str(g['isolates']), f"{g['pairs_with_support_and_oppose']:,}"]})
    add("Directed network structure", [
        "Nodes are all nonempty user names in either role. Edges point from voter to candidate, collapse repeated pairs for topology, include neutral votes, and exclude unknown sources and self-loops. Degree counts measure distinct neighbors, while participation counts measure vote records. These denominators differ intentionally.",
        f"The graph is sparse (density {g['density']:.6f}) and its largest weak component contains {g['largest_weak_component']:,} of {g['nodes']:,} named users. Reciprocity is {g['reciprocity']:.2%}: the fraction of directed edges with a reverse edge. This is voting reciprocity, not agreement or mutual positive sentiment.",
        f"{g['pairs_with_support_and_oppose']:,} distinct named, non-self pairs have both support and opposition somewhere in their history. A signed-graph model should preserve temporal vote labels or define an explicit aggregation rule rather than assigning an unexplained single sign. The file lacks reliable election IDs and complete timestamps, so these histories do not establish within-election vote changes.",
        "SNAP's published graph statistics use a different projection that discards neutral votes. Their graph counts therefore are not directly comparable with this report's inclusive graph.",
    ], table=network_table.set_index("Metric"), figure="05_network_and_pair_histories.png")
    add("Comment lengths and vocabulary", [
        f"On raw whitespace splitting, the median comment length is {s['raw_whitespace_tokens_median']:g} tokens and the mean is {s['raw_whitespace_tokens_mean']:.2f}. After lightweight markup cleaning and alphabetic word tokenization, the median is {s['cleaned_words_median']:g} words and the mean is {s['cleaned_words_mean']:.2f}. These are different token definitions, not comparable measurements of an unchanged tokenizer.",
        "Comment-length summaries include blank comments unless the column explicitly says nonempty. Separate nonempty-comment medians help distinguish writing length from missingness. Markup cleaning is approximate and does not fully render nested templates or MediaWiki syntax.",
        f"Among nonempty comments, the median is {a['comment_stats'].loc['Oppose', 'median_words_nonempty']:g} cleaned words for opposition, {a['comment_stats'].loc['Neutral', 'median_words_nonempty']:g} for neutral votes, and {a['comment_stats'].loc['Support', 'median_words_nonempty']:g} for support. Opposing and neutral comments are typically longer under this tokenization.",
        "Vocabulary analysis uses per-comment term presence, removes English stop words, vote labels, and common wiki boilerplate, and compares support and opposition. Terms must appear in at least 200 support/oppose comments combined. Ranking uses a smoothed prevalence ratio; the chart displays unsmoothed percentages of all comments in each vote class, including blanks.",
        "Term contrasts are exploratory associations. User names, formulaic endorsements, differing comment lengths, repeated participants, and candidacy topics can influence them. They are not causal explanations or validated prediction features.",
    ], table=a["comment_stats"][["records", "empty_comment_percent", "median_words_all", "mean_words_all", "median_words_nonempty"]], figure="06_comment_lengths.png")
    sections.append("![Comment vocabulary contrasts](figures/07_comment_vocabulary.png)")
    html_sections.append('<section><h2>Comment vocabulary contrasts</h2>' + embedded_figure("07_comment_vocabulary.png", "Comment vocabulary contrasts") + '</section>')
    add("Implications for subsequent modeling", [
        "Choose the prediction target before creating features. For vote prediction, using RES would reveal the eventual election outcome, and literal vote words in TXT make a text classifier partly a label-recognition task. Evaluate a text baseline with explicit label masking and inspect remaining boilerplate.",
        "Use chronological splits where reliable timestamps exist and document how missing or conflicting dates are handled. Fit vocabulary, preprocessing, and graph features using training data only. Avoid computing network degree or sentiment aggregates from future votes. Report macro-F1 and per-class precision/recall alongside accuracy because support is the majority class.",
        "A random row split can reuse the same participants, candidates, or repeat candidacies across training and test data. Evaluate candidate-held-out and time-based splits when they match the intended use. Without validated election IDs, do not claim an election-held-out evaluation or train an election-outcome model from candidate–year proxies as if they were exact elections.",
        "This is observational historical data covering a limited period. Names are not normalized identities, unknown voters are unresolved, and outcome labels are known after the voting process. The EDA establishes descriptive patterns, not independent observations, causal effects, or present-day Wikipedia behavior.",
    ])
    add("Reproduce and inspect", [
        "Run python -m votelens.eda from the repository root. The script accepts --data and --output paths and regenerates the report, figures, summary JSON, and CSV tables.",
        "Outputs include report.html, report.md, summary.json, figures/*.png, and tables/*.csv. Table names identify the unit of analysis.",
        "Runtime used: " + "; ".join(f"{key} {value}" for key, value in s["environment"].items()) + ".",
        "Reference for field semantics and documented coverage: " + SOURCE_URL,
    ])
    title = "Wikipedia RfA · Exploratory Data Analysis"
    (out / "report.md").write_text("# " + title + "\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    style = """body{font-family:system-ui,-apple-system,sans-serif;margin:0;color:#243444;background:#f4f7fa;line-height:1.65}main{max-width:1150px;margin:32px auto;background:white;padding:36px 48px;border-radius:16px}h1{font-size:34px;line-height:1.2;color:#173550}h2{margin-top:0;color:#173550;font-size:23px}section{margin:34px 0;padding-top:26px;border-top:1px solid #e3eaf0}p{max-width:100ch}img{max-width:100%;height:auto;margin-top:16px}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto;margin:18px 0;max-width:100%}th,td{padding:8px 12px;text-align:right;border-bottom:1px solid #e2e8ee}th{background:#f0f5f8}td:first-child,th:first-child{text-align:left}.sub{color:#627589}a{color:#167d83}code{word-break:break-all}@media(max-width:700px){main{padding:22px;margin:10px}h1{font-size:28px}}"""
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + title + '</title><style>' + style + '</style><main><h1>' + title + '</h1><p class="sub">198,275 vote records · 2003–2013 · local reproducible analysis</p>' + "".join(html_sections) + f'<footer>Dataset semantics: <a href="{SOURCE_URL}">Stanford SNAP · Wikipedia Requests for Adminship</a></footer></main></html>'
    (out / "report.html").write_text(page, encoding="utf-8")
    return out / "report.html"


def run_analysis(data_path=DEFAULT_DATA, output_dir=None):
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT
    analysis = build_analysis(data_path, output_dir)
    records = analysis["summary"]["records"]
    assert analysis["vote_table"].records.sum() == records
    assert analysis["annual"].records.sum() == records
    assert analysis["outcome_counts"].Total.sum() == records
    assert analysis["voters"].votes_cast.sum() == (~analysis["df"].missing_source).sum()
    assert analysis["candidates"].votes_received.sum() == records
    assert analysis["pair_table"].pairs.sum() == analysis["summary"]["graph"]["edges"]
    figure_names = make_figures(analysis)
    write_reports(analysis, figure_names)
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Local SNAP gzip file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_analysis(args.data or ensure_dataset(), args.output)
    print(json.dumps(result["summary"], indent=2))
    print("Report:", args.output / "report.html")
