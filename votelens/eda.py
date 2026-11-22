"""Analyze Wikipedia RfA counts, support shares, and comment toxicity."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .preprocess import ELECTION_COLUMNS
from .source import OUTPUT_DIR, scored_path

DEFAULT_OUTPUT = OUTPUT_DIR / "eda"
YEARS = tuple(range(2004, 2013))
RESULTS = {1: ("Approved", "#39779f"), -1: ("Rejected", "#b95b4e")}
VOTES = {1: ("Support", "#39779f"), 0: ("Neutral", "#828a98"), -1: ("Oppose", "#b95b4e")}
COLUMNS = ["textid", "tgt", "yea", "res", "vote_start", "src", "vot", "txt", "toxicity"]
FIGURES = {
    "01_rfas_by_year.png": "Number of requests for adminship by outcome",
    "02_support_share.png": "Share of support among support and oppose votes",
    "03_toxicity_by_outcome.png": "Comment toxicity by election outcome",
    "04_toxicity_by_vote.png": "Comment toxicity by vote type",
}


def read_scored_votes(path):
    votes = pd.read_csv(
        path, usecols=COLUMNS, keep_default_na=False,
        na_values={"vote_start": ["", "NA", "NaN"], "toxicity": ["", "NA", "NaN"]},
    )
    for column in ("textid", "yea", "res", "vot", "vote_start", "toxicity"):
        votes[column] = pd.to_numeric(votes[column], errors="raise")
    scores = votes.toxicity.dropna()
    if (not votes.textid.is_unique or not votes.res.isin(RESULTS).all()
            or not votes.vot.isin(VOTES).all() or not scores.between(0, 1).all()):
        raise ValueError("Invalid row ID, vote, outcome, or toxicity score in scored data")
    # The scored input stores vote_start as calendar days since 1970.
    votes["vote_start"] = pd.to_datetime(votes.vote_start, unit="D", origin="unix")
    return votes


def summarize(votes):
    analysis = votes.loc[votes.yea.between(YEARS[0], YEARS[-1])]
    if analysis.empty:
        raise ValueError("The scored data has no votes in 2004–2012")

    elections = analysis.groupby(ELECTION_COLUMNS, dropna=False).size().reset_index(name="votes")
    annual = elections.groupby(["yea", "res"]).size().reset_index(name="rfas")

    signed = analysis.loc[analysis.vot.ne(0)]
    support = signed.groupby(ELECTION_COLUMNS, dropna=False).vot.agg(
        signed_votes="size", support_votes=lambda values: values.eq(1).sum(),
    ).reset_index()
    # Keep RfAs with a support row after counting both signed categories.
    support = support.loc[support.support_votes.gt(0)].copy()
    support["support_share"] = support.support_votes / support.signed_votes
    medians = support.groupby(["yea", "res"]).support_share.median().reset_index(name="median_support_share")

    scored = analysis.dropna(subset=["toxicity"])
    sample = analysis.loc[analysis.txt.str.len().lt(100), ["yea", "src", "tgt", "vot", "txt"]].head(5)
    tables = {
        "sample_votes.csv": sample,
        "rfas_by_year.csv": annual,
        "support_share_by_rfa.csv": support,
        "median_support_share.csv": medians,
    }
    for category, name in (("res", "outcome"), ("vot", "vote")):
        tables[f"toxicity_by_{name}.csv"] = scored.groupby(["yea", category]).toxicity.agg(
            comments="size", median="median", mean="mean",
        ).reset_index()
    summary = {
        "scored_input_records": len(votes),
        "first_year": YEARS[0],
        "last_year": YEARS[-1],
        "votes_in_period": len(analysis),
        "inferred_rfas_in_period": len(elections),
        "approved_rfas": int(elections.res.eq(1).sum()),
        "rejected_rfas": int(elections.res.eq(-1).sum()),
        "rfas_in_support_share_chart": len(support),
        "scored_comments_in_period": len(scored),
        "median_toxicity_by_outcome": {
            RESULTS[key][0]: float(values.median()) for key, values in scored.groupby("res").toxicity
        },
        "median_toxicity_by_vote": {
            VOTES[key][0]: float(values.median()) for key, values in scored.groupby("vot").toxicity
        },
    }
    return tables, scored, summary


def finish_chart(fig, ax, path):
    ax.set_xticks(YEARS)
    ax.set_xlim(YEARS[0] - 0.45, YEARS[-1] + 0.45)
    ax.set_xlabel("Year")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e3e8ee")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_elections(annual, path):
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    for key, (label, color) in RESULTS.items():
        by_year = annual.loc[annual.res.eq(key)]
        ax.plot(by_year.yea, by_year.rfas, marker="o", color=color, label=label)
    ax.set(title=FIGURES[path.name], ylabel="Requests for adminship", ylim=(0, None))
    ax.legend(frameon=False)
    finish_chart(fig, ax, path)


def plot_support(support, medians, path):
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    rng = np.random.default_rng(7)
    for key, (label, color) in RESULTS.items():
        rows = support.loc[support.res.eq(key)]
        grouped = medians.loc[medians.res.eq(key)]
        jitter = rng.uniform(-0.19, 0.19, len(rows))
        ax.scatter(rows.yea + jitter, rows.support_share, s=16, color=color, alpha=0.22, linewidth=0)
        ax.scatter(grouped.yea, grouped.median_support_share, s=48, color=color, edgecolors="white", label=label)
    ax.set(title=FIGURES[path.name], ylabel="Support share", ylim=(0, 1.04))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle="", marker="o", color="#666666", markersize=5))
    labels.append("Larger dots show medians")
    ax.legend(handles, labels, frameon=False, loc="lower right")
    finish_chart(fig, ax, path)


def plot_toxicity(scored, field, categories, path):
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    offsets = np.linspace(-0.24, 0.24, len(categories))
    legend = []
    for (key, (label, color)), offset in zip(categories.items(), offsets):
        by_year = scored.loc[scored[field].eq(key)].groupby("yea").toxicity
        samples = [(year, values.to_numpy()) for year, values in by_year if len(values)]
        if samples:
            years, scores = zip(*samples)
            ax.boxplot(
                scores, positions=np.array(years) + offset, widths=0.17 if len(categories) == 3 else 0.24,
                manage_ticks=False, patch_artist=True,
                boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.82},
                medianprops={"color": "white", "linewidth": 1.4},
                whiskerprops={"color": color}, capprops={"color": color},
                flierprops={"marker": ".", "markersize": 2, "markerfacecolor": color,
                            "markeredgecolor": color, "alpha": 0.15},
            )
        legend.append(Patch(facecolor=color, label=label))
    ax.set(title=FIGURES[path.name], ylabel="TOXICITY score", ylim=(-0.012, 1.012))
    ax.legend(handles=legend, frameon=False)
    finish_chart(fig, ax, path)


def write_report(output, sample, summary):
    charts = "\n".join(
        f'<figure><img src="{file}" alt="{description}"></figure>' for file, description in FIGURES.items()
    )
    html = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wikipedia RfA comment toxicity</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 0; color: #21334c; background: #f2f5f8; }}
main {{ max-width: 920px; margin: 24px auto; background: white; padding: 24px 32px; }}
figure {{ margin: 24px 0; }}
img {{ width: 100%; height: auto; }}
table {{ border-collapse: collapse; font-size: 0.9rem; }}
th, td {{ border: 1px solid #dce4ec; padding: 6px 9px; text-align: left; }}
@media (max-width: 600px) {{ main {{ margin: 0; padding: 16px; }} }}
</style>
<main>
<h1>Wikipedia RfA comment toxicity</h1>
<p>2004–2012 · {summary['inferred_rfas_in_period']:,} inferred RfAs ·
{summary['scored_comments_in_period']:,} comments with toxicity scores</p>
<h2>Example vote records</h2>
{sample.to_html(index=False, border=0)}
{charts}
</main>
</html>
"""
    (output / "report.html").write_text(html, encoding="utf-8")


def run_eda(scored_data, output):
    votes = read_scored_votes(scored_data)
    tables, scored, summary = summarize(votes)
    output.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".eda-", dir=output) as temp:
        stage = Path(temp)
        for name, frame in tables.items():
            frame.to_csv(stage / name, index=False, date_format="%Y-%m-%d")
        plot_elections(tables["rfas_by_year.csv"], stage / "01_rfas_by_year.png")
        plot_support(tables["support_share_by_rfa.csv"], tables["median_support_share.csv"],
                     stage / "02_support_share.png")
        plot_toxicity(scored, "res", RESULTS, stage / "03_toxicity_by_outcome.png")
        plot_toxicity(scored, "vot", VOTES, stage / "04_toxicity_by_vote.png")
        write_report(stage, tables["sample_votes.csv"], summary)
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for path in stage.iterdir():
            path.replace(output / path.name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", type=Path, help="Local ZIP or CSV containing toxicity scores")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run_eda(scored_path(args.scored), args.output), indent=2))


if __name__ == "__main__":
    main()
