"""Orchestrate Task B data preparation, training, and output."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .features import (
    GRAPH_FEATURE_NAMES,
    VOTE_FEATURE_NAMES,
    build_elections,
    build_samples,
)
from .models import DEFAULT_C_VALUES, run_protocol
from .protocol import protocol_windows
from .report import write_report


def run_task_b(
    data_path,
    output,
    protocol="all",
    c_values=DEFAULT_C_VALUES,
    max_features=40000,
    bootstrap_iterations=500,
    seed=7,
    smoke_limit=None,
):
    frame = pd.read_csv(
        data_path,
        usecols=[
            "textid",
            "model_election_id",
            "timestamp",
            "vote_order",
            "voter",
            "candidate",
            "stance",
            "outcome",
            "txt_clean",
            "text_status",
        ],
    )
    elections = build_elections(frame)
    if smoke_limit:
        limited = []
        for split in ("train", "validation", "test"):
            limited.extend(
                [election for election in elections if election.split == split][
                    :smoke_limit
                ]
            )
        elections = limited
    windows = protocol_windows(protocol)
    samples = build_samples(elections, windows)
    result_frames = []
    prediction_frames = []
    for index, window in enumerate(windows):
        window_samples = [sample for sample in samples if sample.window == window.label]
        results, predictions = run_protocol(
            window_samples,
            window.label,
            c_values,
            max_features,
            bootstrap_iterations,
            seed + index * 100,
        )
        results["protocol"] = window.kind
        result_frames.append(results)
        prediction_frames.append(predictions)
    results = pd.concat(result_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    metadata = {
        "source": str(data_path.resolve()),
        "protocol": protocol,
        "elections_2004_2012": len(elections),
        "percentage_cohort_min_votes": 10,
        "fixed_cohort_min_votes": 20,
        "split": {
            "train": "2004-2007",
            "validation": "2008",
            "test": "2009-2012",
        },
        "metric": "two-class macro-F1",
        "c_values": list(c_values),
        "tfidf_max_features": max_features,
        "bootstrap_iterations": bootstrap_iterations,
        "smoke_limit": smoke_limit,
        "vote_features": list(VOTE_FEATURE_NAMES),
        "graph_features": list(GRAPH_FEATURE_NAMES),
    }

    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".task-b-", dir=output) as temp:
        stage = Path(temp)
        results.to_csv(stage / "results.csv", index=False)
        predictions.to_csv(stage / "test_predictions.csv", index=False)
        (stage / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        write_report(stage, results, metadata)
        for path in stage.iterdir():
            path.replace(output / path.name)
    return results, metadata
