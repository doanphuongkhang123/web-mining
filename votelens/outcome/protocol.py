"""Shared Task B protocol definitions and sample records."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


START_YEAR = 2004
END_YEAR = 2012
TRAIN_END = 2008
VALIDATION_END = 2009
PERCENTAGE_WINDOWS = (0.10, 0.25, 0.50, 0.75)
FIXED_WINDOWS = (5, 10, 20)
LABELS = (-1, 1)


@dataclass(frozen=True)
class Window:
    label: str
    kind: str
    value: float


@dataclass
class Election:
    election_id: str
    start: pd.Timestamp
    outcome: int
    rows: pd.DataFrame
    history: dict[str, object]
    split: str


@dataclass
class Sample:
    window: str
    election_id: str
    split: str
    outcome: int
    start: pd.Timestamp
    total_votes: int
    observed_votes: int
    text: str
    vote_features: np.ndarray
    graph_features: np.ndarray


def split_for_year(year):
    if START_YEAR <= year < TRAIN_END:
        return "train"
    if TRAIN_END <= year < VALIDATION_END:
        return "validation"
    if VALIDATION_END <= year <= END_YEAR:
        return "test"
    return "excluded"


def protocol_windows(protocol):
    percentage = [
        Window(f"{int(value * 100)}%", "percentage", value)
        for value in PERCENTAGE_WINDOWS
    ]
    fixed = [
        Window(f"{value} votes", "fixed", value)
        for value in FIXED_WINDOWS
    ]
    if protocol == "percentage":
        return percentage
    if protocol == "fixed":
        return fixed
    return percentage + fixed
