"""Train and evaluate classical text-only stance classifiers for Wikipedia RfA votes."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.svm import LinearSVC

from .source import ROOT

DEFAULT_INPUT = ROOT / "data" / "stance_model.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "rfa_stance"
LABELS = (-1, 0, 1)
LABEL_NAMES = ("oppose", "neutral", "support")
LABEL_TO_ID = {-1: 0, 0: 1, 1: 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
REQUIRED_COLUMNS = {
    "textid", "model_election_id", "timestamp", "stance", "txt_clean",
    "eligible_for_stance_model",
}
SPLITS = ("train", "validation", "test")
LEADING_STANCE_LABEL = (
    r"(?i)^\s*(?:(?:strong|weak|mild|qualified|conditional|provisional|tentative|strongly|"
    r"weakly|very|full|fully|absolute|absolutely|complete|total|wholehearted|wholeheartedly|"
    r"unreserved|unreservedly|nom)\s+|(?:uber|über|nom)-)*"
    r"(support|suppport|suport|supprt|upport|supprot|supoort|suppoer|suuport|suppprt|suppot|"
    r"sypport|spport|supportt|suppoort|suupport|suppoprt|shupport|supporto|supporte|suppory|"
    r"s['’]port|oppose|opppose|opppse|opose|oppoose|opposition|neutral|netural|nuetral|"
    r"neutal|neural|neurtal|abstain)\b"
)
LEADING_LABEL_SEPARATOR = r"^\s*(?:[.:,;!'’\-—–]+\s*)*"


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def load_stance_data(path: Path) -> pd.DataFrame:
    """Load and validate the prepared vote-level stance dataset."""
    path = Path(path)
    columns = pd.read_csv(path, nrows=0).columns
    missing = REQUIRED_COLUMNS.difference(columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    frame = pd.read_csv(
        path,
        usecols=sorted(REQUIRED_COLUMNS),
        dtype={"textid": "int64", "model_election_id": "string", "stance": "int8",
               "txt_clean": "string", "eligible_for_stance_model": "boolean"},
        keep_default_na=False,
        na_values={"timestamp": [""]},
    )
    if not frame["textid"].is_unique:
        raise ValueError("textid must be unique")
    if frame["model_election_id"].isna().any() or frame["model_election_id"].str.strip().eq("").any():
        raise ValueError("Every stance row needs model_election_id")
    if not frame["eligible_for_stance_model"].fillna(False).all():
        raise ValueError("Input contains rows that are not eligible for stance modeling")
    if not frame["stance"].isin(LABELS).all():
        raise ValueError("stance must contain only -1, 0, and 1")
    if frame["txt_clean"].isna().any() or frame["txt_clean"].str.strip().eq("").any():
        raise ValueError("Every stance row needs non-empty txt_clean")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["target"] = frame["stance"].map(LABEL_TO_ID).astype("int8")
    frame = sanitize_explicit_stance_headers(frame)
    return frame.sort_values("textid").reset_index(drop=True)


def chronological_election_split(
    frame: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, dict]:
    """Split whole elections chronologically, excluding elections with no usable timestamp."""
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Split fractions must be positive and sum to less than one")

    starts = frame.groupby("model_election_id", sort=False)["timestamp"].min().dropna().sort_values()
    if len(starts) < 3:
        raise ValueError("At least three dated elections are required")

    train_position = min(max(int(len(starts) * train_fraction), 1), len(starts) - 2)
    validation_position = min(
        max(int(len(starts) * (train_fraction + validation_fraction)), train_position + 1),
        len(starts) - 1,
    )
    train_cutoff = starts.iloc[train_position]
    test_cutoff = starts.iloc[validation_position]
    if train_cutoff >= test_cutoff:
        distinct = starts.drop_duplicates()
        train_cutoff = distinct.iloc[min(int(len(distinct) * train_fraction), len(distinct) - 2)]
        test_cutoff = distinct.iloc[min(
            max(int(len(distinct) * (train_fraction + validation_fraction)), 1), len(distinct) - 1,
        )]
    if train_cutoff >= test_cutoff:
        raise ValueError("Could not find strictly ordered chronological cutoffs")

    election_split = pd.Series("test", index=starts.index, dtype="string")
    election_split.loc[starts < test_cutoff] = "validation"
    election_split.loc[starts < train_cutoff] = "train"

    result = frame.copy()
    result["split"] = result["model_election_id"].map(election_split).fillna("excluded_undated")
    split_starts = pd.DataFrame({"start": starts, "split": election_split})
    dated = result.loc[result["split"].isin(SPLITS)]

    election_sets = {
        split: set(dated.loc[dated["split"].eq(split), "model_election_id"])
        for split in SPLITS
    }
    if any(election_sets[left] & election_sets[right]
           for index, left in enumerate(SPLITS) for right in SPLITS[index + 1:]):
        raise AssertionError("An election crossed split boundaries")
    ordered_ranges = {
        split: split_starts.loc[split_starts["split"].eq(split), "start"]
        for split in SPLITS
    }
    if not (ordered_ranges["train"].max() < ordered_ranges["validation"].min()
            and ordered_ranges["validation"].max() < ordered_ranges["test"].min()):
        raise AssertionError("Chronological split order is invalid")

    audit = {
        "strategy": "chronological_whole_elections",
        "train_fraction_target": train_fraction,
        "validation_fraction_target": validation_fraction,
        "train_cutoff": train_cutoff,
        "test_cutoff": test_cutoff,
        "rows": {split: int(result["split"].eq(split).sum())
                 for split in (*SPLITS, "excluded_undated")},
        "elections": {split: int(election_split.eq(split).sum()) for split in SPLITS},
        "excluded_undated_elections": int(
            result.loc[result["split"].eq("excluded_undated"), "model_election_id"].nunique()
        ),
        "class_counts": {
            split: {
                LABEL_NAMES[LABEL_TO_ID[label]]: int(
                    result.loc[result["split"].eq(split), "stance"].eq(label).sum()
                )
                for label in LABELS
            }
            for split in SPLITS
        },
        "election_start_ranges": {
            split: {"min": values.min(), "max": values.max()}
            for split, values in ordered_ranges.items()
        },
    }
    return result, audit


def duplicate_audit(frame: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Mark normalized texts already seen in train and summarize exact overlap."""
    normalized = frame["txt_clean"].str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
    train_text = set(normalized.loc[frame["split"].eq("train")])
    seen = normalized.isin(train_text) & ~frame["split"].eq("train")
    summary = {}
    for split in ("validation", "test"):
        mask = frame["split"].eq(split)
        summary[split] = {
            "rows": int(mask.sum()),
            "rows_with_text_seen_in_train": int((mask & seen).sum()),
            "share_with_text_seen_in_train": float((mask & seen).sum() / max(mask.sum(), 1)),
        }
    return seen, summary


def sanitize_explicit_stance_headers(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove leading vote-direction headers without consulting the target stance."""
    result = frame.copy()
    result["explicit_headers_removed"] = 0
    for _ in range(100):
        headers = result["txt_clean"].str.extract(LEADING_STANCE_LABEL, expand=False)
        matched = headers.notna()
        if not matched.any():
            break
        result.loc[matched, "explicit_headers_removed"] += 1
        result.loc[matched, "txt_clean"] = (
            result.loc[matched, "txt_clean"].str.replace(LEADING_STANCE_LABEL, "", regex=True)
            .str.replace(LEADING_LABEL_SEPARATOR, "", regex=True).str.strip()
        )
    result["explicit_header_masked"] = result["explicit_headers_removed"].gt(0)
    return result


def explicit_label_audit(frame: pd.DataFrame) -> dict:
    """Summarize removed and residual structural vote labels."""
    residual = frame["txt_clean"].str.extract(LEADING_STANCE_LABEL, expand=False).str.casefold()
    masked = frame.get("explicit_header_masked", pd.Series(False, index=frame.index)).fillna(False)
    removed = frame.get("explicit_headers_removed", pd.Series(0, index=frame.index)).fillna(0)
    return {
        "masked_leading_header_rows": int(masked.sum()),
        "masked_header_occurrences": int(removed.sum()),
        "empty_after_masking": int(frame["txt_clean"].str.strip().eq("").sum()),
        "residual_leading_header_rows": int(residual.notna().sum()),
    }


def _metrics(y_true, y_pred) -> dict:
    report = classification_report(
        y_true, y_pred, labels=range(len(LABELS)), target_names=LABEL_NAMES,
        output_dict=True, zero_division=0,
    )
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(report["accuracy"]),
        "per_class": {
            name: {metric: float(report[name][metric]) for metric in ("precision", "recall", "f1-score")}
            | {"support": int(report[name]["support"])}
            for name in LABEL_NAMES
        },
    }


def _build_estimator(model_name: str, c_value: float):
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=c_value, class_weight="balanced", max_iter=2_000, solver="lbfgs",
            random_state=42,
        )
    if model_name == "linear_svm":
        return LinearSVC(C=c_value, class_weight="balanced", random_state=42)
    raise ValueError(f"Unknown model {model_name}")


def _save_confusion(y_true, y_pred, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=range(len(LABELS)), display_labels=LABEL_NAMES,
        normalize="true", values_format=".2f", cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(title)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def _top_features(vectorizer, estimator, model_name: str, count: int = 30) -> pd.DataFrame:
    features = vectorizer.get_feature_names_out()
    rows = []
    for class_id, class_name in enumerate(LABEL_NAMES):
        coefficients = estimator.coef_[class_id]
        indices = np.argsort(coefficients)[-count:][::-1]
        rows.extend({"model": model_name, "class": class_name, "rank": rank,
                     "feature": features[index], "coefficient": float(coefficients[index])}
                    for rank, index in enumerate(indices, 1))
    return pd.DataFrame(rows)


def _write_report(output: Path, summary: dict, metrics_frame: pd.DataFrame):
    test_rows = metrics_frame.loc[metrics_frame["split"].eq("test")].copy()
    test_rows["macro_f1"] = test_rows["macro_f1"].map(lambda value: f"{value:.4f}")
    test_rows["accuracy"] = test_rows["accuracy"].map(lambda value: f"{value:.4f}")
    figures = "\n".join(
        f'<figure><img src="confusion_matrix_{html.escape(model)}.png" '
        f'alt="Normalized confusion matrix for {html.escape(model)}"></figure>'
        for model in summary["selected_models"]
    )
    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wikipedia RfA stance classification</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 0; color: #21334c; background: #f2f5f8; }}
main {{ max-width: 900px; margin: 24px auto; background: white; padding: 24px 32px; }}
table {{ border-collapse: collapse; }} th,td {{ border:1px solid #dce4ec; padding:6px 9px; text-align:left; }}
img {{ width: min(100%, 620px); height: auto; }} code {{ background:#eef2f6; padding:2px 4px; }}
</style><main>
<h1>Vote-level stance classification</h1>
<p>{summary['dataset']['modeled_rows']:,} cleaned comments; target classes are oppose, neutral, and support.</p>
<p>The split is chronological and keeps each reconstructed election intact. The primary metric is Macro-F1.</p>
<p>Leakage audit: {summary['explicit_label_audit']['masked_header_occurrences']:,} leading stance headers
were removed without consulting the target; {summary['explicit_label_audit']['residual_leading_header_rows']}
remain.</p>
<h2>Held-out test results</h2>{test_rows[['model','macro_f1','accuracy']].to_html(index=False, border=0)}
<p>Exact train-text overlap in test: {summary['duplicate_audit']['test']['share_with_text_seen_in_train']:.2%}.
The sensitivity result excludes those rows.</p>
{figures}
<h2>Reproduce</h2><pre><code>python -m votelens.stance</code></pre>
</main></html>"""
    (output / "report.html").write_text(document, encoding="utf-8")


def run_classical(
    input_path: Path, output: Path, c_values: tuple[float, ...] = (0.5, 1.0, 2.0),
    max_features: int = 200_000, save_models: bool = True,
) -> dict:
    frame = load_stance_data(input_path)
    label_audit = explicit_label_audit(frame)
    if label_audit["residual_leading_header_rows"]:
        raise ValueError("Explicit leading stance labels remain in txt_clean")
    frame, split_audit = chronological_election_split(frame)
    seen_train_text, overlap_audit = duplicate_audit(frame)
    frame["text_seen_in_train"] = seen_train_text
    modeled = frame.loc[frame["split"].isin(SPLITS)].copy()

    train = modeled["split"].eq("train")
    validation = modeled["split"].eq("validation")
    test = modeled["split"].eq("test")
    vectorizer = TfidfVectorizer(
        lowercase=True, min_df=2, max_df=0.995, max_features=max_features,
        ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b", dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform(modeled.loc[train, "txt_clean"])
    validation_matrix = vectorizer.transform(modeled.loc[validation, "txt_clean"])
    search_rows = []
    for model_name in ("logistic_regression", "linear_svm"):
        for c_value in c_values:
            estimator = _build_estimator(model_name, c_value)
            estimator.fit(train_matrix, modeled.loc[train, "target"])
            prediction = estimator.predict(validation_matrix)
            search_rows.append({"model": model_name, "C": c_value,
                                "validation_macro_f1": f1_score(
                                    modeled.loc[validation, "target"], prediction,
                                    average="macro", zero_division=0,
                                )})
    search = pd.DataFrame(search_rows)
    best = (
        search.sort_values(["model", "validation_macro_f1", "C"], ascending=[True, False, True])
        .groupby("model", as_index=False).first().set_index("model")
    )

    final_vectorizer = TfidfVectorizer(**vectorizer.get_params())
    development = train | validation
    development_matrix = final_vectorizer.fit_transform(modeled.loc[development, "txt_clean"])
    test_matrix = final_vectorizer.transform(modeled.loc[test, "txt_clean"])
    y_development = modeled.loc[development, "target"]
    y_test = modeled.loc[test, "target"].to_numpy()

    predictions = modeled.loc[test, ["textid", "model_election_id", "timestamp", "stance",
                                     "text_seen_in_train"]].copy()
    metrics_rows = []
    selected_models = ["majority", "logistic_regression", "linear_svm"]
    fitted = {}

    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(np.zeros((len(y_development), 1)), y_development)
    fitted["majority"] = majority
    majority_prediction = majority.predict(np.zeros((len(y_test), 1))).astype("int8")
    all_predictions = {"majority": majority_prediction}

    for model_name in ("logistic_regression", "linear_svm"):
        estimator = _build_estimator(model_name, float(best.loc[model_name, "C"]))
        estimator.fit(development_matrix, y_development)
        fitted[model_name] = estimator
        all_predictions[model_name] = estimator.predict(test_matrix).astype("int8")

    for model_name, prediction in all_predictions.items():
        predictions[f"prediction_{model_name}"] = [ID_TO_LABEL[int(value)] for value in prediction]
        primary = _metrics(y_test, prediction)
        metrics_rows.append({"model": model_name, "split": "test", "subset": "all",
                             "macro_f1": primary["macro_f1"], "accuracy": primary["accuracy"]})
        unseen = ~predictions["text_seen_in_train"].to_numpy()
        sensitivity = _metrics(y_test[unseen], prediction[unseen])
        metrics_rows.append({"model": model_name, "split": "test", "subset": "unseen_text_only",
                             "macro_f1": sensitivity["macro_f1"],
                             "accuracy": sensitivity["accuracy"]})

    metrics_frame = pd.DataFrame(metrics_rows)
    feature_tables = [
        _top_features(final_vectorizer, fitted[name], name)
        for name in ("logistic_regression", "linear_svm")
    ]
    summary = {
        "dataset": {
            "input": str(Path(input_path).resolve()),
            "input_rows": len(frame),
            "modeled_rows": len(modeled),
            "class_counts": {LABEL_NAMES[LABEL_TO_ID[label]]: int(frame["stance"].eq(label).sum())
                             for label in LABELS},
        },
        "split": split_audit,
        "explicit_label_audit": label_audit,
        "duplicate_audit": overlap_audit,
        "vectorizer": {"kind": "word_tfidf", "ngram_range": [1, 2], "min_df": 2,
                       "max_df": 0.995, "max_features": max_features,
                       "vocabulary_size": len(final_vectorizer.vocabulary_)},
        "selection_metric": "validation_macro_f1",
        "selected_models": selected_models,
        "selected_hyperparameters": {
            name: {"C": float(best.loc[name, "C"]),
                   "validation_macro_f1": float(best.loc[name, "validation_macro_f1"])}
            for name in ("logistic_regression", "linear_svm")
        },
        "test_metrics": {
            name: _metrics(y_test, prediction) for name, prediction in all_predictions.items()
        },
        "test_metrics_unseen_text_only": {
            name: _metrics(y_test[~predictions["text_seen_in_train"].to_numpy()],
                           prediction[~predictions["text_seen_in_train"].to_numpy()])
            for name, prediction in all_predictions.items()
        },
    }

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".stance-", dir=output) as temporary:
        stage = Path(temporary)
        split_columns = ["textid", "model_election_id", "timestamp", "stance", "split",
                         "explicit_header_masked", "explicit_headers_removed", "text_seen_in_train"]
        frame[split_columns].to_csv(stage / "splits.csv.gz", index=False)
        search.to_csv(stage / "validation_search.csv", index=False)
        metrics_frame.to_csv(stage / "metrics.csv", index=False)
        predictions.to_csv(stage / "test_predictions.csv.gz", index=False)
        pd.concat(feature_tables, ignore_index=True).to_csv(stage / "top_features.csv", index=False)
        for model_name, prediction in all_predictions.items():
            _save_confusion(y_test, prediction, f"{model_name}: normalized test confusion matrix",
                            stage / f"confusion_matrix_{model_name}.png")
        if save_models:
            joblib.dump(final_vectorizer, stage / "tfidf_vectorizer.joblib", compress=3)
            for model_name, estimator in fitted.items():
                joblib.dump(estimator, stage / f"{model_name}.joblib", compress=3)
        (stage / "summary.json").write_text(
            json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8"
        )
        _write_report(stage, summary, metrics_frame.loc[metrics_frame["subset"].eq("all")])
        for path in stage.iterdir():
            path.replace(output / path.name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--no-save-models", action="store_true")
    args = parser.parse_args()
    summary = run_classical(
        args.input, args.output, tuple(args.c_values), args.max_features, not args.no_save_models,
    )
    compact = {
        "modeled_rows": summary["dataset"]["modeled_rows"],
        "split_rows": summary["split"]["rows"],
        "selected_hyperparameters": summary["selected_hyperparameters"],
        "test_macro_f1": {
            name: values["macro_f1"] for name, values in summary["test_metrics"].items()
        },
        "report": str((args.output / "report.html").resolve()),
    }
    print(json.dumps(compact, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
