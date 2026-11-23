"""Train and evaluate Task B baseline model families."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from .protocol import LABELS


DEFAULT_C_VALUES = (0.01, 0.1, 1.0, 10.0)


def vectorize_text(train_text, validation_text, test_text, max_features):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    train_matrix = vectorizer.fit_transform(train_text)
    validation_matrix = vectorizer.transform(validation_text)
    fit_matrix = vectorizer.fit_transform(train_text + validation_text)
    test_matrix = vectorizer.transform(test_text)
    return train_matrix, validation_matrix, fit_matrix, test_matrix


def scale_numeric(train_values, validation_values, test_values):
    scaler = StandardScaler()
    train_matrix = csr_matrix(scaler.fit_transform(train_values))
    validation_matrix = csr_matrix(scaler.transform(validation_values))
    refit_scaler = StandardScaler()
    fit_values = np.vstack([train_values, validation_values])
    fit_matrix = csr_matrix(refit_scaler.fit_transform(fit_values))
    test_matrix = csr_matrix(refit_scaler.transform(test_values))
    return train_matrix, validation_matrix, fit_matrix, test_matrix


def logistic_model(c_value):
    return LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
    )


def tune_and_predict(
    train_matrix,
    validation_matrix,
    fit_matrix,
    test_matrix,
    train_targets,
    validation_targets,
    fit_targets,
    c_values,
):
    best_score = -np.inf
    best_c = c_values[0]
    for c_value in c_values:
        model = logistic_model(c_value)
        model.fit(train_matrix, train_targets)
        prediction = model.predict(validation_matrix)
        score = f1_score(
            validation_targets,
            prediction,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
        if score > best_score:
            best_score = score
            best_c = c_value
    final_model = logistic_model(best_c)
    final_model.fit(fit_matrix, fit_targets)
    return final_model.predict(test_matrix), best_c, float(best_score)


def metric_row(
    model_name,
    window,
    samples,
    prediction,
    bootstrap_iterations,
    seed,
):
    actual = np.asarray([sample.outcome for sample in samples], dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    score = f1_score(
        actual, prediction, labels=LABELS, average="macro", zero_division=0
    )
    precision, recall, class_f1, _ = precision_recall_fscore_support(
        actual, prediction, labels=LABELS, zero_division=0
    )
    rng = np.random.default_rng(seed)
    bootstrap = []
    if bootstrap_iterations > 0 and len(actual):
        for _ in range(bootstrap_iterations):
            indexes = rng.integers(0, len(actual), size=len(actual))
            bootstrap.append(
                f1_score(
                    actual[indexes],
                    prediction[indexes],
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                )
            )
    low, high = (
        tuple(float(value) for value in np.quantile(bootstrap, [0.025, 0.975]))
        if bootstrap
        else (np.nan, np.nan)
    )
    return {
        "window": window,
        "model": model_name,
        "macro_f1": float(score),
        "macro_f1_low": low,
        "macro_f1_high": high,
        "test_elections": len(actual),
        "rejected_test": int(np.count_nonzero(actual == -1)),
        "approved_test": int(np.count_nonzero(actual == 1)),
        "rejected_precision": float(precision[0]),
        "rejected_recall": float(recall[0]),
        "rejected_f1": float(class_f1[0]),
        "approved_precision": float(precision[1]),
        "approved_recall": float(recall[1]),
        "approved_f1": float(class_f1[1]),
    }


def run_protocol(
    samples,
    window,
    c_values,
    max_features,
    bootstrap_iterations,
    seed,
):
    train = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split == "validation"]
    test = [sample for sample in samples if sample.split == "test"]
    if not train or not validation or not test:
        raise ValueError(f"Window {window} does not have all chronological splits")

    train_targets = np.asarray([sample.outcome for sample in train])
    validation_targets = np.asarray([sample.outcome for sample in validation])
    fit_targets = np.concatenate([train_targets, validation_targets])
    rows = []
    predictions = []

    train_majority = int(pd.Series(train_targets).mode().iloc[0])
    majority_prediction = np.full(len(test), train_majority, dtype=int)
    rows.append(
        metric_row(
            "majority_class",
            window,
            test,
            majority_prediction,
            bootstrap_iterations,
            seed,
        )
    )
    predictions.extend(
        ("majority_class", sample, int(prediction))
        for sample, prediction in zip(test, majority_prediction)
    )

    vote_train = np.vstack([sample.vote_features for sample in train])
    vote_validation = np.vstack([sample.vote_features for sample in validation])
    vote_test = np.vstack([sample.vote_features for sample in test])
    vote_matrices = scale_numeric(vote_train, vote_validation, vote_test)
    vote_prediction, vote_c, vote_score = tune_and_predict(
        *vote_matrices, train_targets, validation_targets, fit_targets, c_values
    )
    row = metric_row(
        "vote_statistics",
        window,
        test,
        vote_prediction,
        bootstrap_iterations,
        seed + 1,
    )
    row.update(validation_macro_f1=vote_score, selected_c=vote_c)
    rows.append(row)
    predictions.extend(
        ("vote_statistics", sample, int(prediction))
        for sample, prediction in zip(test, vote_prediction)
    )

    train_text = [sample.text for sample in train]
    validation_text = [sample.text for sample in validation]
    test_text = [sample.text for sample in test]
    text_matrices = vectorize_text(
        train_text, validation_text, test_text, max_features
    )
    text_prediction, text_c, text_score = tune_and_predict(
        *text_matrices, train_targets, validation_targets, fit_targets, c_values
    )
    row = metric_row(
        "text_only", window, test, text_prediction, bootstrap_iterations, seed + 2
    )
    row.update(validation_macro_f1=text_score, selected_c=text_c)
    rows.append(row)
    predictions.extend(
        ("text_only", sample, int(prediction))
        for sample, prediction in zip(test, text_prediction)
    )

    graph_train = np.hstack(
        [vote_train, np.vstack([sample.graph_features for sample in train])]
    )
    graph_validation = np.hstack(
        [
            vote_validation,
            np.vstack([sample.graph_features for sample in validation]),
        ]
    )
    graph_test = np.hstack(
        [vote_test, np.vstack([sample.graph_features for sample in test])]
    )
    graph_matrices = scale_numeric(graph_train, graph_validation, graph_test)
    graph_prediction, graph_c, graph_score = tune_and_predict(
        *graph_matrices, train_targets, validation_targets, fit_targets, c_values
    )
    row = metric_row(
        "graph_only",
        window,
        test,
        graph_prediction,
        bootstrap_iterations,
        seed + 3,
    )
    row.update(validation_macro_f1=graph_score, selected_c=graph_c)
    rows.append(row)
    predictions.extend(
        ("graph_only", sample, int(prediction))
        for sample, prediction in zip(test, graph_prediction)
    )

    combined_matrices = tuple(
        hstack([text_matrix, graph_matrix], format="csr")
        for text_matrix, graph_matrix in zip(text_matrices, graph_matrices)
    )
    combined_prediction, combined_c, combined_score = tune_and_predict(
        *combined_matrices,
        train_targets,
        validation_targets,
        fit_targets,
        c_values,
    )
    row = metric_row(
        "text_and_graph",
        window,
        test,
        combined_prediction,
        bootstrap_iterations,
        seed + 4,
    )
    row.update(validation_macro_f1=combined_score, selected_c=combined_c)
    rows.append(row)
    predictions.extend(
        ("text_and_graph", sample, int(prediction))
        for sample, prediction in zip(test, combined_prediction)
    )

    prediction_frame = pd.DataFrame(
        [
            {
                "window": sample.window,
                "model": model_name,
                "election_id": sample.election_id,
                "start": sample.start,
                "total_votes": sample.total_votes,
                "observed_votes": sample.observed_votes,
                "outcome": sample.outcome,
                "prediction": prediction,
            }
            for model_name, sample, prediction in predictions
        ]
    )
    for row in rows:
        row["train_elections"] = len(train)
        row["validation_elections"] = len(validation)
    return pd.DataFrame(rows), prediction_frame
