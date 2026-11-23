"""Fine-tune RoBERTa for Task B election-level text-only prediction."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .features import build_elections, build_samples
from .protocol import protocol_windows


DEFAULT_MODEL_NAME = "roberta-base"


def outcome_to_index(outcome):
    return 0 if int(outcome) == -1 else 1


def read_windows(data_path, protocol, smoke_limit):
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
    windows = protocol_windows(protocol)
    samples = build_samples(elections, windows)
    if smoke_limit:
        selected_counts = Counter()
        selected_samples = []
        for sample in samples:
            key = (sample.window, sample.split)
            if selected_counts[key] < smoke_limit:
                selected_samples.append(sample)
                selected_counts[key] += 1
        samples = selected_samples
    return windows, samples, len(elections)


def bootstrap_f1(actual, predicted, iterations, seed):
    if not iterations:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(iterations):
        indexes = rng.integers(0, len(actual), size=len(actual))
        scores.append(
            f1_score(
                actual[indexes],
                predicted[indexes],
                average="macro",
                zero_division=0,
            )
        )
    low, high = np.quantile(scores, [0.025, 0.975])
    return float(low), float(high)


def metric_row(
    window,
    samples,
    predicted,
    validation_f1,
    best_epoch,
    bootstrap,
    seed,
    model_name,
):
    actual = np.asarray(
        [outcome_to_index(sample.outcome) for sample in samples], dtype=int
    )
    predicted = np.asarray(predicted, dtype=int)
    precision, recall, class_f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=(0, 1), zero_division=0
    )
    low, high = bootstrap_f1(actual, predicted, bootstrap, seed)
    return {
        "window": window,
        "model": model_name.replace("/", "_"),
        "macro_f1": float(
            f1_score(actual, predicted, average="macro", zero_division=0)
        ),
        "macro_f1_low": low,
        "macro_f1_high": high,
        "test_elections": len(actual),
        "rejected_test": int(np.count_nonzero(actual == 0)),
        "approved_test": int(np.count_nonzero(actual == 1)),
        "rejected_precision": float(precision[0]),
        "rejected_recall": float(recall[0]),
        "rejected_f1": float(class_f1[0]),
        "approved_precision": float(precision[1]),
        "approved_recall": float(recall[1]),
        "approved_f1": float(class_f1[1]),
        "validation_macro_f1": float(validation_f1),
        "selected_epoch": int(best_epoch),
    }


def train_epoch(model, loader, optimizer, scheduler, scaler, device, accumulation):
    import torch

    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for step, batch in enumerate(loader):
        labels = batch.pop("labels").to(device)
        weights = batch.pop("weights").to(device)
        batch.pop("sample_indexes")
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(**batch).logits
            chunk_losses = torch.nn.functional.cross_entropy(
                logits, labels, reduction="none"
            )
            loss = (chunk_losses * weights).sum() / weights.sum()
            loss = loss / accumulation
        scaler.scale(loss).backward()
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        losses.append(float(loss.detach().cpu()) * accumulation)
    return float(np.mean(losses))


def predict_epoch(model, loader, device, num_samples):
    import torch

    model.eval()
    logit_sums = np.zeros((num_samples, 2), dtype=np.float64)
    chunk_counts = np.zeros(num_samples, dtype=np.int64)
    actual = np.zeros(num_samples, dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").numpy()
            sample_indexes = batch.pop("sample_indexes").numpy()
            batch.pop("weights")
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(**batch).logits.cpu().numpy()
            np.add.at(logit_sums, sample_indexes, logits)
            np.add.at(chunk_counts, sample_indexes, 1)
            actual[sample_indexes] = labels
    if np.any(chunk_counts == 0):
        raise ValueError("At least one election produced no token chunks")
    return actual, logit_sums.argmax(axis=1)


def build_loaders(tokenizer, train, validation, batch_size, max_length):
    import torch

    class ChunkedElectionDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            label_counts = Counter(outcome_to_index(sample.outcome) for sample in samples)
            class_weights = {
                label: len(samples) / (2 * count)
                for label, count in label_counts.items()
            }
            self.rows = []
            for sample_index, sample in enumerate(samples):
                encoded = tokenizer(
                    sample.text,
                    truncation=True,
                    max_length=max_length,
                    stride=min(64, max(0, max_length // 8)),
                    return_overflowing_tokens=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                label = outcome_to_index(sample.outcome)
                chunk_count = int(encoded["input_ids"].shape[0])
                chunk_weight = class_weights[label] / chunk_count
                for chunk_index in range(chunk_count):
                    self.rows.append(
                        {
                            "input_ids": encoded["input_ids"][chunk_index],
                            "attention_mask": encoded["attention_mask"][chunk_index],
                            "label": label,
                            "sample_index": sample_index,
                            "weight": chunk_weight,
                        }
                    )

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    def collate(batch):
        padded = tokenizer.pad(
            [
                {
                    "input_ids": row["input_ids"],
                    "attention_mask": row["attention_mask"],
                }
                for row in batch
            ],
            padding=True,
            return_tensors="pt",
        )
        padded["labels"] = torch.tensor([row["label"] for row in batch], dtype=torch.long)
        padded["sample_indexes"] = torch.tensor(
            [row["sample_index"] for row in batch], dtype=torch.long
        )
        padded["weights"] = torch.tensor(
            [row["weight"] for row in batch], dtype=torch.float32
        )
        return padded

    train_loader = torch.utils.data.DataLoader(
        ChunkedElectionDataset(train),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    validation_loader = torch.utils.data.DataLoader(
        ChunkedElectionDataset(validation),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    return train_loader, validation_loader


def fit_model(
    model_name,
    train,
    validation,
    device,
    epochs,
    learning_rate,
    batch_size,
    max_length,
    gradient_accumulation,
    warmup_ratio,
    seed,
    num_training_epochs=None,
    select_epoch=True,
):
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader, validation_loader = build_loaders(
        tokenizer, train, validation, batch_size, max_length
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    ).to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    steps_per_epoch = (len(train_loader) + gradient_accumulation - 1) // gradient_accumulation
    total_steps = steps_per_epoch * (num_training_epochs or epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1 = -np.inf
    best_epoch = 0
    validation_scores = []
    requested_epochs = num_training_epochs or epochs
    for epoch in range(1, requested_epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            gradient_accumulation,
        )
        epoch_status = {
            "epoch": epoch,
            "train_loss": train_loss,
            "samples": len(train),
        }
        if select_epoch:
            actual, predicted = predict_epoch(
                model, validation_loader, device, len(validation)
            )
            score = float(
                f1_score(actual, predicted, average="macro", zero_division=0)
            )
            validation_scores.append(score)
            epoch_status["validation_macro_f1"] = score
            if score > best_f1:
                best_f1 = score
                best_epoch = epoch
        print(json.dumps(epoch_status), flush=True)
    if not select_epoch:
        best_epoch = requested_epochs
    return model, tokenizer, best_f1, best_epoch, validation_scores


def run_window(
    window,
    samples,
    model_name,
    output,
    epochs,
    learning_rate,
    batch_size,
    max_length,
    gradient_accumulation,
    warmup_ratio,
    bootstrap,
    seed,
    save_checkpoints,
):
    import torch

    train = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split == "validation"]
    test = [sample for sample in samples if sample.split == "test"]
    if not train or not validation or not test:
        raise ValueError(f"{window.label} is missing one chronological split")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, validation_f1, best_epoch, _ = fit_model(
        model_name,
        train,
        validation,
        device,
        epochs,
        learning_rate,
        batch_size,
        max_length,
        gradient_accumulation,
        warmup_ratio,
        seed,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    final_model, tokenizer, _, _, _ = fit_model(
        model_name,
        train + validation,
        test,
        device,
        best_epoch,
        learning_rate,
        batch_size,
        max_length,
        gradient_accumulation,
        warmup_ratio,
        seed,
        num_training_epochs=best_epoch,
        select_epoch=False,
    )
    _, test_loader = build_loaders(
        tokenizer, test, test, batch_size, max_length
    )
    _, predicted = predict_epoch(final_model, test_loader, device, len(test))
    row = metric_row(
        window.label,
        test,
        predicted,
        validation_f1,
        best_epoch,
        bootstrap,
        seed,
        model_name,
    )
    row.update(
        {
            "train_elections": len(train),
            "validation_elections": len(validation),
            "selected_learning_rate": learning_rate,
            "max_length": max_length,
            "protocol": window.kind,
        }
    )
    predictions = pd.DataFrame(
        [
            {
                "window": sample.window,
                "model": model_name.replace("/", "_"),
                "election_id": sample.election_id,
                "start": sample.start,
                "total_votes": sample.total_votes,
                "observed_votes": sample.observed_votes,
                "outcome": sample.outcome,
                "prediction": 1 if prediction == 1 else -1,
            }
            for sample, prediction in zip(test, predicted)
        ]
    )
    if save_checkpoints:
        checkpoint = output / "checkpoints" / window.label.replace("%", "pct").replace(" ", "_")
        final_model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
    return row, predictions


def prepare_roberta(
    data_path,
    output,
    protocol,
    smoke_limit,
    windows=None,
):
    configured_windows, samples, election_count = read_windows(
        data_path, protocol, smoke_limit
    )
    selected = set(windows) if windows else None
    configured_windows = [
        window for window in configured_windows if selected is None or window.label in selected
    ]
    selected_samples = [
        sample for sample in samples
        if selected is None or sample.window in selected
    ]
    manifest = (
        pd.DataFrame(
            [
                {
                    "window": sample.window,
                    "election_id": sample.election_id,
                    "split": sample.split,
                    "outcome": sample.outcome,
                    "total_votes": sample.total_votes,
                    "observed_votes": sample.observed_votes,
                    "text_characters": len(sample.text),
                }
                for sample in selected_samples
            ]
        )
        .groupby(["window", "split"])
        .agg(
            elections=("election_id", "size"),
            nonempty_text=("text_characters", lambda values: int(values.gt(0).sum())),
            mean_text_characters=("text_characters", "mean"),
        )
        .reset_index()
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / "prepared_windows.csv", index=False)
    return configured_windows, selected_samples, election_count


def run_roberta(
    data_path,
    output,
    protocol="all",
    model_name=DEFAULT_MODEL_NAME,
    windows=None,
    epochs=3,
    learning_rate=2e-5,
    batch_size=4,
    max_length=512,
    gradient_accumulation=4,
    warmup_ratio=0.1,
    bootstrap=500,
    seed=7,
    smoke_limit=None,
    save_checkpoints=False,
    prepare_only=False,
    overwrite=False,
):
    configured_windows, samples, election_count = prepare_roberta(
        data_path, output, protocol, smoke_limit, windows
    )
    if prepare_only:
        metadata = {
            "model": model_name,
            "prepare_only": True,
            "protocol": protocol,
            "selected_windows": [window.label for window in configured_windows],
            "elections_2004_2012": election_count,
        }
        (output / "roberta_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        return pd.DataFrame(), pd.DataFrame(), metadata

    result_path = output / "roberta_results.csv"
    prediction_path = output / "roberta_test_predictions.csv"
    existing_results = (
        pd.read_csv(result_path)
        if not overwrite and result_path.exists()
        else pd.DataFrame()
    )
    existing_predictions = (
        pd.read_csv(prediction_path)
        if not overwrite and prediction_path.exists()
        else pd.DataFrame()
    )
    completed = (
        set(existing_results.window)
        if not existing_results.empty and "window" in existing_results
        else set()
    )
    result_rows = []
    prediction_frames = []
    for index, window in enumerate(configured_windows):
        if window.label in completed:
            print(f"Skipping completed window: {window.label}", flush=True)
            continue
        print(f"\n=== Training RoBERTa for {window.label} ===", flush=True)
        window_samples = [
            sample for sample in samples if sample.window == window.label
        ]
        row, predictions = run_window(
            window,
            window_samples,
            model_name,
            output,
            epochs,
            learning_rate,
            batch_size,
            max_length,
            gradient_accumulation,
            warmup_ratio,
            bootstrap,
            seed + index * 100,
            save_checkpoints,
        )
        result_rows.append(row)
        prediction_frames.append(predictions)
        incremental_results = (
            pd.concat(
                [existing_results, pd.DataFrame(result_rows)], ignore_index=True
            )
            if not existing_results.empty
            else pd.DataFrame(result_rows)
        )
        incremental_predictions = (
            pd.concat(
                [existing_predictions, *prediction_frames], ignore_index=True
            )
            if not existing_predictions.empty
            else pd.concat(prediction_frames, ignore_index=True)
        )
        incremental_results.to_csv(result_path, index=False)
        incremental_predictions.to_csv(prediction_path, index=False)

    results = (
        pd.concat([existing_results, pd.DataFrame(result_rows)], ignore_index=True)
        if not existing_results.empty
        else pd.DataFrame(result_rows)
    )
    predictions = (
        pd.concat([existing_predictions, *prediction_frames], ignore_index=True)
        if not existing_predictions.empty
        else pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    metadata = {
        "model": model_name,
        "source": str(data_path.resolve()),
        "protocol": protocol,
        "selected_windows": [window.label for window in configured_windows],
        "elections_2004_2012": election_count,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": batch_size * gradient_accumulation,
        "max_length": max_length,
        "warmup_ratio": warmup_ratio,
        "bootstrap_iterations": bootstrap,
        "seed": seed,
        "smoke_limit": smoke_limit,
        "overwrite": overwrite,
        "text_only": True,
        "split": {
            "train": "2004-2007",
            "validation": "2008",
            "test": "2009-2012",
        },
    }
    results.to_csv(result_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    (output / "roberta_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return results, predictions, metadata
