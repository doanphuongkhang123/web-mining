"""Fine-tune RoBERTa for vote-level stance classification using the classical split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .stance import (
    DEFAULT_INPUT, DEFAULT_OUTPUT, ID_TO_LABEL, LABEL_NAMES, LABELS, SPLITS,
    chronological_election_split, duplicate_audit, explicit_label_audit, load_stance_data,
)


class WeightedTrainer(Trainer):
    """Trainer with inverse-frequency cross-entropy for the imbalanced stances."""

    def __init__(self, *args, class_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device)
        )
        return (loss, outputs) if return_outputs else loss


def metric_callback(evaluation):
    predictions = np.argmax(evaluation.predictions, axis=-1)
    return {"macro_f1": f1_score(evaluation.label_ids, predictions, average="macro", zero_division=0)}


def to_dataset(frame: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_dict({
        "text": frame["txt_clean"].tolist(),
        "labels": frame["target"].astype(int).tolist(),
        "textid": frame["textid"].astype(int).tolist(),
    })
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True, remove_columns=["text"], desc="Tokenizing comments",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "roberta")
    parser.add_argument("--model-name", default="FacebookAI/roberta-base")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Run this experiment on Colab or Kaggle GPU.")
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    frame = load_stance_data(args.input)
    label_audit = explicit_label_audit(frame)
    if label_audit["residual_leading_header_rows"]:
        raise ValueError("Explicit leading stance labels remain in txt_clean")
    frame, split_audit = chronological_election_split(frame)
    seen, duplicate_summary = duplicate_audit(frame)
    frame["text_seen_in_train"] = seen
    partitions = {split: frame.loc[frame["split"].eq(split)].copy() for split in SPLITS}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label={index: name for index, name in enumerate(LABEL_NAMES)},
        label2id={name: index for index, name in enumerate(LABEL_NAMES)},
    )
    datasets = {
        split: to_dataset(partition, tokenizer, args.max_length)
        for split, partition in partitions.items()
    }
    weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(LABELS)),
        y=partitions["train"]["target"].to_numpy(),
    )
    training_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps", logging_steps=100,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs, warmup_ratio=0.1,
        fp16=True, dataloader_num_workers=2,
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=2, report_to="none", seed=args.seed, data_seed=args.seed,
    )
    trainer = WeightedTrainer(
        model=model, args=training_args, train_dataset=datasets["train"],
        eval_dataset=datasets["validation"], data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer, compute_metrics=metric_callback, class_weights=weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train()
    trainer.save_model(args.output / "model")
    tokenizer.save_pretrained(args.output / "model")

    prediction_output = trainer.predict(datasets["test"])
    prediction_ids = np.argmax(prediction_output.predictions, axis=-1)
    y_test = prediction_output.label_ids
    report = classification_report(
        y_test, prediction_ids, labels=range(len(LABELS)), target_names=LABEL_NAMES,
        output_dict=True, zero_division=0,
    )
    unseen = ~partitions["test"]["text_seen_in_train"].to_numpy()
    result = {
        "model_name": args.model_name,
        "seed": args.seed,
        "max_length": args.max_length,
        "split": split_audit,
        "explicit_label_audit": label_audit,
        "duplicate_audit": duplicate_summary,
        "validation_metrics": trainer.evaluate(datasets["validation"]),
        "test_macro_f1": float(f1_score(y_test, prediction_ids, average="macro", zero_division=0)),
        "test_macro_f1_unseen_text_only": float(
            f1_score(y_test[unseen], prediction_ids[unseen], average="macro", zero_division=0)
        ),
        "test_classification_report": report,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n")

    predictions = partitions["test"][["textid", "model_election_id", "timestamp", "stance",
                                       "text_seen_in_train"]].copy()
    predictions["prediction"] = [ID_TO_LABEL[int(value)] for value in prediction_ids]
    predictions.to_csv(args.output / "test_predictions.csv.gz", index=False)
    print(json.dumps({"test_macro_f1": result["test_macro_f1"],
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
