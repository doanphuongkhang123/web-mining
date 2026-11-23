# Task A experiment: vote-level stance mining

This experiment predicts `stance` (`oppose`, `neutral`, or `support`) using only `txt_clean` from
`data/stance_model.csv`. It never uses voter, candidate, election outcome, or vote metadata as model
features.

## Why preserve election groups?

The prediction unit is still one vote. Election grouping is used only when making the chronological
train/validation/test boundaries. Comments from one live RfA can repeat language, quote each other,
or refer to the same candidate and discussion. Splitting one RfA across a boundary would therefore
make the held-out score less independent. Keeping each election intact also follows the leakage rule
in `overview_and_data_processing.md`.

The model loader removes structural headers such as bold **Support**, **Oppose**, and **Neutral**.
The training command refuses to run if one of those labels is still at the start of a cleaned comment.
Ordinary semantic phrases such as “I cannot support” remain because they are the opinion-bearing text
the task is intended to model.

## Classical experiment

From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m votelens.stance
```

The script uses the oldest 70% of dated elections for training, the next 15% for validation, and the
newest 15% for testing. Elections without any timestamp are excluded and counted in `summary.json`.
The validation set selects `C` from `0.5`, `1.0`, and `2.0`; the test set is evaluated once after
refitting on train plus validation.

Outputs are written to `outputs/rfa_stance/`:

- `report.html` and `summary.json`: experiment summary and full metrics.
- `metrics.csv`: test Macro-F1/accuracy, including sensitivity results after exact train-text overlaps
  are removed.
- `validation_search.csv`: hyperparameter selection evidence.
- `splits.csv.gz` and `test_predictions.csv.gz`: auditable assignments and predictions.
- `confusion_matrix_*.png`: row-normalized test confusion matrices.
- `top_features.csv`: the largest positive class coefficients.
- `*.joblib`: fitted TF-IDF vectorizer and classical estimators.

## RoBERTa experiment (CUDA GPU machine)

The transformer runner reuses the same data validation, chronological election split, label mapping,
class weighting, primary metric, and test-overlap audit as the classical experiment.

Clone or copy the repository to a Linux machine with an NVIDIA GPU and working CUDA driver, then run
from the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-transformer.txt
python -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU not available'; print(torch.cuda.get_device_name(0))"
python -m votelens.stance_transformer \
  --model-name FacebookAI/roberta-base \
  --batch-size 8 \
  --gradient-accumulation 4 \
  --max-length 256 \
  --epochs 3
```

The effective batch size is 32 on one GPU. If memory is limited, reduce `--batch-size` and increase
`--gradient-accumulation` by the same factor. The public checkpoint is ungated, so Hugging Face
authentication is not required. It can be checked before training with:

```sh
hf models info FacebookAI/roberta-base
```

The best validation checkpoint is selected by Macro-F1 with one-epoch early stopping. Results,
predictions, and the saved tokenizer/model are written to `outputs/rfa_stance/roberta/`.
