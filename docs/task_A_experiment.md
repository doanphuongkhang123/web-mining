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
python -m unittest discover -s tests
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

## RoBERTa experiment (GPU; run manually)

The transformer runner reuses the same data validation, chronological election split, label mapping,
class weighting, primary metric, and test-overlap audit as the classical experiment.

Authenticate the compute CLI you intend to use. Both commands open an official browser flow:

```sh
kaggle auth login
colab status
```

Verify with `kaggle kernels list --mine --page-size 1` and `colab sessions`. This host currently has
no OS keyring service, so Colab OAuth may ask again on later commands. It still works, but persistent
login requires either a desktop Secret Service keyring or Google Application Default Credentials; do
not install a plaintext token backend merely to suppress the prompt.

On the GPU machine, copy or clone the repository, then run:

```sh
python -m pip install -r requirements-transformer.txt
python -m votelens.stance_transformer \
  --model-name FacebookAI/roberta-base \
  --batch-size 16 \
  --gradient-accumulation 2 \
  --max-length 256 \
  --epochs 3
```

If a T4 runs out of memory, reduce `--batch-size` to `8` and increase
`--gradient-accumulation` to `4`. The effective batch size remains 32. The public checkpoint is
ungated, so Hugging Face authentication is not required. You can verify it before training with:

```sh
hf models info FacebookAI/roberta-base
```

The best validation checkpoint is selected by Macro-F1 with one-epoch early stopping. Results,
predictions, and the saved tokenizer/model are written to `outputs/rfa_stance/roberta/`.

### Run persistently on Kaggle

This route uploads a private, checksummed copy of the required code and `stance_model.csv`, then
starts a private Kaggle Kernel on GPU T4 ×2. Once `kaggle kernels push` confirms submission, the run
is executing in Kaggle's cloud and the local laptop can be shut down.

Authenticate once and verify the account:

```sh
kaggle auth login
kaggle kernels list --mine --page-size 1
```

Prepare the upload directories using your Kaggle username:

```sh
python scripts/prepare_kaggle_roberta.py --owner YOUR_KAGGLE_USERNAME
```

Create the private input Dataset and wait until its status is ready:

```sh
kaggle datasets create -p kaggle_upload/dataset
kaggle datasets status YOUR_KAGGLE_USERNAME/rfa-task-a-bundle
```

Submit the GPU Kernel. The command uploads and starts it; it does not need to remain attached to the
terminal afterward:

```sh
kaggle kernels push -p kaggle_upload/kernel --accelerator NvidiaTeslaT4
```

After Kaggle confirms the push, the laptop may be turned off. Check it later and download the saved
model, metrics, and predictions with:

```sh
kaggle kernels status YOUR_KAGGLE_USERNAME/rfa-task-a-roberta
kaggle kernels logs YOUR_KAGGLE_USERNAME/rfa-task-a-roberta
kaggle kernels files YOUR_KAGGLE_USERNAME/rfa-task-a-roberta
kaggle kernels output YOUR_KAGGLE_USERNAME/rfa-task-a-roberta \
  -p outputs/kaggle-roberta
```

For a later rerun after changing code or data, rebuild the package and use `kaggle datasets version`
instead of `kaggle datasets create`:

```sh
kaggle datasets version -p kaggle_upload/dataset -m "Update Task A experiment"
kaggle kernels push -p kaggle_upload/kernel --accelerator NvidiaTeslaT4
```

### Run through the Colab CLI

The CLI sends only the runner file to Colab, so first commit and push this Task A implementation to
a branch. Replace `YOUR_BRANCH` below with that pushed branch name:

```sh
colab run --gpu T4 --session rfa-stance --keep --timeout 0 \
  scripts/run_roberta_colab.py \
  --ref YOUR_BRANCH
```

The wrapper clones the branch on a fresh VM, installs the transformer requirements, trains the model,
and creates `/content/rfa_stance_roberta.tar.gz`. Because `--keep` leaves the VM running, download the
archive and then release the session:

```sh
colab download -s rfa-stance \
  /content/rfa_stance_roberta.tar.gz \
  outputs/rfa_stance_roberta.tar.gz
colab stop -s rfa-stance
```

Do not stop the session before downloading: files on an ephemeral Colab VM are not persistent.
