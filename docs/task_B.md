# Task B: Early Election-Outcome Prediction

## Objective

Predict the final Wikipedia Requests for Adminship result from only the early
part of an election.

The main question is:

> How early can the final RfA outcome be predicted?

The comparison includes:

1. Vote-statistics baseline
2. Text-only model
3. Graph/history model
4. Combined text + graph model
5. RoBERTa text-only model

The target is the final election outcome:

```text
 1 = Approved
-1 = Rejected
```

## Evaluation Protocol

### Unit of analysis

One sample is one election, identified by `model_election_id`. Votes inside an
election are sorted by `vote_order` or `timestamp`.

### Chronological split

Random row or election splitting is not used.

```text
Train:      2004-2007
Validation: 2008
Test:       2009-2012
```

Every vote and comment from one election remains in the same split. The primary
metric is two-class macro-F1. Hyperparameters are selected on validation. The
selected configuration is refit on train + validation and evaluated once on
test.

### Percentage windows

The percentage protocol uses:

```text
10%, 25%, 50%, 75%
```

For an election with `N` timestamped votes:

```text
k = max(1, floor(window_fraction * N))
```

Only the first `k` votes and comments are available to the model.

Percentage windows use the common cohort where `N >= 10`:

- Train: 2,188 elections
- Validation: 429 elections
- Test: 641 elections

These windows are a retrospective benchmark because knowing `N` requires
knowing the final election length.

### Fixed-vote windows

The realistic protocol uses:

```text
First 5 votes
First 10 votes
First 20 votes
```

Fixed windows use the common cohort where `N >= 20`:

- Train: 1,673 elections
- Validation: 361 elections
- Test: 537 elections

Fixed windows are deployment-realistic because the cutoff is known at
prediction time.

## Leakage Prevention

- `outcome` is never an input feature.
- A model at a given cutoff never receives later votes or comments.
- The final election length is not an input feature.
- Historical graph features use only events before the target election starts.
- Text-only models do not receive `stance`, vote counts, voter IDs, candidate
  IDs, outcome, or graph features.
- Explicit stance labels are absent from `txt_clean`.
- Only `text_status = ready` comments inside the allowed prefix are used.
- TF-IDF, scalers, and other learned preprocessing steps are fitted on training
  data only.

For an election starting at time `t_E`, an allowed historical graph edge must
satisfy:

```text
edge_timestamp < t_E
```

## Implemented Models

### 1. Majority-class baseline

Predicts the majority training outcome for every test election.

### 2. Vote-statistics baseline

Uses only the observed prefix:

- support count
- oppose count
- neutral count
- signed-support ratio
- unique-voter count
- elapsed time
- vote rate
- early versus late support-share trend

The baseline classifier is logistic regression.

### 3. TF-IDF text-only

Only comments inside the observation prefix are concatenated and represented
with word bigram TF-IDF. The classifier is logistic regression.

It receives no vote signs.

### 4. Graph/history model

Uses the vote-statistics features plus historical network features such as:

- candidate historical support/oppose volume
- candidate historical signed-support share
- voter historical activity
- voter historical support tendency
- voter-candidate history
- alignment between current votes and prior voter behavior

No comment text is included.

### 5. Text + graph

Concatenates the TF-IDF text representation with the numerical graph features.

### 6. RoBERTa-base text-only

Uses cleaned prefix comments only. Long election text is split into overlapping
chunks of at most 512 tokens. Chunk logits are averaged at the election level.
Validation macro-F1 selects the number of epochs; the selected configuration is
then refit on train + validation and evaluated once on test.

## Final Results

### Percentage windows

All values are test macro-F1 percentages.

| Model | 10% | 25% | 50% | 75% |
|---|---:|---:|---:|---:|
| Majority class | 30.17 | 30.17 | 30.17 | 30.17 |
| Vote statistics | 79.87 | 86.62 | **92.37** | **96.02** |
| TF-IDF text-only | 77.47 | 85.40 | 88.42 | 90.75 |
| RoBERTa-base text-only | 78.63 | 87.27 | 91.39 | 93.07 |
| Graph/history | 80.95 | 87.54 | 91.89 | 95.85 |
| Text + graph | **81.68** | **88.20** | 91.89 | 95.85 |

### Fixed-vote windows

All values are test macro-F1 percentages.

| Model | 5 votes | 10 votes | 20 votes |
|---|---:|---:|---:|
| Majority class | 34.03 | 34.03 | 34.03 |
| Vote statistics | 78.77 | **83.48** | 85.59 |
| TF-IDF text-only | 76.13 | 79.65 | 84.73 |
| RoBERTa-base text-only | 77.73 | 80.18 | 86.58 |
| Graph/history | 77.87 | 83.16 | 86.95 |
| Text + graph | **79.73** | 82.74 | **88.25** |

Compare models within one protocol. Percentage and fixed-vote tables use
different election cohorts.

## Brief Discussion

### Early signed votes are a strong signal

Vote statistics reach 78.77 macro-F1 after only five votes and 96.02 at the 75%
window. A simple support-count model is therefore difficult to beat.

### Text contains useful information

TF-IDF already performs well without vote signs. RoBERTa improves over TF-IDF
in all seven windows, with gains between approximately 0.5 and 3.0 macro-F1
points.

### Historical graph features help only modestly

Graph/history features improve the earliest windows in some settings. However,
historical voter behavior and current signed votes overlap strongly, so the
incremental graph signal is limited.

### Combined gains are window-dependent

Text + graph performs best at 10%, 25%, 5 votes, and 20 votes. At 20 votes, its
point estimate is about 1.3 macro-F1 points above the strongest single model.
Several confidence intervals overlap, so this should not yet be claimed as a
statistically significant improvement.

### Later observation windows are dominated by vote composition

At 50% and 75%, vote statistics performs best. Adding text does not improve the
late-window result. Future work should focus on harder early-window cases and
paired statistical tests.

## Practical Conclusion

Signed early voting is the strongest practical signal in this dataset. Text
adds useful evidence, and RoBERTa improves text-only representation. Fusion
provides small, window-dependent gains rather than a large universal
improvement.

The next useful steps are:

1. Analyze election-level errors for 5 and 20 votes.
2. Perform paired bootstrap or permutation tests.
3. Add calibrated probability fusion and late fusion.
4. Separate strict historical-graph-only features from current-prefix vote
   counts.
5. Report confidence intervals and class-specific errors alongside aggregate
   macro-F1.

The baseline suite is CPU-efficient. GPU compute is needed for RoBERTa but is
not required for TF-IDF, logistic regression, or handcrafted graph features.

## Reproducible Commands

Run all CPU baselines:

```bash
python -m votelens.task_b \
  --data data/election_model.csv \
  --output outputs/task_b \
  --protocol all
```

Prepare RoBERTa inputs without using a GPU:

```bash
python -m votelens.task_b_roberta \
  --data data/election_model.csv \
  --output outputs/task_b_roberta \
  --protocol all \
  --prepare-only
```

Run all RoBERTa windows on GPU:

```bash
python -m votelens.task_b_roberta \
  --data data/election_model.csv \
  --output outputs/task_b_roberta \
  --protocol all \
  --model-name roberta-base \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --max-length 512 \
  --bootstrap 500 \
  --seed 7
```

Completed RoBERTa windows are resumed automatically. Use `--overwrite` only
when a completed window should be retrained.

## Implementation Layout

```text
votelens/task_b.py                       baseline CLI
votelens/task_b_roberta.py               RoBERTa CLI
votelens/outcome/protocol.py             splits and observation windows
votelens/outcome/features.py             election prefixes and features
votelens/outcome/models.py               CPU baselines and metrics
votelens/outcome/transformer.py          RoBERTa fine-tuning
votelens/outcome/pipeline.py             baseline orchestration
votelens/outcome/report.py               baseline HTML report
```

## Result Artifacts

```text
outputs/task_b/results.csv
outputs/task_b/test_predictions.csv
outputs/task_b/report.html
outputs/task_b/metadata.json
outputs/task_b_roberta/roberta_results.csv
outputs/task_b_roberta/roberta_test_predictions.csv
outputs/task_b_roberta/roberta_metadata.json
outputs/task_b_final/all_model_results.csv
```

The final merged table contains 42 model-window rows: six models evaluated at
seven observation windows.
