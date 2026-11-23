# Wikipedia Requests for Adminship Web Mining

This project analyzes discussion and voting data from Wikipedia Requests for
Adminship (RfA). Its goals are to study support, opposition, and neutrality in
comments, analyze signed voting relationships, and predict election outcomes
from information available early in each election.

## Data in the Repository

The repository provides three files for downstream tasks:

| File | Purpose |
|---|---|
| `data/stance_model.csv` | Comment-level data for stance classification |
| `data/election_model.csv` | Timestamped vote-level data for election-outcome prediction |
| `data/rfa_modeling.csv.summary.json` | Row counts, schema, text statuses, source checksums, and grouping metadata |

The two CSV files have been cleaned, joined, and assigned a unified
`model_election_id`. Each row represents one vote and, when available, its
associated comment.

## Main Columns

| Column | Meaning |
|---|---|
| `textid` | Unique ID for a vote/comment record from the SNAP data |
| `model_election_id` | Unified election ID after data integration |
| `election_id_source` | Indicates whether the election ID came from the graph or the seven-day grouping rule |
| `graph_election_id` | Election ID from the graph; may be empty when a direct graph match is unavailable |
| `timestamp` | Vote time in UTC |
| `timestamp_available` | Indicates whether the vote has a usable timestamp |
| `vote_order` | Position of the vote within its election |
| `voter` | User who cast the vote |
| `candidate` | Candidate requesting adminship |
| `stance` | Vote stance: `1` support, `0` neutral, `-1` oppose |
| `outcome` | Election outcome: `1` approved, `-1` rejected |
| `txt_clean` | Comment after removing vote labels, signatures, and wikitext |
| `text_status` | `ready`, `needs_review`, `empty`, or `symbols_only` |
| `text_flags` | Cleaning actions or warnings recorded during text preprocessing |
| `eligible_for_stance_model` | Indicates whether the row has usable clean text for stance modeling |
| `eligible_for_election_model` | Indicates whether the row has a timestamp, election ID, and outcome |

## Stance Dataset

Use `data/stance_model.csv`.

- Rows: 170,540.
- Input: `txt_clean`.
- Target: `stance`.
- Every row has `eligible_for_stance_model = True`.
- Every comment has `text_status = ready`.

Stance labels:

```text
 1 = Support
 0 = Neutral
-1 = Oppose
```

Recommended baseline:

```text
txt_clean -> TF-IDF -> Logistic Regression or Linear SVM -> stance
```

The primary metric is three-class macro-F1. Accuracy alone is insufficient
because the stance classes are imbalanced, particularly the Neutral class.

## Election Dataset

Use `data/election_model.csv`.

- Rows: 188,904.
- Every row has a timestamp, election ID, and outcome.
- Every row has `eligible_for_election_model = True`.
- Use `vote_order` to construct prefixes containing the first 5, 10, or 20
  votes.

To predict the election outcome:

1. Group rows by `model_election_id`.
2. Sort by `vote_order` or `timestamp`.
3. Keep only the prefix observed at the prediction time.
4. Build features from text, signed votes, or both.
5. Use `outcome` as the target.

Compare at least:

- Text-only model.
- Graph/vote-only model.
- Combined model.
- Majority-class baseline.
- Early signed-support-ratio baseline.

## Chronological Train/Validation/Test Split

Do not use a random split. A realistic forecasting system can only use past
elections to predict future elections.

Recommended split:

```text
Train:      2004-2007
Validation: 2008
Test:       2009-2012
```

All votes and comments from the same election must remain in the same split.
The primary metric for election-outcome prediction is two-class macro-F1.

## Leakage Prevention

- Never use `outcome` as an input feature.
- Remove explicit `Support`, `Oppose`, and `Neutral` labels from model text.
- Do not use votes occurring after a prediction point.
- Do not use the final election length to construct an early-window feature.
- Build graph features only from elections that occurred before the target
  election.
- Fit TF-IDF, scalers, and other learned preprocessing steps only on the
  training split.

## Election IDs

Election IDs were consolidated as follows:

1. When a vote matches the graph through `textid = record_id`, use the graph's
   `election_id`.
2. Propagate that graph ID to every vote in the same election, including neutral
   votes.
3. For elections without a graph match, use the inferred key:

```text
candidate + year + outcome + vote_start
```

`outcome` is used to distinguish historical inferred groups. During modeling,
it must be stored separately as the target and must never be used as an input
feature.

## Missing Values

Some fields may be empty without indicating a processing error:

- An empty `graph_election_id` means that the election did not match the graph
  directly.
- An empty `timestamp` means that the source data did not provide a valid time.
- An empty `txt_clean` means that no usable comment remained after
  preprocessing.
- An empty `text_flags` usually means that no warning or special cleaning action
  was recorded.

The task-specific files have already been filtered with the corresponding
`eligible_for_*` flag. Do not automatically remove every row containing a
missing value.

## Data Summary

`data/rfa_modeling.csv.summary.json` records:

- 198,275 rows before task-specific filtering.
- 4,120 unified elections.
- 170,540 rows for the stance model.
- 188,904 rows for the election model.
- Text-status counts.
- Source-file SHA-256 checksums.
- The election-ID construction rules.

## Current Scope

The repository currently contains prepared data for modeling. The remaining work
is to:

1. Train the stance model.
2. Create early-window text features.
3. Create time-aware graph features.
4. Train and compare text-only, graph-only, and combined models.
5. Report macro-F1 and performance across observation windows.
