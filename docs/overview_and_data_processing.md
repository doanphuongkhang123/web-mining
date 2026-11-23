The user is working on a **Web Mining capstone project** about:

> **Stance Mining and Early Outcome Prediction in Online Community Voting**

The project uses **Wikipedia Requests for Adminship (RfA)** data and is intended to combine **opinion mining** and **link analysis** from the course.

The course syllabus says capstone topics can include link analysis, opinion mining, or recommendation systems, and expects problem statement, method, evaluation, code, and datasets. 20261_syllabus

The user shared this dataset:

`https://snap.stanford.edu/data/wiki-RfA.html`

It is the Stanford SNAP Wikipedia RfA dataset. 

It was downloaded to data/ folder.

The raw dataset has records with fields:

- `SRC`: voter username
- `TGT`: candidate username
- `VOT`: vote/stance
  - `+1` = support
  - `0` = neutral
  - `-1` = oppose
- `RES`: final result of the whole RfA
  - `+1` = accepted
  - `-1` = rejected
- `YEA`: year the election started
- `DAT`: exact vote timestamp
- `TXT`: the vote comment

The key structural insight is:

> **one raw record = one person casting one vote on one candidate**

It is **not** “one row = one election.”

There are three conceptual levels:

1. **User level** — Wikipedia users.
2. **Vote level** — one `SRC → TGT` vote with `VOT`, `DAT`, `TXT`.
3. **Election level** — many vote records belonging to the same candidate’s RfA episode, sharing one final `RES`.

For example:

```text
Alice  -> Bob   VOT=+1
Carol  -> Bob   VOT=-1
David  -> Bob   VOT=0
...
```

can all belong to one Bob election whose final:

```text
RES=+1
```

The dataset is a very good fit for the capstone because it includes:

- textual comments
- signed votes
- timestamps
- final election outcome
- directed user-to-candidate relationships

Important issue: **there is no explicit `election_id`**.

Also, the same candidate may run multiple times, so you should **not** group only by `TGT`.

The proposed reconstruction strategy is:

\[
\text{election\_id}
\approx
(TGT,\;YEA,\;\text{temporal episode})
\]

Procedure:

1. Parse `DAT`.
2. Sort by `TGT`, then `DAT`.
3. For each candidate, calculate gaps between consecutive votes.
4. Split when:
   - `YEA` changes, or
   - the time gap is clearly larger than a normal RfA episode.
5. Use `RES` mainly as a **consistency check**:
   - ideally all rows inside one reconstructed election have the same `RES`.
6. Do not use `RES` alone for splitting because two separate elections can both be accepted or both rejected.

Recommended improvement: inspect the empirical distribution of vote-to-vote time gaps before choosing a threshold. Test something like 10, 14, 21, 30 days and check reconstruction stability.

After reconstruction, create something like:

```text
election_id | SRC | TGT | DAT | TXT_clean | VOT | RES
```

and an election-level table:

```text
election_id | candidate | start | end | n_votes | RES
```

### Chronological splitting

Do **not** randomly split vote rows.

Reconstruct elections first, order elections by start time, then split entire elections chronologically into train / validation / test.

For example:

```text
older elections      middle       newest
      train        validation      test
```

The exact year boundaries should depend on class balance and election counts.

Important leakage rule:

\[
\max(t_{train}) < \min(t_{test})
\]

and each election must belong to exactly one split.

### Historical graph leakage

For an election beginning at time \(t_E\), the historical network may contain only edges before \(t_E\):

\[
G_E = \{e_i : DAT_i < t_E\}
\]

plus current-election votes that fall inside the allowed observation window.

Do not let a 2012 edge influence a prediction for a 2010 election.

### `RES` and `VOT` leakage distinctions

For stance prediction:

\[
y=VOT
\]

so the model must not directly see that vote sign.

For election outcome prediction:

\[
y=RES
\]

so `RES` must never be an input.

However, **early `VOT`s are valid inputs for outcome prediction**, because those votes genuinely exist at prediction time.

### Suggested practical implementation order

1. Parse raw SNAP file.
2. Parse timestamps.
3. Reconstruct `election_id`.
4. Validate election clusters.
5. Clean `TXT` and mask explicit stance words.
6. Create chronological election-level split.
7. Train stance baseline:
   - TF-IDF + Logistic Regression.
8. Train stronger stance model:
   - BERT/RoBERTa.
9. Create observation windows.
10. Train early-outcome vote-statistics baseline.
11. Build text-only outcome model.
12. Build graph-feature model.
13. Build combined model.
14. Compare Macro-F1 across observation windows.
15. Optionally add GCN/GAT only after the baseline pipeline works.

The current recommended scope is deliberately **not GNN-first**. A strong capstone can be built from rigorous preprocessing, leakage prevention, chronological evaluation, strong baselines, and text/graph fusion. A complex GNN is optional.

The final research questions should be approximately:

- Can voting-comment text predict support / oppose / neutral stance?
- How early can the final RfA result be predicted?
- Is text or network structure more predictive?
- Does combining text and graph information improve performance?
- What disagreement patterns characterize successful vs unsuccessful elections?

The next useful step for a fresh agent would be to help the user turn this into a **concrete implementation plan / notebook structure**, likely beginning with downloading and parsing the SNAP RfA file and empirically reconstructing `election_id`s.
