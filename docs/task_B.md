### Task B: Election-level early outcome prediction

Goal:

\[
\text{first } w\% \text{ of election activity}
\rightarrow RES
\]

with observation windows like:

\[
w \in \{10\%,25\%,50\%,75\%\}
\]

For an election with \(N\) votes:

\[
k_w=\lfloor wN\rfloor
\]

Sort by `DAT` and use only the first \(k_w\) votes/comments.

Do not randomly sample votes.

The main research question is:

> **How early can the final RfA result be predicted?**

The central comparison should be:

1. **Text-only**
2. **Graph-only**
3. **Text + graph**
4. simple vote-statistics baseline

A useful final results table would look like:

```text
Model            10%    25%    50%    75%
-------------------------------------------
Vote baseline      ?      ?      ?      ?
Text-only          ?      ?      ?      ?
Graph-only         ?      ?      ?      ?
Text + Graph       ?      ?      ?      ?
```

The numbers should be discovered experimentally.

### Text-only outcome model

Use only early comments.

For each observed comment:

\[
h_i = Encoder(TXT_i)
\]

aggregate:

\[
h_{text}=Pool(h_1,\dots,h_k)
\]

then:

\[
\hat{RES}=MLP(h_{text})
\]

Do not directly give `VOT` to the text-only model if you want the comparison to remain clean.

### Graph-only outcome model

The dataset naturally defines signed edges:

\[
SRC \xrightarrow{VOT} TGT
\]

The course link-analysis material covers graph ranking, community detection, link prediction, and graph classification. 16-9-en-L03-LinkAnalysis-01

However, a single RfA election graph is often mostly a star around the candidate, so a GNN on just that one election may not add much.

A stronger idea is to use the **historical signed voting network** before the current election.

Possible graph features:

- early number of support votes
- early number of oppose votes
- early neutral count
- support ratio
- candidate historical degree
- voter historical degree
- voter historical support/opposition tendency
- centrality of early supporters/opposers
- community structure
- signed-network features

Start with hand-crafted graph/network features + Logistic Regression/XGBoost before attempting a GNN.

The course GNN material describes node embeddings via neighborhood aggregation. 16-9-en-L03-LinkAnalysis-03_GCN…

### Combined model

Compute:

\[
h_T = \text{text representation}
\]

\[
h_G = \text{graph representation}
\]

then concatenate:

\[
h=[h_T;h_G]
\]

and predict:

\[
\hat{RES}=MLP(h)
\]

The project’s key question is whether combined information outperforms either modality alone.

### Simple outcome baseline

Before deep learning, use early-vote statistics such as:

\[
support\_ratio =
\frac{N_{support}}{N_{support}+N_{oppose}}
\]

plus:

- number of support votes
- number of oppose votes
- number of neutral votes
- number of unique voters
- elapsed time
- vote rate

Then train Logistic Regression.

This baseline is crucial because a candidate with 22 support vs 2 oppose after 25 votes may already be easy to predict.
