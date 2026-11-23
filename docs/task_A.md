### Task A: Vote-level stance mining

Goal:

\[
TXT_i \rightarrow VOT_i
\]

Predict:

- support
- oppose
- neutral

from the comment text.

This aligns with the course’s sentiment/opinion-mining setup, where comments are classified into positive, negative, and neutral classes. 16-9-en-L04-OpinionMining-01

Important leakage problem:

Many comments explicitly contain text like:

```text
'''Support'''
'''Oppose'''
'''Neutral'''
```

Those must be removed or masked before training, otherwise the model can trivially copy the answer.

Suggested model progression:

- majority-class baseline
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- transformer such as BERT/RoBERTa

Primary metric:

\[
Macro\text{-}F1
\]

because stance classes may be imbalanced.
