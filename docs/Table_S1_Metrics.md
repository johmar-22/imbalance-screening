# Table S1. Evaluation metrics and their roles

Referenced from Section 3.6 of the manuscript.

Notation. `N` is the number of compounds in the set being scored: a held-out
fold for Tables 1 and 2, the full 8,924 out-of-fold predictions for the ranking
metrics. `i` indexes compounds; `y_i in {0,1}` is the true class, 1 when
m\*p < 1.0 me; `p_i in [0,1]` is the predicted positive-class probability
(`predict_proba(...)[:,1]`); `t` is the decision threshold, either the fixed 0.5
or the inner-CV tuned `t*`; `y_hat_i = 1[p_i >= t]`. `TP, FP, TN, FN` are the
confusion-matrix counts at `t`; `P` and `R` are precision and recall at `t`;
`beta = 2`. `B_b` is the set of compounds in calibration bin `b` of ten
equal-width bins.

| # | Metric | Definition | Threshold dependent | Role in this study |
|---|---|---|---|---|
| 1 | Precision | `TP / (TP + FP)` | yes | Fraction of flagged compounds that are true positives. Reported for context, never optimized alone. |
| 2 | Recall (sensitivity) | `TP / (TP + FN)` | yes | Fraction of low-mass compounds recovered. The quantity a screening campaign cannot afford to lose. |
| 3 | **F2** | `(1 + 4) P R / (4 P + R)` | yes | **Primary operating-point metric.** Weights recall twice as heavily as precision, because a missed candidate costs an undiscovered material while a false positive costs one downstream calculation. |
| 4 | F1 | `2 P R / (P + R)` | yes | Reported for comparability with the wider literature. Not used for model selection. |
| 5 | MCC | `(TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))` | yes | Balanced single-number summary that stays informative under strong class imbalance. |
| 6 | Accuracy | `(TP + TN) / N` | yes | Reported only to show that it is uninformative here: the majority-class rate already exceeds it in interest. Never used for any decision. |
| 7 | **Average precision (PR-AUC)** | `sum_n (R_n − R_{n−1}) P_n`, with `R_0 = 0` | no | **Primary threshold-free metric.** A screening campaign consumes a ranked list, and PR curves are more informative than ROC under strong imbalance. |
| 8 | ROC-AUC | area under TPR against FPR as `t` sweeps | no | Reported for continuity with prior work. Optimistic under imbalance because the FPR denominator is dominated by negatives. |
| 9 | Brier score | `(1/N) sum_i (p_i − y_i)^2` | no | Mean squared error of the predicted probabilities. Lower is better. Detects resampling that inflates probabilities. |
| 10 | Expected calibration error (ECE) | `sum_b (|B_b| / N) · |acc(B_b) − conf(B_b)|` over 10 equal-width bins | no | Average gap between predicted probability and observed frequency. Lower is better. Matters because candidates are ranked by probability. |
| 11 | Enrichment factor at k | `precision@k / base rate` | no (rank based) | How many times more productive inspecting the top k is than inspecting at random. The practical screening statistic. |
| 12 | Precision@k | fraction of the top k ranked compounds that are true positives | no (rank based) | Direct answer to "if I can afford k calculations, how many hits do I get?" |
| 13 | Recall@k | fraction of all true positives contained in the top k | no (rank based) | Coverage of the candidate space achieved by a budget of k. |

## Notes

- Metrics 1 to 6 are computed at both the fixed 0.5 threshold and the
  inner-cross-validation tuned threshold `t*`. The tuned threshold is selected
  inside the training folds only and never sees held-out data.
- Metrics 7 to 10 are computed on predicted probabilities and are therefore
  independent of any decision threshold.
- Metrics 11 to 13 are computed on out-of-fold predictions, so no compound is
  ever ranked by a model that saw its label.
- All fold-level values are aggregated over 15 estimates (5 folds x 3 repeats)
  and reported as mean plus or minus standard deviation.

## Source

Generated from `results/Main_Results.csv` and
`results/Screening_PrecisionAtK.csv`. See `docs/TABLE_MAP.md` for the full
mapping between manuscript objects and files in this repository.
