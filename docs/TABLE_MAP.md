# Manuscript to repository map

Every numbered object in the manuscript, and the file in this repository that
produces or contains it. Section numbers in the last column refer to the
section headings printed by `scripts/run_analysis.py`.

## Main tables

| Manuscript | Contents | File | Produced by |
|---|---|---|---|
| Table 1 | Operating-point performance over 15 composition-grouped folds: F2 at 0.5, F2 tuned, tuned threshold, precision, recall, F1, MCC, accuracy | `results/Main_Results.csv` | Section 4 |
| Table 2 | Threshold-free ranking and probability quality: average precision, ROC-AUC, Brier, ECE | `results/Main_Results.csv` (same file, remaining columns) | Section 4 |
| Table 3 | Retrospective screening utility: precision@k, enrichment, recall@k | `results/Screening_PrecisionAtK.csv` | Section 9 |
| Table 4 | Fifteen highest-ranked compounds with formula, m\*p, band gap, gap character, E above hull | `results/Top100_Candidates.csv`, first 15 rows | Section 9 (+ Materials Project API) |
| Table 5 | Leave-one-chemical-family-out generalization | `results/LOCO_Generalization.csv` | Section 10 |
| Table 6 | External hold-out on 10,879 Ricci compounds sharing no normalized composition with training | `results/External_Ricci_Validation.csv` | Section 10 |
| Table 7 | Incremental value of n-type transport descriptors over composition | `results/CrossBand_Incremental_Test.csv` | Section 11 |
| Table 8 | Effect of post-hoc recalibration over the same 15 folds | `results/Recalibration_Summary.csv` | Section 4b |

## Supplementary tables promised in the Data availability statement

| Manuscript | Contents | File |
|---|---|---|
| Table S1 | Evaluation metrics and their roles | `docs/Table_S1_Metrics.md` |
| Table S2 | All 15 pairwise comparisons of strategies on fold-level F2 (Wilcoxon signed-rank) | `results/Stat_Wilcoxon_F2.csv` |
| Table S3 | Sensitivity of the benchmark to the positive-class cutoff (0.5 / 0.8 / 1.0 / 1.2 / 1.5 me) | `results/Threshold_Sensitivity.csv` |
| Table S4 | Ten descriptors with the highest permutation importance | `results/Feature_Importances.csv` (sort by permutation importance, take 10) |
| complete candidate list | All 100 ranked compounds | `results/Top100_Candidates.csv` |

## Figures

The figure files were written by the script under working names. They are
renamed here to match the manuscript numbering. Each exists as `.pdf` (vector),
`.png` and `.tiff` (both 600 dpi).

| Manuscript | Caption subject | File in `figures/` | Script working name |
|---|---|---|---|
| Figure 1 | Overview of the analysis pipeline | `Fig1_Pipeline_Overview` | drawn separately, not script output |
| Figure 2 | Imbalance-handling strategies compared over 15 folds | `Fig2_Strategy_Comparison` | `Fig1_Strategy_Comparison` |
| Figure 3 | PR curves and reliability diagrams from out-of-fold predictions | `Fig3_PR_and_Calibration` | `Fig2_PR_and_Calibration` |
| Figure 4 | Sensitivity of the benchmark to the positive-class definition | `Fig4_Threshold_Sensitivity` | `Fig_Threshold_Sensitivity` |
| Figure 5 | SHAP summary for the class-weighted reference model | `Fig5_SHAP_Summary` | `Fig_SHAP_Summary` |
| Figure 6 | Retrospective screening utility from out-of-fold rankings | `Fig6_Enrichment` | `Fig_Enrichment` |
| Figure 7 | Reliability before and after recalibration | `Fig7_Recalibration` | `Fig_Recalibration` |
| Figure S1 | PCA projection, visualization diagnostic only | `FigS1_PCA_Diagnostic` | `Fig3_PCA_Diagnostic` |

`Fig3_PCA_Diagnostic` is produced by the script but is not a numbered figure in
the manuscript. It is carried here as supplementary Figure S1.

## Result files not tied to a numbered table

These support statements in the text rather than a table, and are committed for
completeness.

| File | Supports |
|---|---|
| `results/Composition_Redundancy.csv` | Section 3.3, count of polymorphs per composition and the justification for grouped cross-validation |
| `results/Stat_McNemar.csv` | Section 4.3, paired McNemar tests on pooled out-of-fold predictions |
| `results/Noise_Audit.csv` | Section 4.5, negative control confirming the selection stage never prefers the two Gaussian noise columns |
| `results/SHAP_Importances.csv` | Section 4.5, numeric backing for Figure 5 |
| `results/Best_Hyperparameters.csv` | Section 3.5, hyperparameters selected per fold by the randomized search |
| `results/Recalibration_PerFold.csv` | Section 4.9, fold-level values summarized in Table 8 |
| `results/Recal_Diag_ProbDist.csv` | Section 4.9, predicted-probability distributions before and after recalibration |
| `results/Recal_Diag_TailReliability.csv` | Section 4.9, reliability restricted to the high-probability tail that screening actually uses |
| `results/Recal_Diag_PriorCorrection.csv` | Section 4.9, effect of prior correction on the resampled models |
| `data/magpie_cache.csv` | Section 3.2, the 132-column Magpie feature matrix for all 8,924 compositions |
