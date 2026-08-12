# Manuscript to repository map

Every numbered object in manuscript v32, and the file in this repository that
produces or contains it.

Objects marked **NEW** did not exist when the repository was first published.
Objects marked **RENUMBERED** exist but changed number: Table 3 of the earlier
draft (precision@k) was deleted because Figure 7 carries the same information,
and Figure 1 of the earlier draft became Figure 2 when the leakage-mechanism
figure was added in front of it.

Section numbers in the last column refer to the headings printed by
`scripts/run_analysis.py` and `scripts/run_extended_analysis.py`.

## Main tables

| Manuscript | Contents | File | Produced by |
|---|---|---|---|
| Table 1 | Operating-point performance over 15 composition-grouped folds: F2 at 0.5, F2 tuned, tuned threshold, precision, recall, F1, MCC, accuracy | `results/Main_Results.csv` | `run_analysis.py` Section 4 |
| Table 2 | Threshold-free ranking and probability quality: average precision, ROC-AUC, Brier, ECE | `results/Main_Results.csv`, remaining columns | `run_analysis.py` Section 4 |
| Table 3 **RENUMBERED** (was 4) | Fifteen highest-ranked compounds with formula, m\*p, band gap, gap character, E above hull | `results/Top100_Candidates.csv`, first 15 rows | `run_analysis.py` Section 9 + Materials Project API |
| Table 4 **RENUMBERED** (was 5) | Leave-one-chemical-family-out generalization | `results/LOCO_Generalization.csv` | `run_analysis.py` Section 10 |
| Table 5 **RENUMBERED** (was 6) | External hold-out, 10,879 Ricci compounds sharing no normalized composition with training | `results/External_Ricci_Validation.csv` | `run_analysis.py` Section 10 |
| Table 6 **RENUMBERED** (was 7) | Incremental value of n-type transport descriptors over composition | `results/CrossBand_Incremental_Test.csv` | `run_analysis.py` Section 11 |
| Table 7 **RENUMBERED** (was 8) | Effect of post-hoc recalibration over the same 15 folds | `results/Recalibration_Summary.csv` | `run_analysis.py` Section 4b |
| Table 8 **NEW** | Inflation of average precision from dropping the composition grouping: inflation, 95% bootstrap CI, Hedges' g, Holm-adjusted Mann-Whitney p, and the same split into shared-composition and unique-composition rows | `results/Leakage_Inflation.csv` | `run_extended_analysis.py` 13A |
| Table 9 **NEW** | Strategy scorecard, six strategies ranked 1 to 6 on eight criteria | `docs/Table_9_Scorecard.md` | authored, not script output |

The earlier draft's Table 3, precision@k and enrichment per strategy, was
deleted from the manuscript because Figure 7 plots the same quantities at every
list length. The underlying file is still committed as
`results/Screening_PrecisionAtK.csv` and
`results/Screening_PrecisionAtK_AllStrategies.csv`, since Section 4.6 quotes
values from it.

## Supplementary tables promised in the Data availability statement

The statement names Tables S1 to S8 and says no separate supplementary document
accompanies the article, so all eight must be present here.

| Manuscript | Contents | File |
|---|---|---|
| Table S1 | Evaluation metrics and their roles | `docs/Table_S1_Metrics.md` |
| Table S2 | All 15 pairwise comparisons on fold-level F2 | `results/Stat_Wilcoxon_F2.csv` |
| Table S3 | Sensitivity to the positive-class cutoff, 0.5 to 1.5 m_e | `results/Threshold_Sensitivity.csv` |
| Table S4 | Ten descriptors with the highest permutation importance | `results/Feature_Importances.csv`, sort by permutation importance |
| Table S5 **NEW** | The same 15 comparisons on average precision | `results/Stat_Wilcoxon_AP.csv` |
| Table S6 **NEW** | The same 15 comparisons on expected calibration error | `results/Stat_Wilcoxon_ECE.csv` |
| Table S7 **NEW** | Per-fold leakage ablation, both split families | `results/Leakage_Ablation_PerFold.csv` |
| Table S8 **NEW** | Net-benefit grids, realized and attainable | `results/DecisionCurve_NetBenefit.csv`, `results/DecisionCurve_Ceiling.csv` |

## Figures

Each exists as `.pdf` (vector), `.png` and `.tiff`, all 600 dpi, except
`Fig2_Pipeline_Overview` which is `.png`, `.svg` and `.tiff`.

| Manuscript | Caption subject | File in `figures/` | Working name | Drawn by |
|---|---|---|---|---|
| Figure 1 **NEW** | Representational leakage: the two SnS entries collapsing to one descriptor vector | `Fig1_Leakage_Mechanism` | `Fig_Leakage_Mechanism` | `scripts/make_fig_leakage.py` |
| Figure 2 **RENUMBERED** (was 1) | Overview of the analysis pipeline | `Fig2_Pipeline_Overview` | drawn separately | not script output |
| Figure 3 **RENUMBERED** (was 2) | Imbalance-handling strategies compared over 15 folds | `Fig3_Strategy_Comparison` | `Fig1_Strategy_Comparison` | `run_analysis.py` Section 9b |
| Figure 4 **RENUMBERED** (was 3) | Ranking quality and calibration from out-of-fold predictions | `Fig4_Ranking_and_Calibration` | `Fig2_PR_and_Calibration` | `run_analysis.py` Section 9b |
| Figure 5 **RENUMBERED** (was 4) | Sensitivity to the positive-class definition | `Fig5_Threshold_Sensitivity` | `Fig_Threshold_Sensitivity` | `run_analysis.py` Section 7 |
| Figure 6 **RENUMBERED** (was 5) | SHAP summary for the class-weighted reference model | `Fig6_SHAP_Summary` | `Fig_SHAP_Summary` | `run_analysis.py` Section 8 |
| Figure 7 **RENUMBERED** (was 6) | Retrospective screening utility from out-of-fold rankings | `Fig7_Enrichment` | `Fig_Enrichment` | `run_analysis.py` Section 9 |
| Figure 8 **RENUMBERED** (was 7) | Reliability before and after recalibration | `Fig8_Recalibration` | `Fig_Recalibration` | `run_analysis.py` Section 4b |
| Figure 9 **NEW** | Effect of removing the composition grouping, three panels | `Fig9_Leakage_Ablation` | `Fig_Leakage_Ablation` | `run_extended_analysis.py` 13A |
| Figure 10 **NEW** | Decision-curve analysis and the scale/ranking decomposition | `Fig10_Decision_Curve` | `Fig_Decision_Curve` | `run_extended_analysis.py` 13B |
| Figure S1 | PCA projection, visualization diagnostic only | `FigS1_PCA_Diagnostic` | `Fig3_PCA_Diagnostic` | `run_analysis.py` Section 9b |

`Fig3_PCA_Diagnostic` is produced by the script but is not a numbered figure in
the manuscript. It is carried here as supplementary Figure S1.

## Result files not tied to a numbered object

| File | Supports |
|---|---|
| `results/Composition_Redundancy.csv` | Section 3.3, polymorph count per composition |
| `results/Composition_Structure.csv` **NEW** | Section 3.3, the seven counts quoted in the text |
| `results/Straddling_Compositions.csv` **NEW** | Section 3.3, the identity of the 58 compositions whose polymorphs straddle the cutoff |
| `results/Leakage_Rate.csv` **NEW** | Section 3.3, the 23.8% ± 1.2% exposure under ungrouped splitting and the assertion of zero under grouping |
| `results/Leakage_Ablation.csv` **NEW** | Section 4.10, per-arm summary behind Figure 9 |
| `results/Leakage_Contrast_Test.csv` **NEW** | Section 4.10, the six-strategy contrast test and the note that p = 0.031 is the attainable floor for n = 6 |
| `results/DecisionCurve_Decomposition.csv` **NEW** | Section 4.11, realized against attainable net benefit at five exchange rates |
| `results/DecisionCurve_Peaks.csv` **NEW** | Section 4.11, net benefit and calculations avoided at the five operating points |
| `results/OOF_Probabilities.csv` **NEW** | Sections 4.6 and 4.11, the out-of-fold probability of every compound under every strategy |
| `results/Fold_Metrics_PerFold.csv` **NEW** | Sections 4.1 to 4.3, the 15 per-fold values behind Tables 1, 2, S2, S5 and S6 |
| `results/Screening_PrecisionAtK_AllStrategies.csv` **NEW** | Section 4.6 and Figure 7a, precision@k for all six strategies |
| `results/Stat_McNemar.csv` | Section 4.3 |
| `results/Noise_Audit.csv` | Section 3.2, the negative control |
| `results/SHAP_Importances.csv` | Section 4.5, numeric backing for Figure 6 |
| `results/Best_Hyperparameters.csv` | Section 3.5, and the frozen configuration used by 13A |
| `results/Recalibration_PerFold.csv` | Section 4.9 |
| `results/Recal_Diag_ProbDist.csv` | Section 4.9 |
| `results/Recal_Diag_TailReliability.csv` | Section 4.9 |
| `results/Recal_Diag_PriorCorrection.csv` | Section 4.9 |
| `docs/EVALUATION_CARD.md` **NEW** | Section 5.6, the evaluation card the Data availability statement promises |
| `data/magpie_cache.csv` | Section 3.2 |
