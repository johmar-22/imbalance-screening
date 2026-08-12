# Table 9. Strategy scorecard

Referenced from Section 5.6 of the manuscript.

Each strategy is ranked from 1, best, to 6. Every value is taken from results
already reported elsewhere in the paper; no additional computation enters this
table. Tied ranks share the better number, which is why some columns skip a
value.

| Criterion | None | Class weight | B-SMOTE | SMOTE | ADASYN | RUS |
|---|---|---|---|---|---|---|
| F2 at threshold 0.5 | 6 | **1** | 3 | 2 | 4 | 5 |
| F2 at tuned threshold | **1** | 2 | 3 | 4 | 5 | 5 |
| Average precision | **1** | 2 | 4 | 3 | 5 | 6 |
| ROC-AUC (not ranked) | 0.874 | 0.873 | 0.861 | 0.862 | 0.860 | 0.862 |
| Expected calibration error | **1** | 2 | 3 | 5 | 4 | 6 |
| Precision over top 25 | **1** | 2 | 4 | 3 | 5 | 5 |
| Precision over top 100 | 2 | **1** | 5 | 2 | 4 | 6 |
| F2 retained after Platt scaling | **1** | 2 | 4 | 3 | 5 | 6 |
| External enrichment at k = 100 | 2 | **1** | n/m | n/m | n/m | n/m |
| Net benefit at p_t = 0.10 | **1** | 2 | 3 | 4 | 5 | 6 |
| **Mean rank** (seven ranked criteria) | **1.9** | **1.7** | 3.7 | 3.1 | 4.6 | 5.6 |

`n/m` = not measured. The external hold-out of Section 4.7 was run for the
uncorrected and class-weighted models only.

## How to read it

**ROC-AUC is listed but not ranked.** The six values span 0.014. Ordering within
that span carries no information, and ranking it would manufacture a distinction
the data does not support.

**Only two strategies lead on anything, and they lead on disjoint criteria.**
Class weighting leads on fixed-threshold F2, on precision over the top 100 and
on external enrichment. No correction leads on tuned-threshold F2, on average
precision, on calibration, on precision over the top 25, on stability under
recalibration and on net benefit. The four synthetic and undersampling
strategies lead on nothing.

**The mean rank is a choice, not a fact.** It weights seven ranked criteria
equally: F2 at 0.5, F2 tuned, average precision, expected calibration error,
precision over the top 25, precision over the top 100, and F2 retained after
Platt scaling. External enrichment is excluded because it is missing for four
strategies, and net benefit because it is a restatement of the calibration and
ranking criteria already counted. Do not read the mean rank as the answer.

- A campaign that must act on a **hard label at a fixed threshold** should read
  the first row.
- A campaign that allocates a **calculation budget down a ranked list** should
  read average precision, precision over the top 25 and precision over the top
  100.

## Provenance

| Criterion | Source in the manuscript | File in this repository |
|---|---|---|
| F2 at threshold 0.5 | Table 1 | `results/Main_Results.csv` |
| F2 at tuned threshold | Table 1 | `results/Main_Results.csv` |
| Average precision | Table 2 | `results/Main_Results.csv` |
| ROC-AUC | Table 2 | `results/Main_Results.csv` |
| Expected calibration error | Table 2 | `results/Main_Results.csv` |
| Precision over top 25 and top 100 | Figure 7, Section 4.6 | `results/Screening_PrecisionAtK_AllStrategies.csv` |
| F2 retained after Platt scaling | Table 7 | `results/Recalibration_Summary.csv` |
| External enrichment at k = 100 | Table 5 | `results/External_Ricci_Validation.csv` |
| Net benefit at p_t = 0.10 | Figure 10, Section 4.11 | `results/DecisionCurve_Peaks.csv` |
