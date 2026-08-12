# Evaluation card

Referenced from Section 5.6 of the manuscript, which states that this card
records the estimand, the estimator, the reporting rules, the known limitations
and the conditions under which the benchmark should be considered superseded.

Structure follows the evaluation-card proposal of Alampara, Schilling-Wilhelmi
and Jablonka, *Lessons from the trenches on evaluating machine learning systems
in materials science*, Comput. Mater. Sci. **259**, 114041 (2025).

---

## 1. Estimand

**The question this benchmark answers.** For a rare transport property predicted
from chemical composition alone, does correcting class imbalance improve the
model, and if it appears to, what exactly improved?

**Population.** The 8,924 entries of the `boltztrap_mp` release with complete
values for all six transport quantities and a parseable formula. This is a
convenience population, not a sample from the space of synthesizable inorganic
compounds, and no claim of representativeness is made.

**Target.** The binary label m\*p < 1.0 me, where m\*p is the BoltzTraP
conductivity effective mass at 300 K and 10^18 cm^-3. Prevalence 4.90%,
imbalance ratio 19.4:1.

**Unit of analysis.** One database entry. Because inputs are functions of the
formula alone, entries sharing a composition are not independent units, which is
why every split is grouped on the normalized reduced formula.

**What the estimand is not.** It is not the probability that a compound is a
useful p-type conductor. The label is a transport-property criterion; band gap,
p-type dopability and thermodynamic stability are not assessed.

---

## 2. Estimator

| Element | Specification |
|---|---|
| Inputs | 132 Magpie elemental-property descriptors, formula only, plus 2 Gaussian negative controls |
| Feature selection | ANOVA F filter, k = 30, fitted inside each training fold |
| Classifier | XGBoost, histogram split-finding, log-loss objective, held fixed across strategies |
| Strategies | None, ClassWeight, B-SMOTE, SMOTE, ADASYN, RUS, each a step of an `imblearn` pipeline so resampling is confined to training folds |
| Outer validation | `StratifiedGroupKFold`, 5 folds x 3 repeats = 15 estimates, grouped on normalized reduced formula |
| Inner validation | `StratifiedGroupKFold`, 3 folds, same grouping constraint |
| Hyperparameter search | randomized, 25 draws, selected on average precision |
| Operating points | fixed 0.5, and a threshold maximizing F2 over a 91-point grid on inner-fold predictions only |
| Primary metrics | F2 at a fixed point; average precision threshold-free |
| Probability quality | Brier score, expected calibration error over 10 equal-width bins |
| Comparison | Wilcoxon signed-rank, Nadeau-Bengio corrected resampled t, Cohen's d, Holm across 15 pairwise comparisons |

**Asserted at run time, not assumed.** No composition appears on both sides of
any outer split, of any inner split, or of any internal calibration split. The
assertions are in the code and fail the run if violated.

**Analyses that depart from this protocol.** Sections 4.4, 4.5, 4.7, 4.8 and
4.10 each freeze or reduce part of the configuration, and each says so where it
is reported. Their values are comparable with each other where the protocol
matches and not with Table 1.

---

## 3. Reporting rules

1. **Lead with the threshold-free metric.** A screening campaign consumes a
   ranked list, so average precision governs, not F2 and not accuracy.
2. **Report calibration alongside discrimination.** A workflow that allocates
   effort by predicted probability is exposed to the probability scale, and a
   discrimination-only audit will not reveal the exposure.
3. **Separate the ranking from the cut.** Every strategy is reported at a fixed
   threshold and at a tuned threshold, so a change in the ordering of compounds
   is never confused with a change in where the ordering is cut.
4. **Report effect sizes and multiplicity control, not bare p-values.** Repeated
   cross-validation reuses data across folds, so fold differences are
   correlated; p-values are reported with the corrected resampled t-test
   alongside and are not to be read as if the 15 estimates were independent.
5. **Report null and negative results in full.** The aggregate leakage ablation
   is not significant and is stated before the localized result that is.
6. **Do not report the maximum of a net-benefit curve.** It is attained at
   p_t → 0 by construction, where the false-positive penalty vanishes.
7. **State when a p-value is at its attainable floor.** With six paired values
   the smallest two-sided p a sign test or Wilcoxon test can return is 0.031;
   where that value appears, it evidences consistency of sign, not magnitude.

---

## 4. Known limitations

- **An irreducible ceiling.** 58 compositions have polymorphs on opposite sides
  of the cutoff, so identical inputs carry conflicting labels. No classifier or
  resampler can resolve this, and grouped validation counts it as error rather
  than absorbing it as memorization.
- **The label is not always well defined.** The conductivity effective mass is
  properly meaningful only for semiconductors and the release carries no
  metallicity flag, so metallic entries remain in the pool. Nothing was removed
  on that ground; the exposure was bounded by a band-gap sweep instead.
- **Labels inherit the assumptions of the calculations.** GGA-PBE or GGA+U band
  structures, BoltzTraP under the constant relaxation-time approximation, at
  300 K and 10^18 cm^-3. They are a computational artifact, not a measurement.
- **One database.** All results come from a single transport release, so this
  measures generalization to unseen compositions within one methodology, not
  transfer across methodologies.
- **Recalibration is bounded.** Only three calibration maps were tested and all
  three are monotone, so the invariance of average precision under them is
  algebraic rather than empirical. A non-monotone post-processing step is not
  covered.
- **Attribution is not physical attribution.** Magpie descriptors are strongly
  correlated by construction; the three attribution measures used here disagree,
  and none isolates the contribution of a single descriptor.
- **Correlated folds.** Repeated cross-validation reuses data, so the 15
  estimates are not independent.

---

## 5. Conditions under which this benchmark is superseded

Both of the following remove the problem this protocol was built to control, and
either makes these results a historical baseline rather than a current one.

1. **A descriptor set that is not a function of composition alone.** The
   grouping argument rests on the descriptor map being many-to-one over
   polymorphs. Structure-aware inputs break that premise, and the correct
   grouping key becomes something richer than the reduced formula.
2. **A database release that resolves polymorphs into distinct feature
   vectors.** If entries sharing a formula no longer share an input, the
   representational leakage this protocol controls does not arise, and the cost
   of grouping is no longer justified by it.

Two weaker conditions that would warrant a rerun rather than replacement: a
release in which the metallicity of entries is flagged, so the label can be
restricted to the compounds for which it is defined; and an independent
repository tabulating the same transport quantities under different conventions,
which would provide the external check across methodologies that this design
does not.

---

## 6. Reuse conditions

Code under `scripts/` is MIT licensed. Result tables, figures and cached data
are CC BY 4.0. The underlying transport data remains under the license of Ricci
et al. (2017) and must be cited independently of this work.

If you reuse the protocol rather than the results, the two elements that carry
it are the composition grouping and the separation of the ranking from the cut.
Reusing the strategy comparison without both will reproduce the fixed-threshold
result and miss the reason it is misleading.
