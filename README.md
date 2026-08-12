# Rare-class prediction of low hole effective mass from composition alone

A leakage-controlled benchmark of imbalance-handling strategies for
composition-only screening of p-type semiconductors.

This repository contains the complete source code, result tables and figures for
the manuscript *Class-imbalance corrections for composition-only screening of low hole effective mass: a leakage-controlled benchmarking analysis*.

**Nothing in this repository requires Google Drive, a Colab session, or any
manually downloaded data file.** The primary dataset is fetched from Figshare
through `matminer` on first run and cached locally.

---

## 1. What the study does

The dataset is the `boltztrap_mp` release distributed through `matminer`
(8,924 compounds with complete transport data), derived from the ab initio
electronic transport database of Ricci et al. (2017). The positive class is
defined by the hole effective mass threshold m\*p < 1.0 me, which occurs in a
small minority of entries.

Model inputs are Magpie elemental-property descriptors computed from the
chemical formula only. They are available for any hypothetical composition
before any DFT or BoltzTraP calculation is run, so the task is a genuine
pre-calculation triage decision rather than a re-description of the target.

Six imbalance-handling strategies (uncorrected, class weighting, SMOTE,
BorderlineSMOTE, ADASYN, random undersampling) are compared under one identical
protocol: 5-fold cross-validation repeated 3 times, **grouped by normalized
composition** so that polymorphs sharing a formula never appear on both sides of
a split. Both a fixed 0.5 decision threshold and an inner-CV tuned threshold are
reported. Ranking quality (average precision), probability quality (Brier score,
expected calibration error), retrospective screening utility (enrichment,
precision@k), leave-one-chemical-family-out generalization and an external
hold-out on unseen compositions from the larger Ricci tabular release are all
evaluated.

---

## 2. Quick start

```bash
git clone https://github.com/johmar-22/imbalance-screening.git
cd imbalance-screening

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# ~10 minute smoke test; confirms the environment works.
# The numbers it prints are NOT the numbers in the paper.
python scripts/run_analysis.py --outdir results_reproduced --quick

# Full run behind every reported number. 1.5-2.5 h on a GPU, 6-10 h on CPU.
python scripts/run_analysis.py --outdir results_reproduced
```

Every table and figure of the manuscript is written into `--outdir`. Compare
against the committed copies in `results/` and `figures/`.

### Command line options

| Flag | Default | Meaning |
|---|---|---|
| `--outdir PATH` | `../results` | Where CSVs, figures and checkpoints are written |
| `--quick` | off | Smoke test: 1 CV repeat, 8 search iterations |
| `--seed N` | `42` | Random seed |
| `--no-resume` | off | Ignore existing checkpoints and recompute everything |
| `--data-source {figshare,csv}` | `figshare` | Dataset origin |
| `--datadir PATH` | script directory | Where a local `boltztrap_mp.csv` is looked for, only when `--data-source csv` |

### Checkpointing

Long loops write a small pickle into `<outdir>/_checkpoints/` after each unit of
work. If the process is interrupted, rerunning the same command resumes instead
of restarting. Each checkpoint records a hash of the configuration that produced
it and is ignored automatically if the configuration changed. `--no-resume`
forces a full recomputation.

`_checkpoints/` is listed in `.gitignore` and is not part of the repository.

---

## 3. Optional: Materials Project enrichment

Table 4 of the manuscript reports band gap, gap character and energy above hull
for the fifteen highest-ranked candidates. Those three columns come from the
Materials Project REST API and require a free key from
<https://next-gen.materialsproject.org/api>.

```bash
export MP_API_KEY=your_key_here      # Windows PowerShell: $env:MP_API_KEY="..."
python scripts/run_analysis.py --outdir results_reproduced
```

Without a key the script runs to completion and prints a notice; the band gap,
gap character and stability columns of `Top100_Candidates.csv` are left empty.
No other result depends on the key. The committed `results/Top100_Candidates.csv`
already contains the enriched columns, so a reviewer does not need a key to
check Table 4.

**No API key is stored in this repository.** The script reads `MP_API_KEY` from
the environment and nowhere else.

---

## 4. Repository layout

```
.
├── README.md
├── LICENSE                  MIT, applies to everything under scripts/
├── LICENSE-DATA             CC BY 4.0, applies to results/ figures/ data/
├── CITATION.cff
├── requirements.txt         pinned environment for reproduction
├── environment.yml          conda alternative
├── .gitignore
├── scripts/
│   ├── run_analysis.py      the complete analysis, one file, no Drive
│   └── make_supplementary.py  regenerates docs/ tables from results/
├── notebooks/
│   └── run_analysis.ipynb   identical code as a notebook, for Colab
├── data/
│   └── magpie_cache.csv     precomputed Magpie features (optional, see below)
├── results/                 every CSV behind the manuscript tables
├── figures/                 every figure, in PDF, PNG and TIFF
└── docs/
    ├── TABLE_MAP.md         manuscript table/figure -> file in this repo
    ├── FILE_MANIFEST.md     every file, its source and its destination
    └── Table_S1_Metrics.md  supplementary Table S1
```

### `data/magpie_cache.csv`

The 132-column Magpie feature matrix for all 8,924 compositions, precomputed.
The script writes this file itself on first run, which takes roughly 20 to 40
minutes. It is committed here so a reviewer can skip that step and so the
features used are byte-identical to ours. To use it:

```bash
cp data/magpie_cache.csv results_reproduced/magpie_cache.csv
python scripts/run_analysis.py --outdir results_reproduced
```

To verify the featurization from scratch instead, simply do not copy the file.
The script regenerates it and the two should agree.

---

## 5. Mapping from the manuscript to this repository

Full detail in [`docs/TABLE_MAP.md`](docs/TABLE_MAP.md). Summary:

| Manuscript | File |
|---|---|
| Table 1, Table 2 | `results/Main_Results.csv` |
| Table 3 | `results/Screening_PrecisionAtK.csv` |
| Table 4 | `results/Top100_Candidates.csv` (first 15 rows) |
| Table 5 | `results/LOCO_Generalization.csv` |
| Table 6 | `results/External_Ricci_Validation.csv` |
| Table 7 | `results/CrossBand_Incremental_Test.csv` |
| Table 8 | `results/Recalibration_Summary.csv` |
| Table S1 | `docs/Table_S1_Metrics.md` |
| Table S2 | `results/Stat_Wilcoxon_F2.csv` |
| Table S3 | `results/Threshold_Sensitivity.csv` |
| Table S4 | `results/Feature_Importances.csv` |
| Figure 1 | `figures/Fig1_Pipeline_Overview.*` |
| Figure 2 | `figures/Fig2_Strategy_Comparison.*` |
| Figure 3 | `figures/Fig3_PR_and_Calibration.*` |
| Figure 4 | `figures/Fig4_Threshold_Sensitivity.*` |
| Figure 5 | `figures/Fig5_SHAP_Summary.*` |
| Figure 6 | `figures/Fig6_Enrichment.*` |
| Figure 7 | `figures/Fig7_Recalibration.*` |

---

## 6. Reproducibility notes and known limits

- **Exact bitwise reproduction is not guaranteed.** XGBoost histogram building
  is not bit-identical between CPU and GPU, and `nthread` affects floating-point
  accumulation order. Expect agreement to roughly the third decimal on F2 and
  average precision, which is far below the fold-to-fold standard deviations
  reported in the tables. The strategy ranking is stable.
- **Runtime.** Full run is 1.5 to 2.5 hours on an NVIDIA T4/L4/A100 and 6 to 10
  hours on CPU. The script detects the GPU automatically and falls back to CPU.
- **`shap` and `scipy`.** Recent `shap` releases can conflict with an older
  `scipy` already resident in a session. If the SHAP section is skipped with an
  import error, run `pip install -U scipy shap` and restart the interpreter.
  Only Figure 5 depends on SHAP; every other output is unaffected.
- **Two noise columns** (`Noise_1`, `Noise_2`) are drawn from a standard normal
  and injected deliberately as a negative control on the feature-selection
  stage. They are not a bug.
- **Network access** is needed on first run to download the datasets from
  Figshare. Afterwards `matminer` serves them from its local cache.

---

## 7. Data sources and attribution

- Ricci, F. et al. *An ab initio electronic transport database for inorganic
  materials.* Sci. Data **4**, 170085 (2017). doi:10.1038/sdata.2017.85
  Distributed via `matminer` as `boltztrap_mp` and
  `ricci_boltztrap_mp_tabular`, and on Figshare.
- Ward, L. et al. *Matminer: An open source toolkit for materials data mining.*
  Comput. Mater. Sci. **152**, 60-69 (2018).
- Ward, L., Agrawal, A., Choudhary, A., Wolverton, C. *A general-purpose machine
  learning framework for predicting properties of inorganic materials.*
  npj Comput. Mater. **2**, 16028 (2016). The Magpie descriptor set.
- Jain, A. et al. *The Materials Project.* APL Mater. **1**, 011002 (2013).

## 8. License

Source code under `scripts/` and `notebooks/` is released under the MIT License
(`LICENSE`). Result tables, figures and cached data under `results/`, `figures/`
and `data/` are released under CC BY 4.0 (`LICENSE-DATA`). The underlying
transport data remains under the license of Ricci et al. (2017).

## 9. Citation

See `CITATION.cff`. Please cite the manuscript and the Ricci et al. dataset.
