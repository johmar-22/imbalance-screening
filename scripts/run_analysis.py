"""run_analysis.py

Rare-class prediction of low hole effective mass from composition alone:
a leakage-controlled benchmark of imbalance-handling strategies.

Complete analysis for the manuscript. One file, no Google Drive, no Colab,
no manually downloaded data. The primary dataset is fetched from Figshare
through matminer on first run and cached locally.

BEFORE YOU RUN
--------------
1. Install the environment:

       pip install -r requirements.txt

   Python 3.10 or 3.11. If the SHAP section later reports an import error,
   run `pip install -U scipy shap` and restart the interpreter. Only Figure 5
   depends on SHAP; every other output is unaffected.

2. Optional. A GPU shortens the run from roughly 6-10 h to 1.5-2.5 h. The
   script detects it automatically and falls back to CPU with identical
   results to within floating-point accumulation order.

3. Optional. A free Materials Project key adds band gap, gap character and
   energy above hull to Top100_Candidates.csv (manuscript Table 3):

       export MP_API_KEY=your_key_here      # PowerShell: $env:MP_API_KEY="..."

   The key is read from the environment only. Never write it into this file.

4. Run:

       python run_analysis.py --outdir ../results_reproduced --quick   # smoke test
       python run_analysis.py --outdir ../results_reproduced           # full run

   `--quick` is a smoke test only. The reported numbers come from the full
   configuration: 15 folds (5 x 3 repeats), 25 search iterations, 6 strategies,
   5 target cutoffs.

DESIGN NOTES
------------
Model inputs are Magpie elemental-property descriptors computed from the
chemical formula alone. They are available for any hypothetical composition
before any DFT or BoltzTraP calculation is run, so the task is a genuine
pre-calculation triage decision and not a re-description of the target.

All cross-validation is GROUPED by normalized composition. The database
contains polymorphs sharing one formula (SiO2 occurs dozens of times) and a
composition-only model gives them identical feature vectors, so random
K-fold would score the model on an input it had already memorized. The
script asserts at run time that no composition appears on both sides of any
split. This cuts both ways: polymorphs of one formula can fall on opposite
sides of the m_p cutoff, giving identical inputs different labels. That is
irreducible label noise and it sets a ceiling on achievable performance.

Two Gaussian columns (Noise_1, Noise_2) are injected deliberately as a
negative control on the feature-selection stage. They are not a bug.

Design choices grounded in the literature:
  - Repeated stratified CV, 5 folds x 3 repeats = 15 estimates, because at
    5 folds the minimum Wilcoxon p is 0.0625 and significance is unreachable
    by construction (Raschka 2018).
  - SMOTE in high dimensions is treated as a hypothesis, not an assumption.
    Blagus and Lusa (2013) showed it often fails to help in high-dimensional
    data. The ~132-dimensional feature space here is a case where it may not.
  - Decision-threshold calibration is a first-class strategy, not an
    afterthought (Abdelhamid et al. 2024).
  - Probability calibration is measured (Brier, ECE), because resampling can
    improve discrimination while degrading calibration (Welvaars et al. 2023).
  - Average precision is reported as the threshold-free primary metric, since
    PR curves are more informative than ROC under strong imbalance.

Precedent for composition to transport prediction: Antunes et al. (2022),
Mach. Learn.: Sci. Technol. Composition to band gap: Zhuo et al. (2018),
J. Phys. Chem. Lett.; Vera de la Garza et al. (2026), Next Materials.
"""

from sklearn.calibration import CalibratedClassifierCV, calibration_curve

"""## Setup: configuration, dependencies, GPU detection"""

import os
import sys
import json
import time
import warnings
import subprocess
from functools import partial

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

"""## 0. CONFIGURATION

Command line (all optional):

    python run_analysis.py --outdir ../results
    python run_analysis.py --outdir ../results --quick      # ~10 min smoke test
    python run_analysis.py --outdir ../results --no-resume  # ignore checkpoints

Environment variables:

    MP_API_KEY   optional Materials Project key. Adds band gap, gap character
                 and energy above hull to Top100_Candidates.csv (Table 3).
                 Every other output is produced without it.
"""

import argparse

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # notebook / interactive
    _HERE = os.getcwd()
_DEFAULT_OUT = os.path.join(os.path.dirname(_HERE), 'results')

_p = argparse.ArgumentParser(
    description='Rare-class prediction of low hole effective mass from '
                'composition alone. Writes every table and figure of the '
                'manuscript into --outdir.')
_p.add_argument('--outdir', default=os.environ.get('ML_OUTDIR', _DEFAULT_OUT),
                help='directory for CSVs, figures and checkpoints '
                     '(default: ../results)')
_p.add_argument('--datadir', default=os.environ.get('ML_DATADIR', _HERE),
                help='directory searched for an optional local '
                     'boltztrap_mp.csv; unused unless --data-source csv')
_p.add_argument('--data-source', dest='data_source', default='figshare',
                choices=['figshare', 'csv'],
                help='where the primary dataset comes from (default: figshare)')
_p.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
_p.add_argument('--quick', action='store_true',
                help='smoke test: 1 repeat, 8 search iterations. NOT the '
                     'configuration behind the reported numbers.')
_p.add_argument('--no-resume', action='store_false', dest='resume',
                help='ignore existing checkpoints and recompute everything')
_ARGS, _UNPARSED = _p.parse_known_args()   # parse_known_args keeps Jupyter happy

QUICK_MODE = _ARGS.quick    # True = fast smoke test; False = full paper run
RNG        = _ARGS.seed
TARGET_THRESHOLD = 1.0     # m_p < 1.0 m_e defines the positive (rare) class

N_SPLITS  = 5
N_REPEATS = 1 if QUICK_MODE else 3
N_ITER    = 8 if QUICK_MODE else 25
K_BEST    = 30             # SelectKBest dimensionality reduction

def _ensure(mod, pip_name=None):
    try:
        __import__(mod)
    except ImportError:
        print(f"  installing {pip_name or mod} ...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                               pip_name or mod])

for _m, _p in [('imblearn', 'imbalanced-learn'), ('xgboost', 'xgboost>=2.0'),
               ('pymatgen', 'pymatgen'), ('matminer', 'matminer'),
               ('shap', 'shap'), ('statsmodels', 'statsmodels')]:
    _ensure(_m, _p)

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, fbeta_score, make_scorer, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold,
    cross_val_predict,
)
from sklearn.preprocessing import StandardScaler

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import BorderlineSMOTE, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

from scipy.stats import wilcoxon, spearmanr
from statsmodels.stats.contingency_tables import mcnemar

# -- Paths --------------------------------------------------------------------
# No Google Drive dependency. The output directory comes from --outdir, or the
# ML_OUTDIR environment variable, or defaults to ../results next to this script.
# DRIVE_ROOT is only the search path for an optional local boltztrap_mp.csv; by
# default the dataset is fetched from Figshare, so nothing must be placed there.
save_dir   = _ARGS.outdir
DRIVE_ROOT = _ARGS.datadir
os.makedirs(save_dir, exist_ok=True)
print(f"Outputs -> {os.path.abspath(save_dir)}")

# -- GPU ----------------------------------------------------------------------
def _gpu():
    try:
        return subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0
    except Exception:
        return False

USE_GPU    = _gpu()
XGB_DEVICE = 'cuda' if USE_GPU else 'cpu'
N_JOBS     = 1 if USE_GPU else -1   # one process per GPU avoids device contention
print(f"XGBoost device: {XGB_DEVICE}"
      + ("" if USE_GPU else "  [no GPU found; CPU fallback, results identical]"))

print("\n" + "!"*72)
if QUICK_MODE:
    print("  QUICK_MODE = True  ->  SMOKE TEST ONLY. DO NOT REPORT THESE NUMBERS.")
    print(f"  folds={N_SPLITS}x{N_REPEATS}   search_iter={N_ITER}   "
          f"strategies=4 of 6   cutoffs=3 of 5")
    print("  Set QUICK_MODE = False for the values that go in the manuscript.")
else:
    print("  QUICK_MODE = False  ->  FULL RUN (publication configuration)")
    print(f"  folds={N_SPLITS}x{N_REPEATS}   search_iter={N_ITER}   "
          f"strategies=6   cutoffs=5")
    print("  Expect roughly 1.5 to 2.5 hours on a Colab Pro GPU.")
print("!"*72)

"""## 0a. CHECKPOINTS  (crash / disconnect recovery)

Long loops write a small file to Drive after each unit of work.
A rerun reloads finished units instead of recomputing them.
Set `RESUME = False`, or call `ckpt_clear()`, to force a full recompute.
"""

# =============================================================================
# 0a. CHECKPOINTS  (crash / disconnect recovery only)
#
# Colab drops runtimes. Section 4 alone is 1.5-2.5 h, so losing it to a
# disconnect is expensive. Every long loop below writes a small checkpoint file
# to Drive after each unit of work. On a rerun those units are reloaded instead
# of recomputed, and the notebook picks up where it stopped.
#
# This changes nothing about the analysis. Every unit of work is already a pure
# function of its inputs: RandomizedSearchCV, the resamplers and XGBoost all
# take a fixed random_state, and no loop iteration depends on a previous one.
# A resumed run therefore produces the same numbers as an uninterrupted run.
#
# SAFETY. Each file stores a fingerprint of the settings that affect results
# (QUICK_MODE, RNG, TARGET_THRESHOLD, fold and search counts, K_BEST, the
# strategy list, the search space, and the dataset size). If any of those
# change, the checkpoint is ignored and the work is redone. Checkpoints are
# keyed to configuration, not to code: if you edit the body of a loop, clear
# them yourself with ckpt_clear().
#
#   RESUME = False    ->  ignore existing checkpoints, recompute everything
#   ckpt_clear()      ->  delete all checkpoints
#   ckpt_clear('cv_') ->  delete only the Section 4 fold checkpoints
# =============================================================================
import pickle, hashlib, glob as _glob

RESUME   = _ARGS.resume
CKPT_DIR = os.path.join(save_dir, '_checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

_FP_KEYS = ['QUICK_MODE', 'RNG', 'TARGET_THRESHOLD', 'N_SPLITS', 'N_REPEATS',
            'N_ITER', 'K_BEST']

def _fingerprint():
    """Hash of everything that would change the numbers. Evaluated lazily, so
    it picks up STRATEGIES, PARAM_DIST and the dataset once they exist."""
    g = globals()
    d = {k: g.get(k) for k in _FP_KEYS}
    d['STRATEGIES'] = list(g.get('STRATEGIES') or [])
    d['PARAM_DIST'] = sorted((k, list(v)) for k, v in (g.get('PARAM_DIST') or {}).items())
    d['n_rows'] = int(len(g['X'])) if 'X' in g else None
    d['n_pos'] = int(g['y'].sum()) if 'y' in g else None
    return hashlib.md5(repr(sorted(d.items(), key=str)).encode()).hexdigest()[:12]

def ckpt_path(tag):
    safe = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in str(tag))
    return os.path.join(CKPT_DIR, safe + '.pkl')

def ckpt_save(tag, obj):
    """Write atomically: a disconnect mid-write cannot leave a corrupt file."""
    p = ckpt_path(tag)
    try:
        tmp = p + '.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump({'fp': _fingerprint(), 'obj': obj}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)
    except Exception as e:
        # A failed checkpoint must never kill a long run.
        print(f"    [checkpoint write failed for {tag}: {e}]")

def ckpt_load(tag):
    """Return the stored object, or None on miss, stale fingerprint or damage."""
    if not RESUME:
        return None
    p = ckpt_path(tag)
    if not os.path.exists(p):
        return None
    try:
        with open(p, 'rb') as fh:
            rec = pickle.load(fh)
    except Exception:
        print(f"    [checkpoint {tag} unreadable, recomputing]")
        return None
    if rec.get('fp') != _fingerprint():
        print(f"    [checkpoint {tag} was written under a different "
              f"configuration, recomputing]")
        return None
    return rec['obj']

def ckpt_clear(prefix=''):
    n = 0
    for f in _glob.glob(os.path.join(CKPT_DIR, prefix + '*.pkl')):
        os.remove(f); n += 1
    print(f"  cleared {n} checkpoint file(s) matching '{prefix}*'")

_existing = sorted(os.path.basename(f)[:-4]
                   for f in _glob.glob(os.path.join(CKPT_DIR, '*.pkl')))
print("\n" + "="*72)
print("  0a. CHECKPOINTS")
print("="*72)
print(f"  directory: {CKPT_DIR}")
print(f"  RESUME = {RESUME}")
if _existing:
    print(f"  {len(_existing)} checkpoint file(s) found:")
    for _e in _existing[:24]:
        print(f"    {_e}")
    if len(_existing) > 24:
        print(f"    ... and {len(_existing)-24} more")
    print("  Matching units will be reloaded instead of recomputed.")
else:
    print("  none found; this run starts from scratch and will write them as it goes.")

"""## 0b. PUBLICATION FIGURE STANDARD

Nature-family figure specification:  
width      89 mm (single column) or 183 mm (double column)  
text       sans-serif, editable (TrueType) so the publisher can typeset it  
raster     600 dpi for TIFF/PNG; PDF is vector and resolution independent  
colour     colourblind-safe palette, distinguishable in greyscale  
Font sizes below are 8-9 pt rather than the 5-7 pt minimum some guides quote.  
That is deliberate: legibility on screen and in review matters more than  
hitting the floor, and no journal rejects a figure for readable text. Lower  
FS_BASE to 7 if a specific journal insists.
"""

MM = 1 / 25.4
W1COL, W2COL = 89 * MM, 183 * MM      # 3.50 in, 7.20 in
FS_BASE = 8

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         FS_BASE,
    'axes.labelsize':    FS_BASE + 1,
    'axes.titlesize':    FS_BASE + 1,
    'xtick.labelsize':   FS_BASE,
    'ytick.labelsize':   FS_BASE,
    'legend.fontsize':   FS_BASE - 0.5,
    'axes.linewidth':    0.7,
    'grid.linewidth':    0.4,
    'lines.linewidth':   1.4,
    'lines.markersize':  3.5,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size':  2.6,
    'ytick.major.size':  2.6,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'legend.frameon':    False,
    'figure.dpi':        150,
    'savefig.dpi':       600,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype':      42,     # embed TrueType, keeps text editable
    'ps.fonttype':       42,
})

# Colourblind-safe (Okabe-Ito), ordered so adjacent series stay distinct in
# greyscale as well as colour.
CB = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
      '#E69F00', '#56B4E9', '#000000', '#999999']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

def save_fig(fig, name, formats=('pdf', 'tiff', 'png')):
    """Save one figure at publication quality in several formats.

    PDF is vector (preferred by most publishers); TIFF and PNG are written at
    600 dpi for journals that require raster. Text stays editable in the PDF.
    """
    paths = []
    for ext in formats:
        p = os.path.join(save_dir, f'{name}.{ext}')
        kw = {'dpi': 600, 'bbox_inches': 'tight', 'pad_inches': 0.02}
        if ext == 'tiff':
            kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        try:
            fig.savefig(p, format=ext, **kw)
            paths.append(os.path.basename(p))
        except Exception as e:
            print(f"    [{ext} save skipped: {e}]")
    plt.close(fig)
    print(f"  saved: {', '.join(paths)}")

def panel_label(ax, letter, dx=-0.16, dy=1.04):
    """Bold lowercase panel letter in the Nature style."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=FS_BASE + 2, fontweight='bold', va='top', ha='left')

def make_xgb(**kw):
    return XGBClassifier(eval_metric='logloss', tree_method='hist',
                         device=XGB_DEVICE, random_state=RNG, verbosity=0, **kw)

f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)

"""## 1. DATA LOADING AND CLEANING

"""

print("\n" + "="*72)
print("  1. DATA")
print("="*72)

# -----------------------------------------------------------------------------
# NO MANUAL FILE REQUIRED. matminer hosts this exact dataset on Figshare under
# the name 'boltztrap_mp' (8,924 compounds; columns formula, mpid, m_n, m_p,
# pf_n, pf_p, s_n, s_p, structure). It is the same BoltzTraP Materials Project
# data as the local CSV, so the notebook downloads and caches it on first run.
# The larger sibling 'ricci_boltztrap_mp_tabular' (47,737 compounds) is used in
# Section 10 as a genuinely external hold-out.
#
# Source: Ricci F. et al., Sci. Data 4, 170085 (2017), doi:10.1038/sdata.2017.85
#
# Set DATA_SOURCE = 'csv' to use a local file instead.
# -----------------------------------------------------------------------------
DATA_SOURCE = _ARGS.data_source   # 'figshare' (default) or 'csv'
LOCAL_CSV   = os.path.join(DRIVE_ROOT, 'boltztrap_mp.csv')

SIX = ['m_n', 'PF_n', 'S_n', 'm_p', 'PF_p', 'S_p']

def _standardise(d):
    """Map matminer's lowercase column names onto the names used here."""
    ren = {}
    for c in d.columns:
        lc = str(c).strip().lower()
        if   lc == 'pf_n': ren[c] = 'PF_n'
        elif lc == 'pf_p': ren[c] = 'PF_p'
        elif lc == 's_n':  ren[c] = 'S_n'
        elif lc == 's_p':  ren[c] = 'S_p'
        elif lc == 'm_n':  ren[c] = 'm_n'
        elif lc == 'm_p':  ren[c] = 'm_p'
        elif lc in ('mpid', 'mp_id', 'material_id'): ren[c] = 'mpid'
        elif lc in ('formula', 'pretty_formula', 'full_formula'): ren[c] = 'formula'
    return d.rename(columns=ren)

def load_main():
    if DATA_SOURCE == 'figshare':
        try:
            from matminer.datasets import load_dataset
            d = _standardise(load_dataset('boltztrap_mp'))
            missing = [c for c in SIX + ['formula'] if c not in d.columns]
            if missing:
                raise RuntimeError(f"columns not found after mapping: {missing}. "
                                   f"available: {list(d.columns)}")
            print(f"  source: matminer/Figshare 'boltztrap_mp' ({len(d):,} rows)")
            return d
        except Exception as e:
            print(f"  Figshare load failed ({e})")
            print("  falling back to the local CSV ...")
    d = _standardise(pd.read_csv(LOCAL_CSV))
    print(f"  source: local CSV {LOCAL_CSV} ({len(d):,} rows)")
    return d

df = load_main()
n_raw = len(df)

HAS_MPID = 'mpid' in df.columns
if HAS_MPID:
    df = df.drop_duplicates(subset='mpid', keep='first').reset_index(drop=True)
else:
    print("  WARNING: no 'mpid' column; a surrogate id is created. The external")
    print("  Ricci hold-out cannot remove training overlap without real ids,")
    print("  so that test will be skipped to avoid an optimistic estimate.")
    df['mpid'] = [f'row-{i}' for i in range(len(df))]
df = df.dropna(subset=SIX + ['formula']).reset_index(drop=True)
print(f"  raw {n_raw:,} -> retained {len(df):,}")

df['target'] = (df['m_p'] < TARGET_THRESHOLD).astype(int)
y = df['target']
base_rate = float(y.mean())
print(f"  positives (m_p < {TARGET_THRESHOLD}): {int(y.sum())} "
      f"({100*base_rate:.2f}%)  imbalance {int((y==0).sum())/max(int(y.sum()),1):.1f}:1")

# -----------------------------------------------------------------------------
# 1b. COMPOSITION GROUPING  (mandatory once features are composition-only)
#
# The database contains polymorphs: several mp-ids that share one chemical
# formula (SiO2 occurs dozens of times). A composition-only model assigns those
# entries IDENTICAL feature vectors. Under random K-fold, some copies land in
# training and others in test, so the model can memorise a label and be scored
# on an identical input. This is the "repeated subjects" leakage documented by
# Rosenblatt et al. (2024, Nat. Commun.) and the dataset-redundancy problem
# quantified by Li et al. (2023, Nat. Commun.) and MD-HIT (Li et al., 2024,
# npj Comput. Mater.), both of which show random splitting overestimates
# performance on redundant materials data.
#
# Every split in this script is therefore grouped by normalised composition:
# all polymorphs of a formula go to the same side of every split.
#
# Note this cuts both ways and must be stated in the manuscript: polymorphs of
# one formula can fall on opposite sides of the m_p cutoff, so identical inputs
# carry different labels. That is irreducible label noise for a composition-only
# model and sets a ceiling on achievable performance.
# -----------------------------------------------------------------------------
def norm_formula(f):
    try:
        return Composition(f).reduced_formula
    except Exception:
        return str(f)

df['comp_group'] = df['formula'].apply(norm_formula)
groups = df['comp_group'].values

_vc = df['comp_group'].value_counts()
_n_dup_groups = int((_vc > 1).sum())
_n_dup_rows = int(_vc[_vc > 1].sum())
_straddle = df.groupby('comp_group')['m_p'].agg(
    lambda v: (v.min() < TARGET_THRESHOLD) and (v.max() >= TARGET_THRESHOLD))
print(f"  unique compositions: {df['comp_group'].nunique():,}")
print(f"  compositions with >1 polymorph: {_n_dup_groups:,} "
      f"covering {_n_dup_rows:,} rows ({100*_n_dup_rows/len(df):.1f}%)")
print(f"  compositions whose polymorphs straddle the cutoff: {int(_straddle.sum()):,}")
print("  -> all cross-validation is GROUPED by composition to prevent")
print("     identical feature vectors appearing in train and test.")
pd.DataFrame({'composition': _vc.index, 'n_entries': _vc.values}).to_csv(
    os.path.join(save_dir, 'Composition_Redundancy.csv'), index=False)

# The seven counts quoted in Section 3.3 of the manuscript, plus the identity of
# the compositions whose polymorphs straddle the cutoff. The Data availability
# statement promises both, so they are written to disk rather than only printed.
_agg_ms = df.groupby('comp_group')['m_p'].agg(['min', 'max', 'count'])
_str_tbl = _agg_ms[(_agg_ms['min'] < TARGET_THRESHOLD)
                   & (_agg_ms['max'] >= TARGET_THRESHOLD)]
pd.DataFrame([{
    'Rows': len(df),
    'Unique compositions': int(df['comp_group'].nunique()),
    'Compositions with >1 polymorph': _n_dup_groups,
    'Rows in multi-polymorph compositions': _n_dup_rows,
    'Share of rows': float(_n_dup_rows / len(df)),
    'Compositions straddling the cutoff': int(len(_str_tbl)),
    'Cutoff (m_e)': TARGET_THRESHOLD,
}]).to_csv(os.path.join(save_dir, 'Composition_Structure.csv'), index=False)
(_str_tbl.reset_index()
 .rename(columns={'comp_group': 'composition', 'min': 'm_p_min',
                  'max': 'm_p_max', 'count': 'n_entries'})
 .to_csv(os.path.join(save_dir, 'Straddling_Compositions.csv'), index=False))
print(f"  composition structure -> Composition_Structure.csv, "
      f"Straddling_Compositions.csv ({len(_str_tbl)} straddling)")

"""## 2. FEATURES  (R2-1: composition only, no DFT required)

Magpie elemental-property statistics are functions of the chemical formula  
alone. A practitioner can evaluate them for any hypothetical composition  
without running DFT or BoltzTraP, so a model built on them addresses a  
decision that actually arises before the calculation is performed.
"""

print("\n" + "="*72)
print("  2. MAGPIE COMPOSITION FEATURES (formula only)")
print("="*72)

CACHE = os.path.join(save_dir, 'magpie_cache.csv')
if os.path.exists(CACHE):
    feat = pd.read_csv(CACHE)
    print(f"  loaded cache: {CACHE}")
else:
    comps, keep = [], []
    for i, f in enumerate(df['formula']):
        try:
            comps.append(Composition(f)); keep.append(i)
        except Exception:
            pass
    if len(keep) < len(df):
        print(f"  dropped {len(df)-len(keep)} unparsable formula(s)")
        df = df.iloc[keep].reset_index(drop=True)
        y  = df['target']
    ep = ElementProperty.from_preset('magpie')
    print(f"  featurizing {len(comps):,} compositions ...")
    feat = ep.featurize_dataframe(pd.DataFrame({'composition': comps}),
                                  col_id='composition',
                                  ignore_errors=True).drop(columns=['composition'])
    feat.to_csv(CACHE, index=False)
    print(f"  cached -> {CACHE}")

feat = feat.reset_index(drop=True)
assert len(feat) == len(df), "feature/label length mismatch"

magpie_cols = list(feat.columns)

# Two Gaussian noise columns. Purpose (R1-5): a NEGATIVE CONTROL that verifies
# the selection stage never prefers an uninformative feature. They are not
# evidence that feature selection contributes predictive power.
_rng = np.random.default_rng(RNG)
feat['Noise_1'] = _rng.normal(0, 1, len(feat))
feat['Noise_2'] = _rng.normal(0, 1, len(feat))

ALL_FEATURES = magpie_cols + ['Noise_1', 'Noise_2']
X = feat[ALL_FEATURES]
NTYPE = ['m_n', 'PF_n', 'S_n']

print(f"  X shape: {X.shape}  ({len(magpie_cols)} Magpie + 2 noise controls)")
print(f"  SelectKBest k = {K_BEST}")
print("  NOTE: n-type transport descriptors are NOT model inputs here.")
print("        They appear only in the Section 11 co-occurrence test and as")
print("        reference columns in the candidate table.")

"""## 3. STRATEGIES

Fold-safe resampling is a correctness requirement of the design (R2-3), not a  
contribution: imblearn.Pipeline guarantees every resampler sees training data  
only. scale_pos_weight is likewise computed per training fold (R1-3).
"""

def build(strategy):
    steps = [('imputer', SimpleImputer(strategy='median')),
             ('fs',      SelectKBest(score_func=f_classif, k=K_BEST)),
             ('scaler',  StandardScaler())]
    if strategy == 'B-SMOTE':
        steps.append(('res', BorderlineSMOTE(random_state=RNG, kind='borderline-1')))
    elif strategy == 'SMOTE':
        steps.append(('res', SMOTE(random_state=RNG)))
    elif strategy == 'ADASYN':
        steps.append(('res', ADASYN(random_state=RNG)))
    elif strategy == 'RUS':
        steps.append(('res', RandomUnderSampler(random_state=RNG)))
    steps.append(('xgb', make_xgb()))
    return ImbPipeline(steps)

STRATEGIES = ['None', 'ClassWeight', 'B-SMOTE', 'SMOTE', 'ADASYN', 'RUS']
if QUICK_MODE:
    STRATEGIES = ['None', 'ClassWeight', 'B-SMOTE', 'RUS']

PARAM_DIST = {
    'xgb__max_depth':     [3, 5, 7],
    'xgb__learning_rate': [0.03, 0.05, 0.1, 0.2],
    'xgb__subsample':     [0.6, 0.8, 1.0],
    'xgb__n_estimators':  [100, 200, 400],
    'xgb__min_child_weight': [1, 3, 5],
}

"""## 4. REPEATED NESTED CROSS-VALIDATION

Threshold protocol (R1-4):  
FIXED   : all metrics at the a priori 0.5 probability threshold.  
TUNED   : threshold chosen to maximise F2 on inner-CV predictions made on  
the TRAINING fold only, then frozen and applied to the held-out  
fold. The test fold never participates in threshold selection.  
Calibration (U4) and average precision (U5) are recorded per fold.
"""

print("\n" + "="*72)
print(f"  4. REPEATED NESTED CV  ({N_SPLITS} folds x {N_REPEATS} repeats)")
print("     resampling confined to training folds (design requirement)")
print("     thresholds: fixed 0.5 and inner-CV tuned (leakage-free)")
print("="*72)

def grouped_splits(Xd, yd, gd, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=RNG):
    """Repeated stratified splits that never break a composition group.

    StratifiedGroupKFold keeps every polymorph of a formula on one side of the
    split while approximately preserving the class ratio. Repeats use different
    shuffles to give more paired estimates for the statistical comparison.
    """
    out = []
    for rep in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed + rep)
        out.extend(list(sgkf.split(Xd, yd, groups=gd)))
    return out

SPLITS = grouped_splits(X, y, groups)
THR_GRID = np.linspace(0.05, 0.95, 91)

# Sanity check: no composition may appear on both sides of any split.
for _tr, _te in SPLITS:
    assert not (set(groups[_tr]) & set(groups[_te])), "group leakage in split"
print(f"  {len(SPLITS)} grouped splits verified leakage-free "
      f"(no shared composition between train and test)")

def ece(y_true, prob, bins=10):
    """Expected calibration error."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(prob, edges[1:-1])
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += (m.sum() / len(prob)) * abs(y_true[m].mean() - prob[m].mean())
    return float(tot)

METRICS = ['Accuracy', 'Precision', 'Recall', 'F1', 'F2', 'MCC',
           'AvgPrecision', 'ROC-AUC', 'Brier', 'ECE',
           'F2_tuned', 'Precision_tuned', 'Recall_tuned', 'Threshold']

fold_metrics = {s: {m: [] for m in METRICS} for s in STRATEGIES}
oof_prob = {s: np.zeros(len(X)) for s in STRATEGIES}
oof_pred = {s: np.zeros(len(X)) for s in STRATEGIES}
oof_seen = np.zeros(len(X))
noise_audit, best_params_log = [], []

t0 = time.time()
for fold, (tr, te) in enumerate(SPLITS):
    # -- checkpoint: reload this fold if it completed in an earlier run -------
    _ck = ckpt_load(f'cv_fold{fold+1:02d}')
    if _ck is not None:
        print(f"  fold {fold+1}/{len(SPLITS)} [restored from checkpoint]")
        for s in STRATEGIES:
            for m in METRICS:
                fold_metrics[s][m].append(_ck['metrics'][s][m])
            if fold < N_SPLITS:
                oof_prob[s][_ck['te']] = _ck['prob'][s]
                oof_pred[s][_ck['te']] = _ck['pred'][s]
        if fold < N_SPLITS:
            oof_seen[_ck['te']] = 1
        noise_audit.extend(_ck['noise'])
        best_params_log.extend(_ck['params'])
        continue

    print(f"  fold {fold+1}/{len(SPLITS)} [{time.time()-t0:.0f}s]")
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ytr, yte = y.iloc[tr], y.iloc[te]
    gtr = groups[tr]
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)   # training fold only

    # Inner splits are grouped too, otherwise hyperparameters are selected on
    # leaked estimates even though the outer split is clean.
    inner_cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True,
                                         random_state=RNG).split(Xtr, ytr, gtr))

    for s in STRATEGIES:
        pipe = build(s)
        if s == 'ClassWeight':
            pipe.set_params(xgb__scale_pos_weight=spw)
        # Inner selection uses average precision, a threshold-free ranking
        # metric. Tuning directly on F2 pushes the search toward degenerate
        # high-recall solutions whose apparent F2 is an artefact of the 0.5
        # cut rather than better ranking; the operating threshold is chosen
        # separately below.
        search = RandomizedSearchCV(pipe, PARAM_DIST, n_iter=N_ITER,
                                    cv=inner_cv, scoring='average_precision',
                                    random_state=RNG, n_jobs=N_JOBS, refit=True)
        search.fit(Xtr, ytr)
        best = search.best_estimator_
        best_params_log.append({'Fold': fold+1, 'Strategy': s, **search.best_params_})

        prob = best.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)

        # first repeat only fills the OOF arrays (one prediction per compound)
        if fold < N_SPLITS:
            oof_prob[s][te] = prob
            oof_pred[s][te] = pred
            oof_seen[te] = 1

        yte_v = yte.values
        M = fold_metrics[s]
        M['Accuracy'].append(accuracy_score(yte_v, pred))
        M['Precision'].append(precision_score(yte_v, pred, zero_division=0))
        M['Recall'].append(recall_score(yte_v, pred, zero_division=0))
        M['F1'].append(f1_score(yte_v, pred, zero_division=0))
        M['F2'].append(fbeta_score(yte_v, pred, beta=2, zero_division=0))
        M['MCC'].append(matthews_corrcoef(yte_v, pred))
        M['AvgPrecision'].append(average_precision_score(yte_v, prob))
        M['ROC-AUC'].append(roc_auc_score(yte_v, prob))
        M['Brier'].append(brier_score_loss(yte_v, prob))
        M['ECE'].append(ece(yte_v, prob))

        # -- leakage-free tuned threshold (R1-4, U3) --------------------------
        # Inner predictions come from grouped inner folds, so the threshold is
        # selected without either fold leakage or composition leakage.
        inner = cross_val_predict(clone(best), Xtr, ytr, cv=inner_cv,
                                  method='predict_proba', n_jobs=N_JOBS)[:, 1]
        f2s = [fbeta_score(ytr, (inner >= t).astype(int), beta=2, zero_division=0)
               for t in THR_GRID]
        t_star = float(THR_GRID[int(np.argmax(f2s))])
        pred_t = (prob >= t_star).astype(int)
        M['Threshold'].append(t_star)
        M['F2_tuned'].append(fbeta_score(yte_v, pred_t, beta=2, zero_division=0))
        M['Precision_tuned'].append(precision_score(yte_v, pred_t, zero_division=0))
        M['Recall_tuned'].append(recall_score(yte_v, pred_t, zero_division=0))

        # -- noise control audit (R1-5) --------------------------------------
        sel = np.array(ALL_FEATURES)[best.named_steps['fs'].get_support()]
        noise_audit.append({'Fold': fold+1, 'Strategy': s,
                            'Noise_1': 'Noise_1' in sel,
                            'Noise_2': 'Noise_2' in sel})

    # -- checkpoint: this fold is complete, persist it before starting the next
    ckpt_save(f'cv_fold{fold+1:02d}', {
        'te': te,
        'metrics': {s: {m: fold_metrics[s][m][-1] for m in METRICS}
                    for s in STRATEGIES},
        'prob': {s: oof_prob[s][te].copy() for s in STRATEGIES} if fold < N_SPLITS else {},
        'pred': {s: oof_pred[s][te].copy() for s in STRATEGIES} if fold < N_SPLITS else {},
        'noise': noise_audit[-len(STRATEGIES):],
        'params': best_params_log[-len(STRATEGIES):]})

print(f"  done in {time.time()-t0:.0f}s")

rows = []
for s in STRATEGIES:
    r = {'Strategy': s}
    for m in METRICS:
        v = fold_metrics[s][m]
        r[m] = f"{np.mean(v):.3f} +/- {np.std(v):.3f}"
    rows.append(r)
res = pd.DataFrame(rows).set_index('Strategy')
print("\n--- Main results (mean +/- std over all folds) ---")
print(res[['F2', 'F2_tuned', 'AvgPrecision', 'Precision', 'Recall',
           'MCC', 'Brier', 'ECE', 'Threshold']].to_string())
res.to_csv(os.path.join(save_dir, 'Main_Results.csv'))
pd.DataFrame(best_params_log).to_csv(
    os.path.join(save_dir, 'Best_Hyperparameters.csv'), index=False)

# Per-fold values behind Tables 1, 2, S2, S5 and S6. Exported so that
# run_extended_analysis.py, and any reader, can recompute the paired tests
# without unpickling a checkpoint. Checkpoints are gitignored; this is not.
_pf = []
for _s in STRATEGIES:
    for _i in range(len(SPLITS)):
        _row_pf = {'Strategy': _s, 'Fold': _i + 1}
        for _m in METRICS:
            _row_pf[_m] = fold_metrics[_s][_m][_i]
        _pf.append(_row_pf)
pd.DataFrame(_pf).to_csv(
    os.path.join(save_dir, 'Fold_Metrics_PerFold.csv'), index=False)
print(f"  per-fold metrics -> Fold_Metrics_PerFold.csv ({len(_pf)} rows)")

# Honest automatic read-out of the two literature hypotheses
best_f2  = max(STRATEGIES, key=lambda s: np.mean(fold_metrics[s]['F2']))
best_ap  = max(STRATEGIES, key=lambda s: np.mean(fold_metrics[s]['AvgPrecision']))
print(f"\n  highest mean F2 (fixed 0.5): {best_f2}")
print(f"  highest mean average precision (threshold-free): {best_ap}")
gain_thr = np.mean(fold_metrics[best_f2]['F2_tuned']) - np.mean(fold_metrics[best_f2]['F2'])
print(f"  F2 change from tuned threshold on {best_f2}: {gain_thr:+.3f}")
print("  [U2] If a resampler no longer wins in this ~132-dim space, report that")
print("       plainly: it is consistent with Blagus & Lusa (2013).")
print("  [U4] Compare Brier/ECE across strategies before recommending one for")
print("       probability-ranked screening.")

# =============================================================================
#  SECTION 4b.  DOES POST-HOC RECALIBRATION RESCUE THE RESAMPLED MODELS?
# =============================================================================
#
#  WHERE TO PUT THIS
#  -----------------
#  File: ml_project_v7_colab_checkpointed_v1.py
#
#  (1) Add ONE import to the sklearn import block (near line 216, next to
#      `from sklearn.base import clone`):
#
#          from sklearn.calibration import CalibratedClassifierCV, calibration_curve
#
#  (2) Paste EVERYTHING BELOW THE DASHED LINE immediately BEFORE line 876,
#      i.e. directly above the line that reads:
#
#          """## 5. STATISTICAL COMPARISON
#
#      It must come after Section 4 because it reuses SPLITS, STRATEGIES,
#      build(), ece(), THR_GRID and best_params_log. Nothing later depends
#      on it, so it can also be run on its own after a checkpointed restart.
#
#  WHY IT EXISTS
#  -------------
#  The manuscript reports that imbalance corrections raise expected calibration
#  error by a factor of about 27. The obvious referee question is: why not just
#  recalibrate the resampled model and keep the F2 gain? Recent work on tree
#  ensembles reports that one recalibration step removes much of the penalty.
#  This section answers the question with data rather than a concession.
#
#  THE FOUR THINGS THAT MAKE THE ANSWER METHODOLOGICALLY VALID
#  -----------------------------------------------------------
#  1. GROUPED INTERNAL CALIBRATION FOLDS.  CalibratedClassifierCV would
#     otherwise split the training fold with ordinary stratified K-fold. That
#     puts polymorphs of one composition on both sides of the internal split,
#     and because the composition-only descriptor map is many-to-one, the
#     calibrator would be fitted on points whose identical feature vector the
#     base model has already memorised. We pass `cv=inner_cv`, a precomputed
#     list of StratifiedGroupKFold index pairs, so the calibration split obeys
#     the same grouping constraint as every other split in the study. The
#     assertion in the loop enforces this at run time.
#
#  2. ensemble=False IS REQUIRED, NOT COSMETIC.  With sklearn's default
#     ensemble=True the returned probability is the AVERAGE over k internally
#     fitted models. That averaging is a model change and it moves the ranking,
#     which would confound the exact quantity being measured. With
#     ensemble=False a single calibrator is fitted on cross-validated
#     predictions and applied to one base model refitted on the whole training
#     fold, so the only difference from the uncalibrated model is a monotone
#     map on the scores. Verify with len(cal.calibrated_classifiers_) == 1.
#
#  3. THE CALIBRATOR SEES THE TRUE PRIOR.  imblearn resamples during fit only,
#     never during transform or predict, so the held-out internal partitions
#     used to fit the calibrator keep the natural 4.90% prevalence. The
#     calibration map therefore targets the real class prior, which is the
#     correct target for a screening campaign.
#
#  4. HYPERPARAMETERS ARE REUSED, NOT RE-SEARCHED.  Each fold reuses the
#     hyperparameters Section 4's randomized search already selected on that
#     fold's TRAINING partition (best_params_log). No test-fold information
#     enters, the model is held fixed, and the contrast isolates the
#     calibration layer alone. It also avoids repeating the expensive search.
#
#  WHAT THIS WILL SHOW, AND WHY IT STRENGTHENS THE PAPER
#  ----------------------------------------------------
#  Platt scaling is strictly monotone, so average precision and ROC-AUC come
#  back bit-identical and Spearman rho against the raw scores is exactly
#  1.000000. Isotonic regression is monotone but not strictly, so it merges
#  scores into ties and average precision moves slightly, in testing always
#  downward. Recalibration therefore CANNOT recover ranking quality that
#  resampling destroyed, because average precision is a rank statistic and
#  recalibration is a rank-preserving transformation. Meanwhile the F2
#  advantage the resamplers showed at the fixed 0.5 threshold collapses once
#  the probability scale is restored, which is direct evidence that the
#  advantage was an artefact of the shifted scale and not a better model.
#
#  RUNTIME
#  -------
#  About 12 model fits per strategy per fold: 1 base + 3 for the inner
#  threshold + 4 per calibration method. With 6 strategies and 15 folds that is
#  roughly 1,080 XGBoost fits. Checkpointed per fold, so it resumes safely.
#  To do the first repeat only, replace `enumerate(SPLITS)` with
#  `enumerate(SPLITS[:N_SPLITS])` in the loop header.
# -----------------------------------------------------------------------------

print("\n" + "="*72)
print("  4b. POST-HOC RECALIBRATION  (can it rescue the resampled models?)")
print("      grouped internal calibration folds; ensemble=False")
print("="*72)

CAL_METHODS = ['sigmoid', 'isotonic']          # Platt scaling, isotonic regression

# hyperparameters chosen by Section 4, keyed by (fold number, strategy)
_bp = {}
for _r in best_params_log:
    _bp[(_r['Fold'], _r['Strategy'])] = {k: v for k, v in _r.items()
                                         if k.startswith('xgb__')}

cal_rows = []
cal_oof  = {(s, c): np.zeros(len(X))
            for s in STRATEGIES for c in ['raw'] + CAL_METHODS}
cal_seen = np.zeros(len(X))
_t0 = time.time()

for fold, (tr, te) in enumerate(SPLITS):
    _ck = ckpt_load(f'cal_fold{fold+1:02d}')
    if _ck is not None:
        print(f"  fold {fold+1}/{len(SPLITS)} [restored from checkpoint]")
        cal_rows.extend(_ck['rows'])
        if fold < N_SPLITS:
            for k, v in _ck['oof'].items():
                cal_oof[k][_ck['te']] = v
            cal_seen[_ck['te']] = 1
        continue

    print(f"  fold {fold+1}/{len(SPLITS)} [{time.time()-_t0:.0f}s]")
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ytr, yte = y.iloc[tr], y.iloc[te]
    gtr, yte_v = groups[tr], yte.values
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)   # training fold only

    # Same grouped inner design and seed as Section 4, so the calibration split
    # is identical to the one used for hyperparameter selection.
    inner_cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True,
                                         random_state=RNG).split(Xtr, ytr, gtr))
    for _a, _b in inner_cv:
        assert not (set(gtr[_a]) & set(gtr[_b])), \
            "composition leaked across an internal calibration fold"

    _rows, _oof = [], {}
    for s in STRATEGIES:
        pipe = build(s)
        if s == 'ClassWeight':
            pipe.set_params(xgb__scale_pos_weight=spw)
        _p = _bp.get((fold + 1, s))
        if _p:
            pipe.set_params(**_p)

        # -- uncalibrated reference (reproduces the Section 4 numbers) --------
        base = clone(pipe)
        base.fit(Xtr, ytr)
        p_raw = base.predict_proba(Xte)[:, 1]

        # -- leakage-free tuned threshold, identical protocol to Section 4 ----
        inner = cross_val_predict(clone(pipe), Xtr, ytr, cv=inner_cv,
                                  method='predict_proba', n_jobs=N_JOBS)[:, 1]
        f2s = [fbeta_score(ytr, (inner >= t).astype(int), beta=2,
                           zero_division=0) for t in THR_GRID]
        t_star = float(THR_GRID[int(np.argmax(f2s))])

        def _row(tag, prob, rho, t_tuned):
            return {'Fold': fold + 1, 'Strategy': s, 'Calibration': tag,
                    'F2': fbeta_score(yte_v, (prob >= 0.5).astype(int),
                                      beta=2, zero_division=0),
                    'F2_tuned': (np.nan if t_tuned is None else
                                 fbeta_score(yte_v, (prob >= t_tuned).astype(int),
                                             beta=2, zero_division=0)),
                    'Threshold': (np.nan if t_tuned is None else t_tuned),
                    'AvgPrecision': average_precision_score(yte_v, prob),
                    'ROC-AUC': roc_auc_score(yte_v, prob),
                    'Brier': brier_score_loss(yte_v, prob),
                    'ECE': ece(yte_v, prob),
                    'SpearmanVsRaw': rho,
                    'nUniqueProb': int(len(np.unique(prob)))}

        # F2_tuned is reported for the uncalibrated model only. Calibration is
        # a monotone map, so the tuned operating point is the image of t_star
        # under that map and the tuned-threshold F2 is invariant by
        # construction; recomputing it would double the runtime to confirm an
        # algebraic identity.
        _rows.append(_row('raw', p_raw, 1.0, t_star))
        _oof[(s, 'raw')] = p_raw.copy()

        # -- calibrated variants ----------------------------------------------
        for meth in CAL_METHODS:
            try:
                cal = CalibratedClassifierCV(estimator=clone(pipe), method=meth,
                                             cv=inner_cv, ensemble=False,
                                             n_jobs=N_JOBS)
                cal.fit(Xtr, ytr)
                assert len(cal.calibrated_classifiers_) == 1, \
                    "ensemble=False must yield exactly one calibrator"
                p_cal = cal.predict_proba(Xte)[:, 1]
                rho = float(spearmanr(p_raw, p_cal).statistic)
                _rows.append(_row(meth, p_cal, rho, None))
                _oof[(s, meth)] = p_cal.copy()
            except Exception as e:
                # isotonic can fail if a calibration fold holds too few
                # positives; record the failure rather than abort the run.
                print(f"    [{s}/{meth} failed on fold {fold+1}: {e}]")

    cal_rows.extend(_rows)
    if fold < N_SPLITS:
        for k, v in _oof.items():
            cal_oof[k][te] = v
        cal_seen[te] = 1
    ckpt_save(f'cal_fold{fold+1:02d}',
              {'rows': _rows, 'te': te, 'oof': _oof if fold < N_SPLITS else {}})

print(f"  done in {time.time()-_t0:.0f}s")

cal_df = pd.DataFrame(cal_rows)
cal_df.to_csv(os.path.join(save_dir, 'Recalibration_PerFold.csv'), index=False)

_num = ['F2', 'AvgPrecision', 'ROC-AUC', 'Brier', 'ECE', 'SpearmanVsRaw']
_agg = cal_df.groupby(['Strategy', 'Calibration'])[_num].agg(['mean', 'std'])
cal_summary = pd.DataFrame(index=_agg.index)
for c in _num:
    cal_summary[c] = [f"{m:.3f} +/- {sd:.3f}"
                      for m, sd in zip(_agg[(c, 'mean')], _agg[(c, 'std')])]
_order = pd.MultiIndex.from_product([STRATEGIES, ['raw'] + CAL_METHODS],
                                    names=['Strategy', 'Calibration'])
cal_summary = cal_summary.reindex(_order).dropna(how='all')
cal_summary.to_csv(os.path.join(save_dir, 'Recalibration_Summary.csv'))

print("\n--- Recalibration: mean +/- std over all folds ---")
print(cal_summary.to_string())

# -----------------------------------------------------------------------------
#  Automatic read-out of the three claims this section has to settle
# -----------------------------------------------------------------------------
_g = cal_df.groupby(['Strategy', 'Calibration'])[_num].mean()
print("\n  READ-OUT")
for s in STRATEGIES:
    if (s, 'raw') not in _g.index:
        continue
    raw = _g.loc[(s, 'raw')]
    print(f"  {s:<12s} raw       ECE {raw['ECE']:.3f}   "
          f"AP {raw['AvgPrecision']:.3f}   F2@0.5 {raw['F2']:.3f}")
    for meth in CAL_METHODS:
        if (s, meth) not in _g.index:
            continue
        c = _g.loc[(s, meth)]
        d = 100 * (c['ECE'] - raw['ECE']) / max(raw['ECE'], 1e-12)
        print(f"  {'':<12s} {meth:<9s} ECE {c['ECE']:.3f} ({d:+.0f}%)  "
              f"AP {c['AvgPrecision']:.3f} ({c['AvgPrecision']-raw['AvgPrecision']:+.3f})  "
              f"F2@0.5 {c['F2']:.3f} ({c['F2']-raw['F2']:+.3f})  "
              f"rho {c['SpearmanVsRaw']:.6f}")

if 'sigmoid' in set(cal_df['Calibration']):
    _s, _r = _g.xs('sigmoid', level='Calibration'), _g.xs('raw', level='Calibration')
    _common = _s.index.intersection(_r.index)
    print(f"\n  max |change in average precision| under Platt: "
          f"{(_s.loc[_common,'AvgPrecision'] - _r.loc[_common,'AvgPrecision']).abs().max():.8f}")
    print(f"  max |change in ROC-AUC| under Platt:           "
          f"{(_s.loc[_common,'ROC-AUC'] - _r.loc[_common,'ROC-AUC']).abs().max():.8f}")
    print(f"  min Spearman rho against raw scores (Platt):   "
          f"{_s.loc[_common,'SpearmanVsRaw'].min():.6f}")
    print("  These must be ~0, ~0 and 1.000000. Platt scaling is strictly")
    print("  monotone; average precision is a rank statistic, so recalibration")
    print("  cannot restore a ranking that resampling has already degraded.")

# -----------------------------------------------------------------------------
#  Figure: reliability diagram before and after recalibration (pooled OOF)
# -----------------------------------------------------------------------------
try:
    _m = cal_seen == 1
    _yv = y.values[_m]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, tag, ttl in zip(axes, ['raw', 'sigmoid'],
                            ['Uncalibrated', 'After Platt scaling']):
        ax.plot([0, 1], [0, 1], ':', color='k', lw=1.2, label='Perfect calibration')
        for i, s in enumerate(STRATEGIES):
            p = cal_oof[(s, tag)][_m]
            if p.sum() == 0:
                continue
            try:
                fr, mp = calibration_curve(_yv, p, n_bins=10, strategy='uniform')
            except Exception:
                continue
            ax.plot(mp, fr, marker=MARKERS[i % len(MARKERS)], ms=4, lw=1.2,
                    color=CB[i % len(CB)], label=s)
        ax.set_xlabel('Mean predicted probability')
        ax.set_title(ttl, fontsize=FS_BASE)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel('Observed positive fraction')
    axes[1].legend(fontsize=FS_BASE - 2, frameon=False, loc='upper left')
    for ax, L in zip(axes, 'ab'):
        panel_label(ax, L)
    fig.tight_layout()
    save_fig(fig, 'Fig_Recalibration')
except Exception as e:
    print(f"  [recalibration figure skipped: {e}]")

# =============================================================================
#  SECTION 4c.  DIAGNOSTICS FOR THE RECALIBRATED MODELS
#  Paste immediately after Section 4b, before """## 5. STATISTICAL COMPARISON
#  Requires cal_oof / cal_seen from 4b to be in memory (same session).
#
#  PURPOSE
#  -------
#  Section 4b returns F2 = 0.000 for RUS under Platt scaling but 0.148 under
#  isotonic, from the same base model with Spearman rho 0.989 between them.
#  Two explanations compete and must be separated before either is written up:
#    (a) the posterior genuinely never exceeds 0.5 for this model;
#    (b) it does, but Platt's two-parameter global sigmoid cannot represent it.
#  Platt scaling is exact only when the class-conditional score distributions
#  are Gaussian with equal variance (Kull, Silva Filho & Flach, AISTATS 2017);
#  isotonic regression is non-parametric and can track a sharp upper tail
#  (Zadrozny & Elkan, 2002). Boosted trees are sigmoid-distorted to begin with
#  (Niculescu-Mizil & Caruana, ICML 2005), which is the regime here.
#
#  TEST 1 reports the calibrated probability distribution directly.
#  TEST 2 asks whether the calibrator under-predicts the top of the ranking,
#         by comparing observed positive rate with mean predicted probability
#         among the top-k compounds.
#  TEST 3 applies the closed-form prior-shift correction of Elkan (2001) and
#         Dal Pozzolo et al. (IEEE SSCI 2015), which has no fitted parameters:
#             p = beta*p_s / (beta*p_s - p_s + 1),  beta = n_pos/n_neg
#         For RUS to parity this maps p_s = 0.5 to the base rate exactly, and
#         p > 0.5 requires p_s > 1/(1+beta). VALID ONLY for random
#         undersampling and class weighting, which shift the prior alone.
#         NOT valid for SMOTE/B-SMOTE/ADASYN, which alter the class-conditional
#         density as well; this is the distinction drawn in the tree-ensemble
#         literature and it must be stated in the paper.
# =============================================================================

print("\n" + "="*72)
print("  4c. RECALIBRATION DIAGNOSTICS")
print("="*72)

_m   = cal_seen == 1
_yv  = y.values[_m]
_pos = int(_yv.sum()); _neg = int((_yv == 0).sum())
_beta = _pos / _neg                      # negatives retained by RUS to parity
print(f"  out-of-fold pool: {_m.sum()} compounds, {_pos} positives "
      f"({_yv.mean():.4f}), beta = {_beta:.5f}")
print(f"  analytic correction crosses 0.5 at balanced-model score "
      f"p_s > 1/(1+beta) = {1/(1+_beta):.6f}")

# -- TEST 1: where does the calibrated probability actually live? -------------
print("\n  TEST 1  calibrated probability distribution (pooled out-of-fold)")
print(f"  {'Strategy':<12s} {'Calib':<9s} {'max':>7s} {'p99.9':>7s} {'p99':>7s} "
      f"{'n>0.5':>6s} {'n>0.3':>6s} {'nUniq':>6s}")
diag1 = []
for s in STRATEGIES:
    for c in ['raw'] + CAL_METHODS:
        if (s, c) not in cal_oof:
            continue
        p = cal_oof[(s, c)][_m]
        if not np.any(p):
            continue
        r = {'Strategy': s, 'Calibration': c, 'max': p.max(),
             'p99.9': np.quantile(p, .999), 'p99': np.quantile(p, .99),
             'n_gt_0.5': int((p > .5).sum()), 'n_gt_0.3': int((p > .3).sum()),
             'n_unique': int(len(np.unique(p)))}
        diag1.append(r)
        print(f"  {s:<12s} {c:<9s} {r['max']:7.4f} {r['p99.9']:7.4f} "
              f"{r['p99']:7.4f} {r['n_gt_0.5']:6d} {r['n_gt_0.3']:6d} "
              f"{r['n_unique']:6d}")

# -- TEST 2: is the top of the ranking under-predicted? -----------------------
print("\n  TEST 2  tail reliability: observed positive rate vs mean predicted p")
print(f"  {'Strategy':<12s} {'Calib':<9s} {'k':>5s} {'observed':>9s} "
      f"{'predicted':>10s} {'gap':>7s}")
diag2 = []
for s in STRATEGIES:
    for c in ['raw'] + CAL_METHODS:
        if (s, c) not in cal_oof:
            continue
        p = cal_oof[(s, c)][_m]
        if not np.any(p):
            continue
        order = np.argsort(-p)
        for k in (25, 100, 250):
            idx = order[:k]
            obs, prd = _yv[idx].mean(), p[idx].mean()
            diag2.append({'Strategy': s, 'Calibration': c, 'k': k,
                          'observed': obs, 'predicted': prd, 'gap': obs - prd})
            if c != 'raw' or k == 100:
                print(f"  {s:<12s} {c:<9s} {k:5d} {obs:9.3f} {prd:10.3f} "
                      f"{obs-prd:+7.3f}")

# -- TEST 3: closed-form prior-shift correction (RUS and ClassWeight only) ----
def elkan_correct(ps, beta):
    """Elkan (2001); Dal Pozzolo et al. (2015). Prior-shift only."""
    ps = np.clip(ps, 1e-12, 1 - 1e-12)
    return beta * ps / (beta * ps - ps + 1.0)

print("\n  TEST 3  closed-form prior correction (no fitted parameters)")
print("          valid for prior shift only: RUS, ClassWeight.")
print("          NOT applied to SMOTE/B-SMOTE/ADASYN, which also change the")
print("          class-conditional density.")
diag3 = []
for s in ['RUS', 'ClassWeight']:
    if (s, 'raw') not in cal_oof:
        continue
    p_s = cal_oof[(s, 'raw')][_m]
    if not np.any(p_s):
        continue
    p_c = elkan_correct(p_s, _beta)
    row = {'Strategy': s,
           'raw_max': p_s.max(),
           'n_raw_gt_threshold': int((p_s > 1/(1+_beta)).sum()),
           'corrected_max': p_c.max(),
           'n_corrected_gt_0.5': int((p_c > .5).sum()),
           'ECE_corrected': ece(_yv, p_c),
           'F2_corrected@0.5': fbeta_score(_yv, (p_c >= .5).astype(int),
                                           beta=2, zero_division=0),
           'AP_corrected': average_precision_score(_yv, p_c),
           'rho_vs_raw': float(spearmanr(p_s, p_c).statistic)}
    diag3.append(row)
    print(f"  {s}:")
    print(f"    raw max score            {row['raw_max']:.6f}")
    print(f"    raw scores > {1/(1+_beta):.4f}      {row['n_raw_gt_threshold']}")
    print(f"    corrected max            {row['corrected_max']:.6f}")
    print(f"    corrected > 0.5          {row['n_corrected_gt_0.5']}")
    print(f"    corrected ECE            {row['ECE_corrected']:.4f}")
    print(f"    corrected F2 @ 0.5       {row['F2_corrected@0.5']:.4f}")
    print(f"    corrected AP             {row['AP_corrected']:.4f}"
          f"   (rho vs raw {row['rho_vs_raw']:.6f})")

pd.DataFrame(diag1).to_csv(os.path.join(save_dir, 'Recal_Diag_ProbDist.csv'), index=False)
pd.DataFrame(diag2).to_csv(os.path.join(save_dir, 'Recal_Diag_TailReliability.csv'), index=False)
pd.DataFrame(diag3).to_csv(os.path.join(save_dir, 'Recal_Diag_PriorCorrection.csv'), index=False)

print("\n  HOW TO READ THIS")
print("  If RUS raw scores exceed 1/(1+beta) and the corrected probability")
print("  therefore exceeds 0.5, the Platt F2 of 0.000 reflects the sigmoid")
print("  family, not the model. Report the collapse in F2 as robust in")
print("  DIRECTION across both calibration families, and do not present")
print("  0.000 as the quantity of interest.")
print("  If instead no raw score reaches 1/(1+beta), the posterior genuinely")
print("  never crosses 0.5 and the 0.000 can be reported as a property of the")
print("  model at this prevalence.")

"""## 5. STATISTICAL COMPARISON

Repeated CV gives 15 paired estimates instead of 5, so the Wilcoxon test is  
no longer bounded away from significance by construction (U1). McNemar is run  
on the out-of-fold predictions of the first repeat. Holm-Bonferroni controls  
the family-wise error rate. Note that repeated-CV folds overlap, so p-values  
are optimistic; effect sizes and consistent direction are reported alongside.
"""

print("\n" + "="*72)
print("  5. STATISTICAL COMPARISON")
print("="*72)

import itertools
def holm(ps):
    m = len(ps); adj = [min(1.0, p*(m-i)) for i, p in enumerate(ps)]
    for i in range(1, m):
        adj[i] = max(adj[i], adj[i-1])
    return adj

from scipy.stats import t as _tdist

def corrected_resampled_t(diff, n_train, n_test):
    """Nadeau and Bengio (2003) corrected resampled t-test.

    Repeated cross-validation reuses the same data across folds, so the fold
    differences are positively correlated and the naive paired t-test is
    anti-conservative. The correction inflates the variance by
    (1/k + n_test/n_train). See also Bouckaert and Frank (2004).
    """
    d = np.asarray(diff, dtype=float)
    k = len(d)
    if k < 2 or np.allclose(d, 0):
        return np.nan, 1.0
    var = d.var(ddof=1)
    if var <= 0:
        return np.nan, 1.0
    stat = d.mean() / np.sqrt((1.0/k + n_test/max(n_train, 1)) * var)
    p = 2 * (1 - _tdist.cdf(abs(stat), df=k-1))
    return float(stat), float(p)

_n_test  = int(np.mean([len(te) for _, te in SPLITS]))
_n_train = int(np.mean([len(tr) for tr, _ in SPLITS]))

wil = []
for a, b in itertools.combinations(STRATEGIES, 2):
    sa, sb = np.array(fold_metrics[a]['F2']), np.array(fold_metrics[b]['F2'])
    if np.allclose(sa, sb):
        stat, p = np.nan, 1.0
    else:
        try:
            stat, p = wilcoxon(sa, sb)
        except ValueError:
            stat, p = np.nan, 1.0
    t_stat, t_p = corrected_resampled_t(sa - sb, _n_train, _n_test)
    d = (np.mean(sa)-np.mean(sb)) / (np.sqrt((np.var(sa)+np.var(sb))/2) + 1e-12)
    wil.append({'A': a, 'B': b, 'mean_F2_A': round(float(np.mean(sa)), 4),
                'mean_F2_B': round(float(np.mean(sb)), 4),
                "Cohen_d": round(float(d), 3),
                't_corrected': None if np.isnan(t_stat) else round(t_stat, 3),
                'p_corrected_t': round(t_p, 5),
                'p_raw': p,
                'winner': a if np.mean(sa) > np.mean(sb) else b})
wil = pd.DataFrame(wil).sort_values('p_raw').reset_index(drop=True)
wil['p_holm'] = holm(wil['p_raw'].tolist())
wil['significant'] = wil['p_holm'] < 0.05
print(wil.to_string(index=False))
wil.to_csv(os.path.join(save_dir, 'Stat_Wilcoxon_F2.csv'), index=False)

mcn = []
valid = oof_seen == 1
for a, b in itertools.combinations(STRATEGIES, 2):
    ca = (oof_pred[a][valid] == y.values[valid])
    cb = (oof_pred[b][valid] == y.values[valid])
    b_ = int(np.sum(~ca & cb)); c_ = int(np.sum(ca & ~cb))
    tb = np.array([[int(np.sum(ca & cb)), c_], [b_, int(np.sum(~ca & ~cb))]])
    r = mcnemar(tb, exact=False, correction=True)
    mcn.append({'A': a, 'B': b, 'b': b_, 'c': c_, 'b+c': b_+c_,
                'stat': round(float(r.statistic), 3), 'p_raw': float(r.pvalue),
                'winner': a if c_ > b_ else (b if b_ > c_ else 'tie')})
mcn = pd.DataFrame(mcn).sort_values('p_raw').reset_index(drop=True)
mcn['p_holm'] = holm(mcn['p_raw'].tolist())
mcn['significant'] = mcn['p_holm'] < 0.05
print("\n" + mcn.to_string(index=False))
mcn.to_csv(os.path.join(save_dir, 'Stat_McNemar.csv'), index=False)

"""## 6. NOISE-CONTROL AUDIT  (R1-5)

"""

audit = pd.DataFrame(noise_audit)
hits = int(audit['Noise_1'].sum() + audit['Noise_2'].sum())
print("\n" + "="*72)
print("  6. NOISE CONTROL")
print("="*72)
print(f"  noise columns selected {hits} / {2*len(audit)} opportunities "
      f"({'PASS' if hits == 0 else 'INSPECT CSV'})")
print("  SelectKBest here reduces dimensionality and verifies that")
print("  uninformative synthetic features are excluded. It is not claimed to")
print("  be a source of predictive gain.")
audit.to_csv(os.path.join(save_dir, 'Noise_Audit.csv'), index=False)

"""## 7. TARGET-THRESHOLD SENSITIVITY  (R1-2)

"""

print("\n" + "="*72)
print("  7. TARGET-THRESHOLD SENSITIVITY  (m_p cutoff)")
print("="*72)

CUTOFFS = [0.5, 1.0, 1.5] if QUICK_MODE else [0.5, 0.8, 1.0, 1.2, 1.5]
sens = []
# Grouped splits here too: the cutoff changes the labels, not the polymorph
# structure, so composition grouping must still be enforced.
def grouped_single(yd):
    return list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                     random_state=RNG).split(X, yd, groups=groups))
for cut in CUTOFFS:
    yc = (df['m_p'] < cut).astype(int)
    if yc.sum() < 30:
        print(f"  m_p < {cut}: only {int(yc.sum())} positives, skipped")
        continue
    print(f"\n  m_p < {cut}: {int(yc.sum())} positives ({100*yc.mean():.1f}%)")
    for s in STRATEGIES:
        # -- checkpoint: one file per (cutoff, strategy) ---------------------
        _tag = f'sens_cut{cut}_{s}'
        _ck = ckpt_load(_tag)
        if _ck is not None:
            sens.append(_ck)
            print(f"    {s}: restored from checkpoint")
            continue
        f2v, apv = [], []
        for tr, te in grouped_single(yc):
            ytr2, yte2 = yc.iloc[tr], yc.iloc[te]
            pipe = build(s)
            if s == 'ClassWeight':
                pipe.set_params(xgb__scale_pos_weight=
                                (ytr2 == 0).sum()/max((ytr2 == 1).sum(), 1))
            sr = RandomizedSearchCV(pipe, PARAM_DIST, n_iter=max(5, N_ITER//2),
                                    cv=3, scoring=f2_scorer, random_state=RNG,
                                    n_jobs=N_JOBS)
            try:
                sr.fit(X.iloc[tr], ytr2)
                pr = sr.best_estimator_.predict_proba(X.iloc[te])[:, 1]
                f2v.append(fbeta_score(yte2, (pr >= 0.5).astype(int), beta=2,
                                       zero_division=0))
                apv.append(average_precision_score(yte2, pr))
            except Exception as e:
                print(f"    {s} fold failed: {e}")
        if f2v:
            _row = {'Cutoff': cut, 'Positives': int(yc.sum()),
                    'Strategy': s,
                    'F2': f"{np.mean(f2v):.3f} +/- {np.std(f2v):.3f}",
                    'AvgPrec': f"{np.mean(apv):.3f} +/- {np.std(apv):.3f}",
                    'F2_mean': float(np.mean(f2v)),
                    'AP_mean': float(np.mean(apv))}
            sens.append(_row)
            ckpt_save(_tag, _row)
sens = pd.DataFrame(sens)
if len(sens):
    print("\n" + sens.drop(columns=['F2_mean', 'AP_mean']).to_string(index=False))
    sens.to_csv(os.path.join(save_dir, 'Threshold_Sensitivity.csv'), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(W2COL, W2COL * 0.36))
    for j, (col, lab) in enumerate([('F2_mean', r'$F_2$ score'),
                                    ('AP_mean', 'Average precision')]):
        ax = axes[j]
        for i, s in enumerate(STRATEGIES):
            sub = sens[sens['Strategy'] == s]
            if len(sub):
                ax.plot(sub['Cutoff'], sub[col], marker=MARKERS[i % len(MARKERS)],
                        color=CB[i % len(CB)], label=s)
        ax.set_xlabel(r'Positive-class cutoff on $m^{*}_{p}$ ($m_{e}$)')
        ax.set_ylabel(lab)
        ax.yaxis.grid(True, ls=':', alpha=0.5)
        ax.set_axisbelow(True)
        panel_label(ax, 'ab'[j])
    axes[0].legend(ncol=2, loc='best')
    fig.tight_layout()
    save_fig(fig, 'Fig_Threshold_Sensitivity')

"""## 8. INTERPRETABILITY  (R1-6)

Three views, deliberately reported together because none is sufficient alone:  
MDI          model-internal, biased toward high-cardinality splits  
permutation  computed on HELD-OUT data with the F2 scorer  
SHAP         local attributions, averaged  
Magpie descriptors are strongly correlated, so all three distribute credit  
among correlated inputs. None of them establishes an independent physical  
contribution of a single descriptor.
"""

print("\n" + "="*72)
print("  8. INTERPRETABILITY (MDI + permutation + SHAP)")
print("  Caveat: correlated descriptors share credit; no independent physical")
print("  attribution is implied.")
print("="*72)

ref_strategy = best_f2
tr_idx, te_idx = SPLITS[0]   # grouped split, no shared composition
ref = build(ref_strategy)
if ref_strategy == 'ClassWeight':
    ref.set_params(xgb__scale_pos_weight=
                   (y.iloc[tr_idx] == 0).sum()/max((y.iloc[tr_idx] == 1).sum(), 1))
ref.set_params(xgb__n_estimators=200, xgb__max_depth=5, xgb__learning_rate=0.1)
ref.fit(X.iloc[tr_idx], y.iloc[tr_idx])

sel_mask  = ref.named_steps['fs'].get_support()
sel_names = np.array(ALL_FEATURES)[sel_mask]
mdi = pd.Series(ref.named_steps['xgb'].feature_importances_, index=sel_names)

# -- checkpoint: permutation importance is the slow step in this section ----
_ck = ckpt_load(f'perm_importance_{ref_strategy}')
if _ck is not None:
    print("  permutation importance restored from checkpoint")
    perm_mean = _ck
else:
    perm = permutation_importance(ref, X.iloc[te_idx], y.iloc[te_idx],
                                  scoring=f2_scorer, n_repeats=5,
                                  random_state=RNG, n_jobs=N_JOBS)
    perm_mean = perm.importances_mean
    ckpt_save(f'perm_importance_{ref_strategy}', perm_mean)
perm_s = pd.Series(perm_mean, index=ALL_FEATURES)

imp = pd.DataFrame({'permutation_heldout': perm_s,
                    'MDI': mdi.reindex(perm_s.index)}).fillna(0)
imp = imp.sort_values('permutation_heldout', ascending=False)
imp.to_csv(os.path.join(save_dir, 'Feature_Importances.csv'))
print(imp.head(15).round(4).to_string())

try:
    import shap
    import matplotlib.pyplot as plt

    Xt = ref.named_steps['scaler'].transform(
         ref.named_steps['fs'].transform(
         ref.named_steps['imputer'].transform(X.iloc[te_idx])))
    sample = Xt[:min(1000, len(Xt))]
    expl = shap.TreeExplainer(ref.named_steps['xgb'])
    sv = expl.shap_values(sample)
    shap_imp = pd.Series(np.abs(sv).mean(axis=0), index=sel_names)
    shap_imp.sort_values(ascending=False).to_csv(
        os.path.join(save_dir, 'SHAP_Importances.csv'))
    print("\n  top SHAP features:",
          ', '.join(shap_imp.sort_values(ascending=False).head(5).index))

    fig = plt.figure(figsize=(W1COL * 1.6, W1COL * 1.5))
    shap.summary_plot(sv, sample, feature_names=list(sel_names),
                      show=False, max_display=15, plot_size=None)

    # --- UPDATED FIX ---
    # 1. Remove fig.tight_layout() because it fights with SHAP's colorbar
    # 2. Set aggressive manual margins
    fig.subplots_adjust(bottom=0.35, left=0.35)

    # 3. Shorten the label so it stops taking up excess vertical space
    plt.gca().set_xlabel("SHAP value (impact on model output)")
    # -------------------

    save_fig(fig, 'Fig_SHAP_Summary')
except Exception as e:
    print(f"  [SHAP skipped: {e}]")
    print("  If this is a scipy/numpy binary conflict "
          "('does not export expected C function'), run this ONCE in a fresh cell:")
    print("      !pip install -q -U scipy shap")
    print("  then Runtime -> Restart session, then Run all. The Magpie cache and")
    print("  all other outputs are preserved, so the rerun is not slower.")

"""## 9. RETROSPECTIVE SCREENING UTILITY  (R2-4, R1-7)

Every compound is ranked by a model that never saw its label (out-of-fold  
probability). Enrichment is the quantity a practitioner cares about: how much  
richer in true positives is the top of the ranked list than random selection.  
This is a retrospective estimate on a fully labeled database and is reported  
as such; no prospective discovery is claimed.
"""

print("\n" + "="*72)
print("  9. RETROSPECTIVE SCREENING UTILITY (out-of-fold ranking)")
print("="*72)

prob_rank = oof_prob[best_ap]
order = np.argsort(-prob_rank)
yv = y.values
pk = []
for k in [25, 50, 100, 250, 500, 1000]:
    if k > len(order):
        continue
    top = order[:k]
    p = float(yv[top].mean())
    pk.append({'k': k, 'precision@k': round(p, 3),
               'enrichment': round(p/base_rate, 2),
               'recall@k': round(float(yv[top].sum()/yv.sum()), 3)})
pk = pd.DataFrame(pk)
print(f"  ranking model: {best_ap}   base rate: {base_rate:.3f}")
print(pk.to_string(index=False))
pk.to_csv(os.path.join(save_dir, 'Screening_PrecisionAtK.csv'), index=False)

# ---- precision@k for every strategy (Figure 7a, Section 4.6) ---------------
# The single-model table above backs the values quoted in the text. Section 4.6
# also claims that no correction is at least as precise as all four resamplers
# at every list length, which needs all six curves.
_rows_pk = []
for _s in STRATEGIES:
    _o = np.argsort(-oof_prob[_s])
    for _k in [25, 50, 100, 250, 500, 1000]:
        if _k > len(_o):
            continue
        _top = _o[:_k]
        _p = float(yv[_top].mean())
        _rows_pk.append({'strategy': _s, 'k': _k,
                         'precision@k': round(_p, 3),
                         'enrichment': round(_p / base_rate, 2),
                         'recall@k': round(float(yv[_top].sum() / yv.sum()), 3)})
pk_all = pd.DataFrame(_rows_pk)
print("\n  precision@k by strategy")
print(pk_all.pivot(index='k', columns='strategy',
                   values='precision@k')[list(STRATEGIES)].to_string())
pk_all.to_csv(os.path.join(save_dir,
              'Screening_PrecisionAtK_AllStrategies.csv'), index=False)

# ---- out-of-fold probabilities (Section 4.11, Figure 10) -------------------
# run_extended_analysis.py reads these and refits nothing. CSV so a reader can
# open it without numpy; the .npy is kept for backward compatibility.
pd.DataFrame({_s: oof_prob[_s] for _s in STRATEGIES}).to_csv(
    os.path.join(save_dir, 'OOF_Probabilities.csv'), index=False)
np.save(os.path.join(save_dir, 'oof_prob_all.npy'),
        np.vstack([oof_prob[_s] for _s in STRATEGIES]))
print("  out-of-fold probabilities -> OOF_Probabilities.csv, oof_prob_all.npy")

cum = np.cumsum(yv[order])
ks = np.arange(1, len(order)+1)
fig, axes = plt.subplots(1, 2, figsize=(W2COL, W2COL * 0.36))

# --- AXES 0 ---
ax = axes[0]

handles_bottom, labels_bottom = [], []
handles_upper, labels_upper = [], []
bottom_targets = ['None', 'ClassWeight', 'B-SMOTE', 'SMOTE']

# Below k = 20 the enrichment curve is one compound wide and swings between 0
# and 1/base_rate. The caption of Figure 7 states the k = 20 floor, so the code
# has to impose it or the published panel and this one are different figures.
KMIN = 20
_mk = ks >= KMIN
for i, s in enumerate(STRATEGIES):
    o_s = np.argsort(-oof_prob[s])
    _enr = np.cumsum(yv[o_s]) / ks / base_rate
    line, = ax.plot(ks[_mk], _enr[_mk], color=CB[i % len(CB)], label=s)

    if s in bottom_targets:
        handles_bottom.append(line)
        labels_bottom.append(s)
    else:
        handles_upper.append(line)
        labels_upper.append(s)

line_rand = ax.axhline(1.0, color='k', ls=':', lw=1.0, label='Random selection')
handles_upper.append(line_rand)
labels_upper.append('Random selection')

ax.set_xscale('log')
ax.set_xlim(KMIN, len(ks))
ax.set_xlabel('Compounds inspected, $k$ (log scale)')
ax.set_ylabel('Enrichment factor')

leg_bottom = ax.legend(handles_bottom, labels_bottom, loc='lower center', ncol=2, framealpha=0.95)
ax.add_artist(leg_bottom)
ax.legend(handles_upper, labels_upper, loc='upper right', framealpha=0.95)

ax.yaxis.grid(True, ls=':', alpha=0.5); ax.set_axisbelow(True)
ax.text(-0.15, 1.05, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom', ha='right')


# --- AXES 1 ---
ax = axes[1]
ax.plot(ks, cum/yv.sum(), color=CB[0], label=f'{best_ap} (out-of-fold)')
ax.plot(ks, ks*base_rate/yv.sum(), 'k:', lw=1.0, label='Random selection')
ax.set_xlabel('Compounds inspected, $k$')
ax.set_ylabel('Fraction of positives recovered')
ax.set_xlim(0, min(2000, len(ks)))
ax.legend(loc='best', framealpha=0.95)
ax.yaxis.grid(True, ls=':', alpha=0.5); ax.set_axisbelow(True)
ax.text(-0.15, 1.05, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom', ha='right')


fig.tight_layout()
save_fig(fig, 'Fig_Enrichment')

cand = df[['mpid', 'formula', 'm_p', 'm_n', 'S_p', 'PF_p', 'S_n', 'PF_n']].copy()
cand['oof_probability'] = prob_rank
cand['true_positive'] = yv
cand = cand.sort_values('oof_probability', ascending=False).head(100)


# --- Materials Project enrichment (R1-7) -------------------------------------
# Plain REST call. Deliberately avoids mp-api / emmet-core, whose pyarrow
# dependency clashes with the pyarrow already imported by pandas/matminer
# when installed mid-session (ABI error: "IpcReadOptions size changed").

# --- Materials Project enrichment (R1-7) -------------------------------------
# MP moved to alphabetical canonical material_ids (DB release 2026-06-08).
# Numeric ids from the 2017 BoltzTraP dataset still resolve, but the REST
# response echoes the CANONICAL id, not the alias queried. Merging on the
# returned id therefore matches nothing. We query one id at a time and keep
# the requested id as the join key.
for _c in ['mp_canonical_id', 'mp_formula', 'band_gap_eV', 'direct_gap', 'e_above_hull']:
    cand[_c] = np.nan

if os.environ.get('MP_API_KEY'):
    import requests, time
    _s = requests.Session()
    _s.headers.update({'X-API-KEY': os.environ['MP_API_KEY']})
    _rows, _fail = [], []
    for _id in cand['mpid'].astype(str):
        try:
            _r = _s.get('https://api.materialsproject.org/materials/summary/',
                        params={'material_ids': _id,
                                '_fields': 'material_id,formula_pretty,band_gap,'
                                           'is_gap_direct,energy_above_hull',
                                '_limit': 1},
                        timeout=60)
            _r.raise_for_status()
            _d = _r.json().get('data', [])
            if not _d:
                _fail.append(_id); continue
            _d = _d[0]
            _rows.append({'mpid': _id,
                          'mp_canonical_id': _d.get('material_id'),
                          'mp_formula': _d.get('formula_pretty'),
                          'band_gap_eV': _d.get('band_gap'),
                          'direct_gap': _d.get('is_gap_direct'),
                          'e_above_hull': _d.get('energy_above_hull')})
        except Exception as _e:
            _fail.append(f"{_id} ({type(_e).__name__})")
        time.sleep(0.05)

    if _rows:
        meta = pd.DataFrame(_rows)
        cand = cand.drop(columns=['mp_canonical_id', 'mp_formula', 'band_gap_eV',
                                  'direct_gap', 'e_above_hull']).merge(
                                      meta, on='mpid', how='left')

    _m = int(cand['band_gap_eV'].notna().sum())
    print(f"  MP enrichment: {_m}/{len(cand)} matched ({_m/len(cand):.1%})")

    # VERIFICATION: the returned canonical id must describe the same compound.
    def _red(f):
        try:
            return Composition(str(f)).reduced_formula
        except Exception:
            return None
    _pairs = [(a, b) for a, b in zip(cand['formula'], cand['mp_formula'])
              if pd.notna(b)]
    _agree = sum(_red(a) == _red(b) for a, b in _pairs)
    print(f"  formula cross-check: {_agree}/{len(_pairs)} agree with BoltzTraP")
    if _pairs and _agree < len(_pairs):
        print("  [WARNING: id mapping suspect, do NOT report these columns]")
        for a, b in _pairs:
            if _red(a) != _red(b):
                print(f"    mismatch: dataset {a}  vs  MP {b}")
    if _fail:
        print(f"  unresolved ids ({len(_fail)}): {_fail[:10]}")
else:
    print("  [set MP_API_KEY to add band gap, gap character, stability]")
cand.to_csv(os.path.join(save_dir, 'Top100_Candidates.csv'), index=False)
print(f"  top-100 hit rate: {cand['true_positive'].mean():.3f} "
      f"(base rate {base_rate:.3f})")
print("  saved: Top100_Candidates.csv")

"""## 9b. PUBLICATION FIGURES

Figure 1  strategy comparison (F2 fixed vs tuned threshold, average precision)  
Figure 2  precision-recall curves and reliability (calibration) diagrams  
Figure 3  PCA projection, labelled explicitly as a visualization diagnostic  
On Figure 3 (Referee 1, comment 6): a two-component projection of a ~132  
dimensional descriptor space captures a small share of the total variance.  
It shows where synthetic samples are placed relative to real ones. It is NOT  
evidence of physical separation between the classes, and the caption says so.
"""

print("\n" + "="*72)
print("  9b. PUBLICATION FIGURES (600 dpi, vector PDF + TIFF + PNG)")
print("="*72)

# ---- Figure 1: strategy comparison ------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(W2COL, W2COL * 0.30))
xs = np.arange(len(STRATEGIES))
for j, (key, lab) in enumerate([('F2', r'$F_2$ (threshold = 0.5)'),
                                ('F2_tuned', r'$F_2$ (tuned threshold)'),
                                ('AvgPrecision', 'Average precision')]):
    ax = axes[j]
    mu = [np.mean(fold_metrics[s][key]) for s in STRATEGIES]
    sd = [np.std(fold_metrics[s][key]) for s in STRATEGIES]
    ax.bar(xs, mu, yerr=sd, capsize=2.5, width=0.68,
           color=[CB[i % len(CB)] for i in range(len(STRATEGIES))],
           edgecolor='black', linewidth=0.5,
           error_kw={'elinewidth': 0.7, 'capthick': 0.7})
    if key == 'AvgPrecision':
        ax.axhline(base_rate, color='k', ls=':', lw=1.0)
        ax.text(len(STRATEGIES)-0.5, base_rate, ' no-skill', va='bottom',
                ha='right', fontsize=FS_BASE-1)
    ax.set_xticks(xs)
    ax.set_xticklabels(STRATEGIES, rotation=35, ha='right')
    ax.set_ylabel(lab)
    ax.yaxis.grid(True, ls=':', alpha=0.5); ax.set_axisbelow(True)
    panel_label(ax, 'abc'[j])
fig.tight_layout()
save_fig(fig, 'Fig1_Strategy_Comparison')

# ---- Figure 2: PR curves + reliability diagram ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(W2COL * 0.78, W2COL * 0.36))

ax = axes[0]
for i, s in enumerate(STRATEGIES):
    pr_, rc_, _ = precision_recall_curve(yv, oof_prob[s])
    ax.plot(rc_, pr_, color=CB[i % len(CB)],
            label=f'{s} (AP={average_precision_score(yv, oof_prob[s]):.2f})')
ax.axhline(base_rate, color='k', ls=':', lw=1.0, label=f'No skill ({base_rate:.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(loc='upper right')
panel_label(ax, 'a')

# Reliability diagram (U4): resampling is known to distort calibration, so a
# strategy that ranks well may still produce over-confident probabilities.
ax = axes[1]
bins = np.linspace(0, 1, 11)
for i, s in enumerate(STRATEGIES):
    p = oof_prob[s]
    idx = np.digitize(p, bins[1:-1])
    xs_, ys_ = [], []
    for b in range(10):
        m = idx == b
        if m.sum() >= 20:
            xs_.append(p[m].mean()); ys_.append(yv[m].mean())
    ax.plot(xs_, ys_, marker=MARKERS[i % len(MARKERS)], color=CB[i % len(CB)],
            label=f"{s} (ECE={np.mean(fold_metrics[s]['ECE']):.3f})")
ax.plot([0, 1], [0, 1], 'k:', lw=1.0, label='Perfect calibration')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Observed positive fraction')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc='upper left')
panel_label(ax, 'b')

fig.tight_layout()
save_fig(fig, 'Fig2_PR_and_Calibration')

# ---- Figure 3: PCA visualization diagnostic (R1-6) ---------------------------
try:
    _imp = SimpleImputer(strategy='median')
    _sc  = StandardScaler()
    Xs_  = _sc.fit_transform(_imp.fit_transform(X[magpie_cols]))
    _pca = PCA(n_components=2, random_state=RNG)
    X2   = _pca.fit_transform(Xs_)
    ve   = _pca.explained_variance_ratio_ * 100
    Xr_, yr_ = BorderlineSMOTE(random_state=RNG,
                               kind='borderline-1').fit_resample(Xs_, y)
    Xsyn = _pca.transform(Xr_[len(Xs_):])

    fig, ax = plt.subplots(figsize=(W1COL * 1.55, W1COL * 1.15))
    ax.scatter(X2[yv == 0, 0], X2[yv == 0, 1], s=3, c='#BBBBBB',
               linewidths=0, label=f'Majority (n={int((yv==0).sum()):,})',
               rasterized=True)
    ax.scatter(X2[yv == 1, 0], X2[yv == 1, 1], s=7, c=CB[0],
               linewidths=0, label=f'Minority (n={int(yv.sum()):,})',
               rasterized=True)
    ax.scatter(Xsyn[:, 0], Xsyn[:, 1], s=5, c=CB[1], marker='x',
               linewidths=0.4, alpha=0.6,
               label=f'Synthetic (n={len(Xsyn):,})', rasterized=True)
    ax.set_xlabel(f'PC1 ({ve[0]:.1f}% of variance)')
    ax.set_ylabel(f'PC2 ({ve[1]:.1f}% of variance)')
    ax.legend(loc='best', markerscale=2.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    save_fig(fig, 'Fig3_PCA_Diagnostic')
    print(f"  PCA captures only {ve.sum():.1f}% of variance in "
          f"{len(magpie_cols)} dimensions.")
    print("  CAPTION MUST STATE: visualization diagnostic only; this projection")
    print("  is not evidence of physical separation between classes (R1-6).")
except Exception as e:
    print(f"  [PCA figure skipped: {e}]")

"""## 10. GENERALIZATION  (R2-4)

(i)  Leave-one-chemical-family-out. Random K-fold mostly measures  
interpolation; testing on a chemistry absent from training is the  
stricter question (Meredig et al. 2018; Li et al. 2024 caution that many  
"out-of-distribution" splits are still interpolation, so this is a  
stronger but not a definitive extrapolation test).  
(ii) External hold-out on Ricci et al. (2017) compounds absent from the  
training file. Same methodology, unseen compounds: compound-level  
generalization, not cross-methodology transfer.
"""

print("\n" + "="*72)
print("  10. GENERALIZATION")
print("="*72)

def fixed_model(strategy, ytr):
    p = build(strategy)
    if strategy == 'ClassWeight':
        p.set_params(xgb__scale_pos_weight=(ytr == 0).sum()/max((ytr == 1).sum(), 1))
    p.set_params(xgb__n_estimators=200, xgb__max_depth=5,
                 xgb__learning_rate=0.1, xgb__subsample=0.8)
    return p

def family(formula):
    try:
        els = {e.symbol for e in Composition(formula).elements}
    except Exception:
        return 'other'
    for name, anions in [('halide', {'F', 'Cl', 'Br', 'I'}),
                         ('oxide', {'O'}),
                         ('chalcogenide', {'S', 'Se', 'Te'}),
                         ('pnictide', {'N', 'P', 'As', 'Sb', 'Bi'})]:
        if els & anions:
            return name
    return 'other'

fams = df['formula'].apply(family)
print("  families:", fams.value_counts().to_dict())
loco = []
for fam in sorted(fams.unique()):
    m = (fams == fam).values
    ytr, yte = y[~m], y[m]
    if yte.sum() < 10 or ytr.sum() < 50:
        print(f"  [{fam}] skipped (too few positives)")
        continue
    for s in [best_ap, 'ClassWeight']:
        # -- checkpoint: one file per (held-out family, strategy) ------------
        _tag = f'loco_{fam}_{s}'
        _ck = ckpt_load(_tag)
        if _ck is not None:
            loco.append(_ck)
            print(f"  [{fam}/{s}] restored from checkpoint")
            continue
        mod = fixed_model(s, ytr)
        mod.fit(X[~m], ytr)
        pr = mod.predict_proba(X[m])[:, 1]
        k = min(100, int(m.sum()))
        br = float(yte.mean())
        _row = {'held_out_family': fam, 'strategy': s,
                'n_test': int(m.sum()), 'base_rate': round(br, 3),
                'F2': round(fbeta_score(yte, (pr >= .5).astype(int),
                                        beta=2, zero_division=0), 3),
                'AvgPrec': round(average_precision_score(yte, pr), 3),
                f'enrichment@{k}': round(
                    float(yte.values[np.argsort(-pr)[:k]].mean())/max(br, 1e-9), 2)}
        loco.append(_row)
        ckpt_save(_tag, _row)
loco = pd.DataFrame(loco)
if len(loco):
    print(loco.to_string(index=False))
    loco.to_csv(os.path.join(save_dir, 'LOCO_Generalization.csv'), index=False)

# ---- external Ricci hold-out -------------------------------------------------
# No manual CSV needed: the tabular Ricci database is hosted on Figshare
# (record 14701110) and matminer downloads and caches it automatically.
# The external set is the full Ricci tabular database (47,737 compounds),
# roughly five times the size of the training database and covering chemistries
# absent from it. Column names there use a different convention from
# 'boltztrap_mp', so they are auto-detected and printed; override below if the
# detection misses.
# Detected from the printed column list of ricci_boltztrap_mp_tabular.
# That dataset has no Materials Project id column, so overlap is removed by
# normalised composition alone, which is the stricter test for a
# composition-only model (it also removes polymorphs of training compounds).
RICCI_ID_COL      = None
RICCI_FORMULA_COL = 'pretty_formula'
RICCI_MP_COL      = 'mₑᶜ.p.ε̄ [mₑ]'   # mean p-type conductivity effective mass
FIGSHARE_RICCI = 14701110

def load_ricci():
    try:
        from matminer.datasets import load_dataset
        d = load_dataset('ricci_boltztrap_mp_tabular')
        print(f"  Ricci via matminer/Figshare: {len(d):,} rows")
        return d
    except Exception as e:
        print(f"  matminer loader failed ({e}); trying Figshare API ...")
    import urllib.request
    with urllib.request.urlopen(
            f'https://api.figshare.com/v2/articles/{FIGSHARE_RICCI}',
            timeout=120) as r:
        meta = json.load(r)
    f = max(meta['files'], key=lambda x: x.get('size', 0))
    local = os.path.join(save_dir, f['name'])
    if not os.path.exists(local):
        urllib.request.urlretrieve(f['download_url'], local)
    if local.endswith(('.csv', '.csv.gz')):
        return pd.read_csv(local)
    if local.endswith(('.json', '.json.gz')):
        return pd.read_json(local)
    return pd.read_pickle(local)

print("\n  external validation (Ricci et al. 2017) ...")
try:
    if not HAS_MPID:
        raise RuntimeError("no real mpid column; cannot guarantee the external "
                           "set excludes training compounds")
    ricci = load_ricci()
    cols = [str(c) for c in ricci.columns]
    print(f"  columns: {cols}")

    def pick(keys, manual):
        if manual:
            return manual
        for c in cols:
            if any(k in c.lower() for k in keys):
                return c
        return None

    idc = pick(['mp_id', 'mpid', 'material_id'], RICCI_ID_COL)
    fmc = pick(['formula'], RICCI_FORMULA_COL)
    mpc = RICCI_MP_COL
    if mpc is None:
        for c in cols:
            lc = c.lower()
            if lc.startswith('m') and ('.p' in lc or '_p' in lc or 'p' == lc[-1]):
                mpc = c
                break
    print(f"  detected -> id={idc} formula={fmc} m_p={mpc}")
    if mpc not in ricci.columns:
        mpc = next((c for c in cols if c.startswith('m') and '.p.' in c), None)
    if not all([fmc, mpc]):
        raise RuntimeError("set RICCI_FORMULA_COL / RICCI_MP_COL from the "
                           "printed column list and rerun")
    if idc is None:
        ricci = ricci.copy(); ricci['_noid'] = ''
        idc = '_noid'
        print("  no id column in this dataset; overlap removed by composition only")

    ext = ricci[[idc, fmc, mpc]].copy()
    ext.columns = ['mpid', 'formula', 'm_p_ext']
    ext['m_p_ext'] = pd.to_numeric(ext['m_p_ext'], errors='coerce')
    ext = ext.dropna(subset=['m_p_ext', 'formula'])

    # Zero-overlap external validation. Removing shared mp-ids is not enough:
    # a different mp-id can be a polymorph of a training composition, and a
    # composition-only model sees an identical feature vector. We therefore
    # exclude by normalised composition as well, following the zero-overlap
    # protocol of Vera de la Garza et al. (2026).
    n0 = len(ext)
    ext = ext[~ext['mpid'].astype(str).isin(df['mpid'].astype(str))]
    n1 = len(ext)
    ext['comp_group'] = ext['formula'].apply(norm_formula)
    ext = ext[~ext['comp_group'].isin(set(df['comp_group']))]
    n2 = len(ext)
    ext = ext.drop_duplicates(subset='comp_group', keep='first').reset_index(drop=True)
    print(f"  overlap removal: {n0:,} -> {n1:,} (id match) -> {n2:,} "
          f"(composition match) -> {len(ext):,} (one entry per composition)")
    if QUICK_MODE:
        ext = ext.sample(min(2000, len(ext)), random_state=RNG).reset_index(drop=True)
    print(f"  unseen external compounds: {len(ext):,}")

    # -- checkpoint: featurising ~10k external compositions is slow ---------
    _ck = ckpt_load('external_featurised')
    if _ck is not None:
        ext, fe = _ck['ext'], _ck['fe']
        print(f"  external Magpie features restored from checkpoint "
              f"({len(ext):,} compositions)")
    else:
        comps_e, keep_e = [], []
        for i, f in enumerate(ext['formula']):
            try:
                comps_e.append(Composition(f)); keep_e.append(i)
            except Exception:
                pass
        ext = ext.iloc[keep_e].reset_index(drop=True)
        fe = ElementProperty.from_preset('magpie').featurize_dataframe(
            pd.DataFrame({'composition': comps_e}), col_id='composition',
            ignore_errors=True).drop(columns=['composition'])
        fe = fe[magpie_cols].reset_index(drop=True)
        ckpt_save('external_featurised', {'ext': ext, 'fe': fe})

    r2 = np.random.default_rng(RNG+1)
    fe['Noise_1'] = r2.normal(0, 1, len(fe))
    fe['Noise_2'] = r2.normal(0, 1, len(fe))
    Xe = fe[ALL_FEATURES]
    ye = (ext['m_p_ext'] < TARGET_THRESHOLD).astype(int)
    be = float(ye.mean())
    print(f"  external base rate: {be:.3f}")

    ext_rows = []
    for s in [best_ap, 'ClassWeight', 'None']:
        if s not in STRATEGIES:
            continue
        # -- checkpoint: one file per external-holdout strategy --------------
        _tag = f'external_{s}'
        _ck = ckpt_load(_tag)
        if _ck is not None:
            ext_rows.append(_ck)
            print(f"  external [{s}] restored from checkpoint")
            continue
        mod = fixed_model(s, y)
        mod.fit(X, y)
        pr = mod.predict_proba(Xe)[:, 1]
        row = {'strategy': s, 'n_external': len(ye), 'base_rate': round(be, 4),
               'F2': round(fbeta_score(ye, (pr >= .5).astype(int), beta=2,
                                       zero_division=0), 3),
               'AvgPrec': round(average_precision_score(ye, pr), 3),
               'ROC-AUC': round(roc_auc_score(ye, pr), 3)}
        oe = np.argsort(-pr)
        for k in [100, 500, 1000]:
            if k <= len(ye):
                p = float(ye.values[oe[:k]].mean())
                row[f'precision@{k}'] = round(p, 3)
                row[f'enrichment@{k}'] = round(p/max(be, 1e-9), 2)
        ext_rows.append(row)
    extdf = pd.DataFrame(ext_rows)
    print(extdf.to_string(index=False))
    extdf.to_csv(os.path.join(save_dir, 'External_Ricci_Validation.csv'),
                 index=False)
except Exception as e:
    print(f"  [external validation skipped: {e}]")

"""## 11. IS THE CROSS-BAND SIGNAL REDUCIBLE TO COMPOSITION?  (R2-2)

Referee 2 argued the cross-band association is an expected materials-level
co-occurrence: broadly dispersive band structures give low effective masses
for both carriers. That is a testable claim. Under one identical protocol we
compare three input sets:
(a) composition only   (b) n-type transport only   (c) both

Interpretation, stated in advance:
- if (c) is not better than (a), the n-type descriptors add nothing beyond
  composition and the co-occurrence explanation is sufficient;
- if (c) beats (a) materially, the n-type descriptors carry band-structure
  information that composition does not encode.
Either outcome is reportable. No mechanism is proposed.
"""

print("\n" + "="*72)
print("  11. CROSS-BAND SIGNAL: BEYOND COMPOSITION OR NOT?  (Referee 2, pt 2)")
print("="*72)

for f in NTYPE:
    rho, p = spearmanr(df[f], df['m_p'])
    print(f"  Spearman rho({f}, m_p) = {rho:+.3f}  (p = {p:.2e})")

Xn    = df[NTYPE].reset_index(drop=True)
Xboth = pd.concat([X[magpie_cols], Xn], axis=1)
sets  = {'(a) composition only': X[magpie_cols],
         '(b) n-type only':      Xn,
         '(c) composition + n-type': Xboth}

cb = []
for name, Xs in sets.items():
    # -- checkpoint: one file per input set -----------------------------------
    _tag = 'crossband_' + name.split(')')[0].strip('( ')
    _ck = ckpt_load(_tag)
    if _ck is not None:
        cb.append(_ck)
        print(f"  {name}: restored from checkpoint")
        continue
    f2v, apv = [], []
    for tr, te in grouped_single(y):
        ytr2 = y.iloc[tr]
        p = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('xgb',     make_xgb(n_estimators=200, max_depth=5,
                                 learning_rate=0.1, subsample=0.8,
                                 scale_pos_weight=(ytr2 == 0).sum() /
                                                  max((ytr2 == 1).sum(), 1)))])
        p.fit(Xs.iloc[tr], ytr2)
        pr = p.predict_proba(Xs.iloc[te])[:, 1]
        f2v.append(fbeta_score(y.iloc[te], (pr >= .5).astype(int), beta=2,
                               zero_division=0))
        apv.append(average_precision_score(y.iloc[te], pr))
    _row = {'input set': name,
            'F2': f"{np.mean(f2v):.3f} +/- {np.std(f2v):.3f}",
            'AvgPrec': f"{np.mean(apv):.3f} +/- {np.std(apv):.3f}",
            'F2_mean': float(np.mean(f2v)), 'AP_mean': float(np.mean(apv))}
    cb.append(_row)
    ckpt_save(_tag, _row)

cb = pd.DataFrame(cb)
print("\n" + cb.drop(columns=['F2_mean', 'AP_mean']).to_string(index=False))
cb.to_csv(os.path.join(save_dir, 'CrossBand_Incremental_Test.csv'), index=False)

d_ap = cb.loc[2, 'AP_mean'] - cb.loc[0, 'AP_mean']
print(f"\n  incremental average precision from adding n-type descriptors: {d_ap:+.3f}")
if d_ap < 0.02:
    print("  READING: n-type descriptors add little beyond composition. This")
    print("  supports the materials-level co-occurrence explanation given by")
    print("  Referee 2 and argues against treating the cross-band association")
    print("  as an independent finding.")
else:
    print("  READING: n-type descriptors carry information composition does not")
    print("  encode. Report the magnitude; do not attach an untested mechanism.")

"""## 12. SUMMARY
"""

print("\n" + "="*72)
print("  OUTPUTS")
print("="*72)
for f in sorted(os.listdir(save_dir)):
    print("   ", f)
print(f"\nDirectory: {save_dir}")
print("""
Reporting guidance:
  - Lead with average precision and enrichment, not accuracy.
  - If no resampler beats the class-weighted or threshold-tuned baseline, say
    so plainly; that is a useful result consistent with prior high-dimensional
    findings, and burying it would be the only real error.
  - Report Brier/ECE when recommending a model for probability-ranked screening.
  - Describe LOCO and the external hold-out as compound-level generalization,
    not cross-methodology transfer.
""")
