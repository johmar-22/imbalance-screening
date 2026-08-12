#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_extended_analysis.py
========================

Sections 4.10 and 4.11 of the manuscript, plus the threshold-free significance
tests behind Section 4.3, plus the contrast test behind Section 4.10.

    13A       Leakage ablation: grouped vs ungrouped splits at frozen
              hyperparameters, separated into shared-composition and
              unique-composition held-out rows.       -> Table 8, Fig. 9
    13A-bis   Contrast test over the six strategies.  -> Section 4.10 text
    13B       Decision-curve analysis and the scale/ranking decomposition.
                                                      -> Fig. 10
    13C       Wilcoxon / corrected resampled t on average precision and on
              expected calibration error.             -> Tables S5, S6

NO GOOGLE DRIVE, NO COLAB, NO MANUAL DOWNLOAD.
The primary dataset is fetched from Figshare through matminer on first run and
cached under --outdir. Nothing here reads a Drive path.

WHAT IT NEEDS FROM run_analysis.py
----------------------------------
Run `scripts/run_analysis.py --outdir DIR` first. This script then reads three
files from the same DIR:

    Best_Hyperparameters.csv    the per-fold search results; 13A freezes the
                                modal configuration per strategy
    OOF_Probabilities.csv       out-of-fold probabilities, one column per
                                strategy; 13B refits nothing
    Fold_Metrics_PerFold.csv    the 15 per-fold metric values; 13C refits
                                nothing

The last two are written by run_analysis.py only after the patches described in
docs/UPLOAD_PLAN_v32.md are applied. If OOF_Probabilities.csv is absent the
script falls back to oof_prob_all.npy, and if Fold_Metrics_PerFold.csv is absent
it falls back to the pickles in <outdir>/_checkpoints/.

USAGE
-----
    python scripts/run_extended_analysis.py --outdir results_reproduced

    # ~15 minutes; the ablation is reduced to one repeat
    python scripts/run_extended_analysis.py --outdir results_reproduced --quick

    # run one part only
    python scripts/run_extended_analysis.py --outdir DIR --only dca

RUNTIME
-------
13A is the only part that fits models: 6 strategies x 2 split families x 15
splits = 180 XGBoost fits at frozen hyperparameters, roughly 25-40 minutes on a
GPU and 2-3 hours on CPU. It is checkpointed per (arm, strategy). 13A-bis, 13B
and 13C together take under a minute.

OUTPUTS
-------
    Leakage_Rate.csv                 leaked-row fraction per split family
    Composition_Structure.csv        the counts quoted in Section 3.3
    Straddling_Compositions.csv      the 58 compositions straddling the cutoff
    Leakage_Ablation.csv             per-arm, per-strategy summary
    Leakage_Ablation_PerFold.csv     fold-level values behind it
    Leakage_Inflation.csv            inflation, CI, Hedges' g, Mann-Whitney p
    Leakage_Contrast_Test.csv        the six-strategy contrast test
    DecisionCurve_NetBenefit.csv     net-benefit grid
    DecisionCurve_Ceiling.csv        attainable net benefit (oracle bound)
    DecisionCurve_Decomposition.csv  realized vs attainable at five p_t
    DecisionCurve_Peaks.csv          per-strategy summary
    Stat_Wilcoxon_AP.csv             Table S5
    Stat_Wilcoxon_ECE.csv            Table S6
    Fig_Leakage_Ablation.{pdf,png,tiff}
    Fig_Decision_Curve.{pdf,png,tiff}

Figure 1 of the manuscript, the leakage mechanism illustration, is drawn by
scripts/make_fig_leakage.py. It is kept separate because it renders crystal
structures rather than analysis output.

MIT licence, same as the rest of scripts/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd

# =============================================================================
#  0. CONFIGURATION
# =============================================================================

STRATEGIES = ['None', 'ClassWeight', 'B-SMOTE', 'SMOTE', 'ADASYN', 'RUS']
TARGET_THRESHOLD = 1.0        # m*p < 1.0 m_e defines the positive class
K_BEST = 30                   # SelectKBest, same as the main analysis
PRECISION_AT = 100            # head of the ranked list used in 13A

# Frozen classifier for the ablation is NOT set here: 13A reads the modal
# configuration per strategy from Best_Hyperparameters.csv, which is what the
# manuscript states in Section 4.10.

# Decision-curve grid. Must span the tuned thresholds of Table 1, which reach
# 0.670 for undersampling.
PT_GRID = np.linspace(0.005, 0.95, 380)
OPERATING_POINTS = (0.05, 0.10, 0.15, 0.20, 0.30)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Extended analyses: leakage ablation, decision curves, '
                    'threshold-free significance tests.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--outdir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'results'),
        help='directory holding run_analysis.py output; new files are written '
             'here too')
    p.add_argument('--seed', type=int, default=42, help='random seed')
    p.add_argument('--quick', action='store_true',
                   help='smoke test: 1 CV repeat instead of 3. The numbers it '
                        'prints are NOT the numbers in the paper.')
    p.add_argument('--no-resume', action='store_true',
                   help='ignore existing checkpoints and recompute')
    p.add_argument('--data-source', choices=['figshare', 'csv'],
                   default='figshare', help='dataset origin')
    p.add_argument('--datadir', default=None,
                   help='where boltztrap_mp.csv is looked for when '
                        '--data-source csv; defaults to the script directory')
    p.add_argument('--only', choices=['ablation', 'contrast', 'dca', 'tests'],
                   default=None, help='run one part only')
    p.add_argument('--n-boot', type=int, default=10000,
                   help='bootstrap draws for the inflation interval')
    return p.parse_args(argv)


# =============================================================================
#  1. CHECKPOINTS
#
#  Same scheme as run_analysis.py: a pickle per unit of work under
#  <outdir>/_checkpoints/, tagged with a fingerprint of the configuration that
#  produced it, so a checkpoint from a different configuration is ignored
#  instead of silently reused.
# =============================================================================

class Ckpt:
    def __init__(self, outdir, fingerprint, enabled=True):
        self.dir = os.path.join(outdir, '_checkpoints')
        self.fp = fingerprint
        self.enabled = enabled
        os.makedirs(self.dir, exist_ok=True)

    def path(self, tag):
        return os.path.join(self.dir, f'{tag}.pkl')

    def save(self, tag, obj):
        try:
            with open(self.path(tag), 'wb') as fh:
                pickle.dump({'fingerprint': self.fp, 'obj': obj}, fh)
        except Exception as exc:                      # never fail the run
            print(f'  [checkpoint {tag} not written: {exc}]')

    def load(self, tag):
        if not self.enabled:
            return None
        path = self.path(tag)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as fh:
                blob = pickle.load(fh)
        except Exception:
            return None
        if blob.get('fingerprint') != self.fp:
            return None
        return blob['obj']


def fingerprint(args, n_rows):
    payload = json.dumps({
        'seed': args.seed, 'quick': bool(args.quick),
        'threshold': TARGET_THRESHOLD, 'k_best': K_BEST,
        'strategies': STRATEGIES, 'n_rows': int(n_rows),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# =============================================================================
#  2. FIGURE STANDARD
#
#  Nature-family: 89 mm single column, 183 mm double, Arial, 8 pt, 600 dpi,
#  TrueType fonts embedded so the PDF survives a publisher's preflight.
#  Colour-blind-safe Okabe-Ito palette.
# =============================================================================

MM = 1 / 25.4
W1COL, W2COL = 89 * MM, 183 * MM
FS_BASE = 8
CB = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
      '#E69F00', '#56B4E9', '#000000', '#999999']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']


def setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': FS_BASE,
        'axes.labelsize': FS_BASE,
        'axes.titlesize': FS_BASE,
        'xtick.labelsize': FS_BASE - 1,
        'ytick.labelsize': FS_BASE - 1,
        'legend.fontsize': FS_BASE - 1,
        'axes.linewidth': 0.7,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    return plt


def save_fig(fig, outdir, name, formats=('pdf', 'tiff', 'png')):
    written = []
    for ext in formats:
        path = os.path.join(outdir, f'{name}.{ext}')
        try:
            if ext == 'tiff':
                fig.savefig(path, dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
            else:
                fig.savefig(path, dpi=600)
            written.append(os.path.basename(path))
        except Exception as exc:
            print(f'  [could not write {ext}: {exc}]')
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f'  saved: {", ".join(written)}')


def panel_label(ax, letter, dx=-0.16, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=FS_BASE + 2, fontweight='bold', va='bottom', ha='right')


# =============================================================================
#  3. DATA
#
#  Identical to Section 1 of run_analysis.py. Duplicated rather than imported
#  because run_analysis.py executes its whole analysis at module level, so
#  importing it would launch a two-hour run.
# =============================================================================

SIX = ['m_n', 'PF_n', 'S_n', 'm_p', 'PF_p', 'S_p']


def _standardise(d):
    ren = {}
    for c in d.columns:
        lc = str(c).strip().lower()
        if lc == 'pf_n':
            ren[c] = 'PF_n'
        elif lc == 'pf_p':
            ren[c] = 'PF_p'
        elif lc == 's_n':
            ren[c] = 'S_n'
        elif lc == 's_p':
            ren[c] = 'S_p'
        elif lc == 'm_n':
            ren[c] = 'm_n'
        elif lc == 'm_p':
            ren[c] = 'm_p'
        elif lc in ('mpid', 'mp_id', 'material_id'):
            ren[c] = 'mpid'
        elif lc in ('formula', 'pretty_formula', 'full_formula'):
            ren[c] = 'formula'
    return d.rename(columns=ren)


def load_frame(args):
    if args.data_source == 'figshare':
        try:
            from matminer.datasets import load_dataset
            d = _standardise(load_dataset('boltztrap_mp'))
            missing = [c for c in SIX + ['formula'] if c not in d.columns]
            if missing:
                raise RuntimeError(f'columns not found after mapping: {missing}')
            print(f"  source: matminer/Figshare 'boltztrap_mp' ({len(d):,} rows)")
            return d
        except Exception as exc:
            print(f'  Figshare load failed ({exc}); falling back to local CSV')
    datadir = args.datadir or os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(datadir, 'boltztrap_mp.csv')
    d = _standardise(pd.read_csv(path))
    print(f'  source: local CSV {path} ({len(d):,} rows)')
    return d


def build_dataset(args, with_features=True):
    """Return (df, X, y, groups).

    with_features=False returns X as None and skips the Magpie step entirely.
    Sections 13B and 13C need only the labels, and featurizing 8,924
    compositions takes 20 to 40 minutes when no cache is present.
    """
    from pymatgen.core import Composition

    print('\n' + '=' * 72)
    print('  DATA')
    print('=' * 72)
    df = load_frame(args)
    n_raw = len(df)
    if 'mpid' in df.columns:
        df = df.drop_duplicates(subset='mpid', keep='first').reset_index(drop=True)
    else:
        df['mpid'] = [f'row-{i}' for i in range(len(df))]
    df = df.dropna(subset=SIX + ['formula']).reset_index(drop=True)
    print(f'  raw {n_raw:,} -> retained {len(df):,}')

    df['target'] = (df['m_p'] < TARGET_THRESHOLD).astype(int)
    y = df['target']
    print(f"  positives (m_p < {TARGET_THRESHOLD}): {int(y.sum())} "
          f'({100 * y.mean():.2f}%)')

    def norm_formula(f):
        try:
            return Composition(f).reduced_formula
        except Exception:
            return str(f)

    df['comp_group'] = df['formula'].apply(norm_formula)
    groups = df['comp_group'].values

    if not with_features:
        print('  descriptors not needed for this run; Magpie step skipped')
        return df, None, y, groups

    from matminer.featurizers.composition import ElementProperty

    # ---- Magpie features, cached under --outdir exactly as run_analysis.py
    cache = os.path.join(args.outdir, 'magpie_cache.csv')
    if os.path.exists(cache):
        feat = pd.read_csv(cache)
        print(f'  loaded Magpie cache: {cache}')
        if len(feat) != len(df):
            raise RuntimeError(
                f'{cache} has {len(feat)} rows but the dataset has {len(df)}. '
                'Delete the cache and rerun, or point --outdir at the folder '
                'run_analysis.py used.')
    else:
        comps, keep = [], []
        for i, f in enumerate(df['formula']):
            try:
                comps.append(Composition(f))
                keep.append(i)
            except Exception:
                pass
        if len(keep) < len(df):
            print(f'  dropped {len(df) - len(keep)} unparsable formula(s)')
            df = df.iloc[keep].reset_index(drop=True)
            y = df['target']
            groups = df['comp_group'].values
        ep = ElementProperty.from_preset('magpie')
        print(f'  featurizing {len(comps):,} compositions (20-40 min) ...')
        feat = ep.featurize_dataframe(
            pd.DataFrame({'composition': comps}), col_id='composition',
            ignore_errors=True).drop(columns=['composition'])
        feat.to_csv(cache, index=False)
        print(f'  cached -> {cache}')

    feat = feat.reset_index(drop=True)
    magpie_cols = list(feat.columns)

    # Two Gaussian negative controls, same seed convention as run_analysis.py
    rng = np.random.default_rng(args.seed)
    feat['Noise_1'] = rng.normal(0, 1, len(feat))
    feat['Noise_2'] = rng.normal(0, 1, len(feat))
    all_features = magpie_cols + ['Noise_1', 'Noise_2']
    X = feat[all_features]
    print(f'  X shape: {X.shape}  ({len(magpie_cols)} Magpie + 2 noise)')
    return df, X, y, groups


# =============================================================================
#  4. PIPELINE, identical to run_analysis.py
# =============================================================================

def make_pipeline_factory(seed, device):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import BorderlineSMOTE, SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from xgboost import XGBClassifier

    def make_xgb(**kw):
        return XGBClassifier(eval_metric='logloss', tree_method='hist',
                             device=device, random_state=seed, verbosity=0, **kw)

    def build(strategy):
        steps = [('imputer', SimpleImputer(strategy='median')),
                 ('fs', SelectKBest(score_func=f_classif, k=K_BEST)),
                 ('scaler', StandardScaler())]
        if strategy == 'B-SMOTE':
            steps.append(('res', BorderlineSMOTE(random_state=seed,
                                                 kind='borderline-1')))
        elif strategy == 'SMOTE':
            steps.append(('res', SMOTE(random_state=seed)))
        elif strategy == 'ADASYN':
            steps.append(('res', ADASYN(random_state=seed)))
        elif strategy == 'RUS':
            steps.append(('res', RandomUnderSampler(random_state=seed)))
        steps.append(('xgb', make_xgb()))
        return ImbPipeline(steps)

    return build


def detect_device():
    import subprocess
    try:
        ok = subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0
    except Exception:
        ok = False
    print(f"  XGBoost device: {'cuda' if ok else 'cpu'}"
          + ('' if ok else '  [no GPU found; CPU fallback, results identical]'))
    return ('cuda', 1) if ok else ('cpu', -1)


def ece(y_true, prob, bins=10):
    """Expected calibration error over equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(prob, edges[1:-1])
    tot = 0.0
    y_true = np.asarray(y_true)
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += (m.sum() / len(prob)) * abs(y_true[m].mean() - prob[m].mean())
    return float(tot)


# =============================================================================
#  5. STATISTICS
#
#  Order-safe Holm, so the caller does not have to sort first. The version in
#  run_analysis.py assumes a sorted input; this one does not.
# =============================================================================

def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, p[idx] * (m - rank))
        run = max(run, val)
        adj[idx] = run
    return adj


def corrected_resampled_t(diff, n_train, n_test):
    """Nadeau and Bengio (2003). Variance inflated by (1/k + n_test/n_train)."""
    from scipy.stats import t as tdist
    d = np.asarray(diff, float)
    k = len(d)
    if k < 2 or np.allclose(d, 0):
        return np.nan, 1.0
    var = d.var(ddof=1)
    if var <= 0:
        return np.nan, 1.0
    stat = d.mean() / np.sqrt((1.0 / k + n_test / max(n_train, 1)) * var)
    p = 2 * (1 - tdist.cdf(abs(stat), df=k - 1))
    return float(stat), float(p)


def cohens_d_pooled(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((a.mean() - b.mean())
                 / (np.sqrt((a.var() + b.var()) / 2) + 1e-12))


def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    if sp <= 0:
        return np.nan
    return float((b.mean() - a.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)))


def boot_ci(a, b, rng, n=10000, alpha=0.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.empty(n)
    for i in range(n):
        d[i] = (rng.choice(b, len(b), replace=True).mean()
                - rng.choice(a, len(a), replace=True).mean())
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


# =============================================================================
#  6. SECTION 13A.  THE LEAKAGE ABLATION            -> Table 8, Fig. 9
#
#  Both arms must differ ONLY in how the split is made, so the inner randomized
#  search is switched off and one configuration per strategy is frozen: the
#  modal configuration selected across the 15 nested folds of Section 4.1.
#  This is what Section 4.10 of the manuscript states.
# =============================================================================

def load_frozen_params(outdir):
    """Modal hyperparameter configuration per strategy, from Section 4.

    NOTE keep_default_na=False. Without it pandas reads the strategy called
    'None' as a missing value and that strategy silently disappears.
    """
    path = os.path.join(outdir, 'Best_Hyperparameters.csv')
    if not os.path.exists(path):
        raise SystemExit(
            f'{path} not found.\n'
            'Run scripts/run_analysis.py --outdir <same dir> first; 13A freezes '
            'the modal configuration that search selected.')
    hp = pd.read_csv(path, keep_default_na=False)
    frozen = {}
    for s in STRATEGIES:
        sub = hp[hp['Strategy'] == s].drop(columns=['Fold', 'Strategy'],
                                           errors='ignore')
        if not len(sub):
            raise SystemExit(f'no hyperparameters logged for strategy {s!r}')
        cfg = sub.mode().iloc[0].to_dict()
        frozen[s] = {
            'xgb__subsample': float(cfg['xgb__subsample']),
            'xgb__n_estimators': int(cfg['xgb__n_estimators']),
            'xgb__min_child_weight': int(cfg['xgb__min_child_weight']),
            'xgb__max_depth': int(cfg['xgb__max_depth']),
            'xgb__learning_rate': float(cfg['xgb__learning_rate']),
        }
        print(f'  frozen config {s:<12s} {frozen[s]}')
    return frozen


def make_split_families(X, y, groups, n_splits, n_repeats, seed):
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    out = {'grouped': [], 'random': []}
    for rep in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed + rep)
        out['grouped'].extend(list(sgkf.split(X, y, groups=groups)))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed + rep)
        out['random'].extend(list(skf.split(X, y)))
    return out


def export_composition_structure(df, groups, outdir):
    """The counts quoted in Section 3.3, exported so they are checkable."""
    vc = pd.Series(groups).value_counts()
    dup = vc[vc > 1]
    agg = (pd.DataFrame({'g': groups, 'm_p': df['m_p'].values})
           .groupby('g')['m_p'].agg(['min', 'max', 'count']))
    straddle = agg[(agg['min'] < TARGET_THRESHOLD)
                   & (agg['max'] >= TARGET_THRESHOLD)]
    stats = pd.DataFrame([{
        'Rows': len(df),
        'Unique compositions': int(vc.size),
        'Compositions with >1 polymorph': int(dup.size),
        'Rows in multi-polymorph compositions': int(dup.sum()),
        'Share of rows': float(dup.sum() / len(df)),
        'Compositions straddling the cutoff': int(len(straddle)),
        'Cutoff (m_e)': TARGET_THRESHOLD,
    }])
    print('\n  Composition structure quoted in Section 3.3:')
    print(stats.T.to_string(header=False))
    stats.to_csv(os.path.join(outdir, 'Composition_Structure.csv'), index=False)
    (straddle.reset_index()
     .rename(columns={'g': 'composition', 'min': 'm_p_min',
                      'max': 'm_p_max', 'count': 'n_entries'})
     .to_csv(os.path.join(outdir, 'Straddling_Compositions.csv'), index=False))
    return stats


def leakage_rate(splits, groups, outdir):
    g = np.asarray(groups)
    rows = []
    for kind, folds in splits.items():
        for i, (tr, te) in enumerate(folds):
            seen = set(g[tr])
            leaked = np.fromiter((x in seen for x in g[te]), bool, len(te))
            rows.append({'Split': kind, 'Fold': i + 1, 'Test n': len(te),
                         'Leaked rows': int(leaked.sum()),
                         'Leak rate': float(leaked.mean())})
    leak = pd.DataFrame(rows)
    print('\n  Fraction of held-out rows whose composition is also in training:')
    print(leak.groupby('Split')['Leak rate'].agg(['mean', 'std']).round(4)
          .to_string())
    assert leak.loc[leak.Split == 'grouped', 'Leaked rows'].sum() == 0, \
        'grouped splits are not leakage-free; check the grouping key'
    leak.to_csv(os.path.join(outdir, 'Leakage_Rate.csv'), index=False)
    return leak


def run_ablation(args, X, y, groups, build, frozen, splits, ckpt):
    from sklearn.metrics import (average_precision_score, fbeta_score,
                                 precision_score, recall_score, roc_auc_score)
    from scipy.stats import mannwhitneyu

    g = np.asarray(groups)
    multi = pd.Series(g).duplicated(keep=False).values   # shared-composition row

    def precision_at_k(y_true, prob, k):
        idx = np.argsort(-prob)[:k]
        return float(np.mean(np.asarray(y_true)[idx]))

    metrics = ['F2', 'AvgPrecision', 'ROC-AUC', f'P@{PRECISION_AT}',
               'AP_shared_comp', 'AP_unique_comp']
    summary_rows, fold_rows = [], []
    t0 = time.time()

    for kind, folds in splits.items():
        for s in STRATEGIES:
            tag = f'abl_{kind}_{s}'
            cached = ckpt.load(tag)
            if cached is not None:
                summary_rows.append(cached['summary'])
                fold_rows.extend(cached['folds'])
                print(f'  {kind:<8s} {s:<12s} [restored from checkpoint]')
                continue

            per_fold = []
            for i, (tr, te) in enumerate(folds):
                ytr, yte = y.iloc[tr], y.iloc[te]
                pipe = build(s)
                pipe.set_params(**frozen[s])
                if s == 'ClassWeight':
                    pipe.set_params(xgb__scale_pos_weight=(
                        (ytr == 0).sum() / max((ytr == 1).sum(), 1)))
                pipe.fit(X.iloc[tr], ytr)
                prob = pipe.predict_proba(X.iloc[te])[:, 1]
                pred = (prob >= 0.5).astype(int)

                m_te = multi[te]
                yv = np.asarray(yte)
                rec = {
                    'Split': kind, 'Strategy': s, 'Fold': i + 1,
                    'F2': fbeta_score(yte, pred, beta=2, zero_division=0),
                    'AvgPrecision': average_precision_score(yte, prob),
                    'ROC-AUC': roc_auc_score(yte, prob),
                    f'P@{PRECISION_AT}': precision_at_k(yte, prob, PRECISION_AT),
                    'Precision': precision_score(yte, pred, zero_division=0),
                    'Recall': recall_score(yte, pred, zero_division=0),
                    'AP_shared_comp': (
                        average_precision_score(yv[m_te], prob[m_te])
                        if yv[m_te].sum() > 0 else np.nan),
                    'AP_unique_comp': (
                        average_precision_score(yv[~m_te], prob[~m_te])
                        if yv[~m_te].sum() > 0 else np.nan),
                }
                per_fold.append(rec)

            d = pd.DataFrame(per_fold)
            summary = {'Split': kind, 'Strategy': s}
            for c in metrics:
                summary[c] = f'{d[c].mean():.3f} +/- {d[c].std(ddof=0):.3f}'
                summary[c + '_mean'] = float(d[c].mean())
            summary_rows.append(summary)
            fold_rows.extend(per_fold)
            ckpt.save(tag, {'summary': summary, 'folds': per_fold})
            print(f"  {kind:<8s} {s:<12s} AP={summary['AvgPrecision']}  "
                  f"F2={summary['F2']}  [{time.time() - t0:.0f}s]")

    abl = pd.DataFrame(summary_rows)
    abl_fold = pd.DataFrame(fold_rows)

    # ---- inflation ---------------------------------------------------------
    # The two arms partition the data differently, so fold i of one arm has no
    # counterpart in the other and a PAIRED test is not justified. The
    # comparison is unpaired: Mann-Whitney U, Hedges' g, and a percentile
    # bootstrap on the difference of means. Fold values within an arm remain
    # correlated because the repeats reuse the data, so the p-values are
    # approximate and err conservative. Section 3.7 of the manuscript says this.
    rng = np.random.default_rng(args.seed)
    infl_rows = []
    for s in STRATEGIES:
        row = {'Strategy': s}
        for metric in metrics:
            a = abl_fold.query('Split=="grouped" and Strategy==@s')[metric] \
                        .dropna().values
            b = abl_fold.query('Split=="random"  and Strategy==@s')[metric] \
                        .dropna().values
            if len(a) < 2 or len(b) < 2:
                continue
            lo, hi = boot_ci(a, b, rng, n=args.n_boot)
            row[f'{metric} grouped'] = a.mean()
            row[f'{metric} random'] = b.mean()
            row[f'{metric} inflation'] = b.mean() - a.mean()
            row[f'{metric} inflation %'] = 100 * (b.mean() - a.mean()) / a.mean()
            row[f'{metric} CI low'] = lo
            row[f'{metric} CI high'] = hi
            row[f'{metric} g'] = hedges_g(a, b)
            try:
                row[f'{metric} p'] = mannwhitneyu(
                    a, b, alternative='two-sided').pvalue
            except ValueError:
                row[f'{metric} p'] = np.nan
        infl_rows.append(row)
    infl = pd.DataFrame(infl_rows)
    infl['AvgPrecision p_holm'] = holm(infl['AvgPrecision p'].values)

    print('\n  Inflation from dropping the grouping (random minus grouped),')
    print('  unpaired, 15 fold values per arm, 95% bootstrap CI:')
    show = ['Strategy', 'AvgPrecision grouped', 'AvgPrecision random',
            'AvgPrecision inflation', 'AvgPrecision CI low',
            'AvgPrecision CI high', 'AvgPrecision g', 'AvgPrecision p_holm']
    print(infl[show].round(4).to_string(index=False))

    print('\n  Where the inflation lands (average precision):')
    print(infl[['Strategy', 'AP_shared_comp inflation',
                'AP_unique_comp inflation']].round(4).to_string(index=False))

    out = args.outdir
    abl.drop(columns=[c for c in abl.columns if c.endswith('_mean')]) \
       .to_csv(os.path.join(out, 'Leakage_Ablation.csv'), index=False)
    abl_fold.to_csv(os.path.join(out, 'Leakage_Ablation_PerFold.csv'),
                    index=False)
    infl.to_csv(os.path.join(out, 'Leakage_Inflation.csv'), index=False)
    print('\n  -> Leakage_Ablation.csv, Leakage_Ablation_PerFold.csv, '
          'Leakage_Inflation.csv')
    return abl_fold, infl


def figure_ablation(abl_fold, outdir):
    """Fig. 9. Three panels on one metric: all rows, shared rows, unique rows."""
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(W2COL, W2COL * 0.30), sharey=True)
    xpos = np.arange(len(STRATEGIES))
    wid = 0.38
    panels = [('AvgPrecision', 'All held-out rows'),
              ('AP_shared_comp',
               'Rows sharing a composition\n(leakage can act here)'),
              ('AP_unique_comp',
               'Rows with a unique composition\n(control: it cannot)')]
    for j, (metric, title) in enumerate(panels):
        ax = axes[j]
        gm = [abl_fold.query('Split=="grouped" and Strategy==@s')[metric].mean()
              for s in STRATEGIES]
        gs = [abl_fold.query('Split=="grouped" and Strategy==@s')[metric]
              .std(ddof=0) for s in STRATEGIES]
        rm = [abl_fold.query('Split=="random" and Strategy==@s')[metric].mean()
              for s in STRATEGIES]
        rs = [abl_fold.query('Split=="random" and Strategy==@s')[metric]
              .std(ddof=0) for s in STRATEGIES]
        ax.bar(xpos - wid / 2, gm, wid, yerr=gs, capsize=2, color=CB[0],
               label='Composition-grouped', error_kw={'lw': 0.7})
        ax.bar(xpos + wid / 2, rm, wid, yerr=rs, capsize=2, color=CB[1],
               label='Ungrouped (random)', error_kw={'lw': 0.7})
        ax.set_xticks(xpos)
        ax.set_xticklabels(STRATEGIES, rotation=30, ha='right')
        ax.set_title(title, fontsize=FS_BASE - 1)
        ax.yaxis.grid(True, ls=':', alpha=0.5)
        ax.set_axisbelow(True)
        panel_label(ax, 'abc'[j], dx=-0.10)
    axes[0].set_ylabel('Average precision')
    axes[0].legend(loc='upper right', framealpha=0.95)
    fig.tight_layout()
    save_fig(fig, outdir, 'Fig_Leakage_Ablation')


# =============================================================================
#  7. SECTION 13A-bis.  THE CONTRAST TEST            -> Section 4.10 text
# =============================================================================

def run_contrast_test(outdir, infl=None):
    from scipy.stats import binomtest, wilcoxon

    print('\n' + '=' * 72)
    print('  13A-bis. CONTRAST TEST: SHARED vs UNIQUE COMPOSITION ROWS')
    print('=' * 72)
    if infl is None:
        path = os.path.join(outdir, 'Leakage_Inflation.csv')
        if not os.path.exists(path):
            raise SystemExit(f'{path} not found; run --only ablation first.')
        infl = pd.read_csv(path, keep_default_na=False)
        print(f'  read Leakage_Inflation.csv from {outdir}')

    t = infl.set_index('Strategy').reindex(STRATEGIES)
    shared = t['AP_shared_comp inflation'].astype(float).values
    unique = t['AP_unique_comp inflation'].astype(float).values
    contrast = shared - unique
    shared_grouped = t['AP_shared_comp grouped'].astype(float).values
    rel = shared.mean() / shared_grouped.mean()

    print('\n  Per-strategy inflation (random split minus grouped split):')
    print(pd.DataFrame({
        'shared (grouped)': shared_grouped,
        'shared inflation': shared,
        'unique inflation': unique,
        'contrast': contrast,
        'shared p (Mann-Whitney)': t['AP_shared_comp p'].astype(float).values,
        'unique p (Mann-Whitney)': t['AP_unique_comp p'].astype(float).values,
    }, index=STRATEGIES).round(4).to_string())

    print(f'\n  mean inflation, shared rows : {shared.mean():+.4f}'
          f'   ({100 * rel:.1f}% of the grouped-arm level)')
    print(f'  mean inflation, unique rows : {unique.mean():+.4f}')
    print(f'  mean contrast               : {contrast.mean():+.4f}')
    print(f'  contrasts positive          : {int((contrast > 0).sum())} '
          f'of {len(contrast)}')

    # Six paired values, one per strategy, both tests two-sided. With n = 6 the
    # smallest attainable p is 2 / 2**6 = 0.03125, so an all-positive contrast
    # lands exactly on that floor. Report the floor rather than let the reader
    # mistake it for a measurement of effect size.
    n = len(contrast)
    n_pos = int((contrast > 0).sum())
    sign_p = binomtest(n_pos, n, 0.5, alternative='two-sided').pvalue
    w_stat, w_p = wilcoxon(contrast, alternative='two-sided')
    floor = 2 / 2 ** n

    print('\n  Six paired values, one per strategy, both tests two-sided:')
    print(f'    sign test            p = {sign_p:.4f}  ({n_pos}/{n} positive)')
    print(f'    Wilcoxon signed-rank p = {w_p:.4f}  (W = {w_stat:g})')
    print(f'    floor for n = {n}        p = {floor:.5f}   '
          '<- smallest value either test can return')
    print('\n  The test establishes consistency of sign across strategies, not')
    print('  the magnitude of any one of them; the magnitude is in the')
    print('  per-strategy Mann-Whitney column above.')

    out = pd.DataFrame([{
        'mean_inflation_shared': shared.mean(),
        'mean_inflation_unique': unique.mean(),
        'mean_contrast': contrast.mean(),
        'relative_rise_shared': rel,
        'n_contrast_positive': n_pos,
        'n_strategies': n,
        'sign_test_p': sign_p,
        'wilcoxon_p': float(w_p),
        'min_attainable_p': floor,
    }])
    out.to_csv(os.path.join(outdir, 'Leakage_Contrast_Test.csv'), index=False)
    print('\n  -> Leakage_Contrast_Test.csv')
    return out


# =============================================================================
#  8. SECTION 13B.  DECISION-CURVE ANALYSIS                     -> Fig. 10
#
#  Net benefit (Vickers and Elkin) at threshold probability p_t:
#
#      NB(p_t) = TP/N - (FP/N) * p_t/(1 - p_t)
#
#  The weight p_t/(1-p_t) is the exchange rate the analyst accepts: at
#  p_t = 0.05 one missed candidate is worth 19 wasted calculations. Two
#  reference strategies bound the problem, calculate everything and calculate
#  nothing:
#
#      NB_all(p_t) = pi - (1 - pi) * p_t/(1 - p_t),   NB_none = 0
#
#  Nothing is refitted. The out-of-fold probabilities are read from disk.
# =============================================================================

def load_oof(outdir, n_rows):
    """Out-of-fold probabilities, one column per strategy."""
    csv = os.path.join(outdir, 'OOF_Probabilities.csv')
    if os.path.exists(csv):
        d = pd.read_csv(csv)
        missing = [s for s in STRATEGIES if s not in d.columns]
        if missing:
            raise SystemExit(f'{csv} is missing columns {missing}')
        P = np.vstack([d[s].values for s in STRATEGIES])
        print(f'  read OOF_Probabilities.csv ({P.shape[1]:,} compounds)')
    else:
        npy = os.path.join(outdir, 'oof_prob_all.npy')
        if not os.path.exists(npy):
            raise SystemExit(
                f'neither {csv} nor {npy} found.\n'
                'Apply the run_analysis.py patch in docs/UPLOAD_PLAN_v32.md, '
                'or rerun run_analysis.py with the patched Section 9.')
        P = np.load(npy)
        print(f'  read oof_prob_all.npy ({P.shape[1]:,} compounds)')
    if P.shape != (len(STRATEGIES), n_rows):
        raise SystemExit(f'expected shape {(len(STRATEGIES), n_rows)}, '
                         f'got {P.shape}')
    return P


def net_benefit(y_true, prob, pt):
    w = pt / (1 - pt)
    pred = prob[:, None] >= pt[None, :]
    tp = (pred & (y_true[:, None] == 1)).sum(0)
    fp = (pred & (y_true[:, None] == 0)).sum(0)
    return tp / len(y_true) - (fp / len(y_true)) * w


def attainable_net_benefit(y_true, prob, pt):
    """Best net benefit any cut on this ranking could reach, per exchange rate.

    Sweeping the cut traces one set of confusion matrices, so this quantity is
    invariant to any strictly monotone rescaling of the probabilities and is a
    property of the ranking alone. It is an ORACLE bound: the cut is chosen with
    knowledge of the held-out labels. It is used only as a yardstick common to
    every strategy, never as an achievable operating point.

    IMPLEMENTATION NOTE. An earlier version swept a fixed grid of cut values
    (linspace(0, 1, 501) unioned with the p_t grid). That is only APPROXIMATELY
    rescale-invariant: a strictly monotone map moves the scores relative to a
    fixed grid, so the maximum lands on a different confusion matrix and the
    curve moves by around 2.5e-4. Since the whole point of this quantity is
    that it depends on the ranking alone, the approximation undercuts the
    claim it exists to support.

    This version is exact. It sorts by score, walks the achievable cuts, and
    takes the maximum. Cut positions are restricted to TIE-GROUP BOUNDARIES,
    because a threshold cannot separate two compounds carrying the same score,
    and in this dataset that case is not hypothetical: polymorphs of one
    composition receive identical probabilities by construction. The result is
    exactly invariant under any strictly monotone rescaling, which preserves
    both the order and the ties.
    """
    y_true = np.asarray(y_true)
    n = len(y_true)
    order = np.argsort(-prob, kind='stable')
    ys = y_true[order]
    ps = prob[order]

    # cumulative counts after taking the top k, for k = 0 .. n
    tp = np.concatenate([[0.0], np.cumsum(ys)])
    fp = np.concatenate([[0.0], np.cumsum(1 - ys)])

    # achievable cuts only: k = 0, k = n, and any k where the score changes
    keep = np.zeros(n + 1, dtype=bool)
    keep[0] = keep[n] = True
    if n > 1:
        keep[1:n] = ps[:-1] != ps[1:]

    tp, fp = tp[keep] / n, fp[keep] / n
    w = pt / (1 - pt)
    return (tp[:, None] - fp[:, None] * w[None, :]).max(axis=0)


def run_decision_curves(outdir, y_arr, P):
    print('\n' + '=' * 72)
    print('  13B. DECISION-CURVE ANALYSIS')
    print('=' * 72)
    pt = PT_GRID
    pi = float(y_arr.mean())
    nb_all = pi - (1 - pi) * (pt / (1 - pt))
    nb_none = np.zeros_like(pt)
    ref = np.maximum(nb_all, 0.0)

    dca = {s: net_benefit(y_arr, P[i], pt) for i, s in enumerate(STRATEGIES)}
    ceiling = {s: attainable_net_benefit(y_arr, P[i], pt)
               for i, s in enumerate(STRATEGIES)}

    # The maximum of NB(p_t) is NOT a useful summary: net benefit rises as
    # p_t -> 0, where the false-positive penalty vanishes and calculating
    # everything is optimal by construction. Report net benefit at thresholds a
    # campaign would actually use, and the width of the region where the model
    # beats both references.
    peak = []
    for s in STRATEGIES:
        nb = dca[s]
        above = pt[(nb - ref) > 0]
        row = {'Strategy': s}
        for p0 in OPERATING_POINTS:
            v = float(np.interp(p0, pt, nb))
            r = float(np.interp(p0, pt, ref))
            row[f'NB @ {p0:.2f}'] = v
            row[f'avoided/100 @ {p0:.2f}'] = (v - r) / (p0 / (1 - p0)) * 100
        row['Beats both refs from p_t'] = above.min() if above.size else np.nan
        row['to p_t'] = above.max() if above.size else np.nan
        m = pt >= pi
        trapz = getattr(np, 'trapezoid', np.trapz)
        row['Area over reference (p_t >= base rate)'] = float(
            trapz((nb - ref)[m], pt[m]))
        peak.append(row)
    peak = pd.DataFrame(peak)

    print('\n  Net benefit at thresholds a campaign would actually use:')
    print(peak[['Strategy'] + [f'NB @ {p:.2f}' for p in OPERATING_POINTS]]
          .round(4).to_string(index=False))
    print('\n  Calculations avoided per 100 compounds, over calculating all:')
    print(peak[['Strategy'] + [f'avoided/100 @ {p:.2f}'
                               for p in OPERATING_POINTS]]
          .round(1).to_string(index=False))
    print('\n  Range over which each strategy beats both references:')
    print(peak[['Strategy', 'Beats both refs from p_t', 'to p_t',
                'Area over reference (p_t >= base rate)']]
          .round(4).to_string(index=False))

    # ---- decomposition -----------------------------------------------------
    dec = []
    for s in STRATEGIES:
        row = {'Strategy': s}
        for p0 in OPERATING_POINTS:
            d_ = float(np.interp(p0, pt, dca[s]))
            c_ = float(np.interp(p0, pt, ceiling[s]))
            row[f'default @ {p0:.2f}'] = d_
            row[f'ceiling @ {p0:.2f}'] = c_
            row[f'scale cost @ {p0:.2f}'] = c_ - d_
        dec.append(row)
    dec = pd.DataFrame(dec)

    print("\n  Decomposition. 'default' uses cut = p_t; 'ceiling' is the best")
    print('  any cut on that ranking could reach at the same exchange rate.')
    print(f"\n  {'p_t':>6}{'realized':>10}{'attainable':>12}{'shrinkage':>12}")
    for p0 in OPERATING_POINTS:
        sd = dec[f'default @ {p0:.2f}'].max() - dec[f'default @ {p0:.2f}'].min()
        sc = dec[f'ceiling @ {p0:.2f}'].max() - dec[f'ceiling @ {p0:.2f}'].min()
        sh = 1 - (sc / sd) if sd else np.nan
        print(f'  {p0:>6.2f}{sd:>10.4f}{sc:>12.4f}{sh:>11.0%}')
    print('  A large shrinkage means the strategies differ mainly in where')
    print('  their probability scale sits, not in how well they order compounds.')

    pd.DataFrame({'p_t': pt, 'NB_calculate_all': nb_all,
                  'NB_calculate_none': nb_none,
                  **{f'NB_{s}': dca[s] for s in STRATEGIES}}) \
      .to_csv(os.path.join(outdir, 'DecisionCurve_NetBenefit.csv'), index=False)
    pd.DataFrame({'p_t': pt, **{f'ceiling_{s}': ceiling[s]
                                for s in STRATEGIES}}) \
      .to_csv(os.path.join(outdir, 'DecisionCurve_Ceiling.csv'), index=False)
    dec.to_csv(os.path.join(outdir, 'DecisionCurve_Decomposition.csv'),
               index=False)
    peak.to_csv(os.path.join(outdir, 'DecisionCurve_Peaks.csv'), index=False)
    print('\n  -> DecisionCurve_NetBenefit.csv, DecisionCurve_Ceiling.csv,')
    print('     DecisionCurve_Decomposition.csv, DecisionCurve_Peaks.csv')
    return dca, ceiling, nb_all, nb_none, pi


def figure_decision_curve(outdir, dca, ceiling, nb_all, nb_none, pi):
    """Fig. 10, three panels, one shared legend below so nothing is obscured."""
    from matplotlib.lines import Line2D
    plt = setup_matplotlib()
    pt = PT_GRID
    pt_max = 0.35
    ylim = (-0.04, 0.055)
    m = pt <= pt_max

    fig, axes = plt.subplots(1, 3, figsize=(W2COL, W2COL * 0.34))

    ax = axes[0]
    for i, s in enumerate(STRATEGIES):
        ax.plot(pt[m], dca[s][m], color=CB[i % len(CB)],
                marker=MARKERS[i % len(MARKERS)], markevery=40, ms=3, label=s)
    ax.plot(pt[m], nb_all[m], color='#666666', ls='--', lw=1.0)
    ax.plot(pt[m], nb_none[m], color='#666666', ls=':', lw=1.0)
    ax.axvline(pi, color='#999999', lw=0.7, ls='-.')
    ax.annotate('base rate', xy=(pi, 0), xytext=(pi + 0.012, ylim[1] * 0.72),
                fontsize=FS_BASE - 2, color='#666666')
    ax.set_xlabel('Threshold probability $p_t$')
    ax.set_ylabel('Net benefit')
    ax.set_ylim(*ylim)
    panel_label(ax, 'a')

    # Panel b is drawn only above the base rate: below it calculating everything
    # remains the better reference and the ratio diverges where the two
    # references cross, which is a property of the transformation, not a model.
    ax = axes[1]
    ref = np.maximum(nb_all, 0.0)
    mb = (pt >= pi) & (pt <= pt_max)
    for i, s in enumerate(STRATEGIES):
        avoided = (dca[s] - ref) / (pt / (1 - pt)) * 100
        ax.plot(pt[mb], avoided[mb], color=CB[i % len(CB)],
                marker=MARKERS[i % len(MARKERS)], markevery=40, ms=3, label=s)
    ax.axhline(0, color='#666666', lw=0.7, ls=':')
    ax.set_xlabel('Threshold probability $p_t$')
    ax.set_ylabel('Calculations avoided per 100')
    panel_label(ax, 'b')

    ax = axes[2]
    lo = np.minimum.reduce([ceiling[s] for s in STRATEGIES])
    hi = np.maximum.reduce([ceiling[s] for s in STRATEGIES])
    ax.fill_between(pt[m], lo[m], hi[m], color='#CCCCCC', alpha=0.6, lw=0)
    for i, s in enumerate(STRATEGIES):
        ax.plot(pt[m], dca[s][m], color=CB[i % len(CB)], lw=1.1)
        ax.plot(pt[m], ceiling[s][m], color=CB[i % len(CB)], lw=0.9, ls=':')
    ax.axvline(pi, color='#999999', lw=0.7, ls='-.')
    ax.set_xlabel('Threshold probability $p_t$')
    ax.set_ylabel('Net benefit')
    ax.set_ylim(*ylim)
    panel_label(ax, 'c')

    handles = [Line2D([], [], color=CB[i % len(CB)],
                      marker=MARKERS[i % len(MARKERS)], ms=3, label=s)
               for i, s in enumerate(STRATEGIES)]
    handles += [
        Line2D([], [], color='#666666', ls='--', label='Calculate all'),
        Line2D([], [], color='#666666', ls=':', label='Calculate none'),
        Line2D([], [], color='#000000', lw=1.1, label='realized (cut = $p_t$)'),
        Line2D([], [], color='#000000', lw=0.9, ls=':',
               label='attainable (best cut)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    save_fig(fig, outdir, 'Fig_Decision_Curve')


# =============================================================================
#  9. SECTION 13C.  ARE THE RANKING DIFFERENCES SIGNIFICANT?  -> Tables S5, S6
#
#  Section 5.1 concludes from the average-precision ordering, but the published
#  benchmark only ever tested F2. This applies the identical machinery (paired
#  Wilcoxon, corrected resampled t-test, Cohen's d, Holm across the 15 pairwise
#  comparisons) to average precision and to expected calibration error.
#  No model is refitted.
# =============================================================================

def load_fold_metrics(outdir, ckpt, n_splits, n_repeats):
    """15 per-fold values per strategy for AvgPrecision and ECE."""
    csv = os.path.join(outdir, 'Fold_Metrics_PerFold.csv')
    if os.path.exists(csv):
        d = pd.read_csv(csv, keep_default_na=False)
        fm = {}
        for s in STRATEGIES:
            sub = d[d['Strategy'] == s].sort_values('Fold')
            if not len(sub):
                raise SystemExit(f'{csv} has no rows for strategy {s!r}')
            fm[s] = {'AvgPrecision': sub['AvgPrecision'].astype(float).tolist(),
                     'ECE': sub['ECE'].astype(float).tolist()}
        print(f'  read Fold_Metrics_PerFold.csv ({len(d)} rows)')
        return fm
    print('  Fold_Metrics_PerFold.csv absent; falling back to checkpoints')
    fm = {s: {'AvgPrecision': [], 'ECE': []} for s in STRATEGIES}
    for f in range(1, n_splits * n_repeats + 1):
        blob = ckpt.load(f'cv_fold{f:02d}')
        if blob is None:
            raise SystemExit(
                f'checkpoint cv_fold{f:02d} missing and '
                'Fold_Metrics_PerFold.csv absent.\n'
                'Apply the run_analysis.py patch in docs/UPLOAD_PLAN_v32.md '
                'and rerun, or keep the _checkpoints/ folder from that run.')
        for s in STRATEGIES:
            fm[s]['AvgPrecision'].append(blob['metrics'][s]['AvgPrecision'])
            fm[s]['ECE'].append(blob['metrics'][s]['ECE'])
    return fm


def run_threshold_free_tests(outdir, fm, n_rows, n_splits):
    from scipy.stats import wilcoxon

    print('\n' + '=' * 72)
    print('  13C. ARE THE RANKING AND CALIBRATION DIFFERENCES SIGNIFICANT?')
    print('=' * 72)
    n_tr = int(n_rows * (n_splits - 1) / n_splits)
    n_te = n_rows - n_tr

    def paired_table(metric, higher_is_better=True):
        rows = []
        for a, b in combinations(STRATEGIES, 2):
            va = np.asarray(fm[a][metric], float)
            vb = np.asarray(fm[b][metric], float)
            diff = va - vb
            try:
                p_w = wilcoxon(va, vb).pvalue
            except ValueError:
                p_w = 1.0
            _, p_t = corrected_resampled_t(diff, n_tr, n_te)
            rows.append({
                'A': a, 'B': b,
                f'{metric} A': va.mean(), f'{metric} B': vb.mean(),
                'Mean diff (A-B)': diff.mean(),
                "Cohen's d": cohens_d_pooled(va, vb),
                'Wilcoxon p': p_w,
                'Corrected resampled t p': p_t,
                'Winner': (a if (va.mean() > vb.mean()) == higher_is_better
                           else b),
            })
        t = pd.DataFrame(rows).sort_values('Wilcoxon p').reset_index(drop=True)
        t['Wilcoxon p (Holm)'] = holm(t['Wilcoxon p'].values)
        t['Significant'] = t['Wilcoxon p (Holm)'] < 0.05
        return t

    ap = paired_table('AvgPrecision')
    ec = paired_table('ECE', higher_is_better=False)

    print('\n  Average precision, 15 pairwise comparisons on 15 paired folds:')
    print(ap.round(4).to_string(index=False))
    print(f"\n  {int(ap['Significant'].sum())} of 15 survive Holm on average "
          'precision.')
    print('\n  Expected calibration error, same machinery:')
    print(ec.round(4).to_string(index=False))
    print(f"  {int(ec['Significant'].sum())} of 15 survive Holm on ECE.")

    ap.to_csv(os.path.join(outdir, 'Stat_Wilcoxon_AP.csv'), index=False)
    ec.to_csv(os.path.join(outdir, 'Stat_Wilcoxon_ECE.csv'), index=False)
    print('\n  -> Stat_Wilcoxon_AP.csv, Stat_Wilcoxon_ECE.csv')
    return ap, ec


# =============================================================================
#  10. MAIN
# =============================================================================

def main(argv=None):
    args = parse_args(argv)
    args.outdir = os.path.abspath(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    n_splits = 5
    n_repeats = 1 if args.quick else 3

    print('=' * 72)
    print('  EXTENDED ANALYSES: leakage ablation, decision curves,')
    print('  threshold-free significance tests')
    print('=' * 72)
    print(f'  outdir : {args.outdir}')
    print(f'  folds  : {n_splits} x {n_repeats}')
    if args.quick:
        print('\n  ' + '!' * 60)
        print('  QUICK MODE. Smoke test only. DO NOT REPORT THESE NUMBERS.')
        print('  ' + '!' * 60)

    # Only 13A fits models, so only 13A needs the descriptor matrix. 13B and
    # 13C need the labels alone. Building the features when they are not needed
    # would cost 20 to 40 minutes on a machine without a Magpie cache, for
    # nothing, so the label-only path skips them.
    needs_features = args.only in (None, 'ablation')
    df, X, y, groups = build_dataset(args, with_features=needs_features)
    n_rows = len(df)

    ckpt = Ckpt(args.outdir, fingerprint(args, n_rows),
                enabled=not args.no_resume)
    y_arr = np.asarray(y).astype(int)

    infl = None

    if args.only in (None, 'ablation'):
        print('\n' + '=' * 72)
        print('  13A. HOW MUCH DOES POLYMORPH LEAKAGE INFLATE PERFORMANCE?')
        print('=' * 72)
        device, _ = detect_device()
        build = make_pipeline_factory(args.seed, device)
        frozen = load_frozen_params(args.outdir)
        export_composition_structure(df, groups, args.outdir)
        splits = make_split_families(X, y, groups, n_splits, n_repeats,
                                     args.seed)
        leakage_rate(splits, groups, args.outdir)
        abl_fold, infl = run_ablation(args, X, y, groups, build, frozen,
                                      splits, ckpt)
        figure_ablation(abl_fold, args.outdir)

    if args.only in (None, 'ablation', 'contrast'):
        run_contrast_test(args.outdir, infl)

    if args.only in (None, 'dca'):
        P = load_oof(args.outdir, n_rows)
        dca, ceiling, nb_all, nb_none, pi = run_decision_curves(
            args.outdir, y_arr, P)
        figure_decision_curve(args.outdir, dca, ceiling, nb_all, nb_none, pi)

    if args.only in (None, 'tests'):
        fm = load_fold_metrics(args.outdir, ckpt, n_splits, n_repeats)
        run_threshold_free_tests(args.outdir, fm, n_rows, n_splits)

    print('\n' + '=' * 72)
    print('  EXTENDED ANALYSES COMPLETE')
    print(f'  everything written to {args.outdir}')
    print('=' * 72)
    return 0


if __name__ == '__main__':
    sys.exit(main())
