"""
Figure: representational leakage in composition-only models.

Draws the two SnS entries of the boltztrap_mp release from their real pymatgen
structures, shows that a formula-only descriptor map sends them to one identical
132-dimensional vector carrying conflicting labels, and contrasts what a random
split does with what a composition-grouped split does.

Nothing here is a cartoon: atom positions come from the `structure` column of
the matminer/Figshare `boltztrap_mp` release. Only the 132-cell colour _lk_strip is decorative, and it is drawn
from a fixed seed so the two copies are provably identical.

Output: Fig_Leakage_Mechanism.{pdf,tiff,png}
Style follows the same Nature-family specification as the other figures.

Import-safe: every module-level name is prefixed so that pasting this into a
live notebook kernel cannot shadow the pipeline factory `build()`, the helper
`panel_label()`, or the matplotlib rcParams used by the other figures.
"""
import ast
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

# ----------------------------------------------------------------- style
_LK_MM = 1 / 25.4
_LK_W2COL = 183 * _LK_MM
_LK_FS = 8
_LK_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': _LK_FS,
    'axes.linewidth': 0.7,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}
_LK_CB = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
      '#E69F00', '#56B4E9', '#000000', '#999999']
_LK_BLUE, _LK_VERM, _LK_GREEN, _LK_GREY = _LK_CB[0], _LK_CB[1], _LK_CB[2], _LK_CB[7]
_LK_SN, _LK_S = '#6E6E6E', '#F0E442'
_LK_INK = '#222222'

# Data source, mirroring Section 1 of the notebook. 'auto' tries matminer first
# (which fetches and caches the Figshare copy), then the Figshare API directly,
# then a local CSV. matminer is preferred because it returns the SAME 8,924-row
# release the manuscript analyses; the local boltztrap_mp.csv in the project
# folder is an older 9,036-row snapshot.
_LK_SOURCE = os.environ.get('BOLTZTRAP_SOURCE', 'auto')     # auto|matminer|figshare|csv
_LK_CSV = os.environ.get('BOLTZTRAP_CSV', 'boltztrap_mp.csv')
_LK_OUT = os.environ.get('FIG_OUT', '.')
_LK_FIGSHARE_ARTICLE = 7221410          # matminer's boltztrap_mp deposit


def _lk_standardise(d):
    """Map matminer's lowercase column names onto the ones used here.

    Same mapping as `_standardise()` in Section 1 of the notebook.
    """
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
        elif lc in ('mpid', 'mp_id', 'material_id'):
            ren[c] = 'mpid'
        elif lc in ('formula', 'pretty_formula', 'full_formula'):
            ren[c] = 'formula'
    return d.rename(columns=ren)


def _lk_from_matminer():
    from matminer.datasets import load_dataset
    d = _lk_standardise(load_dataset('boltztrap_mp'))
    print(f"  source: matminer 'boltztrap_mp' ({len(d):,} rows)")
    return d


def _lk_from_figshare():
    """Resolve the download URL from the Figshare API, then stream the CSV."""
    import requests
    api = f'https://api.figshare.com/v2/articles/{_LK_FIGSHARE_ARTICLE}/files'
    files = requests.get(api, timeout=60).json()
    url = next((f['download_url'] for f in files
                if str(f.get('name', '')).lower().startswith('boltztrap_mp')
                and str(f.get('name', '')).lower().endswith('.csv')), None)
    if url is None:
        raise RuntimeError(
            f'no boltztrap_mp CSV in Figshare article {_LK_FIGSHARE_ARTICLE}; '
            f'saw {[f.get("name") for f in files]}')
    print(f'  source: Figshare article {_LK_FIGSHARE_ARTICLE} -> {url}')
    return _lk_standardise(pd.read_csv(url))


def _lk_from_csv():
    print(f'  source: local CSV {_LK_CSV}')
    return _lk_standardise(pd.read_csv(_LK_CSV))


def _lk_load_boltztrap(source=None):
    source = source or _LK_SOURCE
    order = {'auto': ('matminer', 'figshare', 'csv'),
             'matminer': ('matminer',), 'figshare': ('figshare',),
             'csv': ('csv',)}[source]
    errors = []
    for how in order:
        try:
            return {'matminer': _lk_from_matminer,
                    'figshare': _lk_from_figshare,
                    'csv': _lk_from_csv}[how]()
        except Exception as e:                       # noqa: BLE001
            errors.append(f'    {how}: {type(e).__name__}: {e}')
            print(f'  {how} unavailable, falling back')
    raise RuntimeError('could not load boltztrap_mp from any source:\n'
                       + '\n'.join(errors))


def _lk_as_struct_dict(obj):
    """Accept a pymatgen Structure, a dict, or a serialised string."""
    if hasattr(obj, 'as_dict'):                      # pymatgen Structure
        return obj.as_dict()
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:                            # noqa: BLE001
            return ast.literal_eval(obj)
    raise TypeError(f'unrecognised structure payload: {type(obj)}')


def _lk_load_sns(source=None):
    """The two SnS entries of the release, ordered by hole effective mass.

    Identifiers are read from the data rather than hard-coded, so the figure
    always labels whatever the release actually contains.
    """
    d = _lk_load_boltztrap(source)
    need = {'formula', 'm_p', 'structure'}
    if not need.issubset(d.columns):
        raise RuntimeError(f'missing columns {need - set(d.columns)}; '
                           f'got {list(d.columns)}')
    sub = d[d['formula'].astype(str).str.strip() == 'SnS'].copy()
    if len(sub) != 2:
        raise RuntimeError(f'expected exactly 2 SnS entries in this release, '
                           f'found {len(sub)}. The figure argues from a pair; '
                           f'inspect the data before redrawing it.')
    if 'mpid' not in sub.columns:
        sub['mpid'] = [f'entry-{i}' for i in range(len(sub))]
    sub = sub.sort_values('m_p')
    entries = [(str(r['mpid']), float(r['m_p']),
                _lk_as_struct_dict(r['structure'])) for _, r in sub.iterrows()]
    lo, hi = entries[0], entries[-1]
    print(f"  SnS pair: {lo[0]} m*p = {lo[1]:g} me  (positive)   "
          f"{hi[0]} m*p = {hi[1]:g} me  (negative)")
    return entries


def _lk_cartesian(struct, reps=(1, 1, 1)):
    M = np.array(struct['lattice']['matrix'], float)
    base = [(s['label'], np.array(s['abc'], float)) for s in struct['sites']]
    pts = []
    r1, r2, r3 = reps
    for i in range(-r1, r1 + 1):
        for j in range(-r2, r2 + 1):
            for k in range(-r3, r3 + 1):
                for lab, abc in base:
                    pts.append((lab, (abc + np.array([i, j, k])) @ M))
    return M, pts


def _lk_draw_structure(ax, struct, axes2d=(0, 2), window=(11.0, 10.0), cutoff=3.35):
    """Orthographic projection of the deposited cell onto two Cartesian axes.

    Both panels use the same physical window, so the two cells are drawn to a
    common scale and can be compared directly.
    """
    u, v = axes2d
    w = ({0, 1, 2} - {u, v}).pop()
    M, pts = _lk_cartesian(struct, reps=(2, 2, 2))

    order = sorted(range(3), key=lambda i: -(abs(M[i][u]) + abs(M[i][v])))
    a_vec, b_vec = M[order[0]], M[order[1]]
    corners = np.array([[0, 0],
                        [a_vec[u], a_vec[v]],
                        [a_vec[u] + b_vec[u], a_vec[v] + b_vec[v]],
                        [b_vec[u], b_vec[v]]])
    cx, cy = corners[:, 0].mean(), corners[:, 1].mean()

    ww, wh = window
    x0, x1 = cx - ww / 2, cx + ww / 2
    y0, y1 = cy - wh / 2, cy + wh / 2

    depth = np.array([p[w] for _, p in pts])
    dmid = np.median(depth)
    vis = [(lab, p) for lab, p in pts
           if x0 - 0.8 <= p[u] <= x1 + 0.8 and y0 - 0.8 <= p[v] <= y1 + 0.8
           and abs(p[w] - dmid) < 3.2]

    for i in range(len(vis)):
        for j in range(i + 1, len(vis)):
            (la, pa), (lb, pb) = vis[i], vis[j]
            if la == lb:
                continue
            if np.linalg.norm(pa - pb) < cutoff:
                ax.plot([pa[u], pb[u]], [pa[v], pb[v]], '-', color='#a5a5a5',
                        lw=0.8, zorder=1, solid_capstyle='round')

    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=_LK_INK,
                         lw=0.8, ls=(0, (3, 2)), zorder=4))

    vis.sort(key=lambda t: t[1][w])
    for lab, p in vis:
        col = _LK_SN if lab == 'Sn' else _LK_S
        size = 42 if lab == 'Sn' else 26
        ax.scatter(p[u], p[v], s=size, facecolor=col, edgecolor=_LK_INK,
                   linewidths=0.5, zorder=3)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ------------------------------------------------------------- primitives
def _lk_strip(ax, x, y, w, h, n=132, seed=7, label=None, lw=0.6):
    """The 132-dimensional descriptor vector as a row of cells."""
    rng = np.random.default_rng(seed)
    vals = rng.random(n)
    cmap = plt.get_cmap('Blues')
    cw = w / n
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((x + i * cw, y), cw, h,
                               facecolor=cmap(0.18 + 0.62 * v),
                               edgecolor='none', zorder=2))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=_LK_INK,
                           lw=lw, zorder=3))
    if label:
        ax.text(x + w / 2, y + h + 0.012, label, ha='center', va='bottom',
                fontsize=_LK_FS - 1)


def _lk_grey_strip(ax, x, y, w, h, n=132, seed=0, lw=0.5):
    """A descriptor vector belonging to some other composition."""
    rng = np.random.default_rng(seed)
    vals = rng.random(n)
    cmap = plt.get_cmap('Greys')
    cw = w / n
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((x + i * cw, y), cw, h,
                               facecolor=cmap(0.12 + 0.28 * v),
                               edgecolor='none', zorder=2))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='#999999',
                           lw=lw, zorder=3))


def _lk_chip(ax, x, y, w, h, text, color, fontsize=None, bold=False, alpha=0.16):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.006,rounding_size=0.012',
                                facecolor=color, alpha=alpha,
                                edgecolor=color, lw=0.8, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize or _LK_FS - 1, color=_LK_INK,
            fontweight='bold' if bold else 'normal', zorder=3)


def _lk_arrow(ax, p0, p1, color=_LK_INK, lw=0.9, style='-|>', rad=0.0, ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style,
                                 mutation_scale=7, lw=lw, color=color,
                                 linestyle=ls, zorder=4,
                                 connectionstyle=f'arc3,rad={rad}',
                                 shrinkA=1.5, shrinkB=1.5))


def _lk_blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def _lk_panel_label(ax, letter, x=-0.01, y=1.0):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=_LK_FS + 2,
            fontweight='bold', va='top', ha='left')


# ------------------------------------------------------------------ figure
def build_leakage_figure(entries):
    """entries: [(mpid, m_p, structure_dict)] sorted by m_p, from _lk_load_sns()."""
    with plt.rc_context(_LK_RC):
        return _lk_build(entries)


def _lk_build(entries):
    (mp_pos, mp_pos_val, st_pos), (mp_neg, mp_neg_val, st_neg) = entries

    fig = plt.figure(figsize=(_LK_W2COL, _LK_W2COL * 0.615))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.06, 0.80], hspace=0.20)

    # ---------------------------------------------------------- panel (a)
    ax = fig.add_subplot(gs[0])
    _lk_blank(ax)
    _lk_panel_label(ax, 'a', x=-0.005, y=1.14)
    ax.text(0.030, 1.14, 'The descriptor map is many-to-one over polymorphs',
            fontsize=_LK_FS + 0.5, fontweight='bold', va='top', ha='left')

    axs1 = ax.inset_axes([0.030, 0.560, 0.135, 0.370])
    axs2 = ax.inset_axes([0.030, 0.075, 0.135, 0.370])
    _lk_draw_structure(axs1, st_pos, axes2d=(0, 2))
    _lk_draw_structure(axs2, st_neg, axes2d=(0, 2))
    ax.text(0.0975, 0.985, f'SnS   {mp_pos}', ha='center', va='center',
            fontsize=_LK_FS - 0.5, fontweight='bold')
    ax.text(0.0975, 0.512, f'SnS   {mp_neg}', ha='center', va='center',
            fontsize=_LK_FS - 0.5, fontweight='bold')
    ax.text(0.0975, 0.020, 'two entries, one formula', ha='center', va='center',
            fontsize=_LK_FS - 1.5, style='italic', color='#555555')

    fx, fy, fw, fh = 0.245, 0.400, 0.150, 0.210
    _lk_chip(ax, fx, fy, fw, fh, '', _LK_GREY, alpha=0.10)
    ax.text(fx + fw / 2, fy + fh * 0.70, 'Magpie featurizer', ha='center',
            va='center', fontsize=_LK_FS - 0.5, fontweight='bold')
    ax.text(fx + fw / 2, fy + fh * 0.28, r'$\varphi$(formula): 132 statistics',
            ha='center', va='center', fontsize=_LK_FS - 1.5, color='#444444')

    _lk_arrow(ax, (0.175, 0.730), (fx - 0.004, 0.585), color=_LK_BLUE, rad=0.15)
    _lk_arrow(ax, (0.175, 0.250), (fx - 0.004, 0.425), color=_LK_VERM, rad=-0.15)
    ax.text(0.222, 0.880, 'formula\nonly', fontsize=_LK_FS - 2, color='#555555',
            ha='center', va='center')

    sx, sw, sy, sh = 0.450, 0.200, 0.470, 0.095
    _lk_strip(ax, sx, sy, sw, sh, seed=11)
    ax.text(sx + sw / 2, sy + sh + 0.055,
            r'one identical $\mathbf{x}\in\mathbb{R}^{132}$', ha='center',
            va='bottom', fontsize=_LK_FS, fontweight='bold')
    _lk_arrow(ax, (fx + fw + 0.004, 0.505), (sx - 0.008, 0.5175))

    _lk_arrow(ax, (sx + sw + 0.008, 0.545), (0.736, 0.700), color=_LK_BLUE, rad=0.15)
    _lk_arrow(ax, (sx + sw + 0.008, 0.490), (0.736, 0.320), color=_LK_VERM, rad=-0.15)
    _lk_chip(ax, 0.742, 0.640, 0.245, 0.185,
         f'y = 1   positive\nm*$_p$ = {mp_pos_val:g} $m_e$', _LK_BLUE)
    _lk_chip(ax, 0.742, 0.230, 0.245, 0.185,
         f'y = 0   negative\nm*$_p$ = {mp_neg_val:g} $m_e$', _LK_VERM)
    ax.text(0.8645, 0.530, 'same input,\nopposite labels', ha='center',
            va='center', fontsize=_LK_FS - 1, style='italic', color='#333333')

    ax.text(0.435, 0.185,
            '58 of the 812 multi-polymorph compositions\n'
            'straddle the cutoff in exactly this way',
            fontsize=_LK_FS - 1.5, color='#555555', ha='left', va='center')

    ax.scatter([0.448], [0.028], s=42, facecolor=_LK_SN, edgecolor=_LK_INK,
               linewidths=0.5, clip_on=False)
    ax.text(0.463, 0.028, 'Sn', va='center', fontsize=_LK_FS - 1.5)
    ax.scatter([0.503], [0.028], s=26, facecolor=_LK_S, edgecolor=_LK_INK,
               linewidths=0.5, clip_on=False)
    ax.text(0.516, 0.028, 'S', va='center', fontsize=_LK_FS - 1.5)
    ax.text(0.545, 0.028,
            'deposited structures, drawn to a common scale; unit cells dashed',
            va='center', fontsize=_LK_FS - 2, color='#777777')

    # ------------------------------------------------ panels (b) and (c)
    gsb = gs[1].subgridspec(1, 2, wspace=0.13)

    def split_panel(cell, letter, title, grouped):
        a = fig.add_subplot(cell)
        _lk_blank(a)
        _lk_panel_label(a, letter, x=-0.015, y=1.14)
        a.text(0.035, 1.14, title, fontsize=_LK_FS + 0.5, fontweight='bold',
               va='top', ha='left')
        for x0, name, col in [(0.020, 'Training folds', _LK_GREEN),
                              (0.545, 'Held-out fold', _LK_VERM)]:
            a.add_patch(FancyBboxPatch((x0, 0.310), 0.435, 0.610,
                                       boxstyle='round,pad=0.008,rounding_size=0.02',
                                       facecolor='none', edgecolor=col, lw=0.9,
                                       linestyle=(0, (4, 2)), zorder=1))
            a.text(x0 + 0.2175, 0.960, name, ha='center', fontsize=_LK_FS - 0.5,
                   color=col, fontweight='bold')

        SW, SH, GH = 0.335, 0.070, 0.042          # _lk_strip width, SnS height, grey height
        LX, RX = 0.070, 0.595                     # left and right _lk_strip origins

        if not grouped:
            # both folds are full; only the two SnS rows are picked out
            for k, yy in enumerate((0.855, 0.795)):
                _lk_grey_strip(a, LX, yy, SW, GH, seed=100 + k)
                _lk_grey_strip(a, RX, yy, SW, GH, seed=200 + k)
            _lk_strip(a, LX, 0.600, SW, SH, seed=11)
            _lk_strip(a, RX, 0.600, SW, SH, seed=11)
            a.text(LX + SW / 2, 0.700, f'{mp_pos}   y = 1', ha='center',
                   va='center', fontsize=_LK_FS - 1.5, color=_LK_BLUE)
            a.text(RX + SW / 2, 0.700, f'{mp_neg}   y = 0', ha='center',
                   va='center', fontsize=_LK_FS - 1.5, color=_LK_VERM)
            a.text(0.500, 0.635, '=', ha='center', va='center',
                   fontsize=_LK_FS + 4, fontweight='bold', color=_LK_INK)
            for k, yy in enumerate((0.500, 0.440, 0.380)):
                _lk_grey_strip(a, LX, yy, SW, GH, seed=300 + k)
                _lk_grey_strip(a, RX, yy, SW, GH, seed=400 + k)
            a.text(0.020, 0.135,
                   'The held-out vector was seen in training:\n'
                   'memorization is scored as generalization.',
                   fontsize=_LK_FS - 0.5, color=_LK_VERM, ha='left', va='center')
        else:
            for k, yy in enumerate((0.855, 0.795)):
                _lk_grey_strip(a, LX, yy, SW, GH, seed=100 + k)
            a.add_patch(FancyBboxPatch((LX - 0.020, 0.435), SW + 0.040, 0.325,
                                       boxstyle='round,pad=0.008,rounding_size=0.02',
                                       facecolor=_LK_BLUE, alpha=0.07, edgecolor=_LK_BLUE,
                                       lw=0.8, zorder=0))
            _lk_strip(a, LX, 0.640, SW, 0.062, seed=11)
            a.text(LX + SW / 2, 0.728, f'{mp_pos}   y = 1', ha='center',
                   va='center', fontsize=_LK_FS - 1.5, color=_LK_BLUE)
            _lk_strip(a, LX, 0.520, SW, 0.062, seed=11)
            a.text(LX + SW / 2, 0.608, f'{mp_neg}   y = 0', ha='center',
                   va='center', fontsize=_LK_FS - 1.5, color=_LK_VERM)
            a.text(LX + SW / 2, 0.470, 'group key = "SnS"', ha='center',
                   va='center', fontsize=_LK_FS - 1.5, color=_LK_BLUE)
            _lk_grey_strip(a, LX, 0.380, SW, GH, seed=302)
            # the held-out fold is full, it simply contains no SnS
            for k, yy in enumerate((0.855, 0.795, 0.735, 0.675)):
                _lk_grey_strip(a, RX, yy, SW, GH, seed=500 + k)
            for k, yy in enumerate((0.440, 0.380)):
                _lk_grey_strip(a, RX, yy, SW, GH, seed=600 + k)
            a.text(RX + SW / 2, 0.565, '1,785 compounds,\nnone of them SnS',
                   ha='center', va='center', fontsize=_LK_FS - 1.5, color=_LK_VERM,
                   style='italic')
            a.text(0.020, 0.135,
                   'One score serves both: one hit, one miss.\n'
                   'An irreducible ceiling, correctly counted.',
                   fontsize=_LK_FS - 0.5, color=_LK_GREEN, ha='left', va='center')

        if letter == 'b':
            _lk_grey_strip(a, 0.020, 0.020, 0.075, 0.032, seed=999)
            a.text(0.105, 0.036, 'some other composition', va='center',
                   fontsize=_LK_FS - 2, color='#777777')
        return a

    split_panel(gsb[0], 'b',
                'Random split: the leakage (80% of splits)', grouped=False)
    split_panel(gsb[1], 'c', 'Composition-grouped split', grouped=True)
    return fig


def make_leakage_figure(source=None):
    entries = _lk_load_sns(source)
    fig = build_leakage_figure(entries)
    for ext in ('pdf', 'png', 'tiff'):
        kw = {'dpi': 600, 'bbox_inches': 'tight', 'pad_inches': 0.02}
        if ext == 'tiff':
            kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(os.path.join(_LK_OUT, f'Fig_Leakage_Mechanism.{ext}'), **kw)
    print('written to', _LK_OUT)


if __name__ == '__main__':
    make_leakage_figure()
