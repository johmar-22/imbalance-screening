#!/usr/bin/env python3
"""Populate results/, figures/ and data/ from the raw run output folder.

Run this ONCE, after downloading the Google Drive folder
`ml_project_v7_major_revision_checkpointed` and unzipping it somewhere local.
It copies files into the repository layout, renames the figures to the
manuscript numbering, and refuses to copy anything it does not recognize so
that nothing unexpected reaches the public repository.

    python scripts/organize_assets.py --src "C:/path/to/ml_project_v7_major_revision_checkpointed"
    python scripts/organize_assets.py --src ../raw_output --dry-run

Nothing here touches the network and nothing is deleted from the source.
"""

import argparse
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Result CSVs, copied verbatim into results/
# ---------------------------------------------------------------------------
RESULT_CSVS = [
    'Main_Results.csv',                 # Tables 1 and 2
    'Screening_PrecisionAtK.csv',       # Table 3
    'Top100_Candidates.csv',            # Table 4 and the full candidate list
    'LOCO_Generalization.csv',          # Table 5
    'External_Ricci_Validation.csv',    # Table 6
    'CrossBand_Incremental_Test.csv',   # Table 7
    'Recalibration_Summary.csv',        # Table 8
    'Stat_Wilcoxon_F2.csv',             # Table S2
    'Threshold_Sensitivity.csv',        # Table S3
    'Feature_Importances.csv',          # Table S4
    'Stat_McNemar.csv',
    'Noise_Audit.csv',
    'SHAP_Importances.csv',
    'Best_Hyperparameters.csv',
    'Composition_Redundancy.csv',
    'Recalibration_PerFold.csv',
    'Recal_Diag_ProbDist.csv',
    'Recal_Diag_TailReliability.csv',
    'Recal_Diag_PriorCorrection.csv',
]

# Large cached artifacts that belong in data/, not results/
DATA_FILES = ['magpie_cache.csv']

# ---------------------------------------------------------------------------
# Figures: script working name -> manuscript name. Each is copied in every
# extension that exists (pdf, png, tiff).
# ---------------------------------------------------------------------------
FIGURE_RENAMES = {
    'Fig1_Pipeline_Overview':   'Fig1_Pipeline_Overview',
    'Fig1_Strategy_Comparison': 'Fig2_Strategy_Comparison',
    'Fig2_PR_and_Calibration':  'Fig3_PR_and_Calibration',
    'Fig_Threshold_Sensitivity': 'Fig4_Threshold_Sensitivity',
    'Fig_SHAP_Summary':         'Fig5_SHAP_Summary',
    'Fig_Enrichment':           'Fig6_Enrichment',
    'Fig_Recalibration':        'Fig7_Recalibration',
    'Fig3_PCA_Diagnostic':      'FigS1_PCA_Diagnostic',
}
# Every extension that may exist. Not all figures have all four: the seven
# script-generated figures are written as pdf/png/tiff, while
# Fig1_Pipeline_Overview was drawn by hand and exists as png/svg/tiff with no
# PDF. Missing combinations are skipped silently; a figure with no file at all
# is reported as missing.
FIGURE_EXTS = ('pdf', 'svg', 'png', 'tiff')

MAX_GITHUB_MB = 50.0    # GitHub warns above this for a single file


def copy(src, dst, dry):
    size_mb = os.path.getsize(src) / 1e6
    flag = '  <-- OVER 50 MB, use Git LFS or omit' if size_mb > MAX_GITHUB_MB else ''
    print(f"  {os.path.basename(src):<38} -> {os.path.relpath(dst, REPO):<44}"
          f" {size_mb:7.2f} MB{flag}")
    if not dry:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    return size_mb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True,
                    help='the unzipped ml_project_v7_major_revision_checkpointed folder')
    ap.add_argument('--extra-figure-dir', default=None,
                    help='second folder searched for figures, e.g. the one '
                         'holding Fig1_Pipeline_Overview if it was drawn elsewhere')
    ap.add_argument('--dry-run', action='store_true',
                    help='print what would be copied and change nothing')
    a = ap.parse_args()

    if not os.path.isdir(a.src):
        sys.exit(f"source folder not found: {a.src}")

    search_dirs = [a.src] + ([a.extra_figure_dir] if a.extra_figure_dir else [])

    def find(name):
        for d in search_dirs:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
        return None

    total = 0.0
    missing = []

    print("\nRESULT TABLES -> results/")
    for f in RESULT_CSVS:
        p = find(f)
        if p:
            total += copy(p, os.path.join(REPO, 'results', f), a.dry_run)
        else:
            missing.append(f)

    print("\nCACHED DATA -> data/")
    for f in DATA_FILES:
        p = find(f)
        if p:
            total += copy(p, os.path.join(REPO, 'data', f), a.dry_run)
        else:
            missing.append(f)

    print("\nFIGURES -> figures/  (renamed to manuscript numbering)")
    for old, new in FIGURE_RENAMES.items():
        found_any = False
        for ext in FIGURE_EXTS:
            p = find(f'{old}.{ext}')
            if p:
                found_any = True
                total += copy(p, os.path.join(REPO, 'figures', f'{new}.{ext}'),
                              a.dry_run)
        if not found_any:
            missing.append(f'{old}.[pdf|png|tiff]')

    print(f"\nTotal copied: {total:.1f} MB")
    if total > 200:
        print("  WARNING: over 200 MB. GitHub soft-limits repositories at 1 GB "
              "and this will make cloning slow. Consider dropping the TIFFs.")

    if missing:
        print(f"\n{len(missing)} expected file(s) NOT FOUND:")
        for m in missing:
            print(f"  - {m}")
        print("\nIf a file is missing because that section did not run, rerun "
              "run_analysis.py before publishing. Do not publish a table map "
              "that points at files which are not there.")
    else:
        print("\nAll expected files present.")

    print("\nUnrecognized files left behind in the source folder "
          "(NOT copied, review before adding anything by hand):")
    known = set(RESULT_CSVS) | set(DATA_FILES) | {
        f'{o}.{e}' for o in FIGURE_RENAMES for e in FIGURE_EXTS}
    leftovers = sorted(f for f in os.listdir(a.src)
                       if os.path.isfile(os.path.join(a.src, f)) and f not in known)
    for f in leftovers:
        print(f"  - {f}")
    if not leftovers:
        print("  (none)")

    if a.dry_run:
        print("\nDRY RUN: nothing was written. Rerun without --dry-run.")


if __name__ == '__main__':
    main()
