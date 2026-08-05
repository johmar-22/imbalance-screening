#!/usr/bin/env python3
"""Pre-push audit. Run from the repository root BEFORE the first git push.

Checks, in order of how badly each one can hurt:

  1. secrets       API keys, tokens and personal email addresses in any
                   tracked text file
  2. drive         leftover Google Drive / Colab paths and mounts
  3. magics        Jupyter ! and % lines that make a .py file unrunnable
  4. size          files above GitHub's 50 MB warning and 100 MB hard limit
  5. completeness  every file promised by docs/TABLE_MAP.md actually exists
  6. runnable      run_analysis.py parses and its --help works

Exit code is non-zero if any BLOCKER is found.

    python scripts/preflight_check.py
"""

import ast
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_EXT = {'.py', '.ipynb', '.md', '.txt', '.yml', '.yaml', '.cff', '.json',
            '.cfg', '.toml', '.csv'}
SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.ipynb_checkpoints',
             '_checkpoints', 'node_modules'}

# Patterns that must never appear in a public repository.
SECRET_PATTERNS = [
    (r'MP_API_KEY\s*[=:]\s*[\'"][A-Za-z0-9]{16,}[\'"]', 'hardcoded Materials Project key'),
    (r'X-API-KEY[\'"]\s*:\s*[\'"][A-Za-z0-9]{16,}[\'"]', 'hardcoded API key in a header'),
    (r'\bgh[pousr]_[A-Za-z0-9]{30,}', 'GitHub personal access token'),
    (r'AKIA[0-9A-Z]{16}', 'AWS access key id'),
    (r'sk-[A-Za-z0-9]{20,}', 'OpenAI-style secret key'),
    (r'-----BEGIN [A-Z ]*PRIVATE KEY-----', 'private key block'),
    (r'[A-Za-z0-9._%+-]+@(?:gmail|yahoo|hotmail|outlook)\.com', 'personal email address'),
]

DRIVE_PATTERNS = [
    (r'from\s+google\.colab\s+import', 'Colab import'),
    (r'drive\.mount\(', 'Drive mount call'),
    (r'/content/drive', 'hardcoded Drive path'),
    (r'MyDrive', 'hardcoded Drive path'),
]

blockers, warnings_ = [], []


def walk_text_files():
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            p = os.path.join(root, f)
            if os.path.splitext(f)[1].lower() in TEXT_EXT:
                yield p


def scan(patterns, label, bucket, skip_self=True):
    hits = 0
    for p in walk_text_files():
        if skip_self and os.path.abspath(p) == os.path.abspath(__file__):
            continue
        try:
            text = open(p, encoding='utf-8', errors='ignore').read()
        except OSError:
            continue
        for pat, desc in patterns:
            for m in re.finditer(pat, text):
                line = text[:m.start()].count('\n') + 1
                bucket.append(f"{label}: {desc} in "
                              f"{os.path.relpath(p, REPO)}:{line}  ->  "
                              f"{m.group(0)[:60]}")
                hits += 1
    return hits


print("=" * 72)
print("  1. SECRETS")
print("=" * 72)
n = scan(SECRET_PATTERNS, 'BLOCKER', blockers)
print(f"  {n} match(es)")
if n:
    print("  A key that has ever been pushed is compromised even after you "
          "delete it. Revoke it, then remove it, then push.")

print("\n" + "=" * 72)
print("  2. GOOGLE DRIVE AND COLAB DEPENDENCIES")
print("=" * 72)
n = scan(DRIVE_PATTERNS, 'WARNING', warnings_)
print(f"  {n} match(es). Acceptable only inside notebooks/ or inside a "
      f"docstring that explains the Colab path.")

print("\n" + "=" * 72)
print("  3. NOTEBOOK MAGICS IN .py FILES")
print("=" * 72)
n = 0
for p in walk_text_files():
    if not p.endswith('.py'):
        continue
    for i, line in enumerate(open(p, encoding='utf-8', errors='ignore'), 1):
        if re.match(r'^\s*[!%]\w', line):
            blockers.append(f"BLOCKER: notebook magic in "
                            f"{os.path.relpath(p, REPO)}:{i}  ->  {line.strip()[:60]}")
            n += 1
print(f"  {n} match(es)")

print("\n" + "=" * 72)
print("  4. FILE SIZES")
print("=" * 72)
big = []
for root, dirs, files in os.walk(REPO):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for f in files:
        p = os.path.join(root, f)
        mb = os.path.getsize(p) / 1e6
        if mb > 50:
            big.append((mb, os.path.relpath(p, REPO)))
for mb, rel in sorted(big, reverse=True):
    if mb > 100:
        blockers.append(f"BLOCKER: {rel} is {mb:.1f} MB, above GitHub's 100 MB "
                        f"hard limit. Push will be rejected.")
    else:
        warnings_.append(f"WARNING: {rel} is {mb:.1f} MB, above GitHub's 50 MB "
                         f"warning threshold.")
    print(f"  {mb:7.1f} MB  {rel}")
if not big:
    print("  no file above 50 MB")
tot = sum(os.path.getsize(os.path.join(r, f))
          for r, d, fs in os.walk(REPO) if not any(s in r for s in SKIP_DIRS)
          for f in fs) / 1e6
print(f"  repository total: {tot:.1f} MB")

print("\n" + "=" * 72)
print("  5. FILES PROMISED BY docs/TABLE_MAP.md")
print("=" * 72)
tm = os.path.join(REPO, 'docs', 'TABLE_MAP.md')
if os.path.isfile(tm):
    referenced = set(re.findall(r'`((?:results|figures|data|docs)/[^`]+)`',
                                open(tm, encoding='utf-8').read()))
    absent = []
    for r in sorted(referenced):
        r = r.split(' ')[0].rstrip(',')
        if r.endswith('*') or '[' in r:
            stem = r.split('.')[0].rstrip('*')
            if not any(f.startswith(os.path.basename(stem))
                       for f in os.listdir(os.path.join(REPO, os.path.dirname(r)))
                       if os.path.isdir(os.path.join(REPO, os.path.dirname(r)))):
                absent.append(r)
        elif not os.path.exists(os.path.join(REPO, r)):
            absent.append(r)
    for r in absent:
        warnings_.append(f"WARNING: TABLE_MAP.md references {r}, which is missing.")
    print(f"  {len(referenced)} referenced, {len(absent)} missing")
    for r in absent:
        print(f"    missing: {r}")
else:
    warnings_.append("WARNING: docs/TABLE_MAP.md not found.")

print("\n" + "=" * 72)
print("  6. SCRIPT IS RUNNABLE")
print("=" * 72)
ra = os.path.join(REPO, 'scripts', 'run_analysis.py')
if os.path.isfile(ra):
    try:
        ast.parse(open(ra, encoding='utf-8').read())
        print("  run_analysis.py parses")
    except SyntaxError as e:
        blockers.append(f"BLOCKER: run_analysis.py has a syntax error at line {e.lineno}")
        print(f"  SYNTAX ERROR line {e.lineno}: {e.msg}")
    r = subprocess.run([sys.executable, ra, '--help'],
                       capture_output=True, text=True, timeout=600)
    if r.returncode == 0 and '--outdir' in r.stdout:
        print("  --help works and advertises --outdir")
    else:
        warnings_.append("WARNING: `run_analysis.py --help` did not succeed. "
                         "Usually a missing dependency, not a code fault. "
                         "Install requirements.txt and rerun this check.")
        print(f"  --help returned {r.returncode}")
else:
    blockers.append("BLOCKER: scripts/run_analysis.py not found.")

print("\n" + "=" * 72)
print("  SUMMARY")
print("=" * 72)
for b in blockers:
    print("  " + b)
for w in warnings_:
    print("  " + w)
print(f"\n  {len(blockers)} blocker(s), {len(warnings_)} warning(s)")
if blockers:
    print("  DO NOT PUSH until every blocker is resolved.")
    sys.exit(1)
print("  Safe to push.")
