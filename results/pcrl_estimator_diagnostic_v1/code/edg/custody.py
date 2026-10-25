"""EDG1 custody: back up the research branch (git bundle) with a SHA-256 manifest, then restore it into a fresh directory from
the bundle ALONE and run a representative check there:
  - the self-test;
  - representative repetitions recomputed from their seeds and compared exactly with the saved rows;
  - the frozen analysis re-run on the restored outputs, with the summary compared with the committed SUMMARY.json.

    python3 edg_run.py custody backup BACKUP_DIR        (writes BACKUP_DIR/<stamp>/{branch.bundle, MANIFEST.sha256, TIP})
    python3 edg_run.py custody restore BACKUP_SNAPSHOT  (clones into a temporary directory and checks; prints JSON)
Paths in the printed record are relative or redacted; no data other than this study's synthetic outputs is involved."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import RESULTS, WORKTREE

BRANCH = 'research/pcrl-estimator-diagnostic-v1'
REL = 'results/pcrl_estimator_diagnostic_v1'
CHECK_ROWS = ((13, (0, 1, 2)), (37, (0, 1)), (48, (0, 1)))     # (cell, reps) recomputed after the restore


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _same(x, y):
    """Deep value equality of parsed JSON; NaN equals NaN; dict key order ignored."""
    if isinstance(x, dict) and isinstance(y, dict):
        return set(x) == set(y) and all(_same(x[k], y[k]) for k in x)
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(_same(a, b) for a, b in zip(x, y))
    if isinstance(x, float) and isinstance(y, float) and x != x and y != y:
        return True
    return x == y


def backup(dest):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    snap = Path(dest) / f'pcrl_estimator_diagnostic_v1_{stamp}'
    snap.mkdir(parents=True)
    tip = subprocess.run(['git', '-C', str(WORKTREE), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(['git', '-C', str(WORKTREE), 'bundle', 'create', str(snap / 'branch.bundle'), BRANCH], check=True,
                   capture_output=True)
    (snap / 'TIP').write_text(tip + '\n')
    files = sorted(p for p in snap.iterdir() if p.name != 'MANIFEST.sha256')
    (snap / 'MANIFEST.sha256').write_text(''.join(f'{_sha(p)}  {p.name}\n' for p in files))
    return {'snapshot': snap.name, 'tip': tip, 'bundle_bytes': (snap / 'branch.bundle').stat().st_size,
            'utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}


def restore(snap):
    snap = Path(snap)
    man = dict(reversed(l.split('  ')) for l in (snap / 'MANIFEST.sha256').read_text().splitlines())
    out = {'manifest_ok': all(_sha(snap / n) == h for n, h in man.items())}
    tip = (snap / 'TIP').read_text().strip()
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / 'restored'
        subprocess.run(['git', 'clone', '-q', '-b', BRANCH, str(snap / 'branch.bundle'), str(dst)], check=True, capture_output=True)
        got = subprocess.run(['git', '-C', str(dst), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        out['tip_matches'] = got == tip
        out['tip'] = tip
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'EDG_REPORT': str(Path(tmp) / 'report')}
        (Path(tmp) / 'report' / 'figures').mkdir(parents=True)
        run = lambda *a: subprocess.run([sys.executable, str(dst / REL / 'edg_run.py'), *a], capture_output=True, text=True, env=env)
        st = run('selftest')
        out['selftest_exit_status'] = st.returncode
        code = ('import sys, json, gzip; sys.path.insert(0, %r); from edg import simulate as S, laws as L\n'
                'saved = {}\n'
                'for line in gzip.open(%r, "rt"):\n'
                '    r = json.loads(line); saved[(r["cell"], r["rep"])] = line.strip()\n'
                'res = []\n'
                'for cell, reps in %r:\n'
                '    sc = S.scenarios()[cell]; law = L.build(sc["family"], sc["pi"], sc["occ"])\n'
                '    keys = {k: S._wire_keys(v) for k, v in law["views"].items()}\n'
                '    for rep in reps:\n'
                '        row = S.one_rep(sc, law, rep, keys=keys if rep == 0 else None)\n'
                '        res.append([cell, rep, json.dumps(row, allow_nan=True) == saved[(cell, rep)]])\n'
                'print(json.dumps(res))\n') % (str(dst / REL / 'code'), str(dst / REL / 'out' / 'sim_all.jsonl.gz'), CHECK_ROWS)
        rr = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
        out['rows_recomputed_equal'] = json.loads(rr.stdout) if rr.returncode == 0 else rr.stderr[-2000:]
        an = run('analyze')
        out['analyze_exit_status'] = an.returncode
        if an.returncode == 0:
            # amendment A3: value equality (NaN-aware); the byte order of tied bank 'winners' entries follows the string hash seed
            a, b = json.load(open(Path(tmp) / 'report' / 'SUMMARY.json')), json.load(open(dst / REL / 'SUMMARY.json'))
            out['summary_equal_to_committed'] = _same(a, b)
            out['summary_bytes_equal'] = _sha(Path(tmp) / 'report' / 'SUMMARY.json') == _sha(dst / REL / 'SUMMARY.json')
            out['tables_equal_to_committed'] = _sha(Path(tmp) / 'report' / 'TABLES.md') == _sha(dst / REL / 'TABLES.md')
        out['restored_from'] = 'the bundle alone (fresh clone in a temporary directory)'
    out['all_pass'] = bool(out['manifest_ok'] and out['tip_matches'] and out['selftest_exit_status'] == 0
                           and isinstance(out['rows_recomputed_equal'], list) and all(x[2] for x in out['rows_recomputed_equal'])
                           and out.get('summary_equal_to_committed') is True)
    out['utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    return out


def main(argv):
    if argv[0] == 'backup':
        print(json.dumps(backup(argv[1])))
    elif argv[0] == 'restore':
        print(json.dumps(restore(argv[1])))
