"""Tier B provenance audit of prior 2016 use (label-free; reads no 2016 record)."""
from __future__ import annotations
import re
import subprocess
from pathlib import Path

from .common import OUT, ROOT, atomic_json, now

TOKENS = ('ss16p', 'acs_2016', 'PUMSDataDict16', 'csv_pca_2016', 'acs2016', 'ACS_2016', '2016_admission')
HOME = Path('/Users/nathansamson')


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True).stdout


def commits_touching():
    out = {}
    for token in TOKENS:
        log = git('log', '--all', '--format=%H %cI %s', '-S', token)
        out[token] = [line for line in log.splitlines() if line.strip()]
    return out


def tracked_paths():
    refs = [r for r in git('for-each-ref', '--format=%(refname)', 'refs/heads', 'refs/remotes').split() if r]
    hits = {}
    for ref in refs:
        for path in git('ls-tree', '-r', '--name-only', ref).splitlines():
            if any(t.lower() in path.lower() for t in TOKENS) or re.search(r'(^|/)[^/]*2016[^/]*\.(csv|npz|parquet|joblib|pt)$', path):
                hits.setdefault(path, []).append(ref)
    return hits


def filesystem():
    roots = [p for p in HOME.glob('PCRL*') if p.is_dir()]+[ROOT.parents[1]/'.worktrees' if (ROOT.parents[1]/'.worktrees').exists() else ROOT]
    hits = []
    for root in roots:
        proc = subprocess.run(['find', str(root), '-not', '-path', '*/.git/*', '(', '-iname', '*ss16*', '-o', '-iname', '*2016*',
                               ')', '-not', '-path', '*/pcrl_final_prospective_v1/*'], capture_output=True, text=True)
        hits += [line for line in proc.stdout.splitlines() if line]
    return sorted(set(hits))


def s3():
    proc = subprocess.run(['aws', '--profile', 'vein', 's3', 'ls', '--recursive', 's3://pcrl-ux-archive-ed9d21fd/'],
                          capture_output=True, text=True)
    return [line for line in proc.stdout.splitlines() if any(t.lower() in line.lower() for t in TOKENS) or 'ss16' in line]


def main():
    record = {'created_utc': now(), 'tokens': TOKENS, 'git_commits_adding_or_removing_token': commits_touching(),
              'tracked_paths_matching': tracked_paths(), 'filesystem_matches': filesystem(),
              'private_archive_matches': s3(),
              'method': 'git pickaxe over all refs, tree listing of all branches, filesystem find over PCRL worktrees, private archive listing; no 2016 record opened'}
    atomic_json(OUT/'private'/'PROVENANCE_AUDIT_RAW.json', record)
    return record


if __name__ == '__main__':
    r = main()
    for k, v in r['git_commits_adding_or_removing_token'].items():
        print(k, len(v)); [print('   ', x) for x in v[:12]]
    print('tracked', {k: v[:2] for k, v in list(r['tracked_paths_matching'].items())[:40]})
    print('fs', r['filesystem_matches'][:60])
    print('s3', r['private_archive_matches'][:20])
