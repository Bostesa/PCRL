"""Engineering review (label-free, read-only): SHA-256 of every committed version of the campaign's modules, to map the
code_sha256 values recorded in release records (OCC.json / RELEASE.json / VERIFY.json / STEP1.json) to git history.

    python3 results/pcrl_occupation_headroom_overnight_v1/roles/engineering_review/code/code_history.py [OUT.json]

Writes JSON {module: [{commit, commit_time, sha256}], 'working_tree': {module: sha256}}. Reads git objects of this worktree only.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
MODULES = ('ohn/fam_a.py', 'ohn/fam_b.py', 'ohn/fam_c.py', 'ohn/protect.py', 'ohn/fam_inc.py', 'ohn/devrun.py', 'ohn/evaluate.py',
           'ohn/deploy.py', 'ohn/common.py', 'ohn/jobqueue.py', 'ohn/rules.py', 'ohn/devreaders.py', 'ohn/ledger.py', 'ohn/witness.py',
           'ohn/grid_summary.py', 'ohn/nominate.py', 'ohn/tables.py', 'ohn/receipt.py', 'ohn/publish_build.py')


def git(*a, binary=False):
    r = subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True)
    return r.stdout if binary else r.stdout.decode().strip()


def history():
    out = {'working_tree': {}}
    for m in MODULES:
        hist = []
        for line in git('log', '--format=%H %cI', '--', m).splitlines():
            c, t = line.split()
            blob = git('show', f'{c}:{m}', binary=True)
            hist.append({'commit': c[:9], 'commit_time': t, 'sha256': hashlib.sha256(blob).hexdigest()})
        out[m] = hist
        p = ROOT / m
        out['working_tree'][m] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return out


def main():
    text = json.dumps(history(), indent=1)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text + '\n')
    else:
        print(text)


if __name__ == '__main__':
    main()
