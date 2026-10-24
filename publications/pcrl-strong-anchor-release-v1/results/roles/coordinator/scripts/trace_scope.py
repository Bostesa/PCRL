# Coordinator script (label-free) that wrote the custody scope trace (torch venv; output path as argument); saved from the session scratchpad at closeout.
"""Label-free trace of every private file the campaign's bank members and deployment functions open (custody scope).
Run from the worktree root under the torch venv: python trace_scope.py OUT.json"""
import json
import os
import sys
import time

ROOT = str(__import__('pathlib').Path(__file__).resolve().parents[5])   # the repository root
sys.path.insert(0, ROOT)
os.chdir(ROOT)
opened = {}
phase = ['import']


def hook(event, args):
    if event != 'open':
        return
    p = args[0]
    if not isinstance(p, (str, bytes, os.PathLike)):
        return
    try:
        p = os.path.realpath(os.fsdecode(p))
    except Exception:
        return
    if '/.private/' in p or '/.worktrees/' in p or p.startswith('/Volumes/'):
        opened.setdefault(p, set()).add(phase[0])


sys.addaudithook(hook)
t0 = time.process_time()
from sar import common as C, devreaders as DR, deploy as DP  # noqa: E402

BANKS = ('SAR1_REFS', 'SAR1_FB_R1', 'SAR1_FA_R1', 'SAR1_FA_R2', 'SAR1_FB_R1_ROB', 'SAR1_FA_R2_ROB')
errors = {}
for b in BANKS:
    spec = json.loads((C.RESULTS / 'grid_specs' / f'{b}.json').read_text())
    for m, member in spec['members'].items():
        for v in C.VERSIONS:
            phase[0] = f'bank:{b}:{m}'
            try:
                DR.load_member(member, v)
            except Exception as e:
                errors[f'{b}/{m}/V{v}'] = f'{type(e).__name__}: {str(e)[:160]}'
PKGS = [('FA/FA_TFRB_R2', 0), ('FA/FA_TFRB_R2', 1), ('FA/FA_TFRB_R2_TASK_LO', 0), ('FA/FA_TFRB_R2_TASK_LO', 1),
        ('FA/FA_TFRB_R2_TASK_HI', 0), ('FA/FA_TFRO_R1_CONT', 0), ('FA/FA_TFRO_R1_CONT', 1), ('FB/FB_PFO', 0), ('FB/FB_PFO', 1),
        ('FB/FB_PFO_DQ_HI', 0), ('FB/FB_PFO_DQ_HI', 1), ('FB/FB_PFO_RAWJ_HI', 0), ('FB/FB_PFO_RAWJ_HI', 1)]
F = C.features('DEV_REPORT')
for name, v in PKGS:
    phase[0] = f'deploy:{name}:V{v}'
    try:
        DP.from_features(name, v, F['X'][:5], F['feature_names'])
    except Exception as e:
        errors[f'deploy/{name}/V{v}'] = f'{type(e).__name__}: {str(e)[:160]}'
out = {'n_files': len(opened), 'errors': errors, 'cpu_s': round(time.process_time() - t0, 1),
       'files': {p: sorted(s) for p, s in sorted(opened.items())}}
json.dump(out, open(sys.argv[1], 'w'), indent=1)
tops = {}
for p in opened:
    parts = p.split('/')
    k = '/'.join(parts[:9]) if '/.private/' in p else '/'.join(parts[:8])
    tops.setdefault(k, [0, 0])
    tops[k][0] += 1
    tops[k][1] += os.path.getsize(p) if os.path.isfile(p) else 0
for k, (n, b) in sorted(tops.items()):
    print(f'{n:5d} {b / 1e6:9.1f} MB  {k}')
print('errors', json.dumps(errors, indent=1)[:2000])
print('cpu_s', out['cpu_s'])
