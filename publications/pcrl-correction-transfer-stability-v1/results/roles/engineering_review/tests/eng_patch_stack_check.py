"""Apply PATCHES/1-6 (and any fork patches 20-39 present) together to a scratch copy of cts/, byte-compile, and run invented-data
checks of the patched behaviour. Reads no label or person-level file.

    python3 <this file> <scratch dir>        (from the worktree root)
"""
import importlib
import json
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[5]
ER = WT / 'results/pcrl_correction_transfer_stability_v1/roles/engineering_review'
SCR = Path(sys.argv[1]) / 'combo'
if SCR.exists():
    shutil.rmtree(SCR)
SCR.mkdir(parents=True)
shutil.copytree(WT / 'cts', SCR / 'cts', ignore=shutil.ignore_patterns('__pycache__'))
res = {}
diffs = sorted(ER.glob('PATCHES/*.diff'), key=lambda p: int(p.stem))
for d in diffs:
    r = subprocess.run(['patch', '-s', '-p1', '-d', str(SCR), '-i', str(d)], capture_output=True, text=True)
    res[f'apply_{d.name}'] = (r.returncode == 0, r.stdout + r.stderr)
for f in sorted((SCR / 'cts').glob('*.py')):
    py_compile.compile(str(f), doraise=True)
res['compile_all'] = (True, '')

sys.path.insert(0, str(SCR)); sys.path.insert(1, str(WT))
import numpy as np                                                     # noqa: E402
C = importlib.import_module('cts.common')
assert Path(C.__file__).resolve().is_relative_to(SCR)
unit = np.repeat(np.arange(3000), 2)
C.TAG_SUFFIX['D0'] = 'b'
m = C.partition_masks(unit)
res['d0b_follows_d0_suffix'] = (all(np.array_equal(m['D0'][r], m['D0B'][r]) for r in C.PROLES), '')
C.TAG_SUFFIX.clear()
res['repair_frozen_false_without_file'] = (C.repair_frozen() is False, '')
RS = importlib.import_module('cts.rstage')
res['rstage_no_recipe_file_ok'] = (RS.selected_recipe() is None and RS.recipe_directions() == [], '')
CP = importlib.import_module('cts.compression')
src = (SCR / 'cts/compression.py').read_text()
res['compression_refuses_inf_target'] = ('if not np.isfinite(target):' in src, '')
out = {k: {'ok': bool(v[0]), 'detail': v[1][:300]} for k, v in res.items()}
(ER / 'tests' / 'eng_patch_stack_RESULT.json').write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
print(json.dumps({k: v['ok'] for k, v in out.items()}))
print(json.dumps({'all_ok': all(v['ok'] for v in out.values()), 'patches': [d.name for d in diffs]}))
