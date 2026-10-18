"""Engineering review (CTS1): patches 20, 21, 22 (cts/jobqueue.py), 23 (cts/closeout.py) and 25 (cts/deploy.py) apply together to
copies of the committed files and the results byte-compile. No data; temporary directory only.

    python3 results/pcrl_correction_transfer_stability_v1/roles/engineering_review/tests/queue_patch_compose_test.py
"""
import json
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WT = Path(__file__).resolve().parents[5]
P = Path(__file__).resolve().parents[1] / 'PATCHES'
out = {}
with tempfile.TemporaryDirectory() as td:
    (Path(td) / 'cts').mkdir()
    for f in ('jobqueue.py', 'closeout.py', 'deploy.py'):
        shutil.copy(WT / 'cts' / f, Path(td) / 'cts' / f)
    for p in ('20.diff', '21.diff', '22.diff', '23.diff', '25.diff'):
        r = subprocess.run(['patch', '-p1', '-s', '-i', str(P / p)], cwd=td, capture_output=True, text=True)
        out[p] = 'applied' if r.returncode == 0 else (r.stdout + r.stderr)[-300:]
    for f in ('jobqueue.py', 'closeout.py', 'deploy.py'):
        try:
            py_compile.compile(str(Path(td) / 'cts' / f), doraise=True); out[f] = 'compiles'
        except py_compile.PyCompileError as e:
            out[f] = str(e)[-300:]
ok = all(v in ('applied', 'compiles') for v in out.values())
print(json.dumps({'ok': ok, **out}, indent=1))
sys.exit(0 if ok else 1)
