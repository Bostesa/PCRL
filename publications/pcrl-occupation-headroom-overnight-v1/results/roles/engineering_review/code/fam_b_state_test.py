"""Engineering review (label-free): ohn.fam_b.predict does not check STATE.json against OCC.json's state_sha256, so a modified
STATE.json changes the release silently when B is reached outside a package that hashes it (e.g. as the witness inside an INC or
PROT package under the current ohn.deploy manifest). Demonstrated on COPIES of B2_SINGLE seed 0 (STATE.json, OCC.json and the
TREE_V2 base learner files) under .private/<campaign>/runs/DEPLOY_REVIEW/ (deleted at the end); the release directories are never
written. The proposed 4-line check (proposed_ohn_fam_b.patch) is loaded from a file and must reproduce the untouched release
bitwise and refuse the modified copy.

    python3 results/pcrl_occupation_headroom_overnight_v1/roles/engineering_review/code/fam_b_state_test.py PATCHED_FAM_B.py OUT.json
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ohn import common as C  # noqa: E402
from ohn import devrun as D  # noqa: E402
from ohn import fam_b as B  # noqa: E402


def main(patched_file, out):
    spec = importlib.util.spec_from_file_location('ohn._review_fam_b_patched', patched_file)
    P = importlib.util.module_from_spec(spec); sys.modules[spec.name] = P; spec.loader.exec_module(P)
    root = C.RUNS / 'DEPLOY_REVIEW' / 'tamper_root_B'
    if root.exists():
        shutil.rmtree(root)
    src_cfg, src_base = D.occ_dir('B', 'B2_SINGLE', 0), C.RUNS / 'B' / 'BASES' / 'TREE_V2' / 'S0'
    (root / 'B2_SINGLE' / 'S0').mkdir(parents=True, mode=0o700); (root / 'BASES' / 'TREE_V2' / 'S0').mkdir(parents=True, mode=0o700)
    for f in ('STATE.json', 'OCC.json'):
        shutil.copy2(src_cfg / f, root / 'B2_SINGLE' / 'S0' / f)
    for f in ('BASE.json', 'MODEL.pkl'):
        shutil.copy2(src_base / f, root / 'BASES' / 'TREE_V2' / 'S0' / f)
    try:
        F = C.load_role('DEV_REPORT', ('features',))
        ref = D.load_occupation('B', 'B2_SINGLE', 0, roles=('DEV_REPORT',))[0]['DEV_REPORT']
        res = {'untouched_copy': {'current_bitwise': bool(np.array_equal(B.predict(0, 'B2_SINGLE', F['X'], F['feature_names'], root=root), ref)),
                                  'patched_bitwise': bool(np.array_equal(P.predict(0, 'B2_SINGLE', F['X'], F['feature_names'], root=root), ref))}}
        sp = root / 'B2_SINGLE' / 'S0' / 'STATE.json'
        st = json.loads(sp.read_text())
        st['params']['tau'] = float(st['params']['tau']) * 1.5            # a silent change of the deployed temperature
        sp.write_text(json.dumps(st, indent=1, sort_keys=True) + '\n')
        p1 = B.predict(0, 'B2_SINGLE', F['X'], F['feature_names'], root=root)
        res['modified_copy'] = {'current_accepted': True, 'current_max_abs_diff_vs_release': float(np.max(np.abs(p1 - ref))),
                                'current_decisions_changed_share': float(np.mean(p1.argmax(1) != ref.argmax(1)))}
        try:
            P.predict(0, 'B2_SINGLE', F['X'], F['feature_names'], root=root)
            res['modified_copy']['patched'] = 'NOT_REFUSED'
        except PermissionError as e:
            res['modified_copy']['patched'] = f'refused: PermissionError ({e})'
        res['ok'] = bool(res['untouched_copy']['current_bitwise'] and res['untouched_copy']['patched_bitwise']
                         and res['modified_copy']['patched'].startswith('refused'))
        res['patched_file_sha256'] = C.sha(patched_file)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    Path(out).write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    print(json.dumps(res))
    return 0 if res['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
