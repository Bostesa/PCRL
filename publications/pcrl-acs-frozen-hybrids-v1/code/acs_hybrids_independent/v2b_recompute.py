"""V2 supplement: content-level row pairing of every bank service (label-free REPLAY of the frozen models).

The STAGE_B (LOCAL_TREE, JOINT_TREE, TASK_REFIT) and NEURAL_RAW source output files carry no row ids of their own, so
row-id equality (V2) cannot by itself show that their rows are in export order. Here every bank array, all services and
references in all four roles and both seeds, is recomputed from its frozen model file (hash-checked in V1) applied to the
admitted export features, and compared with the bank (bitwise count, max abs diff, decisions). Model evaluation uses the
source's own model definitions (acs_replication.trees / acs_replication.neural); comparison and orchestration are the
verifier's. Writes results/.../independent/V2b_comparator_row_alignment_recompute.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from acs_hybrids import common as C   # hash-checked data loading only

from . import vcommon as V


def main():
    import torch
    from acs_replication import neural, trees
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(V.OUT / 'V2b_comparator_row_alignment_recompute.json'))
    a = ap.parse_args()
    tl = V.jload(V.SRC_RES / 'TRAINING_COMPLETE_LOCK.json')['frozen_files_sha256']
    comps = V.components()
    roles = {r: C.load_role(r, ('features',)) for r in V.ROLES}
    names = roles['DEV_REPORT']['feature_names']
    res = {'check': 'V2 supplement: comparator row alignment by recompute from frozen models',
           'scope': 'REPLAY of frozen defense models on admitted export features (no fitting); label-free', 'per_seed': {}}
    ok = True
    for s in V.SEEDS:
        files = {'teacher_income': f'STAGE_A_{s}/TREE_TEACHER_income.json', 'teacher_occupation': f'STAGE_A_{s}/TREE_TEACHER_occupation.json',
                 'LOCAL_TREE': f'STAGE_B_{s}/LOCAL_TREE__E20.npz', 'JOINT_TREE': f'STAGE_B_{s}/JOINT_TREE__E20.npz',
                 'TASK_REFIT': f'STAGE_B_{s}/TASK_REFIT__E20.npz', 'NEURAL_RAW': f'NEURAL_RAW_{s}/NEURAL_RAW__rho1.0__E2.pt',
                 'U0': f'STAGE_A_{s}/U0_MODEL.pt'}
        hash_ok = {k: V.sha256_file(V.SRCP / 'runs' / f) == tl.get(f) for k, f in files.items()}
        teachers = {t: trees.TreeTeacher(json.loads((V.SRCP / 'runs' / files[f'teacher_{t}']).read_text())) for t in V.TASKS}
        thetas = {}
        for lab in ('LOCAL_TREE', 'JOINT_TREE', 'TASK_REFIT'):
            with np.load(V.SRCP / 'runs' / files[lab]) as z:
                thetas[lab] = {t: z[f'theta_{t}'].copy() for t in V.TASKS}
        nstate = torch.load(V.SRCP / 'runs' / files['NEURAL_RAW'], map_location='cpu', weights_only=True)
        ustate = torch.load(V.SRCP / 'runs' / files['U0'], map_location='cpu', weights_only=True)
        bank, ids = C.load_bank(s)
        per = {}
        for r in V.ROLES:
            X = roles[r]['X']
            if not np.array_equal(ids[f'row_id__{r}'], roles[r]['row_id']):
                raise SystemExit('bank rows differ from export')
            for t in V.TASKS:
                k = V.NCLASS[t]
                leaves = teachers[t].route(X, names)
                comp = {'T_TREE': trees.release(t, teachers[t].raw_from_leaves(leaves, None))['probability']}
                for lab in thetas:
                    comp[lab] = trees.release(t, teachers[t].raw_from_leaves(leaves, thetas[lab][t]))['probability']
                m = neural.network(k, X.shape[1]); m.load_state_dict(nstate[t]); m.eval()
                comp['NEURAL_RAW_rho1.0_E2'] = neural.probabilities(m, X)
                u = neural.network(k, X.shape[1]); u.load_state_dict(ustate[t]); u.eval()
                comp['U0'] = neural.probabilities(u, X)
                comp['Ucal'] = neural.tempered_probabilities(neural.logits(u, X), float(ustate['temperatures'][t]))
                for svc in sorted(comps):
                    lab = comps[svc][V.TASKS.index(t)]
                    p, b = np.asarray(comp[lab], dtype=np.float64), bank[svc][r][t]
                    per[f'{svc}|{r}|{t}'] = {'source_model': lab, 'bitwise': bool(np.array_equal(p, b)), 'max_abs_diff': V.maxabs(p, b),
                                             'decisions_equal': bool(np.array_equal(p.argmax(1), b.argmax(1)))}
        worst = max(v['max_abs_diff'] for v in per.values())
        by_model = {}
        for v in per.values():
            d = by_model.setdefault(v['source_model'], {'arrays': 0, 'bitwise': 0, 'max_abs_diff': 0.0, 'decisions_equal': 0})
            d['arrays'] += 1
            d['bitwise'] += int(v['bitwise'])
            d['max_abs_diff'] = max(d['max_abs_diff'], v['max_abs_diff'])
            d['decisions_equal'] += int(v['decisions_equal'])
        rec = {'model_file_hashes_equal_TRAINING_COMPLETE_LOCK': hash_ok, 'arrays': len(per), 'bitwise': sum(v['bitwise'] for v in per.values()),
               'max_abs_diff': worst, 'all_decisions_equal': all(v['decisions_equal'] for v in per.values()), 'by_source_model': by_model,
               'tolerance': 1e-12}
        rec['pass'] = bool(all(hash_ok.values()) and worst <= 1e-12 and rec['all_decisions_equal'])
        ok &= rec['pass']
        res['per_seed'][f'S{s}'] = rec
    res['status'] = 'PASS' if ok else 'FAIL'
    V.write_json(a.out, res)
    print({'status': res['status'], **{k: (v['bitwise'], v['arrays'], v['max_abs_diff']) for k, v in res['per_seed'].items()}})


if __name__ == '__main__':
    main()
