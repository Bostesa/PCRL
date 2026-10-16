"""Independent check 6: teacher identity and released wires (own routing / softmax code; no acs_replication import).

For both seeds: parse TREE_TEACHER_<task>.json, route every development role's exported features through the oblivious
trees (bit i = float32 x[feature] > float32 border), rebuild T_TREE probabilities and compare with STAGE_A outputs; rebuild
LOCAL_TREE / TASK_REFIT / JOINT_TREE probabilities from the E20 theta files (raw = teacher raw + theta[leaf].sum / T;
income softmax(-d/2, d/2); occupation softmax(centred raw)) and compare with STAGE_B_OUTPUTS_E20; confirm theta = 0
reproduces the teacher; confirm released decisions = first-index argmax and count exact ties.
Only aggregate differences are written.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRIV = ROOT / '.private' / 'acs_replication'
OUT = ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1' / 'independent'
ROLES = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
TASKS = ('income', 'occupation')
TOL = 1e-12


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_features(role):
    with np.load(PRIV / 'exports' / role / 'features.npz', allow_pickle=False) as z:
        return z['X'].astype(np.float32, copy=False), [str(s) for s in z['feature_names']], z['row_id'].copy()


class Teacher:
    def __init__(self, path):
        J = json.loads(Path(path).read_text())
        self.task = J['task']
        self.feat = np.asarray(J['split_features'], dtype=np.int64)
        self.bord = np.asarray(J['split_borders'], dtype=np.float32)
        self.leaf = np.asarray(J['leaf_values'], dtype=np.float64)
        self.T, self.depth = self.feat.shape
        self.L = 2 ** self.depth
        self.dim = self.leaf.shape[2]
        self.scale = float(J['scale'])
        self.bias = np.asarray(J['bias'], dtype=np.float64).reshape(-1)
        self.names = list(J['feature_names'])

    def leaves(self, X):
        idx = np.zeros((X.shape[0], self.T), dtype=np.int64)
        for t in range(self.T):            # explicit per-tree loop (different code path from the lead's vectorised routing)
            code = np.zeros(X.shape[0], dtype=np.int64)
            for i in range(self.depth):
                code += (X[:, self.feat[t, i]] > self.bord[t, i]).astype(np.int64) * (1 << i)
            idx[:, t] = code
        return idx

    def raw(self, idx, theta=None):
        acc = np.zeros((idx.shape[0], self.dim), dtype=np.float64)
        for t in range(self.T):
            acc += self.leaf[t][idx[:, t]]
        out = self.scale * acc + self.bias[None, :]
        if theta is not None:
            th = np.asarray(theta, dtype=np.float64).reshape(self.T, self.L, self.dim)
            corr = np.zeros_like(acc)
            for t in range(self.T):
                corr += th[t][idx[:, t]]
            out = out + corr / self.T
        return out


def probs(task, raw):
    if task == 'income':
        d = raw[:, 0]
        # softmax(-d/2, d/2) = (sigmoid(-d), sigmoid(d)), computed stably
        p1 = np.where(d >= 0, 1.0 / (1.0 + np.exp(-np.abs(d))), np.exp(-np.abs(d)) / (1.0 + np.exp(-np.abs(d))))
        return np.column_stack((1.0 - p1, p1))
    z = raw - raw.mean(axis=1, keepdims=True)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    feats = {r: load_features(r) for r in ROLES}
    names0 = feats['MODEL_FIT'][1]
    ev = {'per_seed': {}}
    ok = True
    for seed in (0, 1):
        sa = PRIV / 'runs' / f'STAGE_A_{seed}'
        sb = PRIV / 'runs' / f'STAGE_B_{seed}'
        rec = json.loads((sb / 'STAGE_B_SEED.json').read_text())
        teachers = {t: Teacher(sa / f'TREE_TEACHER_{t}.json') for t in TASKS}
        with np.load(sa / 'STAGE_A_OUTPUTS.npz') as z:
            A = {k: z[k].copy() for k in z.files}
        with np.load(sb / 'STAGE_B_OUTPUTS_E20.npz') as z:
            Bout = {k: z[k].copy() for k in z.files}
        srec = {'teacher_trees': {t: int(teachers[t].T) for t in TASKS}, 'feature_names_match_export': all(teachers[t].names == names0 for t in TASKS),
                'T_TREE': {}, 'zero_theta': {}, 'E20': {}, 'theta_file_sha256_match': {}, 'decision_ties': {}}
        ok &= srec['feature_names_match_export']
        thetas = {}
        for svc in ('LOCAL_TREE', 'TASK_REFIT', 'JOINT_TREE'):
            ck = rec['checkpoints'].get(svc, {}).get('E20')
            if not ck:
                continue
            p = sb / ck['file']
            srec['theta_file_sha256_match'][svc] = sha(p) == ck['sha256']
            ok &= srec['theta_file_sha256_match'][svc]
            with np.load(p) as z:
                thetas[svc] = {'income': z['theta_income'].copy(), 'occupation': z['theta_occupation'].copy()}
        for role in ROLES:
            X, names, row_id = feats[role]
            if not np.array_equal(A['row_id__' + role], row_id):
                srec.setdefault('row_alignment_fail', []).append(role); ok = False
            for t in TASKS:
                te = teachers[t]
                idx = te.leaves(X)
                pt = probs(t, te.raw(idx))
                ref = A[f'T_TREE__{role}__{t}']
                d = float(np.max(np.abs(pt - ref)))
                dec_eq = bool(np.array_equal(pt.argmax(1), ref.argmax(1)))
                srec['T_TREE'][f'{t}|{role}'] = {'max_abs_prob_diff': d, 'decisions_identical': dec_eq}
                ok &= (d <= TOL) and dec_eq
                pz = probs(t, te.raw(idx, np.zeros((te.T * te.L, te.dim))))
                dz = float(np.max(np.abs(pz - ref)))
                srec['zero_theta'][f'{t}|{role}'] = {'max_abs_prob_diff_vs_stored_T_TREE': dz, 'decisions_identical': bool(np.array_equal(pz.argmax(1), ref.argmax(1)))}
                ok &= dz <= TOL
                for svc, th in thetas.items():
                    key = f'{svc}__{role}__{t}'
                    if key not in Bout:
                        continue
                    pc = probs(t, te.raw(idx, th[t]))
                    refb = Bout[key]
                    db = float(np.max(np.abs(pc - refb)))
                    deq = bool(np.array_equal(pc.argmax(1), refb.argmax(1)))
                    srec['E20'][f'{svc}|{t}|{role}'] = {'max_abs_prob_diff': db, 'decisions_identical': deq,
                                                        'max_abs_diff_from_teacher': float(np.max(np.abs(refb - ref)))}
                    ok &= (db <= TOL) and deq
                    # released decisions = argmax; count exact top-1 ties (first index wins)
                    srt = np.sort(refb, axis=1)
                    srec['decision_ties'][key] = int(np.sum(srt[:, -1] == srt[:, -2]))
                    ok &= bool(np.allclose(refb.sum(1), 1.0, atol=1e-9)) and bool((refb >= 0).all())
        srec['max_T_TREE_diff'] = max(v['max_abs_prob_diff'] for v in srec['T_TREE'].values())
        srec['max_E20_diff'] = max(v['max_abs_prob_diff'] for v in srec['E20'].values())
        srec['max_zero_theta_diff'] = max(v['max_abs_prob_diff_vs_stored_T_TREE'] for v in srec['zero_theta'].values())
        srec['total_decision_ties_E20'] = int(sum(srec['decision_ties'].values()))
        srec['lead_zero_identity_record_all_pass'] = all(v['max_abs_prob_diff'] <= 1e-12 and v['decisions_identical'] for v in rec['zero_identity'].values())
        ev['per_seed'][str(seed)] = srec
    res = {'id': 6, 'name': 'TEACHER IDENTITY + WIRES', 'status': 'PASS' if ok else 'FAIL', 'tolerance_prob': TOL,
           'evidence': {s: {k: v for k, v in r.items() if k in ('teacher_trees', 'feature_names_match_export', 'theta_file_sha256_match', 'max_T_TREE_diff',
                                                                'max_E20_diff', 'max_zero_theta_diff', 'total_decision_ties_E20', 'lead_zero_identity_record_all_pass', 'row_alignment_fail')}
                        for s, r in ev['per_seed'].items()},
           'detail': ev,
           'notes': 'Own per-tree routing loop (float32 x > float32 border, bit i << i), own leaf-sum and softmax; compared on all 6 development roles '
                    '(features only; SEALED has no export). Decisions are not stored separately: the released decision is argmax of the stored probabilities; '
                    'exact top-1 ties are counted (first index wins under numpy argmax).'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'check06_teacher_wires.json').write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(res['evidence'], indent=1), res['status'])


if __name__ == '__main__':
    sys.exit(main())
