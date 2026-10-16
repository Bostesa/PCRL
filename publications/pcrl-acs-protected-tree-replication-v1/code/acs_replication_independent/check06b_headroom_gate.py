"""Independent add-on to check 6: Stage A teacher-usefulness gate on MODEL_SELECT (own metrics).

Gate (SCIENCE_LOCK.models.T_TREE.headroom_gate): for both tasks and both seeds, T_TREE has positive NLL gain and positive accuracy gain
over constant prediction on MODEL_SELECT, and MODEL_FIT task-class support (>= 100 per class) passes. Constant prediction = the MODEL_SELECT
class prior (as in the lead's summary); a MODEL_FIT-prior constant is reported too. Headroom vs U0 / Ucal recomputed. Aggregates only."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRIV = ROOT / '.private' / 'acs_replication'
RES = ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
OUT = RES / 'independent'
K = {'income': 2, 'occupation': 5}


def lab(role, t):
    with np.load(PRIV / 'exports' / role / f'{t}.npz') as z:
        return z[t].copy()


def m(y, p):
    n = len(y); oh = np.zeros_like(p); oh[np.arange(n), y] = 1
    return {'accuracy': float(np.mean(p.argmax(1) == y)), 'nll': float(np.mean(-np.log(np.maximum(p[np.arange(n), y], 1e-12)))),
            'brier': float(np.mean(((p - oh) ** 2).sum(1)))}


def main():
    lead = json.loads((RES / 'STAGE_A_HEADROOM.json').read_text())
    res = {'id': '6b', 'name': 'Stage A headroom gate (add-on to TEACHER IDENTITY)', 'per_seed_task': {}}
    ok = True; useful_all = True; maxdiff = 0.0
    support = {}
    for t in K:
        yf = lab('MODEL_FIT', t)
        support[t] = np.bincount(yf, minlength=K[t]).tolist()
    support_ok = all(min(v) >= 100 for v in support.values())
    for seed in (0, 1):
        with np.load(PRIV / 'runs' / f'STAGE_A_{seed}' / 'STAGE_A_OUTPUTS.npz') as z:
            A = {k: z[k].copy() for k in z.files}
        for t in K:
            y = lab('MODEL_SELECT', t); n = len(y)
            prior_sel = np.bincount(y, minlength=K[t]) / n
            prior_fit = np.bincount(lab('MODEL_FIT', t), minlength=K[t]) / len(lab('MODEL_FIT', t))
            mt = m(y, A[f'T_TREE__MODEL_SELECT__{t}']); mu = m(y, A[f'U0__MODEL_SELECT__{t}']); mc = m(y, A[f'Ucal__MODEL_SELECT__{t}'])
            cs = m(y, np.repeat(prior_sel[None], n, 0)); cf = m(y, np.repeat(prior_fit[None], n, 0))
            rec = {'nll_gain_over_constant': cs['nll'] - mt['nll'], 'accuracy_gain_over_constant': mt['accuracy'] - cs['accuracy'],
                   'nll_gain_over_MODEL_FIT_prior_constant': cf['nll'] - mt['nll'], 'accuracy_gain_over_MODEL_FIT_prior_constant': mt['accuracy'] - cf['accuracy'],
                   'headroom_vs_U0': {k: (mu[k] - mt[k]) if k != 'accuracy' else (mt[k] - mu[k]) for k in mt},
                   'headroom_vs_Ucal': {k: (mc[k] - mt[k]) if k != 'accuracy' else (mt[k] - mc[k]) for k in mt}}
            useful = rec['nll_gain_over_constant'] > 0 and rec['accuracy_gain_over_constant'] > 0
            useful_all &= useful
            ls = lead['per_seed'][str(seed)][t]
            for k in ('accuracy', 'nll', 'brier'):
                maxdiff = max(maxdiff, abs(rec['headroom_vs_U0'][k] - ls['headroom_vs_U0'][k]), abs(rec['headroom_vs_Ucal'][k] - ls['headroom_vs_Ucal'][k]))
            maxdiff = max(maxdiff, abs(rec['nll_gain_over_constant'] - ls['summaries']['T_TREE']['nll_gain_over_constant']),
                          abs(rec['accuracy_gain_over_constant'] - ls['summaries']['T_TREE']['accuracy_gain_over_constant']))
            res['per_seed_task'][f's{seed}|{t}'] = rec
    decision = 'SUPPORTED_CONTINUE' if useful_all and support_ok else 'UNSUPPORTED'
    ok = decision == lead['gate']['decision'] and maxdiff <= 1e-12
    res.update({'MODEL_FIT_class_support': support, 'support_ok': support_ok, 'teacher_useful_all': useful_all, 'decision_recomputed': decision,
                'decision_lead': lead['gate']['decision'], 'max_abs_diff_vs_lead': maxdiff, 'status': 'PASS' if ok else 'FAIL'})
    (OUT / 'check06b_headroom_gate.json').write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
