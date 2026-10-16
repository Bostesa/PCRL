"""Independent add-on to check 7: the Ucal reference is one global temperature per task/seed applied to the centred U0 logits,
and that temperature minimises mean CALIBRATION NLL within [0.05, 20] (own bounded search). Aggregates only."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent.parent
PRIV = ROOT / '.private' / 'acs_replication'
OUT = ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1' / 'independent'
ROLES = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')


def softmax_t(z, t):
    a = z / t; a = a - a.max(1, keepdims=True); e = np.exp(a)
    return e / e.sum(1, keepdims=True)


def main():
    res = {'id': '7b', 'name': 'Ucal reference (temperature of U0) - add-on to NOMINATIONS', 'per_seed_task': {}}
    ok = True
    for seed in (0, 1):
        with np.load(PRIV / 'runs' / f'STAGE_A_{seed}' / 'STAGE_A_OUTPUTS.npz') as z:
            A = {k: z[k].copy() for k in z.files}
        for task in ('income', 'occupation'):
            with np.load(PRIV / 'exports' / 'CALIBRATION' / f'{task}.npz') as z:
                ycal = z[task].copy()
            # implied temperature from CALIBRATION rows: centred log Ucal = centred log U0 / T
            zc = np.log(A[f'U0__CALIBRATION__{task}']); zc = zc - zc.mean(1, keepdims=True)
            uc = np.log(A[f'Ucal__CALIBRATION__{task}']); uc = uc - uc.mean(1, keepdims=True)
            m = np.abs(uc) > 1e-6
            t_implied = float(np.median(zc[m] / uc[m]))
            max_dev = 0.0
            for role in ROLES:
                z0 = np.log(A[f'U0__{role}__{task}']); z0 = z0 - z0.mean(1, keepdims=True)
                max_dev = max(max_dev, float(np.max(np.abs(softmax_t(z0, t_implied) - A[f'Ucal__{role}__{task}']))))

            def nll(t):
                p = softmax_t(zc, t)
                return float(np.mean(-np.log(np.maximum(p[np.arange(len(ycal)), ycal], 1e-300))))
            r = minimize_scalar(nll, method='bounded', bounds=(0.05, 20.0), options={'xatol': 1e-8, 'maxiter': 1000})
            grid = np.linspace(max(0.05, t_implied * 0.5), min(20.0, t_implied * 1.5), 2001)
            tg = float(grid[np.argmin([nll(t) for t in grid])])
            rec = {'temperature_implied_by_released_Ucal': t_implied, 'max_abs_prob_dev_all_roles_single_T': max_dev,
                   'temperature_own_bounded_argmin': float(r.x), 'temperature_grid_argmin': tg,
                   'nll_at_implied': nll(t_implied), 'nll_at_own_argmin': nll(float(r.x)), 'nll_excess_implied_minus_own': nll(t_implied) - nll(float(r.x))}
            res['per_seed_task'][f's{seed}|{task}'] = rec
            # U0 logits are float32-derived; allow small tolerance on probability reconstruction and on the argmin (xatol 1e-6 in the lead's solver)
            ok &= max_dev <= 1e-6 and abs(t_implied - float(r.x)) <= 1e-4 and rec['nll_excess_implied_minus_own'] <= 1e-9
    res['status'] = 'PASS' if ok else 'FAIL'
    res['notes'] = ('Ucal outputs reproduced as softmax(centred log U0 / T) with one T per seed/task on all six roles; T matches an own bounded minimiser '
                    'of mean CALIBRATION NLL (CALIBRATION labels are the registered Ucal access).')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'check07b_ucal_reference.json').write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
