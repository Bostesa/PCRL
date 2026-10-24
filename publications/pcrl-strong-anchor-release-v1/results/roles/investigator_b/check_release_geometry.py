"""Label-free design check (investigator B): geometry of the frozen U0 and NEURAL_RAW release arrays on MODEL_FIT and CALIBRATION
(features-derived outputs only; no label file, nothing from MODEL_SELECT).
  1. How much of the income logit difference is a function of the occupation logit differences (R^2 of linear and quadratic fits):
     if high, a marginally attributing pair critic pushes the income network for occupation-carried signal.
  2. How far each recipient moved from U0 along NEURAL_RAW's rho-1 path (E2, E5, E10), in units of U0's own spread of each coordinate.
Run from the worktree root with python3."""
import json
import sys
from pathlib import Path

import numpy as np

B = Path('<worktrees>/pcrl-acs-protected-tree-replication-v1/.private/acs_replication/runs')
ROLES = ('MODEL_FIT', 'CALIBRATION')


def coords(pi, po):
    li = np.log(pi[:, 1]) - np.log(pi[:, 0])
    lo = np.log(po[:, 1:]) - np.log(po[:, :1])
    return li, lo


def r2(y, F):
    F1 = np.column_stack([np.ones(len(y)), F])
    beta, *_ = np.linalg.lstsq(F1, y, rcond=None)
    res = y - F1 @ beta
    return float(1 - res.var() / y.var())


def quad(F):
    iu = np.triu_indices(F.shape[1])
    return np.column_stack([F, (F[:, :, None] * F[:, None, :])[:, iu[0], iu[1]]])


out = {}
for v in (0, 1):
    with np.load(B / f'STAGE_A_{v}' / 'STAGE_A_OUTPUTS.npz') as z:
        u0 = {r: (z[f'U0__{r}__income'], z[f'U0__{r}__occupation']) for r in ROLES}
    with np.load(B / f'NEURAL_RAW_{v}' / 'NEURAL_RAW_OUTPUTS.npz') as z:
        nr = {e: {r: (z[f'NEURAL_RAW_rho1.0_{e}__{r}__income'], z[f'NEURAL_RAW_rho1.0_{e}__{r}__occupation']) for r in ROLES}
              for e in ('E2', 'E5', 'E10')}
    rec = {}
    for r in ROLES:
        li0, lo0 = coords(*u0[r])
        rr = {'n': int(len(li0)), 'U0_r2_income_on_occ_linear': r2(li0, lo0), 'U0_r2_income_on_occ_quadratic': r2(li0, quad(lo0))}
        for e, d in nr.items():
            li, lo = coords(*d[r])
            rr[f'{e}_r2_income_on_occ_quadratic'] = r2(li, quad(lo))
            rr[f'{e}_move_income_sd_units'] = float(np.sqrt(np.mean((li - li0) ** 2)) / li0.std())
            rr[f'{e}_move_occupation_sd_units'] = float(np.sqrt(np.mean(((lo - lo0) / lo0.std(0)) ** 2)))
        rec[r] = rr
    out[f'V{v}'] = rec
print(json.dumps(out, indent=1))
