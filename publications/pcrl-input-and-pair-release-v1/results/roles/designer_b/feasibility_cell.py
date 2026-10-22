"""Designer B, campaign pcrl_input_and_pair_release_v1: LABEL-FREE plug-in check of the registered candidate CSF-T (mode 'cell').

Same plug-in world and readers as feasibility.py. The release-level propensity of CSF-T is fitted on PSEUDO-SEX ~ Bern(q) (one
fixed draw; the registered fit uses the MODEL_FIT SEX labels), with household folds from the admitted MODEL_FIT units. Compared at
equal flattening spend with uniform CSF (mode 'coord') and with DQ matched on total plug-in KL. Writes FEASIBILITY_CELL.json.
    python3 results/pcrl_input_and_pair_release_v1/roles/designer_b/feasibility_cell.py
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import feasibility as FE  # noqa: E402
import mech_sketch as MS  # noqa: E402


def main():
    t0 = time.process_time()
    from ddr import common as C
    from ddr import ctlkit as CK
    unit = np.asarray(C.features('MODEL_FIT')['unit'])
    out = {'schema': 'IPR-designer-B-feasibility-cell-v1', 'label_free': True, 'versions': {}}
    for v in (0, 1):
        d = FE.load(v)
        Z, pI, pO, q = d['Z'], d['p_inc'], d['p_occ'], d['q']
        splits = FE.make_splits(len(q), q, 2, seed=[2026, 10, 22, v])
        s_pseudo = (np.random.default_rng([2026, 10, 22, v, 5]).random(len(q)) < q).astype(int)
        rows = {}
        st = MS.fit(Z, None, None, s_pseudo, unit, v, 0.10, twin_inc=pI, twin_occ=pO, mode='cell')
        z = MS.release_propensity(np.column_stack([MS.clog(pI), MS.clog(pO)]), s_pseudo, unit, v)
        kI = np.sum(pI * (np.log(pI) - np.log(st['T_inc'][pI.argmax(1)])), 1)
        kO = np.sum(pO * (np.log(pO) - np.log(st['T_occ'][pO.argmax(1)])), 1)
        cell = pI.argmax(1) * 5 + pO.argmax(1)
        basis = CK.dq_fit(pI, pO)
        for pi in (0.10, 0.15):
            cost = kI / MS.COST_NORM[0] + kO / MS.COST_NORM[1]
            stc = dict(st, pi=pi, rate=MS.waterfill(cell, z, cost, pi * float(np.sum(cost))))
            rows[f'CSF_T_{pi:.2f}'] = r = FE.evaluate(d, *MS.release_from_twin(stc, Z, pI, pO), splits)
            r['rate'] = {str(k): round(x, 3) for k, x in stc['rate'].items()}
            rows[f'CSF_{pi:.2f}'] = FE.evaluate(d, *MS.release_from_twin(dict(st, mode='coord', pi=pi), Z, pI, pO), splits)
            tgt = r['spend']['kl_total']
            delta = FE.bisect(lambda x: FE.spend(d, *CK.dq_release(basis, pI, pO, Z, x, f'IPR_B_FEAS_DQ|V{v}'))['kl_total'],
                              tgt, 0.0, 6.0)
            rows[f'DQ_m{pi:.2f}'] = FE.evaluate(d, *CK.dq_release(basis, pI, pO, Z, delta, f'IPR_B_FEAS_DQ|V{v}'), splits)
        out['versions'][str(v)] = {'z_auc_soft': FE.auc_soft(z, q), 'rows': rows}
        for name, row in rows.items():
            A, S = row['auc'], row['spend']
            print(f'V{v} {name:10s} pair {A["pair"]:.4f} inc {A["income"]:.4f} occ {A["occupation"]:.4f} | KL {S["kl_total"]:.4f} '
                  f'men {S["men_occ_brier"]:.4f} S2x {S["s2_ratio"]:.2f}', flush=True)
        print(f'V{v} release-level propensity soft AUC {out["versions"][str(v)]["z_auc_soft"]:.4f}; CSF_T_0.10 rates',
              rows['CSF_T_0.10']['rate'])
    out['cpu_seconds'] = time.process_time() - t0
    print(f'cpu_seconds {out["cpu_seconds"]:.1f}')
    (HERE / 'FEASIBILITY_CELL.json').write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
