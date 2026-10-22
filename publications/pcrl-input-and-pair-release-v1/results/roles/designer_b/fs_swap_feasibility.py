"""Designer B, campaign pcrl_input_and_pair_release_v1: LABEL-FREE plug-in feasibility of FS_SWAP (FS_SWAP_PROPOSAL.md).

Same inputs, plug-in world, readers and spend as feasibility.py (imported). The donor bank is the 20,000 MODEL_FIT people with their
out-of-fold twin vectors (the registered nested selection uses the other folds' people; the deployed release uses the deployed twin
on the MODEL_FIT records). S3 ratios use the protocol's S3 components only (occupation / income NLL 0.012 / 0.006, men's / women's
occupation Brier 0.0045, accuracy 0.008 / 0.006); the summed occupation Brier (a G1 clause, cap 0.005, not an S3 component) is
reported beside. Writes FS_SWAP_FEASIBILITY.json.
    python3 results/pcrl_input_and_pair_release_v1/roles/designer_b/fs_swap_feasibility.py
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
import fs_swap_sketch as SW  # noqa: E402
import mech_sketch as MS  # noqa: E402

S3 = {'kl_occ': 0.012, 'kl_inc': 0.006, 'men_occ_brier': 0.0045, 'women_occ_brier': 0.0045, 'acc_occ': 0.008, 'acc_inc': 0.006}


def s3(sp):
    r = {k: sp[k] / x for k, x in S3.items()}
    sp['s3_ratio'] = max(r.values())
    sp['binding_s3'] = max(r, key=r.get)
    return sp


def main():
    t0 = time.process_time()
    from ddr import ctlkit as CK
    out = {'schema': 'IPR-designer-B-fs-swap-feasibility-v1', 'label_free': True, 'grid': list(SW.STRENGTHS), 'versions': {}}
    for v in (0, 1):
        d = FE.load(v)
        Z, pI, pO = d['Z'], d['p_inc'], d['p_occ']
        splits = FE.make_splits(len(Z), d['q'], 2, seed=[2026, 10, 22, v])
        st = {m: SW.fit(Z, None, None, None, None, v, 0.0, twin_inc=pI, twin_occ=pO, mode=m) for m in ('swap', 'swap_joint', 'occ_flat')}
        rel = lambda m, lam: SW.release_from_twin(dict(st[m], lam=lam), Z, pI, pO)
        ev = lambda R: FE.evaluate(d, *R, splits)
        rows, grid = {'TWIN': ev((pI, pO))}, {}
        for lam in SW.STRENGTHS:                                           # spend only
            grid[f'{lam:.3f}'] = s3(FE.spend(d, *rel('swap', lam)))
        # per-person cost of one swapped person vs CSF's collapse (everyone treated)
        sp_all = FE.spend(d, *rel('swap', 1.0))
        spj_all = FE.spend(d, *rel('swap_joint', 1.0))
        stc = MS.fit(Z, None, None, None, None, v, 0.0, twin_inc=pI, twin_occ=pO, mode='coord')
        csf_all = FE.spend(d, *MS.release_from_twin(dict(stc, pi=1.0), Z, pI, pO))
        per_person = {'swap_kl_occ': sp_all['kl_occ'], 'swap_joint_kl_occ': spj_all['kl_occ'],
                      'collapse_kl_occ': csf_all['kl_occ'], 'collapse_kl_inc': csf_all['kl_inc'],
                      'swap_men_brier': sp_all['men_occ_brier'], 'collapse_men_brier': csf_all['men_occ_brier'],
                      'swap_brier_occ': sp_all['brier_occ'], 'collapse_brier_occ': csf_all['brier_occ']}
        rows['SWAP_ALL'] = ev(rel('swap', 1.0))
        for lam in (0.10, 0.20):
            rows[f'SWAP_{lam:.2f}'] = r = ev(rel('swap', lam))
            s3(r['spend'])
            tgt = r['spend']['kl_total']
            delta = FE.bisect(lambda x: FE.spend(d, *CK.dq_release(CK.dq_fit(pI, pO), pI, pO, Z, x, f'IPR_DQ_S{v}'))['kl_total'],
                              tgt, 0.0, 6.0, it=24)
            rows[f'DQ_m{lam:.2f}'] = ev(CK.dq_release(CK.dq_fit(pI, pO), pI, pO, Z, delta, f'IPR_DQ_S{v}'))
            lf = FE.bisect(lambda x: FE.spend(d, *rel('occ_flat', x))['kl_total'], tgt, 0.0, 1.0, it=24)
            rows[f'OCC_FLAT_m{lam:.2f}'] = ev(rel('occ_flat', lf))
            rows[f'OCC_FLAT_m{lam:.2f}']['lam'] = lf
            for k in (f'DQ_m{lam:.2f}', f'OCC_FLAT_m{lam:.2f}'):
                s3(rows[k]['spend'])
        rows['SWAP_J_0.20'] = ev(rel('swap_joint', 0.20))
        s3(rows['SWAP_J_0.20']['spend'])
        tgt = rows['SWAP_0.20']['spend']['kl_total']
        pc = FE.bisect(lambda x: FE.spend(d, *MS.release_from_twin(dict(stc, pi=x), Z, pI, pO))['kl_total'], tgt, 0.0, 1.0, it=24)
        rows['CSF_m0.20'] = ev(MS.release_from_twin(dict(stc, pi=pc), Z, pI, pO))
        rows['CSF_m0.20']['pi'] = pc
        s3(rows['CSF_m0.20']['spend'])
        out['versions'][str(v)] = {'per_person': per_person, 'spend_grid_swap': grid, 'rows': rows}
        print(f'--- V{v} per-person (everyone treated): {json.dumps({k: round(x, 4) for k, x in per_person.items()})}')
        for name, row in rows.items():
            A, S = row['auc'], row['spend']
            print(f'V{v} {name:14s} pair {A["pair"]:.4f} inc {A["income"]:.4f} occ {A["occupation"]:.4f} | KL inc {S["kl_inc"]:.4f} '
                  f'occ {S["kl_occ"]:.4f} BrO {S["brier_occ"]:.4f} men {S["men_occ_brier"]:.4f} wom {S["women_occ_brier"]:.4f} '
                  f'S3x {S.get("s3_ratio", float("nan")):.2f} {S.get("binding_s3", "")}', flush=True)
        print(f'V{v} swap S3 grid:', {k: (round(g['s3_ratio'], 2), g['binding_s3'], round(g['brier_occ'], 4)) for k, g in grid.items()})
    out['cpu_seconds'] = time.process_time() - t0
    print(f'cpu_seconds {out["cpu_seconds"]:.1f}')
    (HERE / 'FS_SWAP_FEASIBILITY.json').write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
