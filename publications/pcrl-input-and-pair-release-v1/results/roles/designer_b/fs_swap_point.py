"""Designer B, campaign pcrl_input_and_pair_release_v1: LABEL-FREE plug-in reading of FS_SWAP at its plug-in S3 point (lambda 0.05),
4 reader draws, with DQ, occupation-only flattening (OCC_FLAT) and CSF matched on total plug-in KL. Same world as feasibility.py.
Writes FS_SWAP_POINT.json.
    python3 results/pcrl_input_and_pair_release_v1/roles/designer_b/fs_swap_point.py
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import feasibility as FE  # noqa: E402
import fs_swap_feasibility as FF  # noqa: E402
import fs_swap_sketch as SW  # noqa: E402
import mech_sketch as MS  # noqa: E402

LAM = 0.05


def main():
    t0 = time.process_time()
    from ddr import ctlkit as CK
    out = {'schema': 'IPR-designer-B-fs-swap-point-v1', 'label_free': True, 'lam': LAM, 'draws': 4, 'versions': {}}
    for v in (0, 1):
        d = FE.load(v)
        Z, pI, pO = d['Z'], d['p_inc'], d['p_occ']
        splits = FE.make_splits(len(Z), d['q'], 4, seed=[2026, 10, 22, v, 4])
        sw = SW.fit(Z, None, None, None, None, v, LAM, twin_inc=pI, twin_occ=pO, mode='swap')
        sf = SW.fit(Z, None, None, None, None, v, 0.0, twin_inc=pI, twin_occ=pO, mode='occ_flat')
        sc = MS.fit(Z, None, None, None, None, v, 0.0, twin_inc=pI, twin_occ=pO, mode='coord')
        rows = {'TWIN': FE.evaluate(d, pI, pO, splits), 'SWAP': FE.evaluate(d, *SW.release_from_twin(sw, Z, pI, pO), splits)}
        tgt = rows['SWAP']['spend']['kl_total']
        basis = CK.dq_fit(pI, pO)
        delta = FE.bisect(lambda x: FE.spend(d, *CK.dq_release(basis, pI, pO, Z, x, f'IPR_DQ_S{v}'))['kl_total'], tgt, 0.0, 6.0, it=24)
        rows['DQ_m'] = FE.evaluate(d, *CK.dq_release(basis, pI, pO, Z, delta, f'IPR_DQ_S{v}'), splits)
        lf = FE.bisect(lambda x: FE.spend(d, *SW.release_from_twin(dict(sf, lam=x), Z, pI, pO))['kl_total'], tgt, 0.0, 1.0, it=24)
        rows['OCC_FLAT_m'] = FE.evaluate(d, *SW.release_from_twin(dict(sf, lam=lf), Z, pI, pO), splits)
        pc = FE.bisect(lambda x: FE.spend(d, *MS.release_from_twin(dict(sc, pi=x), Z, pI, pO))['kl_total'], tgt, 0.0, 1.0, it=24)
        rows['CSF_m'] = FE.evaluate(d, *MS.release_from_twin(dict(sc, pi=pc), Z, pI, pO), splits)
        for k, x in (('DQ_m', delta), ('OCC_FLAT_m', lf), ('CSF_m', pc)):
            rows[k]['strength'] = x
        for row in rows.values():
            FF.s3(row['spend'])
        out['versions'][str(v)] = rows
        for name, row in rows.items():
            A, S = row['auc'], row['spend']
            sd = {k: round(float(__import__('numpy').std(x)), 4) for k, x in row['auc_draws'].items()}
            print(f'V{v} {name:11s} pair {A["pair"]:.4f} inc {A["income"]:.4f} occ {A["occupation"]:.4f} (draw sd pair {sd["pair"]}) | '
                  f'KL inc {S["kl_inc"]:.4f} occ {S["kl_occ"]:.4f} BrO {S["brier_occ"]:.4f} men {S["men_occ_brier"]:.4f} '
                  f'S3x {S["s3_ratio"]:.2f} {S["binding_s3"]} strength {row.get("strength", "")}', flush=True)
    out['cpu_seconds'] = time.process_time() - t0
    print(f'cpu_seconds {out["cpu_seconds"]:.1f}')
    (HERE / 'FS_SWAP_POINT.json').write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
