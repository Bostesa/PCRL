"""Designer B, campaign pcrl_input_and_pair_release_v1: LABEL-FREE plug-in spend of CSF on the registered pi grid (no readers).

Same inputs and plug-in spend as feasibility.py (whose FEASIBILITY.json used the exploratory grid 0.10-0.40); writes
SPEND_GRID.json.
    python3 results/pcrl_input_and_pair_release_v1/roles/designer_b/spend_grid.py
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
import mech_sketch as MS  # noqa: E402

S3 = {k: 1.5 * x for k, x in FE.BUDGET_S2.items()}


def main():
    t0 = time.process_time()
    out = {'schema': 'IPR-designer-B-spend-grid-v1', 'label_free': True, 'grid': list(MS.STRENGTHS), 'versions': {}}
    for v in (0, 1):
        d = FE.load(v)
        st = MS.fit(d['Z'], None, None, None, None, v, 0.0, twin_inc=d['p_inc'], twin_occ=d['p_occ'], mode='coord')
        rows = {}
        for pi in MS.STRENGTHS:
            s = FE.spend(d, *MS.release_from_twin(dict(st, pi=pi), d['Z'], d['p_inc'], d['p_occ']))
            s['s3_ratio'] = max(s[k] / x for k, x in S3.items())
            s['binding_s2'] = max(FE.BUDGET_S2, key=lambda k: s[k] / FE.BUDGET_S2[k])
            rows[f'{pi:.3f}'] = s
            print(f'V{v} pi {pi:.3f}: KL inc {s["kl_inc"]:.4f} occ {s["kl_occ"]:.4f} men {s["men_occ_brier"]:.4f} '
                  f'S2x {s["s2_ratio"]:.2f} S3x {s["s3_ratio"]:.2f} binding {s["binding_s2"]}')
        out['versions'][str(v)] = rows
    out['cpu_seconds'] = time.process_time() - t0
    print(f'cpu_seconds {out["cpu_seconds"]:.1f}')
    (HERE / 'SPEND_GRID.json').write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
