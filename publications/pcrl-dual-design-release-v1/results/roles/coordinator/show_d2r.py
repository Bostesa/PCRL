"""Print a D2R reading compactly (coordinator; reads claims/<reading>/D2R.json).

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/show_d2r.py DA_REG_D2R DB_REG_D2R
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
for name in sys.argv[1:]:
    d = json.loads((R / 'claims' / name / 'D2R.json').read_text())
    r = d['d2r']
    print(f"===== {name}: {r.get('label')} release={r.get('release')} refit={d['refit_effectiveness'].get('label')} "
          f"decisive={r.get('decisive')} counted_once={r.get('counted_once')}")
    for c, v in (r.get('per_comparator') or {}).items():
        print(f"  {c:12s} {v.get('label')} nrm={v.get('nrm_label')} rob_mean={v.get('robustness_mean')} se={v.get('robustness_se')} "
              f"n_eff={v.get('n_eff_robustness_cells')} all={v.get('all_cells_label')} reasons={v.get('reasons')}")
    for k, v in d['ablation'].items():
        rb = v['robustness']
        print(f"  ablation {k}: registered {v['registered']['benefit']['label']} "
              f"(mean {v['registered']['benefit'].get('aggregate', {}).get('mean')}); robustness {rb['label']} "
              f"(rob mean {rb.get('robustness_cells', {}).get('aggregate', {}).get('mean')}, all {rb.get('all_cells', {}).get('label')})")
    eff = d['refit_effectiveness']
    print('  refit detail:', json.dumps({k: v for k, v in eff.items() if k not in ('seed_checks', 'checks')}, default=str)[:900])
