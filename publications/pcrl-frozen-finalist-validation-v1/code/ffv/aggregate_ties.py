"""One-off publication step (ran once, at about 22:19Z, before the registration commit): replaces the per-person threshold-tie detail
in results/<campaign>/DEPLOYMENT_CHECKS.json by aggregate counts; the detail is kept privately in runs/DEPLOY/DEPLOYMENT_CHECKS_DETAIL.json."""
import json
from pathlib import Path

p = Path('results/pcrl_frozen_finalist_validation_v1/DEPLOYMENT_CHECKS.json'); d = json.loads(p.read_text())
for s, tv in d['threshold_ties'].items():
    for r, v in tv['per_role'].items():
        det = v.pop('detail')
        v['exactly_on_threshold'] = sum(x['exactly_on_threshold'] for x in det)
        v['level_changes_under_exactly_rounded_sum'] = sum(x['level_package_arithmetic'] != x['level_exactly_rounded_sum'] for x in det)
        v['level_changes_under_reversed_sum'] = sum(x['level_package_arithmetic'] != x['level_reversed_sum'] for x in det)
        v['max_abs_d_package_minus_exact'] = max((abs(x['d_package_minus_exact']) for x in det), default=None)
d['threshold_ties_note'] = 'aggregated for publication; the per-person detail (no identifiers) is kept in the private tree'
p.write_text(json.dumps(d, indent=1, sort_keys=True, default=float) + '\n')
print(json.dumps({s: {r: {k: v[k] for k in ('people_within_1e-12_of_a_threshold', 'level_changes_under_exactly_rounded_sum', 'level_changes_under_reversed_sum')}
                      for r, v in tv['per_role'].items() if v['people_within_1e-12_of_a_threshold']} for s, tv in d['threshold_ties'].items()}))
