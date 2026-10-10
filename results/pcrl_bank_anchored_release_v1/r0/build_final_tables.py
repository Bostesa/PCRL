"""Build public aggregate endpoint tables from independently replayed receipts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
selection = json.loads((ROOT / 'r5/R5_REPLAY_PUBLIC_SELECTION_V2_RECEIPT.json').read_text())
intervals = json.loads((ROOT / 'r5/R5_RECOVERY_INTERVALS_K1575_V2.json').read_text())
with (ROOT / 'r3/R3_ORIGINAL_POINT_INTERVAL_ARM_TABLE.csv').open(newline='') as f:
    utility = {row['arm_id']: row for row in csv.DictReader(f)}

arms = ['B0', 'T', 'D', 'P', 'L', 'J', 'S12', 'S21', 'W_Q', 'P_STAR']
rows = []
for arm in arms:
    gate = utility[arm]
    for seed in ('s0', 's1', 'mean'):
        metrics = selection['equal_seed_means'][arm][seed]
        for view in ('r1', 'r2', 'pair'):
            rows.append({
                'arm_id': arm,
                'source_seed': seed,
                'view': view,
                'sex_auc': '' if metrics is None else metrics[view],
                'original_inner_point_passed': gate['original_inner_point_passed'],
                'original_inner_point_total': gate['original_inner_point_total'],
                'point_attack_eligible': gate['point_attack_eligible'],
                'utility_simultaneous_status': gate['full_seed_interval_status'],
                'fit_b0_proxy_passed': gate['fit_b0_proxy_passed'],
                'fit_b0_proxy_total': gate['fit_b0_proxy_total'],
                'recovery_status': 'CONDITIONAL_NULL' if metrics is None else 'FRESH_ATTACK_REPLAYED',
                'recovery_scope': 'exploratory development on reused INNER rows',
            })
out = ROOT / 'PRIMARY_ENDPOINTS.csv'
with out.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

comparisons = [('B0','P','pair'),('B0','J','pair'),('P','J','pair'),('P_STAR','J','pair'),
               ('J','P_STAR','r1'),('J','P_STAR','r2'),('D','J','pair'),('L','J','pair'),
               ('S12','J','pair'),('T','J','pair'),('W_Q','J','pair')]
comp_rows=[]
for left,right,view in comparisons:
    key=f'mean:recovery:{left}:{right}:{view}:auc'
    reverse=False
    if key not in intervals['clauses']:
        key=f'mean:recovery:{right}:{left}:{view}:auc'
        reverse=True
    value=intervals['clauses'][key]
    sign=-1 if reverse else 1
    comp_rows.append({'contrast':f'{left}_minus_{right}','view':view,
                      'auc_difference':sign*value['point'],
                      'simultaneous_lower':-value['upper'] if reverse else value['lower'],
                      'simultaneous_upper':-value['lower'] if reverse else value['upper'],
                      'simultaneous_status':value['status'],
                      'family_K':intervals['fixed_family_K'],
                      'scope':intervals['scope']})
out=ROOT/'HEADLINE_COMPARISONS.csv'
with out.open('w', newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(comp_rows[0])); w.writeheader(); w.writerows(comp_rows)
print(f'{len(rows)} primary rows, {len(comp_rows)} comparisons')
