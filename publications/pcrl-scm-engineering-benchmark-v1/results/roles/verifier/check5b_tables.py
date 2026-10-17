"""Independent verifier, check 5 (part b): RESULTS_TABLES.md sections 2-4 and report/PRIMARY_ENDPOINTS.csv against the verifier's
recomputed endpoints (check5_endpoints.json) and the recomputed recovery values (which equal DEV_REPORT_RESULTS.json to 1e-16).
Compares at the displayed precision (4 decimals). Aggregates only."""
import csv
import json
import re
from pathlib import Path

RES = Path('<repository>/.claude/worktrees/pcrl-scm-engineering-benchmark-v1/results/pcrl_scm_engineering_benchmark_v1')
V = json.loads((RES / 'roles' / 'verifier' / 'check5_endpoints.json').read_text())
R = json.loads((RES / 'report' / 'DEV_REPORT_RESULTS.json').read_text())
md = (RES / 'RESULTS_TABLES.md').read_text()
f4 = lambda x: f'{x:.4f}'
mism = []
n3 = n2 = 0
for line in md.splitlines():
    m = re.match(r'^\| ((?:RECIPE|ATTRIBUTION|COMPETITIVE|UTILITY|SUBGROUP)\|[^ ]+) \| (\S+) \| (\S+) \| \[(\S+), (\S+)\] \| (\S+) \|$', line)
    if m:
        k = m.group(1); e = V['endpoints'][k]; n3 += 1
        bound = e['clause_L1'] if e['clause_L1'] is not None and ('benefit' in k) else e['clause_U1']
        exp = (f4(e['point']), f4(bound), f4(e['family_lower']), f4(e['family_upper']), e['status'])
        got = tuple(m.group(i) for i in range(2, 7))
        if tuple(g.replace('-0.0000', '0.0000') for g in got) != tuple(x.replace('-0.0000', '0.0000') for x in exp):
            mism.append(f'S3 {k}: table={got} verifier={exp}')
    m = re.match(r'^\| (S[01]) \| (\w+) \| (\S+) \((\S+)\) \| (\S+) \| (\S+) \| (\S+) \| (\w+) \| (True|False) \|$', line)
    if m:
        s, svc = m.group(1), m.group(2); n2 += 1
        v = R['per_seed'][s]['services'][svc]
        exp = (f4(v['recovery']['pair']['auc']), f4(v['recovery']['pair']['auc_se']), f4(v['recovery']['income']['auc']), f4(v['recovery']['occupation']['auc']),
               f4(v['recovery_slate7']['pair']['auc']), v['recovery']['pair']['winner'], str(v['utility_point_feasible_24_this_seed']))
        got = tuple(m.group(i) for i in range(3, 10))
        if got != exp:
            mism.append(f'S2 {s} {svc}: table={got} report={exp}')
# PRIMARY_ENDPOINTS.csv vs verifier (full precision)
csvmax = 0.0; csv_status = []
with open(RES / 'report' / 'PRIMARY_ENDPOINTS.csv') as fh:
    for row in csv.DictReader(fh):
        e = V['endpoints'][row['key']]
        for f in ('point', 'se', 'family_lower', 'family_upper', 'clause_L1', 'clause_U1'):
            csvmax = max(csvmax, abs(float(row[f]) - e[f]))
        if row['status'] != e['status'] or row['clause_pass'] != str(e['clause_pass']) or row['point_pass'] != str(e['point_pass']):
            csv_status.append(row['key'])
# section 4 claims lines
claims_md = dict(re.findall(r'^- \*\*(\w+):\*\* (\S+)', md, flags=re.M))
cl = V['claims_recomputed']
claims_cmp = {'A_RECIPE_VS_TWIN': claims_md.get('A_RECIPE_VS_TWIN') == cl['A_RECIPE_VS_TWIN'],
              'ATTRIBUTION_VS_PUB_IBSI_AND_SHUF': claims_md.get('ATTRIBUTION_VS_PUB_IBSI_AND_SHUF') == cl['ATTRIBUTION_VS_PUB_IBSI_AND_SHUF'],
              'B_COMPETITIVE': claims_md.get('B_COMPETITIVE') == cl['B_COMPETITIVE'], 'C_SUBGROUP': claims_md.get('C_SUBGROUP') == cl['C_SUBGROUP']}
out = {'section3_rows_checked': n3, 'section2_rows_checked': n2, 'mismatches_at_display_precision': mism,
       'primary_csv_max_abs_diff_vs_verifier': csvmax, 'primary_csv_status_mismatch_keys': csv_status, 'section4_claims_match': claims_cmp}
(RES / 'roles' / 'verifier' / 'check5b_tables.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
