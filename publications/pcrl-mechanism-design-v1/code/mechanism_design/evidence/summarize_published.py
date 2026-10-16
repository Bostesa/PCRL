"""Summarise published ACS aggregates for the evidence pack (aggregate tables only; no person-level data)."""
import csv, json, sys
from collections import Counter
from mechanism_design import no_real_data
no_real_data.install()
R = 'results/'
src = R + 'pcrl_acs_protected_tree_replication_v1/'
hyb = R + 'pcrl_acs_frozen_hybrids_v1/'
out = {}
rows = list(csv.DictReader(open(src + 'SERVICE_TABLE.csv')))
out['source_service_table_columns'] = list(rows[0].keys())
svc = {}
for r in rows:
    svc.setdefault(r.get('service'), []).append({k: r[k] for k in r if k in ('seed', 'service', 'view', 'auc', 'winner', 'model_select_point_feasible_24', 'dev_report_point_feasible_24')})
out['source_service_names'] = sorted(svc)
out['source_rows_sample'] = rows[:3]
sel = json.load(open(src + 'SELECTION_DETAIL.json'))
nr = sel['family_tables']['NEURAL_RAW']
cnt = Counter()
for c in nr:
    for fc in c['failed_clauses']:
        s, task, ref, metric = fc.split('|')
        cnt[f'{task}|{metric}'] += 1
out['neural_raw_grid'] = [{'configuration': c['configuration'], 'feasible': c['feasible'], 'max_normalized': round(c['max_normalized'], 4),
                           'n_failed': len(c['failed_clauses']),
                           'failed_tasks_metrics': sorted({'|'.join(fc.split('|')[1::2]) for fc in c['failed_clauses']}),
                           'proxy_auc_mean': round(c['proxy_auc_mean'], 4)} for c in nr]
out['neural_raw_failed_clause_counts_by_task_metric'] = dict(cnt)
json.dump(out, sys.stdout, indent=1, default=str)
