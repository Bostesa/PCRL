"""Label-free extraction (investigator B, FS revision design): reads ONLY the committed fitting records
results/pcrl_strong_anchor_release_v1/fs/FIT_CPI_V<v>.json and prints what the FS R1 diagnosis uses. No label file, no MODEL_SELECT.
Run from the worktree root with python3."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
for v in (0, 1):
    r = json.loads((ROOT / f'results/pcrl_strong_anchor_release_v1/fs/FIT_CPI_V{v}.json').read_text())
    print(f'===== V{v} keys {sorted(r)}')
    sel = r['selection']
    print('lambda*', sel.get('lambda_star'), 'cand', json.dumps(r.get('cand'), default=str)[:300])
    print('start_cf_mean_nll', sel.get('start_cf_mean_nll'), 'minus h3 record', sel.get('start_cf_mean_nll_minus_h3_record'))
    for lam, sp in sel['per_lambda'].items():
        print('  lam', lam, {k: (round(sp[k]['estimate'], 5), round(sp[k]['se'], 5)) for k in ('inc_nll', 'inc_brier', 'inc_acc')})
    for lam, logs in r.get('train_logs', {}).items():
        first, last = logs[0], logs[-1]
        print('  log lam', lam, 'first', {k: round(first.get(k, 0), 4) for k in ('pair', 'hinge', 'd2_pair', 'd2_inc', 'd2_inc_start')},
              'last', {k: round(last.get(k, 0), 4) for k in ('pair', 'hinge', 'd2_pair', 'd2_inc', 'd2_inc_start', 'loss')})
    for key, P in r.get('paths', {}).items():
        print(key, 'lo', P.get('lo'), 'hi', P.get('hi'), 'flags', P.get('flags'), 'cand', P.get('cand_spend_over_start'))
    print('checks', r.get('checks'))
    if 'mf_reader_check' in r:
        print('mf readers', json.dumps(r['mf_reader_check'].get('auc'), default=str))
    print('saved', sorted(r.get('saved', {})), 'stop', r.get('stop'))
d = json.loads((ROOT / 'results/pcrl_strong_anchor_release_v1/claims/FS_CPI/DECISIONS.json').read_text())
print('DECISIONS keys', sorted(d)[:30])
print(json.dumps({k: d[k] for k in list(d)[:12]}, default=str)[:2500])
