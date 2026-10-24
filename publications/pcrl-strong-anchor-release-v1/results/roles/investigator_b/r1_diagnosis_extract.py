"""Label-free extraction (investigator B, revision R1 design): reads ONLY the committed MODEL_FIT fit records
results/pcrl_strong_anchor_release_v1/fb/FIT_PFD_V<v>.json (aggregates written by the registered FB_PFD_v1 fits) and prints the
numbers the R1 diagnosis uses. No label file, no MODEL_SELECT. Run from the worktree root with python3."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
COMP = ('inc_nll', 'inc_brier', 'inc_acc', 'occ_nll', 'occ_brier', 'occ_brier_men', 'occ_brier_women', 'occ_acc', 'tot_nll')


def fmt(sp, k):
    e = sp[k]
    return f"{e['estimate']:+.4f}({e['se']:.4f})"


out = {}
for v in (0, 1):
    r = json.loads((ROOT / f'results/pcrl_strong_anchor_release_v1/fb/FIT_PFD_V{v}.json').read_text())
    print(f'===== V{v}: keys {sorted(r)}')
    sel = r['selection']
    print('k*', sel.get('k_star'), 'cand', json.dumps(r.get('cand'), default=str)[:400])
    print('budget', sel['budget'])
    for arm in ('PFD', 'TASK', 'RAWJ'):
        for e in range(0, 11):
            key = f'{arm}_E{e}'
            if key in sel['path'] and e in (0, 1, 2, 3, 5, 10):
                sp = sel['path'][key]
                print(f'  {key:8s}', ' '.join(f'{k}={fmt(sp, k)}' for k in COMP))
    if 'income_share_of_nll_spend' in sel:
        print('income share', json.dumps(sel['income_share_of_nll_spend']))
    for key in ('CTRL', 'ALT'):
        P = r.get('paths', {}).get(key)
        if P:
            print(key, 'lo', P.get('lo'), 'hi', P.get('hi'), 'flags', P.get('flags'), 'cand', P.get('cand_spend_over_start'))
    fr = r.get('full_records', {})
    for arm, rec in fr.items():
        print('full', arm, 'lambda', rec.get('lambda'), 'cal', json.dumps(rec.get('calibration', rec.get('calibration_medians')), default=str)[:300])
    for j, f in enumerate(r.get('fold_records', [])):
        print(' fold', j, {a: (x.get('lambda'), round(x.get('cpu_s', 0), 1)) for a, x in f.items()})
    logs = r.get('full_logs_PFD', [])
    for e in logs[:6]:
        print('  log step', e.get('step'), 'P', round(e.get('P', 0), 4), 'terms', {k: round(x, 4) for k, x in e.get('terms', {}).items()},
              'd2', {k: round(x, 4) for k, x in e.get('d2', {}).items()}, 'tgn', [round(x, 3) for x in e.get('task_grad_norm', [])],
              'wpgn', [round(x, 3) for x in e.get('weighted_privacy_grad_norm', [])])
    if 'mf_reader_check' in r:
        print('mf readers', json.dumps(r['mf_reader_check'].get('auc'), default=str)[:1500])
    print('checks', r.get('checks'), 'saved', sorted(r.get('saved', {})), 'stop', r.get('stop'))
