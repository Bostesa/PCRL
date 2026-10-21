"""Coordinator's summary of the MODEL_SELECT utility records (EVAL_MS.json of this campaign's releases and the references).

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/inspect_eval.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / '.private' / 'pcrl_dual_design_release_v1' / 'runs'


def row(name, ev):
    out = []
    for v in ('0', '1'):
        e = ev[v]
        if e.get('missing'):
            out.append('missing'); continue
        p, r = e['point'], e.get('reserve_without_mc') or e.get('reserve') or {}
        tot = e['absolute']
        out.append(f"{'PASS' if p['point_pass_12_plus_2'] else 'FAIL'} worst {p['max_value_over_cap']:.3f} reserve {r.get('min_whole_ratio', float('nan')):.2f}"
                   + (f" failed {p['failed']}" if p['failed'] else ''))
    return f'{name:14s} ' + ' | '.join(out)


for fam in ('DA', 'DB'):
    for d in sorted((RUNS / fam).glob('*/EVAL_MS.json')):
        ev = json.loads(d.read_text())['versions']
        print(row(d.parent.name, ev))
        for v in ('0', '1'):
            e = ev[v]
            if e.get('missing'):
                continue
            g = {f"{c['task'][:3]}|{c['reference']}|{c['metric'][:4]}": round(float(c['value']), 4) for c in e['utility']['clauses']}
            gd = {f"guard|{x['group']}": round(float(x['value']), 4) for x in e['utility']['guards']}
            key = {k: g[k] for k in g if k.endswith('U0_S' + v + '|nll_') or 'nll' in k}
            print('      V' + v, {k: val for k, val in g.items() if ('U0' in k)}, gd)
            pc = e.get('paired_costs') or {}
            for comp in ('DA/DA_TWIN',):
                if comp in pc and not pc[comp].get('missing'):
                    c = pc[comp]['costs']
                    print('        vs', comp, {k: round(c[k]['value'], 5) for k in ('total|nll', 'occupation|nll', 'income|nll', 'occupation|acc_loss', 'income|acc_loss', 'occupation|men_brier', 'occupation|women_brier')})
ref = json.loads((RUNS / 'REFERENCES' / 'EVAL_MS.json').read_text())['references']
for name, ev in ref.items():
    print(row(name, ev))
