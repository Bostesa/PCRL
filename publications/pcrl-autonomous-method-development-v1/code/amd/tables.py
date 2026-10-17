"""Public development tables from the private EVAL_MS.json files and the baseline table (aggregates only; no person-level data).

COMPARISON_TABLE.csv: one row per family x version x configuration x seed (plus the frozen references): MODEL_SELECT utility
(12 clauses, worst normalized value), within-SEX guards, development reader proxies (pair / income / occupation), recipient
screens, pair benefit against both references, distinct outputs and the promotion status.
DEV tables (UTILITY_TABLE.csv, RECOVERY_TABLE.csv) are written by `dev` after the one logged DEV_REPORT evaluation.
"""
from __future__ import annotations
import csv
import json
import sys

from . import common as C

REFS = ('NEURAL_RAW', 'H_LJ')


def comparison():
    base = {s: json.loads((C.RUNS / 'BASELINES' / f'BASELINES_MS_S{s}.json').read_text()) for s in C.SEEDS}
    prom = json.loads((C.RESULTS / 'PROMOTION.json').read_text()) if (C.RESULTS / 'PROMOTION.json').exists() else {'trials': []}
    status = {(t['family'], t['config_id']): t.get('status') for t in prom['trials']}
    with open(C.RESULTS / 'EXPERIMENT_LEDGER.csv') as f:
        ledger = list(csv.DictReader(f))
    version = {(r['family'], r['config_id']): r['version'] for r in ledger}
    rows = []

    def row(family, ver, cid, s, m, kind):
        cl = {(c['task'], c['reference'], c['metric']): c['value'] for c in m['clauses']}
        pr = {v: m['proxy'][v]['max'] for v in ('pair', 'income', 'occupation')}
        bar = {r: {v: base[s][r][str(s)]['proxy'][v]['max'] for v in ('pair', 'income', 'occupation')} for r in REFS}
        out = {'family': family, 'version': ver, 'config_id': cid, 'kind': kind, 'seed': s,
               'ms_clauses_pass_12': sum(c['point_pass'] for c in m['clauses']), 'ms_worst_normalized': round(m['max_normalized_12'], 4),
               'ms_guard_men': round(m['guard_occupation_brier_excess_vs_U0']['sex0'], 5), 'ms_guard_women': round(m['guard_occupation_brier_excess_vs_U0']['sex1'], 5)}
        for t in C.TASKS:
            for met in ('accuracy_loss', 'nll_excess', 'brier_excess'):
                for ref in ('U0', 'Ucal'):
                    out[f'ms_{t}_{met}_vs_{ref}'] = round(cl[(t, ref, met)], 5)
        out.update({'proxy_pair': round(pr['pair'], 4), 'proxy_income': round(pr['income'], 4), 'proxy_occupation': round(pr['occupation'], 4),
                    'pair_benefit_vs_NEURAL_RAW': round(bar['NEURAL_RAW']['pair'] - pr['pair'], 4), 'pair_benefit_vs_H_LJ': round(bar['H_LJ']['pair'] - pr['pair'], 4),
                    'income_screen_pass': all(pr['income'] <= bar[r]['income'] + 0.010 for r in REFS),
                    'occupation_screen_pass': all(pr['occupation'] <= bar[r]['occupation'] + 0.010 for r in REFS),
                    'distinct_pair': m['distinct_outputs']['pair'], 'distinct_income': m['distinct_outputs']['income'],
                    'promotion_status': status.get((family, cid), '' if kind != 'reference' else 'REFERENCE')})
        return out
    for s in C.SEEDS:
        for name, per in base[s].items():
            rows.append(row('REF', 'frozen', name, s, per[str(s)], 'reference'))
    for fam in ('A', 'B', 'C'):
        d = C.RUNS / fam
        if not d.is_dir():
            continue
        for cfg in sorted(p for p in d.iterdir() if p.is_dir()):
            for s in C.SEEDS:
                ev = cfg / f'S{s}' / 'EVAL_MS.json'
                if ev.is_file():
                    kind = 'control' if any(x in cfg.name.upper() for x in ('CTL', 'CTRL', 'TASKREFIT')) else ('diagnostic' if 'DIAG' in cfg.name.upper() else 'candidate')
                    rows.append(row(fam, version.get((fam, cfg.name), ''), cfg.name, s, json.loads(ev.read_text()), kind))
    with open(C.RESULTS / 'COMPARISON_TABLE.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'COMPARISON_TABLE.csv: {len(rows)} rows')


if __name__ == '__main__':
    {'comparison': comparison}[sys.argv[1]]()
