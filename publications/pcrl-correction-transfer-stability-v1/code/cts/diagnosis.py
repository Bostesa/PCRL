"""Cost-transfer tables of the Track F diagnosis (question A) from the saved ladder, selection and nested records (label-free
aggregation; no label is read here).

    python3 -m cts.diagnosis tables        results/<campaign>/diagnosis/COST_TRANSFER.csv, COST_TRANSFER_SUMMARY.json,
                                           RULE_OUTCOMES.csv

For every partition (D0, D1, D0B when present), routing (FROZEN, REFIT), arm (PAIR, LOC, CONT) and ladder strength, the incremental
costs (men's / women's / all summed-Brier increase, NLL increase, accuracy loss) of each estimator with its household-clustered SE:
OOF (source estimator, within FIT), NESTED_OOF (where run), GUARD, GUARD_ENS, CHECK, CHECK_ENS, MS, MS_ENS; and the errors of the
predictions (OOF, NESTED_OOF, GUARD) against the realized excluded-household costs (CHECK, MS): signed (realized minus predicted),
absolute, and relative to the statistic's reserve slack (the witness's MODEL_SELECT margin minus 2 SE).
"""
from __future__ import annotations
import csv
import json
import sys

from . import common as C
from . import rules as RL
from . import trackf as TF

PARTS = ('D0', 'D1', 'D0B')
PRED = ('OOF', 'NESTED_OOF', 'GUARD', 'GUARD_ENS')
REAL = ('CHECK', 'MS', 'CHECK_ENS', 'MS_ENS')


def _load(P, did):
    f = TF.pdir(P) / 'LADDER' / f'{did}.json'
    if not f.is_file():
        return None
    lad = json.loads(f.read_text())
    est = dict(lad['estimators'])
    nf = TF.pdir(P) / 'NESTED' / f'NESTED_{did.split("_")[1]}.json'
    if did.startswith('REFIT_') and nf.is_file():
        est['NESTED_OOF'] = json.loads(nf.read_text())['ladder_NESTED_OOF']
    return est


def tables():
    rows, summ, rule_rows = [], {}, []
    for P in PARTS:
        selp = TF.pdir(P) / 'SELECT'
        for routing in ('FROZEN', 'REFIT'):
            for arm in ('PAIR', 'LOC', 'CONT'):
                did = f'{routing}_{arm}'
                est = _load(P, did)
                if est is None:
                    continue
                sf = selp / f'{did}.json'
                slack = json.loads(sf.read_text())['baseline'] if sf.is_file() else None
                ss = RL.stat_slack(slack) if slack else None
                for s in TF.LADDER:
                    key = str(s)
                    base = {'partition': P, 'routing': routing, 'arm': arm, 'strength': s}
                    for e, v in est.items():
                        if key not in v:
                            continue
                        r = {**base, 'estimator': e, 'n': v[key]['n']}
                        for k in RL.STATS:
                            r[k] = v[key]['stats'][k]; r[k + '_se'] = v[key]['se'][k]
                        r.update({f'mag_{m}': x for m, x in v[key].get('magnitude', {}).items()})
                        rows.append(r)
                    for p in PRED:
                        for q in REAL:
                            if p not in est or q not in est or key not in est[p] or key not in est[q]:
                                continue
                            if p.endswith('_ENS') != q.endswith('_ENS'):
                                continue
                            for k in RL.STATS:
                                a, b = est[p][key]['stats'][k], est[q][key]['stats'][k]
                                se = (est[p][key]['se'][k] ** 2 + est[q][key]['se'][k] ** 2) ** 0.5
                                err = b - a
                                summ.setdefault(f'{P}|{did}|{p}->{q}|{k}', []).append(
                                    {'s': s, 'predicted': a, 'realized': b, 'signed': err, 'se_diff': se,
                                     'z': err / se if se > 0 else None,
                                     'rel_slack': err / ss[k] if ss and ss[k] > 0 else None})
                if sf.is_file():
                    sel = json.loads(sf.read_text())
                    rf = selp / f'{did}__REALIZED.json'
                    real = json.loads(rf.read_text())['realized'] if rf.is_file() else {}
                    for rule, s in sel['selected'].items():
                        rr = {'partition': P, 'routing': routing, 'arm': arm, 'rule': rule, 'selected': s}
                        for where in ('CHECK', 'MS'):
                            if where in real and rule in real[where]:
                                o = real[where][rule]['outcome']
                                rr[f'{where}_reserve_crossing'] = o['reserve_crossing']
                                rr[f'{where}_point_cap_crossing'] = o['point_cap_crossing']
                                rr[f'{where}_max_slack_use'] = o['max_slack_use']
                        rule_rows.append(rr)
    out = C.RESULTS / 'diagnosis'
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('partition', 'routing', 'arm', 'strength', 'estimator', 'n'), k))
        with open(out / 'COST_TRANSFER.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    if rule_rows:
        keys = sorted({k for r in rule_rows for k in r}, key=lambda k: (k not in ('partition', 'routing', 'arm', 'rule', 'selected'), k))
        with open(out / 'RULE_OUTCOMES.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rule_rows)
    agg = {}
    for key, lst in summ.items():
        P, did, pq, k = key.split('|')
        nz = [x for x in lst if x['s'] > 0]
        agg[key] = {'mean_signed': sum(x['signed'] for x in nz) / len(nz), 'mean_abs': sum(abs(x['signed']) for x in nz) / len(nz),
                    'mean_abs_rel_slack': (sum(abs(x['rel_slack']) for x in nz if x['rel_slack'] is not None) /
                                           max(1, sum(1 for x in nz if x['rel_slack'] is not None))),
                    'max_abs_z': max((abs(x['z']) for x in nz if x['z'] is not None), default=None),
                    'at_s': {str(x['s']): {kk: x[kk] for kk in ('signed', 'se_diff', 'z', 'rel_slack')} for x in lst}}
    C.write_json(out / 'COST_TRANSFER_SUMMARY.json', {'schema': 'CTS1-cost-transfer-v1', 'utc': C.now(), 'pairs': agg,
                                                      'note': 'signed = realized minus predicted (positive = cost under-predicted); '
                                                              'se_diff = sqrt(se_pred^2 + se_real^2) (independent people sets; '
                                                              'OOF and NESTED_OOF are on FIT people, GUARD / CHECK on disjoint '
                                                              'households, MS on another role)'})
    print(json.dumps({'rows': len(rows), 'pairs': len(agg), 'rule_rows': len(rule_rows)}))
    return agg


if __name__ == '__main__':
    tables()
    sys.exit(0)
