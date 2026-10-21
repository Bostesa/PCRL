"""V6b reader-budget contrasts of the claims plans (label-free), own code.

For PRC_S2 against every plan member / reference, per version and registered repetition (R1, R2), on the pair view with the full
reader FIT: the person-weighted median FIT group size (the 0.5 quantile, numpy default, over FIT people of the FIT count of their own
transmitted pair vector) and the share of SCORE people whose pair vector never occurs in FIT; ratio = max / min of the two medians,
unseen difference = |difference of the shares|; READER_BUDGET_ASYMMETRIC iff ratio > 2.0 or difference > 0.10. Reader roles from
my own household split (v7_robustness.splits definition). Compared with the plans' reader_budget_detail and reader_budget flags
(flag read as ASYMMETRIC iff some version / repetition is asymmetric).

    python3 .../v6b_reader_budget.py --tag V1
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402
import v6_grid_claims as G  # noqa: E402

RATIO, UNSEEN = 2.0, 0.10
TOL = 1e-12


def pair_ids(rel):
    a = np.ascontiguousarray(np.column_stack((np.asarray(rel['income'], dtype=np.float64), np.asarray(rel['occupation'], dtype=np.float64))))
    return V.row_ids(a)[0]


def headline(ids, fit, score):
    K = int(ids.max()) + 1
    cnt = np.bincount(ids[fit], minlength=K)
    own = cnt[ids[fit]]
    return {'median_group': float(np.quantile(own.astype(np.float64), 0.5)), 'unseen': float(np.mean(cnt[ids[score]] == 0))}


def run(tag):
    if str(V.ROOT) not in sys.path:
        sys.path.insert(0, str(V.ROOT))
    import frc.common as C
    import v4_utility as U
    unit = np.asarray(C.features('MODEL_SELECT')['unit'])
    roles = {}
    for r in ('R1', 'R2'):
        part = V.household_folds(unit, 10, f'FRC1_devgrid_{r}')
        roles[r] = (part <= 4, part >= 7)
    bank = json.loads((V.RUNS / 'GRID' / 'FRC1_MAIN' / 'S0' / 'GRID.json').read_text())['members']
    banks = {v: json.loads((V.RUNS / 'GRID' / 'FRC1_MAIN' / f'S{v}' / 'GRID.json').read_text())['members'] for v in (0, 1)}
    checks, record = [], {'tag': tag, 'utc': V.utc(), 'per': {}}
    for pid in ('MAIN', 'R1'):
        plan = json.loads((V.RESULTS / 'claims' / f'PLAN_{pid}.json').read_text())
        s2 = plan['prc_s2']
        mm, mine = [], {}
        for m, flag in (plan.get('reader_budget') or {}).items():
            det = (plan.get('reader_budget_detail') or {}).get(m) or {}
            labs = []
            for v in (0, 1):
                ra = G.member_release(s2, banks[v].get(s2) or {}, v, U)
                rb = G.member_release(m, banks[v].get(m) or {}, v, U)
                if ra is None or rb is None:
                    mm.append(f'{m} V{v}: release not loadable')
                    continue
                ia, ib = pair_ids(ra), pair_ids(rb)
                for r in ('R1', 'R2'):
                    fit, score = roles[r]
                    ha, hb = headline(ia, fit, score), headline(ib, fit, score)
                    ratio = max(ha['median_group'], hb['median_group']) / min(ha['median_group'], hb['median_group'])
                    dun = abs(ha['unseen'] - hb['unseen'])
                    lab = 'READER_BUDGET_ASYMMETRIC' if ratio > RATIO or dun > UNSEEN else 'READER_BUDGET_COMPARABLE'
                    labs.append(lab)
                    mine[f'{m}|V{v}{r}'] = {'ratio': ratio, 'unseen_difference': dun, 'label': lab}
                    d = det.get(f'V{v}{r}') or {}
                    if d.get('label') != lab or abs(d.get('group_size_ratio', -1) - ratio) > TOL or \
                            abs(d.get('score_unseen_difference', -1) - dun) > TOL:
                        mm.append(f'{m} V{v}{r}: mine {lab} {ratio} {dun} recorded {d}')
            overall = 'READER_BUDGET_ASYMMETRIC' if 'READER_BUDGET_ASYMMETRIC' in labs else 'READER_BUDGET_COMPARABLE'
            if labs and overall != flag:
                mm.append(f'{m}: overall mine {overall} recorded {flag}')
        record['per'][pid] = {'mine': mine, 'mismatches': mm}
        checks.append({'id': f'V6B.READER_BUDGET.{pid}__{tag}',
                       'what': f'PLAN_{pid}: pair-view reader-budget contrasts of PRC_S2 vs every member (person-weighted median FIT '
                               f'group size ratio, SCORE unseen-share difference, thresholds 2.0 / 0.10) recomputed label-free',
                       'inputs': V.inputs_sha([V.RESULTS / 'claims' / f'PLAN_{pid}.json']), 'result': {'mismatches': mm[:20],
                                                                                                        'n_mismatches': len(mm),
                                                                                                        'n_contrasts': len(mine)},
                       'numerical_status': 'PASS' if not mm else 'FAIL', 'procedural_status': 'NOT_APPLICABLE'})
    V.write_record(f'V6B_READER_BUDGET__{tag}.json', record)
    print(json.dumps(V.append_checks(checks), indent=1))
    for c in checks:
        print(c['id'], c['numerical_status'], c['result']['n_mismatches'], c['result']['mismatches'][:4])
    return 0


def selftest():
    ids = np.array([0, 0, 0, 1, 1, 2, 3, 3])
    fit = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    score = ~fit
    h = headline(ids, fit, score)
    ok = {'median_group': h['median_group'] == 3.0, 'unseen': abs(h['unseen'] - 1.0) < 1e-15}
    ok['all_ok'] = all(ok.values())
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        print(json.dumps(ok, indent=1))
        return 0 if ok['all_ok'] else 1
    return run(a.tag)


if __name__ == '__main__':
    sys.exit(main())
