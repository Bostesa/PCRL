"""Verifier stage V6b (label-free; after adjudication A-SC1): the equal-reserve table and Pareto frontier, the original-benchmark table
and the COARSE_REFERENCE_INCOME_WIRE flags, rebuilt from the verifier's own records (V6 utility, V4 size fits, V5 reader labels) and the
reader grids, vs conditional/CLAIMS.json.

PCRL controlled training-size campaign (TSZ1; declared development readers only; two-SE reserve = development heuristic).

    python3 .../v6_tables.py --util-record <V6_COND_UTILITY record> --size-tag V1 --readers-tag V1 --tag V1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

RES = V.RESULTS
GRID = V.ROOT / '.private' / 'pcrl_controlled_training_size_v1' / 'runs' / 'GRID' / 'COND'
REFS = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ')
COARSE = ('B2TWIN', 'RETIRED_B3CTL')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--util-record', required=True); ap.add_argument('--size-tag', required=True)
    ap.add_argument('--readers-tag', required=True); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    up = V.RECORDS / a.util_record
    uc = next(c for c in json.loads(up.read_text())['checks'] if c['id'].startswith('V6.COND_UTILITY'))['observed']
    rel, refs = uc['releases'], uc['references']
    size = json.loads((V.RECORDS / f'V4_SIZE_MINE__{a.size_tag}.json').read_text())['mine']['fits']
    labels = next(c for c in json.loads((V.RECORDS / f'V5_READERS__{a.readers_tag}.json').read_text())['checks']
                  if c['id'].startswith('V5.READERS.LABELS'))['observed']
    g = {v: json.loads((GRID / f'S{v}' / 'GRID.json').read_text()) for v in (0, 1)}

    def pair_auc(name):
        vals = []
        for v in (0, 1):
            cs = [c for c in g[v]['cells'] if '@' not in c]
            vals.append(sum(g[v]['cells'][c][name]['pair']['score_auc'] for c in cs) / len(cs))
        return sum(vals) / 2

    def ratio(name, v, whole):
        if name in rel:
            x = rel[name][str(v)]
            return x['min_whole_ratio'] if whole else x['min_occ_ratio']
        if name in ('TR100', 'TT100'):
            f = size[f"{'TR' if name == 'TR100' else 'TT'}|S100|V{v}"]
            return min(f['min_occ_ratio'], f['income_min_ratio']) if whole else f['min_occ_ratio']
        x = refs[name][str(v)]
        return x['min_whole_ratio'] if whole else x['min_occ_ratio']
    cl_p = RES / 'conditional' / 'CLAIMS.json'
    cl = json.loads(cl_p.read_text())
    bad, eq_mine = [], {}
    names = ['LOC_SEL', 'PAIR_SEL', 'CONT_AT_LOC', 'CONT_AT_PAIR', 'LOC_AT_PAIR', 'W', 'TR100', 'TT100'] + list(REFS)
    for whole, key in ((True, 'whole_release_28'), (False, 'occupation_side_16')):
        pts = [{'id': n, 'min_ratio': min(ratio(n, 0, whole), ratio(n, 1, whole)), 'pair_auc_mean': pair_auc(n)} for n in names]
        for p in pts:
            p['qualified'] = p['min_ratio'] >= 2.0
        q = [p for p in pts if p['qualified']]
        front = sorted(p['id'] for p in q if not any(o['pair_auc_mean'] <= p['pair_auc_mean'] and o['min_ratio'] >= p['min_ratio'] and
                                                    (o['pair_auc_mean'] < p['pair_auc_mean'] or o['min_ratio'] > p['min_ratio']) for o in q))
        eq_mine[key] = {'points': pts, 'frontier': front}
        theirs = {p['id']: p for p in cl['equal_reserve'][key]['points']}
        for p in pts:
            t = theirs.get(p['id'])
            if t is None or abs(t['min_ratio'] - p['min_ratio']) > 1e-9 or abs(t['pair_auc_mean'] - p['pair_auc_mean']) > 1e-12 or t['qualified'] != p['qualified']:
                bad.append(f'{key}|{p["id"]}')
        if sorted(cl['equal_reserve'][key]['frontier']) != front:
            bad.append(f'{key} frontier {front} vs {sorted(cl["equal_reserve"][key]["frontier"])}')
    # original benchmark rows
    ob = []
    for row in cl['original_benchmark']:
        c, r = row['candidate'], row['reference']
        lab = labels.get(f'{c}|{r}') or {}
        mine = {'candidate_point_both_versions': all(rel[c][str(v)]['point_pass'] for v in (0, 1)),
                'candidate_occ_reserve_both_versions': all(rel[c][str(v)]['occ_reserve_met'] for v in (0, 1)),
                'candidate_whole_reserve_both_versions': all(rel[c][str(v)]['whole_release_reserve_met'] for v in (0, 1)),
                'reference_point_both_versions': all(refs[r][str(v)]['point_pass_12_plus_2'] for v in (0, 1)),
                'reference_whole_reserve_both_versions': all(refs[r][str(v)]['whole_release_reserve_met'] for v in (0, 1)),
                'pair_benefit_label': lab.get('stability'),
                'income_gate': 'RECIPIENT_POINT_PASS' if lab.get('recipient_max', {}).get('income', 1) <= 0.010 else 'RECIPIENT_POINT_FAILED',
                'occupation_gate': 'RECIPIENT_POINT_PASS' if lab.get('recipient_max', {}).get('occupation', 1) <= 0.010 else 'RECIPIENT_POINT_FAILED',
                # V2 correction: the 0.020 target (C6) is defined only against NEURAL_RAW and H_LJ (RULES.md 9-11); V1 compared it for all six
                'target_0020': lab.get('target_label') if r in ('NEURAL_RAW', 'H_LJ') else None}
        diff = [k for k, x in mine.items() if row.get(k) != x]
        if row.get('pair_benefit_mean') is not None and lab.get('mean') is not None and abs(row['pair_benefit_mean'] - lab['mean']) > 1e-12:
            diff.append('pair_benefit_mean')
        if diff:
            ob.append(f'{c}|{r}: {diff}')
    # COARSE_REFERENCE_INCOME_WIRE flags
    fl = []
    for key, comp in cl['comparisons'].items():
        cand, ref = key.split('|')
        if 'pair' not in comp:
            continue
        exp_income = ref in COARSE
        exp_pair = ref in COARSE and any(str(g[v]['cells'][c][ref]['pair']['winner']).startswith('IGNORE_occupation__')
                                         for v in (0, 1) for c in g[v]['cells'] if '@' not in c)
        fv = comp.get('flag_views') or {}
        if bool(fv.get('income', False)) != exp_income or bool(fv.get('pair', False)) != exp_pair:
            fl.append(f'{key}: mine income {exp_income} pair {exp_pair} vs {fv}')
    checks = [{'id': f'V6.TABLES__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
               'what': 'equal-reserve table (whole-release 28 and occupation-side 16: min ratio over versions, mean pair SCORE AUC, qualified iff '
                       'ratio >= 2, Pareto frontier), original-benchmark rows (point and reserve statuses of candidate and reference, pair-benefit '
                       'label and mean, per-view recipient gates, 0.020 target) and the COARSE_REFERENCE_INCOME_WIRE flags (A-SC1), rebuilt from '
                       'own records and the reader grids vs CLAIMS.json',
               'inputs': V.inputs_sha([up, cl_p, V.RECORDS / f'V4_SIZE_MINE__{a.size_tag}.json', V.RECORDS / f'V5_READERS__{a.readers_tag}.json',
                                       GRID / 'S0' / 'GRID.json', GRID / 'S1' / 'GRID.json']),
               'expected': 'agreement', 'observed': {'equal_reserve_mine': {k: {'frontier': v['frontier'], 'n_qualified': sum(p['qualified'] for p in v['points']),
                                                                               'failing': sorted(p['id'] for p in v['points'] if not p['qualified'])}
                                                                           for k, v in eq_mine.items()},
                                                     'equal_reserve_problems': bad, 'original_benchmark_problems': ob, 'coarse_flag_problems': fl},
               'status': 'PASS' if not (bad or ob or fl) else 'FAIL', 'code_sha256': V.code_hashes(__file__, V.__file__)}]
    V.write_record(f'V6_TABLES__{a.tag}.json', {'checks': checks, 'utc': V.utc()})
    print(json.dumps(V.append_checks(checks)))
    print(json.dumps({checks[0]['id']: checks[0]['status'], **checks[0]['observed']}, default=str)[:3000])


if __name__ == '__main__':
    main()
