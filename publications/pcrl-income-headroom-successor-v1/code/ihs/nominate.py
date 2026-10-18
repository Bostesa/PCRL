"""Unit IHS_NOMINATION: applies the frozen nomination rule (roles/statistical_reviewer/NOMINATION_RULE.md, registered at
74376912c) to every configuration of the campaign, from the MODEL_SELECT development evaluations only. No new label is read
here: it uses the EVAL_MS.json records written by ihs.devrun and the cached reference evaluations.

Per configuration: admissibility; (a) POINT_ELIGIBLE (24 clauses + 4 guards at the point, both seeds); (b) two-seed reserve
(ihs.reserve.two_seed_status); c1 PAIR_SUPERIOR (benefit > 0 in every cell vs NEURAL_RAW and H_LJ, both seeds); c2
RECIPIENT_SCREEN (every recipient cell difference <= 0.010); c3 GAIN_OVER_B2TWIN (pair below B2TWIN's in every cell, both
seeds); flags PAIR_NEAR_TIE, READER_UNSTABLE, RECIPIENT_EDGE; (d) Bonferroni utility bounds (reported). Labels in the rule's
order; deterministic choice of at most one finalist. Output: NOMINATION.json and NOMINATION_TABLE.csv (results/<campaign>/).
"""
from __future__ import annotations
import csv
import json

import numpy as np

from . import common as C
from . import devrun as D
from . import evaluate as EV
from . import ledger as L
from . import reserve as R

Z_BONF_28 = 3.1237346
NEAR_TIE = 0.005
REFS = ('NEURAL_RAW', 'H_LJ')


def cell_values(m, view, key):
    return {c: m['proxy']['cells'][c][view][key] for c in m['proxy']['cells']}


def pair_winner(cell):
    """The reader (or single-view alternative) attaining the cell's pair maximum."""
    cands = {f'pair:{n}': cell['pair'][n] for n in EV.DEV_READERS}
    cands.update({f'income:{n}': cell['income'][n] for n in EV.DEV_READERS})
    cands.update({f'occupation:{n}': cell['occupation'][n] for n in EV.DEV_READERS})
    return max(sorted(cands), key=lambda k: cands[k])


def configurations():
    """First COMPLETE evaluation row of each (configuration, seed) (reviewer N1): counts_toward_36 = 1, a fit or control row
    with a reserve status and development proxies; FAILED / STOPPED / decision rows never count."""
    led = [r for r in L.rows() if r['kind'] in ('fit', 'control') and r['counts_toward_36'] == '1'
           and r['reserve_status'] and r['proxy_pair_mean'] and not str(r['decision']).startswith(('FAILED', 'STOPPED'))]
    out = {}
    for r in led:
        key = (r['family'], r['config_id'])
        out.setdefault(key, {'entries': {}, 'rows': {}, 'kind': r['kind']})
        s = int(r['seed'])
        if s not in out[key]['entries']:
            out[key]['entries'][s] = int(r['entry']); out[key]['rows'][s] = r
    return out


CHECK_FIELDS = ('proxy_pair_mean', 'proxy_pair_min', 'proxy_pair_max', 'inc_brier_excess_Ucal', 'inc_nll_excess_Ucal', 'guard_men', 'reserve_status')


def record_matches(m, row):
    """The EVAL_MS.json on disk must reproduce the counted ledger row (reviewer N1: a later run must not have replaced it)."""
    s = L.summarize(m, m.get('comparisons'), m.get('reserve'))
    return [f for f in CHECK_FIELDS if str(s.get(f, '')) != str(row.get(f, ''))]


def evaluate_config(fam, cfg, info, refs_eval):
    rec = {'family': fam, 'config_id': cfg, 'kind': info['kind'], 'seed0_entry': info['entries'].get(0), 'flags': []}
    seeds = sorted(info['entries'])
    if seeds != [0, 1]:
        rec['label'] = 'NOT_CONSIDERED_ONE_SEED_ONLY'; rec['seeds'] = seeds
        return rec
    ms, nfits, rels, mism, inputs = {}, 0, {}, {}, {}
    rule_sha = R.load_rule()['_sha256']
    for s in (0, 1):
        ms[s] = json.loads((D.release_dir(fam, cfg, s) / 'EVAL_MS.json').read_text())
        rels[s] = json.loads((D.release_dir(fam, cfg, s) / 'RELEASE.json').read_text())
        nfits += int(rels[s].get('n_model_fits') or 0)
        if ms[s]['proxy']['version'] != EV.PROXY_VERSION:
            raise AssertionError(f'{fam}/{cfg}/S{s}: evaluated with another grid version')
        if ms[s]['reserve'].get('rule_sha256') != rule_sha:                       # reviewer N3
            raise AssertionError(f'{fam}/{cfg}/S{s}: reserve screened with another rule')
        mism[s] = record_matches(ms[s], info['rows'][s])
        inputs[s] = str(rels[s].get('inputs_used', ''))
    rec['n_model_fits_two_seeds'] = nfits
    rec['inputs_used'] = inputs
    rec['record_mismatch'] = mism
    six = all(bool(inputs[s]) and (all(f in inputs[s] for f in C.RAW_FIELDS) or '145' in inputs[s] or 'encoding' in inputs[s].lower()
                                   or 'six permitted raw fields' in inputs[s]) for s in (0, 1))
    if any(mism.values()) or not six:                                              # reviewer N1 / N2: admissibility
        rec['label'] = 'NOT_ADMISSIBLE' + (' (EVAL_MS differs from the counted ledger row)' if any(mism.values()) else ' (inputs not recorded as the six raw fields)')
        return rec
    # alias of a reference: equal MODEL_SELECT wires for both tasks on both seeds
    alias = None
    for name in D.REFERENCE_SET:
        if all(ms[s]['wire_sha256'] == refs_eval[s][name]['wire_sha256'] for s in (0, 1)):
            alias = name
    # (a)
    a_fail = [f"s{s}|{c['task']}|{c['reference']}|{c['metric']}" for s in (0, 1) for c in ms[s]['utility']['clauses'] if not c['point_pass']]
    a_fail += [f"s{s}|guard|{g['group']}" for s in (0, 1) for g in ms[s]['utility']['guards'] if not g['point_pass']]
    # (b)
    two = R.two_seed_status([ms[0]['reserve'], ms[1]['reserve']])
    # c1, c2 and flags
    c1, c2, rec_cells, near, unstable, edge = True, True, {}, False, False, -np.inf      # reviewer N7
    for r in REFS:
        for s in (0, 1):
            b = ms[s]['comparisons'][r]
            pc = b['pair']['cells']
            c1 = c1 and all(v > 0 for v in pc.values())
            for v in ('income', 'occupation'):
                c2 = c2 and all(x <= 0.010 for x in b[v]['cells'].values())
                edge = max(edge, max(b[v]['cells'].values()))
            near = near or b['pair']['mean'] < NEAR_TIE
            unstable = unstable or (max(pc.values()) - min(pc.values())) > b['pair']['mean']
            rec_cells[f'{r}|s{s}'] = {'pair_benefit_cells': pc, 'pair_benefit_mean': b['pair']['mean'], 'pair_benefit_min': b['pair']['min'],
                                      'income_recipient_cells': b['income']['cells'], 'occupation_recipient_cells': b['occupation']['cells']}
    winners = {s: {c: pair_winner(ms[s]['proxy']['cells'][c]) for c in ms[s]['proxy']['cells']} for s in (0, 1)}
    ref_winners = {f'{r}|s{s}': {c: pair_winner(refs_eval[s][r]['proxy']['cells'][c]) for c in refs_eval[s][r]['proxy']['cells']}
                   for r in REFS + ('B2TWIN',) for s in (0, 1)}                                   # reviewer N4
    unstable_sides = []
    if any(len(set(w.values())) > 1 for w in winners.values()):
        unstable = True; unstable_sides.append('configuration winner varies across cells')
    if any(len(set(w.values())) > 1 for k, w in ref_winners.items() if k.split('|')[0] in REFS):
        unstable_sides.append('an eligible reference winner varies across cells')
    if any((max(rec_cells[k]['pair_benefit_cells'].values()) - min(rec_cells[k]['pair_benefit_cells'].values())) > rec_cells[k]['pair_benefit_mean'] for k in rec_cells):
        unstable_sides.append('a benefit cell range exceeds its mean')
    own = {}                                                                                 # reviewer N5: own twin / component control
    for key in ('matched_twin', 'component_control'):
        other = rels[0].get(key) or rels[1].get(key)
        if other and all((D.release_dir(fam, other, s) / 'EVAL_MS.json').is_file() for s in (0, 1)):
            om = {s: json.loads((D.release_dir(fam, other, s) / 'EVAL_MS.json').read_text()) for s in (0, 1)}
            own[key] = {'config': other, 'pair_cells_other_minus_this': {s: {c: cell_values(om[s], 'pair', 'max_incl_single')[c] - cell_values(ms[s], 'pair', 'max_incl_single')[c]
                                                                              for c in ms[s]['proxy']['cells']} for s in (0, 1)}}
        elif other:
            own[key] = {'config': other, 'status': 'not evaluated on both seeds'}
    # c3: below B2TWIN in every cell on both seeds
    c3_cells = {}
    for s in (0, 1):
        mine = cell_values(ms[s], 'pair', 'max_incl_single'); b2 = cell_values(refs_eval[s]['B2TWIN'], 'pair', 'max_incl_single')
        c3_cells[s] = {c: b2[c] - mine[c] for c in mine}
    c3 = all(v > 0 for s in c3_cells for v in c3_cells[s].values())
    # (d) Bonferroni utility bounds over the 28 conditions
    unresolved = [f"s{s}|{c['task']}|{c['reference']}|{c['metric']}" for s in (0, 1) for c in ms[s]['utility']['clauses']
                  if c['value'] + Z_BONF_28 * c['se'] > c['cap']]
    unresolved += [f"s{s}|guard|{g['group']}" for s in (0, 1) for g in ms[s]['utility']['guards'] if g['value'] + Z_BONF_28 * g['se'] > g['cap']]
    score = float(np.mean([v for s in (0, 1) for v in cell_values(ms[s], 'pair', 'max_incl_single').values()]))
    rec.update({'admissible_alias_of': alias, 'a_point_eligible': not a_fail, 'a_failed': a_fail, 'b_reserve': two,
                'c1_pair_superior': c1, 'c2_recipient_screen': c2, 'c3_gain_over_b2twin': c3, 'c3_cells': c3_cells,
                'reference_cells': rec_cells, 'pair_winner_per_cell': winners, 'reference_pair_winner_per_cell': ref_winners,
                'reader_unstable_sides': unstable_sides, 'own_control_cells': own, 'score_two_seed_mean_pair_proxy': score,
                'min_income_ratio': two['min_income_ratio'], 'max_recipient_cell_difference': edge,
                'utility_confidence': 'UTILITY_CONFIDENCE_RESOLVED' if not unresolved else f'UTILITY_CONFIDENCE_UNRESOLVED ({len(unresolved)} of 28)',
                'utility_confidence_unresolved': unresolved,
                'inherited_constraint_report': {s: ms[s]['reserve'].get('inherited_constraint_report') for s in (0, 1)}})
    if near:
        rec['flags'].append('PAIR_NEAR_TIE')
    if unstable:
        rec['flags'].append('READER_UNSTABLE')
    if edge > 0:
        rec['flags'].append(f'RECIPIENT_EDGE (max cell difference {edge:+.4f})')
    if alias:
        rec['label'] = f'ALIAS_OF_{alias}'
    elif a_fail:
        rec['label'] = 'POINT_INELIGIBLE'
    elif two['status'] != 'RESERVE_MET':
        rec['label'] = 'RESERVE_NOT_MET' + (' + EDGE_PAIR_SUPERIOR_WITHOUT_RESERVE' if c1 else '')
    elif not c1:
        rec['label'] = 'UTILITY_HEADROOM_ONLY'
    elif not c2:
        rec['label'] = 'RECIPIENT_SCREEN_FAILED'
    elif not c3:
        rec['label'] = 'NO_GAIN_OVER_B2TWIN'
    else:
        rec['label'] = 'QUALIFYING'
    return rec


def run(write=True):
    refs_eval = {s: D.reference_eval(s) for s in C.SEEDS}
    recs = [evaluate_config(f, c, info, refs_eval) for (f, c), info in sorted(configurations().items())]
    q = [r for r in recs if r['label'] == 'QUALIFYING']
    choice = None
    if q:
        best = min(r['score_two_seed_mean_pair_proxy'] for r in q)
        tie = [r for r in q if r['score_two_seed_mean_pair_proxy'] <= best + NEAR_TIE]
        tie.sort(key=lambda r: (-(r['min_income_ratio'] if r['min_income_ratio'] is not None else -np.inf), r['n_model_fits_two_seeds'], r['seed0_entry']))
        choice = tie[0]
        for r in q:
            r['label'] = 'NOMINATED_FINALIST' if r is choice else 'QUALIFYING_NOT_CHOSEN'
        if choice['kind'] == 'control':
            choice['flags'].append('CONTROL_DESIGNATED_FINALIST')
        choice['near_tie_set'] = [f"{r['family']}/{r['config_id']}" for r in tie]
        choice['near_tie_detail'] = [{'config': f"{r['family']}/{r['config_id']}", 'score': r['score_two_seed_mean_pair_proxy'],
                                      'score_minus_best': r['score_two_seed_mean_pair_proxy'] - best, 'min_income_ratio': r['min_income_ratio'],
                                      'n_model_fits_two_seeds': r['n_model_fits_two_seeds'], 'seed0_entry': r['seed0_entry']} for r in tie]
        choice['outside_near_tie_window'] = [{'config': f"{r['family']}/{r['config_id']}", 'score_minus_best': r['score_two_seed_mean_pair_proxy'] - best}
                                             for r in q if r not in tie]
    ab = [r for r in recs if r.get('a_point_eligible') and (r.get('b_reserve') or {}).get('status') == 'RESERVE_MET']
    frontier = sorted([{'config': f"{r['family']}/{r['config_id']}", 'label': r['label'], 'score_two_seed_mean_pair_proxy': r['score_two_seed_mean_pair_proxy'],
                        'min_income_ratio': r['min_income_ratio']} for r in recs if 'score_two_seed_mean_pair_proxy' in r],
                      key=lambda x: x['score_two_seed_mean_pair_proxy'])
    out = {'schema': 'IHS1-nomination-v1', 'utc': C.now(), 'rule': 'roles/statistical_reviewer/NOMINATION_RULE.md (registered 74376912c)',
           'reserve_rule_sha256': R.load_rule()['_sha256'], 'configurations': recs,
           'nominated': None if choice is None else f"{choice['family']}/{choice['config_id']}",
           'outcome': 'FINALIST_NOMINATED' if choice else 'NO_QUALIFYING_CONFIGURATION',
           'frontier_two_seed': frontier,                                                       # reviewer N6
           'best_diagnostic_package': None if not ab else min(ab, key=lambda r: r['score_two_seed_mean_pair_proxy'])['config_id'],
           'references_marked': {s: {n: refs_eval[s][n]['proxy']['summary']['pair']['mean'] for n in REFS + ('B2TWIN', 'RETIRED_B3CTL')} for s in (0, 1)}}
    if write:
        C.write_json(C.RESULTS / 'NOMINATION.json', out)
        cols = ['family', 'config_id', 'kind', 'label', 'flags', 'score_two_seed_mean_pair_proxy', 'min_income_ratio', 'a_point_eligible',
                'reserve_status', 'c1_pair_superior', 'c2_recipient_screen', 'c3_gain_over_b2twin', 'max_recipient_cell_difference',
                'utility_confidence', 'n_model_fits_two_seeds']
        with open(C.RESULTS / 'NOMINATION_TABLE.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in recs:
                w.writerow({k: (';'.join(r.get('flags', [])) if k == 'flags' else (r.get('b_reserve', {}) or {}).get('status', '') if k == 'reserve_status'
                                else r.get(k, '')) for k in cols})
    print(json.dumps({'outcome': out['outcome'], 'nominated': out['nominated'],
                      'labels': {f"{r['family']}/{r['config_id']}": r['label'] for r in recs}}, indent=1))
    return out


if __name__ == '__main__':
    run()
