"""Finalist promotion by the registered rule (PROTOCOL.md section 6), from MODEL_SELECT development evaluations only.

Considers every fitted configuration (family A/B/C) that has an EVAL_MS.json on both seeds. `--candidates all` (the registered
text of PROTOCOL.md section 6, which does not exclude any configuration) or `--candidates non_control` (an unregistered
implementation choice of an interim version that treated ids containing CTL, CTRL or TASKREFIT as comparators only); which one
applies was adjudicated by the statistical reviewer (roles/statistical_reviewer). For each candidate and seed:
- eligible: the 12 original clauses and both within-SEX occupation guards hold at the point on MODEL_SELECT;
- recipient screen: income-only and occupation-only proxies <= each reference's (NEURAL_RAW, H_LJ) proxy + 0.010;
- pair: the pair proxy (maximum over the development proxy readers) is below both references' pair proxies;
- robust: every clause value <= 0.8 x its cap and both guards <= 0.8 x 0.005.
Promotable = eligible, screened and pair-below on both seeds. Choice: robust first, then the lowest mean pair proxy; a choice
without the robustness margin is labelled EDGE_SELECTED. Writes results/<campaign>/PROMOTION.json (all trials considered).
"""
from __future__ import annotations
import json

import numpy as np

from . import common as C
from . import evaluate as EV

REFS = ('NEURAL_RAW', 'H_LJ')
MARGIN_R = 0.010
ROBUST = 0.8


def is_control(cid):
    return any(x in cid.upper() for x in ('CTL', 'CTRL', 'TASKREFIT'))


def main(candidates='all'):
    base = {s: json.loads((C.RUNS / 'BASELINES' / f'BASELINES_MS_S{s}.json').read_text()) for s in C.SEEDS}
    bar = {s: {r: {v: base[s][r][str(s)]['proxy'][v]['max'] for v in ('pair', 'income', 'occupation')} for r in REFS} for s in C.SEEDS}
    trials = []
    for fam_dir in sorted(p for p in C.RUNS.iterdir() if p.is_dir() and p.name in ('A', 'B', 'C')):
        for cfg_dir in sorted(p for p in fam_dir.iterdir() if p.is_dir()):
            ev = {s: cfg_dir / f'S{s}' / 'EVAL_MS.json' for s in C.SEEDS}
            have = {s: p.is_file() for s, p in ev.items()}
            designation = 'control' if is_control(cfg_dir.name) else ('diagnostic' if 'DIAG' in cfg_dir.name.upper() else 'candidate')
            row = {'family': fam_dir.name, 'config_id': cfg_dir.name, 'designation': designation, 'control': is_control(cfg_dir.name),
                   'seeds_evaluated': [s for s in C.SEEDS if have[s]]}
            if not all(have.values()):
                row['status'] = 'NOT_CONSIDERED_ONE_SEED_ONLY'; trials.append(row); continue
            per = {s: json.loads(ev[s].read_text()) for s in C.SEEDS}
            row['per_seed'] = {}
            for s in C.SEEDS:
                m = per[s]; pr = {v: m['proxy'][v]['max'] for v in ('pair', 'income', 'occupation')}
                el = EV.eligibility({s: m}, 1.0); rob = EV.eligibility({s: m}, ROBUST)
                screen = {f'{v}_vs_{r}': bool(pr[v] <= bar[s][r][v] + MARGIN_R) for v in ('income', 'occupation') for r in REFS}
                below = {f'pair_below_{r}': bool(pr['pair'] < bar[s][r]['pair']) for r in REFS}
                row['per_seed'][f'S{s}'] = {'proxy': pr, 'eligible': el['eligible'], 'failed': el['failed'], 'worst_normalized': el['worst_normalized'],
                                            'robust': rob['eligible'], 'robust_failed': rob['failed'], 'recipient_screen': screen, 'pair_below': below,
                                            'pair_benefit_vs_refs': {r: bar[s][r]['pair'] - pr['pair'] for r in REFS}}
            ps = row['per_seed'].values()
            row['eligible_both'] = all(p['eligible'] for p in ps)
            row['screen_both'] = all(all(p['recipient_screen'].values()) for p in ps)
            row['pair_below_both'] = all(all(p['pair_below'].values()) for p in ps)
            row['robust_both'] = all(p['robust'] for p in ps)
            row['promotable'] = bool((candidates == 'all' or not row['control']) and row['eligible_both'] and row['screen_both'] and row['pair_below_both'])
            row['mean_pair_proxy'] = float(np.mean([p['proxy']['pair'] for p in ps]))
            row['status'] = ('PROMOTABLE' if row['promotable'] else 'NOT_PROMOTABLE') + (' (control-designated)' if row['control'] else '')
            trials.append(row)
    prom = [t for t in trials if t.get('promotable')]
    robust = [t for t in prom if t['robust_both']]
    pool = robust or prom
    choice = min(pool, key=lambda t: (t['mean_pair_proxy'], t['config_id'])) if pool else None
    rec = {'schema': 'AMD1-promotion-v1', 'utc': C.now(), 'rule': 'PROTOCOL.md section 6', 'candidate_set': candidates,
           'proxy_definition': 'max: on the pair, the maximum over OUT_LR, OUT_HGB, LOG_MLP and the exact-wire LOOKUP; on each single output, the maximum over OUT_LR, OUT_HGB and LOG_MLP (amd.evaluate.proxies; MODEL_SELECT household halves, both directions averaged)',
           'adjudication': 'roles/statistical_reviewer/DEVELOPMENT_REVIEW.md section 0: the registered text applies to every fitted configuration; the control exclusion was an unregistered, post-hoc implementation choice',
           'superseded_records': [{'utc': '2026-10-17T17:23:54+00:00', 'decision': 'FINALIST_PROMOTED', 'finalist': 'B2_CSR_JT_C6A (EDGE_SELECTED, mean pair proxy 0.61314)',
                                   'candidate_set': 'non_control (unregistered exclusion of ids containing CTL / CTRL / TASKREFIT, written 17:23:49Z after B2 controls were scored)',
                                   'trial_set': 'incomplete (before ledger entries 40-44; family B was still running)',
                                   'status': 'SUPERSEDED; the interim file was generated by the coordinator, never committed, and deleted at 17:33Z before the development commit cf042d113; its content is reproduced here'}],
           'reference_bar_model_select': bar,
           'trials': trials, 'n_trials_considered': len(trials), 'n_promotable': len(prom), 'n_promotable_robust': len(robust),
           'finalist': None if choice is None else {'family': choice['family'], 'config_id': choice['config_id'],
                                                     'selection_label': 'ROBUST_SELECTED' if choice['robust_both'] else 'EDGE_SELECTED',
                                                     'mean_pair_proxy': choice['mean_pair_proxy']},
           'decision': 'FINALIST_PROMOTED' if choice else 'NO_FINALIST_PROMOTED'}
    C.write_json(C.RESULTS / 'PROMOTION.json', rec)
    print(json.dumps({'decision': rec['decision'], 'finalist': rec['finalist'], 'promotable': [t['config_id'] for t in prom],
                      'robust': [t['config_id'] for t in robust]}, indent=1))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--candidates', choices=('all', 'non_control'), required=True)
    main(ap.parse_args().candidates)
