"""Verifier stage V5 (label-free comparison): the verifier's own reader-seed effectiveness findings (records/V5_READERS__<tag>.json,
from the reader cache) vs results/<campaign>/readers/READER_SEED_CHECK.json, and the label the registered rule (RULES.md 9.2) gives on
the checked set.

PCRL controlled training-size campaign (TSZ1).

    python3 .../v5_seedcheck_agree.py --readers-tag V1 --tag V1
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
SELECTED = ('W', 'LOC_SEL', 'PAIR_SEL', 'CONT_AT_LOC', 'CONT_AT_PAIR', 'LOC_AT_PAIR')
SEED_DEP = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--readers-tag', required=True); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    rp = V.RECORDS / f'V5_READERS__{a.readers_tag}.json'
    mine = None
    for c in json.loads(rp.read_text())['checks']:
        if c['id'].startswith('V5.READERS.SEED_EFFECTIVENESS'):
            mine = c['observed']
    sp = RES / 'readers' / 'READER_SEED_CHECK.json'
    rec = json.loads(sp.read_text())
    theirs = sorted(f"S{x['v']}{x['R']}|{x['wire']}|{x['view']}|{x['recipe']}" for x in rec['rule']['ineffective_checks'])
    mine_eq = sorted(mine['seed_dependent_equal'])
    agree = theirs == mine_eq
    seeds = rec.get('seeds') or {}
    base = 2026101205
    seeds_ok = seeds == {f'S{v}k{k}': base + 500 + 10 * v + k for v in (0, 1) for k in (0, 1)}
    covered = sorted({x['recipe'] for x in rec.get('per_recipe', [])}) if isinstance(rec.get('per_recipe'), list) else None
    in_selected = [x for x in mine_eq if x.split('|')[1] in SELECTED]
    label_rule_all_checked = 'READER_SEEDS_NOT_EFFECTIVE' if mine_eq else 'READER_SEEDS_EFFECTIVE'
    label_rule_min_set = 'READER_SEEDS_NOT_EFFECTIVE' if in_selected else 'READER_SEEDS_EFFECTIVE'
    chk = {'id': f'V5.READERS.SEED_CHECK_AGREES__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': 'the verifier\'s own cache-based list of seed-dependent recipes with bitwise-equal k0 / k1 SCORE scores vs READER_SEED_CHECK.json '
                   'ineffective_checks; reader seeds; the RULES.md 9.2 label on the production checked set (every bank member) and on the minimum '
                   'set the rule names (W100_v and every selected release)',
           'inputs': V.inputs_sha([rp, sp]), 'expected': {'production_label': rec['rule']['label'], 'production_ineffective': theirs},
           'observed': {'verifier_equalities': mine_eq, 'lists_agree': agree, 'equalities_in_W_or_selected_releases': in_selected,
                        'label_on_all_checked_members': label_rule_all_checked, 'label_on_minimum_set_W_and_selected': label_rule_min_set,
                        'reader_seeds_recorded_as_registered': seeds_ok, 'n_checks_production': rec.get('n_checks'),
                        'missing_entries_production': rec.get('missing_entries')},
           'status': 'PASS' if (agree and seeds_ok and label_rule_all_checked == rec['rule']['label']) else 'FAIL',
           'note': 'agreement check; whether the registered check set is every member (production) or the minimum set is a rule-scope '
                   'question for the coordinator / statistical reviewer: the equalities are all income-view ET5 / ET25 of the references '
                   'B2TWIN and RETIRED_B3CTL', 'code_sha256': V.code_hashes(__file__, V.__file__)}
    V.write_record(f'V5_SEEDCHECK_AGREE__{a.tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], **{k: chk['observed'][k] for k in ('lists_agree', 'label_on_all_checked_members',
                                                                                  'label_on_minimum_set_W_and_selected', 'reader_seeds_recorded_as_registered')}}))


if __name__ == '__main__':
    main()
