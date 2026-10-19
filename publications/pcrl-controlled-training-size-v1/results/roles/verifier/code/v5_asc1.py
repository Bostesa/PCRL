"""Verifier stage V5e (after adjudication A-SC1): grid validity under the amended seed-check scope, recomputed with the verifier's own
data, and the verifier's stored reader labels (records/V5_READERS__<tag>.json) compared with conditional/CLAIMS.json.

PCRL controlled training-size campaign (TSZ1; declared development readers only; reader recipes and overlapping splits are not
independent datasets). Label-free (the MODEL_SELECT SEX array was read in V5 READERS; nothing is read here except bank member
releases, which carry no label).

  A-SC1 (roles/statistical_reviewer/ADJUDICATION_SEEDCHECK.md): READER_SEEDS_EFFECTIVE iff (i) 4 distinct recorded seeds outside the
  OHN1 / CTS1 grid seeds; (ii) no seed-dependent equality on W, LOC_SEL, PAIR_SEL; (iii) every other equality explained: (a) another
  seed-dependent recipe differs in the same (wire, view, v, R), (b) the same recipe differs on another view of the same wire in the
  same (v, R), (c) the view's distinct-input count on MODEL_SELECT recorded label-free; (iv) seed-invariant recipes as item 3.
  Distinct-input counts are recomputed here from the bank releases and compared with readers/WIRE_CARDINALITY.json.

    python3 .../v5_asc1.py --readers-tag V1 --tag V1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402
from v5_readers import load_member  # noqa: E402   (own member loader; references through the source's admitted loader)

RES = V.RESULTS
PRIV = V.ROOT / '.private' / 'pcrl_controlled_training_size_v1'
SD = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')
MIN_SET = ('W', 'LOC_SEL', 'PAIR_SEL')
REFS = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ')


def distinct_rows(a):
    return int(np.unique(np.ascontiguousarray(np.asarray(a, dtype=np.float64)), axis=0).shape[0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--readers-tag', required=True); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    rp = V.RECORDS / f'V5_READERS__{a.readers_tag}.json'
    rr = json.loads(rp.read_text())['checks']
    seed_obs = next(c['observed'] for c in rr if c['id'].startswith('V5.READERS.SEED_EFFECTIVENESS'))
    labels = next(c['observed'] for c in rr if c['id'].startswith('V5.READERS.LABELS'))
    sp, cp = RES / 'readers' / 'READER_SEED_CHECK.json', RES / 'readers' / 'WIRE_CARDINALITY.json'
    seed_rec, card = json.loads(sp.read_text()), json.loads(cp.read_text())
    eq = set(seed_obs['seed_dependent_equal'])                       # 'S<v><R>|member|view|recipe'
    base = 2026101205
    item1 = seed_rec['seeds'] == {f'S{v}k{k}': base + 500 + 10 * v + k for v in (0, 1) for k in (0, 1)}
    ii = not [x for x in eq if x.split('|')[1] in MIN_SET]
    # own distinct-input counts from the bank member releases (MODEL_SELECT rows; label-free)
    spec = json.loads((PRIV / 'runs' / 'GRID' / 'COND' / 'S0' / 'GRID.json').read_text())['spec']
    mine_card, card_bad = {}, []
    for name, m in spec['members'].items():
        for v in (0, 1):
            rel = load_member(m, v)
            d = {'income': distinct_rows(rel['income']), 'occupation': distinct_rows(rel['occupation']),
                 'pair': distinct_rows(np.column_stack((rel['income'], rel['occupation'])))}
            mine_card[f'{name}|V{v}'] = d
            if card['distinct_input_vectors'].get(f'{name}|V{v}') != d:
                card_bad.append(f'{name}|V{v}: mine {d} vs {card["distinct_input_vectors"].get(f"{name}|V{v}")}')
    explained, unexplained = [], []
    for x in sorted(eq):
        cell, member, view, recipe = x.split('|')
        a_ = any(f'{cell}|{member}|{view}|{r}' not in eq for r in SD if r != recipe)
        b_ = any(f'{cell}|{member}|{w}|{recipe}' not in eq for w in ('income', 'occupation', 'pair') if w != view)
        n = mine_card.get(f'{member}|V{cell[1]}', {}).get(view)
        (explained if (a_ and b_ and isinstance(n, int)) else unexplained).append({'check': x, 'iii_a': a_, 'iii_b': b_, 'distinct_inputs': n})
    iv_ok = not seed_obs['seed_invariant_differ']
    effective = bool(item1 and ii and not unexplained)
    cl_p = RES / 'conditional' / 'CLAIMS.json'
    cl = json.loads(cl_p.read_text())
    prod_val = cl.get('reader_seed_check') or {}
    # own labels vs CLAIMS.json comparisons (the stored V5 labels were computed WITHOUT a validity gate: numerically equal to the
    # labels under an effective grid, i.e. under A-SC1)
    lab_bad, n_cmp = [], 0
    for key, x in labels.items():
        t = cl['comparisons'].get(key)
        if not t or 'pair' not in t:
            continue
        n_cmp += 1
        p = t['pair']
        if p.get('label') != x['stability'] or p.get('n_eff') != x['n_eff'] or p.get('reader_seed_qualifier') != x['qualifier'] \
                or abs((p.get('mean') or 0.0) - x['mean']) > 1e-12 or abs((p.get('min') or 0.0) - x['min']) > 1e-12:
            lab_bad.append(f'{key}: mine {x["stability"]}/{x["n_eff"]}/{x["qualifier"]} vs {p.get("label")}/{p.get("n_eff")}/{p.get("reader_seed_qualifier")}')
        tt = (t.get('target') or {}).get('label')
        if tt is not None and tt != x['target_label']:
            lab_bad.append(f'{key}: target {x["target_label"]} vs {tt}')
        gates = t.get('recipient_gate') or {}
        if gates:
            g_all = all((gates.get(vw) or {}).get('pass') is True for vw in ('income', 'occupation'))
            if g_all != x['old_recipient_gate_pass']:
                lab_bad.append(f'{key}: recipient gate {x["old_recipient_gate_pass"]} vs {g_all}')
    chk1 = {'id': f'V5.READERS.ASC1_VALIDITY__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
            'what': 'grid validity under adjudication A-SC1 recomputed from the verifier\'s own cache comparison (all 2,688 checks) and own '
                    'distinct-input counts of every bank member release (vs readers/WIRE_CARDINALITY.json): item 1 seeds, (ii) minimum set, '
                    '(iii a-c) every other equality explained, (iv) seed-invariant recipes; vs the validity recorded in CLAIMS.json',
            'inputs': V.inputs_sha([rp, sp, cp, cl_p]), 'expected': {'literal_registered': seed_rec['rule']['label'], 'amended_A-SC1': 'per CLAIMS.json'},
            'observed': {'item1_seeds': item1, 'ii_minimum_set_no_equality': ii, 'explained': len(explained), 'unexplained': unexplained,
                         'iv_seed_invariant_equal': iv_ok, 'amended_label_mine': 'READER_SEEDS_EFFECTIVE' if effective else 'READER_SEEDS_NOT_EFFECTIVE',
                         'literal_label_mine': 'READER_SEEDS_NOT_EFFECTIVE' if eq else 'READER_SEEDS_EFFECTIVE',
                         'distinct_input_mismatches': card_bad, 'coarse_wires': {k: v for k, v in mine_card.items() if min(v.values()) < 100},
                         'claims_record_validity': prod_val},
            'status': 'PASS' if (not card_bad and (not prod_val or prod_val.get('amended_A-SC1') == ('READER_SEEDS_EFFECTIVE' if effective else 'READER_SEEDS_NOT_EFFECTIVE'))) else 'FAIL',
            'code_sha256': V.code_hashes(__file__, V.__file__)}
    chk2 = {'id': f'V5.READERS.LABELS_VS_CLAIMS__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
            'what': 'the verifier\'s stored reader labels (V5.READERS.LABELS__' + a.readers_tag + ', computed 01:33:09Z without any validity gate, '
                    'i.e. numerically the labels of an effective grid = the A-SC1 status; under the literal registered status no label would '
                    'exist) vs CLAIMS.json comparisons: stability label, n_eff, reader-seed qualifier, mean, min, target label, old recipient '
                    'gate, for every pair present in both',
            'inputs': V.inputs_sha([rp, cl_p]), 'expected': 'agreement', 'observed': {'n_compared': n_cmp, 'problems': lab_bad},
            'status': 'PASS' if (n_cmp > 0 and not lab_bad) else 'FAIL', 'code_sha256': V.code_hashes(__file__, V.__file__)}
    V.write_record(f'V5_ASC1__{a.tag}.json', {'checks': [chk1, chk2], 'utc': V.utc()})
    print(json.dumps(V.append_checks([chk1, chk2])))
    print(json.dumps({chk1['id']: [chk1['status'], chk1['observed']['amended_label_mine'], len(explained), unexplained, card_bad[:3], prod_val.get('amended_A-SC1') if prod_val else None],
                      chk2['id']: [chk2['status'], n_cmp, lab_bad[:10]]}, default=str))


if __name__ == '__main__':
    main()
