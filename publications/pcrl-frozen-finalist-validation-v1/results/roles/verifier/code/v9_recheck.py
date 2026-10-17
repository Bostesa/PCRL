"""Unit VER_RECHECK_V*: final label-free re-check of the reporting documents (check 9) after the coordinator's revisions, plus
the seed-1 conflict role/level assignment published in MAPPING_CONFLICTS.json.

Public-use ACS 2019 Florida person records. No label is read; no person-level value is written (only match flags).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import vcommon as V
from ffv import common as C        # admitted loaders only

DOCS = ('CONFIRMATION_FEASIBILITY.md', 'OUTPUT_MAPPING_CHECK.md', 'RESEARCH_DECISION.md', 'PROTOCOL_DEVIATIONS.md', 'HANDOFF_DECISION.json',
        'START_HERE.md', 'MAPPING_CONFLICTS.json')
# (item, document, phrase that must be ABSENT once corrected, phrase expected when corrected or None)
TESTS = [
    ('V-D1', 'OUTPUT_MAPPING_CHECK.md', 'seed 0: 6 such vectors', 'seed 0: 5 such vectors in the union'),
    ('V-D1', 'OUTPUT_MAPPING_CHECK.md', 'seed 1: 49 such vectors', 'seed 1: 46 such vectors in the union'),
    ('V-D2', 'OUTPUT_MAPPING_CHECK.md', '587 depth-4 income trees and 800 occupation trees per seed', '529 and 799 on seed 1'),
    ('V-D3', 'CONFIRMATION_FEASIBILITY.md', 'at most about 0.2', 'about 0.25 under ORIGINAL'),
    ('V-D3', 'RESEARCH_DECISION.md', 'at most about 0.2', 'about 0.25 under ORIGINAL'),
    ('V-D4', 'CONFIRMATION_FEASIBILITY.md', 'No status label or conclusion changed', 'V2 did change feasibility status labels'),
    ('V-D4', 'RESEARCH_DECISION.md', 'with no label changed', 'some feasibility status labels did'),
    ('V-D4', 'RESEARCH_DECISION.md', 'V2 that changed no label', None),
    ('V-D4', 'PROTOCOL_DEVIATIONS.md', 'no status changed', 'Feasibility status labels did change'),
    ('V-D5', 'CONFIRMATION_FEASIBILITY.md', 'ENHANCED governs; ORIGINAL in brackets where it differs', 'The less favourable scorecard governs'),
    ('V-D6', 'RESEARCH_DECISION.md', "the recovery clauses alone pass at DEV's size", 'ENHANCED recipient clause above does not'),
    ('V-D7', 'RESEARCH_DECISION.md', 'one on each side of the income decision', 'which straddle the income decision'),
    ('V-D8', 'RESEARCH_DECISION.md', 'was read once more', '22:50:37Z'),
]


def main():
    out = {'unit': 'VER_RECHECK', 'labels_read': 'none', 'utc': None}
    texts = {d: (V.RES / d).read_text() if (V.RES / d).is_file() else '' for d in DOCS}
    res = []
    for item, doc, bad, good in TESTS:
        t = texts[doc]
        res.append({'item': item, 'document': doc, 'old_phrase_absent': bad not in t, 'corrected_phrase_present': None if good is None else good in t,
                    'resolved': (bad not in t) and (good is None or good in t)})
    out['phrase_tests'] = res
    out['document_sha256'] = {d: V.sha_file(V.RES / d) for d in DOCS if (V.RES / d).is_file()}
    # MAPPING_CONFLICTS.json: per-role income level of each seed-1 conflict (own recomputation; only match flags are written)
    mc = V.jload(V.RES / 'MAPPING_CONFLICTS.json')
    man = V.jload(V.RES / 'FROZEN_CANDIDATE_MANIFEST.json')
    chk = {}
    for s in V.SEEDS:
        lv = np.array([[float.fromhex(x) for x in row] for row in man['per_seed'][f'S{s}']['per_level_income_probability_hex']])
        lvb = {lv[j].tobytes(): j for j in range(6)}
        with np.load(V.ADM / 'runs' / 'B' / 'B3CTL_UNCOUPLED_MIX50_C6A' / f'S{s}' / 'RELEASE.npz') as z:
            rel = {k: z[k].copy() for k in z.files}
        lev_of, roles_levels, inp_of = defaultdict(set), defaultdict(set), defaultdict(set)
        for r in V.ROLES6:
            X = np.ascontiguousarray(C.load_role(r, ('features',))['X'])
            occ = np.ascontiguousarray(rel[f'{r}__occupation'])
            inc = np.ascontiguousarray(rel[f'{r}__income'])
            for a, b, x in zip(occ, inc, X):
                k = a.tobytes()
                lev_of[k].add(lvb[b.tobytes()])
                roles_levels[k].add((r, lvb[b.tobytes()]))
                inp_of[k].add(x.tobytes())
        mine = sorted(sorted(map(list, roles_levels[k])) for k, v in lev_of.items() if len(v) > 1)
        theirs = sorted(sorted(map(list, c['roles_and_levels'])) for c in mc['per_seed'][f'S{s}']['cross_role_conflicts'])
        sizes = [len(v) for v in inp_of.values() if len(v) > 1]
        p = mc['per_seed'][f'S{s}']
        chk[f'S{s}'] = {'conflict_roles_and_levels_equal': mine == theirs, 'n_conflicts': len(mine),
                        'shared_vectors_equal': len(sizes) == p['union_occupation_vectors_shared_by_distinct_encoded_inputs'],
                        'excess_equal': sum(x - 1 for x in sizes) == p['union_distinct_inputs_minus_distinct_occupation_vectors'],
                        'max_inputs_equal': (max(sizes) if sizes else 0) == p['max_distinct_inputs_on_one_occupation_vector']}
    out['mapping_conflicts_json'] = chk
    import datetime as dt
    out['utc'] = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    V.jdump(V.VPRIV / 'check9_recheck.json', out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
