"""Advancement decision inputs and the decision (coordinator; label-free: saved records and private per-person files only).

    python3 -m fsa.fa.advance route_s --out results/pcrl_finite_solver_adequacy_v1/decisions/ROUTE_S_INPUT.json
    python3 -m fsa.fa.advance decide --e <route E input json or none> --s <route S input json> --out decisions/ADVANCEMENT.json

Route S input (RULES.md 4; fsa.rules.route_s): the fitted full-pool objectives of the original and SV1 PRC chains at every registered
strength (source/SOURCE_CHECK_V<v>.json), and the MODEL_FIT nested comparison at the S2 selection
(nested/NESTED_SOLVERS_V<v>.json and its per-person file): the revised and original select_path choices at S2 (nrm.fa.prc.select_path,
the recorded P1 budgets), the original path with its nested total-NLL spends, and paired differences (revised minus original) of the
held-out pair / income / occupation LOOKUP AUCs (paired household bootstrap, B = 400, SeedSequence([2026102200, v, 4]), the same
weights for both entries) and of the total NLL (household-clustered SE). The cached statistics are derived from MODEL_FIT training
labels; MODEL_SELECT is not read.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from nrm.fa import prc as P
from frc.fa import route as FR
from fsa import common as C
from fsa import rules as RU
from fsa.fa import solve as SV

BOOT = 2026102200
D_CLAIMS = [   # the examined approximations (RULES.md 2: documented, therefore not defects); evidence in the audit and exact references
    {'id': 'shrunk_routing_tables', 'approximation': 'SHRUNK_ROUTING_TABLES', 'location': 'nrm/fa/prc.py set_tables / route',
     'evidence': 'along the tilt line of the final refreshed book at strength 8 the routing dependence score and the exact dependence '
                 'term are anti-ranked (Spearman -0.995 / -0.996, PRC; audit)'},
    {'id': 'cell_mean_codeword_refresh', 'approximation': 'CELL_MEAN_CODEWORD_REFRESH', 'location': 'nrm/fa/prc.py update_book',
     'evidence': 'on the real chains the refresh never raised the objective (largest refresh change -9.1e-7 / -7.1e-7); in law B '
                 'stale codewords raised the objective without changing the release (investigator)'},
    {'id': 'b1_emission_vs_prototypes', 'approximation': 'B1_EMISSION_DIFFERS_FROM_SEARCH_PROTOTYPES',
     'location': 'frc/fa/route.py tables', 'evidence': 'largest |B1 table - prototype| 0.0095 / 0.0066 (PRC_S2), 0.032 / 0.054 (HQ_S2); '
                                                       'emitted training cross-entropy <= prototype cross-entropy (audit)'},
    {'id': 'seven_tilt_first_improvement', 'approximation': 'SEVEN_TILT_FIRST_IMPROVEMENT_SEARCH', 'location': 'nrm/fa/prc.py lloyd',
     'evidence': 'improving tilt windows without a power of two are never tried (laws D and E; version 1 at strength 8 on the real '
                 'statistics, window near t = 0.19 between the tried 0.125 and 0.25)'},
]
GRID = SV.GRID


def _nest_from_npz(path, names):
    with np.load(path) as z:
        unit, sex = z['unit'], z['sex']
        nest = {}
        for nm in names:
            keys = [k.split('|', 1)[1] for k in z.files if k.startswith(nm + '|')]
            nest[nm] = {'_per_person': {k: z[f'{nm}|{k}'].copy() for k in keys}}
    return nest, unit, sex


def _boot_weights(unit, v, B=400):
    g, inv = np.unique(unit, return_inverse=True)
    rng = np.random.default_rng(np.random.SeedSequence([BOOT, int(v), 4]))
    W = np.empty((B, len(unit)))
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g))[inv]
    return W


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_equality(v):
    """Engineering review S4: every per-person array of every original-solver entry of the nested run against the unchanged P1
    replay (all shared keys, not three)."""
    nz, rz = SV._priv(f'NESTED_SOLVERS_V{v}.npz'), SV._priv(f'P1_REPLAY_V{v}.npz')
    if not nz.exists() or not rz.exists():
        return {'compared': 0, 'max_abs_diff': None, 'missing': [str(x.name) for x in (nz, rz) if not x.exists()]}
    out, n = 0.0, 0
    with np.load(nz) as a, np.load(rz) as b:
        keys = [k for k in a.files if '|' in k and 'sv1' not in k.split('|')[0] and k in b.files]
        for k in keys:
            out = max(out, float(np.max(np.abs(a[k] - b[k]))))
            n += 1
        same_units = bool(np.array_equal(a['unit'], b['unit']) and np.array_equal(a['sex'], b['sex']))
    return {'compared': n, 'max_abs_diff': out, 'same_units_and_labels': same_units}


def knife_edge(v, rel=1e-9):
    """Engineering review S1: re-route every saved SV1 / SV1_first / original book of the source check with its tilt perturbed by
    +-rel (relative) on the same cached statistics and count the people whose cell changes (a knife-edge tilt makes the deployed
    partition depend on rounding)."""
    import pickle
    from fsa import common as CC
    with open(SV._priv(f'SOURCE_BOOKS_V{v}.pkl'), 'rb') as f:
        books = pickle.load(f)
    mf = CC.features('MODEL_FIT')
    E, Z, unit, names, row_sha = SV.statistics(v, mf)
    res = {}
    for key, bk in books.items():
        k, l = FR.route_any(E, bk)
        moved = 0
        for sgn in (-1, 1):
            b2 = dict(bk)
            b2['tilt'] = float(bk.get('tilt', 0.0)) * (1 + sgn * rel)
            k2, l2 = FR.route_any(E, b2)
            moved = max(moved, int(((k2 != k) | (l2 != l)).sum()))
        res[key] = moved
    return res


def route_s_input():
    rec = {'revision_id': 'SV1', 'label_derived_statistics_disclosed': True,
           'statistics_note': 'cached MODEL_FIT out-of-fold statistics, derived from MODEL_FIT training labels (income, occupation, '
                              'SEX); nested estimates on MODEL_FIT only; MODEL_SELECT not read',
           'fitted': {}, 'nested': {}}
    ledger = [json.loads(l) for l in open(C.PRIV / 'compute' / 'ledger.jsonl')]
    starts = {e['unit_id']: e['utc'] for e in ledger if e.get('event') == 'unit_start'}
    card = C.card_info(SV.CARD, 'FA')
    units = [u for u in starts if u.startswith(('FSA_SOURCE_', 'FSA_NESTED_'))]
    rec['card_commit_utc'] = card['card_commit_utc']
    rec['unit_starts'] = {u: starts[u] for u in sorted(units)}
    import datetime as dt
    cc = dt.datetime.fromisoformat(card['card_commit_utc'])
    rec['card_committed_before_run'] = bool(units) and all(dt.datetime.fromisoformat(starts[u]) > cc for u in units)
    acc = [json.loads(l) for l in open(C.LABEL_LOG)] if C.LABEL_LOG.is_file() else []
    rec['model_select_opened'] = any(a.get('role') == 'MODEL_SELECT' and str(a.get('unit_id', '')).startswith(('FSA_SOURCE_', 'FSA_NESTED_',
                                                                                                                'FSA_REPLAY_', 'FSA_AUDIT_'))
                                     for a in acc)
    eq = []
    for v in C.VERSIONS:
        src = json.loads((C.RESULTS / 'source' / f'SOURCE_CHECK_V{v}.json').read_text())
        o, s = src['runs']['PRC|original']['per_strength'], src['runs']['PRC|SV1']['per_strength']
        rec['fitted'][str(v)] = {str(lam): {'original': o[str(lam)]['objective'], 'revised': s[str(lam)]['objective'],
                                            'revised_deployable': bool(s[str(lam)]['rule_consistent']),
                                            'distinct_from_original': s[str(lam)]['partition_sha256'] != o[str(lam)]['partition_sha256'],
                                            'distinct_from_twin': s[str(lam)]['rerouted_vs_twin'] > 0,
                                            'original_rerouted_vs_twin': o[str(lam)]['rerouted_vs_twin'],
                                            'revised_rerouted_vs_twin': s[str(lam)]['rerouted_vs_twin']}
                                 for lam in GRID}
        nrec = json.loads((C.RESULTS / 'nested' / f'NESTED_SOLVERS_V{v}.json').read_text())
        eq.append(nrec.get('original_entries_max_abs_diff_vs_replay'))
        sel = json.loads((C.FRC_RESULTS / 'fa' / f'SELECTION_V{v}.json').read_text())
        names = list(nrec['means'])
        nest, unit, sex = _nest_from_npz(SV._priv(f'NESTED_SOLVERS_V{v}.npz'), names)
        lab = {'sex': sex}
        orig = [f'PRC@{g}' for g in GRID]
        revd = [f'PRCsv1@{g}' for g in GRID]
        ch_o, _ = P.select_path(nest, lab, unit, orig, budgets={'S2': sel['budgets']['S2']}, ref='PRC_TWIN')
        ch_r, _ = P.select_path(nest, lab, unit, revd, budgets={'S2': sel['budgets']['S2']}, ref='PRC_TWIN')
        tw = nest['PRC_TWIN']['_per_person']
        tot = lambda nm: nest[nm]['_per_person']['occ_nll'] + nest[nm]['_per_person']['inc_nll']
        spend = {nm: float((tot(nm) - (tw['occ_nll'] + tw['inc_nll'])).mean()) for nm in orig + revd}
        R = ch_r.get('S2')
        W = _boot_weights(unit, v)
        paired = {}
        if R is not None:
            dR = {vw: FR._wauc_draws(sex, nest[R]['_per_person'][f'look_{vw}'], W) for vw in ('pair', 'income', 'occupation')}
            aR = {vw: P.auc(nest[R]['_per_person'][f'look_{vw}'], sex) for vw in ('pair', 'income', 'occupation')}
            for nm in orig:
                e = {}
                for vw in ('pair', 'income', 'occupation'):
                    a = P.auc(nest[nm]['_per_person'][f'look_{vw}'], sex)
                    d = dR[vw] - FR._wauc_draws(sex, nest[nm]['_per_person'][f'look_{vw}'], W)
                    e['d_auc' if vw == 'pair' else f'd_auc_{vw}'] = [aR[vw] - a, float(np.std(d, ddof=1))]
                diff = tot(R) - tot(nm)
                e['d_spend'] = [float(diff.mean()), P.cluster_se(diff, unit)]
                paired[nm] = e
        rec['nested'][str(v)] = {'revised_point': R, 'revised_spend': spend.get(R) if R else None, 'original_s2': ch_o.get('S2'),
                                 'recorded_original_s2': sel['selection']['PRC'].get('S2'),
                                 'original_path': [{'id': nm, 'spend': spend[nm]} for nm in orig],
                                 'revised_path_spends': {nm: spend[nm] for nm in revd},
                                 'paired': paired, 'alias_of_original_s2': None}
    rec['original_entries_max_abs_diff_vs_replay_three_keys'] = eq
    full = {str(v): replay_equality(v) for v in C.VERSIONS}
    rec['original_entries_vs_replay_all_keys'] = full
    rec['original_entries_equal_replay'] = all(x['max_abs_diff'] is not None and x['max_abs_diff'] <= RU.REPLAY_TOL
                                               and x.get('same_units_and_labels') for x in full.values())
    rec['private_file_sha256'] = {f.name: _sha(f) for v in C.VERSIONS for f in (SV._priv(f'NESTED_SOLVERS_V{v}.npz'),
                                  SV._priv(f'P1_REPLAY_V{v}.npz'), SV._priv(f'SOURCE_PARTITIONS_V{v}.npz'),
                                  SV._priv(f'SOURCE_BOOKS_V{v}.pkl')) if f.exists()}
    rec['knife_edge_people_moved_at_relative_tilt_1e-9'] = {str(v): knife_edge(v) for v in C.VERSIONS}
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('route_s'); a1.add_argument('--out', required=True)
    a2 = sub.add_parser('decide'); a2.add_argument('--e', required=True); a2.add_argument('--s', required=True)
    a2.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'route_s':
        rec = route_s_input()
        out = Path(a.out)
        if out.exists():
            raise PermissionError(f'{out.name} exists (write-once)')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, sort_keys=True))
        print(json.dumps(RU.route_s(rec), indent=1, default=str)[:3000])
    else:
        e = None if a.e == 'none' else json.loads(Path(a.e).read_text())
        s = json.loads(Path(a.s).read_text())
        if e is not None:                       # route E's source fields come from the measured source check, never placeholders
            e = dict(e)
            e['runnable_on_source'] = True
            sd = {}
            for v in C.VERSIONS:
                fv = s['fitted'][str(v)]
                sd[v] = any(x['revised_deployable'] and x['distinct_from_original'] and x['distinct_from_twin'] for x in fv.values())
            e['source_distinct'] = sd
            e['source_fields_from'] = 'source/SOURCE_CHECK_V<v>.json via the route S input (SV1 PRC chain on the cached MODEL_FIT statistics)'
        rec = {'registered_revisions': ['SV1'], 'attempted': {'D': True, 'E': e is not None, 'S': True}, 'withdrawn': [],
               'D': {'claims': D_CLAIMS, 'note': 'the objective audit (audit/OBJECTIVE_AUDIT_V<v>.json) and the exact-reference '
                     'investigator (code checks on five registered laws) found no behaviour contradicting card FA_PRC_v1 '
                     'section 1 or a documented contract (objective, route, update_book / set_tables, lloyd, _chain, '
                     'protected_path all reproduced); the four documented approximations were examined and are recorded'},
               'E': {'SV1': e} if e is not None else {}, 'S': {'SV1': s}}
        res = RU.advancement(rec)
        out = Path(a.out)
        if out.exists():
            raise PermissionError(f'{out.name} exists (write-once)')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'input': rec, 'result': res, 'utc': C.now()}, indent=1, sort_keys=True, default=str))
        print(json.dumps(res, indent=1, default=str)[:4000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
