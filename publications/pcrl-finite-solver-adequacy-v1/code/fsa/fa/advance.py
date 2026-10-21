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
    rec['original_entries_max_abs_diff_vs_replay'] = eq
    rec['original_entries_equal_replay'] = all(x is not None and x <= RU.REPLAY_TOL for x in eq)
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
        rec = {'registered_revisions': ['SV1'], 'attempted': {'D': False, 'E': e is not None, 'S': True}, 'withdrawn': [],
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
