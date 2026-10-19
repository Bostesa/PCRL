"""Verifier stage V6a: conditional-stage utility on MODEL_SELECT and the claim labels C1-C8, recomputed with the verifier's own code
and compared with conditional/COND_UTILITY.json and conditional/CLAIMS.json; claim scope sentences checked in the records.

PCRL controlled training-size campaign (TSZ1; exploratory development on previously used ACS 2019 Florida PUMS person records;
MODEL_SELECT is adaptively used development data; declared development readers only).

Inputs: MODEL_SELECT income, occupation, SEX (tsz.common.load operation 'verify', read only after both selection records are
committed), the frozen references U0 / Ucal (frozen bank), H3 income S<v>, the corrected releases (runs/CORRECT/V<v>/RELEASES.npz,
hash-checked), W100_v, the six inherited strong references (the source's admitted loader ohn.devreaders.load_member), the
selections, the verifier's own reader-label record (records/V5_READERS__<tag>.json) and own size record (V4_SIZE_MINE__<tag>.json).

    python3 -m tsz.jobqueue run --unit VER_CLAIMS_V1 --stage verification --rss-gib 3 -- python3 .../v6_claims.py --readers-tag V1 \
        --size-tag V1 --tag V1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from tsz import common as C  # noqa: E402   (the logged label-access layer; operation 'verify')

RES = V.RESULTS
PRIV = ROOT / '.private' / 'pcrl_controlled_training_size_v1'
CORRECT = PRIV / 'runs' / 'CORRECT'
SIZE = PRIV / 'runs' / 'SIZE'
CONTROL = PRIV / 'runs' / 'CONTROL'
REFS6 = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ')
SCOPE_SENTENCES = ('Occupation-only optimization is not a new two-output joint method.',
                   'Lower recovery for declared readers is not arbitrary-reader protection.',
                   'Do not claim a population learning curve, independent replication or proof that size caused every historical difference.',
                   'reader recipes and overlapping splits are not independent datasets')


def skey(s):
    return repr(float(s))


def npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def utility_c2(sel, ut, with_reserve):
    pre = 'UTILITY_' if with_reserve else 'UTILITY_POINT_'
    nonzero = [v for v in (0, 1) if sel[v] > 0]
    if not nonzero:
        return 'NO_NONZERO_SELECTION'
    for v in nonzero:
        if not (ut[v]['point_pass'] and (ut[v]['occ_reserve_met'] or not with_reserve)):
            return pre + 'NOT_PRESERVED'
    return pre + ('PRESERVED_BOTH_VERSIONS' if len(nonzero) == 2 else 'PRESERVED_ONE_VERSION')


def protection(stab, name, both):
    if stab != 'STABLE_POSITIVE':
        return 'NOT_MET'
    return name if both else 'MET_WITHOUT_UTILITY'


def realized_costs(Z, W, y, sex, unit, strengths):
    """Own realized MODEL_SELECT increments of every arm at every strength against W (s = 0), means and clustered / subgroup SEs."""
    men, women = sex == 0, sex == 1
    out = {}
    for arm in ('CONT', 'LOC', 'PAIR'):
        out[arm] = {}
        for s in strengths:
            p = W if float(s) == 0.0 else Z[f'{arm}__s{skey(s)}__MODEL_SELECT']
            a_, b_ = V.person_vectors(y, p, 5), V.person_vectors(y, W, 5)
            db, dn, da = a_['brier'] - b_['brier'], a_['nll'] - b_['nll'], b_['accuracy'] - a_['accuracy']
            out[arm][skey(s)] = {'stats': {'men_brier': float(db[men].mean()), 'women_brier': float(db[women].mean()), 'brier': float(db.mean()),
                                           'nll': float(dn.mean()), 'acc_loss': float(da.mean())},
                                 'se': {'men_brier': V.subgroup_se(db, men, unit), 'women_brier': V.subgroup_se(db, women, unit),
                                        'brier': V.cluster_se(db, unit), 'nll': V.cluster_se(dn, unit), 'acc_loss': V.cluster_se(da, unit)}}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--readers-tag', default=None)
    ap.add_argument('--size-tag', default=None)
    ap.add_argument('--utility-only', action='store_true', help='MODEL_SELECT utility / reserve / realized costs only; no claim label')
    a = ap.parse_args()
    sels = {}
    for v in (0, 1):
        p = RES / 'conditional' / f'SELECTION_V{v}.json'
        if not C.committed(p):
            raise SystemExit(f'{p.name} is not committed: no MODEL_SELECT read of a corrected release before it')
        sels[v] = json.loads(p.read_text())
    S = C.load('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), 'verify',
               'independent verifier V6: utility and reserve of the selected releases and references after the committed selections; claim labels')
    from ohn import common as OC
    from ohn import devrun as OD
    from ohn import devreaders as ODR
    floors = json.loads((ROOT / V.OHN1_RULES).read_text())['reserve']['floor']['se']
    unit, sex = np.asarray(S['unit']), np.asarray(S['sex'])
    labels = {'income': np.asarray(S['income']), 'occupation': np.asarray(S['occupation'])}
    G = len(np.unique(unit))
    util, refs_util, inputs, realized = {}, {}, [], {}
    for v in (0, 1):
        bank, ids = OC.frozen_bank(v, roles=('MODEL_SELECT',))
        if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']):
            raise AssertionError('row order differs')
        refs = {r: {t: np.asarray(bank[r]['MODEL_SELECT'][t], dtype=np.float64) for t in ('income', 'occupation')} for r in ('U0', 'Ucal')}
        inc = np.asarray(OD.anchor_income('H3', v, 'MODEL_SELECT'), dtype=np.float64)
        rec = json.loads((CORRECT / f'V{v}' / 'RELEASES.json').read_text())
        if V.file_sha(CORRECT / f'V{v}' / 'RELEASES.npz') != rec['npz_sha256']:
            raise PermissionError('RELEASES.npz differs from its record')
        Z = npz(CORRECT / f'V{v}' / 'RELEASES.npz')
        W = npz(SIZE / f'A1__S100__V{v}' / 'RELEASE.npz')['MODEL_SELECT']
        inputs += [CORRECT / f'V{v}' / 'RELEASES.npz', SIZE / f'A1__S100__V{v}' / 'RELEASE.npz']
        s_loc, s_pair = float(sels[v]['selected']['LOC']), float(sels[v]['selected']['PAIR'])
        named = {'W': ('W', 0.0), 'LOC_SEL': ('LOC', s_loc), 'PAIR_SEL': ('PAIR', s_pair), 'CONT_AT_LOC': ('CONT', s_loc),
                 'CONT_AT_PAIR': ('CONT', s_pair), 'LOC_AT_PAIR': ('LOC', s_pair)}
        rc_mine = realized_costs(Z, W, labels['occupation'], sex, unit, rec['strengths'])
        realized.setdefault('V', {})[v] = rc_mine
        for name, (arm, s) in named.items():
            occ = W if (arm == 'W' or s == 0.0) else Z[f'{arm}__s{skey(s)}__MODEL_SELECT']
            conds = V.clauses_and_guards(labels, {'income': inc, 'occupation': occ}, refs, sex, unit)
            ro = V.reserve(conds, floors[str(v)], G, names=V.OCC_SIDE)
            rw = V.reserve(conds, floors[str(v)], G, names=V.ALL_CONDITIONS)
            util.setdefault(name, {})[v] = {'arm': arm, 'strength': s, 'point_pass': all(c['point_pass'] for c in conds.values()),
                                            'occ_reserve_met': ro['met'], 'min_occ_ratio': ro['min_ratio'], 'binding': ro['binding'],
                                            'whole_release_reserve_met': rw['met'], 'min_whole_ratio': rw['min_ratio'],
                                            'identity_equals_W': bool(np.array_equal(occ, W)), 'occ_sha256': V.arr_sha(occ)}
        for r in REFS6:
            relr = ODR.load_member({'type': 'reference', 'name': r}, v)
            conds = V.clauses_and_guards(labels, {'income': np.asarray(relr['income'], dtype=np.float64),
                                                  'occupation': np.asarray(relr['occupation'], dtype=np.float64)}, refs, sex, unit)
            ro = V.reserve(conds, floors[str(v)], G, names=V.OCC_SIDE)
            rw = V.reserve(conds, floors[str(v)], G, names=V.ALL_CONDITIONS)
            refs_util.setdefault(r, {})[v] = {'point_pass_12_plus_2': all(c['point_pass'] for c in conds.values()),
                                              'whole_release_reserve_met': rw['met'], 'min_whole_ratio': rw['min_ratio'],
                                              'occ_reserve_met': ro['met'], 'min_occ_ratio': ro['min_ratio']}
    # compare with COND_UTILITY.json
    cu_p = RES / 'conditional' / 'COND_UTILITY.json'
    cu = json.loads(cu_p.read_text())
    ub = []
    for name, per in util.items():
        for v, x in per.items():
            q = cu['releases'][name][str(v)]
            for k in ('point_pass', 'occ_reserve_met', 'whole_release_reserve_met', 'identity_equals_W'):
                if x[k] != q[k]:
                    ub.append(f'{name}|V{v}|{k}: {x[k]} vs {q[k]}')
            for k in ('min_occ_ratio', 'min_whole_ratio'):
                if abs(x[k] - q[k]) > 1e-9:
                    ub.append(f'{name}|V{v}|{k}: {x[k]} vs {q[k]}')
            if x['occ_sha256'] != q['occ_sha256']:
                ub.append(f'{name}|V{v}|occ_sha256')
    for r, per in refs_util.items():
        for v, x in per.items():
            q = cu['references'][r][str(v)]
            for k in ('point_pass_12_plus_2', 'whole_release_reserve_met', 'occ_reserve_met'):
                if x[k] != q[k]:
                    ub.append(f'{r}|V{v}|{k}: {x[k]} vs {q[k]}')
            for k in ('min_whole_ratio', 'min_occ_ratio'):
                if abs(x[k] - q[k]) > 1e-9:
                    ub.append(f'{r}|V{v}|{k}: {x[k]} vs {q[k]}')
    rc_max = {'stats': 0.0, 'se': 0.0}
    for v, per_arm in realized.get('V', {}).items():
        for arm, per_s in per_arm.items():
            for sk, row in per_s.items():
                q = cu['realized_costs'][arm][str(v)][sk]
                for j in V.STATS5:
                    rc_max['stats'] = max(rc_max['stats'], abs(row['stats'][j] - q['stats'][j]))
                    rc_max['se'] = max(rc_max['se'], abs(row['se'][j] - q['se'][j]))
    if rc_max['stats'] > 1e-12 or rc_max['se'] > 1e-12:
        ub.append(f'realized costs differ: {rc_max}')
    if a.utility_only:
        code = V.code_hashes(__file__, V.__file__)
        sel_status = {name: {str(v): {k: util[name][v][k] for k in ('strength', 'point_pass', 'occ_reserve_met', 'min_occ_ratio', 'binding',
                                                                     'whole_release_reserve_met', 'min_whole_ratio', 'identity_equals_W')}
                             for v in (0, 1)} for name in util}
        checks = [{'id': f'V6.COND_UTILITY__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
                   'what': 'MODEL_SELECT utility of the selected corrected releases (identity included), matched continuations, the witness and LOC '
                           'at PAIR\'s strength, and of the six inherited references: 12 clauses + 2 guards at the point, occupation-side and '
                           'whole-release reserve (k = 2, OHN1 floors), minimum ratios; realized increments of CONT / LOC / PAIR at every '
                           'registered strength against W (means and household-clustered SEs); own code vs COND_UTILITY.json (no reader label)',
                   'inputs': V.inputs_sha(inputs + [cu_p]), 'expected': 'agreement',
                   'observed': {'releases': sel_status, 'references': refs_util, 'realized_cost_max_abs_diff': rc_max, 'problems': ub},
                   'status': 'PASS' if not ub else 'FAIL', 'code_sha256': code}]
        import os
        V.write_record(f'V6_COND_UTILITY__{a.tag}__{os.environ.get("TSZ_UNIT_ID") or "local"}.json', {'checks': checks, 'utc': V.utc()})
        print(json.dumps(V.append_checks(checks)))
        print(json.dumps({checks[0]['id']: checks[0]['status'], 'problems': ub[:20], 'rc_max': rc_max,
                          'selected_status': {n: {v: (x['point_pass'], x['occ_reserve_met'], round(x['min_occ_ratio'], 3)) for v, x in d.items()}
                                              for n, d in sel_status.items()}}, default=str))
        return
    # claims from own components
    rl = {}
    for c in json.loads((V.RECORDS / f'V5_READERS__{a.readers_tag}.json').read_text())['checks']:
        if c['id'].startswith('V5.READERS.LABELS'):
            rl = c['observed']
    size_rec = json.loads((V.RECORDS / f'V4_SIZE_MINE__{a.size_tag}.json').read_text())['mine']
    trig = json.loads((RES / 'trigger' / 'TRIGGER.json').read_text())
    sel = {arm: {v: float(sels[v]['selected'][arm]) for v in (0, 1)} for arm in ('LOC', 'PAIR')}
    lab = lambda key: (rl.get(key) or {}).get('stability')
    claims = {'C1': 'MARGIN_RESTORED_BY_TRAINING_SIZE_ON_THIS_POOL' if size_rec['size_explanation'] == 'SUPPORTED_ON_THIS_POOL' else 'NOT_ESTABLISHED',
              'C7': 'NOT_CLAIMABLE', 'C8': 'NOT_CLAIMED', 'arms': {}}
    qual = {}
    for arm, name in (('LOC', 'LOC_SEL'), ('PAIR', 'PAIR_SEL')):
        ut = {v: {'point_pass': util[name][v]['point_pass'], 'occ_reserve_met': util[name][v]['occ_reserve_met']} for v in (0, 1)}
        c2 = utility_c2(sel[arm], ut, True); c2p = utility_c2(sel[arm], ut, False)
        both = c2 == 'UTILITY_PRESERVED_BOTH_VERSIONS'
        r = {'C2': c2, 'C2p': c2p, 'C3': protection(lab(f'{name}|CONT_AT_{arm}'), 'RECOVERY_BELOW_CONTINUATION', both),
             'C3w': protection(lab(f'{name}|W'), 'RECOVERY_BELOW_WITNESS', both)}
        if arm == 'PAIR':
            r['C4'] = protection(lab('PAIR_SEL|LOC_SEL'), 'RECOVERY_BELOW_LOCAL_CORRECTION', both)
        point_all = all(util[name][v]['point_pass'] for v in (0, 1))
        gate_all = all((rl.get(f'{name}|{x}') or {}).get('old_recipient_gate_pass') is True for x in REFS6)
        stable_all = all(lab(f'{name}|{x}') == 'STABLE_POSITIVE' for x in REFS6)
        met5 = point_all and gate_all and stable_all
        r['C5'] = ('COMPETITIVE_WITH_ALL_REQUIRED' if both else 'MET_WITHOUT_UTILITY') if met5 else 'NOT_MET'
        order = ('DEV_GRID_TARGET_NOT_MET', 'DEV_GRID_TARGET_POINT_ONLY', 'DEV_GRID_TARGET_STABLE')
        tl = [(rl.get(f'{name}|{x}') or {}).get('target_label') for x in ('NEURAL_RAW', 'H_LJ')]
        r['C6'] = min((t if t in order else 'DEV_GRID_TARGET_NOT_MET' for t in tl), key=order.index)
        keys = ('C3', 'C3w') + (('C4',) if arm == 'PAIR' else ())
        r['qualifies'] = both and all(r[k] not in ('NOT_MET', 'MET_WITHOUT_UTILITY') for k in keys)
        claims['arms'][arm] = r
        qual[arm] = r['qualifies']
    claims['outcome'] = ('NOT_RUN_TRIGGER_NOT_FIRED' if not trig['trigger']['fires'] else
                         'PRIMARY_ARM_QUALIFIES' if qual['LOC'] else 'SECONDARY_ARM_ONLY' if qual['PAIR'] else 'CORRECTION_RECIPE_CLOSED')
    cl_p = RES / 'conditional' / 'CLAIMS.json'
    cl = json.loads(cl_p.read_text())
    pc = cl['claims']
    cb = []
    if claims['C1'] != pc['C1']['label']:
        cb.append(f"C1 {claims['C1']} vs {pc['C1']['label']}")
    if claims['outcome'] != pc.get('outcome'):
        cb.append(f"outcome {claims['outcome']} vs {pc.get('outcome')}")
    for arm in ('LOC', 'PAIR'):
        mine, theirs = claims['arms'][arm], pc['arms'][arm]
        for k in ('C2', 'C2p'):
            if mine[k] != theirs[k]:
                cb.append(f'{arm} {k} {mine[k]} vs {theirs[k]}')
        for k in ('C3', 'C3w', 'C4', 'C5', 'C6'):
            if k in mine and mine[k] != (theirs[k]['label'] if isinstance(theirs.get(k), dict) else theirs.get(k)):
                cb.append(f'{arm} {k} {mine[k]} vs {theirs.get(k)}')
        if mine['qualifies'] != theirs['qualifies']:
            cb.append(f'{arm} qualifies {mine["qualifies"]} vs {theirs["qualifies"]}')
    # labels of every compared pair vs CLAIMS.json comparisons
    lb = []
    for key, x in rl.items():
        t = (cl['comparisons'].get(key) or {}).get('pair') or {}
        if t and t.get('label') != x['stability']:
            lb.append(f'{key}: {x["stability"]} vs {t.get("label")}')
        if t and (abs((t.get('mean') or 0) - (x['mean'] or 0)) > 1e-12 or t.get('n_eff') != x['n_eff']):
            lb.append(f'{key}: mean / n_eff differ')
        tt = (cl['comparisons'].get(key) or {}).get('target')
        if tt and tt.get('label') != x['target_label']:
            lb.append(f'{key}: target {x["target_label"]} vs {tt.get("label")}')
    # scope sentences present in the claims record
    scope_text = json.dumps(cl)
    missing_scope = [s for s in SCOPE_SENTENCES if s.lower().rstrip('.') not in scope_text.lower()]
    code = V.code_hashes(__file__, V.__file__)
    checks = [
        {'id': f'V6.COND_UTILITY__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
         'what': 'MODEL_SELECT utility of the selected corrected releases (identity included), matched continuations, the witness and LOC at '
                 'PAIR\'s strength, and of the six inherited references: 12 clauses + 2 guards at the point, occupation-side and whole-release '
                 'reserve (k = 2, OHN1 floors), minimum ratios, own code vs COND_UTILITY.json',
         'inputs': V.inputs_sha(inputs + [cu_p]), 'expected': 'agreement', 'observed': {'releases': util, 'references': refs_util, 'problems': ub},
         'status': 'PASS' if not ub else 'FAIL'},
        {'id': f'V6.CLAIMS__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
         'what': 'claim labels C1-C8, arm qualification and the closure outcome recomputed from the verifier\'s own components (own size '
                 'explanation, own utility, own reader stability / target / recipient-gate labels, the committed selections) vs CLAIMS.json; '
                 'every compared pair\'s stability label, mean, n_eff and target label vs CLAIMS.json comparisons',
         'inputs': V.inputs_sha([cl_p, V.RECORDS / f'V5_READERS__{a.readers_tag}.json', V.RECORDS / f'V4_SIZE_MINE__{a.size_tag}.json']),
         'expected': {'outcome': pc.get('outcome')}, 'observed': {'mine': claims, 'claim_problems': cb, 'pair_label_problems': lb},
         'status': 'PASS' if not (cb or lb) else 'FAIL'},
        {'id': f'V6.CLAIM_SCOPE_TEXT__{a.tag}', 'kind': 'PROCEDURAL_COMPLIANCE',
         'what': 'the registered scope sentences (RULES.md sections 3.6, 11, 12) are carried by the claims record',
         'inputs': V.inputs_sha([cl_p]), 'expected': list(SCOPE_SENTENCES), 'observed': {'missing': missing_scope},
         'status': 'PASS' if not missing_scope else 'ATTENTION'}]
    for c in checks:
        c['code_sha256'] = code
    import os
    V.write_record(f'V6_CLAIMS__{a.tag}__{os.environ.get("TSZ_UNIT_ID") or "local"}.json', {'checks': checks, 'utc': V.utc()})
    print(json.dumps(V.append_checks(checks)))
    print(json.dumps({c['id']: c['status'] for c in checks}))
    print(json.dumps({'mine': claims, 'util_problems': ub[:20], 'claim_problems': cb, 'pair_problems': lb[:20], 'missing_scope': missing_scope}, default=str)[:5000])


if __name__ == '__main__':
    main()
