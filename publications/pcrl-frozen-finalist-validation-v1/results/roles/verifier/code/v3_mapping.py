"""Unit VER_MAPPING_V1 (check 3: output mapping). Label-free: no SEX or task label is read.

Public-use ACS 2019 Florida person records. Own recomputation, from the admitted release arrays and encoded inputs, of the
per-role and union exact-byte counts and the seed-1 cross-role conflicts; own recomputation of the occupation, income and joint
routing-cell counts from the frozen teacher JSONs and the package encoder; confirmation of reported invented counterexamples
through the frozen package (amd.fam_b.package_release, allowed by the brief) and through an own routing computation; and an own
bounded search on invented inputs with a different random seed. No person-level value is written to any output.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import vcommon as V
from ffv import common as C        # admitted loaders only
from v1_identity import teacher, raw_scores, softmax_occ, load_npz

CFG = 'B3CTL_UNCOUPLED_MIX50_C6A'
STATE_FIPS = ['001', '002', '004', '005', '006', '008', '009', '010', '011', '012', '013', '015', '016', '017', '018', '019', '020',
              '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037',
              '038', '039', '040', '041', '042', '044', '045', '046', '047', '048', '049', '050', '051', '053', '054', '055', '056']
RAW = ('AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP')


def rows_bytes(a):
    a = np.ascontiguousarray(a)
    return [r.tobytes() for r in a]


def wire_bytes(p):
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    d = np.argmax(p, axis=1).astype(np.int64)
    return [a.tobytes() + b.tobytes() for a, b in zip(p, d)]


def empirical(seed, feats, levelvec):
    rel = load_npz(V.ADM / 'runs' / 'B' / CFG / f'S{seed}' / 'RELEASE.npz')
    lvb = {levelvec[j].tobytes(): j for j in range(6)}
    per, U = {}, defaultdict(list)
    names = feats['DEV_REPORT']['feature_names']
    for r in V.ROLES6:
        X = feats[r]['X']
        occ, inc = rel[f'{r}__occupation'], rel[f'{r}__income']
        ko, ki, kx = wire_bytes(occ), wire_bytes(inc), rows_bytes(X)
        lev = np.array([lvb.get(row.tobytes(), -1) for row in np.ascontiguousarray(inc)])
        cnt = defaultdict(int)
        lev_of, inp_of, occ_of_inp = defaultdict(set), defaultdict(set), defaultdict(set)
        for a, b, l in zip(ko, kx, lev):
            cnt[a] += 1
            lev_of[a].add(int(l))
            inp_of[a].add(b)
            occ_of_inp[b].add(a)
        c = np.array(list(cnt.values()))
        rec = {'records': len(ko), 'distinct_encoded_input_vectors': len(set(kx)), 'distinct_occupation_vectors': len(cnt),
               'distinct_income_vectors': len(set(ki)), 'distinct_linked_pairs': len(set(i + o for i, o in zip(ki, ko))),
               'unmatched_income_vectors': int((lev < 0).sum()),
               'occupation_vectors_with_more_than_one_income_code': int(sum(len(v) > 1 for v in lev_of.values())),
               'occupation_vectors_shared_by_distinct_encoded_inputs': int(sum(len(v) > 1 for v in inp_of.values())),
               'records_whose_occupation_vector_is_shared_with_a_different_input': int(sum(len(inp_of[a]) > 1 for a in ko)),
               'encoded_inputs_mapped_to_more_than_one_occupation_vector': int(sum(len(v) > 1 for v in occ_of_inp.values())),
               'share_vectors_singleton': float((c == 1).mean()), 'share_records_in_singleton_vectors': float(c[c == 1].sum() / c.sum()),
               'max_group': int(c.max())}
        if 'raw_inputs' in feats[r]:
            raw = feats[r]['raw_inputs']
            rec['distinct_raw_six_field_vectors'] = len(set(zip(*[raw[f].tolist() for f in RAW])))
        per[r] = rec
        U['occ'] += ko
        U['inc'] += ki
        U['inp'] += kx
        U['lev'] += lev.tolist()
        U['role'] += [r] * len(ko)
        U['xrow'] += [(r, i) for i in range(len(ko))]
    lev_of, roles_of, inp_of, out_of_inp, inp_roles = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    for o, i, l, r, x in zip(U['occ'], U['inc'], U['lev'], U['role'], U['inp']):
        lev_of[o].add(l)
        roles_of[o].add(r)
        inp_of[o].add(x)
        out_of_inp[x].add(o + i)
        inp_roles[x].add(r)
    union = {'records': len(U['occ']), 'distinct_encoded_input_vectors': len(set(U['inp'])), 'distinct_occupation_vectors': len(lev_of),
             'distinct_income_vectors': len(set(U['inc'])), 'distinct_linked_pairs': len(set(i + o for i, o in zip(U['inc'], U['occ']))),
             'occupation_vectors_with_more_than_one_income_code_any_roles': int(sum(len(v) > 1 for v in lev_of.values())),
             'occupation_vectors_seen_in_two_or_more_roles': int(sum(len(v) > 1 for v in roles_of.values())),
             'cross_role_occupation_vectors_with_more_than_one_income_code': int(sum(len(lev_of[k]) > 1 for k, v in roles_of.items() if len(v) > 1)),
             'encoded_inputs_in_two_or_more_roles': int(sum(len(v) > 1 for v in inp_roles.values())),
             'encoded_inputs_with_more_than_one_output_pair_across_roles': int(sum(len(v) > 1 for v in out_of_inp.values())),
             'occupation_vectors_shared_by_distinct_encoded_inputs': int(sum(len(v) > 1 for v in inp_of.values()))}
    # conflicts (aggregate descriptors only)
    conf = []
    pos = defaultdict(list)
    for j, o in enumerate(U['occ']):
        if len(lev_of[o]) > 1:
            pos[o].append(j)
    for o, js in pos.items():
        rows = [U['xrow'][j] for j in js]
        Xs = np.array([feats[r]['X'][i] for r, i in rows])
        diffcols = sorted({names[c] for a in range(len(Xs)) for b in range(a + 1, len(Xs)) for c in np.flatnonzero(Xs[a] != Xs[b])})
        conf.append({'roles': sorted({U['role'][j] for j in js}), 'n_records': len(js), 'income_levels': sorted(lev_of[o]),
                     'differing_encoded_columns_all_POBP': all(c.startswith('POBP=') for c in diffcols) and bool(diffcols),
                     'n_distinct_inputs': len(inp_of[o]),
                     'income_decisions': sorted({int(np.argmax(levelvec[l])) for l in lev_of[o]})})
    return per, union, conf


def encode(rows, enc):
    names = enc['feature_names']
    col = {n: i for i, n in enumerate(names)}
    X = np.zeros((len(rows), len(names)), dtype=np.float64)
    for i, r in enumerate(rows):
        for c in ('AGEP', 'WKHP'):
            X[i, col[c]] = (float(r[c]) - enc['numeric'][c]['mean']) / enc['numeric'][c]['sd']
        for c in ('COW', 'SCHL', 'MAR', 'POBP'):
            key = f'{c}={r[c]}'
            X[i, col[key] if key in col else col[f'{c}=__OTHER__']] = 1.0
    return X.astype(np.float32)


def valid_values(enc):
    pobp = list(enc['categorical']['POBP']) + [c for c in STATE_FIPS if c not in enc['categorical']['POBP']]
    return {'AGEP': [str(v) for v in range(25, 65)], 'WKHP': [str(v) for v in range(1, 100)], 'COW': [str(v) for v in range(1, 9)],
            'SCHL': [f'{v:02d}' for v in range(1, 25)], 'MAR': [str(v) for v in range(1, 6)], 'POBP': pobp}


def routing(seed):
    pkg = V.ADM / 'deploy' / f'FINALIST_S{seed}'
    meta = V.jload(pkg / 'PACKAGE.json')
    enc = meta['encoder']
    names = enc['feature_names']
    col = {n: i for i, n in enumerate(names)}
    te = {t: V.jload(pkg / meta['teachers'][t]) for t in V.TASKS}
    valid = valid_values(enc)
    base = {f: valid[f][0] for f in RAW}
    per, sig_store = {}, {}
    for v in RAW:
        vals = valid[v]
        X = encode([{**base, v: a} for a in vals], enc)
        vcols = [col[v]] if v in ('AGEP', 'WKHP') else [i for i, n in enumerate(names) if n.startswith(v + '=')]
        encs = [X[i, vcols].tobytes() for i in range(len(vals))]
        sig = {}
        for t in V.TASKS:
            f = np.asarray(te[t]['split_features'], dtype=np.int64).reshape(-1)
            b = np.asarray(te[t]['split_borders'], dtype=np.float32).reshape(-1)
            m = np.isin(f, vcols)
            splits = sorted(set(zip(f[m].tolist(), b[m].tolist())))
            sig[t] = [tuple(bool(X[i, c] > np.float32(bb)) for c, bb in splits) for i in range(len(vals))]
        joint = list(zip(sig['occupation'], sig['income']))

        def classes(sg):
            g = defaultdict(set)
            for e, k in zip(encs, sg):
                g[k].add(e)
            return len(g), sum(len(x) == 1 for x in g.values())
        oc, osing = classes(sig['occupation'])
        per[v] = {'valid_values': len(vals), 'distinct_encodings': len(set(encs)), 'occupation_classes': oc, 'occupation_singleton_classes': osing,
                  'income_classes': classes(sig['income'])[0], 'joint_classes': classes(joint)[0]}
        sig_store[v] = {'vals': vals, 'occ': sig['occupation'], 'inc': sig['income'], 'enc': encs}
    prod = lambda key: int(np.prod([per[v][key] for v in RAW], dtype=object))
    cells = {'encoded_domain_size': prod('distinct_encodings'), 'occupation_routing_cells': prod('occupation_classes'),
             'income_routing_cells': prod('income_classes'), 'joint_routing_cells': prod('joint_classes'),
             'share_of_domain_in_singleton_occupation_cells': float(np.prod([per[v]['occupation_singleton_classes'] / per[v]['distinct_encodings'] for v in RAW]))}
    return per, cells, sig_store, enc


def own_release(seed, rows, enc):
    """Own recomputation of the occupation vector and the income level for invented rows (admitted teacher and theta files)."""
    pkg = V.ADM / 'deploy' / f'FINALIST_S{seed}'
    X = encode(rows, enc)
    names = enc['feature_names']
    te_o, te_i = teacher(seed, 'occupation'), teacher(seed, 'income')
    thJ = load_npz(pkg / 'JOINT_TREE__E20.npz')
    thL = load_npz(pkg / 'LOCAL_TREE__E20.npz')
    H = load_npz(pkg / 'H.npz')
    p_occ = softmax_occ(raw_scores(te_o, thJ['theta_occupation'], X, names))
    d = 0.5 * raw_scores(te_i, thL['theta_income'], X, names).reshape(-1) + 0.5 * raw_scores(te_i, thJ['theta_income'], X, names).reshape(-1)
    lev = np.searchsorted(np.asarray(H['thresholds'], dtype=np.float64), d, side='left')
    return p_occ, lev


def nearest_level(p, levelvec):
    return [int(np.argmin(np.abs(levelvec - row).sum(1))) for row in p]


def main():
    from amd import fam_b                 # allowed by the brief for confirming counterexamples through the frozen package
    rep = V.jload(V.RES / 'OUTPUT_MAPPING_RESULTS.json')
    man = V.jload(V.RES / 'FROZEN_CANDIDATE_MANIFEST.json')
    feats = {r: C.load_role(r, ('features', 'raw_inputs') if r in ('MODEL_FIT', 'DEV_REPORT') else ('features',)) for r in V.ROLES6}
    out = {'unit': 'VER_MAPPING_V1', 'labels_read': 'none', 'per_seed': {}}
    for s in V.SEEDS:
        levelvec = np.array([[float.fromhex(x) for x in row] for row in man['per_seed'][f'S{s}']['per_level_income_probability_hex']])
        per, union, conf = empirical(s, feats, levelvec)
        rs = rep['per_seed'][f'S{s}']['empirical_finalist']
        cmp_role = {r: {k: (v, rs['per_role'][r].get(k)) for k, v in per[r].items() if k in rs['per_role'][r] and v != rs['per_role'][r][k]}
                    for r in V.ROLES6}
        for r in V.ROLES6:
            for k in ('share_vectors_singleton', 'share_records_in_singleton_vectors'):
                a, b = per[r][k], rs['per_role'][r]['occupation_group_size'][k]
                if a != b:
                    cmp_role[r][k] = (a, b)
            if per[r]['max_group'] != rs['per_role'][r]['occupation_group_size']['max']:
                cmp_role[r]['max_group'] = (per[r]['max_group'], rs['per_role'][r]['occupation_group_size']['max'])
        cmp_union = {k: (v, rs['union_of_six_roles'].get(k)) for k, v in union.items() if k in rs['union_of_six_roles'] and v != rs['union_of_six_roles'][k]}
        rper, cells, sig, enc = routing(s)
        ds = rep['per_seed'][f'S{s}']['domain_structure']
        cmp_cells = {k: (v, ds['cells'][k]) for k, v in cells.items() if v != ds['cells'][k]}
        cmp_var = {v: {k: (x, ds['per_variable'][v][k]) for k, x in rper[v].items() if x != ds['per_variable'][v].get(k)} for v in RAW}
        # reported invented examples: through the frozen package and through own routing
        pkg = V.ADM / 'deploy' / f'FINALIST_S{s}'
        conf_ex = []
        srch = ds['search']
        for kind in ('examples_same_occupation_different_income_code', 'examples_distinct_inputs_same_occupation_vector'):
            for ex in srch[kind]:
                a, b = ex['input_a'], ex['input_b']
                pr = fam_b.package_release(pkg, [a, b])
                po, pi = pr['occupation']['probability'], pr['income']['probability']
                own_o, own_l = own_release(s, [a, b], enc)
                rec = {'kind': kind, 'variable': ex['variable'],
                       'package_occupation_bitwise_equal': bool(np.array_equal(po[0], po[1])),
                       'package_income_differs': bool(not np.array_equal(pi[0], pi[1])),
                       'package_income_levels_nearest': nearest_level(pi, levelvec),
                       'package_income_decisions': [int(x) for x in pr['income']['decision']],
                       'own_occupation_bitwise_equal': bool(np.array_equal(own_o[0], own_o[1])),
                       'own_occupation_equals_package': bool(np.array_equal(own_o, po)),
                       'own_income_levels': [int(x) for x in own_l]}
                if 'income_decisions' in ex:
                    rec['reported_income_decisions'] = ex['income_decisions']
                    rec['reported_income_a_equals_package'] = [repr(float(x)) for x in pi[0]] == ex['income_a']
                    rec['reported_income_b_equals_package'] = [repr(float(x)) for x in pi[1]] == ex['income_b']
                conf_ex.append(rec)
        # own bounded search (different seed; invented valid inputs only)
        rng = np.random.default_rng(770000 + s)
        valid = valid_values(enc)
        pairs_by_var = {}
        for v in RAW:
            st = sig[v]
            pv = [(st['vals'][i], st['vals'][j]) for i in range(len(st['vals'])) for j in range(i + 1, len(st['vals']))
                  if st['occ'][i] == st['occ'][j] and st['inc'][i] != st['inc'][j] and st['enc'][i] != st['enc'][j]]
            pairs_by_var[v] = pv
        rows_a, rows_b, vars_ = [], [], []
        for _ in range(400):
            bse = {f: valid[f][rng.integers(len(valid[f]))] for f in RAW}
            for v in RAW:
                if pairs_by_var[v]:
                    a, b = pairs_by_var[v][rng.integers(len(pairs_by_var[v]))]
                    rows_a.append({**bse, v: a}); rows_b.append({**bse, v: b}); vars_.append(v)
        oa, la = own_release(s, rows_a, enc)
        ob, lb = own_release(s, rows_b, enc)
        same_occ = (oa == ob).all(1)
        hit = same_occ & (la != lb)
        flip = hit & ((la >= 3) != (lb >= 3))
        idx = np.flatnonzero(hit)[:3]
        via_pkg = []
        for i in idx:
            pr = fam_b.package_release(pkg, [rows_a[i], rows_b[i]])
            via_pkg.append({'variable': vars_[i], 'occupation_bitwise_equal': bool(np.array_equal(pr['occupation']['probability'][0], pr['occupation']['probability'][1])),
                            'income_levels_nearest': nearest_level(pr['income']['probability'], levelvec)})
        own_search = {'rng_seed': 770000 + s, 'base_rows': 400, 'pairs_tested': len(rows_a), 'same_occupation_vector': int(same_occ.sum()),
                      'different_income_level': int(hit.sum()), 'income_decision_flips': int(flip.sum()),
                      'by_variable': {v: int(sum(1 for i in np.flatnonzero(hit) if vars_[i] == v)) for v in RAW},
                      'value_pairs_same_occ_class_diff_income_class': {v: len(p) for v, p in pairs_by_var.items()},
                      'first_hits_confirmed_through_package': via_pkg}
        out['per_seed'][f'S{s}'] = {'per_role': per, 'union': union, 'conflicts': conf, 'differences_vs_reported_per_role': cmp_role,
                                    'differences_vs_reported_union': cmp_union, 'routing_per_variable': rper, 'routing_cells': cells,
                                    'differences_vs_reported_cells': cmp_cells, 'differences_vs_reported_per_variable': cmp_var,
                                    'reported_examples_confirmed': conf_ex, 'own_bounded_search': own_search}
        print(f'S{s}', 'union', union, 'conflicts', conf, 'cells', cells, flush=True)
        print('diffs role', {r: v for r, v in cmp_role.items() if v}, 'union', cmp_union, 'cells', cmp_cells, 'var', {v: x for v, x in cmp_var.items() if x})
        print('examples', [(e['kind'][:20], e['package_occupation_bitwise_equal'], e.get('package_income_differs'), e['package_income_levels_nearest'], e['own_income_levels']) for e in conf_ex])
        print('own search', {k: v for k, v in own_search.items() if k != 'first_hits_confirmed_through_package'}, own_search['first_hits_confirmed_through_package'])
    V.jdump(V.VPRIV / 'check3_mapping.json', out)


if __name__ == '__main__':
    main()
