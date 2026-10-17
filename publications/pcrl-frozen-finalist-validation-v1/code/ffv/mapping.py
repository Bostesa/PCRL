"""Unit FFV_MAPPING: output-mapping check of the frozen finalist (PROTOCOL.md section 5). Label-free: no SEX or task label is read.

Part 1, empirical (exact transmitted float64 bytes; no rounding): for each admitted role and seed, and for the union of the six
roles, counts of records, distinct encoded permitted-input vectors (and distinct raw six-field vectors where raw inputs are
admitted), distinct occupation vectors, income vectors and linked pairs; occupation-vector group sizes and singleton fractions;
occupation vectors that accompany more than one income code (within a role and across roles); input -> output determinism within
and across roles. Context counts for the other bank releases on the four bank roles.

Part 2, domain-wide structure (kept separate from Part 1): from the frozen oblivious-tree split borders, per raw variable, the
equivalence classes of its valid values under the occupation teacher, the income teacher and both; the exact number of routing
cells over the valid encoded domain; and a bounded search over INVENTED valid inputs (random draws from the valid domain; no real
record is used) for (i) distinct inputs with bitwise-identical occupation vectors and (ii) bitwise-identical occupation vectors
with different income codes. Every reported example is confirmed through the frozen FINALIST deployment package.

Part 3, diagnostic (pair access only): "pair-trained reconstruction with occupation-only evaluation input": a classifier with
the HGB15 reader settings fitted on ATTACK_FIT linked pairs predicts the finalist's income level from its occupation output;
level accuracy on ATTACK_VALIDATION and DEV_REPORT outputs (no SEX used). Not an ordinary occupation-only adversary.
"""
from __future__ import annotations
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from acs_replication import deploy as SRC_DEPLOY
from amd import fam_b
from . import common as C
from . import features as F

FIN = 'B3CTL_UNCOUPLED_MIX50_C6A'
US_STATE_FIPS = ['001', '002', '004', '005', '006', '008', '009', '010', '011', '012', '013', '015', '016', '017', '018', '019', '020', '021',
                 '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039',
                 '040', '041', '042', '044', '045', '046', '047', '048', '049', '050', '051', '053', '054', '055', '056']
VALID = {'AGEP': [str(v) for v in range(25, 65)], 'WKHP': [str(v) for v in range(1, 100)],
         'COW': [str(v) for v in range(1, 9)], 'MAR': [str(v) for v in range(1, 6)], 'SCHL': [f'{v:02d}' for v in range(1, 25)]}


def keyrows(*arrs):
    z = np.ascontiguousarray(np.column_stack([np.asarray(a) for a in arrs]))
    return [r.tobytes() for r in z]


def wire_keys(p):
    """Serialized transmitted output: float64 probability bytes + int64 argmax decision bytes."""
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    d = p.argmax(1).astype(np.int64)
    return [a.tobytes() + b.tobytes() for a, b in zip(p, d)]


def level_of(inc, lv):
    out = np.full(len(inc), -1, dtype=np.int64)
    for j in range(len(lv)):
        out[(inc == lv[j]).all(1)] = j
    return out


def group_stats(keys):
    u, inv, cnt = np.unique(np.array(keys, dtype=object), return_inverse=True, return_counts=True)
    return u, inv, cnt


def empirical(seed):
    with np.load(C.admitted(f'runs/B/{FIN}/S{seed}/RELEASE.npz')) as z:
        rel = {k: z[k].copy() for k in z.files}
    man = json.loads((C.RESULTS / 'FROZEN_CANDIDATE_MANIFEST.json').read_text())['per_seed'][f'S{seed}']
    lv = np.array([[float.fromhex(x) for x in row] for row in man['per_level_income_probability_hex']])
    roles, union = {}, {'occ': [], 'inc': [], 'inp': [], 'role': []}
    for r in C.ROLES6:
        X = C.load_role(r, ('features', 'raw_inputs') if 'raw_inputs' in C.ADMITTED[r] else ('features',))
        fx = np.ascontiguousarray(X['X'].astype(np.float32))
        occ, inc = rel[f'{r}__occupation'], rel[f'{r}__income']
        ko, ki, kx = wire_keys(occ), wire_keys(inc), keyrows(fx)
        kp = [a + b for a, b in zip(ki, ko)]
        lev = level_of(inc, lv)
        uo, invo, cnto = group_stats(ko)
        conflicts = 0
        lev_by_occ = {}
        for i, k in enumerate(ko):
            lev_by_occ.setdefault(k, set()).add(int(lev[i]))
        conflicts = sum(len(v) > 1 for v in lev_by_occ.values())
        inp_by_occ, occ_by_inp = {}, {}
        for a, b in zip(kx, ko):
            inp_by_occ.setdefault(b, set()).add(a); occ_by_inp.setdefault(a, set()).add(b)
        rec = {'records': int(len(occ)), 'distinct_encoded_input_vectors': int(len(set(kx))),
               'distinct_occupation_vectors': int(len(uo)), 'distinct_income_vectors': int(len(set(ki))), 'distinct_linked_pairs': int(len(set(kp))),
               'income_levels_observed': sorted(int(x) for x in np.unique(lev)), 'unmatched_income_vectors': int((lev < 0).sum()),
               'occupation_group_size': {'max': int(cnto.max()), 'mean': float(cnto.mean()), 'share_vectors_singleton': float((cnto == 1).mean()),
                                         'share_records_in_singleton_vectors': float(cnto[cnto == 1].sum() / cnto.sum()),
                                         'histogram': {str(k): int((cnto == k).sum()) for k in sorted(set(cnto.tolist()))[:10]}},
               'occupation_vectors_with_more_than_one_income_code': int(conflicts),
               'occupation_vectors_shared_by_distinct_encoded_inputs': int(sum(len(v) > 1 for v in inp_by_occ.values())),
               'records_whose_occupation_vector_is_shared_with_a_different_input': int(sum(len(inp_by_occ[b]) > 1 for b in ko)),
               'encoded_inputs_mapped_to_more_than_one_occupation_vector': int(sum(len(v) > 1 for v in occ_by_inp.values()))}
        if 'raw_inputs' in X:
            raw = X['raw_inputs']; kr = keyrows(*[raw[f].astype('U3') for f in C.RAW_FIELDS])
            rec['distinct_raw_six_field_vectors'] = int(len(set(kr)))
        roles[r] = rec
        union['occ'] += ko; union['inc'] += ki; union['inp'] += kx; union['role'] += [r] * len(ko)
        union.setdefault('lev', []).extend(lev.tolist())
    lev_by_occ, roles_by_occ, out_by_inp = {}, {}, {}
    for k, l, r, x, ki in zip(union['occ'], union['lev'], union['role'], union['inp'], union['inc']):
        lev_by_occ.setdefault(k, set()).add(l); roles_by_occ.setdefault(k, set()).add(r); out_by_inp.setdefault(x, set()).add(k + ki)
    cross = [k for k, v in roles_by_occ.items() if len(v) > 1]
    u = {'records': len(union['occ']), 'distinct_encoded_input_vectors': len(set(union['inp'])), 'distinct_occupation_vectors': len(set(union['occ'])),
         'distinct_income_vectors': len(set(union['inc'])), 'distinct_linked_pairs': len(set(a + b for a, b in zip(union['inc'], union['occ']))),
         'occupation_vectors_with_more_than_one_income_code_any_roles': int(sum(len(v) > 1 for v in lev_by_occ.values())),
         'occupation_vectors_seen_in_two_or_more_roles': len(cross),
         'cross_role_occupation_vectors_with_more_than_one_income_code': int(sum(len(lev_by_occ[k]) > 1 for k in cross)),
         'encoded_inputs_in_two_or_more_roles': int(sum(1 for x in out_by_inp)) - len(set(union['inp'])) + 0,
         'encoded_inputs_with_more_than_one_output_pair_across_roles': int(sum(len(v) > 1 for v in out_by_inp.values()))}
    inp_roles = {}
    for x, r in zip(union['inp'], union['role']):
        inp_roles.setdefault(x, set()).add(r)
    u['encoded_inputs_in_two_or_more_roles'] = int(sum(len(v) > 1 for v in inp_roles.values()))
    return {'per_role': roles, 'union_of_six_roles': u}


def context_counts(seed):
    bank, _, _ = C.service_bank(seed)
    out = {}
    for svc in C.BANK:
        out[svc] = {}
        for r in ('MODEL_SELECT',) + C.ROLES3:
            ko, ki = wire_keys(bank[svc][r]['occupation']), wire_keys(bank[svc][r]['income'])
            ib = {}
            for a, b in zip(ko, ki):
                ib.setdefault(a, set()).add(b)
            out[svc][r] = {'distinct_occupation': len(set(ko)), 'distinct_income': len(set(ki)), 'distinct_pairs': len(set(a + b for a, b in zip(ki, ko))),
                           'occupation_vectors_with_more_than_one_income_vector': int(sum(len(v) > 1 for v in ib.values()))}
    return out


def domain(seed, n_base=600, max_pairs_per_variable=12, rng_seed=2026101801):
    pkg = C.ADM / f'deploy/FINALIST_S{seed}'
    meta = json.loads((pkg / 'PACKAGE.json').read_text()); enc = meta['encoder']; names = enc['feature_names']; col = {n: i for i, n in enumerate(names)}
    teach = {t: json.loads((pkg / meta['teachers'][t]).read_text()) for t in C.TASKS}
    valid = dict(VALID); valid['POBP'] = list(enc['categorical']['POBP']) + [c for c in US_STATE_FIPS if c not in enc['categorical']['POBP']]
    other_examples = {v: [c for c in valid[v] if c not in enc['categorical'].get(v, valid[v])] for v in ('COW', 'MAR', 'SCHL', 'POBP')}
    enc_rows = lambda rows: SRC_DEPLOY.encode_raw(rows, enc)
    classes, per_var = {}, {}
    for v in C.RAW_FIELDS:
        base = {f: valid[f][0] for f in C.RAW_FIELDS}
        rows = [{**base, v: a} for a in valid[v]]
        Xv = enc_rows(rows)
        sig = {}
        for t in C.TASKS:
            f_, b_ = np.asarray(teach[t]['split_features']), np.asarray(teach[t]['split_borders'], dtype=np.float32)
            vcols = [col[v]] if v in ('AGEP', 'WKHP') else [col[n] for n in names if n.startswith(v + '=')]
            mask = np.isin(f_, vcols)
            splits = sorted(set(zip(f_[mask].tolist(), b_[mask].tolist())))
            sig[t] = [tuple(bool(x[c] > np.float32(b)) for c, b in splits) for x in Xv]
        enc_sig = [x.tobytes() for x in Xv]
        n_enc = len(set(enc_sig))
        occ_cls, inc_cls = sig['occupation'], sig['income']
        joint = list(zip(occ_cls, inc_cls))
        def nclass(s):
            u = {}
            for e, k in zip(enc_sig, s):
                u.setdefault(k, set()).add(e)
            return len(u), sum(1 for x in u.values() if len(x) == 1)
        per_var[v] = {'valid_values': len(valid[v]), 'distinct_encodings': n_enc,
                      'occupation_classes': nclass(occ_cls)[0], 'occupation_singleton_classes': nclass(occ_cls)[1],
                      'income_classes': nclass(inc_cls)[0], 'joint_classes': nclass(joint)[0]}
        pairs = []
        seen = set()
        for i in range(len(valid[v])):
            for j in range(i + 1, len(valid[v])):
                if occ_cls[i] == occ_cls[j] and inc_cls[i] != inc_cls[j] and (occ_cls[i], inc_cls[i], inc_cls[j]) not in seen:
                    seen.add((occ_cls[i], inc_cls[i], inc_cls[j])); pairs.append((valid[v][i], valid[v][j]))
        same_occ = [(valid[v][i], valid[v][j]) for i in range(len(valid[v])) for j in range(i + 1, len(valid[v]))
                    if occ_cls[i] == occ_cls[j] and enc_sig[i] != enc_sig[j]]
        per_var[v]['value_pairs_same_occupation_class_different_income_class'] = len(pairs)
        per_var[v]['value_pairs_same_occupation_class_distinct_encoding'] = len(same_occ)
        classes[v] = {'income_only_pairs': pairs[:max_pairs_per_variable], 'same_occ_pairs': same_occ[:3]}
    prod = lambda key: int(np.prod([per_var[v][key] for v in C.RAW_FIELDS], dtype=object))
    cells = {'encoded_domain_size': prod('distinct_encodings'), 'occupation_routing_cells': prod('occupation_classes'),
             'income_routing_cells': prod('income_classes'), 'joint_routing_cells': prod('joint_classes'),
             'share_of_domain_in_singleton_occupation_cells': float(np.prod([per_var[v]['occupation_singleton_classes'] / per_var[v]['distinct_encodings'] for v in C.RAW_FIELDS]))}
    rng = np.random.default_rng(rng_seed + seed)
    bases = [{f: valid[f][rng.integers(len(valid[f]))] for f in C.RAW_FIELDS} for _ in range(n_base)]
    lv = None
    tests, examples_ce, examples_ni = 0, [], []
    n_ce = 0; ce_by_var = {v: 0 for v in C.RAW_FIELDS}
    for v in C.RAW_FIELDS:
        prs = classes[v]['income_only_pairs']
        if not prs:
            continue
        rows_a = [{**b, v: a} for b in bases for a, _ in prs]; rows_b = [{**b, v: c} for b in bases for _, c in prs]
        both = fam_b.package_release(pkg, rows_a + rows_b)
        n = len(rows_a)
        oa, ob = both['occupation']['probability'][:n], both['occupation']['probability'][n:]
        ia, ib = both['income']['probability'][:n], both['income']['probability'][n:]
        same_occ = (oa == ob).all(1); diff_inc = ~(ia == ib).all(1)
        hit = same_occ & diff_inc
        tests += n; n_ce += int(hit.sum()); ce_by_var[v] = int(hit.sum())
        for i in np.flatnonzero(hit)[:2]:
            examples_ce.append({'variable': v, 'input_a': rows_a[i], 'input_b': rows_b[i], 'occupation_vector_bitwise_equal': True,
                                'income_a': [repr(float(x)) for x in ia[i]], 'income_b': [repr(float(x)) for x in ib[i]],
                                'income_decisions': [int(ia[i].argmax()), int(ib[i].argmax())]})
    for v in C.RAW_FIELDS:
        for a, c in classes[v]['same_occ_pairs'][:1]:
            b = bases[0]; out = fam_b.package_release(pkg, [{**b, v: a}, {**b, v: c}])
            eq = bool((out['occupation']['probability'][0] == out['occupation']['probability'][1]).all())
            examples_ni.append({'variable': v, 'input_a': {**b, v: a}, 'input_b': {**b, v: c}, 'occupation_vector_bitwise_equal': eq})
    return {'per_variable': per_var, 'cells': cells, 'other_column_valid_examples': {k: v[:5] for k, v in other_examples.items()},
            'valid_domain_definition': 'AGEP 25..64; WKHP 1..99; COW 1..8; MAR 1..5; SCHL 01..24 (codes outside the encoder vocabulary encode as SCHL=__OTHER__); POBP: the encoder vocabulary plus U.S. state FIPS codes outside it (encode as POBP=__OTHER__); counts are over distinct encodings',
            'search': {'invented_base_rows': n_base, 'rng_seed': rng_seed + seed, 'value_pairs_per_variable_max': max_pairs_per_variable,
                       'input_pairs_tested': int(tests), 'counterexamples_found': int(n_ce), 'counterexamples_by_variable': ce_by_var,
                       'examples_same_occupation_different_income_code': examples_ce[:6], 'examples_distinct_inputs_same_occupation_vector': examples_ni,
                       'confirmed_through': 'the frozen FINALIST deployment package (amd.fam_b.package_release on the admitted package files)'}}


def reconstruction(seed):
    bank, _, _ = C.service_bank(seed, ('FINALIST',))
    man = json.loads((C.RESULTS / 'FROZEN_CANDIDATE_MANIFEST.json').read_text())['per_seed'][f'S{seed}']
    lv = np.array([[float.fromhex(x) for x in row] for row in man['per_level_income_probability_hex']])
    b = bank['FINALIST']
    X = {r: F.rich(b[r]['occupation']) for r in C.ROLES3}; y = {r: level_of(b[r]['income'], lv) for r in C.ROLES3}
    m = HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, min_samples_leaf=30, l2_regularization=1.0,
                                       early_stopping=False, random_state=2026101205 + seed).fit(X['ATTACK_FIT'], y['ATTACK_FIT'])
    seen = set(wire_keys(b['ATTACK_FIT']['occupation']))
    out = {'label': 'pair-trained reconstruction with occupation-only evaluation input (pair access; diagnostic, not an occupation-only adversary)',
           'fit': 'ATTACK_FIT linked pairs (occupation output -> finalist income level); no SEX used', 'model': 'HGB15 reader settings, multiclass'}
    for r in ('ATTACK_VALIDATION', 'DEV_REPORT'):
        pred = m.predict(X[r]); maj = np.bincount(y['ATTACK_FIT']).argmax()
        out[r] = {'level_accuracy': float((pred == y[r]).mean()), 'majority_level_accuracy': float((y[r] == maj).mean()),
                  'share_occupation_vectors_seen_in_attack_fit': float(np.mean([k in seen for k in wire_keys(b[r]['occupation'])]))}
    out['ATTACK_FIT_in_sample_level_accuracy'] = float((m.predict(X['ATTACK_FIT']) == y['ATTACK_FIT']).mean())
    return out


def main():
    t0 = time.process_time()
    res = {'schema': 'FFV1-output-mapping-v1', 'utc': C.now(), 'labels_read': 'none', 'per_seed': {}}
    for s in C.SEEDS:
        res['per_seed'][f'S{s}'] = {'empirical_finalist': empirical(s), 'context_other_releases': context_counts(s),
                                    'domain_structure': domain(s), 'pair_trained_reconstruction': reconstruction(s)}
    res['cpu_s'] = round(time.process_time() - t0, 1)
    C.write_json_x(C.RESULTS / 'OUTPUT_MAPPING_RESULTS.json', res)
    print(json.dumps({s: {'union': v['empirical_finalist']['union_of_six_roles'], 'cells': v['domain_structure']['cells'],
                          'search': {k: v['domain_structure']['search'][k] for k in ('input_pairs_tested', 'counterexamples_found', 'counterexamples_by_variable')},
                          'reconstruction': {r: v['pair_trained_reconstruction'][r] for r in ('ATTACK_VALIDATION', 'DEV_REPORT')}} for s, v in res['per_seed'].items()}, indent=1))


if __name__ == '__main__':
    main()
