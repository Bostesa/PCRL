"""Label-blind support census for pcrl_shared_context_release_v1 (support agent).

LABEL-BLIND CONTRACT
  * Never reads ``ctx.pools[*].labels`` (task or protected labels) for any role,
    never reads ``prepared['tables']`` (label-derived support tables) and never
    reads ``prepared['erasers']``.  joblib unpickles the whole prepared cache,
    so the label arrays are resident in memory, but no code path touches them.
  * Arrays accessed are recorded in ``ACCESSED`` and written to the output.
  * Output is aggregate only: counts, ESS, singular values, hashes.  No person
    rows, IDs, household IDs, or label values are written.

Run:  OMP_NUM_THREADS=1 PYTHONPATH=. python <this file>
"""
from __future__ import annotations

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import gc
import hashlib
import json
import platform
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

WT = Path('/Users/nathansamson/PCRL/.worktrees/pcrl-shared-context-release-v1')
MAIN = Path('/Users/nathansamson/PCRL')
PROSP = Path('/Users/nathansamson/PCRL/.worktrees/pcrl-final-prospective-v1')
OUT = WT / 'results/pcrl_shared_context_release_v1/agents/support'
PINNED = WT / 'results/pcrl_task_aligned_cuts_v1/REUSABLE_INPUTS_PINNED.json'
ROLE_COUNTS = WT / 'results/pcrl_adaptive_release_v1/DATA_ROLE_COUNTS.json'
INPUT_RESTORE = WT / 'results/pcrl_task_directed_release_v1/INPUT_RESTORE.json'

# Predecessor role rule (experiments/pcrl_adaptive_release_v1/roles.py), re-implemented
# verbatim so the census does not import code that could touch labels.
ROLE_NAMES = ('nuisance_train', 'audit_fit', 'coefficient_split',
              'inner_selection', 'inner_check', 'outer_assessment')
ROLE_UPPER = (.20, .40, .65, .75, .85, 1.0)
SALT = 'pcrl_adaptive_release_v1|'
INNER_POOLS = ('representation_fit', 'downstream_fit', 'downstream_validation', 'attacker_fit')
ALL_POOLS = INNER_POOLS + ('attacker_validation',)
FLOORS = (25, 50, 100)
ACCESSED: set[str] = set()


def role_index(household: object) -> int:
    value = str(household)
    if not value:
        raise ValueError('empty household identifier')
    unit = int.from_bytes(hashlib.sha256((SALT + value).encode('utf-8')).digest()[:8], 'big') / 2**64
    return next(j for j, b in enumerate(ROLE_UPPER) if unit < b)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ess(masses: np.ndarray) -> float:
    m = np.asarray(masses, float)
    s2 = float((m ** 2).sum())
    return float(m.sum() ** 2 / s2) if s2 > 0 else 0.0


def hh_stats(hh_codes: np.ndarray, w: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """People, unique households, household-mass ESS, person ESS, multi-person households."""
    if mask is not None:
        hh_codes, w = hh_codes[mask], w[mask]
    if len(hh_codes) == 0:
        return {'people': 0, 'households': 0, 'hh_weight_ess': 0.0, 'person_weight_ess': 0.0,
                'weight_sum': 0.0, 'multi_person_households': 0}
    uniq, inv, cnt = np.unique(hh_codes, return_inverse=True, return_counts=True)
    mass = np.bincount(inv, weights=w)
    return {'people': int(len(hh_codes)), 'households': int(len(uniq)),
            'hh_weight_ess': round(ess(mass), 2), 'person_weight_ess': round(ess(w), 2),
            'weight_sum': float(w.sum()), 'multi_person_households': int((cnt >= 2).sum())}


def weighted_free_quantiles(v: np.ndarray, k: int) -> np.ndarray:
    return np.quantile(v, np.arange(1, k) / k) if k > 1 else np.array([])


def spectrum(M: np.ndarray, label: str) -> dict:
    M = np.asarray(M, np.float64)
    mu, sd = M.mean(0), M.std(0, ddof=1)
    const = np.flatnonzero(sd <= 1e-12)
    Z = (M[:, sd > 1e-12] - mu[sd > 1e-12]) / sd[sd > 1e-12]
    s = np.linalg.svd(Z, compute_uv=False)
    ev = s ** 2 / (len(Z) - 1)        # eigenvalues of correlation matrix
    tol64 = s.max() * max(Z.shape) * np.finfo(np.float64).eps
    cum = np.cumsum(ev) / ev.sum()
    return {'matrix': label, 'rows': int(M.shape[0]), 'cols': int(M.shape[1]),
            'zero_variance_cols': int(len(const)),
            'rank_float64_default_tol': int((s > tol64).sum()),
            'rank_rel_1e-5': int((s > 1e-5 * s.max()).sum()),
            'rank_rel_1e-3': int((s > 1e-3 * s.max()).sum()),
            'condition_number': float(s.max() / s.min()) if s.min() > 0 else float('inf'),
            'corr_eigenvalues': [round(float(x), 6) for x in ev],
            'components_for_90pct_var': int(np.searchsorted(cum, .90) + 1),
            'components_for_99pct_var': int(np.searchsorted(cum, .99) + 1)}


def load_anchor(anchor: int, pinned: dict):
    path = Path(pinned['anchors'][str(anchor)]['prepared']['path'])
    return joblib.load(path)


def census_anchor(anchor: int, pinned: dict) -> tuple[dict, dict]:
    t0 = time.time()
    prepared = load_anchor(anchor, pinned)
    ctx = prepared['ctx']; ACCESSED.add('ctx.anchor')
    assert ctx['anchor'] == anchor
    pools = ctx['pools']
    pool_names = list(pools)
    # ---- gather per-person label-free arrays (never touching pools[*]['labels'])
    rows = {k: [] for k in ('pool', 'hh', 'id', 'w', 'x', 'ha', 't0')}
    for pi, name in enumerate(pool_names):
        p = pools[name]
        n = len(p['households'])
        rows['pool'].append(np.full(n, pi))
        rows['hh'].append(np.asarray(p['households']).astype(str))
        rows['id'].append(np.asarray(p['ids']).astype(str))
        rows['w'].append(np.asarray(p['weights'], float))
        rows['x'].append(np.asarray(p['x'], np.float64))
        rows['ha'].append(np.asarray(p['ha'], np.float64))
        rows['t0'].append(np.asarray(prepared['encoded'][name]['codes']['T0'], int))
        for k in ('households', 'ids', 'weights', 'x', 'ha'):
            ACCESSED.add(f'ctx.pools.<pool>.{k}')
        ACCESSED.add('encoded.<pool>.codes.T0')
    R = {k: np.concatenate(v) for k, v in rows.items()}
    # task-directed legacy household roles (index lists into representation_fit; label-free)
    td_roles = prepared['roles']; ACCESSED.add('roles (task-directed household index lists)')
    rf_hh = np.asarray(pools['representation_fit']['households']).astype(str)
    legacy = {k: set(rf_hh[np.asarray(v, int)]) for k, v in td_roles.items()}
    del prepared, ctx, pools
    gc.collect()

    n = len(R['hh'])
    role = np.array([role_index(h) for h in R['hh']])
    # households coded as dense ints for bincount
    _, hh_code = np.unique(R['hh'], return_inverse=True)
    pool_ix = {name: i for i, name in enumerate(pool_names)}
    av = R['pool'] == pool_ix.get('attacker_validation', -1)
    inner = ~av

    # structural checks (aggregate booleans only)
    struct = {
        'pools_loaded': {name: int((R['pool'] == i).sum()) for i, name in enumerate(pool_names)},
        'person_ids_unique_within_anchor': bool(len(np.unique(R['id'])) == n),
        'household_extra_pool_memberships_within_anchor': int(
            len(set(zip(hh_code.tolist(), R['pool'].tolist()))) - len(np.unique(hh_code))),
        'person_id_prefixed_by_household_id': bool(all(i.startswith(h) for i, h in zip(R['id'], R['hh']))),
        'weights_finite_nonnegative': bool(np.isfinite(R['w']).all() and (R['w'] >= 0).all()),
        'x_shape_cols': int(R['x'].shape[1]), 'ha_shape_cols': int(R['ha'].shape[1]),
        't0_range': [int(R['t0'].min()), int(R['t0'].max())],
    }

    # ---- 1/2: roles
    roles_out = {}
    for j, rn in enumerate(ROLE_NAMES):
        m_loaded = (role == j) & (inner if rn != 'outer_assessment' else np.ones(n, bool))
        m_inner = (role == j) & inner
        m_av = (role == j) & av
        roles_out[rn] = {
            'predecessor_semantics': hh_stats(hh_code, R['w'], m_loaded),
            'inner_pools_only': hh_stats(hh_code, R['w'], m_inner),
            'attacker_validation_rows_in_role': hh_stats(hh_code, R['w'], m_av),
            'legacy_teacher_fit_households': int(len(set(R['hh'][m_loaded & (R['pool'] == pool_ix['representation_fit'])]) & legacy['teacher_fit'])),
            'legacy_teacher_internal_validation_households': int(len(set(R['hh'][m_loaded & (R['pool'] == pool_ix['representation_fit'])]) & legacy['teacher_internal_validation'])),
        }
    hh_role_sets = {rn: set(R['hh'][role == j]) for j, rn in enumerate(ROLE_NAMES)}
    overlap = {f'{a}|{b}': len(hh_role_sets[a] & hh_role_sets[b])
               for i, a in enumerate(ROLE_NAMES) for b in ROLE_NAMES[i + 1:]}

    # ---- 3: T32 parents for coefficient and nuisance+coefficient (inner pools, predecessor semantics)
    maps = {}
    for key in ('D17', 'D33', 'Q'):
        Q = np.load(pinned['anchors'][str(anchor)]['maps'][key]['Q.npz']['path'])['Q']
        ACCESSED.add(f'maps.{key}.Q.npz:Q')
        det = bool(np.all(np.isclose(Q.max(1), 1.0)))
        maps[key] = {'shape': list(Q.shape), 'deterministic_rows': int(np.isclose(Q.max(1), 1.0).sum()),
                     'is_deterministic': det,
                     'distinct_tokens_used': int((Q.sum(0) > 1e-12).sum()),
                     'argmax': Q.argmax(1)}
    d17 = maps['D17']['argmax']
    tok_groups = defaultdict(list)
    for s, t in enumerate(d17):
        tok_groups[int(t)].append(s)
    d17_info = {'deterministic': maps['D17']['is_deterministic'],
                'distinct_tokens_used': len(tok_groups),
                'state_to_token': [int(t) for t in d17],
                'states_per_token': {str(t): len(v) for t, v in sorted(tok_groups.items())},
                'merge_histogram': dict(sorted(Counter(len(v) for v in tok_groups.values()).items()))}
    maps_summary = {k: {kk: vv for kk, vv in v.items() if kk != 'argmax'} for k, v in maps.items()}

    coef = (role == ROLE_NAMES.index('coefficient_split')) & inner
    nuis = (role == ROLE_NAMES.index('nuisance_train')) & inner
    t32 = {}
    for label, mask in (('coefficient', coef), ('nuisance_plus_coefficient', coef | nuis)):
        states = []
        for s in range(32):
            st = hh_stats(hh_code, R['w'], mask & (R['t0'] == s))
            st['state'] = s
            st['d17_token'] = int(d17[s])
            st['max_child_household_sum'] = st['households'] + st['multi_person_households']
            st['count_feasible_binary_split'] = {
                str(F): {'disjoint_households(H>=2F)': st['households'] >= 2 * F,
                         'upper_bound_if_households_span_children': st['max_child_household_sum'] >= 2 * F}
                for F in FLOORS}
            states.append(st)
        hh = np.array([s['households'] for s in states])
        tokens = []
        for t, ss in sorted(tok_groups.items()):
            st = hh_stats(hh_code, R['w'], mask & np.isin(R['t0'], ss))
            st.update({'token': t, 't32_states': ss})
            tokens.append(st)
        th = np.array([s['households'] for s in tokens])
        t32[label] = {
            'per_state': states,
            'household_distribution': {'min': int(hh.min()), 'p25': float(np.percentile(hh, 25)),
                                       'median': float(np.median(hh)), 'p75': float(np.percentile(hh, 75)),
                                       'max': int(hh.max()), 'empty_states': int((hh == 0).sum())},
            'states_count_feasible_disjoint': {str(F): int((hh >= 2 * F).sum()) for F in FLOORS},
            'states_with_ge_households': {str(F): int((hh >= F).sum()) for F in FLOORS},
            'per_d17_token': tokens,
            'd17_token_household_distribution': {'min': int(th.min()), 'median': float(np.median(th)),
                                                 'max': int(th.max())},
            'd17_tokens_with_ge_households': {str(F): int((th >= F).sum()) for F in (50, 100, 200)},
        }

    # ---- 4: indicative pooled contexts (quantile cells fitted on nuisance role only)
    def pc1(Mn, Mc):
        mu, sd = Mn.mean(0), Mn.std(0, ddof=1)
        sd[sd <= 1e-12] = 1.0
        Zn = (Mn - mu) / sd
        _, sv, vt = np.linalg.svd(Zn, full_matrices=False)
        v = vt[0]
        return Zn @ v, ((Mc - mu) / sd) @ v, float(sv[0] ** 2 / (sv ** 2).sum())
    xn, xc, xvar = pc1(R['x'][nuis], R['x'][coef])
    hn, hc, hvar = pc1(R['ha'][nuis], R['ha'][coef])
    hh_c, w_c, tok_c = hh_code[coef], R['w'][coef], d17[R['t0'][coef]]

    def cells_of(designs):
        out = {}
        for name, (cell_c, K) in designs.items():
            per = [hh_stats(hh_c, w_c, cell_c == k) for k in range(K)]
            cell_hh = []
            for k in range(K):
                for t in tok_groups:
                    m = (cell_c == k) & (tok_c == t)
                    cell_hh.append(len(np.unique(hh_c[m])))
            cell_hh = np.array(cell_hh)
            out[name] = {'K': K,
                         'per_context': [{kk: p[kk] for kk in ('people', 'households', 'hh_weight_ess')} for p in per],
                         'context_x_d17_cells': int(len(cell_hh)),
                         'cells_ge_50_households': int((cell_hh >= 50).sum()),
                         'cells_ge_100_households': int((cell_hh >= 100).sum()),
                         'cell_household_min': int(cell_hh.min()),
                         'cell_household_median': float(np.median(cell_hh)),
                         'extra_context_memberships_of_households': int(
                             len(set(zip(hh_c.tolist(), cell_c.tolist()))) - len(np.unique(hh_c)))}
        return out

    designs = {'K1_all': (np.zeros(coef.sum(), int), 1)}
    for K in (2, 4):
        designs[f'K{K}_XA_PC1_quantiles'] = (np.searchsorted(weighted_free_quantiles(xn, K), xc, side='right'), K)
        designs[f'K{K}_HA_PC1_quantiles'] = (np.searchsorted(weighted_free_quantiles(hn, K), hc, side='right'), K)
    designs['K4_XA_PC1_median_x_HA_PC1_median'] = (
        2 * np.searchsorted(weighted_free_quantiles(xn, 2), xc, side='right')
        + np.searchsorted(weighted_free_quantiles(hn, 2), hc, side='right'), 4)
    contexts = {'note': 'indicative only; cut points = unweighted quantiles of PC1 fitted on nuisance_train (inner pools) of this anchor',
                'XA_PC1_nuisance_variance_share': round(xvar, 4),
                'HA_PC1_nuisance_variance_share': round(hvar, 4),
                'designs': cells_of(designs)}

    # ---- 5: rank on coefficient role
    Xc, Hc = R['x'][coef], R['ha'][coef]
    # linear redundancy of each H_A column given X_A (label-free)
    Z = np.column_stack([np.ones(len(Xc)), Xc])
    beta, *_ = np.linalg.lstsq(Z, Hc, rcond=None)
    resid = Hc - Z @ beta
    r2 = 1 - resid.var(0) / Hc.var(0)
    rank = {'X_A': spectrum(Xc, 'X_A'), 'H_A': spectrum(Hc, 'H_A'),
            'X_A_H_A': spectrum(np.column_stack([Xc, Hc]), '[X_A,H_A]'),
            'H_A_row_sum_stats': {'min': float(Hc.sum(1).min()), 'max': float(Hc.sum(1).max())},
            'H_A_pair_sums_max_abs_dev_from_1': {'cols01': float(np.abs(Hc[:, 0] + Hc[:, 1] - 1).max()),
                                                 'cols23': float(np.abs(Hc[:, 2] + Hc[:, 3] - 1).max())},
            'R2_of_each_H_A_col_linear_on_X_A': [round(float(v), 4) for v in r2]}

    compact = {'hh': R['hh'], 'id': R['id'], 'w': R['w'], 'role': role, 'inner': inner,
               'legacy_teacher_fit': legacy['teacher_fit'],
               'legacy_teacher_any': legacy['teacher_fit'] | legacy['teacher_internal_validation']}
    del R
    gc.collect()
    result = {'structure': struct, 'roles': roles_out, 'household_role_overlap_pairs': overlap,
              'household_role_overlap_total': int(sum(overlap.values())),
              'maps': maps_summary, 'd17': d17_info, 't32_census': t32,
              'context_support': contexts, 'feature_rank_coefficient_role': rank,
              'seconds': round(time.time() - t0, 1)}
    return result, compact


def global_union(compacts: dict) -> dict:
    """Union across anchors; people deduped by person ID, household mass from unique people."""
    out = {}
    for j, rn in enumerate(ROLE_NAMES):
        person_w: dict[str, float] = {}
        person_hh: dict[str, str] = {}
        w_mismatch = hh_mismatch = 0
        anchors_per_person = Counter()
        anchors_per_hh = defaultdict(set)
        for a, c in compacts.items():
            m = (c['role'] == j) & (c['inner'] if rn != 'outer_assessment' else True)
            for pid, hh, w in zip(c['id'][m], c['hh'][m], c['w'][m]):
                if pid in person_w:
                    w_mismatch += person_w[pid] != w
                    hh_mismatch += person_hh[pid] != hh
                else:
                    person_w[pid], person_hh[pid] = float(w), hh
                anchors_per_person[pid] += 1
                anchors_per_hh[hh].add(a)
        mass = defaultdict(float)
        for pid, w in person_w.items():
            mass[person_hh[pid]] += w
        wv = np.array(list(person_w.values()))
        out[rn] = {'unique_people': len(person_w), 'unique_households': len(mass),
                   'hh_weight_ess': round(ess(np.array(list(mass.values()))), 2),
                   'person_weight_ess': round(ess(wv), 2), 'weight_sum': float(wv.sum()),
                   'people_in_n_anchors': {str(k): v for k, v in sorted(Counter(anchors_per_person.values()).items())},
                   'households_in_n_anchors': {str(k): v for k, v in sorted(Counter(len(s) for s in anchors_per_hh.values()).items())},
                   'weight_mismatch_across_anchors': int(w_mismatch),
                   'household_mismatch_across_anchors': int(hh_mismatch)}
    sets = {rn: set() for rn in ROLE_NAMES}
    for c in compacts.values():
        for j, rn in enumerate(ROLE_NAMES):
            sets[rn] |= set(c['hh'][c['role'] == j])
    overlap = {f'{a}|{b}': len(sets[a] & sets[b]) for i, a in enumerate(ROLE_NAMES) for b in ROLE_NAMES[i + 1:]}
    tf = set().union(*(c['legacy_teacher_fit'] for c in compacts.values()))
    ta = set().union(*(c['legacy_teacher_any'] for c in compacts.values()))
    for rn in ROLE_NAMES:
        out[rn]['legacy_task_directed_teacher_fit_households'] = len(sets[rn] & tf)
        out[rn]['legacy_task_directed_teacher_fit_or_internal_validation_households'] = len(sets[rn] & ta)
    return {'roles': out, 'household_role_overlap_total': int(sum(overlap.values())),
            'household_role_overlap_pairs': overlap}


def verify_hashes(pinned: dict) -> dict:
    res = {'reusable_inputs_pinned': [], 'input_restore_raw_and_preprocessing': [],
           'prepared_json_artifact_hashes': {}, 'source_files_vs_worktree': []}

    def check(path: Path, expected: str) -> dict:
        if not path.is_file():
            return {'path': str(path), 'exists': False, 'match': False}
        got = sha256(path)
        return {'path': str(path), 'exists': True, 'bytes': path.stat().st_size, 'match': got == expected}

    def walk(o, prefix):
        if isinstance(o, dict):
            if 'path' in o and 'sha256' in o:
                r = check(Path(o['path']) if o['path'].startswith('/') else WT / o['path'], o['sha256'])
                r['item'] = prefix
                res['reusable_inputs_pinned'].append(r)
                return
            for k, v in o.items():
                walk(v, f'{prefix}/{k}')
    walk(pinned['anchors'], 'anchors')
    env = pinned['compatible_historical_environment']
    res['environment_record_present'] = (WT / env['path']).is_file()
    for rel, h in pinned['source_files'].items():
        r = check(WT / rel, h); r['item'] = rel
        import subprocess
        blob = subprocess.run(['git', '-C', str(WT), 'show', f'HEAD:{rel}'], capture_output=True)
        r['head_blob_match'] = blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == h
        if not r['exists']:
            r['note'] = 'excluded by sparse checkout; verified against the HEAD git blob instead'
        res['source_files_vs_worktree'].append(r)
    for f in json.loads(INPUT_RESTORE.read_text())['files']:
        hits = []
        for root in (MAIN, PROSP, WT):
            p = root / f['relative_path']
            if p.is_file():
                hits.append({'root': str(root), 'match': sha256(p) == f['sha256'], 'bytes': p.stat().st_size})
        res['input_restore_raw_and_preprocessing'].append({'item': f['relative_path'], 'locations': hits,
                                                           'any_verified': any(h['match'] for h in hits)})
    for a in ('0', '1', '2'):
        base = Path(pinned['anchors'][a]['prepared']['path']).parent
        rec = json.loads((base / 'PREPARED.json').read_text())
        ok = missing = bad = 0
        bad_items, missing_items = [], []
        for rel, h in rec['artifact_hashes'].items():
            p = base / rel
            if not p.is_file():
                missing += 1; missing_items.append(rel); continue
            if sha256(p) == h:
                ok += 1
            else:
                bad += 1; bad_items.append(rel)
        res['prepared_json_artifact_hashes'][a] = {'listed': len(rec['artifact_hashes']), 'verified': ok,
                                                   'missing': missing, 'mismatch': bad,
                                                   'missing_items': missing_items[:50], 'mismatch_items': bad_items,
                                                   'prepared_created_utc': rec.get('created_utc')}
    return res


def main():
    pinned = json.loads(PINNED.read_text())
    out = {'schema': 'pcrl-shared-context-support-census-v1',
           'label_blind': True,
           'outer_assessment_labels_accessed': False,
           'any_label_values_accessed': False,
           'role_rule': 'first 64 bits SHA256(UTF8("pcrl_adaptive_release_v1|"+household_id))/2^64; upper bounds ' + str(dict(zip(ROLE_NAMES, ROLE_UPPER))),
           'predecessor_semantics': 'non-outer roles: representation_fit+downstream_fit+downstream_validation+attacker_fit (attacker_validation excluded, as in roles.summarize_roles/pooled_role); outer: all five loaded pools',
           'host': {'platform': platform.platform(), 'python': sys.version.split()[0], 'numpy': np.__version__},
           'anchors': {}}
    compacts = {}
    for a in (0, 1, 2):
        res, compacts[a] = census_anchor(a, pinned)
        out['anchors'][str(a)] = res
        gc.collect()
    # cross-anchor overlap per role
    out['global_union'] = global_union(compacts)
    del compacts
    gc.collect()
    # cross-check vs predecessor DATA_ROLE_COUNTS
    ref = json.loads(ROLE_COUNTS.read_text())
    xc = {'anchors': {}, 'global_households': {}}
    all_ok = True
    for a in ('0', '1', '2'):
        xc['anchors'][a] = {}
        for rn in ROLE_NAMES:
            mine = out['anchors'][a]['roles'][rn]['predecessor_semantics']
            theirs = ref['anchors'][a][rn]
            ok = (mine['people'] == theirs['people'] and mine['households'] == theirs['households']
                  and abs(mine['weight_sum'] - theirs['weight_sum']) < 1e-6)
            all_ok &= ok
            xc['anchors'][a][rn] = {'match': ok, 'people': [mine['people'], theirs['people']],
                                    'households': [mine['households'], theirs['households']]}
    for rn in ROLE_NAMES:
        m, t = out['global_union']['roles'][rn]['unique_households'], ref['global_households'][rn]
        xc['global_households'][rn] = {'match': m == t, 'mine': m, 'reference': t}
        all_ok &= m == t
    mine_t = out['global_union']['roles']['outer_assessment']['legacy_task_directed_teacher_fit_or_internal_validation_households']
    xc['historical_teacher_households_in_outer'] = {'mine': mine_t, 'reference': ref['historical_teacher_households_in_outer'],
                                                    'match': mine_t == ref['historical_teacher_households_in_outer'],
                                                    'definition': 'outer households in task-directed teacher_fit or teacher_internal_validation (representation_fit, any anchor)'}
    all_ok &= mine_t == ref['historical_teacher_households_in_outer']
    xc['all_match'] = bool(all_ok)
    xc['note'] = 'class_support fields in the reference file were not read or recomputed'
    out['cross_check_DATA_ROLE_COUNTS'] = xc
    out['hash_verification'] = verify_hashes(pinned)
    out['arrays_accessed'] = sorted(ACCESSED)
    out['arrays_never_accessed'] = ['ctx.pools.<pool>.labels (all keys, all pools, all roles)',
                                    'prepared.tables (label-derived support/cost tables)',
                                    'prepared.erasers', 'encoded.<pool>.{p,b,r,risk,actions,global_offsets}',
                                    'ctx.pools.<pool>.{hb,J,A0,raw_rows}',
                                    'historical fineC tables (hash only)']
    out['peak_rss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)
    tmp = OUT / 'SUPPORT_CENSUS.json.tmp'
    tmp.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    tmp.replace(OUT / 'SUPPORT_CENSUS.json')
    print('written', OUT / 'SUPPORT_CENSUS.json', 'peak_rss_mb', out['peak_rss_mb'])


if __name__ == '__main__':
    main()
