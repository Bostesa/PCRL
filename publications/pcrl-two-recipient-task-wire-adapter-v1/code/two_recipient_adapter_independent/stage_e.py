"""Stage E: INNER control gates, 39 primary and 45 diagnostic endpoints from the frozen releases/scores with an own bootstrap."""
from __future__ import annotations
import argparse
import json

from .common import bind, npz, write_out, task_metrics_np

SEED = 2026101301
BREP = 2048
ALPHA = 0.05
NULL_SEED = 2026101207
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
ROSTER = ('RAW_PROJ', 'T_BOTH', 'L_BOTH', 'S12', 'S21')
PRIV = (('pair', 'benefit', 0.020), ('r1', 'excess', 0.010), ('r2', 'excess', 0.010))


def boot_weights(unit):
    import numpy as np
    u = np.unique(unit)                       # sorted unique INNER units
    pos = np.searchsorted(u, unit)
    draws = np.random.default_rng(SEED).integers(0, len(u), (BREP, len(u)))
    cnt = np.zeros((BREP, len(u)))
    for k in range(BREP):
        cnt[k] = np.bincount(draws[k], minlength=len(u))
    return np.vstack([np.ones((1, len(unit))), cnt[:, pos]])


def wauc(y, s, Wt):
    """Tie-aware weighted AUC for every weight row: sum_{pos,neg} w w' [s+>s- + .5 tie] / (W+ W-)."""
    import numpy as np
    y = np.asarray(y).astype(bool); s = np.asarray(s, dtype=np.float64)
    vals, inv = np.unique(s, return_inverse=True)       # ascending distinct scores
    order = np.argsort(inv, kind='stable'); g = inv[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(g)) + 1))
    yp = y[order]
    out = np.empty(len(Wt))
    for a in range(0, len(Wt), 512):
        w = Wt[a:a + 512][:, order]
        P = np.add.reduceat(np.where(yp, w, 0.0), starts, axis=1)   # positive weight per distinct score
        N = np.add.reduceat(np.where(yp, 0.0, w), starts, axis=1)   # negative weight per distinct score
        below = np.cumsum(N, 1) - N
        out[a:a + 512] = (P * (below + 0.5 * N)).sum(1) / (P.sum(1) * N.sum(1))
    return out


def shuffle_groups(y, unit):
    """Group-size-stratified permutation of group labels (registered INNER null)."""
    import numpy as np
    rng = np.random.default_rng(NULL_SEED); out = np.empty_like(y); buckets = {}
    for g in np.unique(unit):
        ix = np.flatnonzero(unit == g)
        assert len(np.unique(y[ix])) == 1
        buckets.setdefault(len(ix), []).append(ix)
    for size in sorted(buckets):
        arr = buckets[size]
        for tgt, srcix in zip(arr, rng.permutation(len(arr))):
            out[tgt] = y[arr[int(srcix)]]
    return out


def tcrit(K, G):
    from scipy.stats import t
    return float(t.ppf(1 - ALPHA / (2 * K), G - 1))


def cls(point, lo, hi, thr, direction):
    if direction == 'at_most':
        return 'PASS' if hi <= thr else ('ESTABLISHED_VIOLATION' if lo > thr else 'UNRESOLVED')
    return 'PASS' if lo > thr else ('ESTABLISHED_VIOLATION' if hi < thr else 'UNRESOLVED')


def run(spec):
    import numpy as np
    active, cfg, get, guard = bind(spec, inner=True)
    rep = json.loads(get(cfg['inner_report']).read_text())
    sel = json.loads(get(cfg['selection_lock_item']).read_text())
    from two_recipient_adapter_independent.readers_list import SLATE
    truth = {}
    for n in ('income', 'occupation', 'sex'):
        truth[n] = npz(get(cfg['inner_truth'][n]))
    rid, unit = truth['sex']['row_id'], truth['sex']['unit']
    for n in truth:
        assert np.array_equal(truth[n]['row_id'], rid) and np.array_equal(truth[n]['unit'], unit)
    y = {n: truth[n][n].astype(np.int64) for n in truth}
    refs = {s: npz(get(cfg['inner_references'][s])) for s in ('0', '1')}
    banks = {}
    for key, b in cfg['banks'].items():
        sc = npz(get(b['scores'])); rl = npz(get(b['release']))
        assert np.array_equal(sc['row_id'], rid) and np.array_equal(rl['row_id'], rid)
        banks[key] = {'scores': sc, 'release': rl}
    G = len(np.unique(unit)); N = len(rid)
    Wt = boot_weights(unit); tot = Wt.sum(1)
    guard()
    curves = {}

    def curve(scores):
        k = np.ascontiguousarray(scores, dtype=np.float64).tobytes()
        import hashlib
        h = hashlib.sha256(k).hexdigest()
        if h not in curves:
            curves[h] = wauc(y['sex'], scores, Wt)
        return curves[h]

    env = {}; argmax = {}

    def envelope(label, seed, view):
        key = (label, seed, view)
        if key not in env:
            sc = banks[f'{label}__S{seed}']['scores']
            keys = [(f'{view}__{n}', f'{view}__{n}') for n in SLATE]
            if view == 'pair':
                keys += [(f'pair__IGNORE_{o}__{n}', f'{"r2" if o == "r1" else "r1"}__{n}') for o in ('r1', 'r2') for n in SLATE]
            cs = [curve(sc[src]) for _, src in keys]
            M = np.vstack(cs)
            env[key] = M.max(0)
            j = int(np.argmax(M[:, 0])); argmax[key] = {'reader_key': keys[j][0], 'score_key': keys[j][1], 'auc': float(M[j, 0])}
        return env[key]

    # control gates
    null_y = shuffle_groups(y['sex'], unit)
    gates = {}; gdiff = 0.0
    for seed in ('0', '1'):
        c = banks[f'CONTROLS__S{seed}']['scores']
        for view in ('r1', 'r2', 'pair'):
            sh = [wauc(null_y, c[f'{view}__SHUFFLED__{n}'], Wt[:1])[0] for n in SLATE]
            if view == 'pair':
                sh += [wauc(null_y, c[f'{v}__SHUFFLED__{n}'], Wt[:1])[0] for v in ('r1', 'r2') for n in SLATE]
            pl = [wauc(y['sex'], np.where(y['sex'] == 1, c[f'{view}__PLANTED_1__{n}'], c[f'{view}__PLANTED_0__{n}']), Wt[:1])[0] for n in SLATE]
            vals = {'null_auc': float(wauc(y['sex'], np.full(N, .5), Wt[:1])[0]), 'shuffled_envelope_max_two_sided': float(max(max(a, 1 - a) for a in sh)),
                    'planted_envelope': float(max(pl)), 'planted_weakest_reader': float(min(pl))}
            passed = vals['null_auc'] == .5 and vals['shuffled_envelope_max_two_sided'] <= .60 and vals['planted_envelope'] >= .99
            k = f'S{seed}__{view}'
            gates[k] = {'values': vals, 'passed': bool(passed)}
            cv = rep['control_gates'][k]
            gdiff = max(gdiff, max(abs(vals[x] - cv['values'][x]) for x in vals))
            if bool(passed) != cv['passed']:
                gates[k]['mismatch'] = True
    gates_ok = all(g['passed'] for g in gates.values())

    def util_rows(label, K):
        crit = tcrit(K, G); out = []
        for seed in ('0', '1'):
            rl = banks[f'{label}__S{seed}']['release']
            for i, task in enumerate(('income', 'occupation')):
                k = (2, 6)[i]; w = rl[f'r{i + 1}']; p = w[:, 16 + k:16 + 2 * k]
                assert np.array_equal(w[:, -1], p.argmax(1).astype(np.float64))
                own = task_metrics_np(y[task], p, k)
                for ref in ('U0', 'Ucal'):
                    rv = task_metrics_np(y[task], refs[seed][f'{ref}_{i + 1}'], k)
                    d = {'accuracy_loss': rv['accuracy'] - own['accuracy'], 'nll_excess': own['nll'] - rv['nll'], 'brier_excess': own['brier'] - rv['brier']}
                    for m, cap in CAPS.items():
                        v = (Wt @ d[m]) / tot; pt = float(v[0]); se = float(v[1:].std(ddof=1)); lo, hi = pt - crit * se, pt + crit * se
                        out.append({'endpoint_id': f'utility__{label}__S{seed}__{task}__{ref}__{m}', 'point': pt, 'se': se, 'lower': lo, 'upper': hi,
                                    'status': cls(pt, lo, hi, cap, 'at_most')})
        return out

    cand = 'J_BOTH'; K = 39
    prim = util_rows(cand, K)
    crit = tcrit(K, G)
    eligible = sel['primary_family']['comparators_point_eligible']
    for ctl in ROSTER:
        for view, kind, thr in PRIV:
            eid = f'privacy__{cand}__vs__{ctl}__{view}_{kind}'
            parts = [(envelope(ctl, s, view) - envelope(cand, s, view)) if kind == 'benefit' else (envelope(cand, s, view) - envelope(ctl, s, view)) for s in ('0', '1')]
            v = np.mean(parts, 0); pt = float(v[0]); se = float(v[1:].std(ddof=1)); lo, hi = pt - crit * se, pt + crit * se
            st = cls(pt, lo, hi, thr, 'greater_than' if kind == 'benefit' else 'at_most')
            if not gates_ok:
                st = 'NOT_ESTIMABLE'
            prim.append({'endpoint_id': eid, 'kind': 'privacy', 'control': ctl, 'point': pt, 'se': se, 'lower': lo, 'upper': hi,
                         'status': st if ctl in eligible else 'INCOMPLETE_COMPARATOR_INELIGIBLE', 'per_seed': [float(p[0]) for p in parts]})
    for r in prim:
        r.setdefault('kind', 'utility')
    Kd = 45; critd = tcrit(Kd, G); diag = []
    for c in sel['diagnostic_family']['contrasts']:
        for view in ('r1', 'r2', 'pair'):
            parts = [envelope(c['control'], s, view) - envelope(c['candidate'], s, view) for s in ('0', '1')]
            v = np.mean(parts, 0); pt = float(v[0]); se = float(v[1:].std(ddof=1)); lo, hi = pt - critd * se, pt + critd * se
            st = ('REDUCTION_ESTABLISHED' if lo > 0 else ('INCREASE_ESTABLISHED' if hi < 0 else 'UNRESOLVED')) if gates_ok else 'NOT_ESTIMABLE'
            diag.append({'endpoint_id': f'diagnostic__{c["id"]}__{view}', 'point': pt, 'se': se, 'lower': lo, 'upper': hi, 'status': st})
    # compare
    def compare(mine, theirs, pkey):
        tmap = {r['endpoint_id']: r for r in theirs}
        md = 0.0; smis = []
        for r in mine:
            t = tmap.get(r['endpoint_id'])
            if t is None:
                smis.append(f'{r["endpoint_id"]} missing'); continue
            for a, b in (('point', pkey), ('lower', 'lower'), ('upper', 'upper'), ('se', 'se')):
                md = max(md, abs(r[a] - t[b]))
            if r['status'] != t['status']:
                smis.append(f'{r["endpoint_id"]}: {r["status"]} vs {t["status"]}')
        if len(mine) != len(theirs):
            smis.append(f'count {len(mine)} vs {len(theirs)}')
        return md, smis
    pmd, pmis = compare(prim, rep['primary']['endpoints'], 'point')
    dmd, dmis = compare(diag, rep['diagnostic']['endpoints'], 'recovery_reduction_point')
    # service table complete AUC cross-check
    stmd = 0.0
    for row in rep['service_table']:
        for view in ('r1', 'r2', 'pair'):
            stmd = max(stmd, abs(float(envelope(row['service'], str(row['seed']), view)[0]) - row[f'{view}__complete_auc']))
    counts = {s: sum(r['status'] == s for r in prim) for s in ('PASS', 'ESTABLISHED_VIOLATION', 'UNRESOLVED', 'NOT_ESTIMABLE', 'INCOMPLETE_COMPARATOR_INELIGIBLE')}
    dcounts = {}
    for r in diag:
        dcounts[r['status']] = dcounts.get(r['status'], 0) + 1
    res = {'schema': 'TRA1-independent-stage-E-v1', 'unit_id': active['unit_id'], 'rows': N, 'groups': G,
           'critical_primary': crit, 'critical_diagnostic': critd,
           'control_gates': gates, 'control_gates_all_passed': gates_ok, 'control_gate_value_max_abs_diff': gdiff,
           'control_gate_status_mismatch': [k for k, g in gates.items() if g.get('mismatch')],
           'primary': {'K': len(prim), 'endpoints': prim, 'status_counts': counts, 'max_abs_diff_vs_report': pmd, 'status_mismatches': pmis},
           'diagnostic': {'K': len(diag), 'endpoints': diag, 'status_counts': dcounts, 'max_abs_diff_vs_report': dmd, 'status_mismatches': dmis},
           'service_table_complete_auc_max_abs_diff': stmd,
           'envelope_argmax_point': {f'{k[0]}|S{k[1]}|{k[2]}': v for k, v in argmax.items()},
           'primary_utility_all_pass': all(r['status'] == 'PASS' for r in prim if r['kind'] == 'utility'),
           'primary_privacy_eligible_all_pass': all(r['status'] == 'PASS' for r in prim if r['kind'] == 'privacy' and r['control'] in eligible),
           'comparators_point_eligible': eligible, 'cached_curves': len(curves)}
    res['pass'] = bool(len(prim) == 39 and len(diag) == 45 and pmd <= 1e-9 and dmd <= 1e-9 and not pmis and not dmis and gdiff <= 1e-9
                       and not res['control_gate_status_mismatch'] and gates_ok == rep['control_gates_all_passed'] and stmd <= 1e-9)
    return write_out(cfg, res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True)
    print(json.dumps(run(ap.parse_args().spec)['pass']))
