"""Independent Stage B: reader refits (B1) and INNER endpoint recomputation (B2). Queue-only.

Own feature tables, ATT split, scaler, orientation, tie-aware AUC, bootstrap, envelopes,
critical values and statuses. sklearn estimators are instantiated from the slate definitions.
"""
import argparse
import hashlib
import json
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch

from . import core as C

SPLIT_SALT = 'pcrl-method-portfolio-v1|attack-tune|20261008|'
MODEL_SEED_BASE = 2026101205
INNER_NULL_SEED = 2026101207
BOOT_SEED = 2026101301
B = 2048
ALPHA = 0.05
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
SLATE = ('LR_C1', 'MLP_64x32', 'HGB_15x200', 'DA_MLP_64x32', 'REP_LR_C1', 'REP_MLP_64x32', 'REP_HGB_15x200',
         'OUTPUT_LR_C1', 'OUTPUT_HGB_15x200') + tuple(f'{c}_{m}' for c in ('LOGITS_OUTPUT', 'DECISION', 'CONFIDENCE')
                                                    for m in ('LR_C1', 'MLP_64x32', 'HGB_15x200'))


# ------------------------------------------------------------------ AUC / bootstrap
def wauc(y, s, W):
    """Weighted tie-aware Mann-Whitney AUC for each weight row of W (own implementation)."""
    y = np.asarray(y).astype(bool); s = np.asarray(s, np.float64)
    W = np.atleast_2d(np.asarray(W, np.float64))
    uniq, inv = np.unique(s, return_inverse=True)
    out = np.empty(len(W))
    for a in range(0, len(W), 256):
        w = W[a:a + 256]
        P = np.zeros((len(w), len(uniq))); N = np.zeros((len(w), len(uniq)))
        for j in range(len(w)):
            P[j] = np.bincount(inv[y], weights=w[j, y], minlength=len(uniq))
            N[j] = np.bincount(inv[~y], weights=w[j, ~y], minlength=len(uniq))
        below = np.cumsum(N, 1) - N
        out[a:a + 256] = (P * (below + 0.5 * N)).sum(1) / (P.sum(1) * N.sum(1))
    return out


def auc(y, s):
    return float(wauc(y, s, np.ones((1, len(y))))[0])


def boot_weights(unit, n_boot=B, seed=BOOT_SEED):
    g, inv = np.unique(unit, return_inverse=True)
    draws = np.random.default_rng(seed).integers(0, len(g), (n_boot, len(g)))
    return np.stack([np.bincount(d, minlength=len(g))[inv] for d in draws]).astype(np.float64)


def group_permute(y, unit, seed):
    y = np.asarray(y); unit = np.asarray(unit); rng = np.random.default_rng(seed)
    groups = {}
    for gid in np.unique(unit):
        ix = np.flatnonzero(unit == gid)
        assert len(np.unique(y[ix])) == 1
        groups.setdefault(len(ix), []).append(ix)
    out = np.empty_like(y)
    for size in sorted(groups):
        arr = groups[size]
        perm = rng.permutation(len(arr))
        for target, src in zip(arr, perm):
            out[target] = y[arr[int(src)]]
    return out


def tcrit(K, G):
    from scipy.stats import t
    return float(t.ppf(1 - ALPHA / (2 * K), G - 1))


# ------------------------------------------------------------------ readers
def att_split(unit):
    u = np.unique(unit)
    ranked = sorted((hashlib.sha256((SPLIT_SALT + str(int(i))).encode()).hexdigest(), int(i)) for i in u)
    val_groups = {v for _, v in ranked[:1015]}
    val = np.array([int(x) in val_groups for x in unit])
    return np.flatnonzero(~val), np.flatnonzero(val)


def contract(name):
    for p, k in (('LOGITS_OUTPUT_', 'logits_output'), ('DECISION_', 'decision'), ('CONFIDENCE_', 'confidence'),
                 ('DA_', 'defense'), ('REP_', 'representation'), ('OUTPUT_', 'output')):
        if name.startswith(p):
            return k
    return 'full'


def view_table(r1, r2, view, kind):
    def single(x, k):
        x = np.asarray(x, np.float64); p = x[:, 16 + k:16 + 2 * k]
        return {'full': x, 'representation': x[:, :16], 'output': x[:, 16 + k:], 'logits_output': x[:, 16:],
                'decision': x[:, -1:], 'confidence': p.max(1, keepdims=True),
                'defense': np.column_stack((x, np.log(np.maximum(p, np.finfo(float).eps))))}
    if view == 'r1':
        return single(r1, 2)[kind]
    if view == 'r2':
        return single(r2, 6)[kind]
    a, b = single(r1, 2), single(r2, 6)
    if kind != 'defense':
        return np.column_stack((a[kind], b[kind]))
    r1 = np.asarray(r1, np.float64); r2 = np.asarray(r2, np.float64)
    prod = (r1[:, :16, None] * r2[:, None, :16]).reshape(len(r1), -1)
    lg = (r1[:, 16:18, None] * r2[:, None, 16:22]).reshape(len(r1), -1)
    return np.column_stack((a['defense'], b['defense'], prod, lg))


class Scaler:
    def fit(self, x):
        self.m = x.mean(0); sd = x.std(0); self.s = np.where(sd > 0, sd, 1.0); return self

    def __call__(self, x):
        return (x - self.m) / self.s


def make_estimator(name, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    if 'MLP' in name:
        return MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=1e-4, max_iter=300,
                             early_stopping=False, random_state=seed)
    if 'HGB' in name:
        return HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=.05, early_stopping=False, random_state=seed)
    return LogisticRegression(C=1, max_iter=500, solver='lbfgs')


# ------------------------------------------------------------------ wires
def wires_for(kind, X, src_start, geom, ckpt=None, prior_sd=None, own_B=None):
    """Return (r1, r2) full wires as numpy arrays, built from raw83 with my own forward."""
    Bi = torch.from_numpy(np.asarray(geom['heads'][0]['B'], np.float64).reshape(int(geom['heads'][0]['rank']), 16).copy())
    Bo = torch.from_numpy(np.asarray(geom['heads'][1]['B'], np.float64).reshape(int(geom['heads'][1]['rank']), 16).copy())
    src = src_start if prior_sd is None else C.Source(prior_sd)
    W0, b0 = src.head(0); W1, b1 = src.head(1)
    with torch.no_grad():
        zi = src.final(0, src.hidden(0, X))
        r1 = C.raw_proj_wire(zi, Bi, W0, b0)['full'].numpy()
        h = src.hidden(1, X); z0 = src.final(1, h)
        if kind == 'RAW_PROJ':
            r2 = C.raw_proj_wire(z0, Bo, W1, b1)['full']
        elif kind == 'ADAPTER':
            R = ckpt['R'].to(torch.float64); b = ckpt['b_res'].to(torch.float64)
            r2 = C.occ_wire(h.to(torch.float64) / float(ckpt['scale']), z0.to(torch.float64) @ Bo.T, R, b, Bo, W1, b1)['full']
        elif kind == 'PRIOR_HYBRID':
            cl, p, d = C.outputs_from_z(z0.to(torch.float64), W1, b1)
            r2 = torch.cat((z0.to(torch.float64), cl, p, d.to(torch.float64)[:, None]), 1)
        elif kind == 'OWNPROJ':
            r2 = C.raw_proj_wire(z0, own_B, W1, b1)['full']
        else:
            raise ValueError(kind)
    return r1, r2.numpy()


def run_refit(cfg, V, guard, res, dump):
    seed = int(cfg['seed'])
    att = C.npz(V(cfg['att_features'])); sx = C.npz(V(cfg['att_sex'])); inn = C.npz(V(cfg['inner_features']))
    isx = C.npz(V(cfg['inner_sex']))
    assert np.array_equal(att['row_id'], sx['row_id']) and np.array_equal(inn['row_id'], isx['row_id'])
    y = sx['sex'].astype(np.int64); yi = isx['sex'].astype(np.int64)
    tr, va = att_split(att['unit'])
    res['att_split'] = {'train_rows': len(tr), 'val_rows': len(va)}
    start = C.tload(V(cfg['checkpoint'])); src = C.Source(start['model'])
    names_ok = list(map(str, start['feature_names'])) == list(map(str, att['feature_names'])) == list(map(str, inn['feature_names']))
    res['feature_schema_ok'] = names_ok
    geom = json.loads(V(cfg['geometry']).read_text())
    fits = []
    for svc in cfg['services']:
        guard()
        kind = svc['kind']; ck = prior = own = None
        if kind == 'ADAPTER':
            ck = C.tload(V(svc['checkpoint']))
        if kind in ('PRIOR_HYBRID', 'OWNPROJ'):
            prior = C.tload(V(svc['checkpoint']))['model']
        if kind == 'OWNPROJ':
            g = json.loads(V(svc['ownproj']).read_text())['occupation_geometry']
            own = torch.from_numpy(np.asarray(g['B'], np.float64).reshape(int(g['rank']), int(g['source_dimension'])).copy())
        wa = wires_for(kind, att['X'], src, geom, ck, prior, own)
        wi = wires_for(kind, inn['X'], src, geom, ck, prior, own)
        bank = svc['bank']
        rel = C.npz(V(bank['release'])); scores = C.npz(V(bank['scores'])); rec = json.loads(V(bank['receipt']).read_text())
        wcheck = {'r1_maxabs_vs_campaign_release': C_max(wi[0], rel['r1']), 'r2_maxabs_vs_campaign_release': C_max(wi[1], rel['r2']),
                  'r1_decisions_equal': bool(np.array_equal(wi[0][:, -1], rel['r1'][:, -1])),
                  'r2_decisions_equal': bool(np.array_equal(wi[1][:, -1], rel['r2'][:, -1]))}
        for view in svc['views']:
            vals = rec['ATT_validation'][view]
            best = max(SLATE, key=lambda n: (vals[n]['fixed_orientation_auc'], -SLATE.index(n)))
            kd = contract(best)
            assert kd == vals[best]['contract']
            Xtr = view_table(wa[0], wa[1], view, kd)
            sc = Scaler().fit(Xtr[tr])
            est = make_estimator(best, MODEL_SEED_BASE + seed)
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                est.fit(sc(Xtr[tr]), y[tr])
            assert list(est.classes_) == [0, 1]
            pv = est.predict_proba(sc(Xtr[va]))[:, 1]
            sign = 1 if auc(y[va], pv) >= .5 else -1
            pi = est.predict_proba(sc(view_table(wi[0], wi[1], view, kd)))[:, 1]
            oi = pi if sign == 1 else 1 - pi
            camp = scores[f'{view}__{best}']
            fits.append({'service': svc['label'], 'view': view, 'reader': best, 'contract': kd, 'columns': int(Xtr.shape[1]),
                         'att_val_auc_mine': auc(y[va], pv if sign == 1 else 1 - pv), 'att_val_auc_campaign': vals[best]['fixed_orientation_auc'],
                         'orientation_mine': sign, 'orientation_campaign': vals[best]['orientation'],
                         'inner_auc_mine': auc(yi, oi), 'inner_auc_from_campaign_scores': auc(yi, camp),
                         'inner_score_maxabs_vs_campaign': float(np.max(np.abs(oi - camp))), 'fit_s': time.time() - t0, 'wire_check': wcheck})
            res['fits'] = fits; dump()
    res['fit_count'] = len(fits)


def C_max(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b))) if a.shape == b.shape else float('inf')


def run_report(cfg, V, guard, res, dump):
    sel = json.loads(V(cfg['sel_lock']).read_text())
    camp = json.loads(V(cfg['inner_report']).read_text())
    truth = {k: C.npz(V(cfg['inner_truth'][k])) for k in ('income', 'occupation', 'sex')}
    ids = truth['sex']
    for k in truth:
        assert np.array_equal(truth[k]['row_id'], ids['row_id']) and np.array_equal(truth[k]['unit'], ids['unit'])
    y = {k: truth[k][k].astype(np.int64) for k in truth}
    refs = {s: C.npz(V(cfg['inner_references'][str(s)])) for s in (0, 1)}
    banks = {}
    for key, b in cfg['banks'].items():
        banks[key] = {'scores': C.npz(V(b['scores'])), 'release': C.npz(V(b['release']))}
        assert np.array_equal(banks[key]['scores']['row_id'], ids['row_id'])
    n = len(ids['row_id']); G = len(np.unique(ids['unit']))
    W = np.vstack((np.ones((1, n)), boot_weights(ids['unit'])))
    res['rows'] = n; res['groups'] = G
    cache = {}

    def curve(yv, s):
        k = (hashlib.sha256(np.ascontiguousarray(s, np.float64).tobytes()).hexdigest(), hashlib.sha256(yv.tobytes()).hexdigest())
        if k not in cache:
            cache[k] = wauc(yv, s, W)
        return cache[k]

    def svc_scores(label, s):
        r1 = banks[f'R1__S{s}']['scores']; sv = banks[f'{label}__S{s}']['scores']
        out = {f'r1__{n_}': r1[f'r1__{n_}'] for n_ in SLATE}
        out.update({f'{v}__{n_}': sv[f'{v}__{n_}'] for v in ('r2', 'pair') for n_ in SLATE})
        return out
    env_cache = {}

    def envelope(label, s, view):
        k = (label, s, view)
        if k not in env_cache:
            sc = svc_scores(label, s)
            keys = [f'{view}__{n_}' for n_ in SLATE]
            if view == 'pair':
                keys += [f'r1__{n_}' for n_ in SLATE] + [f'r2__{n_}' for n_ in SLATE]  # IGNORE_r2 / IGNORE_r1 local aliases
            env_cache[k] = np.max(np.stack([curve(y['sex'], sc[q]) for q in keys]), 0)
        return env_cache[k]
    # control gates
    null_truth = group_permute(y['sex'], ids['unit'], INNER_NULL_SEED)
    gates = {}
    for s in (0, 1):
        c = banks[f'CONTROLS__S{s}']['scores']
        for view in ('r1', 'r2', 'pair'):
            sh_views = [view] + (['r1', 'r2'] if view == 'pair' else [])
            sh = [auc(null_truth, c[f'{v}__SHUFFLED__{n_}']) for v in sh_views for n_ in SLATE]
            pl = [auc(y['sex'], np.where(y['sex'] == 1, c[f'{view}__PLANTED_1__{n_}'], c[f'{view}__PLANTED_0__{n_}'])) for n_ in SLATE]
            vals = {'null_auc': auc(y['sex'], np.full(n, .5)), 'shuffled_envelope_max_two_sided': max(max(a, 1 - a) for a in sh),
                    'planted_envelope': max(pl), 'planted_weakest_reader': min(pl)}
            ok = vals['null_auc'] == .5 and vals['shuffled_envelope_max_two_sided'] <= .60 and vals['planted_envelope'] >= .99
            cg = camp['control_gates'][f'S{s}__{view}']
            gates[f'S{s}__{view}'] = {'values': vals, 'passed': ok, 'campaign_passed': cg['passed'],
                                      'maxabs_vs_campaign': max(abs(vals[k] - cg['values'][k]) for k in vals)}
    gates_ok = all(g['passed'] for g in gates.values())
    res['control_gates'] = gates; res['control_gates_all_passed'] = gates_ok; dump()

    def vectors(yv, p):
        p = np.asarray(p, np.float64); k = p.shape[1]
        onehot = (yv[:, None] == np.arange(k)).astype(np.float64)
        return {'accuracy': (p.argmax(1) == yv).astype(np.float64),
                'nll': -np.log(np.maximum(p[np.arange(len(yv)), yv], 1e-12)),
                'brier': ((p - onehot) ** 2).sum(1)}

    def utility(label, s):
        rel = banks[f'{label}__S{s}']['release']; out = {}
        for i, task in enumerate(('income', 'occupation')):
            k = 2 if i == 0 else 6; w = rel[f'r{i + 1}']; p = w[:, 16 + k:16 + 2 * k]
            assert np.array_equal(w[:, -1], p.argmax(1))
            own = vectors(y[task], p)
            for ref in ('U0', 'Ucal'):
                rv = vectors(y[task], refs[s][f'{ref}_{i + 1}'])
                out[(task, ref)] = {'accuracy_loss': rv['accuracy'] - own['accuracy'], 'nll_excess': own['nll'] - rv['nll'],
                                    'brier_excess': own['brier'] - rv['brier']}
        return out

    def status(point, lo, hi, thr, direction):
        if direction == 'at_most':
            return 'PASS' if hi <= thr else ('ESTABLISHED_VIOLATION' if lo > thr else 'UNRESOLVED')
        return 'PASS' if lo > thr else ('ESTABLISHED_VIOLATION' if hi < thr else 'UNRESOLVED')
    tot = W.sum(1)
    prim = sel['primary_family']; K = prim['K']; cr = tcrit(K, G); cand = prim['candidate']
    rows = []
    for s in (0, 1):
        u = utility(cand, s)
        for task in ('income', 'occupation'):
            for ref in ('U0', 'Ucal'):
                for m, cap in CAPS.items():
                    v = (W @ u[(task, ref)][m]) / tot
                    pt, se = float(v[0]), float(v[1:].std(ddof=1)); lo, hi = pt - cr * se, pt + cr * se
                    rows.append({'endpoint_id': f'utility__{cand}__S{s}__{task}__{ref}__{m}', 'point': pt, 'se': se, 'lower': lo, 'upper': hi,
                                 'status': status(pt, lo, hi, cap, 'at_most')})
    for ctl in prim['roster']:
        for view, (kind, thr) in (('pair', ('benefit', .020)), ('r1', ('excess', .010)), ('r2', ('excess', .010))):
            parts = [(envelope(ctl, s, view) - envelope(cand, s, view)) if kind == 'benefit' else (envelope(cand, s, view) - envelope(ctl, s, view)) for s in (0, 1)]
            v = np.mean(parts, 0); pt, se = float(v[0]), float(v[1:].std(ddof=1)); lo, hi = pt - cr * se, pt + cr * se
            st = status(pt, lo, hi, thr, 'greater_than' if kind == 'benefit' else 'at_most') if gates_ok else 'NOT_ESTIMABLE'
            rows.append({'endpoint_id': f'privacy__{cand}__vs__{ctl}__{view}_{kind}', 'point': pt, 'se': se, 'lower': lo, 'upper': hi, 'status': st})
    cmap = {r['endpoint_id']: r for r in camp['primary']['endpoints']}
    pd = {'point': 0.0, 'se': 0.0, 'lower': 0.0, 'upper': 0.0}; smis = []
    for r in rows:
        c = cmap[r['endpoint_id']]
        for k in pd:
            pd[k] = max(pd[k], abs(r[k] - c[k]))
        if r['status'] != c['status']:
            smis.append(r['endpoint_id'])
    counts = {k: sum(r['status'] == k for r in rows) for k in ('PASS', 'ESTABLISHED_VIOLATION', 'UNRESOLVED', 'NOT_ESTIMABLE')}
    res['primary'] = {'K': K, 'critical': cr, 'n_endpoints': len(rows), 'endpoint_ids_match': set(cmap) == {r['endpoint_id'] for r in rows},
                      'maxabs_vs_campaign': pd, 'status_mismatches': smis, 'status_counts': counts,
                      'campaign_status_counts': camp['primary']['status_counts'], 'endpoints': rows}
    dump()
    diag = sel['diagnostic_family']; Kd = diag['K']; crd = tcrit(Kd, G); drows = []
    for c in diag['contrasts']:
        for view in ('r1', 'r2', 'pair'):
            parts = [envelope(c['control'], s, view) - envelope(c['candidate'], s, view) for s in (0, 1)]
            v = np.mean(parts, 0); pt, se = float(v[0]), float(v[1:].std(ddof=1)); lo, hi = pt - crd * se, pt + crd * se
            st = ('REDUCTION_ESTABLISHED' if lo > 0 else ('INCREASE_ESTABLISHED' if hi < 0 else 'UNRESOLVED')) if gates_ok else 'NOT_ESTIMABLE'
            drows.append({'endpoint_id': f'diagnostic__{c["id"]}__{view}', 'point': pt, 'se': se, 'lower': lo, 'upper': hi, 'status': st})
    dmap = {r['endpoint_id']: r for r in camp['diagnostic']['endpoints']}
    dd = 0.0; dmis = []
    for r in drows:
        c = dmap[r['endpoint_id']]
        dd = max(dd, abs(r['point'] - c['recovery_reduction_point']), abs(r['se'] - c['se']), abs(r['lower'] - c['lower']), abs(r['upper'] - c['upper']))
        if r['status'] != c['status']:
            dmis.append(r['endpoint_id'])
    res['diagnostic'] = {'K': Kd, 'critical': crd, 'n_endpoints': len(drows), 'maxabs_vs_campaign': dd, 'status_mismatches': dmis,
                         'status_counts': {k: sum(r['status'] == k for r in drows) for k in ('REDUCTION_ESTABLISHED', 'INCREASE_ESTABLISHED', 'UNRESOLVED', 'NOT_ESTIMABLE')},
                         'endpoints': drows}
    # service table complete AUCs
    sd = 0.0
    for row in camp['service_table']:
        for view in ('r1', 'r2', 'pair'):
            sd = max(sd, abs(float(envelope(row['service'], row['seed'], view)[0]) - row[f'{view}__complete_auc']))
    res['service_table_complete_auc_maxabs'] = sd
    res['cached_curves'] = len(cache)
    dump()


def main(spec_path):
    from minimal_task_wire_adapter.evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'verification', allow_inner_truth=True)
    torch.set_num_threads(1)
    V = lambda it: verify(it['path'], it['sha256'])
    t0 = time.time()
    res = {'unit_id': active['unit_id'], 'operation': cfg['operation']}
    out = Path(cfg['output'])

    def dump():
        res['elapsed_s'] = time.time() - t0
        out.write_text(json.dumps(res, indent=1, sort_keys=True, default=float))
    try:
        (run_refit if cfg['operation'] == 'refit' else run_report)(cfg, V, guard, res, dump)
    except Exception:
        res['error'] = traceback.format_exc()
    dump()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True)
    main(ap.parse_args().spec)
