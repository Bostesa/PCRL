"""Independent replay of final scores on a second host with a separate scorer.

Separate code path from the main report generator:
  * token laws rebuilt directly from Q[code] / declared RR and W formulas (not build_release);
  * H/aux wires assembled directly; predictions from the saved candidate objects;
  * expected loss by an independent einsum formula;
  * headline point estimates by per-anchor weighted means; SEs by a fresh numpy bootstrap
    (different seed, 2,000 draws) for an order-of-magnitude check only.
Cross-host tolerance: 2e-6 absolute on per-person loss (float32 MLP arithmetic differs by platform).
"""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit, logit

from experiments.pcrl_task_directed_release_v1.audits import load_candidate
from experiments.pcrl_task_directed_release_v1.baselines import AffineEraser
from .common import ANCHORS, OUT, PANEL, PRIMARY_ROLES, RESTORED, TASK_ROLE, WEIGHTINGS, atomic_json, now

TOL = 2e-6


def token_law(short, feats, anchor):
    base = RESTORED/f'anchor_{anchor}'
    codes = feats['codes_T0']
    n = len(codes)
    if short in ('Q', 'D17', 'D33', 'RR75', 'W75'):
        name = {'Q': 'T0_L_0.01_a17', 'D33': 'T0_U_unconstrained_a33'}.get(short, 'T0_U_unconstrained_a17')
        q = np.asarray(joblib.load(base/'maps'/name/'solution.joblib')['Q'], float)[codes]
        if short == 'RR75':
            q = .75*q+.25/q.shape[1]
        if short == 'W75':
            q = np.column_stack((.75*q, np.full(n, .25)))
        return q
    return np.ones((n, 1))


def aux(short, feats, anchor):
    if short == 'J':
        return feats['J']
    if short == 'C':
        return feats['p'][:, None]
    if short in ('E', 'S'):
        method = 'leace_supervised' if short == 'E' else 'splince_supervised'
        eraser = AffineEraser.load(RESTORED/f'anchor_{anchor}'/'baseline_supplement/mechanism40'/method)
        x = np.column_stack((feats['x'], logit(np.clip(feats['p'], 1e-5, 1-1e-5))))
        return (x-eraser.mean)@eraser.projection.T+eraser.mean
    return None


def fixed_decoder(short, feats):
    b = feats['b']
    if short in ('Q', 'D17', 'RR75'):
        return feats['actions17']
    if short == 'D33':
        return feats['actions33']
    if short == 'W75':
        return np.column_stack((feats['actions17'], b))
    if short == 'C':
        return feats['p'][:, None]
    if short == 'H':
        return b[:, None]
    return None


def predict(record, root, feats, short, anchor):
    route = record['route']
    tokens = token_law(short, feats, anchor)
    if route['kind'] == 'fixed_decoder':
        pos = fixed_decoder(short, feats)
        return np.stack((1-pos, pos), 2), tokens
    if route['kind'] == 'global_offset':
        pos = feats['global_offsets'][:, [route['action_index']]]
        return np.stack((1-pos, pos), 2), np.ones((len(pos), 1))
    view, wire = route['source_view'], route['wire']
    h = {'A': feats['ha'], 'B': feats['hb'], 'AB': np.column_stack((feats['ha'], feats['hb']))}[view]
    if wire == 'H' or view == 'B':
        return load_candidate(Path(root)/record['model_path']).predict_token_proba(h, 1), np.ones((len(h), 1))
    a = aux(short, feats, anchor)
    if a is not None:
        h = np.column_stack((h, a))
    return load_candidate(Path(root)/record['model_path']).predict_token_proba(h, tokens.shape[1]), tokens


def expected_loss(q, p, y):
    picked = np.take_along_axis(q, y[:, None, None].repeat(q.shape[1], 1), 2)[:, :, 0]
    return np.einsum('nt,nt->n', p, -np.log(np.maximum(picked, 1e-9)))


def replay(root, features, labels, units):
    """features: {anchor: final-pool feature dict}; labels: final label dict; units: list of (short, anchor, role)."""
    root = Path(root)
    rows = []
    for short, anchor, role in units:
        d = root/'units'/f'anchor_{anchor}'/short/role.replace(':', '__').replace('/', '__')
        reg = joblib.load(d/'registry.joblib')
        with np.load(d/'score_final.npz') as z:
            stored = {k: z[k] for k in z.files}
        target = role.split('/')[1]
        y = labels[target]
        mask = y >= 0
        feats = {k: v[mask] for k, v in features[anchor].items()}
        q, p = predict(reg['candidates'][reg['selection']], root, feats, short, anchor)
        loss = expected_loss(q, p, y[mask])
        err = np.abs(loss-stored['loss'])
        rows.append({'release': short, 'anchor': anchor, 'role': role, 'selection': reg['selection'], 'rows': int(mask.sum()),
                     'max_abs_loss_error': float(err.max()), 'rows_over_tolerance': int((err > TOL).sum()),
                     'mean_loss_replay': float(loss.mean()), 'mean_loss_stored': float(stored['loss'].mean())})
    return rows


def point_estimates(root, endpoints):
    """Independent per-anchor weighted means from stored per-person losses."""
    root = Path(root)
    cache = {}

    def get(short, a, role):
        key = (short, a, role)
        if key not in cache:
            d = root/'units'/f'anchor_{a}'/short/role.replace(':', '__').replace('/', '__')
            with np.load(d/'score_final.npz') as z:
                cache[key] = (z['loss'], z['weights'], z['households'])
        return cache[key]
    out = {}
    for e in endpoints:
        vals = []
        for a in ANCHORS:
            lp, w, _ = get(e['plus'], a, e['role']); lm, _, _ = get(e['minus'], a, e['role'])
            w = np.ones_like(lp) if e['weighting'] == 'unweighted' else w
            vals.append(float(np.sum(w*(lp-lm))/np.sum(w)))
        out[e['id']] = {'estimate': float(np.mean(vals)), 'anchors': vals}
    return out, cache


def bootstrap_se(endpoints, cache, n_boot=2000, seed=777):
    hh = cache[next(iter(cache))][2]
    uniq, inv = np.unique(hh, return_inverse=True)
    rng = np.random.default_rng(seed)
    mult = rng.multinomial(len(uniq), np.full(len(uniq), 1/len(uniq)), size=n_boot).T  # households x draws
    out = {}
    for e in endpoints:
        ratios = []
        for a in ANCHORS:
            lp, w, h = cache[(e['plus'], a, e['role'])]; lm = cache[(e['minus'], a, e['role'])][0]
            idx = np.searchsorted(uniq, h)
            if not np.array_equal(uniq[idx], h):
                raise ValueError('Household outside the common final-household union')
            w = np.ones_like(lp) if e['weighting'] == 'unweighted' else w
            num = np.bincount(idx, weights=w*(lp-lm), minlength=len(uniq)); den = np.bincount(idx, weights=w, minlength=len(uniq))
            ratios.append((num@mult)/(den@mult))
        out[e['id']] = float(np.std(np.mean(ratios, 0), ddof=1))
    return out
