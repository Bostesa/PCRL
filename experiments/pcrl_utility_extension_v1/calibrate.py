"""Tier 1b: bounded attacker calibration on the 2018 fitting/validation pools.

At most EIGHT prespecified recipes (frozen here, before any calibration outcome), applied to
H, untouched A0, J and LEACE-on-A0 on the historical attacker subset (4096 rows of
attacker_fit, historical indices), early-stopped on an internal household split OF THAT
SUBSET, compared on attacker_validation. The evaluation (test) split is read only for a
recipe that is adopted, and then identically for every interface.

Prespecified definition of "stronger" (declared before running):
  a recipe is STRONGER iff, averaged over seeds and both weightings,
   (i)  its A0 full-view validation log loss is lower than the standard slate's selected
        validation loss on at least 3 of the 4 sensitive endpoints, and
   (ii) its A0-minus-H validation increment (recipe on both views) is at least the standard
        slate's A0-minus-H validation increment on average over the 4 endpoints.
If several qualify, the one with the lowest mean A0 full-view validation loss is adopted.
An adopted recipe is ADDED to the frozen slate (per endpoint, the candidate with lower
validation loss is selected), never substituted, and is applied to every interface
including H, so it cannot lower a measured increment by replacing a stronger candidate.
If none qualifies, the standard slate is retained and added stress strength is recorded as
NOT established.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import acs_fixed_predictions_audits as oldaudit
from experiments.run_acs_coalition import TARGETS as COALITION_TARGETS
from experiments.run_acs_residual_spectral import load_labels
from experiments.run_acs_transfer import subset_indices
from experiments.pcrl_nonlinear_rank_v1.inputs import HIST_ROOT

ENDPOINTS = ('A/SEX', 'AB/SEX', 'A/RAC1P', 'AB/RAC1P')
CLASSES = {'SEX': 2, 'RAC1P': 9}

RECIPES = {
    'c1_mlp64x32_wd1e-4': dict(hidden=(64, 32), dropout=0.0, wd=1e-4, lr=1e-3),
    'c2_mlp128x64_do.2': dict(hidden=(128, 64), dropout=0.2, wd=1e-4, lr=1e-3),
    'c3_mlp64_wd1e-3': dict(hidden=(64,), dropout=0.0, wd=1e-3, lr=1e-3),
    'c4_mlp64x32_lr3e-4': dict(hidden=(64, 32), dropout=0.0, wd=1e-4, lr=3e-4),
    'c5_mlp256x128_do.3_wd1e-3': dict(hidden=(256, 128), dropout=0.3, wd=1e-3, lr=1e-3),
    'c6_mlp128x128_do.1_lr5e-4': dict(hidden=(128, 128), dropout=0.1, wd=1e-4, lr=5e-4),
    'c7_bag3_mlp64x32': dict(hidden=(64, 32), dropout=0.0, wd=1e-4, lr=1e-3, bag=3),
    'c8_mlp32x16_wd1e-4': dict(hidden=(32, 16), dropout=0.0, wd=1e-4, lr=1e-3),
}
MAX_EPOCHS = 200
PATIENCE = 12
BATCH = 256
FLOOR = 1e-12


def _net(n_in, n_out, hidden, dropout, seed):
    torch.manual_seed(seed)
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else [])
        d = h
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def _fit_predict(x_fit, y_fit, households, x_eval: dict, classes, recipe, seed):
    """Early-stopped on an internal household split of the attacker subset; returns probs."""
    mean, scale = x_fit.mean(0), x_fit.std(0)
    scale[scale < 1e-8] = 1.0
    tr_mask = np.array([int(hashlib.sha256(f'calib/{h}'.encode()).hexdigest()[:8], 16) % 5 != 0
                        for h in households])
    xt = torch.tensor((x_fit - mean) / scale, dtype=torch.float32)
    yt = torch.from_numpy(y_fit.astype(np.int64))
    tr, va = np.flatnonzero(tr_mask), np.flatnonzero(~tr_mask)
    outs = []
    for b in range(recipe.get('bag', 1)):
        net = _net(x_fit.shape[1], classes, recipe['hidden'], recipe['dropout'], seed + 17 * b)
        opt = torch.optim.AdamW(net.parameters(), lr=recipe['lr'], weight_decay=recipe['wd'])
        rng = np.random.default_rng(seed + 17 * b)
        best, best_state, wait, epochs = np.inf, None, 0, 0
        for epoch in range(MAX_EPOCHS):
            net.train()
            order = rng.permutation(tr)
            for s in range(0, len(order), BATCH):
                ix = torch.from_numpy(order[s:s + BATCH])
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(net(xt[ix]), yt[ix]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(F.cross_entropy(net(xt[va]), yt[va]))
            epochs = epoch + 1
            if v < best - 1e-6:
                best, best_state, wait = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            outs.append({k: torch.softmax(net(torch.tensor((x - mean) / scale, dtype=torch.float32)), 1)
                         .numpy().astype(np.float64) for k, x in x_eval.items()})
    return {k: np.mean([o[k] for o in outs], 0) for k in x_eval}, {'epochs_last': epochs, 'inner_best': best}


def log_loss(p, y, w=None):
    p = np.clip(p, FLOOR, 1.0)
    p = p / p.sum(1, keepdims=True)
    ll = -np.log(p[np.arange(len(y)), y])
    return float(ll.mean() if w is None else (ll * w).sum() / w.sum())


def views(release: dict, pool: str) -> dict:
    a, ab = release[f'wire/A/{pool}'], release[f'wire/AB/{pool}']
    return {'A': a, 'AB': ab, 'H_A': a[:, :4], 'H_AB': np.column_stack([a[:, :4], ab[:, -2:]])}


def calibrate_seed(seed: int, releases: dict, recipes=RECIPES, splits=('validation',)) -> dict:
    """releases: {interface_name: npz-like dict}. Returns per-recipe per-interface losses."""
    tick = time.perf_counter()
    frame, pools, labels, weights = load_labels(HIST_ROOT, seed)
    ai = {t: subset_indices(labels['attacker_fit'][t], 4096, 1240000 + 100 * seed + j)
          for j, t in enumerate(COALITION_TARGETS)}
    serial = {p: frame.iloc[pools[p]].SERIALNO.astype(str).to_numpy() for p in pools}
    pool_of = {'validation': 'attacker_validation', 'test': 'test'}
    out = {}
    for rname, recipe in recipes.items():
        out[rname] = {}
        for iname, rel in releases.items():
            per = {}
            for endpoint in ENDPOINTS:
                view, target = endpoint.split('/')
                hview = 'H_A' if view == 'A' else 'H_AB'
                ix = ai[target]
                y_fit = labels['attacker_fit'][target][ix]
                for vname in (view, hview):
                    x_fit = views(rel, 'attacker_fit')[vname][ix]
                    evals = {}
                    for split in splits:
                        pool = pool_of[split]
                        y = labels[pool][target]
                        evals[split] = views(rel, pool)[vname][y >= 0]
                    seed_r = 41000 + 100 * seed + 10 * ENDPOINTS.index(endpoint) + list(recipes).index(rname)
                    probs, info = _fit_predict(x_fit, y_fit, serial['attacker_fit'][ix], evals,
                                               CLASSES[target], recipe, seed_r)
                    for split in splits:
                        pool = pool_of[split]
                        y = labels[pool][target]
                        valid = y >= 0
                        for wname, w in (('unweighted', None), ('person_weighted', weights[pool][valid])):
                            per[f'{endpoint}|{"full" if vname == view else "H"}|{split}|{wname}'] = \
                                log_loss(probs[split], y[valid], w)
                    per[f'{endpoint}|{"full" if vname == view else "H"}|info'] = info
            out[rname][iname] = per
    out['_seconds'] = time.perf_counter() - tick
    return out


def decide(calib: dict, standard: dict, seeds) -> dict:
    """standard[seed][interface][endpoint|weight] = standard-slate selected VALIDATION loss for
    full views; standard[seed]['H'][endpoint|weight] for the H interface. Applies the frozen rule."""
    weights = ('unweighted', 'person_weighted')
    rows = {}
    for rname in RECIPES:
        better, inc_r, inc_s, a0loss = 0, [], [], []
        for e in ENDPOINTS:
            diff = []
            for s in seeds:
                for w in weights:
                    r_full = calib[s][rname]['A0'][f'{e}|full|validation|{w}']
                    r_h = calib[s][rname]['A0'][f'{e}|H|validation|{w}']
                    s_full = standard[s]['A0'][f'{e}|{w}']
                    s_h = standard[s]['H'][f'{e}|{w}']
                    diff.append(s_full - r_full)
                    inc_r.append(r_h - r_full)
                    inc_s.append(s_h - s_full)
                    a0loss.append(r_full)
            better += int(np.mean(diff) > 0)
        rows[rname] = {'A0_endpoints_better_than_standard': better,
                       'recipe_A0_minus_H_increment': float(np.mean(inc_r)),
                       'standard_A0_minus_H_increment': float(np.mean(inc_s)),
                       'mean_A0_full_validation_loss': float(np.mean(a0loss))}
        rows[rname]['stronger'] = bool(better >= 3 and rows[rname]['recipe_A0_minus_H_increment']
                                       >= rows[rname]['standard_A0_minus_H_increment'])
    qualified = sorted((r for r in rows if rows[r]['stronger']), key=lambda r: rows[r]['mean_A0_full_validation_loss'])
    return {'per_recipe': rows, 'adopted': qualified[0] if qualified else None,
            'stress_strength_established': bool(qualified),
            'rule': __doc__.split('Prespecified definition')[1].split('"""')[0].strip()}
