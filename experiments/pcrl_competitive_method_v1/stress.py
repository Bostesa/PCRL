"""The single prespecified stronger attack (PROTOCOL §7), frozen before any panel outcome.

`MLP[256, 256, 128]` — wider and deeper than anything in the training ensembles or the
standard audit slate — Adam `1e-3`, weight decay `1e-5`, batch 256, two fresh
initialisations, 240 epochs with snapshots every 40, fitted on the historical
`attacker_fit` subset, snapshot/initialisation chosen on `attacker_validation`
(unweighted log loss, then candidate id), scored once on `test`. The identical recipe
is run on the `H`-only views, so the increment over `H` uses the same attacker on both
sides. Applied identically to every stressed interface: candidates, their controls and
the external comparators.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.pcrl_direct_adversarial_v1.stress import release_path as hist_release_path
from experiments.run_acs_coalition import TARGETS as COALITION_TARGETS
from experiments.run_acs_residual_spectral import load_labels, wires
from experiments.run_acs_transfer import subset_indices
from experiments.pcrl_nonlinear_rank_v1.inputs import HIST_ROOT

from . import evidence as ev
from .common import (ADVERSARIAL_RESULTS, OUT, limit_threads, read_json, sha_file, utcnow,
                     write_json_atomic)

ROLES = (('A', 'SEX'), ('A', 'RAC1P'), ('AB', 'SEX'), ('AB', 'RAC1P'))
CLASSES = {'SEX': 2, 'RAC1P': 9}
HIDDEN = (256, 256, 128)
EPOCHS, SNAPSHOT, BATCH, LR, WD = 240, 40, 256, 1e-3, 1e-5
INITS = (0, 1)
FLOOR = 1e-12


def network(width: int, classes: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    layers, previous = [], width
    for size in HIDDEN:
        layers += [nn.Linear(previous, size), nn.ReLU()]
        previous = size
    layers.append(nn.Linear(previous, classes))
    return nn.Sequential(*layers)


def log_loss(p: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> float:
    p = np.clip(p, FLOOR, 1.0)
    p = p / p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y])
    return float(loss.mean() if w is None else (loss * w).sum() / w.sum())


def fit_role(xf, yf, xv, yv, classes, seed):
    mean, scale = xf.mean(0), xf.std(0)
    scale[scale <= 1e-12] = 1.0
    norm = lambda x: torch.tensor((x - mean) / scale, dtype=torch.float32)
    tf, tv = norm(xf), norm(xv)
    yf_t = torch.from_numpy(yf.astype(np.int64))
    candidates = []
    for init in INITS:
        net = network(xf.shape[1], classes, seed + init)
        opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
        rng = np.random.default_rng(seed + 100 + init)
        for epoch in range(1, EPOCHS + 1):
            order = rng.permutation(len(xf))
            net.train()
            for start in range(0, len(order), BATCH):
                ix = torch.from_numpy(order[start:start + BATCH])
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(net(tf[ix]), yf_t[ix])
                if not torch.isfinite(loss):
                    raise AssertionError('non-finite stress loss')
                loss.backward()
                opt.step()
            if epoch % SNAPSHOT == 0:
                net.eval()
                with torch.no_grad():
                    pv = torch.softmax(net(tv).double(), 1).numpy()
                candidates.append({'id': f'init{init}_epoch{epoch:03d}',
                                   'validation_log_loss': log_loss(pv, yv),
                                   'state': {k: v.clone() for k, v in net.state_dict().items()}})
    best = min(candidates, key=lambda c: (c['validation_log_loss'], c['id']))
    net = network(xf.shape[1], classes, 0)
    net.load_state_dict(best['state'])
    net.eval()
    return net, norm, best, [{k: c[k] for k in ('id', 'validation_log_loss')}
                             for c in candidates]


def features(w: dict, view: str, pool: str, h_only: bool) -> np.ndarray:
    x = np.asarray(w[view][pool], np.float64)
    if not h_only:
        return x
    return x[:, :4] if view == 'A' else np.column_stack([x[:, :4], x[:, -2:]])


def release_for(seed: int, condition: str) -> Path:
    local = OUT / f'seed_{seed}' / 'releases' / condition / 'releases.npz'
    return local if local.exists() else hist_release_path(ADVERSARIAL_RESULTS, seed, condition)


def run_condition(seed: int, condition: str, frame_cache: dict) -> dict:
    dest = OUT / 'stress' / f'seed_{seed}' / condition
    marker = dest / 'stress_complete.json'
    if marker.exists():
        return read_json(marker)
    tick = time.perf_counter()
    if seed not in frame_cache:
        frame_cache[seed] = load_labels(HIST_ROOT, seed)
    _frame, _pools, labels, weights = frame_cache[seed]
    ai = {t: subset_indices(labels['attacker_fit'][t], 4096, 1240000 + 100 * seed + j)
          for j, t in enumerate(COALITION_TARGETS)}
    h_only = condition == 'H'
    path = release_for(seed, 'ref_A0' if h_only else condition)
    w, _ = wires(path)
    rows, selection = [], {}
    for index, (view, target) in enumerate(ROLES):
        classes = CLASSES[target]
        vv = labels['attacker_validation'][target] >= 0
        vt = labels['test'][target] >= 0
        xf = features(w, view, 'attacker_fit', h_only)[ai[target]]
        yf = labels['attacker_fit'][target][ai[target]]
        keep = yf >= 0
        net, norm, best, cands = fit_role(
            xf[keep], yf[keep], features(w, view, 'attacker_validation', h_only)[vv],
            labels['attacker_validation'][target][vv], classes,
            31000 + 100 * seed + 10 * index)
        selection[f'{view}/{target}'] = {'selected': best['id'],
                                         'validation_log_loss': best['validation_log_loss'],
                                         'candidates': cands}
        with torch.no_grad():
            pt = torch.softmax(net(norm(features(w, view, 'test', h_only)[vt])).double(),
                               1).numpy()
        if not (np.isfinite(pt).all() and np.allclose(pt.sum(1), 1, atol=1e-6)):
            raise AssertionError(f'invalid stress probabilities {condition} {view}/{target}')
        yt = labels['test'][target][vt]
        wt = np.asarray(weights['test'])[vt]
        prior = ev.prior(seed)[target]
        for weight, loss in (('unweighted', log_loss(pt, yt)),
                             ('person_weighted', log_loss(pt, yt, wt))):
            key = 'test' if weight == 'unweighted' else 'test_person_weighted'
            rows.append({'seed': seed, 'condition': condition, 'endpoint': f'{view}/{target}',
                         'weight': weight, 'test_log_loss': loss,
                         'gain_over_prior': prior[key]['log_loss'] - loss})
    dest.mkdir(parents=True, exist_ok=True)
    write_json_atomic(dest / 'selection_before_test.json',
                      {'seed': seed, 'condition': condition, 'selection': selection,
                       'test_pool_read_before_selection': False})
    record = {'seed': seed, 'condition': condition, 'rows': rows,
              'release': str(path.name), 'release_sha256': sha_file(path),
              'recipe': {'hidden': HIDDEN, 'epochs': EPOCHS, 'snapshot_every': SNAPSHOT,
                         'inits': len(INITS), 'lr': LR, 'weight_decay': WD, 'batch': BATCH},
              'runtime_seconds': time.perf_counter() - tick, 'utc': utcnow()}
    write_json_atomic(marker, record)
    print('STRESS_DONE', seed, condition, round(record['runtime_seconds'], 1), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--conditions', nargs='+', required=True)
    args = parser.parse_args()
    limit_threads()
    cache = {}
    for condition in ['H'] + [c for c in args.conditions if c != 'H']:
        run_condition(args.seed, condition, cache)


if __name__ == '__main__':
    main()
