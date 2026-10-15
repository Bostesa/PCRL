"""Equally exposed neural task reference (N_EQ).

Architecture: the validated source per-task encoder (Linear 83->64, ReLU,
Linear 64->64, ReLU, Linear 64->16) followed by a linear K-way head, i.e. the
UsefulModel function class trained end to end. Training protocol: the
inherited validated Recipe A (Adam lr 1e-3, batch 512, at most 30 epochs,
15% group-disjoint internal validation, patience 5, min delta 1e-4, refit from
scratch on the whole fitting pool for the selected epoch count). No temperature
or calibration. Same fitting pool and internal split as the tree teachers.
"""
from __future__ import annotations
import numpy as np

SPEC = {'learning_rate': 1e-3, 'batch_size': 512, 'max_epochs': 30, 'patience': 5, 'min_delta_nll': 1e-4,
        'validation_fraction': 0.15, 'architecture': '83-64-ReLU-64-ReLU-16-linear K', 'calibration': None}


def network(k):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(83, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                               torch.nn.Linear(64, 16), torch.nn.Linear(16, k))


def _epoch(model, opt, X, y, rows, seed):
    import torch
    model.train()
    order = np.random.default_rng(seed).permutation(rows)
    for s in range(0, len(order), SPEC['batch_size']):
        b = order[s:s + SPEC['batch_size']]
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(torch.from_numpy(X[b])), torch.from_numpy(y[b]))
        loss.backward(); opt.step()


def _nll(model, X, y, rows):
    import torch
    model.eval()
    with torch.no_grad():
        return float(torch.nn.functional.cross_entropy(model(torch.from_numpy(X[rows])), torch.from_numpy(y[rows])))


def fit(X, y, k, fit_rows, val_rows, seed):
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    X = np.ascontiguousarray(X, dtype=np.float32); y = np.ascontiguousarray(y, dtype=np.int64)
    torch.manual_seed(seed); m = network(k); opt = torch.optim.Adam(m.parameters(), lr=SPEC['learning_rate'])
    best, best_epoch, bad, trace = float('inf'), 0, 0, []
    for e in range(1, SPEC['max_epochs'] + 1):
        _epoch(m, opt, X, y, fit_rows, seed + 20000 + e)
        v = _nll(m, X, y, val_rows); trace.append(v)
        if v < best - SPEC['min_delta_nll']:
            best, best_epoch, bad = v, e, 0
        else:
            bad += 1
            if bad >= SPEC['patience']:
                break
    torch.manual_seed(seed); final = network(k); opt = torch.optim.Adam(final.parameters(), lr=SPEC['learning_rate'])
    allrows = np.arange(len(X))
    for e in range(1, best_epoch + 1):
        _epoch(final, opt, X, y, allrows, seed + 30000 + e)
    final.eval()
    return final, {'best_epoch': best_epoch, 'best_internal_nll': best, 'internal_trace': trace,
                   'epochs_run': len(trace), 'refit_rows': int(len(X))}


def probabilities(model, X):
    import torch
    with torch.no_grad():
        logits = model(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))).to(torch.float64)
        logits = logits - logits.mean(1, keepdim=True)
        return torch.softmax(logits, 1).numpy()
