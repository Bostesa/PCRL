"""Executable positive controls and traps (synthetic only; no real data).

xor_attacker_control   A, B ~ Bern(1/2) iid, S = A xor B: each bit alone has zero linear and nonlinear recovery,
                       the pair is recovered by a nonlinear attacker.
xor_training_control   X = [Y1, Y2, A, B, noise]; tasks Y1, Y2; S = A xor B. Local critics see nothing; only the
                       coalition critic can push one encoder to drop A or B. Training positive control (J vs L).
task_equals_s          task 1 = S: preserving it while hiding S is impossible; the runner must report the conflict
                       (guards bind, protection steps rejected / utility gate fails), never repair thresholds.
constant_release       constant features/predictions: recovery 0.5 and zero useful gain -> fails usefulness gates.
"""
from __future__ import annotations

import numpy as np
import torch

from jcv import audit as A
from jcv import train as T


def xor_data(n, seed, noise=4, task_eq_s=False):
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 2, n), rng.integers(0, 2, n)
    s = a ^ b
    y1 = s.copy() if task_eq_s else rng.integers(0, 2, n)
    y2 = rng.integers(0, 2, n)
    X = np.column_stack([y1, y2, a, b, rng.normal(size=(n, noise))]).astype(np.float32)
    X[:, :4] = 2 * X[:, :4] - 1
    return X, y1.astype(np.int64), y2.astype(np.int64), s.astype(np.int64), a, b


def split(n, seed):
    p = np.random.default_rng(seed).permutation(n)
    k = n // 3
    return p[:k], p[k:2 * k], p[2 * k:]


def xor_attacker_control(n=6000, seed=0):
    X, y1, y2, s, a, b = xor_data(n, seed)
    f, v, t = split(n, seed)
    out = {}
    for name, cols in (("A_alone", [2]), ("B_alone", [3]), ("pair", [2, 3])):
        Z = X[:, cols]
        best = {}
        for nm, fac in A.slate("inner"):
            m = fac(0).fit(Z[f], s[f])
            best[nm] = A.macro_auc(s[t], A.proba(m, Z[t], 2), [0, 1])
        out[name] = best
    return out


def _engine_data(X, y1, y2, s, idx, seed):
    tX = torch.from_numpy(X[idx])
    return T.Data(tX, {0: torch.from_numpy(y1[idx]), 1: torch.from_numpy(y2[idx])}, torch.from_numpy(s[idx]),
                  np.bincount(s[idx], minlength=2) / len(idx),
                  np.random.default_rng(1).choice(len(idx), min(2048, len(idx)), replace=False), seed)


def run_arms(X, y1, y2, s, arms, beta, seed=0, warm=10, prot=20, masks=None):
    """Train arms on the fixture (fit role), audit coalition/local recovery on held-out thirds with the inner slate."""
    n = len(X)
    f, v, t = split(n, seed + 100)
    trn = np.concatenate([f])
    data = _engine_data(X, y1, y2, s, trn, seed)
    old = (T.HP["warm_epochs"], T.HP["prot_epochs"])
    T.HP["warm_epochs"], T.HP["prot_epochs"] = warm, prot
    try:
        m0 = T.warm_start(X.shape[1], [2, 2], data, seed, masks=masks)
        st = {k: x.clone() for k, x in m0.state_dict().items()}
        B = T.guard_losses(m0, data, [None, None])
        budgets = {i: B[i] + 0.01 for i in B}
        res = {}
        for arm in arms:
            model, diag = T.train_arm(arm, beta, st, X.shape[1], [2, 2], data, seed, budgets, masks=masks)
            # erasure maps used at the end of training (float32 graph maps are enough for the fixture audit)
            spec = T.arm_spec(arm)
            maps = [None, None]
            if spec["erasure"]:
                T.fit_maps(model, data, [0, 1], maps)
            with torch.no_grad():
                Xa = torch.from_numpy(X)
                V = {w: x.numpy() for w, x in T.views(model, Xa, maps, ["v1", "v2", "pair"], frozen=(0, 1)).items()}
                acc = {}
                for i, y in ((0, y1), (1, y2)):
                    _, lg = T.release(model, Xa, maps, i)
                    acc[i] = float((lg.numpy().argmax(1)[t] == y[t]).mean())
            rec = {}
            for w in ("v1", "v2", "pair"):
                vals = []
                for nm, fac in A.slate("inner"):
                    m = fac(0).fit(V[w][v], s[v])
                    vals.append(A.macro_auc(s[t], A.proba(m, V[w][t], 2), [0, 1]))
                rec[w] = max(vals)
            res[arm] = {"recovery_max_inner": rec, "task_acc": acc, "rejected": diag["protection_steps_rejected"],
                        "attempted": diag["protection_steps_attempted"]}
        return res
    finally:
        T.HP["warm_epochs"], T.HP["prot_epochs"] = old


XOR_MASKS = [[0, 2, 4, 5, 6, 7], [1, 3, 4, 5, 6, 7]]   # recipient 1 sees (Y1, A, noise); recipient 2 (Y2, B, noise)


def constant_release_control(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, n)
    y = rng.integers(0, 2, n)
    Z = np.zeros((n, 3))
    f, v, t = split(n, seed)
    m = A._lr(1.0, 0).fit(Z[f], s[f])
    auc = A.macro_auc(s[t], A.proba(m, Z[t], 2), [0, 1])
    maj = int(np.argmax(np.bincount(y[f])))
    acc_const_release = float((np.full(len(t), maj) == y[t]).mean())
    return {"recovery": auc, "useful_gain_over_constant": acc_const_release - float((y[t] == maj).mean())}
