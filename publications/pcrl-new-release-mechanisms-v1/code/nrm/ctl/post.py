"""Generic post-processing controls for any family's task-only twin (challenger; see nrm/ctl/__init__.py).

Every path maps a twin probability vector P (n, K) to a released vector of the same shape; decisions are the argmax of
the released vector. Strength knobs are PREDECLARED module constants; none is tuned on MODEL_SELECT. Fitting uses only
MODEL_FIT and/or CALIBRATION arrays supplied by the caller (role names are checked, other roles are refused).

Paths
  temperature(P, T)          q ~ P^(1/T), T in TEMPS. Keeps every argmax. Invertible for a fixed T, so it cannot lower
                             the best achievable recovery: any recovery drop it shows is reader sensitivity to scale or
                             shape, which is what this control detects.
  smooth(P, eps)             q = (1 - eps) P + eps / K, eps in EPS. Keeps every argmax; also invertible (affine).
  Coarsener(mode='blind')    SEX-blind coarsening. Label-free initial cells on the twin vectors (quantile bins of p1 for
                             K = 2; KL k-means for K > 2; K0 cells), merged greedily by the smallest loss of plug-in task
                             information (agglomerative information bottleneck), snapshotted at the predeclared LEVELS.
                             The released vector of a cell is its posterior: Dirichlet-smoothed task-label frequencies on
                             the estimation sample (posterior='labels': MODEL_FIT out-of-fold twin vectors with MODEL_FIT
                             task labels, or CALIBRATION vectors with CALIBRATION labels) or the cell's mean twin vector
                             (posterior='mean', fully label-free).
  Coarsener(mode='aware')    SEX-aware grouping. Merge score = task-information loss - lam * SEX information removed
                             (lam in LAMBDAS); SEX counts from MODEL_FIT only. The post-hoc SCM/IBSI-type control.
  Coarsener(mode='shuffled') the aware rule with SEX permuted (seeded): separates the SEX signal from the merge pattern.
  Coarsener(mode='random')   random pair merges (seeded) to the same levels: the random-merge (shuffle) floor.
  replace_random(...)        random channel without task adaptation: with probability q a person's vector is replaced by
                             the vector of a random person from a SEX-blind bank (e.g. the twin's MODEL_FIT vectors);
                             same_class=True draws only among bank vectors with the same argmax (every decision kept).
                             Returns an (R, n, K) stack of independent one-draw-per-person repetitions.
  Coarsener.refit_posterior  keeps the fitted cells and merges (MODEL_FIT) and re-estimates the released posteriors on
                             another permitted sample, normally CALIBRATION (calibration parameters only).
  JointCoarsener             the same paths on ONE joint partition of [income | occupation] (for candidates that couple
                             both outputs); each joint cell sends its income posterior and its occupation posterior to
                             the respective recipients; levels JOINT_LEVELS.
  flatten_remainder(P, R, s) decision- and top-probability-preserving flattening: the non-argmax coordinates move toward
                             SEX-blind class shares R[argmax] (fit_remainder_shares, on MODEL_FIT out-of-fold vectors
                             with labels, CALIBRATION, or label-free) with weight min(s, cap), s in REM_S; deterministic
                             and many-to-one in the low-probability log-odds; identity for K = 2.

Selftest (invented arrays only): python3 -m nrm.ctl.post selftest
"""
from __future__ import annotations

import json
import sys

import numpy as np

FIT_ROLES = ('MODEL_FIT', 'CALIBRATION')
TEMPS = (1.25, 1.5, 2.0, 3.0)
EPS = (0.05, 0.10, 0.20, 0.30)
K0 = 256
LEVELS = {2: (64, 32, 16, 8, 6, 4, 2), 5: (128, 64, 32, 16, 8, 5)}
JOINT_LEVELS = (128, 64, 32, 16, 10)
LAMBDAS = (0.5, 1.0, 2.0)
REPLACE_Q = (0.05, 0.10, 0.20, 0.40)
ALPHA = 0.5          # Dirichlet pseudo-count per class for label posteriors (Jeffreys)
FLOOR = 1e-15        # keeps label-free mean posteriors strictly positive
TINY = 1e-300
MODES = ('blind', 'aware', 'shuffled', 'random')


def check_prob(P, name='P'):
    P = np.asarray(P, dtype=np.float64)
    if (P.ndim != 2 or P.shape[1] < 2 or not np.isfinite(P).all() or (P < 0).any()
            or not np.allclose(P.sum(1), 1.0, atol=1e-9, rtol=0)):
        raise ValueError(f'{name}: not an (n, K) probability array')
    return P


def _norm(Q):
    Q = np.asarray(Q, dtype=np.float64)
    return Q / Q.sum(-1, keepdims=True)


def _role_ok(role, what):
    if role not in FIT_ROLES:
        raise PermissionError(f'{what}: role {role!r} refused; controls fit on MODEL_FIT / CALIBRATION only')


def temperature(P, T):
    """Flatten (T > 1) or sharpen (T < 1); argmax unchanged."""
    P = check_prob(P)
    if not T > 0:
        raise ValueError('T must be > 0')
    L = np.log(np.maximum(P, TINY)) / T
    L -= L.max(1, keepdims=True)
    return _norm(np.exp(L))


def smooth(P, eps):
    """Label smoothing toward uniform; argmax unchanged."""
    P = check_prob(P)
    if not 0.0 <= eps < 1.0:
        raise ValueError('eps must be in [0, 1)')
    return _norm((1.0 - eps) * P + eps / P.shape[1])


def _kl_rows(P, C):
    """KL(P_i || C_j) for every i, j: (n, K), (m, K) -> (n, m)."""
    lp = np.log(np.maximum(P, TINY))
    lc = np.log(np.maximum(C, TINY))
    return (P * lp).sum(1, keepdims=True) - P @ lc.T


def _kl_pair(A, M):
    """sum_k A log(A / M) along the last axis, with 0 log 0 = 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        pos = A > 0
        v = np.where(pos, A * (np.log(np.where(pos, A, 1.0)) - np.log(np.where(pos, M, 1.0))), 0.0)
    return v.sum(-1)


def _check_blocks(Z, blocks, name='Z'):
    """Concatenated probability blocks, e.g. [income (2) | occupation (5)]; each block must be a probability vector."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[1] != sum(blocks):
        raise ValueError(f'{name}: expected {sum(blocks)} columns in blocks {blocks}')
    off = 0
    for b in blocks:
        check_prob(Z[:, off:off + b], name)
        off += b
    return Z


def _blocknorm(Q, blocks):
    out, off = [], 0
    for b in blocks:
        out.append(_norm(Q[:, off:off + b]))
        off += b
    return np.hstack(out)


def _kl_kmeans(P, k, seed, iters, blocks=None):
    rng = np.random.default_rng(seed)
    n, K = P.shape
    k = min(k, n)
    idx = [int(rng.integers(n))]
    d = np.maximum(_kl_rows(P, P[idx])[:, 0], 0.0)
    for _ in range(1, k):
        tot = d.sum()
        i = int(rng.choice(n, p=d / tot)) if tot > 0 else int(rng.integers(n))
        idx.append(i)
        d = np.minimum(d, np.maximum(_kl_rows(P, P[[i]])[:, 0], 0.0))
    C = P[idx].copy()
    for _ in range(iters):
        a = np.argmin(_kl_rows(P, C), 1)
        cnt = np.bincount(a, minlength=len(C)).astype(float)
        S = np.stack([np.bincount(a, weights=P[:, j], minlength=len(C)) for j in range(K)], 1)
        new = np.where(cnt[:, None] > 0, S / np.maximum(cnt, 1.0)[:, None], C)
        if np.allclose(new, C, atol=1e-12, rtol=0):
            break
        C = new
    a = np.argmin(_kl_rows(P, C), 1)
    return _blocknorm(C[np.unique(a)], blocks or (K,))


class Cells:
    """Label-free initial partition of twin vectors, fitted on the caller's partition sample. blocks=(2, 5) partitions
    the concatenated [income | occupation] vectors jointly (KL distance = income KL + occupation KL)."""

    def __init__(self, P, k0=K0, seed=0, iters=30, blocks=None):
        self.blocks = tuple(blocks) if blocks is not None else (np.asarray(P).shape[1],)
        P = _check_blocks(P, self.blocks, 'P')
        self.K = P.shape[1]
        if self.blocks == (2,):
            e = np.unique(np.quantile(P[:, 1], np.linspace(0.0, 1.0, k0 + 1)[1:-1]))
            self.edges = e
            a = np.searchsorted(e, P[:, 1], side='right')
            k = len(e) + 1
            cnt = np.bincount(a, minlength=k)
            s = np.bincount(a, weights=P[:, 1], minlength=k)
            lo, hi = np.concatenate(([0.0], e)), np.concatenate((e, [1.0]))
            p1 = np.where(cnt > 0, s / np.maximum(cnt, 1), (lo + hi) / 2.0)
            self.C = np.stack([1.0 - p1, p1], 1)
        else:
            self.edges = None
            self.C = _kl_kmeans(P, k0, seed, iters, self.blocks)
        self.k = len(self.C)

    def assign(self, P):
        P = _check_blocks(P, self.blocks, 'P')
        if self.blocks == (2,):
            return np.searchsorted(self.edges, P[:, 1], side='right')
        return np.argmin(_kl_rows(P, self.C), 1)


def _pair_costs(n, T, i, idx):
    """Loss of plug-in information (total nats) when cell i merges with cell j in idx: n_i KL(T_i||M) + n_j KL(T_j||M),
    M the count-weighted mixture. All cells involved must have positive mass."""
    tot = n[i] + n[idx]
    M = (n[i] * T[i][None, :] + n[idx, None] * T[idx]) / tot[:, None]
    v = n[i] * _kl_pair(T[i][None, :], M) + n[idx] * _kl_pair(T[idx], M)
    if not np.isfinite(v).all():
        raise FloatingPointError('non-finite merge cost')
    return v


def merge_path(n, T, levels, mode='blind', S=None, lam=0.0, seed=0):
    """Greedy merging of k weighted cells. Returns {level: (map k -> 0..level-1, n (level,), T (level, K))}."""
    if mode not in MODES:
        raise ValueError(mode)
    n = np.asarray(n, dtype=np.float64).copy()
    T = np.asarray(T, dtype=np.float64).copy()
    S = None if S is None else np.asarray(S, dtype=np.float64).copy()
    if mode in ('aware', 'shuffled') and S is None:
        raise ValueError('aware/shuffled merging needs SEX distributions')
    k = len(n)
    if not (n > 0).all():
        raise ValueError('every initial cell needs positive mass (merge empty cells first)')
    alive = np.ones(k, bool)
    root = np.arange(k)
    rng = np.random.default_rng(seed)
    want = sorted({int(L) for L in levels if 1 <= int(L) <= k}, reverse=True)

    def row(i):
        c = np.full(k, np.inf)
        idx = np.flatnonzero(alive)
        idx = idx[idx != i]
        if len(idx):
            v = _pair_costs(n, T, i, idx)
            if mode in ('aware', 'shuffled'):
                v = v - lam * _pair_costs(n, S, i, idx)
            c[idx] = v
        return c

    Mx = None if mode == 'random' else np.vstack([row(i) for i in range(k)])
    out = {}

    def snap(L):
        ids, inv = np.unique(root, return_inverse=True)
        out[L] = (inv.astype(np.int64), n[ids].copy(), T[ids].copy())

    live = k
    if live in want:
        snap(live)
    while want and live > want[-1]:
        if mode == 'random':
            a, b = sorted(int(x) for x in rng.choice(np.flatnonzero(alive), 2, replace=False))
        else:
            a, b = sorted(divmod(int(np.argmin(Mx)), k))
        w = n[a] + n[b]
        T[a] = (n[a] * T[a] + n[b] * T[b]) / w
        if S is not None:
            S[a] = (n[a] * S[a] + n[b] * S[b]) / w
        n[a], n[b] = w, 0.0
        alive[b] = False
        root[root == b] = a
        live -= 1
        if Mx is not None:
            Mx[b, :] = np.inf
            Mx[:, b] = np.inf
            r = row(a)
            Mx[a, :] = r
            Mx[:, a] = r
        if live in want:
            snap(live)
    return out


def plugin_mi(groups, labels):
    """Plug-in mutual information (nats) between integer groups and integer labels (diagnostics only)."""
    g = np.unique(np.asarray(groups), return_inverse=True)[1]
    y = np.unique(np.asarray(labels), return_inverse=True)[1]
    J = np.zeros((g.max() + 1, y.max() + 1))
    np.add.at(J, (g, y), 1.0)
    J /= J.sum()
    pg, py = J.sum(1, keepdims=True), J.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.where(J > 0, J * np.log(J / (pg * py)), 0.0).sum())


class Coarsener:
    """One task's coarsening path: fit once on MODEL_FIT / CALIBRATION arrays, apply at any predeclared level."""

    def __init__(self, mode='blind', lam=0.0, posterior='labels', k0=K0, levels=None, alpha=ALPHA, seed=0):
        if mode not in MODES:
            raise ValueError(mode)
        if mode in ('aware', 'shuffled') and not lam > 0:
            raise ValueError('aware/shuffled paths need lam > 0')
        if posterior not in ('labels', 'mean'):
            raise ValueError(posterior)
        self.mode, self.lam, self.posterior = mode, float(lam), posterior
        self.k0, self.levels, self.alpha, self.seed = int(k0), levels, float(alpha), int(seed)

    def fit(self, P_part, part_role, P_est, est_role, y_est=None, sex_est=None):
        """P_part: twin vectors for the label-free partition. P_est (+ y_est, sex_est): estimation sample, normally the
        twin's MODEL_FIT out-of-fold vectors with MODEL_FIT labels, or CALIBRATION vectors with CALIBRATION labels."""
        _role_ok(part_role, 'partition')
        _role_ok(est_role, 'estimation')
        Z_part, B = self._stack(P_part, 'P_part')
        Z_est, B_est = self._stack(P_est, 'P_est')
        if B != B_est:
            raise ValueError('class count mismatch')
        n = len(Z_est)
        s = None
        if self.mode in ('aware', 'shuffled'):
            if est_role != 'MODEL_FIT' or sex_est is None:
                raise PermissionError('SEX-aware paths need MODEL_FIT SEX on the estimation sample')
            s = np.asarray(sex_est).astype(np.int64)
            if len(s) != n or not set(np.unique(s)) <= {0, 1}:
                raise ValueError('sex_est must be 0/1 of length n')
            if self.mode == 'shuffled':
                s = np.random.default_rng(self.seed + 7919).permutation(s)
        elif sex_est is not None:
            raise PermissionError('SEX-blind paths take no SEX input')
        ys = None
        if self.posterior == 'labels':
            if y_est is None:
                raise ValueError("posterior='labels' needs task labels")
            ys = self._labels(y_est, n, B)
        cells = Cells(Z_part, self.k0, self.seed, blocks=B)
        c = cells.assign(Z_est)
        cnt = np.bincount(c, minlength=cells.k).astype(float)
        Y = self._counts(c, cells.k, Z_est, ys, B)
        ne = np.flatnonzero(cnt > 0)
        init = np.argmin(_kl_rows(cells.C, cells.C[ne]), 1)      # empty cells join the nearest non-empty cell
        init[ne] = np.arange(len(ne))
        n0 = cnt[ne]
        T0 = Y[ne] / n0[:, None]
        S0 = None
        if s is not None:
            Sx = np.zeros((cells.k, 2))
            np.add.at(Sx, (c, s), 1.0)
            S0 = Sx[ne] / n0[:, None]
        if self.levels is not None:
            levels = self.levels
        elif len(B) == 1:
            levels = LEVELS.get(B[0], (64, 32, 16, 8, B[0]))
        else:
            levels = JOINT_LEVELS
        snaps = merge_path(n0, T0, levels, self.mode, S0, self.lam, self.seed)
        self.cells, self.blocks, self.K, self.tables, self.task_mi = cells, B, sum(B), {}, {}
        tot = n0.sum()
        for L, (m, nL, TL) in snaps.items():
            raw = self._post(nL[:, None] * TL, nL, B) if self.posterior == 'labels' else np.maximum(TL, FLOOR)
            self.tables[L] = (m[init], _blocknorm(raw, B))
            overall = (nL[:, None] * TL).sum(0) / tot
            self.task_mi[L] = float((nL / tot * _kl_pair(TL, overall[None, :])).sum())   # summed over blocks
        self.roles = {'partition': part_role, 'estimation': est_role}
        self.n_part, self.n_est = len(Z_part), n
        return self

    # ---- block helpers (one block for a single task; JointCoarsener uses (2, 5))
    def _stack(self, P, name='P'):
        P = check_prob(P, name)
        return P, (P.shape[1],)

    def _ylist(self, y):
        return [y]

    def _split(self, Q):
        return Q

    def _labels(self, y, n, blocks):
        ys = [np.asarray(v).astype(np.int64) for v in self._ylist(y)]
        if len(ys) != len(blocks) or any(len(v) != n or v.min() < 0 or v.max() >= b for v, b in zip(ys, blocks)):
            raise ValueError('task labels out of range')
        return ys

    def _counts(self, g, k, Z, ys, blocks):
        """Per-cell label counts (posterior='labels') or summed twin vectors (posterior='mean'), all blocks."""
        if self.posterior == 'labels':
            Y, off = np.zeros((k, sum(blocks))), 0
            for b, v in zip(blocks, ys):
                np.add.at(Y, (g, off + v), 1.0)
                off += b
            return Y
        return np.stack([np.bincount(g, weights=Z[:, j], minlength=k) for j in range(Z.shape[1])], 1)

    def _post(self, Y, cnt, blocks):
        """Unnormalized released posteriors from per-cell totals Y and cell sizes cnt."""
        if self.posterior == 'labels':
            parts, off = [], 0
            for b in blocks:
                parts.append((Y[:, off:off + b] + self.alpha) / (cnt[:, None] + b * self.alpha))
                off += b
            return np.hstack(parts)
        return np.maximum(Y / np.maximum(cnt, 1.0)[:, None], FLOOR)

    def refit_posterior(self, P, role, y=None):
        """Keep every fitted partition and merge map; re-estimate the released posteriors on another permitted sample
        (e.g. CALIBRATION vectors + labels, or label-free means). Cells absent from the sample keep their posterior."""
        _role_ok(role, 'posterior')
        Z, B = self._stack(P, 'P')
        if B != self.blocks:
            raise ValueError('class count mismatch')
        ys = None
        if self.posterior == 'labels':
            if y is None:
                raise ValueError("posterior='labels' needs task labels")
            ys = self._labels(y, len(Z), B)
        self.kept_cells = {}
        for L, (cmap, post) in list(self.tables.items()):
            g = cmap[self.cells.assign(Z)]
            cnt = np.bincount(g, minlength=L).astype(float)
            new = np.where(cnt[:, None] > 0, self._post(self._counts(g, L, Z, ys, B), cnt, B), post)
            self.tables[L] = (cmap, _blocknorm(new, B))
            self.kept_cells[L] = int((cnt == 0).sum())
        self.roles['posterior'] = role
        return self

    def cell_ids(self, P, level):
        cmap, _ = self.tables[level]
        Z, _ = self._stack(P, 'P')
        return cmap[self.cells.assign(Z)]

    def apply(self, P, level):
        _, post = self.tables[level]
        return self._split(post[self.cell_ids(P, level)].copy())

    def record(self):
        return {'control': f'coarsen_{self.mode}' + ('_joint' if len(self.blocks) > 1 else ''), 'blocks': list(self.blocks),
                'posterior': self.posterior, 'lam': self.lam, 'k0_requested': self.k0, 'k0_actual': self.cells.k,
                'levels': sorted(self.tables, reverse=True), 'alpha': self.alpha, 'seed': self.seed, 'roles': self.roles,
                'n_partition': self.n_part, 'n_estimation': self.n_est,
                'plugin_task_mi_nats': {str(L): v for L, v in sorted(self.task_mi.items(), reverse=True)}}


class JointCoarsener(Coarsener):
    """Joint income x occupation coarsening for candidates that couple both outputs. One partition of the concatenated
    twin vectors [income (2) | occupation (5)] (KL distance and merge cost = income + occupation terms; the SEX-aware
    score as in Coarsener). Each joint cell sends its income posterior to the income recipient and its occupation
    posterior to the occupation recipient. Inputs and labels are pairs (income, occupation); apply returns
    (income (n, 2), occupation (n, 5)). Default levels JOINT_LEVELS (10 = 2 x 5 decision pairs)."""

    def _stack(self, P, name='P'):
        if not isinstance(P, (tuple, list)) or len(P) != 2:
            raise ValueError(f'{name}: joint input is (income (n, 2), occupation (n, K))')
        Pi, Po = check_prob(P[0], f'{name}[income]'), check_prob(P[1], f'{name}[occupation]')
        if Pi.shape[1] != 2 or len(Pi) != len(Po):
            raise ValueError(f'{name}: income must be (n, 2) and both blocks must have n rows')
        return np.hstack([Pi, Po]), (2, Po.shape[1])

    def _ylist(self, y):
        if not isinstance(y, (tuple, list)) or len(y) != 2:
            raise ValueError('joint labels are (income, occupation)')
        return list(y)

    def _split(self, Q):
        return Q[:, :2].copy(), Q[:, 2:].copy()


def replace_random(P, bank, q, R, seed, bank_role='MODEL_FIT', same_class=False):
    """(R, n, K) independent draws: each vector replaced with probability q by a random bank vector (SEX-blind).
    same_class=True draws the replacement among bank vectors with the same argmax, so every decision is kept."""
    _role_ok(bank_role, 'bank')
    P, bank = check_prob(P), check_prob(bank, 'bank')
    if P.shape[1] != bank.shape[1] or not 0.0 <= q <= 1.0 or R < 1:
        raise ValueError('bad replacement arguments')
    rng = np.random.default_rng(seed)
    cls, bcls = P.argmax(1), bank.argmax(1)
    pools = {int(c): np.flatnonzero(bcls == c) for c in np.unique(cls)} if same_class else None
    if same_class and any(len(v) == 0 for v in pools.values()):
        raise ValueError('the bank lacks a predicted class present in P')
    out = np.repeat(P[None], R, axis=0)
    for r in range(R):
        m = rng.random(len(P)) < q
        if not same_class:
            out[r, m] = bank[rng.integers(0, len(bank), int(m.sum()))]
            continue
        idx = np.flatnonzero(m)
        for c, pool in pools.items():
            sel = idx[cls[idx] == c]
            out[r, sel] = bank[pool[rng.integers(0, len(pool), len(sel))]]
    return out


REM_S = (0.25, 0.5, 0.75, 1.0)


def fit_remainder_shares(P, role, y=None, alpha=ALPHA):
    """SEX-blind shares of the non-argmax remainder per predicted class: R (K, K), R[c, c] = 0, rows sum to 1.
    With task labels y (MODEL_FIT out-of-fold twin vectors, or CALIBRATION): Jeffreys-smoothed frequencies of the true
    class among people predicted c whose true class is not c. Without labels: the mean remainder shape of the twin's
    own vectors predicted c (label-free)."""
    _role_ok(role, 'remainder shares')
    P = check_prob(P)
    n, K = P.shape
    j = P.argmax(1)
    if y is not None:
        y = np.asarray(y).astype(np.int64)
        if len(y) != n or y.min() < 0 or y.max() >= K:
            raise ValueError('y out of range')
    R = np.zeros((K, K))
    for c in range(K):
        m = j == c
        if y is not None:
            r = np.bincount(y[m & (y != c)], minlength=K).astype(float) + alpha
        else:
            rem = P[m].copy()
            rem[:, c] = 0.0
            tot = rem.sum(1, keepdims=True)
            r = (rem / np.maximum(tot, TINY)).mean(0) if m.any() else np.ones(K)
        r[c] = 0.0
        R[c] = r / r.sum() if r.sum() > 0 else np.where(np.arange(K) == c, 0.0, 1.0 / (K - 1))
    return R


def flatten_remainder(P, R, s):
    """Keep every argmax and its probability; move the other coordinates toward (1 - p_top) R[argmax] with weight
    min(s, w_max), where w_max keeps every other coordinate below the top one (decision kept). Deterministic, SEX-blind
    given R, many-to-one at s = 1 where the cap does not bind (the remainder shape is replaced). Identity for s = 0 and
    for K = 2 (the remainder has one coordinate)."""
    P = check_prob(P)
    n, K = P.shape
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (K, K) or (R < 0).any() or not np.allclose(R.sum(1), 1.0) or not np.allclose(np.diag(R), 0.0):
        raise ValueError('R must be (K, K) remainder shares with a zero diagonal')
    if not 0.0 <= s <= 1.0:
        raise ValueError('s must be in [0, 1]')
    if K == 2 or s == 0.0:
        return P.copy()
    rows = np.arange(n)
    j = P.argmax(1)
    pj = P[rows, j]
    D = (1.0 - pj)[:, None] * R[j] - P
    D[rows, j] = 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        lim = np.where(D > 0, (pj[:, None] * (1.0 - 1e-12) - P) / D, np.inf)
    w = np.minimum(s, np.clip(lim.min(1), 0.0, 1.0))
    return _norm(P + w[:, None] * D)


# ----------------------------------------------------------------------------------------------- selftest (invented)

def _invented(n, seed):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, n)
    z = rng.normal(size=(n, 3))
    pi = 1.0 / (1.0 + np.exp(-(1.2 * z[:, 0] + 0.6 * s - 0.3)))
    W = np.random.default_rng(99).normal(size=(3, 5))
    lo = z @ W + np.outer(s, [0.0, 0.9, -0.9, 0.0, 0.5])
    po = np.exp(lo - lo.max(1, keepdims=True))
    po /= po.sum(1, keepdims=True)
    y_inc = (rng.random(n) < pi).astype(np.int64)
    y_occ = (rng.random(n)[:, None] > np.cumsum(po, 1)).sum(1).clip(0, 4)
    return s, np.stack([1.0 - pi, pi], 1), po, y_inc, y_occ


def _ok(name, cond, results):
    results.append((name, bool(cond)))
    print(('PASS ' if cond else 'FAIL ') + name)


def selftest():
    res = []
    s, Pi, Po, yi, yo = _invented(6000, 1)
    s2, Pi2, Po2, _, _ = _invented(3000, 2)                       # invented 'deployment' rows
    ent = lambda Q: float(-(Q * np.log(np.maximum(Q, TINY))).sum(1).mean())
    _ok('temperature T=1 is the identity', np.allclose(temperature(Po, 1.0), Po, atol=1e-12), res)
    _ok('smooth eps=0 is the identity', np.allclose(smooth(Po, 0.0), Po, atol=1e-15), res)
    for T in TEMPS:
        Q = temperature(Po, T)
        _ok(f'temperature T={T}: valid, argmax kept, flatter',
            np.allclose(Q.sum(1), 1, atol=1e-12) and (Q.argmax(1) == Po.argmax(1)).all() and ent(Q) > ent(Po), res)
    for e in EPS:
        Q = smooth(Pi, e)
        _ok(f'smooth eps={e}: valid, argmax kept, floor eps/K',
            np.allclose(Q.sum(1), 1, atol=1e-12) and (Q.argmax(1) == Pi.argmax(1)).all() and Q.min() >= e / 2 - 1e-15,
            res)
    try:
        Coarsener().fit(Po, 'MODEL_SELECT', Po, 'MODEL_FIT', y_est=yo)
        _ok('MODEL_SELECT partition refused', False, res)
    except PermissionError:
        _ok('MODEL_SELECT partition refused', True, res)
    try:
        Coarsener('aware', lam=1.0).fit(Po, 'MODEL_FIT', Po, 'CALIBRATION', y_est=yo, sex_est=s)
        _ok('SEX-aware path refuses a non-MODEL_FIT estimation role', False, res)
    except PermissionError:
        _ok('SEX-aware path refuses a non-MODEL_FIT estimation role', True, res)
    try:
        Coarsener('blind').fit(Po, 'MODEL_FIT', Po, 'MODEL_FIT', y_est=yo, sex_est=s)
        _ok('SEX-blind path refuses a SEX input', False, res)
    except PermissionError:
        _ok('SEX-blind path refuses a SEX input', True, res)
    lv = (32, 16, 8, 5)
    fits = {m: Coarsener(m, lam=(1.0 if m in ('aware', 'shuffled') else 0.0), k0=64, levels=lv, seed=3)
            .fit(Po, 'MODEL_FIT', Po, 'MODEL_FIT', y_est=yo, sex_est=(s if m in ('aware', 'shuffled') else None))
            for m in MODES}
    b = fits['blind']
    for L in lv:
        Q = b.apply(Po2, L)
        _ok(f'blind occupation L={L}: valid vectors, at most L distinct',
            np.allclose(Q.sum(1), 1, atol=1e-12) and (Q > 0).all() and len(np.unique(Q, axis=0)) <= L, res)
    mi = [b.task_mi[L] for L in lv]
    _ok('blind task information is non-increasing along the path', all(x >= y - 1e-12 for x, y in zip(mi, mi[1:])), res)
    again = Coarsener('blind', k0=64, levels=lv, seed=3).fit(Po, 'MODEL_FIT', Po, 'MODEL_FIT', y_est=yo)
    _ok('refit is bit-identical', all(np.array_equal(again.tables[L][1], b.tables[L][1]) for L in lv), res)
    smi = {m: plugin_mi(f.cell_ids(Po, 8), s) for m, f in fits.items()}
    tmi = {m: f.task_mi[8] for m, f in fits.items()}
    print('  L=8 plug-in SEX MI (nats):', json.dumps({m: round(v, 5) for m, v in smi.items()}))
    print('  L=8 plug-in task MI (nats):', json.dumps({m: round(v, 5) for m, v in tmi.items()}))
    _ok('aware removes more SEX information than blind and shuffled at L=8',
        smi['aware'] < smi['blind'] and smi['aware'] < smi['shuffled'], res)
    _ok('blind keeps more task information than random merging at L=8', tmi['blind'] > tmi['random'], res)
    ci = Coarsener('blind', k0=64, levels=(16, 6, 2), seed=3).fit(Pi, 'CALIBRATION', Pi, 'CALIBRATION', y_est=yi)
    for L in (16, 6, 2):
        Q = ci.apply(Pi2, L)
        _ok(f'blind income L={L}: valid, at most L distinct', np.allclose(Q.sum(1), 1, atol=1e-12)
            and len(np.unique(Q, axis=0)) <= L, res)
    aw = Coarsener('aware', lam=1.0, k0=64, levels=lv, seed=3).fit(Po, 'MODEL_FIT', Po, 'MODEL_FIT', y_est=yo, sex_est=s)
    ids_before = {L: aw.cell_ids(Po2, L) for L in lv}
    _, _, _, yi3, yo3 = _invented(3000, 2)
    aw.refit_posterior(Po2, 'CALIBRATION', y=yo3)
    Q = aw.apply(Po2, 8)
    _ok('posterior re-estimated on CALIBRATION: same cells, valid vectors, roles recorded',
        all(np.array_equal(ids_before[L], aw.cell_ids(Po2, L)) for L in lv) and np.allclose(Q.sum(1), 1, atol=1e-12)
        and aw.record()['roles']['posterior'] == 'CALIBRATION', res)
    try:
        aw.refit_posterior(Po2, 'MODEL_SELECT', y=yo3)
        _ok('MODEL_SELECT posterior refused', False, res)
    except PermissionError:
        _ok('MODEL_SELECT posterior refused', True, res)
    cm = Coarsener('blind', posterior='mean', k0=64, levels=(16, 8), seed=3).fit(Po, 'MODEL_FIT', Po2, 'MODEL_FIT')
    Q = cm.apply(Po2, 8)
    _ok('label-free mean posterior: valid, strictly positive', np.allclose(Q.sum(1), 1, atol=1e-12) and (Q > 0).all(),
        res)
    D = replace_random(Po2, Po, 0.2, 4, seed=11)
    frac = (D != Po2[None]).any(2).mean()
    _ok('random replacement: shape, validity, rate ~ q, draws differ',
        D.shape == (4,) + Po2.shape and np.allclose(D.sum(2), 1, atol=1e-12) and abs(frac - 0.2) < 0.03
        and not np.array_equal(D[0], D[1]), res)
    Dc = replace_random(Po2, Po, 0.4, 3, seed=12, same_class=True)
    _ok('same-class random replacement keeps every decision and still replaces',
        (Dc.argmax(2) == Po2.argmax(1)[None]).all() and (Dc != Po2[None]).any(2).mean() > 0.3, res)
    jl = (32, 16, 10)
    jb = JointCoarsener('blind', k0=64, levels=jl, seed=3).fit((Pi, Po), 'MODEL_FIT', (Pi, Po), 'MODEL_FIT', y_est=(yi, yo))
    for L in jl:
        Qi, Qo = jb.apply((Pi2, Po2), L)
        _ok(f'joint blind L={L}: valid income and occupation vectors, at most L distinct joint values',
            Qi.shape == Pi2.shape and Qo.shape == Po2.shape and np.allclose(Qi.sum(1), 1, atol=1e-12)
            and np.allclose(Qo.sum(1), 1, atol=1e-12) and len(np.unique(np.hstack([Qi, Qo]), axis=0)) <= L, res)
    ja = JointCoarsener('aware', lam=1.0, k0=64, levels=jl, seed=3).fit((Pi, Po), 'MODEL_FIT', (Pi, Po), 'MODEL_FIT',
                                                                        y_est=(yi, yo), sex_est=s)
    smi_j = {'blind': plugin_mi(jb.cell_ids((Pi, Po), 10), s), 'aware': plugin_mi(ja.cell_ids((Pi, Po), 10), s)}
    print('  joint L=10 plug-in SEX MI (nats):', json.dumps({m: round(v, 5) for m, v in smi_j.items()}))
    _ok('joint aware removes more SEX information than joint blind at L=10', smi_j['aware'] < smi_j['blind'], res)
    ids = jb.cell_ids((Pi2, Po2), 16)
    jb.refit_posterior((Pi2, Po2), 'CALIBRATION', y=(yi3, yo3))
    _ok('joint posterior refit on CALIBRATION keeps the joint cells', np.array_equal(ids, jb.cell_ids((Pi2, Po2), 16)), res)
    try:
        JointCoarsener('blind').fit((Pi, Po), 'MODEL_FIT', (Pi, Po), 'MODEL_FIT', y_est=(yi, yo), sex_est=s)
        _ok('joint SEX-blind path refuses a SEX input', False, res)
    except PermissionError:
        _ok('joint SEX-blind path refuses a SEX input', True, res)
    try:
        fit_remainder_shares(Po, 'MODEL_SELECT', yo)
        _ok('remainder shares refuse MODEL_SELECT', False, res)
    except PermissionError:
        _ok('remainder shares refuse MODEL_SELECT', True, res)
    top = Po2.argmax(1)
    ptop = Po2[np.arange(len(Po2)), top]
    for nm, R in (('labels', fit_remainder_shares(Po, 'MODEL_FIT', yo)), ('label-free', fit_remainder_shares(Po, 'CALIBRATION'))):
        good = R.shape == (5, 5) and np.allclose(R.sum(1), 1) and np.allclose(np.diag(R), 0)
        dev0 = None
        for sv in REM_S:
            Q = flatten_remainder(Po2, R, sv)
            rem = Q.copy()
            rem[np.arange(len(Q)), top] = 0.0
            dev = float(np.abs(rem / rem.sum(1, keepdims=True) - R[top]).sum(1).mean())
            if dev0 is None:
                r0 = Po2.copy()
                r0[np.arange(len(Q)), top] = 0.0
                dev0 = float(np.abs(r0 / r0.sum(1, keepdims=True) - R[top]).sum(1).mean())
            good = good and np.allclose(Q.sum(1), 1, atol=1e-12) and (Q.argmax(1) == top).all() \
                and np.allclose(Q[np.arange(len(Q)), top], ptop, atol=1e-12) and dev < dev0
        _ok(f'remainder flattening ({nm} shares): decision and top probability kept, remainder moves to the shares', good, res)
    _ok('remainder flattening is the identity at s=0 and for K=2',
        np.array_equal(flatten_remainder(Po2, fit_remainder_shares(Po, 'MODEL_FIT'), 0.0), Po2)
        and np.array_equal(flatten_remainder(Pi2, fit_remainder_shares(Pi, 'MODEL_FIT'), 1.0), Pi2), res)
    print('  record example:', json.dumps(b.record()))
    nfail = sum(not ok for _, ok in res)
    print(f'SELFTEST {"PASS" if nfail == 0 else "FAIL"}: {len(res) - nfail}/{len(res)} checks')
    return nfail == 0


if __name__ == '__main__':
    if sys.argv[1:] == ['selftest']:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
