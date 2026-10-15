"""Independent numerical core: AUC, bootstrap, influence functions, task losses, statuses, power, receipts.

Everything here is written from first principles (lock text definitions), not from campaign code.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import stats

W = Path(__file__).resolve().parents[1]
RES = W / 'results/pcrl_frozen_tree_control_completion_v1'
OUT = RES / 'independent'
IMPORTS_NOTE = ('numpy/scipy/sklearn/torch + tree_completion.evaluation_queue.queue_config (queue binding and '
                'verify() only); no tree_completion.*/task_headroom.*/task_conditioned_representation.* metric, '
                'inference, bootstrap or power code imported')


# ----------------------------------------------------------------------------- AUC
def auc(y, s):
    """Two-class AUC (Mann-Whitney with midranks): P(S_pos > S_neg) + 0.5 P(=)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    n1 = int(y.sum()); n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        raise ValueError('AUC needs both classes')
    r = stats.rankdata(s, method='average')
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (float(n1) * float(n0)))


def auc_pairs(y, s):
    """Brute-force pairwise AUC (test oracle; O(n1 n0))."""
    y = np.asarray(y).astype(bool); s = np.asarray(s, dtype=np.float64)
    a, b = s[y][:, None], s[~y][None, :]
    return float(((a > b) + 0.5 * (a == b)).mean())


class WeightedAUC:
    """AUC under row weights (bootstrap multiplicities), vectorized over draws, ties counted 1/2."""

    def __init__(self, y, s):
        y = np.asarray(y).astype(bool); s = np.asarray(s, dtype=np.float64)
        self.order = np.argsort(s, kind='mergesort')
        ss = s[self.order]
        self.starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
        self.pos = y[self.order]

    def __call__(self, Wrows):
        Ws = Wrows[:, self.order]
        Wp = np.where(self.pos[None, :], Ws, 0.0)
        Wn = Ws - Wp
        bp = np.add.reduceat(Wp, self.starts, axis=1)
        bn = np.add.reduceat(Wn, self.starts, axis=1)
        below = np.cumsum(bn, axis=1) - bn
        num = (bp * (below + 0.5 * bn)).sum(1)
        return num / (bp.sum(1) * bn.sum(1))


def auc_influence(y, s):
    """Per-row Hajek projection of the AUC U-statistic.

    positive i: (V_i - AUC)/pi1, V_i = P(S_neg < s_i) + 0.5 P(S_neg = s_i)
    negative j: (W_j - AUC)/pi0, W_j = P(S_pos > s_j) + 0.5 P(S_pos = s_j)
    so that AUC - theta ~= mean over rows of IF.
    """
    y = np.asarray(y).astype(bool); s = np.asarray(s, dtype=np.float64)
    n = len(y); n1 = int(y.sum()); n0 = n - n1
    neg = np.sort(s[~y]); pos = np.sort(s[y])
    sp, sn = s[y], s[~y]
    lo = np.searchsorted(neg, sp, side='left'); hi = np.searchsorted(neg, sp, side='right')
    V = (lo + 0.5 * (hi - lo)) / n0
    lo2 = np.searchsorted(pos, sn, side='left'); hi2 = np.searchsorted(pos, sn, side='right')
    Wj = ((n1 - hi2) + 0.5 * (hi2 - lo2)) / n1
    A = float(V.mean())
    out = np.empty(n)
    out[y] = (V - A) / (n1 / n)
    out[~y] = (Wj - A) / (n0 / n)
    return out, A


# ----------------------------------------------------------------------------- task losses
def task_rows(y, p, k):
    """Per-row accuracy (argmax, first maximum), NLL (prob floor 1e-12) and summed Brier."""
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (len(y), k) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('probability array malformed')
    if np.abs(p.sum(1) - 1.0).max() > 1e-9:
        raise ValueError('probability rows do not sum to one')
    idx = np.arange(len(y))
    acc = (p.argmax(1) == y).astype(np.float64)
    nll = -np.log(np.maximum(p[idx, y], 1e-12))
    t = np.zeros_like(p); t[idx, y] = 1.0
    brier = ((p - t) ** 2).sum(1)
    return {'accuracy': acc, 'nll': nll, 'brier': brier}


def clause_point_and_rows(y, p_svc, p_ref, k):
    """Original clause statistics as differences of means + per-row differences (for resampling/IF)."""
    a = task_rows(y, p_svc, k); r = task_rows(y, p_ref, k)
    point = {'accuracy_loss': float(np.mean(r['accuracy'])) - float(np.mean(a['accuracy'])),
             'nll_excess': float(np.mean(a['nll'])) - float(np.mean(r['nll'])),
             'brier_excess': float(np.mean(a['brier'])) - float(np.mean(r['brier']))}
    rows = {'accuracy_loss': r['accuracy'] - a['accuracy'], 'nll_excess': a['nll'] - r['nll'],
            'brier_excess': a['brier'] - r['brier']}
    return point, rows


CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
METRICS = ('accuracy_loss', 'nll_excess', 'brier_excess')
TASKS = ('income', 'occupation')
KCLS = {'income': 2, 'occupation': 6}


# ----------------------------------------------------------------------------- groups and bootstrap
def group_index(unit):
    units, gidx = np.unique(np.asarray(unit), return_inverse=True)
    return gidx.astype(np.int64), len(units)


def multiplicities(rng, G, B):
    """B x G bootstrap multiplicities: G groups drawn uniformly with replacement per draw."""
    draws = rng.integers(0, G, size=(B, G))
    off = (np.arange(B, dtype=np.int64) * G)[:, None]
    return np.bincount((draws + off).ravel(), minlength=B * G).reshape(B, G).astype(np.float64)


def t_critical(K, df, alpha=0.05):
    return float(stats.t.ppf(1.0 - alpha / (2.0 * K), df))


def z_critical_two_sided(K, alpha=0.05):
    return float(stats.norm.ppf(1.0 - alpha / (2.0 * K)))


def benefit_status(lo, hi, margin):
    if lo > margin:
        return 'ESTABLISHED'
    if hi < margin:
        return 'ESTABLISHED_SHORTFALL'
    return 'UNRESOLVED'


def cap_status(lo, hi, cap):
    if hi <= cap:
        return 'PASS'
    if lo > cap:
        return 'ESTABLISHED_VIOLATION'
    return 'UNRESOLVED'


# ----------------------------------------------------------------------------- power (own model implementation)
def psd_factor(C):
    """Symmetric PSD square-root factor F with F F' = C (eigenvalues clipped at 0)."""
    C = 0.5 * (np.asarray(C, dtype=np.float64) + np.asarray(C, dtype=np.float64).T)
    w, Q = np.linalg.eigh(C)
    return Q * np.sqrt(np.clip(w, 0.0, None))[None, :]


def gstar_draws(mu, bound, direction, C, c, rng, draws, chunk=50000):
    """Smallest G at which every clause passes, per standard-normal draw (known-variance normal model).

    Clause j: direction +1 means 'statistic <= bound' (passes iff T + c*sd/sqrt(G) <= bound);
    direction -1 means 'statistic > bound' (passes iff T - c*sd/sqrt(G) > bound); T = mu + e/sqrt(G), e ~ N(0, C).
    Returns None if some clause has non-positive true slack (not achievable at any G).
    """
    mu = np.asarray(mu, float); bound = np.asarray(bound, float); d = np.asarray(direction, float)
    slack = -d * (mu - bound)
    if (slack <= 0).any():
        return None
    sd = np.sqrt(np.clip(np.diag(C), 0, None))
    F = psd_factor(C)
    out = []
    left = draws
    while left > 0:
        m = min(chunk, left); left -= m
        e = rng.standard_normal((m, F.shape[1])) @ F.T
        num = d[None, :] * e + c * sd[None, :]
        out.append(((np.maximum(num, 0.0) / slack[None, :]) ** 2).max(1))
    return np.concatenate(out)


def direct_power(mu, bound, direction, C, c, G, rng, draws, chunk=50000):
    """Direct check: simulate T ~ N(mu, C/G) and apply the decision rule (fraction of all-clause passes)."""
    mu = np.asarray(mu, float); bound = np.asarray(bound, float); d = np.asarray(direction, float)
    sd = np.sqrt(np.clip(np.diag(C), 0, None)) / math.sqrt(G)
    F = psd_factor(C) / math.sqrt(G)
    passed = 0; left = draws
    while left > 0:
        m = min(chunk, left); left -= m
        T = mu[None, :] + rng.standard_normal((m, F.shape[1])) @ F.T
        ok_le = (T + c * sd[None, :] <= bound[None, :])
        ok_gt = (T - c * sd[None, :] > bound[None, :])
        ok = np.where(d[None, :] > 0, ok_le, ok_gt).all(1)
        passed += int(ok.sum())
    return passed / draws


# ----------------------------------------------------------------------------- IO helpers (files opened only via verify)
def load_npz(verify, item, fields=None):
    p = verify(item['path'], item['sha256'])
    with ZipFile(p) as z:
        names = sorted(n[:-4] for n in z.namelist())
    if fields is not None and sorted(fields) != names:
        raise ValueError(f'unexpected fields in {Path(item["path"]).name}: {names}')
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in names}


def npz_fields(verify, item):
    p = verify(item['path'], item['sha256'])
    with ZipFile(p) as z:
        return sorted(n[:-4] for n in z.namelist())


def load_json(verify, item):
    return json.loads(verify(item['path'], item['sha256']).read_text())


def file_sha(verify, item):
    p = verify(item['path'], item['sha256'])
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def wire_sha(p1, d1, p2, d2):
    h = hashlib.sha256()
    for p, d in ((p1, d1), (p2, d2)):
        h.update(np.ascontiguousarray(p, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(d, dtype=np.int64).tobytes())
    return h.hexdigest()


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o).__name__)


class Receipt:
    """Collect PASS/FAIL checks with numbers; write results/.../independent/STAGE_<name>.json."""

    def __init__(self, name, active, cfg):
        self.name = name; self.active = active; self.cfg = cfg
        self.checks = []; self.numbers = {}; self.notes = []; self.t0 = time.process_time()

    def check(self, name, ok, **detail):
        self.checks.append({'check': name, 'pass': bool(ok), **detail})
        return bool(ok)

    def write(self, extra=None):
        failed = [c['check'] for c in self.checks if not c['pass']]
        rec = {'schema': 'TCV1-independent-stage-v1', 'stage': self.name,
               'status': 'PASS' if not failed and self.checks else 'FAIL',
               'n_checks': len(self.checks), 'n_failed': len(failed), 'failed_checks': failed,
               'checks': self.checks, 'numbers': self.numbers, 'notes': self.notes,
               'queue_unit_id': self.active['unit_id'], 'imports': IMPORTS_NOTE,
               'cpu_s': round(time.process_time() - self.t0, 3),
               'inputs_sha256': self.cfg.get('_inputs_summary', {}),
               'code_sha256': self.cfg.get('_code_sha256', {})}
        if extra:
            rec.update(extra)
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f'STAGE_{self.name}.json'
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(json.dumps(rec, indent=2, sort_keys=True, default=_jsonable) + '\n')
        tmp.replace(path)
        return rec


def code_hashes(verify, cfg):
    """Hash this package's source files through verify (they are registered queue sources)."""
    out = {}
    for name, it in sorted(cfg.get('code', {}).items()):
        out[name] = file_sha(verify, it)
    return out


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def lock_item(cfg, name):
    """Locks use the queue_config convention cfg[name] = path, cfg[name + '_sha256'] = digest (re-verified by guard())."""
    return {'path': cfg[name], 'sha256': cfg[name + '_sha256']}
