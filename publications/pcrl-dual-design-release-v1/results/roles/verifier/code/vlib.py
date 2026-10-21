"""Independent verifier library of the PCRL dual-design release campaign (`pcrl_dual_design_release_v1`).

Own implementations written from the registered definitions (PROTOCOL.md, PROTOCOL_AMENDMENT_1.md, the cards DA_STD_v1 / DB_PTC_v1,
RULES.md). Nothing here imports ddr.evaluate, ddr.devreaders, ddr.rules, ddr.claims or a metric / reader / rule function of a
source package. No label is read here: label reads happen only in the verifier scripts themselves (ddr.common.load, operation
'verify', inside a queue unit), because the access layer authorizes the calling script path.

Contents: paths; hashes (file, array, wire, view); first-argmax; household dealing (SHA-256 order of '<tag>|<household>', round
robin); grid split masks; per-person task losses (NLL floor 1e-12, summed Brier, error indicator); household-clustered SE and the
linearized subgroup SE; rank AUC with half credit for ties and its weighted (bootstrap) form; JSON record writer (write-once).
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
VER_DIR = CODE_DIR.parent
ROOT = VER_DIR.parents[3]
CAMPAIGN = 'pcrl_dual_design_release_v1'
RESULTS = ROOT / 'results' / CAMPAIGN
PRIV = ROOT / '.private' / CAMPAIGN
RUNS = PRIV / 'runs'
ACCESS_LOG = PRIV / 'access' / 'LABEL_ACCESS.jsonl'
RECEIPT = PRIV / 'access' / 'REGISTRATION_RECEIPT_VERIFIED.json'
QUEUE_LEDGER = PRIV / 'compute' / 'ledger.jsonl'
CONFIG_LEDGER = RUNS / 'CONFIG_LEDGER.jsonl'
RECORDS = VER_DIR / 'records'

TASKS = ('income', 'occupation')
K = {'income': 2, 'occupation': 5}
REFS = ('U0', 'Ucal')
METRICS = ('accuracy_loss', 'nll_excess', 'brier_excess')
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
GUARD_CAP = 0.005
NLL_FLOOR = 1e-12
VIEWS = ('income', 'occupation', 'pair')
SEED_BASE = 2026101205 + 800                # MODEL_SEED_BASE + 800 (PROTOCOL 5; READER_SPLITS.json reader_seeds)
BOOT_BASE = 2026102200                      # + 100 v + repetition index (PROTOCOL 5)
B_BOOT = 400
REP_INDEX = {'R1': 0, 'R2': 1, 'R3': 2, 'R4': 3}
GRID_VERSION = 'DDR1-devgrid-v1'            # READER_SPLITS.json cache-key definition


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def parse_utc(s):
    t = dt.datetime.fromisoformat(str(s).replace('Z', '+00:00'))
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def arr_sha(a):
    """SHA-256 of the dtype string, the shape string and the C-order bytes (registered array hash)."""
    z = np.ascontiguousarray(a)
    return hashlib.sha256(str(z.dtype).encode() + str(z.shape).encode() + z.tobytes()).hexdigest()


def argmax_first(p):
    """Index of the first maximum of each row (own loop-free implementation)."""
    p = np.asarray(p, dtype=np.float64)
    hit = p == p.max(axis=1, keepdims=True)
    return (np.cumsum(hit, axis=1) == 0).sum(axis=1).astype(np.int64)


def wire_sha(p):
    """One recipient's wire: SHA-256 of the float64 probability bytes, then the int64 first-argmax decision bytes."""
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    h = hashlib.sha256(p.tobytes())
    h.update(np.ascontiguousarray(argmax_first(p)).tobytes())
    return h.hexdigest()


def view_hash(view, inc, occ):
    """The grid's per-view wire hash (READER_SPLITS cache key): view name, then the float64 arrays the view reads."""
    h = hashlib.sha256(view.encode())
    arrays = (inc,) if view == 'income' else (occ,) if view == 'occupation' else (inc, occ)
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


def cache_key(rep, reader_seed, view_wire, sex_sha, frac=1.0):
    return hashlib.sha256(f'{GRID_VERSION}|{rep}|{reader_seed}|{float(frac)}|{view_wire}|{sex_sha}'.encode()).hexdigest()


def reader_seed(v, k):
    return SEED_BASE + 10 * int(v) + int(k)


# ---------------------------------------------------------------- household dealing and grid splits
def dealt(unit, k, tag):
    """Part id per person: households sorted by the SHA-256 hex digest of '<tag>|<household>' and dealt round robin into k."""
    unit = np.asarray(unit)
    groups = np.unique(unit)
    digests = np.array([hashlib.sha256(f'{tag}|{g}'.encode()).hexdigest() for g in groups])
    rank = np.empty(len(groups), dtype=np.int64)
    rank[np.argsort(digests, kind='stable')] = np.arange(len(groups))
    part_of_group = rank % k
    idx = np.searchsorted(groups, unit)
    return part_of_group[idx]


def split_masks(unit, rep):
    part = dealt(unit, 10, f'DDR1_devgrid_{rep}')
    return {'FIT': part <= 4, 'VALIDATION': (part >= 5) & (part <= 6), 'SCORE': part >= 7}


# ---------------------------------------------------------------- per-person losses and SEs
def losses(y, p, k):
    """Per person: error indicator (first argmax), NLL (floor 1e-12), summed Brier."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (len(y), k):
        raise ValueError(f'shape {p.shape} is not ({len(y)}, {k})')
    py = p[np.arange(len(y)), y]
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return {'err': (argmax_first(p) != y).astype(np.float64),
            'nll': -np.log(np.maximum(py, NLL_FLOOR)),
            'brier': ((p - onehot) ** 2).sum(axis=1)}


def clustered_se(d, unit):
    """SE of the mean with household clustering: sqrt(G/(G-1) * sum_g (sum_{i in g} (d_i - dbar))^2) / n."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    groups, inv = np.unique(unit, return_inverse=True)
    G = len(groups)
    s = np.zeros(G)
    np.add.at(s, inv, d - d.mean())
    return float(np.sqrt(G / (G - 1) * np.sum(s * s)) / n)


def subgroup_mean_se(d, mask, unit):
    """Subgroup mean of d over mask and its linearized household-clustered SE (ratio of two means)."""
    d = np.asarray(d, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n = len(d)
    m = float(d[mask].mean())
    p = float(mask.mean())
    z = np.where(mask, (d - m) / p, 0.0)
    groups, inv = np.unique(unit, return_inverse=True)
    G = len(groups)
    s = np.zeros(G)
    np.add.at(s, inv, z)
    return m, float(np.sqrt(G / (G - 1) * np.sum(s * s)) / n)


# ---------------------------------------------------------------- AUC
def auc(y, s):
    """Rank AUC of score s for y == 1 against y == 0, ties counted one half (average ranks)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    ranks = np.empty(len(s))
    i = 0
    n = len(s)
    # average ranks over tied blocks (vectorized by block boundaries)
    bounds = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1, n]
    for a, b in zip(bounds[:-1], bounds[1:]):
        ranks[order[a:b]] = 0.5 * (a + 1 + b)
    n1 = int(y.sum())
    n0 = n - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def weighted_auc(y, s, W):
    """AUC of fixed scores s under person weights W (B x n): sum over value blocks of w_pos * (w_neg below + half w_neg tied),
    divided by total positive weight x total negative weight."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    vals, inv = np.unique(s, return_inverse=True)
    pos = y.astype(np.float64)
    neg = 1.0 - pos
    out = np.empty(W.shape[0])
    for b in range(W.shape[0]):
        wp = np.bincount(inv, weights=W[b] * pos, minlength=len(vals))
        wn = np.bincount(inv, weights=W[b] * neg, minlength=len(vals))
        below = np.cumsum(wn) - wn
        out[b] = float(np.sum(wp * (below + 0.5 * wn)) / (wp.sum() * wn.sum()))
    return out


def household_bootstrap_weights(unit_sc, seed, B=B_BOOT):
    """Household bootstrap of the SCORE people: B draws of G households with replacement (numpy default_rng(seed), integers(0, G,
    G) per draw), each person weighted by its household's count (the registered bootstrap: B = 400, seed 2026102200 + 100 v + rep
    index)."""
    groups, inv = np.unique(unit_sc, return_inverse=True)
    G = len(groups)
    rng = np.random.default_rng(seed)
    W = np.empty((B, len(unit_sc)))
    for b in range(B):
        cnt = np.bincount(rng.integers(0, G, G), minlength=G)
        W[b] = cnt[inv]
    return W


# ---------------------------------------------------------------- releases (read-only; hash-checked here, never via ddr.release)
def release_paths(family, config, v):
    d = RUNS / family / config / f'V{int(v)}'
    return d / 'RELEASE.npz', d / 'RELEASE.json'


def load_release_ms(family, config, v, roles=('MODEL_SELECT',)):
    npz, js = release_paths(family, config, v)
    rec = json.loads(js.read_text())
    got = file_sha(npz)
    if got != rec['release_sha256']:
        raise RuntimeError(f'{family}/{config}/V{v}: RELEASE.npz SHA-256 {got} differs from RELEASE.json')
    out = {}
    with np.load(npz, allow_pickle=False) as z:
        for key in z.files:
            role, task = key.split('__')
            if roles is None or role in roles:
                out.setdefault(role, {})[task] = z[key].copy()
    return out, rec


# ---------------------------------------------------------------- records
def write_record(name, obj):
    """Write-once JSON record under roles/verifier/records/ (a rerun gets a new name). A label-free dry run outside the queue
    (environment VER_DRYRUN_DIR set) writes to that directory instead and never to the records."""
    dry = os.environ.get('VER_DRYRUN_DIR')
    base = Path(dry) if dry else RECORDS
    base.mkdir(parents=True, exist_ok=True)
    p = base / name
    with open(p, 'x') as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=_default)
        f.write('\n')
    return p


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f'not serializable: {type(o)}')


def unit_id():
    return os.environ.get('DDR_UNIT_ID')


def selftest():
    """Invented data only."""
    rng = np.random.default_rng(7)
    ok = {}
    y = rng.integers(0, 2, 500)
    s = rng.normal(size=500) + 0.5 * y
    s[:50] = np.round(s[:50])
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(s[y == 1], s[y == 0]).statistic
    ok['auc_matches_mann_whitney'] = abs(auc(y, s) - u / ((y == 1).sum() * (y == 0).sum())) < 1e-12
    ok['weighted_auc_unit_weights'] = abs(weighted_auc(y, s, np.ones((1, 500)))[0] - auc(y, s)) < 1e-12
    p = np.array([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    ok['argmax_first'] = argmax_first(p).tolist() == [0, 2, 0]
    unit = np.repeat(np.arange(100), 5)
    d = rng.normal(size=500)
    ok['cluster_se_positive'] = clustered_se(d, unit) > 0
    ok['all_ok'] = all(ok.values())
    return ok


if __name__ == '__main__':
    print(json.dumps(selftest(), default=_default))
