"""Independent verifier library of the PCRL input-and-pair release campaign (`pcrl_input_and_pair_release_v1`).

Own implementations written from the registered definitions (PROTOCOL.md, PROTOCOL_AMENDMENT_1.md, the cards IA_SFG_v1, IB_CSF_v1,
CTL_PATHS_v1, CTL_TWIN_v1 and roles/statistical_reviewer/RULES.md). Nothing here imports ipr.evaluate, ipr.devreaders, ipr.rules,
ipr.claims or a metric / reader / rule function of a source package. No label is read here: label reads happen only in the verifier
scripts themselves (ipr.common.load, operation 'verify', inside a queue unit), because the access layer authorizes the calling script
path.

Contents: paths; hashes (file, array, wire, view); first-argmax; household dealing (SHA-256 order of '<tag>|<household>', round
robin); grid split masks; per-person task losses (NLL floor 1e-12, summed Brier, error indicator); household-clustered SE and the
linearized subgroup SE; rank AUC with half credit for ties and its weighted (bootstrap) form; the aggregate screen of the registered
cells (own arithmetic from RULES.md 0 / NRM1 definitions); write-once JSON records.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
VER_DIR = CODE_DIR.parent
ROOT = VER_DIR.parents[3]
CAMPAIGN = 'pcrl_input_and_pair_release_v1'
RESULTS = ROOT / 'results' / CAMPAIGN
PRIV = ROOT / '.private' / CAMPAIGN
RUNS = PRIV / 'runs'
ACCESS_LOG = PRIV / 'access' / 'LABEL_ACCESS.jsonl'
RECEIPT = PRIV / 'access' / 'REGISTRATION_RECEIPT_VERIFIED.json'
QUEUE_LEDGER = PRIV / 'compute' / 'ledger.jsonl'
CONFIG_LEDGER = RUNS / 'CONFIG_LEDGER.jsonl'
RECORDS = VER_DIR / 'records'
GRID_DIR = RUNS / 'GRID' / 'IPR1_MAIN'
CACHE_DIR = RUNS / 'devgrid_cache'

TASKS = ('income', 'occupation')
K = {'income': 2, 'occupation': 5}
REFS = ('U0', 'Ucal')
METRICS = ('accuracy_loss', 'nll_excess', 'brier_excess')
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
GUARD_CAP = 0.005
NLL_FLOOR = 1e-12
VIEWS = ('income', 'occupation', 'pair')
MODEL_SEED_BASE = 2026101205
SEED_BASE = MODEL_SEED_BASE + 900          # PROTOCOL 5: MODEL_SEED_BASE + 900 + 10 v + k
BOOT_BASE = 2026102300                     # + 100 v + repetition index
B_BOOT = 400
REP_INDEX = {'R1': 0, 'R2': 1, 'R3': 2, 'R4': 3}
GRID_VERSION = 'IPR1-devgrid-v1'
REG_CELLS = tuple(f'S{v}{r}k{k}' for v in (0, 1) for r in ('R1', 'R2') for k in (0, 1))
Z = 2.0
SE_FLOOR = 0.001
RECIPIENT_LIMIT = 0.010
TARGET = 0.020


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
    """SHA-256 of the dtype string, the shape string and the C-order bytes (the registered array hash)."""
    z = np.ascontiguousarray(a)
    return hashlib.sha256(str(z.dtype).encode() + str(z.shape).encode() + z.tobytes()).hexdigest()


def argmax_first(p):
    """Index of the first maximum of each row (own loop-free implementation)."""
    p = np.asarray(p, dtype=np.float64)
    hit = p == p.max(axis=1, keepdims=True)
    return (np.cumsum(hit, axis=1) == 0).sum(axis=1).astype(np.int64)


def wire_sha(p):
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    h = hashlib.sha256(p.tobytes())
    h.update(np.ascontiguousarray(argmax_first(p)).tobytes())
    return h.hexdigest()


def view_hash(view, inc, occ):
    """The grid's per-view wire hash (cache key input): view name, then the float64 arrays the view reads."""
    h = hashlib.sha256(view.encode())
    arrays = (inc,) if view == 'income' else (occ,) if view == 'occupation' else (inc, occ)
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


def cache_key(rep, reader_seed, view_wire, sex_sha, frac=1.0):
    tag = rep if rep in ('R1', 'R2') else f'{rep}:IPR1_devgrid_{rep}'
    return hashlib.sha256(f'{GRID_VERSION}|{tag}|{reader_seed}|{float(frac)}|{view_wire}|{sex_sha}'.encode()).hexdigest()


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
    part = dealt(unit, 10, f'IPR1_devgrid_{rep}')
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
    n = len(s)
    bounds = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1, n]
    for a, b in zip(bounds[:-1], bounds[1:]):
        ranks[order[a:b]] = 0.5 * (a + 1 + b)
    n1 = int(y.sum())
    n0 = n - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def weighted_auc(y, s, W):
    """AUC of fixed scores s under person weights W (B x n)."""
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
    """B draws of G SCORE households with replacement (numpy default_rng(seed), integers(0, G, G) per draw); each person weighted by
    its household's count."""
    groups, inv = np.unique(unit_sc, return_inverse=True)
    G = len(groups)
    rng = np.random.default_rng(seed)
    W = np.empty((B, len(unit_sc)))
    for b in range(B):
        cnt = np.bincount(rng.integers(0, G, G), minlength=G)
        W[b] = cnt[inv]
    return W


# ---------------------------------------------------------------- aggregate screen (own arithmetic, RULES.md 0 / NRM1 definitions)
def parse_cell(c):
    return int(c[1]), c[2:4], int(c[5:])


def aggregate(cells, ses, mcs=None, which=REG_CELLS):
    """Blocks = (version, repetition); m_b = mean of the block's reader-seed cells; x = mean of the block means; se_hh = RMS of the
    cell SEs with the weights of x; se_rs = sqrt(sum_b (m_b - x)^2 / n_b); se_mc likewise; se_agg = max(raw, 0.001) unless an exact
    alias (all values and SEs zero). Per version the same quantities from that version's cells."""
    mcs = mcs or {}
    names = [c for c in which if c in cells]
    if sorted(names) != sorted(which):
        return {'complete': False, 'missing': sorted(set(which) - set(names))}
    blocks = {}
    for c in names:
        v, r, k = parse_cell(c)
        blocks.setdefault((v, r), {})[k] = (float(cells[c]), float(ses[c]), float(mcs.get(c) or 0.0))

    def summ(keys):
        nb = len(keys)
        mb = {b: sum(t[0] for t in blocks[b].values()) / len(blocks[b]) for b in keys}
        x = sum(mb.values()) / nb
        hh = math.sqrt(sum(blocks[b][k][1] ** 2 / (nb * len(blocks[b])) for b in keys for k in blocks[b]))
        mc = math.sqrt(sum(blocks[b][k][2] ** 2 / (nb * len(blocks[b])) for b in keys for k in blocks[b]))
        rs = math.sqrt(sum((m - x) ** 2 for m in mb.values()) / nb)
        return {'mean': x, 'se_hh': hh, 'se_rs': rs, 'se_mc': mc, 'raw': math.sqrt(hh * hh + rs * rs + mc * mc)}
    keys = sorted(blocks)
    tot = summ(keys)
    per = {v: summ([b for b in keys if b[0] == v]) for v in (0, 1)}
    alias = all(t[0] == 0.0 and t[1] == 0.0 and t[2] == 0.0 for b in blocks.values() for t in b.values())
    for s in [tot] + list(per.values()):
        s['se_agg'] = 0.0 if alias else max(s['raw'], SE_FLOOR)
        s['lower'], s['upper'] = s['mean'] - Z * s['se_agg'], s['mean'] + Z * s['se_agg']
    return {'complete': True, 'alias': alias, **tot, 'per_version': per}


def benefit_label(agg, threshold=0.0):
    if not agg.get('complete'):
        return 'INCOMPLETE'
    if agg['alias']:
        return 'ALIAS_ZERO'
    if agg['lower'] > threshold and all(agg['per_version'][v]['mean'] > threshold for v in (0, 1)):
        return 'AGG_POSITIVE' if threshold == 0 else 'TARGET_AGG_RESOLVED'
    if agg['mean'] > threshold:
        return 'AGG_POSITIVE_POINT_ONLY' if threshold == 0 else 'TARGET_AGG_POINT_ONLY'
    return 'AGG_NOT_POSITIVE' if threshold == 0 else 'TARGET_AGG_NOT_MET'


def recipient_label(agg, limit=RECIPIENT_LIMIT):
    if not agg.get('complete'):
        return 'INCOMPLETE', False
    pv = agg['per_version']
    if agg['alias']:
        lab = 'ALIAS_ZERO'
    elif any(pv[v]['lower'] > limit for v in (0, 1)):
        lab = 'RECIPIENT_AGG_RESOLVED_WORSE'
    elif not all(pv[v]['mean'] <= limit for v in (0, 1)):
        lab = 'RECIPIENT_AGG_POINT_FAILED'
    elif all(pv[v]['upper'] <= limit for v in (0, 1)):
        lab = 'RECIPIENT_AGG_RESOLVED'
    else:
        lab = 'RECIPIENT_AGG_EDGE'
    return lab, lab in ('RECIPIENT_AGG_RESOLVED', 'RECIPIENT_AGG_EDGE', 'ALIAS_ZERO')


# ---------------------------------------------------------------- records
def write_record(name, obj):
    """Write-once JSON record under roles/verifier/records/ (a rerun gets a new name)."""
    RECORDS.mkdir(parents=True, exist_ok=True)
    p = RECORDS / name
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
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f'not serializable: {type(o)}')


def unit_id():
    return os.environ.get('IPR_UNIT_ID')


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
    cells = {c: 0.01 for c in REG_CELLS}
    ses = {c: 0.002 for c in REG_CELLS}
    a = aggregate(cells, ses)
    ok['aggregate_constant'] = abs(a['mean'] - 0.01) < 1e-15 and abs(a['se_agg'] - 0.002) < 1e-15 and benefit_label(a) == 'AGG_POSITIVE'
    ok['all_ok'] = all(ok.values())
    return ok


if __name__ == '__main__':
    print(json.dumps(selftest(), default=_default))
