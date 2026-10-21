"""Independent verifier library of the PCRL finite-routing comparison campaign (`pcrl_finite_routing_comparison_v1`).

Own implementations written from the registered definitions (PROTOCOL.md, the card FA_FRC_v1.md, RULES.md / RULES.json, the
source card FA_PRC_v1 read as DATA). Nothing here imports a numerical function of frc.*, nrm.*, ohn.evaluate, ihs.evaluate,
scm_bench.selection or any other metric / routing / selection code of the coordinator or of the source campaigns. Labels are never
read here: every label read happens in a verifier script through frc.common.load(operation 'verify'), which logs it.

Structure follows the previous campaign's verifier (results/pcrl_new_release_mechanisms_v1/roles/verifier/code/vlib.py); every
formula below was re-derived and re-typed for this campaign.

Contents
- paths, hashes (file, array, wire), git helpers, time;
- household folds (households ordered by the SHA-256 hex digest of '<tag>|<household id>', dealt round-robin);
- the finite-routing constructions: cross-entropy routing to codewords (explicit per-class sums, first index on ties, with the
  best / second-best gap), sequential cell sums (add.accumulate in row order), clip-and-renormalize, B1 tables, A decoders,
  partition identity (bijection between two labelings), value ids read off released vectors;
- per-person losses (error, NLL with the 1e-12 floor, summed Brier), household-clustered SE, linearized subgroup SE;
- the 12 clauses + 2 guards of one version, QUAL Q1 / Q2 with the S2 increments, whole-release reserve (k = 2, floor SE);
- append-only CHECKS.json writer (numerical and procedural status kept separate), write-once records, notes.
"""
from __future__ import annotations
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
VER_DIR = CODE_DIR.parent
ROOT = VER_DIR.parents[3]
CAMPAIGN = 'pcrl_finite_routing_comparison_v1'
RESULTS = ROOT / 'results' / CAMPAIGN
PRIV = ROOT / '.private' / CAMPAIGN
RUNS = PRIV / 'runs'
FA_CACHE = PRIV / 'fa_cache'
ACCESS_DIR = PRIV / 'access'
ACCESS_LOG = ACCESS_DIR / 'LABEL_ACCESS.jsonl'
RECEIPT = ACCESS_DIR / 'REGISTRATION_RECEIPT_VERIFIED.json'
QUEUE_LEDGER = PRIV / 'compute' / 'ledger.jsonl'
CONFIG_LEDGER = RUNS / 'CONFIG_LEDGER.jsonl'
SRC_CAMPAIGN = 'pcrl_new_release_mechanisms_v1'
SRC_PRIV = ROOT / '.private' / SRC_CAMPAIGN
SRC_RUNS = SRC_PRIV / 'runs'
SRC_FA_CACHE = SRC_PRIV / 'fa_cache'
SRC_RESULTS = ROOT / 'results' / SRC_CAMPAIGN
CHECKS = VER_DIR / 'CHECKS.json'
CHECK_CHANGES = VER_DIR / 'CHECK_CHANGES.json'
RECORDS = VER_DIR / 'records'
HISTORY = RECORDS / 'history'
NOTES = RECORDS / 'NOTES.jsonl'
UNIT_ENV = 'FRC_UNIT_ID'
REMOTE_URL = 'https://github.com/Bostesa/PCRL.git'
PUB_BRANCH = 'publication/pcrl-finite-routing-comparison-v1'
PUB_NAME = 'pcrl-finite-routing-comparison-v1'

TASKS = ('income', 'occupation')
K = {'income': 2, 'occupation': 5}
REFS = ('U0', 'Ucal')
METRICS = ('accuracy_loss', 'nll_excess', 'brier_excess')
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
GUARD_CAP = 0.005
NLL_FLOOR = 1e-12
CLAUSES = tuple(f'{t}|{r}|{m}' for t in TASKS for r in REFS for m in METRICS)
GUARDS = ('guard|sex0', 'guard|sex1')
CONDITIONS = CLAUSES + GUARDS
# QUAL Q2 increments (RULES.md section 3, the S2 budget): read from the protocol text, typed here independently
Q2_INC = {'income|accuracy_loss': 0.004, 'income|nll_excess': 0.004, 'income|brier_excess': 0.0020,
          'occupation|accuracy_loss': 0.006, 'occupation|nll_excess': 0.008, 'occupation|brier_excess': 0.0030,
          'guard': 0.0030}
K_RESERVE = 2.0
FLOOR_HOUSEHOLDS = 5306
CLIP = 1e-6                       # clip-and-renormalize floor of the constructions (FA_PRC_v1 / FA_FRC_v1)
ALPHA = 20.0                      # pseudo-counts of the label-refit decoders A
CAPACITIES = {'C64': (16, 64), 'C128': (24, 128), 'C256': (32, 256)}
SOURCE_TWIN = {'C64': 'PRC_TWIN', 'C128': 'PRC_TWIN_C128'}
OUTER_TAG = 'NRM_FA_ncv_S{v}'
GATE_LIMITS = {'occ_nll': 0.004, 'inc_nll': 0.004, 'occ_acc': 0.003, 'inc_acc': 0.003}
NEST_METRICS = ('inc_nll', 'occ_nll', 'tot_nll', 'inc_brier', 'occ_brier', 'inc_acc', 'occ_acc')
STATUSES = ('PASS', 'FAIL', 'ATTENTION', 'NOT_APPLICABLE', 'NOT_RUN')
ROLES6 = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')


# ---------------------------------------------------------------- time, hashes, git
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


def bytes_sha(b):
    return hashlib.sha256(b).hexdigest()


def arr_sha(a):
    """Registered array hash: SHA-256 of the dtype string, the shape string and the C-order bytes."""
    z = np.ascontiguousarray(a)
    return bytes_sha(str(z.dtype).encode() + str(z.shape).encode() + z.tobytes())


def argmax_first(p):
    """Index of the first maximum of each row (own loop-free implementation; np.argmax is not used)."""
    p = np.asarray(p, dtype=np.float64)
    hit = p == p.max(axis=1, keepdims=True)
    return (hit.cumsum(axis=1) == 0).sum(axis=1).astype(np.int64)


def wire_sha(p):
    """One recipient's wire: SHA-256 of the float64 probability bytes followed by the int64 first-argmax decision bytes."""
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    return bytes_sha(p.tobytes() + np.ascontiguousarray(argmax_first(p)).tobytes())


def git(*args):
    r = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def git_blob_sha256(commit, relpath):
    r = subprocess.run(['git', '-C', str(ROOT), 'show', f'{commit}:{relpath}'], capture_output=True)
    return bytes_sha(r.stdout) if r.returncode == 0 else None


def committed_unchanged(path):
    rel = str(Path(path).resolve().relative_to(ROOT))
    rc, _, _ = git('ls-files', '--error-unmatch', rel)
    if rc != 0:
        return False
    rc, out, _ = git('status', '--porcelain', '--', rel)
    return rc == 0 and out == ''


def commit_time(commit):
    rc, out, _ = git('show', '-s', '--format=%cI', commit)
    return parse_utc(out) if rc == 0 and out else None


def is_ancestor(commit, of='HEAD'):
    rc, _, _ = git('merge-base', '--is-ancestor', commit, of)
    return rc == 0


# ---------------------------------------------------------------- household folds (label-free)
def household_folds(unit, k, tag):
    """Fold id per person: the distinct households are ordered by the hex SHA-256 digest of '<tag>|<household id>' and dealt
    round-robin into k parts (position i -> part i mod k). Household ids must be integers (their decimal text is hashed)."""
    unit = np.asarray(unit)
    if not np.issubdtype(unit.dtype, np.integer):
        raise TypeError(f'household ids must be integers (got {unit.dtype})')
    groups = sorted({int(u) for u in unit.tolist()})
    order = sorted(groups, key=lambda g: hashlib.sha256(f'{tag}|{g}'.encode()).hexdigest())
    part = {g: i % k for i, g in enumerate(order)}
    return np.fromiter((part[int(u)] for u in unit.tolist()), dtype=np.int64, count=len(unit))


# ---------------------------------------------------------------- the finite-routing constructions
def clip_norm(T):
    """Clip every entry to [1e-6, 1], then divide each row by its (numpy) row sum."""
    T = np.minimum(np.maximum(np.asarray(T, dtype=np.float64), CLIP), 1.0)
    return T / T.sum(axis=1, keepdims=True)


def xent_costs(Q, C):
    """Cross-entropy cost of every person (rows of Q) to every codeword (rows of C): -sum_c q_c log C_kc, accumulated class by
    class in class order (explicit per-class products; no matrix product)."""
    Q = np.asarray(Q, dtype=np.float64)
    L = np.log(np.asarray(C, dtype=np.float64))
    cost = np.zeros((Q.shape[0], L.shape[0]))
    for c in range(Q.shape[1]):
        cost -= Q[:, c][:, None] * L[:, c][None, :]
    return cost


def route_nearest(Q, C, chunk=4096):
    """(cell, gap): nearest codeword in cross-entropy, first index on exact ties; gap = second-best minus best cost (inf if one
    codeword), the margin that rounding would have to overcome to change the assignment."""
    n = Q.shape[0]
    cell = np.empty(n, dtype=np.int64)
    gap = np.empty(n)
    for a in range(0, n, chunk):
        cost = xent_costs(Q[a:a + chunk], C)
        j = cost.argmin(axis=1)                      # numpy argmin returns the first minimum
        best = cost[np.arange(len(j)), j]
        if cost.shape[1] > 1:
            c2 = cost.copy()
            c2[np.arange(len(j)), j] = np.inf
            gap[a:a + chunk] = c2.min(axis=1) - best
        else:
            gap[a:a + chunk] = np.inf
        cell[a:a + chunk] = j
    return cell, gap


def seq_cell_sums(cells, Q, Kc):
    """Per-cell sums of the rows of Q, added in row order (left to right: ((q_i1 + q_i2) + q_i3) + ...), via a stable sort by cell
    and np.add.accumulate over each cell's rows; and the per-cell counts."""
    cells = np.asarray(cells, dtype=np.int64)
    Q = np.asarray(Q, dtype=np.float64)
    order = np.argsort(cells, kind='stable')
    cs = cells[order]
    S = np.zeros((Kc, Q.shape[1]))
    n = np.zeros(Kc, dtype=np.int64)
    bounds = np.flatnonzero(np.diff(cs)) + 1
    starts = np.concatenate([[0], bounds]) if len(cs) else np.array([], dtype=np.int64)
    ends = np.concatenate([bounds, [len(cs)]]) if len(cs) else np.array([], dtype=np.int64)
    for s, e in zip(starts, ends):
        c = int(cs[s])
        S[c] = np.add.accumulate(Q[order[s:e]], axis=0)[-1]
        n[c] = e - s
    return S, n


def b1_table(cells, Q, C):
    """B1: mean of the out-of-fold teacher predictions of the people of each occupied cell (sequential sum / count); an unoccupied
    cell keeps its codeword; then clip-and-renormalize of the whole table. Returns (table, counts, unoccupied count)."""
    Kc = np.asarray(C).shape[0]
    S, n = seq_cell_sums(cells, Q, Kc)
    T = np.array(C, dtype=np.float64, copy=True)
    occ = n > 0
    T[occ] = S[occ] / n[occ][:, None]
    return clip_norm(T), n, int((~occ).sum())


def a_decoders(cells, y, C, alpha=ALPHA):
    """A: clip-and-renormalize of (label counts of the cell + alpha x codeword) / (cell count + alpha)."""
    C = np.asarray(C, dtype=np.float64)
    Kc, k = C.shape
    cnt = np.zeros((Kc, k))
    np.add.at(cnt, (np.asarray(cells, dtype=np.int64), np.asarray(y, dtype=np.int64)), 1.0)   # integer counts (exact)
    return clip_norm((cnt + alpha * C) / (cnt.sum(axis=1, keepdims=True) + alpha))


def row_ids(P):
    """Integer id of each row's exact float64 bytes (first-appearance order) and the number of distinct rows."""
    P = np.ascontiguousarray(np.asarray(P, dtype=np.float64))
    seen, ids = {}, np.empty(P.shape[0], dtype=np.int64)
    for i, row in enumerate(P):
        b = row.tobytes()
        ids[i] = seen.setdefault(b, len(seen))
    return ids, len(seen)


def same_partition(a, b):
    """True iff the labelings a and b of the same people induce the same partition (a bijection between their values)."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    pa, pb = {}, {}
    for x, y in zip(a.tolist(), b.tolist()):
        if pa.setdefault(x, y) != y or pb.setdefault(y, x) != x:
            return False
    return True


def view_ids(rel):
    """Value ids read off a release's vectors (income, occupation, linked pair)."""
    ii, ni = row_ids(rel['income'])
    io, no = row_ids(rel['occupation'])
    return {'income': ii, 'occupation': io, 'pair': ii * (no + 1) + io}, {'income': ni, 'occupation': no}


def table_index(rel_vectors, table):
    """For every person, the index of the table row whose bytes equal the person's vector (-1 if none; ValueError if two table
    rows are equal, i.e. the table rows do not identify cells)."""
    T = np.ascontiguousarray(np.asarray(table, dtype=np.float64))
    lut = {}
    for c, row in enumerate(T):
        b = row.tobytes()
        if b in lut:
            raise ValueError(f'table rows {lut[b]} and {c} are bitwise equal')
        lut[b] = c
    V = np.ascontiguousarray(np.asarray(rel_vectors, dtype=np.float64))
    return np.fromiter((lut.get(r.tobytes(), -1) for r in V), dtype=np.int64, count=V.shape[0])


# ---------------------------------------------------------------- per-person losses and standard errors
def check_prob(p, k, n=None):
    p = np.asarray(p, dtype=np.float64)
    out = []
    if p.ndim != 2 or p.shape[1] != k:
        out.append(f'shape {p.shape}, expected (n, {k})')
    elif n is not None and p.shape[0] != n:
        out.append(f'{p.shape[0]} rows, expected {n}')
    if not np.isfinite(p).all():
        out.append('non-finite entries')
    elif (p < 0).any():
        out.append('negative entries')
    elif p.ndim == 2 and np.abs(p.sum(1) - 1.0).max() > 1e-9:
        out.append('row sums deviate from 1')
    return out


def person_losses(y, p, k):
    """Per-person error (1 - correct first argmax), NLL (true-class probability floored at 1e-12) and summed Brier."""
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    bad = check_prob(p, k, len(y))
    if bad:
        raise ValueError(f'invalid probability vectors: {bad}')
    rows = np.arange(len(y))
    err = (argmax_first(p) != y).astype(np.float64)
    nll = -np.log(np.maximum(p[rows, y], NLL_FLOOR))
    onehot = np.zeros_like(p)
    onehot[rows, y] = 1.0
    brier = np.square(p - onehot).sum(axis=1)
    return {'err': err, 'nll': nll, 'brier': brier}


def _cluster_index(unit):
    _, inv = np.unique(np.asarray(unit), return_inverse=True)
    inv = np.asarray(inv).ravel()
    return inv, int(inv.max()) + 1


def cluster_se(d, unit):
    """Household-clustered SE of mean(d): sqrt(G / (G - 1) * sum_g (sum_{i in g} (d_i - mean d))^2) / n (own sandwich)."""
    d = np.asarray(d, dtype=np.float64)
    inv, G = _cluster_index(unit)
    s = np.zeros(G)
    np.add.at(s, inv, d - d.mean())
    return float(math.sqrt(G / (G - 1) * float(np.dot(s, s))) / len(d))


def subgroup_se(d, mask, unit):
    """Linearized household-clustered SE of the subgroup mean of d over mask (a ratio of two means)."""
    d = np.asarray(d, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    m = d[mask].mean()
    p = mask.mean()
    z = np.where(mask, (d - m) / p, 0.0)
    inv, G = _cluster_index(unit)
    s = np.zeros(G)
    np.add.at(s, inv, z)
    return float(math.sqrt(G / (G - 1) * float(np.dot(s, s))) / len(d))


def conditions_version(labels, rel, refs, sex, unit):
    """{condition: {'value', 'se', 'cap', 'point_pass'}} for the 12 clauses (task x U0/Ucal x accuracy loss, NLL excess, summed
    Brier excess; value = mean(own) - mean(ref) for losses) and the 2 guards (occupation summed-Brier excess vs U0 among men
    (sex 0) and women (sex 1)) of one deterministic release and version."""
    out = {}
    own = {t: person_losses(labels[t], rel[t], K[t]) for t in TASKS}
    for t in TASKS:
        for ref in REFS:
            r = person_losses(labels[t], refs[ref][t], K[t])
            vec = {'accuracy_loss': own[t]['err'] - r['err'], 'nll_excess': own[t]['nll'] - r['nll'],
                   'brier_excess': own[t]['brier'] - r['brier']}
            for m in METRICS:
                d = vec[m]
                val = float(d.mean())
                out[f'{t}|{ref}|{m}'] = {'value': val, 'se': cluster_se(d, unit), 'cap': CAPS[m], 'point_pass': bool(val <= CAPS[m])}
    u0 = person_losses(labels['occupation'], refs['U0']['occupation'], 5)
    d = own['occupation']['brier'] - u0['brier']
    for g, name in ((0, 'guard|sex0'), (1, 'guard|sex1')):
        mask = np.asarray(sex) == g
        val = float(d[mask].mean())
        out[name] = {'value': val, 'se': subgroup_se(d, mask, unit), 'cap': GUARD_CAP, 'point_pass': bool(val <= GUARD_CAP)}
    return out, own


def q2_increment(name):
    if name.startswith('guard|'):
        return Q2_INC['guard']
    t, _, m = name.split('|')
    return Q2_INC[f'{t}|{m}']


def qual_version(conds):
    """Q1 (every condition value <= cap) and Q2 (value <= cap - increment, cap - increment in double precision) of one version."""
    rows = {}
    for n in CONDITIONS:
        c = conds[n]
        lim = float(c['cap']) - q2_increment(n)
        rows[n] = {'value': float(c['value']), 'cap': float(c['cap']), 'increment': q2_increment(n), 'slack_limit': lim,
                   'pass': bool(float(c['value']) <= lim), 'slack_after_increment': lim - float(c['value'])}
    q1_failed = sorted(n for n in CONDITIONS if not float(conds[n]['value']) <= float(conds[n]['cap']))
    q2_failed = sorted(n for n, r in rows.items() if not r['pass'])
    binding = min(CONDITIONS, key=lambda n: (rows[n]['slack_after_increment'], CONDITIONS.index(n)))
    return {'q1': not q1_failed, 'q1_failed': q1_failed, 'q2': not q2_failed, 'q2_failed': q2_failed, 'binding': binding,
            'rows': rows}


def reserve_version(conds, floor_se, households, k=K_RESERVE, g0=FLOOR_HOUSEHOLDS):
    """Whole-release reserve of one version: se_used = max(own SE, floor SE x sqrt(g0 / G)); met iff cap - value >= k se_used."""
    scale = math.sqrt(g0 / households)
    recs = {}
    for name in CONDITIONS:
        c = conds[name]
        fl = float(floor_se[name]) * scale
        own = float(c['se'])
        used = max(own, fl)
        margin = float(c['cap']) - float(c['value'])
        recs[name] = {'se_used': used, 'margin': margin, 'ratio': margin / used, 'met': bool(margin >= k * used)}
    ratios = {n: r['ratio'] for n, r in recs.items()}
    binding = min(CONDITIONS, key=lambda n: (ratios[n], CONDITIONS.index(n)))
    occ_side = [n for n in CONDITIONS if n.startswith('occupation') or n.startswith('guard')]
    inc_side = [n for n in CONDITIONS if n.startswith('income')]
    occ_binding = min(occ_side, key=lambda n: (ratios[n], CONDITIONS.index(n)))
    return {'met': all(r['met'] for r in recs.values()), 'min_ratio': ratios[binding], 'binding': binding,
            'min_occ_side_ratio': ratios[occ_binding], 'occ_binding': occ_binding,
            'occ_side_met': all(recs[n]['met'] for n in occ_side), 'min_income_ratio': min(ratios[n] for n in inc_side),
            'income_met': all(recs[n]['met'] for n in inc_side), 'conditions': recs}


# ---------------------------------------------------------------- records (append-only checks, write-once records, notes)
def _jd(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        if o.size > 256:
            raise TypeError('person-level arrays are never written to verifier records')
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def rel(p):
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def inputs_sha(paths):
    out = {}
    for p in paths:
        p = Path(p)
        out[rel(p)] = file_sha(p) if p.is_file() else None
    return out


def _canon(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True, default=_jd).encode()).hexdigest()


def append_checks(new, checks_path=None):
    """Append checks to CHECKS.json; never edits or removes an earlier check (hashes of all earlier entries are re-checked before
    every write); each written version is snapshotted to records/history/. Each check: id, what, inputs, result,
    numerical_status, procedural_status (PASS / FAIL / ATTENTION / NOT_APPLICABLE / NOT_RUN), utc, unit_id."""
    path = Path(checks_path) if checks_path else CHECKS
    hist = path.parent / 'records' / 'history' if checks_path else HISTORY
    hist.mkdir(parents=True, exist_ok=True)
    with open(hist.parent / '.checks.lock', 'a+') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if path.is_file():
                doc = json.loads(path.read_text())
            else:
                doc = {'schema': 'FRC1-verifier-checks-v1', 'campaign': CAMPAIGN,
                       'role': 'independent verifier (own code under roles/verifier/code/; labels only via frc.common.load, '
                               'operation verify, inside queue units, logged)',
                       'rules': ['append-only: an entry is never edited or removed; each written version is snapshotted to '
                                 'records/history/ and the hashes of all earlier entries are re-checked before every write',
                                 'numerical_status (own code recomputes a recorded number) and procedural_status (process, access, '
                                 'record evidence) are recorded separately; values PASS, FAIL, ATTENTION, NOT_APPLICABLE, NOT_RUN',
                                 'an acceptance rule is never widened after a failure; a corrected check gets a new id and an entry '
                                 'in CHECK_CHANGES.json (before, after, cause); the earlier result is kept'],
                       'checks': [], 'entry_sha256': []}
            if [_canon(c) for c in doc['checks']] != doc.get('entry_sha256', []):
                raise RuntimeError('CHECKS.json earlier entries changed (append-only violated); refusing to write')
            ids = {c['id'] for c in doc['checks']}
            for c in new:
                c = dict(c)
                if c['id'] in ids:
                    raise ValueError(f'check id {c["id"]} already used (append-only)')
                for key in ('numerical_status', 'procedural_status'):
                    if c.get(key) not in STATUSES:
                        raise ValueError(f'{c["id"]}: bad {key} {c.get(key)!r}')
                for key in ('what', 'inputs', 'result'):
                    if key not in c:
                        raise ValueError(f'{c["id"]}: missing {key}')
                c.setdefault('utc', utc())
                c.setdefault('unit_id', os.environ.get(UNIT_ENV))
                c = json.loads(json.dumps(c, default=_jd))
                doc['checks'].append(c)
                doc['entry_sha256'].append(_canon(c))
                ids.add(c['id'])
            doc['summary'] = {
                'n_checks': len(doc['checks']),
                'numerical': {s: sum(c['numerical_status'] == s for c in doc['checks']) for s in STATUSES},
                'procedural': {s: sum(c['procedural_status'] == s for c in doc['checks']) for s in STATUSES},
                'updated_utc': utc()}
            tmp = path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(doc, indent=1, default=_jd) + '\n')
            os.replace(tmp, path)
            stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            shutil.copyfile(path, hist / f'{path.stem}__{stamp}.json')
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return doc['summary']


def write_record(name, obj, records_dir=None):
    """Write-once aggregate record under records/ (person-level arrays refused)."""
    p = (Path(records_dir) if records_dir else RECORDS) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'x') as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=_jd)
    return p


def note(text, **kw):
    RECORDS.mkdir(parents=True, exist_ok=True)
    with open(NOTES, 'a') as f:
        f.write(json.dumps({'utc': utc(), 'note': text, **kw}, default=_jd) + '\n')


def jsonl(path):
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def status(ok, attention=False):
    return 'PASS' if ok and not attention else ('ATTENTION' if ok else 'FAIL')


def max_abs(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max()) if a.size else 0.0
