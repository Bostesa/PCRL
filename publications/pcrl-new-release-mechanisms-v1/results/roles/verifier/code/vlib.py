"""Independent verifier library of the PCRL new-release-mechanisms campaign (`pcrl_new_release_mechanisms_v1`).

Own implementations, written from the registered definitions (PROTOCOL.md, READER_SPLITS.json, ACCESS_POLICY.json, the statistical
reviewer's RULES.json and the source rule files read as DATA). Nothing here imports nrm.evaluate, nrm.screen, nrm.devreaders,
ohn.evaluate, ihs.evaluate, tsz.devreaders, task_conditioned_representation.inference or any other metric / AUC / selection code of
the coordinator or the source campaigns. Labels are never read here: every label read happens in a verifier script (operation
'verify', through nrm.common.load, which logs it), and the arrays are passed in.

Contents
- paths, hashes (file, array, wire, view, cache key, household-set), git helpers;
- household folds (SHA-256 ordering of household ids, dealt round-robin), reader roles, screening halves;
- per-person losses (error, NLL with the 1e-12 floor, summed Brier), household-clustered SE, linearized subgroup SE;
- the 12 clauses + 2 guards of one version (draw-averaged for randomized releases), the whole-release reserve (k = 2, floor SE);
- AUC (Mann-Whitney on average ranks), weighted AUC by sample expansion (bootstrap draws), household bootstrap weights;
- winner pick (max VALIDATION AUC, exact ties to the earlier candidate), aggregate-screen statistics;
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
CAMPAIGN = 'pcrl_new_release_mechanisms_v1'
RESULTS = ROOT / 'results' / CAMPAIGN
PRIV = ROOT / '.private' / CAMPAIGN
RUNS = PRIV / 'runs'
ACCESS_DIR = PRIV / 'access'
ACCESS_LOG = ACCESS_DIR / 'LABEL_ACCESS.jsonl'
RECEIPT = ACCESS_DIR / 'REGISTRATION_RECEIPT_VERIFIED.json'
QUEUE_LEDGER = PRIV / 'compute' / 'ledger.jsonl'
CONFIG_LEDGER = RUNS / 'CONFIG_LEDGER.jsonl'
CHECKS = VER_DIR / 'CHECKS.json'
CHECK_CHANGES = VER_DIR / 'CHECK_CHANGES.json'
RECORDS = VER_DIR / 'records'
HISTORY = RECORDS / 'history'
NOTES = RECORDS / 'NOTES.jsonl'
VER_PRIV = PRIV / 'verifier'                     # private verifier scratch (person-level arrays never go to results/)

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
K_RESERVE = 2.0
FLOOR_HOUSEHOLDS = 5306
SRC8 = ('OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB', 'LOOKUP')
NEW8 = ('ET5', 'ET25', 'HGB15', 'HGB31', 'MLP64', 'MLP6432', 'CATI01', 'CATI1')
RECIPES = SRC8 + NEW8
VIEWS = ('income', 'occupation', 'pair')
REG_REPS = ('R1', 'R2')
ALL_REPS = ('R1', 'R2', 'R3', 'R4')
REP_INDEX = {'R1': 0, 'R2': 1, 'R3': 2, 'R4': 3}
MODEL_SEED_BASE = 2026101205
READER_OFFSET = 600
BOOT_B = 400
BOOT_BASE = 2026102000
GRID_VERSION = 'NRM1-devgrid-v1'
NEAR_TIE = 0.010
STATUSES = ('PASS', 'FAIL', 'ATTENTION', 'NOT_APPLICABLE', 'NOT_RUN')


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
    """Registered array hash: SHA-256 of dtype string, shape string and C-order bytes."""
    z = np.ascontiguousarray(a)
    return bytes_sha(str(z.dtype).encode() + str(z.shape).encode() + z.tobytes())


def argmax_first(p):
    """Decision = index of the first maximum per row (own loop-free implementation, no np.argmax)."""
    p = np.asarray(p, dtype=np.float64)
    mx = p.max(axis=1, keepdims=True)
    hit = p == mx
    return (hit.cumsum(axis=1) == 0).sum(axis=1).astype(np.int64)


def wire_sha(p):
    """One recipient's released wire: float64 probability bytes, then the int64 decision bytes."""
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    return bytes_sha(p.tobytes() + np.ascontiguousarray(argmax_first(p)).tobytes())


def view_hash(view, inc, occ):
    """Reader-cache wire hash of a view: the view name, then the float64 arrays the view reads."""
    h = hashlib.sha256(view.encode())
    parts = (inc,) if view == 'income' else (occ,) if view == 'occupation' else (inc, occ)
    for a in parts:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


def reader_seed(version, k):
    return MODEL_SEED_BASE + READER_OFFSET + 10 * int(version) + int(k)


def boot_seed(version, rep):
    return BOOT_BASE + 100 * int(version) + REP_INDEX[rep]


def cache_key(rep, rs, frac, wh, sex_sha):
    tag = rep if rep in REG_REPS else f'{rep}:NRM1_devgrid_{rep}'
    return bytes_sha(f'{GRID_VERSION}|{tag}|{int(rs)}|{float(frac)}|{wh}|{sex_sha}'.encode())


def hh_set_sha(households):
    hh = sorted(int(h) for h in set(np.asarray(households).tolist()))
    return bytes_sha(','.join(str(h) for h in hh).encode())


def git(*args):
    r = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def git_blob_sha256(commit, relpath):
    """SHA-256 of a file's bytes as committed at `commit` (None if absent)."""
    r = subprocess.run(['git', '-C', str(ROOT), 'show', f'{commit}:{relpath}'], capture_output=True)
    return bytes_sha(r.stdout) if r.returncode == 0 else None


def committed_unchanged(path):
    """True iff the file is tracked and identical to HEAD."""
    rel = str(Path(path).resolve().relative_to(ROOT))
    rc, out, _ = git('ls-files', '--error-unmatch', rel)
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


# ---------------------------------------------------------------- household folds and reader roles (label-free)
def household_part(unit, k, tag):
    """Fold id per person: households ordered by SHA-256('<tag>|<household id>') hex digest, dealt round-robin into k parts."""
    unit = np.asarray(unit)
    groups = sorted(set(int(u) for u in unit.tolist()))
    keyed = sorted(groups, key=lambda g: hashlib.sha256(f'{tag}|{g}'.encode()).hexdigest())
    part_of = {g: i % k for i, g in enumerate(keyed)}
    return np.fromiter((part_of[int(u)] for u in unit.tolist()), dtype=np.int64, count=len(unit))


def reader_roles(unit, rep):
    part = household_part(unit, 10, f'NRM1_devgrid_{rep}')
    return {'FIT': part <= 4, 'VALIDATION': (part >= 5) & (part <= 6), 'SCORE': part >= 7}


def screen_halves(unit):
    part = household_part(unit, 2, 'NRM1_screen')
    return part == 0, part == 1


# ---------------------------------------------------------------- per-person losses and standard errors
def check_prob(p, k, n=None, what='release'):
    p = np.asarray(p, dtype=np.float64)
    problems = []
    if p.ndim != 2 or p.shape[1] != k:
        problems.append(f'shape {p.shape}, expected (n, {k})')
    elif n is not None and p.shape[0] != n:
        problems.append(f'{p.shape[0]} rows, expected {n}')
    if not np.isfinite(p).all():
        problems.append('non-finite entries')
    elif (p < 0).any():
        problems.append('negative entries')
    elif p.ndim == 2 and np.abs(p.sum(1) - 1.0).max() > 1e-9:
        problems.append(f'row sums deviate from 1 by up to {np.abs(p.sum(1) - 1.0).max():.3g}')
    return problems


def person_losses(y, p, k):
    """Per-person error (1 - correct argmax), NLL (true-class probability floored at 1e-12) and summed Brier."""
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    if check_prob(p, k, len(y)):
        raise ValueError(f'invalid probability vectors: {check_prob(p, k, len(y))}')
    rows = np.arange(len(y))
    err = (argmax_first(p) != y).astype(np.float64)
    nll = -np.log(np.clip(p[rows, y], NLL_FLOOR, None))
    onehot = np.zeros_like(p)
    onehot[rows, y] = 1.0
    brier = np.square(p - onehot).sum(axis=1)
    return {'err': err, 'nll': nll, 'brier': brier}


def _cluster_index(unit):
    u = np.asarray(unit)
    _, inv = np.unique(u, return_inverse=True)
    return inv.ravel(), int(inv.max()) + 1


def cluster_se(d, unit):
    """SE of mean(d) clustered by household: sqrt(G / (G - 1) * sum_g (sum_{i in g} (d_i - mean d))^2) / n."""
    d = np.asarray(d, dtype=np.float64)
    inv, G = _cluster_index(unit)
    s = np.zeros(G)
    np.add.at(s, inv, d - d.mean())
    return float(math.sqrt(G / (G - 1) * float(np.dot(s, s))) / len(d))


def subgroup_se(d, mask, unit):
    """Linearized household-clustered SE of the subgroup mean of d over mask (ratio of two means)."""
    d = np.asarray(d, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    m = d[mask].mean()
    p = mask.mean()
    z = np.where(mask, (d - m) / p, 0.0)
    inv, G = _cluster_index(unit)
    s = np.zeros(G)
    np.add.at(s, inv, z)
    return float(math.sqrt(G / (G - 1) * float(np.dot(s, s))) / len(d))


def draws_of(rel):
    """List of per-draw {'income', 'occupation'} arrays (a deterministic release is one draw; a stacked task gives R draws)."""
    inc, occ = np.asarray(rel['income']), np.asarray(rel['occupation'])
    if inc.ndim == 2 and occ.ndim == 2:
        return [{'income': inc, 'occupation': occ}]
    R = inc.shape[0] if inc.ndim == 3 else occ.shape[0]
    if inc.ndim == 3 and occ.ndim == 3 and inc.shape[0] != occ.shape[0]:
        raise ValueError('income and occupation draw counts differ')
    return [{'income': inc[r] if inc.ndim == 3 else inc, 'occupation': occ[r] if occ.ndim == 3 else occ} for r in range(R)]


def conditions_one_draw(labels, rel, refs, sex, unit):
    """{condition: (value, se)} for the 12 clauses and 2 guards of one emitted draw."""
    out = {}
    own = {t: person_losses(labels[t], rel[t], K[t]) for t in TASKS}
    for t in TASKS:
        for ref in REFS:
            r = person_losses(labels[t], refs[ref][t], K[t])
            vecs = {'accuracy_loss': own[t]['err'] - r['err'], 'nll_excess': own[t]['nll'] - r['nll'],
                    'brier_excess': own[t]['brier'] - r['brier']}
            for m in METRICS:
                d = vecs[m]
                out[f'{t}|{ref}|{m}'] = (float(d.mean()), cluster_se(d, unit))
    u0 = person_losses(labels['occupation'], refs['U0']['occupation'], 5)
    d = own['occupation']['brier'] - u0['brier']
    for g, name in ((0, 'guard|sex0'), (1, 'guard|sex1')):
        mask = np.asarray(sex) == g
        out[name] = (float(d[mask].mean()), subgroup_se(d, mask, unit))
    return out


def conditions_table(labels, rel, refs, sex, unit):
    """Draw-averaged conditions: value = mean over draws, se = mean per-draw SE, se_mc = SD over draws / sqrt(R) (R > 1)."""
    per = [conditions_one_draw(labels, d, refs, sex, unit) for d in draws_of(rel)]
    R = len(per)
    out = {}
    for name in CONDITIONS:
        vals = np.array([p[name][0] for p in per])
        ses = np.array([p[name][1] for p in per])
        cap = GUARD_CAP if name.startswith('guard') else CAPS[name.split('|')[2]]
        rec = {'value': float(vals.mean()), 'se': float(ses.mean()), 'cap': cap, 'draws': R}
        if R > 1:
            rec['se_mc'] = float(vals.std(ddof=1) / math.sqrt(R))
            rec['draw_values'] = vals.tolist()
            rec['worst_draw_point_pass'] = bool((vals <= cap).all())
        rec['point_pass'] = bool(rec['value'] <= cap)
        out[name] = rec
    return out


def reserve_version(conds, floor_se, households, k=K_RESERVE, g0=FLOOR_HOUSEHOLDS, with_mc=False):
    """Whole-release reserve of one version: se_used = max(own SE, floor SE x sqrt(g0 / G)); met iff cap - value >= k se_used.
    with_mc: own SE = sqrt(se^2 + se_mc^2) (the reviewer's randomized-release variant)."""
    scale = math.sqrt(g0 / households)
    recs = {}
    for name in CONDITIONS:
        c = conds[name]
        own = float(c['se'])
        if with_mc and c.get('se_mc') is not None:
            own = math.sqrt(own ** 2 + float(c['se_mc']) ** 2)
        fl = float(floor_se[name]) * scale
        if not (math.isfinite(fl) and fl > 0):
            raise ValueError(f'{name}: floor SE must be finite and positive')
        value, cap = float(c['value']), float(c['cap'])
        if not (math.isfinite(value) and math.isfinite(own)):
            recs[name] = {'ratio': None, 'met': False}
            continue
        used = max(own, fl)
        margin = cap - value
        recs[name] = {'value': value, 'cap': cap, 'own_se': own, 'floor_se': fl, 'se_used': used, 'margin': margin,
                      'ratio': margin / used, 'met': bool(margin >= k * used)}
    ratios = {n: (r['ratio'] if r['ratio'] is not None else -math.inf) for n, r in recs.items()}
    binding = min(CONDITIONS, key=lambda n: (ratios[n], CONDITIONS.index(n)))
    occ_side = [n for n in CONDITIONS if n.startswith('occupation') or n.startswith('guard')]
    inc_side = [n for n in CONDITIONS if n.startswith('income')]
    occ_binding = min(occ_side, key=lambda n: (ratios[n], CONDITIONS.index(n)))
    return {'met': all(r['met'] for r in recs.values()), 'min_ratio': ratios[binding], 'binding': binding,
            'min_occ_side_ratio': ratios[occ_binding], 'occ_binding': occ_binding, 'occ_side_met': all(recs[n]['met'] for n in occ_side),
            'min_income_ratio': min(ratios[n] for n in inc_side), 'income_met': all(recs[n]['met'] for n in inc_side),
            'conditions': recs}


def per_person_avg(labels, rel, sex=None):
    """Per-person losses averaged over draws ({task: {'acc_loss', 'nll', 'brier'}}) and per-draw person means ('draw_means')."""
    ds = draws_of(rel)
    out = {}
    for t in TASKS:
        per = [person_losses(labels[t], d[t], K[t]) for d in ds]
        out[t] = {'acc_loss': np.mean([p['err'] for p in per], axis=0), 'nll': np.mean([p['nll'] for p in per], axis=0),
                  'brier': np.mean([p['brier'] for p in per], axis=0),
                  'draw_means': {'acc_loss': np.array([p['err'].mean() for p in per]), 'nll': np.array([p['nll'].mean() for p in per]),
                                 'brier': np.array([p['brier'].mean() for p in per])}}
    return out


def distinct_rows(p):
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    return len({row.tobytes() for row in p})


# ---------------------------------------------------------------- AUC and bootstrap
def auc(y, s):
    """Mann-Whitney AUC with average ranks for ties (scipy.stats.rankdata)."""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(np.int64)
    s = np.asarray(s, dtype=np.float64)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError('AUC needs both classes')
    r = rankdata(s, method='average')
    P = int(y.sum())
    N = len(y) - P
    return float((r[y == 1].sum() - P * (P + 1) / 2.0) / (P * N))


def auc_expanded(y, s, counts):
    """AUC of the sample expanded by integer multiplicities (a household-bootstrap draw), by explicit repetition."""
    counts = np.asarray(counts).astype(np.int64)
    idx = np.repeat(np.arange(len(y)), counts)
    return auc(np.asarray(y)[idx], np.asarray(s)[idx])


def boot_counts(unit_sc, seed, B=BOOT_B):
    """Household bootstrap multiplicities (B x n): numpy default_rng(seed); per draw, G household indices drawn uniformly with
    replacement among the sorted distinct SCORE households (READER_SPLITS.json 'bootstrap')."""
    groups, inv = np.unique(np.asarray(unit_sc), return_inverse=True)
    G = len(groups)
    rng = np.random.default_rng(seed)
    out = np.empty((B, len(unit_sc)), dtype=np.int64)
    for b in range(B):
        c = np.zeros(G, dtype=np.int64)
        np.add.at(c, rng.integers(0, G, G), 1)
        out[b] = c[inv.ravel()]
    return out


def pick(cands):
    """cands: [(name, validation AUC, score key, origin)]; winner = max AUC, exact ties to the EARLIER candidate; gap to the best
    other candidate; near tie < 0.010."""
    best = 0
    for i in range(1, len(cands)):
        if cands[i][1] > cands[best][1]:
            best = i
    others = [c[1] for i, c in enumerate(cands) if i != best]
    gap = float(cands[best][1] - max(others)) if others else None
    return {'candidate': cands[best][0], 'reader_key': cands[best][2], 'origin': cands[best][3],
            'val_auc': float(cands[best][1]), 'gap': gap, 'near_tie': bool(gap is not None and gap < NEAR_TIE)}


def pair_candidates(val):
    """val[view][recipe] = validation AUC. Order: per group (source 8, then new 8): pair readers, ignore-occupation (income view),
    ignore-income (occupation view)."""
    out = []
    for group, org in ((SRC8, 'source'), (NEW8, 'new')):
        out += [(n, val['pair'][n], f'pair__{n}', org) for n in group]
        out += [(f'IGNORE_occupation__{n}', val['income'][n], f'income__{n}', org) for n in group]
        out += [(f'IGNORE_income__{n}', val['occupation'][n], f'occupation__{n}', org) for n in group]
    return out


def single_candidates(val, view):
    return [(n, val[view][n], f'{view}__{n}', 'source' if n in SRC8 else 'new') for n in RECIPES]


# ---------------------------------------------------------------- records (append-only checks, write-once records, notes)
def _jd(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        if o.size > 64:
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


def code_sha(*names):
    return {n: file_sha(CODE_DIR / n) for n in names}


def _canon(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True, default=_jd).encode()).hexdigest()


def append_checks(new):
    """Append checks to CHECKS.json; never edits or removes an earlier check (verified by hashing every earlier entry before the
    write); every version is snapshotted to records/history/. Each check: id, what, inputs (with hashes), result,
    numerical_status, procedural_status, utc."""
    RECORDS.mkdir(parents=True, exist_ok=True)
    HISTORY.mkdir(parents=True, exist_ok=True)
    with open(RECORDS / '.checks.lock', 'a+') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if CHECKS.is_file():
                doc = json.loads(CHECKS.read_text())
            else:
                doc = {'schema': 'NRM1-verifier-checks-v1', 'campaign': CAMPAIGN,
                       'role': 'independent verifier (own code under roles/verifier/code/; labels only via nrm.common.load, operation '
                               'verify, logged)',
                       'rules': ['append-only: an entry is never edited or removed; each written version is snapshotted to '
                                 'records/history/ and the hashes of all earlier entries are re-checked before every write',
                                 'numerical_status (own code recomputes a recorded number) and procedural_status (process, access, '
                                 'record evidence) are recorded separately; values PASS, FAIL, ATTENTION, NOT_APPLICABLE, NOT_RUN',
                                 'an acceptance rule is never widened after a failure; a corrected check gets a new id and an entry in '
                                 'CHECK_CHANGES.json (before, after, cause, adjudication request); the earlier result is kept'],
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
                c.setdefault('unit_id', os.environ.get('NRM_UNIT_ID'))
                c = json.loads(json.dumps(c, default=_jd))
                doc['checks'].append(c)
                doc['entry_sha256'].append(_canon(c))
                ids.add(c['id'])
            doc['summary'] = {
                'n_checks': len(doc['checks']),
                'numerical': {s: sum(c['numerical_status'] == s for c in doc['checks']) for s in STATUSES},
                'procedural': {s: sum(c['procedural_status'] == s for c in doc['checks']) for s in STATUSES},
                'updated_utc': utc()}
            tmp = CHECKS.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(doc, indent=1, default=_jd) + '\n')
            os.replace(tmp, CHECKS)
            stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            shutil.copyfile(CHECKS, HISTORY / f'CHECKS__{stamp}.json')
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return doc['summary']


def add_check_change(entry):
    RECORDS.mkdir(parents=True, exist_ok=True)
    HISTORY.mkdir(parents=True, exist_ok=True)
    doc = json.loads(CHECK_CHANGES.read_text()) if CHECK_CHANGES.is_file() else {
        'schema': 'NRM1-verifier-check-changes-v1',
        'rules': 'every correction of a verifier check: the earlier check stays unchanged in CHECKS.json; before, after, cause and the '
                 'adjudication request are recorded here; acceptance rules are never widened', 'changes': []}
    entry = dict(entry)
    entry.setdefault('utc', utc())
    entry['n'] = len(doc['changes']) + 1
    doc['changes'].append(json.loads(json.dumps(entry, default=_jd)))
    tmp = CHECK_CHANGES.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(doc, indent=1, default=_jd) + '\n')
    os.replace(tmp, CHECK_CHANGES)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    shutil.copyfile(CHECK_CHANGES, HISTORY / f'CHECK_CHANGES__{stamp}.json')
    return entry['n']


def write_record(name, obj):
    """Write-once aggregate record under records/ (person-level arrays refused)."""
    p = RECORDS / name
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


def status_of(ok, attention=False):
    return 'PASS' if ok and not attention else ('ATTENTION' if ok else 'FAIL')
