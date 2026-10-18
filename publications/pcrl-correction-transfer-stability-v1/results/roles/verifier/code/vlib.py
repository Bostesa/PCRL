"""Independent verifier library of the PCRL correction-transfer campaign (CTS1; EXPLORATORY DEVELOPMENT on previously used records).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (development roles of the
protected-tree replication). The service releases an occupation-group probability vector (5 groups) and its argmax decision;
SEX (recorded demographic column; sex 0 = SEX code 1 = men, sex 1 = SEX code 2 = women) is a training-time and evaluation label,
never a runtime input.

Written by the independent verifier from the TEXT of PROTOCOL.md, PARTITIONS.json and roles/statistical_reviewer/RULES.md.
Nothing here imports the campaign's metric, selection or partition functions (cts.rules, cts.trackf, cts.select, cts.common
partition helpers); this module reads no data and no label. Every label read happens in the verifier's own __main__ scripts
through cts.common.load(operation='verify').

Contents
  hashing        file_sha, arr_sha (dtype + shape + bytes, as the campaign's records define a row-set hash), payload_sha
  partitions     deal (households ordered by SHA-256('<tag>|<household id>'), dealt round-robin), partition_masks_from_json,
                 household_set_sha
  probabilities  softmax_rows, centred_log, release_full, release_ensemble
  metrics        person_metrics (accuracy; NLL with the 1e-12 floor; summed Brier), clustered_se, subgroup_se, increments,
                 incremental_costs
  rules          baseline_from_conditions, old_scale (101-point grid + 30 bisections), simple_select, uncertainty_select,
                 outcome (reserve / point-cap crossings)
  records        append_checks (append-only CHECKS.json with a history snapshot per version)
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
VER_DIR = CODE_DIR.parent
RESULTS = VER_DIR.parent.parent
ROOT = RESULTS.parent.parent
CHECKS = VER_DIR / 'CHECKS.json'
HISTORY = VER_DIR / 'records' / 'history'
CHECK_CHANGES = VER_DIR / 'records' / 'CHECK_CHANGES.json'

K_OCC = 5
NLL_FLOOR = 1e-12
LADDER = tuple(j / 10 for j in range(11))
STATS = ('men_brier', 'women_brier', 'brier', 'nll', 'acc_loss')
CONDITIONS = ('occupation|U0|accuracy_loss', 'occupation|U0|nll_excess', 'occupation|U0|brier_excess',
              'occupation|Ucal|accuracy_loss', 'occupation|Ucal|nll_excess', 'occupation|Ucal|brier_excess',
              'guard|sex0', 'guard|sex1')
# RULES.md section 1 table: which statistic spends which condition
STAT_FOR = {'guard|sex0': 'men_brier', 'guard|sex1': 'women_brier',
            'occupation|U0|brier_excess': 'brier', 'occupation|Ucal|brier_excess': 'brier',
            'occupation|U0|nll_excess': 'nll', 'occupation|Ucal|nll_excess': 'nll',
            'occupation|U0|accuracy_loss': 'acc_loss', 'occupation|Ucal|accuracy_loss': 'acc_loss'}
POPULATION = {'men_brier': 'men', 'women_brier': 'women', 'brier': 'all', 'nll': 'all', 'acc_loss': 'all'}
CAP = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005, 'guard': 0.005}
K_RES = 2.0
FLOOR_HH = 5306
OLD_F = {'B80': 0.8, 'B50': 0.5}
GAMMA = 0.5
Z_REGISTERED = 1.644853626951472        # RULES.md section 5 / section 12 (one-sided 0.05)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


# ================================================================ hashing
def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def arr_sha(a):
    """Row-set / array hash as the campaign's records define it: SHA-256 of dtype string + shape string + C-order bytes."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def payload_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ================================================================ partitions (PARTITIONS.json text)
def deal(unit, parts, tag):
    """Part id per person: households ordered by SHA-256('<tag>|<household id>') and dealt round-robin into `parts` parts."""
    unit = np.asarray(unit)
    groups, inv = np.unique(unit, return_inverse=True)
    inv = inv.reshape(-1)
    digests = [hashlib.sha256(('%s|%s' % (tag, g)).encode()).hexdigest() for g in groups]
    ranked = sorted(range(len(groups)), key=lambda i: digests[i])
    part_of_group = np.empty(len(groups), dtype=np.int64)
    for position, gi in enumerate(ranked):
        part_of_group[gi] = position % parts
    return part_of_group[inv]


def partition_masks_from_json(unit, spec):
    """Masks over MODEL_FIT people from PARTITIONS.json alone: {'C0', 'C1', 'POOL', <pid>: {'FIT', 'GUARD', 'CHECK'}}."""
    unit = np.asarray(unit)
    blk = deal(unit, int(spec['blocks']['n']), spec['blocks']['tag'])
    out = {name: np.isin(blk, np.asarray(b, dtype=np.int64)) for name, b in spec['panels'].items()}
    pool = np.isin(blk, np.asarray(spec['pool_blocks'], dtype=np.int64))
    out['POOL'] = pool
    pool_idx = np.flatnonzero(pool)
    for pid, ps in spec['partitions'].items():
        part = deal(unit[pool_idx], int(ps['parts']), ps['tag'])
        full = np.full(len(unit), -1, dtype=np.int64)
        full[pool_idx] = part
        m = {'FIT': np.isin(full, ps['FIT_parts']) & pool, 'GUARD': np.isin(full, ps['GUARD_parts']) & pool}
        if isinstance(ps['CHECK'], str):
            m['CHECK'] = out[ps['CHECK']].copy()
        else:
            m['CHECK'] = np.isin(full, ps['CHECK']) & pool
        out[pid] = m
    return out


def household_set_sha(unit, mask):
    hh = np.unique(np.asarray(unit)[mask])
    return hashlib.sha256(','.join(str(h) for h in hh).encode()).hexdigest(), int(len(hh)), int(np.sum(mask))


# ================================================================ probabilities
def softmax_rows(z):
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=1, keepdims=True)
    e = np.exp(z - m)
    return e / np.sum(e, axis=1, keepdims=True)


def centred_log(p):
    lz = np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-300, None))
    return lz - np.mean(lz, axis=1, keepdims=True)


def release_full(p_w, delta, s):
    """r_s = softmax(log p_W + s Delta) (PROTOCOL section 6), with log p_W centred (softmax is invariant to row constants)."""
    return softmax_rows(centred_log(p_w) + float(s) * np.asarray(delta, dtype=np.float64))


def release_ensemble(p_folds, delta_folds, s):
    """e_s = mean_k softmax(log p_Wk + s Delta_k)."""
    acc = None
    for k in range(len(p_folds)):
        r = release_full(p_folds[k], delta_folds[k], s)
        acc = r if acc is None else acc + r
    return acc / len(p_folds)


# ================================================================ metrics
def person_metrics(y, p):
    """accuracy = 1[argmax p = y] (ties: first class); NLL = -log(max(p_y, 1e-12)); summed Brier = sum_k (p_k - 1[y = k])^2."""
    y = np.asarray(y, dtype=np.int64); p = np.asarray(p, dtype=np.float64)
    n = len(y)
    if p.shape != (n, K_OCC) or not np.all(np.isfinite(p)):
        raise ValueError('probability array shape / finiteness')
    if np.max(np.abs(p.sum(axis=1) - 1.0)) > 1e-9 or np.min(p) < 0:
        raise ValueError('rows are not probability vectors')
    idx = np.arange(n)
    py = p[idx, y]
    onehot = np.zeros_like(p); onehot[idx, y] = 1.0
    return {'accuracy': (np.argmax(p, axis=1) == y).astype(np.float64),
            'nll': -np.log(np.maximum(py, NLL_FLOOR)),
            'brier': np.sum((p - onehot) ** 2, axis=1)}


def _cluster_sums(values, unit):
    _, inv = np.unique(np.asarray(unit), return_inverse=True)
    inv = inv.reshape(-1)
    G = int(inv.max()) + 1
    sums = np.zeros(G, dtype=np.float64)
    np.add.at(sums, inv, values)
    return sums, G


def clustered_se(d, unit):
    """Household-clustered SE of a mean: sqrt(G / (G - 1) * sum_g (sum_{i in g} (d_i - dbar))^2) / n."""
    d = np.asarray(d, dtype=np.float64)
    sums, G = _cluster_sums(d - d.mean(), unit)
    return math.sqrt(G / (G - 1) * float(np.dot(sums, sums))) / d.size


def subgroup_se(d, mask, unit):
    """Linearized household-clustered SE of the group mean (ratio of means) computed on the whole set:
    z_i = 1[i in group] (d_i - m_group) / share_group; SE = sqrt(G / (G - 1) * sum_g (sum_{i in g} z_i)^2) / n."""
    d = np.asarray(d, dtype=np.float64); mask = np.asarray(mask, dtype=bool)
    m = d[mask].mean(); share = mask.mean()
    zi = np.where(mask, (d - m) / share, 0.0)
    sums, G = _cluster_sums(zi, unit)
    return math.sqrt(G / (G - 1) * float(np.dot(sums, sums))) / d.size


def increments(p, p0, y):
    a, b = person_metrics(y, p), person_metrics(y, p0)
    return a['brier'] - b['brier'], a['nll'] - b['nll'], b['accuracy'] - a['accuracy']


def incremental_costs(p, p0, y, sex, unit):
    """The five incremental statistics (corrected minus uncorrected on the same people) with household-clustered SEs."""
    db, dn, da = increments(p, p0, y)
    sex = np.asarray(sex)
    men, women = sex == 0, sex == 1
    stats = {'men_brier': float(db[men].mean()), 'women_brier': float(db[women].mean()), 'brier': float(db.mean()),
             'nll': float(dn.mean()), 'acc_loss': float(da.mean())}
    se = {'men_brier': subgroup_se(db, men, unit), 'women_brier': subgroup_se(db, women, unit), 'brier': clustered_se(db, unit),
          'nll': clustered_se(dn, unit), 'acc_loss': clustered_se(da, unit)}
    return {'stats': stats, 'se': se, 'n': int(len(db)), 'n_men': int(men.sum()), 'n_women': int(women.sum()),
            'households': int(len(np.unique(np.asarray(unit))))}


# ================================================================ rules (RULES.md sections 2-5, from the text)
def baseline_from_conditions(conds, floor_se, households):
    """Reserve record per occupation-side condition: margin = cap - value; se_used = max(own SE, floor x sqrt(5306 / G));
    rho = margin - 2 se_used; a non-finite margin / own SE gives rho = -inf."""
    scale = math.sqrt(FLOOR_HH / float(households))
    out = {}
    for c in CONDITIONS:
        x = conds[c]
        cap = CAP['guard'] if c.startswith('guard|') else CAP[c.split('|')[2]]
        if abs(float(x['cap']) - cap) > 0:
            raise ValueError(f'{c}: cap {x["cap"]} differs from the registered cap {cap}')
        fl = float(floor_se[c]) * scale
        if not (math.isfinite(fl) and fl > 0):
            raise ValueError(f'{c}: invalid floor')
        own = float(x['se']) if x.get('se') is not None else math.nan
        margin = cap - float(x['value'])
        if not (math.isfinite(own) and math.isfinite(margin)):
            out[c] = {'margin': margin, 'se_used': None, 'rho': -math.inf}
            continue
        su = max(own, fl)
        out[c] = {'margin': margin, 'own_se': own, 'floor_se': fl, 'se_used': su, 'rho': margin - K_RES * su,
                  'se_source': 'own' if own >= fl else 'floor'}
    return out


def stat_budget(rho, f):
    """OLD budget per statistic: f x the smallest rho over the statistic's mapped conditions."""
    out = {}
    for k in STATS:
        out[k] = f * min(rho[c] for c in CONDITIONS if STAT_FOR[c] == k)
    return out


def old_scale(z0, delta, y, sex, budget, grid=101, iters=30):
    """OLD: the largest s on the grid 0, 1/(grid - 1), ..., 1 with every grid point up to s feasible (all five OOF costs <= their
    budget), refined by `iters` bisection steps toward the first infeasible grid point. Costs against s = 0 on the same people."""
    z0 = np.asarray(z0, dtype=np.float64); delta = np.asarray(delta, dtype=np.float64)
    sex = np.asarray(sex)
    p0 = softmax_rows(z0)
    b = person_metrics(y, p0)
    men, women = sex == 0, sex == 1

    def stats_at(s):
        a = person_metrics(y, softmax_rows(z0 + s * delta))
        db = a['brier'] - b['brier']
        return {'men_brier': float(db[men].mean()), 'women_brier': float(db[women].mean()), 'brier': float(db.mean()),
                'nll': float((a['nll'] - b['nll']).mean()), 'acc_loss': float((b['accuracy'] - a['accuracy']).mean())}

    def ok(st):
        return all(st[k] <= budget[k] for k in STATS)
    lo, hi, curve = 0.0, None, {}
    for i in range(grid):
        s = i / (grid - 1)
        st = stats_at(s)
        curve[s] = st
        if not ok(st):
            hi = s
            break
        lo = s
    if hi is not None:
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if ok(stats_at(mid)):
                lo = mid
            else:
                hi = mid
    return {'selected': lo, 'first_infeasible_grid_point': hi, 'stats_at_selected': stats_at(lo),
            'ladder_stats': {s: curve.get(s, None) for s in LADDER}}


def simple_select(guard_stats, rho, gamma=GAMMA):
    """SIMPLE: accept s iff for all 8 conditions cost_{stat(c)}(s) <= b_c, b_c = gamma rho_c if rho_c > 0 else rho_c;
    ascending over the nonzero ladder; the last accepted strength before the first failure (0 = identity)."""
    selected, first_fail, per = 0.0, None, {}
    for s in LADDER[1:]:
        fails = []
        for c in CONDITIONS:
            r = rho[c]
            thr = gamma * r if r > 0 else r
            x = float(guard_stats[s][STAT_FOR[c]])
            if not (math.isfinite(x) and x <= thr):
                fails.append(c)
        per[s] = fails
        if fails:
            first_fail = s
            break
        selected = s
    return {'selected': selected, 'first_failure': first_fail, 'failing_at_first_failure': per.get(first_fail, [])}


def uncertainty_select(guard_stats, guard_se, base, n_guard, n_ms, z=Z_REGISTERED):
    """UNCERTAINTY: per condition c (stat k, population g): se = max(SE_k(s), phi_k) with phi_acc_loss = 1 / n_GUARD and 0
    otherwise; tau = se sqrt(1 + n_GUARD,g / n_MS,g); U1: Delta + z tau <= rho_c; U2: Delta + z sqrt(se^2 + se_used_c^2) <= margin_c;
    accept s iff U1 and U2 hold for all 8 conditions; ascending fixed sequence; last accepted strength (0 = identity)."""
    phi = {k: 0.0 for k in STATS}
    phi['acc_loss'] = 1.0 / float(n_guard['all'])
    selected, first_fail, fail_detail = 0.0, None, []
    for s in LADDER[1:]:
        fails = []
        for c in CONDITIONS:
            k = STAT_FOR[c]; g = POPULATION[k]
            d, se_g = float(guard_stats[s][k]), float(guard_se[s][k])
            b = base[c]
            if b['se_used'] is None or not (math.isfinite(d) and math.isfinite(se_g)):
                fails.append(c); continue
            se = max(se_g, phi[k])
            tau = se * math.sqrt(1.0 + float(n_guard[g]) / float(n_ms[g]))
            u1 = d + z * tau <= b['rho']
            u2 = d + z * math.sqrt(se * se + b['se_used'] ** 2) <= b['margin']
            if not (u1 and u2):
                fails.append(c)
        if fails:
            first_fail, fail_detail = s, fails
            break
        selected = s
    return {'selected': selected, 'first_failure': first_fail, 'failing_at_first_failure': fail_detail}


def outcome(stats, base):
    """Realized reserve / point-cap crossings of an incremental-cost vector against the witness baseline."""
    rc, pc, use = [], [], []
    for c in CONDITIONS:
        x = float(stats[STAT_FOR[c]])
        rho, m = base[c]['rho'], base[c]['margin']
        if x > rho:
            rc.append(c)
        if x > m:
            pc.append(c)
        use.append(x / rho if rho > 0 else (math.inf if x > rho else 0.0))
    return {'reserve_crossing': bool(rc), 'reserve_crossing_conditions': rc, 'point_cap_crossing': bool(pc),
            'point_cap_crossing_conditions': pc, 'max_slack_use': max(use)}


# ================================================================ comparison helpers
def max_abs_diff(a, b):
    """Max |a - b| over matching numeric leaves of two nested dicts; returns (max diff, n compared, list of structural misses)."""
    worst, n, miss = 0.0, 0, []

    def walk(x, y, path):
        nonlocal worst, n
        if isinstance(x, dict) and isinstance(y, dict):
            for k in x:
                if k not in y:
                    miss.append(path + '/' + str(k)); continue
                walk(x[k], y[k], path + '/' + str(k))
        elif isinstance(x, (int, float, np.floating, np.integer)) and isinstance(y, (int, float, np.floating, np.integer)) \
                and not isinstance(x, bool):
            fx, fy = float(x), float(y)
            if math.isnan(fx) and math.isnan(fy):
                n += 1; return
            if math.isinf(fx) or math.isinf(fy):
                if fx != fy:
                    worst = math.inf
                n += 1; return
            worst = max(worst, abs(fx - fy)); n += 1
        else:
            if x != y:
                miss.append(path + f' ({x!r} vs {y!r})'[:200])
    walk(a, b, '')
    return worst, n, miss


# ================================================================ append-only records
def code_hashes(*paths):
    return {Path(p).name: file_sha(p) for p in paths}


def append_checks(new_checks):
    """Append checks to CHECKS.json (never edits an earlier check) and snapshot the new version to records/history/."""
    VER_DIR.mkdir(parents=True, exist_ok=True); HISTORY.mkdir(parents=True, exist_ok=True)
    lock = VER_DIR / 'records' / '.checks.lock'
    with open(lock, 'a+') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if CHECKS.is_file():
                doc = json.loads(CHECKS.read_text())
            else:
                doc = {'schema': 'CTS1-verifier-checks-v1', 'campaign': 'pcrl_correction_transfer_stability_v1',
                       'role': 'independent verifier (separate metric, partition and selection code under roles/verifier/code/)',
                       'rules': ['append-only: a check entry is never edited or removed; every version is snapshotted to records/history/',
                                 'kind NUMERICAL_REPLAY (own code recomputes a recorded number) and kind PROCEDURAL_COMPLIANCE (process '
                                 'and access evidence) are kept separate',
                                 'a failing criterion is never widened; a genuine bug in a verifier check gets a separate correction in '
                                 'records/CHECK_CHANGES.json for adjudication by the coordinator',
                                 'statuses: PASS, FAIL, ATTENTION (no failure of a registered criterion, but a discrepancy or caveat '
                                 'the coordinator must see), NOT_RUN'],
                       'checks': []}
            ids = {c['id'] for c in doc['checks']}
            for c in new_checks:
                if c['id'] in ids:
                    raise ValueError(f'check id {c["id"]} already used (append-only)')
                if c.get('status') not in ('PASS', 'FAIL', 'ATTENTION', 'NOT_RUN'):
                    raise ValueError(f'bad status {c.get("status")}')
                if c.get('kind') not in ('NUMERICAL_REPLAY', 'PROCEDURAL_COMPLIANCE'):
                    raise ValueError(f'bad kind {c.get("kind")}')
                c.setdefault('utc', utc())
                c.setdefault('unit_id', os.environ.get('CTS_UNIT_ID'))
                doc['checks'].append(c)
                ids.add(c['id'])
            counts = {}
            for kind in ('NUMERICAL_REPLAY', 'PROCEDURAL_COMPLIANCE'):
                counts[kind] = {st: sum(1 for c in doc['checks'] if c['kind'] == kind and c['status'] == st)
                                for st in ('PASS', 'FAIL', 'ATTENTION', 'NOT_RUN')}
            doc['summary'] = {'n_checks': len(doc['checks']), 'by_kind_and_status': counts, 'updated_utc': utc()}
            tmp = CHECKS.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(doc, indent=1, sort_keys=False, default=_json_default) + '\n')
            os.replace(tmp, CHECKS)
            stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            shutil.copyfile(CHECKS, HISTORY / f'CHECKS__{stamp}.json')
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return doc['summary']


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        raise TypeError('person-level arrays are never written to verifier records')
    return str(o)


def write_record(name, obj):
    p = VER_DIR / 'records' / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'x') as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=_json_default)
    return p


# ================================================================ invented-data self-test
def selftest():
    rng = np.random.default_rng(7)
    unit = rng.integers(0, 300, 900)
    a = deal(unit, 5, 'T')
    ok = {'deal_whole_households': all(len(set(a[unit == u])) == 1 for u in np.unique(unit)),
          'deal_balanced': max(np.bincount(np.array([a[unit == u][0] for u in np.unique(unit)]))) -
          min(np.bincount(np.array([a[unit == u][0] for u in np.unique(unit)]))) <= 1}
    p = rng.dirichlet(np.ones(K_OCC), 50); dl = rng.normal(size=(50, K_OCC))
    ok['release_identity'] = bool(np.allclose(release_full(p, dl, 0.0), p, atol=1e-13))
    ok['ensemble_equal_members'] = bool(np.allclose(release_ensemble(np.stack([p, p]), np.stack([dl, dl]), 0.7),
                                                    release_full(p, dl, 0.7), atol=1e-13))
    y = rng.integers(0, K_OCC, 50); sex = rng.integers(0, 2, 50); u = np.arange(50) // 2
    c = incremental_costs(release_full(p, dl, 0.0), p, y, sex, u)
    ok['zero_cost_identity'] = all(abs(v) < 1e-12 for v in c['stats'].values())
    # clustered SE with singleton clusters equals the iid SE x sqrt(n / (n - 1)) ... with n-correction
    d = rng.normal(size=200)
    iid = math.sqrt(np.sum((d - d.mean()) ** 2) * 200 / 199) / 200
    ok['cluster_se_singletons'] = abs(clustered_se(d, np.arange(200)) - iid) < 1e-15
    # simple rule: monotone invented curve
    rho = {cc: 0.005 for cc in CONDITIONS}
    gs = {s: {k: s / 100 for k in STATS} for s in LADDER}
    ok['simple_invented'] = simple_select(gs, rho)['selected'] == 0.2
    rho_neg = dict(rho); rho_neg['guard|sex0'] = -0.001
    ok['simple_negative_slack_identity'] = simple_select(gs, rho_neg)['selected'] == 0.0
    ok = {k: bool(v) for k, v in ok.items()}
    print(json.dumps(ok))
    return all(ok.values())


if __name__ == '__main__':
    raise SystemExit(0 if selftest() else 1)
