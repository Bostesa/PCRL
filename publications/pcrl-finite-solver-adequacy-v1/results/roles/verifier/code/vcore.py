"""Independent verifier core (campaign `pcrl_finite_solver_adequacy_v1`): paths, hashing, git, the append-only CHECKS.json
store, and the verifier's OWN numerical primitives.

Nothing here imports an fsa / frc / nrm numerical function. The primitives are written from their definitions:
- household-clustered SE of a mean: sqrt(G / (G - 1) * sum_g (sum_{i in g} (v_i - vbar))^2) / n (households from the person's unit
  id; persons summed per household by sorting, not by a unique-inverse bincount);
- ratio-estimator (linearised) subgroup mean: estimate mean(d[sub]); SE = clustered SE of z_i = 1[sub_i] (d_i - est) / mean(1[sub]);
- AUC of a score for a binary label: exact pair count P(s_pos > s_neg) + 0.5 P(tie) by searchsorted on the sorted negatives;
- weighted AUC with nonnegative integer person weights (household bootstrap counts): the same pair count weighted by w_i w_j;
- household bootstrap weights: B draws of G households with replacement from rng.integers(0, G, G), households indexed in
  sorted id order, person weight = its household's draw count.

    python3 vcore.py --selftest        # invented data only: every primitive against a brute-force reference, and the store
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[4]
CAMPAIGN = 'pcrl_finite_solver_adequacy_v1'
RES = ROOT / 'results' / CAMPAIGN
VER = RES / 'roles' / 'verifier'
PRIV = ROOT / '.private' / CAMPAIGN
ACCESS = PRIV / 'access'
COMPUTE = PRIV / 'compute'
FRC = 'pcrl_finite_routing_comparison_v1'
NRM = 'pcrl_new_release_mechanisms_v1'
FRC_RES = ROOT / 'results' / FRC
FRC_PRIV = ROOT / '.private' / FRC
NRM_RES = ROOT / 'results' / NRM
NRM_PRIV = ROOT / '.private' / NRM
REMOTE = 'https://github.com/Bostesa/PCRL.git'
PUB_BRANCH = 'publication/pcrl-finite-solver-adequacy-v1'
PUB_NAME = 'pcrl-finite-solver-adequacy-v1'
STATUSES = ('PASS', 'FAIL', 'ATTENTION', 'NOT_APPLICABLE', 'NOT_RUN')
KINDS = ('numerical', 'procedural')


# ---------------------------------------------------------------- small utilities
def admitted_loaders():
    """The admitted loader module fsa.common (features label-free; load logged with operation 'verify'). A script run by path has
    only its own folder on sys.path, so the repository root is appended (after the verifier's folder, never before it)."""
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from fsa import common as C
    return C


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def parse_utc(s):
    if s is None:
        return None
    s = str(s).replace('Z', '+00:00')
    t = dt.datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def arr_sha(a):
    """sha256(dtype string + shape string + C-contiguous bytes): the logged row-set hash definition."""
    z = np.ascontiguousarray(a)
    return hashlib.sha256(str(z.dtype).encode() + str(z.shape).encode() + z.tobytes()).hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def git(*args, binary=False):
    r = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)
    out = r.stdout if binary else r.stdout.decode('utf-8', 'replace').strip()
    return r.returncode, out


def git_blob(commit, path):
    """Bytes of `path` (repository-relative) at `commit`, or None."""
    rc, out = git('cat-file', 'blob', f'{commit}:{path}', binary=True)
    return out if rc == 0 else None


def git_commit_utc(commit):
    rc, out = git('show', '-s', '--format=%cI', commit)
    return parse_utc(out) if rc == 0 and out else None


def is_ancestor(a, b='HEAD'):
    rc, _ = git('merge-base', '--is-ancestor', a, b)
    return rc == 0


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def load_json(p):
    return json.loads(Path(p).read_text())


def write_once_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'x') as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True, default=_json_default) + '\n')
    return sha_file(p)


# ---------------------------------------------------------------- numerical primitives (own definitions)
def household_totals(d, unit):
    """Per-household sums of d (households in sorted id order) and the household count G."""
    d = np.asarray(d, dtype=np.float64)
    unit = np.asarray(unit)
    order = np.argsort(unit, kind='stable')
    u = unit[order]
    starts = np.flatnonzero(np.concatenate(([True], u[1:] != u[:-1])))
    return np.add.reduceat(d[order], starts), len(starts)


def cluster_se(v, unit):
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    tot, G = household_totals(v - v.mean(), unit)
    if G < 2:
        return float('nan')
    return float(np.sqrt(G / (G - 1.0) * np.sum(tot * tot)) / n)


def mean_se(v, unit):
    v = np.asarray(v, dtype=np.float64)
    return float(v.mean()), cluster_se(v, unit)


def subgroup_mean_se(d, sub, unit):
    """Ratio-estimator subgroup mean of d over the boolean mask sub, with its linearised household-clustered SE."""
    d = np.asarray(d, dtype=np.float64)
    sub = np.asarray(sub, dtype=bool)
    est = float(d[sub].mean())
    p = float(sub.mean())
    z = np.where(sub, (d - est) / p, 0.0)
    return est, cluster_se(z, unit)


def auc(score, y):
    """Exact AUC: (#pairs pos > neg + 0.5 #ties) / (n_pos n_neg)."""
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    sp, sn = score[y], np.sort(score[~y])
    if len(sp) == 0 or len(sn) == 0:
        return float('nan')
    lo = np.searchsorted(sn, sp, side='left').astype(np.int64)
    hi = np.searchsorted(sn, sp, side='right').astype(np.int64)
    num = 2 * int(lo.sum()) + int((hi - lo).sum())          # twice the count, integer-exact
    return float(num / 2.0 / (float(len(sp)) * float(len(sn))))


class WeightedAUC:
    """AUC of FIXED scores under many person-weight vectors (bootstrap draws); the sort is done once."""

    def __init__(self, score, y):
        score = np.asarray(score, dtype=np.float64)
        y = np.asarray(y).astype(bool)
        self.pos = np.flatnonzero(y)
        neg = np.flatnonzero(~y)
        o = np.argsort(score[neg], kind='stable')
        self.neg_sorted = neg[o]
        sn = score[self.neg_sorted]
        sp = score[self.pos]
        self.lo = np.searchsorted(sn, sp, side='left')
        self.hi = np.searchsorted(sn, sp, side='right')

    def __call__(self, w):
        w = np.asarray(w, dtype=np.float64)
        wn = w[self.neg_sorted]
        c = np.concatenate(([0.0], np.cumsum(wn)))           # c[j] = weight of the first j sorted negatives
        below = c[self.lo]
        ties = c[self.hi] - c[self.lo]
        wp = w[self.pos]
        num = float(np.sum(wp * (2.0 * below + ties)))       # twice the weighted count
        den = float(wp.sum()) * float(c[-1])
        return num / 2.0 / den


def bootstrap_weights(unit, B, seed_entropy):
    """B x n person weights: per draw, rng.integers(0, G, G) over households in sorted id order (G = #households)."""
    unit = np.asarray(unit)
    groups = np.unique(unit)
    gi = np.searchsorted(groups, unit)
    G = len(groups)
    rng = np.random.default_rng(np.random.SeedSequence(list(seed_entropy)))
    W = np.empty((B, len(unit)), dtype=np.float64)
    for b in range(B):
        cnt = np.zeros(G, dtype=np.int64)
        np.add.at(cnt, rng.integers(0, G, G), 1)
        W[b] = cnt[gi]
    return W


# ---------------------------------------------------------------- brute-force references (selftest only)
def _naive_cluster_se(v, unit):
    v = np.asarray(v, dtype=np.float64)
    m = v.mean()
    groups = {}
    for x, u in zip(v, unit):
        groups[u] = groups.get(u, 0.0) + (x - m)
    G = len(groups)
    return (G / (G - 1) * sum(t * t for t in groups.values())) ** 0.5 / len(v)


def _naive_auc(score, y, w=None):
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    w = np.ones(len(score)) if w is None else np.asarray(w, dtype=np.float64)
    num = den = 0.0
    for i in np.flatnonzero(y):
        for j in np.flatnonzero(~y):
            ww = w[i] * w[j]
            den += ww
            num += ww * (1.0 if score[i] > score[j] else 0.5 if score[i] == score[j] else 0.0)
    return num / den


# ---------------------------------------------------------------- append-only CHECKS.json
class Checks:
    """Append-only check store: CHECKS.json holds entries chained by prev_sha256 (sha256 of the previous entry's canonical JSON);
    every entry's record file is re-hashed before every write; ids are unique (a rerun uses a new __V<n> suffix)."""

    def __init__(self, folder=None):
        self.folder = Path(folder) if folder is not None else VER
        self.path = self.folder / 'CHECKS.json'
        self.lock = self.folder / 'records' / '.checks.lock'

    def _read(self):
        if not self.path.is_file():
            return {'schema': 'FSA1-verifier-checks-v1', 'campaign': CAMPAIGN, 'checks': []}
        return json.loads(self.path.read_text())

    def verify_chain(self, doc=None):
        doc = doc or self._read()
        prev = None
        problems = []
        for i, e in enumerate(doc['checks']):
            if e.get('seq') != i:
                problems.append(f'entry {i}: seq {e.get("seq")}')
            if e.get('prev_sha256') != prev:
                problems.append(f'entry {i} ({e.get("id")}): prev_sha256 breaks the chain')
            rp = e.get('record')
            if rp:
                p = self.folder / rp
                if not p.is_file() or sha_file(p) != e.get('record_sha256'):
                    problems.append(f'entry {i} ({e.get("id")}): record {rp} missing or changed')
            prev = sha_bytes(canon(e).encode())
        return problems

    def ids(self):
        return {e['id'] for e in self._read()['checks']}

    def append(self, items):
        """items: list of dicts with id, block, kind, status, summary, detail and optional record (dict, written write-once)."""
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock, 'a') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                doc = self._read()
                problems = self.verify_chain(doc)
                if problems:
                    raise RuntimeError(f'CHECKS.json chain broken: {problems[:5]}')
                have = {e['id'] for e in doc['checks']}
                prev = sha_bytes(canon(doc['checks'][-1]).encode()) if doc['checks'] else None
                new = []
                for it in items:
                    if it['id'] in have:
                        raise PermissionError(f'check id {it["id"]} exists (append-only; use a new __V<n> suffix)')
                    if it['status'] not in STATUSES or it['kind'] not in KINDS:
                        raise ValueError(f'bad status/kind in {it["id"]}')
                    e = {k: it[k] for k in ('id', 'block', 'kind', 'status', 'summary')}
                    e['detail'] = it.get('detail', {})
                    e['utc'] = utc_now()
                    e['unit_id'] = os.environ.get('FSA_UNIT_ID')
                    e['script'] = it.get('script')
                    e['script_sha256'] = sha_file(ROOT / it['script']) if it.get('script') else None
                    if it.get('record') is not None:
                        rp = Path('records') / f'{it["id"]}.json'
                        e['record'] = str(rp)
                        e['record_sha256'] = write_once_json(self.folder / rp, it['record'])
                    else:
                        e['record'], e['record_sha256'] = None, None
                    e['seq'] = len(doc['checks']) + len(new)
                    e['prev_sha256'] = prev
                    prev = sha_bytes(canon(e).encode())
                    new.append(e)
                    have.add(it['id'])
                doc['checks'].extend(new)
                tmp = self.path.with_suffix('.json.tmp')
                tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=_json_default) + '\n')
                os.replace(tmp, self.path)
                return new
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)


def totals(doc):
    out = {k: {s: 0 for s in STATUSES} for k in KINDS}
    for e in doc['checks']:
        out[e['kind']][e['status']] += 1
    return out


# ---------------------------------------------------------------- selftest (invented data only)
def selftest():
    import tempfile
    rng = np.random.default_rng(7)
    res = []

    def ok(name, cond, info=''):
        res.append((name, bool(cond), info))

    # clustered SE against the brute-force definition; invariance to row order
    n = 400
    unit = rng.integers(0, 150, n) * 7919 + 13
    v = rng.normal(size=n) + (unit % 5)
    a, b = cluster_se(v, unit), _naive_cluster_se(v, unit)
    ok('cluster_se_vs_naive', abs(a - b) < 1e-14, f'{a} {b}')
    perm = rng.permutation(n)
    ok('cluster_se_order_invariant', abs(cluster_se(v[perm], unit[perm]) - a) < 1e-14)
    # ratio-estimator subgroup: the SE equals the clustered SE of the linearised scores (brute force)
    sub = rng.random(n) < 0.45
    est, se = subgroup_mean_se(v, sub, unit)
    z = [((x - v[sub].mean()) / sub.mean()) if s else 0.0 for x, s in zip(v, sub)]
    ok('subgroup_mean', abs(est - float(np.mean(v[sub]))) < 1e-14)
    ok('subgroup_se_vs_naive', abs(se - _naive_cluster_se(np.array(z), unit)) < 1e-13)
    # one-person households: clustered SE equals the iid SE with the n/(n-1) correction
    u1 = np.arange(n)
    iid = float(np.std(v, ddof=1) / np.sqrt(n))
    ok('cluster_se_singletons_iid', abs(cluster_se(v, u1) - iid) < 1e-13)
    # AUC with heavy ties (finite cells) against brute force, and weighted AUC
    y = rng.random(n) < 0.5
    s = np.round(rng.random(n) * 6) / 6 + 0.3 * y * (rng.random(n) < 0.3)
    ok('auc_vs_naive', abs(auc(s, y) - _naive_auc(s, y)) < 1e-14, f'{auc(s, y)} {_naive_auc(s, y)}')
    w = rng.integers(0, 4, n).astype(float)
    wa = WeightedAUC(s, y)
    ok('wauc_vs_naive', abs(wa(w) - _naive_auc(s, y, w)) < 1e-14, f'{wa(w)} {_naive_auc(s, y, w)}')
    ok('wauc_unit_weights_equal_auc', abs(wa(np.ones(n)) - auc(s, y)) < 1e-15)
    ok('auc_perfect_and_null', auc(np.r_[np.zeros(5), np.ones(5)], np.r_[np.zeros(5), np.ones(5)]) == 1.0
       and auc(np.ones(10), np.r_[np.zeros(5), np.ones(5)]) == 0.5)
    # bootstrap weights: sum to n per draw only in expectation; each draw's household counts sum to G; persons of a household share
    W = bootstrap_weights(unit, 5, [1, 2, 3])
    G = len(np.unique(unit))
    hh_ok = all(len(np.unique(W[b][unit == g])) == 1 for b in range(5) for g in np.unique(unit)[:20])
    ht, hg = household_totals(np.ones(n), unit)
    tot_ok = hg == G and len(ht) == G and float(ht.sum()) == float(n)
    counts_ok = True
    groups = np.unique(unit)
    first = np.array([np.flatnonzero(unit == g)[0] for g in groups])
    for b in range(5):
        counts_ok &= int(W[b][first].sum()) == G
    ok('bootstrap_household_constant', hh_ok and tot_ok)
    ok('bootstrap_counts_sum_to_G', counts_ok)
    ok('bootstrap_deterministic', np.array_equal(W, bootstrap_weights(unit, 5, [1, 2, 3])))
    # the draws follow B sequential rng.integers(0, G, G) calls on the sorted households
    rng2 = np.random.default_rng(np.random.SeedSequence([1, 2, 3]))
    d0 = np.bincount(rng2.integers(0, G, G), minlength=G)
    ok('bootstrap_draw_definition', np.array_equal(W[0][first], d0))
    # append-only store
    with tempfile.TemporaryDirectory() as td:
        st = Checks(td)
        st.append([{'id': 'T.A', 'block': 'T', 'kind': 'numerical', 'status': 'PASS', 'summary': 'a', 'record': {'x': 1}}])
        st.append([{'id': 'T.B', 'block': 'T', 'kind': 'procedural', 'status': 'ATTENTION', 'summary': 'b'}])
        ok('store_chain_ok', st.verify_chain() == [])
        try:
            st.append([{'id': 'T.A', 'block': 'T', 'kind': 'numerical', 'status': 'PASS', 'summary': 'dup'}])
            ok('store_refuses_duplicate', False)
        except PermissionError:
            ok('store_refuses_duplicate', True)
        doc = json.loads(st.path.read_text())
        doc['checks'][0]['status'] = 'FAIL'                  # tamper with an earlier entry
        st.path.write_text(json.dumps(doc))
        ok('store_detects_tamper', len(st.verify_chain()) > 0)
        try:
            st.append([{'id': 'T.C', 'block': 'T', 'kind': 'numerical', 'status': 'PASS', 'summary': 'c'}])
            ok('store_refuses_after_tamper', False)
        except RuntimeError:
            ok('store_refuses_after_tamper', True)
        st2 = Checks(Path(td) / 'b')
        st2.append([{'id': 'R.A', 'block': 'R', 'kind': 'numerical', 'status': 'PASS', 'summary': 'r', 'record': {'y': 2}}])
        (Path(td) / 'b' / 'records' / 'R.A.json').write_text('{"y": 3}\n')     # tamper with a record
        ok('store_detects_record_change', len(st2.verify_chain()) > 0)
        ok('totals', totals(json.loads((Path(td) / 'b' / 'CHECKS.json').read_text()))['numerical']['PASS'] == 1)
    n_ok = sum(r[1] for r in res)
    for name, good, info in res:
        print(('PASS ' if good else 'FAIL ') + name + (f'  [{info}]' if not good and info else ''))
    print(f'vcore selftest: {n_ok}/{len(res)}')
    return n_ok == len(res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
