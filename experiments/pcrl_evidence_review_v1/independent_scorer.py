"""Independent re-implementation of the 2017 transport study's scoring and uncertainty.

Written for Terminal B's evidence audit. It shares no computation function with
`scripts/report_acs_spectral_transport.py`:

* per-person log loss is recomputed from the stored prediction arrays with its own
  clip/renormalise/gather arithmetic;
* candidate selection is re-derived from the stored 2017 validation losses rather
  than read from `selected_scopes`;
* the household-cluster bootstrap is evaluated by household-level aggregation
  (counts @ group-sums) instead of materialising a replicate-by-row matrix, which
  is both algebraically equivalent and an order of magnitude smaller in memory;
* the studentised max-|t| adjustment is coded from the protocol text.

Everything is read-only with respect to the study directory. One worker, one
BLAS thread; arrays are loaded per interface and released.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

PROB_FLOOR = 1e-12
BOOT_SEED = 20260917
BOOT_REPLICATES = 2000

TARGETS = ('SEX', 'RAC1P', 'income_binary', 'civilian_at_work',
           'public_coverage', 'same_residence', 'commute_over20')
UTILITY_TASKS = ('income_binary', 'civilian_at_work', 'public_coverage',
                 'same_residence', 'commute_over20')
AUDIT_ROLES = {'A': ('public_coverage', 'commute_over20', 'SEX', 'RAC1P'),
               'B': ('income_binary', 'civilian_at_work', 'same_residence', 'SEX', 'RAC1P'),
               'AB': ('SEX', 'RAC1P')}
FORBIDDEN = tuple(f'{v}/{t}' for v, ts in AUDIT_ROLES.items() for t in ts)
INTERFACES = ('H', 'E', 'A0', 'L025', 'L20', 'J', 'spectral_S0', 'spectral_M025',
              'spectral_M1', 'spectral_L025', 'spectral_L1', 'spectral_C025',
              'spectral_C1', 'spectral_L2')
SCORE_FIELD = {'unweighted': 'test', 'person_weighted': 'test_person_weighted'}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- losses
def person_nll(y, probs):
    """Natural-log loss per person under the declared 1e-12 floor + renormalisation.

    Independent arithmetic: floor first, divide by the floored row sum, then take
    the negative log of the true-class entry via advanced indexing on a copy.
    """
    p = np.array(probs, dtype=np.float64, copy=True)
    np.clip(p, PROB_FLOOR, 1.0, out=p)
    p /= p.sum(axis=1, keepdims=True)
    idx = np.asarray(y, dtype=np.intp)
    return -np.log(p[np.arange(p.shape[0], dtype=np.intp), idx])


class PredictionStore:
    """Lazy reader for a unit's predictions.npz (key -> content-hash -> array)."""

    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path) as z:
            self._index = dict(zip(z['keys'].tolist(), z['ids'].tolist()))
        self._z = np.load(self.path)

    def has(self, key):
        return key in self._index

    def get(self, key):
        return self._z['p/' + self._index[key]]

    def close(self):
        self._z.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- selection
def reselect(rows, role, view, target, budget, scope):
    """Re-derive the scope's selected candidate from stored 2017 validation losses.

    The protocol's rule: minimum unweighted validation log loss, ties broken by
    candidate ID. Membership is taken from the candidate's own origin/space
    attributes, not from the stored `selected_scopes` list.
    """
    pool = [r for r in rows
            if r['role'] == role and r['view'] == view and r['target'] == target
            and r.get('audit_budget') == budget
            and not r.get('diagnostic_only', False)
            and in_scope(r, scope)
            and r.get('validation_log_loss') is not None
            and np.isfinite(r['validation_log_loss'])]
    if not pool:
        return None
    return min(pool, key=lambda r: (r['validation_log_loss'], r['candidate_id']))


# Mode B scope membership is a predicate on (candidate origin, released space),
# transcribed from COMPARISONS.json rather than imported from the study code.
MODE_B_SCOPES = {
    'common_fresh': ({'fresh'}, {'wire'}),
    'fresh_expanded': ({'fresh'}, {'wire', 'derived'}),
    'fresh_catchup': ({'fresh', 'catchup'}, {'wire', 'derived'}),
    'transport_all': ({'fresh', 'catchup', 'frozen2018'}, {'wire', 'derived'}),
}


def in_scope(row, scope):
    origins, spaces = MODE_B_SCOPES[scope]
    return row.get('transport_origin') in origins and row.get('space') in spaces


def stored_selection(rows, role, view, target, budget, scope):
    hits = [r for r in rows
            if r['role'] == role and r['view'] == view and r['target'] == target
            and r.get('audit_budget') == budget
            and scope in r.get('selected_scopes', [])]
    if len(hits) != 1:
        raise ValueError(f'expected 1 stored selection, got {len(hits)} for '
                         f'{role}/{view}/{target}/{budget}/{scope}')
    return hits[0]


# --------------------------------------------------------------------------- bootstrap
class HouseholdBootstrap:
    """Paired household-cluster bootstrap, evaluated by household aggregation.

    For a per-person vector v and person weights w, the replicate ratio estimate is
        sum_i c[g(i)] w_i v_i / sum_i c[g(i)] w_i
    which equals (C @ Gv) / (C @ Gw) with C the (replicates x households) resample
    count matrix and G the household-aggregation operator. Aggregating first keeps
    peak memory at O(replicates x households) rather than O(replicates x rows).
    """

    def __init__(self, serialno, weights, labels, replicates=BOOT_REPLICATES, seed=BOOT_SEED):
        order = sorted(set(map(str, serialno)))
        code = {h: i for i, h in enumerate(order)}
        self.group = np.fromiter((code[str(h)] for h in serialno), dtype=np.int64,
                                 count=len(serialno))
        self.n_groups = len(order)
        self.replicates = replicates
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, self.n_groups, size=(replicates, self.n_groups))
        counts = np.empty((replicates, self.n_groups), dtype=np.int32)
        for b in range(replicates):
            counts[b] = np.bincount(draws[b], minlength=self.n_groups)
        self.counts = counts
        self.weights = np.asarray(weights, dtype=np.float64)
        self.labels = labels
        self._denoms = {}

    def _valid(self, target):
        return self.labels[target] >= 0

    def _denominator(self, target, weight):
        key = (target, weight)
        if key not in self._denoms:
            valid = self._valid(target)
            w = self.weights[valid] if weight == 'person_weighted' else np.ones(int(valid.sum()))
            gw = np.bincount(self.group[valid], weights=w, minlength=self.n_groups)
            self._denoms[key] = (w, gw, self.counts @ gw)
        return self._denoms[key]

    def ratio(self, target, weight, vector):
        """Return (replicate estimates (B,), point estimate) for one per-person vector."""
        valid = self._valid(target)
        w, _, rep_denom = self._denominator(target, weight)
        v = np.asarray(vector, dtype=np.float64)
        if v.shape[0] != int(valid.sum()):
            raise ValueError(f'vector length {v.shape[0]} != valid rows {int(valid.sum())}')
        gv = np.bincount(self.group[valid], weights=w * v, minlength=self.n_groups)
        reps = (self.counts @ gv) / rep_denom
        point = float((w @ v) / w.sum())
        return reps, point


def max_t_intervals(replicates, estimates, level=0.95):
    """Single-step studentised max-|t| simultaneous intervals (protocol section 6)."""
    R = np.asarray(replicates, dtype=np.float64)      # (B, K)
    est = np.asarray(estimates, dtype=np.float64)     # (K,)
    se = R.std(axis=0, ddof=1)
    live = se > 1e-15
    if not live.any():
        return est.copy(), est.copy(), se, float('nan'), live
    t = np.abs(R[:, live] - est[live]) / se[live]
    crit = float(np.quantile(t.max(axis=1), level, method='higher'))
    low = np.where(live, est - crit * se, est)
    high = np.where(live, est + crit * se, est)
    return low, high, se, crit, live


def one_sided_upper(replicates, estimates, level=0.95, crit=None):
    """Upper confidence bound. With `crit` given, the simultaneous studentised bound
    at the family's critical value; otherwise the pointwise bootstrap percentile."""
    R = np.asarray(replicates, dtype=np.float64)
    est = np.asarray(estimates, dtype=np.float64)
    if crit is None:
        return np.quantile(R, level, axis=0, method='higher')
    se = R.std(axis=0, ddof=1)
    return est + crit * se


def one_sided_studentised_crit(replicates, estimates, level=0.95):
    """Critical value for a one-sided simultaneous upper bound: the `level` quantile
    of max_k (theta*_bk - theta_hat_k)/se_k (no absolute value)."""
    R = np.asarray(replicates, dtype=np.float64)
    est = np.asarray(estimates, dtype=np.float64)
    se = R.std(axis=0, ddof=1)
    live = se > 1e-15
    if not live.any():
        return float('nan')
    t = (R[:, live] - est[live]) / se[live]
    return float(np.quantile(t.max(axis=1), level, method='higher'))


# --------------------------------------------------------------------------- study access
class Study:
    """Read-only accessor for the locked study directory."""

    def __init__(self, root, seeds=(0, 1, 2)):
        self.root = Path(root)
        self.seeds = tuple(seeds)
        lab = np.load(self.root / 'final_labels.npz')
        self.labels = {t: lab[f'y/{t}'] for t in TARGETS}
        self.weights = lab['weights']
        self.serialno = lab['serialno']
        self.sporder = lab['sporder']
        self._metrics = {}

    def metrics(self, seed, condition, mode):
        key = (seed, condition, mode)
        if key not in self._metrics:
            path = self.root / f'seed_{seed}' / condition / f'mode_{mode}' / 'metrics.json'
            self._metrics[key] = read_json(path)['raw_metrics']
        return self._metrics[key]

    def predictions(self, seed, condition, mode):
        return PredictionStore(self.root / f'seed_{seed}' / condition / f'mode_{mode}' / 'predictions.npz')

    def context_rows(self, seed):
        return read_json(self.root / f'seed_{seed}' / 'context' / 'metrics.json')['rows']

    def context_predictions(self, seed):
        return PredictionStore(self.root / f'seed_{seed}' / 'context' / 'predictions.npz')

    def prior_loss(self, seed, mode, target):
        """Per-person loss of the fitted prior for this mode and target."""
        with self.context_predictions(seed) as store:
            arr = store.get(f'{mode}/prior/{target}')
        valid = self.labels[target] >= 0
        return person_nll(self.labels[target][valid], arr)

    def utility_loss(self, seed, mode, condition, task):
        rows = self.metrics(seed, condition, mode)
        hit = [r for r in rows if r['role'] == 'utility' and r['target'] == task and r['selected']]
        if len(hit) != 1:
            raise ValueError(f'utility selection ambiguous: {seed}/{condition}/{mode}/{task}')
        r = hit[0]
        with self.predictions(seed, condition, mode) as store:
            arr = store.get(f"utility/{r['view']}/{task}/{r['candidate_id']}")
        valid = self.labels[task] >= 0
        return person_nll(self.labels[task][valid], arr), r

    def attack_loss(self, seed, mode, condition, scope, budget, endpoint, use_stored=True):
        view, target = endpoint.split('/')
        rows = self.metrics(seed, condition, mode)
        r = (stored_selection(rows, 'audit', view, target, budget, scope) if use_stored
             else reselect(rows, 'audit', view, target, budget, scope))
        with self.predictions(seed, condition, mode) as store:
            arr = store.get(f"audit/{budget}/{view}/{target}/{r['candidate_id']}")
        valid = self.labels[target] >= 0
        return person_nll(self.labels[target][valid], arr), r


def csv_gz_write(path, header, rows):
    with gzip.open(path, 'wt', newline='') as fh:
        fh.write(','.join(header) + '\n')
        for row in rows:
            fh.write(','.join('' if v is None else str(v) for v in row) + '\n')
