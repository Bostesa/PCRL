"""RFA1 independent verifier library (pure functions; no data access, no campaign import).

Written by the verifier from the governing DEFINITIONS (rfa.evaluate -> ohn.evaluate.complete_eval -> ihs.evaluate.utility ->
scm_bench.selection.clauses / scm_bench.common.task_vectors; rfa.devreaders; task_conditioned_representation.attacks / .inference;
ffv.readers / ffv.features; scm_bench.readers), without calling them. Used by the later-wave entry scripts (which do the data access
through rfa.common.load(..., operation='verify') from their own __main__) and tested on invented data by v1_prepare_test.py.

(a) usefulness: task_vectors, clause values (12 per version), within-SEX guards (2 per version), household-clustered SEs, G1 (all 28).
(b) grid: household_folds, reader roles, reader seeds, wire / cache keys, cache-entry reader, winner selection (pick), Mann-Whitney
    AUC, household bootstrap weights and weighted AUC, paired per-cell differences and SEs.
(c) readers: feature tables (kinds, rich view, categorical-interaction design), FiniteScaler, the 16 locked recipes as sklearn
    estimators, LOOKUP, VALIDATION orientation, refit -> oriented scores.
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings

import numpy as np

# ---------------------------------------------------------------- (a) usefulness: the 28 original conditions
TASKS = ('income', 'occupation')
K = {'income': 2, 'occupation': 5}
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
GUARD = 0.005
REFS = ('U0', 'Ucal')
NLL_FLOOR = 1e-12


def check_prob(p, k, n=None):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != k or (n is not None and p.shape[0] != n) or not np.isfinite(p).all() or (p < 0).any() \
            or np.max(np.abs(p.sum(1) - 1.0)) > 1e-9:
        raise ValueError('not a valid probability-vector array')
    return p


def per_person(y, p, k):
    """accuracy (argmax == y, first index on ties), NLL (-log max(p_y, 1e-12)), summed Brier sum_k (p_k - 1[y = k])^2."""
    y = np.asarray(y).astype(np.int64)
    p = check_prob(p, k, len(y))
    pick = p[np.arange(len(y)), y]
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return {'accuracy': (np.argmax(p, axis=1) == y).astype(np.float64),
            'nll': -np.log(np.maximum(pick, NLL_FLOOR)),
            'brier': np.sum((p - onehot) ** 2, axis=1)}


def cluster_se(d, unit):
    """SE of mean(d), household clusters, G/(G-1) small-sample factor (sum of within-cluster centred sums)."""
    d = np.asarray(d, dtype=np.float64)
    _, inv = np.unique(np.asarray(unit), return_inverse=True)
    G = int(inv.max()) + 1
    cs = np.zeros(G)
    np.add.at(cs, inv, d - d.mean())
    return float(math.sqrt(G / (G - 1) * float(np.sum(cs * cs))) / len(d))


def subgroup_se(d, mask, unit):
    """Linearized household-clustered SE of the subgroup mean over mask (influence (d - m) / share inside the group, 0 outside)."""
    d = np.asarray(d, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    m = d[mask].mean()
    share = mask.mean()
    infl = np.where(mask, (d - m) / share, 0.0)
    _, inv = np.unique(np.asarray(unit), return_inverse=True)
    G = int(inv.max()) + 1
    cs = np.zeros(G)
    np.add.at(cs, inv, infl)
    return float(math.sqrt(G / (G - 1) * float(np.sum(cs * cs))) / len(d))


def _draws(rel):
    inc, occ = np.asarray(rel['income']), np.asarray(rel['occupation'])
    if inc.ndim == 2 and occ.ndim == 2:
        return [{'income': inc, 'occupation': occ}]
    R = inc.shape[0] if inc.ndim == 3 else occ.shape[0]
    return [{'income': inc[r] if inc.ndim == 3 else inc, 'occupation': occ[r] if occ.ndim == 3 else occ} for r in range(R)]


def usefulness_one_draw(labels, rel, refs, sex, unit):
    """{'clauses': {task|ref|metric: {'value', 'se', 'cap', 'pass'}}, 'guards': {sex0|sex1: {...}}} for one emitted draw."""
    own = {t: per_person(labels[t], rel[t], K[t]) for t in TASKS}
    out = {'clauses': {}, 'guards': {}}
    for t in TASKS:
        for r in REFS:
            rv = per_person(labels[t], refs[r][t], K[t])
            ds = {'accuracy_loss': rv['accuracy'] - own[t]['accuracy'], 'nll_excess': own[t]['nll'] - rv['nll'],
                  'brier_excess': own[t]['brier'] - rv['brier']}
            # the governing value is a DIFFERENCE OF MEANS (scm_bench.selection.clauses); the paired vector gives the clustered SE
            vals = {'accuracy_loss': float(rv['accuracy'].mean() - own[t]['accuracy'].mean()),
                    'nll_excess': float(own[t]['nll'].mean() - rv['nll'].mean()),
                    'brier_excess': float(own[t]['brier'].mean() - rv['brier'].mean())}
            for m, d in ds.items():
                v = vals[m]
                out['clauses'][f'{t}|{r}|{m}'] = {'value': v, 'se': cluster_se(d, unit), 'cap': CAPS[m], 'pass': bool(v <= CAPS[m]),
                                                  'paired_mean_minus_value': float(d.mean()) - v}
    u0 = per_person(labels['occupation'], refs['U0']['occupation'], 5)
    d = own['occupation']['brier'] - u0['brier']
    sex = np.asarray(sex)
    for g in (0, 1):
        mask = sex == g
        v = float(d[mask].mean())
        out['guards'][f'sex{g}'] = {'value': v, 'se': subgroup_se(d, mask, unit), 'cap': GUARD, 'pass': bool(v <= GUARD)}
    return out


def usefulness_version(labels, rel, refs, sex, unit):
    """Draw-averaged values (a deterministic release is one draw); pass = the draw-averaged value <= cap; se = mean per-draw SE."""
    evs = [usefulness_one_draw(labels, d, refs, sex, unit) for d in _draws(rel)]
    out = {'draws': len(evs), 'clauses': {}, 'guards': {}}
    for part in ('clauses', 'guards'):
        for key in evs[0][part]:
            vals = np.array([e[part][key]['value'] for e in evs])
            v = float(vals.mean())
            cap = evs[0][part][key]['cap']
            out[part][key] = {'value': v, 'se': float(np.mean([e[part][key]['se'] for e in evs])), 'cap': cap, 'pass': bool(v <= cap),
                              'se_mc': float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0}
            if part == 'clauses':
                out[part][key]['paired_mean_minus_value'] = float(max(abs(e[part][key]['paired_mean_minus_value']) for e in evs))
    out['n_conditions'] = len(out['clauses']) + len(out['guards'])
    out['all_pass'] = all(c['pass'] for c in out['clauses'].values()) and all(g['pass'] for g in out['guards'].values())
    out['failed'] = [k for k, c in out['clauses'].items() if not c['pass']] + [f'guard|{k}' for k, g in out['guards'].items() if not g['pass']]
    return out


def g1(per_version):
    """G1 = all 28 conditions (2 versions x (12 clauses + 2 guards)) at the point."""
    n = sum(pv['n_conditions'] for pv in per_version.values())
    return {'n_conditions': n, 'G1': bool(n == 28 and all(pv['all_pass'] for pv in per_version.values())),
            'failed': {str(v): pv['failed'] for v, pv in per_version.items()}}


def compare_with_campaign_utility(mine, u, tol=1e-12):
    """mine: usefulness_version output; u: a campaign utility record ('clauses' list with task / reference / metric / value / cap,
    'guards' list with group / value / cap). Returns max abs differences and pass-flag agreement."""
    diffs, flags = [], []
    for c in u['clauses']:
        key = f"{c['task']}|{c['reference']}|{c['metric']}"
        diffs.append(abs(float(c['value']) - mine['clauses'][key]['value']))
        if 'se' in c:
            diffs.append(abs(float(c['se']) - mine['clauses'][key]['se']))
        flags.append((float(c['value']) <= float(c['cap'])) == mine['clauses'][key]['pass'])
    for g in u['guards']:
        diffs.append(abs(float(g['value']) - mine['guards'][g['group']]['value']))
        if 'se' in g:
            diffs.append(abs(float(g['se']) - mine['guards'][g['group']]['se']))
        flags.append((float(g['value']) <= float(g['cap'])) == mine['guards'][g['group']]['pass'])
    return {'max_abs_diff': float(max(diffs)), 'flags_agree': bool(all(flags)), 'n_compared': len(flags),
            'agree': bool(max(diffs) <= tol and all(flags) and len(flags) == 14)}


# ---------------------------------------------------------------- (b) the development reader grid
MODEL_SEED_BASE = 2026101205
SEED_BASE_OFFSET = 1200
GRID_VERSION = 'RFA1-devgrid-v1'
REGISTERED_REPS = ('R1', 'R2')
REP_TAG = {r: f'RFA1_devgrid_{r}' for r in ('R1', 'R2', 'R3', 'R4')}
REP_INDEX = {'R1': 0, 'R2': 1, 'R3': 2, 'R4': 3}
LC_TAG = 'RFA1_devgrid_LC'
B_BOOT = 400
BOOT_SEED = 2026102520
NEAR_TIE = 0.010
SLATE = (('OUT_LR', 'out'), ('OUT_MLP', 'out'), ('OUT_HGB', 'out'), ('LOG_LR', 'log'), ('LOG_MLP', 'log'), ('DEC_LR', 'dec'),
         ('CONF_HGB', 'conf'))
SRC8 = tuple(n for n, _ in SLATE) + ('LOOKUP',)
NEW8 = ('ET5', 'ET25', 'HGB15', 'HGB31', 'MLP64', 'MLP6432', 'CATI01', 'CATI1')
RECIPES = SRC8 + NEW8
VIEWS = ('income', 'occupation', 'pair')


def household_folds(unit, k, tag):
    """Households ordered by SHA-256('<tag>|<household>') and dealt round-robin into k parts; per-person part."""
    unit = np.asarray(unit)
    groups = np.unique(unit)
    keys = np.array([hashlib.sha256(f'{tag}|{g}'.encode()).hexdigest() for g in groups])
    rank = np.empty(len(groups), dtype=np.int64)
    rank[np.argsort(keys, kind='stable')] = np.arange(len(groups))
    part_of_group = rank % k
    return part_of_group[np.searchsorted(groups, unit)]


def reader_roles(unit, rep, frac=1.0):
    part = household_folds(unit, 10, REP_TAG[rep])
    fit = part <= 4
    if float(frac) != 1.0:
        lc = household_folds(unit, 4, LC_TAG)
        fit = fit & ((lc == 0) if float(frac) == 0.25 else (lc <= 1))
    return {'FIT': fit, 'VALIDATION': (part >= 5) & (part <= 6), 'SCORE': part >= 7}


def reader_seed(version, k):
    return MODEL_SEED_BASE + SEED_BASE_OFFSET + 10 * int(version) + int(k)


def cell_name(version, rep, k, frac=1.0):
    return f'S{version}{rep}k{k}' + ('' if float(frac) == 1.0 else f'@{float(frac)}')


def arr_sha(a):
    z = np.ascontiguousarray(a)
    return hashlib.sha256(str(z.dtype).encode() + str(z.shape).encode() + z.tobytes()).hexdigest()


def wire_sha(p):
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    h = hashlib.sha256(p.tobytes())
    h.update(np.ascontiguousarray(p.argmax(1).astype(np.int64)).tobytes())
    return h.hexdigest()


def view_hash(view, inc, occ):
    h = hashlib.sha256(view.encode())
    for a in ((inc,) if view == 'income' else (occ,) if view == 'occupation' else (inc, occ)):
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


def cache_key(rep, rs, frac, vh, sex_sha):
    tag = rep if rep in REGISTERED_REPS else f'{rep}:{REP_TAG[rep]}'
    return hashlib.sha256(f'{GRID_VERSION}|{tag}|{rs}|{float(frac)}|{vh}|{sex_sha}'.encode()).hexdigest()


def read_cache_entry(path):
    """{recipe: (oriented SCORE scores, info)} from one cache npz (no pickle)."""
    with np.load(path, allow_pickle=False) as z:
        info = json.loads(str(z['info']))
        return {n: (z[f'score__{n}'].astype(np.float64).copy(), info[n]) for n in RECIPES}


def pick(cands):
    """cands: [(name, val_auc, reader_key, origin)]: maximum VALIDATION AUC, exact ties to the earlier candidate; gap and near tie."""
    best = 0
    for i in range(1, len(cands)):
        if cands[i][1] > cands[best][1]:
            best = i
    others = [c[1] for i, c in enumerate(cands) if i != best]
    gap = float(cands[best][1] - max(others)) if others else None
    return {'winner': cands[best][0], 'reader_key': cands[best][2], 'origin': cands[best][3], 'val_auc': float(cands[best][1]),
            'gap_to_runner_up': gap, 'near_tie': bool(gap is not None and gap < NEAR_TIE)}


def winners(info):
    """info: {view: {recipe: {'val_auc': ...}}} -> {view: pick}; pair candidates: per group (source 8, new 8) the pair readers, the
    ignore-occupation (income-view) readers, the ignore-income (occupation-view) readers."""
    w = {}
    for view in ('income', 'occupation'):
        if view in info:
            w[view] = pick([(n, info[view][n]['val_auc'], f'{view}__{n}', 'source' if n in SRC8 else 'new') for n in RECIPES])
    if 'pair' in info:
        c = []
        for group, org in ((SRC8, 'source'), (NEW8, 'new')):
            c += [(n, info['pair'][n]['val_auc'], f'pair__{n}', org) for n in group]
            c += [(f'IGNORE_occupation__{n}', info['income'][n]['val_auc'], f'income__{n}', org) for n in group]
            c += [(f'IGNORE_income__{n}', info['occupation'][n]['val_auc'], f'occupation__{n}', org) for n in group]
        w['pair'] = pick(c)
    return w


def auc(y, s):
    """Mann-Whitney AUC with average ranks for ties (positive class = 1)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def boot_weights(unit_sc, version, rep, B=B_BOOT):
    """Household bootstrap of SCORE: B draws of G households with replacement (the registered seed); per-person multiplicities."""
    g, inv = np.unique(np.asarray(unit_sc), return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED + 100 * int(version) + REP_INDEX[rep])
    W = np.empty((B, len(inv)))
    for b in range(B):
        draw = rng.integers(0, len(g), len(g))
        cnt = np.zeros(len(g))
        np.add.at(cnt, draw, 1.0)
        W[b] = cnt[inv]
    return W


def weighted_auc(y, s, W):
    """AUC under nonnegative weights per row (each row of W): sum over (pos, neg) pairs of w_p w_n (1[s_p > s_n] + 1/2 1[=]) / norm."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    W = np.atleast_2d(np.asarray(W, dtype=np.float64))
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
    Wp = W[:, order] * y[order][None, :]
    Wn = W[:, order] * (~y[order])[None, :]
    P = np.add.reduceat(Wp, starts, axis=1)
    N = np.add.reduceat(Wn, starts, axis=1)
    below = np.cumsum(N, axis=1) - N
    return (P * (below + 0.5 * N)).sum(1) / (P.sum(1) * N.sum(1))


def paired_cell(view, cand_score, comp_score, cand_boot, comp_boot):
    """Pair: comparator - candidate (positive favours the candidate); single views: candidate - comparator. SE = SD of the paired
    bootstrap differences (ddof 1)."""
    if view == 'pair':
        d, db = comp_score - cand_score, comp_boot - cand_boot
    else:
        d, db = cand_score - comp_score, cand_boot - comp_boot
    return float(d), float(np.std(db, ddof=1))


# ---------------------------------------------------------------- (c) readers: features, estimators, refit
CLIP = 1e-300


def kinds(p, k):
    p = np.asarray(p, dtype=np.float64)
    d = p.argmax(1)
    one = np.eye(k)[d]
    return {'out': np.column_stack((p, d.astype(np.float64))), 'log': np.column_stack((np.log(np.maximum(p, CLIP)), one)),
            'dec': one, 'conf': p.max(1, keepdims=True)}


def src_table(view, inc, occ, kind):
    a, b = kinds(inc, 2)[kind], kinds(occ, 5)[kind]
    return a if view == 'income' else b if view == 'occupation' else np.column_stack((a, b))


def _lg(p):
    return np.log(np.maximum(np.asarray(p, dtype=np.float64), CLIP))


def rich1(p):
    p = np.asarray(p, dtype=np.float64)
    return np.column_stack((p, _lg(p), np.eye(p.shape[1])[p.argmax(1)]))


def rich_view(view, inc, occ):
    return rich1(inc) if view == 'income' else rich1(occ) if view == 'occupation' else np.column_stack((rich1(inc), rich1(occ)))


def _spline(x, n_knots):
    from sklearn.preprocessing import SplineTransformer
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    kn = np.unique(np.quantile(x[:, 0], np.linspace(0, 1, n_knots)))
    if len(kn) < 2:
        return None
    return SplineTransformer(knots=kn[:, None], degree=3, extrapolation='constant', include_bias=True).fit(x)


def _occ9(p):
    p = np.asarray(p, dtype=np.float64)
    lp = _lg(p)
    return np.column_stack((p, lp[:, 1:] - lp[:, [0]]))


class VIncome:
    """<= 32 distinct exact income vectors on FIT: one indicator each (ordered by P(income = 1), then bytes; unseen -> zeros);
    otherwise 6 FIT quantile groups of P(income = 1) (side 'left') plus continuous p and log p."""

    def fit(self, p):
        p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
        u = np.unique(p, axis=0)
        if len(u) <= 32:
            self.cat = True
            order = sorted(range(len(u)), key=lambda i: (float(u[i, 1]), u[i].tobytes()))
            self.keys = {u[i].tobytes(): j for j, i in enumerate(order)}
        else:
            self.cat = False
            self.cuts = np.quantile(p[:, 1], np.arange(1, 6) / 6)
        return self

    def ind(self, p):
        p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
        if self.cat:
            M = np.zeros((len(p), len(self.keys)))
            for i in range(len(p)):
                j = self.keys.get(p[i].tobytes())
                if j is not None:
                    M[i, j] = 1.0
            return M
        return np.eye(6)[np.searchsorted(self.cuts, p[:, 1], side='left')]

    def cont(self, p):
        if self.cat:
            return np.zeros((len(p), 0))
        p = np.asarray(p, dtype=np.float64)
        return np.column_stack((p, _lg(p)))


class VCati:
    def __init__(self, view):
        self.view = view

    def fit(self, inc, occ):
        if self.view in ('income', 'pair'):
            self.inc = VIncome().fit(inc)
            if self.view == 'income' and not self.inc.cat:
                self.sp_inc = _spline(self._logit(inc), 6)
        if self.view == 'occupation':
            lc = _occ9(occ)[:, 5:]
            self.sp_occ = [_spline(lc[:, j], 5) for j in range(4)]
        return self

    @staticmethod
    def _logit(inc):
        p = np.asarray(inc, dtype=np.float64)
        return (_lg(p[:, 1]) - _lg(p[:, 0])).reshape(-1, 1)

    def transform(self, inc, occ):
        if self.view == 'pair':
            M, O = self.inc.ind(inc), _occ9(occ)
            return np.column_stack((M, self.inc.cont(inc), O, (M[:, :, None] * O[:, None, :]).reshape(len(M), -1)))
        if self.view == 'income':
            M = self.inc.ind(inc)
            if self.inc.cat:
                return M
            blocks = [M, self.inc.cont(inc)]
            if self.sp_inc is not None:
                blocks.append(self.sp_inc.transform(self._logit(inc)))
            return np.column_stack(blocks)
        O = _occ9(occ)
        lc = O[:, 5:]
        spl = [self.sp_occ[j].transform(lc[:, [j]]) for j in range(4) if self.sp_occ[j] is not None]
        prods = [lc[:, i] * lc[:, j] for i in range(4) for j in range(i + 1, 4)]
        return np.column_stack([O] + spl + [np.column_stack(prods), np.eye(5)[np.asarray(occ).argmax(1)]])


class Scaler:
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.mu = x.mean(0)
        sd = x.std(0)
        self.sd = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=np.float64) - self.mu) / self.sd


def estimator(name, rs):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    if name in ('OUT_MLP', 'LOG_MLP'):
        return MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=1e-4, max_iter=300,
                             early_stopping=False, random_state=rs)
    if name in ('OUT_HGB', 'CONF_HGB'):
        return HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, early_stopping=False, random_state=rs)
    if name in ('OUT_LR', 'LOG_LR', 'DEC_LR'):
        return LogisticRegression(C=1, max_iter=500, solver='lbfgs')
    if name in ('ET5', 'ET25'):
        return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5 if name == 'ET5' else 25, max_features=1.0, n_jobs=1,
                                    random_state=rs)
    if name in ('HGB15', 'HGB31'):
        return HistGradientBoostingClassifier(max_leaf_nodes=15 if name == 'HGB15' else 31, max_iter=200, learning_rate=0.05,
                                              min_samples_leaf=30, l2_regularization=1.0, early_stopping=False, random_state=rs)
    if name in ('MLP64', 'MLP6432'):
        return MLPClassifier(hidden_layer_sizes=(64,) if name == 'MLP64' else (64, 32), alpha=0.01, batch_size=256, max_iter=300,
                             early_stopping=False, random_state=rs)
    if name in ('CATI01', 'CATI1'):
        return LogisticRegression(C=0.1 if name == 'CATI01' else 1.0, solver='lbfgs', max_iter=5000)
    raise ValueError(name)


def lookup_scores(view, inc, occ, sex, fit, pseudo=2.0):
    arrs = (inc,) if view == 'income' else (occ,) if view == 'occupation' else (inc, occ)
    z = np.ascontiguousarray(np.column_stack([np.asarray(a, dtype=np.float64) for a in arrs]))
    keys = [r.tobytes() for r in z]
    y = np.asarray(sex)
    pi = float(y[fit].mean())
    tab = {}
    for i in np.flatnonzero(fit):
        n1, n = tab.get(keys[i], (0, 0))
        tab[keys[i]] = (n1 + int(y[i]), n + 1)
    return np.array([(tab[k][0] + pseudo * pi) / (tab[k][1] + pseudo) if k in tab else pi for k in keys], dtype=np.float64)


def features(name, view, inc, occ, fit):
    kind = dict(SLATE).get(name)
    if kind is not None:
        return src_table(view, inc, occ, kind)
    if name.startswith('CATI'):
        d = VCati(view).fit(np.asarray(inc)[fit], np.asarray(occ)[fit])
        return d.transform(inc, occ)
    return rich_view(view, inc, occ)


def refit(name, view, inc, occ, sex, fit, val, rs):
    """Fit one recipe on FIT, orient on VALIDATION (sign + if VALIDATION AUC >= 1/2, else 1 - score), return (oriented scores on all
    rows, VALIDATION AUC of the oriented score, raw VALIDATION AUC)."""
    sex = np.asarray(sex)
    if name == 'LOOKUP':
        raw = lookup_scores(view, inc, occ, sex, fit)
    else:
        X = features(name, view, inc, occ, fit)
        sc = Scaler().fit(X[fit])
        m = estimator(name, rs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m.fit(sc.transform(X[fit]), sex[fit])
        if list(m.classes_) != [0, 1]:
            raise ValueError('class order differs')
        raw = m.predict_proba(sc.transform(X))[:, 1].astype(np.float64)
    a_raw = auc(sex[val], raw[val])
    s = raw if a_raw >= 0.5 else 1.0 - raw
    return s, auc(sex[val], s[val]), a_raw
