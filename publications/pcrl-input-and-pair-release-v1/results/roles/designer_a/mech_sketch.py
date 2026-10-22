"""Designer A, campaign pcrl_input_and_pair_release_v1: SFG, shared SEX-aware field generalization (reference sketch).

Both task models see only h_k(x): the six fields coarsened by the first k steps of a ladder built from a FIXED menu of 12
label-free generalization steps (two coarser levels per field). SEX (MODEL_FIT training label) is used only to ORDER the menu:
steps are taken in decreasing order of cross-fitted SEX information removed per nat of task log-loss spent. The released pair is
a function of h_k(x), so no reader of either view or of the linked pair can exceed what h_k(X) carries about SEX.

  fit(Z, y_inc, y_occ, sex, unit, version, strength, order='sex' | 'blind' | 'attr', attr=None, T=None) -> state
  release(state, Z) -> (P_inc (n, 2), P_occ (n, 5))          # never reads SEX
  select_strength(...)                                         # MODEL_FIT out-of-fold budget rule (estimate + 2 household SE)
  selftest()                                                   # invented data only

Controls with the same code: order='blind' (SEX-blind generalization: cheapest task spend first, no target), order='attr' with
attr = PSEUDO2 (same operation, SEX-independent attribute). Z is the compact coding ohn.fam_a.compact_po (AGEP, WKHP standardized;
COW, MAR category indices; SCHL ordinal; POBP numeric code). Plain numpy + sklearn; no repository imports.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

CAT = np.array([False, False, True, False, True, False])          # COW, MAR native categories (as nrm.fa.prc.CAT_Z)
COL = {'AGEP': 0, 'WKHP': 1, 'COW': 2, 'SCHL': 3, 'MAR': 4, 'POBP': 5}
MENU = {'AGEP': ('AGE5', 'AGE10'), 'WKHP': ('WK6', 'WK3'), 'COW': ('COW5', 'COW3'), 'SCHL': ('SCH5', 'SCH3'),
        'MAR': ('MAR3', 'MAR1'), 'POBP': ('POB4', 'POB1')}
FIELD = {s: f for f, lv in MENU.items() for s in lv}
HGB = {'lr': 0.025, 'leaves': 16, 'msl': 100, 'l2': 3.0}           # A1_PO_16 learner settings
T_DEFAULT = {'inc': 332, 'occ': 168, 'sex': 200}                     # campaign: inner-CV T* rule of nrm.fa.prc.tstar on h_k(Z)
BUDGET = {'nll_inc': 0.012, 'nll_occ': 0.012, 'acc_inc': 0.008, 'acc_occ': 0.008, 'brier_men': 0.0045, 'brier_women': 0.0045}
GRID = (2, 3, 4)


# ---------------------------------------------------------------- the menu (label-free, deterministic from Z)
def field_maps(Z):
    """Age rank (40 levels = ages 25..64) and hours from the standardized WKHP (modal value = 40 hours, minimum = 1 hour)."""
    age_lv = np.unique(Z[:, 0])
    wk_lv, wk_ct = np.unique(Z[:, 1], return_counts=True)
    z40, zmin = float(wk_lv[np.argmax(wk_ct)]), float(wk_lv.min())
    return {'age_levels': age_lv, 'z40': z40, 'zmin': zmin}


def _hours(Z, m):
    return np.round(1 + (Z[:, 1] - m['zmin']) * 39.0 / (m['z40'] - m['zmin']))


def apply_step(Zh, Z, st, m):
    c = Z[:, COL[FIELD[st]]]
    if st in ('AGE5', 'AGE10'):
        idx = np.clip(np.searchsorted(m['age_levels'], c), 0, len(m['age_levels']) - 1)
        Zh[:, 0] = idx // (5 if st == 'AGE5' else 10)
    elif st == 'WK6':
        Zh[:, 1] = np.digitize(_hours(Z, m), [20, 35, 40, 41, 50])
    elif st == 'WK3':
        Zh[:, 1] = np.digitize(_hours(Z, m), [35, 45])
    elif st == 'SCH5':
        Zh[:, 3] = np.digitize(c, [16, 18, 21, 22])
    elif st == 'SCH3':
        Zh[:, 3] = np.digitize(c, [18, 21])
    elif st == 'COW5':                         # for-profit | nonprofit | government (local, state, federal) | self-employed | unpaid
        Zh[:, 2] = np.array([0, 1, 2, 2, 2, 3, 3, 4, 0])[np.clip(c.astype(int), 0, 8)]
    elif st == 'COW3':                         # private (incl. unpaid) | government | self-employed
        Zh[:, 2] = np.array([0, 0, 1, 1, 1, 2, 2, 0, 0])[np.clip(c.astype(int), 0, 8)]
    elif st == 'MAR3':                         # married | widowed, divorced, separated | never married
        Zh[:, 4] = np.array([0, 1, 1, 1, 2, 0])[np.clip(c.astype(int), 0, 5)]
    elif st == 'MAR1':
        Zh[:, 4] = 0
    elif st == 'POB4':                         # Florida | other US state | territories | foreign or other
        Zh[:, 5] = np.where(c == 12, 0, np.where(c < 60, 1, np.where(c < 100, 2, 3)))
    elif st == 'POB1':
        Zh[:, 5] = 0
    else:
        raise KeyError(st)
    return Zh


def generalize(Z, steps, m):
    Zh = np.array(Z, dtype=np.float64, copy=True)
    for st in steps:
        apply_step(Zh, Z, st, m)
    return Zh


def compose(ladder, k):
    """The first k ladder steps; a field keeps its last (coarsest) step."""
    cur = {}
    for st in ladder[:k]:
        cur[FIELD[st]] = st
    return [cur[f] for f in MENU if f in cur]


# ---------------------------------------------------------------- learners and folds
def household_folds(unit, k, seed):
    g = np.unique(unit)
    perm = np.random.default_rng(seed).permutation(len(g))
    fold = np.empty(len(g), dtype=np.int64); fold[perm] = np.arange(len(g)) % k
    return fold[np.searchsorted(g, unit)]


def _hgb(T, seed):
    return HistGradientBoostingClassifier(loss='log_loss', learning_rate=HGB['lr'], max_iter=int(T), max_leaf_nodes=HGB['leaves'],
                                          min_samples_leaf=HGB['msl'], l2_regularization=HGB['l2'], early_stopping=False,
                                          categorical_features=CAT, random_state=int(seed))


def _proba(m, X, K):
    p = np.zeros((len(X), K)); p[:, m.classes_] = m.predict_proba(X)
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum(1, keepdims=True)


def oof(Zh, y, K, folds, T, seed):
    P = np.zeros((len(y), K))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        P[te] = _proba(_hgb(T, seed).fit(Zh[tr], y[tr]), Zh[te], K)
    return P


def nll_rows(P, y):
    return -np.log(P[np.arange(len(y)), y])


def cluster_se(v, unit):
    v = np.asarray(v, dtype=np.float64); n = len(v)
    _, inv = np.unique(unit, return_inverse=True); G = inv.max() + 1
    s = np.bincount(inv, weights=v - v.mean(), minlength=G)
    return float(np.sqrt(G / (G - 1) * np.sum(s ** 2)) / n)


# ---------------------------------------------------------------- the ladder (MODEL_FIT only)
def step_stats(Z, y_inc, y_occ, target, unit, version, T, m, n_folds=5):
    """Cross-fitted single-step effects: task log-loss spent (dT, nats, both tasks) and target information removed (dI, nats)."""
    folds = household_folds(unit, n_folds, 7919 + int(version))
    base = {'inc': nll_rows(oof(Z, y_inc, 2, folds, T['inc'], version), y_inc).mean(),
            'occ': nll_rows(oof(Z, y_occ, 5, folds, T['occ'], version), y_occ).mean()}
    if target is not None:
        base['tgt'] = nll_rows(oof(Z, target, 2, folds, T['sex'], version), target).mean()
    out = {}
    for st in FIELD:
        Zh = generalize(Z, [st], m)
        dT = (nll_rows(oof(Zh, y_inc, 2, folds, T['inc'], version), y_inc).mean() - base['inc']
              + nll_rows(oof(Zh, y_occ, 5, folds, T['occ'], version), y_occ).mean() - base['occ'])
        dI = (nll_rows(oof(Zh, target, 2, folds, T['sex'], version), target).mean() - base['tgt']) if target is not None else 0.0
        out[st] = {'dT': float(dT), 'dI': float(dI)}
    return out


def build_ladder(stats, order):
    """Greedy: each field moves only to a coarser level; incremental ratio vs the field's current level.
    order 'sex' / 'attr': largest dI / dT first; 'blind': smallest incremental dT first (no target)."""
    cur = {f: None for f in MENU}; ladder = []
    while True:
        cand = []
        for f, lv in MENU.items():
            i0 = -1 if cur[f] is None else lv.index(cur[f])
            T0 = 0.0 if cur[f] is None else stats[cur[f]]['dT']
            I0 = 0.0 if cur[f] is None else stats[cur[f]]['dI']
            for st in lv[i0 + 1:]:
                dT = max(stats[st]['dT'] - T0, 1e-5); dI = stats[st]['dI'] - I0
                key = -dT if order == 'blind' else dI / dT
                cand.append((key, -dT, st))
        if not cand:
            return ladder
        st = max(cand)[2]
        cur[FIELD[st]] = st; ladder.append(st)


# ---------------------------------------------------------------- fit / release
def fit(Z, y_inc, y_occ, sex, unit, version, strength, order='sex', attr=None, T=None, ladder=None, **kw):
    T = dict(T or T_DEFAULT)
    m = field_maps(Z)
    if ladder is None:
        target = {'sex': sex, 'attr': attr, 'blind': None}[order]
        if order != 'blind' and target is None:
            raise ValueError('order needs its target')
        stats = step_stats(Z, y_inc, y_occ, None if target is None else np.asarray(target, dtype=np.int64), unit, version, T, m,
                           kw.get('n_folds', 5))
        ladder = build_ladder(stats, order)
    else:
        stats = None
    steps = compose(ladder, int(strength))
    Zh = generalize(Z, steps, m)
    models = {'inc': _hgb(T['inc'], version).fit(Zh, y_inc), 'occ': _hgb(T['occ'], version).fit(Zh, y_occ)}
    return {'maps': m, 'ladder': list(ladder), 'steps': steps, 'strength': int(strength), 'order': order, 'models': models,
            'T': T, 'stats': stats, 'version': int(version)}


def release(state, Z):
    Zh = generalize(Z, state['steps'], state['maps'])
    return _proba(state['models']['inc'], Zh, 2), _proba(state['models']['occ'], Zh, 5)


def spend_clauses(Z, y_inc, y_occ, sex, unit, version, steps, m, T, n_folds=5):
    """MODEL_FIT out-of-fold spend of h = steps over the twin (h = identity), each as (estimate, household-clustered SE)."""
    folds = household_folds(unit, n_folds, 104729 + int(version))
    Zh = generalize(Z, steps, m)
    tw = {'inc': oof(Z, y_inc, 2, folds, T['inc'], version), 'occ': oof(Z, y_occ, 5, folds, T['occ'], version)}
    ca = {'inc': oof(Zh, y_inc, 2, folds, T['inc'], version), 'occ': oof(Zh, y_occ, 5, folds, T['occ'], version)}
    out = {}
    for t, y in (('inc', y_inc), ('occ', y_occ)):
        d = nll_rows(ca[t], y) - nll_rows(tw[t], y); out['nll_' + t] = (float(d.mean()), cluster_se(d, unit))
        a = (tw[t].argmax(1) == y).astype(float) - (ca[t].argmax(1) == y).astype(float)
        out['acc_' + t] = (float(a.mean()), cluster_se(a, unit))
    oh = np.eye(5)[y_occ]
    b = ((ca['occ'] - oh) ** 2).sum(1) - ((tw['occ'] - oh) ** 2).sum(1)
    for nm, s in (('brier_men', 0), ('brier_women', 1)):
        w = sex == s
        out[nm] = (float(b[w].mean()), cluster_se(b[w], unit[w]))
    return out


def select_strength(Z, y_inc, y_occ, sex, unit, version, ladder, grid=GRID, budget=BUDGET, T=None, **kw):
    """k* = the largest k in the grid whose every clause meets estimate + 2 SE <= budget (MODEL_FIT only); None if none."""
    T = dict(T or T_DEFAULT); m = field_maps(Z); best, rec = None, {}
    for k in sorted(grid):
        cl = spend_clauses(Z, y_inc, y_occ, sex, unit, version, compose(ladder, k), m, T, kw.get('n_folds', 5))
        ok = all(e + 2 * s <= budget[c] for c, (e, s) in cl.items())
        rec[k] = {'clauses': cl, 'pass': ok}
        if ok:
            best = k
    return best, rec


# ---------------------------------------------------------------- invented-data self-test
def _invent(n, seed):
    rng = np.random.default_rng(seed)
    sex = (rng.random(n) < 0.49).astype(np.int64)
    age = rng.integers(25, 65, n)
    hours = np.where(rng.random(n) < 0.55, 40, np.clip(np.round(rng.normal(42 - 6 * sex, 9)), 1, 99))
    cow = np.where(rng.random(n) < 0.65, 0, rng.integers(1, 8, n))
    cow = np.where((sex == 1) & (rng.random(n) < 0.15), 1, cow)
    schl = np.clip(np.round(rng.normal(18 + sex, 3)), 1, 24)
    mar = np.where(rng.random(n) < 0.55, 0, rng.integers(1, 5, n))
    mar = np.where((sex == 1) & (age > 55) & (rng.random(n) < 0.3), 1, mar)
    pob = np.where(rng.random(n) < 0.7, rng.choice([12, 13, 36, 48], n), rng.choice([72, 110, 303, 327], n)).astype(float)
    Z = np.column_stack([(age - 44.5) / 11.0, (hours - 40.6) / 11.2, cow, schl, mar, pob]).astype(np.float64)
    li = -2.5 + 0.06 * (hours - 40) + 0.25 * (schl - 18) + 0.03 * (age - 44) + 0.5 * (mar == 0) * (1 - sex) - 0.3 * sex
    y_inc = (rng.random(n) < 1 / (1 + np.exp(-li))).astype(np.int64)
    lo = np.column_stack([np.zeros(n), 0.2 * (schl - 18) + 0.8 * (cow == 1), 1.2 * sex - 0.03 * (hours - 40),
                          -1.5 * sex + 0.02 * (hours - 40), 0.4 * (cow >= 5) - 0.8 * sex])
    po = np.exp(lo - lo.max(1, keepdims=True)); po /= po.sum(1, keepdims=True)
    y_occ = (po.cumsum(1) > rng.random((n, 1))).argmax(1).astype(np.int64)
    unit = np.arange(n) // 2
    return Z, y_inc, y_occ, sex, unit


def selftest():
    Z, yi, yo, s, u = _invent(3000, 0)
    T = {'inc': 40, 'occ': 40, 'sex': 40}
    st = fit(Z, yi, yo, s, u, 0, 3, order='sex', T=T, n_folds=3)
    Pi, Po = release(st, Z)
    assert Pi.shape == (3000, 2) and Po.shape == (3000, 5)
    assert np.allclose(Pi.sum(1), 1) and np.allclose(Po.sum(1), 1) and np.all(Pi > 0) and np.all(Po > 0)
    L = st['ladder']                                       # a field may skip its moderate level, so 6..12 steps
    assert 6 <= len(L) <= 12 and len(set(L)) == len(L) and 1 <= len(st['steps']) <= 3
    assert sorted(compose(L, len(L))) == sorted(lv[-1] for lv in MENU.values())
    Pi2, Po2 = release(st, Z.copy())                       # deterministic and SEX-free at release (no sex argument exists)
    assert np.array_equal(Pi, Pi2) and np.array_equal(Po, Po2)
    Zh = generalize(Z, st['steps'], st['maps'])            # the release is a function of h(Z): equal h -> equal vectors
    _, inv = np.unique(Zh, axis=0, return_inverse=True); inv = np.asarray(inv).ravel()
    first = np.unique(inv, return_index=True)[1]
    assert np.allclose(Po, Po[first][inv]) and np.allclose(Pi, Pi[first][inv])
    bl = fit(Z, yi, yo, s, u, 0, 3, order='blind', T=T, n_folds=3)
    assert bl['stats'] is not None and all(v['dI'] == 0.0 for v in bl['stats'].values())
    rng = np.random.default_rng(5)
    at = fit(Z, yi, yo, s, u, 0, 3, order='attr', attr=(rng.random(3000) < 0.49).astype(int), T=T, n_folds=3)
    reuse = fit(Z, yi, yo, s, u, 0, 2, ladder=st['ladder'], T=T)
    assert reuse['steps'] == compose(st['ladder'], 2)
    k, rec = select_strength(Z, yi, yo, s, u, 0, st['ladder'], grid=(1, 2), T=T, n_folds=3)
    assert set(rec) == {1, 2} and (k is None or k in (1, 2))
    for kk in range(len(L) + 1):                           # one step per field in every composition
        ss = compose(st['ladder'], kk); assert len({FIELD[x] for x in ss}) == len(ss)
    return {'ok': True, 'sex_ladder': st['ladder'][:4], 'blind_ladder': bl['ladder'][:4], 'attr_ladder': at['ladder'][:4],
            'selected_k_on_invented': k}


if __name__ == '__main__':
    import json, os, time
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(k, '1')
    t = time.process_time()
    r = selftest(); r['cpu_s'] = round(time.process_time() - t, 1)
    print(json.dumps(r))
