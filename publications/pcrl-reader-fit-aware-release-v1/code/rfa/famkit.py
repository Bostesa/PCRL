"""Shared, label-free helpers of the family modules of the PCRL strong-anchor release campaign (`pcrl_reader_fit_aware_release_v1`;
EXPLORATORY DEVELOPMENT on previously used public-use ACS 2019 1-Year PUMS Florida person records; not confirmation).

Nothing here reads a label: label reads happen only inside the family entry points (rfa.ia.*, rfa.ib.*, rfa.ctl.*, rfa.fs.*), which
call `rfa.common.load` themselves (the access layer authorizes the calling module). This module holds
- write-once model files beside each release: runs/<FAMILY>/<config>/V<v>/MODEL.pkl + MODEL.json (hash-checked on every load), so
  that every saved release is computed by the configuration's own deployment function from admitted features alone (bitwise);
- the admitted feature matrices of the six roles (label-free) and the compact six-field coding (ohn.fam_a.compact_po);
- per-person task losses, paired spends over a twin with household-clustered SEs (accuracy as a loss; the men's and women's
  occupation Brier rows as linearised subgroup means), and the predeclared strength-selection rule (PROTOCOL section 4: the largest
  grid strength such that it and every smaller strength keep estimate + 2 SE <= the budget on every metric);
- the release record builder required by rfa.release.save.

Sign convention: every paired spend is loss(candidate) - loss(twin) on the same people (positive = the candidate spends more).
"""
from __future__ import annotations
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np

from . import common as C

SCHEMA = 'RFA1-model-v1'
K_INC, K_OCC = 2, 5
CLIP = 1e-6
BUDGET_S2 = {'occ_nll': 0.008, 'inc_nll': 0.004, 'occ_brier_men': 0.0030, 'occ_brier_women': 0.0030, 'occ_acc': 0.006,
             'inc_acc': 0.004}
METRIC_KEYS = ('occ_nll', 'inc_nll', 'occ_brier', 'inc_brier', 'occ_acc', 'inc_acc', 'tot_nll')


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def clipn(P):
    P = np.clip(np.asarray(P, dtype=np.float64), CLIP, 1.0)
    return P / P.sum(-1, keepdims=True)


def cluster_se(v, unit):
    v = np.asarray(v, dtype=np.float64)
    g, inv = np.unique(unit, return_inverse=True)
    s = np.bincount(inv, weights=v - v.mean(), minlength=len(g))
    G = len(g)
    return float(np.sqrt(G / max(G - 1, 1) * np.sum(s ** 2)) / len(v))


def household_folds(unit, k, tag):
    return C.household_folds(unit, k, tag)


# ---------------------------------------------------------------- features (label-free)
def role_features():
    return {r: C.features(r) for r in C.ROLES6}


def compact(X, names):
    from ohn import fam_a as FA
    return FA.compact_po(np.asarray(X, dtype=np.float64), tuple(map(str, names)))


# ---------------------------------------------------------------- model files (write-once, hash-checked)
def model_dir(family, config_id, version):
    return C.RUNS / family / config_id / f'V{int(version)}'


def model_paths(family, config_id, version):
    d = model_dir(family, config_id, version)
    return [d / 'MODEL.pkl', d / 'MODEL.json']


def write_model(family, config_id, version, payload):
    """Write-once MODEL.pkl / MODEL.json. An incomplete earlier attempt (files but no RELEASE.npz) is moved aside and kept."""
    d = model_dir(family, config_id, version)
    if (d / 'RELEASE.npz').exists():
        raise PermissionError(f'{family}/{config_id}/V{version} is complete (append-only; a revision gets a new config id)')
    if d.exists() and any(d.iterdir()):
        d.rename(d.with_name(f'{d.name}__incomplete_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}'))
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = {**payload, 'schema': SCHEMA, 'family': family, 'config_id': config_id, 'version': int(version)}
    buf = pickle.dumps(payload, protocol=5)
    with open(d / 'MODEL.pkl', 'xb') as f:
        f.write(buf)
    meta = {'schema': SCHEMA, 'family': family, 'config_id': config_id, 'version': int(version), 'model_sha256': sha_bytes(buf),
            'feature_names_sha256': sha_bytes(json.dumps(list(payload['feature_names'])).encode()), 'utc': utc(),
            'keys': sorted(payload)}
    with open(d / 'MODEL.json', 'x') as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return meta


def load_model(family, config_id, version):
    d = model_dir(family, config_id, version)
    meta = json.loads((d / 'MODEL.json').read_text())
    buf = (d / 'MODEL.pkl').read_bytes()
    if sha_bytes(buf) != meta['model_sha256'] or meta['config_id'] != config_id or int(meta['version']) != int(version):
        raise PermissionError(f'{family}/{config_id}/V{version}: MODEL.pkl differs from MODEL.json')
    P = pickle.loads(buf)
    if P.get('schema') != SCHEMA or P['config_id'] != config_id or int(P['version']) != int(version) or P['family'] != family:
        raise PermissionError(f'{family}/{config_id}/V{version}: model payload identity differs')
    return P


def check_X(X, names, P):
    names = tuple(map(str, names))
    if names != tuple(P['feature_names']):
        raise PermissionError('feature columns differ from the admitted encoding the configuration was fitted on')
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(names) or not np.isfinite(X).all():
        raise ValueError('X must be a finite (n, columns) matrix of the admitted encoding')
    return X, names


def release_all(predict, config_id, version, roles=None):
    """Every role's release computed by the deployment function itself (the saved wire IS its output, bitwise)."""
    roles = roles or role_features()
    return {r: predict(config_id, version, F['X'], F['feature_names']) for r, F in roles.items()}


# ---------------------------------------------------------------- per-person losses, spends and the selection rule
def person_losses(P_inc, P_occ, y_inc, y_occ):
    out = {}
    for t, P, y, K in (('inc', P_inc, y_inc, K_INC), ('occ', P_occ, y_occ, K_OCC)):
        P = np.asarray(P, dtype=np.float64); y = np.asarray(y, dtype=np.int64)
        oh = np.eye(K)[y]
        out[f'{t}_nll'] = -np.log(np.clip(P[np.arange(len(y)), y], 1e-300, None))
        out[f'{t}_brier'] = ((P - oh) ** 2).sum(1)
        out[f'{t}_acc'] = (P.argmax(1) != y).astype(np.float64)          # accuracy entered as a loss
    out['tot_nll'] = out['inc_nll'] + out['occ_nll']
    return out


def spend(cand, twin, unit, sex=None):
    """Paired spend of `cand` over `twin` (person_losses dicts on the same people): {metric: (estimate, household-clustered SE)}.
    With `sex`, the men's / women's occupation Brier rows (linearised subgroup means, clustered)."""
    out = {}
    for key in ('occ_nll', 'inc_nll', 'tot_nll', 'occ_brier', 'inc_brier', 'occ_acc', 'inc_acc'):
        v = cand[key] - twin[key]
        out[key] = (float(v.mean()), cluster_se(v, unit))
    if sex is not None:
        sx = np.asarray(sex).astype(bool)
        v = cand['occ_brier'] - twin['occ_brier']
        for key, sub in (('occ_brier_men', ~sx), ('occ_brier_women', sx)):
            w = sub / sub.mean()
            out[key] = (float(v[sub].mean()), cluster_se(w * (v - v[sub].mean()), unit))
    return out


def meets(sp, budget, k_se=2.0):
    return all(sp[key][0] + k_se * sp[key][1] <= lim for key, lim in budget.items())


def select_strength(spends, order, budget=BUDGET_S2, k_se=2.0):
    """The LAST strength in `order` (increasing) such that it and every earlier one meet the budget (estimate + k_se SE <= limit on
    every metric). None if the first fails. `spends` = {strength: spend(...)}."""
    pick = None
    for s in order:
        if meets(spends[s], budget, k_se):
            pick = s
        else:
            break
    return pick


def match_spend(spends, order, target, key='tot_nll'):
    """The first strength in `order` whose point spend on `key` is >= target (a spend-matching rule on MODEL_FIT); None if none."""
    for s in order:
        if spends[s][key][0] >= target:
            return s
    return None


def jsonable_spend(sp):
    return {k: {'estimate': v[0], 'se': v[1]} for k, v in sp.items()}


def record(card, kind, description, fits, roles_read, exposure, people, households, cpu, extra=None, randomized=False, **rnd):
    rec = {'mechanism_card': card, 'kind': kind, 'description': description, 'n_model_fits': int(fits[0]),
           'n_calibrator_fits': int(fits[2]) if len(fits) > 2 else 0, 'n_crossfit_fits': int(fits[1]),
           'n_critic_fits': int(fits[3]) if len(fits) > 3 else 0, 'roles_read': roles_read, 'exposure': exposure,
           'training_people': int(people), 'training_households': int(households), 'fit_cpu_s': round(float(cpu), 1),
           'randomized': bool(randomized), **(extra or {})}
    rec.update(rnd)
    return rec


# ---------------------------------------------------------------- MODEL_FIT reader check (descriptive; never a decision input)
def auc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def mf_reader_check(views_by_config, sex, unit, tag, hgb_pair=True):
    """Household 5-fold cross-fitted SEX readers on MODEL_FIT out-of-fold releases (descriptive; the campaign's decisions use the
    MODEL_SELECT grid only): LOG_LR (logistic regression on standardized log-probabilities) for the income, occupation and pair
    views, and HGB15 (15-leaf gradient boosting) for the pair view. views_by_config = {config: (P_inc, P_occ)}."""
    import warnings
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    f = household_folds(unit, 5, tag)
    out = {}
    for cfg, (pi, po) in views_by_config.items():
        feats = {'income': np.log(pi[:, 1:] / pi[:, :1]), 'occupation': np.log(po[:, 1:] / po[:, :1])}
        feats['pair'] = np.column_stack([feats['income'], feats['occupation']])
        rec = {}
        for view, F in feats.items():
            s_lr = np.zeros(len(sex)); s_hgb = np.zeros(len(sex))
            for j in range(5):
                tr, te = f != j, f == j
                mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    m = LogisticRegression(C=1.0, max_iter=2000).fit((F[tr] - mu) / sd, sex[tr])
                s_lr[te] = m.decision_function((F[te] - mu) / sd)
                if hgb_pair and view == 'pair':
                    h = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
                                                       random_state=0).fit(F[tr], sex[tr])
                    s_hgb[te] = h.decision_function(F[te])
            rec[f'LOG_LR_{view}'] = auc(s_lr, sex)
            if hgb_pair and view == 'pair':
                rec['HGB15_pair'] = auc(s_hgb, sex)
        out[cfg] = rec
    return out


def selftest():
    rng = np.random.default_rng(3)
    n = 600
    unit = np.arange(n) // 3
    y_i, y_o, sex = rng.integers(0, 2, n), rng.integers(0, 5, n), rng.integers(0, 2, n)
    Pi, Po = rng.dirichlet(np.ones(2), n), rng.dirichlet(np.ones(5), n)
    a = person_losses(Pi, Po, y_i, y_o)
    z = spend(a, a, unit, sex)
    ok = {'zero_spend_identity': all(v[0] == 0.0 for v in z.values())}
    flat = person_losses(np.full_like(Pi, 0.5), np.full_like(Po, 0.2), y_i, y_o)
    sp = {0.0: z, 1.0: spend(flat, a, unit, sex)}
    ok['selection_keeps_identity'] = select_strength(sp, [0.0], BUDGET_S2) == 0.0
    big = {k: (1.0, 0.0) for k in z}
    ok['selection_stops_at_first_failure'] = select_strength({0.0: z, 1.0: big, 2.0: z}, [0.0, 1.0, 2.0]) == 0.0
    ok['match_spend'] = match_spend({0.0: z, 1.0: big}, [0.0, 1.0], 0.5) == 1.0
    ok['all_ok'] = all(ok.values())
    return ok


if __name__ == '__main__':
    r = selftest()
    print(json.dumps(r))
    raise SystemExit(0 if r['all_ok'] else 1)
