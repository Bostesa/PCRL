"""Independent numerical verifier: shared helpers written for this verification (PCRL frozen-finalist validation study).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, development roles only. The
service predicts income class (2) and occupation group (5) from AGEP, WKHP, COW, SCHL, MAR and POBP; each recipient receives its
own float64 probability vector and argmax decision. SEX is the dataset's recorded demographic column; evaluation readers try to
infer SEX from the released predictions (each output alone and the linked pair). Defensive evaluation of a privacy-protecting
release.

Nothing here imports ffv/report.py, ffv/feasibility.py or ffv/mapping.py. Data access goes through the admitted loaders
(ffv.common.load_role, service_bank, frozen_bank, source_readers) only; everything else is written here.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CAMPAIGN = 'pcrl_frozen_finalist_validation_v1'
RES = ROOT / 'results' / CAMPAIGN
PRIV = ROOT / '.private' / CAMPAIGN
VPRIV = PRIV / 'verifier'
ADM = PRIV / 'admitted' / 'amd'
RUNS = PRIV / 'runs'
SRC_RES = ROOT / 'results' / 'pcrl_autonomous_method_development_v1'

SEEDS = (0, 1)
TASKS = ('income', 'occupation')
K = {'income': 2, 'occupation': 5}
BANK = ('FINALIST', 'NEURAL_RAW', 'H_LJ', 'TASKONLY', 'B2TWIN', 'PUB_IBSI')
ROLES3 = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
ROLES6 = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
VIEWS = ('income', 'occupation', 'pair')
MODEL_SEED_BASE = 2026101205
SRC8 = ('OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB', 'LOOKUP')
SRC_KIND = {'OUT_LR': 'out', 'OUT_MLP': 'out', 'OUT_HGB': 'out', 'LOG_LR': 'log', 'LOG_MLP': 'log', 'DEC_LR': 'dec', 'CONF_HGB': 'conf'}
NEW8 = ('ET5', 'ET25', 'HGB15', 'HGB31', 'MLP64', 'MLP6432', 'CATI01', 'CATI1')
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
GUARD = 0.005
NEAR_TIE = 0.010


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def wire_hash(p):
    """sha256 over the float64 probability bytes followed by the int64 argmax-decision bytes (the transmitted wire)."""
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    h = hashlib.sha256(p.tobytes())
    h.update(np.ascontiguousarray(np.argmax(p, axis=1).astype(np.int64)).tobytes())
    return h.hexdigest()


def jload(p):
    return json.loads(Path(p).read_text())


def jdump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_default) + '\n')


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o))


# ------------------------------------------------------------------ AUC (own implementation: searchsorted over sorted negatives)
def auc_weighted(y, s, W):
    """Weighted Mann-Whitney AUC for each row of W (B x n nonnegative weights): sum_{i pos, j neg} w_i w_j (1[s_i > s_j] +
    0.5 1[s_i == s_j]) / (sum_pos w)(sum_neg w). Uses sorted negative scores and cumulative weights (exact for integer weights)."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    W = np.atleast_2d(np.asarray(W, dtype=np.float64))
    if s.shape != y.shape or W.shape[1] != len(y) or not np.isfinite(s).all():
        raise ValueError('auc input shapes differ')
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) + len(neg) != len(y) or not len(pos) or not len(neg):
        raise ValueError('binary labels with both classes required')
    o = np.argsort(s[neg], kind='mergesort')
    sn = s[neg][o]
    Wn = W[:, neg[o]]
    cum = np.zeros((W.shape[0], len(neg) + 1))
    np.cumsum(Wn, axis=1, out=cum[:, 1:])
    sp = s[pos]
    lo = np.searchsorted(sn, sp, side='left')
    hi = np.searchsorted(sn, sp, side='right')
    below = cum[:, lo]
    ties = cum[:, hi] - below
    Wp = W[:, pos]
    num = (Wp * (below + 0.5 * ties)).sum(1)
    den = Wp.sum(1) * cum[:, -1]
    if np.any(den <= 0):
        raise ValueError('a replicate lacks one SEX class')
    return num / den


def auc(y, s):
    return float(auc_weighted(y, s, np.ones((1, len(y))))[0])


def ratio_draws(d, W, mask=None):
    """Household-bootstrap replicates of a (sub)group mean: sum_i W_bi d_i / sum_i W_bi (BLAS matrix-vector order)."""
    d = np.asarray(d, dtype=np.float64)
    if mask is not None:
        Wm = np.ascontiguousarray(W[:, mask])
        return (Wm @ d[mask]) / Wm.sum(1)
    return (W @ d) / W.sum(1)


def tcrit(df, upper_tail):
    from scipy.stats import t
    return float(t.ppf(1.0 - upper_tail, df))


# ------------------------------------------------------------------ selection (own code)
def pick_max_first(cands):
    """cands: list of (name, auc, key, origin). Maximum AUC; an exact tie keeps the earlier candidate."""
    best = 0
    for i in range(1, len(cands)):
        if cands[i][1] > cands[best][1]:
            best = i
    others = sorted([c[1] for i, c in enumerate(cands) if i != best], reverse=True)
    gap = (cands[best][1] - others[0]) if others else None
    return {'candidate': cands[best][0], 'auc': cands[best][1], 'key': cands[best][2], 'origin': cands[best][3], 'index': best,
            'gap_to_runner_up': gap, 'near_tie': bool(gap is not None and gap < NEAR_TIE), 'n': len(cands)}


def enhanced_candidates(src_rec, enh_rec, svc, view):
    """Candidate list in the registered source-first order, values taken from the stored records (source AV AUCs stored in
    the admitted source READERS_SEED.json; new AV AUCs stored in READERS_ENH_SEED.json)."""
    s = src_rec['services'][svc]
    wire_new = {}
    for wid, mb in enh_rec['marginal_banks'].items():
        for x in mb['services_aliasing_this_wire']:
            wire_new[(x, mb['task'])] = wid
    if view in TASKS:
        swid = s[f'{view}_wire']
        sread = src_rec['marginal_banks'][swid]['readers']
        nwid = wire_new[(svc, view)]
        nread = enh_rec['marginal_banks'][nwid]['readers']
        c = [(n, sread[n]['attack_validation_auc'], f'SRC::MARG__{swid}__{n}', 'source') for n in SRC8]
        c += [(n, nread[n]['attack_validation_auc'], f'NEW::MARG__{nwid}__{n}', 'new') for n in NEW8]
        return c
    c = [(n, s['pair_readers'][n]['attack_validation_auc'], f'SRC::PAIR__{svc}__{n}', 'source') for n in SRC8]
    for drop, keep in (('occupation', 'income'), ('income', 'occupation')):
        swid = s[f'{keep}_wire']
        sread = src_rec['marginal_banks'][swid]['readers']
        c += [(f'IGNORE_{drop}__{n}', sread[n]['attack_validation_auc'], f'SRC::MARG__{swid}__{n}', 'source') for n in SRC8]
    c += [(n, enh_rec['pairs'][svc][n]['attack_validation_auc'], f'NEW::PAIR__{svc}__{n}', 'new') for n in NEW8]
    for drop, keep in (('occupation', 'income'), ('income', 'occupation')):
        nwid = wire_new[(svc, keep)]
        nread = enh_rec['marginal_banks'][nwid]['readers']
        c += [(f'IGNORE_{drop}__{n}', nread[n]['attack_validation_auc'], f'NEW::MARG__{nwid}__{n}', 'new') for n in NEW8]
    return c


def source_candidates(src_rec, svc, view):
    """The source's own 8 (single view) or 24 (pair) candidates, source order."""
    s = src_rec['services'][svc]
    if view in TASKS:
        swid = s[f'{view}_wire']
        sread = src_rec['marginal_banks'][swid]['readers']
        return [(n, sread[n]['attack_validation_auc'], f'MARG__{swid}__{n}', 'source') for n in SRC8]
    c = [(n, s['pair_readers'][n]['attack_validation_auc'], f'PAIR__{svc}__{n}', 'source') for n in SRC8]
    for drop, keep in (('occupation', 'income'), ('income', 'occupation')):
        swid = s[f'{keep}_wire']
        sread = src_rec['marginal_banks'][swid]['readers']
        c += [(f'IGNORE_{drop}__{n}', sread[n]['attack_validation_auc'], f'MARG__{swid}__{n}', 'source') for n in SRC8]
    return c


# ------------------------------------------------------------------ reader features (own code, following the lock text)
CLIP = 1e-300


def logp(p):
    return np.log(np.maximum(np.asarray(p, dtype=np.float64), CLIP))


def onehot(p):
    p = np.asarray(p, dtype=np.float64)
    return (np.argmax(p, axis=1)[:, None] == np.arange(p.shape[1])[None, :]).astype(np.float64)


def rich(p):
    p = np.asarray(p, dtype=np.float64)
    return np.column_stack((p, logp(p), onehot(p)))


def source_kind(p, kind):
    p = np.asarray(p, dtype=np.float64)
    if kind == 'out':
        return np.column_stack((p, np.argmax(p, axis=1).astype(np.float64)))
    if kind == 'log':
        return np.column_stack((logp(p), onehot(p)))
    if kind == 'dec':
        return onehot(p)
    if kind == 'conf':
        return p.max(1, keepdims=True)
    raise ValueError(kind)


class Std:
    """Column standardization fitted on the fit rows only (zero-variance columns keep scale 1)."""

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.m = X.mean(0)
        sd = X.std(0)
        self.s = np.where(sd > 0, sd, 1.0)
        return self

    def tr(self, X):
        return (np.asarray(X, dtype=np.float64) - self.m) / self.s


def occ_features9(p):
    p = np.asarray(p, dtype=np.float64)
    lp = logp(p)
    return np.column_stack((p, lp[:, 1:] - lp[:, [0]]))


class CatiPair:
    """Pair CATI design (lock section 7.3): income indicators (categorical if <= 32 distinct ATTACK_FIT vectors, else six
    ATTACK_FIT quantile groups plus continuous p and log p), the 9 occupation features, and every indicator x feature."""

    def fit(self, inc_fit):
        inc_fit = np.ascontiguousarray(np.asarray(inc_fit, dtype=np.float64))
        rows = {r.tobytes(): r for r in inc_fit}
        if len(rows) <= 32:
            self.kind = 'categorical'
            self.cats = sorted(rows, key=lambda b: (float(rows[b][1]), b))
            self.idx = {b: j for j, b in enumerate(self.cats)}
        else:
            self.kind = 'quantile'
            self.cuts = np.quantile(inc_fit[:, 1], np.arange(1, 6) / 6)
        self.n_distinct = len(rows)
        return self

    def indicators(self, inc):
        inc = np.ascontiguousarray(np.asarray(inc, dtype=np.float64))
        if self.kind == 'categorical':
            M = np.zeros((len(inc), len(self.cats)))
            for i, r in enumerate(inc):
                j = self.idx.get(r.tobytes())
                if j is not None:
                    M[i, j] = 1.0
            return M
        g = np.searchsorted(self.cuts, inc[:, 1], side='left')
        return (g[:, None] == np.arange(6)[None, :]).astype(np.float64)

    def design(self, inc, occ):
        M = self.indicators(inc)
        O = occ_features9(occ)
        cont = np.zeros((len(M), 0)) if self.kind == 'categorical' else np.column_stack((inc, logp(inc)))
        inter = (M[:, :, None] * O[:, None, :]).reshape(len(M), -1)
        return np.column_stack((M, cont, O, inter))

    def state(self):
        if self.kind == 'categorical':
            return {'kind': 'categorical', 'categories_hex': [[float(v).hex() for v in np.frombuffer(b, dtype=np.float64)] for b in self.cats]}
        return {'kind': 'quantile', 'cuts': [float(x) for x in self.cuts]}


def make_model(name, seed):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    if name in ('ET5', 'ET25'):
        return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5 if name == 'ET5' else 25, max_features=1.0, n_jobs=1,
                                    random_state=seed)
    if name in ('HGB15', 'HGB31'):
        return HistGradientBoostingClassifier(max_leaf_nodes=15 if name == 'HGB15' else 31, max_iter=200, learning_rate=0.05,
                                              min_samples_leaf=30, l2_regularization=1.0, early_stopping=False, random_state=seed)
    if name in ('MLP64', 'MLP6432'):
        return MLPClassifier(hidden_layer_sizes=(64,) if name == 'MLP64' else (64, 32), alpha=0.01, batch_size=256, max_iter=300,
                             early_stopping=False, random_state=seed)
    if name in ('CATI01', 'CATI1'):
        return LogisticRegression(C=0.1 if name == 'CATI01' else 1.0, solver='lbfgs', max_iter=5000)
    # source slate (task_conditioned_representation.attacks.fit_reader definitions, as registered in the source lineage)
    if name in ('OUT_MLP', 'LOG_MLP'):
        return MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=1e-4, max_iter=300,
                             early_stopping=False, random_state=seed)
    if name in ('OUT_HGB', 'CONF_HGB'):
        return HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, early_stopping=False,
                                              random_state=seed)
    if name in ('OUT_LR', 'LOG_LR', 'DEC_LR'):
        return LogisticRegression(C=1, max_iter=500, solver='lbfgs')
    raise ValueError(name)


def symbolic_params(name):
    m = make_model(name, 0)
    return {k: ('MODEL_SEED_BASE+s' if k == 'random_state' and not name.startswith('CATI') else repr(v)) for k, v in sorted(m.get_params().items())}


def fit_predict(name, Xfit, yfit, Xs, seed):
    """Standardize on the fit rows, fit, return P(SEX = 1) on each matrix in Xs (and the fitted objects)."""
    import warnings
    sc = Std().fit(Xfit)
    m = make_model(name, seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(sc.tr(Xfit), yfit)
    if not np.array_equal(m.classes_, [0, 1]):
        raise ValueError('class order differs')
    return [m.predict_proba(sc.tr(X))[:, 1].astype(np.float64) for X in Xs], (m, sc)


def orient_scores(y_val, s_val, others):
    sign = 1 if auc(y_val, s_val) >= 0.5 else -1
    f = (lambda a: np.asarray(a)) if sign == 1 else (lambda a: 1.0 - np.asarray(a))
    return sign, f(s_val), [f(o) for o in others]
