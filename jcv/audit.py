"""Attacker slates (fresh fits; never the training critics). Fit on attacker_fit, select on attacker_val log loss,
score the selected attacker. No assessment-based choice and no AUC orientation flip.

Inner (selection) slate, per view:  LR(C=1) | MLP(64,64) | HGB(default)            -> score on attacker_val
Final slate, per view (registered):
  LR      StandardScaler + LogisticRegression, C in {0.01, 0.1, 1, 10, 100}
  MLP     hidden in {(64,), (128,), (64,64), (128,128)}, adam, alpha 1e-4, early stopping on a 10% split of
          attacker_fit, max_iter 300
  HGB     HistGradientBoosting, learning_rate in {0.05, 0.1} x max_leaf_nodes in {15, 31}, max_iter 200
  DA      defense-aware: release canonicalised with its known structure (attacker_fit PCA-whitening that drops the
          directions LEACE collapsed, i.e. variance <= 1e-9 x max) followed by MLP(128,128); for finite releases
          (FARE cells) the exact cell-conditional Bayes attacker (Laplace alpha selected on attacker_val)
  coalition banks also contain the selected recipient-1-only and recipient-2-only attackers (ignore-other-view)
The selected attacker is refitted at attacker seeds 0, 1, 2 (deterministic families give identical refits); recovery
is the macro one-vs-rest AUC (binary SEX: the AUC) averaged over the three refits.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLIP = 1e-12
ATT_SEEDS = (0, 1, 2)
warnings.filterwarnings("ignore")


def _lr(C, seed):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def _mlp(h, seed):
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=h, alpha=1e-4, max_iter=300,
                                                         early_stopping=True, validation_fraction=0.1,
                                                         n_iter_no_change=15, random_state=seed))


def _hgb(lr, leaves, seed):
    return HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaves, max_iter=200, early_stopping=False,
                                          random_state=seed)


class Canon:
    """Defense-aware canonicalisation: centre, PCA on attacker_fit, drop collapsed directions, whiten."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y=None):
        self.pca = PCA(whiten=True, random_state=self.seed).fit(X)
        ev = self.pca.explained_variance_
        self.keep = ev > 1e-9 * ev.max()
        return self

    def transform(self, X):
        return self.pca.transform(X)[:, self.keep]


def _da(seed):
    from sklearn.base import BaseEstimator, ClassifierMixin

    class DA(BaseEstimator, ClassifierMixin):
        def __init__(self, seed=0):
            self.seed = seed

        def fit(self, X, y):
            self.c_ = Canon(self.seed).fit(X)
            self.m_ = MLPClassifier(hidden_layer_sizes=(128, 128), alpha=1e-4, max_iter=300, early_stopping=True,
                                    validation_fraction=0.1, n_iter_no_change=15, random_state=self.seed)
            self.m_.fit(self.c_.transform(X), y)
            self.classes_ = self.m_.classes_
            return self

        def predict_proba(self, X):
            return self.m_.predict_proba(self.c_.transform(X))
    return DA(seed)


class CellConditional:
    """Exact cell-conditional attacker for finite releases: P(s | release row) with Laplace smoothing."""

    def __init__(self, alpha=1.0, K=2):
        self.alpha, self.K = alpha, K

    def fit(self, X, y):
        keys = [r.tobytes() for r in np.ascontiguousarray(np.round(X, 9))]
        self.prior_ = np.bincount(y, minlength=self.K) / len(y)
        self.table_ = {}
        for k, s in zip(keys, y):
            self.table_.setdefault(k, np.zeros(self.K))[s] += 1
        self.classes_ = np.arange(self.K)
        return self

    def predict_proba(self, X):
        out = np.empty((len(X), self.K))
        for i, r in enumerate(np.ascontiguousarray(np.round(X, 9))):
            c = self.table_.get(r.tobytes())
            out[i] = self.prior_ if c is None else (c + self.alpha * self.prior_) / (c.sum() + self.alpha)
        return out


def proba(m, X, K):
    P = m.predict_proba(X)
    cl = list(getattr(m, "classes_", range(P.shape[1])))
    out = np.zeros((len(X), K))
    for j, c in enumerate(cl):
        out[:, int(c)] = P[:, j]
    return out


def logloss(y, P):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], CLIP, 1))))


def macro_auc(y, P, classes):
    if len(classes) == 2 and P.shape[1] == 2:
        return float(roc_auc_score(y == classes[1], P[:, classes[1]]))
    return float(np.mean([roc_auc_score(y == c, P[:, c]) for c in classes]))


def slate(kind, finite=False):
    """List of (name, factory(seed)) for one view."""
    if kind == "inner":
        return [("LR_C1", lambda s: _lr(1.0, s)), ("MLP_64x64", lambda s: _mlp((64, 64), s)),
                ("HGB_0.1_31", lambda s: _hgb(0.1, 31, s))]
    out = [(f"LR_C{C}", (lambda s, C=C: _lr(C, s))) for C in (0.01, 0.1, 1.0, 10.0, 100.0)]
    out += [(f"MLP_{'x'.join(map(str, h))}", (lambda s, h=h: _mlp(h, s))) for h in ((64,), (128,), (64, 64), (128, 128))]
    out += [(f"HGB_{lr}_{lv}", (lambda s, lr=lr, lv=lv: _hgb(lr, lv, s))) for lr in (0.05, 0.1) for lv in (15, 31)]
    out += [("DA_canonical_MLP", lambda s: _da(s))]
    if finite:
        out += [(f"CC_alpha{a}", (lambda s, a=a: CellConditional(a))) for a in (0.1, 1.0, 10.0)]
    return out


def fit_view(Xf, yf, Xv, yv, kind, K=2, finite=False):
    """Fit the slate on attacker_fit, select by attacker_val log loss (ties -> slate order)."""
    table, best = [], None
    for name, fac in slate(kind, finite):
        m = fac(0).fit(Xf, yf)
        Pv = proba(m, Xv, K)
        ll = logloss(yv, Pv)
        table.append({"attacker": name, "attacker_val_log_loss": ll})
        if best is None or ll < best[0] - 1e-12:
            best = (ll, name, fac, m, Pv)
    return {"selected": best[1], "val_log_loss": best[0], "table": table, "factory": best[2], "model0": best[3],
            "Pv": best[4]}


def recovery(sel, Xf, yf, Xs, ys, classes, K=2, refit=True):
    """Selected attacker refitted at attacker seeds 0,1,2; returns (mean AUC, list of per-seed P on Xs)."""
    Ps = []
    for s in (ATT_SEEDS if refit else (0,)):
        m = sel["model0"] if s == 0 else sel["factory"](s).fit(Xf, yf)
        Ps.append(proba(m, Xs, K))
    return float(np.mean([macro_auc(ys, P, classes) for P in Ps])), Ps
