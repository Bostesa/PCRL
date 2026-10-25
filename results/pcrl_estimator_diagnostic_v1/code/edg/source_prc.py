"""Verbatim copies of the two DRF1 nested-estimator helpers from nrm/fa/prc.py (blob of commit 913eb5c91, SHA-256 in
SOURCE_PROVENANCE.json): `auc` (rank AUC, ties 1/2, NaN when a class is missing) and `lookup_scores` (the held-out LOOKUP reader of
drf.dr.nested: Beta(a * rbar, a * (1 - rbar))-shrunk training SEX rate per cell; an unseen cell gets rbar). The function bodies below
must equal the source text; edg.selftest checks that against the git blob."""
import numpy as np


def auc(score, s):
    from scipy.stats import rankdata
    s = np.asarray(s).astype(bool)
    n1, n0 = int(s.sum()), int((~s).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(score)
    return float((r[s].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def lookup_scores(train_cells, train_sex, test_cells, a=1.0):
    """LOOKUP reader: Beta(a * rbar, a * (1 - rbar))-shrunk SEX rate of the cell in the training part."""
    rbar = float(train_sex.mean())
    keys, inv = np.unique(np.concatenate([train_cells, test_cells]), return_inverse=True)
    itr, ite = inv[:len(train_cells)], inv[len(train_cells):]
    n = np.bincount(itr, minlength=len(keys))
    s1 = np.bincount(itr, weights=train_sex.astype(np.float64), minlength=len(keys))
    return ((s1 + a * rbar) / (n + a))[ite]
