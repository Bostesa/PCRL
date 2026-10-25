"""Label-free reader features built ONLY from released probability vectors (PROTOCOL.md section 7; READER_BANK_LOCK.json).

Every transformation with fitted state (categorical income levels, quantile cut points, spline knots, standardization) is fitted
on ATTACK_FIT released outputs alone and then applied unchanged to ATTACK_VALIDATION and DEV_REPORT outputs. No reader feature
uses raw Census inputs, hidden scores, teacher outputs, parameters or another recipient's output (single views), and no label
enters any transformation. Clipping follows the source convention: log(max(p, 1e-300)).
"""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import SplineTransformer

CLIP = 1e-300


def spline_block(x, n_knots):
    """Reviewer rule C2: knots = np.unique(np.quantile(x_ATTACK_FIT, linspace(0, 1, n_knots))) (numpy linear method, what
    knots="quantile" computes); with at least 2 knots, a cubic SplineTransformer on that explicit knot array (constant
    extrapolation, include_bias=True); otherwise the block is omitted. Returns (fitted transformer or None, knot vector)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    base = np.unique(np.quantile(x[:, 0], np.linspace(0, 1, n_knots)))
    if len(base) < 2:
        return None, base
    return SplineTransformer(knots=base[:, None], degree=3, extrapolation='constant', include_bias=True).fit(x), base
MAX_CATEGORIES = 32
N_GROUPS = 6


def _log(p):
    return np.log(np.maximum(np.asarray(p, dtype=np.float64), CLIP))


def rich(p):
    """ET / HGB / MLP features of one received output: probabilities, clipped log-probabilities, one-hot argmax decision."""
    p = np.asarray(p, dtype=np.float64)
    return np.column_stack((p, _log(p), np.eye(p.shape[1])[p.argmax(1)]))


def rich_view(view, inc, occ):
    if view == 'income':
        return rich(inc)
    if view == 'occupation':
        return rich(occ)
    return np.column_stack((rich(inc), rich(occ)))


def occ9(p):
    """The 5 occupation probabilities and the 4 log-contrasts log(p_j / p_0), j = 1..4 (source clipping)."""
    p = np.asarray(p, dtype=np.float64); lp = _log(p)
    return np.column_stack((p, lp[:, 1:] - lp[:, [0]]))


class IncomeRep:
    """Categorical when the received income vector takes <= 32 distinct exact float64 values on ATTACK_FIT (one indicator per
    ATTACK_FIT vector, ordered by P(income = 1), then bytes; an unseen vector gets all indicators zero); otherwise six label-free
    ATTACK_FIT quantile groups of P(income = 1) (side='left': a value equal to a cut point takes the lower group) plus the
    continuous p and log p."""

    def fit(self, p):
        p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
        u = np.unique(p, axis=0)
        if len(u) <= MAX_CATEGORIES:
            order = sorted(range(len(u)), key=lambda i: (float(u[i, 1]), u[i].tobytes()))
            self.kind, self.cats = 'categorical', [u[i].tobytes() for i in order]
            self.index = {k: j for j, k in enumerate(self.cats)}
        else:
            self.kind = 'quantile'
            self.cuts = np.quantile(p[:, 1], np.arange(1, N_GROUPS) / N_GROUPS)
        self.n_distinct_fit = int(len(u))
        return self

    def indicators(self, p):
        p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
        if self.kind == 'categorical':
            M = np.zeros((len(p), len(self.cats)))
            for i, row in enumerate(p):
                j = self.index.get(row.tobytes())
                if j is not None:
                    M[i, j] = 1.0
            return M
        return np.eye(N_GROUPS)[np.searchsorted(self.cuts, p[:, 1], side='left')]

    def continuous(self, p):
        if self.kind == 'categorical':
            return np.zeros((len(p), 0))
        p = np.asarray(p, dtype=np.float64)
        return np.column_stack((p, _log(p)))

    def describe(self):
        return {'kind': self.kind, 'n_distinct_attack_fit_vectors': self.n_distinct_fit,
                'n_indicators': len(self.cats) if self.kind == 'categorical' else N_GROUPS,
                'cuts': None if self.kind == 'categorical' else [float(x) for x in self.cuts]}


class CatiDesign:
    """Regularized categorical-interaction design for one view (PROTOCOL.md section 7.3)."""

    def __init__(self, view):
        self.view = view

    def fit(self, inc, occ):
        if self.view in ('income', 'pair'):
            self.inc = IncomeRep().fit(inc)
            if self.view == 'income' and self.inc.kind == 'quantile':
                self.inc_spline, self.inc_knots = spline_block(self._logit(inc), 6)
        if self.view == 'occupation':
            lc = occ9(occ)[:, 5:]
            fitted = [spline_block(lc[:, j], 5) for j in range(4)]
            self.occ_splines = [f[0] for f in fitted]; self.occ_knots = [f[1] for f in fitted]
        return self

    @staticmethod
    def _logit(inc):
        p = np.asarray(inc, dtype=np.float64)
        return (_log(p[:, 1]) - _log(p[:, 0])).reshape(-1, 1)

    def transform(self, inc, occ):
        if self.view == 'pair':
            M = self.inc.indicators(inc); O = occ9(occ)
            inter = (M[:, :, None] * O[:, None, :]).reshape(len(M), -1)
            return np.column_stack((M, self.inc.continuous(inc), O, inter))
        if self.view == 'income':
            M = self.inc.indicators(inc)
            if self.inc.kind == 'categorical':
                return M
            blocks = [M, self.inc.continuous(inc)]
            if self.inc_spline is not None:
                blocks.append(self.inc_spline.transform(self._logit(inc)))
            return np.column_stack(blocks)
        O = occ9(occ); lc = O[:, 5:]
        spl = [self.occ_splines[j].transform(lc[:, [j]]) for j in range(4) if self.occ_splines[j] is not None]
        prods = [lc[:, i] * lc[:, j] for i in range(4) for j in range(i + 1, 4)]
        dec = np.eye(5)[np.asarray(occ).argmax(1)]
        return np.column_stack([O] + spl + [np.column_stack(prods), dec])

    def describe(self):
        d = {'view': self.view}
        if hasattr(self, 'inc'):
            d['income'] = self.inc.describe()
            if self.inc.kind == 'categorical':
                d['income']['categories_hex'] = [[float(v).hex() for v in np.frombuffer(k, dtype=np.float64)] for k in self.inc.cats]
        if hasattr(self, 'inc_knots'):
            d['income_logit_spline_knots'] = [float(x) for x in self.inc_knots]
        if hasattr(self, 'occ_knots'):
            d['occupation_log_contrast_spline_knots'] = [[float(x) for x in k] for k in self.occ_knots]
        return d

    def unseen(self, inc):
        """Count of scoring rows whose categorical income vector did not occur on ATTACK_FIT (all-zero indicators)."""
        if not hasattr(self, 'inc') or self.inc.kind != 'categorical':
            return None
        return int((self.inc.indicators(inc).sum(1) == 0).sum())
