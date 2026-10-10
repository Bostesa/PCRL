"""Direct categorical sensitive-attribute readers for D's public hard codes.

The pair fallback is fixed before outcomes: the arithmetic mean of the two
smoothed local posteriors. An unseen local code falls back to the fit prior.
All smoothing uses the fit prior, not the screen prevalence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hard_code import ALPHABET


ALPHAS = (0.5, 1.0, 5.0)


def _validate_codes(codes: np.ndarray) -> np.ndarray:
    x = np.asarray(codes)
    if (x.ndim not in (1, 2) or (x.ndim == 2 and x.shape[1] != 2)
            or not np.issubdtype(x.dtype, np.integer)
            or np.any(x < 0) or np.any(x >= ALPHABET)):
        raise ValueError("local IDs or aligned two-column pair IDs required")
    return x.astype(np.int64, copy=False)


@dataclass(frozen=True)
class CellReader:
    alpha: float
    width: int
    prior: float
    counts: dict[tuple[int, ...], tuple[int, int]]
    marginals: tuple[dict[int, tuple[int, int]], ...]

    @classmethod
    def fit(cls, codes: np.ndarray, sex: np.ndarray, alpha: float) -> "CellReader":
        x = _validate_codes(codes)
        y = np.asarray(sex)
        if (alpha not in ALPHAS or len(x) == 0 or y.shape != (len(x),)
                or not np.issubdtype(y.dtype, np.integer)
                or not np.isin(y, (0, 1)).all() or len(np.unique(y)) != 2):
            raise ValueError("predeclared alpha and binary attacker-fitting labels required")
        if x.ndim == 1:
            x = x[:, None]
        counts: dict[tuple[int, ...], list[int]] = {}
        marginals: list[dict[int, list[int]]] = [dict() for _ in range(x.shape[1])]
        for row, label in zip(x, y):
            key = tuple(int(c) for c in row)
            counts.setdefault(key, [0, 0])[int(label)] += 1
            for j, code in enumerate(key):
                marginals[j].setdefault(code, [0, 0])[int(label)] += 1
        return cls(float(alpha), x.shape[1], float(np.mean(y)),
                   {k: tuple(v) for k, v in counts.items()},
                   tuple({k: tuple(v) for k, v in m.items()} for m in marginals))

    def _p1(self, count: tuple[int, int] | None) -> float:
        if count is None:
            return self.prior
        n0, n1 = count
        return (n1 + self.alpha * self.prior) / (n0 + n1 + self.alpha)

    def predict_proba(self, codes: np.ndarray) -> np.ndarray:
        x = _validate_codes(codes)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[1] != self.width:
            raise ValueError("reader and release code widths differ")
        p1 = np.empty(len(x), dtype=np.float64)
        for i, row in enumerate(x):
            key = tuple(int(c) for c in row)
            observed = self.counts.get(key)
            if observed is not None:
                p1[i] = self._p1(observed)
            elif self.width == 1:
                p1[i] = self.prior
            else:
                p1[i] = sum(self._p1(self.marginals[j].get(code))
                            for j, code in enumerate(key)) / 2.0
        return np.column_stack((1.0 - p1, p1))


def fit_bank(codes: np.ndarray, sex: np.ndarray) -> dict[float, CellReader]:
    return {alpha: CellReader.fit(codes, sex, alpha) for alpha in ALPHAS}
