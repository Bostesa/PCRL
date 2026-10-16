"""Designer 1 cross-critique check (data-free): is counter-shift cancellation robust beyond the criterion's reader class?

Setting: one released coordinate h (the occupation direction orthogonal to income) with SEX signal, plus a driver u
correlated with SEX. The counter-shift release is h' = h + k*u, with k chosen so that E[h'|S=1] = E[h'|S=0] (zero linear
covariance). Every case below is built so that the class-conditional VARIANCES of h' are also equal, so a linear and a
quadratic (second-moment) criterion both read exactly zero after the shift. We then compute the exact Bayes AUC of S from
h' by numerical integration on a fine grid (likelihood-ratio ordering, ties 1/2). Bayes AUC = 0.5 means the cancellation
is genuine for every reader; anything above 0.5 is signal left for nonlinear readers (MLP/HGB) that the criterion cannot
see. P(S=1) does not enter AUC.
"""
from mechanism_design import no_real_data
no_real_data.install()

import json
import math
from pathlib import Path

import numpy as np

GRID = np.linspace(-25.0, 25.0, 200001)
DX = GRID[1] - GRID[0]
OUT = Path(__file__).resolve().parent / 'D1_XC_DRIVER_RESULTS.json'


def phi(x, m=0.0, s=1.0):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def bayes_auc(p1, p0):
    m1, m0 = p1 * DX, p0 * DX
    m1, m0 = m1 / m1.sum(), m0 / m0.sum()
    lr = np.where(m0 > 0, m1 / np.maximum(m0, 1e-300), np.inf)
    order = np.argsort(lr, kind='mergesort')
    lr_s, a1, a0 = lr[order], m1[order], m0[order]
    # group exact ties
    auc, cum0, i, n = 0.0, 0.0, 0, len(lr_s)
    while i < n:
        j = i
        while j + 1 < n and lr_s[j + 1] == lr_s[i]:
            j += 1
        g1, g0 = a1[i:j + 1].sum(), a0[i:j + 1].sum()
        auc += g1 * cum0 + 0.5 * g1 * g0
        cum0 += g0
        i = j + 1
    return float(auc)


def moments(p):
    w = p * DX; w = w / w.sum()
    m = float((GRID * w).sum()); v = float(((GRID - m) ** 2 * w).sum())
    return m, v


def discrete_driver_case(name, d, levels, w1, w0):
    """h|S ~ N(mu_S, 1), mu1 - mu0 = d; u discrete with P(u=level|S). k cancels the mean difference."""
    levels, w1, w0 = map(np.asarray, (levels, w1, w0))
    eu1, eu0 = float((levels * w1).sum()), float((levels * w0).sum())
    k = -d / (eu1 - eu0)
    p1 = sum(w * phi(GRID, d + k * l) for l, w in zip(levels, w1))
    p0 = sum(w * phi(GRID, 0.0 + k * l) for l, w in zip(levels, w0))
    (m1, v1), (m0, v0) = moments(p1), moments(p0)
    driver_auc = bayes_auc(sum(w * phi(GRID, l, 1e-3) for l, w in zip(levels, w1)),
                           sum(w * phi(GRID, l, 1e-3) for l, w in zip(levels, w0)))
    return {'case': name, 'h_alone_auc': bayes_auc(phi(GRID, d), phi(GRID, 0.0)), 'driver_alone_auc': driver_auc,
            'k': k, 'mean_diff_after': m1 - m0, 'var_diff_after': v1 - v0,
            'bayes_auc_after': bayes_auc(p1, p0)}


def gaussian_driver_case(name, d, du):
    """Control: u|S ~ N(+-du/2, 1) (a location family). Cancellation should be exact for every reader."""
    k = -d / du
    s = math.sqrt(1 + k * k)
    p1 = phi(GRID, d + k * du / 2, s); p0 = phi(GRID, 0.0 - k * du / 2, s)
    (m1, v1), (m0, v0) = moments(p1), moments(p0)
    return {'case': name, 'h_alone_auc': bayes_auc(phi(GRID, d), phi(GRID, 0.0)),
            'driver_alone_auc': bayes_auc(phi(GRID, du / 2), phi(GRID, -du / 2)), 'k': k,
            'mean_diff_after': m1 - m0, 'var_diff_after': v1 - v0, 'bayes_auc_after': bayes_auc(p1, p0)}


def mixture_task_case(name, sep, wdelta, du):
    """Task coordinate is a class mixture whose weights depend on SEX (as occupation classes do); Gaussian driver.
    h|S ~ (1-w_S) N(0,1) + w_S N(sep,1), w_S = 0.5 +- wdelta (symmetric -> equal variances); u|S ~ N(+-du/2, 1)."""
    w1, w0 = 0.5 + wdelta, 0.5 - wdelta
    d = sep * (w1 - w0)
    k = -d / du
    s = math.sqrt(1 + k * k)
    p1 = (1 - w1) * phi(GRID, 0 + k * du / 2, s) + w1 * phi(GRID, sep + k * du / 2, s)
    p0 = (1 - w0) * phi(GRID, 0 - k * du / 2, s) + w0 * phi(GRID, sep - k * du / 2, s)
    h1 = (1 - w1) * phi(GRID, 0) + w1 * phi(GRID, sep); h0 = (1 - w0) * phi(GRID, 0) + w0 * phi(GRID, sep)
    (m1, v1), (m0, v0) = moments(p1), moments(p0)
    return {'case': name, 'h_alone_auc': bayes_auc(h1, h0), 'driver_alone_auc': bayes_auc(phi(GRID, du / 2), phi(GRID, -du / 2)),
            'k': k, 'mean_diff_after': m1 - m0, 'var_diff_after': v1 - v0, 'bayes_auc_after': bayes_auc(p1, p0)}


def main():
    d = math.sqrt(2) * 0.3866   # h-alone Gaussian AUC 0.650: occupation's partial signal with income frozen at T_TREE (D2's d'^2 0.299)
    rows = [gaussian_driver_case('G: Gaussian driver (location family), control', d, 0.5),
            gaussian_driver_case('G: Gaussian driver, weaker (du 0.25)', d, 0.25)]
    for delta in (0.05, 0.10, 0.20):
        rows.append(discrete_driver_case(f'B: binary category driver, P(u=1|S)=0.5+-{delta}', d, [0, 1],
                                         [0.5 - delta, 0.5 + delta], [0.5 + delta, 0.5 - delta]))
    rows.append(discrete_driver_case('C3: three-level category driver (0.25,0.35,0.40) vs reversed', d, [-1, 0, 1],
                                     [0.25, 0.35, 0.40], [0.40, 0.35, 0.25]))
    rows.append(mixture_task_case('H: class-mixture task coordinate (sep 2, w 0.5+-0.1) with Gaussian driver du 0.5', 2.0, 0.10, 0.5))
    for r in rows:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    OUT.write_text(json.dumps(rows, indent=1))
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
