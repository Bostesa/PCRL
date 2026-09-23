"""Rational finite laws fixed by PROTOCOL.md before synthetic outcomes."""

from fractions import Fraction as F

import numpy as np


def _partly_coupled(shift=F(0), target_cell=None, prior_shift=F(0)):
    law = np.zeros((2, 2, 2, 2, 2), dtype=float)
    u = F(prior_shift)
    masses = {(0, 0): F(3, 8) - u / 2, (1, 0): F(1, 8) + u / 2,
              (0, 1): F(1, 8) + u / 2, (1, 1): F(3, 8) - u / 2}
    if target_cell is not None:
        old = masses[(1, 0)]
        for key in masses:
            if key == (1, 0):
                masses[key] = F(target_cell)
            else:
                masses[key] *= (1 - F(target_cell)) / (1 - old)
    for (s, ha), mass in masses.items():
        hb = s ^ ha
        pt = F(1 + 2 * s + ha, 5) + F(shift)
        assert 0 <= pt <= 1
        for t in range(2):
            mt = pt if t else 1 - pt
            for y in range(2):
                my = F(4, 5) if y == t else F(1, 5)
                law[s, ha, hb, t, y] += float(mass * mt * my)
    return law


def fixtures():
    xor = np.zeros((2, 2, 1, 2, 2))
    for s in range(2):
        for ha in range(2):
            t = s ^ ha
            xor[s, ha, 0, t, t] = 0.25

    safe_independent = np.zeros((2, 2, 2, 2, 2))
    safe_correlated = np.zeros((2, 2, 1, 2, 2))
    for s in range(2):
        for y in range(2):
            safe_correlated[s, s ^ y, 0, y, y] = 0.25
            for ha in range(2):
                for hb in range(2):
                    safe_independent[s, ha, hb, y, y] = 1 / 16

    rational = np.zeros((2, 1, 1, 3, 2))
    ps = [F(1, 10), F(9, 10), F(9, 10)]
    py = [F(1, 5), F(1, 5), F(4, 5)]
    for t in range(3):
        for s in range(2):
            for y in range(2):
                rational[s, 0, 0, t, y] = float(F(1, 3) *
                                                 (ps[t] if s else 1 - ps[t]) *
                                                 (py[t] if y else 1 - py[t]))

    nominal = _partly_coupled()
    prior_laws = [nominal]
    for t_shift in (F(-1, 20), F(0), F(1, 20)):
        for prior_shift in (F(-1, 20), F(0), F(1, 20)):
            if t_shift != 0 or prior_shift != 0:
                prior_laws.append(_partly_coupled(t_shift, prior_shift=prior_shift))
    return {
        "xor": {"laws": [xor], "outside": []},
        "safe_independent": {"laws": [safe_independent], "outside": []},
        "safe_correlated": {"laws": [safe_correlated], "outside": []},
        "partly_coupled": {"laws": [nominal], "outside": []},
        "rational_separation": {"laws": [rational], "outside": []},
        "finite_uncertainty": {"laws": [nominal, _partly_coupled(F(1, 20)),
                                        _partly_coupled(F(-1, 20))],
                               "outside": [_partly_coupled(F(1, 10))]},
        "unsupported": {"laws": [_partly_coupled(target_cell=F(0))], "outside": []},
        "rare": {"laws": [_partly_coupled(target_cell=F(1, 1000))], "outside": []},
        "prior_sensitivity": {"laws": prior_laws,
                              "outside": [_partly_coupled(F(1, 10), prior_shift=F(1, 10))]},
    }


def task_coefficients(law):
    """Coefficient of Q[T,Z] in binary-action Hamming loss."""
    if law.shape[-1] != 2:
        raise ValueError("binary task required")
    pty = law.sum(axis=tuple(range(law.ndim - 2)))
    return np.stack((pty[:, 1], pty[:, 0]), axis=1)


def task_cost(law, q):
    return float(np.sum(task_coefficients(law) * q))


def task_information(law, q):
    """Exact finite-law I(Y;Z|H_A) for a one-token channel."""
    joint = np.tensordot(law.sum(axis=(0, 2)), q, axes=([1], [0]))
    total = 0.0
    for ha in range(joint.shape[0]):
        table = joint[ha]
        mass = table.sum()
        if mass == 0:
            continue
        p = table / mass
        expected = p.sum(axis=1)[:, None] * p.sum(axis=0)[None, :]
        mask = p > 0
        total += mass * float(np.sum(p[mask] * np.log(p[mask] / expected[mask])))
    return total
