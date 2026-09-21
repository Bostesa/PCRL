"""Exact finite example for PCRL experiment design, not an ACS result.

Run with Python's standard library. Prints deterministic JSON.
The example is a constructed diagnostic, not a claim of novel theory.
"""

from fractions import Fraction as F
from itertools import product
import json
import math


def partitions(n):
    def rec(prefix):
        if len(prefix) == n:
            yield tuple(prefix)
            return
        for label in range(max(prefix, default=-1) + 2):
            yield from rec(prefix + [label])
    yield from rec([])


PX = [F(1, 3)] * 3
PY = [F(1, 5), F(1, 5), F(4, 5)]
PS = [F(1, 10), F(9, 10), F(9, 10)]


def joint(label_probs, kernel):
    k = len(kernel[0])
    assert all(sum(row) == 1 and all(v >= 0 for v in row) for row in kernel)
    return [[sum(PX[x] * (label_probs[x] if s else 1-label_probs[x])
                 * kernel[x][z] for x in range(3)) for z in range(k)]
            for s in range(2)]


def independent(table):
    rows = list(map(sum, table))
    cols = [sum(row[z] for row in table) for z in range(len(table[0]))]
    return all(table[s][z] == rows[s] * cols[z]
               for s in range(2) for z in range(len(cols)))


def mi(table):
    rows = list(map(sum, table))
    cols = [sum(row[z] for row in table) for z in range(len(table[0]))]
    return sum(float(p) * math.log(float(p / (rows[s] * cols[z])))
               for s, row in enumerate(table) for z, p in enumerate(row) if p)


def main():
    # S and Y are independent conditional on X; these marginals specify the joint.
    joint_xys = {(x, y, s): PX[x] * (PY[x] if y else 1-PY[x])
                 * (PS[x] if s else 1-PS[x])
                 for x, y, s in product(range(3), range(2), range(2))}
    assert sum(joint_xys.values()) == 1
    # T=P(Y=1|X) identifies {x0,x1} and {x2}; it is exactly sufficient for Y.
    assert PY[0] == PY[1]
    p_s_t0 = (PX[0]*PS[0]+PX[1]*PS[1])/(PX[0]+PX[1])
    p_s_t1 = PS[2]
    assert (p_s_t0, p_s_t1) == (F(1, 2), F(9, 10))
    # For each output z, privacy implies (-4/45)q(t0,z)+(4/45)q(t1,z)=0.
    p_s = sum(PX[x]*PS[x] for x in range(3))
    a0 = (PX[0]+PX[1])*(p_s_t0-p_s)
    a1 = PX[2]*(p_s_t1-p_s)
    assert (a0, a1) == (-F(4, 45), F(4, 45))
    assert a0 != 0 and a0 == -a1
    deterministic = []
    for labels in partitions(3):
        k = max(labels)+1
        kernel = [[F(int(labels[x] == z)) for z in range(k)] for x in range(3)]
        sy, yy = joint(PS, kernel), joint(PY, kernel)
        private = independent(sy)
        if private:
            assert independent(yy) and k == 1
        deterministic.append({"partition": list(labels), "exactly_private": private,
                              "task_information_nats": mi(yy),
                              "sensitive_information_nats": mi(sy)})
    assert len(deterministic) == 5
    assert sum(d["exactly_private"] for d in deterministic) == 1
    q_one = [F(1, 2), F(0), F(1)]
    kernel = [[1-q, q] for q in q_one]
    sy, yy = joint(PS, kernel), joint(PY, kernel)
    assert independent(sy) and not independent(yy)
    p_z1 = sum(row[1] for row in yy)
    assert p_s == F(19, 30) and p_z1 == F(1, 2)
    assert sy[1][1] == F(19, 60)
    assert yy[1][1]/p_z1 == F(3, 5)
    assert yy[1][0]/(1-p_z1) == F(1, 5)
    assert math.isclose(mi(yy), 0.0863046217355342, abs_tol=1e-14)
    # The public kernel row is NOT the sampled token. Distinct rows reveal X.
    row_release = [[F(int(x == z)) for z in range(3)] for x in range(3)]
    assert not independent(joint(PS, row_release))
    result = {
        "scope": "Exact constructed finite distribution; no ACS data or novelty claim",
        "probability_arithmetic": "fractions.Fraction; logarithms only for displayed MI",
        "coarse_posterior_code_sufficient_for_Y": True,
        "coarse_private_kernel_requires_identical_rows_for_every_output": True,
        "coarse_privacy_equation_coefficients": [str(a0), str(a1)],
        "deterministic_partitions": deterministic,
        "refined_stochastic_channel": {
            "P_Z1_given_X": [str(q) for q in q_one],
            "P_S1": str(p_s), "P_Z1": str(p_z1), "P_S1_Z1": str(sy[1][1]),
            "exact_independence_S_Z": True,
            "P_Y1_given_Z0": "1/5", "P_Y1_given_Z1": "3/5",
            "task_information_nats": mi(yy), "sensitive_information_nats": mi(sy),
            "releasing_kernel_row_breaks_privacy": True,
        },
        "conclusion": "Prediction sufficiency need not preserve privacy-utility options; ACS benefit remains untested",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
