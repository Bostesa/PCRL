"""F10 -- scope of statements made conditional on a COARSENED output.

Exact enumeration over small discrete joint distributions (no sampling).
Y is the full model output (a score with 4 values); C = 1{Y > 0.5} is the
coarsened output (the released / thresholded label). S is the protected bit,
Z an additional released representation. Mutual information in bits.

E1 (output leak hidden by coarsening):    I(S; C) = 0   but  I(S; Y) > 0.
E2 (residual leak beyond the output, vanishes with the full output):
      Z = Y:   I(S; Z | C) > 0   but  I(S; Z | Y) = 0.
E3 (the reverse, by explaining-away):
      S, Z independent fair bits, Y = (S xor Z, noise-free) and C constant:
      I(S; Z | C) = 0   but  I(S; Z | Y) = 1 bit.
Also reported: Bayes accuracy of guessing S from C versus from Y in E1.
"""
import itertools
import json
from collections import defaultdict
from math import log2


def H(p):
    return -sum(v * log2(v) for v in p.values() if v > 0)


def marg(joint, idx):
    m = defaultdict(float)
    for k, v in joint.items():
        m[tuple(k[i] for i in idx)] += v
    return m


def MI(joint, a, b, cond=()):
    # I(A;B|C) = H(A,C) + H(B,C) - H(A,B,C) - H(C)
    return (H(marg(joint, a + cond)) + H(marg(joint, b + cond)) - H(marg(joint, a + b + cond))
            - (H(marg(joint, cond)) if cond else 0.0))


def bayes_acc(joint, target, given):
    m = defaultdict(lambda: defaultdict(float))
    for k, v in joint.items():
        m[tuple(k[i] for i in given)][k[target]] += v
    return sum(max(d.values()) for d in m.values())


def main():
    # E1/E2: Y in {.2,.4,.6,.8} equally likely; P(S=1|Y) = .8,.2,.8,.2 ; C = 1{Y>.5}; Z = Y
    pS1 = {0.2: 0.8, 0.4: 0.2, 0.6: 0.8, 0.8: 0.2}
    joint = {}  # key = (S, Y, C, Z)
    for y, p1 in pS1.items():
        for s in (0, 1):
            joint[(s, y, int(y > 0.5), y)] = 0.25 * (p1 if s else 1 - p1)
    S, Y, C, Z = (0,), (1,), (2,), (3,)
    e1 = {"I_S_C": MI(joint, S, C), "I_S_Y": MI(joint, S, Y),
          "bayes_acc_S_from_C": bayes_acc(joint, 0, (2,)), "bayes_acc_S_from_Y": bayes_acc(joint, 0, (1,))}
    e2 = {"I_S_Z_given_C": MI(joint, S, Z, C), "I_S_Z_given_Y": MI(joint, S, Z, Y)}
    # E3: S,Z iid fair bits; Y = S xor Z; C constant 0
    j3 = {}
    for s, z in itertools.product((0, 1), repeat=2):
        j3[(s, z, s ^ z, 0)] = 0.25
    e3 = {"I_S_Z": MI(j3, (0,), (1,)), "I_S_Z_given_C": MI(j3, (0,), (1,), (3,)),
          "I_S_Z_given_Y": MI(j3, (0,), (1,), (2,))}
    print(json.dumps({"id": "F10", "E1_output_leak_hidden_by_coarsening": e1,
                      "E2_residual_beyond_coarse_output_vanishes_given_full_output": e2,
                      "E3_conditioning_on_full_output_creates_dependence": e3}, indent=1))


if __name__ == "__main__":
    main()
