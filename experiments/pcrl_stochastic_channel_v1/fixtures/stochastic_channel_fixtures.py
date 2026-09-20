"""Exact finite-distribution fixtures for PCRL design review; no ACS data.

Run with Python 3.10+: python stochastic_channel_fixtures.py
Only the standard library is used. Rational arithmetic establishes independence;
floating-point logarithms report mutual information in nats. These are mathematical
counterexamples, not empirical evidence that the mechanism works on PCRL data.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path


def information(joint, axes_a, axes_b):
    """Mutual information under a finite exact rational joint distribution."""
    pa, pb, pab = defaultdict(F), defaultdict(F), defaultdict(F)
    for state, mass in joint.items():
        a = tuple(state[i] for i in axes_a)
        b = tuple(state[i] for i in axes_b)
        pa[a] += mass
        pb[b] += mass
        pab[a, b] += mass
    return math.fsum(float(p) * math.log(float(p / (pa[a] * pb[b])))
                     for (a, b), p in pab.items() if p)


def independent(joint, axes_a, axes_b):
    pa, pb, pab = defaultdict(F), defaultdict(F), defaultdict(F)
    for state, mass in joint.items():
        a = tuple(state[i] for i in axes_a)
        b = tuple(state[i] for i in axes_b)
        pa[a] += mass
        pb[b] += mass
        pab[a, b] += mass
    return all(pab[a, b] == pa[a] * pb[b] for a in pa for b in pb)


def conditional_information(joint, axes_a, axes_b, axes_c):
    groups, pc = defaultdict(dict), defaultdict(F)
    for state, mass in joint.items():
        c = tuple(state[i] for i in axes_c)
        groups[c][state] = mass
        pc[c] += mass
    return math.fsum(float(pc[c]) * information(
        {state: p / pc[c] for state, p in sub.items()}, axes_a, axes_b)
        for c, sub in groups.items() if pc[c])


def partitions(n):
    """Canonical restricted-growth strings enumerate every deterministic channel.

    Any deterministic map from n states induces one of these partitions, even if
    its nominal output alphabet is larger than n.
    """
    def extend(prefix):
        if len(prefix) == n:
            yield tuple(prefix)
            return
        for label in range(max(prefix) + 2):
            yield from extend(prefix + [label])
    yield from extend([0])


def joint_for_channel(s_probs, q_probs):
    joint = {}
    for t in range(3):
        for s in range(2):
            for z in range(2):
                p = F(1, 3) * (s_probs[t] if s else 1 - s_probs[t])
                p *= q_probs[t] if z else 1 - q_probs[t]
                if p:
                    joint[t, s, int(t == 1), z] = p
    assert sum(joint.values()) == 1
    return joint


def check_toy(name, s_probs, q_probs):
    joint = joint_for_channel(s_probs, q_probs)
    deterministic = []
    for labels in partitions(3):
        exact = {}
        for t in range(3):
            for s in range(2):
                p = F(1, 3) * (s_probs[t] if s else 1 - s_probs[t])
                if p:
                    exact[t, s, int(t == 1), labels[t]] = p
        deterministic.append({
            "partition": labels,
            "exact_private": independent(exact, [1], [3]),
            "sensitive_information_nats": information(exact, [1], [3]),
            "task_information_nats": information(exact, [2], [3]),
        })
    assert len(deterministic) == 5
    feasible = [d for d in deterministic if d["exact_private"]]
    assert len(feasible) == 1 and feasible[0]["partition"] == (0, 0, 0)
    assert feasible[0]["task_information_nats"] == 0
    assert independent(joint, [1], [3])
    utility = information(joint, [2], [3])
    assert utility > 0.3

    # Publishing q(.|t) instead of sampling Z reveals t, because its three values
    # are distinct. The privacy property belongs to the random release, not logits.
    assert len(set(q_probs)) == 3
    probability_vector_leak = information(joint, [1], [0])
    assert probability_vector_leak > 0

    return {
        "name": name,
        "input_probability": ["1/3"] * 3,
        "sensitive_probability_given_input": list(map(str, s_probs)),
        "release_one_probability_given_input": list(map(str, q_probs)),
        "target": "Y = 1[T == 1]",
        "random_release_sensitive_independence_exact": True,
        "random_release_sensitive_information_nats": information(joint, [1], [3]),
        "random_release_task_information_nats": utility,
        "all_deterministic_partitions": deterministic,
        "largest_exact_private_deterministic_task_information_nats": 0,
        "releasing_probability_vector_sensitive_information_nats": probability_vector_leak,
    }


def xor_counterexamples():
    # A sees Z; B sees C. S and C are independent fair bits, Z=S xor C.
    joint = {(s, c, s ^ c): F(1, 4) for s in range(2) for c in range(2)}
    assert independent(joint, [0], [1])
    assert independent(joint, [0], [2])
    local = information(joint, [0], [2])
    coalition = conditional_information(joint, [0], [2], [1])
    assert local == 0 and abs(coalition - math.log(2)) < 1e-15

    # Converse: B already sees S; the appended Z=S adds nothing to B but reveals
    # S to A. Conditional privacy for a coalition does not imply local privacy.
    reverse = {(s, s, s): F(1, 2) for s in range(2)}
    reverse_local = information(reverse, [0], [2])
    reverse_coalition = conditional_information(reverse, [0], [2], [1])
    assert abs(reverse_local - math.log(2)) < 1e-15
    assert reverse_coalition == 0

    # In the first example, treating the true baseline C as a constant coarse bin
    # falsely certifies I(S;Z|bin(C))=0 while true baseline-conditioned leakage=ln2.
    return {
        "individually_private_jointly_disclosing": {
            "definition": "S,C independent fair bits; Z=S xor C",
            "A_increment_nats": local,
            "coalition_increment_given_C_nats": coalition,
            "increment_conditioned_on_constant_context_bin_nats": local,
        },
        "coalition_increment_zero_local_increment_positive": {
            "definition": "S fair bit; B=S; Z=S",
            "A_increment_nats": reverse_local,
            "coalition_increment_given_B_nats": reverse_coalition,
        },
    }


def reconstruction_counterexample():
    # Y and S independent centered unit-variance signs. Teacher=(Y,10*S).
    # One-channel candidate Z=S achieves teacher MSE 1, while Z=Y achieves MSE
    # 100. Teacher fidelity explicitly rewards the sensitive coordinate here.
    joint = {(y, s): F(1, 4) for y in (-1, 1) for s in (-1, 1)}
    assert independent(joint, [0], [1])
    return {
        "definition": "Independent Y,S in {-1,+1}; teacher=(Y,10*S)",
        "release_Y": {"teacher_total_squared_error": 100,
                      "task_information_nats": math.log(2),
                      "sensitive_information_nats": 0},
        "release_S": {"teacher_total_squared_error": 1,
                      "task_information_nats": 0,
                      "sensitive_information_nats": math.log(2)},
        "scope": "Counterexample to teacher fidelity implying useful private transfer; not a diagnosis of ACS.",
    }


def main():
    results = {
        "scope": "Synthetic exact finite models only; no ACS or protected evaluation data read.",
        "units": "Natural-log information (nats). Independence uses exact fractions.",
        "toys": [
            check_toy("rational_symmetric_output", [F(0), F(1, 3), F(1)],
                      [F(5, 6), F(0), F(2, 3)]),
            check_toy("strictly_interior_sensitive_probabilities",
                      [F(1, 10), F(4, 10), F(9, 10)],
                      [F(1), F(0), F(11, 13)]),
        ],
        "view_and_coarsening_counterexamples": xor_counterexamples(),
        "utility_proxy_counterexample": reconstruction_counterexample(),
        "interpretation": [
            "Stochastic channels can strictly improve over every deterministic channel under exact privacy in a finite model.",
            "This does not show that randomization improves the ACS PCRL frontier.",
            "A release probability, its logits, repeated independent samples, or joint use of two private channels need separate audits.",
            "Conditional privacy for coarsened contexts does not establish conditional privacy for the actual released context.",
            "Training with Y labels creates a supervised diagnostic, not evidence of held-out-task transfer.",
        ],
    }
    source = Path(__file__)
    results["script_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    target = source.with_suffix(".json")
    target.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"status": "all assertions passed", "output": str(target),
                      "toy_task_information_nats": [x["random_release_task_information_nats"]
                                                    for x in results["toys"]]}, indent=2))


if __name__ == "__main__":
    main()
