"""Exact, data-free finite-law oracle for public hard codes and sampled scores.

This deliberately imports no candidate, runner, or common endpoint code. Small
enumerations use rational arithmetic except for natural-log NLL.
"""

from __future__ import annotations

import math
from fractions import Fraction as F
from typing import Callable, Hashable, Sequence


def binary_hamming_kernel(left: Sequence[int], right: Sequence[int], rho: F) -> F:
    """Rational binary kernel rho ** Hamming(left, right), 0 < rho <= 1."""
    if not left or len(left) != len(right) or any(bit not in (0, 1) for bit in (*left, *right)):
        raise ValueError("equal nonempty hard-binary codes required")
    rho = F(rho)
    if not 0 < rho <= 1:
        raise ValueError("rho must be in (0, 1]")
    return rho ** sum(a != b for a, b in zip(left, right))


def empirical_biased_mmd2(
    group0: Sequence[Hashable], group1: Sequence[Hashable], kernel: Callable[[Hashable, Hashable], F]
) -> F:
    """Biased MMD² with all ordered pairs and within-group diagonals."""
    if not group0 or not group1:
        raise ValueError("both groups must be nonempty")
    n0, n1 = len(group0), len(group1)
    k00 = sum((F(kernel(a, b)) for a in group0 for b in group0), F(0)) / (n0 * n0)
    k11 = sum((F(kernel(a, b)) for a in group1 for b in group1), F(0)) / (n1 * n1)
    k01 = sum((F(kernel(a, b)) for a in group0 for b in group1), F(0)) / (n0 * n1)
    return k00 + k11 - 2 * k01


def public_output_key(probabilities: Sequence[F]) -> tuple[tuple[F, ...], int]:
    """Canonical public identity: exact probability vector and first argmax."""
    vector = tuple(F(value) for value in probabilities)
    if len(vector) < 2 or any(value < 0 or value > 1 for value in vector) or sum(vector) != 1:
        raise ValueError("a normalized multiclass probability vector is required")
    decision = max(range(len(vector)), key=lambda index: vector[index])
    return vector, decision


def canonical_public_bank(vectors: Sequence[Sequence[F]]) -> dict[tuple[tuple[F, ...], int], int]:
    """Assign IDs by global public value, independent of token/action order."""
    keys = sorted({public_output_key(vector) for vector in vectors})
    if not keys:
        raise ValueError("bank must be nonempty")
    return {key: index for index, key in enumerate(keys)}


def deduplicate_before_normalizing(
    choices: Sequence[Sequence[F]], unnormalized_masses: Sequence[F]
) -> tuple[tuple[tuple[F, ...], ...], tuple[F, ...]]:
    """Keep the first instance of each public choice, then normalize masses.

    This is a tiny oracle for a fixed first-occurrence convention. It does not
    authorize using input-local action positions as transmitted identities.
    """
    if not choices or len(choices) != len(unnormalized_masses):
        raise ValueError("one nonnegative mass per nonempty choice is required")
    unique: dict[tuple[tuple[F, ...], int], F] = {}
    for choice, raw_mass in zip(choices, unnormalized_masses):
        key = public_output_key(choice)
        mass = F(raw_mass)
        if mass < 0:
            raise ValueError("negative action mass")
        if key not in unique:
            unique[key] = mass
    total = sum(unique.values(), F(0))
    if total <= 0:
        raise ValueError("deduplicated law has zero total mass")
    return tuple(key[0] for key in unique), tuple(mass / total for mass in unique.values())


def _validated_channel(
    weights: Sequence[F], vectors: Sequence[Sequence[F]], label: int
) -> tuple[tuple[F, ...], tuple[tuple[F, ...], ...]]:
    if not weights or len(weights) != len(vectors):
        raise ValueError("one probability per nonempty public choice is required")
    q = tuple(F(value) for value in weights)
    p = tuple(public_output_key(vector)[0] for vector in vectors)
    if any(value < 0 for value in q) or sum(q) != 1 or any(len(vector) != len(p[0]) for vector in p):
        raise ValueError("normalized channel and common class width required")
    if not isinstance(label, int) or not 0 <= label < len(p[0]):
        raise ValueError("label outside class range")
    return q, p


def expected_sampled_cost(
    weights: Sequence[F], vectors: Sequence[Sequence[F]], label: int, clip: float = 1e-12
) -> tuple[float, F]:
    """Expected one-draw natural-log NLL and summed multiclass Brier."""
    q, p = _validated_channel(weights, vectors, label)
    if not math.isfinite(clip) or clip <= 0:
        raise ValueError("positive finite NLL clip required")
    nll = sum(float(weight) * -math.log(max(float(vector[label]), clip)) for weight, vector in zip(q, p))
    brier = sum((weight * sum(((value - int(k == label)) ** 2 for k, value in enumerate(vector)), F(0))
                 for weight, vector in zip(q, p)), F(0))
    return nll, brier


def loss_of_averaged_vector(
    weights: Sequence[F], vectors: Sequence[Sequence[F]], label: int, clip: float = 1e-12
) -> tuple[float, F]:
    """Diagnostic only: cost of a mean vector the sampled recipient did not get."""
    q, p = _validated_channel(weights, vectors, label)
    average = tuple(sum((weight * vector[k] for weight, vector in zip(q, p)), F(0)) for k in range(len(p[0])))
    return expected_sampled_cost((F(1),), (average,), label, clip)


def aggregate_public_pairs(
    actions: Sequence[tuple[F, Sequence[F], Sequence[F]]],
    bank1: dict[tuple[tuple[F, ...], int], int],
    bank2: dict[tuple[tuple[F, ...], int], int],
) -> dict[tuple[int, int], F]:
    """Sum joint action mass by the actual globally named public output pair."""
    if not actions or sum((F(mass) for mass, _, _ in actions), F(0)) != 1:
        raise ValueError("joint actions must have unit total mass")
    law: dict[tuple[int, int], F] = {}
    for mass, output1, output2 in actions:
        mass = F(mass)
        if mass < 0:
            raise ValueError("negative joint action mass")
        key = (bank1[public_output_key(output1)], bank2[public_output_key(output2)])
        law[key] = law.get(key, F(0)) + mass
    return law


def empirical_public_pair_laws(
    rows: Sequence[tuple[int, Sequence[tuple[F, Sequence[F], Sequence[F]]]]],
    bank1: dict[tuple[tuple[F, ...], int], int],
    bank2: dict[tuple[tuple[F, ...], int], int],
) -> dict[int, dict[tuple[int, int], F]]:
    """Average each person's public action law within the two SEX groups."""
    sums: dict[int, dict[tuple[int, int], F]] = {0: {}, 1: {}}
    counts = {0: 0, 1: 0}
    for sex, actions in rows:
        if sex not in (0, 1):
            raise ValueError("SEX must be binary")
        counts[sex] += 1
        for key, mass in aggregate_public_pairs(actions, bank1, bank2).items():
            sums[sex][key] = sums[sex].get(key, F(0)) + mass
    if not all(counts.values()):
        raise ValueError("both SEX groups must be nonempty")
    return {sex: {key: mass / counts[sex] for key, mass in sums[sex].items()} for sex in (0, 1)}
