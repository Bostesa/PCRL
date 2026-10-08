"""Frozen common-law checks for Team A's finite score channel.

The five symbolic laws are specified in shared/SYNTHETIC_LAWS.json. These
constructors enumerate their atoms directly; no Adult data are read.
"""

from itertools import product

import numpy as np
import pytest

from finite_channel import (
    FiniteLaw,
    class_floor,
    evaluate,
    input_recovery,
    sample_release,
    solve_coordinate,
)


def _common_law(kind):
    # Tokens are (task bit, clue bit), except L1, which uses just task bits.
    n = 2 if kind == "L1" else 4
    mass = np.zeros((2, n, n, 2, 2), dtype=float)
    if kind == "L1":
        for s, b1, b2 in product(range(2), repeat=3):
            mass[s, b1, b2, b1, b2] += 1 / 8
        classes = np.array([0, 1])
        bank = np.array([[.9, .1], [.1, .9]])
    elif kind in ("L2", "L3", "L4"):
        if kind == "L3":
            atoms = ((s, b1, b2, u, 1 / 16)
                     for s, b1, b2, u in product(range(2), repeat=4))
        elif kind == "L4":
            atoms = ((s, b1, b2, s ^ e, (1 / 8) * (.8 if e == 0 else .2))
                     for s, b1, b2, e in product(range(2), repeat=4))
        else:
            atoms = ((s, b1, b2, s, 1 / 8)
                     for s, b1, b2 in product(range(2), repeat=3))
        for s, b1, b2, c, w in atoms:
            c2 = c ^ s if kind == "L3" else c
            mass[s, 2 * b1 + c, 2 * b2 + c2, b1, b2] += w
        classes = np.array([0, 0, 1, 1])
        bank = np.array([[.8, .2], [.9, .1], [.2, .8], [.1, .9]])
    else:
        raise ValueError(kind)
    return FiniteLaw(mass, bank, bank, classes, classes)


def _l5_confidence():
    # Shared L5: S=H; Y1|H has rates .60/.95; Y2 is an independent fair bit.
    mass = np.zeros((2, 2, 2, 2, 2), dtype=float)
    for h, y2 in product(range(2), repeat=2):
        p1 = .60 if h == 0 else .95
        for y1 in range(2):
            mass[h, h, y2, y1, y2] += .25 * (p1 if y1 else 1 - p1)
    bank1 = np.array([[.40, .60], [.05, .95]])
    bank2 = np.array([[.90, .10], [.10, .90]])
    return FiniteLaw(mass, bank1, bank2,
                     np.array([1, 1]), np.array([0, 1]))


def _identity(law):
    return np.eye(law.n_input[0], law.n_output[0]), np.eye(law.n_input[1], law.n_output[1])


@pytest.mark.parametrize("kind,local,pair", [
    ("L1", .5, .5), ("L2", 1.0, 1.0), ("L3", .5, 1.0), ("L4", .8, .8),
])
def test_frozen_common_law_recovery(kind, local, pair):
    law = _common_law(kind)
    q1, q2 = _identity(law)
    assert np.isclose(law.mass.sum(), 1)
    assert np.isclose(class_floor(law)["pair"], .5)
    assert np.isclose(input_recovery(law)["pair"], pair)
    result = evaluate(law, q1, q2)
    assert np.isclose(result["accuracy"]["r1"], local)
    assert np.isclose(result["accuracy"]["r2"], local)
    assert np.isclose(result["accuracy"]["pair"], pair)


def test_removable_distinction_keeps_task_outputs_and_erases_sex():
    law = _common_law("L2")
    identity = _identity(law)
    # Keep the task bit and discard the SEX clue; same-class bank probabilities.
    q = np.zeros((4, 4))
    q[[0, 1, 2, 3], [0, 0, 2, 2]] = 1
    before = evaluate(law, *identity)
    after = evaluate(law, q, q)
    assert np.isclose(after["accuracy"]["pair"], .5)
    assert np.isclose(after["accuracy"]["r1"], .5)
    # The frozen score bank differs with clue, so task loss may change; this
    # assertion checks exact loss accounting, not a claimed utility pass.
    assert np.isfinite(after["utility"]["r1"]["nll"])
    assert before["accuracy"]["pair"] == 1


def test_confidence_is_useful_even_when_decision_is_constant():
    law = _l5_confidence()
    q1, q2 = _identity(law)
    variable = evaluate(law, q1, q2)
    # The class-only probability .775 is not in the output bank; enumerate its
    # proper loss directly to test utility eligibility of the privacy floor.
    p = .775
    class_nll = -.775 * np.log(p) - .225 * np.log(1 - p)
    class_brier = 2 * (.775 * (1 - p) ** 2 + .225 * p ** 2)
    assert np.isclose(class_floor(law)["r1"], .5)
    assert variable["utility"]["r1"]["nll"] < class_nll - .01
    assert variable["utility"]["r1"]["brier"] < class_brier - .005


def test_coordinate_lp_respects_mean_caps_and_class_locality():
    law = _common_law("L2")
    q1, q2 = _identity(law)
    baseline = evaluate(law, q1, q2)["utility"]
    # Loose cap admits a class-preserving privacy repair in one coordinate.
    caps = {r: {m: baseline[r][m] + 1.0 for m in ("nll", "brier")}
            for r in ("r1", "r2")}
    fit = solve_coordinate(law, q1, q2, recipient="r1", caps=caps)
    assert fit["success"]
    assert np.allclose(fit["q"].sum(axis=1), 1, atol=1e-8)
    assert np.all(fit["q"] >= -1e-9)
    forbidden = law.input_class[0][:, None] != law.output_class[0][None, :]
    assert np.all(fit["q"][forbidden] == 0)
    measured = evaluate(law, fit["q"], q2)
    assert measured["utility"]["r1"]["nll"] <= caps["r1"]["nll"] + 1e-8
    assert measured["utility"]["r1"]["brier"] <= caps["r1"]["brier"] + 1e-8
    assert measured["accuracy"]["pair"] >= measured["accuracy"]["r2"] - 1e-10


def test_sampled_release_is_actual_bank_vector_and_code_id():
    law = _l5_confidence()
    q1, _ = _identity(law)
    released = sample_release(law, recipient="r1", input_tokens=np.array([0, 1]),
                              q=q1, rng=np.random.default_rng(7))
    assert np.array_equal(released["code_id"], np.array([0, 1]))
    assert np.array_equal(released["probability"], law.bank[0])
    assert np.array_equal(released["decision"], np.array([1, 1]))
