"""Data-free arithmetic checks for actual emitted A laws."""

from __future__ import annotations

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r04.independent_a_law import IndependentAPolicy


def _toy(kind: str, *, order: str = "12") -> IndependentAPolicy:
    bank = np.array([[0.8, 0.2], [0.9, 0.1]], dtype=np.float64)
    arrays = {"bank1_probability": bank, "bank2_probability": bank,
              "bank1_decision": np.zeros(2, dtype=np.int64),
              "bank2_decision": np.zeros(2, dtype=np.int64)}
    if kind in ("stochastic_joint_v1", "deterministic_argmax_v1"):
        arrays.update({"w1": np.zeros((8, 16)), "b1": np.zeros(16),
                       "w2": np.zeros((16, 16)), "b2": np.zeros(16)})
    else:
        for prefix in (("factor1", "factor2") if kind in
                       ("central_product_v1", "strict_local_product_v1")
                       else ("first", "second")):
            width = (4 if kind == "strict_local_product_v1" else 8)
            arrays.update({f"{prefix}_w1": np.zeros((width, 16)),
                           f"{prefix}_b1": np.zeros(16),
                           f"{prefix}_w2": np.zeros((16, 4)),
                           f"{prefix}_b2": np.zeros(4)})
        if kind == "freeze_first_conditional_v1":
            arrays["second_embedding"] = np.zeros((2, 16))
    return IndependentAPolicy({"law_kind": kind, "mechanism_descriptor": {"order": order}},
                              arrays, ((0, 1),), ((0, 1),))


def test_uniform_stochastic_product_local_sequential_laws():
    score = np.array([0.85, 0.15], dtype=np.float64)
    for kind in ("stochastic_joint_v1", "central_product_v1",
                 "strict_local_product_v1", "freeze_first_conditional_v1"):
        law = _toy(kind).law(score, score, 0, 0)
        assert [(i, j) for i, j, _ in law] == [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert np.allclose([p for _, _, p in law], 0.25)
    assert _toy("deterministic_argmax_v1").law(score, score, 0, 0) == ((0, 0, 1.0),)


def test_sequential_draw_uses_actual_first_code():
    score = np.array([0.85, 0.15], dtype=np.float64)
    # Order 21 spends the uniform on recipient 2, then conditionally on r1.
    # At u=.375 this yields pair (1,0); a flat sorted-pair CDF yields (0,1).
    policy = _toy("freeze_first_conditional_v1", order="21")
    assert policy.sample(score, score, 0, 0, 0.375) == (1, 0)


def test_expected_loss_scores_emitted_vectors():
    score = np.array([0.85, 0.15], dtype=np.float64)
    policy = _toy("stochastic_joint_v1")
    law = policy.law(score, score, 0, 0)
    actual = policy.expected_losses(law, 0, 0)
    expected_brier = 0.5 * (2 * .2 ** 2 + 2 * .1 ** 2)
    expected_nll = -.5 * (np.log(.8) + np.log(.9))
    assert abs(actual["r1"]["brier"] - expected_brier) < 1e-14
    assert abs(actual["r1"]["nll"] - expected_nll) < 1e-14
    assert actual["r1"]["brier"] > 2 * .15 ** 2
