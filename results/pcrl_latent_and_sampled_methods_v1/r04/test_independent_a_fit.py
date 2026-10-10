"""Tiny exact checks for expected sampled output and protected FIT arithmetic."""

from __future__ import annotations

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r04.independent_a_fit import expected_fit_metrics
from results.pcrl_latent_and_sampled_methods_v1.r04.independent_a_inputs import AdmittedARole
from results.pcrl_latent_and_sampled_methods_v1.r04.independent_a_law import IndependentAPolicy


def test_expected_two_output_law_and_exact_public_bayes():
    score = np.array([[.8, .2], [.8, .2]], dtype=np.float64)
    bank = np.array([[.8, .2], [.9, .1]], dtype=np.float64)
    policy = IndependentAPolicy(
        {"law_kind": "stochastic_joint_v1"},
        {"bank1_probability": bank, "bank2_probability": bank,
         "bank1_decision": np.zeros(2, np.int64),
         "bank2_decision": np.zeros(2, np.int64),
         "w1": np.zeros((8, 16)), "b1": np.zeros(16),
         "w2": np.zeros((16, 16)), "b2": np.zeros(16)},
        ((0, 1),), ((0, 1),))
    source = AdmittedARole(
        0, "OSF_DEFENSE_FIT", np.array([0, 1], np.int64),
        np.array([0, 1], np.int64), score, score,
        np.zeros(2, np.int64), np.zeros(2, np.int64),
        np.zeros(2, np.int64), np.zeros(2, np.int64), {}, {}, {},
        {"income": np.zeros(2, np.int64), "occupation": np.zeros(2, np.int64)},
        {}, "a" * 64, "b" * 64)
    metrics = expected_fit_metrics(policy, source, np.array([0, 1], np.int64))
    assert abs(metrics["r1_nll"] - (-np.log(.8) - np.log(.9)) / 2) < 1e-14
    assert abs(metrics["r1_brier"] - (.08 + .02) / 2) < 1e-14
    assert abs(metrics["r1_bayes"] - .5) < 1e-14
    assert abs(metrics["pair_bayes"] - .5) < 1e-14
    assert abs(metrics["protected_objective"] - 1.0) < 1e-14
