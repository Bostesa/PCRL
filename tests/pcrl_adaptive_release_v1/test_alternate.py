import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1.alternate import (
    coverage_law,
    select_frozen_decoder,
    channel_update,
)


class FixedDecoder:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_token_proba(self, h, n_tokens):
        return np.repeat(self.probabilities[None, :, :], len(h), axis=0)


def test_coverage_is_training_law_and_exact_expected_decoder_selection():
    q = np.array([[1., 0.], [0., 1.]])
    ref = np.full((2, 2), 0.5)
    t = np.array([0, 1])
    law = coverage_law(t, q, ref)
    assert np.allclose(law, (q[t] + ref[t] + 0.5) / 3)
    h = np.zeros((2, 4))
    y = np.array([0, 1])
    good = FixedDecoder([[0.9, 0.1], [0.1, 0.9]])
    bad = FixedDecoder([[0.1, 0.9], [0.9, 0.1]])
    chosen = select_frozen_decoder({"good": good, "bad": bad}, h, t, y,
                                   np.ones(2), q)
    assert chosen["id"] == "good"
    assert chosen["scores"]["good"]["U"] == pytest.approx(-np.log(0.9))
    assert chosen["scores"]["bad"]["U"] == pytest.approx(-np.log(0.1))


def test_channel_update_uses_separate_u_w_cost_and_calibrated_bank():
    ref = np.eye(2)
    c_u = np.array([[0., 1.], [1., 0.]])
    c_w = np.array([[0.2, 0.], [0., 0.2]])
    bank = [{"id": "h", "role": "A/SEX", "weighting": "U",
             "coeff": np.full((2, 2), 0.25),
             "coefficient_pool_sha256": "same", "class_order": [0, 1],
             "weight_normalization": "1/n"}]
    record = channel_update({"U": c_u, "W": c_w}, ref, bank, 0.001)
    assert record["solution"]["feasible"]
    assert record["cost_U"][0, 0] == 0
    assert record["cost_W"][0, 0] == pytest.approx(0.2)
    assert record["solution"]["replay"]["maximum_cut_violation"] < 1e-8
