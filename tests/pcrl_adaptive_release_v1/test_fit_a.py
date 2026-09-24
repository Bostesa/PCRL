import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1.fit_a import (
    aggregate_loss_coefficients,
    build_attack_bank,
    select_final_bank_feasible,
    load_frozen_bank,
    person_loss_from_attack,
    run_center_from_roles,
    source_laws,
    task_person_losses,
)


class FixedTask:
    def predict_token_proba(self, h, n_tokens):
        p = np.full((len(h), n_tokens, 2), 0.5)
        p[:, 0, :] = (0.9, 0.1)
        p[:, 1, :] = (0.2, 0.8)
        return p


def test_original_person_weighting_and_no_second_state_mass():
    codes = np.array([0, 0, 1])
    loss = np.array([[1., 2.], [3., 4.], [5., 6.]])
    pair = aggregate_loss_coefficients(codes, loss, np.array([1., 2., 3.]), 2)
    q = np.array([[0.25, 0.75], [0.6, 0.4]])
    person = np.sum(q[codes] * loss, axis=1)
    assert np.sum(pair['U'] * q) == pytest.approx(person.mean())
    assert np.sum(pair['W'] * q) == pytest.approx(np.dot(person, [1/6, 2/6, 3/6]))


def test_source_laws_keep_h_only_one_token_and_17_token_coverage():
    codes = np.array([0, 1])
    d17 = np.zeros((2, 17)); d17[0, 0] = 1; d17[1, 1] = 1
    hist = np.zeros((2, 17)); hist[0, 2] = 1; hist[1, 3] = 1
    laws = source_laws(codes, d17, hist)
    assert laws['H'].shape == (2, 1)
    assert np.array_equal(laws['H'], np.ones((2, 1)))
    assert np.array_equal(laws['D17'], d17[codes])
    assert np.array_equal(laws['Q'], hist[codes])
    assert np.allclose(laws['coverage'], (d17[codes] + hist[codes] + 1/17) / 3)


def test_task_loss_is_expected_loss_of_each_token_not_averaged_prediction():
    rows = {'ha': np.zeros((2, 4)), 'token_codes': np.array([0, 1]),
            'labels': {'same_residence': np.array([0, 1])},
            'weights': np.array([1., 1.])}
    valid, losses = task_person_losses(FixedTask(), rows, n_tokens=2)
    assert valid.tolist() == [True, True]
    assert losses.shape == (2, 2)
    assert losses[0, 0] == pytest.approx(-np.log(0.9))
    assert losses[0, 1] == pytest.approx(-np.log(0.2))
    q = np.full((2, 2), .5)
    exact = np.sum(q * losses, axis=1)
    assert exact[0] > -np.log(.55)


def _toy_role_rows(seed, count, offset):
    rng = np.random.default_rng(seed)
    ha = rng.normal(size=(count, 4))
    return {"ha": ha, "hb": rng.normal(size=(count, 2)),
            "token_codes": rng.integers(0, 2, size=count, dtype=np.int64),
            "labels": {"SEX": np.tile([0, 1], count // 2),
                       "RAC1P": np.arange(count) % 9,
                       "same_residence": (ha[:, 0] > 0).astype(int)},
            "weights": np.ones(count),
            "ids": np.arange(offset, offset + count),
            "households": np.asarray([f"h{offset + i}" for i in range(count)])}


def test_real_fitted_h_ancestor_remains_action_invariant_after_banking(tmp_path):
    role_dict = {"audit_fit": _toy_role_rows(11, 40, 0),
                 "inner_selection": _toy_role_rows(12, 20, 1000),
                 "coefficient_split": _toy_role_rows(13, 30, 2000)}
    d17 = np.zeros((32, 17)); d17[:, 0] = 1
    sources = {"H": None, "D17": d17}
    bank = build_attack_bank(role_dict, sources, tmp_path / "private" / "bank",
                             seed=41, target_roles=("A/SEX",))
    assert bank["cuts"]
    assert {cut["weighting"] for cut in bank["cuts"]} == {"U", "W"}
    assert all(cut["coeff"].shape == (32, 17) for cut in bank["cuts"])
    h_spec = next(spec for spec in bank["attack_specs"] if spec["wire"] == "H")
    valid, losses = person_loss_from_attack(h_spec, role_dict["coefficient_split"])
    assert valid.sum() == 30
    assert np.array_equal(losses, np.repeat(losses[:, :1], 17, axis=1))
    reloaded = load_frozen_bank(tmp_path / "private" / "bank")
    assert len(reloaded["cuts"]) == len(bank["cuts"])
    assert np.array_equal(reloaded["cuts"][0]["coeff"], bank["cuts"][0]["coeff"])
    assert reloaded["attack_specs"][0]["model_sha256"] == bank["attack_specs"][0]["model_sha256"]


def test_synthetic_fit_to_calibrated_release_and_immutable_resume(tmp_path):
    role_dict = {
        "nuisance_train": _toy_role_rows(21, 60, 0),
        "audit_fit": _toy_role_rows(22, 40, 1000),
        "coefficient_split": _toy_role_rows(23, 30, 2000),
        "inner_selection": _toy_role_rows(24, 20, 3000),
        "inner_check": _toy_role_rows(25, 20, 4000),
    }
    d17 = np.zeros((32, 17)); d17[:, 0] = 1
    hist = np.zeros((32, 17)); hist[:, 1] = 1
    out = tmp_path / "private" / "center"
    args = dict(anchor=0, delta=.001, role_dict=role_dict,
                q_ref=d17, historical_q=hist, encoder_sha256="0"*64,
                output_dir=out, max_rounds=0, target_roles=("A/SEX",),
                initial_sources=("H", "D17"))
    first = run_center_from_roles(**args)
    assert first["status"] == "COMPLETE"
    assert first["selected_channel"].shape == (32, 17)
    assert first["rounds"][0]["reference_max_cut_violation"] <= 1e-10
    assert first["rounds"][0]["maximum_cut_violation"] <= 1e-7
    receipt = (out / "COMPLETE.json").read_bytes()
    second = run_center_from_roles(**args)
    assert np.array_equal(second["selected_channel"], first["selected_channel"])
    assert (out / "COMPLETE.json").read_bytes() == receipt
    assert first["outer_labels_accessed"] is False


def test_prior_round_with_better_selection_loss_cannot_escape_final_bank():
    earlier = np.array([[0., 1.], [1., 0.]])
    final = np.eye(2)
    cost = np.array([[0., 1.], [0., 1.]])
    cut = {"id": "new_best_response", "coeff": np.array([[1., 0.], [0., 0.]]),
           "floor": .8}
    rounds = [{"round": 0, "inner_selection_fixed_decoder_task": {"U": .1, "W": .1}},
              {"round": 1, "inner_selection_fixed_decoder_task": {"U": .2, "W": .2}}]
    selected = select_final_bank_feasible([earlier, final], rounds, cost, [cut])
    assert selected["selected_round"] == 1
    assert selected["final_bank_checks"][0]["feasible"] is False
    assert selected["final_bank_checks"][1]["feasible"] is True
