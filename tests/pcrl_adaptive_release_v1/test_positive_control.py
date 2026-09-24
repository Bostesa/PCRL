"""Nondeployable audit-power controls: exact interaction and role barriers."""
import math
import json

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import audit, positive_control, roles


@pytest.mark.parametrize("view", ["A", "AB"])
def test_xor_interaction_is_invisible_marginally_but_visible_to_legal_full_view(view):
    fixture = positive_control.xor_fixture(view)
    bit = fixture["sensitive"]
    token = fixture["token"]
    h = fixture["h_bit"]
    assert len(bit) == 4
    assert np.array_equal(np.bincount(bit, minlength=2), [2, 2])
    for conditioned in (h, token):
        for value in (0, 1):
            assert np.array_equal(np.bincount(bit[conditioned == value], minlength=2), [1, 1])
    full = audit.expected_token_loss(fixture["full_prediction"], fixture["law"], bit)
    marginal = audit.expected_token_loss(fixture["marginal_prediction"], fixture["law"], bit)
    assert np.mean(full) == pytest.approx(-math.log(.99))
    assert np.mean(marginal) == pytest.approx(math.log(2))
    if view == "AB":
        assert np.array_equal(fixture["ha"][:, 0], np.zeros(4))
        assert np.array_equal(fixture["hb"][:, 0], h)
    else:
        assert np.array_equal(fixture["ha"][:, 0], h)


def test_revealing_law_uses_existing_tokens_and_refuses_unsupported_label():
    labels = np.array([0, 1, 8])
    law = positive_control.revealing_token_law(labels, target="RAC1P")
    assert law.shape == (3, 17)
    assert np.array_equal(np.argmax(law, axis=1), labels)
    assert np.array_equal(law.sum(axis=1), np.ones(3))
    with pytest.raises(ValueError):
        positive_control.revealing_token_law(np.array([9]), target="RAC1P")
    with pytest.raises(ValueError):
        positive_control.revealing_token_law(np.array([0]), target="same_residence")


def _role_household(role):
    return next(f"positive_control_test_{i}" for i in range(10000)
                if roles.role_of(f"positive_control_test_{i}") == role)


def _inner_rows(role):
    household = _role_household(role)
    return {"ha": np.zeros((4, 4)), "hb": np.zeros((4, 2)),
            "labels": {"SEX": np.array([0, 1, 0, 1]),
                       "RAC1P": np.array([0, 1, 0, 1])},
            "weights": np.ones(4), "ids": np.array([f"{role}_{i}" for i in range(4)]),
            "households": np.repeat(household, 4)}


def test_inner_diagnostic_rejects_outer_resource_before_fitting(tmp_path):
    inner = {role: _inner_rows(role) for role in
             ("audit_fit", "inner_selection", "inner_check")}
    inner["outer_assessment"] = _inner_rows("outer_assessment")
    with pytest.raises(ValueError, match="only.*inner|three inner"):
        positive_control.run_inner_diagnostic(
            inner, role="attack:A/SEX", anchor=0,
            output_dir=tmp_path / "private" / "positive_control")


def test_inner_diagnostic_rejects_wrong_household_assignment(tmp_path):
    inner = {role: _inner_rows(role) for role in
             ("audit_fit", "inner_selection", "inner_check")}
    inner["inner_check"]["households"] = inner["audit_fit"]["households"].copy()
    with pytest.raises(ValueError, match="household|role"):
        positive_control.run_inner_diagnostic(
            inner, role="attack:AB/RAC1P", anchor=1,
            output_dir=tmp_path / "private" / "positive_control")


def test_inner_diagnostic_keeps_label_oracle_private_and_selects_legal_ab_routes(tmp_path, monkeypatch):
    inner = {role: _inner_rows(role) for role in
             ("audit_fit", "inner_selection", "inner_check")}
    seen = []

    def fake_fit(fit_rows, fit_law, validation_rows, validation_law,
                 role, output_dir, seed, *, release_id, slate):
        target = role.split("/")[1]
        assert slate == "standard"
        assert len(fit_law) == len(fit_rows["ids"])
        if release_id == "H":
            assert fit_law.shape[1] == 1
        else:
            assert fit_law.shape[1] == 17
            assert np.array_equal(np.argmax(fit_law, axis=1), fit_rows["labels"][target])
        seen.append((role, release_id))
        view = role.split(":")[1].split("/")[0]
        output_dir.mkdir(parents=True)
        (output_dir / "synthetic_model.txt").write_text("fixture")
        return {"role": role, "release_id": release_id, "slate": slate,
                "models": {"fixture": {"kind": "model", "target": target,
                    "source_view": view, "wire": "H" if release_id == "H" else "release",
                    "source_release_id": release_id, "model_sha256": "a"*64}}}

    def fake_select(rows, law, role, routes, *, release_id):
        assert "own/fixture" in routes and "H/fixture" in routes
        if role.startswith("attack:AB/"):
            assert {"A/fixture", "B/fixture"}.issubset(routes)
        return {"role": role, "release_id": release_id,
                "selected": "own/fixture", "route": routes["own/fixture"],
                "candidate_count": len(routes), "scores": {"own/fixture": 0.1}}

    def fake_score(rows, law, lock):
        loss = np.full(len(rows["ids"]),
                       math.log(2) if lock["release_id"] == "H" else -math.log(.99))
        return {"ids": rows["ids"], "households": rows["households"],
                "weights": rows["weights"], "loss": loss,
                "scores": {"U": float(loss.mean()), "PWGTP": float(loss.mean())},
                "selected_candidate": lock["selected"], "role": lock["role"]}

    monkeypatch.setattr(audit, "fit_role_slate", fake_fit)
    monkeypatch.setattr(audit, "select_frozen_routes", fake_select)
    monkeypatch.setattr(audit, "score_frozen_route", fake_score)
    root = tmp_path / "private" / "diagnostic"
    result = positive_control.run_inner_diagnostic(
        inner, role="attack:AB/SEX", anchor=0, output_dir=root)
    assert result["nondeployable_label_oracle"] is True
    assert result["detected_both_weightings"] is True
    assert result["improvement_nats"]["U"] > .6
    assert ("attack:A/SEX", "POSITIVE_CONTROL_LABEL_ORACLE_NONDEPLOYABLE") in seen
    assert ("attack:B/SEX", "H") in seen
    saved = json.loads((root / "POSITIVE_CONTROL.json").read_text())
    assert saved["role"] == "attack:AB/SEX"
    assert "labels" not in saved and "token_law" not in saved
    assert (root / "COMPLETE.json").exists()
    with pytest.raises(FileExistsError):
        positive_control.run_inner_diagnostic(
            inner, role="attack:AB/SEX", anchor=0, output_dir=root)
