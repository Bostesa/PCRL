"""Positive-control wrapper: roles, admission, detection rule (synthetic, fitted routes faked)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import audit, roles
from experiments.pcrl_shared_context_release_v1 import poscontrol


def _household(role, k):
    found = [f"sc_poscontrol_{i}" for i in range(20000) if roles.role_of(f"sc_poscontrol_{i}") == role]
    return found[k]


def _rows(role, n=6, missing=False):
    sex = np.array([0, 1] * (n // 2))
    race = np.arange(n) % 9
    if missing:
        sex[0] = -1
    return {"ha": np.zeros((n, 4)), "hb": np.zeros((n, 2)), "weights": np.ones(n),
            "ids": np.array([f"{role}_{i}" for i in range(n)]),
            "households": np.array([_household(role, i // 2) for i in range(n)]),
            "labels": {"SEX": sex, "RAC1P": race, "same_residence": np.zeros(n, int)},
            "x": np.zeros((n, 32))}


def _fake(monkeypatch, leak_gain):
    def fake_fit(fit_rows, fit_law, validation_rows, validation_law, role, output_dir, seed, *,
                 release_id, slate):
        target = role.split("/")[1]
        if release_id != "H":
            assert np.array_equal(np.argmax(fit_law, 1), fit_rows["labels"][target])
        output_dir.mkdir(parents=True)
        (output_dir / "m.txt").write_text("fixture")
        view = role.split(":")[1].split("/")[0]
        return {"role": role, "release_id": release_id, "slate": slate,
                "models": {"fx": {"kind": "model", "target": target, "source_view": view,
                                  "wire": "H" if release_id == "H" else "release",
                                  "source_release_id": release_id, "model_sha256": "a" * 64}}}

    def fake_select(rows, law, role, routes, *, release_id):
        return {"role": role, "release_id": release_id, "selected": "own/fx", "route": routes["own/fx"],
                "candidate_count": len(routes), "scores": {"own/fx": 0.1}}

    def fake_score(rows, law, lock):
        base = math.log(2)
        loss = np.full(len(rows["ids"]), base if lock["release_id"] == "H" else base - leak_gain)
        return {"ids": rows["ids"], "households": rows["households"], "weights": rows["weights"],
                "loss": loss, "scores": {"U": float(loss.mean()), "PWGTP": float(loss.mean())},
                "selected_candidate": lock["selected"], "role": lock["role"]}

    monkeypatch.setattr(audit, "fit_role_slate", fake_fit)
    monkeypatch.setattr(audit, "select_frozen_routes", fake_select)
    monkeypatch.setattr(audit, "score_frozen_route", fake_score)


@pytest.mark.parametrize("gain,detected", [(0.5, True), (0.005, False)])
def test_detection_rule_is_ar_threshold(monkeypatch, tmp_path, gain, detected):
    _fake(monkeypatch, gain)
    role_dict = {name: _rows(name) for name in poscontrol.INNER}
    s = poscontrol.run_control(0, "AB/SEX", tmp_path, role_dict=role_dict)
    assert s["detected"] is detected and s["threshold_nats"] == 0.01
    saved = json.loads((tmp_path / "a0_AB_SEX_SUMMARY.json").read_text())
    assert saved["nondeployable_label_oracle"] and not saved["outer_labels_accessed"]
    assert "private" in saved["unit_dir"]


def test_missing_labels_are_excluded_and_counted(monkeypatch, tmp_path):
    _fake(monkeypatch, 0.5)
    role_dict = {name: _rows(name, missing=(name == "audit_fit")) for name in poscontrol.INNER}
    s = poscontrol.run_control(2, "AB/SEX", tmp_path, role_dict=role_dict)
    assert s["excluded_missing_label_rows"] == {"audit_fit": 1, "inner_selection": 0, "inner_check": 0}


def test_registered_roles_cover_both_coalition_targets():
    assert poscontrol.ROLES == ("AB/SEX", "AB/RAC1P")
    with pytest.raises(ValueError):
        poscontrol.run_control(0, "B/SEX", "/tmp/unused", role_dict={})
