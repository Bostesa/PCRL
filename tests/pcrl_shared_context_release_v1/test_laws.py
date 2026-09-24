"""Synthetic laws through the dispatcher; no ACS rows."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from experiments.pcrl_shared_context_release_v1 import laws

N = 6


def legal(n=N, seed=0):
    rng = np.random.default_rng(seed)
    return {"x": rng.normal(size=(n, 32)), "ha": rng.normal(size=(n, 4)),
            "token_codes": np.arange(n) % 32, "teacher_p": rng.uniform(.1, .9, n),
            "residual": rng.normal(size=n), "risk": rng.uniform(size=(n, 11))}


def d17_matrix():
    q = np.zeros((32, 17))
    q[np.arange(32), np.arange(32) % 17] = 1.0
    return q


def fake_module(monkeypatch, module, law_fn):
    """Register a stand-in for release.py / rd.py / adv.py exposing load_law."""
    fake = types.ModuleType(f"{laws.PACKAGE}.{module}")
    calls = []

    def load_law(unit_dir):
        calls.append(unit_dir)

        def law(inputs):
            assert set(inputs) == set(laws.LEGAL_INPUTS)
            return law_fn(inputs)
        return law
    fake.load_law = load_law
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    return calls


def unit(tmp_path, kind, name="u"):
    root = tmp_path / name
    root.mkdir()
    (root / "model.bin").write_bytes(b"frozen")
    laws.write_descriptor(root, kind, release_id=name, anchor=0)
    return root


def test_nested_deterministic_and_adv_units_dispatch_by_descriptor(tmp_path, monkeypatch):
    stochastic = lambda inputs: np.full((len(inputs["x"]), 17), 1 / 17)
    onehot = lambda inputs: np.eye(17)[inputs["token_codes"] % 17]
    fake_module(monkeypatch, "release", stochastic)
    fake_module(monkeypatch, "rd", onehot)
    fake_module(monkeypatch, "adv", stochastic)
    for kind, expected in (("nested", 1 / 17), ("adv_mlp", 1 / 17)):
        law = laws.person_law(unit(tmp_path, kind, kind), legal())
        assert law.shape == (N, 17) and np.allclose(law, expected)
    law = laws.person_law(unit(tmp_path, "deterministic_policy", "rd"), legal())
    assert np.array_equal(law.argmax(1), np.arange(N) % 17)
    assert laws.support_record(law)["max_support"] == 1


def test_historical_map_indexed_by_t0_and_h_only_wire(tmp_path):
    path = tmp_path / "Q.npz"
    np.savez(path, Q=d17_matrix())
    spec = laws.historical_spec("D17", 0, map_path=path, map_sha256=laws.sha256_file(path))
    law = laws.person_law(spec, legal())
    assert np.array_equal(law, d17_matrix()[np.arange(N) % 32])
    with pytest.raises(ValueError, match="SHA-256"):
        laws.person_law({**spec, "map_sha256": "0" * 64}, legal())
    h = laws.person_law(laws.h_only_spec(), legal())
    assert h.shape == (N, 1) and np.all(h == 1)


@pytest.mark.parametrize("key", ["hb", "labels", "ids", "households", "weights", "extra"])
def test_hidden_or_undeclared_inputs_are_refused_before_loading(tmp_path, monkeypatch, key):
    calls = fake_module(monkeypatch, "release", lambda i: np.full((len(i["x"]), 17), 1 / 17))
    root = unit(tmp_path, "nested")
    bad = {**legal(), key: np.zeros(N)}
    with pytest.raises(PermissionError):
        laws.person_law(root, bad)
    assert calls == []  # refused before the release module loaded anything
    with pytest.raises(PermissionError):
        laws.load(root)(bad)


def test_legal_inputs_projects_role_rows_and_rejects_bad_shapes():
    rows = {**legal(), "hb": np.zeros((N, 2)), "labels": {"SEX": np.zeros(N)},
            "ids": np.arange(N), "households": np.arange(N), "weights": np.ones(N)}
    assert set(laws.legal_inputs(rows)) == set(laws.LEGAL_INPUTS)
    with pytest.raises(ValueError):
        laws.check_inputs({**legal(), "risk": np.zeros((N, 10))})
    with pytest.raises(ValueError):
        laws.check_inputs({**legal(), "token_codes": np.full(N, 32)})
    missing = legal()
    del missing["risk"]
    with pytest.raises(ValueError, match="missing"):
        laws.check_inputs(missing)


@pytest.mark.parametrize("broken", [
    lambda n: np.full((n, 16), 1 / 16),                      # wrong columns
    lambda n: np.full((n + 1, 17), 1 / 17),                  # wrong rows
    lambda n: np.where(np.eye(17)[np.zeros(n, int)] > 0, 1.0, np.nan),  # nonfinite
    lambda n: np.eye(17)[np.zeros(n, int)] * 2 - np.eye(17)[np.ones(n, int)],  # negative
    lambda n: np.full((n, 17), 1 / 17) * (1 + 1e-8),         # row sums off by 1e-8
])
def test_every_returned_law_is_validated(tmp_path, monkeypatch, broken):
    fake_module(monkeypatch, "release", lambda inputs: broken(len(inputs["x"])))
    with pytest.raises(ValueError):
        laws.person_law(unit(tmp_path, "nested"), legal())


def test_row_sum_tolerance_is_1e_minus_9():
    ok = np.full((2, 17), 1 / 17)
    laws.validate_law(ok * (1 + 5e-11), 2)
    with pytest.raises(ValueError):
        laws.validate_law(ok * (1 + 5e-9), 2)


def test_descriptor_pins_are_verified_and_write_once(tmp_path, monkeypatch):
    fake_module(monkeypatch, "release", lambda i: np.full((len(i["x"]), 17), 1 / 17))
    root = unit(tmp_path, "nested")
    (root / "model.bin").write_bytes(b"swapped")
    with pytest.raises(ValueError, match="differs"):
        laws.person_law(root, legal())
    with pytest.raises(FileExistsError):
        laws.write_descriptor(root, "nested", release_id="other", anchor=0)
    with pytest.raises(ValueError):
        laws.write_descriptor(tmp_path, "J", release_id="J", anchor=0)


def test_law_identity_is_exact_bytes():
    a = np.full((3, 17), 1 / 17)
    assert laws.law_identity(a) == laws.law_identity(a.copy())
    b = a.copy()
    b[0, 0] = np.nextafter(b[0, 0], 1)
    assert laws.law_identity(a) != laws.law_identity(b)
