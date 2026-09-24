"""Release loader and keyed-persistent session on a hand-built pinned unit."""
import json

import numpy as np
import pytest

from experiments.pcrl_task_aligned_cuts_v1.data import sha256_file
from experiments.pcrl_shared_context_release_v1 import channel, contexts, policies, release


class RiskFromX:
    def probabilities(self, inputs):
        p = 1 / (1 + np.exp(-np.asarray(inputs.x_a)[:, 1]))
        return {"SEX": np.column_stack((p, 1 - p)), "RAC1P": np.full((len(p), 9), 1 / 9)}


class ParityPolicy:
    """Deterministic legal-input policy (token from sign of x0)."""
    name = "parity"

    def predict(self, inputs):
        legal = policies.check_legal(inputs)
        return np.where(np.asarray(legal["x"])[:, 0] > 0, 16, 1).astype(np.int64)


def _inputs(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return {"x": rng.normal(size=(n, 32)), "ha": rng.normal(size=(n, 4)),
            "token_codes": rng.integers(0, 32, n), "teacher_p": rng.uniform(.1, .9, n),
            "residual": rng.normal(size=n), "risk": rng.dirichlet(np.ones(11), n)}


@pytest.fixture()
def unit(tmp_path):
    root = tmp_path / "private"
    d17 = np.zeros((32, 17)); d17[np.arange(32), np.arange(32) % 17] = 1.
    bank_obj = policies.PolicyBank(("D17", "parity"), (policies.D17Policy("D17", np.argmax(d17, 1)), ParityPolicy()))
    bank_dir = root / "bank"
    saved = policies.save_bank(bank_dir / "policies", bank_obj)
    rows = {**_inputs(400, 5), "weights": np.ones(400)}
    rules = contexts.fit_context_rules(rows, RiskFromX(), "r" * 64)
    ctx_saved = contexts.save_rules(bank_dir / "contexts", rules)
    (bank_dir / "BANK_COMPLETE.json").write_text(json.dumps(
        {"policy_bank_sha256": saved["sha256"], "contexts_sha256": ctx_saved["sha256"]}))
    unit_dir = root / "unit"
    unit_dir.mkdir()
    B = 0.6 * d17
    A = np.array([[0.4, 0.], [0., 0.4], [0.2, 0.2], [0.1, 0.3]])
    np.savez_compressed(unit_dir / "PARAMS.npz", B=B, A=A, eta=np.asarray(0.4))
    spec = {"schema": "pcrl-sc-nested-release-v1", "kind": "nested", "K": 4,
            "policy_columns": ["D17", "parity"], "params_relative": "PARAMS.npz",
            "params_sha256": sha256_file(unit_dir / "PARAMS.npz"),
            "bank_dir": str(bank_dir), "bank_dir_relative": "../bank",
            "bank_manifest_sha256": sha256_file(bank_dir / "BANK_COMPLETE.json"),
            "policy_bank_sha256": saved["sha256"], "contexts_sha256": ctx_saved["sha256"]}
    (unit_dir / "RELEASE_SPEC.json").write_text(json.dumps(spec))
    return {"unit": unit_dir, "B": B, "A": A, "rules": rules, "bank": bank_obj}


def test_load_law_replays_pinned_parameters(unit):
    law = release.load_law(unit["unit"])
    x = _inputs()
    q = law(x)
    expected = channel.person_law(unit["B"], unit["A"], 0.4, x["token_codes"],
                                  unit["rules"][4].assign(x), unit["bank"].predict(x))
    assert np.array_equal(q, expected)
    assert np.max(np.abs(q.sum(1) - 1)) <= 1e-12
    assert not channel.deterministic_emission(q)["all_rows_one_hot"]
    # aliases for T0/p/r are accepted; forbidden inputs are not
    alias = {"x": x["x"], "ha": x["ha"], "T0": x["token_codes"], "p": x["teacher_p"],
             "r": x["residual"], "risk": x["risk"]}
    assert np.array_equal(law(alias), q)
    for key in ("hb", "labels", "ids", "households", "weights"):
        with pytest.raises(PermissionError):
            law({**x, key: np.zeros(len(q))})


def test_tampered_pins_are_rejected(unit):
    params = unit["unit"] / "PARAMS.npz"
    raw = params.read_bytes()
    np.savez_compressed(params, B=unit["B"], A=unit["A"] * 1.0, eta=np.asarray(0.40000001))
    with pytest.raises(ValueError):
        release.load_law(unit["unit"])
    params.write_bytes(raw)
    release.load_law(unit["unit"])
    manifest = unit["unit"].parent / "bank" / "BANK_COMPLETE.json"
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(FileNotFoundError):
        release.load_law(unit["unit"])


def test_session_emits_one_persistent_keyed_token(unit):
    law = release.load_law(unit["unit"])
    x = _inputs(4000, 2)
    ids = [f"p{i}" for i in range(4000)]
    session = release.SharedContextRelease(law, replay_key=b"s" * 32, release_id="r1")
    wire = session.emit(x, ids)
    assert set(wire) == {"h_a", "token"}
    assert wire["h_a"].tobytes() == np.asarray(x["ha"]).tobytes()
    q = law(x)
    assert np.all(q[np.arange(4000), wire["token"]] > 0)
    # empirical token frequencies track the private law (one draw per person)
    freq = np.bincount(wire["token"], minlength=17) / 4000
    assert np.max(np.abs(freq - q.mean(0))) < 0.03
    # persistence within and across sessions with the same key; changed inputs rejected
    assert np.array_equal(session.emit(x, ids)["token"], wire["token"])
    again = release.SharedContextRelease(law, replay_key=b"s" * 32, release_id="r1")
    assert np.array_equal(again.emit(x, ids)["token"], wire["token"])
    other = release.SharedContextRelease(law, replay_key=b"t" * 32, release_id="r1")
    assert not np.array_equal(other.emit(x, ids)["token"], wire["token"])
    changed = {**x, "residual": x["residual"] + 1.}
    with pytest.raises(ValueError):
        session.emit(changed, ids)
    with pytest.raises(ValueError):
        session.emit({k: v for k, v in x.items() if k != "risk"}, ids)
    with pytest.raises(ValueError):
        release.SharedContextRelease(law, replay_key=b"short", release_id="r1")
