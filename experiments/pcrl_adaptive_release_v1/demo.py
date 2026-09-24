"""Public synthetic A/B fit -> one-token release -> exact-token audit.

This fixture never opens ACS, a private archive or an assessment pool. Its
small roles deliberately cannot meet B's frozen 100-household child floor,
so B may correctly retain an A alias. It tests the actual fitting and wire
contracts; it is not evidence of a task/privacy advantage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
from . import audit, fit_a, fit_b, nuisance, refinement, roles


class SyntheticEncoder:
    """Deterministic allowed-input encoder for this public fixture only."""

    def encode(self, inputs: RuntimeInputs) -> dict:
        if not isinstance(inputs, RuntimeInputs):
            raise TypeError("fixture encoder accepts RuntimeInputs(X_A,H_A) only")
        signal = np.asarray(inputs.x_a[:, 0], dtype=np.float64)
        t = np.clip(np.floor((signal + 2.)*8.), 0, 31).astype(np.int64)
        p = 1./(1.+np.exp(-signal))
        return {"codes": {"T0": t}, "p": p, "r": signal.copy()}


def _households(role: str, n: int, seed: int) -> np.ndarray:
    selected = []
    candidate = 0
    while len(selected) < n:
        label = f"public-synthetic-{role}-{seed}-{candidate}"
        if roles.role_of(label) == role:
            selected.append(label)
        candidate += 1
    return np.asarray(selected)


def _role_rows(role: str, n: int, seed: int, encoder: SyntheticEncoder) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .3, size=(n, 32)).astype(np.float32)
    x[:, 0] = np.linspace(-1.5, 1.5, n, dtype=np.float32)
    ha = rng.normal(0, .5, size=(n, 4)).astype(np.float32)
    hb = rng.normal(0, .5, size=(n, 2)).astype(np.float32)
    inputs = RuntimeInputs(x, ha)
    encoded = encoder.encode(inputs)
    return {"x": x, "ha": ha, "hb": hb,
            "token_codes": encoded["codes"]["T0"],
            "teacher_p": encoded["p"], "residual": encoded["r"],
            "labels": {"same_residence": (x[:, 0] + .5*ha[:, 0] > 0).astype(np.int64),
                       "SEX": (x[:, 0] - ha[:, 0] > 0).astype(np.int64),
                       "RAC1P": (np.arange(n) + seed) % 9},
            "weights": (1 + np.arange(n) % 5).astype(np.float64),
            "ids": np.asarray([f"public-person-{role}-{seed}-{i}" for i in range(n)]),
            "households": _households(role, n, seed)}


def _exact_task_audit(decoder, rows: dict, law: np.ndarray) -> dict:
    """Compare production expected-token CE with an independent direct sum."""
    y = np.asarray(rows["labels"]["same_residence"], dtype=np.int64)
    predictions = np.asarray(decoder.predict_token_proba(rows["ha"], 17),
                             dtype=np.float64)
    person = audit.expected_token_loss(predictions, law, y)
    weights = np.asarray(rows["weights"], dtype=np.float64)
    score = audit.score_weightings(person, weights)
    independent = np.asarray([
        sum(float(law[i, z]) * -np.log(max(float(predictions[i, z, y[i]]),
                                          audit.FLOOR))
            for z in range(17))
        for i in range(len(y))], dtype=np.float64)
    difference = float(np.max(np.abs(person - independent)))
    if difference > 1e-12:
        raise AssertionError("exact token expectation disagrees with direct enumeration")
    return {"scores": score, "difference": difference}


def run_demo(output_dir: str | Path) -> dict:
    """Fit and replay synthetic units; write only an aggregate public receipt."""
    root = Path(output_dir)
    # macOS resolves /tmp under /private; require a study-owned private
    # directory component in addition to that system mount prefix.
    if "private" not in root.parts[2:]:
        raise ValueError("fitted synthetic objects require a private output path")
    root.mkdir(parents=True, exist_ok=True)
    encoder = SyntheticEncoder()
    role_sizes = {"nuisance_train": 60, "audit_fit": 40,
                  "coefficient_split": 30, "inner_selection": 20,
                  "inner_check": 20}
    role_dict = {name: _role_rows(name, count, 20260924 + j, encoder)
                 for j, (name, count) in enumerate(role_sizes.items())}
    q_ref = np.eye(17, dtype=np.float64)[np.arange(32) % 17]
    historical_q = np.eye(17, dtype=np.float64)[(np.arange(32) + 1) % 17]
    encoder_sha = hashlib.sha256(
        b"PUBLIC_SYNTHETIC_ENCODER_V1:X_A0_TO_T32_P_RESIDUAL").hexdigest()
    a_dir, b_dir = root / "fit_A", root / "fit_B"
    a_complete = fit_a.run_center_from_roles(
        0, .001, role_dict, q_ref, historical_q, encoder_sha, a_dir, 0,
        target_roles=("A/SEX",), initial_sources=("H", "D17"))
    a_selected = fit_b.load_a_selected(a_dir, anchor=0, delta=.001)
    b_result = fit_b.run_center_from_roles(
        0, .001, role_dict, q_ref, historical_q, encoder_sha,
        a_selected, b_dir, max_states=33, max_rounds=0)
    b_complete = json.loads((b_dir / "COMPLETE.json").read_text())
    if b_complete["status"] != b_result["status"]:
        raise AssertionError("B selected receipt status differs")
    frozen_nuisance, nuisance_receipt = nuisance.load_frozen_nuisance(
        b_dir / "nuisance")
    partition = refinement.NestedPartition.from_record(
        json.loads((b_dir / "PARTITION.json").read_text()))
    with np.load(b_dir / b_complete["selected_channel_relative"],
                 allow_pickle=False) as archive:
        q = np.asarray(archive["Q"], dtype=np.float64).copy()
    if not np.array_equal(q, b_result["selected_channel"]):
        raise AssertionError("selected B channel differs from immutable fit receipt")
    # A public fixture key proves repeatable one-release mechanics; real keys
    # must be private, random and never committed with a trained release.
    synthetic_key = hashlib.sha256(b"PUBLIC_SYNTHETIC_REPLAY_KEY_V1").digest()
    check = role_dict["inner_check"]
    inputs = RuntimeInputs(check["x"], check["ha"])
    session = fit_b.RefinedReleaseSession(
        q, partition, encoder, frozen_nuisance,
        synthetic_fixture=True, replay_key=synthetic_key)
    wire = session.release(inputs, person_ids=check["ids"])
    repeated = session.release(inputs, person_ids=check["ids"])
    fresh = fit_b.RefinedReleaseSession(
        q, partition, encoder, frozen_nuisance,
        synthetic_fixture=True, replay_key=synthetic_key)
    restarted = fresh.release(inputs, person_ids=check["ids"])
    changed = np.asarray(inputs.h_a).copy()
    changed[0, 0] += 1.
    changed_input_rejected = False
    try:
        session.release(RuntimeInputs(inputs.x_a, changed), person_ids=check["ids"])
    except ValueError as exc:
        changed_input_rejected = "changed" in str(exc)
    if (set(wire) != {"h_a", "token"} or
            wire["h_a"].dtype != inputs.h_a.dtype or
            wire["h_a"].tobytes() != inputs.h_a.tobytes() or
            not np.array_equal(wire["token"], repeated["token"]) or
            not np.array_equal(wire["token"], restarted["token"]) or
            not changed_input_rejected or
            np.any((wire["token"] < 0) | (wire["token"] >= 17))):
        raise AssertionError("synthetic one-token release contract failed")
    law = session.private_expected_law(inputs)
    if law.shape != (len(inputs.h_a), 17):
        raise AssertionError("private law changed person or token count")
    exact = _exact_task_audit(a_selected["decoder"], check, law)
    receipt = {"schema": "pcrl-adaptive-public-synthetic-demo-v1",
               "fit_A_status": a_complete["status"],
               "fit_B_status": b_complete["status"],
               "fit_B_realized_internal_states": int(len(q)),
               "B_support_floor_households_per_child": 100,
               "n_original_check_people": int(len(inputs.h_a)),
               "token_count": 17, "wire_fields": ["h_a", "token"],
               "service_byte_equal": True,
               "same_token_within_session": True,
               "same_token_after_restart": True,
               "changed_input_rejected": changed_input_rejected,
               "exact_task_loss_nats": exact["scores"],
               "exact_enumeration_max_abs_difference": exact["difference"],
               "decoder_training_scope": "synthetic nuisance_train and inner_selection only",
               "audit_scope": "fixed fitted decoder on synthetic inner_check; no fresh attacker slate",
               "independent_acs_claim": False,
               "outer_assessment_accessed": False,
               "public_fixture_key_is_not_deployment_secret": True,
               "model_status": "EXPERIMENTAL_SYNTHETIC_DEMONSTRATION"}
    target = root / "SYNTHETIC_DEMO.json"
    encoded = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False)+"\n"
    if target.exists() and target.read_text() != encoded:
        raise ValueError("existing synthetic demo summary differs on immutable resume")
    if not target.exists():
        target.write_text(encoded)
    return receipt


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True,
                        help="new private directory for synthetic fitted units")
    args = parser.parse_args(argv)
    receipt = run_demo(args.output_dir)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return receipt


if __name__ == "__main__":
    main()
