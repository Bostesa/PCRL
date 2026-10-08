"""Run one locked Team B arm/seed as one S3 real-data queue unit.

This file has no import-time data access. It refuses direct invocation outside
the matching running S3 child process and an effective S1 science seal. Each
attempt is versioned: completed phases and failed stage checkpoints remain
private for inspection or a separately admitted retry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import stat
import time
from pathlib import Path

import numpy as np
import torch

from attack_bridge import build_attack_view
from bottleneck import (BottleneckPair, ReleaseAdapter, SavedRelease, SexCritics,
                        model_fingerprint, schema_fingerprint)
from calibration_bridge import fit_calibration
from control_artifacts import critic_fingerprint, file_sha256
from control_pipeline import fit_arm
from control_plan import ARMS, stages_for_arm
from control_runner import ControlHyperparameters
from fit_input_bridge import build_fit_input
from train_bottleneck import FrozenPurposeView


class StopRequested(Exception):
    pass


def _sha(path: str | Path) -> str:
    return file_sha256(Path(path))


def _write_json(path: Path, record: dict) -> None:
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _check_gate(args) -> tuple[dict, dict]:
    from compute_queue import Queue

    lock_path = Path(args.campaign_lock).resolve()
    config_path = Path(args.screen_config).resolve()
    manifest_path = Path(args.data_manifest).resolve()
    for path, expected in ((lock_path, args.lock_sha256),
                           (config_path, args.config_sha256),
                           (manifest_path, args.manifest_sha256)):
        if not path.is_file() or _sha(path) != expected:
            raise PermissionError("A sealed lock/config/data manifest byte identity differs")
    lock = json.loads(lock_path.read_text())
    config = json.loads(config_path.read_text())
    if (lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE"
            or lock.get("adult_fit_permission") is not True
            or lock.get("fit_admission") != "B_ONLY"
            or lock.get("families", {}).get("B", {}).get("fit_admission") != "ALLOWED"
            or args.arm not in lock.get("admitted_arms", {}).get("B", ())
            or lock["families"]["B"].get("screen_config_sha256") != args.config_sha256
            or lock.get("data_access_manifest_sha256") != args.manifest_sha256
            or config.get("schema") != "pcrl-team-b-screen-config-v1"
            or config.get("status") != "EFFECTIVE"
            or config.get("science_permission") is not True):
        raise PermissionError("S1 has not sealed this exact Team B science configuration")
    if args.arm not in ARMS or args.arm not in config["arms"] or args.seed not in config["seeds"]:
        raise PermissionError("Arm or seed is not in the effective Team B screen lock")
    queue = Queue(args.queue_root)
    unit = None
    state = None
    for _ in range(30):
        state = queue.status()
        unit = state["units"].get(args.unit_id)
        if unit and unit.get("child_pgid") == os.getpgrp():
            break
        time.sleep(0.1)
    if (not unit or unit.get("status") != "running"
            or unit.get("child_pgid") != os.getpgrp()
            or unit["spec"].get("mode") != "real_data"
            or unit["spec"].get("stage") != "screen"
            or unit["spec"].get("team") != "B"
            or unit["spec"].get("arm") != args.arm
            or unit["spec"].get("seed") != args.seed
            or not any(Path(arg).name == "screen_driver.py" for arg in unit["spec"]["argv"])):
        raise PermissionError("This process is not the matching running S3 screen unit")
    gate = state["gate"]
    if (not gate.get("safe_inputs") or not gate.get("occupancy_confirmed")
            or not queue._local_seal_ok(gate)):
        raise PermissionError("S3 real-data science gate is not open")
    synthetic = lock.get("synthetic_attack_plumbing", {})
    receipt_path = Path(args.synthetic_attack_plumbing_receipt)
    unit = state["units"].get(synthetic.get("s3_unit_id"))
    if (not receipt_path.is_file()
            or _sha(receipt_path) != args.synthetic_attack_plumbing_sha256
            or synthetic.get("receipt_sha256") != args.synthetic_attack_plumbing_sha256
            or not unit or unit.get("status") != "done" or unit.get("returncode") != 0
            or unit["spec"].get("mode") != "prelock_synthetic"
            or unit["spec"].get("team") != "S2"
            or unit["spec"].get("test_sha256") != synthetic.get("smoke_script_sha256")):
        raise PermissionError("Queued synthetic attack-plumbing smoke is not admitted")
    smoke_script = Path(unit["spec"]["test_file"])
    if (smoke_script.name != "smoke_attack_controls.py"
            or _sha(smoke_script) != synthetic.get("smoke_script_sha256")
            or _sha(smoke_script.with_name("attack_controls.py"))
            != synthetic.get("attack_controls_module_sha256")):
        raise PermissionError("Synthetic attack-plumbing source bytes changed")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "pcrl-portfolio-synthetic-attack-control-smoke-v1"
            or receipt.get("status") != "SYNTHETIC_PASS"
            or receipt.get("adult_role_values_accessed") is not False
            or receipt.get("probe", {}).get("status") != "PASS"
            or set(receipt.get("probe", {}).get("gates", {})) !=
            {"null_exact", "shuffled_at_most_0.60", "planted_at_least_0.99"}
            or not all(receipt["probe"]["gates"].values())):
        raise PermissionError("Synthetic attack-plumbing controls failed")
    return lock, config


def _make_initial_models(data, seed: int, config: dict):
    classes = {"income": 2, "occupation": 6}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config["model_initialization_base_seed"]) + seed)
        model = BottleneckPair(
            feature_names=data.feature_names,
            forbidden_names=("SEX", "RACE", "income", "occupation", "fnlwgt",
                             "income_label", "occupation_label"),
            class_counts=classes,
            log_variance_bounds=tuple(config["log_variance_bounds"]),
            hidden_dim=int(config["hidden_dim"]), latent_dim=int(config["latent_dim"]),
        )
        critics = SexCritics(latent_dim=int(config["latent_dim"]),
                             class_counts=classes, hidden_dim=int(config["hidden_dim"]))
    return model, critics


def _hyper(config: dict) -> ControlHyperparameters:
    values = dict(config["hyperparameters"])
    values["adam_betas"] = tuple(values["adam_betas"])
    return ControlHyperparameters(**values)


def _new_secret(path: Path) -> bytes:
    value = secrets.token_bytes(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    return value


def _release_role(store, role: str, model: BottleneckPair, *, source_id: str,
                  secret: bytes, temperatures: dict[str, float],
                  calibration_records: dict[str, dict],
                  calibration_record_sha256: dict[str, str],
                  attempt: Path, stochastic: bool):
    features = store.read(role, "features")
    ids = np.asarray(features["row_id"])
    groups = np.asarray(features["unit"])
    if ids.dtype != np.int64 or groups.dtype != np.int64:
        raise ValueError("Admitted release role IDs must be int64")
    columns = tuple(str(name) for name in features["feature_names"])
    if columns != model.feature_names:
        raise ValueError("Release role has different ordered predictors")
    adapter = ReleaseAdapter(
        model, stochastic=stochastic, sampler_secret=secret,
        temperatures=temperatures, source_id=source_id,
        expected_model_sha256=model_fingerprint(model),
        expected_schema_sha256=schema_fingerprint(model.feature_names),
        calibration_records=calibration_records,
        calibration_record_sha256=calibration_record_sha256,
    )
    for index, person in enumerate(ids):
        key = str(int(person))
        for purpose in ("income", "occupation"):
            adapter.release(purpose, key, features["X"][index], columns=columns)
    target = attempt / f"release__{role}.json"
    adapter.finalize(target)
    saved = SavedRelease(target)
    rows = [(str(int(person)), {purpose: saved.get(purpose, str(int(person)))
                                for purpose in ("income", "occupation")})
            for person in ids]
    return features, rows, _sha(target)


def _verify_sequential_deployment(fitted, data, *, secret: bytes,
                                  source_id: str, attempt: Path) -> dict:
    """Replay the final first-purpose raw draw against stage 2's frozen view."""
    if fitted.arm not in ("B-SEQ12", "B-SEQ21"):
        return {"checked": False, "first_purpose_forward_passes": 0}
    second = fitted.stage_receipts[1]
    frozen_path = attempt / "fit" / "first_fixed_draw.pt"
    if _sha(frozen_path) != fitted.private_artifact_sha256["first_fixed_draw_file"]:
        raise ValueError("Sequential frozen-view file bytes changed")
    frozen = FrozenPurposeView.load(
        frozen_path, expected_sha256=second["frozen_view_sha256"])
    first_purpose = "income" if fitted.arm == "B-SEQ12" else "occupation"
    if (frozen.purpose != first_purpose or frozen.source_id != source_id
            or frozen.row_order_sha256 != data.row_order_sha256
            or frozen.group_order_sha256 != data.group_order_sha256
            or frozen.feature_schema_sha256 != data.feature_schema_sha256):
        raise ValueError("Sequential final deployment has wrong frozen-view identity")
    adapter = ReleaseAdapter(
        fitted.model, stochastic=True, sampler_secret=secret,
        temperatures={"income": 1.0, "occupation": 1.0}, raw_mode=True,
        source_id=source_id,
        expected_model_sha256=model_fingerprint(fitted.model),
        expected_schema_sha256=data.feature_schema_sha256,
    )
    for index, person_key in enumerate(data.row_ids):
        actual = adapter.release(first_purpose, person_key, data.x[index].numpy(),
                                 columns=data.feature_names)
        with torch.no_grad():
            raw_from_frozen_logits = torch.softmax(
                frozen.logits[index].to(torch.float64).reshape(1, -1), dim=1
            ).squeeze(0).numpy()
        # FrozenPurposeView stores its T=1 probabilities as float32 for stage-2
        # training; deployment retains float64. Compare the exact transmitted
        # z/logits and their corresponding serialized float32 probability.
        if (not np.array_equal(actual["features"], frozen.z[index].numpy())
                or not np.array_equal(actual["logits"], frozen.logits[index].numpy())
                or not np.array_equal(actual["probabilities"], raw_from_frozen_logits)
                or not np.array_equal(actual["probabilities"].astype(np.float32),
                                      frozen.probabilities[index].numpy())):
            raise ValueError("Sequential final first recipient differs from fixed stage-2 view")
    return {"checked": True, "first_purpose_forward_passes": 1,
            "frozen_view_sha256": frozen.sha256}


def _task_and_attack_aggregates(store, *, fit_data, rows_inner, sex_inner,
                                fitted, inner_features, seed: int):
    from attack_eval import score_slate
    from common_eval import (assert_same_people, fit_constant_probability,
                             task_metrics, utility_against_references)

    income = store.read("INNER_SELECTION", "income", "screen")
    occupation = store.read("INNER_SELECTION", "occupation", "screen")
    references = store.read("INNER_SELECTION", f"references_s{seed}")
    assert_same_people(inner_features, income, occupation, references, sex_inner)
    task = {}
    for purpose, label_record, suffix in (("income", income, 1),
                                          ("occupation", occupation, 2)):
        actual = np.asarray([views[purpose]["probabilities"] for _, views in rows_inner])
        y = label_record[purpose]
        constant = fit_constant_probability(fit_data.labels[purpose].numpy(), task=purpose)
        candidate = task_metrics(actual, y, task=purpose, constant_prob=constant)
        u0 = task_metrics(references[f"U0_{suffix}"], y, task=purpose)
        ucal = task_metrics(references[f"Ucal_{suffix}"], y, task=purpose)
        task[purpose] = {"candidate": candidate, "U0": u0, "Ucal": ucal,
                         "comparison": utility_against_references(candidate, u0, ucal)}
    return task, {view: score_slate(fit, sex_inner) for view, fit in fitted.items()}


def _save_endpoint_inputs(store, *, seed: int, arm: str, rows_inner,
                          inner_features, sex_inner, fitted, attempt: Path) -> str:
    """Keep the exact S2 margin inputs private for paired S4 replay."""
    from common_eval import assert_same_people

    income = store.read("INNER_SELECTION", "income", "screen")
    occupation = store.read("INNER_SELECTION", "occupation", "screen")
    references = store.read("INNER_SELECTION", f"references_s{seed}")
    assert_same_people(inner_features, income, occupation, sex_inner, references)
    arrays = {"row_id": np.asarray(inner_features["row_id"], dtype=np.int64),
              "unit": np.asarray(inner_features["unit"], dtype=np.int64),
              "income": np.asarray(income["income"], dtype=np.int64),
              "occupation": np.asarray(occupation["occupation"], dtype=np.int64),
              "sex": np.asarray(sex_inner["sex"], dtype=np.int64)}
    for purpose, number in (("income", 1), ("occupation", 2)):
        arrays[f"prob_{number}"] = np.asarray(
            [view[purpose]["probabilities"] for _, view in rows_inner], dtype=np.float64)
        for ref in ("U0", "Ucal"):
            arrays[f"{ref}_{number}"] = np.asarray(
                references[f"{ref}_{number}"], dtype=np.float64)
    reader_keys = {}
    for view, short in (("income", "r1"), ("occupation", "r2"), ("pair", "pair")):
        reader_keys[short] = {}
        for index, (reader, score) in enumerate(fitted[view]["predictions_inner"].items()):
            key = f"reader__{short}__{index}"
            values = np.asarray(score, dtype=np.float64)
            if values.shape != arrays["row_id"].shape or not np.isfinite(values).all():
                raise ValueError("Fresh reader prediction has wrong admitted people")
            arrays[key] = values
            reader_keys[short][reader] = key
    path = attempt / "ENDPOINT_INPUTS.npz"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    identity = {"schema": "pcrl-team-b-common-endpoint-input-v1", "arm": arm,
                "seed": seed, "npz_sha256": _sha(path), "reader_keys": reader_keys,
                "role": "INNER_SELECTION", "contains_private_person_arrays": True}
    _write_json(attempt / "ENDPOINT_INPUTS.json", identity)
    return _sha(attempt / "ENDPOINT_INPUTS.json")


def _save_attempt_index(*, attempt: Path, arm: str, seed: int,
                        screen_unit_id: str,
                        source_id: str, manifest_sha256: str,
                        campaign_lock_sha256: str, screen_config_sha256: str,
                        fitted, calibration, release_sha256: dict[str, str],
                        attack_receipts: dict[str, dict],
                        endpoint_receipt_sha256: str) -> str:
    """Bind private release, reader and endpoint bytes for independent replay."""
    endpoint = json.loads((attempt / "ENDPOINT_INPUTS.json").read_text())
    if (endpoint["npz_sha256"] != _sha(attempt / "ENDPOINT_INPUTS.npz")
            or endpoint_receipt_sha256 != _sha(attempt / "ENDPOINT_INPUTS.json")):
        raise ValueError("Private endpoint arrays changed before custody index")
    for role, digest in release_sha256.items():
        if _sha(attempt / f"release__{role}.json") != digest:
            raise ValueError("Saved release changed before custody index")
    for purpose, digest in calibration.record_file_sha256.items():
        if _sha(attempt / "calibration" / f"{purpose}.json") != digest:
            raise ValueError("Fitted calibrator changed before custody index")
    for view, receipt in attack_receipts.items():
        if (_sha(attempt / f"readers__{view}" / "READERS_RECEIPT.json")
                != receipt["private_receipt_sha256"]):
            raise ValueError("Fresh reader receipt changed before custody index")
    identity = {
        "schema": "pcrl-team-b-private-attempt-index-v1",
        "status": "FROZEN_FOR_INDEPENDENT_REPLAY",
        "arm": arm, "seed": seed, "screen_unit_id": screen_unit_id,
        "source_sha256": source_id,
        "data_manifest_sha256": manifest_sha256,
        "campaign_lock_sha256": campaign_lock_sha256,
        "screen_config_sha256": screen_config_sha256,
        "fit_model_sha256": model_fingerprint(fitted.model),
        "fit_artifact_sha256": fitted.private_artifact_sha256,
        "calibration_record_sha256": calibration.record_file_sha256,
        "saved_release_sha256": release_sha256,
        "reader_receipt_sha256": {view: receipt["private_receipt_sha256"]
                                   for view, receipt in attack_receipts.items()},
        "endpoint_input_receipt_sha256": endpoint_receipt_sha256,
        "endpoint_arrays_sha256": endpoint["npz_sha256"],
        "aggregate_sha256": _sha(attempt / "AGGREGATE.json"),
        "source_code_sha256": {
            "release_adapter": _sha(Path(__file__).with_name("bottleneck.py")),
            "attack_view_builder": _sha(Path(__file__).with_name("attack_bridge.py")),
            "wire_flattener": _sha(Path(__file__).with_name("attack_views.py")),
            "screen_driver": _sha(Path(__file__)),
        },
        "private_person_arrays": True,
    }
    path = attempt / "ATTEMPT_INDEX.json"
    if path.exists():
        raise FileExistsError(path)
    _write_json(path, identity)
    return _sha(path)


def run_one(args) -> dict:
    from attack_eval import add_ignore_recipient_aliases, fit_slate, save_fitted_slate
    from common_eval import assert_same_people
    from safe_data_guard import SafeRoleStore

    lock, config = _check_gate(args)
    attempt = Path(args.private_attempt).resolve()
    if ".private" not in attempt.parts or attempt.exists():
        raise ValueError("A new versioned Team B private attempt is required")
    attempt.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if stat.S_IMODE(attempt.parent.stat().st_mode) != 0o700:
        raise ValueError("Attempt parent must have mode 0700")
    attempt.mkdir(mode=0o700)
    status = {"schema": "pcrl-team-b-screen-attempt-v1", "status": "STARTED",
              "arm": args.arm, "seed": args.seed, "queue_unit_id": args.unit_id,
              "campaign_lock_sha256": args.lock_sha256,
              "screen_config_sha256": args.config_sha256,
              "data_manifest_sha256": args.manifest_sha256,
              "completed_phases": []}
    _write_json(attempt / "STATUS.json", status)

    def phase(name: str, **identity):
        status["completed_phases"].append(name)
        status.update(identity)
        _write_json(attempt / "STATUS.json", status)

    def stop(_signum, _frame):
        raise StopRequested("S3 requested interruption; private checkpoints are retained")

    old_term = signal.signal(signal.SIGTERM, stop)
    try:
        store = SafeRoleStore(args.data_manifest)
        data, admitted = build_fit_input(store, arm=args.arm, seed=args.seed)
        if admitted.source_id != store.manifest["source"]["source_npz_sha256"]:
            raise ValueError("Team B source differs from S2 admission")
        phase("FIT_INPUT", fit_data_sha256=data.sha256,
              fit_admission_manifest_sha256=admitted.admission_manifest_sha256)
        base, critics = _make_initial_models(data, args.seed, config)
        secret = _new_secret(attempt / "sampler_secret.bin")
        sampler_plan = hashlib.sha256(json.dumps(
            {"schema": "TeamBOneDrawPlanV1", "arm": args.arm, "seed": args.seed,
             "source": admitted.source_id, "sampler_secret_sha256": hashlib.sha256(secret).hexdigest()},
            sort_keys=True).encode()).hexdigest()
        hyper = _hyper(config)
        fitted = fit_arm(
            args.arm, args.seed, data=data, base_model=base, base_critics=critics,
            hyper=hyper, source_id=admitted.source_id, role_id=admitted.role_id,
            admitted_fit=admitted, private_attempt_dir=attempt / "fit",
            expected_base_model_sha256=model_fingerprint(base),
            expected_base_critics_sha256=critic_fingerprint(critics),
            sampler_secret=secret, sampler_plan_sha256=sampler_plan,
        )
        phase("FIT", fit_stage_artifacts=fitted.private_artifact_sha256,
              fit_model_sha256=model_fingerprint(fitted.model))
        sequential_replay = _verify_sequential_deployment(
            fitted, data, secret=secret, source_id=admitted.source_id, attempt=attempt)
        phase("SEQUENTIAL_DEPLOYMENT_REPLAY", **sequential_replay)
        cal_features = store.read("CALIBRATION_HELDOUT", "features")
        cal_labels = {purpose: store.read("CALIBRATION_HELDOUT", purpose, "calibration")
                      for purpose in ("income", "occupation")}
        cal_pair_sha = store.manifest["real_file_specs"]["CALIBRATION_HELDOUT:features"]["row_unit_pair_sha256"]
        stochastic = all(stage.stochastic for stage in stages_for_arm(args.arm, args.seed,
                                                                      hyper.steps_per_encoder))
        calibration = fit_calibration(
            fitted.model, arm_id=args.arm, model_seed=args.seed,
            feature_record=cal_features, task_label_records=cal_labels,
            source_id=admitted.source_id, stochastic=stochastic, sampler_secret=secret,
            expected_model_sha256=model_fingerprint(fitted.model),
            expected_schema_sha256=data.feature_schema_sha256,
            admitted_cal_pair_sha256=cal_pair_sha,
            private_directory=attempt / "calibration",
        )
        phase("CALIBRATION", calibration_record_sha256=calibration.record_file_sha256,
              calibration_replay_max_error=calibration.replay_max_probability_error,
              calibration_paired_forward_passes=calibration.paired_calibration_forward_passes)
        att_features, att_rows, att_release_sha = _release_role(
            store, "ATTACK_FIT_NEW", fitted.model, source_id=admitted.source_id,
            secret=secret, temperatures=calibration.adapter.temperatures,
            calibration_records=calibration.records,
            calibration_record_sha256=calibration.record_file_sha256,
            attempt=attempt, stochastic=stochastic)
        inner_features, inner_rows, inner_release_sha = _release_role(
            store, "INNER_SELECTION", fitted.model, source_id=admitted.source_id,
            secret=secret, temperatures=calibration.adapter.temperatures,
            calibration_records=calibration.records,
            calibration_record_sha256=calibration.record_file_sha256,
            attempt=attempt, stochastic=stochastic)
        phase("RELEASE", release_sha256={"ATTACK_FIT_NEW": att_release_sha,
                                         "INNER_SELECTION": inner_release_sha})
        sex_att = store.read("ATTACK_FIT_NEW", "sex", "attack")
        assert_same_people(att_features, sex_att)
        attack_views, fitted_readers = {}, {}
        for view in ("income", "occupation", "pair"):
            att_view = build_attack_view(
                att_rows, view=view, row_id=att_features["row_id"],
                unit=att_features["unit"], public_model=fitted.model,
                expected_source_id=admitted.source_id,
                calibration_records=calibration.records,
                calibration_record_sha256=calibration.record_file_sha256,
            )
            inner_view = build_attack_view(
                inner_rows, view=view, row_id=inner_features["row_id"],
                unit=inner_features["unit"], public_model=fitted.model,
                expected_source_id=admitted.source_id,
                calibration_records=calibration.records,
                calibration_record_sha256=calibration.record_file_sha256,
            )
            attack_views[view] = (att_view, inner_view)
            fitted_readers[view] = fit_slate(
                att_view, sex_att, inner_view,
                required_readers=tuple(config["attack_readers"]),
                attacker_seed=int(config["attack_seed_offset"]) + args.seed,
                progress_dir=attempt / f"readers_progress__{view}",
            )
        fitted_readers["pair"] = add_ignore_recipient_aliases(
            fitted_readers["pair"], fitted_readers["income"],
            fitted_readers["occupation"])
        attack_receipts = {}
        for view, fit in fitted_readers.items():
            att_view, inner_view = attack_views[view]
            attack_receipts[view] = save_fitted_slate(
                fit, attempt / f"readers__{view}",
                att_view=att_view, inner_view=inner_view)
        phase("ATTACK_FIT", reader_receipt_sha256={view: receipt["private_receipt_sha256"]
                                                   for view, receipt in attack_receipts.items()})
        sex_inner = store.read("INNER_SELECTION", "sex", "screen")
        task, recovery = _task_and_attack_aggregates(
            store, fit_data=data, rows_inner=inner_rows,
            sex_inner=sex_inner, fitted=fitted_readers,
            inner_features=inner_features, seed=args.seed,
        )
        endpoint_input_sha = _save_endpoint_inputs(
            store, seed=args.seed, arm=args.arm, rows_inner=inner_rows,
            inner_features=inner_features, sex_inner=sex_inner,
            fitted=fitted_readers, attempt=attempt)
        aggregate = {"schema": "pcrl-team-b-screen-aggregate-v1",
                     "status": "EXPLORATORY_DEVELOPMENT",
                     "arm": args.arm, "seed": args.seed,
                     "campaign_lock_sha256": args.lock_sha256,
                     "screen_config_sha256": args.config_sha256,
                     "source_sha256": admitted.source_npz_sha256,
                     "fit_model_sha256": model_fingerprint(fitted.model),
                     "private_endpoint_input_receipt_sha256": endpoint_input_sha,
                     "task": task, "recovery": recovery,
                     "role_counts": {role: store.manifest["roles"][role]["rows"]
                                     for role in ("OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT",
                                                  "ATTACK_FIT_NEW", "INNER_SELECTION")}}
        _write_json(attempt / "AGGREGATE.json", aggregate)
        phase("SCORED", aggregate_sha256=_sha(attempt / "AGGREGATE.json"))
        index_sha = _save_attempt_index(
            attempt=attempt, arm=args.arm, seed=args.seed,
            screen_unit_id=args.unit_id,
            source_id=admitted.source_npz_sha256,
            manifest_sha256=args.manifest_sha256,
            campaign_lock_sha256=args.lock_sha256,
            screen_config_sha256=args.config_sha256,
            fitted=fitted, calibration=calibration,
            release_sha256={"ATTACK_FIT_NEW": att_release_sha,
                            "INNER_SELECTION": inner_release_sha},
            attack_receipts=attack_receipts,
            endpoint_receipt_sha256=endpoint_input_sha,
        )
        phase("CUSTODY_INDEX", private_attempt_index_sha256=index_sha)
        status["status"] = "COMPLETED"
        _write_json(attempt / "STATUS.json", status)
        return aggregate
    except BaseException as exc:
        status["status"] = "INCOMPLETE" if isinstance(exc, StopRequested) else "FAILED"
        status["error_type"] = type(exc).__name__
        status["error_message"] = str(exc)[:240]
        _write_json(attempt / "STATUS.json", status)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_term)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--campaign-lock", required=True)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--screen-config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--synthetic-attack-plumbing-receipt", required=True)
    parser.add_argument("--synthetic-attack-plumbing-sha256", required=True)
    parser.add_argument("--private-attempt", required=True)
    args = parser.parse_args()
    run_one(args)


if __name__ == "__main__":
    main()
