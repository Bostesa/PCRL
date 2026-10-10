"""Queue-only conditional T+/L/J/S12/S21/C fitting and private wire exports.

The worker refuses real role reads until R0's committed capacity gate and
protection stage lock match the active shared queue unit and immutable T0.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import numpy as np

from code_witness.learner import FitData, TrainConfig, make_batch_plan
from code_witness.model import HardCodeRelease
from code_witness.service import load_service, save_service
from code_repair.protection import (calibrate_kernel_strength, fit_adversarial,
                                    fit_continue, fit_sequential, state_sha256)
from results.pcrl_latent_and_sampled_methods_v1.r01.frozen_u import FrozenUEncoder
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people, sha_file,
)
from results.pcrl_code_utility_witness_v1.r0.compute_queue import assert_active_from_env


SCHEMA = "pcrl-code-utility-protection-fit-receipt-v1"
LOCK_SCHEMA = "pcrl-code-utility-protection-lock-v1"
ARMS = ("T+", "L", "J", "S12", "S21", "C")
RELEASE_KEYS = frozenset({"row_id", "unit", "code0", "code1", "prob0", "prob1",
                          "decision0", "decision1"})


def scientific_arm_id(short: str, bits: int) -> str:
    if short not in ARMS or bits not in (10, 16):
        raise ValueError("unknown conditional arm or capacity")
    return f"{'TPLUS' if short == 'T+' else short}_B{bits}"


def _write_once(path: Path, writer) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())


def _json_once(path: Path, value: dict) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    _write_once(path, lambda stream: stream.write(payload))
    return hashlib.sha256(payload).hexdigest()


def write_release_npz(path: Path, row_id: np.ndarray, unit: np.ndarray,
                      first: dict, second: dict, bits: int) -> str:
    """Write exactly the private R4 hard-code wire schema, once."""
    ids, groups = np.asarray(row_id), np.asarray(unit)
    if (bits not in (10, 16) or ids.dtype != np.int64 or groups.dtype != np.int64
            or ids.ndim != 1 or groups.shape != ids.shape or not len(ids)
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("ordered admitted IDs/groups and selected bit width required")
    n = len(ids)
    for purpose, wire in enumerate((first, second)):
        classes = (2, 6)[purpose]
        code, prob, decision = (np.asarray(wire[key]) for key in
                                ("code", "probabilities", "decision"))
        if (code.dtype != np.uint8 or code.shape != (n, bits)
                or not np.isin(code, (0, 1)).all()
                or prob.dtype != np.float64 or prob.shape != (n, classes)
                or not np.isfinite(prob).all() or np.any(prob < 0)
                or not np.allclose(prob.sum(axis=1), 1, rtol=0, atol=1e-6)
                or decision.dtype != np.int64 or decision.shape != (n,)
                or not np.array_equal(decision, prob.argmax(axis=1))
                or wire.get("metadata", {}).get("bits") != bits
                or wire.get("metadata", {}).get("purpose") != purpose):
            raise ValueError("release is not the exact code-only selected-capacity wire")
    arrays = {"row_id": ids, "unit": groups,
              "code0": first["code"], "code1": second["code"],
              "prob0": first["probabilities"], "prob1": second["probabilities"],
              "decision0": first["decision"], "decision1": second["decision"]}
    if set(arrays) != RELEASE_KEYS:
        raise AssertionError("private release key inventory changed")
    _write_once(Path(path), lambda stream: np.savez_compressed(stream, **arrays))
    return sha_file(Path(path))


def _committed_at_head(path: Path, expected_sha256: str, root: Path) -> None:
    if sha_file(path) != expected_sha256:
        raise PermissionError("sealed file byte hash differs")
    try:
        rel = path.relative_to(root)
    except ValueError as exc:
        raise PermissionError("sealed file must be inside research worktree") from exc
    shown = subprocess.run(["git", "show", f"HEAD:{rel.as_posix()}"], cwd=root,
                           capture_output=True, check=False)
    if (shown.returncode != 0
            or hashlib.sha256(shown.stdout).hexdigest() != expected_sha256):
        raise PermissionError("stage seal or gate is not committed at local HEAD")


def _read_json_pinned(path: Path, expected_sha256: str) -> dict:
    if sha_file(path) != expected_sha256:
        raise PermissionError("pinned JSON byte hash differs")
    return json.loads(path.read_text())


def _preflight(args: argparse.Namespace, queue: dict) -> tuple[dict, dict, dict, Path]:
    root = Path(__file__).resolve().parents[1]
    stage_path = Path(args.stage_lock).resolve(strict=True)
    gate_path = Path(args.gate_receipt).resolve(strict=True)
    _committed_at_head(stage_path, args.stage_lock_sha256, root)
    _committed_at_head(gate_path, args.gate_receipt_sha256, root)
    stage = _read_json_pinned(stage_path, args.stage_lock_sha256)
    gate = _read_json_pinned(gate_path, args.gate_receipt_sha256)
    seed_key = str(args.seed)
    if (stage.get("schema") != LOCK_SCHEMA
            or stage.get("status") != "LOCALLY_SEALED_PROTECTION"
            or stage.get("gate_receipt_sha256") != args.gate_receipt_sha256
            or stage.get("source_lock_sha256") != queue["seal_sha256"]
            or stage.get("selected_bits") not in (10, 16)
            or stage.get("seeds") != [0, 1]
            or stage.get("arms") != [scientific_arm_id(arm, stage["selected_bits"])
                                         for arm in ARMS]
            or stage.get("lambda_rule") != "FIRST_SHARED_FIT_BATCH_GRADIENT_RATIO_V1"
            or stage.get("continuation_batch_seed_offset") is None
            or int(stage.get("continuation_epochs_by_seed", {}).get(seed_key, 0)) < 1
            or stage.get("role_manifest_sha256") != sha_file(Path(args.role_manifest))
            or stage.get("source_model_sha256_by_seed", {}).get(seed_key)
               != sha_file(Path(args.source_model))
            or stage.get("t0_manifest_sha256_by_seed", {}).get(seed_key)
               != sha_file(Path(args.t0_manifest))
            or stage.get("source_files_sha256", {}).get("code_repair/run_conditional.py")
               != sha_file(Path(__file__))
            or gate.get("schema") != "pcrl-code-witness-capacity-gate-v1"
            or gate.get("status") != "POINT_GATE_PASS"
            or gate.get("all_two_seed_task_reference_point_caps_pass") is not True
            or gate.get("explicit_r0_go") is not True
            or gate.get("source_lock_sha256") != queue["seal_sha256"]
            or gate.get("selected_bits") != stage["selected_bits"]
            or gate.get("t0_by_seed", {}).get(seed_key, {}).get("manifest_sha256")
               != stage["t0_manifest_sha256_by_seed"][seed_key]):
        raise PermissionError("R0 capacity gate and committed protection seal disagree")
    queue_state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    spec = queue_state.get("units", {}).get(args.unit_id, {}).get("spec", {})
    if (spec.get("stage_lock_sha256") != args.stage_lock_sha256
            or spec.get("gate_receipt_sha256") != args.gate_receipt_sha256
            or spec.get("seed") != args.seed):
        raise PermissionError("active queue unit does not bind R0 protection gate and lock")
    expected_sources = stage["source_files_sha256"]
    for relative, expected in expected_sources.items():
        if sha_file(root / relative) != expected:
            raise PermissionError("stage-sealed implementation source changed")
    out = Path(args.private_output).absolute()
    if (out.exists() or out.parent.name != ".private"
            or out.parent.parent.resolve() != stage_path.parent.resolve()
            or out.name != f"protection_s{args.seed}"):
        raise FileExistsError("private protection output path is occupied or outside package")
    public = Path(args.public_receipt).absolute()
    if (public.exists() or public.parent.resolve() != stage_path.parent / "r3"
            or public.name != f"PROTECTION_S{args.seed}_RECEIPT.json"):
        raise FileExistsError("public protection receipt path is occupied or unexpected")
    admission = {"schema": gate["schema"], "status": gate["status"],
                 "all_two_seed_task_reference_point_caps_pass": True,
                 "explicit_r0_go": True, "selected_bits": stage["selected_bits"],
                 "seeds": stage["seeds"],
                 "t0_state_sha256_by_seed": {seed_key: gate["t0_by_seed"][seed_key]["model_sha256"]},
                 "source_lock_sha256": queue["seal_sha256"],
                 "stage_lock_sha256": args.stage_lock_sha256}
    return stage, gate, admission, out


def _calibration_module(stage: dict, root: Path):
    base = "results/pcrl_method_portfolio_v1/shared/"
    paths = {name: root / (base + name) for name in
             ("common_eval.py", "safe_data_guard.py")}
    if any(stage["source_files_sha256"].get(base + name) != sha_file(path)
           for name, path in paths.items()):
        raise PermissionError("inherited H-GLOBAL-TEMP source/dependency not pinned")
    guard_spec = importlib.util.spec_from_file_location(
        "safe_data_guard", paths["safe_data_guard.py"])
    common_spec = importlib.util.spec_from_file_location(
        "code_repair_sealed_common_eval", paths["common_eval.py"])
    if (guard_spec is None or guard_spec.loader is None
            or common_spec is None or common_spec.loader is None):
        raise ImportError("sealed calibration module unavailable")
    guard = importlib.util.module_from_spec(guard_spec)
    common = importlib.util.module_from_spec(common_spec)
    previous = sys.modules.get("safe_data_guard")
    try:
        sys.modules["safe_data_guard"] = guard
        guard_spec.loader.exec_module(guard)
        common_spec.loader.exec_module(common)
    finally:
        if previous is None:
            sys.modules.pop("safe_data_guard", None)
        else:
            sys.modules["safe_data_guard"] = previous
    return common


def _fit_data(store: RoleOnlyStore, encoder: FrozenUEncoder, seed: int
              ) -> tuple[FitData, FitData, tuple[str, ...]]:
    role = "OSF_DEFENSE_FIT"
    features = store.read(role, "features")
    income = store.read(role, "income", "task_fit")
    occupation = store.read(role, "occupation", "task_fit")
    references = store.read(role, f"references_s{seed}")
    sex = store.read(role, "sex", "defense_fit")
    assert_same_people(features, income, occupation, references, sex)
    names = tuple(str(x) for x in features["feature_names"])
    latents = encoder.encode_pair(features["X"], names, role)
    args = (*latents, income["income"], occupation["occupation"],
            references["Ucal_1"], references["Ucal_2"])
    task = FitData(*args, None, features["row_id"], role)
    protected = FitData(*args, sex["sex"], features["row_id"], role)
    return task, protected, names


def _cal_role(store: RoleOnlyStore, encoder: FrozenUEncoder, names: tuple[str, ...]):
    role = "CALIBRATION_HELDOUT"
    features = store.read(role, "features")
    income = store.read(role, "income", "calibration")
    occupation = store.read(role, "occupation", "calibration")
    assert_same_people(features, income, occupation)
    if tuple(str(x) for x in features["feature_names"]) != names:
        raise ValueError("CAL predictor schema differs from FIT")
    return (encoder.encode_pair(features["X"], names, role),
            (income["income"], occupation["occupation"]),
            features["row_id"], features["unit"])


def _fit_temperature(common, model, scaler, source_sha256: str,
                     cal_latent, cal_labels, cal_ids, cal_units,
                     purpose: int, arm_id: str, seed: int) -> dict:
    raw = HardCodeRelease(model, scaler, f"W-{arm_id}-s{seed}", source_sha256)
    code = raw.recipient(purpose, cal_latent[purpose])["code"]
    p = raw.uncalibrated_probabilities_from_code(purpose, code)
    return common.fit_global_temperature(
        p, cal_labels[purpose], task=("income", "occupation")[purpose],
        arm_id=arm_id, model_seed=seed, row_id=cal_ids, unit=cal_units)


def _occupied(code: np.ndarray) -> dict:
    bits = code.shape[1]
    ids = (code.astype(np.uint64) * (1 << np.arange(bits, dtype=np.uint64))).sum(1)
    _, counts = np.unique(ids, return_counts=True)
    p = counts / counts.sum()
    return {"capacity_states": 1 << bits, "occupied_states": len(counts),
            "entropy_bits": float(-(p * np.log2(p)).sum()),
            "rare_state_row_fraction": float(counts[counts < 5].sum() / len(code))}


def _require_t0_scaler_pin(t0: HardCodeRelease, gate: dict, stage: dict,
                           seed: int) -> None:
    """Bind restored scaler bytes to both R0 seals, beyond package self-checks."""
    key = str(seed)
    digest = t0.scaler_sha256
    if (digest != gate.get("t0_by_seed", {}).get(key, {}).get("scaler_sha256")
            or digest != stage.get("t0_scaler_sha256_by_seed", {}).get(key)):
        raise PermissionError("immutable T0 scaler/gate identity differs")


def _require_t0_schema_pin(package_manifest: dict,
                           role_source_schema_sha256: str) -> None:
    if package_manifest.get("source_schema_sha256") != role_source_schema_sha256:
        raise PermissionError("immutable T0 source schema differs from admitted role")


def _batch_plan_sha256(plan: tuple, row_ids: np.ndarray) -> str:
    """Hash exact epoch, batch boundary and ordered FIT-person sequence."""
    digest = hashlib.sha256(b"pcrl-code-witness-batch-plan-v1\0")
    ids = np.asarray(row_ids, dtype=np.int64)
    digest.update(len(plan).to_bytes(8, "little"))
    for epoch in plan:
        digest.update(len(epoch).to_bytes(8, "little"))
        for chunk in epoch:
            ordered = np.ascontiguousarray(ids[np.asarray(chunk, dtype=np.int64)])
            digest.update(len(ordered).to_bytes(8, "little"))
            digest.update(ordered.tobytes())
    return digest.hexdigest()


def _prepare_private_parent(out: Path) -> None:
    """Ensure a local owner-only package root exists before expensive FIT work."""
    parent = out.parent
    if parent.name != ".private" or parent.is_symlink():
        raise PermissionError("private parent must be a direct directory")
    parent.mkdir(mode=0o700, exist_ok=True)
    info = parent.stat()
    if (not parent.is_dir() or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise PermissionError("private parent must be owner-only")


def run(args: argparse.Namespace) -> dict:
    # This check must precede every supplied path open and every role read.
    queue = assert_active_from_env(args.unit_id, "protection")
    stage, gate, admission, out = _preflight(args, queue)
    _prepare_private_parent(out)
    seed, bits = args.seed, stage["selected_bits"]
    root = Path(__file__).resolve().parents[1]
    common = _calibration_module(stage, root)
    store = RoleOnlyStore(Path(args.role_manifest))
    features = store.read("OSF_DEFENSE_FIT", "features")
    names = tuple(str(x) for x in features["feature_names"])
    encoder = FrozenUEncoder.load(Path(args.source_model),
                                  stage["source_model_sha256_by_seed"][str(seed)],
                                  names, store.manifest["source_schema_sha256"], seed)
    task, protected, names = _fit_data(store, encoder, seed)
    cal_latent, cal_labels, cal_ids, cal_units = _cal_role(store, encoder, names)
    t0 = load_service(Path(args.t0_manifest),
                      expected_manifest_sha256=stage["t0_manifest_sha256_by_seed"][str(seed)],
                      expected_source_sha256=encoder.model_sha256,
                      expected_bits=bits)
    _require_t0_schema_pin(
        _read_json_pinned(Path(args.t0_manifest),
                          stage["t0_manifest_sha256_by_seed"][str(seed)]),
        store.manifest["source_schema_sha256"])
    initial = {name: value.detach().cpu().clone()
               for name, value in t0.model.state_dict().items()}
    if state_sha256(initial) != admission["t0_state_sha256_by_seed"][str(seed)]:
        raise PermissionError("immutable T0 restored model differs from capacity gate")
    if t0.scaler.dim != protected.h0.shape[1]:
        raise PermissionError("immutable T0 scaler dimension differs")
    _require_t0_scaler_pin(t0, gate, stage, seed)
    ec = int(stage["continuation_epochs_by_seed"][str(seed)])
    config = TrainConfig(epochs=ec, patience=max(ec, 1))
    plan = make_batch_plan(np.arange(protected.rows),
                           seed=seed + int(stage["continuation_batch_seed_offset"]),
                           epochs=ec, batch_size=config.batch_size,
                           sex=protected.sex)
    if len(plan[0][0]) != 256:
        raise ValueError("sealed gradient-ratio first FIT batch must have 256 people")
    assert_active_from_env(args.unit_id, "protection")
    gradient = calibrate_kernel_strength(
        protected, config=config, seed=seed, bits=bits, init_state=initial,
        scaler=t0.scaler, first_batch=plan[0][0], admission=admission)
    lam = gradient["privacy_weight"]
    common_fit = dict(config=config, seed=seed, bits=bits,
                      init_state=initial, scaler=t0.scaler,
                      batch_plan=plan, admission=admission)
    fits = {}
    assert_active_from_env(args.unit_id, "protection")
    fits["T+"] = fit_continue(task, mode="T+", privacy_weight=0, **common_fit)
    for arm in ("L", "J"):
        assert_active_from_env(args.unit_id, "protection")
        fits[arm] = fit_continue(protected, mode=arm,
                                 privacy_weight=lam, **common_fit)
    frozen_calibrators = {}
    for arm in ("S12", "S21"):
        assert_active_from_env(args.unit_id, "protection")
        def calibrate_first(model, purpose, arm_id=arm):
            return _fit_temperature(common, model, t0.scaler, encoder.model_sha256,
                                    cal_latent, cal_labels, cal_ids, cal_units,
                                    purpose, scientific_arm_id(arm_id, bits), seed)
        fits[arm] = fit_sequential(protected, mode=arm, privacy_weight=lam,
                                    calibrate_first=calibrate_first, **common_fit)
        frozen_calibrators[arm] = fits[arm].first_calibrator
    assert_active_from_env(args.unit_id, "protection")
    fits["C"] = fit_adversarial(protected, coefficient=1.0, **common_fit)
    out.mkdir(mode=0o700)
    packages = {}
    releases = {}
    fit_receipt = {"seed": seed, "bits": bits,
                   "t0_model_sha256": state_sha256(initial),
                   "continuation_epochs": ec,
                   "batch_size": config.batch_size,
                   "batch_plan_sha256": _batch_plan_sha256(plan, protected.row_ids),
                   "batch_seed_offset": int(stage["continuation_batch_seed_offset"]),
                   "gradient_ratio": gradient,
                   "optimizer_updates": {}, "critic_updates": {},
                   "training_history": {},
                   "sequential_first_code_sha256": {},
                   "sequential_first_state_sha256": {}}
    for arm in ARMS:
        assert_active_from_env(args.unit_id, "protection")
        fitted = fits[arm]
        public_id = scientific_arm_id(arm, bits)
        if arm in ("S12", "S21"):
            model = fitted.model
            first = 0 if arm == "S12" else 1
            records = [None, None]
            records[first] = frozen_calibrators[arm]
            records[1 - first] = _fit_temperature(
                common, model, t0.scaler, encoder.model_sha256,
                cal_latent, cal_labels, cal_ids, cal_units, 1 - first, public_id, seed)
            fit_receipt["optimizer_updates"][public_id] = list(fitted.updates_by_purpose)
            fit_receipt["training_history"][public_id] = {
                "first": fitted.first_history, "second": fitted.second_history}
            fit_receipt["sequential_first_code_sha256"][public_id] = fitted.first_code_sha256
            fit_receipt["sequential_first_state_sha256"][public_id] = fitted.first_state_sha256
        else:
            model = fitted.model
            records = [_fit_temperature(common, model, t0.scaler,
                                        encoder.model_sha256, cal_latent, cal_labels,
                                        cal_ids, cal_units, i, public_id, seed)
                       for i in (0, 1)]
            fit_receipt["optimizer_updates"][public_id] = fitted.optimizer_updates
            fit_receipt["training_history"][public_id] = fitted.history
            if arm == "C":
                fit_receipt["critic_updates"][public_id] = fitted.critic_updates
        release = HardCodeRelease(model, t0.scaler, f"W-{public_id}-s{seed}-v1",
                                  encoder.model_sha256, tuple(records))
        releases[public_id] = release
    # Store an identical complete stage receipt inside every private package.
    for public_id, release in releases.items():
        assert_active_from_env(args.unit_id, "protection")
        package = save_service(out / public_id, release, source_seed=seed,
                               arm_id=public_id,
                               role_manifest_sha256=stage["role_manifest_sha256"],
                               source_schema_sha256=store.manifest["source_schema_sha256"],
                               fit_receipt=fit_receipt)
        packages[public_id] = {key: value for key, value in package.items()
                               if key != "manifest_path"}
    role_receipts = {}
    for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        assert_active_from_env(args.unit_id, "protection")
        features = store.read(role, "features")
        if tuple(str(x) for x in features["feature_names"]) != names:
            raise ValueError("release role predictor order differs")
        latents = encoder.encode_pair(features["X"], names, role)
        role_receipts[role] = {}
        for arm, release in releases.items():
            first = release.recipient(0, latents[0])
            second = release.recipient(1, latents[1])
            digest = write_release_npz(
                out / f"release__{arm}__{role}.npz", features["row_id"],
                features["unit"], first, second, bits)
            role_receipts[role][arm] = {
                "rows": len(features["row_id"]),
                "private_release_sha256": digest,
                "manifest_sha256": packages[arm]["manifest_sha256"],
                "weights_sha256": packages[arm]["weights_sha256"],
                "recipient0": _occupied(first["code"]),
                "recipient1": _occupied(second["code"]),
            }
    _json_once(out / "private_index.json", {
        "schema": SCHEMA, "seed": seed, "unit_id": args.unit_id,
        "package_paths": {arm: str(out / arm / "manifest.json") for arm in releases},
        "release_paths": {role: {arm: str(out / f"release__{arm}__{role}.npz")
                                for arm in releases} for role in role_receipts},
    })
    public = {"schema": SCHEMA, "status": "PROTECTION_RELEASES_READY",
              "seed": seed, "bits": bits, "unit_id": args.unit_id,
              "source_lock_sha256": queue["seal_sha256"],
              "stage_lock_sha256": args.stage_lock_sha256,
              "gate_receipt_sha256": args.gate_receipt_sha256,
              "source_sha256": encoder.model_sha256,
              "role_manifest_sha256": stage["role_manifest_sha256"],
              "t0_manifest_sha256": stage["t0_manifest_sha256_by_seed"][str(seed)],
              "fit": fit_receipt, "packages": packages, "roles": role_receipts,
              "private_payload_scope": "six code-only model packages and ATT/INNER row releases"}
    _json_once(Path(args.public_receipt), public)
    return public


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("unit-id", "stage-lock", "stage-lock-sha256", "gate-receipt",
                 "gate-receipt-sha256", "role-manifest", "source-model",
                 "t0-manifest", "private-output", "public-receipt"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    result = run(parser.parse_args())
    print(json.dumps({"status": result["status"], "seed": result["seed"],
                      "unit_id": result["unit_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
