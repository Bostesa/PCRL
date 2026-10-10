"""Queue-only, seed-generic D follow-up for L, S12, S21, C and F.

This worker requires an already completed T/D-K first package for the same
seed. Seed 1 therefore awaits its own sealed first pair under a dated R00
amendment. It does not read any role before the active d_followup queue check.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from results.pcrl_latent_and_sampled_methods_v1.r01.frozen_u import FrozenUEncoder
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people, sha_file,
)
from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import assert_active_from_env
from results.pcrl_utility_reference_repair_v1.r1.fit_splits import outer_split
from ..impl.fit import TrainConfig, make_batch_plan
from ..impl.model import FixedScaler, HardCodeModel, HardCodeRelease
from ..impl.service import save_service
from ..impl.run_first_seed import (
    _batch_plan_receipt, _common_eval, _fit_data, _json_once, _npz_once,
    _occupied,
)
from .fit_c import fit_c_stage
from .fit_f import fit_f_reference
from .f_service import save_f_service
from .local_sequential import fit_local_or_sequential
from .unchanged import UnchangedLatentRelease


SCHEMA = "latent-repair-followup-controls-receipt-v1"


def _warm_state(path: Path, expected_sha256: str, dim: int) -> dict[str, torch.Tensor]:
    if sha_file(path) != expected_sha256:
        raise PermissionError("first-pair warm-state byte hash mismatch")
    model = HardCodeModel(dim, (2, 6))
    template = model.state_dict()
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(template):
            raise ValueError("first-pair warm-state tensor inventory mismatch")
        arrays = {name: np.asarray(archive[name]).copy() for name in template}
    for name, value in arrays.items():
        if (value.shape != tuple(template[name].shape) or value.dtype != np.float32
                or not np.isfinite(value).all()):
            raise ValueError("first-pair warm-state tensor invalid")
    return {name: torch.from_numpy(value) for name, value in arrays.items()}


def _amendment_guard(args: argparse.Namespace, queue: dict, seal: Path) -> None:
    """Bind this worker and both first receipts to R03's latest D amendment."""
    amendment_path = Path(args.amendment).resolve(strict=True)
    if (amendment_path.parent != seal.parent
            or sha_file(amendment_path) != args.amendment_sha256):
        raise PermissionError("dated D controls amendment path/hash mismatch")
    queue_state = Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json"
    state = json.loads(queue_state.read_text())
    latest = [entry for entry in state.get("amendments", [])
              if "d_followup" in entry.get("stages", [])]
    unit = state.get("units", {}).get(args.unit_id, {})
    if (not latest or latest[-1].get("sha256") != args.amendment_sha256
            or Path(latest[-1]["path"]).resolve() != amendment_path
            or unit.get("spec", {}).get("amendment_sha256") != args.amendment_sha256
            or queue["seal_sha256"] != args.seal_sha256):
        raise PermissionError("R03 does not bind the latest D controls amendment")
    amendment = json.loads(amendment_path.read_text())
    receipts = amendment.get("first_receipt_sha256_by_seed", {})
    sources = amendment.get("source_seed_sha256_by_seed", {})
    if (amendment.get("schema") != "pcrl-d-followup-amendment-v1"
            or amendment.get("status") != "LOCALLY_SEALED_D_FOLLOWUP"
            or amendment.get("effective_seal_sha256") != args.seal_sha256
            or amendment.get("controls_worker_sha256") != sha_file(Path(__file__))
            or set(receipts) != {"0", "1"}
            or any(not isinstance(value, str) or len(value) != 64
                   for value in receipts.values())
            or receipts.get(str(args.seed)) != args.first_receipt_sha256
            or amendment.get("role_manifest_sha256") != args.role_manifest_sha256
            or set(sources) != {"0", "1"}
            or any(not isinstance(value, str) or len(value) != 64
                   for value in sources.values())
            or sources.get(str(args.seed)) != args.source_sha256):
        raise PermissionError("A2 amendment does not bind controls and seed sources")


def _admit(args: argparse.Namespace) -> tuple[dict, dict, Path, Path]:
    queue = assert_active_from_env(args.unit_id, "d_followup")
    seal = Path(args.seal).resolve(strict=True)
    if (sha_file(seal) != args.seal_sha256
            or queue["seal_sha256"] != args.seal_sha256):
        raise PermissionError("follow-up effective seal byte hash mismatch")
    protocol = json.loads(seal.read_text())
    if (protocol.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1"
            or protocol.get("status") != "LOCALLY_SEALED_D_STAGE"
            or args.seed not in (0, 1)):
        raise PermissionError("D follow-up effective protocol is unavailable")
    _amendment_guard(args, queue, seal)
    sources = protocol.get("source_files_sha256", {})
    if (sources.get("source_role_manifest") != args.role_manifest_sha256
            or sources.get("inherited_calibration_code") != args.common_eval_sha256):
        raise PermissionError("role/CAL source not bound by effective seal")
    for relative, expected in sources.items():
        if relative in ("source_role_manifest", "inherited_calibration_code"):
            continue
        if sha_file(seal.parent / relative) != expected:
            raise PermissionError(f"effective source changed: {relative}")
    manifest = Path(args.role_manifest).resolve(strict=True)
    if sha_file(manifest) != args.role_manifest_sha256:
        raise PermissionError("role manifest byte hash mismatch")
    role = json.loads(manifest.read_text())
    identity = protocol.get("role_identity_and_file_hashes", {})
    if (role.get("role_file_sha256") != identity.get("role_file_sha256")
            or role.get("roles") != identity.get("role_membership")
            or role.get("source_schema_sha256") != identity.get("ordered_predictors_sha256")
            or identity.get("source_model_sha256", {}).get(str(args.seed)) != args.source_sha256):
        raise PermissionError("admitted roles or U source differ from seal")
    first_receipt_path = Path(args.first_receipt).resolve(strict=True)
    first = json.loads(first_receipt_path.read_text())
    if (first.get("schema") != "latent-repair-first-paired-seed-receipt-v1"
            or first.get("status") != "T_DK_RELEASES_READY"
            or first.get("seed") != args.seed
            or args.first_seal_sha256 != args.seal_sha256
            or first.get("role_manifest_sha256") != args.role_manifest_sha256
            or first.get("source_sha256") != args.source_sha256
            or first.get("common_eval_sha256") != args.common_eval_sha256
            or first.get("seal_sha256") != args.first_seal_sha256
            or set(first.get("packages", {})) != {"T", "D-K"}
            or len(first.get("warm_full_state_sha256", "")) != 64
            or sha_file(first_receipt_path) != args.first_receipt_sha256):
        raise PermissionError("first T/D-K receipt is not the pinned seed/source")
    first_private = Path(args.first_private_output).resolve(strict=True)
    if (first_private.parent.name != ".private"
            or first_private.parent.parent.resolve() != seal.parent
            or first_private.name != f"d_first_s{args.seed}"):
        raise PermissionError("first-pair private package has wrong location")
    if not (first_private / "private_index.json").is_file():
        raise PermissionError("first-pair private index missing")
    out = Path(args.private_output).absolute()
    if (out.exists() or out.parent.name != ".private"
            or out.parent.parent.resolve() != seal.parent
            or out.name != f"d_followup_s{args.seed}"):
        raise FileExistsError("follow-up output must be a fresh owner-only directory")
    return protocol, first, first_private, out


def run(args: argparse.Namespace) -> dict:
    protocol, first, first_private, out = _admit(args)
    common = _common_eval(Path(args.shared_dir).resolve(strict=True),
                          args.common_eval_sha256)
    model_pt = Path(args.model_pt).resolve(strict=True)
    if sha_file(model_pt) != args.source_sha256:
        raise PermissionError("U checkpoint source byte hash mismatch")
    # The first pair's selected T lengths and full warm state are immutable.
    e0 = int(first["fit"]["warm_selected_epoch"])
    ec = int(first["fit"]["continuation_epoch"])
    lam = float(first["fit"]["gradient_ratio"]["privacy_weight"])
    if (e0 not in (5, 10, 15, 20) or ec not in (5, 10, 15, 20)
            or not np.isfinite(lam) or lam < 0):
        raise PermissionError("first-pair continuation choices are invalid")
    out.mkdir(mode=0o700)
    store = RoleOnlyStore(Path(args.role_manifest))
    feature = store.read("OSF_DEFENSE_FIT", "features")
    names = tuple(str(x) for x in feature["feature_names"])
    encoder = FrozenUEncoder.load(model_pt, args.source_sha256, names,
                                  store.manifest["source_schema_sha256"], args.seed)
    _, protected, unit, _ = _fit_data(store, encoder, args.seed)
    scaler = FixedScaler.fit(protected.h0, protected.h1)
    warm = _warm_state(first_private / "warm_full_state.npz",
                       first["warm_full_state_sha256"], scaler.dim)
    plan = make_batch_plan(np.arange(protected.rows), seed=args.seed, epochs=ec,
                           batch_size=256, sex=protected.sex)
    config = TrainConfig(epochs=ec, privacy_weight=lam)
    local = {}
    for mode in ("L", "S12", "S21"):
        assert_active_from_env(args.unit_id, "d_followup")
        local[mode] = fit_local_or_sequential(protected, config, args.seed,
                                              warm, scaler, mode,
                                              args.source_sha256, plan)
    assert_active_from_env(args.unit_id, "d_followup")
    critic = fit_c_stage(protected, TrainConfig(epochs=ec), args.seed,
                         warm, scaler, np.arange(protected.rows),
                         coefficient=1.0, critic_steps_per_update=2,
                         batch_plan=plan)
    fold_train, fold_val, fold_receipt = outer_split(unit, protected.y0,
                                                      protected.y1, args.seed)
    assert_active_from_env(args.unit_id, "d_followup")
    f = fit_f_reference(protected, TrainConfig(epochs=40, privacy_weight=0,
                        checkpoints=tuple(range(5, 41, 5))), args.seed,
                        scaler, fold_train, fold_val, args.source_sha256)
    cal_feature = store.read("CALIBRATION_HELDOUT", "features")
    cal_income = store.read("CALIBRATION_HELDOUT", "income", "calibration")
    cal_occupation = store.read("CALIBRATION_HELDOUT", "occupation", "calibration")
    assert_same_people(cal_feature, cal_income, cal_occupation)
    cal_latent = encoder.encode_pair(cal_feature["X"], names, "CALIBRATION_HELDOUT")
    raw_releases = {
        **{mode: HardCodeRelease(result.fit.model, scaler,
                                f"D-{mode}-s{args.seed}-v1", args.source_sha256)
           for mode, result in local.items()},
        "C": HardCodeRelease(critic.model, scaler, f"D-C-s{args.seed}-v1",
                             args.source_sha256),
    }
    releases: dict[str, HardCodeRelease | UnchangedLatentRelease] = {}
    packages = {}
    fit_receipt = {
        "source_seed": args.seed, "first_receipt_sha256": args.first_receipt_sha256,
        "warm_selected_epoch": e0, "continuation_epoch": ec,
        "privacy_weight": lam, "full_batch_plan": _batch_plan_receipt(plan, protected.rows),
        "optimizer_updates": {mode: result.updates_by_purpose
                              for mode, result in local.items()},
        "sequential_first_fit_code_sha256": {
            mode: result.first_fit_code_sha256 for mode, result in local.items()
            if result.first_fit_code_sha256 is not None},
        "c_optimizer_updates": critic.optimizer_updates,
        "c_critic_updates": critic.critic_updates,
        "f_fold": fold_receipt,
        "f_selected_epochs": f.selected_epochs,
        "f_full_fit_updates_by_purpose": f.full_fit_updates_by_purpose,
    }
    for arm_id, raw in {**raw_releases, "F": f.release}.items():
        records = []
        for purpose, labels, task_name in ((0, cal_income["income"], "income"),
                                           (1, cal_occupation["occupation"], "occupation")):
            assert_active_from_env(args.unit_id, "d_followup")
            if arm_id == "F":
                scaled = raw.scaler.transform(purpose, cal_latent[purpose])
                p = raw.uncalibrated_probabilities_from_features(purpose, scaled)
            else:
                code = raw.recipient(purpose, cal_latent[purpose])["code"]
                p = raw.uncalibrated_probabilities_from_code(purpose, code)
            records.append(common.fit_global_temperature(
                p, labels, task=task_name, arm_id=arm_id, model_seed=args.seed,
                row_id=cal_feature["row_id"], unit=cal_feature["unit"]))
        if arm_id == "F":
            release = UnchangedLatentRelease(raw.heads, scaler, raw.version,
                                             args.source_sha256, tuple(records))
            package = save_f_service(out / arm_id, release, source_seed=args.seed,
                                     role_manifest_sha256=args.role_manifest_sha256,
                                     fit_receipt=fit_receipt)
        else:
            release = HardCodeRelease(raw.model, scaler, raw.version,
                                      args.source_sha256, tuple(records))
            package = save_service(out / arm_id, release, source_seed=args.seed,
                                   arm_id=arm_id,
                                   role_manifest_sha256=args.role_manifest_sha256,
                                   fit_receipt=fit_receipt)
        releases[arm_id] = release
        packages[arm_id] = package
    role_receipts = {}
    for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        feature = store.read(role, "features")
        if tuple(str(x) for x in feature["feature_names"]) != names:
            raise ValueError("follow-up service feature schema changed")
        latents = encoder.encode_pair(feature["X"], names, role)
        role_receipts[role] = {}
        for arm_id, release in releases.items():
            r0, r1 = (release.recipient(purpose, latents[purpose])
                      for purpose in (0, 1))
            arrays = {"row_id": feature["row_id"], "unit": feature["unit"],
                      "prob0": r0["probabilities"], "decision0": r0["decision"],
                      "prob1": r1["probabilities"], "decision1": r1["decision"]}
            if arm_id == "F":
                arrays.update(features0=r0["features"], features1=r1["features"])
            else:
                arrays.update(code0=r0["code"], code1=r1["code"])
            digest = _npz_once(out / f"release__{arm_id}__{role}.npz", arrays)
            entry = {"rows": len(feature["row_id"]),
                     "private_release_sha256": digest}
            if arm_id != "F":
                entry["recipient0"] = _occupied(r0["code"])
                entry["recipient1"] = _occupied(r1["code"])
            role_receipts[role][arm_id] = entry
    receipt = {"schema": SCHEMA, "status": "D_FOLLOWUP_RELEASES_READY",
               "seed": args.seed, "unit_id": args.unit_id,
               "seal_sha256": args.seal_sha256,
               "first_receipt_sha256": args.first_receipt_sha256,
               "source_sha256": args.source_sha256,
               "role_manifest_sha256": args.role_manifest_sha256,
               "common_eval_sha256": args.common_eval_sha256,
               "fit": fit_receipt,
               "packages": {arm: {k: v for k, v in package.items()
                                  if k != "manifest_path"}
                            for arm, package in packages.items()},
               "roles": role_receipts,
               "private_payload_scope": "five owner-only model packages and role releases"}
    _json_once(out / "private_index.json", {
        "schema": SCHEMA, "unit_id": args.unit_id,
        "package_paths": {arm: package["manifest_path"]
                          for arm, package in packages.items()},
        "release_files": [str(out / f"release__{arm}__{role}.npz")
                          for role in role_receipts for arm in releases],
    })
    _json_once(Path(args.public_receipt), receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("seal", "seal-sha256", "unit-id", "role-manifest",
                 "role-manifest-sha256", "shared-dir", "common-eval-sha256",
                 "model-pt", "source-sha256", "first-receipt",
                 "first-receipt-sha256", "first-seal-sha256",
                 "first-private-output", "private-output", "public-receipt",
                 "amendment", "amendment-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "seed": result["seed"],
                      "unit_id": result["unit_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
