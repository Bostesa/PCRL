"""Run B's real-role null/shuffled/planted controls after a saved screen arm.

One S3 real-data unit covers the three exact recipient/coalition views and
charges six new LR fits. It opens only admitted ATTACK_FIT_NEW and
INNER_SELECTION role files plus the named private B screen attempt. A failed
view leaves its report in private custody and blocks that arm's endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

import numpy as np
import torch

from attack_bridge import build_attack_view
from bottleneck import SavedRelease, model_fingerprint
from control_plan import ARMS, stages_for_arm
from screen_driver import _make_initial_models, _sha, _write_json


def _parent_attempt_matches(index: dict, parent: dict, *, screen_unit_id: str,
                            attempt: Path) -> bool:
    """Bind a real-control unit to the exact completed screen attempt."""
    if index.get("screen_unit_id") != screen_unit_id:
        return False
    argv = parent.get("spec", {}).get("argv", ())
    locations = [position for position, token in enumerate(argv)
                 if token == "--private-attempt"]
    if len(locations) != 1 or locations[0] + 1 >= len(argv):
        return False
    declared = Path(argv[locations[0] + 1])
    return declared.is_absolute() and declared.resolve() == attempt.resolve()


def _gate(args):
    import attack_eval
    import attack_controls
    from compute_queue import Queue

    if ".private" not in args.attempt.parts:
        raise PermissionError("B real controls require a named private saved-release attempt")
    paths = ((args.campaign_lock, args.lock_sha256),
             (args.screen_config, args.config_sha256),
             (args.data_manifest, args.manifest_sha256),
             (args.attempt / "ATTEMPT_INDEX.json", args.attempt_index_sha256))
    for path, digest in paths:
        if not Path(path).is_file() or _sha(path) != digest:
            raise PermissionError("B postrelease probe input identity changed")
    lock = json.loads(Path(args.campaign_lock).read_text())
    config = json.loads(Path(args.screen_config).read_text())
    index = json.loads((args.attempt / "ATTEMPT_INDEX.json").read_text())
    key = f"{args.arm}:s{args.seed}"
    family = lock.get("families", {}).get("B", {})
    if (lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE"
            or lock.get("adult_fit_permission") is not True
            or lock.get("fit_admission") != "B_ONLY"
            or family.get("fit_admission") != "ALLOWED"
            or args.arm not in lock.get("admitted_arms", {}).get("B", ())
            or family.get("real_control_unit_ids", {}).get(key) != args.unit_id
            or family.get("real_control_probe_sha256") != _sha(Path(__file__))
            or family.get("screen_config_sha256") != args.config_sha256
            or lock.get("data_access_manifest_sha256") != args.manifest_sha256
            or lock.get("attack_protocol", {}).get("source_module_sha256")
               != _sha(Path(attack_eval.__file__))
            or lock.get("shared_attack_controls_sha256") != _sha(Path(attack_controls.__file__))
            or config.get("status") != "EFFECTIVE"
            or config.get("science_permission") is not True
            or args.arm not in config.get("arms", ())
            or args.seed not in config.get("seeds", ())
            or index.get("schema") != "pcrl-team-b-private-attempt-index-v1"
            or index.get("status") != "FROZEN_FOR_INDEPENDENT_REPLAY"
            or index.get("arm") != args.arm or index.get("seed") != args.seed
            or index.get("campaign_lock_sha256") != args.lock_sha256
            or index.get("screen_config_sha256") != args.config_sha256
            or index.get("data_manifest_sha256") != args.manifest_sha256):
        raise PermissionError("B real-control unit is absent from the effective S1 seal")
    state = Queue(args.queue_root).status()
    unit = state["units"].get(args.unit_id)
    parent = state["units"].get(args.screen_unit_id)
    if (not unit or unit.get("status") != "running"
            or unit.get("child_pgid") != os.getpgrp()
            or unit["spec"].get("mode") != "real_data"
            or unit["spec"].get("stage") != "postrelease_control"
            or unit["spec"].get("team") != "B"
            or unit["spec"].get("arm") != args.arm
            or unit["spec"].get("seed") != args.seed
            or not any(Path(arg).resolve() == Path(__file__).resolve()
                       for arg in unit["spec"].get("argv", ()))
            or args.screen_unit_id not in unit["spec"].get("deps", ())
            or not parent or parent.get("status") != "done"
            or parent["spec"].get("team") != "B"
            or parent["spec"].get("stage") != "screen"
            or parent["spec"].get("arm") != args.arm
            or parent["spec"].get("seed") != args.seed
            or not _parent_attempt_matches(index, parent,
                                           screen_unit_id=args.screen_unit_id,
                                           attempt=args.attempt)
            or not state["gate"].get("safe_inputs")
            or not state["gate"].get("occupancy_confirmed")
            or not Queue._local_seal_ok(state["gate"])):
        raise PermissionError("S3 postrelease probe queue/parent gate is not open")
    return config, index


def _restore_public_model(args, config: dict, index: dict, feature_names: tuple[str, ...]):
    if (index["source_code_sha256"]["release_adapter"]
            != _sha(Path(__file__).with_name("bottleneck.py"))
            or index["source_code_sha256"]["attack_view_builder"]
            != _sha(Path(__file__).with_name("attack_bridge.py"))
            or index["source_code_sha256"]["wire_flattener"]
            != _sha(Path(__file__).with_name("attack_views.py"))
            or index["source_code_sha256"]["screen_driver"]
            != _sha(Path(__file__).with_name("screen_driver.py"))):
        raise ValueError("B release/view source changed after screen fit")
    model, _ = _make_initial_models(
        type("Schema", (), {"feature_names": feature_names})(), args.seed, config)
    stages = stages_for_arm(args.arm, args.seed,
                            int(config["hyperparameters"]["steps_per_encoder"]))
    if args.arm == "B-LOCAL":
        name, digest = "B-LOCAL__assembled.pt", index["fit_artifact_sha256"]["assembled_local_pair"]
        expected_format = "TeamBAssembledLocalPairV1"
    else:
        final = stages[-1]
        name = final.stage_id.replace("/", "__") + ".pt"
        digest = index["fit_artifact_sha256"][final.stage_id]
        expected_format = "TeamBStageCheckpointV1"
    path = args.attempt / "fit" / name
    if _sha(path) != digest:
        raise ValueError("B final checkpoint bytes changed")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != expected_format:
        raise ValueError("B final checkpoint format changed")
    model.load_state_dict(state["model"], strict=True)
    model.eval().requires_grad_(False)
    if model_fingerprint(model) != index["fit_model_sha256"]:
        raise ValueError("B final public model differs from saved release")
    return model


def run(args):
    from attack_controls import run_control_probes
    from common_eval import assert_same_people, load_calibrator
    from safe_data_guard import SafeRoleStore, sha_arr

    config, index = _gate(args)
    output_dir = args.private_output.resolve()
    if ".private" not in output_dir.parts or output_dir.exists():
        raise ValueError("A new private B real-control output is required")
    output_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if stat.S_IMODE(output_dir.parent.stat().st_mode) != 0o700:
        raise ValueError("Private real-control parent must have mode 0700")
    output_dir.mkdir(mode=0o700)
    store = SafeRoleStore(args.data_manifest)
    if store.manifest.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED":
        raise PermissionError("B real-control inputs lack S4 role-only admission")
    att_features = store.read("ATTACK_FIT_NEW", "features")
    inner_features = store.read("INNER_SELECTION", "features")
    att_sex = store.read("ATTACK_FIT_NEW", "sex", "attack")
    inner_sex = store.read("INNER_SELECTION", "sex", "screen")
    assert_same_people(att_features, att_sex)
    assert_same_people(inner_features, inner_sex)
    names = tuple(str(name) for name in att_features["feature_names"])
    if names != tuple(str(name) for name in inner_features["feature_names"]):
        raise ValueError("B attack/screen feature schemas differ")
    model = _restore_public_model(args, config, index, names)
    calibration, releases, rows = {}, {}, {}
    for purpose in ("income", "occupation"):
        path = args.attempt / "calibration" / f"{purpose}.json"
        calibration[purpose] = load_calibrator(path, index["calibration_record_sha256"][purpose])
    for role, features in (("ATTACK_FIT_NEW", att_features),
                           ("INNER_SELECTION", inner_features)):
        path = args.attempt / f"release__{role}.json"
        if _sha(path) != index["saved_release_sha256"][role]:
            raise ValueError("B saved release bytes changed")
        releases[role] = SavedRelease(path)
        if (releases[role].model_sha256 != index["fit_model_sha256"]
                or releases[role].source_id != index["source_sha256"]):
            raise ValueError("B saved release model/source changed")
        rows[role] = [(str(int(person)), {
            purpose: releases[role].get(purpose, str(int(person)))
            for purpose in ("income", "occupation")})
            for person in np.asarray(features["row_id"], dtype=np.int64)]
    report_hashes = {}
    for view in ("income", "occupation", "pair"):
        att_view = build_attack_view(
            rows["ATTACK_FIT_NEW"], view=view,
            row_id=att_features["row_id"], unit=att_features["unit"],
            public_model=model, expected_source_id=index["source_sha256"],
            calibration_records=calibration,
            calibration_record_sha256=index["calibration_record_sha256"])
        inner_view = build_attack_view(
            rows["INNER_SELECTION"], view=view,
            row_id=inner_features["row_id"], unit=inner_features["unit"],
            public_model=model, expected_source_id=index["source_sha256"],
            calibration_records=calibration,
            calibration_record_sha256=index["calibration_record_sha256"])
        receipt_path = args.attempt / f"readers__{view}" / "READERS_RECEIPT.json"
        if _sha(receipt_path) != index["reader_receipt_sha256"][view]:
            raise ValueError("B trained-reader receipt changed")
        receipt = json.loads(receipt_path.read_text())
        if (att_view.content_hashes() != receipt["att_view_content_sha256"]
                or inner_view.content_hashes() != receipt["inner_view_content_sha256"]):
            raise ValueError("Real-control view differs from original fresh-reader view")
        probe = run_control_probes(att_view, att_sex, inner_view, inner_sex)
        report = {"schema": "pcrl-team-b-real-attack-controls-v1",
                  "status": probe["status"], "arm": args.arm, "seed": args.seed,
                  "view": view, "queue_unit_id": args.unit_id,
                  "screen_unit_id": args.screen_unit_id,
                  "attempt_index_sha256": args.attempt_index_sha256,
                  "att_row_unit_pair_sha256": sha_arr(np.column_stack((att_view.row_id, att_view.unit))),
                  "inner_row_unit_pair_sha256": sha_arr(np.column_stack((inner_view.row_id, inner_view.unit))),
                  "att_view_content_sha256": att_view.content_hashes(),
                  "inner_view_content_sha256": inner_view.content_hashes(),
                  "probe": probe,
                  "role_fields_accessed": ["ATTACK_FIT_NEW:features", "ATTACK_FIT_NEW:sex",
                                           "INNER_SELECTION:features", "INNER_SELECTION:sex"]}
        path = output_dir / f"REAL_CONTROL__{view}.json"
        _write_json(path, report)
        report_hashes[view] = _sha(path)
    status = "PASS" if all(json.loads((output_dir / f"REAL_CONTROL__{view}.json").read_text())["status"] == "PASS"
                           for view in ("income", "occupation", "pair")) else "FAIL"
    _write_json(output_dir / "REAL_CONTROL_INDEX.json",
                {"schema": "pcrl-team-b-real-control-index-v1",
                 "status": status, "arm": args.arm, "seed": args.seed,
                 "queue_unit_id": args.unit_id, "screen_unit_id": args.screen_unit_id,
                 "attempt_index_sha256": args.attempt_index_sha256,
                 "view_report_sha256": report_hashes, "additional_fresh_reader_fits": 6})
    if status != "PASS":
        raise RuntimeError("B real-role attack controls failed for at least one view")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("arm", "unit_id", "screen_unit_id", "queue_root", "campaign_lock",
                 "lock_sha256", "screen_config", "config_sha256", "data_manifest",
                 "manifest_sha256", "attempt", "attempt_index_sha256", "private_output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    args.attempt = Path(args.attempt).resolve()
    args.private_output = Path(args.private_output).resolve()
    if args.arm not in ARMS or args.seed not in (0, 1, 2):
        raise ValueError("Unknown B arm or seed")
    run(args)


if __name__ == "__main__":
    main()
