"""R03-only independent A complete-view, frozen-reader and AUC replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .independent_a_inputs import read_admitted_a_role
from .independent_a_law import load_independent_policy
from .independent_a_q import load_independent_q_table
from .independent_a_source import bound_json, bound_npz, sha_file
from .independent_a_views import compare_frozen_view_arrays, encode_sampled_views
from .independent_reader_custody import (verify_control_custody,
                                         verify_ignore_aliases,
                                         verify_reader_custody,
                                         verify_tune_selection)
from .independent_replay import reader_recovery, weighted_binary_auc
from .replay_a_prefit import queue_guard


VIEWS = ("r1", "r2", "pair")


def _sex(manifest_path: Path, manifest: dict, role: str,
         row: np.ndarray, unit: np.ndarray) -> np.ndarray:
    key = f"{role}:sex"
    path = manifest_path.parent.parent / manifest["role_file_relative_path"][key]
    record = bound_npz(path, manifest["role_file_sha256"][key],
                       {"row_id", "unit", "sex"})
    if (not np.array_equal(record["row_id"], row)
            or not np.array_equal(record["unit"], unit)
            or record["sex"].dtype != np.int64
            or not np.isin(record["sex"], (0, 1)).all()):
        raise ValueError("A attack SEX role does not align with frozen source")
    return record["sex"]


def _codes(directory: Path, row: np.ndarray, unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(directory / "release_codes.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"row_id", "unit", "code_r1", "code_r2"}:
            raise ValueError("A attack release code member set differs")
        if (not np.array_equal(archive["row_id"], row)
                or not np.array_equal(archive["unit"], unit)):
            raise ValueError("A attack code/person order differs")
        return (np.array(archive["code_r1"], copy=True),
                np.array(archive["code_r2"], copy=True))


def _saved_view(root: Path, role: str, row: np.ndarray, unit: np.ndarray,
                receipt: dict) -> dict:
    name = f"view_{role}.npz"
    if sha_file(root / name) != receipt["files"][name]:
        raise ValueError("A saved frozen view file SHA differs")
    with np.load(root / name, allow_pickle=False) as archive:
        fields = {key: np.array(archive[key], copy=True) for key in archive.files}
    if (not np.array_equal(fields.pop("row_id"), row)
            or not np.array_equal(fields.pop("unit"), unit)):
        raise ValueError("A frozen view row/group identity differs")
    return fields


def _q_reader(root: Path, att_q: np.ndarray, inner_q: np.ndarray,
              reader_dir: Path, tune: dict, reader_receipt: dict,
              att_row: np.ndarray, att_unit: np.ndarray,
              inner_row: np.ndarray, inner_unit: np.ndarray,
              table_sha: str, lock_sha: str) -> dict:
    receipt_path = root / "Q_POST_READER_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    model_path = root / "model__Q_POST_LR_C1.joblib"
    npz_path = root / "public_q_features_and_predictions.npz"
    if (receipt.get("schema") != "SCV1-Q-POST-reader-receipt-v1"
            or receipt.get("name") != "Q_POST_LR_C1"
            or receipt.get("public_q_table_sha256") != table_sha
            or receipt.get("lock_sha256") != lock_sha
            or receipt["model_sha256"] != sha_file(model_path)
            or receipt["private_matrix_sha256"] != sha_file(npz_path)):
        raise ValueError("A Q reader private model/matrix custody differs")
    with np.load(npz_path, allow_pickle=False) as archive:
        if (not np.array_equal(archive["att_row_id"], att_row)
                or not np.array_equal(archive["att_unit"], att_unit)
                or not np.array_equal(archive["inner_row_id"], inner_row)
                or not np.array_equal(archive["inner_unit"], inner_unit)):
            raise ValueError("A Q reader ATT/INNER person order differs")
        if not np.array_equal(archive["att_q"], att_q) or not np.array_equal(
                archive["inner_q"], inner_q):
            raise ValueError("A Q reader used different actual-code features")
        pred_inner = np.array(archive["prediction_inner"], copy=True)
        pred_tune = np.array(archive["prediction_tune"], copy=True)
    model = joblib.load(model_path)
    if np.max(np.abs(model.predict_proba(inner_q)[:, 1] - pred_inner)) > 1e-10:
        raise ValueError("A Q reader model does not forward saved INNER features")
    external = reader_receipt.get("external_readers", [])
    if (len(external) != 1 or external[0].get("name") != "Q_POST_LR_C1"
            or external[0].get("model_sha256") != receipt["model_sha256"]
            or external[0].get("custody_file") != "model__Q_POST_LR_C1.bin"
            or sha_file(reader_dir / "model__Q_POST_LR_C1.bin") !=
               receipt["model_sha256"]):
        raise ValueError("A Q reader differs from frozen external model custody")
    with np.load(reader_dir / "predictions_inner.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["Q_POST_LR_C1"], pred_inner):
            raise ValueError("A Q reader INNER prediction differs from frozen slate")
    with np.load(reader_dir / "predictions_tune.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["Q_POST_LR_C1"], pred_tune):
            raise ValueError("A Q reader TUNE prediction differs from frozen slate")
    return {"model_sha256": receipt["model_sha256"],
            "matrix_sha256": receipt["private_matrix_sha256"],
            "receipt_sha256": sha_file(receipt_path),
            "forward_exact_within_1e_10": True,
            "tune_selection_checked": tune["status"] == "TUNE_SELECTION_PASS"}


def run(args: argparse.Namespace) -> dict:
    lock_path = Path(args.original_lock).resolve(strict=True)
    queue_guard(lock_path, args.original_lock_sha256)
    amendment = bound_json(Path(args.a_amendment), args.a_amendment_sha256)
    if (amendment.get("status") != "LOCALLY_SEALED_A_STAGE"
            or amendment.get("original_lock_sha256") != args.original_lock_sha256):
        raise PermissionError("A attack replay requires effective A seal")
    attack_root = Path(args.attack_root).resolve(strict=True)
    index = bound_json(attack_root / "ATTACK_ATTEMPT_INDEX.json",
                       args.attack_index_sha256)
    if (index.get("status") not in ("COMPLETE", "CONTROL_FAIL")
            or index.get("a_amendment_sha256") != args.a_amendment_sha256
            or index.get("original_lock_sha256") != args.original_lock_sha256):
        raise ValueError("A attack index differs from effective amendment")
    arm, seed, draw = index["arm_id"], index["source_seed"], index["realization"]
    plan_id = f"A-ATTACK-{arm}-S{seed}-R{draw}"
    if amendment["unit_plan"].get(plan_id, {}).get("kind") != "attack":
        raise ValueError("A attack absent from sealed plan")
    policy = load_independent_policy(Path(args.fit_dir), args.policy_receipt_sha256)
    if (policy.receipt["arm_id"] != arm or policy.receipt["source_seed"] != seed
            or index["policy_receipt_sha256"] != args.policy_receipt_sha256):
        raise ValueError("A attack selected policy differs")
    manifest_path = Path(args.role_manifest).resolve(strict=True)
    manifest = bound_json(manifest_path, args.role_manifest_sha256)
    role_source = {}
    for role, short in (("ATTACK_FIT_NEW", "att"),
                        ("INNER_SELECTION", "inner")):
        role_source[short] = read_admitted_a_role(
            seed=seed, role=role, portfolio=Path(args.portfolio_package),
            manifest_path=manifest_path, manifest_sha256=args.role_manifest_sha256,
            source_audit_path=Path(args.source_audit),
            source_audit_sha256=args.source_audit_sha256)
    att_dir, inner_dir = Path(args.att_release_dir), Path(args.inner_release_dir)
    att_replay = bound_json(Path(args.att_release_replay), args.att_release_replay_sha256)
    inner_replay = bound_json(Path(args.inner_release_replay), args.inner_release_replay_sha256)
    for receipt, role in ((att_replay, "ATTACK_FIT_NEW"),
                          (inner_replay, "INNER_SELECTION")):
        if (receipt.get("status") != "SAMPLED_RELEASE_WIRE_REPLAY_PASS"
                or receipt.get("arm_id") != arm or receipt.get("source_seed") != seed
                or receipt.get("realization") != draw or receipt.get("role") != role):
            raise ValueError("A source release lacks independent wire replay")
    att_codes = _codes(att_dir, role_source["att"].row_id, role_source["att"].unit)
    inner_codes = _codes(inner_dir, role_source["inner"].row_id, role_source["inner"].unit)
    q_table = None
    q_receipt_sha = None
    if args.q_table_dir:
        q_table, q_identity = load_independent_q_table(
            Path(args.q_table_dir), args.q_table_receipt_sha256,
            args.policy_receipt_sha256)
        q_receipt_sha = args.q_table_receipt_sha256
        if (index["q_table_sha256"] !=
                json.loads((Path(args.q_table_dir) / "Q_TABLE_RECEIPT.json").read_text())
                ["table_content_sha256"]
                or q_identity["mechanism_sha256"] != policy.receipt["mechanism_sha256"]
                or q_identity["source_binding_sha256"] !=
                   policy.receipt["source_binding_sha256"]
                or q_identity["release_id"] != policy.receipt["release_id"]
                or q_identity["bank1_sha256"] !=
                   policy.receipt["bank_sha256_by_recipient"]["r1"]
                or q_identity["bank2_sha256"] !=
                   policy.receipt["bank_sha256_by_recipient"]["r2"]):
            raise ValueError("A attack public Q identity differs")
    elif index.get("q_table_sha256") is not None:
        raise ValueError("A attack Q table missing from independent replay")
    if ("Q_POST_LR_C1" in amendment["attack_plan"]["reader_ids"]) != (q_table is not None):
        raise ValueError("A attack Q reader differs from predeclared slate")
    views = encode_sampled_views(
        policy, att_code1=att_codes[0], att_code2=att_codes[1],
        inner_code1=inner_codes[0], inner_code2=inner_codes[1], q_table=q_table)
    common = attack_root / "common"
    freeze_path = common / "FROZEN_COMPLETE.json"
    freeze = bound_json(freeze_path, index["reader_freeze_sha256"])
    if (freeze.get("status") != "FROZEN_COMPLETE"
            or freeze.get("family") != "A" or freeze.get("seed") != seed
            or freeze.get("effective_lock_sha256") != args.original_lock_sha256):
        raise ValueError("A pre-INNER reader freeze incomplete")
    view_report, reader_receipts = {}, {}
    core_order = tuple(amendment["attack_plan"]["reader_ids"])
    if core_order != ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
                      "CELL_A1", "CODE_BIT_LR_C1", "PROB_LR_C1",
                      "PROB_HGB_15x200", "Q_POST_LR_C1"):
        raise ValueError("A nine fixed sampled readers differ from prefit lock")
    for view in VIEWS:
        reader_dir = common / view / "readers"
        reader_receipt_path = reader_dir / "READERS_RECEIPT.json"
        expected = freeze["reader_receipt_sha256"][view]
        reader_receipts[view] = bound_json(reader_receipt_path, expected)
        expected_order = (core_order if view != "pair" else
                          (*core_order, *(f"IGNORE_R2__{name}" for name in core_order),
                           *(f"IGNORE_R1__{name}" for name in core_order)))
        if tuple(reader_receipts[view]["reader_order"]) != expected_order:
            raise ValueError("A fixed reader/coalition alias order differs")
        saved = {role: _saved_view(reader_dir, role,
                                  role_source[role].row_id,
                                  role_source[role].unit,
                                  reader_receipts[view])
                 for role in ("att", "inner")}
        replay = compare_frozen_view_arrays(views[view], saved,
                                            reader_receipts[view])
        if (reader_receipts[view]["att_view_content_sha256"] !=
                index["view_content_sha256"][view]["att"]
                or reader_receipts[view]["inner_view_content_sha256"] !=
                   index["view_content_sha256"][view]["inner"]
                or reader_receipts[view]["att_view_content_sha256"] !=
                   freeze["att_view_sha256"][view]
                or reader_receipts[view]["inner_view_content_sha256"] !=
                   freeze["inner_view_sha256"][view]):
            raise ValueError("A attack indexed view differs from frozen reader view")
        view_report[view] = replay
    # All frozen view/model receipts are verified before INNER SEX is opened.
    y_att = _sex(manifest_path, manifest, "ATTACK_FIT_NEW",
                 role_source["att"].row_id, role_source["att"].unit)
    y_inner = _sex(manifest_path, manifest, "INNER_SELECTION",
                   role_source["inner"].row_id, role_source["inner"].unit)
    reader_report, controls, auc = {}, {}, {}
    for view in VIEWS:
        reader_dir = common / view / "readers"
        reader_report[view] = verify_reader_custody(
            reader_dir, expected_row_id=role_source["inner"].row_id,
            expected_unit=role_source["inner"].unit)
        tune = verify_tune_selection(reader_dir, role_source["att"].row_id,
                                     role_source["att"].unit, y_att)
        reader_report[view]["tune"] = tune
        with np.load(reader_dir / "predictions_inner.npz", allow_pickle=False) as archive:
            prediction = {name: np.array(archive[name], copy=True)
                          for name in archive.files}
        auc[view] = reader_recovery(prediction, y_inner)
        scored = index["scores"][view]
        if (abs(auc[view]["auc"] - scored["primary_envelope_auc"]) > 1e-10
                or tune["winner"] != scored["validation_winner"]
                or tune["orientation"] != scored["validation_winner_orientation"]
                or set(auc[view]["reader_auc"]) != set(scored["readers"])):
            raise ValueError("A fixed reader AUC envelope or ATT selection differs")
        for name, value in auc[view]["reader_auc"].items():
            if abs(value - scored["readers"][name]["fixed_alias_max_auc"]) > 1e-10:
                raise ValueError("A individual frozen reader AUC differs")
        selected = weighted_binary_auc(prediction[tune["winner"]], y_inner)
        if tune["orientation"] == "reverse":
            selected = 1 - selected
        if abs(selected - scored["validation_winner_inner_auc"]) > 1e-10:
            raise ValueError("A ATT-selected reader INNER AUC differs")
        controls[view] = verify_control_custody(
            common / view / "controls", role_source["inner"].row_id,
            role_source["inner"].unit)
        if q_table is not None:
            q_dir = attack_root / "q_readers" / view
            reader_report[view]["q_reader"] = _q_reader(
                q_dir, views[view]["att"]["x_defense_aware"][:,
                      views[view]["att"]["x"].shape[1]:],
                views[view]["inner"]["x_defense_aware"][:,
                      views[view]["inner"]["x"].shape[1]:],
                reader_dir, tune, reader_receipts[view],
                role_source["att"].row_id, role_source["att"].unit,
                role_source["inner"].row_id, role_source["inner"].unit,
                index["q_table_sha256"], args.original_lock_sha256)
    aliases = verify_ignore_aliases(common / "pair" / "readers",
                                    common / "r1" / "readers",
                                    common / "r2" / "readers")
    return {"schema": "pcrl-r04-a-attack-independent-replay-v1",
            "status": "COMPLETE_A_ATTACK_REPLAY_PASS",
            "arm_id": arm, "source_seed": seed, "realization": draw,
            "original_lock_sha256": args.original_lock_sha256,
            "a_amendment_sha256": args.a_amendment_sha256,
            "attack_index_sha256": args.attack_index_sha256,
            "policy_receipt_sha256": args.policy_receipt_sha256,
            "att_release_replay_sha256": args.att_release_replay_sha256,
            "inner_release_replay_sha256": args.inner_release_replay_sha256,
            "q_table_receipt_sha256": q_receipt_sha,
            "freeze_sha256": index["reader_freeze_sha256"],
            "complete_views": view_report, "readers": reader_report,
            "zero_fit_pair_aliases": aliases, "engineering_controls": controls,
            "sex_recovery": auc}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-lock", "original-lock-sha256", "a-amendment",
                 "a-amendment-sha256", "attack-root", "attack-index-sha256",
                 "fit-dir", "policy-receipt-sha256", "att-release-dir",
                 "inner-release-dir", "att-release-replay",
                 "att-release-replay-sha256", "inner-release-replay",
                 "inner-release-replay-sha256", "q-table-dir",
                 "q-table-receipt-sha256", "portfolio-package",
                 "role-manifest", "role-manifest-sha256", "source-audit",
                 "source-audit-sha256", "public-receipt"):
        parser.add_argument("--" + name, required=name not in
                            ("q-table-dir", "q-table-receipt-sha256"))
    args = parser.parse_args()
    output = Path(args.public_receipt)
    if output.exists():
        raise FileExistsError("A independent attack replay receipt exists")
    result = run(args)
    output.write_text(json.dumps(result, sort_keys=True, indent=2,
                                 allow_nan=False) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
