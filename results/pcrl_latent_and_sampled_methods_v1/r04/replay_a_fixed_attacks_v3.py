"""R03-only independent complete-view and frozen-reader replay of fixed A attacks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from .audit_diagnostic_q_parity import _canonical
from .independent_a_fixed_views import RELEASE_FIELDS, build_fixed_views
from .independent_a_q import load_independent_q_table
from .independent_a_source import bound_json, bound_npz, sha_array, sha_file
from .independent_a_views import compare_frozen_view_arrays
from .independent_reader_custody import (verify_control_custody,
                                         verify_ignore_aliases,
                                         verify_reader_custody,
                                         verify_tune_selection)
from .independent_replay import reader_recovery, weighted_binary_auc
from .replay_a_attack import _saved_view, _sex
from .replay_a_prefit import queue_guard


ARMS = ("T_STAR", "P_STAR", "U0", "Ucal", "D0_LOCAL", "D0_SEQ12",
        "D0_SEQ21", "A_IDENTITY", "A_CLASS_ONLY")
VIEWS = ("r1", "r2", "pair")
FINITE = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
          "CELL_A1", "CODE_BIT_LR_C1", "PROB_LR_C1", "PROB_HGB_15x200",
          "Q_POST_LR_C1")
CONTINUOUS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
              "PROB_LR_C1", "PROB_HGB_15x200")


def _fixed_external_q(external: list[dict], expected_model_sha: str) -> dict:
    """Require the fixed finite slate's two probability readers and one Q reader."""
    names = ("PROB_LR_C1", "PROB_HGB_15x200", "Q_POST_LR_C1")
    if (len(external) != 3 or tuple(row.get("name") for row in external) != names or
            any(row.get("custody_file") != f"model__{row.get('name')}.bin" or
                row.get("new_fit") is not True or
                row.get("fit_role") != "ATTACK_FIT_NEW/ATTACK_TRAIN"
                for row in external) or
            external[-1].get("model_sha256") != expected_model_sha):
        raise ValueError("fixed finite external reader inventory differs")
    return external[-1]


def _fixed_q_reader(root: Path, att_q: np.ndarray, inner_q: np.ndarray,
                    reader_dir: Path, tune: dict, reader_receipt: dict,
                    att_row: np.ndarray, att_unit: np.ndarray,
                    inner_row: np.ndarray, inner_unit: np.ndarray,
                    table_sha: str, lock_sha: str) -> dict:
    """Reforward Q and bind its model within the exact three-reader fixed slate."""
    receipt_path = root / "Q_POST_READER_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    model_path = root / "model__Q_POST_LR_C1.joblib"
    npz_path = root / "public_q_features_and_predictions.npz"
    if (receipt.get("schema") != "SCV1-Q-POST-reader-receipt-v1" or
            receipt.get("name") != "Q_POST_LR_C1" or
            receipt.get("public_q_table_sha256") != table_sha or
            receipt.get("lock_sha256") != lock_sha or
            receipt.get("model_sha256") != sha_file(model_path) or
            receipt.get("private_matrix_sha256") != sha_file(npz_path)):
        raise ValueError("fixed Q private model/matrix custody differs")
    with np.load(npz_path, allow_pickle=False) as archive:
        if (not np.array_equal(archive["att_row_id"], att_row) or
                not np.array_equal(archive["att_unit"], att_unit) or
                not np.array_equal(archive["inner_row_id"], inner_row) or
                not np.array_equal(archive["inner_unit"], inner_unit) or
                not np.array_equal(archive["att_q"], att_q) or
                not np.array_equal(archive["inner_q"], inner_q)):
            raise ValueError("fixed Q reader actual-code features/person order differ")
        pred_inner = np.array(archive["prediction_inner"], copy=True)
        pred_tune = np.array(archive["prediction_tune"], copy=True)
    model = joblib.load(model_path)
    if np.max(np.abs(model.predict_proba(inner_q)[:, 1] - pred_inner)) > 1e-10:
        raise ValueError("fixed Q model does not forward INNER features")
    external = _fixed_external_q(reader_receipt.get("external_readers", []),
                                 receipt["model_sha256"])
    if sha_file(reader_dir / external["custody_file"]) != receipt["model_sha256"]:
        raise ValueError("fixed Q copied frozen model differs")
    with np.load(reader_dir / "predictions_inner.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["Q_POST_LR_C1"], pred_inner):
            raise ValueError("fixed Q INNER prediction differs from frozen slate")
    with np.load(reader_dir / "predictions_tune.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["Q_POST_LR_C1"], pred_tune):
            raise ValueError("fixed Q TUNE prediction differs from frozen slate")
    return {"receipt_sha256": sha_file(receipt_path),
            "model_sha256": receipt["model_sha256"],
            "matrix_sha256": receipt["private_matrix_sha256"],
            "forward_exact_within_1e_10": True,
            "tune_selection_checked": tune["status"] == "TUNE_SELECTION_PASS"}


def _effective_bank(arm: str, historical: np.ndarray) -> np.ndarray:
    """Diagnostic code IDs name globally sorted public actions, not old T* slots."""
    if arm not in ("A_IDENTITY", "A_CLASS_ONLY"):
        return historical
    bank, _, _ = _canonical(
        historical, np.arange(len(historical), dtype=np.int64),
        np.argmax(historical, axis=1).astype(np.int64))
    return bank


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _wire_sha(release: dict, receipt: dict, recipient: int,
              bank_size: int | None) -> str:
    digest = hashlib.sha256()
    label = f"r{recipient}"
    meta = {"model_id": receipt["release_id"],
            "source_binding_sha256": receipt["source_binding_sha256"],
            "bank_sha256": receipt["bank_sha256_by_recipient"][label],
            "code_length_bits": None if bank_size is None else (bank_size - 1).bit_length()}
    for q, d, c in zip(release[f"prob_{label}"], release[f"decision_{label}"],
                       release[f"code_{label}"], strict=True):
        wire = {"schema": "SCV1-reference-score-release-v1", "recipient": recipient,
                "probability": q.tolist(), "decision": int(d),
                "release_id": receipt["release_id"], "public_metadata": meta}
        if bank_size is not None:
            wire["code_id"] = int(c)
        digest.update(json.dumps(wire, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode() + b"\n")
    return digest.hexdigest()


def _release(campaign: Path, arm: str, seed: int, short: str,
             expected_sha: str) -> tuple[dict, dict, Path]:
    prefix = "A-DIAG-RELEASE" if arm.startswith("A_") else "A-REFERENCE-RELEASE"
    unit_id = f"{prefix}-{arm}-S{seed}-{short}"
    root = campaign / "r03/.private/a_fixed_units" / unit_id
    receipt = bound_json(root / "REFERENCE_RELEASE_RECEIPT.json", expected_sha)
    require(receipt.get("schema") == "SCV1-private-reference-release-v1" and
            receipt.get("status") == "FROZEN_DETERMINISTIC" and
            receipt.get("arm_id") == arm and receipt.get("source_seed") == seed and
            receipt.get("role") == ("ATTACK_FIT_NEW" if short == "ATT" else "INNER_SELECTION"),
            "fixed release identity differs")
    package = bound_npz(root / "reference_release.npz", receipt["codes_file_sha256"],
                        RELEASE_FIELDS)
    require(len(package["row_id"]) == receipt["rows"] and
            len(np.unique(package["unit"])) == receipt["groups"] and
            sha_array(np.column_stack((package["row_id"], package["unit"]))) ==
            receipt["row_unit_pair_sha256"], "fixed release row/group differs")
    return package, receipt, root


def _one(campaign: Path, cell: dict, producer: dict, manifest_path: Path,
         manifest: dict, seed: int, original_sha: str, producer_sha: str) -> dict:
    arm = cell["scientific_cell_id"].split("-S")[0].split("ATTACK-")[-1]
    uid = cell["physical_unit_id"]
    root = campaign / "r03/.private/a_fixed_units" / uid
    index = bound_json(root / "ATTACK_ATTEMPT_INDEX.json", cell["attack_index_sha256"])
    identity = index["reference_identity"]
    require(index.get("schema") == "SCV1-reference-attack-attempt-v1" and
            index.get("status") == "COMPLETE" and index.get("unit_id") == uid and
            index.get("a_amendment_sha256") == producer_sha and
            index.get("original_lock_sha256") == original_sha and
            identity.get("reference_id") == arm and identity.get("source_seed") == seed and
            index.get("control_status") == {v: "PASS" for v in VIEWS} and
            index.get("reader_freeze_sha256") == cell["reader_freeze_sha256"],
            "fixed attack completion/index differs")
    finite = identity["interface_kind"] == "finite"
    require(identity["interface_kind"] in ("finite", "continuous"),
            "fixed attack interface differs")
    att, ar, _ = _release(campaign, arm, seed, "ATT", index["att_release_receipt_sha256"])
    inner, ir, _ = _release(campaign, arm, seed, "INNER",
                          index["inner_release_receipt_sha256"])
    require(ar["release_id"] == ir["release_id"] == identity["release_id"] and
            ar["source_binding_sha256"] == ir["source_binding_sha256"] ==
            identity["source_binding_sha256"] and
            ar["bank_sha256_by_recipient"] == ir["bank_sha256_by_recipient"] and
            ar["interface_kind"] == ir["interface_kind"] == identity["interface_kind"] and
            ar["row_unit_pair_sha256"] ==
            manifest["roles"]["ATTACK_FIT_NEW"]["row_unit_pair_sha256"] and
            ir["row_unit_pair_sha256"] ==
            manifest["roles"]["INNER_SELECTION"]["row_unit_pair_sha256"],
            "fixed ATT/INNER source identity differs")
    banks, q_table = None, None
    if finite:
        map_arm = "T_STAR" if arm.startswith("A_") else arm
        map_root = campaign / "r01" / f"A_FIXED_MAP_{map_arm}_S{seed}"
        map_identity = identity["fixed_map_identity"]
        fixed_map = bound_json(map_root.with_suffix(".json"), map_identity["receipt_sha256"])
        arrays = bound_npz(map_root.with_suffix(".npz"), map_identity["npz_sha256"],
                           {"bank_r1", "bank_r2", "fine_to_code_r1", "fine_to_code_r2"})
        banks = tuple(_effective_bank(arm, arrays[f"bank_r{r}"]) for r in (1, 2))
        require(fixed_map["npz_sha256"] == map_identity["npz_sha256"] and
                fixed_map["source_binding_sha256"] == ar["source_binding_sha256"],
                "fixed map/source identity differs")
        q_root = campaign / "r03/.private/a_q_units" / f"A-Q-PREPROCESS-{arm}-S{seed}"
        q_identity = identity["q_table_identity"]
        q_table, _ = load_independent_q_table(q_root, q_identity["receipt_sha256"],
                                               fixed_map["original_bank_receipt_sha256"])
        require(index["q_table_content_sha256"] == q_identity["content_sha256"] and
                json.loads((q_root / "Q_TABLE_RECEIPT.json").read_text())
                ["table_content_sha256"] == q_identity["content_sha256"],
                "fixed Q identity differs")
    else:
        require(identity["fixed_map_identity"] is None and
                identity["q_table_identity"] is None and
                index["q_table_content_sha256"] is None,
                "continuous fixed reference has a code or Q table")
    for release, receipt in ((att, ar), (inner, ir)):
        for r in (1, 2):
            q, d, code = (release[f"prob_r{r}"], release[f"decision_r{r}"],
                          release[f"code_r{r}"])
            require(q.dtype == np.float64 and d.dtype == code.dtype == np.int64 and
                    np.array_equal(d, np.argmax(q, axis=1)),
                    "fixed release task decision differs")
            if finite:
                bank = banks[r - 1]
                require(np.all((code >= 0) & (code < len(bank))) and
                        np.array_equal(q, bank[code]) and
                        receipt["bank_sha256_by_recipient"][f"r{r}"] == sha_array(bank),
                        "fixed release public code/bank differs")
            else:
                require(np.all(code == -1), "continuous release transmitted code")
            require(_wire_sha(release, receipt, r,
                              len(banks[r - 1]) if finite else None) ==
                    receipt["wire_sha256_by_recipient"][f"r{r}"],
                    "fixed release public wire SHA differs")
    views = build_fixed_views(att, inner, ar,
                              bank_sizes=None if banks is None else (len(banks[0]), len(banks[1])),
                              q_table=q_table)
    common = root / "common"
    freeze = bound_json(common / "FROZEN_COMPLETE.json", index["reader_freeze_sha256"])
    require(freeze.get("status") == "FROZEN_COMPLETE" and freeze.get("family") == "A" and
            freeze.get("seed") == seed and freeze.get("effective_lock_sha256") == original_sha,
            "fixed reader freeze differs")
    reader_receipts, view_report = {}, {}
    order = FINITE if finite else CONTINUOUS
    for view in VIEWS:
        reader_dir = common / view / "readers"
        rr = bound_json(reader_dir / "READERS_RECEIPT.json",
                        freeze["reader_receipt_sha256"][view])
        expected = order if view != "pair" else (*order,
                    *(f"IGNORE_R2__{name}" for name in order),
                    *(f"IGNORE_R1__{name}" for name in order))
        require(tuple(rr["reader_order"]) == expected and
                rr["att_view_content_sha256"] ==
                index["view_content_sha256"][view]["att"] and
                rr["inner_view_content_sha256"] ==
                index["view_content_sha256"][view]["inner"],
                "fixed reader order/view index differs")
        saved = {role: _saved_view(reader_dir, role,
                                  att["row_id"] if role == "att" else inner["row_id"],
                                  att["unit"] if role == "att" else inner["unit"], rr)
                 for role in ("att", "inner")}
        view_report[view] = compare_frozen_view_arrays(views[view], saved, rr)
        reader_receipts[view] = rr
    # Every released wire, complete view and model receipt is bound before SEX.
    y_att = _sex(manifest_path, manifest, "ATTACK_FIT_NEW", att["row_id"], att["unit"])
    y_inner = _sex(manifest_path, manifest, "INNER_SELECTION", inner["row_id"], inner["unit"])
    reports, controls, aucs = {}, {}, {}
    for view in VIEWS:
        reader_dir = common / view / "readers"
        reports[view] = verify_reader_custody(
            reader_dir, expected_row_id=inner["row_id"], expected_unit=inner["unit"],
            reforward_names=("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
                             "CELL_A1", "CODE_BIT_LR_C1"))
        tune = verify_tune_selection(reader_dir, att["row_id"], att["unit"], y_att)
        reports[view]["tune"] = tune
        with np.load(reader_dir / "predictions_inner.npz", allow_pickle=False) as archive:
            pred = {name: np.array(archive[name], copy=True) for name in archive.files}
        aucs[view] = reader_recovery(pred, y_inner)
        score = index["scores"][view]
        require(abs(aucs[view]["auc"] - score["primary_envelope_auc"]) < 1e-10 and
                tune["winner"] == score["validation_winner"] and
                tune["orientation"] == score["validation_winner_orientation"] and
                set(aucs[view]["reader_auc"]) == set(score["readers"]),
                "fixed AUC envelope or ATT selection differs")
        for name, value in aucs[view]["reader_auc"].items():
            require(abs(value - score["readers"][name]["fixed_alias_max_auc"]) < 1e-10,
                    "fixed individual reader AUC differs")
        selected = weighted_binary_auc(pred[tune["winner"]], y_inner)
        if tune["orientation"] == "reverse":
            selected = 1 - selected
        require(abs(selected - score["validation_winner_inner_auc"]) < 1e-10,
                "fixed ATT-selected INNER AUC differs")
        controls[view] = verify_control_custody(common / view / "controls",
                                                inner["row_id"], inner["unit"])
        if finite:
            base = views[view]
            reports[view]["q_reader"] = _fixed_q_reader(
                root / "q_readers" / view,
                base["att"]["x_defense_aware"][:, base["att"]["x"].shape[1]:],
                base["inner"]["x_defense_aware"][:, base["inner"]["x"].shape[1]:],
                reader_dir, tune, reader_receipts[view], att["row_id"], att["unit"],
                inner["row_id"], inner["unit"], index["q_table_content_sha256"],
                original_sha)
    aliases = verify_ignore_aliases(common / "pair" / "readers",
                                    common / "r1" / "readers", common / "r2" / "readers")
    return {"arm": arm, "seed": seed, "physical_unit_id": uid,
            "attack_index_sha256": cell["attack_index_sha256"],
            "freeze_sha256": index["reader_freeze_sha256"],
            "release_wire_sha256": {"ATT": ar["wire_sha256_by_recipient"],
                                    "INNER": ir["wire_sha256_by_recipient"]},
            "complete_views": view_report, "readers": reports,
            "aliases": aliases, "controls": controls, "sex_recovery": aucs}


def run(args: argparse.Namespace) -> dict:
    campaign = Path(args.campaign).resolve(strict=True)
    lock = campaign / "CAMPAIGN_LOCK.json"
    queue_guard(lock, args.original_lock_sha256)
    producer = bound_json(Path(args.producer), args.producer_sha256)
    completion = bound_json(Path(args.completion), args.completion_sha256)
    amendment = bound_json(Path(args.verification_amendment), args.verification_amendment_sha256)
    require(producer.get("status") == "LOCALLY_SEALED_A_STAGE" and
            producer.get("original_lock_sha256") == args.original_lock_sha256 and
            completion.get("status") ==
            "ALL_18_SCIENTIFIC_CELLS_HAVE_COMPLETE_PHYSICAL_ATTACK" and
            completion.get("scientific_cell_count") == 18 and
            completion.get("producer_amendment_sha256") == args.producer_sha256 and
            set(amendment.get("authorized_unit_ids", [])) == {
                "R04-A-FIXED-ATTACK-S0-003", "R04-A-FIXED-ATTACK-S1-003"} and
            args.unit_id in amendment["authorized_unit_ids"],
            "fixed verification provenance differs")
    manifest_path = Path(args.role_manifest).resolve(strict=True)
    manifest = bound_json(manifest_path, args.role_manifest_sha256)
    cells = [cell for cell in completion["cells"] if cell["scientific_cell_id"].endswith(
        f"-S{args.seed}")]
    require(len(cells) == 9 and len({cell["scientific_cell_id"] for cell in cells}) == 9,
            "fixed seed scientific cell set differs")
    rows = [_one(campaign, cell, producer, manifest_path, manifest, args.seed,
                 args.original_lock_sha256, args.producer_sha256) for cell in cells]
    require({row["arm"] for row in rows} == set(ARMS), "fixed arm coverage differs")
    return {"schema": "r04-a-fixed-attack-independent-replay-v1",
            "status": "NINE_FIXED_ATTACKS_INDEPENDENT_REPLAY_PASS", "seed": args.seed,
            "producer_sha256": args.producer_sha256,
            "completion_sha256": args.completion_sha256,
            "verification_amendment_sha256": args.verification_amendment_sha256,
            "cells": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "original-lock-sha256", "producer", "producer-sha256",
                 "completion", "completion-sha256", "verification-amendment",
                 "verification-amendment-sha256", "role-manifest",
                 "role-manifest-sha256", "unit-id", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1))
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("fixed attack replay output exists")
    result = run(args)
    output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
