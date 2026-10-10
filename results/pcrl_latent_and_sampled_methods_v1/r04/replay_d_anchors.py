"""Queue-only independent complete-view replay of four D score anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .independent_anchor_view import build_anchor_view
from .independent_d_view import compare_saved_view
from .independent_reader_custody import (verify_control_custody,
                                         verify_ignore_aliases,
                                         verify_reader_custody,
                                         verify_tune_selection)
from .independent_replay import reader_recovery, weighted_binary_auc
from .replay_first_pair import (_npz, _queue_guard, _read_json,
                                _role_npz, sha256)


ANCHORS = ("U0", "Ucal")
VIEWS = ("r1", "r2", "pair")
ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")
REF_KEYS = {"row_id", "unit", "U0_1", "U0_2", "Ucal_1", "Ucal_2", "d_1", "d_2"}


def run(args: argparse.Namespace) -> dict:
    seal_path = Path(args.seal).resolve(strict=True)
    _queue_guard(seal_path, args.seal_sha256)
    seal = _read_json(seal_path, args.seal_sha256)
    if seal.get("status") != "LOCALLY_SEALED_D_STAGE":
        raise PermissionError("effective D seal differs")
    amendment = _read_json(Path(args.anchor_amendment), args.anchor_amendment_sha256)
    if (amendment.get("status") != "LOCALLY_SEALED_D_FOLLOWUP"
            or amendment.get("effective_seal_sha256") != args.seal_sha256
            or amendment.get("r01_source_map_sha256") != args.source_map_sha256
            or amendment.get("role_manifest_sha256") != args.role_manifest_sha256
            or set(amendment.get("authorized_unit_ids", [])) !=
               {f"D_U_ANCHOR_{anchor.upper()}_S{seed}"
                for anchor in ANCHORS for seed in (0, 1)}):
        raise PermissionError("D anchor amendment differs")
    role_path = Path(args.role_manifest).resolve(strict=True)
    role = _read_json(role_path, args.role_manifest_sha256)
    if args.role_manifest_sha256 != seal["source_files_sha256"]["source_role_manifest"]:
        raise ValueError("anchor source manifest differs from seal")
    source_map = _read_json(Path(args.source_map), args.source_map_sha256)
    if (source_map.get("status") != "READ_ONLY_SOURCE_VERIFIED_NO_READER_FITS"
            or Path(source_map["repair_manifest"]).resolve() != role_path):
        raise ValueError("R01 anchor source map differs")
    anchor_map = _read_json(Path(args.anchor_map), args.anchor_map_sha256)
    if set(anchor_map) != {f"{anchor}_s{seed}" for anchor in ANCHORS for seed in (0, 1)}:
        raise ValueError("all four D anchor attacks required")
    public = {"schema": "pcrl-r04-d-anchor-replay-v1",
              "status": "FOUR_SCORE_ANCHOR_REPLAY_PASS",
              "effective_lock_sha256": args.seal_sha256,
              "anchor_amendment_sha256": args.anchor_amendment_sha256,
              "role_manifest_sha256": args.role_manifest_sha256,
              "source_map_sha256": args.source_map_sha256,
              "anchors": {}}
    for seed in (0, 1):
        refs = {role_name: _role_npz(role, role_path, role_name,
                                     f"references_s{seed}") for role_name in ROLES}
        sex = {role_name: _role_npz(role, role_path, role_name, "sex")
               for role_name in ROLES}
        for role_name in ROLES:
            source = source_map["rows"][role_name]
            if (set(refs[role_name]) != REF_KEYS
                    or len(refs[role_name]["row_id"]) != source["count"]
                    or role["role_file_sha256"][f"{role_name}:references_s{seed}"]
                       != source[f"references_s{seed}_sha256"]
                    or role["roles"][role_name]["row_unit_pair_sha256"]
                       != source["row_unit_pair_sha256"]
                    or not np.array_equal(refs[role_name]["row_id"], sex[role_name]["row_id"])
                    or not np.array_equal(refs[role_name]["unit"], sex[role_name]["unit"])):
                raise ValueError("R01 anchor reference/SEX role identity differs")
        for anchor in ANCHORS:
            key = f"{anchor}_s{seed}"
            item = anchor_map[key]
            if set(item) != {"receipt", "receipt_sha256", "private_root"}:
                raise ValueError("anchor map requires receipt/hash/private root")
            receipt = _read_json(Path(item["receipt"]), item["receipt_sha256"])
            private = Path(item["private_root"]).resolve(strict=True)
            if ".private" not in private.parts:
                raise PermissionError("anchor attack custody is not private")
            if (receipt.get("status") != "COMPLETE_PRIVATE_DEVELOPMENT_ONLY"
                    or receipt.get("anchor") != anchor
                    or receipt.get("source_seed") != seed
                    or receipt.get("effective_lock_sha256") != args.seal_sha256
                    or receipt.get("d_anchor_amendment_sha256") !=
                       args.anchor_amendment_sha256
                    or receipt.get("role_manifest_sha256") != args.role_manifest_sha256
                    or receipt.get("r01_source_map_sha256") != args.source_map_sha256):
                raise ValueError("anchor attack receipt/source differs")
            source_binding = _read_json(private / "SOURCE_BINDING.json",
                                        receipt["private_source_binding_sha256"])
            if (source_binding.get("anchor") != anchor
                    or source_binding.get("source_seed") != seed
                    or source_binding.get("d_anchor_amendment_sha256") !=
                       args.anchor_amendment_sha256
                    or source_binding.get("r01_source_map_sha256") != args.source_map_sha256
                    or source_binding.get("view_schema_sha256") !=
                       receipt["attack"]["view_schema_sha256"]
                    or receipt["attack"]["plan"]["fit_forecast"]["mandatory_fits"] != 24
                    or sha256(private / "FROZEN_COMPLETE.json")
                       != receipt["attack"]["freeze_receipt_sha256"]
                    or source_binding["freeze_receipt_sha256"]
                       != receipt["attack"]["freeze_receipt_sha256"]):
                raise ValueError("anchor source/freeze custody differs")
            for role_name in ROLES:
                reference_sha = source_map["rows"][role_name][f"references_s{seed}_sha256"]
                if (receipt["source_archive_sha256"][role_name] != reference_sha
                        or source_binding["roles"][role_name]["reference_archive_sha256"]
                           != reference_sha
                        or receipt["ordered_row_unit_sha256"][role_name]
                           != source_map["rows"][role_name]["row_unit_pair_sha256"]):
                    raise ValueError("anchor role archive or row/group hash differs")
            result = {"attack_receipt_sha256": item["receipt_sha256"],
                      "source_binding_sha256": receipt["private_source_binding_sha256"],
                      "views": {}, "readers": {}, "controls": {}, "sex_recovery": {}}
            att, inner = refs["ATTACK_FIT_NEW"], refs["INNER_SELECTION"]
            for view in VIEWS:
                att_built, inner_built = build_anchor_view(att, inner, anchor, view)
                reader_dir = private / view / "readers"
                result["views"][view] = {
                    "att": compare_saved_view(str(reader_dir / "view_att.npz"),
                                               att_built, att["row_id"], att["unit"]),
                    "inner": compare_saved_view(str(reader_dir / "view_inner.npz"),
                                                 inner_built, inner["row_id"], inner["unit"])}
                reader = verify_reader_custody(
                    reader_dir, expected_row_id=inner["row_id"],
                    expected_unit=inner["unit"])
                reader["tune"] = verify_tune_selection(
                    reader_dir, att["row_id"], att["unit"],
                    sex["ATTACK_FIT_NEW"]["sex"])
                if (reader["receipt_sha256"] !=
                        receipt["attack"]["reader_custody"][view]["private_receipt_sha256"]
                        or reader["receipt_sha256"] !=
                        source_binding["reader_receipt_sha256"][view]):
                    raise ValueError("anchor frozen reader receipt differs")
                result["readers"][view] = reader
                with np.load(reader_dir / "predictions_inner.npz", allow_pickle=False) as archive:
                    prediction = {name: np.array(archive[name], copy=True)
                                  for name in archive.files}
                recovery = reader_recovery(prediction, sex["INNER_SELECTION"]["sex"])
                published = receipt["attack"]["scores"][view]
                if (abs(recovery["auc"] - published["primary_envelope_auc"]) > 1e-10
                        or set(recovery["reader_auc"]) != set(published["readers"])
                        or reader["tune"]["winner"] != published["validation_winner"]
                        or reader["tune"]["orientation"] !=
                           published["validation_winner_orientation"]):
                    raise ValueError("anchor public recovery or selection differs")
                for name, value in recovery["reader_auc"].items():
                    if abs(value - published["readers"][name]["fixed_alias_max_auc"]) > 1e-10:
                        raise ValueError("anchor fixed reader AUC differs")
                selected = weighted_binary_auc(
                    prediction[reader["tune"]["winner"]], sex["INNER_SELECTION"]["sex"])
                if reader["tune"]["orientation"] == "reverse":
                    selected = 1 - selected
                if abs(selected - published["validation_winner_inner_auc"]) > 1e-10:
                    raise ValueError("anchor ATT-selected INNER AUC differs")
                result["sex_recovery"][view] = recovery
                control = verify_control_custody(
                    private / view / "controls", inner["row_id"], inner["unit"])
                if (control["receipt_sha256"] != receipt["attack"]["controls"][view]
                        ["custody"]["private_receipt_sha256"]):
                    raise ValueError("anchor V2 custody differs")
                for name in ("null_auc", "shuffled_att_independent_inner_auc", "planted_auc"):
                    replay_name = ("shuffled_independent_inner_auc" if name ==
                                   "shuffled_att_independent_inner_auc" else name)
                    if abs(control[replay_name] - receipt["attack"]["controls"][view]
                           ["scores"][name]) > 1e-10:
                        raise ValueError("anchor V2 control AUC differs")
                result["controls"][view] = control
            result["readers"]["aliases"] = verify_ignore_aliases(
                private / "pair" / "readers", private / "r1" / "readers",
                private / "r2" / "readers")
            public["anchors"][key] = result
    output = Path(args.public_receipt)
    if output.exists():
        raise FileExistsError("anchor replay receipt already exists")
    output.write_text(json.dumps(public, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return {"status": public["status"], "public_receipt_sha256": sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "seal-sha256", "anchor-amendment",
                 "anchor-amendment-sha256", "role-manifest",
                 "role-manifest-sha256", "source-map", "source-map-sha256",
                 "anchor-map", "anchor-map-sha256", "public-receipt"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
