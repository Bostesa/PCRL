"""One isolated mechanical export of permitted Adult development roles.

This custodian program is the sole authorized exception for reading the mixed-role
source NPZ. Its compressed X/label members deserialize whole arrays, including
excluded rows, before immediate positional selection. No excluded value is
inspected, computed on, printed or written. Scientific workers use only the
verified role-specific outputs through SafeRoleStore.

Run only after static review:
  PYTHONPATH=. python3 results/pcrl_method_portfolio_v1/shared/export_allowed_roles.py \
      --output results/pcrl_method_portfolio_v1/shared/.private/role_export_v1
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from safe_data_guard import sha_arr, sha_file


REPO = Path(__file__).resolve().parents[3]
CAMPAIGN = Path(__file__).resolve().parent.parent
PRIVATE_ROOT = Path(__file__).resolve().parent / ".private"
SOURCE = Path.home() / "PCRL_eval_cache_private/jcv_v1/inputs/adult_jcv.npz"
SOURCE_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
ROLE_MANIFEST = REPO / "results/pcrl_heldout_calibration_v1/ROLE_MANIFEST.json"
ROLE_MANIFEST_SHA = "0994e84d426f2de87f7fe3ad2b70b49b19504bfd209419d308c17337f57f36f0"
NUMERIC_ADMISSION = REPO / "results/pcrl_joint_complete_view_method_v1/DATA_ADMISSION.json"
NUMERIC_ADMISSION_SHA = "fa49b4fd1cc72b497fc7f0268d1748bd6bb8da623a46575a0b71d75247591c55"
REFERENCE_ROOT = Path.home() / "PCRL_eval_cache_private/ccm_v1/ref"
EXPECTED_SOURCE_MEMBERS = {"row_id", "unit", "role", "X", "feature_names", "sex", "race", "y_income",
                           "y_occupation_group"}
OLD_ROLES = ("OSF_DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION",
             "OSF_DEVELOPMENT_ASSESSMENT")
EXPORT_ROLES = ("OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT", "ATTACK_FIT_NEW", "INNER_SELECTION")
LABELS = {
    "OSF_DEFENSE_FIT": ("income", "occupation", "sex"),
    "CALIBRATION_HELDOUT": ("income", "occupation"),
    "ATTACK_FIT_NEW": ("sex",),
    "INNER_SELECTION": ("income", "occupation", "sex"),
}
SOURCE_LABEL = {"income": "y_income", "occupation": "y_occupation_group", "sex": "sex"}
REF_MEMBERS = {"row_id", "positions", "U0_1", "U0_2", "Ucal_1", "Ucal_2", "d_1", "d_2"}
OUTPUT_REF_MEMBERS = {"row_id", "unit", "U0_1", "U0_2", "Ucal_1", "Ucal_2", "d_1", "d_2"}
FEATURE_NAMES_SHA = "7101f2fb37b623a08221c32b33eedf132cf0c4c376bb4042ab9cf16da08ad975"


class ExportRefused(RuntimeError):
    pass


def _exact_members(path: Path, expected: set[str]) -> None:
    with zipfile.ZipFile(path) as archive:
        names = [item.filename for item in archive.infolist()]
    if len(names) != len(expected) or set(names) != {x + ".npy" for x in expected}:
        raise ExportRefused("source/reference archive member schema mismatch")


def _same_role_record(got: dict, want: dict) -> None:
    for key in ("rows", "groups", "row_id_sha256", "group_id_sha256"):
        if got.get(key) != want.get(key):
            raise ExportRefused("pinned role membership/count/hash mismatch")


def _role_positions(source: zipfile.ZipFile | object, role_manifest: dict):
    """Read only source identities here. Return verified source-position arrays."""
    # Pure role functions are imported only after file/code review. Their modules
    # define constants at import and do not call any loader at import time.
    from osf import data as OD
    from hcal import data as HD

    old_role = np.asarray(source["role"])
    unit = np.asarray(source["unit"], dtype=np.int64)
    row_id = np.asarray(source["row_id"], dtype=np.int64)
    if old_role.shape != unit.shape or unit.shape != row_id.shape or len(row_id) != 39205:
        raise ExportRefused("source identity schema mismatch")
    if np.unique(row_id).size != row_id.size or not np.all(np.diff(row_id) > 0):
        raise ExportRefused("source row order or uniqueness mismatch")
    assigned, subrole, pool, _info = OD.assign_roles(old_role, unit)
    OD.check_partition(assigned, subrole, pool, unit, old_role)
    keep = np.flatnonzero(assigned != "")
    D = {"row_id": row_id[keep], "unit": unit[keep],
         "idx": {r: np.flatnonzero(assigned[keep] == r) for r in OLD_ROLES}}
    derived, meta = HD.build_roles(D)
    receipt = HD.role_receipt(D, derived, meta)
    if not receipt["ok"]:
        raise ExportRefused("pinned hcal role receipt failed")
    for role in OLD_ROLES:
        _same_role_record(receipt["inherited"][role], role_manifest["inherited"][role])
    for role in derived:
        _same_role_record(receipt["roles"][role], role_manifest["roles"][role])
    for role in ("CALIBRATION_HELDOUT", "CALIBRATION_TRAIN_MATCHED"):
        for key in ("selected_groups_in_rank_order_sha256", "representatives_in_rank_order_sha256"):
            if receipt["roles"][role][key] != role_manifest["roles"][role][key]:
                raise ExportRefused("pinned hcal representative/rank hash mismatch")
    if receipt["disjointness"] != role_manifest["disjointness"] or receipt["containment"] != role_manifest["containment"]:
        raise ExportRefused("pinned group-disjointness or containment mismatch")
    positions = {"OSF_DEFENSE_FIT": keep[D["idx"]["OSF_DEFENSE_FIT"]],
                 "CALIBRATION_HELDOUT": keep[derived["CALIBRATION_HELDOUT"]],
                 "ATTACK_FIT_NEW": keep[derived["ATTACK_FIT_NEW"]],
                 "INNER_SELECTION": keep[D["idx"]["INNER_SELECTION"]]}
    barred = np.flatnonzero(np.isin(assigned, ("OSF_DEVELOPMENT_ASSESSMENT", "HEAD_VALIDATION")))
    excluded = np.flatnonzero(assigned == "")
    all_selected = np.concatenate(list(positions.values()))
    if np.unique(all_selected).size != all_selected.size or np.intersect1d(all_selected, barred).size or np.intersect1d(all_selected, excluded).size:
        raise ExportRefused("permitted roles overlap or include barred/excluded rows")
    if len(positions["ATTACK_FIT_NEW"]) != 4064 or np.unique(unit[positions["ATTACK_FIT_NEW"]]).size != 4061:
        raise ExportRefused("ATTACK_FIT_NEW count mismatch")
    if len(positions["CALIBRATION_HELDOUT"]) != 2000:
        raise ExportRefused("CALIBRATION_HELDOUT representative count mismatch")
    if len(derived["CALIBRATION_HELDOUT_GROUP_ROWS"]) != 2001:
        raise ExportRefused("CALIBRATION_HELDOUT inherited group-member count mismatch")
    return positions, row_id, unit, assigned, receipt


def _write_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Fixed member order and ZIP metadata make identical input byte-identical."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(arrays):
            content = io.BytesIO()
            np.lib.format.write_array(content, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content.getvalue(), compresslevel=6)


def _receipt(path: Path, root: Path, role: str, field: str, arrays: dict) -> dict:
    ids = arrays["row_id"]
    unit = arrays["unit"]
    return {"role": role, "field": field, "relative_path": str(path.relative_to(root)),
            "sha256": sha_file(path), "members": sorted(arrays), "rows": int(ids.size),
            "groups": int(np.unique(unit).size), "row_id_sha256": sha_arr(np.sort(ids)),
            "group_id_sha256": sha_arr(np.unique(unit)),
            "row_unit_pair_sha256": sha_arr(np.column_stack((ids, unit)).astype(np.int64))}


def _export_features_labels(source, stage: Path, positions: dict, row_id, unit, assigned) -> list[dict]:
    names = np.asarray(source["feature_names"])
    name_strings = [str(x) for x in names]
    if len(names) != 83 or hashlib.sha256("\n".join(name_strings).encode()).hexdigest() != FEATURE_NAMES_SHA:
        raise ExportRefused("pinned 83-column feature schema mismatch")
    if any(n.lower().split("=")[0] in {"sex", "race", "income", "occupation", "fnlwgt", "row_id", "record"}
           for n in name_strings):
        raise ExportRefused("forbidden label or ID predictor column")
    selected = np.sort(np.concatenate(list(positions.values())))
    # NPZ member decompression transiently materializes the all-role X array.
    # Positional selection is the only operation on it; the all-role array is
    # discarded before feature preprocessing or any scientific computation.
    whole_x = source["X"]
    x_selected = np.array(whole_x[selected], copy=True)
    del whole_x
    if x_selected.shape != (selected.size, 83):
        raise ExportRefused("selected feature shape mismatch")
    from smf import data as SD
    fitted, _numeric_receipt = SD.refit_numeric(x_selected, names, assigned[selected] == "OSF_DEFENSE_FIT")
    del x_selected
    receipts = []
    for role in EXPORT_ROLES:
        src_pos = positions[role]
        local_pos = np.searchsorted(selected, src_pos)
        if not np.array_equal(selected[local_pos], src_pos):
            raise ExportRefused("selected feature alignment mismatch")
        arrays = {"row_id": row_id[src_pos].astype(np.int64, copy=True),
                  "unit": unit[src_pos].astype(np.int64, copy=True),
                  "X": fitted[local_pos].copy(), "feature_names": names.copy()}
        path = stage / role / "features.npz"
        _write_archive(path, arrays)
        receipts.append(_receipt(path, stage, role, "features", arrays))
    del fitted
    for field, source_key in SOURCE_LABEL.items():
        target_roles = [role for role in EXPORT_ROLES if field in LABELS[role]]
        # The archive format forces transient full-member deserialization. Slice
        # only authorized positions, then discard; never inspect other values.
        whole_label = source[source_key]
        for role in target_roles:
            src_pos = positions[role]
            arrays = {"row_id": row_id[src_pos].astype(np.int64, copy=True),
                      "unit": unit[src_pos].astype(np.int64, copy=True),
                      field: np.asarray(whole_label[src_pos], dtype=np.int64).copy()}
            path = stage / role / f"{field}.npz"
            _write_archive(path, arrays)
            receipts.append(_receipt(path, stage, role, field, arrays))
        del whole_label
    return receipts


def _export_references(stage: Path, positions: dict, row_id, unit) -> list[dict]:
    receipts = []
    expected_union = np.sort(np.concatenate([row_id[positions[r]] for r in EXPORT_ROLES]))
    if expected_union.size != 23733 or np.unique(expected_union).size != expected_union.size:
        raise ExportRefused("reference-role union mismatch")
    for seed in (0, 1, 2):
        ref_path = REFERENCE_ROOT / f"ref__s{seed}.npz"
        receipt_path = REFERENCE_ROOT / f"ref__s{seed}.json"
        rec = json.loads(receipt_path.read_text())
        if rec.get("seed") != seed or rec.get("rows") != 23733 or set(rec.get("roles", [])) != set(EXPORT_ROLES):
            raise ExportRefused("frozen reference receipt identity mismatch")
        if sha_file(ref_path) != rec.get("npz_sha256"):
            raise ExportRefused("frozen reference archive hash mismatch")
        _exact_members(ref_path, REF_MEMBERS)
        with np.load(ref_path, allow_pickle=False) as ref:
            ids = np.asarray(ref["row_id"], dtype=np.int64)
            if not np.array_equal(ids, expected_union):
                raise ExportRefused("frozen reference row membership/order mismatch")
            if sha_arr(ids) != rec["array_sha256"]["row_id"]:
                raise ExportRefused("frozen reference row hash mismatch")
            for role in EXPORT_ROLES:
                role_ids = row_id[positions[role]]
                ix = np.searchsorted(ids, role_ids)
                if not np.array_equal(ids[ix], role_ids):
                    raise ExportRefused("frozen reference role alignment mismatch")
                arrays = {"row_id": role_ids.astype(np.int64, copy=True),
                          "unit": unit[positions[role]].astype(np.int64, copy=True)}
                for key in sorted(REF_MEMBERS - {"row_id", "positions"}):
                    whole = ref[key]
                    if sha_arr(whole) != rec["array_sha256"][key]:
                        raise ExportRefused("frozen reference member hash mismatch")
                    arrays[key] = np.asarray(whole[ix]).copy()
                    del whole
                for i, k in ((1, 2), (2, 6)):
                    for kind in ("U0", "Ucal"):
                        prob = arrays[f"{kind}_{i}"]
                        if prob.shape != (len(role_ids), k) or not np.all(np.isfinite(prob)) or \
                                np.any(prob < 0) or not np.allclose(prob.sum(axis=1), 1, atol=1e-8):
                            raise ExportRefused("frozen reference probability schema mismatch")
                    if not np.array_equal(np.argmax(arrays[f"U0_{i}"], axis=1), arrays[f"d_{i}"]) or \
                            not np.array_equal(np.argmax(arrays[f"Ucal_{i}"], axis=1), arrays[f"d_{i}"]):
                        raise ExportRefused("frozen reference decision mismatch")
                path = stage / role / f"references_s{seed}.npz"
                _write_archive(path, arrays)
                receipts.append(_receipt(path, stage, role, f"references_s{seed}", arrays))
    return receipts


def export(output: Path) -> dict:
    if output.resolve() != (PRIVATE_ROOT / "role_export_v1").resolve():
        raise ExportRefused("output must be the campaign private versioned role-export root")
    if output.exists():
        raise ExportRefused("versioned output already exists; refusing overwrite")
    if sha_file(SOURCE) != SOURCE_SHA or sha_file(ROLE_MANIFEST) != ROLE_MANIFEST_SHA or \
            sha_file(NUMERIC_ADMISSION) != NUMERIC_ADMISSION_SHA:
        raise ExportRefused("source, role manifest or numeric admission hash mismatch")
    role_manifest = json.loads(ROLE_MANIFEST.read_text())
    _exact_members(SOURCE, EXPECTED_SOURCE_MEMBERS)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="role_export_v1.staging.", dir=output.parent))
    try:
        with np.load(SOURCE, allow_pickle=False) as source:
            positions, row_id, unit, assigned, role_receipt = _role_positions(source, role_manifest)
            files = _export_features_labels(source, stage, positions, row_id, unit, assigned)
            files += _export_references(stage, positions, row_id, unit)
        expected_files = 13 + 12
        if len(files) != expected_files:
            raise ExportRefused("role-only output file count mismatch")
        summary = {"schema": "pcrl-method-portfolio-data-export-v1", "status": "STAGED_FOR_INDEPENDENT_ADMISSION",
                   "source_sha256": SOURCE_SHA, "role_manifest_sha256": ROLE_MANIFEST_SHA,
                   "numeric_admission_sha256": NUMERIC_ADMISSION_SHA,
                   "exporter_sha256": sha_file(Path(__file__)),
                   "mechanical_transient_deserialization": "source X and selected label members deserialize full arrays before direct positional selection; no excluded values are inspected, computed on, logged or exported; race member is never requested",
                   "roles": {role: {"rows": int(len(pos)), "groups": int(np.unique(unit[pos]).size),
                                    "row_id_sha256": sha_arr(np.sort(row_id[pos])),
                                    "group_id_sha256": sha_arr(np.unique(unit[pos])),
                                    "row_unit_pair_sha256": sha_arr(np.column_stack((row_id[pos], unit[pos])).astype(np.int64))}
                             for role, pos in positions.items()},
                   "hcal_receipt_ok": bool(role_receipt["ok"]), "files": files}
        (stage / "EXPORT_RECEIPT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        for p in stage.rglob("*"):
            if p.is_file(): p.chmod(0o600)
        os.replace(stage, output)
        return summary
    except BaseException:
        # Preserve a failed staging directory for private inspection, but never
        # emit real arrays, source values or the exception message to stdout.
        (stage / "FAILED_NO_SCIENTIFIC_USE").write_text("Custodian export failed; inspect code and private files only.\n")
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        receipt = export(args.output.resolve())
    except BaseException as exc:
        print(f"EXPORT_REFUSED:{type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps({"status": receipt["status"], "roles": {k: [v["rows"], v["groups"]]
                                                               for k, v in receipt["roles"].items()},
                      "files": len(receipt["files"]), "exporter_sha256": receipt["exporter_sha256"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
