"""Independent prefit role-export audit; no mixed-source features or labels read.

This verifier does not import the custodian exporter, inherited role loaders,
candidate code, or common evaluator. It opens only row_id/unit/role members of
the mixed archive. All output is aggregate metadata and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np


SOURCE_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
ROLE_MANIFEST_SHA = "0994e84d426f2de87f7fe3ad2b70b49b19504bfd209419d308c17337f57f36f0"
FEATURE_NAMES_SHA = "7101f2fb37b623a08221c32b33eedf132cf0c4c376bb4042ab9cf16da08ad975"
EXPORTER_SHA = "75a945486f510d5280897da219038ea75c7b0d045185454d837583b33fac0d69"
ROLES = ("OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT", "ATTACK_FIT_NEW", "INNER_SELECTION")
LABELS = {
    "OSF_DEFENSE_FIT": ("income", "occupation", "sex"),
    "CALIBRATION_HELDOUT": ("income", "occupation"),
    "ATTACK_FIT_NEW": ("sex",),
    "INNER_SELECTION": ("income", "occupation", "sex"),
}
SOURCE_MEMBERS = {"row_id", "unit", "role", "X", "feature_names", "sex", "race", "y_income",
                  "y_occupation_group"}
REFERENCE_MEMBERS = {"row_id", "positions", "U0_1", "U0_2", "Ucal_1", "Ucal_2", "d_1", "d_2"}


def require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode() + str(arr.shape).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def members(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        names = [item.filename for item in archive.infolist()]
    require(len(names) == len(set(names)), "duplicate_archive_member")
    require(all(name.endswith(".npy") for name in names), "unexpected_archive_member")
    return {name[:-4] for name in names}


def group_fraction(seed: int, salt: str, group: int) -> float:
    digest = hashlib.sha256(f"{seed}|{salt}|{group}".encode()).hexdigest()
    return int(digest[:16], 16) / 2**64


def reconstruct(row_id: np.ndarray, unit: np.ndarray, old: np.ndarray) -> tuple[dict, np.ndarray]:
    """Reimplement the pinned rgj/smf/osf/hcal split from identity members."""
    n = len(row_id)
    require(n == 39205 and unit.shape == row_id.shape and old.shape == row_id.shape,
            "source_identity_shape")
    require(len(np.unique(row_id)) == n and np.all(np.diff(row_id) > 0), "source_id_order")
    rgj = np.full(n, "", dtype="<U32")
    for original, target in (("defense_train", "DEFENSE_FIT"), ("attacker_fit", "AUDIT_FIT"),
                             ("attacker_val", "INNER_SELECTION")):
        rgj[old == original] = target
    ix = np.flatnonzero(old == "defense_val")
    rgj[ix] = np.where(np.array([group_fraction(20261004, "dev", int(u)) for u in unit[ix]]) < .30,
                       "HEAD_VALIDATION", "DEVELOPMENT_ASSESSMENT")
    smf = np.full(n, "", dtype="<U32")
    ix = np.flatnonzero(rgj == "DEFENSE_FIT")
    smf[ix] = np.where(np.array([group_fraction(20261005, "assess", int(u)) for u in unit[ix]]) < .20,
                       "NEW_DEVELOPMENT_ASSESSMENT", "NEW_DEFENSE_FIT")
    for target in ("HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"):
        smf[rgj == target] = target
    osf = np.full(n, "", dtype="<U32")
    osf[smf == "NEW_DEFENSE_FIT"] = "OSF_DEFENSE_FIT"
    for target in ("HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"):
        osf[smf == target] = target
    pool = (np.isin(old, ("assessment", "cert")) |
            (rgj == "DEVELOPMENT_ASSESSMENT") |
            (smf == "NEW_DEVELOPMENT_ASSESSMENT"))
    require(not np.any(pool & (osf != "")), "pool_overlaps_fitting")
    blocked_groups = np.unique(unit[np.isin(osf, ("OSF_DEFENSE_FIT", "HEAD_VALIDATION",
                                                     "AUDIT_FIT", "INNER_SELECTION")) |
                                    np.isin(old, ("excluded_exposure", "excluded_dup"))])
    pool &= ~np.isin(unit, blocked_groups)
    osf[pool] = "OSF_DEVELOPMENT_ASSESSMENT"
    audit = np.flatnonzero(osf == "AUDIT_FIT")
    audit_groups = set(int(u) for u in unit[audit])
    ranked = sorted(audit_groups,
                    key=lambda u: (hashlib.sha256(f"hcal-v1|heldout|20261011|{u}".encode()).hexdigest(), u))
    chosen = set(ranked[:2000])
    all_held = audit[np.isin(unit[audit], list(chosen))]
    held = np.array([min((int(i) for i in all_held if int(unit[i]) == u), key=lambda i: int(row_id[i]))
                     for u in ranked[:2000]], dtype=np.int64)
    held = np.sort(held)
    attack = audit[~np.isin(unit[audit], list(chosen))]
    positions = {"OSF_DEFENSE_FIT": np.flatnonzero(osf == "OSF_DEFENSE_FIT"),
                 "CALIBRATION_HELDOUT": held, "ATTACK_FIT_NEW": attack,
                 "INNER_SELECTION": np.flatnonzero(osf == "INNER_SELECTION")}
    require(len(all_held) == 2001 and len(held) == 2000 and len(attack) == 4064,
            "hcal_heldout_attack_count")
    return positions, osf


def role_descriptor(ids: np.ndarray, groups: np.ndarray) -> dict:
    return {"rows": int(len(ids)), "groups": int(len(np.unique(groups))),
            "row_id_sha256": array_sha(np.sort(ids)), "group_id_sha256": array_sha(np.unique(groups)),
            "row_unit_pair_sha256": array_sha(np.column_stack((ids, groups)).astype(np.int64))}


def audit(source: Path, export_root: Path, manifest_path: Path, reference_root: Path) -> dict:
    export_root = export_root.resolve()
    require(file_sha(source) == SOURCE_SHA, "source_hash")
    require(file_sha(manifest_path) == ROLE_MANIFEST_SHA, "role_manifest_hash")
    require(members(source) == SOURCE_MEMBERS, "source_member_schema")
    role_manifest = json.loads(manifest_path.read_text())
    with np.load(source, allow_pickle=False) as archive:
        row_id = np.asarray(archive["row_id"], dtype=np.int64)
        unit = np.asarray(archive["unit"], dtype=np.int64)
        old = np.asarray(archive["role"])
    positions, osf = reconstruct(row_id, unit, old)
    for role, pinned in role_manifest["inherited"].items():
        role_pos = np.flatnonzero(osf == role)
        actual = role_descriptor(row_id[role_pos], unit[role_pos])
        for key in ("rows", "groups", "row_id_sha256", "group_id_sha256"):
            require(actual[key] == pinned[key], "inherited_role_" + role)
    held_groups = unit[positions["CALIBRATION_HELDOUT"]]
    all_held_pos = np.flatnonzero((osf == "AUDIT_FIT") & np.isin(unit, held_groups))
    held_group_desc = role_descriptor(row_id[all_held_pos], unit[all_held_pos])
    for key in ("rows", "groups", "row_id_sha256", "group_id_sha256"):
        require(held_group_desc[key] == role_manifest["roles"]["CALIBRATION_HELDOUT_GROUP_ROWS"][key],
                "heldout_group_member_" + key)
    receipt_path = export_root / "EXPORT_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("source_sha256") == SOURCE_SHA and
            receipt.get("role_manifest_sha256") == ROLE_MANIFEST_SHA and
            receipt.get("exporter_sha256") == EXPORTER_SHA, "export_receipt_identity")
    require(len(receipt.get("files", [])) == 25, "export_file_count")
    expected_paths = set()
    role_result = {}
    for role in ROLES:
        pos = positions[role]
        expected_ids, expected_groups = row_id[pos], unit[pos]
        desc = role_descriptor(expected_ids, expected_groups)
        pinned = (role_manifest["roles"] if role in ("CALIBRATION_HELDOUT", "ATTACK_FIT_NEW")
                  else role_manifest["inherited"])[role]
        for key in ("rows", "groups", "row_id_sha256", "group_id_sha256"):
            require(desc[key] == pinned[key], "pinned_role_" + role)
            require(desc[key] == receipt["roles"][role][key], "receipt_role_" + role)
        require(desc["row_unit_pair_sha256"] == receipt["roles"][role]["row_unit_pair_sha256"],
                "receipt_pair_" + role)
        fields = ("features", *LABELS[role], "references_s0", "references_s1", "references_s2")
        for field in fields:
            path = export_root / role / (field + ".npz")
            expected_paths.add(path.resolve())
            required = ({"row_id", "unit", "X", "feature_names"} if field == "features" else
                        {"row_id", "unit", "U0_1", "U0_2", "Ucal_1", "Ucal_2", "d_1", "d_2"}
                        if field.startswith("references_s") else {"row_id", "unit", field})
            require(members(path) == required, "export_member_schema")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: np.asarray(archive[key]) for key in required}
            require(np.array_equal(arrays["row_id"], expected_ids) and
                    np.array_equal(arrays["unit"], expected_groups), "export_identity_pair")
            if field == "features":
                names = [str(x) for x in arrays["feature_names"]]
                require(len(names) == 83 and hashlib.sha256("\n".join(names).encode()).hexdigest() == FEATURE_NAMES_SHA,
                        "feature_schema")
                require(not any(n.lower().split("=")[0] in {"sex", "race", "income", "occupation", "fnlwgt",
                                                          "row_id", "record"} for n in names), "forbidden_predictor")
                require(arrays["X"].shape == (len(pos), 83) and np.all(np.isfinite(arrays["X"])),
                        "feature_values_schema")
            elif field.startswith("references_s"):
                seed = int(field[-1])
                source_ref = reference_root / f"ref__s{seed}.npz"
                source_receipt = json.loads((reference_root / f"ref__s{seed}.json").read_text())
                require(file_sha(source_ref) == source_receipt["npz_sha256"] and
                        members(source_ref) == REFERENCE_MEMBERS, "reference_source_identity")
                with np.load(source_ref, allow_pickle=False) as ref:
                    ids = np.asarray(ref["row_id"], dtype=np.int64)
                    ix = np.searchsorted(ids, expected_ids)
                    require(np.array_equal(ids[ix], expected_ids), "reference_row_alignment")
                    for key in required - {"row_id", "unit"}:
                        require(np.array_equal(arrays[key], ref[key][ix]), "reference_array_alignment")
                for recipient, width in ((1, 2), (2, 6)):
                    d = arrays[f"d_{recipient}"]
                    require(d.shape == (len(pos),), "reference_decision_shape")
                    for kind in ("U0", "Ucal"):
                        prob = arrays[f"{kind}_{recipient}"]
                        require(prob.shape == (len(pos), width) and np.all(np.isfinite(prob)) and
                                np.all(prob >= 0) and np.allclose(prob.sum(axis=1), 1, atol=1e-8) and
                                np.array_equal(np.argmax(prob, axis=1), d), "reference_probability_schema")
            else:
                y = arrays[field]
                cap = {"income": 2, "occupation": 6, "sex": 2}[field]
                require(y.shape == (len(pos),) and np.issubdtype(y.dtype, np.integer) and
                        np.all((y >= 0) & (y < cap)), "label_encoding")
        role_result[role] = desc
    actual_paths = {path.resolve() for path in export_root.rglob("*.npz")}
    require(actual_paths == expected_paths, "unexpected_export_file")
    by_path = {entry["relative_path"]: entry for entry in receipt["files"]}
    require(len(by_path) == 25 and set(by_path) == {str(path.relative_to(export_root)) for path in expected_paths},
            "receipt_file_list")
    for path in expected_paths:
        entry = by_path[str(path.relative_to(export_root))]
        require(file_sha(path) == entry["sha256"], "export_file_hash")
    selected = np.concatenate(list(positions.values()))
    require(len(np.unique(selected)) == len(selected), "selected_role_overlap")
    barred = np.flatnonzero(np.isin(osf, ("OSF_DEVELOPMENT_ASSESSMENT", "HEAD_VALIDATION")) |
                           np.isin(old, ("excluded_exposure", "excluded_dup")))
    require(np.intersect1d(selected, barred).size == 0, "barred_identity_overlap")
    for i, first in enumerate(ROLES):
        for second in ROLES[i + 1:]:
            require(np.intersect1d(unit[positions[first]], unit[positions[second]]).size == 0,
                    "export_group_overlap")
    return {"schema": "pcrl-s4-independent-role-export-audit-v1", "status": "PASS",
            "scope": "source identity members only; role export feature/label/reference files; no mixed-source X/label read",
            "source_sha256": SOURCE_SHA, "role_manifest_sha256": ROLE_MANIFEST_SHA,
            "exporter_sha256": EXPORTER_SHA, "export_receipt_sha256": file_sha(receipt_path),
            "roles": role_result, "export_npz_count": 25, "barred_id_overlap": 0,
            "export_group_overlap": 0, "full_export_rerun": "NOT_RUN; deterministic writer checked by S2 synthetic test and static code"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.source, args.export, args.role_manifest, args.reference_root)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except BaseException as error:
        print("S4_ROLE_AUDIT_FAILED:" + type(error).__name__, file=sys.stderr)
        return 1
    print("S4_ROLE_AUDIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
