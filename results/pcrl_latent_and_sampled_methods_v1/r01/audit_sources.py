"""Read-only R01 admission audit; never decompresses X, task, or SEX arrays.

Usage: python3 audit_sources.py REPAIR_WORKTREE PORTFOLIO_WORKTREE PRIVATE_CACHE
The paths are command-line custody inputs and are omitted from JSON output.
"""

import hashlib
import io
import json
import sys
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode() + str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def member(path, name):
    # A single named member is opened. Excluded ZIP members are never decompressed.
    with ZipFile(path) as archive:
        return np.load(io.BytesIO(archive.read(name + ".npy")), allow_pickle=False)


def main():
    repair, portfolio, cache = map(Path, sys.argv[1:4])
    rpackage = repair / "results/pcrl_utility_reference_repair_v1"
    ppackage = portfolio / "results/pcrl_method_portfolio_v1"
    manifest_path = rpackage / "m1/DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    pmanifest_path = ppackage / "DATA_ACCESS_MANIFEST.json"
    pmanifest = json.loads(pmanifest_path.read_text())
    assert sha_file(pmanifest_path) == manifest["portfolio_manifest_sha256"]
    export_root = rpackage / ".private/role_export_v1"
    checked = {}
    for key, relative in manifest["role_file_relative_path"].items():
        path = rpackage / relative
        actual = sha_file(path)
        assert actual == manifest["role_file_sha256"][key]
        assert actual == pmanifest["real_file_specs"][key]["sha256"]
        checked[key] = actual
    role_arrays = {}
    role_summary = {}
    for role, meta in manifest["roles"].items():
        path = export_root / role / "features.npz"
        with ZipFile(path) as archive:
            assert set(archive.namelist()) == {"X.npy", "feature_names.npy", "row_id.npy", "unit.npy"}
        row = member(path, "row_id")
        group = member(path, "unit")
        names = member(path, "feature_names")
        assert row.dtype == group.dtype == np.int64
        assert row.shape == group.shape == (meta["rows"],)
        assert len(np.unique(group)) == meta["groups"]
        assert np.all(np.diff(row) > 0)
        assert sha_array(np.sort(row)) == meta["row_id_sha256"]
        assert sha_array(np.unique(group)) == meta["group_id_sha256"]
        assert sha_array(np.column_stack((row, group))) == meta["row_unit_pair_sha256"]
        names = [str(value) for value in names]
        assert len(names) == len(set(names)) == 83
        assert hashlib.sha256("\n".join(names).encode()).hexdigest() == manifest["source_schema_sha256"]
        assert not any(name.lower().split("=")[0] in {"sex", "race", "income", "occupation", "fnlwgt"} for name in names)
        role_arrays[role] = (row, group)
        role_summary[role] = {"rows": len(row), "groups": len(np.unique(group)),
                              "row_id_sha256": meta["row_id_sha256"],
                              "group_id_sha256": meta["group_id_sha256"],
                              "row_unit_pair_sha256": meta["row_unit_pair_sha256"]}
    overlap = {}
    for left, right in combinations(role_arrays, 2):
        lr, lg = role_arrays[left]
        rr, rg = role_arrays[right]
        row_n = len(np.intersect1d(lr, rr))
        group_n = len(np.intersect1d(lg, rg))
        assert row_n == group_n == 0
        overlap[left + "__" + right] = {"rows": row_n, "groups": group_n}
    source_npz = cache / "jcv_v1/inputs/adult_jcv.npz"
    assert sha_file(source_npz) == manifest["source_npz_sha256"]
    model_manifest = json.loads((repair / "results/pcrl_heldout_calibration_v1/MODEL_MANIFEST.json").read_text())
    u0_manifest = next(package for package in model_manifest["packages"] if package["release"] == "SRC|U")
    models = {}
    calibration = {}
    for seed in (0, 1):
        directory = cache / f"hcal_v1/admitted/rel__s{seed}__U"
        complete = json.loads((directory / "COMPLETE.json").read_text())
        record = json.loads((directory / "record.json").read_text())
        assert record["admitted_from"] == f"smf tl__s{seed}__e40"
        assert complete["id"] == f"rel__s{seed}__U"
        component_hashes = {}
        for name in ("model.pt", "head_0.joblib", "head_1.joblib", "record.json"):
            actual = sha_file(directory / name)
            assert actual == complete["files"][name]
            component_hashes[name] = actual
        assert component_hashes["model.pt"] == u0_manifest["model_pt_sha256"][str(seed)]
        models[str(seed)] = {"source_id": complete["id"], "components_sha256": component_hashes,
                             "fit_role": "OSF_DEFENSE_FIT", "continuation_epoch": 40}
        cal_dir = cache / f"hcal_v1/run/units/calU__s{seed}"
        cal_complete = json.loads((cal_dir / "COMPLETE.json").read_text())
        cal_path = cal_dir / "record.json"
        assert cal_complete["id"] == f"calU__s{seed}"
        assert sha_file(cal_path) == cal_complete["files"]["record.json"]
        cal_record = json.loads(cal_path.read_text())
        assert cal_record["schema"] == "hcal-calU-v1" and cal_record["seed"] == seed
        assert {"H-GLOBAL-TEMP|r1", "H-GLOBAL-TEMP|r2"} <= set(cal_record["tables"])
        calibration[str(seed)] = {"source_id": cal_complete["id"],
                                  "record_sha256": cal_complete["files"]["record.json"],
                                  "parameter_fit_role": "CALIBRATION_HELDOUT",
                                  "family_selection_exposure": "INNER_SELECTION"}
    bank_summary = {}
    for seed in (0, 1):
        root = ppackage / f"shared/.private/tstar_score_bank_s{seed}_v1"
        receipt_path = root / "BANK_ADMISSION_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["release_id"] == "U|FINE-TASK|i8o64|H-GLOBAL-TEMP"
        assert receipt["source_json_sha256"]["teacher_model_sha256"] == models[str(seed)]["components_sha256"]["model.pt"]
        assert receipt["source_binding"]["feature_names_sha256_newline_join"] == manifest["source_schema_sha256"]
        assert receipt["status"] == "ROLE_ONLY_BANK_ADMITTED"
        assert receipt["preprocessing_only"] is True
        assert receipt["data_access_manifest_sha256"] == sha_file(pmanifest_path)
        source_json = {
            "fine_sha256": cache / f"hcal_v1/admitted/units/fine__s{seed}/fine.json",
            "policy_sha256": cache / f"hcal_v1/admitted/units/pol__s{seed}__U_FINE-TASK_i8o64/policy.json",
            "decoder_file_sha256": cache / f"hcal_v1/package/decoder__s{seed}__U_FINE-TASK_i8o64_H-GLOBAL-TEMP.json",
        }
        for key, path in source_json.items():
            assert sha_file(path) == receipt["source_json_sha256"][key]
        binding_bytes = json.dumps(receipt["source_binding"], sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
        assert hashlib.sha256(binding_bytes).hexdigest() == receipt["source_binding_sha256"]
        for filename, expected in receipt["private_output_sha256"].items():
            assert sha_file(root / filename) == expected
        for role, role_receipt in receipt["role_receipts"].items():
            assert role_receipt["row_unit_pair_sha256"] == role_summary[role]["row_unit_pair_sha256"]
            assert role_receipt["source_reference_sha256"] == checked[f"{role}:references_s{seed}"]
        # Bank tables are fixed public decoding parameters; no per-person token arrays are opened.
        dims = {}
        for recipient in (1, 2):
            bank_path = root / f"bank_r{recipient}.npz"
            probability = member(bank_path, "probability")
            expected_classes = 2 if recipient == 1 else 6
            assert probability.ndim == 2 and probability.shape[1] == expected_classes
            dims[f"r{recipient}"] = {"output_tokens": len(probability), "task_classes": expected_classes}
        bank_summary[str(seed)] = {"receipt_sha256": sha_file(receipt_path),
                                   "source_binding_sha256": receipt["source_binding_sha256"],
                                   "source_json_sha256": {key: receipt["source_json_sha256"][key]
                                                          for key in source_json},
                                   "file_sha256": receipt["private_output_sha256"],
                                   "dimensions": dims,
                                   "independent_s4_parity": "PENDING_IN_PREDECESSOR_HANDOFF"}
    result = {"schema": "pcrl-new-campaign-r01-source-audit-v1",
              "status": "READ_ONLY_HASH_AND_MEMBERSHIP_PASS",
              "scope": "file hashes; feature-name and row/unit ZIP members; fixed bank probabilities; no X, per-person tokens, or label member decompression; no fit",
              "source_npz_sha256": manifest["source_npz_sha256"],
              "source_schema_sha256": manifest["source_schema_sha256"],
              "role_file_count": len(checked), "role_file_sha256": checked,
              "roles": role_summary, "overlap": overlap, "u0_models": models,
              "ucal_calibration": calibration,
              "tstar_banks": bank_summary,
              "reference_parity_inherited": "48/48 bitwise U0/Ucal cells on FIT/CAL/INNER, two seeds; not freshly recomputed in this read-only audit"}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
