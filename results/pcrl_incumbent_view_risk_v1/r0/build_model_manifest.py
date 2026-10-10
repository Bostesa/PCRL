"""Collect private model byte bindings without deserializing fitted arrays."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    index_path = ROOT / "r2/FIT_UNIT_INDEX.json"
    index = json.loads(index_path.read_text())
    candidates = {}
    for path in (ROOT / "r2/.private/units").rglob("POLICY_RECEIPT.json"):
        if path.is_symlink():
            raise ValueError("Symlinked policy receipt")
        candidates[sha(path)] = path
    models = []
    for record in index["records"]:
        expected = record["selected_policy_receipt_sha256"]
        path = candidates[expected]
        policy = json.loads(path.read_text())
        assert policy["arm_id"] == record["arm_id"]
        assert policy["source_seed"] == record["source_seed"]
        package = path.parent / "policy.npz"
        assert sha(package) == policy["policy_package_sha256"]
        public_receipt = ROOT / "r2" / (record["unit_id"] + "_UNIT_RECEIPT.json")
        assert sha(public_receipt) == record["unit_receipt_sha256"]
        models.append({
            "arm_id": record["arm_id"], "source_seed": record["source_seed"],
            "status": record["status"], "private_directory": str(path.parent.relative_to(ROOT)),
            "policy_receipt_sha256": expected,
            "policy_package_sha256": policy["policy_package_sha256"],
            "law_kind": policy["law_kind"], "eta1": policy["eta1"], "eta2": policy["eta2"],
            "parameter_count": policy["parameter_count"], "exact_p0_alias": policy["exact_p0_alias"],
            "source_binding_sha256": policy["source_binding_sha256"],
            "bank_sha256": policy["public_bank_sha256"],
            "fit_receipt": str(public_receipt.relative_to(ROOT)),
            "fit_receipt_sha256": sha(public_receipt),
        })
    assert len(models) == 14
    output = {
        "schema": "IVR1-final-model-manifest-v1",
        "scope": "Frozen source and fourteen actually fitted selected policies; metadata/hash reads only",
        "p0": {"identity": "U|JOINT|i8o64|l0.1|H-CLASS-TEMP", "manifest": "r1/P0_SOURCE_MANIFEST.json",
               "sha256": sha(ROOT / "r1/P0_SOURCE_MANIFEST.json"), "new_fit": False},
        "policies": models, "fit_index_sha256": sha(index_path),
        "returned_new_family": "V_L", "joint_diagnostic": "V_J", "strongest_control": "V_S21",
        "trained_policy_units": 14, "nonidentity_policies": 12, "p0_alias_policies": 2,
        "new_calibration_fits": 0, "new_teacher_or_task_head_fits": 0,
        "models_and_per_person_arrays_public": False,
        "source_roles": ["OSF_DEFENSE_FIT", "ATTACK_FIT_NEW", "INNER_SELECTION"],
        "deployment": "raw permitted83 -> frozen P0 router -> selected channel -> original metadata-bearing wire",
    }
    (ROOT / "MODEL_MANIFEST.json").write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"selected_policies": len(models), "manifest_sha256": sha(ROOT / "MODEL_MANIFEST.json")}))


if __name__ == "__main__":
    main()
