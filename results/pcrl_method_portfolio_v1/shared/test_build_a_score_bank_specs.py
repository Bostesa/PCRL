"""Synthetic identity tests for the A source-bank S3 spec builder."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import build_a_score_bank_specs as builder
from safe_data_guard import sha_file


SHARED = Path(__file__).resolve().parent
CAMPAIGN = SHARED.parent
ROOT = CAMPAIGN.parents[1]
PLAN = SHARED / "A_SCORE_BANK_PREPROCESS_SPEC_DRAFT.json"


def _future_lock(plan: dict) -> dict:
    source = plan["source_manifests_sha256"]
    u_by_seed = {}
    for unit in plan["units"]:
        seed = unit["seed"]
        u_path = f"hcal_v1/admitted/rel__s{seed}__U/model.pt"
        u_by_seed[f"s{seed}"] = unit["source_files_sha256"][u_path]
    return {
        "status": "LOCAL_GIT_SEALED_EFFECTIVE", "adult_fit_permission": True,
        "fit_admission": "B_ONLY",
        "families": {"A": {"preprocessing_admission": "ALLOWED",
                           "fit_admission": "DENIED"}},
        "admitted_a_score_bank_unit_ids": plan["required_lock"]["admitted_a_score_bank_unit_ids"],
        "a_score_bank_cache_root": plan["cache_root_relative_to_coordinator"],
        "a_score_bank_runner_sha256": plan["runner_sha256"],
        "a_score_bank_preprocess_plan_sha256": builder.PLAN_SHA256,
        "data_admission": {"manifest_sha256": plan["data_access_manifest_sha256"]},
        "source_model_admission": {
            "hcal_model_manifest_sha256": source["hcal_MODEL_MANIFEST"],
            "hcal_source_admission_sha256": source["hcal_SOURCE_ADMISSION"],
            "lra_model_manifest_sha256": source["lra_MODEL_MANIFEST"],
            "U_model_pt_sha256_by_seed": u_by_seed},
    }


class AScoreBankSpecsTests(unittest.TestCase):
    def test_exact_ten_specs_and_no_role_values(self) -> None:
        plan = json.loads(PLAN.read_text())
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "CAMPAIGN_LOCK.json"
            lock_path.write_text(json.dumps(_future_lock(plan)))
            lock_sha = sha_file(lock_path)
            queue_root = (Path(directory) / "queue").resolve()
            specs = builder.build_specs(PLAN, lock_path, lock_sha,
                                        ROOT, queue_root)
        self.assertEqual(len(specs), 10)
        self.assertEqual({spec["unit_id"] for spec in specs},
                         set(plan["required_lock"]["admitted_a_score_bank_unit_ids"]))
        by_id = {unit["unit_id"]: unit for unit in plan["units"]}
        for spec in specs:
            unit = by_id[spec["unit_id"]]
            self.assertEqual(spec["mode"], "real_data")
            self.assertEqual(spec["stage"], "a_score_bank_preprocess")
            self.assertEqual(spec["data_access"], "A_ROLE_ONLY_REFERENCES_PREPROCESS")
            self.assertEqual((spec["family"], spec["team"], spec["arm"], spec["arm_id"]),
                             ("SHARED", "SHARED", "A_SCORE_BANK_PREPROCESS",
                              "A_SCORE_BANK_PREPROCESS"))
            self.assertEqual(spec["deps"], [])
            self.assertEqual((spec["max_wall_s"], spec["cpu_reserve_s"],
                              spec["max_rss_gib"]), (300, 300, 1.5))
            self.assertEqual(spec["runner_sha256"], plan["runner_sha256"])
            self.assertEqual(spec["preprocess_plan_sha256"], builder.PLAN_SHA256)
            self.assertEqual(spec["cwd"], str(ROOT))
            self.assertEqual(spec["argv"], [
                str((ROOT / "../../.venv/bin/python").resolve()),
                str((ROOT / plan["runner_relative_to_coordinator"]).resolve()),
                "--cache-root", str((ROOT / plan["cache_root_relative_to_coordinator"]).resolve()),
                "--seed", str(unit["seed"]), "--release", unit["release_key"],
                "--campaign-lock-sha256", lock_sha, "--queue-root", str(queue_root)])
            self.assertEqual(spec["outputs"], [
                str((ROOT / unit["private_output_receipt_relative_to_coordinator"]).resolve())])
            self.assertEqual(spec["env"]["OMP_NUM_THREADS"], "1")
            self.assertEqual(spec["env"]["OPENBLAS_NUM_THREADS"], "1")
            self.assertNotIn("sex", json.dumps(spec).lower())
            self.assertNotIn("assessment", json.dumps(spec).lower())

    def test_changed_lock_identity_is_refused(self) -> None:
        plan = json.loads(PLAN.read_text())
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "CAMPAIGN_LOCK.json"
            lock = _future_lock(plan)
            lock["families"]["A"]["fit_admission"] = "ALLOWED"
            lock_path.write_text(json.dumps(lock))
            with self.assertRaises((PermissionError, ValueError)):
                builder.build_specs(PLAN, lock_path, sha_file(lock_path),
                                    ROOT, Path(directory) / "queue")

    def test_private_spec_write_requires_committed_seal(self) -> None:
        with self.assertRaises(PermissionError):
            builder.write_specs([], "a" * 64, "0" * 64, Path("/synthetic/queue"))


if __name__ == "__main__":
    unittest.main()
