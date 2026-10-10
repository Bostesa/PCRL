"""Data-free checks for R4 stage-spec admission and binding."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from .spec_builder import (ARMS, _selected_spec, eligible_arms,
                           planned_units, source_pins)


def write_json(path: Path, value: dict, mode=0o644) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    path.chmod(mode)
    return {"path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class AttackSpecBuilderTests(unittest.TestCase):
    def test_fixed_unit_family_and_matched_attacker_seeds(self):
        self.assertEqual(len(planned_units(set(ARMS))), 44)
        self.assertEqual(len(planned_units({"B0", "W_Q", "P_STAR"})), 6)
        self.assertEqual(len(planned_units({"B0", "W_Q", "P_STAR", "J"})), 12)
        self.assertEqual([r for a, s, r in planned_units({"B0", "D", "T"})
                          if a == "D"], [0, 0])
        with self.assertRaises(ValueError):
            planned_units({"J"})

    def test_live_worker_source_sets_are_exact(self):
        self.assertEqual(len(source_pins("policy")), 14)
        self.assertEqual(len(source_pins("witness")), 11)
        self.assertEqual(len(source_pins("pstar")), 9)

    def test_distinct_b0_wq_and_pstar_gate_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seal = "a" * 64
            gates = {}
            for arm in ARMS:
                if arm == "P_STAR":
                    continue
                if arm == "W_Q":
                    body = {"schema": "pcrl-bank-r3-historical-original-inner-eligibility-v1",
                            "arm_id": arm,
                            "all_two_seed_task_reference_point_caps_pass": True,
                            "original_inner_point_clauses_passed": 24,
                            "original_inner_point_clauses_total": 24,
                            "fit_bank_proxy_applied": False,
                            "fit_bank_proxy_pass": None}
                else:
                    passed = arm == "B0"
                    body = {"schema": "pcrl-bank-r3-arm-point-eligibility-v1",
                            "arm_id": arm, "source_seal_sha256": seal,
                            "all_two_seed_task_reference_point_caps_pass": passed,
                            "fit_bank_proxy_pass": passed,
                            "inner_original_caps_pass": passed}
                gates[arm] = write_json(root / f"{arm}.json", body)
            historical = write_json(root / "PSTAR_HIST.json", {
                "schema": "pcrl-bank-r3-historical-original-inner-eligibility-v1",
                "arm_id": "P_STAR", "all_two_seed_task_reference_point_caps_pass": True,
                "original_inner_point_clauses_passed": 24,
                "original_inner_point_clauses_total": 24,
                "fit_bank_proxy_applied": False, "fit_bank_proxy_pass": None})
            independent = write_json(root / "PSTAR_R5.json", {
                "schema": "pcrl-bank-r5-pstar-common-inner-replay-v1",
                "status": "PASS_UTILITY_POINT_ONLY",
                "all_24_point_clauses_pass": True})
            config = {"campaign_seal_sha256": seal, "gate_by_arm": gates,
                      "pstar_historical_gate": historical,
                      "pstar_gate": independent}
            self.assertEqual(eligible_arms(config), {"B0", "W_Q", "P_STAR"})
            bad = json.loads(Path(historical["path"]).read_text())
            bad["fit_bank_proxy_applied"] = True
            config["pstar_historical_gate"] = write_json(
                root / "PSTAR_HIST_BAD.json", bad)
            with self.assertRaises(PermissionError):
                eligible_arms(config)

    def test_selected_spec_binds_both_role_receipts_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / ".private" / "policies" / "J_s0"
            policy.mkdir(parents=True, mode=0o700)
            (policy / "policy.npz").write_bytes(b"synthetic-policy-no-person-array")
            (policy / "policy.npz").chmod(0o600)
            policy_pin = write_json(policy / "POLICY_RECEIPT.json", {
                "schema": "BAR1-selected-policy-v1", "arm_id": "J",
                "source_seed": 0, "source_binding_sha256": "b" * 64,
                "policy_file_sha256": hashlib.sha256(
                    (policy / "policy.npz").read_bytes()).hexdigest()}, 0o600)
            release_root = root / ".private" / "releases"
            for role, rows in (("ATTACK_FIT_NEW", 4064),
                               ("INNER_SELECTION", 2235)):
                r = release_root / f"J_s0_{role}_r1"
                r.mkdir(parents=True, mode=0o700)
                for name in ("law.npz", "release_codes.npz"):
                    (r / name).write_bytes((role + name).encode())
                    (r / name).chmod(0o600)
                write_json(r / "RELEASE_RECEIPT.json", {
                    "schema": "BAR1-role-release-v1", "arm_id": "J",
                    "source_seed": 0, "role": role, "realization": 1,
                    "source_binding_sha256": "b" * 64,
                    "policy_receipt_sha256": policy_pin["sha256"],
                    "rows": rows,
                    "law_sha256": hashlib.sha256((r / "law.npz").read_bytes()).hexdigest(),
                    "release_codes_sha256": hashlib.sha256(
                        (r / "release_codes.npz").read_bytes()).hexdigest()}, 0o600)
            config = {
                "private_root": str(root / ".private"),
                "public_root": str(root),
                "campaign_seal_path": str(root / "CAMPAIGN_LOCK.json"),
                "campaign_seal_sha256": "a" * 64,
                "role_manifest_path": str(root / "ROLE.json"),
                "role_manifest_sha256": "c" * 64,
                "table_by_seed": {"0": {"path": str(root / "TABLE.json"),
                                        "sha256": "d" * 64}},
                "gate_by_arm": {"J": {"path": str(root / "GATE.json"),
                                      "sha256": "e" * 64}},
                "selected_policy_root": str(policy.parent),
                "selected_release_root": str(release_root),
            }
            role = {"roles": {"ATTACK_FIT_NEW": {"rows": 4064},
                              "INNER_SELECTION": {"rows": 2235}}}
            spec = _selected_spec(config, "J", 0, 1, role)
            self.assertEqual(spec["release_receipt_sha256_by_role"].keys(),
                             set(("ATTACK_FIT_NEW", "INNER_SELECTION")))
            self.assertEqual(spec["attacker_seed"], 100001)
            bad = json.loads((release_root / "J_s0_INNER_SELECTION_r1" /
                              "RELEASE_RECEIPT.json").read_text())
            bad["source_binding_sha256"] = "f" * 64
            write_json(release_root / "J_s0_INNER_SELECTION_r1" /
                       "RELEASE_RECEIPT.json", bad, 0o600)
            with self.assertRaises(PermissionError):
                _selected_spec(config, "J", 0, 1, role)


if __name__ == "__main__":
    unittest.main()
