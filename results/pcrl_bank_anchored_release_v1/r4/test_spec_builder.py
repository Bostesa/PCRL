"""Data-free checks for R4 stage-spec admission and binding."""

import hashlib
import json
import os
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.bank import FixedBank
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import GateWeights, JointPolicy
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_laws import BoundaryJointPolicy
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_materialize import materialize_arrays

from . import spec_builder
from .spec_builder import (ARMS, _selected_spec, eligible_arms,
                           planned_units, source_pins)


def write_json(path: Path, value: dict, mode=0o644) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    path.chmod(mode)
    return {"path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class AttackSpecBuilderTests(unittest.TestCase):
    def test_prefit_queue_bounds_are_hash_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            bounds = {"schema": "pcrl-bank-r4-attack-queue-bounds-v1",
                      "max_wall_s_per_release_unit": 300,
                      "cpu_reserve_s_per_release_unit": 300,
                      "max_rss_gib_per_release_unit": 2.0,
                      "projected_output_bytes_per_release_unit": 134217728}
            pin = write_json(Path(directory) / "BOUNDS.json", bounds)
            config = {"bounds_pin": pin, "max_wall_s": 300,
                      "cpu_reserve_s": 300, "max_rss_gib": 2.0,
                      "projected_output_bytes": 134217728}
            self.assertEqual(spec_builder._validate_bounds(config), pin["sha256"])
            with self.assertRaises(PermissionError):
                spec_builder._validate_bounds(config | {"max_wall_s": 301})

    def test_queue_python_preserves_working_venv_entrypoint(self):
        python = Path("/Users/nathansamson/PCRL/.venv/bin/python")
        if not python.exists():
            self.skipTest("campaign virtual environment unavailable")
        self.assertTrue(python.is_symlink())
        self.assertEqual(spec_builder._python_entrypoint(str(python)), str(python))
        with self.assertRaises(PermissionError):
            spec_builder._python_entrypoint("/bin/false")

    def test_r2_exact_law_archive_is_realization_invariant(self):
        b1 = FixedBank(np.array([[.8, .2], [.7, .3]], dtype=np.float64),
                       np.array([0, 0], dtype=np.int64), (0, 1))
        b2 = FixedBank(np.array([[.6, .1, .1, .1, .05, .05],
                                 [.5, .1, .1, .1, .1, .1]], dtype=np.float64),
                       np.array([0, 0], dtype=np.int64), (0, 1))
        base = JointPolicy(b1, b2, ((0, 1),), ((0, 1),),
                           GateWeights.zeros(input_width=12))
        policy = BoundaryJointPolicy(base, .4)
        rows = {"row_id": np.array([10, 11], dtype=np.int64),
                "unit": np.array([5, 6], dtype=np.int64),
                "score1": np.array([[.8, .2], [.7, .3]], dtype=np.float64),
                "score2": np.array([[.6, .1, .1, .1, .05, .05],
                                    [.5, .1, .1, .1, .1, .1]], dtype=np.float64),
                "cell1": np.zeros(2, dtype=np.int64),
                "cell2": np.zeros(2, dtype=np.int64)}
        law_hashes = []
        for realization in (0, 1, 2):
            law, _ = materialize_arrays(policy, **rows, source_seed=0,
                role="INNER_SELECTION", realization=realization,
                private_root=bytes(range(32)))
            stream = io.BytesIO()
            np.savez_compressed(stream, **law)
            law_hashes.append(hashlib.sha256(stream.getvalue()).hexdigest())
        self.assertEqual(len(set(law_hashes)), 1)

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
            fit = write_json(root / "R2_J_S0_FIT_RECEIPT.json", {
                "schema": "BAR1-r2-fit-receipt-v1", "unit_id": "R2_J_S0",
                "arm_id": "J", "source_seed": 0,
                "status": "INCOMPLETE_SEARCH", "source_lock_sha256": seal})
            unit = write_json(root / "R2_J_S0_UNIT_RECEIPT.json", {
                "schema": "BAR1-r2-unit-receipt-v1", "unit_id": "R2_J_S0",
                "status": "INCOMPLETE_SEARCH",
                "fit_receipt_sha256": fit["sha256"]})
            incomplete = {
                "schema": "pcrl-bank-r4-incomplete-arm-gate-v1",
                "arm_id": "J", "source_seal_sha256": seal,
                "status": "INCOMPLETE_SEARCH",
                "all_two_seed_task_reference_point_caps_pass": False,
                "reason": "R2 search reached its locked ceiling",
                "source_receipts_by_seed": {
                    "0": {"unit": unit, "fit": fit}, "1": None},
                "missing_r3_unit_ids": []}
            config["gate_by_arm"]["J"] = write_json(
                root / "J_INCOMPLETE_GATE.json", incomplete)
            self.assertEqual(eligible_arms(config), {"B0", "W_Q", "P_STAR"})
            missing = {**incomplete, "status": "MISSING_UTILITY",
                       "missing_r3_unit_ids": ["R3_J_S1_INNER"]}
            config["gate_by_arm"]["J"] = write_json(
                root / "J_MISSING_UTILITY_GATE.json", missing)
            self.assertEqual(eligible_arms(config), {"B0", "W_Q", "P_STAR"})
            (root / "R3_J_S1_INNER_RECEIPT.json").write_text("{}")
            with self.assertRaises(PermissionError):
                eligible_arms(config)
            bad_marker = {**incomplete,
                          "all_two_seed_task_reference_point_caps_pass": True}
            config["gate_by_arm"]["J"] = write_json(
                root / "J_BAD_INCOMPLETE_GATE.json", bad_marker)
            with self.assertRaises(PermissionError):
                eligible_arms(config)
            config["gate_by_arm"]["J"] = write_json(
                root / "J_INCOMPLETE_GATE.json", incomplete)
            bad = json.loads(Path(historical["path"]).read_text())
            bad["fit_bank_proxy_applied"] = True
            config["pstar_historical_gate"] = write_json(
                root / "PSTAR_HIST_BAD.json", bad)
            with self.assertRaises(PermissionError):
                eligible_arms(config)

    def test_selected_spec_binds_both_role_receipts_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            r2 = root / "r2"
            unit_root = r2 / ".private" / "units" / "R2_J_S0"
            policy = unit_root / "policy"
            policy.mkdir(parents=True, mode=0o700)
            (policy / "policy.npz").write_bytes(b"synthetic-policy-no-person-array")
            (policy / "policy.npz").chmod(0o600)
            fit_receipt_sha = "f" * 64
            policy_pin = write_json(policy / "POLICY_RECEIPT.json", {
                "schema": "BAR1-selected-policy-v1", "arm_id": "J",
                "source_seed": 0, "source_binding_sha256": "b" * 64,
                "fit_receipt_sha256": fit_receipt_sha,
                "policy_file_sha256": hashlib.sha256(
                    (policy / "policy.npz").read_bytes()).hexdigest()}, 0o600)
            release_items = []
            law_by_role = {}
            for role, realization, rows in (("OSF_DEFENSE_FIT", 0, 15434),
                                             ("ATTACK_FIT_NEW", 1, 4064),
                                             ("INNER_SELECTION", 1, 2235)):
                r = unit_root / f"{role}_R{realization}"
                r.mkdir(parents=True, mode=0o700)
                for name in ("law.npz", "release_codes.npz"):
                    (r / name).write_bytes((role + name).encode())
                    (r / name).chmod(0o600)
                law_sha = hashlib.sha256((r / "law.npz").read_bytes()).hexdigest()
                release_pin = write_json(r / "RELEASE_RECEIPT.json", {
                    "schema": "BAR1-role-release-v1", "arm_id": "J",
                    "source_seed": 0, "role": role, "realization": realization,
                    "source_binding_sha256": "b" * 64,
                    "policy_receipt_sha256": policy_pin["sha256"],
                    "rows": rows, "law_sha256": law_sha,
                    "release_codes_sha256": hashlib.sha256(
                        (r / "release_codes.npz").read_bytes()).hexdigest()}, 0o600)
                release_items.append({"role": role, "realization": realization,
                                      "private_relative_dir": r.name,
                                      "receipt_sha256": release_pin["sha256"],
                                      "law_sha256": law_sha})
                law_by_role[role] = law_sha
            unit_pin = write_json(r2 / "R2_J_S0_UNIT_RECEIPT.json", {
                "schema": "BAR1-r2-unit-receipt-v1", "unit_id": "R2_J_S0",
                "arm_id": "J", "source_seed": 0,
                "status": "COMPLETE_SELECTED_PROTECTED",
                "source_lock_sha256": "a" * 64,
                "source_binding_sha256": "b" * 64,
                "policy_receipt_sha256": policy_pin["sha256"],
                "fit_receipt_sha256": fit_receipt_sha,
                "role_releases": release_items})
            old_pin = write_json(r2 / "R2_J_S0_E_UNIT_RECEIPT.json", {
                "schema": "BAR1-r2-unit-receipt-v1", "unit_id": "R2_J_S0_E",
                "arm_id": "J", "source_seed": 0, "status": "SUPERSEDED",
                "source_lock_sha256": "a" * 64})
            r3 = root / "r3"
            sources = {}
            for role, short in (("OSF_DEFENSE_FIT", "fit"),
                                ("INNER_SELECTION", "inner")):
                unit_id = f"R3_J_S0_{short.upper()}"
                csv_pin = write_json(r3 / f"{unit_id}.csv", {"synthetic": True})
                utility_pin = write_json(r3 / f"{unit_id}_RECEIPT.json", {
                    "schema": "pcrl-bank-r3-utility-receipt-v1", "arm": "J",
                    "seed": 0, "role": role, "unit_id": unit_id,
                    "source_seal_sha256": "a" * 64,
                    "role_manifest_sha256": "c" * 64,
                    "public_table_sha256": "d" * 64,
                    "law_sha256": law_by_role[role],
                    "output_csv_sha256": csv_pin["sha256"],
                    "primary_point_pass": True})
                sources[f"s0_{short}"] = {
                    "unit_id": unit_id, "receipt_sha256": utility_pin["sha256"],
                    "csv_sha256": csv_pin["sha256"]}
            gate_pin = write_json(r3 / "J_GATE.json", {
                "schema": "pcrl-bank-r3-arm-point-eligibility-v1", "arm_id": "J",
                "all_two_seed_task_reference_point_caps_pass": True,
                "source_receipts": sources})
            config = {
                "private_root": str(root / ".private"),
                "public_root": str(root),
                "campaign_seal_path": str(root / "CAMPAIGN_LOCK.json"),
                "campaign_seal_sha256": "a" * 64,
                "role_manifest_path": str(root / "ROLE.json"),
                "role_manifest_sha256": "c" * 64,
                "table_by_seed": {"0": {"path": str(root / "TABLE.json"),
                                        "sha256": "d" * 64}},
                "gate_by_arm": {"J": gate_pin},
                "selected_r2_root": str(r2),
                "selected_unit_by_arm_seed": {"J_s0": unit_pin},
            }
            role = {"roles": {"ATTACK_FIT_NEW": {"rows": 4064},
                              "INNER_SELECTION": {"rows": 2235}}}
            spec = _selected_spec(config, "J", 0, 1, role)
            self.assertEqual(spec["release_receipt_sha256_by_role"].keys(),
                             set(("ATTACK_FIT_NEW", "INNER_SELECTION")))
            self.assertEqual(spec["attacker_seed"], 100001)
            self.assertEqual(spec["utility_receipts_by_role"]["INNER_SELECTION"]
                             ["receipt_sha256"], sources["s0_inner"]["receipt_sha256"])
            config["selected_unit_by_arm_seed"]["J_s0"] = old_pin
            with self.assertRaises(PermissionError):
                _selected_spec(config, "J", 0, 1, role)
            config["selected_unit_by_arm_seed"]["J_s0"] = unit_pin
            bad = json.loads((r3 / "R3_J_S0_INNER_RECEIPT.json").read_text())
            bad["law_sha256"] = "0" * 64
            new_pin = write_json(r3 / "R3_J_S0_INNER_RECEIPT.json", bad)
            gate = json.loads((r3 / "J_GATE.json").read_text())
            gate["source_receipts"]["s0_inner"]["receipt_sha256"] = new_pin["sha256"]
            config["gate_by_arm"]["J"] = write_json(r3 / "J_GATE.json", gate)
            with self.assertRaises(PermissionError):
                _selected_spec(config, "J", 0, 1, role)


if __name__ == "__main__":
    unittest.main()
