"""Engineering stage of the held-out calibration study (hcal; ENGINEERING_LOCK; synthetic data only).

The verdict is ENGINEERING_READY on CORRECTNESS ALONE: every registered synthetic check (ENGINEERING_CHECKS.json, the
exact pytest node IDs collected before ENGINEERING_LOCK and bound into it) passes, and every required correctness
category of the prompt (section 11 B) is covered by at least one passing check. No check requires a favourable effect,
reducing information below the decision floor, beating a task-only optimum, or any Adult result; a known-law
zero-leakage CLASS is never hidden. No real data, label or private artifact is loaded.

    <PRIVATE_CACHE>/hcal_v1/run/work.sh ENGINEERING_LOCK engineering
      -> results/pcrl_heldout_calibration_v1/ENGINEERING_RESULT.json (bound into SCIENCE_LOCK)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from hcal import ids as I

TEST_DIR = "tests/pcrl_heldout_calibration_v1"
CHECKS = "ENGINEERING_CHECKS.json"
RESULT = "ENGINEERING_RESULT.json"
# prompt section 11 B: required correctness categories -> node-ID substrings that cover them
CATEGORIES = {
    "identity probabilities and predictions": ["test_calib.py::test_identity_paths_tokens_and_u",
                                               "test_calib.py::test_solve_returning_exactly_one_uses_identity"],
    "calibration example with a known non-identity optimum": ["test_calib.py::test_temperature_known_optimum_k2",
                                                              "test_calib.py::test_token32_k2_against_bounded_brent"],
    "zero and small token counts, class fallbacks, boundary alpha, absent classes":
        ["test_calib.py::test_zero_calibration_rows", "test_calib.py::test_fit_token32_statuses_counts_and_fallbacks",
         "test_calib.py::test_class_temp_fallbacks_absent_and_threshold", "test_calib.py::test_temperature_boundaries",
         "test_calib.py::test_u_class_temp_fallback"],
    "NLL derivatives / convexity and per-token certificates against an independent solve":
        ["test_calib.py::test_nll_derivative_curvature_and_convexity", "test_calib.py::test_token32_k2_against",
         "test_calib.py::test_token32_k3_against_slsqp", "test_calib.py::test_token32_bitwise_equal_to_lra"],
    "exact group partitioning and duplicate handling": ["test_data_select.py::test_split",
                                                         "test_data_select.py::test_rank_is_sha256",
                                                         "test_data_select.py::test_receipt_disjointness_ok"],
    "unchanged primary privacy predictions across decoder variants of one partition":
        ["test_bank.py::test_common_record_is_shared_by_every_decoder_variant"],
    "attacker orientation, unseen tokens / pairs and ignoring one recipient":
        ["test_bank.py::test_orientation_is_fixed", "test_bank.py::test_unseen_token_uses_fit_prior",
         "test_bank.py::test_pair_bank_holds_coalition_and_both_ignore_recipient_banks"],
    "denominator / seed alignment and all valid / missing nominee / comparator status cases":
        ["test_infer.py::test_seed_alignment_refused", "test_infer.py::test_no_nominee",
         "test_infer.py::test_missing_comparator", "test_infer.py::test_nominee_scores_all_p_slots",
         "test_data_select.py::test_selection_missing", "test_data_select.py::test_selection_valid_nominee",
         "test_data_select.py::test_family_size", "test_bank.py::test_fresh_family_record_contract_and_seed_alignment"],
    "deployment bindings and refusals": ["test_deploy.py::"],
    "real-data control machinery on synthetic plants": ["test_controls.py::test_common_null_passes_and_every_plant_is_detected",
                                                         "test_controls.py::test_a_leaky_release_fails_the_common_null"],
    "attack fit rows strictly ATTACK_FIT_NEW and frozen-winner refit replay":
        ["test_bank.py::test_fit_rows_are_strictly_attack_fit_new",
         "test_bank.py::test_refit_selected_reproduces_stored_inner_predictions_bitwise"],
}


def collect(test_dir=TEST_DIR):
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", test_dir], cwd=str(I.WT),
                       capture_output=True, text=True)
    nodes = [ln.strip() for ln in r.stdout.splitlines() if "::" in ln and not ln.startswith(" ")]
    return sorted(nodes)


def coverage(nodes, passed=None):
    out = {}
    for cat, pats in CATEGORIES.items():
        hits = [n for n in nodes if any(p in n for p in pats)]
        ok = [n for n in hits if passed is None or n in passed]
        out[cat] = {"checks": hits, "covered": bool(ok)}
    return out


def write_checks():
    nodes = collect()
    cov = coverage(nodes)
    body = {"schema": "hcal-engineering-checks-v1", "rule": "ENGINEERING_READY iff every registered node passes and "
            "every category is covered by a passing node; correctness only (no favourable-effect requirement)",
            "test_dir": TEST_DIR, "nodes": nodes, "n_nodes": len(nodes),
            "nodes_sha256": hashlib.sha256("\n".join(nodes).encode()).hexdigest(), "categories": cov,
            "uncovered_categories": [c for c, v in cov.items() if not v["covered"]]}
    (I.PKG / CHECKS).write_text(json.dumps(body, indent=1) + "\n")
    return body


def run_checks(test_dir=TEST_DIR):
    with tempfile.TemporaryDirectory() as td:
        xml = f"{td}/junit.xml"
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test_dir,
                            f"--junitxml={xml}"], cwd=str(I.WT), capture_output=True, text=True)
        wall = time.time() - t0
        root = ET.parse(xml).getroot()
    passed, failed = set(), {}
    for tc in root.iter("testcase"):
        cls, name = tc.get("classname", ""), tc.get("name", "")
        node = f"{cls.replace('.', '/')}.py::{name}"
        bad = [c for c in tc if c.tag in ("failure", "error", "skipped")]
        if bad:
            failed[node] = bad[0].tag
        else:
            passed.add(node)
    return passed, failed, r.returncode, wall, r.stdout[-2000:]


def stage_engineering(D=None, shard_spec=None):
    reg = json.loads((I.PKG / CHECKS).read_text())
    passed, failed, rc, wall, tail = run_checks()
    nodes = reg["nodes"]
    missing = [n for n in nodes if n not in passed and n not in failed]
    cov = coverage(nodes, passed)
    ready = (rc == 0 and not failed and not missing and all(v["covered"] for v in cov.values()))
    res = {"schema": "hcal-engineering-result-v1", "verdict": "ENGINEERING_READY" if ready else "ENGINEERING_BLOCKED",
           "registered_nodes": len(nodes), "registered_nodes_sha256": reg["nodes_sha256"], "passed": len(passed),
           "failed": failed, "missing_registered": missing, "unregistered_passed": sorted(passed - set(nodes)),
           "categories": {c: v["covered"] for c, v in cov.items()}, "pytest_returncode": rc, "wall_s": round(wall, 1),
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "basis": "correctness only: synthetic and known-law checks; no Adult data, labels or favourable effects",
           "pytest_tail": tail}
    (I.PKG / RESULT).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("verdict", "registered_nodes", "passed", "failed", "missing_registered")}))


if __name__ == "__main__":
    if sys.argv[1:] == ["write-checks"]:
        b = write_checks()
        print(b["n_nodes"], "nodes; uncovered:", b["uncovered_categories"])
