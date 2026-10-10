"""Independent saved INNER utility for T*, identity and class-only A diagnostics.

Run only under an R03 verification unit. Read no SEX or assessment archive.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .audit_diagnostic_q_parity import _canonical, _distance
from .independent_a_source import BANK_FIELDS, TOKEN_FIELDS, bound_npz, sha_array, sha_file
from .independent_replay import task_metrics, task_point_gate
from .replay_a_fixed_tstar_pair import wire_digest


ARMS = ("T_STAR", "A_IDENTITY", "A_CLASS_ONLY")
RELEASE_FIELDS = {"row_id", "unit", "prob_r1", "prob_r2", "decision_r1",
                  "decision_r2", "code_r1", "code_r2"}
REF_FIELDS = {"row_id", "unit", "U0_1", "U0_2", "Ucal_1", "Ucal_2", "d_1", "d_2"}


def check(condition: bool, name: str) -> None:
    if not condition:
        raise ValueError(name)


def queue_guard(amendment_sha256: str) -> None:
    check(os.environ.get("PCRL_QUEUE_STAGE") == "verification", "R03 verification stage")
    root = Path(os.environ["PCRL_QUEUE_ROOT"])
    uid = os.environ["PCRL_QUEUE_UNIT_ID"]
    state = json.loads((root / "state.json").read_text())
    unit = state.get("units", {}).get(uid, {})
    latest = next((entry for entry in reversed(state.get("amendments", []))
                   if "verification" in entry.get("stages", [])), None)
    check(not state.get("closed") and unit.get("status") == "running" and
          unit.get("child_pid") == os.getpid() and
          unit.get("spec", {}).get("stage") == "verification" and
          unit.get("spec", {}).get("amendment_sha256") == amendment_sha256 and
          latest is not None and latest.get("sha256") == amendment_sha256,
          "active R03 verification/amendment")


def load_role(manifest_path: Path, manifest: dict, kind: str) -> dict[str, np.ndarray]:
    key = f"INNER_SELECTION:{kind}"
    path = manifest_path.parent.parent / manifest["role_file_relative_path"][key]
    check(sha_file(path) == manifest["role_file_sha256"][key], f"{kind} role source SHA")
    with np.load(path, allow_pickle=False) as npz:
        return {field: np.array(npz[field], copy=True) for field in npz.files}


def main() -> None:
    ap = argparse.ArgumentParser()
    for flag in ("campaign", "portfolio", "repair-manifest", "output", "amendment-sha256"):
        ap.add_argument("--" + flag, required=True)
    args = ap.parse_args()
    queue_guard(args.amendment_sha256)
    camp = Path(args.campaign).resolve(strict=True)
    portfolio = Path(args.portfolio).resolve(strict=True)
    manifest_path = Path(args.repair_manifest).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    source_audit = json.loads((camp / "r01/SOURCE_AUDIT.json").read_text())
    prior_path = camp / "r04/A_PREFIT_INDEPENDENT_REPLAY.json"
    prior = json.loads(prior_path.read_text())
    check(prior["status"] == "PREFIT_SOURCE_DECODER_IDENTITY_PASS", "prior T* source replay")
    income = load_role(manifest_path, manifest, "income")
    occupation = load_role(manifest_path, manifest, "occupation")
    check(np.array_equal(income["row_id"], occupation["row_id"]) and
          np.array_equal(income["unit"], occupation["unit"]), "task role alignment")
    labels = {"income": income["income"], "occupation": occupation["occupation"]}
    majority = {task: int(np.argmax(np.bincount(y, minlength=k)))
                for task, y, k in (("income", labels["income"], 2),
                                   ("occupation", labels["occupation"], 6))}
    result = {"schema": "r04-a-fixed-inner-utility-replay-v1",
              "status": "TSTAR_IDENTITY_CLASS_ONLY_INNER_UTILITY_REPLAY_PASS",
              "scope": "saved INNER q/decision/code and role-only task labels/references; no SEX, assessment or reader fitting",
              "amendment_sha256": args.amendment_sha256,
              "role_manifest_sha256": sha_file(manifest_path),
              "r01_source_audit_sha256": sha_file(camp / "r01/SOURCE_AUDIT.json"),
              "prior_independent_tstar_source_replay_sha256": sha_file(prior_path),
              "seeds": {}}
    for seed in (0, 1):
        bank_root = portfolio / f"shared/.private/tstar_score_bank_s{seed}_v1"
        source = source_audit["tstar_banks"][str(seed)]
        check(sha_file(bank_root / "BANK_ADMISSION_RECEIPT.json") == source["receipt_sha256"],
              "T* bank receipt SHA")
        token = bound_npz(bank_root / "tokens_INNER_SELECTION.npz",
                          source["file_sha256"]["tokens_INNER_SELECTION.npz"], TOKEN_FIELDS)
        references = load_role(manifest_path, manifest, f"references_s{seed}")
        check(set(references) == REF_FIELDS and
              np.array_equal(token["row_id"], income["row_id"]) and
              np.array_equal(token["unit"], income["unit"]) and
              np.array_equal(references["row_id"], income["row_id"]) and
              np.array_equal(references["unit"], income["unit"]), "INNER source alignment")
        banks = {}
        canonical_banks = {}
        codes = {arm: {} for arm in ARMS}
        probabilities = {arm: {} for arm in ARMS}
        decisions = {arm: {} for arm in ARMS}
        for r in (1, 2):
            name = f"bank_r{r}.npz"
            old = bound_npz(bank_root / name, source["file_sha256"][name], BANK_FIELDS)
            banks[r] = old
            old_code = token[f"token_{r}"]
            check(np.array_equal(old_code, old["fine_to_hcal_token"][token[f"cell_{r}"]]),
                  f"seed{seed} r{r} source fine/code")
            canonical, canonical_decision, old_to_global = _canonical(
                old["probability"], old["token_id"], old["token_class"])
            canonical_banks[r] = canonical
            by_class = {}
            for cls in np.unique(canonical_decision):
                ids = np.flatnonzero(canonical_decision == cls)
                by_class[int(cls)] = min((sum(_distance(canonical[i], canonical[j])
                                              for j in ids), int(i)) for i in ids)[1]
            codes["T_STAR"][r] = old_code
            probabilities["T_STAR"][r] = old["probability"][old_code]
            decisions["T_STAR"][r] = old["token_class"][old_code]
            codes["A_IDENTITY"][r] = old_to_global[old_code]
            codes["A_CLASS_ONLY"][r] = np.array([
                by_class[int(canonical_decision[code])] for code in codes["A_IDENTITY"][r]],
                dtype=np.int64)
            for arm in ("A_IDENTITY", "A_CLASS_ONLY"):
                probabilities[arm][r] = canonical[codes[arm][r]]
                decisions[arm][r] = canonical_decision[codes[arm][r]]
            check(np.array_equal(decisions["A_IDENTITY"][r],
                                 decisions["A_CLASS_ONLY"][r]) and
                  np.array_equal(decisions["T_STAR"][r], references[f"d_{r}"]),
                  f"seed{seed} r{r} diagnostic decision preservation")
        baseline = {}
        for task, r in (("income", 1), ("occupation", 2)):
            baseline[task] = {ref: task_metrics(references[f"{ref}_{r}"], labels[task],
                                                 majority_class=majority[task])
                              for ref in ("U0", "Ucal")}
        seed_report = {"inner_rows": len(income["row_id"]),
                       "inner_groups": len(np.unique(income["unit"])),
                       "baseline": {task: {ref: {m: baseline[task][ref][m]
                                                for m in ("accuracy", "nll", "brier")}
                                           for ref in ("U0", "Ucal")}
                                    for task in ("income", "occupation")},
                       "arms": {}}
        for arm in ARMS:
            prefix = "A-REFERENCE-RELEASE" if arm == "T_STAR" else "A-DIAG-RELEASE"
            root = camp / "r03/.private/a_fixed_units" / f"{prefix}-{arm}-S{seed}-INNER"
            receipt_path = root / "REFERENCE_RELEASE_RECEIPT.json"
            receipt = json.loads(receipt_path.read_text())
            release = bound_npz(root / "reference_release.npz",
                                receipt["codes_file_sha256"], RELEASE_FIELDS)
            check(receipt["status"] == "FROZEN_DETERMINISTIC" and
                  receipt["arm_id"] == arm and receipt["source_seed"] == seed and
                  receipt["role"] == "INNER_SELECTION" and
                  receipt["source_binding_sha256"] == source["source_binding_sha256"] and
                  np.array_equal(release["row_id"], income["row_id"]) and
                  np.array_equal(release["unit"], income["unit"]),
                  f"{arm} seed{seed} saved INNER identity")
            current = {}
            observed = {}
            for task, r in (("income", 1), ("occupation", 2)):
                check(np.array_equal(release[f"code_r{r}"], codes[arm][r]) and
                      np.array_equal(release[f"prob_r{r}"], probabilities[arm][r]) and
                      np.array_equal(release[f"decision_r{r}"], decisions[arm][r]) and
                      receipt["bank_sha256_by_recipient"][f"r{r}"] ==
                        sha_array(banks[r]["probability"] if arm == "T_STAR" else
                                  canonical_banks[r]),
                      f"{arm} seed{seed} r{r} saved source/decision")
                bank_count = (len(banks[r]["token_id"]) if arm == "T_STAR" else
                              len(canonical_banks[r]))
                check(wire_digest(release[f"prob_r{r}"], release[f"decision_r{r}"],
                                  release[f"code_r{r}"], recipient=r,
                                  release_id=receipt["release_id"],
                                  source_binding=receipt["source_binding_sha256"],
                                  bank_hash=receipt["bank_sha256_by_recipient"][f"r{r}"],
                                  bank_size=bank_count) ==
                      receipt["wire_sha256_by_recipient"][f"r{r}"],
                      f"{arm} seed{seed} r{r} saved public wire")
                current[task] = task_metrics(release[f"prob_r{r}"], labels[task],
                                             decision=release[f"decision_r{r}"],
                                             majority_class=majority[task])
                observed[f"r{r}"] = len(np.unique(release[f"code_r{r}"]))
            gate = task_point_gate(current, baseline)
            if arm == "T_STAR":
                old = prior["seeds"][str(seed)]["roles"]["INNER_SELECTION"]
                for task in ("income", "occupation"):
                    for metric in ("accuracy", "nll", "brier"):
                        check(abs(current[task][metric] - old["identity_task"][task][metric]) <= 1e-12,
                              f"T* seed{seed} {task} prior independent metric parity")
                check(gate["point_pass"] == old["identity_point_gate"]["point_pass"],
                      f"T* seed{seed} prior independent point gate")
            pair = release["code_r1"].astype(np.int64) * (
                len(banks[2]["token_id"]) if arm == "T_STAR" else
                len(canonical_banks[2])) + release["code_r2"]
            seed_report["arms"][arm] = {
                "release_receipt_sha256": sha_file(receipt_path),
                "release_npz_sha256": sha_file(root / "reference_release.npz"),
                "observed_code_ids": observed,
                "observed_complete_view_pairs": len(np.unique(pair)),
                "task": {task: {m: current[task][m] for m in ("accuracy", "nll", "brier")}
                         for task in ("income", "occupation")},
                "point_gate": gate}
        result["seeds"][str(seed)] = seed_report
    output = Path(args.output)
    check(not output.exists(), "output already exists")
    output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
