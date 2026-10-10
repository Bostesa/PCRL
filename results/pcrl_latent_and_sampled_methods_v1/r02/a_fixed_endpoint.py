"""Proposal-only A endpoint for the 18 fixed diagnostic/reference attacks.

The original 58-instance registry and ordered K=6,885 family remain intact.
Only an R03 verification unit under a later dated amendment may read INNER
labels or release arrays. FIT custody checks here use aggregate receipts.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

from .a_endpoint_driver import (A_ARMS, A_REFS, CAMPAIGN, STOCHASTIC_ARMS,
                                _argv_options, expected_attack_cells,
                                load_completed_cell, point_rows, private_cell_path,
                                save_private_cell, sha)
from .a_inference import run_a_inference
from .prepare_a_endpoint_unit import (LOCK, QUEUE_STATE, SHARED_NAMES,
                                      SOURCE_NAMES, WORKTREE, _latest_a, _source)
from ..r01.role_store import RoleOnlyStore
from ..r03.compute_queue import assert_active_from_env

UNIT_ID = "R02-A-K6885-FIXED-ENDPOINT-001"
PRIVATE_OUT = CAMPAIGN / ".private/a_k6885_fixed_endpoint"
PUBLIC_JSON = CAMPAIGN / "r02/A_K6885_FIXED_INFERENCE.json"
PUBLIC_CSV = CAMPAIGN / "r02/A_FIXED_POINT_ROWS.csv"
PUBLIC_RECEIPT = CAMPAIGN / "r02/A_FIXED_ENDPOINT_RECEIPT.json"
R03_CLOSEOUT = CAMPAIGN / "r03/A_PRODUCER_FIT_CLOSEOUT.json"
R04_FIT_AUDIT = CAMPAIGN / "r04/A_FIT_16_STATUS_INDEPENDENT_AUDIT.json"
SOURCE_NAMES_FIXED = (*SOURCE_NAMES, "r02/a_fixed_endpoint.py",
                      "r03/A_PRODUCER_FIT_CLOSEOUT.json",
                      "r04/A_FIT_16_STATUS_INDEPENDENT_AUDIT.json")
FIXED_ARMS = ("A_IDENTITY", "A_CLASS_ONLY", *A_REFS)
FITTED_ARMS = tuple(arm for arm in A_ARMS if arm not in FIXED_ARMS)
NULL = "FINALIZED_NO_FEASIBLE_POLICY"
FAILED = "FAILED_STAGE1_RESOURCE"
FIT_SCHEMAS = {
    "results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.run_admitted":
        "SCV1-private-stochastic-fit-v1",
    "results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.run_admitted_controls":
        "SCV1-private-control-fit-v1",
}


def _fit_receipt_from_queued_unit(fit_id: str, unit: Mapping, plan: Mapping,
                                  producer_sha256: str) -> tuple[Path, dict, str]:
    spec = unit.get("spec", {})
    argv = spec.get("argv", [])
    options = _argv_options(argv)
    if (plan.get("kind") != "fit" or spec.get("stage") != "a" or
            spec.get("unit_id") != fit_id or spec.get("output_id") != fit_id or
            spec.get("amendment_sha256") != producer_sha256 or
            argv[2] not in FIT_SCHEMAS or
            options.get("unit-id") != fit_id or
            options.get("arm-id") != plan.get("arm_id") or
            options.get("seed") != str(plan.get("seed")) or
            options.get("a-amendment-sha256") != producer_sha256):
        raise PermissionError("FIT queue spec differs from sealed arm/seed")
    root = Path(options["private-output"])
    receipt_path = root / "FIT_RECEIPT.json"
    if (not root.is_dir() or ".private" not in root.parts or root.is_symlink() or
            receipt_path.is_symlink() or
            spec.get("outputs") != [str(receipt_path)] or
            not receipt_path.is_file() or
            any((root / name).exists() for name in
                ("POLICY_RECEIPT.json", "policy.npz", "INCOMPLETE_RESOURCE.json"))):
        raise PermissionError("FIT custody contains a selected or resource artifact")
    digest = sha(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != FIT_SCHEMAS[argv[2]] or
            receipt.get("unit_id") != fit_id or
            receipt.get("arm_id") != plan["arm_id"] or
            receipt.get("source_seed") != plan["seed"] or
            receipt.get("a_amendment_sha256") != producer_sha256):
        raise PermissionError("FIT receipt identity differs")
    return receipt_path, receipt, digest


def _failed_stage1_is_unselected(receipt: Mapping) -> bool:
    fit = receipt.get("fit", {})
    starts = fit.get("stage1_starts", [])
    return (receipt.get("status") == "INCOMPLETE_RESOURCE" and
            receipt.get("schema") == "SCV1-private-control-fit-v1" and
            fit.get("success") is False and fit.get("complete") is False and
            fit.get("stage1_complete") is True and fit.get("stage2_starts") == [] and
            type(starts) is list and len(starts) == 2 and
            {item.get("start") for item in starts} ==
            {"identity_biased", "uniform_logit"} and
            all(item.get("completed") is True and item.get("updates") == 150 and
                type(item.get("checkpoints")) is list and item["checkpoints"] and
                all(checkpoint.get("feasible") is False
                    for checkpoint in item["checkpoints"])
                for item in starts))


def classify_fixed_scope(unit_plan: Mapping, units: Mapping,
                         r03_closeout: Mapping, r04_audit: Mapping,
                         producer_sha256: str, queue_root: Path):
    """Require all 16 terminal FIT lineages and the fixed 18-instance plan."""
    cells = expected_attack_cells(unit_plan)
    fixed = {key: uid for key, uid in cells.items() if key[0] in FIXED_ARMS}
    missing = {key: uid for key, uid in cells.items() if key[0] in FITTED_ARMS}
    if (len(fixed) != 18 or len(missing) != 40 or
            {key[0] for key in fixed} != set(FIXED_ARMS) or
            {key[0] for key in missing} != set(FITTED_ARMS)):
        raise PermissionError("fixed 18/40 A cell partition differs")
    for uid in fixed.values():
        if units.get(uid, {}).get("status") != "done" or units[uid].get("returncode") != 0:
            raise PermissionError("fixed attack parent is not R03 done")
    for uid in missing.values():
        if units.get(uid, {}).get("status") not in (None, "blocked", "cancelled", "superseded"):
            raise PermissionError("fitted attack descendant unexpectedly ran or failed")
    fit_ids = {unit_plan[uid]["fit_unit_id"] for uid in missing.values()}
    if len(fit_ids) != 16 or any(unit_plan[uid].get("kind") != "fit" for uid in fit_ids):
        raise PermissionError("the 16 FIT parent IDs differ")
    r03_rows = {row["unit_id"]: row for row in r03_closeout.get("rows", [])}
    r04_rows = {row["unit_id"]: row for row in r04_audit.get("rows", [])}
    if (r03_closeout.get("schema") != "pcrl-r03-a-producer-fit-closeout-v1" or
            r04_audit.get("schema") != "pcrl-r04-a-fit-closeout-linkage-v1" or
            r03_closeout.get("effective_a_amendment_sha256") != producer_sha256 or
            r04_audit.get("producer_amendment_sha256") != producer_sha256 or
            r03_closeout.get("fit_plan_count") != 16 or
            r03_closeout.get("fit_done_rc0_count") != 14 or
            r03_closeout.get("fit_failed_rc1_count") != 2 or
            r03_closeout.get("selected_policy_count") != 0 or
            r04_audit.get("planned_fit_count") != 16 or
            r04_audit.get("complete_null_count") != 14 or
            r04_audit.get("stage1_stop_failed_count") != 2 or
            r04_audit.get("selected_policy_count") != 0 or
            len(r03_closeout.get("rows", [])) != 16 or
            len(r04_audit.get("rows", [])) != 16 or
            set(r03_rows) != fit_ids or set(r04_rows) != fit_ids):
        raise PermissionError("R03/R04 terminal FIT inventory differs")
    reports = []
    for fit_id in sorted(fit_ids):
        plan = unit_plan[fit_id]
        unit = units.get(fit_id, {})
        path, receipt, digest = _fit_receipt_from_queued_unit(
            fit_id, unit, plan, producer_sha256)
        queue_receipt = queue_root / "units" / fit_id / "RECEIPT.json"
        queue_digest = sha(queue_receipt)
        r03, r04 = r03_rows[fit_id], r04_rows[fit_id]
        if (r03.get("private_fit_receipt_sha256") != digest or
                r04.get("fit_receipt_sha256") != digest or
                r03.get("queue_receipt_sha256") != queue_digest or
                r04.get("queue_receipt_sha256") != queue_digest or
                r03.get("arm_id") != plan["arm_id"] or
                r04.get("arm_id") != plan["arm_id"] or
                r03.get("seed") != plan["seed"] or
                r04.get("seed") != plan["seed"] or
                r03.get("queue_status") != unit.get("status") or
                r04.get("queue_status") != unit.get("status") or
                r03.get("returncode") != unit.get("returncode") or
                r04.get("returncode") != unit.get("returncode") or
                r03.get("selected_policy_present") is not False):
            raise PermissionError("R03/R04 FIT custody cross-check differs")
        if plan["arm_id"] == "A_SEQ21":
            if (unit.get("status") != "failed" or unit.get("returncode") != 1 or
                    not _failed_stage1_is_unselected(receipt) or
                    r03.get("scientific_status") != "INCOMPLETE_RESOURCE" or
                    r04.get("fit_status") != "INCOMPLETE_RESOURCE"):
                raise PermissionError("SEQ21 stage1 resource failure differs")
            reason = FAILED
        else:
            selected_null = (receipt.get("selected", "unexpected") is None
                             if receipt["schema"].endswith("stochastic-fit-v1") else
                             receipt.get("fit", {}).get("success") is False and
                             receipt.get("fit", {}).get("complete") is True)
            if (unit.get("status") != "done" or unit.get("returncode") != 0 or
                    receipt.get("status") != "COMPLETE_NO_FEASIBLE_POLICY" or
                    not selected_null or
                    unit.get("output_sha256", {}).get(str(path)) != digest or
                    r03.get("scientific_status") != "COMPLETE_NO_FEASIBLE_POLICY" or
                    r04.get("fit_status") != "COMPLETE_NO_FEASIBLE_POLICY"):
                raise PermissionError("finalized no-feasible FIT differs")
            reason = NULL
        reports.append({"unit_id": fit_id, "arm_id": plan["arm_id"],
                        "seed": plan["seed"], "category": reason,
                        "fit_receipt_sha256": digest,
                        "queue_receipt_sha256": queue_digest,
                        "queue_status": unit["status"],
                        "returncode": unit["returncode"],
                        "private_fit_receipt_path": str(path)})
    if (sum(row["category"] == NULL for row in reports) != 14 or
            sum(row["category"] == FAILED for row in reports) != 2):
        raise PermissionError("A 14-null/2-failed FIT reason census differs")
    # Every planned Q/release/attack child of an unavailable FIT must be absent
    # or explicitly blocked; an independent technical failure is not a null.
    descendants = set(fit_ids)
    while True:
        more = {uid for uid, plan in unit_plan.items()
                if plan.get("fit_unit_id") in descendants or
                any(dep in descendants for dep in plan.get("deps", []))}
        if more.issubset(descendants):
            break
        descendants.update(more)
    for uid in descendants.difference(fit_ids):
        if units.get(uid, {}).get("status") not in (None, "blocked", "cancelled", "superseded"):
            raise PermissionError("unavailable FIT descendant has another outcome")
    return cells, fixed, missing, reports


def _analysis(state: Mapping, path: Path, digest: str, producer_sha: str) -> dict:
    latest = next((entry for entry in reversed(state.get("amendments", []))
                   if "verification" in entry.get("stages", [])), None)
    if (latest is None or latest.get("sha256") != digest or
            Path(latest["path"]).resolve(strict=True) != path.resolve(strict=True) or
            sha(path) != digest):
        raise PermissionError("latest verification amendment differs")
    document = json.loads(path.read_text())
    if (document.get("schema") != "pcrl-a-fixed-endpoint-analysis-amendment-v1" or
            document.get("status") != "LOCALLY_SEALED_A_FIXED_ENDPOINT" or
            document.get("producer_amendment_sha256") != producer_sha or
            document.get("authorized_unit_ids") != [UNIT_ID] or
            document.get("fixed_family_clause_count") != 6885 or
            document.get("registered_attack_instances") != 58 or
            document.get("fixed_attack_instances") != 18):
        raise PermissionError("fixed endpoint analysis amendment differs")
    return document


def build_spec(*, amendment_path: str | Path, amendment_sha256: str,
               analysis_amendment_path: str | Path, analysis_amendment_sha256: str,
               role_manifest_path: str | Path, role_manifest_sha256: str,
               allow_registered: bool = False) -> dict:
    """Data-free queue builder; failed FITs are pinned sources, never R03 deps."""
    state = json.loads(QUEUE_STATE.read_text())
    seal = state.get("seal") or {}
    if (state.get("closed") or seal.get("path") != str(LOCK) or
            sha(LOCK) != seal.get("sha256")):
        raise PermissionError("R03 effective original seal differs")
    producer_path = Path(amendment_path).resolve(strict=True)
    producer = _latest_a(state, producer_path, amendment_sha256)
    analysis_path = Path(analysis_amendment_path).resolve(strict=True)
    analysis = _analysis(state, analysis_path, analysis_amendment_sha256,
                         amendment_sha256)
    if (analysis.get("effective_seal_sha256") != seal["sha256"] or
            analysis.get("source_sha256", {}).get(
                "r03/A_PRODUCER_FIT_CLOSEOUT.json") != sha(R03_CLOSEOUT) or
            analysis.get("source_sha256", {}).get(
                "r04/A_FIT_16_STATUS_INDEPENDENT_AUDIT.json") != sha(R04_FIT_AUDIT) or
            json.loads(R04_FIT_AUDIT.read_text()).get("r03_closeout_sha256") !=
                sha(R03_CLOSEOUT)):
        raise PermissionError("fixed endpoint source lineage differs")
    if not allow_registered and UNIT_ID in state.get("units", {}):
        raise FileExistsError("fixed endpoint already registered")
    cells, fixed, missing, reports = classify_fixed_scope(
        producer["unit_plan"], state["units"],
        json.loads(R03_CLOSEOUT.read_text()), json.loads(R04_FIT_AUDIT.read_text()),
        amendment_sha256, QUEUE_STATE.parent)
    if (analysis.get("fixed_attack_unit_ids") != sorted(fixed.values()) or
            analysis.get("unavailable_attack_unit_ids") != sorted(missing.values()) or
            analysis.get("fit_reason_rows") != [
                {key: row[key] for key in ("unit_id", "arm_id", "seed", "category",
                                             "fit_receipt_sha256", "queue_receipt_sha256")}
                for row in reports]):
        raise PermissionError("dated 18/40 inventory or FIT reasons differ")
    family = CAMPAIGN / "r02/A_FAMILY_ORDER_S1.json"
    if sha(family) != producer["interval_family"]["manifest_sha256"]:
        raise PermissionError("ordered K6885 family changed")
    for name in SOURCE_NAMES_FIXED:
        if sha(CAMPAIGN / name) != analysis.get("source_sha256", {}).get(name):
            raise PermissionError(f"analysis does not pin fixed endpoint source {name}")
    for name in SHARED_NAMES:
        relative = f"results/pcrl_method_portfolio_v1/shared/{name}"
        if sha(WORKTREE / relative) != analysis.get("source_sha256", {}).get(relative):
            raise PermissionError("inherited evaluator source differs")
    role_path = Path(role_manifest_path).resolve(strict=True)
    if (sha(role_path) != role_manifest_sha256 or
            role_manifest_sha256 != json.loads(LOCK.read_text())[
                "source_files_sha256"]["source_role_manifest"] or
            analysis.get("role_manifest_sha256") != role_manifest_sha256):
        raise PermissionError("role-only manifest differs")
    role_doc = json.loads(role_path.read_text())
    role_package = role_path.parent.parent
    units = state["units"]
    deps = set(fixed.values())
    release_ids = set()
    for uid in fixed.values():
        plan = producer["unit_plan"][uid]
        deps.update(plan.get("deps", []))
        release_ids.update(dep for dep in plan.get("deps", []) if
                           producer["unit_plan"][dep].get("kind") in
                           ("reference_release", "diagnostic_release"))
    deps.update(release_ids)
    if (len(release_ids) != 36 or
            any(units.get(uid, {}).get("status") != "done" or
                units[uid].get("returncode") != 0 for uid in deps)):
        raise PermissionError("36 fixed releases or fixed attack/Q parents incomplete")
    if any(path.exists() for path in (PRIVATE_OUT, PUBLIC_JSON, PUBLIC_CSV, PUBLIC_RECEIPT)):
        raise FileExistsError("fixed endpoint output already exists")
    paths = [LOCK, producer_path, analysis_path, role_path, *
             (CAMPAIGN / name for name in SOURCE_NAMES_FIXED), *
             (WORKTREE / "results/pcrl_method_portfolio_v1/shared" / name
              for name in SHARED_NAMES)]
    for field in ("income", "occupation", "sex"):
        paths.append(role_package / role_doc["role_file_relative_path"][
            f"INNER_SELECTION:{field}"])
    for uid in sorted(deps):
        for path, digest in units[uid].get("output_sha256", {}).items():
            if sha(path) != digest:
                raise PermissionError("fixed parent output hash changed")
            paths.append(Path(path))
    for row in reports:
        paths.append(Path(row["private_fit_receipt_path"]))
        paths.append(QUEUE_STATE.parent / "units" / row["unit_id"] / "RECEIPT.json")
    for uid in fixed.values():
        root = Path(_argv_options(units[uid]["spec"]["argv"])["private-root"])
        for view in ("r1", "r2", "pair"):
            control = root / "common" / view / "controls"
            paths += [control / name for name in ("CUSTODY_RECEIPT.json",
                                                 "shuffled_model.joblib",
                                                 "planted_model.joblib",
                                                 "probe_replay.npz")]
    unique = {str(path.resolve(strict=True)): path for path in paths}
    python = Path(units[next(iter(fixed.values()))]["spec"]["argv"][0]
                  ).resolve(strict=True)
    argv = [str(python), "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r02.a_fixed_endpoint",
            "--unit-id", UNIT_ID, "--amendment", str(producer_path),
            "--amendment-sha256", amendment_sha256,
            "--analysis-amendment", str(analysis_path),
            "--analysis-amendment-sha256", analysis_amendment_sha256,
            "--role-manifest", str(role_path),
            "--role-manifest-sha256", role_manifest_sha256,
            "--private-output", str(PRIVATE_OUT),
            "--public-json", str(PUBLIC_JSON), "--public-csv", str(PUBLIC_CSV),
            "--public-receipt", str(PUBLIC_RECEIPT)]
    return {"unit_id": UNIT_ID, "stage": "verification", "mode": "real_data",
            "argv": argv, "cwd": str(WORKTREE), "deps": sorted(deps),
            "sources": [_source(unique[key]) for key in sorted(unique)],
            "outputs": [str(PRIVATE_OUT / "A_SOURCE_BINDING.json"),
                        *[str(private_cell_path(PRIVATE_OUT / "cells", *key))
                          for key in fixed],
                        str(PUBLIC_JSON), str(PUBLIC_CSV), str(PUBLIC_RECEIPT)],
            "max_wall_s": 1800, "cpu_reserve_s": 1800, "max_rss_gib": 3.0,
            "projected_output_bytes": 100000000,
            "env": {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                               "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                               "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")},
            "amendment_sha256": analysis_amendment_sha256}


def _admit(args: argparse.Namespace):
    active = assert_active_from_env(unit_id=args.unit_id, stage="verification")
    if (args.unit_id != UNIT_ID or Path(args.private_output) != PRIVATE_OUT or
            Path(args.public_json) != PUBLIC_JSON or Path(args.public_csv) != PUBLIC_CSV or
            Path(args.public_receipt) != PUBLIC_RECEIPT):
        raise PermissionError("fixed endpoint unit/output identity differs")
    parent = PRIVATE_OUT.parent
    if (not parent.resolve(strict=True).is_relative_to(CAMPAIGN.resolve(strict=True)) or
            any(path.is_symlink() for path in (parent, CAMPAIGN, *CAMPAIGN.parents)) or
            not parent.is_dir() or parent.stat().st_uid != os.getuid() or
            stat.S_IMODE(parent.stat().st_mode) & 0o077):
        raise PermissionError("fixed endpoint private parent is not owner-only")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    registered = state["units"].get(UNIT_ID, {})
    expected = build_spec(amendment_path=args.amendment,
                          amendment_sha256=args.amendment_sha256,
                          analysis_amendment_path=args.analysis_amendment,
                          analysis_amendment_sha256=args.analysis_amendment_sha256,
                          role_manifest_path=args.role_manifest,
                          role_manifest_sha256=args.role_manifest_sha256,
                          allow_registered=True)
    spec = registered.get("spec")
    if (registered.get("status") != "running" or spec != expected or
            registered.get("spec_sha256") != active["spec_sha256"] or
            spec["argv"] != [str(Path(sys.executable).resolve()), "-m",
                            "results.pcrl_latent_and_sampled_methods_v1.r02.a_fixed_endpoint",
                            *sys.argv[1:]]):
        raise PermissionError("R03 fixed endpoint admission differs")
    prefix = "results.pcrl_latent_and_sampled_methods_v1."
    for relative in SOURCE_NAMES_FIXED:
        if not relative.endswith(".py"):
            continue
        module = importlib.import_module(prefix + relative[:-3].replace("/", "."))
        if Path(module.__file__).resolve(strict=True) != (CAMPAIGN / relative).resolve(strict=True):
            raise PermissionError("fixed endpoint module origin differs")
    for name in SHARED_NAMES:
        module = importlib.import_module(name[:-3])
        if Path(module.__file__).resolve(strict=True) != (
                WORKTREE / "results/pcrl_method_portfolio_v1/shared" / name
                ).resolve(strict=True):
            raise PermissionError("inherited evaluator module origin differs")
    return state, json.loads(Path(args.amendment).read_text())


def run(args: argparse.Namespace) -> dict:
    state, producer = _admit(args)
    _, fixed, missing, reports = classify_fixed_scope(
        producer["unit_plan"], state["units"],
        json.loads(R03_CLOSEOUT.read_text()), json.loads(R04_FIT_AUDIT.read_text()),
        args.amendment_sha256, QUEUE_STATE.parent)
    store = RoleOnlyStore(args.role_manifest)
    labels = {name: store.read("INNER_SELECTION", name, procedure="screen")
              for name in ("income", "occupation", "sex")}
    first = labels["income"]
    if any(not np.array_equal(first[key], labels[field][key])
           for field in ("occupation", "sex") for key in ("row_id", "unit")):
        raise PermissionError("INNER labels differ in ordered row/group identity")
    truth = {"row_id": first["row_id"], "unit": first["unit"],
             **{name: labels[name][name] for name in labels}}
    records = {seed: {"truth": truth, "releases": {}} for seed in (0, 1)}
    provenance = []
    for (arm, seed, draw), uid in fixed.items():
        assert_active_from_env(unit_id=UNIT_ID, stage="verification")
        cell, source = load_completed_cell(uid, state["units"][uid],
                                           producer["unit_plan"][uid],
                                           queue_units=state["units"])
        if (not np.array_equal(cell["row_id"], truth["row_id"]) or
                not np.array_equal(cell["unit"], truth["unit"])):
            raise PermissionError("fixed release differs from INNER people")
        records[seed]["releases"].setdefault(arm, {})[draw] = cell
        release_parents = sorted(
            dep for dep in producer["unit_plan"][uid].get("deps", []) if
            producer["unit_plan"][dep].get("kind") in
            ("reference_release", "diagnostic_release"))
        if len(release_parents) != 2:
            raise PermissionError("fixed attack lacks ATT/INNER release pair")
        source["attack_spec_sha256"] = state["units"][uid]["spec_sha256"]
        source["release_parents"] = [
            {"unit_id": dep, "spec_sha256": state["units"][dep]["spec_sha256"],
             "output_sha256": state["units"][dep]["output_sha256"]}
            for dep in release_parents]
        provenance.append(source)
    interval = producer["interval_family"]
    if (set(interval.get("deterministic_draw_alias_arms", [])) !=
            {arm for arm in A_ARMS if arm not in STOCHASTIC_ARMS} or
            set(interval.get("deterministic_draw_alias_references", [])) != set(A_REFS) or
            not str(interval.get("deterministic_draw_alias_rule", "")).startswith(
                "draw0 complete ATT+INNER wire")):
        raise PermissionError("sealed deterministic draw alias differs")
    aliases = {arm: {"draw1": "draw0", "draw2": "draw0"}
               for arm in FIXED_ARMS}
    PRIVATE_OUT.mkdir(mode=0o700)
    cell_dir = PRIVATE_OUT / "cells"
    cell_dir.mkdir(mode=0o700)
    for source in provenance:
        arm, seed, draw = source["arm"], source["seed"], source["draw"]
        source["normalized_private_cell"] = save_private_cell(
            private_cell_path(cell_dir, arm, seed, draw),
            records[seed]["releases"][arm][draw])
    absent = [{"unit_id": uid, "arm": arm, "seed": seed, "draw": draw,
               "fit_unit_id": producer["unit_plan"][uid]["fit_unit_id"],
               "category": next(row["category"] for row in reports if
                                row["unit_id"] == producer["unit_plan"][uid]["fit_unit_id"])}
              for (arm, seed, draw), uid in missing.items()]
    binding = {"schema": "pcrl-a-k6885-private-fixed-source-binding-v1",
               "status": "INCOMPLETE_FITTED_ARM_SOURCES", "a_amendment_sha256": args.amendment_sha256,
               "analysis_amendment_sha256": args.analysis_amendment_sha256,
               "role_manifest_sha256": args.role_manifest_sha256,
               "ordered_inner_row_group_sha256": store.manifest["roles"][
                   "INNER_SELECTION"]["row_unit_pair_sha256"],
               "registered_attack_instances": 58, "fixed_cells": provenance,
               "unavailable_fitted_cells": absent, "fit_status_rows": reports,
               "deterministic_aliases": aliases,
               "original_lock_sha256": sha(LOCK),
               "fixed_family_manifest_sha256": sha(CAMPAIGN / "r02/A_FAMILY_ORDER_S1.json"),
               "r03_fit_closeout_sha256": sha(R03_CLOSEOUT),
               "r04_fit_audit_sha256": sha(R04_FIT_AUDIT)}
    binding_path = PRIVATE_OUT / "A_SOURCE_BINDING.json"
    fd = os.open(binding_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(binding, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    assert_active_from_env(unit_id=UNIT_ID, stage="verification")
    intervals = run_a_inference(records)
    if (intervals["clause_count"] != 6885 or intervals["available_clause_count"] != 1161 or
            intervals["missing_clause_count"] != 5724 or intervals["family_complete"]):
        raise AssertionError("fixed-scope K6885 availability changed")
    rows = point_rows(records)
    with PUBLIC_JSON.open("x") as stream:
        json.dump(intervals, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    with PUBLIC_CSV.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "arm", "draw", "scope",
                                                    "metric", "point", "draw_sd"))
        writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    receipt = {"schema": "pcrl-a-fixed-endpoint-aggregate-v1",
               "status": "INCOMPLETE_FITTED_ARM_SOURCES", "family_complete": False,
               "full_conjunction_status": "INCOMPLETE_NOT_ESTIMABLE",
               "registered_attack_instances": 58, "fixed_attack_instances": 18,
               "fixed_att_inner_release_instances": 36,
               "unavailable_fitted_attack_instances": 40,
               "finalized_null_fit_count": 14, "failed_stage1_resource_fit_count": 2,
               "clauses": 6885, "available_clause_count": 1161,
               "missing_clause_count": 5724,
               "missing_clause_interpretation": "not estimable; retained in the fixed family",
               "private_binding_sha256": sha(binding_path),
               "interval_json_sha256": sha(PUBLIC_JSON),
               "point_csv_sha256": sha(PUBLIC_CSV),
               "source_manifest_sha256": sha(CAMPAIGN / "r02/A_FAMILY_ORDER_S1.json")}
    with PUBLIC_RECEIPT.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("unit-id", "amendment", "amendment-sha256",
                 "analysis-amendment", "analysis-amendment-sha256", "role-manifest",
                 "role-manifest-sha256", "private-output", "public-json",
                 "public-csv", "public-receipt"):
        parser.add_argument("--" + flag, required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
