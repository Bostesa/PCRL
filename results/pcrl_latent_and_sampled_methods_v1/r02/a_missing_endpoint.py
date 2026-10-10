"""Proposal-only A endpoint for finalized null FIT descendants.

The sealed 58-cell registry and K=6,885 family stay fixed. This alternate
verification unit can be admitted only by a later dated analysis amendment.
It never treats a failed or still-running FIT as a missing scientific arm.
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
                                      SOURCE_NAMES, WORKTREE, _latest_a,
                                      _source)
from ..r01.role_store import RoleOnlyStore
from ..r03.compute_queue import assert_active_from_env

UNIT_ID = "R02-A-K6885-MISSING-ENDPOINT-001"
PRIVATE_OUT = CAMPAIGN / ".private/a_k6885_missing_endpoint"
PUBLIC_JSON = CAMPAIGN / "r02/A_K6885_MISSING_INFERENCE.json"
PUBLIC_CSV = CAMPAIGN / "r02/A_MISSING_POINT_ROWS.csv"
PUBLIC_RECEIPT = CAMPAIGN / "r02/A_MISSING_ENDPOINT_RECEIPT.json"
SOURCE_NAMES_MISSING = (*SOURCE_NAMES, "r02/a_missing_endpoint.py")
NULL_STATUS = "COMPLETE_NO_FEASIBLE_POLICY"
FIT_SCHEMAS = {
    "results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.run_admitted":
        "SCV1-private-stochastic-fit-v1",
    "results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.run_admitted_controls":
        "SCV1-private-control-fit-v1",
}


def classify_null_descendants(unit_plan: Mapping, units: Mapping,
                              null_receipts: list[dict], producer_sha256: str):
    """Partition the fixed 58 attacks using only finalized, hash-pinned FIT nulls."""
    cells = expected_attack_cells(unit_plan)
    if type(null_receipts) is not list or not null_receipts:
        raise PermissionError("dated analysis must list at least one finalized null FIT")
    null_fits = {}
    for entry in null_receipts:
        if type(entry) is not dict or set(entry) != {
                "unit_id", "arm_id", "seed", "receipt_sha256"}:
            raise PermissionError("null FIT declaration schema differs")
        fit_id = entry["unit_id"]
        if fit_id in null_fits or fit_id not in unit_plan:
            raise PermissionError("null FIT identity absent or duplicated")
        plan = unit_plan[fit_id]
        unit = units.get(fit_id, {})
        receipt_outputs = [(Path(path), digest)
                           for path, digest in unit.get("output_sha256", {}).items()
                           if Path(path).name == "FIT_RECEIPT.json"]
        if len(receipt_outputs) != 1:
            raise PermissionError("null FIT has no unique queued receipt output")
        path, queued_digest = receipt_outputs[0]
        spec = unit.get("spec", {})
        argv = spec.get("argv", [])
        options = _argv_options(argv)
        if (plan.get("kind") != "fit" or plan.get("arm_id") != entry["arm_id"] or
                plan.get("seed") != entry["seed"] or
                entry["arm_id"] not in A_ARMS or
                unit.get("status") != "done" or unit.get("returncode") != 0 or
                spec.get("unit_id") != fit_id or spec.get("stage") != "a" or
                spec.get("outputs") != [str(path)] or
                spec.get("amendment_sha256") != producer_sha256 or
                argv[2] not in FIT_SCHEMAS or
                options.get("unit-id") != fit_id or
                options.get("arm-id") != entry["arm_id"] or
                options.get("seed") != str(entry["seed"]) or
                options.get("a-amendment-sha256") != producer_sha256 or
                options.get("private-output") != str(path.parent) or
                queued_digest != entry["receipt_sha256"] or
                sha(path) != entry["receipt_sha256"]):
            raise PermissionError("null FIT is not a completed exact queued output")
        receipt = json.loads(path.read_text())
        complete_without_selection = (receipt.get("selected", "unexpected") is None
                                      if argv[2].endswith(".impl.run_admitted") else
                                      receipt.get("fit", {}).get("success") is False and
                                      receipt.get("fit", {}).get("complete") is True)
        if (receipt.get("schema") != FIT_SCHEMAS[argv[2]] or
                receipt.get("status") != NULL_STATUS or not complete_without_selection or
                receipt.get("unit_id") != fit_id or
                receipt.get("arm_id") != entry["arm_id"] or
                receipt.get("source_seed") != entry["seed"] or
                receipt.get("a_amendment_sha256") != producer_sha256 or
                any((path.parent / name).exists() for name in
                    ("INCOMPLETE_RESOURCE.json", "POLICY_RECEIPT.json"))):
            raise PermissionError("FIT receipt does not certify no feasible policy")
        null_fits[fit_id] = entry
    descendants = set(null_fits)
    changed = True
    while changed:
        before = len(descendants)
        descendants.update(uid for uid, plan in unit_plan.items()
                           if uid not in descendants and
                           (plan.get("fit_unit_id") in descendants or
                            any(dep in descendants for dep in plan.get("deps", []))))
        changed = len(descendants) != before
    for uid in descendants.difference(null_fits):
        if units.get(uid, {}).get("status") not in (None, "blocked", "cancelled", "superseded"):
            raise PermissionError("null FIT descendant has a technical failure or execution")
    missing = {key: uid for key, uid in cells.items()
               if unit_plan[uid].get("fit_unit_id") in null_fits}
    if not missing:
        raise PermissionError("declared null FIT has no registered descendants")
    for fit_id, entry in null_fits.items():
        arm_seed = (entry["arm_id"], entry["seed"])
        actual = {key[2] for key in missing if key[:2] == arm_seed}
        expected = set(range(3)) if arm_seed[0] in STOCHASTIC_ARMS else {0}
        if actual != expected:
            raise PermissionError("null FIT must remove its entire arm and source seed")
    for key, uid in cells.items():
        unit = units.get(uid, {})
        if key in missing:
            if unit.get("status") in ("done", "running"):
                raise PermissionError("null FIT descendant has a completed or running attack")
        elif unit.get("status") != "done" or unit.get("returncode") != 0:
            raise PermissionError("non-null attack remains pending or failed")
    available = {key: uid for key, uid in cells.items() if key not in missing}
    return cells, available, missing, null_fits


def _analysis(state: Mapping, path: Path, digest: str, producer_sha: str) -> dict:
    latest = next((entry for entry in reversed(state.get("amendments", []))
                   if "verification" in entry.get("stages", [])), None)
    if (latest is None or latest.get("sha256") != digest or
            Path(latest["path"]).resolve(strict=True) != path.resolve(strict=True) or
            sha(path) != digest):
        raise PermissionError("latest verification amendment differs")
    document = json.loads(path.read_text())
    if (document.get("schema") != "pcrl-a-missing-endpoint-analysis-amendment-v1" or
            document.get("status") != "LOCALLY_SEALED_A_MISSING_ENDPOINT" or
            document.get("producer_amendment_sha256") != producer_sha or
            document.get("authorized_unit_ids") != [UNIT_ID]):
        raise PermissionError("missing endpoint amendment identity differs")
    return document


def build_spec(*, amendment_path: str | Path, amendment_sha256: str,
               analysis_amendment_path: str | Path, analysis_amendment_sha256: str,
               role_manifest_path: str | Path, role_manifest_sha256: str,
               allow_registered: bool = False) -> dict:
    """Data-free queue spec: only completed attacks plus certified null FITs are deps."""
    state = json.loads(QUEUE_STATE.read_text())
    seal = state.get("seal") or {}
    if (state.get("closed") or seal.get("path") != str(LOCK) or
            sha(LOCK) != seal.get("sha256")):
        raise PermissionError("R03 effective seal differs")
    amendment_path = Path(amendment_path).resolve(strict=True)
    amendment = _latest_a(state, amendment_path, amendment_sha256)
    analysis_path = Path(analysis_amendment_path).resolve(strict=True)
    analysis = _analysis(state, analysis_path, analysis_amendment_sha256,
                         amendment_sha256)
    if analysis.get("effective_seal_sha256") != seal["sha256"]:
        raise PermissionError("analysis original seal differs")
    if not allow_registered and UNIT_ID in state.get("units", {}):
        raise FileExistsError("missing endpoint unit already registered")
    units = state.get("units", {})
    cells, available, missing, null_fits = classify_null_descendants(
        amendment["unit_plan"], units, analysis.get("null_fit_receipts"),
        amendment_sha256)
    if (analysis.get("missing_attack_unit_ids") != sorted(missing.values()) or
            analysis.get("registered_attack_unit_count") != 58 or
            analysis.get("fixed_family_clause_count") != 6885):
        raise PermissionError("dated missing-cell inventory differs from 58-cell family")
    family = CAMPAIGN / "r02/A_FAMILY_ORDER_S1.json"
    if sha(family) != amendment["interval_family"]["manifest_sha256"]:
        raise PermissionError("K6885 ordered family changed")
    for name in SOURCE_NAMES_MISSING:
        if sha(CAMPAIGN / name) != analysis.get("source_sha256", {}).get(name):
            raise PermissionError(f"missing endpoint source pin differs: {name}")
    for name in SHARED_NAMES:
        relative = f"results/pcrl_method_portfolio_v1/shared/{name}"
        if sha(WORKTREE / relative) != analysis.get("source_sha256", {}).get(relative):
            raise PermissionError("inherited evaluator source pin differs")
    if (analysis["source_sha256"].get("r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json") !=
            amendment.get("source_sha256", {}).get(
                "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json") or
            any(analysis["source_sha256"].get(f"r01/A_PUBLIC_FINE_CENTROIDS_S{s}.npz") !=
                amendment.get("attack_plan", {}).get(
                    "public_centroid_npz_sha256_by_seed", {}).get(f"s{s}")
                for s in (0, 1))):
        raise PermissionError("producer source pins differ")
    role_path = Path(role_manifest_path).resolve(strict=True)
    if (sha(role_path) != role_manifest_sha256 or
            role_manifest_sha256 != json.loads(LOCK.read_text())[
                "source_files_sha256"]["source_role_manifest"] or
            analysis.get("role_manifest_sha256") != role_manifest_sha256):
        raise PermissionError("role-only manifest pin differs")
    role_doc = json.loads(role_path.read_text())
    role_package = role_path.parent.parent
    parent_ids = set(null_fits)
    for uid in available.values():
        plan = amendment["unit_plan"][uid]
        parent_ids.add(uid)
        parent_ids.update(plan.get("deps", []))
        parent_ids.update(plan.get("release_unit_ids", {}).values())
    if any(units.get(uid, {}).get("status") != "done" or
           units[uid].get("returncode") != 0 for uid in parent_ids):
        raise PermissionError("available attack/source parent is incomplete")
    if any(path.exists() for path in (PRIVATE_OUT, PUBLIC_JSON, PUBLIC_CSV, PUBLIC_RECEIPT)):
        raise FileExistsError("missing endpoint output already exists")
    paths = [LOCK, amendment_path, analysis_path, role_path, *
             (CAMPAIGN / name for name in SOURCE_NAMES_MISSING), *
             (WORKTREE / "results/pcrl_method_portfolio_v1/shared" / name
              for name in SHARED_NAMES)]
    for field in ("income", "occupation", "sex"):
        paths.append(role_package / role_doc["role_file_relative_path"][
            f"INNER_SELECTION:{field}"])
    for uid in sorted(parent_ids):
        for path, digest in units[uid].get("output_sha256", {}).items():
            if sha(path) != digest:
                raise PermissionError("parent output hash changed")
            paths.append(Path(path))
    for uid in available.values():
        root = Path(_argv_options(units[uid]["spec"]["argv"])["private-root"])
        for view in ("r1", "r2", "pair"):
            control = root / "common" / view / "controls"
            paths += [control / name for name in ("CUSTODY_RECEIPT.json",
                                                 "shuffled_model.joblib",
                                                 "planted_model.joblib",
                                                 "probe_replay.npz")]
    unique = {str(path.resolve(strict=True)): path for path in paths}
    python = Path(units[next(iter(available.values()))]["spec"]["argv"][0]
                  ).resolve(strict=True)
    argv = [str(python), "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r02.a_missing_endpoint",
            "--unit-id", UNIT_ID, "--amendment", str(amendment_path),
            "--amendment-sha256", amendment_sha256,
            "--analysis-amendment", str(analysis_path),
            "--analysis-amendment-sha256", analysis_amendment_sha256,
            "--role-manifest", str(role_path),
            "--role-manifest-sha256", role_manifest_sha256,
            "--private-output", str(PRIVATE_OUT),
            "--public-json", str(PUBLIC_JSON), "--public-csv", str(PUBLIC_CSV),
            "--public-receipt", str(PUBLIC_RECEIPT)]
    return {"unit_id": UNIT_ID, "stage": "verification", "mode": "real_data",
            "argv": argv, "cwd": str(WORKTREE), "deps": sorted(parent_ids),
            "sources": [_source(unique[key]) for key in sorted(unique)],
            "outputs": [str(PRIVATE_OUT / "A_SOURCE_BINDING.json"),
                        *[str(private_cell_path(PRIVATE_OUT / "cells", *key))
                          for key in available],
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
        raise PermissionError("missing endpoint unit/output identity differs")
    parent = PRIVATE_OUT.parent
    if (not parent.resolve(strict=True).is_relative_to(CAMPAIGN.resolve(strict=True)) or
            any(path.is_symlink() for path in (parent, CAMPAIGN, *CAMPAIGN.parents)) or
            not parent.is_dir() or parent.stat().st_uid != os.getuid() or
            stat.S_IMODE(parent.stat().st_mode) & 0o077):
        raise PermissionError("missing endpoint private parent is not owner-only")
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
                            "results.pcrl_latent_and_sampled_methods_v1.r02.a_missing_endpoint",
                            *sys.argv[1:]]):
        raise PermissionError("R03 missing endpoint admission differs")
    prefix = "results.pcrl_latent_and_sampled_methods_v1."
    for relative in SOURCE_NAMES_MISSING:
        if not relative.endswith(".py"):
            continue
        module = importlib.import_module(prefix + relative[:-3].replace("/", "."))
        if Path(module.__file__).resolve(strict=True) != (CAMPAIGN / relative).resolve(strict=True):
            raise PermissionError("missing endpoint module origin differs")
    for name in SHARED_NAMES:
        module = importlib.import_module(name[:-3])
        if Path(module.__file__).resolve(strict=True) != (
                WORKTREE / "results/pcrl_method_portfolio_v1/shared" / name
                ).resolve(strict=True):
            raise PermissionError("inherited evaluator module origin differs")
    return state, json.loads(Path(args.amendment).read_text()), expected


def run(args: argparse.Namespace) -> dict:
    state, amendment, spec = _admit(args)
    analysis = json.loads(Path(args.analysis_amendment).read_text())
    cells, available, missing, null_fits = classify_null_descendants(
        amendment["unit_plan"], state["units"], analysis["null_fit_receipts"],
        args.amendment_sha256)
    store = RoleOnlyStore(args.role_manifest)
    labels = {name: store.read("INNER_SELECTION", name, procedure="screen")
              for name in ("income", "occupation", "sex")}
    first = labels["income"]
    if any(not np.array_equal(first[key], labels[field][key])
           for field in ("occupation", "sex") for key in ("row_id", "unit")):
        raise PermissionError("INNER labels differ in row/group order")
    truth = {"row_id": first["row_id"], "unit": first["unit"],
             **{name: labels[name][name] for name in labels}}
    records = {seed: {"truth": truth, "releases": {}} for seed in (0, 1)}
    provenance = []
    for (arm, seed, draw), uid in available.items():
        assert_active_from_env(unit_id=UNIT_ID, stage="verification")
        cell, source = load_completed_cell(uid, state["units"][uid],
                                           amendment["unit_plan"][uid],
                                           queue_units=state["units"])
        if (not np.array_equal(cell["row_id"], truth["row_id"]) or
                not np.array_equal(cell["unit"], truth["unit"])):
            raise PermissionError("available release differs from INNER people")
        records[seed]["releases"].setdefault(arm, {})[draw] = cell
        provenance.append(source)
    if len(provenance) != len(available) or len(cells) != 58:
        raise AssertionError("fixed A source inventory changed")
    interval = amendment["interval_family"]
    if (set(interval.get("deterministic_draw_alias_arms", [])) !=
            {arm for arm in A_ARMS if arm not in STOCHASTIC_ARMS} or
            set(interval.get("deterministic_draw_alias_references", [])) != set(A_REFS) or
            not str(interval.get("deterministic_draw_alias_rule", "")).startswith(
                "draw0 complete ATT+INNER wire")):
        raise PermissionError("sealed draw0 aliases differ")
    aliases = {arm: {"draw1": "draw0", "draw2": "draw0"}
               for arm in (*A_ARMS, *A_REFS) if arm not in STOCHASTIC_ARMS}
    PRIVATE_OUT.mkdir(mode=0o700)
    cell_dir = PRIVATE_OUT / "cells"
    cell_dir.mkdir(mode=0o700)
    for source in provenance:
        arm, seed, draw = source["arm"], source["seed"], source["draw"]
        source["normalized_private_cell"] = save_private_cell(
            private_cell_path(cell_dir, arm, seed, draw),
            records[seed]["releases"][arm][draw])
    missing_rows = [{"arm": arm, "seed": seed, "draw": draw, "unit_id": uid,
                     "fit_unit_id": amendment["unit_plan"][uid]["fit_unit_id"]}
                    for (arm, seed, draw), uid in missing.items()]
    binding = {"schema": "pcrl-a-k6885-private-missing-source-binding-v1",
               "status": "INCOMPLETE_FINALIZED_NULL_FIT", "a_amendment_sha256": args.amendment_sha256,
               "analysis_amendment_sha256": args.analysis_amendment_sha256,
               "role_manifest_sha256": args.role_manifest_sha256,
               "ordered_inner_row_group_sha256": store.manifest["roles"][
                   "INNER_SELECTION"]["row_unit_pair_sha256"],
               "deterministic_aliases": aliases,
               "registered_attack_instances": 58, "available_cells": provenance,
               "missing_cells": missing_rows, "null_fit_receipts": list(null_fits.values())}
    binding_path = PRIVATE_OUT / "A_SOURCE_BINDING.json"
    fd = os.open(binding_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(binding, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    assert_active_from_env(unit_id=UNIT_ID, stage="verification")
    intervals = run_a_inference(records)
    if intervals["family_complete"] or intervals["missing_clause_count"] == 0:
        raise AssertionError("null descendant cannot yield complete K6885 family")
    rows = point_rows(records)
    for path, contents in ((PUBLIC_JSON, intervals),):
        with path.open("x") as stream:
            json.dump(contents, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
    with PUBLIC_CSV.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "arm", "draw", "scope",
                                                    "metric", "point", "draw_sd"))
        writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    receipt = {"schema": "pcrl-a-missing-endpoint-aggregate-v1",
               "status": "INCOMPLETE_FINALIZED_NULL_FIT", "family_complete": False,
               "full_conjunction_status": "INCOMPLETE", "clauses": 6885,
               "registered_attack_instances": 58,
               "available_attack_instances": len(available),
               "missing_attack_instances": len(missing),
               "missing_arms_by_seed": intervals["missing_arms_by_seed"],
               "missing_clause_count": intervals["missing_clause_count"],
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
