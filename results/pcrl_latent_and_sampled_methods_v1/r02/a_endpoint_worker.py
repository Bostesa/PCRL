"""Queue-gated final A endpoint: saved complete views, points, K=6,885 bounds."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np

from .a_endpoint_driver import (A_ARMS, A_REFS, CAMPAIGN, STOCHASTIC_ARMS,
                                expected_attack_cells, load_completed_cell,
                                private_cell_path, save_private_cell, sha,
                                write_aggregate)
from .prepare_a_endpoint_unit import (PRIVATE_OUT, PUBLIC_CSV, PUBLIC_JSON,
                                      PUBLIC_RECEIPT, SOURCE_NAMES, SHARED_NAMES,
                                      UNIT_ID, WORKTREE, build_spec)
from ..r01.role_store import RoleOnlyStore
from ..r03.compute_queue import assert_active_from_env


def _assert_module_origins() -> None:
    """Tie imported replay code to the exact files in the analysis source map."""
    prefix = "results.pcrl_latent_and_sampled_methods_v1."
    for relative in SOURCE_NAMES:
        if not relative.endswith(".py"):
            continue
        module = importlib.import_module(prefix + relative[:-3].replace("/", "."))
        expected = (CAMPAIGN / relative).resolve(strict=True)
        if Path(module.__file__).resolve(strict=True) != expected:
            raise PermissionError("A endpoint replay module origin differs")
    for name in SHARED_NAMES:
        module = importlib.import_module(name[:-3])
        expected = (WORKTREE / "results/pcrl_method_portfolio_v1/shared" / name
                    ).resolve(strict=True)
        if Path(module.__file__).resolve(strict=True) != expected:
            raise PermissionError("A endpoint inherited evaluator origin differs")


def _admit(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    active = assert_active_from_env(unit_id=args.unit_id, stage="verification")
    if (args.unit_id != UNIT_ID or Path(args.private_output) != PRIVATE_OUT or
            Path(args.public_json) != PUBLIC_JSON or
            Path(args.public_csv) != PUBLIC_CSV or
            Path(args.public_receipt) != PUBLIC_RECEIPT or
            any(path.exists() for path in (PRIVATE_OUT, PUBLIC_JSON, PUBLIC_CSV,
                                           PUBLIC_RECEIPT))):
        raise PermissionError("A endpoint output or unit identity differs")
    parent = PRIVATE_OUT.parent
    campaign = CAMPAIGN.resolve(strict=True)
    if (not parent.resolve(strict=True).is_relative_to(campaign) or
            any(path.is_symlink() for path in (parent, CAMPAIGN, *CAMPAIGN.parents)) or
            not parent.is_dir() or parent.stat().st_uid != os.getuid() or
            stat.S_IMODE(parent.stat().st_mode) & 0o077):
        raise PermissionError("A endpoint private parent must be owner-only")
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
                            "results.pcrl_latent_and_sampled_methods_v1.r02.a_endpoint_worker",
                            *sys.argv[1:]]):
        raise PermissionError("running R03 endpoint spec differs from exact builder")
    amendment = json.loads(Path(args.amendment).read_text())
    if (sha(args.amendment) != args.amendment_sha256 or
            amendment.get("interval_family", {}).get("ordered_clause_count") != 6885):
        raise PermissionError("full A amendment source differs")
    if spec["amendment_sha256"] != args.analysis_amendment_sha256:
        raise PermissionError("analysis stage source amendment differs")
    _assert_module_origins()
    return state, amendment, active


def run(args: argparse.Namespace) -> dict:
    state, amendment, active = _admit(args)
    cells = expected_attack_cells(amendment["unit_plan"])
    # Only after the active slot, exact spec, all parents and A seal are checked
    # may the worker open the three INNER role-only label archives.
    store = RoleOnlyStore(args.role_manifest)
    labels = {name: store.read("INNER_SELECTION", name, procedure="screen")
              for name in ("income", "occupation", "sex")}
    first = labels["income"]
    if any(not np.array_equal(first[key], labels[field][key])
           for field in ("occupation", "sex") for key in ("row_id", "unit")):
        raise PermissionError("INNER role labels differ in ordered row/group identity")
    truth = {"row_id": first["row_id"], "unit": first["unit"],
             **{name: labels[name][name] for name in labels}}
    seed_records = {seed: {"truth": truth, "releases": {}}
                    for seed in (0, 1)}
    provenance = []
    for (arm, seed, draw), unit_id in cells.items():
        assert_active_from_env(unit_id=UNIT_ID, stage="verification")
        cell, source = load_completed_cell(
            unit_id, state["units"][unit_id], amendment["unit_plan"][unit_id],
            queue_units=state["units"])
        if (not np.array_equal(cell["row_id"], truth["row_id"]) or
                not np.array_equal(cell["unit"], truth["unit"])):
            raise PermissionError("A saved release INNER rows/groups differ from admitted labels")
        seed_records[seed]["releases"].setdefault(arm, {})[draw] = cell
        provenance.append(source)
    if len(provenance) != 58:
        raise AssertionError("A normalized source inventory changed")
    aliases = {arm: {"draw1": "draw0", "draw2": "draw0"}
               for arm in (*A_ARMS, *A_REFS) if arm not in STOCHASTIC_ARMS}
    interval = amendment["interval_family"]
    if (set(interval.get("deterministic_draw_alias_arms", [])) !=
            {arm for arm in A_ARMS if arm not in STOCHASTIC_ARMS} or
            set(interval.get("deterministic_draw_alias_references", [])) != set(A_REFS) or
            not str(interval.get("deterministic_draw_alias_rule", "")).startswith(
                "draw0 complete ATT+INNER wire")):
        raise PermissionError("deterministic draw0 alias rule differs from sealed A amendment")
    # The scorer receives one exact saved draw0 object for each deterministic
    # arm/reference. d1/d2 are logical aliases only; no model or output copy.
    PRIVATE_OUT.mkdir(mode=0o700)
    cell_dir = PRIVATE_OUT / "cells"
    cell_dir.mkdir(mode=0o700)
    for source in provenance:
        arm, seed, draw = source["arm"], source["seed"], source["draw"]
        path = private_cell_path(cell_dir, arm, seed, draw)
        source["normalized_private_cell"] = save_private_cell(
            path, seed_records[seed]["releases"][arm][draw])
    binding = {"schema": "pcrl-a-k6885-private-source-binding-v1",
               "status": "ALL_58_COMPLETE_VIEWS_VERIFIED",
               "a_amendment_sha256": args.amendment_sha256,
               "analysis_amendment_sha256": args.analysis_amendment_sha256,
               "role_manifest_sha256": args.role_manifest_sha256,
               "ordered_inner_row_group_sha256":
                   store.manifest["roles"]["INNER_SELECTION"]["row_unit_pair_sha256"],
               "deterministic_aliases": aliases, "cells": provenance}
    binding_path = PRIVATE_OUT / "A_SOURCE_BINDING.json"
    with binding_path.open("x") as stream:
        json.dump(binding, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    binding_path.chmod(0o600)
    assert_active_from_env(unit_id=UNIT_ID, stage="verification")
    return write_aggregate(seed_records, args.public_json, args.public_csv,
                           args.public_receipt, binding_sha256=sha(binding_path),
                           source_manifest_sha256=sha(CAMPAIGN / "r02/A_FAMILY_ORDER_S1.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("unit-id", "amendment", "amendment-sha256",
                 "analysis-amendment", "analysis-amendment-sha256", "role-manifest",
                 "role-manifest-sha256", "private-output", "public-json",
                 "public-csv", "public-receipt"):
        parser.add_argument("--" + flag, required=True)
    result = run(parser.parse_args())
    print(json.dumps({"status": result["status"],
                      "clause_count": result["clauses"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
