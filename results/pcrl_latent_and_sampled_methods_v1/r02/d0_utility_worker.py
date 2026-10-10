"""R03 verification worker for one historical D0 INNER utility cell.

The worker reads only the admitted role-only INNER score token, fixed public
score banks, INNER task labels and U0/Ucal reference arrays. It fits nothing,
opens no assessment archive, and publishes aggregates only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from ..r01.role_store import RoleOnlyStore, assert_same_people, sha_array, sha_file
from ..r03.compute_queue import assert_active_from_env
from .d0_utility import evaluate_inner_utility
from .prepare_d0_utility_queue_units import CAMPAIGN, WORKTREE, source_inventory


KEYS = {"LOCAL": "hcal_local_d0", "SEQ12": "hcal_seq12_d0",
        "SEQ21": "lra_seq21"}
STAGE = "verification"
DEPENDENCY = "R04-D-INTERVALS-001"


def _admit(args: argparse.Namespace, active: dict, *, role_manifest: Path,
           audit: Path, output: Path) -> dict:
    """Check the exact running spec and all current source bytes before reads."""
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    unit = state.get("units", {}).get(args.unit_id, {})
    spec = unit.get("spec", {})
    expected_argv = [str(Path(sys.executable).resolve()), "-m",
                     "results.pcrl_latent_and_sampled_methods_v1.r02.d0_utility_worker",
                     *sys.argv[1:]]
    actual_argv = list(spec.get("argv", []))
    if actual_argv:
        actual_argv[0] = str(Path(actual_argv[0]).resolve())
    if (unit.get("status") != "running" or unit.get("spec_sha256") != active["spec_sha256"]
            or spec.get("unit_id") != args.unit_id or spec.get("stage") != STAGE
            or actual_argv != expected_argv or spec.get("cwd") != str(WORKTREE)
            or spec.get("deps") != [DEPENDENCY]
            or state.get("units", {}).get(DEPENDENCY, {}).get("status") != "done"
            or spec.get("outputs") != [str(output)]
            or float(spec.get("max_wall_s", -1)) != 180
            or float(spec.get("cpu_reserve_s", -1)) != 180
            or float(spec.get("max_rss_gib", -1)) != 1.0
            or int(spec.get("projected_output_bytes", -1)) != 100000):
        raise PermissionError("D0 utility R03 verification spec differs")
    sources = source_inventory(role_manifest=role_manifest, audit=audit,
                               key=KEYS[args.arm], seed=args.seed,
                               seal=Path(args.seal))
    if spec.get("sources") != sources:
        raise PermissionError("D0 utility R03 source inventory differs")
    return state


def run(args: argparse.Namespace) -> dict:
    if (args.stage != STAGE or args.arm not in KEYS or args.seed not in (0, 1)
            or args.unit_id != f"R02-D0-UTILITY-{args.arm}-S{args.seed}"):
        raise PermissionError("unregistered historical D0 utility cell")
    active = assert_active_from_env(unit_id=args.unit_id, stage=STAGE)
    if active["seal_sha256"] != args.seal_sha256:
        raise PermissionError("queue and worker effective seal differ")
    seal = Path(args.seal).resolve(strict=True)
    if sha_file(seal) != args.seal_sha256 or seal != CAMPAIGN / "CAMPAIGN_LOCK.json":
        raise PermissionError("D0 utility effective seal differs")
    lock = json.loads(seal.read_text())
    if (lock.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1"
            or lock.get("status") != "LOCALLY_SEALED_D_STAGE"):
        raise PermissionError("D0 utility source has no effective seal")
    role_manifest = Path(args.role_manifest).resolve(strict=True)
    if (sha_file(role_manifest) != args.role_manifest_sha256
            or lock["source_files_sha256"]["source_role_manifest"] != args.role_manifest_sha256):
        raise PermissionError("role-only manifest differs from effective seal")
    audit = Path(args.audit).resolve(strict=True)
    if (audit != CAMPAIGN / "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json"
            or sha_file(audit) != args.audit_sha256):
        raise PermissionError("R01 D0 score-source audit bytes differ")
    output = Path(args.output).resolve(strict=False)
    if output != CAMPAIGN / "r02" / f"{args.unit_id}_RECEIPT.json":
        raise PermissionError("D0 utility public output path differs")
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("D0 utility receipt must be a new file")
    _admit(args, active, role_manifest=role_manifest, audit=audit, output=output)

    def guard() -> None:
        current = assert_active_from_env(unit_id=args.unit_id, stage=STAGE)
        if current["seal_sha256"] != args.seal_sha256:
            raise PermissionError("D0 utility slot seal changed")

    guard()
    source_audit = json.loads(audit.read_text())
    cell = source_audit["d0"][KEYS[args.arm]][f"s{args.seed}"]
    from ..r01.d0_source import D0SourceStore

    portfolio = Path(cell["receipt_path"]).parents[3]
    cache = Path(cell["source_paths_sha256"]["model.pt"]["path"]).parents[3]
    source = D0SourceStore(portfolio, role_manifest, audit, cache,
                           KEYS[args.arm], args.seed)
    wire1 = source.read_role_wire("INNER_SELECTION", 1)
    wire2 = source.read_role_wire("INNER_SELECTION", 2)
    store = RoleOnlyStore(role_manifest)
    field = f"references_s{args.seed}"
    refs = store.read("INNER_SELECTION", field)
    income = store.read("INNER_SELECTION", "income", procedure="screen")
    occupation = store.read("INNER_SELECTION", "occupation", procedure="screen")
    assert_same_people(wire1, wire2, refs, income, occupation)
    if (not np.array_equal(wire1["decision"], np.argmax(wire1["probabilities"], axis=1))
            or not np.array_equal(wire2["decision"], np.argmax(wire2["probabilities"], axis=1))):
        raise ValueError("D0 actual public decision is not strict argmax")
    release = {"q1": wire1["probabilities"], "q2": wire2["probabilities"],
               "d1": wire1["decision"], "d2": wire2["decision"]}
    references = {kind: {"q1": refs[f"{kind}_1"], "q2": refs[f"{kind}_2"]}
                  for kind in ("U0", "Ucal")}
    labels = {"income": income["income"], "occupation": occupation["occupation"]}
    utility = evaluate_inner_utility(release, references, labels)
    guard()
    role_doc = store.manifest
    public = {
        "schema": "pcrl-r02-d0-inner-utility-receipt-v1", "status": "COMPLETE",
        "unit_id": args.unit_id, "stage": STAGE, "arm": args.arm,
        "release_key": KEYS[args.arm], "release_id": cell["release_id"],
        "seed": args.seed, "role": "INNER_SELECTION",
        "effective_seal_sha256": args.seal_sha256,
        "r01_d0_source_audit_sha256": args.audit_sha256,
        "role_manifest_sha256": args.role_manifest_sha256,
        "source_binding_sha256": cell["source_binding_sha256"],
        "source_sha256": {
            "receipt": cell["receipt_sha256"],
            "bank_r1": cell["banks"]["r1"]["sha256"],
            "bank_r2": cell["banks"]["r2"]["sha256"],
            "tokens_inner": cell["roles"]["INNER_SELECTION"]["sha256"],
            "references_inner": role_doc["role_file_sha256"][f"INNER_SELECTION:{field}"],
            "income_inner": role_doc["role_file_sha256"]["INNER_SELECTION:income"],
            "occupation_inner": role_doc["role_file_sha256"]["INNER_SELECTION:occupation"],
        },
        "ordered_row_unit_pair_sha256": sha_array(np.column_stack((refs["row_id"], refs["unit"]))),
        "probability_array_sha256": {"r1": sha_array(release["q1"]),
                                     "r2": sha_array(release["q2"])},
        "decision_array_sha256": {"r1": sha_array(release["d1"]),
                                  "r2": sha_array(release["d2"])},
        "utility": utility,
        "gate": "each task versus both U0/Ucal, accuracy delta >= -0.010, NLL delta <= +0.010, Brier delta <= +0.005; each seed reported separately",
        "no_attacker_fit": True,
        "no_assessment_access": True,
    }
    with output.open("x") as stream:
        json.dump(public, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"unit_id": args.unit_id, "output_sha256": sha_file(output),
            "point_eligible": utility["point_eligible_all_tasks_refs"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seal", required=True)
    parser.add_argument("--seal-sha256", required=True)
    parser.add_argument("--role-manifest", required=True)
    parser.add_argument("--role-manifest-sha256", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
