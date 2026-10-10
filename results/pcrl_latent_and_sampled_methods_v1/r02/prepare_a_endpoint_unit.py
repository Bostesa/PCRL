"""Data-free R03 spec builder for the final A K=6,885 endpoint worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .a_endpoint_driver import (CAMPAIGN, expected_attack_cells,
                                private_cell_path, sha)

WORKTREE = CAMPAIGN.parents[1]
QUEUE_STATE = CAMPAIGN / "r03/.private/compute/state.json"
LOCK = CAMPAIGN / "CAMPAIGN_LOCK.json"
UNIT_ID = "R02-A-K6885-ENDPOINT-001"
PRIVATE_OUT = CAMPAIGN / ".private/a_k6885_endpoint"
PUBLIC_JSON = CAMPAIGN / "r02/A_K6885_INFERENCE.json"
PUBLIC_CSV = CAMPAIGN / "r02/A_POINT_ROWS.csv"
PUBLIC_RECEIPT = CAMPAIGN / "r02/A_ENDPOINT_RECEIPT.json"
SOURCE_NAMES = (
    "r02/a_endpoint_driver.py", "r02/a_endpoint_worker.py",
    "r02/prepare_a_endpoint_unit.py", "r02/a_inference.py",
    "r02/A_FAMILY_ORDER_S1.json", "r02/complete_view.py",
    "r01/role_store.py", "r03/compute_queue.py",
    "sampled_channel_v1/attacks/runner.py",
    "sampled_channel_v1/attacks/reference_wire.py",
    "sampled_channel_v1/attacks/run_reference_attack.py",
    "sampled_channel_v1/attacks/q_artifact.py",
    "sampled_channel_v1/attacks/q_aware.py",
    "sampled_channel_v1/attacks/wire.py",
)
SHARED_NAMES = ("attack_eval.py", "common_eval.py")


def _source(path: Path) -> dict:
    target = path.resolve(strict=True)
    if not target.is_file():
        raise FileNotFoundError("A endpoint source must be a file")
    return {"path": str(target), "sha256": sha(target)}


def _latest_a(state: dict, amendment_path: Path, amendment_sha256: str) -> dict:
    registered = next((entry for entry in reversed(state.get("amendments", []))
                       if "a" in entry.get("stages", [])), None)
    if (registered is None or registered.get("sha256") != amendment_sha256 or
            Path(registered["path"]).resolve(strict=True) != amendment_path.resolve(strict=True) or
            sha(amendment_path) != amendment_sha256):
        raise PermissionError("latest effective A amendment differs")
    amendment = json.loads(amendment_path.read_text())
    if (amendment.get("schema") != "pcrl-latent-sampled-A-stage-amendment-v1" or
            not str(amendment.get("status", "")).startswith("LOCALLY_SEALED") or
            amendment.get("interval_family", {}).get("ordered_clause_count") != 6885 or
            amendment.get("interval_family", {}).get("references") !=
            ["T_STAR", "P_STAR", "U0", "Ucal", "D0_LOCAL", "D0_SEQ12", "D0_SEQ21"]):
        raise PermissionError("full seven-reference A amendment is not effective")
    return amendment


def _analysis_amendment(state: dict, path: Path, digest: str,
                        producer_sha: str) -> dict:
    latest = next((entry for entry in reversed(state.get("amendments", []))
                   if "verification" in entry.get("stages", [])), None)
    if (latest is None or latest.get("sha256") != digest or
            Path(latest["path"]).resolve(strict=True) != path.resolve(strict=True) or
            sha(path) != digest):
        raise PermissionError("latest verification amendment differs")
    document = json.loads(path.read_text())
    if (document.get("schema") != "pcrl-a-endpoint-analysis-amendment-v1" or
            document.get("status") != "LOCALLY_SEALED_A_ENDPOINT" or
            document.get("producer_amendment_sha256") != producer_sha or
            document.get("authorized_unit_ids") != [UNIT_ID]):
        raise PermissionError("A endpoint analysis amendment is not effective")
    return document


def build_spec(*, amendment_path: str | Path, amendment_sha256: str,
               analysis_amendment_path: str | Path,
               analysis_amendment_sha256: str,
               role_manifest_path: str | Path, role_manifest_sha256: str,
               allow_registered: bool = False) -> dict:
    """Require all attack and release parents DONE; never open an Adult array."""
    state = json.loads(QUEUE_STATE.read_text())
    seal = state.get("seal") or {}
    if (state.get("closed") or seal.get("path") != str(LOCK) or
            sha(LOCK) != seal.get("sha256")):
        raise PermissionError("R03 effective seal is absent or changed")
    amendment_path = Path(amendment_path).resolve(strict=True)
    amendment = _latest_a(state, amendment_path, amendment_sha256)
    analysis_amendment_path = Path(analysis_amendment_path).resolve(strict=True)
    analysis = _analysis_amendment(state, analysis_amendment_path,
                                   analysis_amendment_sha256, amendment_sha256)
    if analysis.get("effective_seal_sha256") != seal["sha256"]:
        raise PermissionError("A endpoint analysis amendment original seal differs")
    cells = expected_attack_cells(amendment["unit_plan"])
    if not allow_registered and UNIT_ID in state.get("units", {}):
        raise FileExistsError("A endpoint unit already registered")
    family_path = CAMPAIGN / "r02/A_FAMILY_ORDER_S1.json"
    if (sha(family_path) != amendment["interval_family"]["manifest_sha256"] or
            sha(CAMPAIGN / "r02/a_inference.py") !=
            analysis.get("source_sha256", {}).get("r02/a_inference.py")):
        raise PermissionError("K6885 manifest/scorer differs from full A amendment")
    for name in SOURCE_NAMES:
        actual = sha(CAMPAIGN / name)
        if actual != analysis.get("source_sha256", {}).get(name):
            raise PermissionError(f"A analysis amendment does not pin {name}")
    for name in SHARED_NAMES:
        relative = f"results/pcrl_method_portfolio_v1/shared/{name}"
        if sha(WORKTREE / relative) != analysis.get("source_sha256", {}).get(relative):
            raise PermissionError("A analysis amendment inherited evaluator differs")
    role_manifest_path = Path(role_manifest_path).resolve(strict=True)
    if sha(role_manifest_path) != role_manifest_sha256:
        raise PermissionError("role-only manifest byte hash differs")
    lock = json.loads(LOCK.read_text())
    if (role_manifest_sha256 != lock["source_files_sha256"]["source_role_manifest"] or
            analysis.get("role_manifest_sha256") != role_manifest_sha256):
        raise PermissionError("role-only manifest differs from original and analysis seals")
    role_doc = json.loads(role_manifest_path.read_text())
    role_package = role_manifest_path.parent.parent
    parent_ids = set(cells.values())
    for attack_id in cells.values():
        plan = amendment["unit_plan"][attack_id]
        parent_ids.update(plan.get("deps", []))
        parent_ids.update(plan.get("release_unit_ids", {}).values())
    units = state.get("units", {})
    if any(units.get(uid, {}).get("status") != "done" or
           units[uid].get("returncode") != 0 for uid in parent_ids):
        raise PermissionError("A endpoint waits for all 58 attacks and release/source parents")
    for path in (PRIVATE_OUT, PUBLIC_JSON, PUBLIC_CSV, PUBLIC_RECEIPT):
        if path.exists():
            raise FileExistsError("A endpoint output already exists")
    source_paths = [LOCK, amendment_path, analysis_amendment_path, role_manifest_path]
    source_paths += [CAMPAIGN / name for name in SOURCE_NAMES]
    source_paths += [WORKTREE / "results/pcrl_method_portfolio_v1/shared" / name
                     for name in SHARED_NAMES]
    for field in ("income", "occupation", "sex"):
        key = f"INNER_SELECTION:{field}"
        source_paths.append(role_package / role_doc["role_file_relative_path"][key])
    for uid in sorted(parent_ids):
        unit = units[uid]
        for path, digest in unit.get("output_sha256", {}).items():
            if sha(path) != digest:
                raise PermissionError(f"A parent output changed: {uid}")
            source_paths.append(Path(path))
    unique = {str(path.resolve(strict=True)): path for path in source_paths}
    sources = [_source(unique[key]) for key in sorted(unique)]
    python = Path(units[next(iter(cells.values()))]["spec"]["argv"][0]).resolve(strict=True)
    argv = [str(python), "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r02.a_endpoint_worker",
            "--unit-id", UNIT_ID, "--amendment", str(amendment_path),
            "--amendment-sha256", amendment_sha256,
            "--analysis-amendment", str(analysis_amendment_path),
            "--analysis-amendment-sha256", analysis_amendment_sha256,
            "--role-manifest", str(role_manifest_path),
            "--role-manifest-sha256", role_manifest_sha256,
            "--private-output", str(PRIVATE_OUT),
            "--public-json", str(PUBLIC_JSON),
            "--public-csv", str(PUBLIC_CSV),
            "--public-receipt", str(PUBLIC_RECEIPT)]
    return {"unit_id": UNIT_ID, "stage": "verification", "mode": "real_data",
            "argv": argv, "cwd": str(WORKTREE), "deps": sorted(parent_ids),
            "sources": sources,
            "outputs": [str(PRIVATE_OUT / "A_SOURCE_BINDING.json"),
                        *[str(private_cell_path(PRIVATE_OUT / "cells", arm, seed, draw))
                          for arm, seed, draw in cells],
                        str(PUBLIC_JSON), str(PUBLIC_CSV), str(PUBLIC_RECEIPT)],
            "max_wall_s": 1800, "cpu_reserve_s": 1800, "max_rss_gib": 3.0,
            "projected_output_bytes": 100000000,
            "env": {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                               "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                               "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")},
            "amendment_sha256": analysis_amendment_sha256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", required=True)
    parser.add_argument("--amendment-sha256", required=True)
    parser.add_argument("--analysis-amendment", required=True)
    parser.add_argument("--analysis-amendment-sha256", required=True)
    parser.add_argument("--role-manifest", required=True)
    parser.add_argument("--role-manifest-sha256", required=True)
    args = parser.parse_args()
    spec = build_spec(amendment_path=args.amendment,
                      amendment_sha256=args.amendment_sha256,
                      analysis_amendment_path=args.analysis_amendment,
                      analysis_amendment_sha256=args.analysis_amendment_sha256,
                      role_manifest_path=args.role_manifest,
                      role_manifest_sha256=args.role_manifest_sha256)
    path = CAMPAIGN / "r02/.private/A_K6885_ENDPOINT_SPEC.json"
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with path.open("x") as stream:
        json.dump(spec, stream, sort_keys=True, indent=2)
        stream.write("\n")
    path.chmod(0o600)
    print(json.dumps({"unit_id": UNIT_ID, "spec_sha256": sha(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
