"""Plan six R03 verification units for historical D0 INNER utility only.

This planner hashes source files but never opens source arrays or labels. Each
unit has one aggregate JSON file output and depends on completed D K=567.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..r01.role_store import sha_file


CAMPAIGN = Path(__file__).resolve().parents[1]
WORKTREE = CAMPAIGN.parents[1]
LOCK = CAMPAIGN / "CAMPAIGN_LOCK.json"
QUEUE_STATE = CAMPAIGN / "r03/.private/compute/state.json"
AUDIT = CAMPAIGN / "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json"
DEPENDENCY = "R04-D-INTERVALS-001"
KEYS = {"LOCAL": "hcal_local_d0", "SEQ12": "hcal_seq12_d0",
        "SEQ21": "lra_seq21"}
CODE_SOURCES = (
    "r01/role_store.py", "r01/pstar_source.py", "r01/d0_source.py", "r02/d0_utility.py",
    "r02/d0_utility_worker.py", "r02/prepare_d0_utility_queue_units.py",
    "r03/compute_queue.py",
)


def _source(path: Path, expected_sha: str | None = None) -> dict:
    path = path.resolve(strict=True)
    sha = sha_file(path)
    if expected_sha is not None and sha != expected_sha:
        raise PermissionError("historical D0 source byte hash differs")
    return {"path": str(path), "sha256": sha}


def source_inventory(*, role_manifest: Path, audit: Path, key: str,
                     seed: int, seal: Path) -> list[dict]:
    if key not in KEYS.values() or seed not in (0, 1):
        raise ValueError("unknown historical D0 source identity")
    lock = json.loads(seal.read_text())
    role_doc = json.loads(role_manifest.read_text())
    audit_doc = json.loads(audit.read_text())
    cell = audit_doc["d0"][key][f"s{seed}"]
    if (role_doc.get("schema") != "pcrl-utility-repair-role-admission-v1"
            or audit_doc.get("schema") != "pcrl-r01-a-historical-score-source-audit-v1"
            or audit_doc.get("status") != "ROLE_ONLY_HISTORICAL_SCORE_SOURCES_VERIFIED"
            or cell["roles"]["INNER_SELECTION"]["row_unit_pair_sha256"] !=
               role_doc["roles"]["INNER_SELECTION"]["row_unit_pair_sha256"]
            or cell["roles"]["INNER_SELECTION"]["source_reference_sha256"] !=
               role_doc["role_file_sha256"][f"INNER_SELECTION:references_s{seed}"]):
        raise PermissionError("D0 source audit differs from admitted INNER role")
    pins = lock["implementation_source_sha256"]
    for name in ("r01/role_store.py", "r03/compute_queue.py"):
        relative = f"results/pcrl_latent_and_sampled_methods_v1/{name}"
        if sha_file(CAMPAIGN / name) != pins.get(relative):
            raise PermissionError("original lock-pinned reader/scheduler source changed")
    if sha_file(role_manifest) != lock["source_files_sha256"]["source_role_manifest"]:
        raise PermissionError("role-only manifest differs from original effective seal")
    sources = [_source(seal), _source(role_manifest), _source(audit)]
    sources.extend(_source(CAMPAIGN / name) for name in CODE_SOURCES)
    role_root = role_manifest.parent.parent
    for field in ("income", "occupation", f"references_s{seed}"):
        item = f"INNER_SELECTION:{field}"
        sources.append(_source(role_root / role_doc["role_file_relative_path"][item],
                               role_doc["role_file_sha256"][item]))
    sources.append(_source(Path(cell["receipt_path"]), cell["receipt_sha256"]))
    portfolio_manifest = Path(cell["receipt_path"]).parents[3] / "DATA_ACCESS_MANIFEST.json"
    sources.append(_source(portfolio_manifest, role_doc["portfolio_manifest_sha256"]))
    for recipient in ("r1", "r2"):
        item = cell["banks"][recipient]
        sources.append(_source(Path(item["path"]), item["sha256"]))
    token = cell["roles"]["INNER_SELECTION"]
    sources.append(_source(Path(token["path"]), token["sha256"]))
    for name in ("fine.json", "model.pt", "policy.json"):
        item = cell["source_paths_sha256"][name]
        sources.append(_source(Path(item["path"]), item["sha256"]))
    if len({item["path"] for item in sources}) != len(sources):
        raise ValueError("duplicate D0 queue source path")
    return sources


def build_specs() -> dict[str, dict]:
    state = json.loads(QUEUE_STATE.read_text())
    seal = state.get("seal") or {}
    if seal.get("path") != str(LOCK) or sha_file(LOCK) != seal.get("sha256"):
        raise PermissionError("R03 effective D seal differs")
    if state.get("units", {}).get(DEPENDENCY, {}).get("status") != "done":
        raise PermissionError("D K=567 interval predecessor is not complete")
    lock = json.loads(LOCK.read_text())
    role_sha = lock["source_files_sha256"]["source_role_manifest"]
    role_manifest = Path(json.loads((CAMPAIGN / "r01/U_ANCHOR_SOURCE_MAP.json").read_text())
                         ["repair_manifest"]).resolve(strict=True)
    if sha_file(role_manifest) != role_sha:
        raise PermissionError("role manifest differs from original seal")
    audit_sha = sha_file(AUDIT)
    python = WORKTREE.parents[1] / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError("sealed project interpreter missing")
    specs = {}
    for arm, key in KEYS.items():
        for seed in (0, 1):
            unit_id = f"R02-D0-UTILITY-{arm}-S{seed}"
            if unit_id in state.get("units", {}):
                raise FileExistsError("D0 utility unit already registered")
            output = CAMPAIGN / "r02" / f"{unit_id}_RECEIPT.json"
            if output.exists():
                raise FileExistsError("D0 aggregate output already exists")
            argv = [str(python), "-m",
                    "results.pcrl_latent_and_sampled_methods_v1.r02.d0_utility_worker",
                    "--stage", "verification", "--unit-id", unit_id,
                    "--arm", arm, "--seed", str(seed),
                    "--seal", str(LOCK), "--seal-sha256", seal["sha256"],
                    "--role-manifest", str(role_manifest),
                    "--role-manifest-sha256", role_sha,
                    "--audit", str(AUDIT), "--audit-sha256", audit_sha,
                    "--output", str(output)]
            specs[unit_id] = {
                "unit_id": unit_id, "stage": "verification", "argv": argv,
                "cwd": str(WORKTREE), "max_wall_s": 180, "cpu_reserve_s": 180,
                "max_rss_gib": 1.0, "projected_output_bytes": 100000,
                "sources": source_inventory(role_manifest=role_manifest, audit=AUDIT,
                                             key=key, seed=seed, seal=LOCK),
                "deps": [DEPENDENCY], "outputs": [str(output)],
                "env": {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            }
    return specs


def main() -> None:
    specs = build_specs()
    root = CAMPAIGN / ".private/d0_utility_specs"
    root.mkdir(mode=0o700, exist_ok=True)
    if root.stat().st_mode & 0o077:
        raise PermissionError("D0 spec directory must be owner-only")
    for unit_id, spec in specs.items():
        path = root / f"{unit_id}_SPEC.json"
        with path.open("x") as stream:
            json.dump(spec, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        path.chmod(0o600)
        print(json.dumps({"unit_id": unit_id, "spec_path": str(path),
                          "spec_sha256": sha_file(path), "attacker_fits": 0},
                         sort_keys=True))


if __name__ == "__main__":
    main()
