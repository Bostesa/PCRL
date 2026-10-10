"""Create four immutable R03 specs after R00 seals the D-anchor amendment.

Planning only: reads hashes and queue state, never opens role array values or
fits a reader. The anchor worker itself enforces the active queue slot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[1]
WORKTREE = CAMPAIGN.parents[1]
LOCK = CAMPAIGN / "CAMPAIGN_LOCK.json"
QUEUE_STATE = CAMPAIGN / "r03/.private/compute/state.json"
SOURCE_MAP = CAMPAIGN / "r01/U_ANCHOR_SOURCE_MAP.json"
R01_SOURCE_MAP_SHA256 = "9f8ba768c9e2ce6f7eeb0dc9d816a665d766301f95bfeca0fc43f7b1c5a6bd4d"
WORKER_SOURCES = (
    "r02/anchor_worker.py", "r02/reference_anchor.py", "r02/prepare_anchor_queue_units.py",
    "r02/protocol.py",
    "r02/complete_view.py", "r01/role_store.py", "r03/compute_queue.py",
)
SHARED_SOURCES = ("attack_eval.py", "attack_controls_v2.py", "common_eval.py",
                  "safe_data_guard.py")
UNIT_IDS = tuple(f"D_U_ANCHOR_{anchor.upper()}_S{seed}"
                 for seed in (0, 1) for anchor in ("U0", "Ucal"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(path: Path, expected: str | None = None) -> dict:
    path = path.resolve(strict=True)
    actual = _sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f"queued anchor source bytes changed: {path}")
    return {"path": str(path), "sha256": actual}


def _latest_d_amendment(state: dict) -> tuple[dict, dict]:
    entry = next((item for item in reversed(state.get("amendments", []))
                  if "d_followup" in item["stages"]), None)
    if entry is None:
        raise PermissionError("D anchor amendment has not been registered")
    doc = json.loads(Path(entry["path"]).read_text())
    if (_sha(Path(entry["path"])) != entry["sha256"]
            or doc.get("schema") != "pcrl-d-followup-amendment-v1"
            or doc.get("status") != "LOCALLY_SEALED_D_FOLLOWUP"
            or set(doc.get("authorized_unit_ids", [])) != set(UNIT_IDS)):
        raise PermissionError("latest D followup amendment does not authorize four anchor units")
    return entry, doc


def build_specs(*, after_unit_id: str) -> dict[str, dict]:
    state = json.loads(QUEUE_STATE.read_text())
    seal = state.get("seal") or {}
    if seal.get("path") != str(LOCK) or _sha(LOCK) != seal.get("sha256"):
        raise PermissionError("R03 effective lock changed")
    if state.get("units", {}).get(after_unit_id, {}).get("status") != "done":
        raise PermissionError("D anchor predecessor unit is not complete")
    entry, amendment = _latest_d_amendment(state)
    if amendment.get("effective_seal_sha256") != seal["sha256"]:
        raise PermissionError("anchor amendment effective seal differs")
    if _sha(SOURCE_MAP) != R01_SOURCE_MAP_SHA256 or amendment.get("r01_source_map_sha256") != R01_SOURCE_MAP_SHA256:
        raise PermissionError("R01 source map differs from anchor amendment")
    source_map = json.loads(SOURCE_MAP.read_text())
    role_manifest = Path(source_map["repair_manifest"]).resolve(strict=True)
    lock = json.loads(LOCK.read_text())
    pinned = lock["implementation_source_sha256"]
    for name in ("r01/role_store.py", "r02/complete_view.py", "r02/protocol.py",
                 "r03/compute_queue.py"):
        relative = f"results/pcrl_latent_and_sampled_methods_v1/{name}"
        if _sha(CAMPAIGN / name) != pinned.get(relative):
            raise PermissionError("original lock-pinned anchor source changed")
    for name in ("attack_eval.py", "common_eval.py"):
        relative = f"results/pcrl_method_portfolio_v1/shared/{name}"
        if _sha(WORKTREE / relative) != pinned.get(relative):
            raise PermissionError("original lock-pinned shared evaluator changed")
    for name in ("attack_controls_v2.py", "safe_data_guard.py"):
        if _sha(WORKTREE / "results/pcrl_method_portfolio_v1/shared" / name) != \
                amendment.get("anchor_additional_shared_sha256", {}).get(name):
            raise PermissionError("anchor amendment-pinned shared control/guard changed")
    role_sha = lock["source_files_sha256"]["source_role_manifest"]
    if (_sha(role_manifest) != role_sha
            or amendment.get("role_manifest_sha256") != role_sha):
        raise PermissionError("anchor role manifest is not lock/amendment pinned")
    if amendment.get("anchor_worker_sha256") != _sha(CAMPAIGN / "r02/anchor_worker.py") or \
            amendment.get("reference_anchor_sha256") != _sha(CAMPAIGN / "r02/reference_anchor.py") or \
            amendment.get("anchor_spec_helper_sha256") != _sha(Path(__file__)):
        raise PermissionError("anchor worker, adapter or planner differs from amendment")
    role_doc = json.loads(role_manifest.read_text())
    role_package = role_manifest.parent.parent
    common = [_source(LOCK, seal["sha256"]), _source(Path(entry["path"]), entry["sha256"]),
              _source(SOURCE_MAP, R01_SOURCE_MAP_SHA256), _source(role_manifest, role_sha),
              _source(CAMPAIGN / "r01/SOURCE_AUDIT.json",
                      lock["source_files_sha256"]["r01/SOURCE_AUDIT.json"])]
    common += [_source(CAMPAIGN / name) for name in WORKER_SOURCES]
    common += [_source(WORKTREE / "results/pcrl_method_portfolio_v1/shared" / name)
               for name in SHARED_SOURCES]
    for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        key = f"{role}:sex"
        path = role_package / role_doc["role_file_relative_path"][key]
        common.append(_source(path, role_doc["role_file_sha256"][key]))
    python = Path(state["units"][after_unit_id]["spec"]["argv"][0])
    if not python.is_absolute() or not python.is_file():
        raise FileNotFoundError("queued interpreter missing")
    specs = {}
    for seed in (0, 1):
        references = []
        for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
            key = f"{role}:references_s{seed}"
            path = role_package / role_doc["role_file_relative_path"][key]
            references.append(_source(path, role_doc["role_file_sha256"][key]))
        for anchor in ("U0", "Ucal"):
            unit_id = f"D_U_ANCHOR_{anchor.upper()}_S{seed}"
            if unit_id in state["units"]:
                raise FileExistsError(f"anchor queue unit already registered: {unit_id}")
            private_out = CAMPAIGN / ".private" / f"d_u_anchor_{anchor.lower()}_s{seed}"
            public_out = CAMPAIGN / "r02" / f"{unit_id}_RECEIPT.json"
            if private_out.exists() or public_out.exists():
                raise FileExistsError(f"anchor output already exists: {unit_id}")
            argv = [str(python), "-m", "results.pcrl_latent_and_sampled_methods_v1.r02.anchor_worker",
                    "--seal", str(LOCK), "--seal-sha256", seal["sha256"],
                    "--unit-id", unit_id, "--stage", "d_followup",
                    "--amendment", entry["path"], "--amendment-sha256", entry["sha256"],
                    "--role-manifest", str(role_manifest), "--role-manifest-sha256", role_sha,
                    "--anchor", anchor, "--seed", str(seed), "--attacker-seed", str(seed),
                    "--private-output", str(private_out), "--public-receipt", str(public_out)]
            outputs = [public_out, private_out / "FROZEN_COMPLETE.json",
                       private_out / "SOURCE_BINDING.json"]
            outputs += [private_out / view / "readers/READERS_RECEIPT.json"
                        for view in ("r1", "r2", "pair")]
            outputs += [private_out / view / "controls/CUSTODY_RECEIPT.json"
                        for view in ("r1", "r2", "pair")]
            specs[unit_id] = {
                "unit_id": unit_id, "stage": "d_followup", "argv": argv,
                "cwd": str(WORKTREE), "max_wall_s": 600, "cpu_reserve_s": 600,
                "max_rss_gib": 3.0, "projected_output_bytes": 300000000,
                "sources": [*common, *references], "deps": [after_unit_id],
                "outputs": [str(path) for path in outputs],
                "env": {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                "amendment_sha256": entry["sha256"],
            }
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-unit-id", required=True)
    args = parser.parse_args()
    specs = build_specs(after_unit_id=args.after_unit_id)
    out = CAMPAIGN / ".private/d_u_anchor_specs"
    out.mkdir(mode=0o700, exist_ok=True)
    if out.stat().st_mode & 0o077:
        raise PermissionError("anchor queue specification directory must be owner-only")
    for unit_id, spec in specs.items():
        path = out / f"{unit_id}_SPEC.json"
        with path.open("x") as stream:
            json.dump(spec, stream, sort_keys=True, indent=2)
            stream.write("\n")
        path.chmod(0o600)
        print(json.dumps({"unit_id": unit_id, "spec_path": str(path),
                          "spec_sha256": _sha(path), "reader_fits": 24}, sort_keys=True))


if __name__ == "__main__":
    main()
