"""Independent data-free source and Git custody check before A downstream work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .independent_a_source import sha_file


MANIFEST_KEYS = {"schema", "status", "dated_utc", "producer_amendment_sha256",
                 "original_lock_sha256", "unit_plan_sha256", "new_source_sha256",
                 "attack_plan", "stage_scope", "scientific_scope_unchanged"}
REGISTRATION_KEYS = {"schema", "status", "registered_utc", "producer_amendment_sha256",
                     "manifest_relative_path", "manifest_sha256",
                     "manifest_commit_full_sha", "stage_scope",
                     "independent_review_sha256", "source_map_sha256"}
NEW = {"r17/q_specs.py", "sampled_channel_v1/attacks/render_attack_specs.py",
       "sampled_channel_v1/attacks/run_downstream_attack.py",
       "sampled_channel_v1/controls/reference_diag_specs.py",
       "sampled_channel_v1/impl/downstream_sources.py"}
SHARED = {"attack_controls_v2.py", "attack_eval.py", "common_eval.py",
          "safe_data_guard.py"}
STAGES = ["attack", "diagnostic", "q", "reference", "release"]


def _git_bytes(worktree: Path, commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"],
                                   cwd=worktree)


def run(campaign: Path) -> dict:
    worktree = campaign.parents[1]
    manifest_path = campaign / "A_DOWNSTREAM_SOURCE_MANIFEST_20261010_001.json"
    registration_path = campaign / "A_DOWNSTREAM_SOURCE_REGISTRATION_20261010_001.json"
    producer_path = campaign / "AMENDMENT_20261010_A_PRODUCER_STAGE_FIT_RETRY_001.json"
    lock_path = campaign / "CAMPAIGN_LOCK.json"
    review_path = campaign / "sampled_channel_v1/review/R15_DOWNSTREAM_DISPATCH_REVIEW.md"
    manifest = json.loads(manifest_path.read_text())
    registration = json.loads(registration_path.read_text())
    producer = json.loads(producer_path.read_text())
    if (set(manifest) != MANIFEST_KEYS or set(registration) != REGISTRATION_KEYS
            or manifest.get("schema") != "pcrl-a-downstream-source-manifest-v1"
            or manifest.get("status") != "LOCALLY_SEALED_DOWNSTREAM"
            or registration.get("schema") != "pcrl-a-downstream-source-registration-v1"
            or registration.get("status") != "LOCALLY_REGISTERED_BY_R00"
            or manifest.get("scientific_scope_unchanged") is not True
            or manifest.get("stage_scope") != STAGES
            or registration.get("stage_scope") != STAGES):
        raise ValueError("A downstream manifest or local registration shape differs")
    plan_digest = hashlib.sha256(json.dumps(
        producer["unit_plan"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if (len(producer["unit_plan"]) != 220
            or manifest.get("producer_amendment_sha256") != sha_file(producer_path)
            or registration.get("producer_amendment_sha256") != sha_file(producer_path)
            or manifest.get("original_lock_sha256") != sha_file(lock_path)
            or manifest.get("unit_plan_sha256") != plan_digest):
        raise ValueError("A downstream manifest changes original producer lineage")
    new = manifest.get("new_source_sha256")
    attack = manifest.get("attack_plan")
    inherited = attack.get("inherited_shared_source_sha256") if type(attack) is dict else None
    if (type(new) is not dict or set(new) != NEW
            or type(attack) is not dict or set(attack) != {"inherited_shared_source_sha256"}
            or type(inherited) is not dict or set(inherited) != SHARED
            or set(new) & set(producer["implementation_source_sha256"])):
        raise ValueError("A downstream source allowlist differs")
    source_map = {"new_source_sha256": new,
                  "inherited_shared_source_sha256": inherited}
    map_digest = hashlib.sha256(json.dumps(
        source_map, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    manifest_rel = manifest_path.relative_to(worktree).as_posix()
    registration_rel = registration_path.relative_to(worktree).as_posix()
    if (registration.get("manifest_sha256") != sha_file(manifest_path)
            or registration.get("manifest_relative_path") != manifest_rel
            or registration.get("source_map_sha256") != map_digest
            or registration.get("independent_review_sha256") != sha_file(review_path)
            or _git_bytes(worktree, registration["manifest_commit_full_sha"],
                          manifest_rel) != manifest_path.read_bytes()
            or _git_bytes(worktree, "HEAD", registration_rel) != registration_path.read_bytes()
            or subprocess.run(["git", "merge-base", "--is-ancestor",
                               registration["manifest_commit_full_sha"], "HEAD"],
                              cwd=worktree, check=False).returncode != 0):
        raise ValueError("A downstream registration lacks exact Git custody")
    for relative, digest in new.items():
        path = campaign / relative
        committed = _git_bytes(worktree, registration["manifest_commit_full_sha"],
                               path.relative_to(worktree).as_posix())
        if sha_file(path) != digest or hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError(f"A downstream new source differs: {relative}")
    for name, digest in inherited.items():
        path = worktree / "results/pcrl_method_portfolio_v1/shared" / name
        committed = _git_bytes(worktree, registration["manifest_commit_full_sha"],
                               path.relative_to(worktree).as_posix())
        if sha_file(path) != digest or hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError(f"A downstream inherited reader differs: {name}")
    state = json.loads((campaign / "r03/.private/compute/state.json").read_text())
    registered_downstream = [uid for uid, entry in producer["unit_plan"].items()
                             if entry.get("kind") != "fit" and uid in state["units"]]
    if registered_downstream:
        raise ValueError("A downstream Adult unit was registered before this admission audit")
    return {"schema": "pcrl-r04-a-downstream-source-registration-audit-v1",
            "status": "EXACT_DOWNSTREAM_SOURCE_REGISTRATION_PASS",
            "scope": "data-free source/Git/queue admission check; no Adult downstream job launched",
            "manifest_sha256": sha_file(manifest_path),
            "registration_sha256": sha_file(registration_path),
            "manifest_commit_full_sha": registration["manifest_commit_full_sha"],
            "registration_review_sha256": sha_file(review_path),
            "original_lock_sha256": sha_file(lock_path),
            "producer_amendment_sha256": sha_file(producer_path),
            "canonical_unit_plan_sha256": plan_digest,
            "source_map_sha256": map_digest,
            "new_source_live_and_manifest_commit_match_count": len(new),
            "inherited_shared_live_and_manifest_commit_match_count": len(inherited),
            "manifest_top_level_key_count": len(manifest),
            "registration_top_level_key_count": len(registration),
            "stage_scope": STAGES,
            "adult_downstream_units_registered_at_audit": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--public-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.campaign.resolve(strict=True))
    path = args.public_receipt
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
