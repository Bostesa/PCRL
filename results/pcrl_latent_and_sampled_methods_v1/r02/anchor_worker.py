"""R03-queued U0/Ucal recovery anchor attacker for the frozen D family.

One unit fits one reference/source seed using only R01's role-specific ATT and
INNER reference archives. It cannot run without an active R03 d_followup slot.
INNER SEX is opened only after all three reader banks have private custody and
the fsynced FROZEN_COMPLETE receipt. This worker does no D candidate fit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..r01.role_store import RoleOnlyStore, assert_same_people, sha_array, sha_file
from ..r03.compute_queue import assert_active_from_env
from .protocol import run_three_view_attack
from .reference_anchor import prepared_anchor_views

ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")
ANCHORS = ("U0", "Ucal")
STAGE = "d_followup"
R01_SOURCE_MAP_SHA256 = "9f8ba768c9e2ce6f7eeb0dc9d816a665d766301f95bfeca0fc43f7b1c5a6bd4d"
CAMPAIGN = Path(__file__).resolve().parents[1]
WORKTREE = CAMPAIGN.parents[1]


def _write_once(path: Path, data: dict, *, private: bool) -> str:
    with path.open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if private:
        path.chmod(0o600)
    return sha_file(path)


def _private_output(path: Path) -> Path:
    target = path.resolve(strict=False)
    if (".private" not in target.parts or target.exists() or not target.parent.is_dir()
            or target.parent.stat().st_mode & 0o077):
        raise PermissionError("anchor custody must be a new owner-only .private directory")
    return target


def _verify_queue_admission(args: argparse.Namespace, active: dict, *,
                            seal: Path, lock: dict, role_manifest: Path,
                            source_map_path: Path, audit_path: Path,
                            private_out: Path, public_out: Path) -> str:
    """Bind the running process to R00's latest four-unit amendment and spec."""
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    latest = next((entry for entry in reversed(state.get("amendments", []))
                   if STAGE in entry["stages"]), None)
    amendment = Path(args.amendment).resolve(strict=True)
    if (latest is None or latest.get("sha256") != args.amendment_sha256
            or Path(latest["path"]).resolve() != amendment
            or sha_file(amendment) != args.amendment_sha256):
        raise PermissionError("latest registered D anchor amendment differs")
    dated = json.loads(amendment.read_text())
    if (dated.get("schema") != "pcrl-d-followup-amendment-v1"
            or dated.get("status") != "LOCALLY_SEALED_D_FOLLOWUP"
            or dated.get("effective_seal_sha256") != args.seal_sha256
            or set(dated.get("authorized_unit_ids", ())) !=
               {f"D_U_ANCHOR_{arm.upper()}_S{seed}" for seed in (0, 1) for arm in ANCHORS}
            or args.unit_id not in dated["authorized_unit_ids"]
            or dated.get("role_manifest_sha256") != args.role_manifest_sha256
            or dated.get("r01_source_map_sha256") != R01_SOURCE_MAP_SHA256
            or dated.get("anchor_worker_sha256") != sha_file(Path(__file__))
            or dated.get("reference_anchor_sha256") != sha_file(CAMPAIGN / "r02/reference_anchor.py")
            or dated.get("anchor_spec_helper_sha256") != sha_file(CAMPAIGN / "r02/prepare_anchor_queue_units.py")):
        raise PermissionError("R00 amendment does not bind this exact anchor worker/slate")
    original_pins = ("r01/role_store.py", "r02/complete_view.py", "r02/protocol.py",
                     "r03/compute_queue.py")
    shared_pins = ("attack_eval.py", "common_eval.py")
    pinned = lock["implementation_source_sha256"]
    for name in original_pins:
        relative = f"results/pcrl_latent_and_sampled_methods_v1/{name}"
        if sha_file(CAMPAIGN / name) != pinned.get(relative):
            raise PermissionError("original lock-pinned anchor source changed")
    for name in shared_pins:
        relative = f"results/pcrl_method_portfolio_v1/shared/{name}"
        if sha_file(WORKTREE / relative) != pinned.get(relative):
            raise PermissionError("original lock-pinned shared evaluator changed")
    for name in ("attack_controls_v2.py", "safe_data_guard.py"):
        if sha_file(WORKTREE / "results/pcrl_method_portfolio_v1/shared" / name) != \
                dated.get("anchor_additional_shared_sha256", {}).get(name):
            raise PermissionError("A5 amendment-pinned shared control/guard changed")
    unit = state.get("units", {}).get(args.unit_id, {})
    spec = unit.get("spec", {})
    argv = spec.get("argv", [])
    if (unit.get("status") != "running" or unit.get("spec_sha256") != active["spec_sha256"]
            or spec.get("unit_id") != args.unit_id or spec.get("stage") != STAGE
            or spec.get("amendment_sha256") != args.amendment_sha256
            or spec.get("cwd") != str(WORKTREE)
            or len(argv) < 4 or Path(argv[0]).resolve() != Path(sys.executable).resolve()
            or argv[1:3] != ["-m", "results.pcrl_latent_and_sampled_methods_v1.r02.anchor_worker"]
            or argv[3:] != sys.argv[1:]
            or float(spec.get("max_wall_s", -1)) != 600
            or float(spec.get("cpu_reserve_s", -1)) != 600
            or float(spec.get("max_rss_gib", -1)) != 3.0
            or int(spec.get("projected_output_bytes", -1)) != 300000000):
        raise PermissionError("active R03 anchor spec does not match running worker")
    deps = spec.get("deps", [])
    if len(deps) != 1 or state.get("units", {}).get(deps[0], {}).get("status") != "done":
        raise PermissionError("anchor predecessor dependency is not complete")
    required_outputs = {str(public_out), str(private_out / "FROZEN_COMPLETE.json"),
                        str(private_out / "SOURCE_BINDING.json")}
    required_outputs |= {str(private_out / view / "readers/READERS_RECEIPT.json")
                         for view in ("r1", "r2", "pair")}
    required_outputs |= {str(private_out / view / "controls/CUSTODY_RECEIPT.json")
                         for view in ("r1", "r2", "pair")}
    if set(spec.get("outputs", [])) != required_outputs:
        raise PermissionError("R03 anchor file output inventory differs")
    role_doc = json.loads(role_manifest.read_text())
    role_package = role_manifest.parent.parent
    code = ("r02/anchor_worker.py", "r02/reference_anchor.py",
            "r02/prepare_anchor_queue_units.py", "r02/protocol.py", "r02/complete_view.py",
            "r01/role_store.py", "r03/compute_queue.py")
    shared = ("attack_eval.py", "attack_controls_v2.py", "common_eval.py", "safe_data_guard.py")
    required_sources = [seal, amendment, role_manifest, source_map_path, audit_path]
    required_sources += [CAMPAIGN / name for name in code]
    required_sources += [WORKTREE / "results/pcrl_method_portfolio_v1/shared" / name
                         for name in shared]
    for role in ROLES:
        for field in ("sex", f"references_s{args.seed}"):
            key = f"{role}:{field}"
            required_sources.append(role_package / role_doc["role_file_relative_path"][key])
    expected = {str(path.resolve()): sha_file(path) for path in required_sources}
    sources = spec.get("sources", [])
    actual = {item.get("path"): item.get("sha256") for item in sources}
    if len(sources) != len(expected) or actual != expected:
        raise PermissionError("R03 anchor source inventory/hash differs")
    return args.amendment_sha256


def run(args: argparse.Namespace) -> dict:
    if (args.stage != STAGE or args.seed not in (0, 1) or args.anchor not in ANCHORS
            or args.attacker_seed != args.seed
            or args.unit_id != f"D_U_ANCHOR_{args.anchor.upper()}_S{args.seed}"):
        raise PermissionError("unregistered D anchor unit identity")
    active = assert_active_from_env(unit_id=args.unit_id, stage=STAGE)
    if active["seal_sha256"] != args.seal_sha256:
        raise PermissionError("queue and worker effective seal differ")
    seal = Path(args.seal).resolve(strict=True)
    if sha_file(seal) != args.seal_sha256:
        raise PermissionError("effective seal bytes changed")
    lock = json.loads(seal.read_text())
    if (lock.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1"
            or lock.get("status") != "LOCALLY_SEALED_D_STAGE"):
        raise PermissionError("D anchor source has no effective seal")
    role_manifest = Path(args.role_manifest).resolve(strict=True)
    if (sha_file(role_manifest) != args.role_manifest_sha256
            or lock["source_files_sha256"]["source_role_manifest"] != args.role_manifest_sha256):
        raise PermissionError("admitted role manifest differs from effective seal")
    source_map_path = CAMPAIGN / "r01/U_ANCHOR_SOURCE_MAP.json"
    if sha_file(source_map_path) != R01_SOURCE_MAP_SHA256:
        raise PermissionError("R01 U-anchor source map bytes differ")
    source_map = json.loads(source_map_path.read_text())
    if (source_map.get("schema") != "pcrl-new-campaign-r01-u-anchor-source-map-v1"
            or Path(source_map["repair_manifest"]).resolve() != role_manifest):
        raise PermissionError("R01 U-anchor source map identity differs")
    audit_path = CAMPAIGN / "r01/SOURCE_AUDIT.json"
    audit_sha = lock["source_files_sha256"]["r01/SOURCE_AUDIT.json"]
    if sha_file(audit_path) != audit_sha:
        raise PermissionError("lock-pinned R01 source audit bytes differ")
    source_audit = json.loads(audit_path.read_text())
    provenance = {
        "u0_model_components": source_audit["u0_models"][str(args.seed)],
        "ucal_calibration": source_audit["ucal_calibration"][str(args.seed)],
    }
    if provenance["u0_model_components"]["components_sha256"]["model.pt"] != \
            lock["role_identity_and_file_hashes"]["source_model_sha256"][str(args.seed)]:
        raise PermissionError("U0 model provenance differs from effective seal")
    private_out = _private_output(Path(args.private_output))
    public_out = Path(args.public_receipt).resolve(strict=False)
    if public_out.exists() or not public_out.parent.is_dir():
        raise FileExistsError("anchor public receipt must be a new file")
    amendment_sha = _verify_queue_admission(
        args, active, seal=seal, lock=lock, role_manifest=role_manifest,
        source_map_path=source_map_path, audit_path=audit_path,
        private_out=private_out, public_out=public_out)

    def guard() -> None:
        current = assert_active_from_env(unit_id=args.unit_id, stage=STAGE)
        if current["seal_sha256"] != args.seal_sha256:
            raise PermissionError("active slot seal changed during anchor attack")

    guard()
    store = RoleOnlyStore(role_manifest)
    field = f"references_s{args.seed}"
    for role in ROLES:
        expected = source_map["rows"][role]
        if (store.manifest["role_file_sha256"][f"{role}:{field}"] != expected[f"{field}_sha256"]
                or store.manifest["roles"][role]["row_unit_pair_sha256"] != expected["row_unit_pair_sha256"]
                or store.manifest["roles"][role]["rows"] != expected["count"]):
            raise PermissionError("R01 anchor archive or ordered role identity differs")
    refs = {role: store.read(role, field) for role in ROLES}
    prepared = prepared_anchor_views(refs["ATTACK_FIT_NEW"], refs["INNER_SELECTION"], args.anchor)
    sex_att = store.read("ATTACK_FIT_NEW", "sex", procedure="attack")
    assert_same_people(sex_att, refs["ATTACK_FIT_NEW"])

    def load_inner_sex() -> dict:
        guard()
        if not (private_out / "FROZEN_COMPLETE.json").is_file():
            raise PermissionError("INNER SEX cannot open before anchor reader freeze")
        sex_inner = store.read("INNER_SELECTION", "sex", procedure="screen")
        assert_same_people(sex_inner, refs["INNER_SELECTION"])
        return sex_inner

    attack = run_three_view_attack(
        prepared, sex_att, load_inner_sex=load_inner_sex, family="D",
        seed=args.attacker_seed, private_root=private_out,
        lock_path=seal, lock_sha256=args.seal_sha256, queue_guard=guard)
    source = {
        "schema": "pcrl-d-u-anchor-source-binding-v1", "anchor": args.anchor,
        "source_seed": args.seed, "unit_id": args.unit_id, "stage": STAGE,
        "effective_lock_sha256": args.seal_sha256,
        "d_anchor_amendment_sha256": amendment_sha,
        "role_manifest_path": str(role_manifest),
        "role_manifest_sha256": args.role_manifest_sha256,
        "r01_source_map_sha256": R01_SOURCE_MAP_SHA256,
        "lock_pinned_source_audit_sha256": audit_sha,
        "fixed_source_provenance": provenance,
        "wire_contract": "score-probabilities-and-argmax-decision-only; no latent, code, or per-person metadata",
        "fixed_public_identity": {"anchor": args.anchor, "source_seed": args.seed},
        "roles": {role: {
            "reference_archive_path": str((store.package_root / store.manifest["role_file_relative_path"][f"{role}:{field}"]).resolve()),
            "reference_archive_sha256": store.manifest["role_file_sha256"][f"{role}:{field}"],
            "ordered_row_unit_sha256": store.manifest["roles"][role]["row_unit_pair_sha256"],
            "rows": store.manifest["roles"][role]["rows"],
            "groups": store.manifest["roles"][role]["groups"],
            "probability_sha256": {str(recipient): sha_array(refs[role][f"{args.anchor}_{recipient}"])
                                   for recipient in (1, 2)},
            "decision_sha256": {str(recipient): sha_array(refs[role][f"d_{recipient}"])
                                for recipient in (1, 2)},
        } for role in ROLES},
        "sex_archive_sha256": {
            role: store.manifest["role_file_sha256"][f"{role}:sex"] for role in ROLES},
        "view_schema_sha256": attack["public"]["view_schema_sha256"],
        "reader_receipt_sha256": {
            view: attack["public"]["reader_custody"][view]["private_receipt_sha256"]
            for view in ("r1", "r2", "pair")},
        "freeze_receipt_sha256": attack["public"]["freeze_receipt_sha256"],
        "reader_plan": attack["public"]["plan"],
    }
    source_sha = _write_once(private_out / "SOURCE_BINDING.json", source, private=True)
    public = {"schema": "pcrl-d-u-anchor-attack-receipt-v1",
              "status": "COMPLETE_PRIVATE_DEVELOPMENT_ONLY", "anchor": args.anchor,
              "source_seed": args.seed, "attacker_seed": args.attacker_seed,
              "unit_id": args.unit_id, "stage": STAGE,
              "effective_lock_sha256": args.seal_sha256,
              "d_anchor_amendment_sha256": amendment_sha,
              "role_manifest_sha256": args.role_manifest_sha256,
              "r01_source_map_sha256": R01_SOURCE_MAP_SHA256,
              "private_artifact_id": args.unit_id,
              "private_source_binding_sha256": source_sha,
              "source_archive_sha256": {role: source["roles"][role]["reference_archive_sha256"]
                                        for role in ROLES},
              "ordered_row_unit_sha256": {role: source["roles"][role]["ordered_row_unit_sha256"]
                                          for role in ROLES},
              "attack": attack["public"]}
    public_sha = _write_once(public_out, public, private=False)
    return {"status": public["status"], "unit_id": args.unit_id,
            "anchor": args.anchor, "source_seed": args.seed,
            "public_receipt_sha256": public_sha}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "seal-sha256", "unit-id", "stage", "role-manifest",
                 "role-manifest-sha256", "amendment", "amendment-sha256",
                 "anchor", "private-output", "public-receipt"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--attacker-seed", type=int, required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
