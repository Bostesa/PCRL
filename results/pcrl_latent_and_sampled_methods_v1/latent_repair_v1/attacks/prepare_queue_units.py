"""Materialize paired D attack queue specs after source completion.

This is a read-only planning helper until it writes the two JSON specifications.
It never opens role labels or runs an attacker. R03 admits and runs each spec.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CAMPAIGN = Path(__file__).resolve().parents[2]
WORKTREE = CAMPAIGN.parents[1]
QUEUE_ROOT = CAMPAIGN / "r03/.private/compute"
LOCK = CAMPAIGN / "CAMPAIGN_LOCK.json"
SOURCE_CODE = (
    "latent_repair_v1/attacks/worker.py",
    "latent_repair_v1/attacks/adapter.py",
    "latent_repair_v1/attacks/hard_code.py",
    "latent_repair_v1/impl/service.py",
    "latent_repair_v1/impl/model.py",
    "r01/role_store.py",
    "r02/complete_view.py",
    "r02/protocol.py",
    "r03/compute_queue.py",
)
SHARED_CODE = ("attack_eval.py", "attack_controls_v2.py", "safe_data_guard.py")
RELEASE_ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")
ARMS = (("T", "T"), ("D-K", "DK"))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _arg(argv: list[str], name: str) -> str:
    flag = "--" + name
    if argv.count(flag) != 1:
        raise ValueError(f"source argv needs exactly one {flag}")
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise ValueError(f"source argv lacks value for {flag}")
    return argv[index + 1]


def _source(path: Path, expected: str) -> dict:
    path = path.resolve(strict=True)
    actual = _sha(path)
    if actual != expected:
        raise ValueError(f"source SHA differs: {path}")
    return {"path": str(path), "sha256": actual}


def build_specs(source_unit_id: str, *, seed: int = 0,
                after_unit_id: str | None = None) -> dict[str, dict]:
    """Require completed release, then create exact read-only attack plans."""
    if seed not in (0, 1):
        raise ValueError("only sealed D source seeds 0 and 1 are admitted")
    stage = "d_first" if seed == 0 else "d_followup"
    state = json.loads((QUEUE_ROOT / "state.json").read_text())
    seal = state.get("seal") or {}
    if seal.get("path") != str(LOCK) or _sha(LOCK) != seal.get("sha256"):
        raise PermissionError("R03 effective lock is missing or changed")
    lock = json.loads(LOCK.read_text())
    if lock.get("status") != "LOCALLY_SEALED_D_STAGE":
        raise PermissionError("D stage is not locally sealed")
    unit = state.get("units", {}).get(source_unit_id, {})
    if unit.get("status") != "done" or unit.get("stage") != stage:
        raise PermissionError("the source release queue unit must be done")
    source_spec = unit["spec"]
    if source_spec.get("stage") != stage or source_spec.get("unit_id") != source_unit_id:
        raise ValueError("source spec identity differs")
    if after_unit_id and state.get("units", {}).get(after_unit_id, {}).get("status") != "done":
        raise PermissionError("ordered predecessor queue unit must be done")
    source_argv = source_spec["argv"]
    receipt_path = Path(_arg(source_argv, "public-receipt")).resolve(strict=True)
    receipt_sha = _sha(receipt_path)
    if unit.get("output_sha256", {}).get(str(receipt_path)) != receipt_sha:
        raise PermissionError("queue completion did not hash the source receipt")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "latent-repair-first-paired-seed-receipt-v1"
            or receipt.get("status") != "T_DK_RELEASES_READY"
            or receipt.get("seed") != seed
            or receipt.get("unit_id") != source_unit_id
            or receipt.get("seal_sha256") != seal["sha256"]):
        raise ValueError("source receipt is not the finished paired seed-zero release")
    source_root = Path(_arg(source_argv, "private-output")).resolve(strict=True)
    if ".private" not in source_root.parts or source_root.stat().st_mode & 0o077:
        raise PermissionError("source release must remain under owner-only private custody")
    role_manifest = Path(_arg(source_argv, "role-manifest")).resolve(strict=True)
    role_sha = _sha(role_manifest)
    if (role_sha != receipt.get("role_manifest_sha256")
            or role_sha != lock["source_files_sha256"]["source_role_manifest"]):
        raise ValueError("source role manifest SHA differs")
    if set(receipt.get("packages", {})) != {"T", "D-K"}:
        raise ValueError("paired service packages are incomplete")
    if set(receipt.get("roles", {})) != set(RELEASE_ROLES):
        raise ValueError("paired ATT/INNER roles are incomplete")
    amendment = next((item["sha256"] for item in reversed(state.get("amendments", []))
                      if stage in item["stages"]), None)
    static_sources = [_source(LOCK, seal["sha256"]), _source(role_manifest, role_sha),
                      _source(receipt_path, receipt_sha)]
    for name in SOURCE_CODE:
        path = CAMPAIGN / name
        static_sources.append(_source(path, lock["implementation_source_sha256"][f"results/pcrl_latent_and_sampled_methods_v1/{name}"]))
    shared = WORKTREE / "results/pcrl_method_portfolio_v1/shared"
    for name in SHARED_CODE:
        path = shared / name
        static_sources.append(_source(path, _sha(path)))
    endpoint = source_root / "endpoint__INNER_SELECTION.npz"
    static_sources.append(_source(endpoint,
        receipt["roles"]["INNER_SELECTION"]["endpoint_labels_references_sha256"]))
    specs = {}
    for arm, short_id in ARMS:
        unit_id = f"D_ATTACK_{short_id}_S{seed}"
        private_name = f"d_attack_{short_id.lower()}_s{seed}"
        if unit_id in state["units"]:
            raise FileExistsError(f"attack unit ID already exists: {unit_id}")
        private_output = CAMPAIGN / ".private" / private_name
        public_receipt = Path(__file__).resolve().parent / f"{unit_id}_RECEIPT.json"
        if private_output.exists() or public_receipt.exists():
            raise FileExistsError(f"attack output already exists for {arm}")
        sources = list(static_sources)
        package = source_root / arm
        sources.append(_source(package / "manifest.json", receipt["packages"][arm]["manifest_sha256"]))
        sources.append(_source(package / "weights.npz", receipt["packages"][arm]["weights_sha256"]))
        for role in RELEASE_ROLES:
            sources.append(_source(source_root / f"release__{arm}__{role}.npz",
                                   receipt["roles"][role][arm]["private_release_sha256"]))
        python = Path(source_argv[0])
        if not python.is_file() or not python.is_absolute():
            raise FileNotFoundError("queued source interpreter is missing")
        argv = [str(python), "-m",
                "results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.attacks.worker",
                "--seal", str(LOCK), "--seal-sha256", seal["sha256"],
                "--unit-id", unit_id, "--stage", stage,
                "--role-manifest", str(role_manifest), "--role-manifest-sha256", role_sha,
                "--source-private", str(source_root), "--source-receipt", str(receipt_path),
                "--source-receipt-sha256", receipt_sha,
                "--source-unit-id", source_unit_id, "--arm-id", arm,
                "--private-output", str(private_output), "--public-receipt", str(public_receipt),
                "--seed", str(seed), "--attacker-seed", str(seed)]
        outputs = [public_receipt, private_output / "FROZEN_COMPLETE.json"]
        outputs += [private_output / view / "readers/READERS_RECEIPT.json"
                    for view in ("r1", "r2", "pair")]
        outputs += [private_output / view / "controls/CUSTODY_RECEIPT.json"
                    for view in ("r1", "r2", "pair")]
        spec = {"unit_id": unit_id, "stage": stage, "argv": argv,
                "cwd": str(WORKTREE), "max_wall_s": 600, "cpu_reserve_s": 600,
                "max_rss_gib": 3.0, "projected_output_bytes": 300000000,
                "sources": sources, "deps": [source_unit_id] + ([after_unit_id] if after_unit_id else []),
                "outputs": [str(path) for path in outputs],
                "env": {"PYTHONHASHSEED": "0"}}
        if amendment:
            spec["amendment_sha256"] = amendment
        specs[arm] = spec
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-unit-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--after-unit-id")
    args = parser.parse_args()
    specs = build_specs(args.source_unit_id, seed=args.seed,
                        after_unit_id=args.after_unit_id)
    spec_dir = CAMPAIGN / ".private" / "d_attack_specs"
    spec_dir.mkdir(mode=0o700, exist_ok=True)
    if spec_dir.stat().st_mode & 0o077:
        raise PermissionError("queue specifications require owner-only custody")
    for arm, spec in specs.items():
        path = spec_dir / f"{spec['unit_id']}_SPEC.json"
        with path.open("x") as stream:
            json.dump(spec, stream, sort_keys=True, indent=2)
            stream.write("\n")
        path.chmod(0o600)
        print(json.dumps({"arm": arm, "spec_path": str(path), "spec_sha256": _sha(path),
                          "reader_fits": 30}, sort_keys=True))


if __name__ == "__main__":
    main()
