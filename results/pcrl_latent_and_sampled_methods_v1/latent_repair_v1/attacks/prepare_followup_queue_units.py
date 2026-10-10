"""Write private R03 specs for sealed L/S12/S21/C/F complete-view attacks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .followup_worker import ARMS


CAMPAIGN = Path(__file__).resolve().parents[2]
WORKTREE = CAMPAIGN.parents[1]
QUEUE_ROOT = CAMPAIGN / "r03/.private/compute"
LOCK = CAMPAIGN / "CAMPAIGN_LOCK.json"
SOURCE_CODE = (
    "latent_repair_v1/attacks/followup_worker.py",
    "latent_repair_v1/attacks/worker.py",
    "latent_repair_v1/attacks/adapter.py",
    "latent_repair_v1/attacks/hard_code.py",
    "latent_repair_v1/controls/f_adapter.py",
    "latent_repair_v1/controls/f_service.py",
    "latent_repair_v1/controls/unchanged.py",
    "latent_repair_v1/impl/service.py",
    "latent_repair_v1/impl/model.py",
    "r01/role_store.py",
    "r02/complete_view.py",
    "r02/protocol.py",
    "r03/compute_queue.py",
)
SHARED_CODE = ("attack_eval.py", "attack_controls_v2.py", "safe_data_guard.py")
ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source(path: Path, expected: str) -> dict:
    path = path.resolve(strict=True)
    digest = _sha(path)
    if digest != expected:
        raise ValueError(f"source file SHA differs: {path}")
    return {"path": str(path), "sha256": digest}


def _arg(argv: list[str], name: str) -> str:
    flag = "--" + name
    if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
        raise ValueError(f"source argv lacks one {flag}")
    return argv[argv.index(flag) + 1]


def build_specs(amendment_path: Path, *, seed: int) -> dict[str, dict]:
    if seed not in (0, 1):
        raise ValueError("only sealed source seeds 0 and 1 admitted")
    state = json.loads((QUEUE_ROOT / "state.json").read_text())
    seal = state.get("seal") or {}
    if seal.get("path") != str(LOCK) or _sha(LOCK) != seal.get("sha256"):
        raise PermissionError("R03 effective lock changed")
    current = next((a["sha256"] for a in reversed(state.get("amendments", []))
                    if "d_followup" in a["stages"]), None)
    amendment_path = amendment_path.resolve(strict=True)
    amendment_sha = _sha(amendment_path)
    if current != amendment_sha:
        raise PermissionError("A3 is not the active d_followup amendment")
    dated = json.loads(amendment_path.read_text())
    worker = CAMPAIGN / "latent_repair_v1/attacks/followup_worker.py"
    if (dated.get("schema") != "pcrl-d-followup-amendment-v1"
            or dated.get("status") != "LOCALLY_SEALED_D_FOLLOWUP"
            or dated.get("effective_seal_sha256") != seal["sha256"]
            or dated.get("control_attack_worker_sha256") != _sha(worker)):
        raise PermissionError("A3 control-attack worker seal differs")
    source_unit_id = f"D_FOLLOWUP_CONTROLS_S{seed}"
    source_unit = state.get("units", {}).get(source_unit_id, {})
    if source_unit.get("status") != "done" or source_unit.get("stage") != "d_followup":
        raise PermissionError("five-arm source queue unit is incomplete")
    source_argv = source_unit["spec"]["argv"]
    receipt_path = Path(_arg(source_argv, "public-receipt")).resolve(strict=True)
    receipt_sha = _sha(receipt_path)
    if (source_unit.get("output_sha256", {}).get(str(receipt_path)) != receipt_sha
            or dated.get("control_receipt_sha256_by_seed", {}).get(str(seed)) != receipt_sha):
        raise PermissionError("A3 and queue did not bind the control release receipt")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "latent-repair-followup-controls-receipt-v1"
            or receipt.get("status") != "D_FOLLOWUP_RELEASES_READY"
            or receipt.get("seed") != seed
            or receipt.get("unit_id") != source_unit_id
            or receipt.get("seal_sha256") != seal["sha256"]
            or set(receipt.get("packages", {})) != set(ARMS)
            or set(receipt.get("roles", {})) != set(ROLES)):
        raise ValueError("control source receipt shape/identity differs")
    source_root = Path(_arg(source_argv, "private-output")).resolve(strict=True)
    if ".private" not in source_root.parts or source_root.stat().st_mode & 0o077:
        raise PermissionError("control source must remain owner-only")
    role_manifest = Path(_arg(source_argv, "role-manifest")).resolve(strict=True)
    role_sha = _sha(role_manifest)
    if role_sha != receipt.get("role_manifest_sha256") or role_sha != dated.get("role_manifest_sha256"):
        raise ValueError("admitted role manifest hash differs")
    python = Path(source_argv[0])
    if not python.is_absolute() or not python.is_file():
        raise FileNotFoundError("source queued virtualenv Python missing")
    shared = WORKTREE / "results/pcrl_method_portfolio_v1/shared"
    common_sources = [_source(LOCK, seal["sha256"]),
                      _source(amendment_path, amendment_sha),
                      _source(role_manifest, role_sha),
                      _source(receipt_path, receipt_sha)]
    common_sources.extend(_source(CAMPAIGN / name, _sha(CAMPAIGN / name))
                          for name in SOURCE_CODE)
    common_sources.extend(_source(shared / name, _sha(shared / name))
                          for name in SHARED_CODE)
    specs = {}
    for arm in ARMS:
        unit_id = f"D_ATTACK_{arm}_S{seed}"
        if unit_id in state["units"] or unit_id not in dated.get("authorized_unit_ids", ()):
            raise PermissionError(f"A3 unit missing or already enqueued: {unit_id}")
        private_output = CAMPAIGN / ".private" / f"d_attack_{arm.lower()}_s{seed}"
        public_receipt = CAMPAIGN / "latent_repair_v1/attacks" / f"{unit_id}_RECEIPT.json"
        if private_output.exists() or public_receipt.exists():
            raise FileExistsError(f"control attack output already exists: {unit_id}")
        sources = list(common_sources)
        package = source_root / arm
        sources += [_source(package / "manifest.json",
                            receipt["packages"][arm]["manifest_sha256"]),
                    _source(package / "weights.npz",
                            receipt["packages"][arm]["weights_sha256"])]
        for role in ROLES:
            sources.append(_source(source_root / f"release__{arm}__{role}.npz",
                                   receipt["roles"][role][arm]["private_release_sha256"]))
        argv = [str(python), "-m",
                "results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.attacks.followup_worker",
                "--seal", str(LOCK), "--seal-sha256", seal["sha256"],
                "--attack-amendment", str(amendment_path),
                "--attack-amendment-sha256", amendment_sha,
                "--unit-id", unit_id, "--stage", "d_followup",
                "--role-manifest", str(role_manifest),
                "--role-manifest-sha256", role_sha,
                "--source-private", str(source_root),
                "--source-receipt", str(receipt_path),
                "--source-receipt-sha256", receipt_sha,
                "--source-unit-id", source_unit_id, "--arm-id", arm,
                "--private-output", str(private_output),
                "--public-receipt", str(public_receipt),
                "--seed", str(seed), "--attacker-seed", str(seed)]
        outputs = [public_receipt, private_output / "FROZEN_COMPLETE.json"]
        outputs += [private_output / view / "readers/READERS_RECEIPT.json"
                    for view in ("r1", "r2", "pair")]
        outputs += [private_output / view / "controls/CUSTODY_RECEIPT.json"
                    for view in ("r1", "r2", "pair")]
        specs[arm] = {"unit_id": unit_id, "stage": "d_followup", "argv": argv,
                      "cwd": str(WORKTREE), "max_wall_s": 600,
                      "cpu_reserve_s": 600, "max_rss_gib": 3.0,
                      "projected_output_bytes": 300000000,
                      "sources": sources, "deps": [source_unit_id],
                      "outputs": [str(path) for path in outputs],
                      "env": {"PYTHONHASHSEED": "0"},
                      "amendment_sha256": amendment_sha}
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    specs = build_specs(args.amendment, seed=args.seed)
    spec_dir = CAMPAIGN / ".private/d_attack_specs"
    spec_dir.mkdir(mode=0o700, exist_ok=True)
    if spec_dir.stat().st_mode & 0o077:
        raise PermissionError("queue specification custody is not owner-only")
    for arm, spec in specs.items():
        path = spec_dir / f"{spec['unit_id']}_SPEC.json"
        with path.open("x") as stream:
            json.dump(spec, stream, sort_keys=True, indent=2)
            stream.write("\n")
        path.chmod(0o600)
        print(json.dumps({"arm": arm, "spec_path": str(path),
                          "spec_sha256": _sha(path),
                          "reader_fits": 24 if arm == "F" else 30}, sort_keys=True))


if __name__ == "__main__":
    main()
