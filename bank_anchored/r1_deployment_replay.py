"""Queue-only real CLI replay of a hash-bound selected bank release.

This worker uses R1's admitted 32-row raw FIT sample and runs the actual
one-recipient CLI twice. It publishes only aggregate evidence and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from zipfile import ZipFile

import numpy as np

from code_witness.bank import bits_to_tokens
from code_witness.bank_deploy import load_bank_service
from dpc.deploy import load_permitted_input, schema_names
from results.pcrl_bank_anchored_release_v1.r0.compute_queue import assert_active_from_env
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_laws import offline_factor_uniforms
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_package import load_policy
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.release import offline_uniform

from .r1_runtime import release_pair_from_raw


WIRE_KEYS = frozenset({"code", "prob", "decision"})
STDOUT_KEYS = frozenset({"schema", "arm_id", "seed", "purpose", "rows",
                         "policy_receipt_sha256", "public_table_sha256",
                         "release_sha256"})


def derive_uniforms(root: bytes | None, *, seed: int, arm_id: str,
                    units: np.ndarray) -> np.ndarray | None:
    """Recreate realization-0 cached FIT coins from the sealed unit stream."""
    if arm_id in ("B0", "D"):
        return None
    value = np.asarray(units)
    if (arm_id not in ("J", "T", "P", "L", "S12", "S21") or
            type(root) is not bytes or len(root) != 32 or seed not in (0, 1) or
            value.dtype != np.int64 or value.ndim != 1 or len(value) != 32 or
            np.any(value < 0)):
        raise ValueError("cached FIT coin source or unit schema differs")
    if arm_id in ("P", "L"):
        return np.asarray([offline_factor_uniforms(root, source_seed=seed,
                           role="OSF_DEFENSE_FIT", realization=0, unit=int(unit))
                           for unit in value], dtype=np.float64)
    return np.asarray([offline_uniform(root, source_seed=seed,
                       role="OSF_DEFENSE_FIT", realization=0, unit=int(unit))
                       for unit in value], dtype=np.float64)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pin(path_text: str, sources: dict[Path, str]) -> Path:
    original = Path(path_text)
    if original.is_symlink():
        raise PermissionError("source symlink refused")
    path = original.resolve(strict=True)
    if sources.get(path) != _sha(path):
        raise PermissionError("deployment source is not byte-pinned by queue")
    return path


def audit_wire(path: Path, table: np.ndarray, baseline: Path, *,
               identity: bool) -> dict:
    """Require the exact 10-bit fixed-table wire and unchanged task decisions."""
    fixed = np.asarray(table)
    if (fixed.dtype != np.float64 or fixed.ndim != 2 or
            not 0 < len(fixed) <= 1024 or
            len(np.unique(fixed, axis=0)) != len(fixed) or
            not np.isfinite(fixed).all()):
        raise ValueError("fixed public bank table differs")
    with ZipFile(path) as archive:
        if {item.filename for item in archive.infolist()} != {
                f"{key}.npy" for key in WIRE_KEYS}:
            raise ValueError("emitted wire has hidden field")
    with np.load(path, allow_pickle=False) as archive:
        wire = {key: np.array(archive[key], copy=True) for key in WIRE_KEYS}
    code, prob, decision = (wire[key] for key in ("code", "prob", "decision"))
    if (code.dtype != np.uint8 or code.ndim != 2 or code.shape[1] != 10 or
            not np.all((code == 0) | (code == 1)) or
            prob.dtype != np.float64 or prob.shape != (len(code), fixed.shape[1]) or
            decision.dtype != np.int64 or decision.shape != (len(code),)):
        raise ValueError("emitted wire type or shape differs")
    tokens = bits_to_tokens(code)
    if (np.any(tokens >= len(fixed)) or
            not np.array_equal(prob, fixed[tokens]) or
            not np.array_equal(decision, fixed[tokens].argmax(axis=1))):
        raise ValueError("emitted wire is not decoded from fixed bank table")
    with ZipFile(baseline) as archive:
        if {item.filename for item in archive.infolist()} != {
                f"{key}.npy" for key in WIRE_KEYS}:
            raise ValueError("B0 baseline wire has hidden field")
    with np.load(baseline, allow_pickle=False) as archive:
        old = {key: np.array(archive[key], copy=True) for key in WIRE_KEYS}
    if not np.array_equal(decision, old["decision"]):
        raise ValueError("selected channel changed frozen task decision")
    if identity and any(not np.array_equal(wire[key], old[key]) for key in WIRE_KEYS):
        raise ValueError("B0 selected CLI differs from original bank identity")
    return {"rows": len(code), "distinct_public_codes": int(len(np.unique(tokens))),
            "fixed_table_decode": "PASS", "frozen_decision": "PASS",
            "exact_B0_identity": "PASS" if identity else "NOT_APPLICABLE"}


def run(config_path: Path) -> dict:
    active = assert_active_from_env(stage="verification")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    registered = {Path(item["path"]).resolve(strict=True): item["sha256"]
                  for item in state["units"][active["unit_id"]]["spec"]["sources"]}
    config_path = _pin(str(config_path), registered)
    if ".private" not in config_path.parts or stat.S_IMODE(config_path.stat().st_mode) & 0o077:
        raise PermissionError("owner-only private replay config required")
    cfg = json.loads(config_path.read_text())
    if (cfg.get("schema") != "pcrl-bank-r1-deployment-replay-config-v1" or
            cfg.get("unit_id") != active["unit_id"] or
            cfg.get("seed") not in (0, 1) or
            cfg.get("arm_id") not in ("B0", "T", "D", "P", "L", "J", "S12", "S21")):
        raise ValueError("deployment replay identity differs")
    pins = {name: _pin(value, registered) for name, value in cfg["pins"].items()}
    required = {"source_config", "source_receipt", "raw_sample", "schema_file",
                "b0_r0", "b0_r1", "public_table", "policy_receipt", "policy_npz",
                "campaign_lock", "role_features", "cached_fit_release",
                "cached_fit_receipt"}
    stochastic = cfg["arm_id"] not in ("B0", "D")
    if set(pins) != required | ({"stream_roots"} if stochastic else set()):
        raise ValueError("deployment replay source set differs")
    source = json.loads(pins["source_config"].read_text())
    witness = json.loads(pins["source_receipt"].read_text())
    policy_receipt = json.loads(pins["policy_receipt"].read_text())
    lock = json.loads(pins["campaign_lock"].read_text())
    cached_receipt = json.loads(pins["cached_fit_receipt"].read_text())
    if (source.get("seed") != cfg["seed"] or source.get("unit_id") !=
            f"R1_BANK_S{cfg['seed']}_FIT_SAMPLE32" or
            source["pins"]["public_table"] != str(pins["public_table"]) or
            witness.get("status") != "PASS" or witness.get("seed") != cfg["seed"] or
            witness.get("sample_input_sha256") != _sha(pins["raw_sample"]) or
            witness.get("new_identity_runtime_exact_parity") is not True or
            policy_receipt.get("arm_id") != cfg["arm_id"] or
            policy_receipt.get("source_seed") != cfg["seed"] or
            policy_receipt.get("source_binding_sha256") != cfg["source_binding_sha256"] or
            policy_receipt.get("policy_file_sha256") != _sha(pins["policy_npz"]) or
            _sha(pins["policy_receipt"]) != cfg["policy_receipt_sha256"] or
            _sha(pins["campaign_lock"]) != active["seal_sha256"] or
            lock.get("selected_bank") != "W_TSTAR" or
            source["pins"]["role_features"] != str(pins["role_features"]) or
            cached_receipt.get("schema") != "BAR1-role-release-v1" or
            cached_receipt.get("role") != "OSF_DEFENSE_FIT" or
            cached_receipt.get("realization") != 0 or
            cached_receipt.get("arm_id") != cfg["arm_id"] or
            cached_receipt.get("source_seed") != cfg["seed"] or
            cached_receipt.get("source_binding_sha256") != cfg["source_binding_sha256"] or
            cached_receipt.get("policy_receipt_sha256") != cfg["policy_receipt_sha256"] or
            cached_receipt.get("release_codes_sha256") != _sha(pins["cached_fit_release"])):
        raise ValueError("raw sample, bank, or selected policy source differs")
    for name, value in source["pins"].items():
        if _pin(value, registered) != Path(value).resolve(strict=True):
            raise ValueError(f"frozen source {name} differs")
    public = json.loads(pins["public_table"].read_text())
    if (public.get("schema") != "code-utility-public-bank-tables-v1" or
            public.get("seed") != cfg["seed"] or
            public.get("bit_count") != 10 or
            public.get("families", {}).get("W_TSTAR") is None):
        raise ValueError("selected public table differs")
    indices = np.asarray(source["indices"], dtype=np.int64)
    if (indices.shape != (32,) or not np.array_equal(
            indices, np.linspace(0, 15433, 32, dtype=np.int64))):
        raise ValueError("frozen 32-row FIT sample positions differ")
    with ZipFile(pins["role_features"]) as archive:
        if {item.filename for item in archive.infolist()} != {
                "row_id.npy", "unit.npy", "X.npy", "feature_names.npy"}:
            raise ValueError("role-only FIT feature archive fields differ")
    with np.load(pins["role_features"], allow_pickle=False) as archive:
        row_id = np.asarray(archive["row_id"])
        unit = np.asarray(archive["unit"])
        x = np.asarray(archive["X"])
    names = tuple(schema_names(pins["schema_file"]))
    raw = load_permitted_input(pins["raw_sample"], list(names))
    if (row_id.dtype != np.int64 or unit.dtype != np.int64 or
            row_id.shape != (15434,) or unit.shape != row_id.shape or
            x.shape != (15434, 83) or not np.array_equal(raw, x[indices])):
        raise ValueError("raw sample does not reproduce admitted FIT rows")
    with ZipFile(pins["cached_fit_release"]) as archive:
        if {item.filename for item in archive.infolist()} != {
                "row_id.npy", "unit.npy", "code_r1.npy", "code_r2.npy"}:
            raise ValueError("R2 cached FIT release has hidden field")
    with np.load(pins["cached_fit_release"], allow_pickle=False) as archive:
        if (not np.array_equal(archive["row_id"], row_id) or
                not np.array_equal(archive["unit"], unit)):
            raise ValueError("R2 cached FIT people differ from raw sample")
        cached = {r: np.array(archive[f"code_r{r}"][indices], copy=True)
                  for r in (1, 2)}
    root = None
    if stochastic:
        roots = json.loads(pins["stream_roots"].read_text())
        if (roots.get("schema") != "scv1-private-stream-roots-v1" or
                _sha(pins["stream_roots"]) !=
                lock["sampling_root_hashes"]["private_roots_source_sha256"]):
            raise ValueError("sealed cached release roots differ")
        root = bytes.fromhex(roots["release_realization_roots_hex"]["0"])
        if (len(root) != 32 or hashlib.sha256(root).hexdigest() !=
                lock["sampling_root_hashes"]["joint_SCv1"][0]):
            raise ValueError("realization-0 root hash differs")
    uniforms = derive_uniforms(root, seed=cfg["seed"], arm_id=cfg["arm_id"],
                               units=unit[indices])
    service = load_bank_service(
        source_root=Path(source["source_root"]), qpc_root=Path(source["qpc_root"]),
        table_path=pins["public_table"],
        expected_table_sha256=_sha(pins["public_table"]),
        expected_source_sha256=witness["source_model_sha256"],
        expected_schema_sha256=source["schema_sha256"],
        ordered_names=names, bank="W_TSTAR", seed=cfg["seed"])
    policy, _ = load_policy(pins["policy_receipt"].parent,
                            receipt_sha256=cfg["policy_receipt_sha256"])
    runtime_uniforms = (np.full(32, 0.5, dtype=np.float64)
                        if cfg["arm_id"] == "D" else uniforms)
    preflight = release_pair_from_raw(service, policy, raw, names,
                                       runtime_uniforms)
    if any(not np.array_equal(bits_to_tokens(preflight[r-1]["code"]), cached[r])
           for r in (1, 2)):
        raise ValueError("reconstructed FIT cached release differs from raw provider route")
    output = Path(cfg["private_output_dir"])
    if (".private" not in output.parts or output.exists() or
            not output.parent.is_dir() or
            stat.S_IMODE(output.parent.stat().st_mode) & 0o077):
        raise PermissionError("new owner-only replay output directory required")
    output.mkdir(mode=0o700)
    draws = None
    if uniforms is not None:
        draws = output / "cached_fit_uniforms.npz"
        with os.fdopen(os.open(draws, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                       "wb") as stream:
            np.savez_compressed(stream, uniforms=uniforms)
            stream.flush(); os.fsync(stream.fileno())
    wire_receipts = {}
    for purpose in (0, 1):
        assert_active_from_env(unit_id=active["unit_id"], stage="verification")
        wire = output / f"recipient{purpose}.npz"
        command = [cfg["python"], "-m", "bank_anchored.r1_deploy",
                   "--purpose", str(purpose), "--seed", str(cfg["seed"]),
                   "--source-root", source["source_root"],
                   "--source-sha256", witness["source_model_sha256"],
                   "--qpc-root", source["qpc_root"],
                   "--public-table", str(pins["public_table"]),
                   "--public-table-sha256", _sha(pins["public_table"]),
                   "--schema", str(pins["schema_file"]),
                   "--schema-sha256", source["schema_sha256"],
                   "--policy-dir", str(pins["policy_receipt"].parent),
                   "--policy-receipt-sha256", cfg["policy_receipt_sha256"],
                   "--source-binding-sha256", cfg["source_binding_sha256"],
                   "--input", str(pins["raw_sample"]), "--output", str(wire)]
        if draws is not None:
            command.extend(("--draws", str(draws),
                            "--draws-sha256", _sha(draws)))
        result = subprocess.run(command, cwd=cfg["cwd"], check=True,
                                capture_output=True, text=True, timeout=90)
        stdout = json.loads(result.stdout)
        if (set(stdout) != STDOUT_KEYS or
                stdout["schema"] != "pcrl-bank-anchored-deployment-v1" or
                stdout["arm_id"] != cfg["arm_id"] or
                stdout["seed"] != cfg["seed"] or
                stdout["purpose"] != purpose or
                stdout["rows"] != 32 or
                stdout["policy_receipt_sha256"] != cfg["policy_receipt_sha256"] or
                stdout["public_table_sha256"] != _sha(pins["public_table"]) or
                stdout["release_sha256"] != _sha(wire)):
            raise ValueError("selected CLI aggregate receipt differs")
        fixed = np.asarray(public["families"]["W_TSTAR"][f"table{purpose}"],
                           dtype=np.float64)
        audited = audit_wire(wire, fixed, pins[f"b0_r{purpose}"],
                             identity=cfg["arm_id"] == "B0")
        if audited["rows"] != 32:
            raise ValueError("selected CLI row count differs")
        wire_receipts[str(purpose)] = {**audited, "release_sha256": _sha(wire),
                                       "cli_stdout": stdout}
    receipt = {
        "schema": "pcrl-bank-r1-selected-deployment-replay-v1", "status": "PASS",
        "unit_id": active["unit_id"], "seed": cfg["seed"], "arm_id": cfg["arm_id"],
        "rows": 32, "source_lock_sha256": active["seal_sha256"],
        "source_witness_receipt_sha256": _sha(pins["source_receipt"]),
        "policy_receipt_sha256": cfg["policy_receipt_sha256"],
        "public_table_sha256": _sha(pins["public_table"]),
        "cached_fit_release_sha256": _sha(pins["cached_fit_release"]),
        "private_draws_sha256": _sha(draws) if draws is not None else None,
        "reconstructed_cached_code_parity": True,
        "wire_by_recipient": wire_receipts,
        "scope": "Actual selected-policy CLI on frozen 32-row raw FIT sample; no labels or coins emitted",
    }
    private_receipt = output / "RECEIPT.json"
    private_receipt.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    os.chmod(private_receipt, 0o600)
    public_receipt = Path(cfg["public_receipt"])
    expected = (Path(cfg["cwd"]) / "results/pcrl_bank_anchored_release_v1/r1" /
                f"R1_DEPLOY_S{cfg['seed']}_{cfg['arm_id']}_RECEIPT.json")
    if public_receipt.resolve(strict=False) != expected.resolve(strict=False):
        raise PermissionError("public receipt must stay in R1 namespace")
    fd = os.open(public_receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), sort_keys=True))


if __name__ == "__main__":
    main()
