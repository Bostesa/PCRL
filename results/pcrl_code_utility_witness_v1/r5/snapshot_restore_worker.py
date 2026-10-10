"""Queue-only restore test using the versioned local snapshot as sole payload.

The bundle is cloned into an ephemeral directory. Deployment subprocesses run
from that clone and receive only model, table and sample paths in the copied
snapshot. Endpoint checking uses independent R5 code and public tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from results.pcrl_code_utility_witness_v1.r0.compute_queue import assert_active_from_env
from .independent_decoder import probabilities_from_hard_bits


HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def _command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None,
             expect_success: bool = True) -> str:
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if expect_success and result.returncode != 0:
        raise RuntimeError(f"snapshot deployment command exited {result.returncode}")
    if not expect_success and result.returncode == 0:
        raise ValueError("invalid snapshot deployment input was accepted")
    return result.stdout.strip()


def _wire(path: Path, bits: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"code", "prob", "decision"}:
            raise ValueError("deployment exported unexpected member")
        result = {key: np.array(archive[key], copy=True) for key in archive.files}
    if (result["code"].dtype != np.uint8
            or result["code"].shape != (32, bits)
            or result["prob"].dtype != np.float64
            or result["decision"].shape != (32,)
            or not np.isin(result["code"], (0, 1)).all()
            or not np.isfinite(result["prob"]).all()
            or np.max(np.abs(result["prob"].sum(axis=1) - 1)) > 1e-8):
        raise ValueError("restored wire shape or values differ")
    return result


def _bank_decode(code: np.ndarray, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    token = (code.astype(np.int64) * (1 << np.arange(10, dtype=np.int64))).sum(axis=1)
    if np.any(token >= len(table)):
        raise ValueError("restored bank code outside table")
    prob = table[token]
    return prob, prob.argmax(axis=1).astype(np.int64)


def run(config_path: Path) -> dict:
    active = assert_active_from_env(stage="verification")
    config_path = config_path.resolve(strict=True)
    if ".private" not in config_path.parts or stat.S_IMODE(config_path.stat().st_mode) & 0o077:
        raise PermissionError("owner-only snapshot restore config required")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    registered = {Path(r["path"]).resolve(strict=True): r["sha256"]
                  for r in state["units"][active["unit_id"]]["spec"]["sources"]}
    if registered.get(config_path) != digest(config_path):
        raise PermissionError("restore config not queue pinned")
    config = json.loads(config_path.read_text())
    if (config.get("schema") != "pcrl-r5-snapshot-restore-config-v1"
            or config.get("unit_id") != active["unit_id"]):
        raise PermissionError("restore unit identity differs")
    snapshot = Path(config["snapshot"]).resolve(strict=True)
    manifest_path = snapshot / "MANIFEST.json"
    if (registered.get(manifest_path) != config["manifest_sha256"]
            or digest(manifest_path) != config["manifest_sha256"]):
        raise PermissionError("snapshot manifest not pinned")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "pcrl-code-utility-local-restore-v1"
            or manifest.get("file_count") != 62 or len(manifest.get("files", [])) != 62):
        raise PermissionError("snapshot inventory differs")
    listed = set()
    bytes_checked = 0
    for record in manifest["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or relative in listed:
            raise PermissionError("snapshot path inventory invalid")
        listed.add(relative)
        original = snapshot / relative
        if original.is_symlink():
            raise PermissionError("snapshot symlink refused")
        path = original.resolve(strict=True)
        if (not path.is_relative_to(snapshot) or not path.is_file()
                or path.stat().st_size != record["size_bytes"]
                or registered.get(path) != record["sha256"]
                or digest(path) != record["sha256"]):
            raise PermissionError("snapshot member byte pin differs")
        bytes_checked += record["size_bytes"]
    if bytes_checked != manifest["total_bytes"]:
        raise PermissionError("snapshot total byte count differs")
    if registered.get(Path(config["campaign_lock"]).resolve(strict=True)) != active["seal_sha256"]:
        raise PermissionError("campaign seal differs")
    with TemporaryDirectory(dir=HERE / ".private", prefix="snapshot_restore_") as temp_name:
        temp = Path(temp_name)
        clone = temp / "checkout"
        if manifest.get("source_branch") != "research/pcrl-code-utility-witness-v1":
            raise PermissionError("snapshot source branch differs")
        _command(["git", "-c", "pack.threads=1", "-c", "index.threads=1",
                  "-c", "checkout.workers=1", "clone", "--quiet",
                  "--branch", manifest["source_branch"],
                  str(snapshot / "git/campaign.bundle"), str(clone)], cwd=temp)
        tip = _command(["git", "rev-parse", "HEAD"], cwd=clone)
        if tip != manifest["git_tip"]:
            raise PermissionError("bundle clone tip differs")
        env = os.environ.copy()
        env.update({"PYTHONPATH": str(clone), "PYTHONNOUSERSITE": "1",
                    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
        source_location = _command([config["python"], "-c",
                                    "import code_witness.deploy; print(code_witness.deploy.__file__)"],
                                   cwd=clone, env=env)
        if not Path(source_location).resolve(strict=True).is_relative_to(clone):
            raise PermissionError("deployment code did not load from bundle clone")
        sample = snapshot / "campaign_private/samples"
        raw = sample / "raw_samples.npz"
        schema = sample / "schema.json"
        with np.load(raw, allow_pickle=False) as archive:
            if set(archive.files) != {"X", "feature_names"} or archive["X"].shape != (32, 83):
                raise ValueError("snapshot raw sample inventory differs")
            feature_names = tuple(str(x) for x in archive["feature_names"])
            raw_matrix = np.array(archive["X"], copy=True)
        schema_record = json.loads(schema.read_text())
        if (set(schema_record) != {"feature_names"}
                or tuple(schema_record["feature_names"]) != feature_names):
            raise ValueError("snapshot ordered schema differs")
        with np.load(sample / "expected_wire.npz", allow_pickle=False) as archive:
            t_expected = {key: np.array(archive[key], copy=True) for key in archive.files}
        with np.load(sample / "expected_bank_wire.npz", allow_pickle=False) as archive:
            b_expected = {key: np.array(archive[key], copy=True) for key in archive.files}
        hcal = snapshot / "sources/hcal"
        qpc = snapshot / "sources/qpc"
        source_model = hcal / "admitted/rel__s1__U/model.pt"
        source_sha = digest(source_model)
        service = (snapshot / "campaign_private/r1/T0_B16_S1_R1" /
                   "service__T0_B16__s1/manifest.json")
        service_manifest = json.loads(service.read_text())
        if (service_manifest.get("source_sha256") != source_sha
                or service_manifest.get("source_schema_sha256")
                   != config["schema_sha256"]):
            raise PermissionError("snapshot service source/schema differs")
        weights = service.with_name("weights.npz")
        with np.load(weights, allow_pickle=False) as archive:
            state = {key: np.array(archive[key], copy=True) for key in archive.files}
        max_t_gap = 0.0
        for purpose in (0, 1):
            assert_active_from_env(unit_id=active["unit_id"], stage="verification")
            output = temp / f"T0_{purpose}.npz"
            t_command = [config["python"], "-m", "code_witness.deploy",
                         "--service-manifest", str(service),
                         "--service-manifest-sha256", digest(service),
                         "--source-model", str(source_model),
                         "--source-model-sha256", source_sha,
                         "--schema", str(schema), "--schema-sha256",
                         config["schema_sha256"], "--input", str(raw),
                         "--output", str(output), "--bits", "16",
                         "--purpose", str(purpose)]
            _command(t_command, cwd=clone, env=env)
            wire = _wire(output, 16)
            if (not np.array_equal(wire["code"], t_expected[f"code{purpose}"])
                    or not np.array_equal(wire["decision"], t_expected[f"decision{purpose}"])):
                raise ValueError("snapshot T0 bits or decision differ")
            independent = probabilities_from_hard_bits(
                state, wire["code"], purpose, 16,
                float(service_manifest["calibrators"][purpose]["alpha"]))
            if not np.array_equal(wire["prob"], independent):
                raise ValueError("snapshot T0 probability not decoded from emitted bits")
            gap = float(np.max(np.abs(wire["prob"] - t_expected[f"prob{purpose}"])))
            if gap > 2e-7:
                raise ValueError("snapshot T0 saved probability gap exceeds A06")
            max_t_gap = max(max_t_gap, gap)
            wrong = t_command.copy()
            wrong[wrong.index("--source-model-sha256") + 1] = "0" * 64
            wrong[wrong.index("--output") + 1] = str(temp / f"wrong_source_T0_{purpose}.npz")
            _command(wrong, cwd=clone, env=env, expect_success=False)
        public = snapshot / "public_tables/W_BANK_PUBLIC_TABLES_S1.json"
        bank_tables = json.loads(public.read_text())
        max_bank_gap = 0.0
        for bank in ("W_Q", "W_TSTAR"):
            for purpose in (0, 1):
                assert_active_from_env(unit_id=active["unit_id"], stage="verification")
                output = temp / f"{bank}_{purpose}.npz"
                b_command = [config["python"], "-m", "code_witness.bank_deploy",
                             "--bank", bank, "--purpose", str(purpose), "--seed", "1",
                             "--source-root", str(hcal), "--source-sha256", source_sha,
                             "--qpc-root", str(qpc), "--public-table", str(public),
                             "--public-table-sha256", digest(public),
                             "--schema", str(schema), "--schema-sha256",
                             config["schema_sha256"], "--input", str(raw),
                             "--output", str(output)]
                _command(b_command, cwd=clone, env=env)
                wire = _wire(output, 10)
                prefix = f"{bank}_"
                table = np.asarray(bank_tables["families"][bank][f"table{purpose}"],
                                   dtype=np.float64)
                decoded_prob, decoded_decision = _bank_decode(wire["code"], table)
                if (not np.array_equal(wire["code"], b_expected[f"{prefix}code{purpose}"])
                        or not np.array_equal(wire["prob"], b_expected[f"{prefix}prob{purpose}"])
                        or not np.array_equal(wire["decision"], b_expected[f"{prefix}decision{purpose}"])
                        or not np.array_equal(wire["prob"], decoded_prob)
                        or not np.array_equal(wire["decision"], decoded_decision)):
                    raise ValueError("snapshot bank wire/table differs")
                max_bank_gap = max(max_bank_gap, float(np.max(np.abs(
                    wire["prob"] - b_expected[f"{prefix}prob{purpose}"]))))
        # Invalid ordered schema, excluded extra field and wrong source are all
        # refused by restored code, using synthetic modifications of sample rows.
        bad_schema = temp / "reordered_schema.json"
        bad_names = list(feature_names)
        bad_names[:2] = bad_names[1::-1]
        bad_schema.write_text(json.dumps({"feature_names": bad_names}))
        bad_output = temp / "bad_schema.npz"
        bad = b_command.copy()
        bad[bad.index("--schema") + 1] = str(bad_schema)
        bad[bad.index("--output") + 1] = str(bad_output)
        _command(bad, cwd=clone, env=env, expect_success=False)
        extra = temp / "extra_sex.npz"
        np.savez_compressed(extra, X=np.column_stack((raw_matrix, np.zeros(32, np.float32))),
                            feature_names=np.asarray([*feature_names, "SEX"], dtype=str))
        bad_extra = b_command.copy()
        bad_extra[bad_extra.index("--input") + 1] = str(extra)
        bad_extra[bad_extra.index("--output") + 1] = str(temp / "bad_extra.npz")
        _command(bad_extra, cwd=clone, env=env, expect_success=False)
        wrong_bank = b_command.copy()
        wrong_bank[wrong_bank.index("--source-sha256") + 1] = "0" * 64
        wrong_bank[wrong_bank.index("--output") + 1] = str(temp / "wrong_bank.npz")
        _command(wrong_bank, cwd=clone, env=env, expect_success=False)
    result = {"schema": "pcrl-r5-independent-snapshot-restore-v1", "status": "PASS",
              "unit_id": active["unit_id"], "manifest_sha256": config["manifest_sha256"],
              "git_bundle_sha256": digest(snapshot / "git/campaign.bundle"),
              "git_tip": manifest["git_tip"], "files_checked": len(listed),
              "bytes_checked": bytes_checked, "code_loaded_from_bundle_clone": True,
              "snapshot_payload_only": True, "sample_rows": 32,
              "t0_b16_seed1_max_saved_probability_gap": max_t_gap,
              "t0_b16_exact_bits_decisions_and_independent_code_decode": True,
              "bank_seed1_max_saved_probability_gap": max_bank_gap,
              "bank_exact_code_probability_decision_and_public_table": True,
              "invalid_source_reordered_schema_extra_sex_refused": True,
              "scope": "same-device snapshot restore; no task or protected labels accessed"}
    output = Path(config["public_output"]).resolve(strict=False)
    if output.exists() or output.parent != HERE:
        raise PermissionError("new public restore receipt required")
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return {"status": "PASS", "receipt_sha256": digest(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().config), sort_keys=True))
