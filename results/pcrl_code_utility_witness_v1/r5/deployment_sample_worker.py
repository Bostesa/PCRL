"""Queued deployment behavior check and private sample for offline restore.

This verifies the production raw-input API. Independent endpoint arithmetic
lives in replay_worker.py and never imports this service implementation.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory

import numpy as np

from code_witness import deploy
from code_witness.service import load_service
from results.pcrl_code_utility_witness_v1.r0.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.r01.frozen_u import FrozenUEncoder
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people, sha_file,
)


HERE = Path(__file__).resolve().parent
WIRE = frozenset({"code0", "prob0", "decision0", "code1", "prob1", "decision1"})


def _pin(record: dict, sources: dict[Path, str]) -> Path:
    if set(record) != {"path", "sha256"}:
        raise PermissionError("deployment source record differs")
    original = Path(record["path"])
    if original.is_symlink():
        raise PermissionError("deployment source symlink refused")
    path = original.resolve(strict=True)
    if sources.get(path) != record["sha256"] or sha_file(path) != record["sha256"]:
        raise PermissionError("deployment source not queue byte pinned")
    return path


def _write_npz(path: Path, arrays: dict) -> str:
    if path.exists():
        raise FileExistsError(path)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return sha_file(path)


def run(config_path: Path) -> dict:
    active = assert_active_from_env(stage="verification")
    config_path = config_path.resolve(strict=True)
    if (".private" not in config_path.parts
            or stat.S_IMODE(config_path.stat().st_mode) & 0o077):
        raise PermissionError("owner-only deployment config required")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    sources = {Path(item["path"]).resolve(strict=True): item["sha256"]
               for item in state["units"][active["unit_id"]]["spec"]["sources"]}
    if sources.get(config_path) != sha_file(config_path):
        raise PermissionError("deployment config not registered with queue")
    config = json.loads(config_path.read_text())
    if (config.get("schema") != "pcrl-r5-deployment-sample-config-v1"
            or config.get("unit_id") != active["unit_id"]
            or config.get("seed") not in (0, 1)
            or config.get("bits") not in (10, 16)
            or config.get("arm") != f"T0_B{config['bits']}"):
        raise PermissionError("deployment config identity differs")
    for key in ("worker", "model_source", "service_source", "deploy_source",
                "frozen_u_source", "role_store_source", "dpc_deploy_source"):
        _pin(config[key], sources)
    seal = json.loads(_pin(config["seal"], sources).read_text())
    if config["seal"]["sha256"] != active["seal_sha256"]:
        raise PermissionError("deployment campaign seal differs")
    role_path = _pin(config["role_manifest"], sources)
    if config["role_manifest"]["sha256"] != seal["role_manifest_sha256"]:
        raise PermissionError("deployment role admission differs")
    store = RoleOnlyStore(role_path)
    role_key = "INNER_SELECTION:features"
    feature_path = store.package_root / store.manifest["role_file_relative_path"][role_key]
    if (_pin(config["features"], sources) != feature_path.resolve(strict=True)
            or config["features"]["sha256"] != store.manifest["role_file_sha256"][role_key]):
        raise PermissionError("INNER feature role source differs")
    seed = config["seed"]
    source_path = _pin(config["source_model"], sources)
    source_sha = seal["source_model_files_sha256_by_seed"][str(seed)]["model_pt_sha256"]
    if config["source_model"]["sha256"] != source_sha:
        raise PermissionError("source checkpoint differs from campaign seal")
    producer = json.loads(_pin(config["producer_receipt"], sources).read_text())
    if (producer.get("status") != "COMPLETE" or producer.get("stage") != "fit"
            or producer.get("seed") != seed
            or producer.get("source_model_sha256") != source_sha
            or producer.get("role_manifest_sha256") != config["role_manifest"]["sha256"]
            or producer.get("queue_seal_sha256") != active["seal_sha256"]
            or producer["result"].get("arm_id") != config["arm"]
            or producer["result"].get("bit_count") != config["bits"]):
        raise PermissionError("deployment producer lineage differs")
    manifest_path = _pin(config["service_manifest"], sources)
    weights_path = _pin(config["service_weights"], sources)
    if (manifest_path.with_name("weights.npz") != weights_path
            or producer["result"]["package"]["manifest_sha256"]
               != config["service_manifest"]["sha256"]
            or producer["result"]["package"]["weights_sha256"]
               != config["service_weights"]["sha256"]):
        raise PermissionError("deployment package differs from producer")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("source_schema_sha256") != seal["source_schema_sha256"]
            or manifest.get("source_sha256") != source_sha
            or manifest.get("role_manifest_sha256") != seal["role_manifest_sha256"]
            or manifest.get("arm_id") != config["arm"]
            or manifest.get("source_seed") != seed):
        raise PermissionError("deployment manifest provenance differs")
    release_path = _pin(config["saved_release"], sources)
    role_record = producer["result"]["roles"]["INNER_SELECTION"]
    if (role_record["private_release_sha256"] != config["saved_release"]["sha256"]
            or role_record["private_release_basename"] != release_path.name):
        raise PermissionError("deployment saved release differs from producer")
    with np.load(release_path, allow_pickle=False) as archive:
        if set(archive.files) != WIRE | {"row_id", "unit"}:
            raise ValueError("saved release wire inventory differs")
        saved = {key: np.array(archive[key], copy=True) for key in archive.files}
    features = store.read("INNER_SELECTION", "features")
    assert_same_people(saved, features)
    names = tuple(str(name) for name in features["feature_names"])
    indices = np.linspace(0, len(features["X"]) - 1, num=32, dtype=np.int64)
    if len(np.unique(indices)) != 32:
        raise ValueError("representative indices collide")
    raw = np.ascontiguousarray(features["X"][indices], dtype=np.float32)
    expected = {key: saved[key][indices].copy() for key in WIRE}
    service = load_service(manifest_path,
                           expected_manifest_sha256=config["service_manifest"]["sha256"],
                           expected_source_sha256=source_sha,
                           expected_bits=config["bits"])
    encoder = FrozenUEncoder.load(source_path, source_sha, names,
                                  seal["source_schema_sha256"], seed)
    max_gap = 0.0
    required_metadata = {"schema", "purpose", "bits", "bit_order", "capacity_states",
                         "version", "source_sha256", "model_sha256", "scaler_sha256",
                         "head_sha256", "calibrator_sha256", "calibration_family"}
    for purpose in (0, 1):
        assert_active_from_env(unit_id=active["unit_id"], stage="verification")
        got = service.recipient_from_raw(purpose, raw, names, encoder)
        if set(got) != {"code", "probabilities", "decision", "metadata"}:
            raise ValueError("raw deployment exported unexpected content")
        if set(got["metadata"]) != required_metadata:
            raise ValueError("raw deployment metadata inventory differs")
        if got["metadata"] != role_record["metadata"][purpose]:
            raise ValueError("raw deployment metadata differs from frozen release")
        for source_key, wire_key in (("code", f"code{purpose}"),
                                     ("probabilities", f"prob{purpose}"),
                                     ("decision", f"decision{purpose}")):
            if not np.array_equal(got[source_key], expected[wire_key]):
                raise ValueError(f"raw deployment differs from saved {wire_key}")
            if source_key == "probabilities":
                max_gap = max(max_gap, float(np.max(np.abs(got[source_key]
                                                       - expected[wire_key]))))
    class WrongSource:
        model_sha256 = "0" * 64
        def encode_one(self, *_args, **_kwargs):
            raise AssertionError("wrong source should refuse before encode")
    try:
        service.recipient_from_raw(0, raw, names, WrongSource())
    except ValueError:
        wrong_source_refused = True
    else:
        raise ValueError("wrong source checkpoint was accepted")
    bad_names = list(names)
    bad_names[0], bad_names[1] = bad_names[1], bad_names[0]
    if len(set(bad_names)) != 83:
        raise ValueError("schema control malformed")
    for schema_names, matrix, expected_error in (
        (bad_names, raw[:, np.r_[1, 0, np.arange(2, 83)]], "reordered"),
        (list(names) + ["SEX"], np.column_stack((raw, np.zeros(len(raw), np.float32))),
         "exactly the 83 permitted"),
    ):
        with TemporaryDirectory(dir=Path(config["private_output_dir"]).parent) as temp_dir:
            temp_input = Path(temp_dir) / "bad.npz"
            np.savez_compressed(temp_input, X=matrix,
                                feature_names=np.asarray(schema_names, dtype=str))
            try:
                deploy.load_permitted_input(temp_input, list(names))
            except Exception as exc:
                if expected_error not in str(exc):
                    raise ValueError(f"unexpected schema refusal: {exc}") from exc
            else:
                raise ValueError("invalid raw predictor schema accepted")
    private = Path(config["private_output_dir"]).resolve(strict=False)
    if private.exists() or private.parent != HERE / ".private":
        raise PermissionError("new owner-only R5 sample directory required")
    private.mkdir(mode=0o700)
    raw_file = private / "raw_samples.npz"
    expected_file = private / "expected_wire.npz"
    raw_sha = _write_npz(raw_file, {"X": raw, "feature_names": np.asarray(names, dtype=str)})
    expected_sha = _write_npz(expected_file, expected)
    with TemporaryDirectory(dir=private) as temp_dir:
        schema_path = Path(temp_dir) / "schema.json"
        schema_path.write_text(json.dumps(list(names)))
        cli_path = Path(temp_dir) / "deployed.npz"
        args = ["--service-manifest", str(manifest_path),
                "--service-manifest-sha256", config["service_manifest"]["sha256"],
                "--source-model", str(source_path), "--source-model-sha256", source_sha,
                "--schema", str(schema_path), "--schema-sha256", seal["source_schema_sha256"],
                "--input", str(raw_file), "--output", str(cli_path),
                "--bits", str(config["bits"])]
        with redirect_stdout(io.StringIO()):
            deploy.main(args)
        with np.load(cli_path, allow_pickle=False) as archive:
            if set(archive.files) != WIRE:
                raise ValueError("CLI exported forbidden member")
            if any(not np.array_equal(archive[key], expected[key]) for key in WIRE):
                raise ValueError("CLI output differs from saved release")
    public = {
        "schema": "pcrl-r5-raw-deployment-sample-receipt-v1",
        "status": "PASS", "unit_id": active["unit_id"],
        "scope": "technical restore/deployment sample; no task label, SEX or fresh utility claim",
        "arm": config["arm"], "seed": seed, "bits": config["bits"],
        "rows": len(raw), "source_model_sha256": source_sha,
        "source_schema_sha256": seal["source_schema_sha256"],
        "service_manifest_sha256": config["service_manifest"]["sha256"],
        "service_weights_sha256": config["service_weights"]["sha256"],
        "role_manifest_sha256": config["role_manifest"]["sha256"],
        "raw_samples_sha256": raw_sha, "expected_wire_sha256": expected_sha,
        "raw_vs_saved_max_probability_gap": max_gap,
        "cli_vs_saved_exact": True, "wrong_source_refused": wrong_source_refused,
        "reordered_columns_refused": True, "extra_excluded_column_refused": True,
        "public_wire_members": sorted(WIRE),
        "private_sample_directory_basename": private.name,
    }
    output = Path(config["public_output"]).resolve(strict=False)
    if output.exists() or output.parent != HERE:
        raise PermissionError("new public R5 deployment receipt required")
    output.write_text(json.dumps(public, sort_keys=True, indent=2) + "\n")
    return {"status": "PASS", "unit_id": active["unit_id"],
            "receipt_sha256": sha_file(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True, type=Path)
    print(json.dumps(run(parser.parse_args().config), sort_keys=True))
