"""Queued, one-row D-K deployment acceptance from admitted role-only features.

This worker writes one private receipt. It opens no label, SEX or mixed-role
source and must run under R03's inherited locked process slot.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.service import load_service
from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.r18.acceptance_contract import (
    FEATURE_NAMES_SHA256, check_d_public_release, check_raw_schema,
    check_source_checkpoint, sha256_file,
)
from results.pcrl_latent_and_sampled_methods_v1.r18.independent_d_head import IndependentDHead
from results.pcrl_latent_and_sampled_methods_v1.r18.role_input import load_one_admitted_feature_row


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parents[2]
ROLE = "INNER_SELECTION"
ARM = "D-K"


def _queue_spec(unit_id: str) -> tuple[dict, dict, str]:
    active = assert_active_from_env(unit_id=unit_id, stage="d_followup")
    root = Path(os.environ["PCRL_QUEUE_ROOT"]).resolve(strict=True)
    state = json.loads((root / "state.json").read_text())
    unit = state["units"][unit_id]
    spec = unit["spec"]
    if (unit["status"] != "running" or unit["spec_sha256"] != active["spec_sha256"]
            or spec["unit_id"] != unit_id or spec["stage"] != "d_followup"
            or Path(spec["cwd"]).resolve() != Path.cwd().resolve()
            or Path(spec["argv"][0]).resolve() != Path(sys.executable).resolve()
            or len(spec["argv"]) < 2
            or Path(spec["argv"][1]).resolve() != Path(__file__).resolve()
            or spec["argv"][1:] != sys.argv):
        raise PermissionError("R18 D worker invocation differs from active queue spec")
    return state, spec, active["seal_sha256"]


def _registered_sha(spec: dict, path: Path) -> str:
    target = Path(path).resolve(strict=True)
    if path.is_symlink() or not target.is_file():
        raise PermissionError("R18 pinned source is not a plain file")
    matches = [record["sha256"] for record in spec["sources"]
               if Path(record["path"]).resolve(strict=True) == target]
    if len(matches) != 1 or sha256_file(target) != matches[0]:
        raise PermissionError("R18 input is absent from exact queued source pins")
    return matches[0]


def _encoder(source_path: Path):
    model_spec = importlib.util.find_spec("jcv.train")
    if (model_spec is None or model_spec.origin is None
            or Path(model_spec.origin).resolve() != (CAMPAIGN_ROOT / "jcv" / "train.py").resolve()):
        raise PermissionError("frozen U architecture import differs from the campaign pin")
    module_spec = importlib.util.spec_from_file_location("r18_frozen_u_encoder", source_path)
    if module_spec is None or module_spec.loader is None:
        raise PermissionError("pinned U encoder source cannot be loaded")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.FrozenUEncoder


def _flag(argv: list[str], name: str) -> str:
    if argv.count(name) != 1:
        raise PermissionError("D first-fit argument provenance differs")
    index = argv.index(name)
    if index + 1 >= len(argv):
        raise PermissionError("D first-fit argument value is absent")
    return argv[index + 1]


def _plain_release(value: dict) -> dict:
    return {"code": np.asarray(value["code"], dtype=np.uint8).tolist(),
            "probabilities": np.asarray(value["probabilities"], dtype=np.float64).tolist(),
            "decision": np.asarray(value["decision"], dtype=np.int64).tolist(),
            "metadata": dict(value["metadata"])}


def _write_private_once(path: Path, value: dict) -> None:
    parent = path.parent
    if (parent.resolve(strict=True) != (HERE / ".private" / "deploy").resolve(strict=True)
            or stat.S_IMODE(parent.stat().st_mode) != 0o700 or path.exists()):
        raise PermissionError("R18 output must be fresh in its private deployment directory")
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def run(args: argparse.Namespace) -> dict:
    state, spec, seal_sha = _queue_spec(args.unit_id)
    paths = {name: Path(getattr(args, name)).resolve(strict=True) for name in (
        "fit_receipt", "role_manifest", "source_checkpoint", "service_manifest",
        "encoder_source")}
    for name, path in paths.items():
        _registered_sha(spec, path)
    for source in (
        Path(__file__), HERE / "acceptance_contract.py",
        HERE / "independent_d_head.py", HERE / "role_input.py",
        HERE.parent / "latent_repair_v1" / "impl" / "model.py",
        HERE.parent / "latent_repair_v1" / "impl" / "service.py",
        HERE.parent / "r03" / "compute_queue.py", CAMPAIGN_ROOT / "jcv" / "train.py",
    ):
        _registered_sha(spec, source)
    output = Path(args.private_output).absolute()
    if spec["outputs"] != [str(output)] or output.exists() or args.row_index < 0:
        raise PermissionError("R18 D output or row selection differs from queued plan")
    seal = state["seal"]
    lock = json.loads(Path(seal["path"]).read_text())
    if (lock.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1"
            or lock.get("status") != "LOCALLY_SEALED_D_STAGE"
            or sha256_file(Path(seal["path"])) != seal_sha):
        raise PermissionError("R00 D seal is not effective")
    implementation = lock["implementation_source_sha256"]
    locked_sources = {
        "results/pcrl_latent_and_sampled_methods_v1/r01/frozen_u.py": paths["encoder_source"],
        "results/pcrl_latent_and_sampled_methods_v1/latent_repair_v1/impl/model.py":
            HERE.parent / "latent_repair_v1" / "impl" / "model.py",
        "results/pcrl_latent_and_sampled_methods_v1/latent_repair_v1/impl/service.py":
            HERE.parent / "latent_repair_v1" / "impl" / "service.py",
        "jcv/train.py": CAMPAIGN_ROOT / "jcv" / "train.py",
    }
    if any(Path(path).resolve() != (CAMPAIGN_ROOT / name).resolve()
           or _registered_sha(spec, path) != implementation.get(name)
           for name, path in locked_sources.items()):
        raise PermissionError("D deployment encoder, model or service differs from the effective lock")
    fit_sha = _registered_sha(spec, paths["fit_receipt"])
    fit = json.loads(paths["fit_receipt"].read_text())
    source = lock["role_identity_and_file_hashes"]
    role_sha = _registered_sha(spec, paths["role_manifest"])
    source_sha = source["source_model_sha256"]["0"]
    package = fit.get("packages", {}).get(ARM, {})
    if (fit.get("schema") != "latent-repair-first-paired-seed-receipt-v1"
            or fit.get("status") != "T_DK_RELEASES_READY"
            or fit.get("seed") != 0
            or fit.get("seal_sha256") != seal_sha
            or fit.get("role_manifest_sha256") != role_sha
            or fit.get("source_sha256") != source_sha
            or source["ordered_predictors_sha256"] != FEATURE_NAMES_SHA256
            or lock["source_files_sha256"]["source_role_manifest"] != role_sha
            or package.get("manifest_sha256") != _registered_sha(spec, paths["service_manifest"])
            or package.get("weights_sha256") != _registered_sha(
                spec, paths["service_manifest"].with_name("weights.npz"))
            or _registered_sha(spec, paths["source_checkpoint"]) != source_sha):
        raise PermissionError("D-K postfit receipt, service or source pin differs")
    parent = state["units"].get(fit.get("unit_id"), {})
    if (parent.get("status") != "done" or fit["unit_id"] not in spec["deps"]
            or parent.get("output_sha256", {}).get(str(paths["fit_receipt"])) != fit_sha
            or parent.get("output_sha256", {}).get(str(paths["service_manifest"]))
            != package["manifest_sha256"]):
        raise PermissionError("D-K fit receipt is not a completed queued parent")
    parent_argv = parent["spec"]["argv"]
    if (Path(_flag(parent_argv, "--role-manifest")).resolve() != paths["role_manifest"]
            or Path(_flag(parent_argv, "--model-pt")).resolve() != paths["source_checkpoint"]
            or _flag(parent_argv, "--source-sha256") != source_sha
            or _flag(parent_argv, "--role-manifest-sha256") != role_sha
            or Path(_flag(parent_argv, "--private-output")).resolve()
            != paths["service_manifest"].parents[1]
            or paths["service_manifest"].name != "manifest.json"
            or paths["service_manifest"].parent.name != ARM):
        raise PermissionError("D-K service/input paths differ from completed first-fit argv")
    admitted = json.loads(paths["role_manifest"].read_text())
    if (admitted.get("role_file_sha256") != source["role_file_sha256"]
            or admitted.get("roles") != source["role_membership"]
            or admitted.get("source_schema_sha256") != FEATURE_NAMES_SHA256):
        raise PermissionError("role-only source does not match effective lock")
    service_record = json.loads(paths["service_manifest"].read_text())
    if (service_record.get("arm_id") != ARM or service_record.get("source_seed") != 0
            or service_record.get("role_manifest_sha256") != role_sha
            or service_record.get("source_sha256") != source_sha
            or service_record.get("model_sha256") != package.get("model_sha256")):
        raise PermissionError("D-K service manifest differs from queued fit output")
    check_source_checkpoint(paths["source_checkpoint"], source_sha)
    restored = load_service(paths["service_manifest"],
                            expected_manifest_sha256=package["manifest_sha256"],
                            expected_source_sha256=source_sha)
    independent = IndependentDHead(paths["service_manifest"],
                                   manifest_sha256=package["manifest_sha256"],
                                   weights_sha256=package["weights_sha256"],
                                   source_sha256=source_sha,
                                   model_sha256=package["model_sha256"])
    role_root = Path(args.role_package_root).resolve(strict=True)
    export_path = role_root / admitted["role_file_relative_path"][f"{ROLE}:features"]
    if (_registered_sha(spec, export_path) != source["role_file_sha256"][f"{ROLE}:features"]
            or not paths["role_manifest"].is_relative_to(role_root)):
        raise PermissionError("one-row role export is not the sealed INNER feature file")
    row, columns, row_id, unit = load_one_admitted_feature_row(
        role_root, paths["role_manifest"], role_sha, ROLE, args.row_index)
    check_raw_schema(row, columns, columns, FEATURE_NAMES_SHA256)
    row_bytes = np.ascontiguousarray(row, dtype="<f4").tobytes()
    source_encoder = _encoder(paths["encoder_source"]).load(
        paths["source_checkpoint"], source_sha, columns, FEATURE_NAMES_SHA256, 0)
    releases = {}
    for purpose in (0, 1):
        first = restored.recipient_from_raw(purpose, row, columns, source_encoder)
        repeated = restored.recipient_from_raw(purpose, row, columns, source_encoder)
        expected = independent.expected_metadata(purpose)
        check_d_public_release(first, purpose, expected, independent.probabilities)
        check_d_public_release(repeated, purpose, expected, independent.probabilities)
        if _plain_release(first) != _plain_release(repeated):
            raise ValueError("D one-row service is not deterministically reproducible")
        releases[str(purpose)] = _plain_release(first)
    result = {"schema": "pcrl-r18-real-d-deployment-acceptance-v1",
              "status": "REAL_D_K_ONE_ROW_PASS", "unit_id": args.unit_id,
              "role": ROLE, "row_index": args.row_index, "row_id": row_id, "group_id": unit,
              "seal_sha256": seal_sha, "fit_receipt_sha256": fit_sha,
              "service_manifest_sha256": package["manifest_sha256"],
              "source_checkpoint_sha256": source_sha,
              "role_manifest_sha256": role_sha,
              "input_row_sha256": hashlib.sha256(row_bytes).hexdigest(),
              "input_row_float32_le_hex": row_bytes.hex(),
              "input_columns": list(columns),
              "releases": releases,
              "checks": {"role_only_83_ordered": True, "excluded_columns_absent": True,
                         "source_checkpoint_exact": True,
                         "code_only_calibrated_reconstruction_exact": True,
                         "repeat_release_exact": True,
                         "hidden_wire_fields_absent": True}}
    _write_private_once(output, result)
    return {"status": result["status"], "unit_id": args.unit_id,
            "output_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--fit-receipt", required=True)
    parser.add_argument("--role-package-root", required=True)
    parser.add_argument("--role-manifest", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--service-manifest", required=True)
    parser.add_argument("--encoder-source", required=True)
    parser.add_argument("--row-index", required=True, type=int)
    parser.add_argument("--private-output", required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
