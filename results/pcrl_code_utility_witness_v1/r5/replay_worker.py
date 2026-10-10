"""Queued, independent INNER task/release replay from role-only archives.

The owner-only config lists exact byte-pinned inputs. This verifier imports no
candidate service, training, runner endpoint, attack, or selection module.
It writes aggregates and hashes only; no row vectors leave private memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np

from results.pcrl_code_utility_witness_v1.r0.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import RoleOnlyStore
from .independent_checks import task_means, task_rows
from .independent_release import (FEATURE_MEMBERS, HARD_MEMBERS, replay_bank,
                                  replay_feature, replay_hard, tensor_digest)


HERE = Path(__file__).resolve().parent
SCHEMA = "pcrl-r5-independent-inner-replay-config-v1"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _private(path: Path) -> None:
    value = path.resolve(strict=True)
    if (".private" not in value.parts or stat.S_IMODE(value.stat().st_mode) & 0o077
            or stat.S_IMODE(value.parent.stat().st_mode) & 0o077):
        raise PermissionError("R5 role/package source requires owner-only private custody")


def _pin(record: dict, registered: dict[Path, str], *, private: bool = False) -> Path:
    if set(record) != {"path", "sha256"}:
        raise ValueError("source record must contain exact path and SHA")
    original = Path(record["path"])
    if original.is_symlink():
        raise PermissionError("source symlink refused")
    path = original.resolve(strict=True)
    if (len(record["sha256"]) != 64 or not path.is_file()
            or registered.get(path) != record["sha256"]
            or _sha(path) != record["sha256"]):
        raise PermissionError("source is not byte-pinned by verification unit")
    if private:
        _private(path)
    return path


def _registered(unit_id: str, config_path: Path) -> dict[Path, str]:
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    source = {Path(item["path"]).resolve(strict=True): item["sha256"]
              for item in state["units"][unit_id]["spec"]["sources"]}
    if source.get(config_path) != _sha(config_path):
        raise PermissionError("R5 config is not byte-pinned by queued unit")
    return source


def _role_path(store: RoleOnlyStore, role: str, field: str,
               registered: dict[Path, str]) -> None:
    key = f"{role}:{field}"
    record = {"path": str(store.package_root / store.manifest["role_file_relative_path"][key]),
              "sha256": store.manifest["role_file_sha256"][key]}
    _pin(record, registered, private=True)


def _release(record: dict, registered: dict[Path, str], kind: str) -> dict:
    path = _pin(record, registered, private=True)
    expected = FEATURE_MEMBERS if kind == "feature" else HARD_MEMBERS
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError("R5 saved wire has missing or extra members")
        return {key: np.array(archive[key], copy=True) for key in expected}


def _package(case: dict, registered: dict[Path, str], role_sha: str,
             source_model_sha: str, schema_sha: str) -> tuple[dict, dict]:
    manifest_path = _pin(case["manifest"], registered, private=True)
    weights_path = _pin(case["weights"], registered, private=True)
    if weights_path != manifest_path.with_name("weights.npz"):
        raise PermissionError("weights are not sibling of manifest")
    manifest = json.loads(manifest_path.read_text())
    kind = case["kind"]
    schema = ("latent-repair-f-service-package-v1" if kind == "feature" else
              "code-witness-service-package-v1" if case["arm"].startswith("T0_") else
              "latent-repair-service-package-v1")
    if (manifest.get("schema") != schema or manifest.get("arm_id") != case["arm"]
            or manifest.get("source_seed") != case["seed"]
            or manifest.get("source_sha256") != source_model_sha
            or manifest.get("role_manifest_sha256") != role_sha
            or manifest.get("classes") != [2, 6]
            or manifest.get("latent_dim") != 16
            or manifest.get("calibration_family") != "H-GLOBAL-TEMP"
            or manifest.get("weights_sha256") != case["weights"]["sha256"]):
        raise PermissionError("R5 package source/role/arm/seed differs")
    if case["arm"].startswith("T0_") and manifest.get("source_schema_sha256") != schema_sha:
        raise PermissionError("R5 new package source schema differs")
    if kind == "hard" and (manifest.get("bit_count") != case["bits"]
                           or manifest.get("bit_order") != "c0_least_significant"):
        raise PermissionError("R5 code package capacity/order differs")
    with np.load(weights_path, allow_pickle=False) as archive:
        if set(archive.files) != set(manifest["weights_keys"]):
            raise ValueError("R5 serialized tensor inventory differs")
        weights = {key: np.array(archive[key], copy=True) for key in archive.files}
    state = {key: value for key, value in weights.items() if not key.startswith("scaler.")}
    if kind == "hard":
        if tensor_digest(state) != manifest.get("model_sha256"):
            raise ValueError("R5 independently computed model digest differs")
        for purpose in (0, 1):
            head = {key: value for key, value in state.items()
                    if key.startswith(f"heads.{purpose}.")}
            if tensor_digest(head) != manifest["head_sha256"][purpose]:
                raise ValueError("R5 independently computed head digest differs")
    calibrators = manifest.get("calibrators")
    if (not isinstance(calibrators, list) or len(calibrators) != 2
            or any(item.get("schema") != "pcrl-portfolio-global-temp-v1"
                   or item.get("task") != ("income", "occupation")[i]
                   or not .25 <= float(item.get("alpha", float("nan"))) <= 4.0
                   for i, item in enumerate(calibrators))):
        raise ValueError("R5 fixed global calibration record differs")
    return manifest, state


def _reference_metrics(ref: dict, labels: tuple[np.ndarray, np.ndarray]) -> dict:
    out = {}
    for kind in ("U0", "Ucal"):
        out[kind] = {}
        for purpose, task, classes in ((0, "income", 2), (1, "occupation", 6)):
            rows = task_rows(labels[purpose], ref[f"{kind}_{purpose+1}"], classes,
                             ref[f"d_{purpose+1}"])
            values = task_means(rows)
            count = np.bincount(labels[purpose], minlength=classes)
            majority = int(np.argmax(count))
            decision = ref[f"d_{purpose+1}"]
            constant = float(np.mean(labels[purpose] == majority))
            values.update({"constant_accuracy": constant,
                           "accuracy_gain_over_constant": values["accuracy"] - constant,
                           "class_counts": count.tolist(),
                           "per_class_recall": [float(np.mean(decision[labels[purpose] == cls] == cls))
                                                if count[cls] else None for cls in range(classes)]})
            out[kind][task] = values
    return out


def _r1_lineage(case: dict, registered: dict[Path, str], role_sha: str,
                source_model_sha: str) -> None:
    """Bind a new W/T0 wire to the immutable producer receipt, independently."""
    receipt = json.loads(_pin(case["producer_receipt"], registered).read_text())
    arm, seed, kind = case["arm"], case["seed"], case["kind"]
    expected_stage = "bank" if kind == "bank" else "fit"
    if (receipt.get("schema") != "code-utility-witness-worker-v1"
            or receipt.get("status") != "COMPLETE"
            or receipt.get("stage") != expected_stage
            or receipt.get("seed") != seed
            or receipt.get("role_manifest_sha256") != role_sha
            or receipt.get("source_model_sha256") != source_model_sha):
        raise PermissionError("R1 producer receipt lineage differs")
    result = receipt["result"]
    if kind == "bank":
        if arm not in result["arm_ids"]:
            raise PermissionError("R1 bank arm absent from producer")
        record = result["roles"]["INNER_SELECTION"]["arms"][arm]
        if record["public_table_sha256"] != case["public_table"]["sha256"]:
            raise PermissionError("R1 bank source table differs")
    else:
        if result["arm_id"] != arm or result["bit_count"] != case["bits"]:
            raise PermissionError("R1 learned arm/capacity differs")
        record = result["roles"]["INNER_SELECTION"]
        package = result["package"]
        if (package["manifest_sha256"] != case["manifest"]["sha256"]
                or package["weights_sha256"] != case["weights"]["sha256"]):
            raise PermissionError("R1 learned package receipt differs")
    if record["private_release_sha256"] != case["release"]["sha256"]:
        raise PermissionError("R1 saved release hash differs from producer")


def _old_lineage(case: dict, registered: dict[Path, str], role_sha: str,
                 source_model_sha: str) -> None:
    receipt = json.loads(_pin(case["producer_receipt"], registered).read_text())
    first = case["arm"] in ("T", "D-K")
    if (receipt.get("schema") != ("latent-repair-first-paired-seed-receipt-v1" if first
                                  else "latent-repair-followup-controls-receipt-v1")
            or receipt.get("status") != ("T_DK_RELEASES_READY" if first
                                        else "D_FOLLOWUP_RELEASES_READY")
            or receipt.get("seed") != case["seed"]
            or receipt.get("source_sha256") != source_model_sha
            or receipt.get("role_manifest_sha256") != role_sha):
        raise PermissionError("old F/T/D-K producer lineage differs")
    arm = case["arm"]
    package = receipt["packages"][arm]
    release = receipt["roles"]["INNER_SELECTION"][arm]
    if (package["manifest_sha256"] != case["manifest"]["sha256"]
            or package["weights_sha256"] != case["weights"]["sha256"]
            or release["private_release_sha256"] != case["release"]["sha256"]):
        raise PermissionError("old F/T/D-K package/release hash differs")


def run(config_path: Path) -> dict:
    active = assert_active_from_env(stage="verification")
    config_path = config_path.resolve(strict=True)
    _private(config_path)
    registered = _registered(active["unit_id"], config_path)
    config = json.loads(config_path.read_text())
    if config.get("schema") != SCHEMA or config.get("unit_id") != active["unit_id"]:
        raise PermissionError("R5 verification config/unit identity differs")
    expected_sources = {"replay_worker.py", "independent_release.py",
                        "independent_decoder.py", "independent_checks.py"}
    if set(config.get("verifier_sources_sha256", {})) != expected_sources:
        raise PermissionError("R5 verifier source inventory differs")
    for name, sha in config["verifier_sources_sha256"].items():
        _pin({"path": str(HERE / name), "sha256": sha}, registered)
    seal_path = _pin(config["seal"], registered)
    if config["seal"]["sha256"] != active["seal_sha256"]:
        raise PermissionError("R5 queue seal differs")
    seal = json.loads(seal_path.read_text())
    role_path = _pin(config["role_manifest"], registered)
    if seal.get("role_manifest_sha256") != config["role_manifest"]["sha256"]:
        raise PermissionError("R5 role admission differs from seal")
    store = RoleOnlyStore(role_path)
    if store.manifest["source_schema_sha256"] != seal["source_schema_sha256"]:
        raise PermissionError("R5 source schema differs")
    for field in ("income", "occupation", "references_s0", "references_s1"):
        _role_path(store, "INNER_SELECTION", field, registered)
    income = store.read("INNER_SELECTION", "income", "screen")
    occupation = store.read("INNER_SELECTION", "occupation", "screen")
    if (not np.array_equal(income["row_id"], occupation["row_id"])
            or not np.array_equal(income["unit"], occupation["unit"])):
        raise PermissionError("R5 INNER task labels are not aligned")
    row_id, unit = income["row_id"], income["unit"]
    labels = (income["income"], occupation["occupation"])
    reference = {}
    for seed in (0, 1):
        ref = store.read("INNER_SELECTION", f"references_s{seed}")
        if (not np.array_equal(ref["row_id"], row_id)
                or not np.array_equal(ref["unit"], unit)):
            raise PermissionError("R5 references differ from INNER people")
        reference[str(seed)] = _reference_metrics(ref, labels)
    case_result = {}
    for case in config["cases"]:
        assert_active_from_env(unit_id=active["unit_id"], stage="verification")
        arm, seed, kind = case["arm"], case["seed"], case["kind"]
        key = f"{arm}__s{seed}"
        if (key in case_result or seed not in (0, 1)
                or kind not in ("bank", "hard", "feature")):
            raise ValueError("duplicate or invalid independent replay case")
        source_model_sha = seal["source_model_files_sha256_by_seed"][str(seed)]["model_pt_sha256"]
        if kind == "bank" or arm.startswith("T0_"):
            _r1_lineage(case, registered, config["role_manifest"]["sha256"],
                        source_model_sha)
        else:
            _old_lineage(case, registered, config["role_manifest"]["sha256"],
                         source_model_sha)
        wire = _release(case["release"], registered, kind)
        if kind == "bank":
            table = json.loads(_pin(case["public_table"], registered).read_text())
            if (table.get("schema") != "code-utility-public-bank-tables-v1"
                    or table.get("seed") != seed
                    or table.get("source_model_sha256") != source_model_sha
                    or table.get("feature_schema_sha256") != seal["source_schema_sha256"]):
                raise PermissionError("R5 public W bank source/table differs")
            family = table["families"][arm]
            result = replay_bank(wire, row_id, unit, labels,
                                 (np.asarray(family["table0"], dtype=np.float64),
                                  np.asarray(family["table1"], dtype=np.float64)))
        else:
            manifest, state = _package(case, registered,
                                       config["role_manifest"]["sha256"],
                                       source_model_sha, seal["source_schema_sha256"])
            alphas = tuple(float(item["alpha"]) for item in manifest["calibrators"])
            if kind == "hard":
                result = replay_hard(wire, row_id, unit, labels, state,
                                     case["bits"], alphas)
            else:
                result = replay_feature(wire, row_id, unit, labels, state, alphas)
        case_result[key] = {"arm": arm, "seed": seed, "kind": kind,
                            "bits": case.get("bits"),
                            "release_sha256": case["release"]["sha256"],
                            **result}
    output = Path(config["output"]).resolve(strict=False)
    if output.exists() or output.parent != HERE:
        raise PermissionError("R5 public aggregate output must be new in owned namespace")
    public = {"schema": "pcrl-r5-independent-inner-replay-v1", "status": "PASS",
              "unit_id": active["unit_id"], "campaign_seal_sha256": active["seal_sha256"],
              "role_manifest_sha256": config["role_manifest"]["sha256"],
              "inner_rows": int(len(row_id)), "inner_groups": int(np.unique(unit).size),
              "reference": reference, "cases": case_result,
              "scope": "reused INNER development; complete saved task wires and public decoder reconstruction; no attack/interval inference"}
    with output.open("x") as stream:
        json.dump(public, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"status": public["status"], "cases": len(case_result),
            "output_sha256": _sha(output)}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().config), sort_keys=True))


if __name__ == "__main__":
    main()
