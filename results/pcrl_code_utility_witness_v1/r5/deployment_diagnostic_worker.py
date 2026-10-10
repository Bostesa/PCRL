"""Queued aggregate diagnostics for the failed raw deployment exact comparison.

Opens INNER features and saved complete wire only; never opens task/SEX labels.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from code_witness.service import load_service
from results.pcrl_code_utility_witness_v1.r0.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.r01.frozen_u import FrozenUEncoder
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people, sha_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    active = assert_active_from_env(stage="verification")
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text())
    if (config.get("schema") != "pcrl-r5-deployment-sample-config-v1"
            or config.get("unit_id") != "R5_T0_B16_S1_DEPLOYMENT_SAMPLE"):
        raise PermissionError("diagnostic source config differs")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    sources = {Path(x["path"]).resolve(strict=True): x["sha256"]
               for x in state["units"][active["unit_id"]]["spec"]["sources"]}
    for name in ("config", "diagnostic", "role_manifest", "features", "source_model",
                 "service_manifest", "service_weights", "saved_release",
                 "producer_receipt", "seal"):
        if name == "config":
            path = config_path
        elif name == "diagnostic":
            path = Path(__file__).resolve()
        else:
            path = Path(config[name]["path"]).resolve(strict=True)
        if sources.get(path) != sha_file(path):
            raise PermissionError(f"diagnostic source {name} not queue pinned")
    seal = json.loads(Path(config["seal"]["path"]).read_text())
    if config["seal"]["sha256"] != active["seal_sha256"]:
        raise PermissionError("diagnostic seal differs")
    store = RoleOnlyStore(config["role_manifest"]["path"])
    names_sha = store.manifest["source_schema_sha256"]
    if names_sha != seal["source_schema_sha256"]:
        raise PermissionError("diagnostic schema differs")
    feature = store.read("INNER_SELECTION", "features")
    with np.load(config["saved_release"]["path"], allow_pickle=False) as archive:
        saved = {name: np.array(archive[name], copy=True) for name in archive.files}
    assert_same_people(feature, saved)
    names = tuple(str(name) for name in feature["feature_names"])
    source_sha = config["source_model"]["sha256"]
    service = load_service(config["service_manifest"]["path"],
                           expected_manifest_sha256=config["service_manifest"]["sha256"],
                           expected_source_sha256=source_sha,
                           expected_bits=config["bits"])
    encoder = FrozenUEncoder.load(config["source_model"]["path"], source_sha,
                                  names, names_sha, config["seed"])
    index = np.linspace(0, len(feature["X"]) - 1, 32, dtype=np.int64)
    raw = feature["X"][index]
    out = {}
    for purpose in (0, 1):
        assert_active_from_env(unit_id=active["unit_id"], stage="verification")
        subset = service.recipient_from_raw(purpose, raw, names, encoder)
        all_rows = service.recipient_from_raw(purpose, feature["X"], names, encoder)
        from_code = service.probabilities_from_code(purpose, saved[f"code{purpose}"][index])
        actual = saved[f"prob{purpose}"][index]
        def gap(left, right):
            return float(np.max(np.abs(left - right)))
        out[str(purpose)] = {
            "subset_code_exact": bool(np.array_equal(subset["code"], saved[f"code{purpose}"][index])),
            "full_code_exact": bool(np.array_equal(all_rows["code"], saved[f"code{purpose}"])),
            "subset_decision_exact": bool(np.array_equal(subset["decision"], saved[f"decision{purpose}"][index])),
            "full_decision_exact": bool(np.array_equal(all_rows["decision"], saved[f"decision{purpose}"])),
            "subset_prob_exact": bool(np.array_equal(subset["probabilities"], actual)),
            "full_prob_exact": bool(np.array_equal(all_rows["probabilities"], saved[f"prob{purpose}"])),
            "subset_prob_max_abs_gap": gap(subset["probabilities"], actual),
            "full_prob_max_abs_gap": gap(all_rows["probabilities"], saved[f"prob{purpose}"]),
            "same_bits_head_subset_max_abs_gap": gap(from_code, actual),
            "raw_subset_vs_code_subset_max_abs_gap": gap(from_code, subset["probabilities"]),
            "calibration_alpha": float(service.calibrators[purpose]["alpha"]),
        }
    result = {"schema": "pcrl-r5-deployment-batch-diagnostic-v1",
              "status": "AGGREGATE_DIAGNOSTIC", "unit_id": active["unit_id"],
              "failed_predecessor_unit_id": config["unit_id"],
              "rows_subset": len(index), "rows_full": len(feature["X"]),
              "service_manifest_sha256": config["service_manifest"]["sha256"],
              "saved_release_sha256": config["saved_release"]["sha256"],
              "source_model_sha256": source_sha, "purpose": out,
              "scope": "technical batch-size and release parity only; no labels opened"}
    output = Path(config["diagnostic_output"]).resolve(strict=False)
    if output.exists() or output.parent != Path(__file__).parent:
        raise PermissionError("new public diagnostic output required")
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "unit_id": active["unit_id"],
                      "receipt_sha256": sha_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
