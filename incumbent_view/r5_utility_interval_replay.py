"""Queue-only independent replay of the 192 simultaneous utility clauses."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env
from .r5_attack_math import ARMS, K
from .r5_utility_interval_math import FIELDS, interval_family


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_vectors(path):
    with ZipFile(path) as archive:
        members = [x.filename for x in archive.infolist()]
        if len(members) != len(FIELDS) or set(members) != {x + ".npy" for x in FIELDS}:
            raise PermissionError("INNER task cost vector member inventory differs")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in FIELDS}


def run(config_path):
    active = assert_active_from_env(stage="verification")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    source_index = {str(Path(x["path"]).resolve(strict=True)): x["sha256"]
                    for x in state["units"][active["unit_id"]]["spec"]["sources"]}

    def pinned(record):
        raw = Path(record["path"])
        if raw.is_symlink():
            raise PermissionError("interval replay source symlink refused")
        path = raw.resolve(strict=True)
        if record["sha256"] != sha(path) or source_index.get(str(path)) != record["sha256"]:
            raise PermissionError("interval source differs from queue pin")
        return path

    cfgpath = Path(config_path).resolve(strict=True)
    if source_index.get(str(cfgpath)) != sha(cfgpath):
        raise PermissionError("R5 interval config unpinned")
    cfg = json.loads(cfgpath.read_text())
    if cfg.get("schema") != "IVR1-r5-utility-interval-replay-config-v1" or cfg.get("unit_id") != active["unit_id"]:
        raise PermissionError("R5 interval identity differs")
    admission_sha = cfg["admission_lock"]["sha256"]
    science_sha = cfg["campaign_seal"]["sha256"]
    pinned(cfg["admission_lock"])
    pinned(cfg["campaign_seal"])
    if (admission_sha != active["seal_sha256"] or admission_sha != state["seal"]["sha256"] or
            science_sha not in {a["sha256"] for a in state["amendments"] if "protection" in a["stages"]}):
        raise PermissionError("R5 interval dual-lock binding differs")
    verification = [a for a in state["amendments"] if "verification" in a["stages"]]
    if (not verification or
            verification[-1]["sha256"] != state["units"][active["unit_id"]]["spec"].get("amendment_sha256") or
            json.loads(pinned(cfg["verification_amendment"]).read_text()).get("campaign_lock_sha256") != science_sha):
        raise PermissionError("R5 interval replay amendment differs")
    result_path = pinned(cfg["r3_result"])
    reported = json.loads(result_path.read_text())
    if (reported.get("schema") != "pcrl-incumbent-r3-utility-intervals-v1" or
            reported.get("K") != K or reported.get("utility_slots") != 192 or
            reported.get("campaign_seal_sha256") != science_sha or
            reported.get("admission_lock_sha256") != admission_sha):
        raise PermissionError("reported simultaneous utility family differs")
    if len(cfg["records"]) != 16:
        raise PermissionError("all seed-by-arm task cost vectors required")
    records = {}
    for source in cfg["records"]:
        receipt_path = pinned(source["receipt"])
        vector_path = pinned(source["vectors"])
        receipt = json.loads(receipt_path.read_text())
        key = (receipt.get("arm_id"), receipt.get("source_seed"))
        source_seal = admission_sha if key[0] == "P0" else science_sha
        if (key[0] not in ARMS or key[1] not in (0, 1) or key in records or
                receipt.get("role") != "INNER_SELECTION" or
                receipt.get("source_pins", {}).get("seal") != source_seal or
                receipt.get("queue", {}).get("seal_sha256") != admission_sha or
                receipt.get("private_vectors_sha256") != sha(vector_path) or
                reported["source_receipts"].get(f"{key[0]}:s{key[1]}") != sha(receipt_path)):
            raise PermissionError("paired R3 source identity/custody differs")
        if key[0] != "P0" and receipt["source_pins"].get("admission_lock") != admission_sha:
            raise PermissionError("protected task vector lacks admission binding")
        records[key] = read_vectors(vector_path)
    if set(records) != {(arm, seed) for arm in ARMS for seed in (0, 1)}:
        raise PermissionError("fixed 16 utility records incomplete")
    independent = interval_family(records)
    if set(independent) != set(reported["slots"]):
        raise ArithmeticError("utility slot ID family differs")
    max_gap = 0.0
    disagreements = []
    for name, value in independent.items():
        comparison = reported["slots"][name]
        if value["status"] != comparison["status"]:
            disagreements.append(name)
        for field in ("point", "se", "lower", "upper"):
            x, y = value[field], comparison[field]
            if (x is None) != (y is None):
                disagreements.append(f"{name}:{field}:null")
            elif x is not None:
                max_gap = max(max_gap, abs(float(x) - float(y)))
    if disagreements or max_gap > 1e-11:
        raise ArithmeticError(f"independent simultaneous utility interval mismatch: {disagreements[:4]}, {max_gap}")
    output = Path(cfg["output"]).resolve()
    owner = Path(__file__).resolve().parents[1] / "results/pcrl_incumbent_view_risk_v1/r5"
    if output.parent != owner or output.exists() or output.is_symlink():
        raise PermissionError("write-once R5 utility replay output differs")
    payload = {"schema": "IVR1-r5-utility-interval-replay-v1", "status": "PASS",
               "unit_id": active["unit_id"], "K": K, "slots_checked": len(independent),
               "status_disagreements": disagreements,
               "max_numeric_gap": max_gap, "r3_result_sha256": sha(result_path),
               "all_16_inner_sources_checked": True,
               "old_assessment_read": False, "SEX_labels_read": False,
               "queue": active, "not_confirmation": True}
    with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "w") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return payload


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps({"unit_id": result["unit_id"], "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
