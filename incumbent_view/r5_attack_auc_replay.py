"""Queue-only independent AUC replay of one frozen complete-view release.

This consumes frozen predictions.  It never fits or selects a reader and does
not import the attack runner's view or scoring functions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env
from .r5_attack_math import BOOTSTRAPS, GROUPS, VIEWS, paired_plan, reader_envelope


def sha(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def arrays(path: str | Path, names: set[str]) -> dict[str, np.ndarray]:
    with ZipFile(path) as archive:
        actual = [x.filename for x in archive.infolist()]
        if len(actual) != len(names) or set(actual) != {x + ".npy" for x in names}:
            raise PermissionError("private archive members differ from frozen allowlist")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in names}


def row_unit_sha(row: np.ndarray, unit: np.ndarray) -> str:
    joined = np.ascontiguousarray(np.column_stack((row, unit)))
    hash_ = hashlib.sha256()
    hash_.update(str(joined.dtype).encode())
    hash_.update(str(tuple(joined.shape)).encode())
    hash_.update(joined.tobytes())
    return hash_.hexdigest()


def write_once_json(path: Path, record: dict, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode), "w") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def run(config_path: str | Path) -> dict:
    active = assert_active_from_env(stage="verification")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    source_index = {str(Path(s["path"]).resolve(strict=True)): s["sha256"]
                    for s in state["units"][active["unit_id"]]["spec"]["sources"]}

    def pinned(item: dict) -> Path:
        raw = Path(item["path"])
        if raw.is_symlink():
            raise PermissionError("symlinked private source refused")
        path = raw.resolve(strict=True)
        if item["sha256"] != sha(path) or source_index.get(str(path)) != item["sha256"]:
            raise PermissionError("source differs from active queue byte pin")
        return path

    cfgpath = Path(config_path).resolve(strict=True)
    if source_index.get(str(cfgpath)) != sha(cfgpath):
        raise PermissionError("R5 config is not pinned by active queue")
    cfg = json.loads(cfgpath.read_text())
    if cfg.get("schema") != "IVR1-r5-frozen-auc-replay-config-v1" or cfg.get("unit_id") != active["unit_id"]:
        raise PermissionError("R5 release identity differs")
    if cfg["queue_seal"]["sha256"] != active["seal_sha256"] or cfg["queue_seal"]["sha256"] != state["seal"]["sha256"]:
        raise PermissionError("active queue seal differs")
    pinned(cfg["queue_seal"])
    science_path = pinned(cfg["campaign_seal"])
    science = cfg["campaign_seal"]["sha256"]
    if science not in {a["sha256"] for a in state["amendments"] if "protection" in a["stages"]}:
        raise PermissionError("campaign science lock is not registered")
    verification = [a for a in state["amendments"] if "verification" in a["stages"]]
    if (not verification or state["units"][active["unit_id"]]["spec"].get("amendment_sha256") != verification[-1]["sha256"]
            or json.loads(pinned(cfg["verification_amendment"]).read_text()).get("campaign_lock_sha256") != science):
        raise PermissionError("latest R5 inference amendment differs")
    receipt_path = pinned(cfg["r4_receipt"])
    receipt = json.loads(receipt_path.read_text())
    identity = (cfg["arm_id"], cfg["source_seed"], cfg["realization"])
    if (receipt.get("schema") != "pcrl-incumbent-r4-fresh-attack-v1" or
            receipt.get("status") != "COMPLETE_PRIVATE_DEVELOPMENT_ONLY" or
            (receipt.get("arm_id"), receipt.get("source_seed"), receipt.get("realization")) != identity or
            receipt.get("campaign_seal_sha256") != science or
            receipt.get("queue_seal_sha256") != active["seal_sha256"]):
        raise PermissionError("frozen R4 release identity differs")
    frozen_path = pinned(cfg["frozen"])
    frozen = json.loads(frozen_path.read_text())
    if (frozen.get("status") != "FROZEN_COMPLETE" or frozen.get("effective_lock_sha256") != science or
            frozen.get("family") != "A" or frozen.get("seed") != receipt.get("attacker_seed") or
            receipt["attack"]["freeze_receipt_sha256"] != sha(frozen_path)):
        raise PermissionError("reader freeze must precede INNER SEX")
    role_manifest = json.loads(pinned(cfg["role_manifest"]).read_text())
    expected_sex_sha = role_manifest["roles"]["INNER_SELECTION"]["source_files"]["sex"]["sha256"]
    if cfg["inner_sex"]["sha256"] != expected_sex_sha:
        raise PermissionError("INNER SEX source is outside role allowlist")

    # Check all three complete views, frozen reader inventories, predictions,
    # and engineering controls before deserializing any protected label.
    readers = {}
    for view in VIEWS:
        source = cfg["views"][view]
        reader_path = pinned(source["reader_receipt"])
        prediction_path = pinned(source["predictions"])
        view_path = pinned(source["view"])
        record = json.loads(reader_path.read_text())
        controls = receipt["attack"]["controls"][view]["scores"]
        if controls.get("status") != "PASS" or controls.get("gates") != {
            "null_exact": True, "planted_at_least_0.99": True,
            "shuffled_independent_inner_at_most_0.60": True,
        }:
            raise PermissionError("fresh reader control failed")
        if (record.get("status") != "FROZEN" or
                record["files"].get("predictions_inner.npz") != sha(prediction_path) or
                record["files"].get("view_inner.npz") != sha(view_path) or
                frozen["reader_receipt_sha256"][view] != sha(reader_path) or
                receipt["attack"]["reader_custody"][view]["private_receipt_sha256"] != sha(reader_path)):
            raise PermissionError("reader prediction/view custody differs")
        order = tuple(record["reader_order"])
        if not order or len(order) != len(set(order)):
            raise PermissionError("frozen reader slate differs")
        readers[view] = (order, prediction_path, view_path)

    truth = arrays(pinned(cfg["inner_sex"]), {"row_id", "unit", "sex"})
    row, unit, y = truth["row_id"], truth["unit"], truth["sex"]
    if (row.dtype != np.int64 or unit.dtype != np.int64 or not np.issubdtype(y.dtype, np.integer) or
            row.shape != (2235,) or unit.shape != row.shape or y.shape != row.shape or
            len(np.unique(row)) != len(row) or len(np.unique(unit)) != GROUPS or
            not np.isin(y, (0, 1)).all() or
            row_unit_sha(row, unit) != role_manifest["roles"]["INNER_SELECTION"]["row_unit_pair_sha256"]):
        raise PermissionError("INNER person/SEX alignment differs")
    y = y.astype(np.int64, copy=False)
    inverse, group_weights, draw_indices = paired_plan(unit)
    output_vectors = {}
    point = {}
    for view, (order, prediction_path, view_path) in readers.items():
        assert_active_from_env(unit_id=active["unit_id"], stage="verification")
        with ZipFile(view_path) as archive:
            names = {x.filename[:-4] for x in archive.infolist() if x.filename.endswith(".npy")}
            if len(names) != len(archive.infolist()) or not {"row_id", "unit"}.issubset(names):
                raise PermissionError("complete view archive identity differs")
        with np.load(view_path, allow_pickle=False) as archive:
            if not np.array_equal(archive["row_id"], row) or not np.array_equal(archive["unit"], unit):
                raise PermissionError("complete view rows do not align with INNER SEX")
        predictions = arrays(prediction_path, set(order))
        envelope = reader_envelope(y, predictions, order, inverse, group_weights)
        stated = float(receipt["attack"]["scores"][view]["primary_envelope_auc"])
        if abs(float(envelope[0]) - stated) > 1e-11:
            raise ArithmeticError("independent frozen reader AUC differs from R4")
        output_vectors[view] = envelope
        point[view] = float(envelope[0])
    private_path = Path(cfg["output_private"]).resolve()
    public_path = Path(cfg["output_public"]).resolve()
    owner = Path(__file__).resolve().parents[1] / "results/pcrl_incumbent_view_risk_v1/r5"
    if (private_path.parent != owner / ".private" or public_path.parent != owner or
            private_path.exists() or public_path.exists()):
        raise PermissionError("write-once R5 output path differs")
    with os.fdopen(os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        np.savez_compressed(stream, **output_vectors)
    public = {"schema": "IVR1-r5-frozen-auc-replay-v1", "unit_id": active["unit_id"],
              "arm_id": identity[0], "source_seed": identity[1], "realization": identity[2],
              "status": "PASS", "point_envelope_auc": point, "bootstrap_replicates": BOOTSTRAPS,
              "group_count": GROUPS, "frozen_r4_receipt_sha256": sha(receipt_path),
              "private_auc_vectors_sha256": sha(private_path), "inner_sex_role_sha256": expected_sex_sha,
              "queue": active, "not_confirmation": True}
    write_once_json(public_path, public)
    return public


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps({"unit_id": result["unit_id"], "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
