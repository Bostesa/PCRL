"""Queue-only independent 567-clause D interval replay from private arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .independent_d_intervals import ARMS, ANCHORS, VIEWS, fixed_d_intervals
from .independent_replay import d_finite_family_ids
from .replay_first_pair import (_npz, _queue_guard, _read_json, _role_npz,
                                sha256)


ENDPOINT_KEYS = {"row_id", "unit", "income", "occupation",
                 "U0_1", "U0_2", "Ucal_1", "Ucal_2"}
HARD_KEYS = {"row_id", "unit", "code0", "code1", "prob0", "prob1",
             "decision0", "decision1"}
F_KEYS = {"row_id", "unit", "features0", "features1", "prob0", "prob1",
          "decision0", "decision1"}
ANCHOR_KEYS = {"row_id", "unit", "U0_1", "U0_2", "Ucal_1", "Ucal_2",
               "d_1", "d_2"}
ARM_LABELS = {"D_T": "T", "D_K": "D-K", "D_L": "L", "D_S12": "S12",
              "D_S21": "S21", "D_C": "C", "D_F": "F"}


def _file(item: dict, *, private: bool = False) -> Path:
    if set(item) != {"path", "sha256"}:
        raise ValueError("interval source map requires path and byte hash")
    path = Path(item["path"]).resolve(strict=True)
    if private and ".private" not in path.parts:
        raise PermissionError("per-person D interval array outside private custody")
    if sha256(path) != item["sha256"]:
        raise ValueError("D interval input changed from frozen source map")
    return path


def _reader_bank(private: Path, attack_receipt: dict,
                 row_id: np.ndarray, unit: np.ndarray) -> dict:
    bank = {}
    for view in VIEWS:
        directory = private / view / "readers"
        published = attack_receipt["attack"]["reader_custody"][view]
        receipt_path = directory / "READERS_RECEIPT.json"
        receipt = _read_json(receipt_path, published["private_receipt_sha256"])
        prediction_path = directory / "predictions_inner.npz"
        predicted = _npz(prediction_path,
                         receipt["files"]["predictions_inner.npz"],
                         set(receipt["reader_order"]))
        view_path = directory / "view_inner.npz"
        if sha256(view_path) != receipt["files"]["view_inner.npz"]:
            raise ValueError("fixed reader view byte identity differs")
        with np.load(view_path, allow_pickle=False) as view_record:
            if (not np.array_equal(view_record["row_id"], row_id)
                    or not np.array_equal(view_record["unit"], unit)):
                raise ValueError("fixed reader bank ordered INNER identity differs")
        for name, values in predicted.items():
            if values.shape != row_id.shape or not np.isfinite(values).all():
                raise ValueError("fixed reader score count or value differs")
        bank[view] = predicted
    return bank


def _point_parity(report: dict, first: dict, allarm: dict,
                  anchors: dict, seed: int) -> dict:
    by_name = {clause["name"]: clause["point"] for clause in report["clauses"]}
    largest = 0.0
    for arm in ARMS:
        label = ARM_LABELS[arm]
        task = allarm["all_seven_point"][label]["task"]
        for task_name in ("income", "occupation"):
            for ref in ANCHORS:
                base = first["references"][task_name][ref]
                for metric in ("accuracy", "nll", "brier"):
                    key = f"s{seed}:utility:{arm}:{task_name}:{ref}:{metric}"
                    expected = task[task_name][metric] - base[metric]
                    largest = max(largest, abs(by_name[key] - expected))
        for ref in ANCHORS:
            reference = anchors["anchors"][f"{ref}_s{seed}"]["sex_recovery"]
            candidate = allarm["all_seven_point"][label]["sex_recovery"]
            for view in VIEWS:
                key = f"s{seed}:u_anchor:{arm}:{ref}:{view}:auc"
                expected = candidate[view]["auc"] - reference[view]["auc"]
                largest = max(largest, abs(by_name[key] - expected))
    if largest > 1e-10:
        raise ValueError("independent interval point differs from saved endpoint replay")
    return {"status": "POINT_PARITY_PASS", "max_abs_difference": largest}


def run(args: argparse.Namespace) -> dict:
    seal_path = Path(args.seal).resolve(strict=True)
    _queue_guard(seal_path, args.seal_sha256)
    seal = _read_json(seal_path, args.seal_sha256)
    if seal.get("status") != "LOCALLY_SEALED_D_STAGE":
        raise PermissionError("D interval effective seal differs")
    role_path = Path(args.role_manifest).resolve(strict=True)
    role = _read_json(role_path, args.role_manifest_sha256)
    if args.role_manifest_sha256 != seal["source_files_sha256"]["source_role_manifest"]:
        raise ValueError("D interval role source differs")
    family = _read_json(Path(args.family_manifest), args.family_manifest_sha256)
    if family["ordered_ids"] != list(d_finite_family_ids()) or len(family["ordered_ids"]) != 567:
        raise ValueError("sealed 567-clause family identity differs")
    first = {seed: _read_json(Path(getattr(args, f"first_s{seed}")),
                              getattr(args, f"first_s{seed}_sha256")) for seed in (0, 1)}
    allarm = {seed: _read_json(Path(getattr(args, f"allarm_s{seed}")),
                               getattr(args, f"allarm_s{seed}_sha256")) for seed in (0, 1)}
    anchors = _read_json(Path(args.anchor_verification), args.anchor_verification_sha256)
    if (anchors.get("status") != "FOUR_SCORE_ANCHOR_REPLAY_PASS"
            or anchors.get("effective_lock_sha256") != args.seal_sha256
            or any(first[s].get("status") != "TECHNICAL_REPLAY_PASS"
                   or allarm[s].get("status") != "FOLLOWUP_COMPLETE_VIEW_REPLAY_PASS"
                   for s in (0, 1))):
        raise ValueError("prerequisite independent release/reader replay incomplete")
    mapping = _read_json(Path(args.input_map), args.input_map_sha256)
    if mapping.get("schema") != "pcrl-r04-d-interval-input-v1" or set(mapping["seeds"]) != {"0", "1"}:
        raise ValueError("fixed two-seed D interval source map differs")
    records = {}
    for seed in (0, 1):
        item = mapping["seeds"][str(seed)]
        if item["endpoint"]["sha256"] != first[seed]["endpoint_sha256"]:
            raise ValueError("interval truth/reference endpoint differs from first replay")
        endpoint = _npz(_file(item["endpoint"], private=True),
                        item["endpoint"]["sha256"], ENDPOINT_KEYS)
        sex = _role_npz(role, role_path, "INNER_SELECTION", "sex")
        if (not np.array_equal(endpoint["row_id"], sex["row_id"])
                or not np.array_equal(endpoint["unit"], sex["unit"])):
            raise ValueError("interval endpoint/SEX ordered rows differ")
        truth = {key: endpoint[key] for key in ("row_id", "unit", "income", "occupation")}
        truth["sex"] = sex["sex"]
        release = {}
        if set(item["releases"]) != set(ARMS).union(ANCHORS):
            raise ValueError("interval release registry incomplete")
        for name in (*ARMS, *ANCHORS):
            entry = item["releases"][name]
            if set(entry) != {"kind", "path", "sha256", "attack_receipt",
                              "attack_receipt_sha256", "private_root"}:
                raise ValueError("interval release source entry differs")
            attack = _read_json(Path(entry["attack_receipt"]),
                                entry["attack_receipt_sha256"])
            if name in ANCHORS:
                expected_sha = anchors["anchors"][f"{name}_s{seed}"]["attack_receipt_sha256"]
            elif name in ("D_T", "D_K"):
                expected_sha = first[seed]["attack_receipt_sha256"][ARM_LABELS[name]]
            else:
                expected_sha = allarm[seed]["arms"][ARM_LABELS[name]]["attack_receipt_sha256"]
            if entry["attack_receipt_sha256"] != expected_sha:
                raise ValueError("interval reader source differs from R04 replay")
            private = Path(entry["private_root"]).resolve(strict=True)
            if ".private" not in private.parts:
                raise PermissionError("interval reader bank outside private custody")
            raw = _npz(_file({"path": entry["path"], "sha256": entry["sha256"]},
                             private=True), entry["sha256"],
                       ANCHOR_KEYS if name in ANCHORS else
                       F_KEYS if name == "D_F" else HARD_KEYS)
            if name in ANCHORS:
                if entry["kind"] != "anchor":
                    raise ValueError("reference score-wire kind differs")
                q1, q2 = raw[f"{name}_1"], raw[f"{name}_2"]
            else:
                if entry["kind"] != ("continuous" if name == "D_F" else "hard"):
                    raise ValueError("D arm wire kind differs")
                q1, q2 = raw["prob0"], raw["prob1"]
            release[name] = {"row_id": raw["row_id"], "unit": raw["unit"],
                             "prob_1": q1, "prob_2": q2,
                             "reader_predictions": _reader_bank(private, attack,
                                                                raw["row_id"], raw["unit"])}
        records[seed] = {"truth": truth, "releases": release}
    report = fixed_d_intervals(records)
    report["point_replay"] = {f"s{seed}": _point_parity(report, first[seed],
                                                       allarm[seed], anchors, seed)
                              for seed in (0, 1)}
    report["effective_lock_sha256"] = args.seal_sha256
    report["family_manifest_sha256"] = args.family_manifest_sha256
    report["input_map_sha256"] = args.input_map_sha256
    report["anchor_verification_sha256"] = args.anchor_verification_sha256
    output = Path(args.public_receipt)
    if output.exists():
        raise FileExistsError("D interval replay receipt already exists")
    output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return {"status": report["status"], "public_receipt_sha256": sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "seal-sha256", "role-manifest", "role-manifest-sha256",
                 "family-manifest", "family-manifest-sha256",
                 "first-s0", "first-s0-sha256", "first-s1", "first-s1-sha256",
                 "allarm-s0", "allarm-s0-sha256", "allarm-s1", "allarm-s1-sha256",
                 "anchor-verification", "anchor-verification-sha256",
                 "input-map", "input-map-sha256", "public-receipt"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
