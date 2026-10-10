"""Queue-only independent replay of frozen bank attack AUC endpoints.

This reads the allowed role-only SEX exports after verifying the R4 reader
freeze. It imports no R4, attack-evaluation, candidate or selection code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .r5_independent import independent_auc


SALT = "pcrl-method-portfolio-v1|attack-tune|20261008|"
VIEWS = ("r1", "r2", "pair")
SCORE_FIELDS = ("positive_alias_auc", "reverse_alias_auc", "fixed_alias_max_auc")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update((str(arr.dtype) + str(arr.shape)).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _archive(path: Path, expected: set[str] | None = None) -> dict:
    with ZipFile(path) as source:
        names = {item.filename for item in source.infolist()}
        if len(names) != len(source.infolist()) or any(not name.endswith(".npy") for name in names):
            raise PermissionError("NPZ member names are duplicated or unexpected")
        keys = {name[:-4] for name in names}
        if expected is not None and keys != expected:
            raise PermissionError("role/prediction archive member inventory differs")
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in keys}


def _binary(value: np.ndarray, rows: int) -> np.ndarray:
    y = np.asarray(value)
    if y.dtype != np.int64 or y.shape != (rows,) or not np.isin(y, (0, 1)).all():
        raise ValueError("role-only binary SEX label schema differs")
    return y


def _auc_triplet(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.asarray(p)
    if p.dtype != np.float64 or p.shape != y.shape:
        raise ValueError("frozen reader probability schema differs")
    auc = independent_auc(y, p)
    return {"positive_alias_auc": auc, "reverse_alias_auc": 1.0 - auc,
            "fixed_alias_max_auc": max(auc, 1.0 - auc)}


def replay_reader_scores(y_tune: np.ndarray, y_inner: np.ndarray,
                         tune: dict[str, np.ndarray],
                         inner: dict[str, np.ndarray],
                         reader_order: tuple[str, ...]) -> dict:
    if (not reader_order or len(set(reader_order)) != len(reader_order)
            or set(tune) != set(reader_order) or set(inner) != set(reader_order)):
        raise ValueError("frozen reader inventory differs")
    val = {name: _auc_triplet(y_tune, tune[name]) for name in reader_order}
    scored = {name: _auc_triplet(y_inner, inner[name]) for name in reader_order}
    winner, best = reader_order[0], -1.0
    for name in reader_order:
        value = val[name]["fixed_alias_max_auc"]
        if value > best + 1e-12:
            winner, best = name, value
    orientation = ("positive" if val[winner]["positive_alias_auc"] >=
                   val[winner]["reverse_alias_auc"] else "reverse")
    result = {"validation": val, "validation_winner": winner,
              "validation_winner_orientation": orientation,
              "readers": scored,
              "primary_envelope_auc": max(
                  value["fixed_alias_max_auc"] for value in scored.values())}
    result["validation_winner_inner_auc"] = scored[winner][
        "positive_alias_auc" if orientation == "positive" else "reverse_alias_auc"]
    return result


def _same_number(actual: float, expected: float) -> None:
    if abs(float(actual) - float(expected)) > 1e-11:
        raise ValueError("independent frozen-reader AUC differs")


def _role_labels(manifest_path: Path, manifest_sha: str, role: str) -> dict:
    if sha(manifest_path) != manifest_sha:
        raise PermissionError("role-only manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if role not in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        raise PermissionError("unapproved attack role")
    key = f"{role}:sex"
    archive_path = manifest_path.parent.parent / manifest["role_file_relative_path"][key]
    if sha(archive_path) != manifest["role_file_sha256"][key]:
        raise PermissionError("role-only SEX export changed")
    record = _archive(archive_path, {"row_id", "unit", "sex"})
    if (record["row_id"].dtype != np.int64 or
            record["unit"].dtype != np.int64 or
            record["row_id"].ndim != 1 or
            record["unit"].shape != record["row_id"].shape or
            len(record["row_id"]) != manifest["roles"][role]["rows"] or
            len(np.unique(record["unit"])) != manifest["roles"][role]["groups"]):
        raise PermissionError("role-only SEX ordered people differ")
    _binary(record["sex"], len(record["row_id"]))
    return record


def _tune_positions(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = np.unique(unit)
    if len(groups) != 4061:
        raise PermissionError("ATTACK_FIT_NEW groups differ")
    ranked = sorted((hashlib.sha256((SALT + str(int(group))).encode()).hexdigest(),
                     int(group)) for group in groups)
    chosen = np.asarray([group for _, group in ranked[:1015]], dtype=np.int64)
    return np.flatnonzero(np.isin(unit, chosen)), np.sort(chosen)


def replay(spec_path: Path, spec_sha: str, receipt_path: Path,
           receipt_sha: str) -> dict:
    if sha(spec_path) != spec_sha or sha(receipt_path) != receipt_sha:
        raise PermissionError("R4 attack spec/receipt bytes differ")
    spec = json.loads(spec_path.read_text())
    public = json.loads(receipt_path.read_text())
    if (public.get("status") != "COMPLETE_PRIVATE_DEVELOPMENT_ONLY" or
            public.get("unit_id") != spec.get("unit_id") or
            public.get("unit_spec_sha256") != spec_sha or
            public.get("campaign_seal_sha256") != spec.get("campaign_seal_sha256") or
            public.get("arm_id") != spec.get("arm_id") or
            public.get("source_seed") != spec.get("source_seed") or
            public.get("realization") != spec.get("realization")):
        raise PermissionError("R4 frozen attack unit identity differs")
    attack = public["attack"]
    if attack.get("schema") != "pcrl-next-frozen-attack-unit-v1" or \
            set(attack.get("scores", {})) != set(VIEWS):
        raise PermissionError("R4 complete three-view score bank differs")
    private = Path(spec["private_output_path"])
    freeze_path = private / "FROZEN_COMPLETE.json"
    if sha(freeze_path) != attack["freeze_receipt_sha256"]:
        raise PermissionError("R4 reader freeze bytes differ")
    freeze = json.loads(freeze_path.read_text())
    if (freeze.get("schema") != "pcrl-next-reader-freeze-v1" or
            freeze.get("status") != "FROZEN_COMPLETE" or
            set(freeze.get("reader_receipt_sha256", {})) != set(VIEWS) or
            freeze.get("effective_lock_sha256") != spec["campaign_seal_sha256"]):
        raise PermissionError("R4 readers were not frozen before INNER label")
    for view in VIEWS:
        expected = attack["reader_custody"][view]["private_receipt_sha256"]
        path = private / view / "readers" / "READERS_RECEIPT.json"
        if sha(path) != expected or freeze["reader_receipt_sha256"][view] != expected:
            raise PermissionError("R4 reader receipt is outside frozen bank")
        custody = json.loads(path.read_text())
        if custody.get("status") != "FROZEN" or \
                custody.get("schema") != "pcrl-portfolio-private-reader-custody-v1":
            raise PermissionError("private reader custody incomplete")
        for name, digest in custody["files"].items():
            if sha(path.parent / name) != digest:
                raise PermissionError("private reader model/view/prediction bytes differ")
    # The two allowed SEX exports are opened only after all three private
    # reader receipts and the shared freeze have passed byte verification.
    manifest = Path(spec["role_manifest_path"])
    sex_att = _role_labels(manifest, spec["role_manifest_sha256"], "ATTACK_FIT_NEW")
    sex_inner = _role_labels(manifest, spec["role_manifest_sha256"], "INNER_SELECTION")
    tune_pos, tune_groups = _tune_positions(sex_att["unit"])
    results = {}
    max_gap = 0.0
    for view in VIEWS:
        reader_dir = private / view / "readers"
        custody = json.loads((reader_dir / "READERS_RECEIPT.json").read_text())
        att = _archive(reader_dir / "view_att.npz")
        inner_view = _archive(reader_dir / "view_inner.npz")
        for role_view, labels in ((att, sex_att), (inner_view, sex_inner)):
            if (not np.array_equal(role_view["row_id"], labels["row_id"]) or
                    not np.array_equal(role_view["unit"], labels["unit"])):
                raise PermissionError("reader view and SEX export people differ")
        order = tuple(custody["reader_order"])
        tune = _archive(reader_dir / "predictions_tune.npz", set(order))
        inner = _archive(reader_dir / "predictions_inner.npz", set(order))
        scored = replay_reader_scores(sex_att["sex"][tune_pos],
                                      sex_inner["sex"], tune, inner, order)
        if (custody["validation_winner"] != scored["validation_winner"] or
                custody["validation_winner_orientation"] !=
                scored["validation_winner_orientation"] or
                custody["split"]["tune_group_id_sha256"] != array_sha(tune_groups)):
            raise ValueError("ATT-only selected reader or split differs")
        for name in order:
            for key in SCORE_FIELDS:
                before = float(custody["validation"][name][key])
                after = scored["validation"][name][key]
                max_gap = max(max_gap, abs(before - after))
                _same_number(before, after)
                before = float(attack["scores"][view]["readers"][name][key])
                after = scored["readers"][name][key]
                max_gap = max(max_gap, abs(before - after))
                _same_number(before, after)
            if (custody["validation"][name]["selected_orientation"] !=
                    ("positive" if scored["validation"][name]["positive_alias_auc"] >=
                     scored["validation"][name]["reverse_alias_auc"] else "reverse")):
                raise ValueError("ATT-only reader orientation differs")
        for key in ("primary_envelope_auc", "validation_winner_inner_auc"):
            before = float(attack["scores"][view][key])
            after = scored[key]
            max_gap = max(max_gap, abs(before - after))
            _same_number(before, after)
        if (attack["scores"][view]["validation_winner"] != scored["validation_winner"] or
                attack["scores"][view]["validation_winner_orientation"] !=
                scored["validation_winner_orientation"]):
            raise ValueError("public selected-reader identity differs")
        results[view] = {"reader_count": len(order),
                         "envelope_auc": scored["primary_envelope_auc"],
                         "validation_winner": scored["validation_winner"]}
    return {"schema": "BAR1-r5-independent-attack-auc-replay-v1",
            "status": "PASS", "source_unit_id": spec["unit_id"],
            "arm_id": spec["arm_id"], "source_seed": spec["source_seed"],
            "realization": spec["realization"], "r4_receipt_sha256": receipt_sha,
            "checked_views": results, "max_absolute_auc_gap": max_gap,
            "scope": "development INNER only; all frozen readers and ATT selection replayed"}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--r4-spec", type=Path, required=True)
    parser.add_argument("--r4-spec-sha256", required=True)
    parser.add_argument("--r4-receipt", type=Path, required=True)
    parser.add_argument("--r4-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from results.pcrl_bank_anchored_release_v1.r0.compute_queue import assert_active_from_env
    active = assert_active_from_env(stage="verification")
    out = replay(args.r4_spec, args.r4_spec_sha256,
                 args.r4_receipt, args.r4_receipt_sha256)
    out["unit_id"] = active["unit_id"]
    target = args.output.resolve(strict=False)
    expected = Path(__file__).resolve().parents[1] / "results/pcrl_bank_anchored_release_v1/r5"
    if target.parent != expected or target.exists():
        raise PermissionError("R5 independent attack receipt path differs")
    with target.open("x") as stream:
        json.dump(out, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
