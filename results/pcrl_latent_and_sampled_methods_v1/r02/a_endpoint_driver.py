"""Custody-checked A endpoint assembly from actual saved releases and readers.

This module has no role or queue access at import time. The verification worker
calls it only after admission. Row-level arrays stay in private custody; public
results contain aggregate point rows and the sealed K=6,885 intervals.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Mapping

import numpy as np

from .a_inference import (A_ARMS, A_REFS, STOCHASTIC_ARMS, VIEWS, TASKS,
                          _release_scores, run_a_inference)

CAMPAIGN = Path(__file__).resolve().parents[1]
READER_FILES = ("view_att.npz", "view_inner.npz", "predictions_tune.npz",
                "predictions_inner.npz")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
FINITE_READERS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
                  "CELL_A1", "CODE_BIT_LR_C1", "PROB_LR_C1",
                  "PROB_HGB_15x200", "Q_POST_LR_C1")
CONTINUOUS_READERS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
                      "PROB_LR_C1", "PROB_HGB_15x200")


def sha(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def expected_attack_cells(unit_plan: Mapping) -> dict[tuple[str, int, int], str]:
    """Fail closed on the prospective 40+14+4 attack-unit inventory."""
    kinds = {"attack", "reference_attack", "diagnostic_attack"}
    entries = {}
    for unit_id, plan in unit_plan.items():
        if plan.get("kind") not in kinds:
            continue
        arm = plan.get("arm_id", plan.get("reference_id"))
        key = (arm, plan.get("seed"), plan.get("realization"))
        if (not isinstance(unit_id, str) or plan.get("output_id") != unit_id or
                arm not in (*A_ARMS, *A_REFS) or plan.get("seed") not in (0, 1) or
                plan.get("realization") not in (0, 1, 2) or key in entries):
            raise ValueError("A attack unit identity/duplicate differs")
        expected_kind = ("reference_attack" if arm in A_REFS else
                         "diagnostic_attack" if arm in ("A_IDENTITY", "A_CLASS_ONLY")
                         else "attack")
        if plan["kind"] != expected_kind:
            raise ValueError("A attack kind differs from arm identity")
        entries[key] = unit_id
    expected = {(arm, seed, draw) for arm in (*A_ARMS, *A_REFS)
                for seed in (0, 1)
                for draw in (range(3) if arm in STOCHASTIC_ARMS else (0,))}
    if set(entries) != expected or len(entries) != 58:
        raise ValueError("A attack inventory is not the sealed 58 instances")
    return entries


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _view_arrays(view) -> dict[str, np.ndarray]:
    arrays = {"row_id": view.row_id, "unit": view.unit, "x": view.x}
    for name in ("x_defense_aware", "cell_key", "x_metadata", "x_code_bits", "x_raw_only"):
        value = getattr(view, name)
        if value is not None:
            arrays[name] = value
    return arrays


def verify_reader_custody(root: str | Path, prepared: Mapping, index: Mapping,
                          *, index_sha256: str) -> dict[str, dict[str, np.ndarray]]:
    """Compare saved matrix bytes and all reader vectors to reconstructed views."""
    root = Path(root)
    if ".private" not in root.resolve().parts or set(prepared) != set(VIEWS):
        raise PermissionError("private three-view attack custody required")
    index_path = root / "ATTACK_ATTEMPT_INDEX.json"
    if sha(index_path) != index_sha256 or json.loads(index_path.read_text()) != index:
        raise ValueError("attack attempt index byte/content hash differs")
    if index.get("status") != "COMPLETE" or index.get("control_status") != {v: "PASS" for v in VIEWS}:
        raise ValueError("attack attempt/control status incomplete")
    common = root / "common"
    freeze_path = common / "FROZEN_COMPLETE.json"
    if sha(freeze_path) != index.get("reader_freeze_sha256"):
        raise ValueError("ATT reader freeze hash differs")
    freeze = json.loads(freeze_path.read_text())
    if (freeze.get("status") != "FROZEN_COMPLETE" or
            freeze.get("effective_lock_sha256") != index.get("original_lock_sha256")):
        raise ValueError("ATT reader freeze status differs")
    if index.get("schema") == "SCV1-sampled-attack-attempt-v1":
        finite = True
    elif index.get("schema") == "SCV1-reference-attack-attempt-v1" and \
            index.get("reference_identity", {}).get("interface_kind") in ("finite", "continuous"):
        finite = index["reference_identity"]["interface_kind"] == "finite"
    else:
        raise ValueError("A attack attempt schema/interface differs")
    mandatory = FINITE_READERS if finite else CONTINUOUS_READERS
    predictions = {}
    for name in VIEWS:
        item = prepared[name]
        if freeze["att_view_sha256"][name] != item.att.content_hashes() or \
                freeze["inner_view_sha256"][name] != item.inner.content_hashes() or \
                index["view_content_sha256"][name] != {"att": item.att.content_hashes(),
                                                           "inner": item.inner.content_hashes()}:
            raise ValueError("complete ATT/INNER view reconstruction differs")
        reader_root = common / name / "readers"
        receipt_path = reader_root / "READERS_RECEIPT.json"
        if sha(receipt_path) != freeze["reader_receipt_sha256"][name]:
            raise ValueError("reader custody receipt hash differs")
        receipt = json.loads(receipt_path.read_text())
        expected_order = (mandatory if name != "pair" else
                          (*mandatory, *(f"IGNORE_R2__{reader}" for reader in mandatory),
                           *(f"IGNORE_R1__{reader}" for reader in mandatory)))
        expected_files = set(READER_FILES)
        for reader_id in mandatory:
            suffix = ".bin" if reader_id.startswith(("PROB_", "Q_POST_")) else ".joblib"
            expected_files.add(f"model__{reader_id}{suffix}")
        if (receipt.get("schema") != "pcrl-portfolio-private-reader-custody-v1" or
                receipt.get("status") != "FROZEN" or
                receipt["att_view_content_sha256"] != item.att.content_hashes() or
                receipt["inner_view_content_sha256"] != item.inner.content_hashes() or
                tuple(receipt.get("reader_order", ())) != expected_order):
            raise ValueError("reader receipt view identity differs")
        if set(receipt.get("files", {})) != expected_files:
            raise ValueError("reader receipt required view/prediction/model inventory differs")
        expected_alias = ([{"alias": f"IGNORE_R2__{reader}",
                            "source_view": "r1", "source_reader": reader,
                            "new_fit": False} for reader in mandatory] +
                          [{"alias": f"IGNORE_R1__{reader}",
                            "source_view": "r2", "source_reader": reader,
                            "new_fit": False} for reader in mandatory]) if name == "pair" else []
        if receipt.get("ignore_recipient_aliases") != expected_alias:
            raise ValueError("fixed reader ignore-recipient alias plan differs")
        for filename, digest in receipt["files"].items():
            if Path(filename).name != filename or sha(reader_root / filename) != digest:
                raise ValueError("reader model/view/prediction artifact hash differs")
        for role in ("att", "inner"):
            actual = _npz(reader_root / f"view_{role}.npz")
            expected = _view_arrays(getattr(item, role))
            if actual.keys() != expected.keys() or any(
                    not np.array_equal(actual[k], expected[k]) for k in expected):
                raise ValueError("saved complete attack matrix differs from public wire")
        bank = _npz(reader_root / "predictions_inner.npz")
        if set(bank) != set(receipt["reader_order"]) or not bank:
            raise ValueError("fixed reader identity differs")
        n = len(item.inner.row_id)
        for reader_id, values in bank.items():
            if (values.dtype != np.float64 or values.shape != (n,) or
                    not np.isfinite(values).all() or np.any((values < 0) | (values > 1))):
                raise ValueError(f"saved INNER reader vector invalid: {reader_id}")
        predictions[name] = bank
        control_root = common / name / "controls"
        control_path = control_root / "CUSTODY_RECEIPT.json"
        control = json.loads(control_path.read_text())
        if (control.get("schema") != "pcrl-portfolio-private-control-v2-custody-v1" or
                control.get("status") != "FROZEN" or
                control.get("att_view_content_sha256") != item.att.content_hashes() or
                control.get("inner_view_content_sha256") != item.inner.content_hashes() or
                set(control.get("files", {})) != {"shuffled_model.joblib",
                                                    "planted_model.joblib", "probe_replay.npz"}):
            raise ValueError("V2 null/planted probe custody differs")
        for filename, digest in control["files"].items():
            if sha(control_root / filename) != digest:
                raise ValueError("V2 null/planted probe file hash differs")
    pair = predictions["pair"]
    for local, prefix in (("r1", "IGNORE_R2__"), ("r2", "IGNORE_R1__")):
        for reader_id, values in predictions[local].items():
            alias = prefix + reader_id
            if alias not in pair or not np.array_equal(pair[alias], values):
                raise ValueError("zero-fit ignore-recipient alias differs")
    return predictions


def normalized_cell(row_id: np.ndarray, unit: np.ndarray, probability_1: np.ndarray,
                    probability_2: np.ndarray, predictions: Mapping) -> dict:
    row, groups = np.asarray(row_id), np.asarray(unit)
    q1, q2 = np.asarray(probability_1), np.asarray(probability_2)
    n = len(row)
    if (row.dtype != np.int64 or groups.dtype != np.int64 or row.shape != (n,) or
            groups.shape != (n,) or not np.all(np.diff(row) > 0) or
            q1.dtype != np.float64 or q2.dtype != np.float64 or
            q1.shape != (n, 2) or q2.shape != (n, 6)):
        raise ValueError("A release ordered IDs or task probability shape differs")
    for q in (q1, q2):
        if (not np.isfinite(q).all() or np.any(q < 0) or
                not np.allclose(q.sum(axis=1), 1, rtol=0, atol=1e-8)):
            raise ValueError("A emitted task probability simplex differs")
    if set(predictions) != set(VIEWS):
        raise ValueError("three frozen reader prediction banks required")
    return {"row_id": row, "unit": groups, "prob_1": q1, "prob_2": q2,
            "reader_predictions": predictions}


def _argv_options(argv: list[str]) -> dict[str, str]:
    if len(argv) < 3 or argv[1] != "-m" or len(argv[3:]) % 2:
        raise ValueError("attack worker argv shape differs")
    args = argv[3:]
    if any(not args[i].startswith("--") for i in range(0, len(args), 2)):
        raise ValueError("attack worker options differ")
    options = dict(zip((name[2:] for name in args[::2]), args[1::2]))
    if len(options) != len(args) // 2:
        raise ValueError("repeated attack worker option")
    return options


def _q_table(directory: str, identity: dict, policy_sha: str):
    from ..sampled_channel_v1.attacks.q_artifact import load_q_table_artifact
    if type(identity) is not dict or identity.keys() != {"receipt_sha256", "content_sha256"}:
        raise ValueError("Q table identity missing")
    return load_q_table_artifact(
        directory, expected_receipt_sha256=identity["receipt_sha256"],
        expected_content_sha256=identity["content_sha256"],
        expected_policy_receipt_sha256=policy_sha)


def _sampled_cell(spec: Mapping, options: Mapping):
    from ..sampled_channel_v1.attacks.runner import (load_fixed_saved_release_pair,
                                                      prepare_sampled_three_views)
    identity = spec["attack_identity"]
    att, inner, bank1, bank2, policy = load_fixed_saved_release_pair(
        options["policy-package-dir"], options["att-release-dir"],
        options["inner-release-dir"], attack_identity=identity)
    q_identity = spec.get("q_table_identity")
    q_table = None if q_identity is None else _q_table(
        options["q-table-dir"], q_identity, identity["policy_receipt_sha256"])
    views = prepare_sampled_three_views(att, inner, bank1, bank2, q_table=q_table)
    return att, inner, views.prepared, {"att": att.release_receipt_sha256,
                                        "inner": inner.release_receipt_sha256,
                                        "q": None if q_table is None else q_table.table_sha256,
                                        "policy": identity["policy_receipt_sha256"],
                                        "wire_content_sha256": views.wire_content_sha256}


def _reference_cell(spec: Mapping, options: Mapping, q_spec: Mapping | None):
    from .complete_view import pair_batch, prepare_views
    from ..sampled_channel_v1.attacks.reference_wire import (
        load_public_fixed_map, load_reference_role_package,
        reference_recipient_batch, with_reference_q_features)
    from ..sampled_channel_v1.attacks.run_reference_attack import (
        _diagnostic_q_mechanism, _fixed_q_mechanism)
    from ..sampled_channel_v1.attacks.run_q_preprocess import load_clean_public_centroids
    identity = spec["reference_identity"]
    finite = identity["interface_kind"] == "finite"
    bank1 = bank2 = fixed_receipt = q_table = mechanism_sha = None
    if finite:
        source_arm = ("T_STAR" if identity["reference_id"] in
                      ("A_IDENTITY", "A_CLASS_ONLY") else identity["reference_id"])
        fixed, fixed_receipt = load_public_fixed_map(
            options["fixed-map-receipt"], options["fixed-map-npz"],
            expected_receipt_sha256=identity["fixed_map_identity"]["receipt_sha256"],
            expected_archive_sha256=identity["fixed_map_identity"]["npz_sha256"],
            arm_id=source_arm, seed=identity["source_seed"])
        bank1, bank2 = fixed["bank_r1"], fixed["bank_r2"]
        q_table = _q_table(options["q-table-dir"], identity["q_table_identity"],
                           fixed_receipt["original_bank_receipt_sha256"])
        if source_arm != identity["reference_id"]:
            producer = json.loads(Path(options["a-amendment-path"]).read_text())
            if sha(options["a-amendment-path"]) != spec["amendment_sha256"]:
                raise ValueError("diagnostic producer amendment changed")
            audit_path = CAMPAIGN / "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json"
            if sha(audit_path) != producer["source_sha256"][
                    "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json"]:
                raise ValueError("diagnostic public source audit changed")
            audit = json.loads(audit_path.read_text())
            seed = identity["source_seed"]
            centroid_path = CAMPAIGN / f"r01/A_PUBLIC_FINE_CENTROIDS_S{seed}.npz"
            if sha(centroid_path) != producer["attack_plan"][
                    "public_centroid_npz_sha256_by_seed"][f"s{seed}"]:
                raise ValueError("diagnostic public centroids changed")
            centroids = load_clean_public_centroids(
                centroid_path, audit["public_centroids"][f"s{seed}"])
            mechanism_sha, descriptor_sha, _, bank1, bank2 = _diagnostic_q_mechanism(
                fixed, fixed_receipt, centroids,
                ref=identity["reference_id"], seed=seed,
                release_id=identity["release_id"])
        else:
            mechanism_sha, descriptor_sha = _fixed_q_mechanism(fixed, fixed_receipt)
        if (q_spec is None or q_spec.get("mechanism_sha256") != mechanism_sha or
                q_spec.get("source_descriptor_sha256") != descriptor_sha or
                q_table.identity["mechanism_sha256"] != mechanism_sha):
            raise ValueError("finite reference Q descriptor/mechanism differs")
    att = load_reference_role_package(
        options["att-release-dir"],
        receipt_sha256=identity["release_receipt_sha256_by_role"]["ATTACK_FIT_NEW"],
        full_bank1=bank1, full_bank2=bank2)
    inner = load_reference_role_package(
        options["inner-release-dir"],
        receipt_sha256=identity["release_receipt_sha256_by_role"]["INNER_SELECTION"],
        full_bank1=bank1, full_bank2=bank2)
    batches = {}
    for role, package in (("att", att), ("inner", inner)):
        left = reference_recipient_batch(package, 1)
        right = reference_recipient_batch(package, 2)
        views = {"r1": left, "r2": right, "pair": pair_batch(left, right)}
        if finite:
            views = {name: with_reference_q_features(
                batch, q_table, view=name, full_bank1=bank1, full_bank2=bank2,
                role_release=package, expected_mechanism_sha256=mechanism_sha)
                for name, batch in views.items()}
        batches[role] = views
    prepared = {name: prepare_views(batches["att"][name], batches["inner"][name])
                for name in VIEWS}
    return att, inner, prepared, {"att": att.receipt_sha256,
                                  "inner": inner.receipt_sha256,
                                  "q": None if q_table is None else q_table.table_sha256,
                                  "fixed_map": None if fixed_receipt is None else
                                  identity["fixed_map_identity"]["receipt_sha256"]}


def load_completed_cell(unit_id: str, unit: Mapping, plan: Mapping,
                        *, queue_units: Mapping | None = None) -> tuple[dict, dict]:
    """Rebuild one actual ATT/INNER wire and verify frozen reader custody."""
    if (unit.get("status") != "done" or unit.get("returncode") != 0 or
            unit.get("spec", {}).get("unit_id") != unit_id or
            unit.get("spec", {}).get("stage") != "a"):
        raise PermissionError("A attack parent is not a completed queued unit")
    spec = unit["spec"]
    options = _argv_options(spec["argv"])
    kind = plan["kind"]
    root = Path(options["private-root"])
    index_path = root / "ATTACK_ATTEMPT_INDEX.json"
    digest = unit.get("output_sha256", {}).get(str(index_path))
    if (type(digest) is not str or HEX64.fullmatch(digest) is None or
            spec.get("outputs") != [str(index_path)] or sha(index_path) != digest):
        raise PermissionError("A attack parent index output hash differs")
    index = json.loads(index_path.read_text())
    if kind == "attack":
        att, inner, prepared, source = _sampled_cell(spec, options)
        if (index.get("arm_id") != plan["arm_id"] or
                index.get("source_seed") != plan["seed"] or
                index.get("realization") != plan["realization"] or
                index.get("wire_content_sha256") != source["wire_content_sha256"] or
                index.get("q_table_sha256") != source["q"]):
            raise ValueError("sampled release/attempt identity differs")
    elif kind in ("reference_attack", "diagnostic_attack"):
        q_spec = None if plan.get("q_unit_id") is None else (
            (queue_units or {})[plan["q_unit_id"]]["spec"])
        att, inner, prepared, source = _reference_cell(spec, options, q_spec)
        if (index.get("reference_identity") != spec["reference_identity"] or
                index.get("att_release_receipt_sha256") != source["att"] or
                index.get("inner_release_receipt_sha256") != source["inner"] or
                inner.receipt["arm_id"] != plan.get("arm_id", plan.get("reference_id")) or
                inner.receipt["source_seed"] != plan["seed"] or
                index.get("q_table_content_sha256") != source["q"]):
            raise ValueError("reference release/attempt identity differs")
    else:
        raise ValueError("unregistered A attack kind")
    if index.get("a_amendment_sha256") != spec.get("amendment_sha256"):
        raise ValueError("attack attempt amendment differs")
    predictions = verify_reader_custody(root, prepared, index, index_sha256=digest)
    wires1, wires2 = inner.wires1, inner.wires2
    cell = normalized_cell(inner.row_id, inner.unit,
                           np.asarray([wire["probability"] for wire in wires1], dtype=np.float64),
                           np.asarray([wire["probability"] for wire in wires2], dtype=np.float64),
                           predictions)
    for wire_bank, q in ((wires1, cell["prob_1"]), (wires2, cell["prob_2"])):
        if not np.array_equal(np.asarray([w["decision"] for w in wire_bank]),
                              np.argmax(q, axis=1)):
            raise ValueError("emitted decision differs from score argmax")
    provenance = {"unit_id": unit_id, "arm": plan.get("arm_id", plan.get("reference_id")),
                  "seed": plan["seed"], "draw": plan["realization"],
                  "private_attack_root": str(root),
                  "private_att_release_root": options["att-release-dir"],
                  "private_inner_release_root": options["inner-release-dir"],
                  "attack_index_sha256": digest,
                  "reader_freeze_sha256": index["reader_freeze_sha256"],
                  "source": source,
                  "view_content_sha256": index["view_content_sha256"],
                  "reader_receipt_sha256": json.loads((root / "common/FROZEN_COMPLETE.json").read_text())[
                      "reader_receipt_sha256"],
                  "v2_control_receipt_sha256": {
                      view: sha(root / "common" / view / "controls/CUSTODY_RECEIPT.json")
                      for view in VIEWS}}
    return cell, provenance


def private_cell_path(directory: Path, arm: str, seed: int, draw: int) -> Path:
    if arm not in (*A_ARMS, *A_REFS) or seed not in (0, 1) or draw not in (0, 1, 2):
        raise ValueError("private normalized cell identity differs")
    return directory / f"{arm}_s{seed}_d{draw}.npz"


def save_private_cell(path: Path, cell: Mapping) -> dict:
    """Retain ordered per-person q and every fixed reader for independent R04 replay."""
    if ".private" not in path.resolve().parts or path.exists():
        raise PermissionError("normalized cell needs new private file")
    arrays = {name: cell[name] for name in ("row_id", "unit", "prob_1", "prob_2")}
    readers = {}
    for view in VIEWS:
        for reader_id, values in cell["reader_predictions"][view].items():
            key = f"reader__{view}__{reader_id}"
            if key in arrays or any(char in key for char in ("/", "\\", ".")):
                raise ValueError("normalized reader key differs")
            arrays[key] = values
            readers.setdefault(view, []).append(reader_id)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    return {"private_file": str(path), "sha256": sha(path),
            "npz_keys": sorted(arrays), "reader_ids": readers}


def point_rows(seed_records: Mapping[int, Mapping]) -> list[dict]:
    """Aggregate task/view points and observed between-draw spread."""
    rows = []
    for seed in (0, 1):
        item = seed_records[seed]
        pos = np.arange(len(item["truth"]["row_id"]), dtype=np.int64)
        for arm in (*A_ARMS, *A_REFS):
            realized = item["releases"].get(arm)
            if realized is None:
                continue
            values = []
            for draw, release in sorted(realized.items()):
                score = _release_scores(release, item["truth"], pos)
                values.append(score)
                for task, _ in TASKS:
                    for metric, point in score["task"][task].items():
                        if metric in ("accuracy", "nll", "brier"):
                            rows.append({"seed": seed, "arm": arm, "draw": draw,
                                         "scope": task, "metric": metric,
                                         "point": float(point), "draw_sd": None})
                for view in VIEWS:
                    rows.append({"seed": seed, "arm": arm, "draw": draw,
                                 "scope": view, "metric": "auc_envelope",
                                 "point": float(score["auc"][view]), "draw_sd": None})
            for scope, metric in ((*[(t, m) for t, _ in TASKS for m in ("accuracy", "nll", "brier")],
                                   *[(v, "auc_envelope") for v in VIEWS])):
                seq = [v["task"][scope][metric] if metric != "auc_envelope" else v["auc"][scope]
                       for v in values]
                rows.append({"seed": seed, "arm": arm, "draw": "mean",
                             "scope": scope, "metric": metric,
                             "point": float(np.mean(seq)),
                             "draw_sd": float(np.std(seq, ddof=1)) if len(seq) > 1 else 0.0})
    return rows


def write_aggregate(seed_records: Mapping[int, Mapping], public_json: str | Path,
                    public_csv: str | Path, public_receipt: str | Path,
                    *, binding_sha256: str, source_manifest_sha256: str) -> dict:
    """Write only aggregate outputs after all 58 normalized cells are verified."""
    for path in (public_json, public_csv, public_receipt):
        if Path(path).exists():
            raise FileExistsError("A endpoint output must be write once")
    intervals = run_a_inference(seed_records)
    rows = point_rows(seed_records)
    with Path(public_json).open("x") as stream:
        json.dump(intervals, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    with Path(public_csv).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "arm", "draw", "scope",
                                                    "metric", "point", "draw_sd"))
        writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    receipt = {"schema": "pcrl-a-endpoint-aggregate-v1", "status": "COMPLETE",
               "family": "A_10_ARMS_7_REFS_9_SUMMARIES_S1", "clauses": 6885,
               "attack_instances": 58, "point_rows": len(rows),
               "source_manifest_sha256": source_manifest_sha256,
               "private_binding_sha256": binding_sha256,
               "interval_json_sha256": sha(public_json), "point_csv_sha256": sha(public_csv),
               "bootstrap_replicates": 2048,
               "missing_clause_count": intervals["missing_clause_count"]}
    with Path(public_receipt).open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    return receipt
