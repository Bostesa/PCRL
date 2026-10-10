"""Queue-only utility scorer for bank-anchored laws on admitted role exports.

No scientific role data is read on import. The command entry point is added
below once R0's queue configuration is sealed. These boundary helpers refuse
mixed or mismatched per-person artifacts before any endpoint is calculated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np

from .r3_utility import METRICS, TASKS, _costs, evaluate_law, point_clause


LAW_FIELDS = {"row_id", "unit", "code_r1", "code_r2", "mass", "valid",
              "b0_code_r1", "b0_code_r2"}
WIRE_FIELDS = {"row_id", "unit", "code0", "code1", "prob0", "prob1",
               "decision0", "decision1"}
CONFIG_FIELDS = {"schema", "unit_id", "stage", "role", "arm", "seed", "seal",
                 "role_manifest", "public_table", "bank_receipt", "bank_wire",
                 "r5_source_replay", "law", "outputs"}
OUTPUT_FIELDS = {"csv", "private_vectors", "receipt"}
CSV_FIELDS = ("role", "arm", "seed", "task", "reference", "metric",
              "candidate_value", "reference_value", "margin", "cap",
              "primary_gate", "point_status", "diagnostic_point_status")


def load_law_npz(path: Path, row_id: np.ndarray, unit: np.ndarray) -> dict:
    """Open only the fixed private law members and exact admitted rows."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != LAW_FIELDS:
            raise PermissionError("law archive member inventory differs")
        law = {key: np.array(archive[key], copy=True) for key in LAW_FIELDS}
    if (not np.array_equal(law["row_id"], row_id)
            or not np.array_equal(law["unit"], unit)):
        raise PermissionError("law row/group identity differs")
    return law


def identity_law_from_wire(wire: dict) -> dict:
    """Construct the exact B0 law from its saved public ten-bit codes."""
    if set(wire) != WIRE_FIELDS:
        raise PermissionError("B0 wire archive inventory differs")
    n = len(wire["row_id"])
    shifts = (1 << np.arange(10, dtype=np.int64))[None, :]
    out = {"row_id": np.asarray(wire["row_id"]), "unit": np.asarray(wire["unit"])}
    for purpose in (0, 1):
        bits = np.asarray(wire[f"code{purpose}"])
        if (bits.dtype != np.uint8 or bits.shape != (n, 10)
                or not np.isin(bits, (0, 1)).all()):
            raise PermissionError("B0 ten-bit source code differs")
        ids = np.sum(bits.astype(np.int64) * shifts, axis=1)
        out[f"b0_code_r{purpose + 1}"] = ids
        out[f"code_r{purpose + 1}"] = np.repeat(ids[:, None], 16, axis=1)
    out["mass"] = np.zeros((n, 16), dtype=np.float64)
    out["mass"][:, 0] = 1.
    out["valid"] = np.zeros((n, 16), dtype=np.bool_)
    out["valid"][:, 0] = True
    return out


def verify_bank_wire(wire: dict, law: dict,
                     tables: tuple[np.ndarray, np.ndarray],
                     row_id: np.ndarray, unit: np.ndarray) -> None:
    """Match law's immutable B0 IDs to the admitted ten-bit source release."""
    n = len(row_id)
    if (set(wire) != WIRE_FIELDS or not np.array_equal(wire["row_id"], row_id)
            or not np.array_equal(wire["unit"], unit)):
        raise PermissionError("B0 wire member or row/group identity differs")
    shifts = (1 << np.arange(10, dtype=np.int64))[None, :]
    for purpose, width in ((0, 2), (1, 6)):
        bits = np.asarray(wire[f"code{purpose}"])
        prob = np.asarray(wire[f"prob{purpose}"])
        decision = np.asarray(wire[f"decision{purpose}"])
        table = np.asarray(tables[purpose])
        if (bits.dtype != np.uint8 or bits.shape != (n, 10)
                or not np.isin(bits, (0, 1)).all()
                or prob.dtype != np.float64 or prob.shape != (n, width)
                or decision.dtype != np.int64 or decision.shape != (n,)):
            raise PermissionError("B0 ten-bit wire schema differs")
        ids = np.sum(bits.astype(np.int64) * shifts, axis=1)
        if (np.any(ids >= len(table))
                or not np.array_equal(ids, law[f"b0_code_r{purpose + 1}"])
                or not np.array_equal(prob, table[ids])
                or not np.array_equal(decision, prob.argmax(axis=1))):
            raise PermissionError("B0 code, probability or decision differs")


def score_rows(law: dict, tables: tuple[np.ndarray, np.ndarray],
               labels: tuple[np.ndarray, np.ndarray], references: dict,
               *, role: str, arm: str, seed: int) -> tuple[list[dict], dict]:
    """Build signed aggregate margins and private per-person cost vectors.

    FIT's only primary constraint reference is B0. INNER's only primary
    evaluation references are U0 and Ucal. All other rows are diagnostics.
    """
    if role not in ("OSF_DEFENSE_FIT", "INNER_SELECTION") or seed not in (0, 1):
        raise ValueError("utility role or paired seed invalid")
    if not isinstance(arm, str) or not arm or set(references) != {
            "U0_1", "Ucal_1", "U0_2", "Ucal_2"}:
        raise ValueError("arm or exact reference inventory invalid")
    evaluated = evaluate_law(law, tables, labels)
    n = len(labels[0])
    rows = []
    vectors = {}
    for task, purpose, width in TASKS:
        y = np.asarray(labels[purpose])
        vals = {"candidate": evaluated[task]["expected"],
                "B0": evaluated[task]["b0"]}
        for name in ("U0", "Ucal"):
            prob = np.asarray(references[f"{name}_{purpose + 1}"])
            if (prob.dtype != np.float64 or prob.shape != (n, width)
                    or not np.isfinite(prob).all() or np.any(prob < 0)
                    or not np.allclose(prob.sum(axis=1), 1, rtol=0, atol=1e-8)):
                raise ValueError("reference probability schema invalid")
            vals[name] = _costs(prob, y)
        for name, costs in vals.items():
            for metric in METRICS:
                vectors[f"{name}_{task}_{metric}"] = costs[metric]
        for reference in ("B0", "U0", "Ucal"):
            primary = (reference == "B0" if role == "OSF_DEFENSE_FIT"
                       else reference in ("U0", "Ucal"))
            regime = "fit" if role == "OSF_DEFENSE_FIT" and reference == "B0" else "inner"
            for metric in METRICS:
                candidate = float(np.mean(vals["candidate"][metric]))
                comparator = float(np.mean(vals[reference][metric]))
                difference = candidate - comparator
                clause = point_clause(metric, difference, regime)
                rows.append({"role": role, "arm": arm, "seed": seed,
                             "task": task, "reference": reference,
                             "metric": metric, "candidate_value": candidate,
                             "reference_value": comparator,
                             "margin": difference, "cap": clause["cap"],
                             "primary_gate": primary,
                             "point_status": clause["status"] if primary else "DIAGNOSTIC_ONLY",
                             "diagnostic_point_status": clause["status"] if not primary else ""})
    if len(rows) != 18:
        raise AssertionError("two tasks by three metrics by three references required")
    return rows, vectors


def validate_job_config(config: dict, active: dict) -> None:
    """Reject any undeclared role, queue stage, seed or extra input field."""
    stage = config.get("stage")
    arm = config.get("arm")
    if (set(config) != CONFIG_FIELDS
            or config.get("schema") != "pcrl-bank-r3-utility-job-v1"
            or config.get("unit_id") != active.get("unit_id")
            or stage != active.get("stage")
            or stage not in ("witness", "protection")
            or config.get("role") not in ("OSF_DEFENSE_FIT", "INNER_SELECTION")
            or config.get("seed") not in (0, 1)
            or (arm != "B0" if stage == "witness" else
                arm not in ("B0", "T", "D", "P", "L", "J", "S12", "S21"))
            or (config.get("law") is not None if stage == "witness" else
                not isinstance(config.get("law"), dict))
            or set(config.get("outputs", {})) != OUTPUT_FIELDS):
        raise PermissionError("R3 job config, stage or role differs")
    for name in ("seal", "role_manifest", "public_table", "bank_receipt",
                 "bank_wire", "r5_source_replay") + (() if stage == "witness" else ("law",)):
        item = config.get(name)
        if (not isinstance(item, dict) or set(item) != {"path", "sha256"}
                or not isinstance(item["sha256"], str)
                or len(item["sha256"]) != 64):
            raise PermissionError("R3 input record is not byte-pinned")


def validate_source_replay(replay: dict) -> None:
    bank = replay.get("summary", {}).get("W_TSTAR", {})
    if (replay.get("schema") != "pcrl-r5-bank-old-point-audit-v1"
            or replay.get("status") != "PASS"
            or bank.get("clauses_pass") != 24
            or bank.get("clauses_total") != 24):
        raise PermissionError("independent source bank replay not admitted")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pinned(record: dict, queue_sources: dict[Path, str]) -> Path:
    path = Path(record["path"])
    if path.is_symlink():
        raise PermissionError("R3 source symlink refused")
    path = path.resolve(strict=True)
    if queue_sources.get(path) != record["sha256"] or _sha(path) != record["sha256"]:
        raise PermissionError("R3 source differs from active queue pin")
    return path


def _new_file(path: Path, *, private: bool):
    path = path.resolve(strict=False)
    here = Path(__file__).resolve().parents[1] / "results/pcrl_bank_anchored_release_v1/r3"
    expected_parent = here / ".private" if private else here
    if path.parent != expected_parent or path.is_symlink():
        raise PermissionError("R3 output must remain in owner namespace")
    if private and (not expected_parent.is_dir()
                    or stat.S_IMODE(expected_parent.stat().st_mode) & 0o077):
        raise PermissionError("R3 private output directory is not owner-only")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    return os.fdopen(fd, "wb")


def run(config_path: Path) -> dict:
    """Execute one byte-pinned FIT or INNER unit under the global queue."""
    from results.pcrl_bank_anchored_release_v1.r0.compute_queue import assert_active_from_env
    from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
        RoleOnlyStore, assert_same_people, sha_array,
    )

    active = assert_active_from_env()
    config_path = config_path.resolve(strict=True)
    if (".private" not in config_path.parts or config_path.is_symlink()
            or stat.S_IMODE(config_path.stat().st_mode) & 0o077):
        raise PermissionError("R3 config must be private")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    sources = {Path(record["path"]).resolve(strict=True): record["sha256"]
               for record in state["units"][active["unit_id"]]["spec"]["sources"]}
    if sources.get(config_path) != _sha(config_path):
        raise PermissionError("R3 config is not byte-pinned in active queue")
    config = json.loads(config_path.read_text())
    validate_job_config(config, active)
    if (config["seal"]["sha256"] != active["seal_sha256"]
            or not {Path(__file__).resolve(),
                    (Path(__file__).with_name("r3_utility.py")).resolve()}.issubset(sources)):
        raise PermissionError("R3 seal or worker source not registered")
    _pinned(config["seal"], sources)
    for source in (Path(__file__).resolve(), Path(__file__).with_name("r3_utility.py").resolve()):
        if _sha(source) != sources[source]:
            raise PermissionError("R3 worker source changed after queue registration")
    role_path = _pinned(config["role_manifest"], sources)
    store = RoleOnlyStore(role_path)
    role = config["role"]
    for field in ("income", "occupation", f"references_s{config['seed']}"):
        key = f"{role}:{field}"
        path = store.package_root / store.manifest["role_file_relative_path"][key]
        _pinned({"path": str(path), "sha256": store.manifest["role_file_sha256"][key]}, sources)
    procedure = "task_fit" if role == "OSF_DEFENSE_FIT" else "screen"
    income = store.read(role, "income", procedure)
    occupation = store.read(role, "occupation", procedure)
    references = store.read(role, f"references_s{config['seed']}")
    assert_same_people(income, occupation, references)
    row, unit = income["row_id"], income["unit"]
    labels = (income["income"], occupation["occupation"])

    replay = json.loads(_pinned(config["r5_source_replay"], sources).read_text())
    validate_source_replay(replay)
    table_doc = json.loads(_pinned(config["public_table"], sources).read_text())
    if (table_doc.get("schema") != "code-utility-public-bank-tables-v1"
            or table_doc.get("seed") != config["seed"]
            or table_doc.get("bit_count") != 10):
        raise PermissionError("fixed W_TSTAR table identity differs")
    family = table_doc["families"]["W_TSTAR"]
    tables = (np.asarray(family["table0"], dtype=np.float64),
              np.asarray(family["table1"], dtype=np.float64))
    receipt = json.loads(_pinned(config["bank_receipt"], sources).read_text())
    admitted = receipt["result"]["roles"][role]
    bank = admitted["arms"]["W_TSTAR"]
    if (receipt.get("status") != "COMPLETE"
            or receipt.get("seed") != config["seed"]
            or receipt.get("role_manifest_sha256") != config["role_manifest"]["sha256"]
            or receipt.get("source_model_sha256") != table_doc["source_model_sha256"]
            or admitted["row_unit_pair_sha256"] != sha_array(np.column_stack((row, unit)))
            or bank["public_table_sha256"] != config["public_table"]["sha256"]
            or bank["decoder_sha256"] != family["decoder_sha256"]
            or bank["policy_sha256"] != family["policy_sha256"]
            or bank["private_release_sha256"] != config["bank_wire"]["sha256"]):
        raise PermissionError("B0 source receipt does not bind admitted role/table/wire")
    wire_path = _pinned(config["bank_wire"], sources)
    if ".private" not in wire_path.parts:
        raise PermissionError("B0 person wire is not private")
    with np.load(wire_path, allow_pickle=False) as archive:
        if set(archive.files) != WIRE_FIELDS:
            raise PermissionError("B0 wire archive inventory differs")
        wire = {key: np.array(archive[key], copy=True) for key in WIRE_FIELDS}
    if any(sha_array(wire[key]) != value
           for key, value in bank["array_sha256"].items()):
        raise PermissionError("B0 wire content differs from source receipt")
    if active["stage"] == "witness":
        law = identity_law_from_wire(wire)
    else:
        law_path = _pinned(config["law"], sources)
        if ".private" not in law_path.parts:
            raise PermissionError("policy law is not private")
        law = load_law_npz(law_path, row, unit)
    verify_bank_wire(wire, law, tables, row, unit)
    rows, vectors = score_rows(law, tables, labels, references, role=role,
                               arm=config["arm"], seed=config["seed"])
    primary = [item for item in rows if item["primary_gate"]]
    if len(primary) != (6 if role == "OSF_DEFENSE_FIT" else 12):
        raise AssertionError("R3 primary gate inventory differs")

    outputs = config["outputs"]
    csv_path = Path(outputs["csv"])
    private_path = Path(outputs["private_vectors"])
    receipt_path = Path(outputs["receipt"])
    with _new_file(private_path, private=True) as stream:
        np.savez_compressed(stream, row_id=row, unit=unit, **vectors)
        stream.flush()
        os.fsync(stream.fileno())
    with _new_file(csv_path, private=False) as stream:
        text = __import__("io").TextIOWrapper(stream, encoding="utf-8", newline="", write_through=True)
        writer = csv.DictWriter(text, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        text.flush()
        os.fsync(stream.fileno())
        text.detach()
    receipt_out = {
        "schema": "pcrl-bank-r3-utility-receipt-v1", "unit_id": active["unit_id"],
        "stage": active["stage"], "role": role, "seed": config["seed"],
        "arm": config["arm"], "source_seal_sha256": active["seal_sha256"],
        "role_manifest_sha256": config["role_manifest"]["sha256"],
        "source_bank_replay_sha256": config["r5_source_replay"]["sha256"],
        "bank_receipt_sha256": config["bank_receipt"]["sha256"],
        "bank_wire_sha256": config["bank_wire"]["sha256"],
        "law_sha256": None if config["law"] is None else config["law"]["sha256"],
        "public_table_sha256": config["public_table"]["sha256"],
        "row_unit_pair_sha256": sha_array(np.column_stack((row, unit))),
        "rows": len(row), "groups": len(np.unique(unit)),
        "primary_point_pass": all(item["point_status"] == "PASS" for item in primary),
        "primary_clause_count": len(primary),
        "primary_clause_pass_count": sum(item["point_status"] == "PASS" for item in primary),
        "output_csv_sha256": _sha(csv_path),
        "private_vector_sha256": _sha(private_path),
        "scope": "exploratory FIT proxy or common INNER utility; no SEX label opened",
    }
    with _new_file(receipt_path, private=False) as stream:
        stream.write((json.dumps(receipt_out, sort_keys=True, indent=2) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    return receipt_out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config), sort_keys=True))


if __name__ == "__main__":
    main()
