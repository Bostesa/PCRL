"""Role-sized, data-free timing of the complete A one-draw attack slate.

Only R03 may launch this as a registered stage-a synthetic queue unit. All
scores, labels, IDs and public centroids below are generated in memory; no
Adult role store or source export is imported or opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r02 import complete_view, protocol
from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import assert_active_from_env
from . import q_aware, runner, wire


ATT_ROWS = 4064
INNER_ROWS = 2235
FINE_CELLS = (64, 641)
PUBLIC_CODES = (16, 321)
EXPECTED_FITS = 33  # 3 Q, 18 core, 6 probability, 6 V2 controls.


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _admit(args: argparse.Namespace) -> tuple[Path, Path]:
    active = assert_active_from_env(unit_id=args.unit_id, stage="a")
    lock = Path(args.original_lock).resolve(strict=True)
    amendment_path = Path(args.amendment).resolve(strict=True)
    if (_sha(lock) != args.original_lock_sha256 or
            _sha(amendment_path) != args.amendment_sha256 or
            active["seal_sha256"] != args.original_lock_sha256):
        raise PermissionError("synthetic A smoke seal SHA differs")
    original = json.loads(lock.read_text())
    if (original.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1" or
            original.get("status") != "LOCALLY_SEALED_D_STAGE"):
        raise PermissionError("original D-stage campaign seal is absent")
    amendment = json.loads(amendment_path.read_text())
    if (amendment.get("schema") != "pcrl-latent-sampled-dated-amendment-v1" or
            amendment.get("status") != "LOCALLY_SEALED_SYNTHETIC_A" or
            amendment.get("input_scope") != "SYNTHETIC_ONLY" or
            amendment.get("stage_scope") != ["a"] or
            amendment.get("authorized_unit_id") != args.unit_id or
            amendment.get("original_lock_sha256") != args.original_lock_sha256 or
            amendment.get("implementation_source_sha256") !=
                _sha(Path(__file__).resolve(strict=True))):
        raise PermissionError("dated A synthetic-only amendment is absent")
    queue_root = Path(os.environ["PCRL_QUEUE_ROOT"]).resolve(strict=True)
    state = json.loads((queue_root / "state.json").read_text())
    spec = state["units"][args.unit_id]["spec"]
    latest = [item for item in state["amendments"] if "a" in item.get("stages", [])]
    if (not latest or latest[-1].get("sha256") != args.amendment_sha256 or
            Path(latest[-1].get("path", "")).resolve() != amendment_path or
            spec.get("amendment_sha256") != args.amendment_sha256 or
            spec.get("stage") != "a" or spec.get("data_access") != "SYNTHETIC_ONLY" or
            spec.get("deps") != ["R04-A-PREFIT-SOURCE-001"]):
        raise PermissionError("synthetic A unit differs from latest queue amendment")
    out = Path(args.private_root).resolve()
    if (Path(args.private_root).is_symlink() or ".private" not in out.parts or
            out.exists() or spec.get("outputs") != [str(out / "SMOKE_RECEIPT.json")]):
        raise PermissionError("synthetic smoke output differs from immutable queue custody")
    shared = Path(protocol.__file__).resolve().parents[2] / "pcrl_method_portfolio_v1" / "shared"
    required = (lock, amendment_path,
                *(Path(module.__file__).resolve(strict=True) for module in
                  (complete_view, protocol, q_aware, runner, wire)),
                *(shared / name for name in ("attack_eval.py", "attack_controls_v2.py",
                                              "common_eval.py", "safe_data_guard.py")),
                Path(__file__).resolve(strict=True))
    declared = {Path(item["path"]).resolve(): item["sha256"]
                for item in spec.get("sources", [])}
    for path in required:
        if declared.get(path) != _sha(path):
            raise PermissionError("synthetic smoke source differs from immutable queue spec")
    return lock, out


def _scores(count: int, classes: int, codes: int) -> tuple[np.ndarray, np.ndarray]:
    """Unique, strict-decision public vectors and class-balanced centroids."""
    bank = np.empty((codes, classes), dtype=np.float64)
    for code in range(codes):
        winner = code % classes
        strength = 0.55 + 0.0002 * (code // classes + 1)
        bank[code] = (1.0 - strength) / (classes - 1)
        bank[code, winner] = strength
    centroids = np.empty((count, classes), dtype=np.float64)
    for i in range(count):
        winner = i % classes
        strength = 0.65 + 0.0001 * (i // classes + 1)
        centroids[i] = (1.0 - strength) / (classes - 1)
        centroids[i, winner] = strength
    return bank, centroids


def _bank(recipient: int, probability: np.ndarray) -> wire.PublicBank:
    release_id = "SYNTHETIC_ROLE_SIZED_SMOKE"
    metadata = {
        "model_id": release_id,
        "source_binding_sha256": "a" * 64,
        "canonical_bank_sha256": wire.canonical_bank_sha256(probability),
        "mechanism_sha256": "b" * 64,
        "code_length_bits": (len(probability) - 1).bit_length(),
        "fallback": False,
        "sampling_contract": wire.SAMPLING_CONTRACT,
    }
    return wire.PublicBank(recipient, probability, release_id, metadata)


def _role(name: str, first: int, count: int, banks: tuple[wire.PublicBank, wire.PublicBank],
          rng: np.random.Generator) -> tuple[runner.SampledRoleRelease, dict]:
    duplicates = 3 if name == "ATTACK_FIT_NEW" else 1
    if count <= duplicates:
        raise ValueError("synthetic role needs more rows than repeated groups")
    row_id = np.arange(first, first + count, dtype=np.int64)
    unit = row_id.copy()
    unit[-duplicates:] = unit[:duplicates]
    code1 = rng.integers(0, PUBLIC_CODES[0], size=count)
    code2 = rng.integers(0, PUBLIC_CODES[1], size=count)
    code1[-duplicates:] = code1[:duplicates]
    code2[-duplicates:] = code2[:duplicates]
    wires = []
    for bank, codes in zip(banks, (code1, code2), strict=True):
        wires.append(tuple({
            "schema": wire.SCHEMA,
            "recipient": bank.recipient,
            "code_id": int(code),
            "probability": [float(value) for value in bank.probability[code]],
            "decision": int(np.argmax(bank.probability[code])),
            "release_id": bank.release_id,
            "public_metadata": dict(bank.public_metadata),
        } for code in codes))
    cache = wire.CacheIdentity(0, name, 0, banks[0].release_id,
                               "a" * 64, "c" * 64)
    package = runner.SampledRoleRelease(name, wires[0], wires[1], row_id, unit,
                                        cache, cache, "d" * 64, "e" * 64)
    strength = 0.2 + 0.6 * ((code1 % 3 == 0) | (code2 % 5 == 0))
    sex = (rng.random(count) < strength).astype(np.int64)
    sex[-duplicates:] = sex[:duplicates]
    return package, {"row_id": row_id, "unit": unit, "sex": sex}


def _synthetic_table(banks: tuple[wire.PublicBank, wire.PublicBank],
                     centroids: tuple[np.ndarray, np.ndarray]) -> q_aware.PrototypeFeatureTable:
    by_class = []
    for bank in banks:
        by_class.append({label: np.flatnonzero(np.argmax(bank.probability, axis=1) == label)
                         for label in range(bank.probability.shape[1])})

    def sparse_law(score1: np.ndarray, score2: np.ndarray, i: int, j: int):
        left = by_class[0][int(np.argmax(score1))]
        right = by_class[1][int(np.argmax(score2))]
        return ((int(left[(i // 2) % len(left)]),
                 int(right[(j // 6) % len(right)]), 1.0),)

    return q_aware.compile_uniform_prototype_features(
        banks[0], banks[1], centroids[0], centroids[1], sparse_law,
        public_centroids_sha256="f" * 64)


def run(args: argparse.Namespace) -> dict:
    lock, out = _admit(args)
    guard = lambda: assert_active_from_env(unit_id=args.unit_id, stage="a")
    guard()
    start_wall, start_cpu = time.monotonic(), time.process_time()
    rng = np.random.default_rng(20261008)
    probability1, centroids1 = _scores(FINE_CELLS[0], 2, PUBLIC_CODES[0])
    probability2, centroids2 = _scores(FINE_CELLS[1], 6, PUBLIC_CODES[1])
    banks = (_bank(1, probability1), _bank(2, probability2))
    q_start = time.monotonic()
    table = _synthetic_table(banks, (centroids1, centroids2))
    q_wall = time.monotonic() - q_start
    guard()
    att, sex_att = _role("ATTACK_FIT_NEW", 0, ATT_ROWS, banks, rng)
    inner, sex_inner = _role("INNER_SELECTION", ATT_ROWS, INNER_ROWS, banks, rng)
    if (np.unique(att.unit).size != 4061 or
            np.unique(inner.unit).size != 2234):
        raise RuntimeError("synthetic role group counts differ from inherited split")
    views = runner.prepare_sampled_three_views(att, inner, *banks, q_table=table)
    out.mkdir(parents=True, mode=0o700)
    q_readers = {}
    for name in ("r1", "r2", "pair"):
        guard()
        q_readers[name] = (q_aware.fit_q_posterior_reader(
            views.att_batches[name], views.inner_batches[name], sex_att, table,
            view=name, attacker_seed=0, private_dir=out / "q_readers" / name,
            lock_path=lock, lock_sha256=args.original_lock_sha256,
            expected_table_sha256=table.table_sha256,
            queue_unit_id=args.unit_id, queue_guard=guard),)

    def load_inner_sex() -> dict:
        guard()
        if not (out / "common" / "FROZEN_COMPLETE.json").is_file():
            raise RuntimeError("synthetic INNER labels opened before reader freeze")
        return sex_inner

    result = protocol.run_three_view_attack(
        views.prepared, sex_att, load_inner_sex=load_inner_sex,
        family="A", seed=0, private_root=out / "common",
        lock_path=lock, lock_sha256=args.original_lock_sha256,
        queue_guard=guard, specialists_by_view=q_readers)
    counts = protocol.unit_fit_forecast(protocol.reader_plan(family="A", code_bits=True))
    if counts["mandatory_fits"] + len(q_readers) != EXPECTED_FITS:
        raise RuntimeError("synthetic smoke reader-fit count differs from sealed plan")
    receipt = {
        "schema": "SCV1-data-free-A-attack-smoke-v1", "status": "COMPLETE",
        "unit_id": args.unit_id, "synthetic_only": True,
        "adult_role_values_read": False, "att_rows": ATT_ROWS, "inner_rows": INNER_ROWS,
        "fine_cells": list(FINE_CELLS), "public_codes": list(PUBLIC_CODES),
        "q_prototype_count": table.prototype_count,
        "q_table_sha256": table.table_sha256,
        "q_compile_wall_s": q_wall,
        "fit_count": EXPECTED_FITS,
        "fit_breakdown": {**counts, "q_specialist_fits": len(q_readers)},
        "wall_s": time.monotonic() - start_wall,
        "cpu_s": time.process_time() - start_cpu,
        "ru_maxrss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "reader_freeze_sha256": result["public"]["freeze_receipt_sha256"],
        "view_schema_sha256": result["public"]["view_schema_sha256"],
        "source_sha256": _sha(Path(__file__).resolve()),
    }
    receipt_path = out / "SMOKE_RECEIPT.json"
    fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--original-lock", required=True)
    parser.add_argument("--original-lock-sha256", required=True)
    parser.add_argument("--amendment", required=True)
    parser.add_argument("--amendment-sha256", required=True)
    parser.add_argument("--private-root", required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "fit_count": result["fit_count"],
                      "wall_s": result["wall_s"], "cpu_s": result["cpu_s"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
