"""Data-free role-sized timing of matched controls; no Adult artifact reads.

Run only through R03's global heavy-process queue with one numerical thread.
"""

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "impl"))
from bank import FixedBank
from channel import GateWeights
from fit import FitRecipe, ProtectedData, TaskOnlyData, prepare_law

from deterministic_trainer import (DeterministicConfig, batch_from_prepared,
                                   FitResult, fit_deterministic)
from factor_controls import (FactorGate, ProductPolicy, SequentialPolicy,
                             public_probability_embedding)
from fit_local import LocalFactorData, fit_local_pair
from fit_product import fit_product
from seq_train import fit_sequential


def _bank(rng, counts):
    classes = np.repeat(np.arange(len(counts)), counts)
    raw = rng.uniform(.01, .04, size=(len(classes), len(counts)))
    raw[np.arange(len(classes)), classes] += 1.0 + np.arange(len(classes)) * 1e-5
    probability = np.asarray(raw / raw.sum(axis=1, keepdims=True), dtype=np.float64)
    return FixedBank.from_source(probability, np.arange(len(classes), dtype=np.int64),
                                 classes.astype(np.int64))


def _role(rng, bank, fine_count, n):
    f_class = np.arange(fine_count, dtype=np.int64) % bank.probability.shape[1]
    identity = []
    actions = []
    for cell, predicted in enumerate(f_class):
        allowed = np.flatnonzero(bank.decision == predicted)
        first = int(allowed[(cell // bank.probability.shape[1]) % len(allowed)])
        # Vary 1–4 actions, preserving singleton output classes.
        ordered = [first] + [int(v) for v in allowed if v != first]
        action = tuple(ordered[:min(len(ordered), 1 + cell % 4)])
        identity.append(first)
        actions.append(action)
    cell = rng.integers(0, fine_count, size=n, dtype=np.int64)
    score = np.asarray(bank.probability[np.asarray(identity)[cell]], dtype=np.float64)
    label = np.asarray(np.argmax(score, axis=1), dtype=np.int64)
    return score, cell, label, tuple(actions)


def _result_status(result):
    """The hard trainer returns a dataclass; factor trainers return mappings."""
    if isinstance(result, FitResult):
        return result.complete, result.success
    if (type(result) is dict and type(result.get("complete")) is bool and
            type(result.get("success")) is bool):
        return result["complete"], result["success"]
    raise TypeError("synthetic timing worker returned no complete/success status")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("det", "product", "local",
                                          "seq12", "seq21"), required=True)
    parser.add_argument("--rows", type=int, default=15434)
    parser.add_argument("--cpu-limit", type=float, default=900)
    args = parser.parse_args()
    if args.rows != 15434 or args.cpu_limit <= 0:
        raise ValueError("registered role-sized timing uses exactly 15434 synthetic rows")
    rng = np.random.default_rng(20261009)
    started = time.process_time()
    bank1, bank2 = _bank(rng, (8, 8)), _bank(rng, (64, 64, 64, 64, 64, 1))
    score1, cell1, y1, actions1 = _role(rng, bank1, 64, args.rows)
    score2, cell2, y2, actions2 = _role(rng, bank2, 641, args.rows)
    sex = rng.integers(0, 2, size=args.rows, dtype=np.int64)
    units = tuple(f"synthetic-{i}" for i in range(args.rows))
    caps = dict(r1_nll=10.0, r1_brier=10.0, r2_nll=10.0, r2_brier=10.0)
    gate = GateWeights.zeros(input_width=12)
    dummy = FactorGate(np.zeros((12, 16)), np.zeros(16),
                       np.zeros((16, 4)), np.zeros(4))
    policy = ProductPolicy(bank1, bank2, actions1, actions2, gate, dummy, dummy)
    if args.arm == "product":
        task = TaskOnlyData(score1, score2, cell1, cell2, y1, y2)
        result = fit_product(ProtectedData(task, sex), policy, caps=caps,
                             recipe=FitRecipe(updates=300, checkpoint_every=25,
                                              cpu_seconds=args.cpu_limit,
                                              seed=2701))
    elif args.arm == "local":
        data1 = LocalFactorData(score1, cell1, y1, sex, bank1, actions1, units)
        data2 = LocalFactorData(score2, cell2, y2, sex, bank2, actions2, units)
        result = fit_local_pair(data1, data2, gate=gate,
                                caps1=(10.0, 10.0), caps2=(10.0, 10.0),
                                recipe1=FitRecipe(updates=150, checkpoint_every=25,
                                                  cpu_seconds=args.cpu_limit,
                                                  seed=2701),
                                recipe2=FitRecipe(updates=150, checkpoint_every=25,
                                                  cpu_seconds=args.cpu_limit,
                                                  seed=2801))
    elif args.arm.startswith("seq"):
        order = args.arm[-2:]
        first_bank = bank1 if order == "12" else bank2
        second = FactorGate(np.zeros((12, 16)), np.zeros(16),
                            np.zeros((16, 4)), np.zeros(4),
                            public_probability_embedding(first_bank.probability))
        seq = SequentialPolicy(bank1, bank2, actions1, actions2, gate,
                               dummy, second, order=order,
                               first_checkpoint_sha256="0" * 64)
        task = TaskOnlyData(score1, score2, cell1, cell2, y1, y2)
        with tempfile.TemporaryDirectory() as directory:
            os.chmod(directory, 0o700)
            stage_ceiling = args.cpu_limit / 2
            result = fit_sequential(
                ProtectedData(task, sex), seq, caps=caps,
                recipe_stage1=FitRecipe(updates=150, checkpoint_every=25,
                                        cpu_seconds=stage_ceiling, seed=2701),
                recipe_stage2=FitRecipe(updates=150, checkpoint_every=25,
                                        cpu_seconds=stage_ceiling, seed=2801),
                total_cpu_seconds=args.cpu_limit, unit_ids=units,
                private_root=b"synthetic-fourth-root-only-00000",
                source_seed=0, source_binding_sha256="0" * 64,
                private_draw_cache_path=Path(directory) / "seq-fit-draw.json")
    else:
        task = TaskOnlyData(score1, score2, cell1, cell2, y1, y2)
        prepared = prepare_law(task, policy)
        identity = np.asarray([(actions1[int(c1)][0], actions2[int(c2)][0])
                               for c1, c2 in zip(cell1, cell2)], dtype=np.int64)
        batch = batch_from_prepared(prepared, identity_public_pair=identity)
        result = fit_deterministic(batch, DeterministicConfig(
            mode="task", steps=300, learning_rate=.01,
            temperature_start=1.0, temperature_end=.5, rho=100.0,
            dual_every=25, replay_every=25, grad_clip=5.0,
            cpu_limit_seconds=args.cpu_limit, seed=2701,
            bias_deltas=(-.5, -.1, .1, .5)), caps=(10.0,) * 4)
    complete, success = _result_status(result)
    print(json.dumps({"synthetic_only": True, "arm": args.arm,
                      "rows": args.rows, "fine_cells": [64, 641],
                      "bank_sizes": [len(bank1.probability), len(bank2.probability)],
                      "complete": complete, "success": success,
                      "cpu_seconds_total": time.process_time() - started,
                      "maxrss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
                     sort_keys=True))


if __name__ == "__main__":
    main()
