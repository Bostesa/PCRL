"""Role-sized synthetic Team D resource timing; no Adult inputs or science scores.

S3 must admit this script through the global heavy-process queue. It measures
one shared 1024-code initializer, two complete epochs for protected D and the
matched task-only code, then batched release generation. It does not fit an
attacker or replace S2's calibration/attacker smoke timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from trainer import DTrainConfig, DTrainingData, pooled_initializer, train_d


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(shifted)
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


def _sample_task(rng: np.random.Generator, h: np.ndarray, classes: int) -> tuple[np.ndarray, np.ndarray]:
    weight = rng.normal(size=(h.shape[1], classes)).astype(np.float32) / math.sqrt(h.shape[1])
    p = _softmax(h @ weight)
    uniforms = rng.random(len(h))
    y = np.sum(uniforms[:, None] > np.cumsum(p, axis=1), axis=1).astype(np.int64)
    y = np.minimum(y, classes - 1)
    return y, p


def _synthetic_data(rows: int, seed: int) -> tuple[DTrainingData, DTrainingData, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    h0 = rng.standard_normal((rows, 16), dtype=np.float32)
    h1 = (0.35 * h0 + math.sqrt(1 - 0.35 ** 2) *
          rng.standard_normal((rows, 16), dtype=np.float32)).astype(np.float32)
    y0, teacher0 = _sample_task(rng, h0, 2)
    y1, teacher1 = _sample_task(rng, h1, 6)
    sex = (h0[:, 0] + h1[:, 1] > 0).astype(np.int64)
    row_ids = np.arange(rows, dtype=np.int64)
    protected = DTrainingData(h0, h1, y0, y1, teacher0, teacher1,
                              sex, row_ids, row_ids, role="SYNTHETIC")
    task_only = DTrainingData(h0, h1, y0, y1, teacher0, teacher1,
                              None, row_ids, row_ids, role="SYNTHETIC")
    return protected, task_only, (h0, h1)


def _resource_snapshot() -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    return {"cpu_seconds": usage.ru_utime + usage.ru_stime,
            "peak_rss_bytes": int(rss)}


def _timed(label: str, operation, receipt: list[dict]):
    before = _resource_snapshot()
    start = time.perf_counter()
    result = operation()
    after = _resource_snapshot()
    receipt.append({"stage": label,
                    "wall_seconds": time.perf_counter() - start,
                    "cpu_seconds": after["cpu_seconds"] - before["cpu_seconds"],
                    "process_peak_rss_bytes_after": after["peak_rss_bytes"]})
    return result


def _batched_release(fit, latents: tuple[np.ndarray, np.ndarray], rows: int) -> dict:
    seen = [0, 0]
    for purpose, latent in enumerate(latents):
        for start in range(0, rows, 256):
            out = fit.release.recipient(purpose, latent[start:min(start + 256, rows)])
            seen[purpose] += len(out["code_ids"])
    return {"recipient_rows": seen, "batch_size": 256}


def run(rows: int, epochs: int, eval_rows: int, seed: int) -> dict:
    if rows < 2048 or epochs < 2 or eval_rows < 1 or eval_rows > rows:
        raise ValueError("role-sized smoke requires at least 2048 rows, two epochs and valid evaluation rows")
    torch.set_num_threads(1)
    receipt: list[dict] = []
    protected, task_only, latents = _timed(
        "synthetic_generation", lambda: _synthetic_data(rows, seed), receipt)
    initial = _timed("pooled_1024_code_kmeans",
                     lambda: pooled_initializer(task_only, seed, 1024), receipt)
    config = replace(DTrainConfig(), epochs=epochs)
    task_fit = _timed("task_only_complete_training",
                      lambda: train_d(task_only, seed, False, config, initializer=initial), receipt)
    protected_fit = _timed("protected_complete_training",
                           lambda: train_d(protected, seed, True, config, initializer=initial), receipt)
    task_forward = _timed("task_only_batched_release_forward",
                          lambda: _batched_release(task_fit, latents, eval_rows), receipt)
    protected_forward = _timed("protected_batched_release_forward",
                               lambda: _batched_release(protected_fit, latents, eval_rows), receipt)
    return {
        "schema": "team-d-role-sized-synthetic-timing-v1",
        "scope": "synthetic resource timing only; no Adult fit or privacy/utility endpoint",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": seed, "rows": rows, "latent_dim": 16, "task_classes": [2, 6],
        "codes": 1024, "epochs_per_arm": epochs, "fit_batch_size": 256,
        "eval_rows_per_recipient": eval_rows, "threads": torch.get_num_threads(),
        "stages": receipt,
        "task_only_forward": task_forward,
        "protected_forward": protected_forward,
        "final_process_peak_rss_bytes": _resource_snapshot()["peak_rss_bytes"],
        "limitation": "Two-epoch synthetic timing with fitted heads; 20-epoch Adult and S2 calibration/attacker costs require conservative extrapolation and separate timing.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=15434)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--eval-rows", type=int, default=6299)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.rows, args.epochs, args.eval_rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
