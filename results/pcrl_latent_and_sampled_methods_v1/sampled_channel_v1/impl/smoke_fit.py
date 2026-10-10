"""Role-sized DATA-FREE stochastic trainer timing, for the R03 queue only.

This fixed generator has Adult-like array shapes but no Adult rows, scores,
labels or source checkpoints. It supplies resource evidence, not a method
result. R00/R03 must hash-register it before execution.
"""

from __future__ import annotations

import json
import resource
import time

import numpy as np

from bank import FixedBank, build_actions
from channel import GateWeights, JointPolicy
from fit import TaskOnlyData, ProtectedData, FitRecipe, fit_gate


ROWS = 15_434
FIXTURE_SEED = 20261009


def _bank(class_counts):
    classes = len(class_counts)
    vectors = []
    for predicted, count in enumerate(class_counts):
        for within in range(count):
            peak = .35 + .50 * (within + 1) / (count + 1)
            peak = max(peak, 1 / classes + .02)
            vector = np.full(classes, (1 - peak) / (classes - 1), np.float64)
            vector[predicted] = peak
            vectors.append(vector)
    p = np.stack(vectors)
    return FixedBank.from_source(p, np.arange(len(p), dtype=np.int64),
                                 np.argmax(p, axis=1).astype(np.int64))


def fixture():
    rng = np.random.default_rng(FIXTURE_SEED)
    b1 = _bank((8, 8))
    b2 = _bank((64, 64, 64, 64, 64, 1))
    c1 = np.repeat(np.arange(2, dtype=np.int64), 32)
    c2 = np.r_[np.repeat(np.arange(5, dtype=np.int64), 128),
               np.array([5], dtype=np.int64)]
    m1 = np.array([c * 8 + (t % 8) for t, c in enumerate(c1)], dtype=np.int64)
    m2 = np.array([c * 64 + (t % 64) if c < 5 else 320
                   for t, c in enumerate(c2)], dtype=np.int64)
    a1 = build_actions(b1, input_class=c1, fine_to_old_token=m1)
    a2 = build_actions(b2, input_class=c2, fine_to_old_token=m2)
    policy = JointPolicy(b1, b2, a1, a2, GateWeights.zeros(input_width=12))
    t1 = rng.integers(0, len(c1), size=ROWS, dtype=np.int64)
    t2 = rng.integers(0, len(c2), size=ROWS, dtype=np.int64)
    s1 = b1.probability[[b1.old_to_global[m1[t]] for t in t1]].copy()
    s2 = b2.probability[[b2.old_to_global[m2[t]] for t in t2]].copy()
    y1 = np.array([rng.choice(2, p=p) for p in s1], dtype=np.int64)
    y2 = np.array([rng.choice(6, p=p) for p in s2], dtype=np.int64)
    sex = ((t1 & 1) ^ (t2 & 1)).astype(np.int64)
    task = TaskOnlyData(s1, s2, t1, t2, y1, y2)
    return ProtectedData(task, sex), policy


def main():
    start_cpu = time.process_time()
    start_wall = time.monotonic()
    data, policy = fixture()
    # Fixed smoke caps are deliberately loose; they are not campaign caps.
    caps = {"r1_nll": 3.0, "r1_brier": 2.0,
            "r2_nll": 3.0, "r2_brier": 2.0}
    recipe = FitRecipe(updates=300, checkpoint_every=25, cpu_seconds=600,
                       seed=FIXTURE_SEED, learning_rate=.01, rho=100,
                       dual_every=25, gradient_norm=5,
                       init_scale=.01, identity_logit_bias=4)
    result = fit_gate(data, policy, objective_kind="protected_bayes",
                      caps=caps, recipe=recipe)
    print(json.dumps({
        "schema": "SCV1-role-sized-data-free-fit-smoke-v1",
        "rows": ROWS, "fine_cells": [64, 641], "public_bank": [16, 321],
        "classes": [2, 6], "max_actions_per_person": 16,
        "updates_per_start": recipe.updates, "starts": 2,
        "completed": result["complete"], "selected": result["success"],
        "completed_updates_by_start": [max(c["update"] for c in s["checkpoints"])
                                        for s in result["starts"]],
        "process_cpu_seconds": time.process_time() - start_cpu,
        "wall_seconds": time.monotonic() - start_wall,
        "max_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
