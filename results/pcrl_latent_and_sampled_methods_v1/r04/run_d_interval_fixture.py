"""Queue-only data-free check of independent D AUC and 567-slot arithmetic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .independent_d_intervals import (ANCHORS, ARMS, VIEWS, _Reader,
                                      _Release, _margins)
from .independent_replay import d_finite_family_ids
from .replay_first_pair import _queue_guard, sha256


def _brute_auc(score: np.ndarray, sex: np.ndarray, weight: np.ndarray) -> float:
    win = 0.0
    pairs = 0.0
    for i in range(len(score)):
        for j in range(len(score)):
            if sex[i] == 1 and sex[j] == 0:
                mass = float(weight[i] * weight[j])
                pairs += mass
                win += mass * (1.0 if score[i] > score[j] else
                               0.5 if score[i] == score[j] else 0.0)
    return max(win / pairs, 1 - win / pairs)


def run(args: argparse.Namespace) -> dict:
    _queue_guard(Path(args.seal), args.seal_sha256)
    score = np.array([0.1, 0.1, 0.8, 0.8, 0.4, 0.4])
    sex = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    weight = np.array([2, 1, 1, 3, 4, 2], dtype=np.float64)
    actual = _Reader(score, sex).oriented_auc(weight)
    expected = _brute_auc(score, sex, weight)
    if abs(actual - expected) > 1e-15:
        raise AssertionError("tie/duplicate weighted AUC differs from brute pair law")
    n = len(sex)
    truth = {"row_id": np.arange(n, dtype=np.int64),
             "unit": np.arange(n, dtype=np.int64),
             "income": np.array([0, 1, 0, 1, 0, 1], dtype=np.int64),
             "occupation": np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)}
    q1 = np.tile(np.array([.6, .4]), (n, 1))
    q2 = np.tile(np.array([.3, .2, .2, .1, .1, .1]), (n, 1))
    prepared = {}
    for seed in (0, 1):
        releases = {}
        for name in (*ARMS, *ANCHORS):
            probabilities = np.tile(np.array([.8, .2]), (n, 1)) if name == "D_T" else q1
            reader_score = sex.astype(np.float64) if name == "D_T" else np.full(n, .5)
            raw = {"row_id": truth["row_id"], "unit": truth["unit"],
                   "prob_1": probabilities, "prob_2": q2,
                   "reader_predictions": {view: {"reader": reader_score} for view in VIEWS}}
            releases[name] = _Release(raw, truth, sex)
        prepared[seed] = releases
    margins = _margins(prepared, np.ones(n))
    names = d_finite_family_ids()
    if margins.shape != (567,) or len(names) != 567:
        raise AssertionError("full D finite family missing")
    by_name = dict(zip(names, margins, strict=True))
    manual_nll = float(np.mean(-np.log(np.array([.8, .2] * 3))) -
                       np.mean(-np.log(np.array([.6, .4] * 3))))
    if (abs(by_name["s0:utility:D_T:income:U0:nll"] - manual_nll) > 1e-15
            or abs(by_name["s0:arm_pair:D_T:D_K:r1:auc"] - .5) > 1e-15
            or abs(by_name["mean:arm_pair:D_T:D_K:r1:auc"] - .5) > 1e-15):
        raise AssertionError("signed task/AUC or mean-slot arithmetic differs")
    output = Path(args.public_receipt)
    if output.exists():
        raise FileExistsError("data-free D interval fixture already exists")
    receipt = {"schema": "pcrl-r04-d-interval-data-free-fixture-v1",
               "status": "DATA_FREE_INTERVAL_ARITHMETIC_PASS",
               "auc_brute_replay_max_abs": abs(actual - expected),
               "family_slots": len(names),
               "nll_manual_max_abs": abs(by_name["s0:utility:D_T:income:U0:nll"] - manual_nll),
               "pair_manual_max_abs": abs(by_name["s0:arm_pair:D_T:D_K:r1:auc"] - .5)}
    output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return {"status": receipt["status"], "public_receipt_sha256": sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", required=True)
    parser.add_argument("--seal-sha256", required=True)
    parser.add_argument("--public-receipt", required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
