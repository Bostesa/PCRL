"""Data-free checks for aggregate-only old F publication."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from results.pcrl_utility_reference_repair_v1.r1.publish_old_f_diagnostic import (
    FIELDS, TASKS, REFERENCES, build_rows,
)


def fixture_receipts():
    old, new = {}, {}
    replay = {
        "schema": "pcrl-utility-r3-old-f-inner-replay-v1",
        "status": "OBSERVATIONAL_DIAGNOSTIC_REPLAYED", "by_seed": {},
    }
    for seed in (0, 1):
        old_metrics = {"old_validation_F": {}, "U0": {}, "Ucal": {}}
        new_metrics = {"F": {}, "U0": {}, "Ucal": {}}
        independent = {"metrics": {"old_validation_F": {}, "new_all_FIT_F": {}},
                       "reference_metrics": {"U0": {}, "Ucal": {}},
                       "old_minus_new": {},
                       "max_abs_old_receipt_metric_difference": 0.0}
        for task_i, task in enumerate(TASKS):
            a = {"accuracy": .75 + .01 * seed, "nll": .45 + .02 * task_i,
                 "brier": .25 + .01 * task_i}
            b = {"accuracy": a["accuracy"] + .01,
                 "nll": a["nll"] - .02, "brier": a["brier"] - .005}
            old_metrics["old_validation_F"][task] = a
            new_metrics["F"][task] = b
            independent["metrics"]["old_validation_F"][task] = a.copy()
            independent["metrics"]["new_all_FIT_F"][task] = b.copy()
            independent["old_minus_new"][task] = {
                key: a[key] - b[key] for key in ("accuracy", "nll", "brier")}
            for ref_i, reference in enumerate(REFERENCES):
                r = {"accuracy": a["accuracy"] - .01,
                     "nll": a["nll"] + .01 * (ref_i + 1),
                     "brier": a["brier"] + .005 * (ref_i + 1)}
                old_metrics[reference][task] = r
                new_metrics[reference][task] = r.copy()
                independent["reference_metrics"][reference][task] = r.copy()
        old[seed] = {"seed": seed, "causal_attribution": "NOT_SUPPORTED",
                     "old_fit_rows": {task: 11574 + seed for task in TASKS},
                     "metrics": old_metrics}
        new[seed] = {"seed": seed, "selected_recipe": "A",
                     "metrics": new_metrics}
        replay["by_seed"][str(seed)] = independent
    return old, new, replay


def test_build_rows_all_seeds_tasks_references():
    old, new, replay = fixture_receipts()
    rows = build_rows(old, new, replay)
    assert len(rows) == 8
    assert all(set(row) == set(FIELDS) for row in rows)
    assert [(row["seed"], row["task"], row["reference"]) for row in rows] == [
        (seed, task, reference) for seed in (0, 1) for task in TASKS
        for reference in REFERENCES]
    assert rows[0]["old_minus_new_nll"] == pytest.approx(.02)
    assert rows[0]["new_nll_excess_vs_reference"] == pytest.approx(-.03)
    assert rows[0]["old_fit_rows"] == 11574
    assert rows[-1]["new_fit_rows"] == 15434


def test_build_rows_requires_independent_numeric_agreement():
    old, new, replay = fixture_receipts()
    changed = copy.deepcopy(replay)
    changed["by_seed"]["1"]["metrics"]["new_all_FIT_F"]["income"]["nll"] += .001
    with pytest.raises(PermissionError, match="endpoint differs"):
        build_rows(old, new, changed)


def test_publisher_has_no_person_array_loader():
    source = (Path(__file__).with_name("publish_old_f_diagnostic.py")).read_text()
    assert "np.load" not in source
    assert "numpy" not in source
    assert "OLD_F_INNER_PREDICTIONS.npz" in source  # queue identity only
    assert "sha(old_predictions" not in source
