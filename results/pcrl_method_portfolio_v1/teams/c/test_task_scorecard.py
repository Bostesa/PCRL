"""Synthetic custody check for the task-label-only C control scorecard."""

import inspect

import numpy as np

import task_scorecard
from prototype import AdultGroupedSchema


def test_task_only_scorecard_repairs_only_task_classes(monkeypatch):
    assert "sex" not in inspect.signature(task_scorecard.build_task_scorecard).parameters
    schema = AdultGroupedSchema()
    n = 120
    rng = np.random.default_rng(91)
    X = rng.normal(size=(n, 83)).astype(np.float32)
    income = np.arange(n, dtype=np.int64) % 2
    occupation = np.arange(n, dtype=np.int64) % 6
    unit = np.arange(n, dtype=np.int64)
    captured = []
    original = task_scorecard.preserve_fit_class_support

    def capture(unit_arg, fit, selection, labels, vocabularies):
        captured.append((tuple(sorted(labels)), tuple(sorted(vocabularies))))
        return original(unit_arg, fit, selection, labels, vocabularies)

    monkeypatch.setattr(task_scorecard, "preserve_fit_class_support", capture)
    result = task_scorecard.build_task_scorecard(
        X, schema.feature_names, income, occupation, unit, seed=19,
    )
    assert captured == [(("income", "occupation_group"),
                         ("income", "occupation_group"))]
    assert result.fit_counts == {
        "task_proxy": 20, "sensitive_proxy": 0, "total": 20,
    }
    assert len(result.mask_metrics) == 10
    assert result.split_receipt["sensitive_label_input"] is False
    assert result.split_receipt["overlap_groups"] == 0
    assert set(result.task_deltas) == {"income", "occupation_group"}
    assert set(result.task_deltas["income"]) == set(schema.atoms)
