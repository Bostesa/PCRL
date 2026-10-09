"""Data-free checks of the independent task-only source retrain."""

import numpy as np
import torch

from source_task import (choose_source_C, fit_source_heads, predict_source, restore_source,
                         save_source, select_source_head_C,
                         train_source_encoder)


def _fixture():
    rng = np.random.default_rng(841)
    x = rng.normal(size=(72, 83)).astype(np.float32)
    yi = (np.arange(72) % 2).astype(np.int64)
    yo = (np.arange(72) % 6).astype(np.int64)
    return x, yi, yo


def test_task_only_kernel_ignores_unused_dummy_sensitive_field():
    x, yi, yo = _fixture()
    left = train_source_encoder(x, yi, yo, 0, dummy_bit=0)
    right = train_source_encoder(x, yi, yo, 0, dummy_bit=1)
    assert all(torch.equal(left.state_dict()[name], right.state_dict()[name])
               for name in left.state_dict())


def test_grid_prediction_and_hashed_restore(tmp_path):
    x, yi, yo = _fixture()
    model = train_source_encoder(x[:60], yi[:60], yo[:60], 1)
    choice = select_source_head_C(model, x[:60], yi[:60], yo[:60],
                                  x[60:], yi[60:], yo[60:])
    heads = fit_source_heads(model, x[:60], yi[:60], yo[:60],
                            {task: row["selected_C"] for task, row in choice.items()})
    before = predict_source(model, heads, x[60:])
    shas = save_source(model, heads, tmp_path / "bundle")
    restored, restored_heads = restore_source(tmp_path / "bundle", 1, shas)
    after = predict_source(restored, restored_heads, x[60:])
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


def test_source_grid_near_tie_keeps_first_lower_C():
    table = [{"C": c, "nll": 0.4 - (5e-13 if i == 1 else 0.0)}
             for i, c in enumerate((0.01, 0.1, 1.0, 10.0, 100.0))]
    assert choose_source_C(table)["C"] == 0.01
