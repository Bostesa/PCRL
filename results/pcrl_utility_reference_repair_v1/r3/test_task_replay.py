"""Data-free checks of independent recipe selection and F restore."""

import json

import numpy as np
import torch

from results.pcrl_utility_reference_repair_v1.r3.source_reference import file_sha
from results.pcrl_utility_reference_repair_v1.r3.task_replay import (
    gate_metrics, restore_f_predictions, select_recipe,
)


def test_raw_screen_selection_and_inclusive_gate():
    row = np.arange(12, dtype=np.int64)
    unit = row + 100
    income = row % 2
    occupation = row % 6
    good2 = np.eye(2)[income] * 0.8 + 0.2 / 2
    good6 = np.eye(6)[occupation] * 0.8 + 0.2 / 6
    bad2 = np.full((12, 2), 0.5)
    bad6 = np.full((12, 6), 1 / 6)
    screen = {"row_id": row, "unit": unit, "income": income,
              "occupation": occupation,
              "A_income": good2, "A_occupation_group": good6,
              "B_income": bad2, "B_occupation_group": bad6}
    selected = select_recipe({0: screen, 1: screen})
    assert selected["selected_recipe"] == "A"
    assert selected["total_raw_nll"]["A"] < selected["total_raw_nll"]["B"]
    gate = {"row_id": row, "unit": unit, "income": income,
            "occupation": occupation,
            "F_income": good2, "F_occupation_group": good6,
            "U0_1": good2, "U0_2": good6,
            "Ucal_1": good2, "Ucal_2": good6}
    assert gate_metrics(gate)["point_feasible"]
    gate["F_income"] = bad2
    assert not gate_metrics(gate)["point_feasible"]


def test_independent_a_weight_and_calibrator_restore(tmp_path):
    torch.manual_seed(19)
    x = np.zeros((7, 83), dtype=np.float32)
    root = tmp_path / "private"
    root.mkdir()
    packages = {}
    for task, width, cal_task in (("income", 2, "income"),
                                  ("occupation_group", 6, "occupation")):
        model = torch.nn.Sequential(
            torch.nn.Linear(83, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, width),
        ).eval()
        state = model.state_dict()
        path = root / f"A_{task}_head.npz"
        meta = {"schema": "team-c-head-checkpoint-v1", "purpose": task,
                "selected_columns": 83,
                "receipt": {"selected_columns": 83},
                "spec": {"hidden_width": 64, "max_epochs": 30},
                "state_keys": list(state)}
        arrays = {"meta": np.array(json.dumps(meta))}
        for index, weight in enumerate(state.values()):
            arrays[f"state_{index}"] = weight.numpy()
        np.savez_compressed(path, **arrays)
        packages[task] = {"relative_path": path.name, "sha256": file_sha(path)}
        cal_path = root / f"{task}_calibrator.json"
        cal_path.write_text(json.dumps({
            "schema": "pcrl-portfolio-global-temp-v1", "task": cal_task,
            "model_seed": 0, "fit_role": "CALIBRATION_HELDOUT", "alpha": 1.0,
        }))
        packages[f"calibrator_{task}"] = {
            "relative_path": cal_path.name, "sha256": file_sha(cal_path)}
        with torch.no_grad():
            expected = torch.softmax(model(torch.from_numpy(x)).double(), dim=1).numpy()
        packages[f"expected_{task}"] = expected
    expected = (packages.pop("expected_income"),
                packages.pop("expected_occupation_group"))
    receipt = {"selected_recipe": "A", "seed": 0, "packages": packages}
    actual = restore_f_predictions(receipt, root, x)
    assert all(np.array_equal(left, right)
               for left, right in zip(actual, expected))
