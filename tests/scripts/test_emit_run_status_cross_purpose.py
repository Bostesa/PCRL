"""Synthetic-fixture tests for the cross-purpose run-status emitter."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts import emit_run_status_cross_purpose as ers  # noqa: E402


def _write_training(root: Path, ds: str, tag: str, attr_results_by_seed,
                    task_acc_mean=None, status="HEALTHY"):
    d = root / "results" / f"v2_{ds}_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    per_seed = [
        {"seed": i, "attribute_results": cells,
         "task_accuracies": {}, "per_purpose_health": {}}
        for i, cells in enumerate(attr_results_by_seed)
    ]
    summary = {"STATUS": status, "task_acc_mean": task_acc_mean or {}}
    (d / "per_seed_results.json").write_text(
        json.dumps({"per_seed": per_seed, "summary": summary}))


def _write_eval(root: Path, ds: str, tag: str, concat_r2_mean, attack_matrix,
                attack_overall="0/0"):
    d = root / "results" / f"v2_{ds}_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps({
        "summary": {
            "concat_r2_mean": concat_r2_mean,
            "attack_per_arch_per_attr": attack_matrix,
            "attack_flagged_above_1pp": attack_overall,
        },
        "per_seed": [],
    }))


def test_per_pair_pass_count(tmp_path):
    cells = [
        {"purpose": "p", "attribute": "a", "linear_r2": 0.01},  # pass
        {"purpose": "p", "attribute": "b", "linear_r2": 0.10},  # fail
    ]
    _write_training(tmp_path, "adult", "CROSS_PURPOSE_AB",
                    [cells, cells, cells], {"income": 0.85})
    line = ers._dataset_line(tmp_path, "adult", "CROSS_PURPOSE_AB")
    assert "per-pair 3/6" in line
    assert "health=HEALTHY" in line
    assert "income=85.0%" in line


def test_eval_pending_when_results_missing(tmp_path):
    cells = [{"purpose": "p", "attribute": "a", "linear_r2": 0.01}]
    _write_training(tmp_path, "hmda", "CROSS_PURPOSE_AB", [cells])
    line = ers._dataset_line(tmp_path, "hmda", "CROSS_PURPOSE_AB")
    assert "per-pair 1/1" in line
    assert "h_concat (eval pending)" in line
    assert "attack (eval pending)" in line


def test_concat_and_attack_lines_when_eval_present(tmp_path):
    cells = [{"purpose": "p", "attribute": "a", "linear_r2": 0.01}]
    _write_training(tmp_path, "diabetes", "CROSS_PURPOSE_DIABETES", [cells])
    _write_eval(
        tmp_path, "diabetes", "CROSS_PURPOSE_DIABETES",
        concat_r2_mean={
            "race": {"mean_r2": 0.03, "max_r2": 0.05, "n_pass": 3, "n": 3},
            "gender": {"mean_r2": 0.12, "max_r2": 0.15, "n_pass": 0, "n": 3},
        },
        attack_matrix={
            "LR":  {"race": {"mean_delta_pp": 0.5, "flag_1pp_mean": False}},
            "MLP": {"race": {"mean_delta_pp": 2.5, "flag_1pp_mean": True}},
            "XGB": {"race": {"mean_delta_pp": 3.0, "flag_1pp_mean": True}},
        },
        attack_overall="2/3",
    )
    line = ers._dataset_line(tmp_path, "diabetes", "CROSS_PURPOSE_DIABETES")
    assert "h_concat 3/6" in line
    assert "race(3/3)" in line and "gender(0/3)" in line
    assert "attack 2/3" in line
    assert "LR(0/1)" in line and "MLP(1/1)" in line and "XGB(1/1)" in line


def test_missing_dataset_reports_gracefully(tmp_path):
    line = ers._dataset_line(tmp_path, "adult", "CROSS_PURPOSE_AB")
    assert "MISSING" in line and "adult" in line


def test_build_status_lists_only_requested_datasets(tmp_path):
    cells = [{"purpose": "p", "attribute": "a", "linear_r2": 0.01}]
    _write_training(tmp_path, "adult", "CROSS_PURPOSE_AB", [cells])
    _write_training(tmp_path, "hmda", "CROSS_PURPOSE_AB", [cells])
    text = ers.build_status(tmp_path, "i-0dead", "CROSS_PURPOSE_AB",
                            ["adult", "hmda"])
    assert "i-0dead" in text
    assert "CROSS_PURPOSE_AB" in text
    assert "adult" in text and "hmda" in text
    assert "diabetes" not in text  # not requested


def test_main_writes_file(tmp_path, monkeypatch):
    cells = [{"purpose": "p", "attribute": "a", "linear_r2": 0.01}]
    _write_training(tmp_path, "adult", "CROSS_PURPOSE_AB", [cells])
    out = tmp_path / "STATUS.txt"
    monkeypatch.setattr("sys.argv", [
        "emit_run_status_cross_purpose.py", "--instance-id", "i-0abc",
        "--tag", "CROSS_PURPOSE_AB", "--datasets", "adult",
        "--root", str(tmp_path), "--out", str(out),
    ])
    rc = ers.main()
    assert rc == 0
    assert out.is_file()
    body = out.read_text()
    assert "i-0abc" in body
    assert "CROSS_PURPOSE_AB" in body
