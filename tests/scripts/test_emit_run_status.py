"""Synthetic-fixture tests for the run-status emitter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))
from scripts import emit_run_status as ers  # noqa: E402


def _write_dataset(root: Path, ds: str, cells_by_seed, task_acc_mean, status="HEALTHY"):
    d = root / "results" / f"laftr_hard_r2_{ds}_LAFTR_HARD_R2"
    d.mkdir(parents=True)
    per_seed = [
        {"seed": i, "total_pairs": len(cells), "attribute_results": cells,
         "task_accuracies": {}, "per_purpose_health": {}}
        for i, cells in enumerate(cells_by_seed)
    ]
    (d / "per_seed_results.json").write_text(json.dumps({"per_seed": per_seed, "summary": {}}))
    (d / "summary.json").write_text(json.dumps({
        "STATUS": status, "dataset": ds, "task_acc_mean": task_acc_mean,
    }))


def test_dataset_line_reports_strict_pass_and_means(tmp_path):
    cells = [
        {"purpose": "p", "attribute": "a", "linear_r2": 0.01},
        {"purpose": "p", "attribute": "b", "linear_r2": 0.10},
    ]
    _write_dataset(tmp_path, "adult", [cells, cells, cells], {"income": 0.85})
    line = ers._dataset_line(tmp_path, "adult")
    # 1 pass / 2 per seed × 3 seeds = 3/6
    assert "strict-pass 3/6" in line
    # mean over all cells = (0.01+0.10)/2 = 0.055
    assert "meanR²(all)=0.0550" in line
    # mean over passing = 0.01
    assert "meanR²(pass)=0.0100" in line
    assert "health=HEALTHY" in line
    assert "income=85.0%" in line


def test_dataset_line_missing_is_graceful(tmp_path):
    line = ers._dataset_line(tmp_path, "hmda")
    assert "MISSING" in line
    assert "hmda" in line


def test_dataset_line_no_passing_cells_shows_dash(tmp_path):
    cells = [{"purpose": "p", "attribute": "a", "linear_r2": 0.30}]
    _write_dataset(tmp_path, "diabetes", [cells], {"readmit": 0.31}, status="COLLAPSED")
    line = ers._dataset_line(tmp_path, "diabetes")
    assert "strict-pass 0/1" in line
    assert "meanR²(pass)=—" in line
    assert "health=COLLAPSED" in line


def test_build_status_includes_instance_and_all_datasets(tmp_path):
    cells = [{"purpose": "p", "attribute": "a", "linear_r2": 0.01}]
    _write_dataset(tmp_path, "adult", [cells], {"income": 0.9})
    # hmda + diabetes intentionally absent → reported MISSING
    text = ers.build_status(tmp_path, "i-0deadbeef")
    assert "i-0deadbeef" in text
    assert "completed_utc:" in text
    assert "adult" in text
    assert "hmda" in text and "MISSING" in text
    assert "diabetes" in text


def test_main_writes_file(tmp_path, monkeypatch):
    cells = [{"purpose": "p", "attribute": "a", "linear_r2": 0.01}]
    _write_dataset(tmp_path, "adult", [cells], {"income": 0.9})
    out = tmp_path / "STATUS.txt"
    monkeypatch.setattr("sys.argv", [
        "emit_run_status.py", "--instance-id", "i-0abc",
        "--root", str(tmp_path), "--out", str(out),
    ])
    rc = ers.main()
    assert rc == 0
    assert out.is_file()
    assert "i-0abc" in out.read_text()
