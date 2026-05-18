"""Synthetic-fixture tests for the LAFTR hard-R² aggregator.

Each test sets up a temp repo root with the on-disk artifacts the
aggregator expects, runs the pure functions (no CLI), and verifies
the arithmetic against hand-computed values.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

# Import as a module so we can call the pure functions directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))
from scripts import aggregate_laftr_hard_r2 as agg  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Fixture builders
# ──────────────────────────────────────────────────────────────────────────────


def _make_per_seed_results(cells_by_seed: list[list[dict]], task_accs_by_seed: list[dict]) -> dict:
    """Build a per_seed_results.json structure matching what
    experiments/run_laftr_hard_r2.py writes.
    """
    per_seed = []
    for seed_idx, (cells, task_accs) in enumerate(zip(cells_by_seed, task_accs_by_seed)):
        per_seed.append({
            "seed": seed_idx,
            "total_pairs": len(cells),
            "attribute_results": cells,
            "task_accuracies": task_accs,
            "per_purpose_health": {},
        })
    return {"per_seed": per_seed, "summary": {}}


def _write_laftr_hard_r2_results(root: Path, dataset: str, per_seed_blob: dict) -> None:
    d = root / "results" / f"laftr_hard_r2_{dataset}_LAFTR_HARD_R2"
    d.mkdir(parents=True)
    (d / "per_seed_results.json").write_text(json.dumps(per_seed_blob))


def _minimal_frozen_baselines() -> dict:
    """Bare-bones frozen baselines: only the keys the aggregator reads."""
    return {
        "_source": "test fixture",
        "_strict_pass_threshold": 0.05,
        "methods": {
            "PCRL_paper": {
                "label": "PCRL (paper)",
                "short_label": "PCRL",
                "per_dataset": {
                    "adult": {
                        "strict_pass_rate": 0.875, "mean_r2_on_passing": 0.014,
                        "task_acc": 0.926, "n_cells_per_seed": 8, "n_seeds": 3, "n_passing": 21,
                    },
                    "hmda": {
                        "strict_pass_rate": 0.888889, "mean_r2_on_passing": 0.005,
                        "task_acc": 0.677, "n_cells_per_seed": 6, "n_seeds": 3, "n_passing": 16,
                    },
                    "diabetes": {
                        "strict_pass_rate": 0.944444, "mean_r2_on_passing": 0.005,
                        "task_acc": 0.732, "n_cells_per_seed": 6, "n_seeds": 3, "n_passing": 17,
                    },
                },
            },
            "LAFTR_appendixQ": {
                "label": "LAFTR (Q)", "short_label": "LAFTR-Q",
                "per_dataset": {
                    "adult": {
                        "strict_pass_rate": 0.0, "mean_r2_on_passing": None,
                        "task_acc": 0.95, "n_cells_per_seed": 8, "n_seeds": 3, "n_passing": 0,
                    },
                    "hmda": None,
                    "diabetes": None,
                },
            },
        },
    }


def _write_frozen_baselines(root: Path, blob: dict) -> None:
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts" / "paper_baseline_numbers.json").write_text(json.dumps(blob))


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests for pure functions
# ──────────────────────────────────────────────────────────────────────────────


def test_compute_laftr_hard_r2_metrics_with_mixed_pass():
    """3 cells × 3 seeds = 9 cells. Cells 0,1 pass (R²=0.01, 0.02); cell 2 fails (R²=0.10).
    Strict pass rate = 6/9. Mean R² on passing = (0.01+0.02)*3 / 6 = 0.015."""
    cells_seed = [
        {"purpose": "p", "attribute": "a", "linear_r2": 0.01},
        {"purpose": "p", "attribute": "b", "linear_r2": 0.02},
        {"purpose": "p", "attribute": "c", "linear_r2": 0.10},
    ]
    blob = _make_per_seed_results(
        cells_by_seed=[cells_seed, cells_seed, cells_seed],
        task_accs_by_seed=[{"task": 0.80}, {"task": 0.80}, {"task": 0.80}],
    )
    m = agg.compute_laftr_hard_r2_metrics(blob)
    assert m["n_cells_per_seed"] == 3
    assert m["n_seeds"] == 3
    assert m["n_passing"] == 6
    assert abs(m["strict_pass_rate"] - 6 / 9) < 1e-9
    assert abs(m["mean_r2_on_passing"] - (0.01 * 3 + 0.02 * 3) / 6) < 1e-9
    assert abs(m["task_acc"] - 0.80) < 1e-9


def test_compute_laftr_hard_r2_metrics_with_no_passing():
    """All cells fail. Mean R² on passing must be None (not NaN, not 0)."""
    cells_seed = [
        {"purpose": "p", "attribute": "a", "linear_r2": 0.30},
        {"purpose": "p", "attribute": "b", "linear_r2": 0.25},
    ]
    blob = _make_per_seed_results(
        cells_by_seed=[cells_seed],
        task_accs_by_seed=[{"task": 0.7}],
    )
    m = agg.compute_laftr_hard_r2_metrics(blob)
    assert m["n_passing"] == 0
    assert m["strict_pass_rate"] == 0.0
    assert m["mean_r2_on_passing"] is None
    assert abs(m["task_acc"] - 0.7) < 1e-9


def test_compute_laftr_hard_r2_metrics_with_all_passing():
    """All cells pass. Mean R² on passing equals mean over all cells."""
    cells_seed = [
        {"purpose": "p", "attribute": "a", "linear_r2": 0.01},
        {"purpose": "p", "attribute": "b", "linear_r2": 0.03},
    ]
    blob = _make_per_seed_results(
        cells_by_seed=[cells_seed, cells_seed],
        task_accs_by_seed=[{"task": 0.9}, {"task": 0.9}],
    )
    m = agg.compute_laftr_hard_r2_metrics(blob)
    assert m["n_passing"] == 4
    assert m["strict_pass_rate"] == 1.0
    assert abs(m["mean_r2_on_passing"] - 0.02) < 1e-9


def test_aggregate_across_datasets_weighted_correctly():
    """Adult 24 cells (21 pass at R²=0.014); HMDA 18 (16 pass at R²=0.005);
    Diabetes 18 (17 pass at R²=0.005). Aggregated pass rate must be
    (21+16+17)/(24+18+18) = 54/60; depth-of-compliance is the weighted mean
    over passing cells."""
    blocks = {
        "adult": {
            "strict_pass_rate": 0.875, "mean_r2_on_passing": 0.014,
            "task_acc": 0.926, "n_cells_per_seed": 8, "n_seeds": 3, "n_passing": 21,
        },
        "hmda": {
            "strict_pass_rate": 16 / 18, "mean_r2_on_passing": 0.005,
            "task_acc": 0.677, "n_cells_per_seed": 6, "n_seeds": 3, "n_passing": 16,
        },
        "diabetes": {
            "strict_pass_rate": 17 / 18, "mean_r2_on_passing": 0.005,
            "task_acc": 0.732, "n_cells_per_seed": 6, "n_seeds": 3, "n_passing": 17,
        },
    }
    aggr = agg.aggregate_across_datasets(blocks)
    assert aggr["n_cells_total"] == 60
    assert aggr["n_passing"] == 54
    assert abs(aggr["strict_pass_rate"] - 54 / 60) < 1e-9
    expected_depth = (0.014 * 21 + 0.005 * 16 + 0.005 * 17) / (21 + 16 + 17)
    assert abs(aggr["mean_r2_on_passing"] - expected_depth) < 1e-9
    # Task acc weighted by total cells per dataset
    expected_acc = (0.926 * 24 + 0.677 * 18 + 0.732 * 18) / 60
    assert abs(aggr["task_acc"] - expected_acc) < 1e-9


def test_aggregate_with_missing_dataset_skipped():
    """LAFTR-Q has only Adult; HMDA/Diabetes are None. Aggregated should only
    include Adult and report n_cells_total=24."""
    blocks = {
        "adult": {
            "strict_pass_rate": 0.0, "mean_r2_on_passing": None,
            "task_acc": 0.95, "n_cells_per_seed": 8, "n_seeds": 3, "n_passing": 0,
        },
        "hmda": None,
        "diabetes": None,
    }
    aggr = agg.aggregate_across_datasets(blocks)
    assert aggr["n_cells_total"] == 24
    assert aggr["n_passing"] == 0
    assert aggr["strict_pass_rate"] == 0.0
    assert aggr["mean_r2_on_passing"] is None
    assert abs(aggr["task_acc"] - 0.95) < 1e-9


def test_aggregate_with_all_missing_returns_nan():
    blocks = {"adult": None, "hmda": None, "diabetes": None}
    aggr = agg.aggregate_across_datasets(blocks)
    assert aggr["n_cells_total"] == 0
    assert aggr["n_passing"] == 0
    assert math.isnan(aggr["strict_pass_rate"])


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end: build_rows + render_tex + render_headline against a synthetic root
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_root(tmp_path):
    """A temp repo with frozen baselines + synthetic LAFTR-hard-R² for Adult only."""
    _write_frozen_baselines(tmp_path, _minimal_frozen_baselines())
    # Synthetic LAFTR-hard-R² Adult: 8 cells × 3 seeds = 24, 18 pass.
    pass_cells = [{"purpose": "p", "attribute": f"a{i}", "linear_r2": 0.01} for i in range(6)]
    fail_cells = [{"purpose": "p", "attribute": f"a{i}", "linear_r2": 0.10} for i in range(6, 8)]
    cells_per_seed = pass_cells + fail_cells
    blob = _make_per_seed_results(
        cells_by_seed=[cells_per_seed, cells_per_seed, cells_per_seed],
        task_accs_by_seed=[{"income": 0.85, "occ": 0.84, "edu": 0.94}] * 3,
    )
    _write_laftr_hard_r2_results(tmp_path, "adult", blob)
    yield tmp_path


def test_build_rows_with_synthetic_root(synthetic_root):
    rows, payload = agg.build_rows(synthetic_root)
    # 3 rows: PCRL, LAFTR-Q, LAFTR-hard-R²
    assert len(rows) == 3
    pcrl_label, pcrl_short, _, _ = rows[0]
    assert pcrl_short == "PCRL"
    laftrq_label, laftrq_short, _, _ = rows[1]
    assert laftrq_short == "LAFTR-Q"
    hr2_label, hr2_short, hr2_per, hr2_aggr = rows[2]
    assert hr2_short == "LAFTR-hard-R²"
    # Adult populated, HMDA/Diabetes None (no on-disk results)
    assert hr2_per["adult"] is not None
    assert hr2_per["hmda"] is None
    assert hr2_per["diabetes"] is None
    # Adult: 18 passing / 24 = 0.75
    assert abs(hr2_per["adult"]["strict_pass_rate"] - 18 / 24) < 1e-9
    assert abs(hr2_per["adult"]["mean_r2_on_passing"] - 0.01) < 1e-9
    # Aggregated: only Adult counts (24 cells, 18 passing)
    assert hr2_aggr["n_cells_total"] == 24
    assert hr2_aggr["n_passing"] == 18
    # Payload mirrors rows
    assert len(payload["rows"]) == 3
    assert payload["rows"][2]["short_label"] == "LAFTR-hard-R²"


def test_render_tex_does_not_crash_with_partial_data(synthetic_root):
    rows, _ = agg.build_rows(synthetic_root)
    tex = agg.render_tex(rows)
    # Sanity: contains the three method labels and the 3 metric rows per method.
    # Match the exact body-row labels (the caption mentions some of these too).
    assert "PCRL (paper)" in tex
    assert "LAFTR (Q)" in tex
    assert "LAFTR-hard-R²" in tex
    assert tex.count("Strict pass ($R^2<0.05$)") == 3
    assert tex.count(" & Mean $R^2$ on passing & ") == 3
    assert tex.count(" & Task acc & ") == 3
    # The HMDA/Diabetes LAFTR-Q cells must render as "—" (em-dash)
    assert "—" in tex


def test_render_headline_does_not_crash_with_partial_data(synthetic_root):
    rows, _ = agg.build_rows(synthetic_root)
    h = agg.render_headline(rows)
    assert "PCRL" in h
    assert "LAFTR-Q" in h
    assert "LAFTR-hard-R²" in h
    # LAFTR-Q HMDA/Diabetes must render "—"
    assert "—" in h


def test_full_main_writes_three_files(synthetic_root, monkeypatch):
    """End-to-end: invoke main() against the synthetic root and verify the three artifacts."""
    out = synthetic_root / "results" / "laftr_hard_r2"
    monkeypatch.setattr(
        "sys.argv",
        ["aggregate_laftr_hard_r2.py", "--root", str(synthetic_root), "--out-dir", str(out)],
    )
    rc = agg.main()
    assert rc == 0
    assert (out / "comparison_table.tex").is_file()
    assert (out / "comparison.json").is_file()
    assert (out / "HEADLINE.txt").is_file()
    payload = json.loads((out / "comparison.json").read_text())
    assert len(payload["rows"]) == 3
    assert payload["_strict_pass_threshold"] == 0.05
