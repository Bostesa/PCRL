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
    over passing cells. With no mask, task-acc denominator is the full 60."""
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
    # Task acc weighted by total cells per dataset (no mask → all 60)
    expected_acc = (0.926 * 24 + 0.677 * 18 + 0.732 * 18) / 60
    assert abs(aggr["task_acc"] - expected_acc) < 1e-9
    assert aggr["task_acc_n_cells"] == 60


def test_aggregate_with_missing_dataset_skipped():
    """LAFTR-Q has only Adult; HMDA/Diabetes are None. Aggregated should only
    include Adult and report n_cells_total=24 / task_acc_n_cells=24."""
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
    assert aggr["task_acc_n_cells"] == 24


def test_aggregate_with_all_missing_returns_nan():
    blocks = {"adult": None, "hmda": None, "diabetes": None}
    aggr = agg.aggregate_across_datasets(blocks)
    assert aggr["n_cells_total"] == 0
    assert aggr["n_passing"] == 0
    assert math.isnan(aggr["strict_pass_rate"])
    assert aggr["task_acc_n_cells"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# Apples-to-apples task-acc mask
# ──────────────────────────────────────────────────────────────────────────────


def _block(strict, r2_pass, acc, n_per_seed, n_seeds, n_passing):
    return {
        "strict_pass_rate": strict, "mean_r2_on_passing": r2_pass,
        "task_acc": acc, "n_cells_per_seed": n_per_seed, "n_seeds": n_seeds,
        "n_passing": n_passing,
    }


def test_common_task_acc_datasets_all_populated():
    pcrl = {ds: _block(0.9, 0.01, 0.85, 8, 3, 22) for ds in ["adult", "hmda", "diabetes"]}
    laftrq = {ds: _block(0.0, None, 0.90, 8, 3, 0) for ds in ["adult", "hmda", "diabetes"]}
    hr2 = {ds: _block(0.8, 0.02, 0.88, 8, 3, 19) for ds in ["adult", "hmda", "diabetes"]}
    common = agg._common_task_acc_datasets([("PCRL", pcrl), ("LAFTR-Q", laftrq), ("LAFTR-hard-R²", hr2)])
    assert common == {"adult", "hmda", "diabetes"}


def test_common_task_acc_datasets_one_null_excludes_dataset():
    """LAFTR-Q HMDA task_acc=null → HMDA dropped from common across ALL methods."""
    pcrl = {ds: _block(0.9, 0.01, 0.85, 8, 3, 22) for ds in ["adult", "hmda", "diabetes"]}
    laftrq = {
        "adult": _block(0.0, None, 0.95, 8, 3, 0),
        "hmda": _block(0.0, None, None, 6, 3, 0),    # null task_acc
        "diabetes": _block(0.83, None, 0.31, 6, 3, 15),
    }
    hr2 = {ds: _block(0.8, 0.02, 0.88, 8, 3, 19) for ds in ["adult", "hmda", "diabetes"]}
    common = agg._common_task_acc_datasets([("PCRL", pcrl), ("LAFTR-Q", laftrq), ("LAFTR-hard-R²", hr2)])
    assert common == {"adult", "diabetes"}


def test_common_task_acc_datasets_method_entirely_none_excludes_all():
    """LAFTR-hard-R² entirely None (pre-AWS) → common is empty set."""
    pcrl = {ds: _block(0.9, 0.01, 0.85, 8, 3, 22) for ds in ["adult", "hmda", "diabetes"]}
    laftrq = {ds: _block(0.0, None, 0.90, 8, 3, 0) for ds in ["adult", "hmda", "diabetes"]}
    hr2 = {"adult": None, "hmda": None, "diabetes": None}
    common = agg._common_task_acc_datasets([("PCRL", pcrl), ("LAFTR-Q", laftrq), ("LAFTR-hard-R²", hr2)])
    assert common == set()


def test_aggregate_with_mask_excludes_correct_datasets():
    """With mask={adult, diabetes}, HMDA should be excluded from task_acc only.
    Strict-pass and mean-R²-on-passing are unaffected by the mask."""
    blocks = {
        "adult": _block(0.875, 0.014, 0.926, 8, 3, 21),
        "hmda": _block(16 / 18, 0.005, 0.677, 6, 3, 16),
        "diabetes": _block(17 / 18, 0.005, 0.732, 6, 3, 17),
    }
    mask = {"adult", "diabetes"}
    aggr = agg.aggregate_across_datasets(blocks, task_acc_dataset_mask=mask)
    # strict-pass still over all 60
    assert aggr["n_cells_total"] == 60
    assert aggr["n_passing"] == 54
    assert abs(aggr["strict_pass_rate"] - 54 / 60) < 1e-9
    # mean R² on passing still over all passing cells
    expected_depth = (0.014 * 21 + 0.005 * 16 + 0.005 * 17) / 54
    assert abs(aggr["mean_r2_on_passing"] - expected_depth) < 1e-9
    # task_acc only over Adult + Diabetes = 24 + 18 = 42 cells
    expected_acc = (0.926 * 24 + 0.732 * 18) / 42
    assert abs(aggr["task_acc"] - expected_acc) < 1e-9
    assert aggr["task_acc_n_cells"] == 42


def test_aggregate_with_empty_mask_yields_nan_task_acc():
    """Empty mask → no cells contribute to task_acc → NaN with task_acc_n_cells=0."""
    blocks = {
        "adult": _block(0.9, 0.01, 0.85, 8, 3, 22),
        "hmda": _block(0.9, 0.01, 0.80, 6, 3, 16),
        "diabetes": _block(0.9, 0.01, 0.75, 6, 3, 17),
    }
    aggr = agg.aggregate_across_datasets(blocks, task_acc_dataset_mask=set())
    # strict-pass etc. unaffected
    assert aggr["n_cells_total"] == 60
    # task_acc collapses
    assert math.isnan(aggr["task_acc"])
    assert aggr["task_acc_n_cells"] == 0


def test_build_rows_applies_mask_to_all_three_methods(tmp_path):
    """Construct a frozen baseline where LAFTR-Q HMDA task_acc is null and
    LAFTR-hard-R² has data for all three datasets. Mask should be
    {adult, diabetes}. PCRL aggregated task_acc must be computed over 42
    cells (Adult+Diabetes), NOT over 60 — that's the silent-favoritism fix.
    """
    baselines = {
        "_source": "test-mask",
        "_strict_pass_threshold": 0.05,
        "methods": {
            "PCRL_paper": {
                "label": "PCRL (paper)", "short_label": "PCRL",
                "per_dataset": {
                    "adult": _block(0.9, 0.01, 0.926, 8, 3, 22),
                    "hmda": _block(0.9, 0.01, 0.677, 6, 3, 16),     # has acc
                    "diabetes": _block(0.9, 0.01, 0.732, 6, 3, 17),
                },
            },
            "LAFTR_appendixQ": {
                "label": "LAFTR (Q)", "short_label": "LAFTR-Q",
                "per_dataset": {
                    "adult": _block(0.0, None, 0.95, 8, 3, 0),
                    "hmda": _block(0.0, None, None, 6, 3, 0),       # null acc
                    "diabetes": _block(0.83, None, 0.31, 6, 3, 15),
                },
            },
        },
    }
    _write_frozen_baselines(tmp_path, baselines)
    # Synthetic LAFTR-hard-R² for ALL three datasets so it doesn't constrain the mask.
    for ds in ["adult", "hmda", "diabetes"]:
        # 8 or 6 cells/seed × 3 seeds; just give 1 pass per seed for shape.
        ncells = 8 if ds == "adult" else 6
        cells = [{"purpose": "p", "attribute": f"a{i}", "linear_r2": 0.01 if i == 0 else 0.10}
                 for i in range(ncells)]
        blob = _make_per_seed_results(
            cells_by_seed=[cells, cells, cells],
            task_accs_by_seed=[{"t": 0.80}] * 3,
        )
        _write_laftr_hard_r2_results(tmp_path, ds, blob)

    rows, payload, common = agg.build_rows(tmp_path)
    # Mask excludes HMDA (LAFTR-Q HMDA task_acc is null)
    assert common == {"adult", "diabetes"}
    # Every method's aggregated task_acc denominator must be 42, not 60
    for _, short, _, aggr in rows:
        assert aggr["task_acc_n_cells"] == 42, (
            f"method {short} task_acc_n_cells = {aggr['task_acc_n_cells']}, "
            f"expected 42 (Adult+Diabetes only after HMDA exclusion)"
        )
    # PCRL aggregated task_acc must match the apples-to-apples computation
    pcrl_aggr = rows[0][3]
    expected_pcrl = (0.926 * 24 + 0.732 * 18) / 42
    assert abs(pcrl_aggr["task_acc"] - expected_pcrl) < 1e-9
    # Strict-pass aggregation is unaffected by the mask
    assert pcrl_aggr["n_cells_total"] == 60
    # Payload exposes the mask + exclusion list
    assert payload["task_acc_common_datasets"] == ["adult", "diabetes"]
    assert payload["task_acc_asymmetric"] is True
    excluded = payload["task_acc_excluded_datasets"]
    assert len(excluded) == 1
    assert excluded[0]["dataset"] == "hmda"
    assert "LAFTR-Q" in excluded[0]["missing_in_methods"]


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
    rows, payload, common_ds = agg.build_rows(synthetic_root)
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
    # In this synthetic root: LAFTR-Q has Adult only, LAFTR-hard-R² has Adult only,
    # so the common task-acc mask is {adult} — every method's aggregated task_acc
    # is computed over the 24 Adult cells.
    assert common_ds == {"adult"}
    for _, _, _, aggr in rows:
        assert aggr["task_acc_n_cells"] == 24
    # Payload mirrors rows + exposes mask
    assert len(payload["rows"]) == 3
    assert payload["rows"][2]["short_label"] == "LAFTR-hard-R²"
    assert payload["task_acc_common_datasets"] == ["adult"]
    assert payload["task_acc_asymmetric"] is True


def test_render_tex_does_not_crash_with_partial_data(synthetic_root):
    rows, _, common_ds = agg.build_rows(synthetic_root)
    tex = agg.render_tex(rows, common_ds=common_ds)
    # Sanity: contains the three method labels and the 3 metric rows per method.
    assert "PCRL (paper)" in tex
    assert "LAFTR (Q)" in tex
    assert "LAFTR-hard-R²" in tex
    assert tex.count("Strict pass ($R^2<0.05$)") == 3
    assert tex.count(" & Mean $R^2$ on passing & ") == 3
    assert tex.count(" & Task acc & ") == 3
    # The HMDA/Diabetes LAFTR-Q cells must render as "—" (em-dash)
    assert "—" in tex


def test_render_tex_includes_footnote_when_asymmetric(synthetic_root):
    """When common_ds is a proper subset of DATASETS, the caption must
    auto-include a \\footnote explaining the exclusion."""
    rows, _, common_ds = agg.build_rows(synthetic_root)
    assert common_ds != {"adult", "hmda", "diabetes"}
    tex = agg.render_tex(rows, common_ds=common_ds)
    assert r"\protect\footnote{" in tex
    # The footnote should name HMDA + Diabetes as excluded (only Adult survives mask)
    assert "HMDA" in tex
    assert "Diabetes" in tex
    # It should mention which method(s) caused each exclusion. In this fixture
    # both LAFTR-Q and LAFTR-hard-R² are missing HMDA/Diabetes.
    assert "LAFTR-Q" in tex or "LAFTR-hard-R²" in tex


def test_render_tex_footnote_fires_when_common_ds_is_empty_set(tmp_path):
    """Regression: ``common_ds=set()`` (one method entirely missing) must
    still fire the footnote — a `common_ds or set(DATASETS)` fallback would
    silently skip it because empty sets are falsy.
    """
    baselines = {
        "_source": "test-empty-mask",
        "_strict_pass_threshold": 0.05,
        "methods": {
            "PCRL_paper": {
                "label": "PCRL (paper)", "short_label": "PCRL",
                "per_dataset": {
                    "adult": _block(0.9, 0.01, 0.926, 8, 3, 22),
                    "hmda": _block(0.9, 0.01, 0.677, 6, 3, 16),
                    "diabetes": _block(0.9, 0.01, 0.732, 6, 3, 17),
                },
            },
            "LAFTR_appendixQ": {
                "label": "LAFTR (Q)", "short_label": "LAFTR-Q",
                "per_dataset": {
                    "adult": _block(0.0, None, 0.95, 8, 3, 0),
                    "hmda": _block(0.0, None, 0.81, 6, 3, 0),
                    "diabetes": _block(0.83, None, 0.31, 6, 3, 15),
                },
            },
        },
    }
    _write_frozen_baselines(tmp_path, baselines)
    # Note: NO LAFTR-hard-R² results on disk — every dataset is "missing" for
    # that method → common mask is empty set, distinct from None.
    rows, _, common_ds = agg.build_rows(tmp_path)
    assert common_ds == set()  # empty, not None
    tex = agg.render_tex(rows, common_ds=common_ds)
    # Footnote MUST fire (all three datasets are excluded)
    assert r"\protect\footnote{" in tex
    # And the footnote names all three datasets
    assert "Adult" in tex and "HMDA" in tex and "Diabetes" in tex


def test_render_tex_no_footnote_when_symmetric(tmp_path):
    """When all three methods report task_acc on every dataset, no footnote."""
    baselines = {
        "_source": "test-symmetric",
        "_strict_pass_threshold": 0.05,
        "methods": {
            "PCRL_paper": {
                "label": "PCRL (paper)", "short_label": "PCRL",
                "per_dataset": {
                    "adult": _block(0.9, 0.01, 0.926, 8, 3, 22),
                    "hmda": _block(0.9, 0.01, 0.677, 6, 3, 16),
                    "diabetes": _block(0.9, 0.01, 0.732, 6, 3, 17),
                },
            },
            "LAFTR_appendixQ": {
                "label": "LAFTR (Q)", "short_label": "LAFTR-Q",
                "per_dataset": {
                    "adult": _block(0.0, None, 0.95, 8, 3, 0),
                    "hmda": _block(0.0, None, 0.81, 6, 3, 0),
                    "diabetes": _block(0.83, None, 0.31, 6, 3, 15),
                },
            },
        },
    }
    _write_frozen_baselines(tmp_path, baselines)
    # All three datasets populated for LAFTR-hard-R²
    for ds in ["adult", "hmda", "diabetes"]:
        n = 8 if ds == "adult" else 6
        cells = [{"purpose": "p", "attribute": f"a{i}", "linear_r2": 0.01 if i == 0 else 0.10}
                 for i in range(n)]
        blob = _make_per_seed_results(
            cells_by_seed=[cells] * 3, task_accs_by_seed=[{"t": 0.80}] * 3,
        )
        _write_laftr_hard_r2_results(tmp_path, ds, blob)

    rows, _, common_ds = agg.build_rows(tmp_path)
    assert common_ds == {"adult", "hmda", "diabetes"}
    tex = agg.render_tex(rows, common_ds=common_ds)
    assert r"\protect\footnote{" not in tex


def test_render_tex_aggregated_task_acc_shows_cell_count(synthetic_root):
    """Aggregated task acc cells must show \"X.X\\% (N cells)\" format."""
    rows, _, common_ds = agg.build_rows(synthetic_root)
    tex = agg.render_tex(rows, common_ds=common_ds)
    # The synthetic root has only Adult contributing — denominator = 24.
    # All three methods report task_acc on Adult, so all three aggregated
    # task-acc cells should read "X.X\\% (24 cells)".
    assert "(24 cells)" in tex


def test_render_headline_does_not_crash_with_partial_data(synthetic_root):
    rows, _, common_ds = agg.build_rows(synthetic_root)
    h = agg.render_headline(rows, common_ds=common_ds)
    assert "PCRL" in h
    assert "LAFTR-Q" in h
    assert "LAFTR-hard-R²" in h
    # LAFTR-Q HMDA/Diabetes must render "—"
    assert "—" in h


def test_render_headline_shows_aggregated_task_acc_with_cells(synthetic_root):
    rows, _, common_ds = agg.build_rows(synthetic_root)
    h = agg.render_headline(rows, common_ds=common_ds)
    # Aggregated task-acc mini-block prints "(24 cells)" per method
    assert "(24 cells)" in h
    # And the exclusion note names HMDA + Diabetes
    assert "HMDA" in h
    assert "Diabetes" in h


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
    assert "task_acc_common_datasets" in payload
    assert "task_acc_asymmetric" in payload
    assert "task_acc_excluded_datasets" in payload
