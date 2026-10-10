import json
from pathlib import Path

import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.r02.a_inference import (
    A_ARMS, A_REFS, STOCHASTIC_ARMS, SUMMARIES, a_margin_at,
    a_margin_names, crossed_resample, run_a_inference, summarize_a_bootstrap,
)


def _release(row, delta=0.0):
    p1 = np.array([[.8, .2], [.2, .8], [.7, .3], [.3, .7]], dtype=float)
    p1[:, 1] += delta
    p1[:, 0] -= delta
    p6 = np.eye(6)[[0, 1, 0, 1]] * .9 + .1 / 6
    scores = np.array([.2, .8, .3, .7]) + delta
    return {"row_id": row, "unit": row, "prob_1": p1, "prob_2": p6,
            "reader_predictions": {v: {"LR_C1": scores} for v in ("r1", "r2", "pair")}}


def _seed():
    row = np.arange(4, dtype=np.int64)
    truth = {"row_id": row, "unit": row, "income": np.array([0, 1, 0, 1]),
             "occupation": np.array([0, 1, 0, 1]), "sex": np.array([0, 1, 0, 1])}
    releases = {arm: {draw: _release(row) for draw in range(3)}
                if arm in STOCHASTIC_ARMS else {0: _release(row)}
                for arm in (*A_ARMS, *A_REFS)}
    return {"truth": truth, "releases": releases}


def test_a_family_fixed_6885_and_includes_historical_d0_score_controls():
    names = a_margin_names()
    assert len(names) == len(set(names)) == 6885
    assert len(SUMMARIES) == 9
    assert sum(":utility:" in x for x in names) == 3780
    assert sum(":arm_pair:" in x for x in names) == 1215
    assert sum(":ref_pair:" in x for x in names) == 1890
    assert "mean:utility:A_JOINT_STOCH:income:P_STAR:nll" in names
    assert "s0_d0:utility:A_JOINT_STOCH:occupation:D0_LOCAL:nll" in names
    assert "mean:ref_pair:A_JOINT_STOCH:D0_SEQ12:pair:auc" in names
    assert "s1_d2:ref_pair:A_JOINT_STOCH:D0_SEQ21:r1:auc" in names
    assert "s1_d2:ref_pair:A_JOINT_STOCH:T_STAR:pair:auc" in names
    manifest = json.loads(Path(__file__).with_name("A_FAMILY_ORDER_S1.json").read_text())
    assert manifest["ordered_clause_ids"] == list(names)
    assert manifest["clause_count"] == 6885
    assert manifest["supersedes_manifest_sha256"] == "d0e8a5b22568ee64844b039cc5468e98a36f9c03c30adb305279e73bdc5b8107"


def test_a_margin_reuses_actual_draws_and_only_resamples_mean_draw_indices():
    records = {0: _seed(), 1: _seed()}
    names, value = a_margin_at(records, np.arange(4))
    assert names == a_margin_names()
    assert value.shape == (6885,)
    assert np.allclose(value, 0)
    records[0]["releases"]["A_JOINT_STOCH"][2] = _release(np.arange(4), .05)
    _, original = a_margin_at(records, np.arange(4))
    _, repeated = a_margin_at(records, np.arange(4), {0: (2, 2, 2), 1: (0, 1, 2)})
    assert np.array_equal(original[:6 * 765], repeated[:6 * 765])
    assert not np.array_equal(original[6 * 765:], repeated[6 * 765:])
    records[1]["releases"]["A_LOCAL"][1]["row_id"] = np.array([1, 0, 2, 3])
    with pytest.raises(ValueError, match="ordered"):
        a_margin_at(records, np.arange(4))


@pytest.mark.parametrize("missing_ref", ["P_STAR", "D0_LOCAL", "D0_SEQ21"])
def test_missing_reference_preserves_full_registry_and_nan_dependent_clauses(missing_ref):
    records = {0: _seed(), 1: _seed()}
    for seed in records:
        del records[seed]["releases"][missing_ref]
    names, value = a_margin_at(records, np.arange(4))
    missing = np.isnan(value)
    assert len(names) == 6885
    assert missing.sum() == 10 * (2 * 3 + 3) * 9
    assert all(missing_ref in names[j] for j in np.flatnonzero(missing))
    assert np.all(np.isfinite(value[~missing]))


def test_group_and_release_draw_sampling_are_paired_and_not_new_people():
    rng_people = np.random.Generator(np.random.PCG64DXSM(7))
    rng_release = np.random.Generator(np.random.PCG64DXSM(8))
    idx, draws = crossed_resample(np.array([3, 3, 4, 5]), rng_people, rng_release)
    assert len(draws) == 2 and all(len(draws[s]) == 3 for s in (0, 1))
    assert len(idx) >= 3
    assert set(idx).issubset({0, 1, 2, 3})
    assert sum(np.isin(idx, [0, 1])) % 2 == 0


def test_runner_rejects_non_inner_group_count_before_heavy_inference():
    records = {0: _seed(), 1: _seed()}
    with pytest.raises(ValueError, match="2234"):
        run_a_inference(records)


def test_missing_clause_uses_full_k_bonferroni_and_keeps_incomplete_slot():
    names = a_margin_names()
    point = np.zeros(6885)
    point[1] = np.nan
    boot = np.repeat(np.linspace(-.01, .01, 2048)[:, None], 6884, axis=1)
    clauses, method, critical = summarize_a_bootstrap(names, point, boot)
    assert method == "two-sided-fixed-K-Bonferroni-t-approximation"
    assert critical > 4
    assert len(clauses) == 6885
    assert clauses[1]["status"] == "INCOMPLETE_MISSING_ARM"
    assert clauses[0]["status"] == "BOUND_ONLY"
    # A three-release mean has just two release degrees of freedom. Its
    # fallback interval must be wider than a fixed-draw group-only interval.
    assert clauses[6 * 765]["upper"] > clauses[0]["upper"]
    assert critical > 100
