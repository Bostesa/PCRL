"""Exact, data-free Team B control budget checks."""

import pytest

from control_plan import (
    ARMS, DEPENDENCY_HASH_FIELDS, stages_for_arm, suite_unit_counts,
    validate_suite, verify_stage_dependency,
)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_arm_has_two_purpose_fits_and_matched_updates(seed):
    validate_suite(seed, 17)
    assert sum(len(stages_for_arm(arm, seed, 17)) for arm in ARMS) == 11


def test_sequential_orders_freeze_first_encoder_and_train_pair_critic():
    for arm, first in (("B-SEQ12", "income"), ("B-SEQ21", "occupation")):
        stage1, stage2 = stages_for_arm(arm, 0, 11)
        assert stage1.train_purposes == (first,)
        assert stage1.pair_critic_updates == 0
        assert stage2.frozen_purposes == (first,)
        assert stage2.depends_on == stage1.stage_id
        assert stage2.pair_critic_updates == 22


def test_complete_screen_and_finalist_counts():
    screen = suite_unit_counts((0, 1), 17)
    assert screen == {
        "arm_seed_cells": 16,
        "training_process_jobs": 22,
        "encoder_fit_stages": 32,
        "encoder_optimizer_updates": 32 * 17,
        "critic_model_paths": 34,
        "critic_optimizer_updates": 48 * 17,
        "calibration_scalar_fits": 32,
        "fresh_attacker_model_fits": 192,
        "fresh_score_readers_including_complements": 384,
        "pair_ignore_local_reader_aliases": 256,
        "frozen_first_stage_defense_fit_forward_passes": 4,
        "paired_role_forward_passes": 48,
        "recipient_role_forward_passes": 96,
    }
    finalist = suite_unit_counts((0, 1, 2), 17)
    assert finalist["training_process_jobs"] == 33
    assert finalist["fresh_attacker_model_fits"] == 288


def test_bad_or_duplicate_seed_and_missing_budget_fail():
    with pytest.raises(ValueError):
        suite_unit_counts((0, 0), 17)
    with pytest.raises(ValueError):
        stages_for_arm("B-JVIB", 0, 0)


def test_sequential_stage_refuses_changed_first_draw_or_row_order():
    first, second = stages_for_arm("B-SEQ12", 0, 11)
    assert second.first_stage_fixed_draw_required
    receipt = {"stage_id": first.stage_id} | {field: "a" * 64 for field in DEPENDENCY_HASH_FIELDS}
    verify_stage_dependency(second, receipt, dict(receipt))
    actual = dict(receipt)
    actual["fixed_first_release_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="first_release"):
        verify_stage_dependency(second, receipt, actual)
    actual = dict(receipt)
    actual["defense_fit_row_ids_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="row_ids"):
        verify_stage_dependency(second, receipt, actual)
