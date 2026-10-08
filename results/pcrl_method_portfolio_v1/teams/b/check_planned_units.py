"""Check Team B's public planned-unit arithmetic against executable stages."""

from __future__ import annotations

import json
from pathlib import Path

from control_plan import ARMS, suite_unit_counts

PATH = Path(__file__).with_name("PLANNED_UNITS.json")


def check() -> None:
    planned = json.loads(PATH.read_text(encoding="utf-8"))
    if planned["release_arms"] != list(ARMS):
        raise AssertionError("Arm order or identities differ from executable control plan")
    t = planned["provisional_T"]
    for section, seeds in (
        ("screen_two_seed_unique", (0, 1)),
        ("full_three_seed_unique_if_B_is_finalist", (0, 1, 2)),
        ("additional_seed_2_if_finalist", (2,)),
    ):
        actual = suite_unit_counts(seeds, t)
        expected = planned[section]
        mapping = {
            "training_process_jobs": "training_process_jobs",
            "encoder_fit_stages": "encoder_fit_stages",
            "critic_model_paths": "critic_model_paths",
            "calibration_scalar_fits": "calibration_scalar_fits",
            "fresh_attacker_model_fits": "fresh_attacker_model_fits",
            "pair_ignore_local_reader_aliases": "ignore_recipient_prediction_aliases",
            "frozen_first_stage_defense_fit_forward_passes": "frozen_first_stage_defense_fit_forward_passes",
            "paired_role_forward_passes": "paired_role_forward_passes",
            "recipient_role_forward_passes": "recipient_role_forward_passes",
        }
        for actual_name, expected_name in mapping.items():
            if actual[actual_name] != expected[expected_name]:
                raise AssertionError(f"{section}: {actual_name} actual={actual[actual_name]} expected={expected[expected_name]}")
        if actual["encoder_optimizer_updates"] != expected["encoder_optimizer_updates_if_T600"]:
            raise AssertionError(f"{section}: encoder optimizer updates differ")
        if actual["critic_optimizer_updates"] != expected["critic_optimizer_updates_if_T600"]:
            raise AssertionError(f"{section}: critic optimizer updates differ")
    one = suite_unit_counts((0,), t)
    for actual_name, expected_name in (
        ("training_process_jobs", "training_process_jobs"),
        ("encoder_fit_stages", "encoder_fit_stages"),
        ("critic_model_paths", "critic_model_paths"),
        ("calibration_scalar_fits", "calibration_scalar_fits"),
        ("fresh_attacker_model_fits", "fresh_attacker_model_fits"),
        ("pair_ignore_local_reader_aliases", "ignore_recipient_prediction_aliases"),
        ("paired_role_forward_passes", "paired_role_forward_passes"),
    ):
        if one[actual_name] != planned["per_seed"][expected_name]:
            raise AssertionError(f"per_seed: {actual_name} differs")


if __name__ == "__main__":
    check()
    print("Team B planned units match executable control stages")
