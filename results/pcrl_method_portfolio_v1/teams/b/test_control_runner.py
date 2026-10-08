"""Synthetic mapping checks; actual fits need S3 queue admission."""

import pytest

from control_plan import DEPENDENCY_HASH_FIELDS, stages_for_arm
from control_runner import ControlHyperparameters, stage_config


def hyper():
    return ControlHyperparameters(
        steps_per_encoder=10, batch_size=256,
        learning_rate=0.001, critic_learning_rate=0.001,
        adam_betas=(0.9, 0.999), adam_eps=1e-8, weight_decay=0,
        task_ce_weight=1, distillation_weight=0.25,
        bottleneck_kl_weight=0.001, local_sex_weight=0.05,
        pair_sex_weight=0.05,
    )


@pytest.mark.parametrize("arm,stochastic,kl,local,pair", [
    ("B-JVIB", True, 0.001, 1, 2),
    ("B-DVIB", False, 0.001, 1, 2),
    ("B-TASK", True, 0.001, 0, 0),
    ("B-RAW-TASK", False, 0.0, 0, 0),
    ("B-RAWJ", False, 0.0, 1, 2),
])
def test_paired_arm_config_matches_locked_schedule(arm, stochastic, kl, local, pair):
    config = stage_config(stages_for_arm(arm, 0, 10)[0], hyper())
    assert config.fit_role_id == "OSF_DEFENSE_FIT"
    assert config.stochastic is stochastic
    assert config.weights.bottleneck_kl == kl
    assert config.local_critic_updates == local
    assert config.pair_critic_updates == pair


def test_local_and_sequential_stage_configs_match_critic_budget():
    for stage in stages_for_arm("B-LOCAL", 0, 10):
        cfg = stage_config(stage, hyper())
        assert cfg.local_critic_updates == 2
        assert cfg.pair_critic_updates == 0
        assert cfg.weights.local_sex == 0.025
    first, second = stages_for_arm("B-SEQ12", 0, 10)
    first_cfg = stage_config(first, hyper())
    assert first_cfg.local_critic_updates == 1
    assert first_cfg.weights.local_sex == 0.025
    receipt = {"stage_id": first.stage_id} | {field: "a" * 64 for field in DEPENDENCY_HASH_FIELDS}
    cfg = stage_config(second, hyper(), first_stage_receipt=receipt,
                       recomputed_first_stage_hashes=receipt)
    assert cfg.frozen_purposes == ("income",)
    assert cfg.local_critic_updates == 1
    assert cfg.pair_critic_updates == 2
    assert cfg.weights.local_sex == 0.025
    with pytest.raises(ValueError, match="requires recomputed"):
        stage_config(second, hyper())
