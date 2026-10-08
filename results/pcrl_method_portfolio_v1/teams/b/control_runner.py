"""Map the sealed Team B control plan onto the generic StageRunner API.

This module performs no fitting at import time and never loads Adult source
data. An outer queue worker supplies admitted FitData, a verified checkpoint,
and the fixed first-stage view for sequential stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from bottleneck import LossWeights
from control_plan import Stage, verify_stage_dependency
from train_bottleneck import StageConfig


@dataclass(frozen=True)
class ControlHyperparameters:
    steps_per_encoder: int
    batch_size: int
    learning_rate: float
    critic_learning_rate: float
    adam_betas: tuple[float, float]
    adam_eps: float
    weight_decay: float
    task_ce_weight: float
    distillation_weight: float
    bottleneck_kl_weight: float
    local_sex_weight: float
    pair_sex_weight: float


def stage_config(
    stage: Stage,
    hyper: ControlHyperparameters,
    *,
    first_stage_receipt: Mapping[str, str] | None = None,
    recomputed_first_stage_hashes: Mapping[str, str] | None = None,
) -> StageConfig:
    """Build one runnable config without aliasing or changing arm budgets."""
    t = hyper.steps_per_encoder
    if stage.task_updates_per_trained_encoder != t or t <= 0:
        raise ValueError("Stage and sealed per-encoder optimizer budget differ")
    local_counts = (stage.income_critic_updates, stage.occupation_critic_updates)
    active_local = tuple(purpose for purpose, count in zip(("income", "occupation"), local_counts)
                         if count > 0)
    ratios = {count // t for count in local_counts if count > 0 and count % t == 0}
    if any(count > 0 and count % t for count in local_counts) or len(ratios) > 1:
        raise ValueError("Local critic counts must have one exact per-step ratio")
    local_updates_per_step = next(iter(ratios), 0)
    if stage.pair_critic_updates % t:
        raise ValueError("Pair critic count must be divisible by encoder steps")
    pair_updates_per_step = stage.pair_critic_updates // t
    privacy_views = active_local + (("pair",) if pair_updates_per_step else ())

    if stage.first_stage_fixed_draw_required:
        if first_stage_receipt is None or recomputed_first_stage_hashes is None:
            raise ValueError("Sequential second stage requires recomputed first-stage identities")
        verify_stage_dependency(stage, first_stage_receipt, recomputed_first_stage_hashes)
        frozen_view_sha256 = first_stage_receipt["fixed_first_release_sha256"]
        first_checkpoint_sha256 = first_stage_receipt["checkpoint_sha256"]
        sampler_plan_sha256 = first_stage_receipt["sampler_plan_sha256"]
    else:
        if first_stage_receipt is not None or recomputed_first_stage_hashes is not None:
            raise ValueError("Unexpected first-stage dependency for nonsequential stage")
        frozen_view_sha256 = first_checkpoint_sha256 = sampler_plan_sha256 = None

    weights = LossWeights(
        task_ce=hyper.task_ce_weight,
        distillation=hyper.distillation_weight,
        bottleneck_kl=hyper.bottleneck_kl_weight if stage.use_bottleneck_kl else 0.0,
        # Joint .05*mean(CE1, CE2) is .025 per purpose. One-purpose local or
        # sequential stages use that same per-purpose pressure, not .05.
        local_sex=(hyper.local_sex_weight / 2 if len(active_local) == 1
                   else hyper.local_sex_weight if len(active_local) == 2 else 0.0),
        coalition_sex=hyper.pair_sex_weight if pair_updates_per_step else 0.0,
    )
    return StageConfig(
        stage_id=stage.stage_id,
        fit_role_id="OSF_DEFENSE_FIT",
        train_purposes=stage.train_purposes,
        frozen_purposes=stage.frozen_purposes,
        frozen_view_sha256=frozen_view_sha256,
        first_checkpoint_sha256=first_checkpoint_sha256,
        sampler_plan_sha256=sampler_plan_sha256,
        privacy_views=privacy_views,
        stochastic=stage.stochastic,
        steps=t,
        batch_size=hyper.batch_size,
        local_critic_updates=local_updates_per_step,
        pair_critic_updates=pair_updates_per_step,
        weights=weights,
        learning_rate=hyper.learning_rate,
        critic_learning_rate=hyper.critic_learning_rate,
        adam_betas=hyper.adam_betas,
        adam_eps=hyper.adam_eps,
        weight_decay=hyper.weight_decay,
        seed=stage.seed,
    )
