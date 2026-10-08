"""Exact prefit stage plan for the eight Team B feature-release arms.

This module contains no data loader or training loop. The trainer consumes each
stage and supplies the checkpoint listed by ``depends_on`` before a dependent
stage starts. ``steps_per_encoder`` is set from the sealed campaign config; no
default can silently change a scientific budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from collections.abc import Mapping
from typing import Literal

Purpose = Literal["income", "occupation"]
Arm = Literal["B-JVIB", "B-DVIB", "B-TASK", "B-RAW-TASK", "B-LOCAL", "B-SEQ12", "B-SEQ21", "B-RAWJ"]
PURPOSES: tuple[Purpose, Purpose] = ("income", "occupation")
ARMS: tuple[Arm, ...] = (
    "B-JVIB", "B-DVIB", "B-TASK", "B-RAW-TASK", "B-LOCAL", "B-SEQ12", "B-SEQ21", "B-RAWJ"
)


@dataclass(frozen=True)
class Stage:
    arm: Arm
    seed: int
    stage_id: str
    train_purposes: tuple[Purpose, ...]
    frozen_purposes: tuple[Purpose, ...]
    stochastic: bool
    use_bottleneck_kl: bool
    task_updates_per_trained_encoder: int
    income_critic_updates: int
    occupation_critic_updates: int
    pair_critic_updates: int
    depends_on: str | None = None
    first_stage_fixed_draw_required: bool = False

    @property
    def encoder_update_total(self) -> int:
        return self.task_updates_per_trained_encoder * len(self.train_purposes)

    @property
    def critic_update_total(self) -> int:
        return (self.income_critic_updates + self.occupation_critic_updates
                + self.pair_critic_updates)

    def receipt_fields(self) -> dict:
        return asdict(self) | {
            "encoder_update_total": self.encoder_update_total,
            "critic_update_total": self.critic_update_total,
        }


def stages_for_arm(arm: Arm, seed: int, steps_per_encoder: int) -> tuple[Stage, ...]:
    """Return every scheduled training job, with no optimistic aliasing."""
    if arm not in ARMS:
        raise ValueError(f"Unknown Team B arm: {arm}")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("A nonnegative integer model seed is required")
    if not isinstance(steps_per_encoder, int) or steps_per_encoder <= 0:
        raise ValueError("Positive integer steps_per_encoder is required")
    t = steps_per_encoder
    if arm in ("B-JVIB", "B-DVIB", "B-TASK", "B-RAW-TASK", "B-RAWJ"):
        protected = arm not in ("B-TASK", "B-RAW-TASK")
        return (Stage(
            arm=arm, seed=seed, stage_id=f"{arm}/s{seed}/both",
            train_purposes=PURPOSES, frozen_purposes=(),
            stochastic=arm in ("B-JVIB", "B-TASK"),
            use_bottleneck_kl=arm not in ("B-RAW-TASK", "B-RAWJ"),
            task_updates_per_trained_encoder=t,
            income_critic_updates=t if protected else 0,
            occupation_critic_updates=t if protected else 0,
            pair_critic_updates=2*t if protected else 0,
        ),)
    if arm == "B-LOCAL":
        return tuple(Stage(
            arm=arm, seed=seed, stage_id=f"{arm}/s{seed}/{purpose}",
            train_purposes=(purpose,), frozen_purposes=(),
            stochastic=True, use_bottleneck_kl=True,
            task_updates_per_trained_encoder=t,
            income_critic_updates=2*t if purpose == "income" else 0,
            occupation_critic_updates=2*t if purpose == "occupation" else 0,
            pair_critic_updates=0,
        ) for purpose in PURPOSES)
    order = PURPOSES if arm == "B-SEQ12" else tuple(reversed(PURPOSES))
    first, second = order
    first_id = f"{arm}/s{seed}/stage1_{first}"
    return (
        Stage(
            arm=arm, seed=seed, stage_id=first_id,
            train_purposes=(first,), frozen_purposes=(),
            stochastic=True, use_bottleneck_kl=True,
            task_updates_per_trained_encoder=t,
            income_critic_updates=t if first == "income" else 0,
            occupation_critic_updates=t if first == "occupation" else 0,
            pair_critic_updates=0,
        ),
        Stage(
            arm=arm, seed=seed, stage_id=f"{arm}/s{seed}/stage2_{second}",
            train_purposes=(second,), frozen_purposes=(first,),
            stochastic=True, use_bottleneck_kl=True,
            task_updates_per_trained_encoder=t,
            income_critic_updates=t if second == "income" else 0,
            occupation_critic_updates=t if second == "occupation" else 0,
            pair_critic_updates=2*t, depends_on=first_id,
            first_stage_fixed_draw_required=True,
        ),
    )


def suite_stages(seed: int, steps_per_encoder: int) -> tuple[Stage, ...]:
    return tuple(stage for arm in ARMS for stage in stages_for_arm(arm, seed, steps_per_encoder))


def suite_unit_counts(seeds: tuple[int, ...], steps_per_encoder: int) -> dict[str, int]:
    """Account for unique jobs, internal fits, forwards and fresh attack fits."""
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("At least one unique paired seed is required")
    stages = tuple(stage for seed in seeds for stage in suite_stages(seed, steps_per_encoder))
    arm_seed = len(ARMS) * len(seeds)
    return {
        "arm_seed_cells": arm_seed,
        "training_process_jobs": len(stages),
        "encoder_fit_stages": sum(len(s.train_purposes) for s in stages),
        "encoder_optimizer_updates": sum(s.encoder_update_total for s in stages),
        "critic_model_paths": sum(
            int(s.income_critic_updates > 0) + int(s.occupation_critic_updates > 0)
            + int(s.pair_critic_updates > 0) for s in stages
        ),
        "critic_optimizer_updates": sum(s.critic_update_total for s in stages),
        "calibration_scalar_fits": 2 * arm_seed,
        "fresh_attacker_model_fits": 3 * 4 * arm_seed,
        "fresh_score_readers_including_complements": 3 * 4 * 2 * arm_seed,
        "pair_ignore_local_reader_aliases": 2 * 4 * 2 * arm_seed,
        "frozen_first_stage_defense_fit_forward_passes": 2 * len(seeds),
        "paired_role_forward_passes": 3 * arm_seed,
        "recipient_role_forward_passes": 2 * 3 * arm_seed,
    }


def validate_suite(seed: int, steps_per_encoder: int) -> None:
    """Guard against a control silently receiving fewer task updates."""
    stages = suite_stages(seed, steps_per_encoder)
    for arm in ARMS:
        own = tuple(s for s in stages if s.arm == arm)
        if sum(s.encoder_update_total for s in own) != 2 * steps_per_encoder:
            raise AssertionError(f"Mismatched encoder budget: {arm}")
        expected_critic = 0 if arm in ("B-TASK", "B-RAW-TASK") else 4 * steps_per_encoder
        if sum(s.critic_update_total for s in own) != expected_critic:
            raise AssertionError(f"Mismatched critic budget: {arm}")
        trained = sorted(p for s in own for p in s.train_purposes)
        if trained != sorted(PURPOSES):
            raise AssertionError(f"Missing or repeated purpose fit: {arm}")
        if any(s.depends_on and s.depends_on not in {x.stage_id for x in own} for s in own):
            raise AssertionError(f"Missing stage dependency: {arm}")


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
DEPENDENCY_HASH_FIELDS = (
    "checkpoint_sha256", "fixed_first_release_sha256", "defense_fit_row_ids_sha256",
    "defense_fit_group_ids_sha256", "feature_schema_sha256", "sampler_plan_sha256",
)


def verify_stage_dependency(
    stage: Stage, receipt: Mapping[str, str], actual: Mapping[str, str]
) -> None:
    """Require the exact frozen first view used by a sequential pair critic.

    ``actual`` hashes must be recomputed from files by the trainer immediately
    before stage 2, not copied from the stage-1 receipt. The receipt includes
    only hashes and a nonsecret sampler-plan ID, never the private sampler key.
    """
    if not stage.first_stage_fixed_draw_required or stage.depends_on is None:
        raise ValueError("Stage does not require a frozen first-stage view")
    if receipt.get("stage_id") != stage.depends_on:
        raise ValueError("Wrong first-stage checkpoint dependency")
    for field in DEPENDENCY_HASH_FIELDS:
        claimed = receipt.get(field)
        observed = actual.get(field)
        if not isinstance(claimed, str) or not _SHA256.fullmatch(claimed):
            raise ValueError(f"Missing or invalid first-stage {field}")
        if observed != claimed:
            raise ValueError(f"First-stage dependency hash mismatch: {field}")
