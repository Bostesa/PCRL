"""Convert S3 smoke receipts into a conservative Team B CPU reservation.

Input times are measured CPU seconds per *unique* unit from a complete
representative seed. This is an estimator, never a substitute for actual queue
receipts. Missing measurements fail closed before a campaign lock is sealed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from control_plan import ARMS, suite_stages, suite_unit_counts

ATTACK_MODELS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32")
VIEWS = ("income", "occupation", "pair")
ROLES = ("CALIBRATION_HELDOUT", "ATTACK_FIT_NEW", "INNER_SELECTION")


def required_smoke_keys(steps_per_encoder: int) -> tuple[str, ...]:
    stages = suite_stages(0, steps_per_encoder)
    return tuple(
        [f"train:{stage.stage_id}" for stage in stages]
        + [f"attack:{view}:{model}" for view in VIEWS for model in ATTACK_MODELS]
        + [f"calibrate:{purpose}" for purpose in ("income", "occupation")]
        + [f"forward:{role}" for role in ROLES]
        + ["forward:CALIBRATION_HELDOUT_REPLAY"]
        + ["forward:OSF_DEFENSE_FIT_FIRST_STAGE"]
        + ["control:constant_reader", "control:shuffled_attacker", "control:planted_attacker"]
    )


def estimate(smoke: dict, *, steps_per_encoder: int, finalist: bool) -> dict:
    """Scale a one-seed matched smoke to two or three paired seeds.

    A separate constant setup/replay reservation is required; the continuation
    budget belongs to the whole portfolio, so this function does not pronounce
    global admission. The S3 queue checks cumulative CPU receipts.
    """
    keys = required_smoke_keys(steps_per_encoder)
    missing = sorted(set(keys) - set(smoke))
    if missing:
        raise ValueError(f"Missing required S3 smoke CPU receipts: {missing}")
    for key in keys:
        value = smoke[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid CPU seconds for {key}")
    seed_count = 3 if finalist else 2
    training_one_seed = sum(smoke[f"train:{stage.stage_id}"] for stage in suite_stages(0, steps_per_encoder))
    attacks_one_arm_seed = sum(smoke[f"attack:{view}:{model}"] for view in VIEWS for model in ATTACK_MODELS)
    calibration_one_arm_seed = sum(smoke[f"calibrate:{purpose}"] for purpose in ("income", "occupation"))
    forwards_one_arm_seed = sum(smoke[f"forward:{role}"] for role in ROLES)
    # The registered control battery is two adapter classes × three views ×
    # four fits per shuffled/plant condition, plus six constant scorings.
    control_cpu_s = (
        24 * smoke["control:shuffled_attacker"]
        + 24 * smoke["control:planted_attacker"]
        + 6 * smoke["control:constant_reader"]
    )
    release_arm_count = len(ARMS)
    defense_cpu_s = seed_count * training_one_seed
    attack_cpu_s = seed_count * release_arm_count * attacks_one_arm_seed
    calibration_cpu_s = seed_count * release_arm_count * calibration_one_arm_seed
    forward_cpu_s = (seed_count * release_arm_count *
                     (forwards_one_arm_seed + smoke["forward:CALIBRATION_HELDOUT_REPLAY"])
                     + seed_count * 2 * smoke["forward:OSF_DEFENSE_FIT_FIRST_STAGE"])
    total_cpu_s = sum((defense_cpu_s, attack_cpu_s, calibration_cpu_s, forward_cpu_s, control_cpu_s))
    return {
        "source": "S3 measured one-seed unit receipts; extrapolation conditional on same fixed architecture and rows",
        "steps_per_encoder": steps_per_encoder,
        "seed_count": seed_count,
        "unit_counts": suite_unit_counts(tuple(range(seed_count)), steps_per_encoder),
        "defense_cpu_s": defense_cpu_s,
        "attack_cpu_s": attack_cpu_s,
        "calibration_cpu_s": calibration_cpu_s,
        "forward_cpu_s": forward_cpu_s,
        "control_cpu_s": control_cpu_s,
        "total_cpu_s": total_cpu_s,
        "total_cpu_h": total_cpu_s / 3600,
        "excludes": ["shared role export/preprocessing", "other families", "independent replay", "failed attempts"],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoke_json", type=Path)
    parser.add_argument("--steps-per-encoder", type=int, required=True)
    parser.add_argument("--finalist", action="store_true")
    args = parser.parse_args()
    measured = json.loads(args.smoke_json.read_text(encoding="utf-8"))
    print(json.dumps(estimate(measured, steps_per_encoder=args.steps_per_encoder,
                              finalist=args.finalist), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
