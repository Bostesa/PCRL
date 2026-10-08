"""Queue-only synthetic timing for one full Team B joint training stage.

This script uses no Adult rows, labels, predictors, or checkpoints. S3 admits
the exact file hash and command before execution. Printed values are aggregate
resource receipts, not empirical method results.
"""

from __future__ import annotations

import json
import resource
import sys
import time

import torch

from bottleneck import BottleneckPair, LossWeights, SexCritics, schema_fingerprint
from train_bottleneck import FitData, StageConfig, StageRunner


def main() -> None:
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError("Synthetic timing requires one numerical thread")
    n = 15_434
    generator = torch.Generator(device="cpu").manual_seed(20_260_108)
    start = time.monotonic()
    x = torch.randn(n, 83, generator=generator)
    income_y = torch.randint(2, (n,), generator=generator)
    occupation_y = torch.randint(6, (n,), generator=generator)
    sex = torch.randint(2, (n,), generator=generator)
    income_teacher = torch.softmax(torch.randn(n, 2, generator=generator), dim=1)
    occupation_teacher = torch.softmax(torch.randn(n, 6, generator=generator), dim=1)
    feature_names = tuple(f"synthetic_feature_{index:02d}" for index in range(83))
    data = FitData(
        x=x, labels={"income": income_y, "occupation": occupation_y},
        teacher_probabilities={"income": income_teacher,
                               "occupation": occupation_teacher},
        sex=sex,
        feature_names=feature_names, permitted_feature_names=feature_names,
        expected_schema_sha256=schema_fingerprint(feature_names),
        row_ids=tuple(f"synthetic-row-{index:05d}" for index in range(n)),
        group_ids=tuple(f"synthetic-group-{min(index, 15_427):05d}" for index in range(n)),
        teacher_identity={"income": "Ucal_1", "occupation": "Ucal_2"},
    )
    torch.manual_seed(20_260_109)
    model = BottleneckPair(
        feature_names=feature_names,
        forbidden_names=("SEX", "income", "occupation"),
        class_counts={"income": 2, "occupation": 6},
        log_variance_bounds=(-8.0, 4.0), hidden_dim=64, latent_dim=16,
    )
    critics = SexCritics(latent_dim=16, class_counts=model.class_counts, hidden_dim=64)
    config = StageConfig(
        stage_id="B-JVIB-SYNTHETIC-TIMING", fit_role_id="SYNTHETIC_TIMING_ONLY",
        train_purposes=("income", "occupation"),
        frozen_purposes=(), frozen_view_sha256=None, first_checkpoint_sha256=None,
        sampler_plan_sha256=None, privacy_views=("income", "occupation", "pair"),
        stochastic=True, steps=600, batch_size=256,
        local_critic_updates=1, pair_critic_updates=2,
        weights=LossWeights(task_ce=1.0, distillation=0.25, bottleneck_kl=0.001,
                            local_sex=0.05, coalition_sex=0.05),
        learning_rate=0.001, critic_learning_rate=0.001,
        adam_betas=(0.9, 0.999), adam_eps=1e-8, weight_decay=0.0,
        seed=20_260_110,
    )
    runner = StageRunner(model, critics, data, config,
                         source_id="SYNTHETIC_TIMING_ONLY", role_id="SYNTHETIC_TIMING_ONLY")
    setup_seconds = time.monotonic() - start
    cpu_start = time.process_time()
    train_start = time.monotonic()
    receipt = runner.run()
    training_seconds = time.monotonic() - train_start
    cpu_seconds = time.process_time() - cpu_start
    expected = {"income": 600, "occupation": 600}
    if (receipt["encoder_steps"] != expected
            or receipt["local_critic_steps"] != expected
            or receipt["pair_critic_steps"] != 1_200
            or receipt["status"] != "COMPLETED"):
        raise AssertionError("Joint timing unit did not run the declared update schedule")
    print(json.dumps({
        "kind": "TEAM_B_SYNTHETIC_TIMING_ONLY", "rows": n, "columns": 83,
        "setup_wall_seconds": setup_seconds, "training_wall_seconds": training_seconds,
        "training_cpu_seconds": cpu_seconds,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "completed_steps": receipt["completed_steps"],
        "encoder_steps": receipt["encoder_steps"],
        "local_critic_steps": receipt["local_critic_steps"],
        "pair_critic_steps": receipt["pair_critic_steps"],
        "code_sha256": receipt["code_sha256"], "config_sha256": receipt["config_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
