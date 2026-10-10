"""Equal-continuation ordinary adversarial hard-code compressor C.

The caller supplies the already task-trained state and the common continuation
length. This stage never reads any role itself, and only accepts admitted
FitData. Its model uses the same hard-code service as T and D-K.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.fit import (
    FitData,
    TrainConfig,
    _task_loss,
    make_batch_plan,
)
from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.model import (
    FixedScaler,
    HardCodeModel,
)
from .adversary import HardCodeCritics


@dataclass
class CStageResult:
    model: HardCodeModel
    critics: HardCodeCritics
    history: list[dict]
    seed: int
    train_rows: int
    optimizer_updates: int
    critic_updates: int


def fit_c_stage(data: FitData, config: TrainConfig, seed: int,
                init_state: dict[str, torch.Tensor], scaler: FixedScaler,
                train_idx: np.ndarray, coefficient: float = 1.0,
                critic_learning_rate: float = 0.001,
                critic_steps_per_update: int = 2,
                batch_plan: tuple[tuple[np.ndarray, ...], ...] | None = None) -> CStageResult:
    """Continue the T model with local and coalition adversaries.

    Each epoch traverses the identical paired-row permutation and batches as
    the hard-code trainer. Critics see the actual hard forward code, then the
    mapper maximizes their balanced BCE with a straight-through gradient.
    """
    if data.sex is None:
        raise ValueError("C requires fitting-role SEX supervision")
    if (not np.isfinite(coefficient) or coefficient < 0
            or not np.isfinite(critic_learning_rate) or critic_learning_rate <= 0
            or critic_steps_per_update < 1):
        raise ValueError("invalid frozen adversarial schedule")
    train = np.asarray(train_idx)
    if (train.ndim != 1 or len(train) < 2 or not np.issubdtype(train.dtype, np.integer)
            or np.any(train < 0) or np.any(train >= data.rows)
            or len(np.unique(train)) != len(train)):
        raise ValueError("explicit unique permitted fitting indices required")
    if scaler.dim != data.h0.shape[1]:
        raise ValueError("shared scaler and latent dimensions differ")
    expected_plan = make_batch_plan(train, seed=seed, epochs=config.epochs,
                                    batch_size=config.batch_size, sex=data.sex)
    if batch_plan is None:
        batch_plan = expected_plan
    elif (len(batch_plan) != len(expected_plan)
          or any(len(given) != len(expected) or any(
              not np.array_equal(a, b) for a, b in zip(given, expected))
                 for given, expected in zip(batch_plan, expected_plan))):
        raise ValueError("C batch plan differs from frozen paired FIT plan")
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    model = HardCodeModel(scaler.dim, (2, 6))
    model.load_state_dict(init_state, strict=True)
    # Critic initialization uses a separate declared stream. No mapper/head
    # random operation follows the loaded state.
    torch.manual_seed(int(seed) + 300007)
    critics = HardCodeCritics()
    mapper_optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    critic_optimizer = torch.optim.Adam(critics.parameters(), lr=critic_learning_rate)
    x = tuple(torch.from_numpy(scaler.transform(i, h)) for i, h in enumerate((data.h0, data.h1)))
    y = tuple(torch.from_numpy(v) for v in (data.y0, data.y1))
    teacher = tuple(torch.from_numpy(v) for v in (data.teacher0, data.teacher1))
    sex = torch.from_numpy(data.sex)
    history: list[dict] = []
    optimizer_updates = critic_updates = 0
    for epoch in range(config.epochs):
        model.train()
        total_task = total_adv = 0.0
        batches = 0
        for chunk in batch_plan[epoch]:
            ix = torch.from_numpy(chunk)
            group = sex[ix]
            if not bool((group == 0).any()) or not bool((group == 1).any()):
                raise ValueError("C minibatch lacks one SEX group")
            with torch.no_grad():
                detached_bits = tuple(model(i, x[i][ix], False)[0] for i in (0, 1))
            for _ in range(critic_steps_per_update):
                critics.critic_step(detached_bits[0], detached_bits[1], group,
                                    critic_optimizer)
                critic_updates += 1
            bits, logits = zip(*(model(i, x[i][ix], True) for i in (0, 1)))
            task = _task_loss(logits, (y[0][ix], y[1][ix]),
                              (teacher[0][ix], teacher[1][ix]), config.distill_weight)
            adv = critics.mapper_loss(bits[0], bits[1], group, coefficient)
            mapper_optimizer.zero_grad(set_to_none=True)
            (task + adv).backward()
            mapper_optimizer.step()
            total_task += float(task.detach())
            total_adv += float(adv.detach())
            optimizer_updates += 1
            batches += 1
        history.append({"epoch": epoch + 1, "updates": optimizer_updates,
                        "critic_updates": critic_updates,
                        "task_loss": total_task / batches,
                        "adversarial_loss": total_adv / batches})
    model.eval()
    critics.eval()
    return CStageResult(model, critics, history, int(seed), len(train),
                        optimizer_updates, critic_updates)
