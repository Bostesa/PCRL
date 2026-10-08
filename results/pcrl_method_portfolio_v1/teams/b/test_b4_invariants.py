"""Independent reviewer checks for Team B's paired stochastic controls.

Synthetic data only. Numerical execution requires S3's shared queue.
"""

import hashlib

import pytest
import torch

from bottleneck import BottleneckPair, LossWeights, SexCritics, schema_fingerprint
from train_bottleneck import FitData, FrozenPurposeView, StageConfig, StageRunner


NAMES = tuple(f"f{i}" for i in range(83))


def fixture():
    torch.manual_seed(509)
    model = BottleneckPair(
        feature_names=NAMES,
        forbidden_names=("SEX", "income_label", "occupation_label"),
        class_counts={"income": 2, "occupation": 6},
        log_variance_bounds=(-8.0, 4.0),
        hidden_dim=8,
        latent_dim=3,
    )
    critics = SexCritics(latent_dim=3, class_counts=model.class_counts, hidden_dim=8)
    data = FitData(
        x=torch.linspace(0, 1, 83).reshape(1, 83),
        labels={"income": torch.tensor([0]), "occupation": torch.tensor([2])},
        teacher_probabilities={
            "income": torch.tensor([[0.8, 0.2]]),
            "occupation": torch.tensor([[0.05, 0.05, 0.75, 0.05, 0.05, 0.05]]),
        },
        sex=torch.tensor([1]),
        feature_names=NAMES,
        permitted_feature_names=NAMES,
        expected_schema_sha256=schema_fingerprint(NAMES),
        row_ids=("private-row-1",),
        group_ids=("private-group-1",),
        teacher_identity={"income": "Ucal_1", "occupation": "Ucal_2"},
    )
    return model, critics, data


def test_fit_data_refuses_reordered_or_label_substituted_columns():
    _model, _critics, data = fixture()
    common = dict(
        x=data.x, labels=data.labels,
        teacher_probabilities=data.teacher_probabilities,
        sex=data.sex,
        permitted_feature_names=NAMES,
        expected_schema_sha256=schema_fingerprint(NAMES),
        row_ids=data.row_ids, group_ids=data.group_ids,
        teacher_identity=data.teacher_identity,
    )
    swapped = list(NAMES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match="exact S2 permitted ordered schema"):
        FitData(feature_names=tuple(swapped), **common)
    contaminated = ("SEX",) + NAMES[1:]
    with pytest.raises(ValueError, match="exact S2 permitted ordered schema"):
        FitData(feature_names=contaminated, **common)


def test_stage_refuses_differently_named_valid_fit_data():
    model, critics, data = fixture()
    alternate = ("different_feature",) + NAMES[1:]
    alternate_data = FitData(
        x=data.x, labels=data.labels,
        teacher_probabilities=data.teacher_probabilities,
        sex=data.sex,
        feature_names=alternate,
        permitted_feature_names=alternate,
        expected_schema_sha256=schema_fingerprint(alternate),
        row_ids=data.row_ids, group_ids=data.group_ids,
        teacher_identity=data.teacher_identity,
    )
    with pytest.raises(ValueError, match="feature schemas differ"):
        StageRunner(
            model, critics, alternate_data,
            config(privacy_views=(), steps=1),
            source_id="synthetic-source", role_id="defense-fit",
        )


def test_stage_refuses_role_substitution_or_missing_active_task_label():
    model, critics, data = fixture()
    with pytest.raises(ValueError, match="sealed stage role"):
        StageRunner(
            model, critics, data,
            config(privacy_views=(), steps=1),
            source_id="synthetic-source", role_id="audit-fit",
        )
    missing_labels = {name: values.clone() for name, values in data.labels.items()}
    missing_labels["income"][0] = -1
    missing_data = FitData(
        x=data.x, labels=missing_labels,
        teacher_probabilities=data.teacher_probabilities,
        sex=data.sex,
        feature_names=NAMES,
        permitted_feature_names=NAMES,
        expected_schema_sha256=schema_fingerprint(NAMES),
        row_ids=data.row_ids, group_ids=data.group_ids,
        teacher_identity=data.teacher_identity,
    )
    with pytest.raises(ValueError, match="Active task labels must be complete"):
        StageRunner(
            model, critics, missing_data,
            config(privacy_views=(), steps=1),
            source_id="synthetic-source", role_id="defense-fit",
        )


def config(*, privacy_views, frozen_view=None, frozen_purpose=(), steps=2):
    protected = bool(privacy_views)
    return StageConfig(
        stage_id="b4-synthetic-review",
        fit_role_id="defense-fit",
        train_purposes=("occupation",) if frozen_purpose else ("income", "occupation"),
        frozen_purposes=frozen_purpose,
        frozen_view_sha256=frozen_view.sha256 if frozen_view else None,
        first_checkpoint_sha256=frozen_view.first_checkpoint_sha256 if frozen_view else None,
        sampler_plan_sha256=frozen_view.sampler_plan_sha256 if frozen_view else None,
        privacy_views=privacy_views,
        stochastic=True,
        steps=steps,
        batch_size=1,
        local_critic_updates=1 if any(v != "pair" for v in privacy_views) else 0,
        pair_critic_updates=2 if "pair" in privacy_views else 0,
        weights=LossWeights(
            task_ce=1.0,
            distillation=0.25,
            bottleneck_kl=0.001,
            local_sex=0.05 if protected and any(v != "pair" for v in privacy_views) else 0.0,
            coalition_sex=0.05 if "pair" in privacy_views else 0.0,
        ),
        learning_rate=0.001,
        critic_learning_rate=0.001,
        adam_betas=(0.9, 0.999),
        adam_eps=1e-8,
        weight_decay=0.0,
        seed=41,
    )


def test_sequential_pair_critic_always_sees_one_frozen_first_draw():
    model, critics, data = fixture()
    checkpoint_sha = hashlib.sha256(b"synthetic-first-checkpoint").hexdigest()
    sampler_sha = hashlib.sha256(b"synthetic-first-sampler-plan").hexdigest()
    frozen = FrozenPurposeView.from_model(
        model, data,
        purpose="income", source_id="synthetic-source",
        first_checkpoint_sha256=checkpoint_sha,
        sampler_plan_sha256=sampler_sha,
        sampler_secret=b"private-review-key" * 2,
    )
    seen = []
    hook = critics.coalition.register_forward_pre_hook(
        lambda _module, inputs: seen.append(inputs[0][:, :3].detach().clone())
    )
    try:
        runner = StageRunner(
            model, critics, data,
            config(privacy_views=("occupation", "pair"), frozen_view=frozen,
                   frozen_purpose=("income",)),
            source_id="synthetic-source", role_id="defense-fit",
            frozen_views={"income": frozen},
        )
        runner.run()
    finally:
        hook.remove()
    assert len(seen) >= 4
    assert all(torch.equal(observed, frozen.z) for observed in seen)


def test_encoder_draw_is_independent_of_extra_critic_updates():
    model_task, critics_task, data_task = fixture()
    model_joint, critics_joint, data_joint = fixture()
    task_draws, joint_draws = [], []
    hook_task = model_task.heads["income"].register_forward_pre_hook(
        lambda _module, inputs: task_draws.append(inputs[0].detach().clone())
        if torch.is_grad_enabled() else None
    )
    hook_joint = model_joint.heads["income"].register_forward_pre_hook(
        lambda _module, inputs: joint_draws.append(inputs[0].detach().clone())
        if torch.is_grad_enabled() else None
    )
    try:
        StageRunner(
            model_task, critics_task, data_task,
            config(privacy_views=(), steps=1),
            source_id="synthetic-source", role_id="defense-fit",
        ).run()
        StageRunner(
            model_joint, critics_joint, data_joint,
            config(privacy_views=("income", "occupation", "pair"), steps=1),
            source_id="synthetic-source", role_id="defense-fit",
        ).run()
    finally:
        hook_task.remove()
        hook_joint.remove()
    assert len(task_draws) == len(joint_draws) == 1
    assert torch.equal(task_draws[0], joint_draws[0])
