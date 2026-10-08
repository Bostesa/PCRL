"""Synthetic-only numerical contract tests for Team B's stage trainer."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from bottleneck import BottleneckPair, LossWeights, SexCritics, schema_fingerprint
from train_bottleneck import FitData, FrozenPurposeView, StageConfig, StageRunner


NAMES = tuple(f"f{i}" for i in range(83))


def fixtures(*, with_sex=True):
    torch.manual_seed(21)
    model = BottleneckPair(
        feature_names=NAMES, forbidden_names=("SEX", "income_label", "occupation_label"),
        class_counts={"income": 2, "occupation": 6}, log_variance_bounds=(-8.0, 4.0),
        hidden_dim=8, latent_dim=3,
    )
    critics = SexCritics(latent_dim=3, class_counts=model.class_counts, hidden_dim=8)
    x = torch.arange(10 * 83, dtype=torch.float32).reshape(10, 83) / 1000
    labels = {
        "income": torch.tensor([0, 1] * 5),
        "occupation": torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3]),
    }
    teachers = {
        name: 0.8 * F.one_hot(y, num_classes=model.class_counts[name]).float()
        + 0.2 / model.class_counts[name]
        for name, y in labels.items()
    }
    sex = torch.tensor([0, 1] * 5) if with_sex else None
    data = FitData(x=x, labels=labels, teacher_probabilities=teachers, sex=sex,
                   feature_names=NAMES, permitted_feature_names=NAMES,
                   expected_schema_sha256=schema_fingerprint(NAMES),
                   row_ids=tuple(f"private-row-{i}" for i in range(10)),
                   group_ids=tuple(f"private-group-{i}" for i in range(10)),
                   teacher_identity={"income": "Ucal_1", "occupation": "Ucal_2"})
    return model, critics, data


def config(**overrides):
    base = StageConfig(
        stage_id="synthetic-joint", fit_role_id="defense-fit",
        train_purposes=("income", "occupation"),
        frozen_purposes=(), frozen_view_sha256=None,
        first_checkpoint_sha256=None, sampler_plan_sha256=None,
        privacy_views=("income", "occupation", "pair"), stochastic=True,
        steps=4, batch_size=5, local_critic_updates=1, pair_critic_updates=2,
        weights=LossWeights(task_ce=1.0, distillation=0.25, bottleneck_kl=0.001,
                            local_sex=0.05, coalition_sex=0.05),
        learning_rate=0.001, critic_learning_rate=0.001,
        adam_betas=(0.9, 0.999), adam_eps=1e-8, weight_decay=0.0, seed=31,
    )
    return replace(base, **overrides)


def runner(model, critics, data, cfg, frozen_views=None):
    return StageRunner(model, critics, data, cfg, source_id="synthetic-source",
                       role_id="defense-fit", frozen_views=frozen_views)


def test_joint_stage_updates_each_purpose_and_charges_exact_critic_exposures():
    """Catches an untrained head or a missing local/pair critic update."""
    model, critics, data = fixtures()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    fit = runner(model, critics, data, config())
    receipt = fit.run()
    assert receipt["completed_steps"] == 4
    assert receipt["encoder_steps"] == {"income": 4, "occupation": 4}
    assert receipt["local_critic_steps"] == {"income": 4, "occupation": 4}
    assert receipt["pair_critic_steps"] == 8
    assert not torch.equal(before["heads.income.weight"], model.state_dict()["heads.income.weight"])
    assert not torch.equal(before["heads.occupation.weight"], model.state_dict()["heads.occupation.weight"])


def test_task_only_stage_does_not_require_sex_or_update_critics():
    """Catches a task-only control silently using protected labels."""
    model, critics, data = fixtures(with_sex=False)
    before = {key: value.clone() for key, value in critics.state_dict().items()}
    cfg = config(stage_id="synthetic-task", privacy_views=(), local_critic_updates=0,
                 pair_critic_updates=0,
                 weights=LossWeights(task_ce=1, distillation=0.25, bottleneck_kl=0.001,
                                     local_sex=0, coalition_sex=0))
    receipt = runner(model, critics, data, cfg).run()
    assert receipt["local_critic_steps"] == {"income": 0, "occupation": 0}
    assert receipt["pair_critic_steps"] == 0
    assert all(torch.equal(before[key], value) for key, value in critics.state_dict().items())


def test_sequential_second_stage_keeps_first_encoder_and_draw_frozen(tmp_path):
    """Catches joint backpropagation through the previously released purpose."""
    model, critics, data = fixtures()
    income_before = {key: value.clone() for key, value in model.state_dict().items()
                     if key.startswith("encoders.income.") or key.startswith("heads.income.")}
    frozen = FrozenPurposeView.from_model(
        model, data, purpose="income", source_id="synthetic-source",
        first_checkpoint_sha256="a" * 64, sampler_plan_sha256="b" * 64,
        sampler_secret=b"k" * 32,
    )
    frozen_path = tmp_path / "private-first-view.pt"
    frozen.save(frozen_path)
    assert frozen_path.stat().st_mode & 0o777 == 0o600
    frozen = FrozenPurposeView.load(frozen_path, expected_sha256=frozen.sha256)
    cfg = config(stage_id="synthetic-second", train_purposes=("occupation",),
                 frozen_purposes=("income",), frozen_view_sha256=frozen.sha256,
                 first_checkpoint_sha256="a" * 64, sampler_plan_sha256="b" * 64,
                 privacy_views=("occupation", "pair"))
    model.encoders["income"].forward = lambda _: (_ for _ in ()).throw(AssertionError("first redraw"))
    receipt = runner(model, critics, data, cfg, frozen_views={"income": frozen}).run()
    assert receipt["encoder_steps"] == {"income": 0, "occupation": 4}
    assert receipt["local_critic_steps"] == {"income": 0, "occupation": 4}
    assert receipt["pair_critic_steps"] == 8
    assert receipt["frozen_view_sha256"] == frozen.sha256
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in income_before.items())
    with pytest.raises(ValueError, match="digest mismatch"):
        FrozenPurposeView.load(frozen_path, expected_sha256="0" * 64)


def test_checkpoint_resume_is_bitwise_identical_and_rejects_changed_fit_data(tmp_path):
    """Catches a reset optimizer, minibatch order or sampler after restart."""
    model_a, critics_a, data = fixtures()
    uninterrupted = runner(model_a, critics_a, data, config())
    uninterrupted.run()

    model_b, critics_b, data_b = fixtures()
    interrupted = runner(model_b, critics_b, data_b, config())
    interrupted.run(steps=2)
    path = tmp_path / "private-stage.pt"
    interrupted.save_checkpoint(path)
    assert path.stat().st_mode & 0o777 == 0o600
    model_c, critics_c, data_c = fixtures()
    resumed = StageRunner.load_checkpoint(path, model_c, critics_c, data_c, config(),
                                          source_id="synthetic-source", role_id="defense-fit")
    resumed.run()
    assert all(torch.equal(value, model_c.state_dict()[key]) for key, value in model_a.state_dict().items())
    assert all(torch.equal(value, critics_c.state_dict()[key]) for key, value in critics_a.state_dict().items())
    changed_x = data_c.x.clone()
    changed_x[0, 0] += 1
    changed_data = FitData(x=changed_x, labels=data_c.labels,
                           teacher_probabilities=data_c.teacher_probabilities, sex=data_c.sex,
                           feature_names=data_c.feature_names,
                           permitted_feature_names=data_c.feature_names,
                           expected_schema_sha256=data_c.feature_schema_sha256,
                           row_ids=data_c.row_ids, group_ids=data_c.group_ids,
                           teacher_identity=data_c.teacher_identity)
    model_d, critics_d, _ = fixtures()
    with pytest.raises(ValueError, match="data identity"):
        StageRunner.load_checkpoint(path, model_d, critics_d, changed_data, config(),
                                    source_id="synthetic-source", role_id="defense-fit")


def test_single_purpose_objective_uses_only_its_task_loss():
    """Catches averaging a frozen inactive head into sequential utility."""
    model, critics, data = fixtures(with_sex=False)
    cfg = config(stage_id="single-task", train_purposes=("occupation",),
                 privacy_views=(), stochastic=False, local_critic_updates=0,
                 pair_critic_updates=0,
                 weights=LossWeights(task_ce=1, distillation=0, bottleneck_kl=0,
                                     local_sex=0, coalition_sex=0))
    fit = runner(model, critics, data, cfg)
    indices = fit.order[:5]
    with torch.no_grad():
        outputs = model.training_outputs(data.x[indices], stochastic=False,
                                         generator=torch.Generator().manual_seed(5))
        expected = F.cross_entropy(outputs["occupation"]["logits"], data.labels["occupation"][indices])
    fit.run(steps=1)
    assert fit.history[0]["task_ce"] == pytest.approx(expected.item(), abs=1e-7)


def test_single_purpose_local_stage_never_evaluates_inactive_encoder():
    """Local fitting spends forward compute only on the released purpose."""
    model, critics, data = fixtures()
    cfg = config(stage_id="synthetic-local-income", train_purposes=("income",),
                 privacy_views=("income",), local_critic_updates=2,
                 pair_critic_updates=0,
                 weights=LossWeights(task_ce=1, distillation=0.25,
                                     bottleneck_kl=0.001, local_sex=0.025,
                                     coalition_sex=0))
    model.encoders["occupation"].forward = lambda _: (_ for _ in ()).throw(
        AssertionError("inactive encoder forward"))
    receipt = runner(model, critics, data, cfg).run(steps=1)
    assert receipt["encoder_steps"] == {"income": 1, "occupation": 0}
    assert receipt["local_critic_steps"] == {"income": 2, "occupation": 0}


def test_stage_refuses_wrong_role_and_missing_active_task_labels():
    model, critics, data = fixtures()
    with pytest.raises(ValueError, match="sealed stage role"):
        StageRunner(model, critics, data, config(), source_id="synthetic-source",
                    role_id="inner-selection")
    missing = data.labels["income"].clone()
    missing[0] = -1
    bad_data = FitData(
        x=data.x, labels={"income": missing, "occupation": data.labels["occupation"]},
        teacher_probabilities=data.teacher_probabilities, sex=data.sex,
        feature_names=data.feature_names, permitted_feature_names=data.feature_names,
        expected_schema_sha256=data.feature_schema_sha256,
        row_ids=data.row_ids, group_ids=data.group_ids,
        teacher_identity=data.teacher_identity,
    )
    with pytest.raises(ValueError, match="Active task labels must be complete"):
        runner(model, critics, bad_data, config())


def test_encoder_noise_stream_does_not_depend_on_critic_update_count():
    """Catches different privacy arms consuming different task-sampler draws."""
    model_a, critics_a, data_a = fixtures()
    model_b, critics_b, data_b = fixtures()
    joint = runner(model_a, critics_a, data_a, config())
    task = runner(model_b, critics_b, data_b,
                  config(stage_id="task", privacy_views=(), local_critic_updates=0,
                         pair_critic_updates=0,
                         weights=LossWeights(task_ce=1, distillation=0.25,
                                             bottleneck_kl=0.001, local_sex=0, coalition_sex=0)))
    assert all(torch.equal(joint.encoder_generators[p].get_state(),
                           task.encoder_generators[p].get_state()) for p in ("income", "occupation"))
    joint.run(steps=1)
    task.run(steps=1)
    assert all(torch.equal(joint.encoder_generators[p].get_state(),
                           task.encoder_generators[p].get_state()) for p in ("income", "occupation"))
