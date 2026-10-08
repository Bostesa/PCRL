"""End-to-end synthetic control provenance checks; run through S3 queue."""

import copy

import pytest
import torch

from bottleneck import BottleneckPair, SexCritics, schema_fingerprint
from control_artifacts import (
    assemble_local_pair, file_sha256, paired_initial_copies,
    verify_sequential_first_artifacts,
)
from control_plan import stages_for_arm
from control_runner import ControlHyperparameters, stage_config
from train_bottleneck import FitData, FrozenPurposeView, StageRunner


def fixtures():
    torch.manual_seed(7)
    names = tuple(f"f{i}" for i in range(83))
    model = BottleneckPair(
        feature_names=names,
        forbidden_names=("SEX", "income_label", "occupation_label"),
        class_counts={"income": 2, "occupation": 6},
        log_variance_bounds=(-8.0, 4.0),
    )
    critics = SexCritics(latent_dim=16, class_counts=model.class_counts)
    n = 8
    data = FitData(
        x=torch.randn(n, 83),
        feature_names=names, permitted_feature_names=names,
        expected_schema_sha256=schema_fingerprint(names),
        labels={"income": torch.tensor([0, 1] * 4),
                "occupation": torch.tensor([0, 1, 2, 3, 4, 5, 0, 1])},
        teacher_probabilities={"income": torch.full((n, 2), 0.5),
                               "occupation": torch.full((n, 6), 1/6)},
        sex=torch.tensor([0, 1] * 4),
        row_ids=tuple(str(i) for i in range(n)),
        group_ids=tuple(str(i) for i in range(n)),
        teacher_identity={"income": "Ucal_1", "occupation": "Ucal_2"},
    )
    hyper = ControlHyperparameters(
        steps_per_encoder=1, batch_size=8,
        learning_rate=0.001, critic_learning_rate=0.001,
        adam_betas=(0.9, 0.999), adam_eps=1e-8, weight_decay=0,
        task_ce_weight=1, distillation_weight=0.25,
        bottleneck_kl_weight=0.001, local_sex_weight=0.05,
        pair_sex_weight=0.05,
    )
    return model, critics, data, hyper


def fit_one(base_model, base_critics, data, hyper, stage, target):
    model, critics, _ = paired_initial_copies(base_model, base_critics)
    runner = StageRunner(model, critics, data, stage_config(stage, hyper),
                         source_id="synthetic-source", role_id="OSF_DEFENSE_FIT")
    receipt = runner.run()
    runner.save_checkpoint(target)
    return runner.model, receipt


def test_local_models_assemble_without_transferring_other_purpose(tmp_path):
    base, critics, data, hyper = fixtures()
    income_stage, occupation_stage = stages_for_arm("B-LOCAL", 0, 1)
    income_path, occupation_path = tmp_path / "income.pt", tmp_path / "occupation.pt"
    income_model, income_receipt = fit_one(base, critics, data, hyper, income_stage, income_path)
    occupation_model, occupation_receipt = fit_one(base, critics, data, hyper, occupation_stage, occupation_path)
    combined, provenance = assemble_local_pair(
        base, income_model=income_model, occupation_model=occupation_model,
        income_receipt=income_receipt, occupation_receipt=occupation_receipt,
        income_checkpoint=income_path, occupation_checkpoint=occupation_path,
        steps_per_encoder=1,
    )
    assert provenance["income_stage_checkpoint_sha256"] == file_sha256(income_path)
    assert provenance["occupation_stage_checkpoint_sha256"] == file_sha256(occupation_path)
    for purpose, own in (("income", income_model), ("occupation", occupation_model)):
        for name, value in own.encoders[purpose].state_dict().items():
            assert torch.equal(combined.encoders[purpose].state_dict()[name], value)
    altered = copy.deepcopy(income_model)
    with torch.no_grad():
        altered.heads["occupation"].bias.add_(1)
    with pytest.raises(ValueError, match="other purpose"):
        assemble_local_pair(
            base, income_model=altered, occupation_model=occupation_model,
            income_receipt=income_receipt, occupation_receipt=occupation_receipt,
            income_checkpoint=income_path, occupation_checkpoint=occupation_path,
            steps_per_encoder=1,
        )


def test_sequential_second_stage_requires_actual_first_checkpoint(tmp_path):
    base, critics, data, hyper = fixtures()
    first, second = stages_for_arm("B-SEQ12", 0, 1)
    checkpoint = tmp_path / "first.pt"
    first_model, first_receipt = fit_one(base, critics, data, hyper, first, checkpoint)
    frozen = FrozenPurposeView.from_model(
        first_model, data, purpose="income", source_id="synthetic-source",
        first_checkpoint_sha256=file_sha256(checkpoint),
        sampler_plan_sha256="a" * 64, sampler_secret=b"s" * 32,
    )
    identity = verify_sequential_first_artifacts(
        checkpoint_path=checkpoint, first_model=first_model,
        first_receipt=first_receipt, fixed_first_view=frozen,
        fit_data=data, expected_sampler_plan_sha256="a" * 64,
        expected_first_purpose="income", expected_steps=1,
        expected_stage_id=first.stage_id,
        expected_config_sha256=stage_config(first, hyper).sha256,
        expected_source_id="synthetic-source", expected_role_id="OSF_DEFENSE_FIT",
    )
    stage2 = stage_config(second, hyper, first_stage_receipt=identity,
                          recomputed_first_stage_hashes=identity)
    assert stage2.frozen_view_sha256 == frozen.sha256
    changed_model = copy.deepcopy(first_model)
    with torch.no_grad():
        changed_model.heads["income"].bias.add_(1)
    with pytest.raises(ValueError, match="Supplied trained model|different trained model"):
        verify_sequential_first_artifacts(
            checkpoint_path=checkpoint, first_model=changed_model,
            first_receipt=first_receipt, fixed_first_view=frozen,
            fit_data=data, expected_sampler_plan_sha256="a" * 64,
            expected_first_purpose="income", expected_steps=1,
            expected_stage_id=first.stage_id,
            expected_config_sha256=stage_config(first, hyper).sha256,
            expected_source_id="synthetic-source", expected_role_id="OSF_DEFENSE_FIT",
        )
    partial = dict(first_receipt)
    partial["status"] = "RUNNING"
    with pytest.raises(ValueError, match="not a complete"):
        verify_sequential_first_artifacts(
            checkpoint_path=checkpoint, first_model=first_model,
            first_receipt=partial, fixed_first_view=frozen,
            fit_data=data, expected_sampler_plan_sha256="a" * 64,
            expected_first_purpose="income", expected_steps=1,
            expected_stage_id=first.stage_id,
            expected_config_sha256=stage_config(first, hyper).sha256,
            expected_source_id="synthetic-source", expected_role_id="OSF_DEFENSE_FIT",
        )
    with pytest.raises(ValueError, match="registered sequential"):
        verify_sequential_first_artifacts(
            checkpoint_path=checkpoint, first_model=first_model,
            first_receipt=first_receipt, fixed_first_view=frozen,
            fit_data=data, expected_sampler_plan_sha256="a" * 64,
            expected_first_purpose="income", expected_steps=1,
            expected_stage_id="B-LOCAL/s0/income",
            expected_config_sha256=stage_config(first, hyper).sha256,
            expected_source_id="synthetic-source", expected_role_id="OSF_DEFENSE_FIT",
        )
