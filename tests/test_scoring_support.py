"""Regression checks for undefined R², class coverage and fail-closed privacy."""

import math

import numpy as np
import pytest
import torch

from pcrl.evaluation.certificates import compute_dominant_axis_r2, compute_mlp_ovr_delta
from pcrl.purposes.verification import LinearComplianceCertificate, NullSpaceCertificate
from pcrl.training.losses import VerificationRegularizer
from pcrl.training.proxy_lagrangian import Constraint, ProxyLagrangianOptimizer


def test_fixed_schema_absent_middle_and_tail_classes_are_undefined():
    labels = torch.tensor([0, 2, 0, 2])
    features = labels.double()[:, None].requires_grad_()
    verifier = VerificationRegularizer(num_classes=4)
    result = verifier.per_class_statistics(features, labels)
    assert result.support.tolist() == [2, 0, 2, 0]
    assert result.valid_mask.tolist() == [True, False, True, False]
    assert result.variance.tolist() == [0.25, 0.0, 0.25, 0.0]
    assert torch.isnan(result.r_squared[[1, 3]]).all()
    assert torch.all(result.r_squared[[0, 2]] > 0.99)
    assert torch.isnan(verifier(features, labels))
    result.r_squared[result.valid_mask].sum().backward()
    assert torch.isfinite(features.grad).all()


@pytest.mark.parametrize("label", [0, 2])
def test_constant_class_cannot_pass_any_linear_score(label):
    features = np.random.default_rng(9).normal(size=(12, 3))
    labels = np.full(12, label, dtype=np.int64)
    verifier = VerificationRegularizer(num_classes=3)
    assert torch.isnan(verifier(torch.tensor(features), torch.tensor(labels)))
    assert torch.isnan(verifier.forward_per_class(torch.tensor(features), torch.tensor(labels))).all()
    for audit in (LinearComplianceCertificate(), NullSpaceCertificate()):
        result = audit.check(features, labels, num_classes=3)
        assert math.isnan(result.r_squared)
        assert not result.certified
        assert not result.coverage_complete
        assert not result.score_defined
    dominant = compute_dominant_axis_r2(features, labels, num_classes=3)
    assert math.isnan(dominant["r2_da"])
    assert dominant["argmax_class"] == -1
    assert dominant["valid_mask"] == [False, False, False]


def test_partial_schema_fails_aggregate_and_dominant_axis_with_coverage():
    labels = np.array([0, 2] * 10)
    features = np.zeros((20, 2))
    result = LinearComplianceCertificate().check(features, labels, num_classes=4)
    dominant = compute_dominant_axis_r2(features, labels, num_classes=4)
    assert result.class_support == [10, 0, 10, 0]
    assert not result.certified
    assert math.isnan(dominant["r2_da"])
    assert dominant["observed_r2_da"] == pytest.approx(0.0)
    assert dominant["class_support"] == result.class_support
    assert dominant["valid_mask"] == [True, False, True, False]


def test_proxy_skips_undefined_updates_and_rejects_missing_evidence():
    constraint = Constraint("private", threshold=0.05, lambda_init=2.0)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    proxy = ProxyLagrangianOptimizer(torch.optim.SGD([parameter], lr=0.1), [constraint])
    for value in (float("nan"), float("inf"), -float("inf")):
        proxy.dual_step({"private": value})
        assert constraint.lambda_value == 2.0
        assert not proxy.all_satisfied({"private": value})
    assert not proxy.all_satisfied({})
    assert proxy.all_satisfied({"private": 0.0})
    loss = proxy.lagrangian_loss(parameter.square(), {"private": torch.tensor(float("nan"))})
    loss.backward()
    assert parameter.grad.item() == pytest.approx(2.0)


def test_float_continuous_target_is_not_truncated_into_classes():
    target = np.array([-0.9, -0.3, 0.2, 1.0])
    result = LinearComplianceCertificate().check(target[:, None], target)
    assert result.r_squared > 0.999
    assert result.class_support is None


def test_zero_predictor_does_not_remove_arbitrary_null_space_directions():
    features = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    labels = np.array([0, 0, 1, 1])
    result = NullSpaceCertificate().check(features, labels, num_classes=2)
    assert result.r_squared == pytest.approx(0.0)
    assert result.variance_preserved == pytest.approx(1.0)


def test_bios_unsupported_slices_do_not_drive_duals():
    from pcrl.language.per_class_ovr_constraints import build_phase2_constraints, evaluate_constraints

    features = torch.randn(8, 3)
    labels = torch.zeros(8, dtype=torch.long)
    occupation = torch.zeros(8, dtype=torch.long)
    result = evaluate_constraints(features, labels, occupation, VerificationRegularizer())
    assert not result.differentiable
    assert all(math.isnan(value) for value in result.scalars.values())
    assert not any(result.valid_mask.values())
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    proxy = ProxyLagrangianOptimizer(torch.optim.SGD([parameter], lr=0.1), build_phase2_constraints())
    previous = {name: c.lambda_value for name, c in proxy.constraints.items()}
    proxy.dual_step(result.scalars)
    assert {name: c.lambda_value for name, c in proxy.constraints.items()} == previous
    assert not proxy.all_satisfied(result.scalars)


def test_ema_controller_undefined_variance_does_not_move_lambda():
    from pcrl.language.dual_controllers import EmaCrossCovPIController

    controller = EmaCrossCovPIController(d=3, lambda_init=8.0)
    original = controller.lam
    value, r2 = controller.step()
    assert value == original
    assert math.isnan(r2)


@pytest.mark.parametrize("missing_split", ["train", "test"])
def test_mlp_ovr_missing_schema_class_is_undefined_in_either_split(missing_split):
    full = np.array([0, 1, 2, 0, 1, 2])
    partial = np.array([0, 1, 0, 1, 0, 1])
    train = partial if missing_split == "train" else full
    test = partial if missing_split == "test" else full
    features = np.zeros((6, 2), dtype=np.float32)
    result = compute_mlp_ovr_delta(
        features, train, features, test, num_classes=4, hidden=2, epochs=0,
    )
    assert result["valid_mask"] == [True, True, False, False]
    assert not result["coverage_complete"]
    assert math.isnan(result["mlp_da_delta"])
    assert math.isfinite(result["observed_mlp_da_delta"])
    for key in ("per_class_delta", "per_class_acc", "per_class_baseline"):
        assert math.isnan(result[key][2])
        assert math.isnan(result[key][3])
    assert result[f"{missing_split}_class_support"] == [3, 3, 0, 0]


def test_mlp_majority_predictor_is_fit_on_training_labels_only():
    features = np.zeros((10, 2), dtype=np.float32)
    train = np.array([1] * 8 + [0] * 2)
    test = np.array([1] * 2 + [0] * 8)
    result = compute_mlp_ovr_delta(
        features, train, features, test, num_classes=2, hidden=2, epochs=0,
    )
    assert result["majority_prediction"] == [0, 1]
    assert result["per_class_baseline"] == pytest.approx([0.2, 0.2])
    assert result["coverage_complete"]


def test_mlp_all_unsupported_classes_have_no_observed_maximum():
    features = np.zeros((4, 2), dtype=np.float32)
    labels = np.zeros(4, dtype=np.int64)
    result = compute_mlp_ovr_delta(
        features, labels, features, labels, num_classes=2, hidden=2, epochs=0,
    )
    assert result["argmax_class"] == -1
    assert math.isnan(result["mlp_da_delta"])
    assert math.isnan(result["observed_mlp_da_delta"])
