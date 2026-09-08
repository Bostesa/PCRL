"""Regression for the retired universal classification-accuracy guarantee."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pcrl.evaluation.certificates import (
    ComplianceReport,
    EmpiricalAudit,
    generate_report,
    print_compliance_table,
)
from pcrl.purposes.verification import (
    LinearComplianceCertificate,
    NonlinearComplianceCertificate,
    certified_accuracy_bound,
)


def test_exact_zero_covariance_counterexample_has_90_percent_threshold_accuracy():
    """Equal conditional means do not stop an affine threshold classifier."""
    h = np.array([1.0] * 9 + [-9.0] + [-1.0] * 9 + [9.0])[:, None]
    labels = np.array([1] * 10 + [0] * 10)
    assert h[labels == 1].mean() == 0.0
    assert h[labels == 0].mean() == 0.0
    assert (h[:, 0] @ (labels - labels.mean())) == 0.0

    # Fit a genuine affine least-squares predictor (intercept included).
    design = np.column_stack([np.ones(len(h)), h])
    beta = np.linalg.lstsq(design, labels, rcond=None)[0]
    prediction = design @ beta
    r_squared = 1 - np.sum((labels - prediction) ** 2) / np.sum(
        (labels - labels.mean()) ** 2
    )
    assert beta[0] == pytest.approx(0.5)
    assert beta[1] == pytest.approx(0.0, abs=1e-15)
    assert r_squared == pytest.approx(0.0, abs=1e-14)
    assert np.mean((h[:, 0] > 0).astype(int) == labels) == 0.9
    assert np.max(np.bincount(labels)) / len(labels) == 0.5

    result = LinearComplianceCertificate(epsilon=0.01).check(h, labels)
    assert result.r_squared == pytest.approx(0.0, abs=1e-14)
    assert result.certified  # Passes only the empirical least-squares criterion.


@pytest.mark.parametrize("r_squared", [0.0, 0.01, 1.0])
def test_accuracy_bound_explicitly_disabled(r_squared):
    with pytest.raises(NotImplementedError, match="retired"):
        certified_accuracy_bound(r_squared, 0.5, 2)


def test_derived_nonlinear_guarantee_explicitly_disabled():
    with pytest.raises(NotImplementedError, match="retired"):
        NonlinearComplianceCertificate().check(np.zeros((4, 1)), np.array([0, 1, 0, 1]))


def test_report_prints_empirical_results_without_accuracy_bound(capsys):
    print_compliance_table([
        ComplianceReport(
            purpose_name="p1", attr_name="attribute", linear_r2=0.0,
            linear_certified=True, empirical_best_acc=0.9,
            class_support=[10, 10], valid_mask=[True, True], coverage_complete=True,
        )
    ])
    output = capsys.readouterr().out
    assert "90.0%" in output
    assert "retired as invalid" in output
    assert "NL Bound" not in output


def test_generate_report_retains_scores_without_retired_guarantees(monkeypatch):
    # Keep the integration check tiny while exercising the real extraction,
    # fixed-schema score, report-generation and support-coverage paths.
    class IdentityEncoder(torch.nn.Module):
        def forward(self, features, purpose_idx):
            return features

    monkeypatch.setattr(
        EmpiricalAudit, "audit", lambda *args: (0.5, {"fixture": {"accuracy": 0.5}}),
    )
    purpose = SimpleNamespace(
        name="p1", disallowed_attrs=["a"], disallowed_attr_dims={"a": 3},
    )
    registry = SimpleNamespace(purposes=[purpose])
    fitting_batches = [{
        "features": torch.tensor([[-1.0], [0.0], [1.0]]),
        "sensitive_attrs": {"a": torch.tensor([0, 1, 2])},
    }]
    test_batches = [{
        "features": torch.tensor([[-1.0], [1.0], [-2.0], [2.0]]),
        "sensitive_attrs": {"a": torch.tensor([0, 1, 0, 1])},
    }]
    [report] = generate_report(
        IdentityEncoder(), fitting_batches, test_batches, registry,
        compute_mlp_da=True, mlp_da_kwargs={"hidden": 2, "epochs": 0},
    )
    assert report.nonlinear_bound is None
    assert report.nonlinear_best_sigma is None
    assert report.accuracy_guarantee_status == "retired_invalid"
    assert report.num_classes == 3
    assert report.class_support == [2, 2, 0]
    assert report.valid_mask == [True, True, False]
    assert not report.coverage_complete
    assert not report.certified
    assert np.isnan(report.mlp_da_delta)
    assert report.mlp_da_coverage["train_class_support"] == [1, 1, 1]
    assert report.mlp_da_coverage["test_class_support"] == [2, 2, 0]
    assert report.mlp_da_coverage["valid_mask"] == [True, True, False]
    assert not report.mlp_da_coverage["coverage_complete"]
