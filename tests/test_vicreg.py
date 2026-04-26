import torch

from pcrl.training.independence.vicreg import (
    covariance_loss,
    variance_loss,
    vicreg_loss,
)


def test_variance_loss_collapsed_high():
    """X all zeros (collapse), variance_loss should equal gamma."""
    X = torch.zeros(100, 16)
    loss = variance_loss(X, gamma=1.0)
    assert loss.item() > 0.99, (
        f"Collapsed variance_loss should be ~1.0, got {loss.item()}"
    )


def test_variance_loss_healthy_zero():
    """X with std > 1 per dim, variance_loss should be 0."""
    X = torch.randn(100, 16) * 2.0
    loss = variance_loss(X, gamma=1.0)
    assert loss.item() < 0.01, (
        f"Healthy variance_loss should be ~0, got {loss.item()}"
    )


def test_covariance_loss_decorrelated_zero():
    """X with identity covariance, covariance_loss should be near 0."""
    torch.manual_seed(0)
    X = torch.randn(1000, 16)
    loss = covariance_loss(X)
    assert loss.item() < 0.1, (
        f"Decorrelated covariance_loss should be ~0, got {loss.item()}"
    )


def test_covariance_loss_correlated_high():
    """X with strong correlations, covariance_loss should be > 0.1."""
    torch.manual_seed(0)
    base = torch.randn(500, 4)
    X = torch.cat(
        [
            base,
            base + 0.01 * torch.randn(500, 4),
            base + 0.01 * torch.randn(500, 4),
            base + 0.01 * torch.randn(500, 4),
        ],
        dim=1,
    )
    loss = covariance_loss(X)
    assert loss.item() > 0.1, (
        f"Correlated covariance_loss should be > 0.1, got {loss.item()}"
    )


def test_vicreg_gradient_flows():
    """Gradient flows for backprop."""
    X = torch.randn(100, 8, requires_grad=True)
    loss = vicreg_loss(X)
    loss.backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
