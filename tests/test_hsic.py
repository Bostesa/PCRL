import torch

from pcrl.training.independence.hsic import gaussian_kernel, hsic


def test_hsic_independent_returns_near_zero():
    """X and Z drawn independently; HSIC should be near 0."""
    torch.manual_seed(0)
    X = torch.randn(500, 16)
    Z = torch.randn(500)
    h = hsic(X, Z)
    assert h.item() < 0.05, f"Independent HSIC too high: {h.item()}"


def test_hsic_dependent_returns_positive():
    """Z = sign of first coordinate; HSIC should be > 0.05."""
    torch.manual_seed(0)
    X = torch.randn(500, 16)
    Z = torch.sign(X[:, 0]) + 0.1 * torch.randn(500)
    h = hsic(X, Z)
    assert h.item() > 0.05, f"Dependent HSIC too low: {h.item()}"


def test_hsic_categorical_z():
    """Z as integer class indices; uses delta kernel path."""
    torch.manual_seed(0)
    X = torch.randn(200, 16)
    Z = torch.randint(0, 4, (200,))
    h = hsic(X, Z)
    assert h.dim() == 0
    assert torch.isfinite(h)


def test_hsic_gradient_flows():
    """Verify gradient flows through HSIC for backprop."""
    X = torch.randn(100, 8, requires_grad=True)
    Z = torch.randn(100)
    h = hsic(X, Z)
    h.backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_hsic_handles_small_batch():
    """Batch size 2 should not crash."""
    X = torch.randn(2, 4)
    Z = torch.tensor([0, 1])
    h = hsic(X, Z)
    assert torch.isfinite(h)


def test_hsic_median_heuristic():
    """Verify sigma is set when None and produces valid kernel."""
    X = torch.randn(100, 8)
    K_auto = gaussian_kernel(X, sigma=None)
    K_fixed = gaussian_kernel(X, sigma=1.0)
    assert (K_auto >= 0).all() and (K_auto <= 1).all()
    assert (K_fixed >= 0).all() and (K_fixed <= 1).all()
    assert torch.allclose(torch.diag(K_auto), torch.ones(100), atol=1e-5)
