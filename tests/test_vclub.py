import torch

from pcrl.training.independence.vclub import VCLUB


def test_vclub_independent_bound_low():
    """Independent X and Z, well-trained q, bound should be near 0."""
    torch.manual_seed(0)
    X = torch.randn(500, 16)
    Z = torch.randn(500, 4)

    vclub = VCLUB(x_dim=16, z_dim=4, z_categorical=False)
    opt = torch.optim.Adam(vclub.parameters(), lr=1e-3)

    for _ in range(200):
        opt.zero_grad()
        loss = vclub.learning_loss(X, Z)
        loss.backward()
        opt.step()

    bound = vclub.mi_upper_bound(X, Z).item()
    assert bound < 0.5, f"Independent bound too high: {bound}"


def test_vclub_dependent_bound_high():
    """Z = X projection, well-trained q, bound should be substantially > 0."""
    torch.manual_seed(0)
    X = torch.randn(500, 16)
    W = torch.randn(16, 4)
    Z = X @ W + 0.1 * torch.randn(500, 4)

    vclub = VCLUB(x_dim=16, z_dim=4, z_categorical=False)
    opt = torch.optim.Adam(vclub.parameters(), lr=1e-3)

    for _ in range(500):
        opt.zero_grad()
        loss = vclub.learning_loss(X, Z)
        loss.backward()
        opt.step()

    bound = vclub.mi_upper_bound(X, Z).item()
    assert bound > 0.5, f"Dependent bound too low: {bound}"


def test_vclub_categorical_z():
    """Works with class index Z (categorical mode)."""
    torch.manual_seed(0)
    X = torch.randn(300, 16)
    Z = torch.randint(0, 5, (300,))

    vclub = VCLUB(x_dim=16, z_dim=5, z_categorical=True)
    opt = torch.optim.Adam(vclub.parameters(), lr=1e-3)

    for _ in range(100):
        opt.zero_grad()
        loss = vclub.learning_loss(X, Z)
        loss.backward()
        opt.step()

    bound = vclub.mi_upper_bound(X, Z)
    assert bound.dim() == 0
    assert torch.isfinite(bound)


def test_vclub_gradient_flows():
    """Gradient flows through bound for encoder optimization."""
    X = torch.randn(100, 8, requires_grad=True)
    Z = torch.randn(100, 4)
    vclub = VCLUB(x_dim=8, z_dim=4, z_categorical=False)
    bound = vclub.mi_upper_bound(X, Z)
    bound.backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_vclub_q_training_converges():
    """Training q for 200 steps, learning_loss should decrease."""
    torch.manual_seed(0)
    X = torch.randn(300, 16)
    Z = torch.randn(300, 4)
    vclub = VCLUB(x_dim=16, z_dim=4, z_categorical=False)
    opt = torch.optim.Adam(vclub.parameters(), lr=1e-3)

    losses = []
    for _ in range(200):
        opt.zero_grad()
        loss = vclub.learning_loss(X, Z)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
    )
