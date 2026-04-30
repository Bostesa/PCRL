import pytest
import torch

from pcrl.training.proxy_lagrangian import Constraint, ProxyLagrangianOptimizer


def test_constraint_violation_upper_bound():
    """For <= constraint, violation is value - threshold."""
    c = Constraint("test", threshold=1.0, direction="<=")
    assert c.violation(2.0) == 1.0
    assert c.violation(0.5) == -0.5
    assert c.violation(1.0) == 0.0


def test_constraint_violation_lower_bound():
    """For >= constraint, violation is threshold - value."""
    c = Constraint("test", threshold=0.5, direction=">=")
    assert c.violation(0.3) == pytest.approx(0.2)
    assert c.violation(0.7) == pytest.approx(-0.2)


def test_constraint_update_lambda_violated():
    """When violated, lambda increases."""
    c = Constraint("test", threshold=1.0, direction="<=", eta_lambda=0.1, lambda_init=1.0)
    c.update_lambda(2.0)
    assert c.lambda_value > 1.0
    assert c.lambda_value == pytest.approx(1.1)


def test_constraint_update_lambda_satisfied():
    """When satisfied with margin, lambda decreases."""
    c = Constraint("test", threshold=1.0, direction="<=", eta_lambda=0.1, lambda_init=1.0)
    c.update_lambda(0.5)
    assert c.lambda_value < 1.0
    assert c.lambda_value == pytest.approx(0.95)


def test_constraint_lambda_clipped_nonnegative():
    """Lambda is clipped at 0."""
    c = Constraint("test", threshold=10.0, direction="<=", eta_lambda=1.0, lambda_init=0.5)
    c.update_lambda(0.0)
    assert c.lambda_value == 0.0


def test_constraint_lambda_clipped_at_max():
    """Lambda is capped at lambda_max."""
    c = Constraint(
        "test", threshold=0.0, direction="<=", eta_lambda=10.0,
        lambda_init=1.0, lambda_max=5.0,
    )
    c.update_lambda(100.0)
    assert c.lambda_value == 5.0


def test_constraint_lambda_min_floor_blocks_descent():
    """Fix R1: with lambda_min > 0, dual descent is floored.

    Even when the constraint is satisfied with margin (which would push
    lambda toward 0), the projected lambda cannot fall below lambda_min.
    """
    c = Constraint(
        "test", threshold=1.0, direction="<=", eta_lambda=1.0,
        lambda_init=10.0, lambda_max=100.0, lambda_min=5.0,
    )
    # value=0.0 means violation = -1.0; descent of 1.0 would put lambda at 9.0
    c.update_lambda(0.0)
    assert c.lambda_value == 9.0
    # Several big descents in a row — eventually clamped to lambda_min, not 0.
    for _ in range(20):
        c.update_lambda(0.0)
    assert c.lambda_value == pytest.approx(5.0)


def test_constraint_lambda_min_does_not_block_ascent():
    """Lambda floor is one-sided: ascent from anywhere still works."""
    c = Constraint(
        "test", threshold=1.0, direction="<=", eta_lambda=0.5,
        lambda_init=5.0, lambda_max=100.0, lambda_min=5.0,
    )
    c.update_lambda(3.0)  # violation = 2.0, ascent of 1.0
    assert c.lambda_value == pytest.approx(6.0)


def test_constraint_lambda_init_promoted_to_floor():
    """If lambda_init < lambda_min, lambda starts at lambda_min."""
    c = Constraint(
        "test", threshold=1.0, direction="<=", eta_lambda=0.1,
        lambda_init=1.0, lambda_max=100.0, lambda_min=5.0,
    )
    assert c.lambda_value == 5.0


def test_constraint_lambda_min_invalid():
    """lambda_min must be in [0, lambda_max]."""
    with pytest.raises(AssertionError):
        Constraint(
            "test", threshold=1.0, direction="<=",
            lambda_max=5.0, lambda_min=10.0,
        )
    with pytest.raises(AssertionError):
        Constraint(
            "test", threshold=1.0, direction="<=",
            lambda_max=5.0, lambda_min=-1.0,
        )


def test_constraint_lagrangian_term_differentiable():
    """Lagrangian term is differentiable w.r.t. value_tensor."""
    c = Constraint("test", threshold=1.0, direction="<=", lambda_init=2.0)
    value = torch.tensor(1.5, requires_grad=True)
    term = c.lagrangian_term(value)
    term.backward()
    assert value.grad is not None
    assert value.grad.item() == pytest.approx(2.0)


def test_optimizer_lagrangian_loss_combines_terms():
    """lagrangian_loss combines base_loss and all constraint terms."""
    constraints = [
        Constraint("c1", threshold=1.0, direction="<=", lambda_init=2.0),
        Constraint("c2", threshold=0.5, direction=">=", lambda_init=3.0),
    ]
    fake_params = [torch.nn.Parameter(torch.zeros(1))]
    primal_opt = torch.optim.SGD(fake_params, lr=0.1)
    opt = ProxyLagrangianOptimizer(primal_opt, constraints)

    base = torch.tensor(5.0, requires_grad=True)
    cv = {
        "c1": torch.tensor(2.0, requires_grad=True),
        "c2": torch.tensor(0.3, requires_grad=True),
    }
    total = opt.lagrangian_loss(base, cv)
    assert total.item() == pytest.approx(7.6)


def test_optimizer_dual_step_updates_lambdas():
    """dual_step updates each constraint's lambda."""
    constraints = [
        Constraint("c1", threshold=1.0, direction="<=", eta_lambda=0.1, lambda_init=1.0),
        Constraint("c2", threshold=0.5, direction=">=", eta_lambda=0.1, lambda_init=1.0),
    ]
    fake_params = [torch.nn.Parameter(torch.zeros(1))]
    primal_opt = torch.optim.SGD(fake_params, lr=0.1)
    opt = ProxyLagrangianOptimizer(primal_opt, constraints)

    opt.dual_step({"c1": 2.0, "c2": 0.3})

    assert opt.constraints["c1"].lambda_value == pytest.approx(1.1)
    assert opt.constraints["c2"].lambda_value == pytest.approx(1.02)


def test_optimizer_all_satisfied():
    """all_satisfied returns True only when all constraints are satisfied."""
    constraints = [
        Constraint("c1", threshold=1.0, direction="<="),
        Constraint("c2", threshold=0.5, direction=">="),
    ]
    fake_params = [torch.nn.Parameter(torch.zeros(1))]
    opt = ProxyLagrangianOptimizer(torch.optim.SGD(fake_params, lr=0.1), constraints)

    assert opt.all_satisfied({"c1": 0.5, "c2": 0.7}) is True
    assert opt.all_satisfied({"c1": 2.0, "c2": 0.7}) is False
    assert opt.all_satisfied({"c1": 0.5, "c2": 0.3}) is False


def test_proxy_lagrangian_simple_convergence():
    """Minimize x^2 s.t. x >= 0.5; expect x to settle near 0.5."""
    torch.manual_seed(0)
    x = torch.nn.Parameter(torch.tensor(2.0))

    primal_opt = torch.optim.SGD([x], lr=0.05)
    constraint = Constraint(
        "c1", threshold=0.5, direction=">=", eta_lambda=0.1, lambda_init=0.1
    )
    opt = ProxyLagrangianOptimizer(primal_opt, [constraint])

    for _ in range(200):
        primal_opt.zero_grad()
        base_loss = x ** 2
        constraint_values = {"c1": x}
        loss = opt.lagrangian_loss(base_loss, constraint_values)
        loss.backward()
        primal_opt.step()
        opt.dual_step({"c1": x.item()})

    assert 0.3 < x.item() < 0.8, f"x did not converge near 0.5: x={x.item()}"
