"""Proxy-Lagrangian constraint optimizer.

Replaces fixed-lambda Lagrangian with adaptive dual variables. Each
constraint has its own lambda that increases when the constraint is
violated and decreases when it is satisfied (with margin), projected
into [0, lambda_max].

This addresses the Wei & Niethammer 2022 negative result on linear
scalarization of non-convex Pareto frontiers: a single fixed lambda
cannot recover Pareto-optimal points whose tangent does not match the
chosen scalarization direction. Adapting lambda based on observed
constraint satisfaction sidesteps the issue.

Refs:
    Cotter et al., JMLR 2019, "Optimization with Non-Differentiable
        Constraints with Applications to Fairness, Recall, Churn, and
        Other Goals."
    Wei & Niethammer 2022, "The Fairness-Accuracy Pareto Front."
"""

from typing import Dict, List

import torch


class Constraint:
    """A single constraint of the form value <= threshold or value >= threshold.

    The Lagrangian contribution is `lambda * (value - threshold)` for `<=`
    constraints and `lambda * (threshold - value)` for `>=` constraints.
    Lambda is updated by gradient ascent on the dual:
        lambda <- clip(lambda + eta_lambda * violation, 0, lambda_max)

    Args:
        name: human-readable constraint name (e.g., "hsic_purpose_0").
        threshold: constraint threshold.
        direction: '<=' for upper-bound (e.g., HSIC <= eps),
                   '>=' for lower-bound (e.g., task_acc >= floor).
        eta_lambda: dual variable learning rate.
        lambda_init: initial lambda value.
        lambda_max: cap on lambda to prevent runaway growth.
        lambda_min: floor for lambda. Default 0.0. Set positive to prevent
            dual relaxation during temporary feasible windows; addresses
            the LATE-DRIFT / STARVATION mode observed in Round 4 where
            lambda decayed to <1 during a feasible interval and then
            could not catch a re-rising R² before training ended.
    """

    def __init__(
        self,
        name: str,
        threshold: float,
        direction: str = "<=",
        eta_lambda: float = 0.01,
        lambda_init: float = 1.0,
        lambda_max: float = 100.0,
        lambda_min: float = 0.0,
    ):
        assert direction in ("<=", ">="), "direction must be '<=' or '>='"
        assert 0.0 <= lambda_min <= lambda_max, (
            f"require 0 <= lambda_min ({lambda_min}) <= lambda_max ({lambda_max})"
        )
        self.name = name
        self.threshold = threshold
        self.direction = direction
        self.eta_lambda = eta_lambda
        self.lambda_value = max(lambda_min, lambda_init)
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.value_history: List[float] = []
        self.lambda_history: List[float] = []
        self.violation_history: List[float] = []

    def violation(self, value: float) -> float:
        """Constraint violation amount; positive means violated."""
        if self.direction == "<=":
            return value - self.threshold
        return self.threshold - value

    def is_satisfied(self, value: float) -> bool:
        return self.violation(value) <= 0

    def lagrangian_term(self, value_tensor: torch.Tensor) -> torch.Tensor:
        """lambda * violation(value), differentiable in value_tensor.

        Lambda is treated as a constant during the primal step (it is a
        plain Python float, not a tensor with grad).
        """
        if self.direction == "<=":
            return self.lambda_value * (value_tensor - self.threshold)
        return self.lambda_value * (self.threshold - value_tensor)

    def update_lambda(self, value: float) -> None:
        """Dual ascent step: lambda += eta_lambda * violation, projected to
        ``[lambda_min, lambda_max]``."""
        violation = self.violation(value)
        self.lambda_value += self.eta_lambda * violation
        self.lambda_value = max(
            self.lambda_min, min(self.lambda_value, self.lambda_max)
        )
        self.value_history.append(value)
        self.lambda_history.append(self.lambda_value)
        self.violation_history.append(violation)


class ProxyLagrangianOptimizer:
    """Wraps a primal optimizer and a list of constraints.

    Per-step usage:

        total = opt.lagrangian_loss(base_loss, {"c1": value_tensor_1, ...})
        primal_opt.zero_grad()
        total.backward()
        primal_opt.step()
        opt.dual_step({"c1": value_scalar_1, ...})

    Args:
        primal_optimizer: torch.optim.Optimizer for the model parameters.
        constraints: list of `Constraint` instances. Names must be unique.
    """

    def __init__(
        self,
        primal_optimizer: torch.optim.Optimizer,
        constraints: List[Constraint],
    ):
        self.primal_optimizer = primal_optimizer
        self.constraints: Dict[str, Constraint] = {c.name: c for c in constraints}
        assert len(self.constraints) == len(constraints), (
            "Constraint names must be unique"
        )

    def lagrangian_loss(
        self,
        base_loss: torch.Tensor,
        constraint_values: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return base_loss + sum_c lambda_c * violation_c(value_c).

        constraint_values maps constraint name -> differentiable value tensor.
        """
        total = base_loss
        for name, value_tensor in constraint_values.items():
            assert name in self.constraints, f"Unknown constraint: {name}"
            total = total + self.constraints[name].lagrangian_term(value_tensor)
        return total

    def primal_step(self) -> None:
        """Step the primal optimizer; assumes loss has been backwarded."""
        self.primal_optimizer.step()
        self.primal_optimizer.zero_grad()

    def dual_step(self, constraint_values: Dict[str, float]) -> None:
        """Update each constraint's lambda using observed (scalar) values."""
        for name, value in constraint_values.items():
            assert name in self.constraints, f"Unknown constraint: {name}"
            self.constraints[name].update_lambda(value)

    def all_satisfied(self, constraint_values: Dict[str, float]) -> bool:
        """True iff every constraint is currently satisfied."""
        return all(
            self.constraints[name].is_satisfied(value)
            for name, value in constraint_values.items()
        )

    def diagnostics(self) -> Dict[str, Dict[str, object]]:
        """Per-constraint diagnostic snapshot for logging."""
        out: Dict[str, Dict[str, object]] = {}
        for name, c in self.constraints.items():
            last_value = c.value_history[-1] if c.value_history else None
            last_viol = c.violation_history[-1] if c.violation_history else None
            satisfied = c.is_satisfied(last_value) if last_value is not None else None
            out[name] = {
                "lambda": c.lambda_value,
                "current_value": last_value,
                "current_violation": last_viol,
                "satisfied": satisfied,
            }
        return out
