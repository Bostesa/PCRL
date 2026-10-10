"""Bounded central-product fit on a fixed bank and admitted protected FIT rows."""

import time

import numpy as np

from fit import (FitRecipe, ProtectedData, TASK_CAP_KEYS, prepare_law,
                 exact_metrics, _identity_guards, _torch_metrics)
from factor_controls import FactorGate, ProductPolicy, initial_factor_gate


def fit_product(data: ProtectedData, policy: ProductPolicy, *,
                caps: dict[str, float], recipe: FitRecipe) -> dict:
    """Two separately initialized product fits; exact deployed law selects.

    This is a heuristic matched control. The caller must bind the source FIT
    row order, bank/action package and common CPU ceiling before invocation.
    """
    if (type(data) is not ProtectedData or type(policy) is not ProductPolicy or
            type(recipe) is not FitRecipe or type(caps) is not dict or
            caps.keys() != TASK_CAP_KEYS or
            any(not np.isfinite(v) for v in caps.values())):
        raise ValueError("central product needs protected rows and exact four caps")
    clock = time.process_time()
    import torch
    torch.set_num_threads(1)
    prepared = prepare_law(data.task, policy)
    guards = _identity_guards(prepared, data.sex)
    limits = {**caps, "r1_bayes": guards["r1_bayes"],
              "r2_bayes": guards["r2_bayes"]}
    feature = torch.as_tensor(prepared.features.copy())
    valid1 = np.zeros((len(feature), 4), dtype=bool)
    valid2 = np.zeros_like(valid1)
    for row in range(len(feature)):
        valid1[row, :len(policy.actions1[int(data.task.cell1[row])])] = True
        valid2[row, :len(policy.actions2[int(data.task.cell2[row])])] = True
    mask1, mask2 = torch.as_tensor(valid1), torch.as_tensor(valid2)
    starts = []
    best = None
    interrupted = False
    for start_index, start_name in enumerate(("identity_biased", "uniform_logit")):
        if time.process_time() - clock >= recipe.cpu_seconds:
            interrupted = True
            break
        rng = np.random.default_rng(recipe.seed + start_index)
        initial = []
        for _ in (1, 2):
            factor = initial_factor_gate(feature.shape[1], rng,
                identity_biased=start_index == 0,
                init_scale=recipe.init_scale,
                identity_bias=recipe.identity_logit_bias)
            initial.extend((factor.w1, factor.b1, factor.w2, factor.b2))
        weights = [torch.nn.Parameter(torch.as_tensor(v.copy(), dtype=torch.float64))
                   for v in initial]
        optimizer = torch.optim.Adam(weights, lr=recipe.learning_rate)
        dual = {key: 0.0 for key in limits}
        receipt = {"start": start_name, "checkpoints": [], "updates": 0,
                   "completed": False}

        def frozen_policy():
            arrays = [value.detach().numpy().copy() for value in weights]
            f1 = FactorGate(*arrays[:4])
            f2 = FactorGate(*arrays[4:])
            return ProductPolicy(policy.bank1, policy.bank2, policy.actions1,
                                 policy.actions2, policy.gate, f1, f2)

        def checkpoint(update):
            nonlocal best
            fitted = frozen_policy()
            metrics = exact_metrics(data, fitted)
            residual = {key: metrics[key] - limits[key] for key in limits}
            feasible = all(metrics[key] <= limits[key] for key in limits)
            receipt["checkpoints"].append({"update": update,
                                            "metrics": metrics,
                                            "residual": residual,
                                            "feasible": feasible})
            rank = (metrics["protected_objective"], start_index, update)
            if feasible and (best is None or rank < best[0]):
                best = (rank, fitted, metrics)

        checkpoint(0)
        for update in range(1, recipe.updates + 1):
            if time.process_time() - clock >= recipe.cpu_seconds:
                interrupted = True
                break
            optimizer.zero_grad()
            logits1 = (torch.tanh(feature @ weights[0] + weights[1]) @
                       weights[2] + weights[3]).masked_fill(~mask1, -torch.inf)
            logits2 = (torch.tanh(feature @ weights[4] + weights[5]) @
                       weights[6] + weights[7]).masked_fill(~mask2, -torch.inf)
            q1 = torch.softmax(logits1, dim=1)
            q2 = torch.softmax(logits2, dim=1)
            q = (q1[:, :, None] * q2[:, None, :]).reshape(-1, 16)
            values = _torch_metrics(prepared, q, sex=data.sex, torch=torch)
            penalty = values["objective"]
            for key in limits:
                residual = values[key] - limits[key]
                positive = torch.clamp(residual, min=0)
                penalty = penalty + dual[key] * residual + \
                    .5 * recipe.rho * positive.square()
            penalty.backward()
            torch.nn.utils.clip_grad_norm_(weights, recipe.gradient_norm)
            optimizer.step()
            receipt["updates"] = update
            if update % recipe.dual_every == 0:
                for key in limits:
                    dual[key] = max(0, dual[key] + recipe.rho *
                                    (float(values[key].detach()) - limits[key]))
            if update % recipe.checkpoint_every == 0 or update == recipe.updates:
                checkpoint(update)
        receipt["completed"] = not interrupted
        starts.append(receipt)
        if interrupted:
            break
    complete = not interrupted and len(starts) == 2 and all(s["completed"] for s in starts)
    return {"complete": complete, "success": bool(complete and best is not None),
            "starts": starts, "policy": best[1] if complete and best else None,
            "metrics": best[2] if complete and best else None,
            "local_identity_guards": guards,
            "diagnostic_best": None if best is None else best[0],
            "elapsed_cpu_seconds": time.process_time() - clock}
