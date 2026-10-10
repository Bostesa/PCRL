"""Checkpoint-retaining frozen-first copy with an exact B0 first route.

Stage 1 is checkpoint-frozen before one private HMAC draw per FIT unit. Stage 2
trains on those actual first public codes, but only exact full-law replay may
select a final checkpoint. Returned first codes are private and must not be
serialized into a public receipt. A real runner must supply an owner-only
cache path and externally bind FIT row/order and frozen source/action hashes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time

import numpy as np
from scipy.optimize import minimize_scalar

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.factor_controls import (
    FactorGate, SequentialPolicy, _factor_law, public_probability_embedding,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.fit import (
    FitRecipe, ProtectedData, TASK_CAP_KEYS, _identity_guards,
    _torch_metrics, exact_metrics, prepare_law,
)
from .r2_sequential import BoundarySequentialPolicy


CHECKPOINT_DOMAIN = b"PCRL-A-SEQ-FIRST-CHECKPOINT-v1\0"
DRAW_SCHEMA = "SCV1-seq-fit-first-v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _units_and_hash(data, unit_ids):
    task = data.task
    units = tuple(unit_ids)
    if len(units) != len(task.y1) or any(
        type(x) not in (str, int, np.int64) or (type(x) is str and not x) or
        (type(x) in (int, np.int64) and x < 0) for x in units
    ):
        raise ValueError("one stable FIT unit ID required per aligned row")
    seen = {}
    for row, unit in enumerate(units):
        key = (type(unit).__name__, str(unit))
        source = (task.score1[row].tobytes(), task.score2[row].tobytes(),
                  int(task.cell1[row]), int(task.cell2[row]),
                  int(task.y1[row]), int(task.y2[row]), int(data.sex[row]))
        if key in seen and seen[key] != source:
            raise ValueError("duplicate FIT unit has different scores, cells, labels or SEX")
        seen[key] = source
    digest = hashlib.sha256(_canonical([(type(x).__name__, str(x)) for x in units])).hexdigest()
    return units, digest


def _first_source(data, template):
    if template.order == "12":
        return ("r1", template.bank1, template.actions1,
                data.task.cell1, data.task.y1)
    return ("r2", template.bank2, template.actions2,
            data.task.cell2, data.task.y2)


def _first_arrays(data, template):
    name, bank, actions, cells, labels = _first_source(data, template)
    n = len(cells)
    valid = np.zeros((n, 4), dtype=bool)
    ids = np.zeros((n, 4), dtype=np.int64)
    nll = np.zeros((n, 4), dtype=np.float64)
    brier = np.zeros((n, 4), dtype=np.float64)
    for row, cell in enumerate(cells):
        for slot, code in enumerate(actions[int(cell)]):
            probability = bank.probability[code]
            target = np.eye(len(probability))[int(labels[row])]
            valid[row, slot], ids[row, slot] = True, int(code)
            nll[row, slot] = -np.log(max(probability[int(labels[row])], 1e-12))
            brier[row, slot] = np.sum((probability - target) ** 2)
    return name, bank, actions, cells, valid, ids, nll, brier


def _first_exact(data, prepared, gate, arrays):
    name, bank, actions, cells, _, ids, nll, brier = arrays
    q = np.zeros_like(nll)
    for row, cell in enumerate(cells):
        for code, mass in _factor_law(gate, prepared.features[row], actions[int(cell)]):
            q[row, actions[int(cell)].index(code)] = mass
    metrics = {f"{name}_nll": float(np.mean(np.sum(q * nll, axis=1))),
               f"{name}_brier": float(np.mean(np.sum(q * brier, axis=1)))}
    counts = np.zeros((2, len(bank.probability)), dtype=np.float64)
    for sensitive in (0, 1):
        selected = data.sex == sensitive
        np.add.at(counts[sensitive], ids[selected], q[selected] / len(q))
    metrics[f"{name}_bayes"] = float(np.maximum(counts[0], counts[1]).sum())
    return metrics


def _project_first(data, prepared, gate, arrays, limits):
    """Select a FIT-only exact-B0 boundary for one retained first gate."""
    name, bank, actions, cells, _, ids, nll, brier = arrays
    q1 = np.zeros_like(nll)
    q0 = np.zeros_like(nll); q0[:, 0] = 1.0
    for row, cell in enumerate(cells):
        for slot, (_, mass) in enumerate(_factor_law(
                gate, prepared.features[row], actions[int(cell)])):
            q1[row, slot] = mass
    upper = 1.0
    for key, cost in (("nll", nll), ("brier", brier)):
        baseline = float(np.mean(np.sum(q0*cost, axis=1)))
        full = float(np.mean(np.sum(q1*cost, axis=1)))
        delta = full-baseline
        if delta > 0:
            upper = min(upper, max(0.0, (limits[f"{name}_{key}"]-baseline)/delta))
    counts = []
    for q in (q0, q1):
        c = np.zeros((2, len(bank.probability)), dtype=np.float64)
        for sensitive in (0, 1):
            selected = data.sex == sensitive
            np.add.at(c[sensitive], ids[selected], q[selected]/len(q))
        counts.append(c)
    g = lambda eta: float(np.maximum(*((1-eta)*counts[0]+eta*counts[1])).sum())
    guard = g(0.0)
    if g(upper) > guard+1e-12:
        left, right = 0.0, upper
        for _ in range(60):
            mid = (left+right)/2
            if g(mid) <= guard+1e-12:
                left = mid
            else:
                right = mid
        upper = left
    candidates = [0.0, upper]
    if upper > 0:
        result = minimize_scalar(g, bounds=(0.0, upper), method="bounded",
                                 options={"xatol": 1e-10, "maxiter": 80})
        if result.success:
            candidates.append(float(result.x))
    for value, eta in sorted((g(eta), eta) for eta in candidates):
        q = (1-eta)*q0 + eta*q1
        costs = {f"{name}_nll": float(np.mean(np.sum(q*nll, axis=1))),
                 f"{name}_brier": float(np.mean(np.sum(q*brier, axis=1)))}
        if all(costs[k] <= limits[k]+1e-12 for k in costs) and value <= guard+1e-12:
            return {"eta": eta, "objective": value, "bayes": value,
                    "costs": costs, "identity_alias": eta == 0.0}
    raise RuntimeError("exact first-stage identity was excluded from feasible set")


def _stage1_hash(gate, template, source_sha):
    _, bank, actions, _, _ = _first_source_dummy(template)
    digest = hashlib.sha256(CHECKPOINT_DOMAIN + bytes.fromhex(source_sha) +
                            template.order.encode("ascii"))
    digest.update(_canonical(actions))
    for value in (bank.probability, bank.decision,
                  gate.w1, gate.b1, gate.w2, gate.b2):
        array = np.ascontiguousarray(value)
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _first_source_dummy(template):
    if template.order == "12":
        return "r1", template.bank1, template.actions1, None, None
    return "r2", template.bank2, template.actions2, None, None


def _draw_uniform(root, source_seed, order, unit):
    normalized_unit = int(unit) if type(unit) is np.int64 else unit
    message = json.dumps([DRAW_SCHEMA, source_seed, order, normalized_unit],
                         separators=(",", ":"), ensure_ascii=True).encode("ascii")
    value = int.from_bytes(hmac.new(root, message, hashlib.sha256).digest()[:8], "big")
    return min((value + .5) / (1 << 64), np.nextafter(1.0, 0.0))


def _persist_private_draw(path, payload):
    content = _canonical(payload)
    digest = hashlib.sha256(content).hexdigest()
    if path is None:
        return digest
    target = Path(path)
    if target.is_symlink():
        raise ValueError("private draw cache may not be a symlink")
    unresolved_parent = target.parent
    parent = unresolved_parent.resolve(strict=True)
    if (unresolved_parent.is_symlink() or not parent.is_dir() or
            stat.S_IMODE(parent.stat().st_mode) & 0o077):
        raise PermissionError("private draw cache parent must be owner-only")

    def verify():
        if (target.is_symlink() or not target.is_file() or
                stat.S_IMODE(target.stat().st_mode) & 0o077 or
                target.read_bytes() != content):
            raise ValueError("frozen FIT draw cache differs from sealed identity")

    if target.exists():
        verify()
        return digest
    fd, temp = tempfile.mkstemp(prefix=".seq-draw-", dir=parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, target)
        except FileExistsError:
            verify()
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return digest


def _freeze_fit_draw(data, template, prepared, gate, eta_first, checkpoint_sha,
                     units, units_sha, root, source_seed, source_sha, cache_path):
    _, bank, actions, cells, _ = _first_source(data, template)
    codes = np.empty(len(units), dtype=np.int64)
    chosen = {}
    laws = []
    for row, unit in enumerate(units):
        source_law = _factor_law(gate, prepared.features[row],
                                 actions[int(cells[row])])
        masses = {int(code): eta_first*float(mass)
                  for code, mass in source_law}
        b0 = int(actions[int(cells[row])][0])
        masses[b0] = masses.get(b0, 0.0) + (1-eta_first)
        law = tuple(sorted(masses.items()))
        laws.append(law)
        key = (type(unit).__name__, str(unit))
        if key not in chosen:
            uniform = _draw_uniform(root, source_seed, template.order, unit)
            cumulative = np.cumsum([mass for _, mass in law])
            index = min(int(np.searchsorted(cumulative, uniform, side="right")),
                        len(law) - 1)
            chosen[key] = law[index][0]
        codes[row] = chosen[key]
        if codes[row] not in actions[int(cells[row])]:
            raise AssertionError("first draw escaped the frozen action mask")
    seen = set(map(int, codes))
    unseen = tuple(code for code in range(len(bank.probability)) if code not in seen)
    coverage = {"seen_first_codes": tuple(sorted(seen)),
                "unseen_first_codes": unseen,
                "unseen_first_code_count": len(unseen),
                "unique_fit_units": len(chosen),
                "frozen_first_law_fit_mass_on_unseen": float(np.mean([
                    sum(mass for code, mass in law if code in unseen) for law in laws]))}
    payload = {"schema": "BAR1-seq-fit-first-boundary-v1", "order": template.order,
               "source_seed": source_seed, "hmac_message_schema": DRAW_SCHEMA,
               "eta_first": eta_first,
               "source_binding_sha256": source_sha,
               "first_checkpoint_sha256": checkpoint_sha,
               "private_root_sha256": hashlib.sha256(root).hexdigest(),
               "unit_ids_sha256": units_sha, "codes": codes.tolist()}
    digest = _persist_private_draw(cache_path, payload)
    codes.setflags(write=False)
    return codes, digest, coverage


def _freeze_identity_draw(data, template, checkpoint_sha, units, units_sha,
                          root, source_seed, source_sha, cache_path):
    """Persist B0 first codes under the same private FIT draw custody."""
    _, bank, actions, cells, _ = _first_source(data, template)
    codes = np.asarray([actions[int(cell)][0] for cell in cells], dtype=np.int64)
    seen = set(map(int, codes))
    unseen = tuple(code for code in range(len(bank.probability)) if code not in seen)
    coverage = {"seen_first_codes": tuple(sorted(seen)),
                "unseen_first_codes": unseen,
                "unseen_first_code_count": len(unseen),
                "unique_fit_units": len(set((type(unit).__name__, str(unit))
                                            for unit in units)),
                "frozen_first_law_fit_mass_on_unseen": 0.0}
    payload = {"schema": "BAR1-seq-fit-exact-first-v1", "order": template.order,
               "source_seed": source_seed, "source_binding_sha256": source_sha,
               "first_checkpoint_sha256": checkpoint_sha,
               "private_root_sha256": hashlib.sha256(root).hexdigest(),
               "unit_ids_sha256": units_sha, "codes": codes.tolist()}
    digest = _persist_private_draw(cache_path, payload)
    codes.setflags(write=False)
    return codes, digest, coverage


def fit_sequential(data: ProtectedData, template: SequentialPolicy, *,
                   caps: dict[str, float], recipe_stage1: FitRecipe,
                   recipe_stage2: FitRecipe, total_cpu_seconds: float,
                   unit_ids, private_root: bytes, source_seed: int,
                   source_binding_sha256: str,
                   private_draw_cache_path=None) -> dict:
    """Fit two starts per stage within one ceiling; return private draw separately.

    In-memory draw mode is for tiny fixtures only. The real runner must pass an
    owner-only cache path and may publish hashes/counts, never returned codes.
    """
    if (type(data) is not ProtectedData or type(template) is not SequentialPolicy or
        type(caps) is not dict or caps.keys() != TASK_CAP_KEYS or
        any(not np.isfinite(v) for v in caps.values()) or
        type(recipe_stage1) is not FitRecipe or type(recipe_stage2) is not FitRecipe or
        not np.isfinite(total_cpu_seconds) or total_cpu_seconds <= 0 or
        recipe_stage1.cpu_seconds + recipe_stage2.cpu_seconds > total_cpu_seconds or
        not isinstance(private_root, bytes) or len(private_root) < 32 or
        type(source_seed) is not int or source_seed not in (0, 1) or
        not isinstance(source_binding_sha256, str) or
        HEX64.fullmatch(source_binding_sha256) is None):
        raise ValueError("sequential fit requires fixed protected inputs and a total CPU ceiling")
    units, units_sha = _units_and_hash(data, unit_ids)
    clock = time.process_time()
    import torch
    torch.set_num_threads(1)
    prepared = prepare_law(data.task, template)
    arrays = _first_arrays(data, template)
    name, bank, _, _, valid1, public1, nll, brier = arrays
    guards = _identity_guards(prepared, data.sex)
    first_limits = {f"{name}_nll": caps[f"{name}_nll"],
                    f"{name}_brier": caps[f"{name}_brier"],
                    f"{name}_bayes": guards[f"{name}_bayes"]}
    feature = torch.as_tensor(prepared.features.copy())
    valid1_t = torch.as_tensor(valid1)
    public1_t = torch.as_tensor(public1)
    sex_t = torch.as_tensor(data.sex.copy())
    nll_t, brier_t = torch.as_tensor(nll), torch.as_tensor(brier)

    def within(start, recipe):
        now = time.process_time()
        return now - start < recipe.cpu_seconds and now - clock < total_cpu_seconds

    def init_weights(recipe, start_index, embedding=None):
        rng = np.random.default_rng(recipe.seed + start_index)
        arrays = [rng.normal(0, recipe.init_scale, size=(feature.shape[1], 16)),
                  rng.normal(0, recipe.init_scale, size=16),
                  np.zeros((16, 4)), np.zeros(4)]
        if start_index == 0:
            arrays[3][0] += recipe.identity_logit_bias
        if embedding is not None:
            arrays.append(embedding.copy())
        return [torch.nn.Parameter(torch.as_tensor(value, dtype=torch.float64))
                for value in arrays]

    first_start = time.process_time()
    first_starts, best_first, interrupted = [], None, False
    stage1_checkpoint_gates = []
    for start_index, start_name in enumerate(("identity_biased", "uniform_logit")):
        if not within(first_start, recipe_stage1):
            interrupted = True
            break
        weights = init_weights(recipe_stage1, start_index)
        optimizer = torch.optim.Adam(weights, lr=recipe_stage1.learning_rate,
                                     weight_decay=0)
        dual = {key: 0.0 for key in first_limits}
        receipt = {"start": start_name, "checkpoints": [], "updates": 0,
                   "completed": False}

        def checkpoint_first(update):
            nonlocal best_first
            gate = FactorGate(*(weight.detach().numpy().copy() for weight in weights))
            stage1_checkpoint_gates.append((start_index, update, gate))
            metrics = _first_exact(data, prepared, gate, arrays)
            projected = _project_first(data, prepared, gate, arrays, first_limits)
            feasible = all(metrics[key] <= first_limits[key] for key in first_limits)
            receipt["checkpoints"].append({"update": update, "metrics": metrics,
                                           "feasible": feasible,
                                           "projected": projected})
            rank = (projected["objective"], start_index, update, projected["eta"])
            if best_first is None or rank < best_first[0]:
                best_first = (rank, gate, projected)

        checkpoint_first(0)
        for update in range(1, recipe_stage1.updates + 1):
            if not within(first_start, recipe_stage1):
                interrupted = True
                break
            optimizer.zero_grad()
            logits = torch.tanh(feature @ weights[0] + weights[1]) @ weights[2] + weights[3]
            q = torch.softmax(logits.masked_fill(~valid1_t, -torch.inf), dim=1)
            values = {f"{name}_nll": (q * nll_t).sum(dim=1).mean(),
                      f"{name}_brier": (q * brier_t).sum(dim=1).mean()}
            counts = []
            for sensitive in (0, 1):
                weighted = q * (sex_t == sensitive).to(torch.float64)[:, None] / len(q)
                counts.append(torch.zeros(len(bank.probability), dtype=torch.float64).scatter_add(
                    0, public1_t.reshape(-1), weighted.reshape(-1)))
            values[f"{name}_bayes"] = torch.maximum(*counts).sum()
            penalty = values[f"{name}_bayes"]
            for key in first_limits:
                residual = values[key] - first_limits[key]
                penalty = penalty + dual[key] * residual + \
                    .5 * recipe_stage1.rho * residual.clamp_min(0).square()
            penalty.backward()
            torch.nn.utils.clip_grad_norm_(weights, recipe_stage1.gradient_norm)
            optimizer.step()
            receipt["updates"] = update
            if update % recipe_stage1.dual_every == 0:
                for key in first_limits:
                    dual[key] = max(0., dual[key] + recipe_stage1.rho *
                                    (float(values[key].detach()) - first_limits[key]))
            if update % recipe_stage1.checkpoint_every == 0 or update == recipe_stage1.updates:
                checkpoint_first(update)
        receipt["completed"] = not interrupted
        first_starts.append(receipt)
        if interrupted:
            break
    first_complete = (not interrupted and len(first_starts) == 2 and
                      all(s["completed"] for s in first_starts) and
                      within(first_start, recipe_stage1))
    result = {"complete": False, "success": False, "stage1_complete": first_complete,
              "stage1_starts": first_starts, "stage2_starts": [],
              "stage1_checkpoint_gates": stage1_checkpoint_gates,
              "elapsed_cpu_seconds": time.process_time() - clock}
    if not first_complete:
        return result
    first_gate = best_first[1]
    first_eta = float(best_first[2]["eta"])
    source_first_sha = _stage1_hash(first_gate, template, source_binding_sha256)
    first_sha = hashlib.sha256(b"BAR1-SEQ-FIRST-BOUNDARY-v1\0" +
                               bytes.fromhex(source_first_sha) +
                               np.asarray([first_eta], dtype="<f8").tobytes()).hexdigest()
    codes, draw_sha, coverage = _freeze_fit_draw(
        data, template, prepared, first_gate, first_eta, first_sha,
        units, units_sha, private_root, source_seed,
        source_binding_sha256, private_draw_cache_path)
    stage1_metrics = best_first[2]
    result.update({"first_checkpoint_sha256": first_sha,
                   "draw_cache_sha256": draw_sha,
                   "private_first_codes": codes,
                   "coverage": coverage, "stage1_metrics": stage1_metrics,
                   "exact_first": first_eta == 0.0,
                   "first_eta": first_eta})

    # Freeze first gate. The sampled code is a fixed categorical input to the
    # second gradient program; full-law integration is reserved for replay.
    second_actions = template.actions2 if template.order == "12" else template.actions1
    second_cells = data.task.cell2 if template.order == "12" else data.task.cell1
    first_actions = template.actions1 if template.order == "12" else template.actions2
    first_cells = data.task.cell1 if template.order == "12" else data.task.cell2
    valid2 = np.zeros((len(codes), 4), dtype=bool)
    slots = np.zeros((len(codes), 4), dtype=np.int64)
    for row, code in enumerate(codes):
        first_index = first_actions[int(first_cells[row])].index(int(code))
        valid2[row, :len(second_actions[int(second_cells[row])])] = True
        for index in range(4):
            slots[row, index] = (4 * first_index + index if template.order == "12"
                                 else 4 * index + first_index)
    valid2_t, slots_t = torch.as_tensor(valid2), torch.as_tensor(slots)
    codes_t = torch.as_tensor(codes.copy())
    e0 = public_probability_embedding(bank.probability)
    unseen = coverage["unseen_first_codes"]
    unseen_t = torch.as_tensor(unseen, dtype=torch.long)
    final_limits = {**caps, "r1_bayes": guards["r1_bayes"],
                    "r2_bayes": guards["r2_bayes"]}
    second_start = time.process_time()
    second_starts, best_final, interrupted = [], None, False
    checkpoint_policies = []
    for start_index, start_name in enumerate(("identity_biased", "uniform_logit")):
        if not within(second_start, recipe_stage2):
            interrupted = True
            break
        weights = init_weights(recipe_stage2, start_index, embedding=e0)
        optimizer = torch.optim.Adam(weights, lr=recipe_stage2.learning_rate,
                                     weight_decay=0)
        dual = {key: 0.0 for key in final_limits}
        receipt = {"start": start_name, "checkpoints": [], "updates": 0,
                   "completed": False}

        def checkpoint_final(update):
            nonlocal best_final
            arrays = [weight.detach().numpy().copy() for weight in weights]
            second_gate = FactorGate(*arrays[:4], embedding=arrays[4])
            policy = SequentialPolicy(template.bank1, template.bank2,
                                      template.actions1, template.actions2,
                                      template.gate, first_gate, second_gate,
                                      order=template.order,
                                      first_checkpoint_sha256=first_sha)
            checkpoint_policies.append((start_index, update, policy))
            deployed = BoundarySequentialPolicy(policy, first_eta, 1.0)
            metrics = exact_metrics(data, deployed)
            feasible = all(metrics[key] <= final_limits[key] for key in final_limits)
            receipt["checkpoints"].append({"update": update, "metrics": metrics,
                                           "feasible": feasible})
            rank = (metrics["protected_objective"], start_index, update)
            if feasible and (best_final is None or rank < best_final[0]):
                best_final = (rank, policy, metrics)

        checkpoint_final(0)
        for update in range(1, recipe_stage2.updates + 1):
            if not within(second_start, recipe_stage2):
                interrupted = True
                break
            optimizer.zero_grad()
            hidden = torch.tanh(feature @ weights[0] + weights[1] + weights[4][codes_t])
            logits = hidden @ weights[2] + weights[3]
            q_second = torch.softmax(logits.masked_fill(~valid2_t, -torch.inf), dim=1)
            q_pair = torch.zeros((len(codes), 16), dtype=torch.float64).scatter(
                1, slots_t, q_second)
            values = _torch_metrics(prepared, q_pair, sex=data.sex, torch=torch)
            penalty = values["objective"]
            for key in final_limits:
                residual = values[key] - final_limits[key]
                penalty = penalty + dual[key] * residual + \
                    .5 * recipe_stage2.rho * residual.clamp_min(0).square()
            penalty.backward()
            if len(unseen_t):
                weights[4].grad[unseen_t] = 0
            torch.nn.utils.clip_grad_norm_(weights, recipe_stage2.gradient_norm)
            optimizer.step()
            if len(unseen_t):
                with torch.no_grad():
                    weights[4][unseen_t] = torch.as_tensor(e0[list(unseen)])
            receipt["updates"] = update
            if update % recipe_stage2.dual_every == 0:
                for key in final_limits:
                    dual[key] = max(0., dual[key] + recipe_stage2.rho *
                                    (float(values[key].detach()) - final_limits[key]))
            if update % recipe_stage2.checkpoint_every == 0 or update == recipe_stage2.updates:
                checkpoint_final(update)
        receipt["completed"] = not interrupted
        second_starts.append(receipt)
        if interrupted:
            break
    final_complete = (not interrupted and len(second_starts) == 2 and
                      all(s["completed"] for s in second_starts) and
                      within(second_start, recipe_stage2))
    result.update({"stage2_starts": second_starts,
                   "checkpoint_policies": checkpoint_policies,
                   "complete": bool(final_complete),
                   "success": bool(final_complete and best_final is not None),
                   "policy": best_final[1] if final_complete and best_final else None,
                   "metrics": best_final[2] if final_complete and best_final else None,
                   "elapsed_cpu_seconds": time.process_time() - clock})
    return result
