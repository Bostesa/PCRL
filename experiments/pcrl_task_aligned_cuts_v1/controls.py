"""Matched finite-channel controls for the 2018 PCRL development study.

Every coefficient matrix already includes its normalized row mass.  A cut is a
fixed-predictor expected loss floor, ``sum(coeff * Q) >= floor``.  Optimizing a
fixed bank does not certify a best-response oracle or population privacy.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, lil_matrix


@dataclass(frozen=True)
class Bank:
    cost: np.ndarray
    coefficients: np.ndarray
    floors: np.ndarray
    ids: tuple[str, ...]


def make_bank(cost: np.ndarray, cuts: Sequence[Mapping[str, Any]]) -> Bank:
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2 or not min(c.shape) or not np.all(np.isfinite(c)):
        raise ValueError('cost must be a finite nonempty state-by-token matrix')
    coefficients = []
    floors = []
    ids = []
    for cut in cuts:
        a = np.asarray(cut['coeff'], dtype=np.float64)
        floor = float(cut['floor'])
        ident = str(cut['id'])
        if a.shape != c.shape or not np.all(np.isfinite(a)) or not np.isfinite(floor):
            raise ValueError(f'invalid cut {ident}')
        if not ident or ident in ids:
            raise ValueError(f'duplicate or empty cut ID {ident}')
        coefficients.append(a)
        floors.append(floor)
        ids.append(ident)
    aa = np.stack(coefficients) if coefficients else np.empty((0, *c.shape))
    return Bank(c, aa, np.asarray(floors, dtype=np.float64), tuple(ids))


def replay(q: np.ndarray, bank: Bank, *, deterministic: bool = False,
           tolerance: float = 1e-7) -> dict[str, Any]:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != bank.cost.shape or not np.all(np.isfinite(q)):
        raise ValueError('invalid channel shape or nonfinite entry')
    row_error = float(np.max(np.abs(q.sum(axis=1) - 1)))
    negative = float(max(0.0, -np.min(q)))
    above_one = float(max(0.0, np.max(q) - 1.0))
    binary_error = float(np.max(np.minimum(np.abs(q), np.abs(q - 1)))) if deterministic else 0.0
    losses = np.einsum('ktz,tz->k', bank.coefficients, q)
    slacks = losses - bank.floors
    violation = float(max(0.0, -np.min(slacks))) if len(slacks) else 0.0
    return {
        'objective': float(np.sum(bank.cost * q)),
        'row_error': row_error, 'negative_entry': negative,
        'above_one': above_one, 'binary_error': binary_error,
        'cut_losses': dict(zip(bank.ids, map(float, losses))),
        'cut_slacks': dict(zip(bank.ids, map(float, slacks))),
        'maximum_cut_violation': violation,
        'feasible': max(row_error, negative, above_one, binary_error, violation) <= tolerance,
    }


def rowwise_minimizer(cost: np.ndarray) -> np.ndarray:
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2 or not min(c.shape) or not np.all(np.isfinite(c)):
        raise ValueError('invalid fixed task cost')
    q = np.zeros_like(c)
    q[np.arange(len(c)), np.argmin(c, axis=1)] = 1.0
    return q


def constant_maps(n_states: int, n_tokens: int) -> list[np.ndarray]:
    if n_states < 1 or n_tokens < 1:
        raise ValueError('empty map shape')
    return [np.tile(np.eye(n_tokens)[z], (n_states, 1)) for z in range(n_tokens)]


def randomized_response(base: np.ndarray, publish_probability: float) -> np.ndarray:
    """Visible 17-token randomized response, including the unmodified endpoint."""
    q = np.asarray(base, dtype=np.float64)
    p = float(publish_probability)
    if q.ndim != 2 or not 0 <= p <= 1 or not np.all(np.isfinite(q)):
        raise ValueError('invalid deterministic base or RR probability')
    if np.max(np.abs(q.sum(axis=1) - 1)) > 1e-12 or not np.all((q == 0) | (q == 1)):
        raise ValueError('RR base must be a deterministic stochastic matrix')
    return p * q + (1 - p) / q.shape[1]


def withholding(base: np.ndarray, publish_probability: float) -> np.ndarray:
    """Append one explicit visible missing-token column at index ``n_tokens``."""
    q = np.asarray(base, dtype=np.float64)
    p = float(publish_probability)
    if q.ndim != 2 or not 0 <= p <= 1 or not np.all(np.isfinite(q)):
        raise ValueError('invalid deterministic base or withholding probability')
    if np.max(np.abs(q.sum(axis=1) - 1)) > 1e-12 or not np.all((q == 0) | (q == 1)):
        raise ValueError('withholding base must be deterministic')
    return np.column_stack((p * q, np.full(len(q), 1 - p)))


def solve_deterministic_p1(
    cost: np.ndarray, cuts: Sequence[Mapping[str, Any]], *,
    time_limit_seconds: float, relative_gap: float = 1e-6,
    feasibility_tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Solve the shared-bank binary map MILP with a *valid* HiGHS lower bound.

    A timeout is not an infeasibility finding.  A returned incumbent is accepted
    only after an independent coefficient replay.  scipy.milp does not expose a
    warm-start API, so deterministic search should be budgeted separately.
    """
    bank = make_bank(cost, cuts)
    if not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError('time limit must be positive and finite')
    n_states, n_tokens = bank.cost.shape
    n_vars = n_states * n_tokens
    n_cuts = len(bank.ids)
    a = lil_matrix((n_states + n_cuts, n_vars), dtype=np.float64)
    for t in range(n_states):
        a[t, t * n_tokens:(t + 1) * n_tokens] = 1.0
    for j, coeff in enumerate(bank.coefficients):
        a[n_states + j, :] = coeff.ravel()
    lower = np.concatenate((np.ones(n_states), bank.floors))
    upper = np.concatenate((np.ones(n_states), np.full(n_cuts, np.inf)))
    start = time.perf_counter()
    result = milp(
        bank.cost.ravel(), integrality=np.ones(n_vars, dtype=np.int8),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=LinearConstraint(csr_matrix(a), lower, upper),
        options={'time_limit': float(time_limit_seconds), 'mip_rel_gap': float(relative_gap)},
    )
    elapsed = time.perf_counter() - start
    raw_bound = getattr(result, 'mip_dual_bound', None)
    bound = float(raw_bound) if raw_bound is not None and np.isfinite(raw_bound) else None
    out = {
        'status_code': int(result.status), 'solver_message': str(result.message),
        'elapsed_seconds': elapsed, 'time_limit_seconds': float(time_limit_seconds),
        'relative_gap_target': float(relative_gap), 'fixed_bank_cut_ids': list(bank.ids),
        'lower_bound': bound, 'incumbent_objective': None, 'incumbent_valid': False,
        'mip_gap_reported': float(result.mip_gap) if getattr(result, 'mip_gap', None) is not None and np.isfinite(result.mip_gap) else None,
        'Q': None, 'replay': None,
    }
    if result.x is not None:
        x = np.asarray(result.x, dtype=np.float64).reshape(bank.cost.shape)
        raw_integrality_error = float(np.max(np.minimum(np.abs(x), np.abs(x - 1))))
        rounded = np.zeros_like(x)
        rounded[np.arange(n_states), np.argmax(x, axis=1)] = 1.0
        report = replay(rounded, bank, deterministic=True, tolerance=feasibility_tolerance)
        valid = bool(report['feasible'] and raw_integrality_error <= max(feasibility_tolerance, 1e-5))
        out.update(Q=rounded if valid else None, replay=report,
                   incumbent_objective=report['objective'] if valid else None,
                   incumbent_valid=valid,
                   raw_solver_objective=float(result.fun) if result.fun is not None else None,
                   raw_integrality_error=raw_integrality_error)
    return out


def deterministic_coordinate_search(
    cost: np.ndarray, cuts: Sequence[Mapping[str, Any]], *,
    seconds: float, seed: int, starts: Sequence[np.ndarray] = (),
) -> dict[str, Any]:
    """Bounded fallback search; an incumbent is a control, never a lower bound."""
    bank = make_bank(cost, cuts)
    if seconds <= 0 or not np.isfinite(seconds):
        raise ValueError('seconds must be positive')
    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()
    n_states, n_tokens = bank.cost.shape
    seeds = [rowwise_minimizer(bank.cost), *constant_maps(n_states, n_tokens), *starts]
    best_q = None
    best_metric = (np.inf, np.inf)
    attempts = 0
    while time.perf_counter() - start_time < seconds or attempts == 0:
        if attempts < len(seeds):
            q0 = np.asarray(seeds[attempts], dtype=np.float64)
            if q0.shape != bank.cost.shape:
                raise ValueError('invalid heuristic start shape')
            z = np.argmax(q0, axis=1)
        else:
            z = rng.integers(n_tokens, size=n_states)
        attempts += 1
        current = np.zeros_like(bank.cost)
        current[np.arange(n_states), z] = 1
        for _ in range(3):
            changed = False
            for t in rng.permutation(n_states):
                best_action = int(z[t]); local_metric = (np.inf, np.inf)
                for action in range(n_tokens):
                    current[t, :] = 0; current[t, action] = 1
                    r = replay(current, bank, deterministic=True)
                    metric = (r['maximum_cut_violation'], r['objective'])
                    if metric < local_metric:
                        best_action, local_metric = action, metric
                current[t, :] = 0; current[t, best_action] = 1
                if best_action != z[t]:
                    z[t] = best_action; changed = True
            if not changed:
                break
        r = replay(current, bank, deterministic=True)
        metric = (r['maximum_cut_violation'], r['objective'])
        if metric < best_metric:
            best_metric = metric; best_q = current.copy()
    return {
        'Q': best_q, 'replay': replay(best_q, bank, deterministic=True),
        'attempts': attempts, 'elapsed_seconds': time.perf_counter() - start_time,
        'valid_lower_bound': None,
    }


def optimize_gradient_bank(
    cost: np.ndarray, cuts: Sequence[Mapping[str, Any]], *,
    steps: int, learning_rate: float, penalty: float,
    seed: int, initial: np.ndarray | None = None,
    dual_learning_rate: float = 0.0,
) -> dict[str, Any]:
    """Exact-expectation softmax-logit baseline against a frozen attack bank.

    Repeated attack best responses and validation tuning live in the common
    exchange controller.  This routine performs one matched fixed-bank channel
    update; its absence of oracle violation is not a privacy theorem.
    """
    bank = make_bank(cost, cuts)
    if (steps < 1 or learning_rate <= 0 or penalty < 0 or
            dual_learning_rate < 0 or not all(map(np.isfinite,
                    (learning_rate, penalty, dual_learning_rate)))):
        raise ValueError('invalid gradient budget')
    rng = np.random.default_rng(seed)
    if initial is None:
        logits = rng.normal(0, 0.01, size=bank.cost.shape)
    else:
        q0 = np.asarray(initial, dtype=np.float64)
        if q0.shape != bank.cost.shape or np.min(q0) < 0 or np.max(np.abs(q0.sum(axis=1) - 1)) > 1e-8:
            raise ValueError('invalid initial channel')
        logits = np.log(np.clip(q0, 1e-8, 1.0))
    m = np.zeros_like(logits); v = np.zeros_like(logits)
    beta1, beta2 = 0.9, 0.999
    multipliers = np.zeros(len(bank.ids), dtype=np.float64)
    best = None; best_key = (np.inf, np.inf)
    for step in range(1, steps + 1):
        centered = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(centered); q = exp / exp.sum(axis=1, keepdims=True)
        losses = np.einsum('ktz,tz->k', bank.coefficients, q)
        residual = np.maximum(bank.floors - losses, 0.0)
        if len(residual):
            multipliers = np.maximum(0.0, multipliers + dual_learning_rate *
                                     (bank.floors - losses))
        grad_q = bank.cost.copy()
        if len(residual):
            grad_q -= np.einsum('k,ktz->tz', multipliers + 2 * penalty * residual,
                               bank.coefficients)
        grad = q * (grad_q - np.sum(grad_q * q, axis=1, keepdims=True))
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad * grad
        logits -= learning_rate * (m / (1 - beta1 ** step)) / (np.sqrt(v / (1 - beta2 ** step)) + 1e-8)
        if step == 1 or step == steps or step % max(1, steps // 20) == 0:
            report = replay(q, bank)
            key = (report['maximum_cut_violation'], report['objective'])
            if key < best_key:
                best_key = key; best = q.copy()
    return {'Q': best, 'replay': replay(best, bank), 'steps': steps,
            'learning_rate': learning_rate, 'penalty': penalty, 'seed': seed,
            'dual_learning_rate': dual_learning_rate,
            'final_multipliers': dict(zip(bank.ids, map(float, multipliers))),
            'optimizer': 'softmax_adam_exact_fixed_bank_augmented_lagrangian'}


def gradient_control_grid(cost: np.ndarray, cuts: Sequence[Mapping[str, Any]], *,
                          steps_per_run: int, seeds: Sequence[int],
                          penalties: Sequence[float],
                          learning_rates: Sequence[float] = (0.05,),
                          dual_learning_rates: Sequence[float] = (1.0,)) -> list[dict[str, Any]]:
    """Run a preregistered matched gradient grid; do not select on assessment.

    The caller must freeze this grid and choose a route on the allowed inner
    validation resource.  All runs and budgets are returned, including losses.
    Exchange best responses are supplied by the shared oracle controller and
    must be budgeted identically to the LP/MILP arms.
    """
    if not seeds or not penalties or not learning_rates or not dual_learning_rates:
        raise ValueError('empty gradient comparison grid')
    outputs = []
    for seed in seeds:
        for penalty in penalties:
            for learning_rate in learning_rates:
                for dual_lr in dual_learning_rates:
                    outputs.append(optimize_gradient_bank(
                        cost, cuts, steps=steps_per_run,
                        learning_rate=learning_rate, penalty=penalty,
                        seed=seed, dual_learning_rate=dual_lr))
    return outputs


def _array_sha(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def materialize_simple_controls(index_path: str | Path, anchor: int,
                                output_dir: str | Path, *,
                                u1_cost_path: str | Path | None = None) -> dict[str, Any]:
    """Materialize all preregistered simple curves without selecting a point.

    These maps require independent audits; writing them is not a performance
    result.  Withholding uses an eighteenth *visible* token category.
    """
    from . import data
    idx = data.index(index_path)
    d17 = data.load_map(idx, anchor, 'D17')
    d33 = data.load_map(idx, anchor, 'D33')
    historical_q = data.load_map(idx, anchor, 'Q')
    bases = {'D17': d17}
    if u1_cost_path is not None:
        with np.load(u1_cost_path, allow_pickle=False) as archive:
            if 'cost' not in archive:
                raise ValueError('U1 cost archive must contain cost')
            u1_cost = np.asarray(archive['cost'], dtype=np.float64)
        if u1_cost.shape != (32, 17):
            raise ValueError('U1 cost has wrong fixed T0/action dimensions')
        bases['D_U1'] = rowwise_minimizer(u1_cost)
    maps: dict[str, np.ndarray] = {'Q_historical': historical_q, 'D33_external': d33}
    rates = (0.5, 0.75, 0.9)
    for name, base in bases.items():
        maps[name] = base
        for p in rates:
            maps[f'{name}_RR_{p:.2f}'] = randomized_response(base, p)
            maps[f'{name}_withhold_{p:.2f}'] = withholding(base, p)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('control output directory is not empty; preserve prior attempt')
    manifest: dict[str, Any] = {'schema':'pcrl-simple-controls-v1','anchor':anchor,
                                'rates':list(rates), 'map_records':{},
                                'u1_cost_sha256': _array_sha(u1_cost) if u1_cost_path is not None else None,
                                'independent_audit_required':True}
    seen: dict[str, str] = {}
    files = []
    for name, q in maps.items():
        if np.min(q) < -1e-8 or np.max(np.abs(q.sum(axis=1)-1)) > 1e-8:
            raise ValueError(f'invalid control simplex: {name}')
        array_sha = _array_sha(q)
        record = {'array_sha256':array_sha,'shape':list(q.shape),
                  'visible_missing_token':bool('withhold' in name),
                  'missing_token_index':17 if 'withhold' in name else None}
        if array_sha in seen:
            record['alias_of'] = seen[array_sha]
        else:
            filename = name + '.npz'
            np.savez_compressed(output/filename, Q=q)
            record['file'] = filename
            record['file_sha256'] = hashlib.sha256((output/filename).read_bytes()).hexdigest()
            files.append(filename)
            seen[array_sha] = name
        manifest['map_records'][name] = record
    manifest_path = output/'MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    files.append('MANIFEST.json')
    if os.environ.get('PCRL_UNIT_ID') and os.environ.get('PCRL_QUEUE_SHA256'):
        from .scheduler import write_completion
        write_completion(output,os.environ['PCRL_UNIT_ID'],
                         os.environ['PCRL_QUEUE_SHA256'],files)
    return manifest


def load_saved_bank(fit_dir: str | Path) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    """Reconstruct the exact private bank serialized by ``fit_p1_center``."""
    from . import solver
    directory = Path(fit_dir)
    meta = json.loads((directory/'bank.json').read_text())
    archive = directory/'coefficients.npz'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != meta['coefficient_archive_sha256']:
        raise ValueError('coefficient archive differs from bank manifest')
    with np.load(archive, allow_pickle=False) as arrays:
        cost = np.asarray(arrays['cost'], dtype=np.float64)
        cuts = []
        for index, record in enumerate(meta['cuts']):
            key = f'cut_{index:04d}'
            if key not in arrays:
                raise ValueError(f'missing frozen bank coefficient: {key}')
            cuts.append({**record, 'coeff': np.asarray(arrays[key], dtype=np.float64)})
    if solver.bank_sha256(cuts, cost.shape) != meta['bank_sha256']:
        raise ValueError('frozen bank hash mismatch')
    make_bank(cost, cuts)
    return cost, cuts, meta['bank_sha256']


def fit_saved_bank_controls(fit_dir: str | Path, output_dir: str | Path, *,
                            milp_seconds: float, heuristic_seconds: float,
                            gradient_steps: int, gradient_seeds: Sequence[int],
                            gradient_penalties: Sequence[float],
                            gradient_learning_rates: Sequence[float],
                            gradient_dual_rates: Sequence[float]) -> dict[str, Any]:
    """Run matched fixed-bank deterministic and gradient controls.

    The bank, cost and reference floors are immutable; this function neither
    fits a new attack nor chooses among gradient settings on assessment data.
    An exchange controller can call it after each common-bank update with the
    *same* retained cuts and equal oracle budget.
    """
    cost, cuts, bank_hash = load_saved_bank(fit_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('control fit output is not empty; preserve attempt')
    deterministic = solve_deterministic_p1(cost, cuts,
                                           time_limit_seconds=milp_seconds)
    heuristic = deterministic_coordinate_search(cost, cuts,
                                                seconds=heuristic_seconds,
                                                seed=20260923)
    gradient = gradient_control_grid(cost, cuts,
                                     steps_per_run=gradient_steps,
                                     seeds=gradient_seeds,
                                     penalties=gradient_penalties,
                                     learning_rates=gradient_learning_rates,
                                     dual_learning_rates=gradient_dual_rates)
    artifacts = []
    if deterministic['Q'] is not None:
        np.savez_compressed(output/'MILP_Q.npz', Q=deterministic['Q'])
        artifacts.append('MILP_Q.npz')
    np.savez_compressed(output/'HEURISTIC_Q.npz', Q=heuristic['Q'])
    artifacts.append('HEURISTIC_Q.npz')
    for index, candidate in enumerate(gradient):
        filename = f'GRADIENT_{index:03d}_Q.npz'
        np.savez_compressed(output/filename, Q=candidate['Q'])
        artifacts.append(filename)
    report = {
        'schema':'pcrl-matched-bank-controls-v1',
        'bank_sha256':bank_hash,
        'cut_count':len(cuts),
        'milp':{k:v for k,v in deterministic.items() if k != 'Q'},
        'heuristic':{k:v for k,v in heuristic.items() if k != 'Q'},
        'gradient':[{k:v for k,v in candidate.items() if k != 'Q'}
                    for candidate in gradient],
        'same_frozen_cost_and_cuts':True,
        'fresh_oracle_error':'unresolved; this command uses a fixed bank only',
    }
    (output/'CONTROLS.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    artifacts.append('CONTROLS.json')
    if os.environ.get('PCRL_UNIT_ID') and os.environ.get('PCRL_QUEUE_SHA256'):
        from .scheduler import write_completion
        write_completion(output,os.environ['PCRL_UNIT_ID'],
                         os.environ['PCRL_QUEUE_SHA256'],artifacts)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('simple','fit-bank'))
    parser.add_argument('--index',type=Path)
    parser.add_argument('--anchor',type=int)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--u1-cost',type=Path)
    parser.add_argument('--fit-dir',type=Path)
    parser.add_argument('--milp-seconds',type=float)
    parser.add_argument('--heuristic-seconds',type=float)
    parser.add_argument('--gradient-steps',type=int)
    parser.add_argument('--gradient-seed',type=int,action='append')
    parser.add_argument('--gradient-penalty',type=float,action='append')
    parser.add_argument('--gradient-learning-rate',type=float,action='append')
    parser.add_argument('--gradient-dual-rate',type=float,action='append')
    args = parser.parse_args(argv)
    if args.command == 'simple':
        if args.index is None or args.anchor is None:
            parser.error('simple requires --index and --anchor')
        result = materialize_simple_controls(args.index,args.anchor,args.output_dir,
                                             u1_cost_path=args.u1_cost)
        print(json.dumps({'anchor':result['anchor'],'map_labels':len(result['map_records']),
                          'distinct_maps':sum('alias_of' not in r for r in result['map_records'].values())},
                         sort_keys=True))
    else:
        required = (args.fit_dir,args.milp_seconds,args.heuristic_seconds,
                    args.gradient_steps,args.gradient_seed,args.gradient_penalty,
                    args.gradient_learning_rate,args.gradient_dual_rate)
        if any(value is None for value in required):
            parser.error('fit-bank requires fit directory and explicit compute/hyperparameter budget')
        result = fit_saved_bank_controls(args.fit_dir,args.output_dir,
                                         milp_seconds=args.milp_seconds,
                                         heuristic_seconds=args.heuristic_seconds,
                                         gradient_steps=args.gradient_steps,
                                         gradient_seeds=args.gradient_seed,
                                         gradient_penalties=args.gradient_penalty,
                                         gradient_learning_rates=args.gradient_learning_rate,
                                         gradient_dual_rates=args.gradient_dual_rate)
        print(json.dumps({'bank_sha256':result['bank_sha256'],
                          'milp_valid':result['milp']['incumbent_valid'],
                          'gradient_runs':len(result['gradient'])},sort_keys=True))


if __name__ == '__main__':
    main()
