"""Equal-update local MMD and freeze-first hard-code controls.

All row-level anchor bits remain private. They are computed by the same
HardCodeRelease path used at service time, then indexed by the original
aligned FIT row order for the second training stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.fit import (
    FitData,
    FitResult,
    TrainConfig,
    fit_arm,
    make_batch_plan,
)
from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.model import (
    FixedScaler,
    HardCodeRelease,
)


@dataclass
class LocalSequentialResult:
    fit: FitResult
    mode: str
    first_release: HardCodeRelease | None
    first_fit_codes: np.ndarray | None
    first_fit_code_sha256: str | None
    fit_row_order_sha256: str
    updates_by_purpose: tuple[int, int]


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def fit_local_or_sequential(data: FitData, config: TrainConfig, seed: int,
                            warm_state: dict[str, torch.Tensor], scaler: FixedScaler,
                            mode: str, source_sha256: str,
                            batch_plan: tuple[tuple[np.ndarray, ...], ...] | None = None
                            ) -> LocalSequentialResult:
    """Continue a shared warm state for exactly e_c epochs per recipient."""
    if mode not in ("L", "S12", "S21"):
        raise ValueError("control mode must be L, S12 or S21")
    if data.sex is None or warm_state is None:
        raise ValueError("privacy controls require FIT SEX and shared T warm state")
    expected_plan = make_batch_plan(np.arange(data.rows), seed=seed,
                                    epochs=config.epochs,
                                    batch_size=config.batch_size, sex=data.sex)
    if batch_plan is None:
        batch_plan = expected_plan
    elif (len(batch_plan) != len(expected_plan)
          or any(len(given) != len(expected) or any(
              not np.array_equal(a, b) for a, b in zip(given, expected))
                 for given, expected in zip(batch_plan, expected_plan))):
        raise ValueError("control batch plan differs from frozen paired FIT plan")
    row_sha = _hash_array(data.row_ids)
    if mode == "L":
        fit = fit_arm(data, config, seed, mode="L", init_state=warm_state,
                      scaler=scaler, full_fit=True, batch_plan=batch_plan)
        return LocalSequentialResult(fit, mode, None, None, None, row_sha,
                                     (fit.optimizer_updates, fit.optimizer_updates))
    first = 0 if mode == "S12" else 1
    second = 1 - first
    first_fit = fit_arm(data, config, seed, mode="L", init_state=warm_state,
                        scaler=scaler, active_purposes=(first,), full_fit=True,
                        batch_plan=batch_plan)
    first_release = HardCodeRelease(first_fit.model, scaler,
                                    f"{mode}-first", source_sha256)
    first_latent = (data.h0, data.h1)[first]
    first_code = first_release.recipient(first, first_latent)["code"]
    if first_code.shape != (data.rows, 10) or not np.isin(first_code, (0, 1)).all():
        raise AssertionError("first release did not emit aligned ten hard bits")
    final_fit = fit_arm(data, config, seed, mode="D-K", init_state=first_fit.state_dict,
                        scaler=scaler, active_purposes=(second,),
                        pair_anchor=first_code, full_fit=True,
                        batch_plan=batch_plan)
    for prefix in (f"mappers.{first}.", f"heads.{first}."):
        first_state, final_state = first_fit.state_dict, final_fit.state_dict
        for name, before in first_state.items():
            if name.startswith(prefix) and not torch.equal(before, final_state[name]):
                raise AssertionError("frozen first recipient changed in second stage")
    updates = [0, 0]
    updates[first] = first_fit.optimizer_updates
    updates[second] = final_fit.optimizer_updates
    return LocalSequentialResult(final_fit, mode, first_release,
                                 first_code.copy(), _hash_array(first_code),
                                 row_sha, tuple(updates))
