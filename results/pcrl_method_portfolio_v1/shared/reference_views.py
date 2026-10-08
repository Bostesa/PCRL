"""Pure complete-view builders for fixed unprotected track anchors.

Caller supplies only S4-admitted role-specific arrays. Adult role loading and
fresh attacker fits remain separate queued jobs after the effective seal.
"""

from __future__ import annotations

import hashlib
import numpy as np

from attack_eval import AttackView
from common_eval import assert_same_people


FEATURE_NAMES_SHA256 = "7101f2fb37b623a08221c32b33eedf132cf0c4c376bb4042ab9cf16da08ad975"
ARM_ID = "FEATURE_REF_RAW83_U"


def feature_ref_raw83_u(feature_record: dict, reference_record: dict, *,
                        source_model_sha256: str) -> dict[str, AttackView]:
    """RAW83 plus Ucal task output for each recipient and aligned coalition.

    U0 is a public-model-derived defense-aware feature because the recipient
    already receives all 83 permitted source inputs. Its numeric parity to
    the published source model is an independent S4 admission requirement.
    """
    assert_same_people(feature_record, reference_record)
    names = [str(name) for name in feature_record["feature_names"]]
    if len(names) != 83 or hashlib.sha256("\n".join(names).encode()).hexdigest() != FEATURE_NAMES_SHA256:
        raise ValueError("unprotected feature reference schema mismatch")
    if len(source_model_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_model_sha256):
        raise ValueError("unprotected feature reference source model SHA absent")
    raw = np.asarray(feature_record["X"], dtype=np.float64)
    n = raw.shape[0]
    if raw.shape != (n, 83) or not np.all(np.isfinite(raw)):
        raise ValueError("unprotected feature reference X invalid")
    fields = {}
    for recipient, k in ((1, 2), (2, 6)):
        calibrated = np.asarray(reference_record[f"Ucal_{recipient}"], dtype=np.float64)
        uncalibrated = np.asarray(reference_record[f"U0_{recipient}"], dtype=np.float64)
        if (calibrated.shape != (n, k) or uncalibrated.shape != (n, k) or
                np.any(calibrated < 0) or np.any(uncalibrated < 0) or
                not np.all(np.isfinite(calibrated)) or not np.all(np.isfinite(uncalibrated)) or
                not np.allclose(calibrated.sum(1), 1, atol=1e-8, rtol=0) or
                not np.allclose(uncalibrated.sum(1), 1, atol=1e-8, rtol=0)):
            raise ValueError("unprotected feature reference probability invalid")
        decision = np.argmax(calibrated, axis=1)
        if not np.array_equal(decision, np.asarray(reference_record[f"d_{recipient}"], dtype=np.int64)):
            raise ValueError("unprotected feature reference decision mismatch")
        fields[recipient] = (calibrated, np.eye(k, dtype=np.float64)[decision], uncalibrated)

    def build(recipients: tuple[int, ...]) -> AttackView:
        arrays, spans, component_names, constants, da_parts = [], [], [], [], []
        col = 0
        for r in recipients:
            q, d, u0 = fields[r]
            for suffix, value in (("selected_values", raw), ("probability", q), ("decision", d)):
                name = f"r{r}.{suffix}"
                arrays.append(value)
                spans.append((name, col, col + value.shape[1]))
                component_names.append(name)
                col += value.shape[1]
            constants.extend(((f"r{r}.model_id", ARM_ID),
                              (f"r{r}.source_model_sha256", source_model_sha256),
                              (f"r{r}.calibration", "Ucal_H_GLOBAL_TEMP"),
                              (f"r{r}.fallback", "false")))
            da_parts.append(u0)
        x = np.column_stack(arrays)
        return AttackView(
            row_id=np.asarray(feature_record["row_id"], dtype=np.int64),
            unit=np.asarray(feature_record["unit"], dtype=np.int64),
            x=x, components=tuple(component_names),
            wire_components=tuple(component_names) + tuple(name for name, _ in constants),
            component_columns=tuple(spans), wire_constants=tuple(constants),
            x_defense_aware=np.column_stack((x, *da_parts)),
            x_raw_only=raw,
        ).validate()

    return {"r1": build((1,)), "r2": build((2,)), "pair": build((1, 2))}
