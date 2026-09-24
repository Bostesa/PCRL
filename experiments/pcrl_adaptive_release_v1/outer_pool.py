"""Five-pool, post-unlock 2018 assessment loader.

The ordinary prepared loader intentionally strips attacker-validation labels.
Only the outer gate calls this module, after verifying a remotely published
selection lock. The original prepared object is SHA-pinned before unpickling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from experiments.pcrl_task_aligned_cuts_v1 import data
from . import roles


FIFTH_POOL = "attacker_validation"
FIELDS = ("x", "ha", "hb", "weights", "ids", "households", "J")
ENCODED = ("p", "r", "risk")


def _same_bytes(left: Any, right: Any) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.hasobject:
        # Object-array bytes are process-local pointers, not serialized IDs.
        return all(type(x) is type(y) and x == y for x, y in zip(a.flat, b.flat))
    return np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def pooled_locked_outer(prepared: dict[str, Any], census: Mapping[str, Any]) -> dict:
    """Append the fifth pool in its original order and replay census counts."""
    pool = prepared["ctx"]["pools"].get(FIFTH_POOL)
    if not isinstance(pool, dict) or set(pool.get("labels", {})) != set(roles.CLASS_COUNT):
        raise ValueError("original label-bearing fifth pool required after unlock")
    encoded = prepared["encoded"][FIFTH_POOL]
    n = len(pool["ids"])
    if (any(len(np.asarray(pool[field])) != n for field in
            ("x", "ha", "hb", "weights", "households")) or
            any(len(np.asarray(pool["labels"][target])) != n for target in roles.CLASS_COUNT) or
            len(np.asarray(encoded["codes"]["T0"])) != n or
            any(len(np.asarray(encoded[field])) != n for field in ENCODED)):
        raise ValueError("fifth pool row alignment changed")
    mask = np.asarray([roles.role_of(h) == "outer_assessment"
                       for h in pool["households"]], dtype=bool)
    main = roles._pooled_role(prepared, "outer_assessment", allow_outer=True)
    fields = {"x": "x", "ha": "ha", "hb": "hb", "weights": "weights",
              "ids": "ids", "households": "households",
              "token_codes": None, "teacher_p": "p", "residual": "r", "risk": "risk"}
    result = {}
    for destination, source in fields.items():
        additional = (encoded["codes"]["T0"] if destination == "token_codes" else
                      encoded[source] if destination in ("teacher_p", "residual", "risk")
                      else pool[source])
        result[destination] = np.concatenate((main[destination], np.asarray(additional)[mask]))
    result["labels"] = {
        target: np.concatenate((main["labels"][target],
                                np.asarray(pool["labels"][target])[mask]))
        for target in roles.CLASS_COUNT}
    if len({str(identifier) for identifier in result["ids"]}) != len(result["ids"]):
        raise ValueError("duplicate original person across five outer pools")
    if (len(result["ids"]) != census.get("people") or
            len({str(h) for h in result["households"]}) != census.get("households") or
            not np.isclose(np.sum(result["weights"], dtype=np.float64),
                           float(census.get("weight_sum", float("nan"))),
                           rtol=1e-10, atol=1e-5)):
        raise ValueError("five-pool outer rows differ from registered census")
    if (result["ha"].shape[1] != 4 or result["hb"].shape[1] != 2 or
            result["x"].shape[1] != 32 or result["risk"].shape[1] != 11 or
            np.any(result["token_codes"] < 0) or np.any(result["token_codes"] >= 32)):
        raise ValueError("five-pool frozen representation or service schema changed")
    for target, classes in roles.CLASS_COUNT.items():
        observed = result["labels"][target]
        if np.any(observed < 0) or np.any(observed >= classes):
            raise ValueError("five-pool protected/task class schema changed")
    return result


def load_verified_outer(index: dict, anchor: int, census: Mapping[str, Any],
                        original_restore_root: str | Path) -> dict:
    """Unpickle only a pinned 2018 original after the caller's outer gate."""
    sanitized = data.load_prepared(index, anchor)
    if "labels" in sanitized["ctx"]["pools"][FIFTH_POOL]:
        raise ValueError("sanitized working copy unexpectedly exposes fifth-pool labels")
    source = data.verified_member(data.member_record(index, anchor, "prepared"),
                                  restore_root=original_restore_root)
    original = joblib.load(source)
    if (original.get("ctx", {}).get("anchor") != anchor or
            tuple(original["ctx"]["pools"]) != data.POOLS):
        raise ValueError("original prepared anchor or pool schema changed")
    left = sanitized["ctx"]["pools"][FIFTH_POOL]
    right = original["ctx"]["pools"][FIFTH_POOL]
    for field in FIELDS:
        if field in left or field in right:
            if field not in left or field not in right or not _same_bytes(left[field], right[field]):
                raise ValueError(f"fifth-pool original/sanitized {field} differs")
    for field in ENCODED:
        if not _same_bytes(sanitized["encoded"][FIFTH_POOL][field],
                           original["encoded"][FIFTH_POOL][field]):
            raise ValueError("fifth-pool stored encoding differs")
    if not _same_bytes(sanitized["encoded"][FIFTH_POOL]["codes"]["T0"],
                       original["encoded"][FIFTH_POOL]["codes"]["T0"]):
        raise ValueError("fifth-pool stored T0 codes differ")
    sanitized["ctx"]["pools"][FIFTH_POOL] = right
    return pooled_locked_outer(sanitized, census)
