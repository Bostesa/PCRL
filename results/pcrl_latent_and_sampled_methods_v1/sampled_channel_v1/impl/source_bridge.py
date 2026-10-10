"""Pure binding of admitted role arrays to the sampled score fit.

R01's role-only stores must hash-check every source file before calling here.
This module reads no files and does not grant Adult fit admission. Its
ExpectedIdentity values must come from a later R00 A-stage lock.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

import numpy as np

if __package__:
    from .bank import FixedBank, build_actions
    from .channel import GateWeights, JointPolicy
    from .fit import ProtectedData, TaskOnlyData, prepare_law
    from .release import _bank_hash
else:
    from bank import FixedBank, build_actions
    from channel import GateWeights, JointPolicy
    from fit import ProtectedData, TaskOnlyData, prepare_law
    from release import _bank_hash


HEX64 = re.compile(r"[0-9a-f]{64}\Z")
BANK_KEYS = frozenset({"probability", "token_class", "token_id", "input_class",
                       "fine_to_hcal_token", "canonical_action_id"})
CELL_KEYS = frozenset({"row_id", "unit", "cell_1", "cell_2", "token_1", "token_2"})
REFERENCE_KEYS = frozenset({"row_id", "unit", "U0_1", "U0_2", "Ucal_1",
                            "Ucal_2", "d_1", "d_2"})


def sha_array(value) -> str:
    """Match R01's admitted array digest convention, including dtype/shape."""
    array = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(array.dtype).encode() + str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def action_map_sha256(actions) -> str:
    normalized = [[int(code) for code in row] for row in actions]
    payload = json.dumps(normalized, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return hashlib.sha256(b"SCV1-action-map-v1\n" + payload).hexdigest()


@dataclass(frozen=True)
class ExpectedIdentity:
    source_binding_sha256: str
    row_unit_pair_sha256: str
    bank_sha256: tuple[str, str]
    action_sha256: tuple[str, str]

    def __post_init__(self):
        values = (self.source_binding_sha256, self.row_unit_pair_sha256,
                  *self.bank_sha256, *self.action_sha256)
        if (len(self.bank_sha256) != 2 or len(self.action_sha256) != 2 or
                any(type(v) is not str or HEX64.fullmatch(v) is None
                    for v in values)):
            raise ValueError("all source, row, bank and action hashes must be sealed")


def _exact_keys(record, keys, name):
    if type(record) is not dict or record.keys() != keys:
        raise ValueError(f"{name} has extra or missing fields")


def _row_pair(record, n):
    row, unit = record["row_id"], record["unit"]
    if (not isinstance(row, np.ndarray) or row.dtype != np.int64 or
            not isinstance(unit, np.ndarray) or unit.dtype != np.int64 or
            row.shape != (n,) or unit.shape != (n,) or
            not np.all(np.diff(row) > 0)):
        raise ValueError("role row and group identities differ")
    return sha_array(np.column_stack((row, unit)).astype(np.int64))


def _reference_cost(score, labels):
    n, k = score.shape
    if (score.dtype != np.float64 or labels.dtype != np.int64 or
            labels.shape != (n,) or not np.isfinite(score).all() or
            np.any(score < 0) or
            np.max(np.abs(score.sum(axis=1) - 1)) > 1e-12 or
            np.any(labels < 0) or np.any(labels >= k)):
        raise ValueError("reference score/label alignment differs")
    selected = score[np.arange(n), labels]
    nll = float(np.mean(-np.log(np.maximum(selected, 1e-12))))
    target = np.eye(k)[labels]
    brier = float(np.mean(np.sum((score - target) ** 2, axis=1)))
    return nll, brier


def bind_task_inputs(bank_record_1, bank_record_2, cells, references,
                     income, occupation, *, source_binding_sha256: str,
                     expected: ExpectedIdentity):
    """Bind one FIT role without accepting or materializing a SEX member."""
    if not isinstance(expected, ExpectedIdentity) or source_binding_sha256 != \
            expected.source_binding_sha256:
        raise ValueError("source binding differs from later A-stage lock")
    for name, record in (("r1 bank", bank_record_1), ("r2 bank", bank_record_2)):
        _exact_keys(record, BANK_KEYS, name)
    _exact_keys(cells, CELL_KEYS, "fine-cell role")
    _exact_keys(references, REFERENCE_KEYS, "reference role")
    _exact_keys(income, frozenset({"row_id", "unit", "income"}), "income label")
    _exact_keys(occupation, frozenset({"row_id", "unit", "occupation"}),
                "occupation label")
    n = len(cells["row_id"])
    if (n < 1 or any(_row_pair(record, n) != expected.row_unit_pair_sha256
                     for record in (cells, references, income, occupation)) or
            any(not np.array_equal(cells[key], record[key])
                for record in (references, income, occupation)
                for key in ("row_id", "unit"))):
        raise ValueError("task-only role rows differ from sealed FIT people")
    bank_records = (bank_record_1, bank_record_2)
    banks, actions = [], []
    for r, record in enumerate(bank_records, start=1):
        bank = FixedBank.from_source(record["probability"], record["token_id"],
                                     record["token_class"])
        input_class, fine_map = record["input_class"], record["fine_to_hcal_token"]
        if (record["canonical_action_id"].dtype != np.int64 or
                not np.array_equal(record["canonical_action_id"], record["token_id"]) or
                input_class.dtype != np.int64 or fine_map.dtype != np.int64 or
                input_class.ndim != 1 or fine_map.shape != input_class.shape or
                set(bank.decision.tolist()) != set(range(bank.probability.shape[1])) or
                np.any((fine_map < 0) | (fine_map >= len(bank.old_to_global))) or
                not np.array_equal(input_class, record["token_class"][fine_map])):
            raise ValueError("fine-cell identity or class coverage differs")
        choices = build_actions(bank, input_class=input_class,
                                fine_to_old_token=fine_map)
        if (_bank_hash(bank) != expected.bank_sha256[r - 1] or
                action_map_sha256(choices) != expected.action_sha256[r - 1]):
            raise ValueError("bank or action map differs from later A-stage lock")
        cell, token = cells[f"cell_{r}"], cells[f"token_{r}"]
        if (cell.dtype != np.int64 or token.dtype != np.int64 or
                cell.shape != (n,) or token.shape != (n,) or
                np.any((cell < 0) | (cell >= len(choices))) or
                not np.array_equal(token, fine_map[cell])):
            raise ValueError("private role cells differ from frozen identity map")
        banks.append(bank)
        actions.append(choices)
    task = TaskOnlyData(score1=references["U0_1"], score2=references["U0_2"],
                        cell1=cells["cell_1"], cell2=cells["cell_2"],
                        y1=income["income"], y2=occupation["occupation"])
    policy = JointPolicy(banks[0], banks[1], actions[0], actions[1],
                         GateWeights.zeros(input_width=task.score1.shape[1] +
                                                        task.score2.shape[1] + 4))
    # The full-row pass validates each source decision against its action class.
    prepare_law(task, policy)
    reference = {}
    caps = {}
    for r, labels in ((1, task.y1), (2, task.y2)):
        decision = references[f"d_{r}"]
        if (decision.dtype != np.int64 or decision.shape != (n,) or
                not np.array_equal(decision, np.argmax(references[f"U0_{r}"], axis=1))):
            raise ValueError("frozen U0 decision identity differs")
        for name in ("U0", "Ucal"):
            score = references[f"{name}_{r}"]
            if not np.array_equal(np.argmax(score, axis=1), decision):
                raise ValueError("calibration-matched decision identity differs")
            nll, brier = _reference_cost(score, labels)
            for metric, value in (("nll", nll), ("brier", brier)):
                reference.setdefault(f"r{r}_{metric}", {})[name] = value
        for metric, allowance in (("nll", .010), ("brier", .005)):
            key = f"r{r}_{metric}"
            caps[key] = min(reference[key].values()) + allowance
    receipt = {"source_binding_sha256": source_binding_sha256,
               "row_unit_pair_sha256": expected.row_unit_pair_sha256,
               "bank_sha256": expected.bank_sha256,
               "action_sha256": expected.action_sha256,
               "references": reference,
               "role_rows": n}
    return task, policy, caps, receipt


def attach_protected(task: TaskOnlyData, sex_record: dict, *,
                     expected_row_unit_sha256: str) -> ProtectedData:
    """Protected worker only; caller first obtains admitted FIT SEX role file."""
    _exact_keys(sex_record, frozenset({"row_id", "unit", "sex"}), "SEX label")
    if _row_pair(sex_record, len(task.y1)) != expected_row_unit_sha256:
        raise ValueError("protected label people differ from bound task FIT rows")
    return ProtectedData(task=task, sex=sex_record["sex"])
