"""Independent arithmetic for the eight fitted A public score laws.

This module reads no roles and imports no candidate or endpoint scorer. R03
workers supply hash-bound policy arrays and source scores after admission.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .independent_a_source import sha_file


KINDS = {"A_JOINT_STOCH": "stochastic_joint_v1",
         "A_TASK_STOCH": "stochastic_joint_v1",
         "A_TASK_DET": "deterministic_argmax_v1",
         "A_JOINT_DET": "deterministic_argmax_v1",
         "A_CENTRAL_PRODUCT": "central_product_v1",
         "A_LOCAL": "strict_local_product_v1",
         "A_SEQ12": "freeze_first_conditional_v1",
         "A_SEQ21": "freeze_first_conditional_v1"}
BASE = {"bank1_probability", "bank1_decision", "bank1_old_to_global",
        "bank2_probability", "bank2_decision", "bank2_old_to_global",
        "actions1", "actions2"}
FACTOR = ("w1", "b1", "w2", "b2")


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _softmax(logit: np.ndarray) -> np.ndarray:
    z = np.exp(logit - np.max(logit))
    return z / z.sum()


def _features(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    def summary(score: np.ndarray) -> tuple[float, float]:
        entropy = float(-np.sum(score * np.log(np.clip(score, 1e-12, 1))))
        ordered = np.sort(score)
        return entropy, float(ordered[-1] - ordered[-2])
    return np.r_[s1, s2, summary(s1), summary(s2)]


def _bank_hash(p: np.ndarray, d: np.ndarray) -> str:
    digest = hashlib.sha256(b"SCV1-canonical-bank-v1\n")
    digest.update(_json(list(p.shape)) + b"\n")
    digest.update(np.ascontiguousarray(p, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(d, dtype="<i8").tobytes())
    return digest.hexdigest()


def _action_hash(actions: tuple[tuple[int, ...], ...]) -> str:
    payload = json.dumps([list(row) for row in actions], separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return hashlib.sha256(b"SCV1-action-map-v1\n" + payload).hexdigest()


def _mechanism_hash(receipt: dict, arrays: dict, actions1, actions2) -> str:
    kind = receipt["law_kind"]
    if kind in ("stochastic_joint_v1", "deterministic_argmax_v1"):
        names = list(FACTOR)
        descriptor = {"law_kind": kind}
    else:
        prefixes = (("factor1", "factor2") if kind in
                    ("central_product_v1", "strict_local_product_v1")
                    else ("first", "second"))
        names = [f"{prefix}_{name}" for prefix in prefixes for name in FACTOR]
        if kind == "freeze_first_conditional_v1":
            names.append("second_embedding")
        descriptor = {"law_kind": kind,
                      "parameter_count": sum(arrays[name].size for name in names)}
        if kind == "freeze_first_conditional_v1":
            registered = receipt["mechanism_descriptor"]
            descriptor["order"] = registered["order"]
            descriptor["first_checkpoint_sha256"] = registered["first_checkpoint_sha256"]
    if "mechanism_descriptor" in receipt and receipt["mechanism_descriptor"] != descriptor:
        raise ValueError("A selected mechanism descriptor differs")
    digest = hashlib.sha256(b"SCV1-joint-mechanism-v1\n")
    digest.update(_json({"policy": descriptor, "parameter_count": len(names),
                         "bank1": _bank_hash(arrays["bank1_probability"], arrays["bank1_decision"]),
                         "bank2": _bank_hash(arrays["bank2_probability"], arrays["bank2_decision"]),
                         "actions1": actions1, "actions2": actions2}) + b"\n")
    for name in names:
        value = arrays[name]
        if value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError("A mechanism weights must be finite float64")
        canonical = np.ascontiguousarray(value, dtype="<f8")
        digest.update(_json(list(canonical.shape)) + b"\n")
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _actions(padded: np.ndarray, classes: np.ndarray) -> tuple[tuple[int, ...], ...]:
    if padded.dtype != np.int64 or padded.ndim != 2 or padded.shape[1] != 4:
        raise ValueError("A action map is not int64 fine-cell by four")
    rows = []
    for row in padded:
        used = row[row >= 0]
        if (not len(used) or len(set(used.tolist())) != len(used)
                or np.any(used >= len(classes))
                or not np.array_equal(row, np.r_[used, np.full(4 - len(used), -1)])
                or len(set(classes[used].tolist())) != 1):
            raise ValueError("A fine-cell actions are not canonical same-class choices")
        rows.append(tuple(int(x) for x in used))
    return tuple(rows)


def _factor_logits(arrays: dict, prefix: str, x: np.ndarray,
                   *, code: int | None = None) -> np.ndarray:
    w1, b1, w2, b2 = (arrays[f"{prefix}_{name}"] for name in FACTOR)
    hidden = x @ w1 + b1
    embedding = arrays.get(f"{prefix}_embedding")
    if embedding is not None:
        if code is None or not 0 <= code < len(embedding):
            raise ValueError("A conditional factor first code missing")
        hidden = hidden + embedding[code]
    elif code is not None:
        raise ValueError("A unconditional factor received first code")
    return np.tanh(hidden) @ w2 + b2


@dataclass(frozen=True)
class IndependentAPolicy:
    receipt: dict
    arrays: dict[str, np.ndarray]
    actions1: tuple[tuple[int, ...], ...]
    actions2: tuple[tuple[int, ...], ...]

    @property
    def kind(self) -> str:
        return self.receipt["law_kind"]

    def law(self, score1: np.ndarray, score2: np.ndarray, cell1: int, cell2: int
            ) -> tuple[tuple[int, int, float], ...]:
        s1 = np.asarray(score1, dtype=np.float64)
        s2 = np.asarray(score2, dtype=np.float64)
        p1, p2 = self.arrays["bank1_probability"], self.arrays["bank2_probability"]
        if (s1.shape != (p1.shape[1],) or s2.shape != (p2.shape[1],)
                or not np.isfinite(s1).all() or not np.isfinite(s2).all()
                or np.any(s1 < 0) or np.any(s2 < 0)
                or abs(s1.sum() - 1) > 1e-12 or abs(s2.sum() - 1) > 1e-12
                or not 0 <= cell1 < len(self.actions1)
                or not 0 <= cell2 < len(self.actions2)):
            raise ValueError("A input score or fine cell invalid")
        a1, a2 = self.actions1[cell1], self.actions2[cell2]
        if (int(np.argmax(s1)) != int(self.arrays["bank1_decision"][a1[0]])
                or int(np.argmax(s2)) != int(self.arrays["bank2_decision"][a2[0]])):
            raise ValueError("A source decision differs from public action class")
        central = _features(s1, s2)
        kind = self.kind
        if kind in ("stochastic_joint_v1", "deterministic_argmax_v1"):
            g = self.arrays
            logits = np.tanh(central @ g["w1"] + g["b1"]) @ g["w2"] + g["b2"]
            slots = [4 * i + j for i in range(len(a1)) for j in range(len(a2))]
            pairs = [(i, j) for i in a1 for j in a2]
            selected = logits[slots]
            if kind == "deterministic_argmax_v1":
                i, j = pairs[int(np.argmax(selected))]
                return ((int(i), int(j), 1.0),)
            return _aggregate(pairs, _softmax(selected))
        if kind in ("central_product_v1", "strict_local_product_v1"):
            x1, x2 = ((central, central) if kind == "central_product_v1"
                      else (np.r_[s1, _features(s1, s1)[-2:]],
                            np.r_[s2, _features(s2, s2)[-2:]]))
            m1 = _softmax(_factor_logits(self.arrays, "factor1", x1)[:len(a1)])
            m2 = _softmax(_factor_logits(self.arrays, "factor2", x2)[:len(a2)])
            return _aggregate([(i, j) for i in a1 for j in a2],
                              np.outer(m1, m2).ravel())
        if kind == "freeze_first_conditional_v1":
            order = self.receipt["mechanism_descriptor"]["order"]
            first_choices, second_choices = (a1, a2) if order == "12" else (a2, a1)
            first = _softmax(_factor_logits(self.arrays, "first", central)[:len(first_choices)])
            pairs, masses = [], []
            for first_code, first_mass in zip(first_choices, first):
                conditional = _softmax(_factor_logits(
                    self.arrays, "second", central, code=first_code)[:len(second_choices)])
                for second_code, second_mass in zip(second_choices, conditional):
                    pairs.append((first_code, second_code) if order == "12"
                                 else (second_code, first_code))
                    masses.append(first_mass * second_mass)
            return _aggregate(pairs, masses)
        raise ValueError("A policy law kind unregistered")

    def sample(self, score1: np.ndarray, score2: np.ndarray,
               cell1: int, cell2: int, uniform: float) -> tuple[int, int]:
        law = self.law(score1, score2, cell1, cell2)
        if not np.isfinite(uniform) or not 0 <= uniform < 1:
            raise ValueError("A realization uniform outside [0,1)")
        if self.kind != "freeze_first_conditional_v1":
            cumulative = np.cumsum([mass for _, _, mass in law])
            index = min(int(np.searchsorted(cumulative, uniform, side="right")), len(law) - 1)
            return law[index][:2]
        # Sequential policies spend one uniform on the first public code and
        # conditionally rescale its remainder for the second recipient.
        order = self.receipt["mechanism_descriptor"]["order"]
        choices1, choices2 = self.actions1[cell1], self.actions2[cell2]
        first_choices, second_choices = ((choices1, choices2) if order == "12"
                                         else (choices2, choices1))
        central = _features(np.asarray(score1), np.asarray(score2))
        first_mass = _softmax(_factor_logits(
            self.arrays, "first", central)[:len(first_choices)])
        first = sorted(zip(first_choices, first_mass))
        prior = 0.0
        for index, (code, mass) in enumerate(first):
            if uniform < prior + mass or index == len(first) - 1:
                first_code = code
                rest = min(np.nextafter(1.0, 0.0), max(0.0, (uniform - prior) / mass))
                break
            prior += mass
        second_mass = _softmax(_factor_logits(
            self.arrays, "second", central, code=first_code)[:len(second_choices)])
        second = sorted(zip(second_choices, second_mass))
        cumulative = 0.0
        for index, (code, mass) in enumerate(second):
            cumulative += mass
            if rest < cumulative or index == len(second) - 1:
                return ((first_code, code) if order == "12" else (code, first_code))
        raise AssertionError("A sequential second-stage law empty")

    def expected_losses(self, law, y1: int, y2: int) -> dict:
        p1, p2 = self.arrays["bank1_probability"], self.arrays["bank2_probability"]
        out = {"r1": {"nll": 0.0, "brier": 0.0},
               "r2": {"nll": 0.0, "brier": 0.0}}
        for i, j, mass in law:
            for name, p, code, truth in (("r1", p1, i, y1), ("r2", p2, j, y2)):
                vector = p[code]
                out[name]["nll"] += mass * -np.log(max(vector[truth], 1e-12))
                out[name]["brier"] += mass * float(np.sum(vector * vector) + 1 - 2 * vector[truth])
        return out


def _aggregate(pairs, mass) -> tuple[tuple[int, int, float], ...]:
    grouped = {}
    for pair, value in zip(pairs, mass):
        grouped[pair] = grouped.get(pair, 0.0) + float(value)
    result = tuple((int(i), int(j), grouped[i, j]) for i, j in sorted(grouped))
    if not result or abs(sum(row[2] for row in result) - 1) > 1e-12:
        raise ValueError("A emitted pair law not normalized")
    return result


def load_independent_policy(directory: Path, expected_receipt_sha256: str) -> IndependentAPolicy:
    root = Path(directory).resolve(strict=True)
    receipt_path, package = root / "POLICY_RECEIPT.json", root / "policy.npz"
    if sha_file(receipt_path) != expected_receipt_sha256:
        raise ValueError("A selected policy receipt SHA differs")
    receipt = json.loads(receipt_path.read_text())
    arm = receipt.get("arm_id")
    if (arm not in KINDS or receipt.get("status") != "SELECTED_FIT_POLICY"
            or receipt.get("law_kind") != KINDS[arm]
            or sha_file(package) != receipt.get("policy_file_sha256")):
        raise ValueError("A selected policy identity differs")
    kind = receipt["law_kind"]
    keys = set(BASE)
    if kind in ("stochastic_joint_v1", "deterministic_argmax_v1"):
        keys |= set(FACTOR)
    else:
        keys |= {f"gate_{name}" for name in FACTOR}
        prefixes = (("factor1", "factor2") if kind in
                    ("central_product_v1", "strict_local_product_v1")
                    else ("first", "second"))
        keys |= {f"{prefix}_{name}" for prefix in prefixes for name in FACTOR}
        if kind == "freeze_first_conditional_v1":
            keys.add("second_embedding")
    with ZipFile(package) as archive:
        if {member.filename for member in archive.infolist()} != {name + ".npy" for name in keys}:
            raise ValueError("A selected policy NPZ members differ")
    with np.load(package, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in keys}
    actions = {}
    for r in (1, 2):
        p, d = arrays[f"bank{r}_probability"], arrays[f"bank{r}_decision"]
        old = arrays[f"bank{r}_old_to_global"]
        if (p.dtype != np.float64 or d.dtype != np.int64 or old.dtype != np.int64
                or p.ndim != 2 or d.shape != (len(p),) or old.ndim != 1
                or np.any((old < 0) | (old >= len(p)))
                or not np.isfinite(p).all() or np.any(p < 0)
                or np.max(np.abs(p.sum(axis=1) - 1)) > 1e-12
                or not np.array_equal(np.argmax(p, axis=1), d)
                or _bank_hash(p, d) != receipt["bank_sha256_by_recipient"][f"r{r}"]):
            raise ValueError("A canonical public bank differs")
        actions[r] = _actions(arrays[f"actions{r}"], d)
        if _action_hash(actions[r]) != receipt["action_sha256_by_recipient"][f"r{r}"]:
            raise ValueError("A input-local action digest differs")
    if _mechanism_hash(receipt, arrays, actions[1], actions[2]) != receipt["mechanism_sha256"]:
        raise ValueError("A selected mechanism hash differs")
    return IndependentAPolicy(receipt, arrays, actions[1], actions[2])
