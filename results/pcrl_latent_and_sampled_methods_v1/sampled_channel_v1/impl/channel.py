"""Small central score channel over actual, global public output identities.

The private provider uses two scores and fine-cell IDs. Recipients receive
only a sampled global bank ID with its fixed probability and decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

if __package__:
    from .bank import FixedBank
else:  # Standalone data-free engineering tests.
    from bank import FixedBank


CLIP = 1e-12
WIDTH = 16


def _score(value, classes: int) -> np.ndarray:
    score = np.asarray(value)
    if (score.dtype != np.dtype("float64") or score.shape != (classes,) or
            not np.isfinite(score).all() or
            np.any(score < 0) or not np.isclose(score.sum(), 1, atol=1e-12, rtol=0)):
        raise ValueError("private source score must be a probability vector")
    return score


def _entropy_margin(score: np.ndarray) -> tuple[float, float]:
    entropy = float(-np.sum(score * np.log(np.clip(score, CLIP, 1))))
    ordered = np.sort(score)
    margin = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 1.0
    return entropy, margin


@dataclass(frozen=True)
class GateWeights:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def __post_init__(self):
        arrays = tuple(np.asarray(x, dtype=np.float64) for x in
                       (self.w1, self.b1, self.w2, self.b2))
        if (arrays[0].ndim != 2 or arrays[0].shape[1] != WIDTH or
                arrays[1].shape != (WIDTH,) or arrays[2].shape != (WIDTH, WIDTH) or
                arrays[3].shape != (WIDTH,) or
                not all(np.isfinite(x).all() for x in arrays)):
            raise ValueError("gate requires finite input×16, 16, 16×16, 16 weights")
        for name, value in zip(("w1", "b1", "w2", "b2"), arrays):
            frozen = value.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)

    @classmethod
    def zeros(cls, *, input_width: int) -> "GateWeights":
        if type(input_width) is not int or input_width < 6:
            raise ValueError("gate needs two score vectors and four summaries")
        return cls(np.zeros((input_width, WIDTH)), np.zeros(WIDTH),
                   np.zeros((WIDTH, WIDTH)), np.zeros(WIDTH))

    def logits(self, features: np.ndarray) -> np.ndarray:
        if features.shape != (self.w1.shape[0],):
            raise ValueError("gate feature width differs from frozen parameters")
        hidden = np.tanh(features @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2


def aggregate_public_pairs(pairs, probability) -> tuple[tuple[int, int, float], ...]:
    """Sum hidden action masses sharing an ACTUAL emitted pair of global IDs."""
    pairs = tuple(pairs)
    q = np.asarray(probability, dtype=np.float64)
    if (q.shape != (len(pairs),) or len(pairs) < 1 or
            not np.isfinite(q).all() or np.any(q < 0) or
            not np.isclose(q.sum(), 1, atol=1e-12, rtol=0)):
        raise ValueError("joint action probabilities must sum to one")
    grouped = {}
    for (i, j), mass in zip(pairs, q):
        if (type(i) not in (int, np.int64) or type(j) not in (int, np.int64) or
                i < 0 or j < 0):
            raise ValueError("public code IDs must be nonnegative integers")
        key = (int(i), int(j))
        grouped[key] = grouped.get(key, 0.0) + float(mass)
    return tuple((i, j, grouped[i, j]) for i, j in sorted(grouped))


class JointPolicy:
    law_kind = "stochastic_joint_v1"

    def mechanism_descriptor(self) -> dict:
        """Public structural choices; control subclasses extend this."""
        return {"law_kind": self.law_kind}

    def mechanism_arrays(self) -> tuple[np.ndarray, ...]:
        """Every fitted numeric parameter that can change the public law."""
        return (self.gate.w1, self.gate.b1, self.gate.w2, self.gate.b2)

    def __init__(self, bank1: FixedBank, bank2: FixedBank,
                 actions1, actions2, gate: GateWeights):
        if not isinstance(bank1, FixedBank) or not isinstance(bank2, FixedBank):
            raise TypeError("a joint policy requires two fixed public banks")
        if not isinstance(gate, GateWeights):
            raise TypeError("a joint policy requires frozen gate weights")
        if gate.w1.shape[0] != (bank1.probability.shape[1] +
                                bank2.probability.shape[1] + 4):
            raise ValueError("gate input width differs from the two task scores")
        parsed = []
        for bank, all_actions in ((bank1, actions1), (bank2, actions2)):
            if len(all_actions) < 1:
                raise ValueError("every fine-cell partition needs actions")
            rows = []
            for choices in all_actions:
                if any(type(i) not in (int, np.int64) for i in choices):
                    raise ValueError("action IDs must be exact integers, not rounded values")
                choices = tuple(int(i) for i in choices)
                if (not 1 <= len(choices) <= 4 or len(set(choices)) != len(choices) or
                        any(i < 0 or i >= len(bank.probability) for i in choices) or
                        any(bank.decision[i] != bank.decision[choices[0]] for i in choices)):
                    raise ValueError("each fine cell needs <=4 unique same-class codes")
                rows.append(choices)
            parsed.append(tuple(rows))
        self.bank1, self.bank2 = bank1, bank2
        self.actions1, self.actions2 = parsed
        self.gate = gate

    def _pairs_and_probabilities(self, score1, score2, cell1: int, cell2: int):
        s1 = _score(score1, self.bank1.probability.shape[1])
        s2 = _score(score2, self.bank2.probability.shape[1])
        if (type(cell1) not in (int, np.int64) or type(cell2) not in (int, np.int64) or
                cell1 < 0 or cell1 >= len(self.actions1) or
                cell2 < 0 or cell2 >= len(self.actions2)):
            raise ValueError("private fine-cell ID is outside the frozen action map")
        choices1, choices2 = self.actions1[cell1], self.actions2[cell2]
        if (int(np.argmax(s1)) != int(self.bank1.decision[choices1[0]]) or
                int(np.argmax(s2)) != int(self.bank2.decision[choices2[0]])):
            raise ValueError("source-score decision differs from fixed action class")
        pairs = tuple(product(choices1, choices2))
        if len(pairs) > 16:
            raise ValueError("joint action count exceeds sixteen")
        features = np.r_[s1, s2, _entropy_margin(s1), _entropy_margin(s2)]
        slots = [4 * a + b for a in range(len(choices1))
                 for b in range(len(choices2))]
        logits = self.gate.logits(features)[slots]
        q = np.exp(logits - np.max(logits))
        q /= q.sum()
        return pairs, q

    def sparse_law(self, score1, score2, cell1: int, cell2: int):
        """Private law over emitted global output IDs, not local action slots."""
        pairs, q = self._pairs_and_probabilities(score1, score2, cell1, cell2)
        return aggregate_public_pairs(pairs, q)

    def sample_public_pair(self, score1, score2, cell1: int, cell2: int,
                           uniform: float) -> tuple[int, int]:
        """Sample this policy's actual law; sequential controls override it."""
        if __package__:
            from .release import draw_pair
        else:
            from release import draw_pair
        return draw_pair(self.sparse_law(score1, score2, cell1, cell2), uniform)


def expected_cost(bank1: FixedBank, bank2: FixedBank, triples, *, y1: int, y2: int):
    """Expected loss of one actual sampled release, never an averaged vector."""
    if (type(y1) not in (int, np.int64) or type(y2) not in (int, np.int64) or
            y1 < 0 or y1 >= bank1.probability.shape[1] or
            y2 < 0 or y2 >= bank2.probability.shape[1]):
        raise ValueError("task labels are outside the bank class orders")
    rows = tuple(triples)
    if (len(rows) < 1 or not np.isclose(sum(p for _, _, p in rows), 1,
                                       atol=1e-12, rtol=0)):
        raise ValueError("sampled output law must be normalized")
    answer = {"r1": {"nll": 0.0, "brier": 0.0},
              "r2": {"nll": 0.0, "brier": 0.0}}
    for i, j, mass in rows:
        for name, bank, code, label in (("r1", bank1, i, y1),
                                        ("r2", bank2, j, y2)):
            if code < 0 or code >= len(bank.probability) or mass < 0:
                raise ValueError("output code or mass differs from the fixed bank")
            vector = bank.probability[code]
            answer[name]["nll"] += mass * -np.log(np.clip(vector[label], CLIP, 1))
            answer[name]["brier"] += mass * float(np.sum(
                (vector - np.eye(len(vector))[label]) ** 2))
    return answer


def bayes_recovery(releases, sex) -> dict[str, float]:
    """Exact empirical binary-SEX Bayes accuracy for three public views."""
    rows = tuple(tuple(row) for row in releases)
    labels = np.asarray(sex)
    if (labels.dtype != np.dtype("int64") or labels.shape != (len(rows),) or
            len(rows) < 1 or not np.isin(labels, (0, 1)).all()):
        raise ValueError("binary SEX labels must align with finite-law people")
    counts = {name: {} for name in ("r1", "r2", "pair")}
    for row, sensitive in zip(rows, labels):
        if (len(row) < 1 or
                any(type(i) not in (int, np.int64) or
                    type(j) not in (int, np.int64) or i < 0 or j < 0 or
                    not np.isfinite(p) or p < 0 for i, j, p in row) or
                not np.isclose(sum(p for _, _, p in row), 1, atol=1e-12, rtol=0)):
            raise ValueError("one person's sampled law is not normalized")
        for i, j, mass in row:
            for name, key in (("r1", i), ("r2", j), ("pair", (i, j))):
                value = counts[name].setdefault(key, np.zeros(2, dtype=np.float64))
                value[int(sensitive)] += mass / len(rows)
    return {name: float(sum(np.max(value) for value in cells.values()))
            for name, cells in counts.items()}
