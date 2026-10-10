"""Deployable product, local and freeze-first laws on the fixed public bank.

These are policy classes and arithmetic, not fitted-control evidence. The
fitting worker must supply separately optimized and frozen FactorGate weights.
"""

from dataclasses import dataclass
import re

import numpy as np

if __package__:
    from ..impl.channel import JointPolicy, _entropy_margin, aggregate_public_pairs
else:
    from channel import JointPolicy, _entropy_margin, aggregate_public_pairs


@dataclass(frozen=True)
class FactorGate:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    embedding: np.ndarray | None = None

    def __post_init__(self):
        arrays = tuple(np.asarray(v, dtype=np.float64) for v in
                       (self.w1, self.b1, self.w2, self.b2))
        if (arrays[0].ndim != 2 or arrays[0].shape[1] != 16 or
                arrays[1].shape != (16,) or arrays[2].shape != (16, 4) or
                arrays[3].shape != (4,) or
                not all(np.isfinite(v).all() for v in arrays)):
            raise ValueError("factor gate must have input→16→4 finite parameters")
        for name, value in zip(("w1", "b1", "w2", "b2"), arrays):
            frozen = np.ascontiguousarray(value).copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        if self.embedding is not None:
            embed = np.asarray(self.embedding, dtype=np.float64)
            if (embed.ndim != 2 or embed.shape[1] != 16 or len(embed) < 1 or
                    not np.isfinite(embed).all()):
                raise ValueError("first-code embedding must be finite Z×16")
            frozen = np.ascontiguousarray(embed).copy()
            frozen.setflags(write=False)
            object.__setattr__(self, "embedding", frozen)

    @property
    def parameter_count(self):
        return sum(v.size for v in (self.w1, self.b1, self.w2, self.b2)) + \
            (0 if self.embedding is None else self.embedding.size)

    def logits(self, features, *, first_code=None):
        x = np.asarray(features, dtype=np.float64)
        if x.shape != (self.w1.shape[0],):
            raise ValueError("factor input width differs from fitted gate")
        hidden = x @ self.w1 + self.b1
        if self.embedding is not None:
            if type(first_code) not in (int, np.int64) or not 0 <= first_code < len(self.embedding):
                raise ValueError("sampled first code missing or outside public bank")
            hidden = hidden + self.embedding[first_code]
        elif first_code is not None:
            raise ValueError("unconditioned factor cannot receive a first code")
        return np.tanh(hidden) @ self.w2 + self.b2


def initial_factor_gate(input_width, rng, *, identity_biased,
                        init_scale=.01, identity_bias=4.0, embedding=None):
    """Frozen two-start geometry: random hidden layer, exact-zero output W2."""
    if (type(input_width) is not int or input_width < 1 or
            not isinstance(rng, np.random.Generator) or
            type(identity_biased) is not bool or
            not np.isfinite(init_scale) or init_scale <= 0 or
            not np.isfinite(identity_bias) or identity_bias <= 0):
        raise ValueError("invalid fixed factor initialization")
    w1 = rng.normal(0, init_scale, size=(input_width, 16))
    b1 = rng.normal(0, init_scale, size=(16,))
    w2 = np.zeros((16, 4), dtype=np.float64)
    b2 = np.zeros(4, dtype=np.float64)
    if identity_biased:
        b2[0] = identity_bias
    return FactorGate(w1, b1, w2, b2, embedding)


def _factor_law(gate, features, choices, *, first_code=None):
    logits = gate.logits(features, first_code=first_code)[:len(choices)]
    mass = np.exp(logits - np.max(logits))
    mass /= mass.sum()
    return tuple((int(code), float(value)) for code, value in zip(choices, mass))


def _central_features(score1, score2):
    return np.r_[score1, score2, _entropy_margin(score1), _entropy_margin(score2)]


def _arrays(gate):
    values = [gate.w1, gate.b1, gate.w2, gate.b2]
    if gate.embedding is not None:
        values.append(gate.embedding)
    return tuple(values)


def public_probability_embedding(bank_probability):
    """Fixed E0 for every first public code, including codes unseen in FIT draws."""
    probability = np.asarray(bank_probability, dtype=np.float64)
    if (probability.ndim != 2 or min(probability.shape) < 1 or
            not np.isfinite(probability).all() or
            np.any(probability < 0) or
            not np.allclose(probability.sum(axis=1), 1, atol=1e-12, rtol=0)):
        raise ValueError("first public bank probability table is invalid")
    return np.ascontiguousarray(probability[:, np.arange(16) % probability.shape[1]])


class ProductPolicy(JointPolicy):
    law_kind = "central_product_v1"

    def __init__(self, bank1, bank2, actions1, actions2, gate,
                 factor1: FactorGate, factor2: FactorGate):
        super().__init__(bank1, bank2, actions1, actions2, gate)
        width = bank1.probability.shape[1] + bank2.probability.shape[1] + 4
        if (not isinstance(factor1, FactorGate) or not isinstance(factor2, FactorGate) or
                factor1.w1.shape[0] != width or factor2.w1.shape[0] != width or
                factor1.embedding is not None or factor2.embedding is not None):
            raise ValueError("central factors must read the same two-score feature width")
        self.factor1, self.factor2 = factor1, factor2

    def mechanism_descriptor(self):
        return {"law_kind": self.law_kind,
                "parameter_count": self.factor1.parameter_count + self.factor2.parameter_count}

    def mechanism_arrays(self):
        return _arrays(self.factor1) + _arrays(self.factor2)

    def _factor_features(self, score1, score2):
        both = _central_features(score1, score2)
        return both, both

    def sparse_law(self, score1, score2, cell1, cell2):
        self._pairs_and_probabilities(score1, score2, cell1, cell2)  # fixed-source validation
        x1, x2 = self._factor_features(score1, score2)
        q1 = _factor_law(self.factor1, x1, self.actions1[cell1])
        q2 = _factor_law(self.factor2, x2, self.actions2[cell2])
        pair_mass = tuple(((i, j), p1 * p2) for i, p1 in q1 for j, p2 in q2)
        return aggregate_public_pairs((pair for pair, _ in pair_mass),
                                      np.asarray([mass for _, mass in pair_mass],
                                                 dtype=np.float64))


class LocalPolicy(ProductPolicy):
    law_kind = "strict_local_product_v1"

    def __init__(self, bank1, bank2, actions1, actions2, gate, factor1, factor2):
        # JointPolicy validates the shared bank/action service. Each trained
        # factor itself receives only its own score and fine cell.
        JointPolicy.__init__(self, bank1, bank2, actions1, actions2, gate)
        if (not isinstance(factor1, FactorGate) or not isinstance(factor2, FactorGate) or
                factor1.w1.shape[0] != bank1.probability.shape[1] + 2 or
                factor2.w1.shape[0] != bank2.probability.shape[1] + 2 or
                factor1.embedding is not None or factor2.embedding is not None):
            raise ValueError("local factors must each use only own score plus summaries")
        self.factor1, self.factor2 = factor1, factor2

    def _factor_features(self, score1, score2):
        return (np.r_[score1, _entropy_margin(score1)],
                np.r_[score2, _entropy_margin(score2)])


class SequentialPolicy(JointPolicy):
    law_kind = "freeze_first_conditional_v1"

    def __init__(self, bank1, bank2, actions1, actions2, gate,
                 first: FactorGate, second: FactorGate, *,
                 order: str, first_checkpoint_sha256: str):
        super().__init__(bank1, bank2, actions1, actions2, gate)
        width = bank1.probability.shape[1] + bank2.probability.shape[1] + 4
        z_first = len(bank1.probability) if order == "12" else len(bank2.probability)
        if (order not in ("12", "21") or not isinstance(first, FactorGate) or
                not isinstance(second, FactorGate) or
                first.w1.shape[0] != width or second.w1.shape[0] != width or
                first.embedding is not None or second.embedding is None or
                second.embedding.shape != (z_first, 16) or
                re.fullmatch(r"[0-9a-f]{64}", first_checkpoint_sha256) is None):
            raise ValueError("freeze-first policy needs central factors and public-code embedding")
        self.first, self.second = first, second
        self.order = order
        self.first_checkpoint_sha256 = first_checkpoint_sha256

    def mechanism_descriptor(self):
        return {"law_kind": self.law_kind, "order": self.order,
                "first_checkpoint_sha256": self.first_checkpoint_sha256,
                "parameter_count": self.first.parameter_count + self.second.parameter_count}

    def mechanism_arrays(self):
        return _arrays(self.first) + _arrays(self.second)

    def sparse_law(self, score1, score2, cell1, cell2):
        self._pairs_and_probabilities(score1, score2, cell1, cell2)
        features = _central_features(score1, score2)
        if self.order == "12":
            first_choices, second_choices = self.actions1[cell1], self.actions2[cell2]
        else:
            first_choices, second_choices = self.actions2[cell2], self.actions1[cell1]
        first = _factor_law(self.first, features, first_choices)
        masses = {}
        for first_code, first_mass in first:
            second = _factor_law(self.second, features, second_choices,
                                 first_code=first_code)
            for second_code, conditional_mass in second:
                pair = ((first_code, second_code) if self.order == "12"
                        else (second_code, first_code))
                masses[pair] = masses.get(pair, 0.0) + first_mass * conditional_mass
        return tuple((i, j, mass) for (i, j), mass in sorted(masses.items()))

    def sample_public_pair(self, score1, score2, cell1, cell2, uniform):
        """Use actual first public draw as the second-stage runtime input."""
        self._pairs_and_probabilities(score1, score2, cell1, cell2)
        if (not np.isfinite(uniform) or uniform < 0 or uniform >= 1):
            raise ValueError("sampling uniform must lie in [0,1)")
        features = _central_features(score1, score2)
        if self.order == "12":
            first_choices, second_choices = self.actions1[cell1], self.actions2[cell2]
        else:
            first_choices, second_choices = self.actions2[cell2], self.actions1[cell1]
        first = sorted(_factor_law(self.first, features, first_choices))
        prior = 0.0
        for code, mass in first:
            if uniform < prior + mass or code == first[-1][0]:
                first_code = code
                conditional_uniform = min(np.nextafter(1.0, 0.0),
                                          max(0.0, (uniform - prior) / mass))
                break
            prior += mass
        second = sorted(_factor_law(self.second, features, second_choices,
                                    first_code=first_code))
        cumulative = 0.0
        for code, mass in second:
            cumulative += mass
            if conditional_uniform < cumulative or code == second[-1][0]:
                return ((first_code, code) if self.order == "12"
                        else (code, first_code))
        raise AssertionError("second-stage conditional law was empty")
