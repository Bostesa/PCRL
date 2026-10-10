"""Exact bank-identity boundaries around the inherited sampled channel.

The source bank, actions, and fitted neural factors are unchanged. This module
only adds an exact identity point and, for product/local controls, independent
recipient-wise mixing. It is safe to import for data-free and admitted-role
checks; no label loader exists here.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import hmac
import json
from math import isfinite

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import (
    JointPolicy,
    _score,
    aggregate_public_pairs,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.release import (
    _uniform_from_uint64,
)


FACTOR_STREAM_SCHEMA = "BAR1-factor-uniform-v1"


def _eta(value: float) -> float:
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("identity boundary eta must be in [0,1]")
    return value


def identity_pair(base: JointPolicy, score1, score2, cell1: int, cell2: int) -> tuple[int, int]:
    """Return the source bank's exact first actions after full class checks."""
    if not isinstance(base, JointPolicy):
        raise TypeError("identity needs a fixed-bank inherited policy")
    a = _score(score1, base.bank1.probability.shape[1])
    b = _score(score2, base.bank2.probability.shape[1])
    if (type(cell1) not in (int, np.int64) or type(cell2) not in (int, np.int64) or
            not 0 <= cell1 < len(base.actions1) or
            not 0 <= cell2 < len(base.actions2)):
        raise ValueError("private fine cell is outside frozen action map")
    z1 = int(base.actions1[cell1][0])
    z2 = int(base.actions2[cell2][0])
    if (int(np.argmax(a)) != int(base.bank1.decision[z1]) or
            int(np.argmax(b)) != int(base.bank2.decision[z2])):
        raise ValueError("source decision differs from identity bank action")
    return z1, z2


def _checked_law(law):
    law = tuple((int(i), int(j), float(mass)) for i, j, mass in law)
    if (not law or any(not isfinite(p) or p < 0.0 for _, _, p in law) or
            not np.isclose(sum(p for _, _, p in law), 1.0, rtol=0.0, atol=1e-12)):
        raise ValueError("source policy law is invalid")
    return law


def mix_joint(identity: tuple[int, int], source_law, eta: float):
    """(1-eta) exact identity + eta source joint law over global public IDs."""
    eta = _eta(eta)
    if eta == 0.0:
        return ((int(identity[0]), int(identity[1]), 1.0),)
    source_law = _checked_law(source_law)
    if eta == 1.0:
        return aggregate_public_pairs(
            [(i, j) for i, j, _ in source_law], [p for _, _, p in source_law]
        )
    pairs = [identity] + [(i, j) for i, j, _ in source_law]
    mass = [1.0 - eta] + [eta * p for _, _, p in source_law]
    return aggregate_public_pairs(pairs, mass)


def mix_product(identity: tuple[int, int], source_law, eta1: float, eta2: float):
    """Recipient-wise private mixtures; the resulting law remains a product."""
    eta1, eta2 = _eta(eta1), _eta(eta2)
    if eta1 == eta2 == 0.0:
        return ((int(identity[0]), int(identity[1]), 1.0),)
    source_law = _checked_law(source_law)
    r1, r2 = defaultdict(float), defaultdict(float)
    for i, j, mass in source_law:
        r1[i] += mass
        r2[j] += mass
    # Refuse to launder a correlated joint law into a product control.
    observed = {(i, j): mass for i, j, mass in source_law}
    for i in r1:
        for j in r2:
            if not np.isclose(observed.get((i, j), 0.0), r1[i] * r2[j],
                              rtol=0.0, atol=1e-12):
                raise ValueError("source law is not a recipient product")
    r1 = {i: eta1 * mass for i, mass in r1.items()}
    r2 = {j: eta2 * mass for j, mass in r2.items()}
    r1[identity[0]] = r1.get(identity[0], 0.0) + (1.0 - eta1)
    r2[identity[1]] = r2.get(identity[1], 0.0) + (1.0 - eta2)
    pairs = [(i, j) for i in sorted(r1) for j in sorted(r2)]
    mass = [r1[i] * r2[j] for i, j in pairs]
    return aggregate_public_pairs(pairs, mass)


def offline_factor_uniforms(private_root: bytes, *, source_seed: int, role: str,
                            realization: int, unit) -> tuple[float, float]:
    """Two independent HMAC domains from one sealed realization root."""
    if (type(private_root) is not bytes or len(private_root) != 32 or
            type(source_seed) is not int or source_seed < 0 or
            type(role) is not str or not role or
            type(realization) is not int or realization < 0 or
            not (type(unit) is str and unit or
                 type(unit) in (int, np.int64) and unit >= 0)):
        raise ValueError("factor stream identity is invalid")
    answer = []
    for recipient in (1, 2):
        message = json.dumps([FACTOR_STREAM_SCHEMA, source_seed, role,
                              realization, int(unit) if type(unit) in (int, np.int64)
                              else unit, recipient], separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")
        digest = hmac.new(private_root, message, hashlib.sha256).digest()
        answer.append(_uniform_from_uint64(int.from_bytes(digest[:8], "big")))
    return answer[0], answer[1]


def _draw_marginal(marginal, uniform: float) -> int:
    if not isfinite(uniform) or not 0.0 <= uniform < 1.0:
        raise ValueError("sampling uniform must lie in [0,1)")
    ids = sorted(marginal)
    mass = np.asarray([marginal[i] for i in ids], dtype=np.float64)
    index = int(np.searchsorted(np.cumsum(mass), uniform, side="right"))
    if index >= len(ids):
        index = int(np.flatnonzero(mass > 0)[-1])
    return ids[index]


class IdentityPolicy(JointPolicy):
    """Exact immutable B0 control; no learned logits or random draw."""

    law_kind = "bank_identity_v1"

    def __init__(self, base: JointPolicy):
        super().__init__(base.bank1, base.bank2, base.actions1, base.actions2, base.gate)

    def sparse_law(self, score1, score2, cell1: int, cell2: int):
        i, j = identity_pair(self, score1, score2, cell1, cell2)
        return ((i, j, 1.0),)

    def sample_public_pair(self, score1, score2, cell1, cell2, uniform=None):
        return identity_pair(self, score1, score2, cell1, cell2)

    def mechanism_arrays(self):
        return ()


class BoundaryJointPolicy(JointPolicy):
    """Joint or task-only source gate with an executable exact B0 boundary."""

    law_kind = "bank_boundary_joint_v1"

    def __init__(self, base: JointPolicy, eta: float):
        super().__init__(base.bank1, base.bank2, base.actions1, base.actions2, base.gate)
        self.eta = _eta(eta)

    def sparse_law(self, score1, score2, cell1: int, cell2: int):
        identity = identity_pair(self, score1, score2, cell1, cell2)
        if self.eta == 0.0:
            return ((identity[0], identity[1], 1.0),)
        source = super().sparse_law(score1, score2, cell1, cell2)
        return mix_joint(identity, source, self.eta)

    def mechanism_descriptor(self):
        return {"law_kind": self.law_kind, "eta": self.eta}

    def mechanism_arrays(self):
        return super().mechanism_arrays() + (np.asarray([self.eta], dtype=np.float64),)


class BoundaryFactorPolicy(JointPolicy):
    """A product or local source policy with independent recipient boundaries."""

    law_kind = "bank_boundary_factor_v1"

    def __init__(self, base: JointPolicy, eta1: float, eta2: float):
        if base.law_kind not in ("central_product_v1", "strict_local_product_v1"):
            raise TypeError("factor boundary requires an inherited product/local policy")
        super().__init__(base.bank1, base.bank2, base.actions1, base.actions2, base.gate)
        self.base = base
        self.eta1, self.eta2 = _eta(eta1), _eta(eta2)

    def sparse_law(self, score1, score2, cell1: int, cell2: int):
        identity = identity_pair(self, score1, score2, cell1, cell2)
        if self.eta1 == self.eta2 == 0.0:
            return ((identity[0], identity[1], 1.0),)
        return mix_product(identity, self.base.sparse_law(score1, score2, cell1, cell2),
                           self.eta1, self.eta2)

    def sample_public_pair(self, score1, score2, cell1, cell2, uniform):
        """Require separate private recipient coins, including at eta=0."""
        if (not isinstance(uniform, tuple) or len(uniform) != 2 or
                any(not isinstance(u, (int, float, np.floating)) for u in uniform)):
            raise ValueError("factor policy requires two independent uniforms")
        law = self.sparse_law(score1, score2, cell1, cell2)
        r1, r2 = defaultdict(float), defaultdict(float)
        for i, j, mass in law:
            r1[i] += mass
            r2[j] += mass
        return _draw_marginal(r1, float(uniform[0])), _draw_marginal(r2, float(uniform[1]))

    def mechanism_descriptor(self):
        return {"law_kind": self.law_kind, "base_law_kind": self.base.law_kind,
                "eta1": self.eta1, "eta2": self.eta2}

    def mechanism_arrays(self):
        return self.base.mechanism_arrays() + (
            np.asarray([self.eta1, self.eta2], dtype=np.float64),)
