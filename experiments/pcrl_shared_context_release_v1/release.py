"""Pinned per-person law loader and one-token keyed-persistent release session.

`load_law(unit_dir)` returns a callable mapping LEGAL per-person inputs
(x, ha, T0 code, p, r, risk) to the private (n, 17) token law of a finished
nested unit (NM*, T32*, DET_SEL*).  The law, context id, policy tokens and
parameters are private; `SharedContextRelease.emit` puts only H_A and one
integer token per record on the wire, drawn once per record with a keyed
HMAC (the TAC/AR session pattern).  Repeated fresh draws are NOT this
contract.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import hmac
import json
from pathlib import Path

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1.data import sha256_file
from . import channel, contexts, policies

KIND = "nested"


def _resolve_bank(unit_root: Path, spec: dict, bank_dir):
    candidates = [Path(bank_dir)] if bank_dir is not None else [
        unit_root / spec["bank_dir_relative"], Path(spec["bank_dir"])]
    for candidate in candidates:
        manifest = candidate / "BANK_COMPLETE.json"
        if manifest.is_file() and sha256_file(manifest) == spec["bank_manifest_sha256"]:
            return candidate.resolve()
    raise FileNotFoundError("pinned frozen policy bank not found (or its manifest hash differs)")


class NestedLaw:
    """Frozen callable q(z|x); legal inputs only."""

    def __init__(self, spec, B, A, eta, policy_bank, rule):
        self.spec = spec
        self.B, self.A, self.eta = channel.validate_params(B, A, eta)
        self.policy_bank = policy_bank
        self.rule = rule
        if len(policy_bank.names) != self.A.shape[1] or rule.K != self.A.shape[0]:
            raise ValueError("pinned policy columns/contexts differ from parameters")
        self.digest = hashlib.sha256(b"SC_NESTED_LAW_V1" + json.dumps(
            [spec["params_sha256"], spec["policy_bank_sha256"], spec["contexts_sha256"],
             spec["policy_columns"], spec["K"]], separators=(",", ":")).encode()).hexdigest()

    def __call__(self, inputs) -> np.ndarray:
        legal = policies.check_legal(inputs)
        codes = np.asarray(legal["token_codes"])
        return channel.person_law(self.B, self.A, self.eta, codes,
                                  self.rule.assign(legal), self.policy_bank.predict(legal))


def load_law(unit_dir, *, bank_dir=None) -> NestedLaw:
    """Load a finished unit's RELEASE_SPEC.json with every hash verified."""
    root = Path(unit_dir).resolve()
    spec = json.loads((root / "RELEASE_SPEC.json").read_text())
    if spec.get("kind") != KIND or spec.get("schema") != "pcrl-sc-nested-release-v1":
        raise ValueError("not a nested shared-context release spec")
    params_path = root / spec["params_relative"]
    if sha256_file(params_path) != spec["params_sha256"]:
        raise ValueError("pinned nested parameter file hash differs")
    with np.load(params_path, allow_pickle=False) as arrays:
        B, A, eta = arrays["B"].copy(), arrays["A"].copy(), float(arrays["eta"])
    bank_root = _resolve_bank(root, spec, bank_dir)
    manifest = json.loads((bank_root / "BANK_COMPLETE.json").read_text())
    if (manifest["policy_bank_sha256"] != spec["policy_bank_sha256"]
            or manifest["contexts_sha256"] != spec["contexts_sha256"]):
        raise ValueError("bank policy/context pins differ from release spec")
    bank = policies.load_bank(bank_root / "policies", spec["policy_bank_sha256"]).subset(spec["policy_columns"])
    rule = contexts.load_rules(bank_root / "contexts", spec["contexts_sha256"])[int(spec["K"])]
    return NestedLaw(spec, B, A, eta, bank, rule)


def _identifier(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("boolean release ID is invalid")
    if isinstance(value, str) and value:
        return ("str", value)
    if isinstance(value, (int, np.integer)):
        return ("int", int(value))
    raise ValueError("release IDs must be nonempty strings or integers")


def _row_digest(legal, i):
    digest = hashlib.sha256()
    for key in policies.LEGAL_INPUTS:
        value = np.ascontiguousarray(np.asarray(legal[key])[i])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class SharedContextRelease:
    """One keyed-persistent token per record; the wire is (H_A, token) only."""

    def __init__(self, law: NestedLaw, *, replay_key: bytes, release_id: str):
        if not isinstance(law, NestedLaw):
            raise TypeError("NestedLaw required")
        if not isinstance(replay_key, bytes) or len(replay_key) < 32:
            raise ValueError("private replay key must contain at least 32 bytes")
        if not isinstance(release_id, str) or not release_id:
            raise ValueError("nonempty release_id required")
        self._law = law
        self._key = replay_key
        binding = hmac.new(replay_key, b"PCRL_SC_KEY_BINDING_V1", hashlib.sha256).hexdigest()
        self._config = hashlib.sha256(json.dumps(
            ["PCRL_SC_NESTED_RELEASE_V1", law.digest, release_id, binding],
            separators=(",", ":")).encode()).hexdigest()
        self._cache = {}

    def _uniform(self, identifier, row_digest):
        payload = json.dumps(["PCRL_SC_KEYED_REPLAY_V1", self._config, identifier, row_digest],
                             separators=(",", ":")).encode()
        draw = hmac.new(self._key, payload, hashlib.sha256).digest()
        return int.from_bytes(draw[:8], "big") / 2**64

    def emit(self, inputs, release_ids: Sequence) -> dict:
        legal = policies.check_legal(inputs)
        missing = [k for k in policies.LEGAL_INPUTS if k not in legal]
        if missing:
            raise ValueError(f"deployment requires every legal input; missing {missing}")
        ids = [_identifier(v) for v in release_ids]
        h_a = np.asarray(legal["ha"])
        if len(ids) != len(h_a) or len(set(ids)) != len(ids):
            raise ValueError("one distinct release ID per original person required")
        law = self._law(legal)
        tokens = np.empty(len(ids), dtype=np.int64)
        for i, identifier in enumerate(ids):
            digest = _row_digest(legal, i)
            cumulative = np.cumsum(law[i])
            cumulative[-1] = 1.
            token = int(np.searchsorted(cumulative, self._uniform(identifier, digest), side="right"))
            if identifier in self._cache:
                previous_digest, previous = self._cache[identifier]
                if previous_digest != digest:
                    raise ValueError("release ID reused with changed inputs")
                if previous != token:
                    raise AssertionError("keyed persistent draw changed")
            self._cache[identifier] = (digest, token)
            tokens[i] = token
        h = np.array(h_a, copy=True)
        if h.tobytes() != h_a.tobytes():
            raise AssertionError("H_A service byte parity failed")
        h.flags.writeable = False
        tokens.flags.writeable = False
        return {"h_a": h, "token": tokens}
