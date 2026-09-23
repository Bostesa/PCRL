"""Deployable one-token T0 release with an in-process one-release cache.

The wire contains exactly unchanged H_A and one integer token.  A caller may
hold H_B for the coalition view, but this interface never reads H_B, Y or S.
OneReleaseSession reuses a token for a repeated caller-held person ID while it
is alive; optional private HMAC keyed replay gives stable pseudorandom draws
across process restarts for immutable inputs and channel. It does not claim
security for independent repeated releases or altered input snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path

import joblib
import numpy as np

from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
from .method import validate_channel

ARTIFACT_SCHEMA = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_encoder_verified(path, expected_sha256):
    """Verify a frozen trusted object before joblib deserialization."""
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("expected encoder SHA-256 is required")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError("frozen encoder SHA-256 mismatch")
    return joblib.load(path)


@dataclass(frozen=True)
class ChannelArtifact:
    Q: np.ndarray
    encoder_sha256: str
    source: str
    model_status: str = "EXPERIMENTAL_UNVALIDATED"

    def __post_init__(self):
        q = validate_channel(self.Q, n_tokens=17).copy()
        q.flags.writeable = False
        object.__setattr__(self, "Q", q)
        if not isinstance(self.encoder_sha256, str) or len(self.encoder_sha256) != 64:
            raise ValueError("pinned frozen encoder SHA-256 required")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source configuration required")
        if not isinstance(self.model_status, str) or not self.model_status:
            raise ValueError("model status required")

    def save(self, directory):
        """Save public channel and hash manifest without a private RNG seed."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(path / "Q.npz", Q=self.Q)
        manifest = {"schema": ARTIFACT_SCHEMA, "channel_file": "Q.npz",
                    "channel_sha256": sha256_file(path / "Q.npz"),
                    "encoder_sha256": self.encoder_sha256,
                    "source": self.source, "model_status": self.model_status,
                    "shape": list(self.Q.shape),
                    "wire": "H_A unchanged plus one sampled integer token",
                    "runtime_inputs": "PCA32 X_A plus four H_A columns only",
                    "one_release_policy": "in-process cache or private keyed replay; bind immutable person inputs and mechanism across restarts",
                    "privacy_scope": "one token/person; no repeated-query or composition guarantee"}
        with (path / "manifest.json").open("x") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        return manifest

    @classmethod
    def load(cls, directory):
        path = Path(directory)
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("schema") != ARTIFACT_SCHEMA or manifest.get("channel_file") != "Q.npz":
            raise ValueError("unsupported or malformed channel manifest")
        if sha256_file(path / "Q.npz") != manifest.get("channel_sha256"):
            raise ValueError("channel file SHA-256 mismatch")
        with np.load(path / "Q.npz", allow_pickle=False) as archive:
            if list(archive.files) != ["Q"]:
                raise ValueError("channel archive must contain only Q")
            q = archive["Q"].copy()
        if list(q.shape) != manifest.get("shape"):
            raise ValueError("channel shape/manifest mismatch")
        return cls(q, manifest["encoder_sha256"], manifest["source"],
                   manifest["model_status"])

    def expected_token_law(self, encoder, inputs: RuntimeInputs):
        """Internal audit law; never append this hidden person's Q row to wire."""
        if not isinstance(inputs, RuntimeInputs):
            raise TypeError("only RuntimeInputs(X_A,H_A) are accepted")
        if encoder.code.n_states("T0") != len(self.Q):
            raise ValueError("frozen encoder T0 alphabet differs from channel")
        codes = np.asarray(encoder.encode(inputs)["codes"]["T0"])
        if (codes.shape != (len(inputs.h_a),) or not np.isfinite(codes).all() or
                np.any(codes != np.floor(codes)) or np.any(codes < 0) or
                np.any(codes >= len(self.Q))):
            raise ValueError("encoder returned invalid T0 codes")
        return self.Q[codes.astype(np.int64)]


def _row_digest(x, h):
    digest = hashlib.sha256()
    for a in (x, h):
        a = np.ascontiguousarray(a)
        digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def _identifier(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("boolean person IDs are invalid")
    if isinstance(value, str) and value:
        return ("str", value)
    if isinstance(value, (int, np.integer)):
        return ("int", int(value))
    raise ValueError("person IDs must be nonempty strings or integers")


class OneReleaseSession:
    """Sample once per caller-held ID, with optional private keyed replay.

    In keyed mode the caller supplies an external secret of at least 32 bytes.
    HMAC turns the person ID, input digest and channel identity into a stable
    pseudorandom uniform across process restarts. The key is never serialized
    or returned. Deployment must bind IDs to immutable input snapshots;
    changing an input across restarts changes the channel row and keyed draw.
    """

    def __init__(self, artifact: ChannelArtifact, encoder, *,
                 rng: np.random.Generator | None = None, replay_key: bytes | None = None):
        if not isinstance(artifact, ChannelArtifact):
            raise TypeError("ChannelArtifact required")
        if (rng is None) == (replay_key is None):
            raise ValueError("supply exactly one explicit RNG or private replay key")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a NumPy Generator")
        if replay_key is not None and (not isinstance(replay_key, bytes) or len(replay_key) < 32):
            raise ValueError("private replay key must have at least 32 bytes")
        if encoder.code.n_states("T0") != len(artifact.Q):
            raise ValueError("encoder/channel state-alphabet mismatch")
        self.artifact = artifact
        self.encoder = encoder
        self._rng = rng
        self._replay_key = replay_key
        self._channel_digest = hashlib.sha256(
            b"PCRL_ONE_RELEASE_CHANNEL_V1"+
            np.ascontiguousarray(artifact.Q).tobytes()+
            artifact.encoder_sha256.encode()).hexdigest()
        self._cache = {}  # caller-held ID -> (input digest, sampled token)

    @classmethod
    def from_verified_encoder_path(cls, artifact: ChannelArtifact, encoder_path, *,
                                   rng=None, replay_key=None):
        """Preferred construction path: hash-check frozen encoder before load."""
        encoder = load_encoder_verified(encoder_path, artifact.encoder_sha256)
        return cls(artifact, encoder, rng=rng, replay_key=replay_key)

    def _uniform(self, person_key, input_digest):
        if self._replay_key is None:
            return float(self._rng.random())
        message = json.dumps(["PCRL_KEYED_REPLAY_V1", self._channel_digest,
                              person_key, input_digest], separators=(",", ":")).encode()
        draw = hmac.new(self._replay_key, message, hashlib.sha256).digest()
        return int.from_bytes(draw[:8], "big") / 2**64

    def emit(self, inputs: RuntimeInputs, release_ids):
        """Return only bitwise-preserved H_A and one token per original person."""
        if not isinstance(inputs, RuntimeInputs):
            raise TypeError("only RuntimeInputs(X_A,H_A) are accepted")
        ids = list(release_ids)
        if len(ids) != len(inputs.h_a):
            raise ValueError("one release ID required per original person")
        keys = [_identifier(value) for value in ids]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate IDs within one batch are not allowed")
        law = self.artifact.expected_token_law(self.encoder, inputs)
        tokens = np.empty(len(keys), dtype=np.int64)
        for i, key in enumerate(keys):
            current = _row_digest(inputs.x_a[i], inputs.h_a[i])
            if key in self._cache:
                previous, token = self._cache[key]
                if previous != current:
                    raise ValueError("person ID reused with changed private or service inputs")
            else:
                cumulative = np.cumsum(law[i])
                cumulative[-1] = 1.
                token = int(np.searchsorted(cumulative, self._uniform(key, current), side="right"))
                self._cache[key] = (current, token)
            tokens[i] = token
        h = np.array(inputs.h_a, copy=True)
        if h.dtype != np.asarray(inputs.h_a).dtype or h.tobytes() != np.asarray(inputs.h_a).tobytes():
            raise AssertionError("H_A service parity failed")
        h.flags.writeable = False
        tokens.flags.writeable = False
        return {"h_a": h, "token": tokens}
