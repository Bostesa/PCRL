"""Public-prototype channel features for one-draw sampled-score attackers.

The prototype prior is UNIFORM over the Cartesian grid of published fine-cell
centroids. It is an approximate attack feature, not an Adult source posterior.
The compiler accepts no role rows, labels, fitted private frequencies or
individual provider scores. Its sparse-law callable must be the published
mechanism evaluated at those public centroid pairs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r02.complete_view import (
    WireBatch, prepare_views,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.protocol import (
    _factory, _label_record, attack_fit_tune_split, sha_file,
)
from .wire import PublicBank


DERIVED_NAME = "q_uniform_proto_v1"
SCHEMA = "SCV1-q-uniform-prototype-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value, dtype="<f8")
        digest.update(_canonical_json(list(a.shape)) + b"\n")
        digest.update(a.tobytes())
    return digest.hexdigest()


def _centroids(value: object, classes: int) -> np.ndarray:
    a = np.asarray(value)
    if (a.dtype != np.dtype("float64") or a.ndim != 2 or a.shape[1] != classes
            or a.shape[0] < 1 or not np.isfinite(a).all() or np.any(a < 0)
            or not np.allclose(a.sum(axis=1), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("public fine-cell centroids are not fixed score vectors")
    out = np.ascontiguousarray(a.copy(), dtype="<f8")
    out.setflags(write=False)
    return out


class _Accumulator:
    def __init__(self, shape: tuple[int, ...], score_width: int):
        self.mass = np.zeros(shape, dtype=np.float64)
        self.qlq = np.zeros(shape, dtype=np.float64)
        self.support = np.zeros(shape, dtype=np.float64)
        self.first = np.zeros((*shape, score_width), dtype=np.float64)
        self.second = np.zeros((*shape, score_width), dtype=np.float64)

    def add(self, key: tuple[int, ...], q: float, score: np.ndarray):
        if q <= 0:
            return
        self.mass[key] += q
        self.qlq[key] += q * math.log(q)
        self.support[key] += 1
        self.first[key] += q * score
        self.second[key] += q * score * score

    def finish(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        prior_mass = self.mass / count
        positive = self.mass > 0
        width = self.first.shape[-1]
        features = np.zeros((*prior_mass.shape, 3 + 2 * width), dtype=np.float64)
        features[..., 0][positive] = np.log(prior_mass[positive])
        if count > 1:
            entropy = np.zeros_like(self.mass)
            entropy[positive] = (np.log(self.mass[positive]) -
                                 self.qlq[positive] / self.mass[positive]) / math.log(count)
            features[..., 1] = np.clip(entropy, 0, 1)
        features[..., 2] = self.support / count
        mean = np.divide(self.first, self.mass[..., None],
                         out=np.zeros_like(self.first), where=self.mass[..., None] > 0)
        moment2 = np.divide(self.second, self.mass[..., None],
                            out=np.zeros_like(self.second), where=self.mass[..., None] > 0)
        features[..., 3:3 + width] = mean
        features[..., 3 + width:] = np.maximum(moment2 - mean * mean, 0)
        if not np.isfinite(features).all():
            raise ValueError("public Q prototype features are nonfinite")
        features.setflags(write=False)
        prior_mass.setflags(write=False)
        return features, prior_mass


@dataclass(frozen=True)
class PrototypeFeatureTable:
    r1: np.ndarray
    r2: np.ndarray
    pair: np.ndarray
    r1_mass: np.ndarray
    r2_mass: np.ndarray
    pair_mass: np.ndarray
    prototype_count: int
    identity: dict
    table_sha256: str


def compile_uniform_prototype_features(
    bank1: PublicBank, bank2: PublicBank,
    centroids1: np.ndarray, centroids2: np.ndarray,
    sparse_law: Callable[[np.ndarray, np.ndarray, int, int], Sequence[tuple[int, int, float]]],
    *, public_centroids_sha256: str, expected_identity: Mapping[str, object] | None = None,
) -> PrototypeFeatureTable:
    """Compile sparse public q on every fixed fine-cell centroid pair.

    This is CPU preprocessing and must acquire R03's shared heavy slot on the
    real 64×641 grid. A lock must pin the returned identity/table hash before
    any ATT reader fit; a caller cannot adapt the prototype prior to outcomes.
    """
    if bank1.recipient != 1 or bank2.recipient != 2:
        raise ValueError("public Q needs recipient 1 and 2 banks")
    if (bank1.release_id != bank2.release_id or
            bank1.public_metadata["source_binding_sha256"] !=
            bank2.public_metadata["source_binding_sha256"] or
            bank1.public_metadata["mechanism_sha256"] !=
            bank2.public_metadata["mechanism_sha256"]):
        raise ValueError("public Q banks have different paired source/mechanism identities")
    if (not isinstance(public_centroids_sha256, str) or len(public_centroids_sha256) != 64
            or any(c not in "0123456789abcdef" for c in public_centroids_sha256)):
        raise ValueError("public centroid artifact SHA-256 is required")
    c1 = _centroids(centroids1, bank1.probability.shape[1])
    c2 = _centroids(centroids2, bank2.probability.shape[1])
    n = len(c1) * len(c2)
    identity = {"schema": SCHEMA, "release_id": bank1.release_id,
                "source_binding_sha256": bank1.public_metadata["source_binding_sha256"],
                "mechanism_sha256": bank1.public_metadata["mechanism_sha256"],
                "bank1_sha256": bank1.public_metadata["canonical_bank_sha256"],
                "bank2_sha256": bank2.public_metadata["canonical_bank_sha256"],
                "public_centroids_sha256": public_centroids_sha256,
                "centroid_array_sha256": _digest_arrays(c1, c2),
                "centroid_shape": [list(c1.shape), list(c2.shape)],
                "uniform_prior": True, "prototype_count": n,
                "output_shape": [len(bank1.probability), len(bank2.probability)]}
    if expected_identity is not None and dict(expected_identity) != identity:
        raise ValueError("public Q source identity differs from predeclared artifact")
    a1 = _Accumulator((len(bank1.probability),), c1.shape[1] + c2.shape[1])
    a2 = _Accumulator((len(bank2.probability),), c1.shape[1] + c2.shape[1])
    ap = _Accumulator((len(bank1.probability), len(bank2.probability)),
                      c1.shape[1] + c2.shape[1])
    for i, score1 in enumerate(c1):
        for j, score2 in enumerate(c2):
            triples = tuple(sparse_law(score1, score2, i, j))
            if not triples or len({(x, y) for x, y, _ in triples}) != len(triples):
                raise ValueError("public prototype q omitted or repeated a global output pair")
            q1: dict[int, float] = {}
            q2: dict[int, float] = {}
            total = 0.0
            score = np.concatenate((score1, score2))
            for z1, z2, q in triples:
                if (type(z1) not in (int, np.int64) or type(z2) not in (int, np.int64)
                        or not 0 <= z1 < len(bank1.probability)
                        or not 0 <= z2 < len(bank2.probability)
                        or not isinstance(q, (int, float, np.floating))
                        or not math.isfinite(q) or q < 0):
                    raise ValueError("public prototype q has invalid code/mass")
                if q > 0 and (np.argmax(score1) != np.argmax(bank1.probability[z1])
                              or np.argmax(score2) != np.argmax(bank2.probability[z2])):
                    raise ValueError("public prototype q changes a source decision")
                total += q
                q1[z1] = q1.get(z1, 0.0) + q
                q2[z2] = q2.get(z2, 0.0) + q
                ap.add((z1, z2), float(q), score)
            if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-12):
                raise ValueError("public prototype q is not normalized")
            for z, q in q1.items():
                a1.add((z,), q, score)
            for z, q in q2.items():
                a2.add((z,), q, score)
    r1, m1 = a1.finish(n)
    r2, m2 = a2.finish(n)
    pair, mp = ap.finish(n)
    if not all(math.isclose(float(m.sum()), 1.0, rel_tol=0, abs_tol=1e-10)
               for m in (m1, m2, mp)):
        raise ValueError("public prototype lookup does not have unit output mass")
    digest = hashlib.sha256(_canonical_json(identity) + b"\n")
    for value in (r1, r2, pair, m1, m2, mp):
        digest.update(_canonical_json(list(value.shape)) + b"\n")
        digest.update(np.ascontiguousarray(value, dtype="<f8").tobytes())
    return PrototypeFeatureTable(r1, r2, pair, m1, m2, mp, n,
                                 identity, digest.hexdigest())


def with_q_features(batch: WireBatch, table: PrototypeFeatureTable, *, view: str) -> WireBatch:
    """Attach a public deterministic code lookup to the complete attack view."""
    if view not in ("r1", "r2", "pair") or not isinstance(table, PrototypeFeatureTable):
        raise ValueError("public Q lookup view/table is invalid")
    names = ("r1.code_id", "r2.code_id") if view == "pair" else ("code_id",)
    if any(name not in batch.fields for name in names):
        raise ValueError("actual emitted public code field is absent")
    codes = tuple(np.asarray(batch.fields[name].values) for name in names)
    n = len(batch.row_id)
    if any(a.shape != (n,) or a.dtype != np.dtype("int64") for a in codes):
        raise ValueError("actual emitted code identity has wrong shape/type")
    if view == "r1":
        value, mass = table.r1, table.r1_mass
    elif view == "r2":
        value, mass = table.r2, table.r2_mass
    else:
        value, mass = table.pair, table.pair_mass
    if any(np.any((a < 0) | (a >= size)) for a, size in zip(codes, mass.shape)):
        raise ValueError("actual emitted code lies outside public Q output table")
    if np.any(mass[codes] <= 0):
        raise ValueError("actual emitted code has zero public prototype support")
    prefix = "r1." if view == "pair" else ""
    second_prefix = "r2." if view == "pair" else ""
    expected = table.identity
    if view in ("r1", "pair"):
        pairs = (("release_id", expected["release_id"]),
                 ("public_metadata.model_id", expected["release_id"]),
                 ("public_metadata.canonical_bank_sha256", expected["bank1_sha256"]),
                 ("public_metadata.mechanism_sha256", expected["mechanism_sha256"]),
                 ("public_metadata.source_binding_sha256", expected["source_binding_sha256"]))
        for field, literal in pairs:
            name = prefix + field
            if name not in batch.fields or not np.all(batch.fields[name].values == literal):
                raise ValueError("actual wire differs from public Q source identity")
    if view in ("r2", "pair"):
        pairs = (("release_id", expected["release_id"]),
                 ("public_metadata.model_id", expected["release_id"]),
                 ("public_metadata.canonical_bank_sha256", expected["bank2_sha256"]),
                 ("public_metadata.mechanism_sha256", expected["mechanism_sha256"]),
                 ("public_metadata.source_binding_sha256", expected["source_binding_sha256"]))
        for field, literal in pairs:
            name = second_prefix + field
            if name not in batch.fields or not np.all(batch.fields[name].values == literal):
                raise ValueError("actual wire differs from public Q source identity")
    features = np.ascontiguousarray(value[codes], dtype=np.float64)
    if DERIVED_NAME in (batch.public_derived or {}):
        if not np.array_equal(np.asarray(batch.public_derived[DERIVED_NAME]), features):
            raise ValueError("existing public Q feature parity differs from code/table")
        return batch
    derived = dict(batch.public_derived or {})
    derived[DERIVED_NAME] = features
    return WireBatch(batch.row_id, batch.unit, batch.fields, batch.code_fields, derived)


def fit_q_posterior_reader(
    att_batch: WireBatch, inner_batch: WireBatch,
    sex_att: dict, table: PrototypeFeatureTable,
    *, view: str, attacker_seed: int, private_dir: str | Path,
    lock_path: str | Path, lock_sha256: str, expected_table_sha256: str,
    queue_guard: Callable[[], None],
) -> dict:
    """ATT-only Q_POST_LR_C1; return the R02 external-reader record.

    This function intentionally has no INNER SEX argument. The caller scores
    its frozen INNER predictions later through R02's common attack protocol.
    """
    lock = Path(lock_path)
    if (not isinstance(lock_sha256, str) or len(lock_sha256) != 64 or
            not lock.is_file() or sha_file(lock) != lock_sha256):
        raise ValueError("effective A lock file/hash is missing or changed")
    if (not isinstance(table, PrototypeFeatureTable) or
            table.table_sha256 != expected_table_sha256):
        raise ValueError("public Q table differs from sealed artifact")
    if not callable(queue_guard):
        raise ValueError("active shared compute queue guard is required")
    out = Path(private_dir)
    if ".private" not in out.resolve().parts or out.exists():
        raise ValueError("Q reader custody must be a new private directory")
    att = with_q_features(att_batch, table, view=view)
    inner = with_q_features(inner_batch, table, view=view)
    prepared = prepare_views(att, inner)
    x_att = np.asarray(att.public_derived[DERIVED_NAME], dtype=np.float64)
    x_inner = np.asarray(inner.public_derived[DERIVED_NAME], dtype=np.float64)
    if x_att.shape[1] != x_inner.shape[1]:
        raise ValueError("public Q ATT/INNER feature widths differ")
    y = _label_record(sex_att, prepared.att)
    split = attack_fit_tune_split(prepared.att.unit)
    fit, tune = split["fit"], split["tune"]
    if np.unique(y[fit]).size != 2 or np.unique(y[tune]).size != 2:
        raise ValueError("Q reader ATT fit/tune lacks both SEX classes")
    queue_guard()
    out.mkdir(parents=True, mode=0o700)
    model = _factory("LR_C1", attacker_seed)
    model.fit(x_att[fit], y[fit])
    queue_guard()
    p_tune = np.asarray(model.predict_proba(x_att[tune])[:, 1], dtype=np.float64)
    p_inner = np.asarray(model.predict_proba(x_inner)[:, 1], dtype=np.float64)
    model_path = out / "model__Q_POST_LR_C1.joblib"
    matrix_path = out / "public_q_features_and_predictions.npz"
    joblib.dump(model, model_path, compress=3)
    np.savez_compressed(matrix_path, att_row_id=att.row_id, att_unit=att.unit,
                        inner_row_id=inner.row_id, inner_unit=inner.unit,
                        att_q=x_att, inner_q=x_inner,
                        prediction_tune=p_tune, prediction_inner=p_inner)
    model_path.chmod(0o600)
    matrix_path.chmod(0o600)
    receipt = {"schema": "SCV1-Q-POST-reader-receipt-v1", "name": "Q_POST_LR_C1",
               "view": view, "fit_role": "ATTACK_FIT_NEW/ATTACK_TRAIN",
               "tune_role": "ATTACK_FIT_NEW/ATTACK_TUNE",
               "score_role": "INNER_SELECTION",
               "lock_sha256": lock_sha256, "public_q_table_sha256": table.table_sha256,
               "public_q_identity": table.identity,
               "att_view_sha256": prepared.att.content_hashes(),
               "inner_view_sha256": prepared.inner.content_hashes(),
               "fit_rows": int(len(fit)), "tune_rows": int(len(tune)),
               "attacker_seed": int(attacker_seed),
               "model_sha256": sha_file(model_path),
               "private_matrix_sha256": sha_file(matrix_path)}
    receipt_path = out / "Q_POST_READER_RECEIPT.json"
    fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return {"name": "Q_POST_LR_C1",
            "fit_role": "ATTACK_FIT_NEW/ATTACK_TRAIN",
            "model_artifact_path": str(model_path),
            "model_sha256": receipt["model_sha256"],
            "prediction_tune": p_tune,
            "prediction_inner": p_inner,
            "private_receipt_path": str(receipt_path),
            "private_receipt_sha256": sha_file(receipt_path)}
