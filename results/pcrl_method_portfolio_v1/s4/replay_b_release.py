"""S4 reconstruction of a saved Team B release and attack matrices.

This module is independent of Team B and S2 implementations. Adult calls may
run only as a registered S3 numerical replay unit. It imports neither their
release builders nor evaluator, endpoint, or selection code.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


PURPOSES = ("income", "occupation")
WIDTH = {"income": 2, "occupation": 6}
WIRE_KEYS = {"features", "logits", "probabilities", "decision", "metadata"}
SPLIT_SALT = "pcrl-method-portfolio-v1|attack-tune|20261008|"


def sha_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update((str(a.dtype) + str(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def ordered_split(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute the fixed ATT train/tune membership without S2 code."""
    group = np.asarray(groups, dtype=np.int64)
    unique = np.unique(group)
    if group.ndim != 1 or len(unique) != 4061:
        raise ValueError("ATT group count differs from lock")
    ranked = sorted((hashlib.sha256((SPLIT_SALT + str(int(u))).encode()).hexdigest(),
                     int(u)) for u in unique)
    tune_groups = np.asarray([u for _, u in ranked[:1015]], dtype=np.int64)
    tune = np.isin(group, tune_groups)
    fit = ~tune
    if len(np.unique(group[fit])) != 3046 or len(np.unique(group[tune])) != 1015:
        raise ValueError("ATT split sizes differ")
    return np.flatnonzero(fit), np.flatnonzero(tune), tune_groups


def model_sha(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        a = state[key].detach().cpu().contiguous().numpy()
        h.update(key.encode())
        h.update(str(a.shape).encode("ascii"))
        h.update(str(a.dtype).encode("ascii"))
        h.update(a.tobytes())
    return h.hexdigest()


def load_final_state(attempt: Path, index: dict) -> dict[str, torch.Tensor]:
    """Find the final state in pinned checkpoints without constructing a candidate model."""
    recorded = index["fit_artifact_sha256"]
    matches = []
    for name, digest in recorded.items():
        if name == "first_fixed_draw_file":
            frozen = attempt / "fit" / "first_fixed_draw.pt"
            if not frozen.is_file() or sha_file(frozen) != digest:
                raise ValueError("sequential frozen first-view file/hash differs")
            continue
        filename = "B-LOCAL__assembled.pt" if name == "assembled_local_pair" else \
            name.replace("/", "__") + ".pt"
        path = attempt / "fit" / filename
        if not path.is_file() or sha_file(path) != digest:
            raise ValueError("fitted checkpoint file/hash differs")
        obj = torch.load(path, map_location="cpu", weights_only=True)
        state = obj.get("model")
        if not isinstance(state, dict):
            raise ValueError("fitted checkpoint has no model state")
        if model_sha(state) == index["fit_model_sha256"]:
            matches.append(state)
    if len(matches) != 1:
        raise ValueError("there is not exactly one final pinned public model")
    return matches[0]


def head_state(state: dict[str, torch.Tensor]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    heads = {}
    for purpose in PURPOSES:
        weight = state[f"heads.{purpose}.weight"].detach().cpu().numpy()
        bias = state[f"heads.{purpose}.bias"].detach().cpu().numpy()
        if weight.shape != (WIDTH[purpose], 16) or bias.shape != (WIDTH[purpose],):
            raise ValueError("public head shape differs")
        heads[purpose] = (weight.astype(np.float32, copy=False),
                          bias.astype(np.float32, copy=False))
    return heads


def schema_sha(feature_names: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(feature_names, separators=(",", ":")).encode()).hexdigest()


def purpose_sha(state: dict[str, torch.Tensor], purpose: str,
                feature_names: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    h.update(b"TeamBPurposeModelV1")
    h.update(purpose.encode("ascii"))
    h.update(schema_sha(feature_names).encode("ascii"))
    for module in ("encoders", "heads"):
        prefix = f"{module}.{purpose}."
        for key in sorted(key for key in state if key.startswith(prefix)):
            a = state[key].detach().cpu().contiguous().numpy()
            h.update(key.encode())
            h.update(str(a.shape).encode("ascii"))
            h.update(str(a.dtype).encode("ascii"))
            h.update(a.tobytes())
    return h.hexdigest()


def load_saved_wire(path: Path, row_id: np.ndarray, x_rows: np.ndarray,
                    expected_sha: str, expected_source: str,
                    expected_model: str,
                    expected_calibration: dict[str, str]) -> list[dict[str, Any]]:
    """Return actual local wire records in admitted row order; keys stay private."""
    if sha_file(path) != expected_sha:
        raise ValueError("saved release byte hash differs")
    wrapped = json.loads(path.read_text())
    if set(wrapped) != {"payload", "payload_sha256"}:
        raise ValueError("saved release wrapper has unexpected fields")
    payload = wrapped["payload"]
    if set(payload) != {"format", "source_id", "model_sha256",
                        "purpose_model_sha256", "schema_sha256", "temperatures",
                        "raw_mode", "calibration_record_sha256", "stochastic",
                        "records"}:
        raise ValueError("saved provider payload fields differ")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(raw).hexdigest() != wrapped["payload_sha256"]:
        raise ValueError("saved release payload digest differs")
    if (payload.get("format") != "TeamBPrivateReleaseV1"
            or payload.get("source_id") != expected_source
            or payload.get("model_sha256") != expected_model
            or payload.get("raw_mode") is not False):
        raise ValueError("saved release source/model/calibration mode differs")
    if (payload["calibration_record_sha256"] != expected_calibration
            or set(payload["purpose_model_sha256"]) != set(PURPOSES)
            or set(payload["temperatures"]) != set(PURPOSES)
            or type(payload["stochastic"]) is not bool):
        raise ValueError("public calibration/sampling identity differs")
    mapping: dict[str, dict[str, Any]] = {}
    for entry in payload["records"]:
        if set(entry) != {"purpose", "private_person_key", "input_sha256", "view"}:
            raise ValueError("saved provider record has unexpected field")
        purpose, person = entry["purpose"], entry["private_person_key"]
        if purpose not in PURPOSES or not isinstance(person, str):
            raise ValueError("saved provider key/purpose differs")
        if purpose in mapping.setdefault(person, {}):
            raise ValueError("duplicate person/purpose")
        view = entry["view"]
        if set(view) != WIRE_KEYS or not isinstance(view["metadata"], dict):
            raise ValueError("recipient wire fields incomplete or extra")
        meta = view["metadata"]
        if (meta.get("calibration_record_sha256") != expected_calibration[purpose]
                or meta.get("purpose_model_sha256") != payload["purpose_model_sha256"][purpose]
                or meta.get("schema_sha256") != payload["schema_sha256"]
                or meta.get("temperature") != payload["temperatures"][purpose]
                or meta.get("stochastic") != payload["stochastic"]):
            raise ValueError("recipient metadata differs from sealed provider law")
        mapping[person][purpose] = (view, entry["input_sha256"])
    ordered = []
    x = np.asarray(x_rows)
    if x.shape != (len(row_id), 83):
        raise ValueError("admitted predictor rows differ")
    for i, person in enumerate(np.asarray(row_id, dtype=np.int64)):
        key = str(int(person))
        if key not in mapping or set(mapping[key]) != set(PURPOSES):
            raise ValueError("saved release person alignment differs")
        pair = mapping.pop(key)
        expected_input = hashlib.sha256(
            np.ascontiguousarray(x[i].astype(np.float32)).tobytes()).hexdigest()
        if any(pair[p][1] != expected_input for p in PURPOSES):
            raise ValueError("saved release input hash differs from admitted 83 predictors")
        ordered.append({p: pair[p][0] for p in PURPOSES})
    if mapping:
        raise ValueError("saved release has extra people")
    return ordered


def replay_encoder(rows: list[dict], x_rows: np.ndarray, row_id: np.ndarray,
                   state: dict[str, torch.Tensor], secret: bytes,
                   source_id: str, feature_names: tuple[str, ...]) -> None:
    """Recompute actual one-shot z from permitted X and private sampling law."""
    if len(secret) < 32 or len(rows) != len(row_id) or \
            np.asarray(x_rows).shape != (len(row_id), 83):
        raise ValueError("sampler/row input shape differs")
    whole_model_sha = model_sha(state)
    feature_schema_sha = schema_sha(feature_names)
    for purpose in PURPOSES:
        prefix = f"encoders.{purpose}."
        ps = purpose_sha(state, purpose, feature_names)
        for i, person in enumerate(row_id):
            view = rows[i][purpose]
            meta = view["metadata"]
            if (meta.get("source_id") != source_id or meta.get("schema_sha256") !=
                    feature_schema_sha or meta.get("purpose_model_sha256") != ps
                    or meta.get("model_sha256") != whole_model_sha
                    or meta.get("purpose") != purpose
                    or meta.get("latent_dim") != 16
                    or meta.get("log_variance_bounds") !=
                    [float(state[prefix + "logvar_min"]),
                     float(state[prefix + "logvar_max"])]
                    or meta.get("sampler_domain") != "TeamBPurposeDrawV1"
                    or meta.get("release_contract") != "immutable_one_shot_batch"):
                raise ValueError("released model/schema/sampling metadata differs")
            if meta.get("draws_per_person") != (1 if meta.get("stochastic") is True else 0):
                raise ValueError("released draw count differs from sampling flag")
            x = torch.from_numpy(np.asarray(x_rows[i], dtype=np.float32)).reshape(1, 83)
            with torch.no_grad():
                h = torch.relu(torch.nn.functional.linear(
                    x, state[prefix + "body.0.weight"], state[prefix + "body.0.bias"]))
                h = torch.relu(torch.nn.functional.linear(
                    h, state[prefix + "body.2.weight"], state[prefix + "body.2.bias"]))
                mean = torch.nn.functional.linear(
                    h, state[prefix + "mean.weight"], state[prefix + "mean.bias"])
                logvar = torch.nn.functional.linear(
                    h, state[prefix + "log_variance.weight"],
                    state[prefix + "log_variance.bias"])
                logvar = logvar.clamp(state[prefix + "logvar_min"],
                                      state[prefix + "logvar_max"])
                if meta.get("stochastic") is True:
                    domain = json.dumps(["TeamBPurposeDrawV1", source_id, ps,
                                         purpose, str(int(person))],
                                        separators=(",", ":")).encode()
                    seed = int.from_bytes(hmac.new(secret, domain, hashlib.sha256).digest()[:8], "big")
                    noise = torch.randn(mean.shape,
                                        generator=torch.Generator(device="cpu").manual_seed(seed))
                    z = mean + torch.exp(0.5 * logvar) * noise
                elif meta.get("stochastic") is False:
                    z = mean
                else:
                    raise ValueError("sampling flag missing")
            actual = np.asarray(view["features"], dtype=np.float32)
            if not np.array_equal(actual, z.squeeze(0).numpy()):
                raise ValueError("released z differs from actual permitted input/model/draw")


def _walk(prefix: str, value: Any, out: dict[str, float | str]) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise ValueError("invalid transmitted mapping key")
            _walk(f"{prefix}.{key}", value[key], out)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _walk(f"{prefix}[{i}]", item, out)
    elif isinstance(value, bool):
        out[prefix] = float(value)
    elif isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("nonfinite transmitted value")
        out[prefix] = numeric
    elif isinstance(value, str):
        out[prefix] = value
    elif value is None:
        out[prefix] = "<NULL>"
    else:
        raise ValueError("unsupported transmitted field type")


def _flat(view: dict, purpose: str) -> dict[str, float | str]:
    if set(view) != WIRE_KEYS:
        raise ValueError("wire key set differs")
    z = np.asarray(view["features"], dtype=np.float64)
    logits = np.asarray(view["logits"], dtype=np.float64)
    q = np.asarray(view["probabilities"], dtype=np.float64)
    d = view["decision"]
    if (z.shape != (16,) or logits.shape != (WIDTH[purpose],)
            or q.shape != (WIDTH[purpose],) or not np.isfinite(z).all()
            or not np.isfinite(logits).all() or not np.isfinite(q).all()
            or np.any(q < 0) or abs(float(q.sum()) - 1.0) > 1e-8
            or type(d) is not int or d != int(np.argmax(q))):
        raise ValueError("recipient probability/decision/wire shape differs")
    out: dict[str, float | str] = {}
    for key in sorted(view):
        _walk(key, view[key], out)
    return out


def _calibrate(raw: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.25 <= alpha <= 4.0:
        raise ValueError("calibration alpha out of frozen bounds")
    if alpha == 1.0:
        return raw.copy()
    p = np.maximum(raw, 1e-12)
    p /= p.sum(axis=1, keepdims=True)
    logp = np.log(p)
    scaled = alpha * logp
    scaled -= scaled.max(axis=1, keepdims=True)
    e = np.exp(scaled)
    return e / e.sum(axis=1, keepdims=True)


def _public_replay(rows: list[dict], purpose: str,
                   head: tuple[np.ndarray, np.ndarray], cal: dict) -> np.ndarray:
    w, b = head
    projected = []
    raw = []
    for views in rows:
        view = views[purpose]
        z = torch.from_numpy(np.asarray(view["features"], dtype=np.float32)).reshape(1, -1)
        with torch.no_grad():
            logits = torch.nn.functional.linear(z, torch.from_numpy(w),
                                                torch.from_numpy(b)).squeeze(0)
            q0 = torch.softmax(logits.to(torch.float64), dim=0)
        lp = logits.numpy().astype(np.float64)
        if not np.array_equal(lp, np.asarray(view["logits"], dtype=np.float64)):
            raise ValueError("released logits bypass public head on released z")
        projected.append(lp)
        raw.append(q0.numpy())
    projected = np.asarray(projected, dtype=np.float64)
    q = _calibrate(np.asarray(raw, dtype=np.float64), float(cal["alpha"]))
    actual = np.asarray([views[purpose]["probabilities"] for views in rows], dtype=np.float64)
    decision = np.asarray([views[purpose]["decision"] for views in rows], dtype=np.int64)
    if not np.array_equal(q, actual) or not np.array_equal(q.argmax(axis=1), decision):
        raise ValueError("released probability/decision bypasses fitted calibration")
    cal_sha = hashlib.sha256((json.dumps(cal, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False) + "\n").encode()).hexdigest()
    for views in rows:
        meta = views[purpose]["metadata"]
        if (meta.get("purpose") != purpose or meta.get("calibration_family") != "global_temperature"
                or meta.get("temperature") != 1.0 / float(cal["alpha"])
                or meta.get("calibration_record_sha256") != cal_sha):
            raise ValueError("transmitted calibration metadata differs")
    return np.concatenate((projected, projected - projected.max(axis=1, keepdims=True)), axis=1)


def rebuild_view(rows: list[dict], view: str,
                 heads: dict[str, tuple[np.ndarray, np.ndarray]],
                 calibrators: dict[str, dict]) -> dict[str, Any]:
    """Recreate S2's full numeric/constant wire declaration independently."""
    if view not in (*PURPOSES, "pair") or not rows:
        raise ValueError("unknown/empty disclosure view")
    flat = []
    for views in rows:
        if set(views) != set(PURPOSES):
            raise ValueError("aligned pair missing")
        if view == "pair":
            item = {f"{p}.{key}": value for p in PURPOSES
                    for key, value in _flat(views[p], p).items()}
        else:
            item = _flat(views[view], view)
        flat.append(item)
    keys = tuple(sorted(flat[0]))
    if any(tuple(sorted(item)) != keys for item in flat):
        raise ValueError("person-varying wire schema")
    components, wire_constants, bindings, columns, metadata_columns = [], [], [], [], []
    for key in keys:
        values = [item[key] for item in flat]
        if all(isinstance(x, str) for x in values):
            if len(set(values)) != 1:
                raise ValueError("person-varying string needs a categorical reader")
            wire_constants.append((key, values[0]))
        elif all(not isinstance(x, str) for x in values):
            col = np.asarray(values, dtype=np.float64).reshape(-1, 1)
            if not np.isfinite(col).all():
                raise ValueError("nonfinite numeric wire")
            start = len(columns)
            columns.append(col)
            components.append(key)
            bindings.append((key, start, start + 1))
            if key.startswith("metadata.") or ".metadata." in key:
                metadata_columns.append(start)
        else:
            raise ValueError("mixed string/numeric wire field")
    x = np.concatenate(columns, axis=1)
    relevant = PURPOSES if view == "pair" else (view,)
    projected = np.concatenate([_public_replay(rows, p, heads[p], calibrators[p])
                                for p in relevant], axis=1)
    return {"x": x, "x_defense_aware": np.concatenate((x, projected), axis=1),
            "x_metadata": x[:, metadata_columns] if metadata_columns else None,
            "components": tuple(components), "wire_components": keys,
            "component_columns": tuple(bindings),
            "wire_constants": tuple(wire_constants)}


def compare_frozen_view(rebuilt: dict, path: Path, receipt: dict) -> None:
    """Check every transmitted column and optional public derived view."""
    with np.load(path, allow_pickle=False) as saved:
        expected = {"row_id", "unit", "x", "x_defense_aware"}
        if rebuilt["x_metadata"] is not None:
            expected.add("x_metadata")
        if set(saved.files) != expected:
            raise ValueError("frozen attack view has extra/missing matrix")
        for name in ("x", "x_defense_aware", "x_metadata"):
            value = rebuilt[name]
            if value is not None and not np.array_equal(value, saved[name]):
                raise ValueError(f"reconstructed {name} differs from fitted view")
    schema = {key: rebuilt[key] for key in ("components", "wire_components",
                                            "component_columns", "wire_constants")}
    receipt_schema = receipt["view_schema"]
    if json.loads(json.dumps(schema)) != receipt_schema:
        raise ValueError("reader receipt omits wire field/binding")
