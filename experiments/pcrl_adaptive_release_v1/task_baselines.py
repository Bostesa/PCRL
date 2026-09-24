"""Same-input task-only token and matched nested-partition controls.

These are 2018 development baselines, not claims of population privacy. They
reuse the Branch B legal feature set, T32 parents, 17-token wire, training
roles, candidate thresholds and child support floor. No assessment row is
opened by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
from . import fit_b, refinement, roles


SUPPORT_FLOOR = 100
CHECK_GAIN_RATIO = fit_b.MIN_CHECK_GAIN_RATIO
PARTITION_POLICIES = ("task_only", "joint_risk", "random_eligible")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_hash(rows: Mapping, valid: np.ndarray) -> str:
    digest = hashlib.sha256()
    for household in np.asarray(rows["households"]):
        value = str(household).encode()
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    for name, value in (("x", rows["x"]), ("ha", rows["ha"]),
                        ("weights", rows["weights"]),
                        ("task_labels", rows["labels"]["same_residence"]),
                        ("valid", valid)):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    value = values[order]
    weight = weights[order]
    midpoint = (np.cumsum(weight)-.5*weight)/weight.sum()
    cuts = np.interp(np.arange(1, 17)/17, midpoint, value,
                     left=float(value[0]), right=float(value[-1]))
    if cuts.shape != (16,) or not np.isfinite(cuts).all() or np.any(np.diff(cuts) < 0):
        raise ValueError("invalid frozen task quantizer thresholds")
    return cuts


@dataclass
class TaskOnlyPredictor:
    mean: np.ndarray
    scale: np.ndarray
    model: LogisticRegression
    cuts: np.ndarray
    training_sha256: str

    def predict_probability(self, inputs: RuntimeInputs) -> np.ndarray:
        if not isinstance(inputs, RuntimeInputs):
            raise TypeError("task-only predictor accepts RuntimeInputs(X_A,H_A) only")
        x = np.asarray(inputs.features(), dtype=np.float64)
        if self.mean.shape != (36,) or self.scale.shape != (36,) or np.any(self.scale <= 0):
            raise ValueError("invalid frozen task standardizer")
        p = np.asarray(self.model.predict_proba((x-self.mean)/self.scale)[:, 1], dtype=float)
        if p.shape != (len(x),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("invalid frozen task probabilities")
        return p

    def token_codes(self, inputs: RuntimeInputs) -> np.ndarray:
        if self.cuts.shape != (16,) or not np.isfinite(self.cuts).all() or np.any(np.diff(self.cuts) < 0):
            raise ValueError("invalid 17-token task quantizer")
        return np.searchsorted(self.cuts, self.predict_probability(inputs),
                               side="right").astype(np.int64)

    def release(self, inputs: RuntimeInputs) -> dict:
        token = self.token_codes(inputs)
        return {"h_a": np.asarray(inputs.h_a).copy(), "token": token}


def fit_task_only(rows: Mapping, *, seed: int) -> tuple[TaskOnlyPredictor, dict]:
    """Fit once on global nuisance_train households; never use inner audit."""
    if not {"x", "ha", "weights", "households", "labels"}.issubset(rows):
        raise ValueError("task-only fit needs nuisance_train local inputs and task labels")
    inputs = RuntimeInputs(np.asarray(rows["x"]), np.asarray(rows["ha"]))
    n = len(inputs.x_a)
    weights = np.asarray(rows["weights"], dtype=float)
    households = np.asarray(rows["households"])
    y = np.asarray(rows["labels"]["same_residence"])
    if (n < 2 or weights.shape != (n,) or households.shape != (n,) or
            y.shape != (n,) or y.dtype.kind not in "iu" or
            np.any((y < -1) | (y > 1)) or not np.isfinite(weights).all() or
            np.any(weights < 0)):
        raise ValueError("invalid nuisance-training task rows")
    if any(roles.role_of(h) != "nuisance_train" for h in households):
        raise ValueError("task-only predictor may fit nuisance_train households only")
    valid = (y >= 0) & (weights > 0)
    if not valid.any() or len(np.unique(y[valid])) != 2:
        raise ValueError("both residence classes need positive-weight nuisance support")
    x = np.asarray(inputs.features(), dtype=np.float64)
    mean = x[valid].mean(0)
    scale = x[valid].std(0)
    scale[scale < 1e-12] = 1.
    standardized = (x[valid]-mean)/scale
    fit_weights = weights[valid] * (int(valid.sum())/weights[valid].sum())
    model = LogisticRegression(C=1., max_iter=500, solver="lbfgs",
                               random_state=int(seed))
    model.fit(standardized, y[valid], sample_weight=fit_weights)
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("residence class order changed")
    fitted_p = model.predict_proba(standardized)[:, 1]
    cuts = _weighted_quantiles(fitted_p, weights[valid])
    training_sha = _training_hash(rows, valid)
    fitted = TaskOnlyPredictor(mean, scale, model, cuts, training_sha)
    receipt = {"schema": "pcrl-task-only-token-v1", "training_role": "nuisance_train",
               "training_sha256": training_sha, "seed": int(seed),
               "fitted_people": int(valid.sum()),
               "fitted_households": int(len(np.unique(households[valid]))),
               "excluded_missing_or_zero_weight": int((~valid).sum()),
               "class_support": np.bincount(y[valid], minlength=2).tolist(),
               "input_contract": "RuntimeInputs(X_A32,H_A4) only",
               "supervision": "same_residence on nuisance_train only",
               "recipe": "standardized logistic C=1 lbfgs max_iter=500; PWGTP-normalized fit weights",
               "quantile_thresholds_from": "nuisance_train predictions only",
               "quantizer": "16 PWGTP quantile thresholds; searchsorted side=right; 17 existing token IDs",
               "distinct_thresholds": int(len(np.unique(cuts))),
               "service_preserved": "H_A copied byte-for-byte by release",
               "outer_labels_accessed": False}
    return fitted, receipt


def save_task_only(directory: str | Path, model: TaskOnlyPredictor,
                   receipt: Mapping) -> dict:
    root = Path(directory)
    if "private" not in root.parts:
        raise ValueError("trained task-only model requires private path")
    if root.exists() and any(root.iterdir()):
        old, existing = load_task_only(root)
        if (old.training_sha256 != model.training_sha256 or
                any(existing.get(key) != value for key, value in receipt.items())):
            raise ValueError("existing task-only model is a different unit")
        complete_path = root / "COMPLETE.json"
        if complete_path.exists():
            complete = json.loads(complete_path.read_text())
            if complete.get("status") != "COMPLETE" or complete.get("artifacts") != _inventory(root):
                raise ValueError("task-only complete artifact inventory differs")
        else:
            if set(_inventory(root)) != {"TASK_ONLY.json", "task_only.joblib"}:
                raise FileExistsError("partial task-only model retained for technical diagnosis")
            _write_json_once(complete_path, {
                "schema": "pcrl-task-only-complete-v1", "status": "COMPLETE",
                "training_sha256": existing["training_sha256"],
                "model_sha256": existing["model_sha256"],
                "artifacts": _inventory(root)})
        return existing
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    temp = root / f"task_only.joblib.tmp.{os.getpid()}"
    joblib.dump(model, temp, compress=3)
    os.chmod(temp, 0o600)
    os.replace(temp, root / "task_only.joblib")
    saved = {**receipt, "model_sha256": _sha_file(root / "task_only.joblib")}
    temporary = root / f"TASK_ONLY.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(saved, sort_keys=True, indent=2, allow_nan=False)+"\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, root / "TASK_ONLY.json")
    load_task_only(root)
    _write_json_once(root / "COMPLETE.json", {
        "schema": "pcrl-task-only-complete-v1", "status": "COMPLETE",
        "training_sha256": saved["training_sha256"],
        "model_sha256": saved["model_sha256"],
        "artifacts": _inventory(root)})
    return saved


def load_task_only(directory: str | Path) -> tuple[TaskOnlyPredictor, dict]:
    root = Path(directory)
    receipt = json.loads((root / "TASK_ONLY.json").read_text())
    path = root / "task_only.joblib"
    if receipt.get("schema") != "pcrl-task-only-token-v1" or _sha_file(path) != receipt.get("model_sha256"):
        raise ValueError("task-only model hash/receipt mismatch")
    model = joblib.load(path)
    if not isinstance(model, TaskOnlyPredictor) or model.training_sha256 != receipt["training_sha256"]:
        raise ValueError("task-only model training hash differs")
    return model, receipt


def task_only_control_law(model: TaskOnlyPredictor, inputs: RuntimeInputs, *,
                          mode: str, publish: float,
                          constant_token: int = 0) -> np.ndarray:
    """Private per-person law for exact token audits of task-only controls.

    The recipient receives one sampled token, never this probability vector.
    Task-only tokens may differ inside a T32 cell, so these controls cannot be
    substituted by a T32×17 channel table without changing the method.
    """
    rate = float(publish)
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("publish probability outside [0,1]")
    if mode not in ("unmodified", "constant_replacement", "randomized_response"):
        raise ValueError("unknown task-only control law")
    if not isinstance(constant_token, int) or not 0 <= constant_token < 17:
        raise ValueError("constant token outside existing 17-token alphabet")
    token = model.token_codes(inputs)
    law = np.eye(17, dtype=float)[token]
    if mode == "constant_replacement":
        law = rate*law
        law[:, constant_token] += 1-rate
    elif mode == "randomized_response":
        law = rate*law + (1-rate)/17
    if not np.isfinite(law).all() or np.any(law < 0) or np.max(np.abs(law.sum(1)-1)) > 1e-12:
        raise ValueError("invalid exact task-only control law")
    return law


def _runtime_row_digest(x: np.ndarray, ha: np.ndarray) -> str:
    """Bind a replay ID to its immutable local inputs without storing them."""
    digest = hashlib.sha256()
    for array in (x, ha):
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _task_model_digest(model: TaskOnlyPredictor) -> str:
    if not isinstance(model, TaskOnlyPredictor):
        raise TypeError("frozen TaskOnlyPredictor required")
    digest = hashlib.sha256(model.training_sha256.encode())
    for name, array in (("mean", model.mean), ("scale", model.scale),
                        ("cuts", model.cuts), ("coef", model.model.coef_),
                        ("intercept", model.model.intercept_),
                        ("classes", model.model.classes_)):
        value = np.ascontiguousarray(array)
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _release_identifier(value) -> tuple[str, str | int]:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("boolean release ID is invalid")
    if isinstance(value, str) and value:
        return ("str", value)
    if isinstance(value, (int, np.integer)):
        return ("int", int(value))
    raise ValueError("release IDs must be nonempty strings or integers")


class TaskOnlyReleaseSession:
    """Sample one registered 17-token control per immutable person/release.

    HMAC gives stable pseudorandom draws across restarts without exposing the
    secret. An optional private cache also rejects ID/input changes across
    restarts; without it that rejection is confined to this process. Neither
    the person-specific law nor the model scores appear on the wire.
    """

    def __init__(self, model: TaskOnlyPredictor, *, mode: str, publish: float,
                 replay_key: bytes, release_id: str,
                 cache_path: str | Path | None = None):
        if not isinstance(replay_key, bytes) or len(replay_key) < 32:
            raise ValueError("private replay key must contain at least 32 bytes")
        if not isinstance(release_id, str) or not release_id:
            raise ValueError("nonempty release_id required")
        if mode not in ("unmodified", "constant_replacement", "randomized_response"):
            raise ValueError("unknown registered control mode")
        if publish not in (.5, .75, .9, 1.) or (mode == "unmodified" and publish != 1.):
            raise ValueError("publish must be a registered control rate")
        self._model = model
        self._model_sha256 = _task_model_digest(model)
        self._mode = mode
        self._publish = float(publish)
        self._replay_key = replay_key
        self._release_id = release_id
        self._cache = {}
        key_binding = hmac.new(replay_key, b"PCRL_TASK_ONLY_CACHE_KEY_BINDING_V1",
                               hashlib.sha256).hexdigest()
        self._config_sha256 = hashlib.sha256(json.dumps(
            ["PCRL_TASK_ONLY_RELEASE_V1", self._model_sha256, mode,
             self._publish, release_id, key_binding],
            separators=(",", ":")).encode()).hexdigest()
        if cache_path is None:
            self._cache_path = None
        else:
            requested = Path(cache_path)
            target = requested.resolve()
            if "private" not in requested.parts or "private" not in target.parts:
                raise ValueError("persistent replay cache must remain private")
            self._cache_path = target

    def _id_key(self, identifier: tuple[str, str | int]) -> str:
        payload = json.dumps(["ID", self._release_id, identifier],
                             separators=(",", ":")).encode()
        return hmac.new(self._replay_key, payload, hashlib.sha256).hexdigest()

    def _uniform(self, identifier: tuple[str, str | int], row_digest: str) -> float:
        payload = json.dumps(["PCRL_TASK_ONLY_REPLAY_V1", self._config_sha256,
                              identifier, row_digest], separators=(",", ":")).encode()
        draw = hmac.new(self._replay_key, payload, hashlib.sha256).digest()
        return int.from_bytes(draw[:8], "big") / 2**64

    def _read_persistent(self) -> dict:
        if self._cache_path is None or not self._cache_path.exists():
            return {}
        document = json.loads(self._cache_path.read_text())
        if (document.get("schema") != "pcrl-task-only-replay-cache-v1" or
                document.get("config_sha256") != self._config_sha256 or
                not isinstance(document.get("records"), dict)):
            raise ValueError("private replay cache has a different model or release")
        return document["records"]

    def _write_persistent(self, records: dict) -> None:
        target = self._cache_path
        assert target is not None
        document = {"schema": "pcrl-task-only-replay-cache-v1",
                    "config_sha256": self._config_sha256,
                    "records": records}
        temporary = target.with_name(target.name+f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(document, sort_keys=True,
                                        separators=(",", ":"))+"\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)

    def emit(self, inputs: RuntimeInputs, release_ids: Sequence) -> dict:
        if not isinstance(inputs, RuntimeInputs):
            raise TypeError("task-only release accepts RuntimeInputs(X_A,H_A) only")
        if _task_model_digest(self._model) != self._model_sha256:
            raise ValueError("frozen task-only model changed after session construction")
        ids = [_release_identifier(value) for value in release_ids]
        if len(ids) != len(inputs.h_a):
            raise ValueError("one release ID required per original person")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate release IDs within a batch")
        law = task_only_control_law(self._model, inputs, mode=self._mode,
                                    publish=self._publish, constant_token=0)
        if self._cache_path is None:
            records = dict(self._cache)
            tokens = self._sample_batch(inputs, ids, law, records)
            self._cache = records
        else:
            if self._cache_path.parent.exists():
                if stat.S_IMODE(self._cache_path.parent.stat().st_mode) & 0o077:
                    raise ValueError("private replay cache directory is accessible to others")
            else:
                self._cache_path.parent.mkdir(parents=True, mode=0o700)
            lock_path = self._cache_path.with_name(self._cache_path.name+".lock")
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            with os.fdopen(descriptor, "r+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                records = self._read_persistent()
                tokens = self._sample_batch(inputs, ids, law, records)
                self._write_persistent(records)
                self._cache = records.copy()
        h = np.array(inputs.h_a, copy=True)
        if h.dtype != np.asarray(inputs.h_a).dtype or h.tobytes() != np.asarray(inputs.h_a).tobytes():
            raise AssertionError("H_A service byte parity failed")
        h.flags.writeable = False
        tokens.flags.writeable = False
        return {"h_a": h, "token": tokens}

    def _sample_batch(self, inputs: RuntimeInputs, ids: list, law: np.ndarray,
                      records: dict) -> np.ndarray:
        tokens = np.empty(len(ids), dtype=np.int64)
        for i, identifier in enumerate(ids):
            key = self._id_key(identifier)
            row_digest = _runtime_row_digest(inputs.x_a[i], inputs.h_a[i])
            if key in records:
                previous = records[key]
                if previous["input_sha256"] != row_digest:
                    raise ValueError("release ID reused with changed local or service inputs")
                token = int(previous["token"])
                if not 0 <= token < 17:
                    raise ValueError("private replay cache has invalid token")
                cumulative = np.cumsum(law[i]); cumulative[-1] = 1.
                expected = int(np.searchsorted(
                    cumulative, self._uniform(identifier, row_digest), side="right"))
                if token != expected:
                    raise ValueError("private replay cache token differs from keyed law")
            else:
                cumulative = np.cumsum(law[i])
                cumulative[-1] = 1.
                token = int(np.searchsorted(
                    cumulative, self._uniform(identifier, row_digest), side="right"))
                records[key] = {"input_sha256": row_digest, "token": token}
            tokens[i] = token
        return tokens


def _risk_matrix(features: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([features[f"sex_risk_{j}"] for j in range(2)] +
                           [features[f"race_risk_{j}"] for j in range(9)])


def _joint_risk_gain(rows: Mapping, risk: np.ndarray, active: np.ndarray,
                     left: np.ndarray) -> float:
    weight = np.asarray(rows["weights"], dtype=float)
    if weight.shape != active.shape or not np.isfinite(weight).all() or np.any(weight < 0) or weight.sum() <= 0:
        raise ValueError("invalid original-person weights for joint-risk control")
    mass = .5/len(weight) + .5*weight/weight.sum()
    right = active & ~left
    if not np.any(left) or not np.any(right):
        return 0.
    ml, mr = mass[left].sum(), mass[right].sum()
    if ml <= 0 or mr <= 0:
        return 0.
    mean_l = np.average(risk[left], axis=0, weights=mass[left])
    mean_r = np.average(risk[right], axis=0, weights=mass[right])
    return float(ml*mr/(ml+mr) * np.square(mean_l-mean_r).sum())


def task_contributions(frozen_decoder, rows: Mapping) -> np.ndarray:
    """Exact 17-action task CE with half-U/half-PWGTP person normalization.

    These values may guide task-only split selection on development roles.
    They are never a deployment feature or an assessment-label lookahead.
    """
    from .fit_a import task_person_losses

    valid, losses = task_person_losses(frozen_decoder, rows)
    if not np.any(valid):
        raise ValueError("task-only partition needs supported task coefficient rows")
    weights = np.asarray(rows["weights"], dtype=float)
    if (weights.shape != valid.shape or not np.isfinite(weights).all() or
            np.any(weights < 0) or weights[valid].sum() <= 0):
        raise ValueError("invalid original-person task weights")
    out = np.zeros((len(valid), 17), dtype=float)
    mass = .5/int(valid.sum()) + .5*weights[valid]/weights[valid].sum()
    out[valid] = mass[:, None]*losses
    return out


def select_partition_control(policy: str, coefficient_rows: Mapping,
                             selection_rows: Mapping, frozen_nuisance, *,
                             task_fit_rows: np.ndarray | None = None,
                             task_selection_rows: np.ndarray | None = None,
                             target_states: int = 64,
                             feature_names: Sequence[str] = tuple(sorted(refinement.ALLOWED_FEATURES)),
                             seed: int, min_households: int = SUPPORT_FLOOR,
                             min_effective_households: float = SUPPORT_FLOOR) -> dict:
    """Build a matched nested T32 partition without privacy-price selection.

    Task-only uses fixed decoder task contributions without lambda. Joint-risk
    uses only frozen local nuisance probabilities. Random chooses uniformly
    among supported train-quantile candidates. No policy reads inner_check.
    """
    if policy not in PARTITION_POLICIES:
        raise ValueError("unknown registered partition control")
    if not isinstance(target_states, int) or not 32 <= target_states <= 128:
        raise ValueError("registered matched leaf budget is 32..128")
    if min_households != SUPPORT_FLOOR or min_effective_households != SUPPORT_FLOOR:
        raise ValueError("registered 100-household support floor cannot change")
    if not feature_names or any(name not in refinement.ALLOWED_FEATURES for name in feature_names):
        raise ValueError("only declared deployable split features are allowed")
    fit_t, fit_features = fit_b.stored_deployable_features(coefficient_rows, frozen_nuisance)
    select_t, select_features = fit_b.stored_deployable_features(selection_rows, frozen_nuisance)
    nfit, nselect = len(fit_t), len(select_t)
    fit_hh = np.asarray(coefficient_rows["households"])
    select_hh = np.asarray(selection_rows["households"])
    if fit_hh.shape != (nfit,) or select_hh.shape != (nselect,) or np.intersect1d(fit_hh, select_hh).size:
        raise ValueError("partition control resources require disjoint households")
    if policy == "task_only":
        g_fit = np.asarray(task_fit_rows, dtype=float)
        g_select = np.asarray(task_selection_rows, dtype=float)
        if (g_fit.shape != (nfit, 17) or g_select.shape != (nselect, 17) or
                not np.isfinite(g_fit).all() or not np.isfinite(g_select).all()):
            raise ValueError("task-only split requires aligned frozen 17-action task contributions")
    else:
        g_fit = np.zeros((nfit, 17))
        g_select = np.zeros((nselect, 17))
    risk_fit = _risk_matrix(fit_features)
    risk_select = _risk_matrix(select_features)
    rng = np.random.default_rng(seed)
    partition = refinement.NestedPartition.base(32)
    history = []
    candidate_history = []
    status = "max_states_reached" if target_states == 32 else "support_limited"
    while len(partition.parents) < target_states:
        fit_leaf = partition.route(fit_t, fit_features)
        select_leaf = partition.route(select_t, select_features)
        candidates = []
        for leaf in range(len(partition.parents)):
            ranked = refinement.rank_splits(
                leaf_ids=fit_leaf, leaf=leaf, features=fit_features,
                priced_rows=g_fit, households=fit_hh,
                weights=np.asarray(coefficient_rows["weights"]),
                min_households=SUPPORT_FLOOR,
                min_effective_weight=SUPPORT_FLOOR,
                feature_names=feature_names,
                checking=(select_leaf, select_features, g_select, select_hh))
            for candidate in ranked:
                active_fit = fit_leaf == leaf
                left_fit = active_fit & (fit_features[candidate.feature_name] <= candidate.threshold)
                active_select = select_leaf == leaf
                left_select = active_select & (select_features[candidate.feature_name] <= candidate.threshold)
                right_select = active_select & ~left_select
                if not left_select.any() or not right_select.any():
                    continue
                if policy == "task_only":
                    gain, checking_gain = candidate.gain, candidate.checking_gain
                elif policy == "joint_risk":
                    gain = _joint_risk_gain(coefficient_rows, risk_fit, active_fit, left_fit)
                    checking_gain = _joint_risk_gain(selection_rows, risk_select,
                                                     active_select, left_select)
                else:
                    gain = checking_gain = None
                candidates.append({"candidate": candidate, "gain": gain,
                                   "checking_gain": checking_gain})
        candidate_history.append(len(candidates))
        if not candidates:
            status = "support_limited"
            break
        if policy != "random_eligible":
            eligible = [row for row in candidates
                        if row["gain"] is not None and
                        row["checking_gain"] is not None and
                        row["checking_gain"] + 1e-12 >= CHECK_GAIN_RATIO*row["gain"]]
            if not eligible:
                status = "no_stable_objective_gain"
                break
            picked = min(eligible, key=lambda row: (
                -row["gain"], row["candidate"].leaf,
                row["candidate"].feature_name, row["candidate"].threshold))
        else:
            candidates.sort(key=lambda row: (row["candidate"].leaf,
                                             row["candidate"].feature_name,
                                             row["candidate"].threshold))
            picked = candidates[int(rng.integers(len(candidates)))]
        c = picked["candidate"]
        partition = partition.split(c.leaf, c.feature_name, c.threshold)
        history.append({"leaf": c.leaf, "new_leaf": len(partition.parents)-1,
                        "feature_name": c.feature_name, "threshold": c.threshold,
                        "fitting_gain": picked["gain"],
                        "inner_selection_gain": picked["checking_gain"],
                        "children_unique_households": list(c.children_households),
                        "children_effective_households": list(c.children_effective_weight),
                        "selection_rule": policy})
        status = "max_states_reached" if len(partition.parents) == target_states else "support_limited"
    return {"policy": policy, "partition": partition, "history": history,
            "candidate_counts": candidate_history,
            "status": status, "realized_states": len(partition.parents),
            "target_states": target_states, "seed": int(seed),
            "support_floor_unique_households": SUPPORT_FLOOR,
            "support_floor_effective_households": SUPPORT_FLOOR,
            "outer_labels_accessed": False,
            "inner_check_labels_accessed": False}


def _role_hash(rows: Mapping) -> str:
    digest = hashlib.sha256()
    for key in ("x", "ha", "token_codes", "teacher_p", "residual", "weights"):
        array = np.ascontiguousarray(rows[key])
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    for household in rows["households"]:
        value = str(household).encode()
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    array = np.ascontiguousarray(rows["labels"]["same_residence"])
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _inventory(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): _sha_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "COMPLETE.json"}


def _write_json_once(path: Path, value: Mapping) -> None:
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError("existing immutable partition-control artifact differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    temporary.write_text(text)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def fit_partition_controls_from_roles(coefficient_rows: Mapping,
                                      selection_rows: Mapping, frozen_nuisance,
                                      frozen_decoder, target_partition: refinement.NestedPartition,
                                      output_dir: str | Path, *, seed: int,
                                      source_pins: Mapping[str, str],
                                      feature_names: Sequence[str] = tuple(sorted(refinement.ALLOWED_FEATURES))) -> dict:
    """Save three independently selected, leaf-matched control partitions.

    Only aggregate split rules and hashes are stored. Subsequent channel fits
    and audits belong to the central scientific queue, not this routine.
    """
    root = Path(output_dir)
    if "private" not in root.parts:
        raise ValueError("fitted partition controls require a private output path")
    if not isinstance(target_partition, refinement.NestedPartition):
        raise TypeError("target must be a frozen B nested partition")
    if not source_pins or any(not isinstance(value, str) or len(value) != 64
                              for value in source_pins.values()):
        raise ValueError("full frozen source SHA-256 pins required")
    target_states = len(target_partition.parents)
    target_record = target_partition.to_record()
    target_sha = hashlib.sha256(json.dumps(
        target_record, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    expected = {"schema": "pcrl-matched-partition-controls-v1",
                "target_states": target_states, "seed": int(seed),
                "target_partition_sha256": target_sha,
                "feature_names": list(feature_names),
                "source_pins": dict(source_pins),
                "coefficient_role_sha256": _role_hash(coefficient_rows),
                "inner_selection_role_sha256": _role_hash(selection_rows),
                "module_sha256": _sha_file(Path(__file__)),
                "support_floor_unique_households": SUPPORT_FLOOR,
                "support_floor_effective_households": SUPPORT_FLOOR,
                "inner_check_labels_accessed": False,
                "outer_labels_accessed": False}
    complete_path = root / "COMPLETE.json"
    if complete_path.exists():
        prior = json.loads(complete_path.read_text())
        if any(prior.get(key) != value for key, value in expected.items()):
            raise ValueError("completed partition controls are a different scientific unit")
        if prior.get("artifact_sha256") != _inventory(root):
            raise ValueError("partition-control artifact inventory differs")
        return prior
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("partial partition-control unit retained for diagnosis")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    _write_json_once(root / "INPUTS.json", expected)
    task_fit = task_contributions(frozen_decoder, coefficient_rows)
    task_select = task_contributions(frozen_decoder, selection_rows)
    controls = {}
    for policy in PARTITION_POLICIES:
        result = select_partition_control(
            policy, coefficient_rows, selection_rows, frozen_nuisance,
            task_fit_rows=task_fit, task_selection_rows=task_select,
            target_states=target_states, feature_names=feature_names,
            seed=int(seed))
        record = {key: value for key, value in result.items() if key != "partition"}
        record["partition"] = result["partition"].to_record()
        record["leaf_count_matches_B"] = result["realized_states"] == target_states
        path = root / "policies" / f"{policy}.json"
        _write_json_once(path, record)
        controls[policy] = {"status": result["status"],
                            "realized_states": result["realized_states"],
                            "leaf_count_matches_B": record["leaf_count_matches_B"],
                            "relative_path": str(path.relative_to(root)),
                            "artifact_sha256": _sha_file(path)}
    receipt = {**expected, "status": "COMPLETE", "controls": controls,
               "all_leaf_counts_matched": all(item["leaf_count_matches_B"]
                                              for item in controls.values()),
               "artifact_sha256": _inventory(root)}
    _write_json_once(complete_path, receipt)
    return receipt


def run_partition_controls(anchor: int, delta: float, index_path: str | Path,
                           a_center_dir: str | Path, b_center_dir: str | Path,
                           output_dir: str | Path, *, seed: int) -> dict:
    """Queue entry: reuse pinned B nuisance/partition and A decoder only."""
    from experiments.pcrl_task_aligned_cuts_v1 import data
    from . import fit_a, nuisance

    b_root = Path(b_center_dir)
    b_complete_path = b_root / "COMPLETE.json"
    b_complete = json.loads(b_complete_path.read_text())
    if (b_complete.get("anchor") != anchor or b_complete.get("delta") != delta or
            b_complete.get("status") not in ("COMPLETE", "SUPPORT_LIMITED_ALIAS_A",
                                              "NO_ACCEPTED_SPLIT_ALIAS_A") or
            b_complete.get("artifact_sha256") != fit_a._inventory(b_root)):
        raise ValueError("frozen B center is incomplete or changed")
    sanitized = (Path(__file__).resolve().parents[2] /
                 data.SANITIZED_RELATIVE / "SANITIZATION.json")
    if not sanitized.is_file():
        raise RuntimeError("sanitized label-stripped 2018 receipt required before deserialization")
    index = data.index(index_path)
    if data._sanitized_record(index, anchor) is None:
        raise RuntimeError("sanitized prepared member required before deserialization")
    prepared = data.load_prepared(index, anchor)
    if "labels" in prepared["ctx"]["pools"]["attacker_validation"]:
        raise RuntimeError("outer labels appeared before assessment lock")
    coefficient = roles.pooled_role(prepared, "coefficient_split")
    selection = roles.pooled_role(prepared, "inner_selection")
    for name, rows in (("coefficient_split", coefficient),
                       ("inner_selection", selection)):
        if fit_a._role_fingerprint(rows) != b_complete["role_input_sha256"][name]:
            raise ValueError("partition-control role differs from B center")
    frozen_nuisance, nuisance_receipt = nuisance.load_frozen_nuisance(b_root / "nuisance")
    if nuisance_receipt["model_sha256"] != b_complete["nuisance_model_sha256"]:
        raise ValueError("frozen B nuisance hash differs")
    partition = refinement.NestedPartition.from_record(
        json.loads((b_root / "PARTITION.json").read_text()))
    selected_a = fit_b.load_a_selected(a_center_dir, anchor=anchor, delta=delta)
    if selected_a["complete_receipt_sha256"] != b_complete["a_complete_receipt_sha256"]:
        raise ValueError("frozen A decoder source differs from B fit")
    return fit_partition_controls_from_roles(
        coefficient, selection, frozen_nuisance, selected_a["decoder"],
        partition, output_dir, seed=seed,
        source_pins={"frozen_decoder_sha256": selected_a["decoder_model_sha256"],
                     "frozen_nuisance_sha256": nuisance_receipt["model_sha256"],
                     "b_complete_sha256": data.sha256_file(b_complete_path)})


def run_task_fit(anchor: int, index_path: str | Path,
                 output_dir: str | Path, *, seed: int) -> dict:
    """Queue-friendly 2018 task-only fit with hash-verified sanitized input."""
    from experiments.pcrl_task_aligned_cuts_v1 import data

    sanitized = (Path(__file__).resolve().parents[2] /
                 data.SANITIZED_RELATIVE / "SANITIZATION.json")
    if not sanitized.is_file():
        raise RuntimeError("sanitized label-stripped 2018 receipt required")
    index = data.index(index_path)
    prepared = data.load_prepared(index, anchor)
    if "labels" in prepared["ctx"]["pools"]["attacker_validation"]:
        raise RuntimeError("outer labels appeared before assessment lock")
    rows = roles.pooled_role(prepared, "nuisance_train")
    fitted, receipt = fit_task_only(rows, seed=seed)
    receipt.update({"anchor": anchor,
                    "module_sha256": _sha_file(Path(__file__)),
                    "outer_labels_accessed": False})
    return save_task_only(output_dir, fitted, receipt)


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Private 2018 task-only and partition baselines")
    commands = parser.add_subparsers(dest="command", required=True)
    task = commands.add_parser("fit-task")
    task.add_argument("--anchor", type=int, required=True)
    task.add_argument("--index-path", required=True)
    task.add_argument("--output-dir", required=True)
    task.add_argument("--seed", type=int, default=20260924)
    partitions = commands.add_parser("fit-partitions")
    partitions.add_argument("--anchor", type=int, required=True)
    partitions.add_argument("--delta", type=float, required=True)
    partitions.add_argument("--index-path", required=True)
    partitions.add_argument("--a-center-dir", required=True)
    partitions.add_argument("--b-center-dir", required=True)
    partitions.add_argument("--output-dir", required=True)
    partitions.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args(argv)
    if args.command == "fit-task":
        receipt = run_task_fit(args.anchor, args.index_path, args.output_dir,
                               seed=args.seed)
        summary = {"schema": receipt["schema"], "anchor": receipt["anchor"],
                   "model_sha256": receipt["model_sha256"],
                   "fitted_people": receipt["fitted_people"]}
    else:
        receipt = run_partition_controls(
            args.anchor, args.delta, args.index_path, args.a_center_dir,
            args.b_center_dir, args.output_dir, seed=args.seed)
        summary = {"schema": receipt["schema"],
                   "all_leaf_counts_matched": receipt["all_leaf_counts_matched"],
                   "realized_states": {key: value["realized_states"]
                                       for key, value in receipt["controls"].items()}}
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
