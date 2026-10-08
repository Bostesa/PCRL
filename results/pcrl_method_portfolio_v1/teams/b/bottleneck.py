"""Team B prototype: purpose-specific Gaussian feature bottlenecks.

This module contains no data loader. Its release builder is for one immutable
batch: once the batch is finalized, downstream systems serve the saved views,
not a newly sampled release. No repeated-query protection is claimed.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class GaussianEncoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, latent_dim: int,
        log_variance_bounds: tuple[float, float],
    ):
        super().__init__()
        self.register_buffer("logvar_min", torch.tensor(log_variance_bounds[0], dtype=torch.float32))
        self.register_buffer("logvar_max", torch.tensor(log_variance_bounds[1], dtype=torch.float32))
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_variance = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.mean(h), self.log_variance(h).clamp(self.logvar_min, self.logvar_max)


class BottleneckPair(nn.Module):
    """Two local encoders and heads, with no teacher or attribute runtime input."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        forbidden_names: Sequence[str],
        class_counts: Mapping[str, int],
        log_variance_bounds: tuple[float, float],
        hidden_dim: int = 64,
        latent_dim: int = 16,
    ):
        super().__init__()
        names = tuple(feature_names)
        if len(names) != 83 or len(set(names)) != len(names):
            raise ValueError("Exactly 83 distinct permitted feature names are required")
        forbidden = {name.casefold() for name in forbidden_names}
        if not forbidden:
            raise ValueError("A forbidden-name list is required")
        if any(name.casefold() in forbidden for name in names):
            raise ValueError("A forbidden label appears in the feature schema")
        if set(class_counts) != {"income", "occupation"}:
            raise ValueError("Both purpose class counts are required")
        if any(count < 2 for count in class_counts.values()):
            raise ValueError("Each purpose needs at least two classes")
        if (len(log_variance_bounds) != 2
                or any(not math.isfinite(value) for value in log_variance_bounds)
                or log_variance_bounds[0] >= log_variance_bounds[1]):
            raise ValueError("Finite increasing log-variance bounds are required")
        self.feature_names = names
        self.latent_dim = latent_dim
        self.log_variance_bounds = tuple(log_variance_bounds)
        self.class_counts = dict(class_counts)
        self.encoders = nn.ModuleDict({
            purpose: GaussianEncoder(len(names), hidden_dim, latent_dim, self.log_variance_bounds)
            for purpose in ("income", "occupation")
        })
        self.heads = nn.ModuleDict({
            purpose: nn.Linear(latent_dim, class_counts[purpose])
            for purpose in ("income", "occupation")
        })

    def training_outputs(
        self, x: torch.Tensor, *, stochastic: bool,
        generator: torch.Generator | Mapping[str, torch.Generator],
        purposes: tuple[str, ...] = ("income", "occupation"),
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Produce the exact sampled features used by heads and critics."""
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError("Training input must have the pinned 83 columns")
        if not torch.isfinite(x).all():
            raise FloatingPointError("Training input contains nonfinite values")
        if not purposes or len(set(purposes)) != len(purposes) or any(
            purpose not in self.encoders for purpose in purposes
        ):
            raise ValueError("A nonempty unique purpose subset is required")
        outputs = {}
        for purpose in purposes:
            mean, log_variance = self.encoders[purpose](x)
            if stochastic:
                own_generator = generator[purpose] if isinstance(generator, Mapping) else generator
                noise = torch.randn(mean.shape, generator=own_generator, device=mean.device)
                z = mean + torch.exp(0.5 * log_variance) * noise
            else:
                z = mean
            logits = self.heads[purpose](z)
            if not all(torch.isfinite(tensor).all() for tensor in (mean, log_variance, z, logits)):
                raise FloatingPointError("Training model produced nonfinite values")
            outputs[purpose] = {
                "mean": mean,
                "log_variance": log_variance,
                "z": z,
                "logits": logits,
                "probabilities": torch.softmax(logits, dim=1),
            }
        return outputs


def diagonal_gaussian_kl(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """Mean over people of KL(q(z|x)||N(0,I)); sum over coordinates."""
    if mean.shape != log_variance.shape or mean.ndim != 2 or not mean.shape[0]:
        raise ValueError("Mean and log variance must be equal nonempty matrices")
    return 0.5 * (mean.square() + log_variance.exp() - log_variance - 1).sum(1).mean()


def complete_view_tensor(
    outputs: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Differentiable full numeric views for local and coalition critics.

    The public decision is one-hot encoded. Constant metadata contributes no
    per-person signal. During encoder training, probabilities use T=1 because
    the held-out temperature is fitted only after the encoder is frozen; fresh
    audit attacks receive the actual calibrated deployment probabilities.
    """
    if set(outputs) != {"income", "occupation"}:
        raise ValueError("Both purpose outputs are required for the coalition")
    local = {purpose: local_view_tensor(outputs[purpose])
             for purpose in ("income", "occupation")}
    return local, torch.cat((local["income"], local["occupation"]), dim=1)


def local_view_tensor(output: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """One recipient's exact numeric critic view without evaluating its peer."""
    probabilities = output["probabilities"]
    decisions = F.one_hot(probabilities.argmax(dim=1),
                          num_classes=probabilities.shape[1]).to(probabilities.dtype)
    return torch.cat((output["z"], output["logits"], probabilities, decisions), dim=1)


class SexCritics(nn.Module):
    """Training-only local and coalition critics for the complete feature view."""

    def __init__(self, *, latent_dim: int, class_counts: Mapping[str, int], hidden_dim: int = 64):
        super().__init__()
        if set(class_counts) != {"income", "occupation"}:
            raise ValueError("Critics require both purposes")
        widths = {purpose: latent_dim + 3 * class_counts[purpose] for purpose in class_counts}
        self.local = nn.ModuleDict({
            purpose: nn.Sequential(nn.Linear(widths[purpose], hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
            for purpose in ("income", "occupation")
        })
        self.coalition = nn.Sequential(
            nn.Linear(sum(widths.values()), hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)
        )

    def forward(self, outputs: Mapping[str, Mapping[str, torch.Tensor]]) -> tuple[dict, torch.Tensor]:
        local_view, coalition_view = complete_view_tensor(outputs)
        return ({purpose: self.local[purpose](local_view[purpose]) for purpose in local_view},
                self.coalition(coalition_view))


def training_terms(
    outputs: Mapping[str, Mapping[str, torch.Tensor]],
    critics: SexCritics,
    labels: Mapping[str, torch.Tensor],
    teacher_probabilities: Mapping[str, torch.Tensor],
    sex: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable terms; caller controls alternating optimizer steps.

    Labels and teacher scores are fitting inputs only. They never enter the
    deployment adapter or recipient view.
    """
    if set(labels) != {"income", "occupation"} or set(teacher_probabilities) != set(labels):
        raise ValueError("Both task labels and teachers are required")
    local_logits, coalition_logits = critics(outputs)
    task_losses, distill_losses, kl_losses = [], [], []
    n = sex.shape[0]
    if sex.shape != (n,) or not bool(torch.all((sex == 0) | (sex == 1))):
        raise ValueError("SEX training labels must be a binary vector")
    for purpose in ("income", "occupation"):
        out = outputs[purpose]
        task_y = labels[purpose]
        teacher = teacher_probabilities[purpose]
        if task_y.shape != (n,) or out["logits"].shape != teacher.shape:
            raise ValueError(f"Incorrect task or teacher shape for {purpose}")
        if not bool(torch.all((task_y >= 0) & (task_y < out["logits"].shape[1]))):
            raise ValueError(f"Invalid task class for {purpose}")
        if not torch.isfinite(teacher).all() or bool(torch.any(teacher < 0)) or not torch.allclose(
            teacher.sum(1), torch.ones(n, device=teacher.device), atol=1e-5
        ):
            raise ValueError(f"Invalid teacher probabilities for {purpose}")
        task_losses.append(F.cross_entropy(out["logits"], task_y))
        distill_losses.append(F.kl_div(F.log_softmax(out["logits"], dim=1), teacher, reduction="batchmean"))
        kl_losses.append(diagonal_gaussian_kl(out["mean"], out["log_variance"]))
    local_sex = [F.cross_entropy(local_logits[purpose], sex) for purpose in ("income", "occupation")]
    return {
        "task_ce": torch.stack(task_losses).mean(),
        "distillation": torch.stack(distill_losses).mean(),
        "bottleneck_kl": torch.stack(kl_losses).mean(),
        "local_sex_ce": torch.stack(local_sex).mean(),
        "coalition_sex_ce": F.cross_entropy(coalition_logits, sex),
    }


@dataclass(frozen=True)
class LossWeights:
    """Explicit prelock inputs: the prototype has no fitted weight defaults."""

    task_ce: float
    distillation: float
    bottleneck_kl: float
    local_sex: float
    coalition_sex: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value < 0 for value in vars(self).values()):
            raise ValueError("All objective weights must be finite and nonnegative")

    def encoder_objective(self, terms: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return (self.task_ce * terms["task_ce"] + self.distillation * terms["distillation"]
                + self.bottleneck_kl * terms["bottleneck_kl"]
                - self.local_sex * terms["local_sex_ce"]
                - self.coalition_sex * terms["coalition_sex_ce"])

    def critic_objective(self, terms: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return terms["local_sex_ce"] + terms["coalition_sex_ce"]


def schema_fingerprint(feature_names: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(tuple(feature_names), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def model_fingerprint(model: BottleneckPair) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def purpose_model_fingerprint(model: BottleneckPair, purpose: str) -> str:
    """Stable sampler identity for one unchanged encoder and its task head."""
    if purpose not in ("income", "occupation"):
        raise ValueError("Unknown purpose")
    digest = hashlib.sha256()
    digest.update(b"TeamBPurposeModelV1")
    digest.update(purpose.encode("ascii"))
    digest.update(schema_fingerprint(model.feature_names).encode("ascii"))
    for module_name in ("encoders", "heads"):
        module = getattr(model, module_name)[purpose]
        for name, value in sorted(module.state_dict().items()):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(f"{module_name}.{purpose}.{name}".encode("utf-8"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


class ReleaseAdapter:
    """Build a single aligned release batch, caching one draw per row and purpose.

    ``person_key`` remains provider-side. It is never included in a recipient
    view. A saved batch must be served unchanged; this builder is not a query
    endpoint, and making a new builder constitutes a new release experiment.
    """

    def __init__(
        self,
        model: BottleneckPair,
        *,
        stochastic: bool,
        sampler_secret: bytes,
        temperatures: Mapping[str, float],
        source_id: str,
        expected_model_sha256: str,
        expected_schema_sha256: str,
        calibration_records: Mapping[str, Mapping] | None = None,
        calibration_record_sha256: Mapping[str, str] | None = None,
        raw_mode: bool = False,
    ):
        if set(temperatures) != {"income", "occupation"} or any(
            not math.isfinite(value) or value <= 0 for value in temperatures.values()
        ):
            raise ValueError("A positive frozen global temperature is required per purpose")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("A pinned source identity is required")
        if not isinstance(sampler_secret, bytes) or len(sampler_secret) < 32:
            raise ValueError("A private sampler key of at least 32 bytes is required")
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        if self.model.class_counts != {"income": 2, "occupation": 6}:
            raise ValueError("Deployment requires pinned Adult task class counts")
        self.stochastic = bool(stochastic)
        self._sampler_secret = sampler_secret
        self.temperatures = dict(temperatures)
        self.raw_mode = bool(raw_mode)
        if self.raw_mode:
            if calibration_records is not None or calibration_record_sha256 is not None or any(
                value != 1.0 for value in self.temperatures.values()
            ):
                raise ValueError("Explicit raw mode requires T=1 and no fitted calibrator")
            self.calibration_records = None
            self.calibration_record_sha256 = None
        else:
            if (calibration_records is None or calibration_record_sha256 is None
                    or set(calibration_records) != {"income", "occupation"}
                    or set(calibration_record_sha256) != {"income", "occupation"}):
                raise ValueError("Scored deployment requires both fitted S2 calibrator records and hashes")
            from common_eval import CAL_SCHEMA, TEMP_BOUNDS
            self.calibration_records = copy.deepcopy(dict(calibration_records))
            self.calibration_record_sha256 = dict(calibration_record_sha256)
            for purpose in ("income", "occupation"):
                record = self.calibration_records[purpose]
                if record.get("schema") != CAL_SCHEMA or record.get("task") != purpose:
                    raise ValueError("Wrong fitted S2 calibrator identity")
                alpha = float(record["alpha"])
                if (not math.isfinite(alpha) or not TEMP_BOUNDS[0] <= alpha <= TEMP_BOUNDS[1]
                        or not math.isclose(self.temperatures[purpose], 1.0 / alpha,
                                            rel_tol=0, abs_tol=1e-12)):
                    raise ValueError("Temperature differs from fitted S2 calibrator")
                canonical = json.dumps(record, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False) + "\n"
                actual_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                if self.calibration_record_sha256[purpose] != actual_sha:
                    raise ValueError("Fitted S2 calibrator SHA mismatch")
        self.source_id = source_id
        self.model_sha256 = model_fingerprint(self.model)
        self.purpose_model_sha256 = {
            purpose: purpose_model_fingerprint(self.model, purpose)
            for purpose in ("income", "occupation")
        }
        self.schema_sha256 = schema_fingerprint(self.model.feature_names)
        if self.model_sha256 != expected_model_sha256:
            raise ValueError("Pinned model identity mismatch")
        if self.schema_sha256 != expected_schema_sha256:
            raise ValueError("Pinned schema identity mismatch")
        self.cache: dict[tuple[str, str], tuple[str, dict]] = {}
        self.closed = False

    @staticmethod
    def _copy_view(view: dict) -> dict:
        return copy.deepcopy(view)

    def release(
        self, purpose: str, person_key: str, x: np.ndarray, *, columns: Sequence[str]
    ) -> dict:
        if self.closed:
            raise RuntimeError("This one-shot release batch has been finalized")
        if purpose not in self.model.encoders:
            raise ValueError("Unknown purpose")
        if not isinstance(person_key, str) or not person_key:
            raise ValueError("A nonempty private provider row key is required")
        if tuple(columns) != self.model.feature_names:
            raise ValueError("Incorrect permitted column order or names")
        array = np.asarray(x, dtype=np.float32)
        if array.shape != (len(self.model.feature_names),):
            raise ValueError(f"Expected exactly {len(self.model.feature_names)} feature values")
        if not np.isfinite(array).all():
            raise ValueError("Feature values must be finite")
        array = np.ascontiguousarray(array)
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        key = (purpose, person_key)
        if key in self.cache:
            old_digest, old_view = self.cache[key]
            if digest != old_digest:
                raise ValueError("Input changed for a cached person and purpose")
            return self._copy_view(old_view)
        with torch.no_grad():
            tensor = torch.from_numpy(array).unsqueeze(0)
            mean, log_variance = self.model.encoders[purpose](tensor)
            if self.stochastic:
                domain = json.dumps(
                    ["TeamBPurposeDrawV1", self.source_id,
                     self.purpose_model_sha256[purpose], purpose, person_key],
                    separators=(",", ":"),
                ).encode("utf-8")
                seed_bytes = hmac.new(self._sampler_secret, domain, hashlib.sha256).digest()[:8]
                generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(seed_bytes, "big"))
                noise = torch.randn(mean.shape, generator=generator)
                z = mean + torch.exp(0.5 * log_variance) * noise
            else:
                z = mean
            logits = self.model.heads[purpose](z)
            raw_probabilities = torch.softmax(logits.to(torch.float64), dim=1)
            if not all(torch.isfinite(tensor).all() for tensor in (mean, log_variance, z, logits, raw_probabilities)):
                raise FloatingPointError("Release model produced nonfinite values")
            raw_array = raw_probabilities.squeeze(0).numpy().astype(np.float64)
            if self.raw_mode:
                probabilities = raw_array
            else:
                from common_eval import apply_global_temperature
                probabilities = apply_global_temperature(
                    raw_array.reshape(1, -1), self.calibration_records[purpose]
                )[0]
            if not np.isfinite(probabilities).all():
                raise FloatingPointError("Release calibration produced nonfinite values")
        view = {
            "features": z.squeeze(0).numpy().copy(),
            "logits": logits.squeeze(0).numpy().copy(),
            "probabilities": probabilities.copy(),
            "decision": int(np.argmax(probabilities)),
            "metadata": {
                "purpose": purpose,
                "stochastic": self.stochastic,
                "draws_per_person": 1 if self.stochastic else 0,
                "latent_dim": self.model.latent_dim,
                "log_variance_bounds": self.model.log_variance_bounds,
                "calibration_family": "raw_unfitted" if self.raw_mode else "global_temperature",
                "temperature": self.temperatures[purpose],
                "calibration_record_sha256": (None if self.raw_mode else
                                               self.calibration_record_sha256[purpose]),
                "source_id": self.source_id,
                "model_sha256": self.model_sha256,
                "purpose_model_sha256": self.purpose_model_sha256[purpose],
                "sampler_domain": "TeamBPurposeDrawV1",
                "schema_sha256": self.schema_sha256,
                "release_contract": "immutable_one_shot_batch",
            },
        }
        self.cache[key] = digest, view
        return self._copy_view(view)

    def finalize(self, path: str | Path) -> None:
        """Atomically seal the private batch; never overwrite an earlier draw.

        This file is provider-side and contains private record keys and
        person-level views. It must not be placed in public results or Git.
        Recipients are served copies of ``SavedRelease.get`` results.
        """
        if self.closed:
            raise RuntimeError("This one-shot release batch has been finalized")
        if self.raw_mode:
            raise ValueError("Unfitted raw calibration views cannot be finalized for deployment")
        target = Path(path)
        if target.exists():
            raise FileExistsError(target)
        people = {person_key for _, person_key in self.cache}
        if not people or any(
            ("income", person_key) not in self.cache or ("occupation", person_key) not in self.cache
            for person_key in people
        ):
            raise ValueError("An aligned pair of recipient views is required for every person")
        if any(self.cache["income", person_key][0] != self.cache["occupation", person_key][0]
               for person_key in people):
            raise ValueError("Both recipient views must use the same input for each person")
        target.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for (purpose, person_key), (input_sha256, view) in sorted(self.cache.items()):
            records.append({
                "purpose": purpose,
                "private_person_key": person_key,
                "input_sha256": input_sha256,
                "view": {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in view.items()
                },
            })
        payload = {
            "format": "TeamBPrivateReleaseV1",
            "source_id": self.source_id,
            "model_sha256": self.model_sha256,
            "purpose_model_sha256": self.purpose_model_sha256,
            "schema_sha256": self.schema_sha256,
            "temperatures": self.temperatures,
            "raw_mode": self.raw_mode,
            "calibration_record_sha256": self.calibration_record_sha256,
            "stochastic": self.stochastic,
            "records": records,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        wrapped = json.dumps({
            "payload_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "payload": payload,
        }, sort_keys=True, separators=(",", ":"))
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=target.parent,
                prefix=".team-b-release-", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                os.chmod(temporary, 0o600)
                stream.write(wrapped)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.closed = True


class SavedRelease:
    """Read-only provider-side replay of a finalized release batch."""

    def __init__(self, path: str | Path):
        wrapped = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = wrapped["payload"]
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != wrapped["payload_sha256"]:
            raise ValueError("Saved release digest mismatch")
        if payload["format"] != "TeamBPrivateReleaseV1":
            raise ValueError("Unsupported saved release format")
        self.source_id = payload["source_id"]
        self.model_sha256 = payload["model_sha256"]
        self.records = {}
        for record in payload["records"]:
            key = (record["purpose"], record["private_person_key"])
            if key in self.records:
                raise ValueError("Duplicate saved person and purpose")
            raw = record["view"]
            if set(raw) != {"features", "logits", "probabilities", "decision", "metadata"}:
                raise ValueError("Saved release contains an unexpected transmitted view field")
            view = {
                "features": np.asarray(raw["features"], dtype=np.float32),
                "logits": np.asarray(raw["logits"], dtype=np.float32),
                "probabilities": np.asarray(raw["probabilities"], dtype=np.float64),
                "decision": int(raw["decision"]),
                "metadata": dict(raw["metadata"]),
            }
            self.records[key] = view

    def get(self, purpose: str, private_person_key: str) -> dict:
        """Return only transmitted fields; the lookup key remains provider-side."""
        return ReleaseAdapter._copy_view(self.records[(purpose, private_person_key)])
