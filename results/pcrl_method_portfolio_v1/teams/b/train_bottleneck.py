"""Explicit, resumable CPU stage trainer for Team B's bottleneck family.

The caller supplies already admitted role-only tensors. This module has no
source-data loader, no hidden scientific defaults, and no numerical work at
import time. S3 must admit each actual fit through the shared queue.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from bottleneck import (
    BottleneckPair, LossWeights, ReleaseAdapter, SexCritics,
    complete_view_tensor, diagonal_gaussian_kl, local_view_tensor, model_fingerprint,
    schema_fingerprint,
)


PURPOSES = ("income", "occupation")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _ordered_id_digest(values: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("utf-8")).hexdigest()


def _tensor_digest(hasher, name: str, value: torch.Tensor) -> None:
    array = value.detach().cpu().contiguous().numpy()
    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(array.tobytes())


class FitData:
    """In-memory data from one S2-admitted fitting role only."""

    def __init__(
        self, *, x: torch.Tensor, labels: Mapping[str, torch.Tensor],
        teacher_probabilities: Mapping[str, torch.Tensor],
        sex: torch.Tensor | None,
        feature_names: tuple[str, ...], permitted_feature_names: tuple[str, ...],
        expected_schema_sha256: str,
        row_ids: tuple[str, ...], group_ids: tuple[str, ...],
        teacher_identity: Mapping[str, str],
    ):
        self.x = torch.as_tensor(x, dtype=torch.float32).detach().cpu().clone().contiguous()
        if self.x.ndim != 2 or self.x.shape[1] != 83 or not len(self.x):
            raise ValueError("FitData requires a nonempty 83-column matrix")
        if not torch.isfinite(self.x).all():
            raise ValueError("FitData features must be finite")
        self.feature_names = tuple(feature_names)
        permitted_names = tuple(permitted_feature_names)
        if (len(self.feature_names) != 83 or len(set(self.feature_names)) != 83
                or any(not isinstance(name, str) or not name for name in self.feature_names)
                or len(permitted_names) != 83 or len(set(permitted_names)) != 83
                or self.feature_names != permitted_names):
            raise ValueError("FitData columns must equal the exact S2 permitted ordered schema")
        self.feature_schema_sha256 = schema_fingerprint(self.feature_names)
        if (not isinstance(expected_schema_sha256, str)
                or not _SHA256.fullmatch(expected_schema_sha256)
                or self.feature_schema_sha256 != expected_schema_sha256):
            raise ValueError("FitData schema differs from pinned S2 manifest hash")
        if set(labels) != set(PURPOSES) or set(teacher_probabilities) != set(PURPOSES):
            raise ValueError("Both task labels and frozen teachers are required")
        n = len(self.x)
        self.row_ids = tuple(row_ids)
        self.group_ids = tuple(group_ids)
        if (len(self.row_ids) != n or len(set(self.row_ids)) != n
                or len(self.group_ids) != n or any(not isinstance(value, str) or not value
                                                    for value in self.row_ids + self.group_ids)):
            raise ValueError("Ordered private row and group IDs are required")
        self.row_order_sha256 = _ordered_id_digest(self.row_ids)
        self.group_order_sha256 = _ordered_id_digest(self.group_ids)
        self.teacher_identity = dict(teacher_identity)
        if self.teacher_identity != {"income": "Ucal_1", "occupation": "Ucal_2"}:
            raise ValueError("Frozen Ucal_1/Ucal_2 teacher identity is required")
        self.labels = {}
        self.teacher_probabilities = {}
        for purpose, classes in (("income", 2), ("occupation", 6)):
            raw_y = torch.as_tensor(labels[purpose])
            if raw_y.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError(f"{purpose} task labels must have an integer dtype")
            y = raw_y.to(dtype=torch.int64).detach().cpu().clone().contiguous()
            teacher = torch.as_tensor(
                teacher_probabilities[purpose], dtype=torch.float32
            ).detach().cpu().clone().contiguous()
            if y.shape != (n,) or bool(torch.any((y < -1) | (y >= classes))):
                raise ValueError(f"Invalid {purpose} task labels")
            if teacher.shape != (n, classes) or not torch.isfinite(teacher).all() or bool(
                torch.any(teacher < 0)
            ) or not torch.allclose(teacher.sum(1), torch.ones(n), atol=1e-5):
                raise ValueError(f"Invalid {purpose} teacher probabilities")
            self.labels[purpose] = y
            self.teacher_probabilities[purpose] = teacher
        if sex is None:
            self.sex = None
        else:
            raw_sex = torch.as_tensor(sex)
            if raw_sex.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError("SEX fitting labels must have an integer dtype")
            self.sex = raw_sex.to(dtype=torch.int64).detach().cpu().clone().contiguous()
            if self.sex.shape != (n,) or not bool(torch.all((self.sex == 0) | (self.sex == 1))):
                raise ValueError("SEX fitting labels must be binary and complete")
        digest = hashlib.sha256()
        digest.update(self.feature_schema_sha256.encode("ascii"))
        digest.update(self.row_order_sha256.encode("ascii"))
        digest.update(self.group_order_sha256.encode("ascii"))
        digest.update(json.dumps(self.teacher_identity, sort_keys=True).encode("utf-8"))
        _tensor_digest(digest, "x", self.x)
        for purpose in PURPOSES:
            _tensor_digest(digest, f"{purpose}_labels", self.labels[purpose])
            _tensor_digest(digest, f"{purpose}_teacher", self.teacher_probabilities[purpose])
        if self.sex is not None:
            _tensor_digest(digest, "sex", self.sex)
        else:
            digest.update(b"sex:none")
        self.sha256 = digest.hexdigest()


class FrozenPurposeView:
    """One aligned, private stage-1 draw reused in every sequential stage-2 step."""

    def __init__(
        self, *, purpose: str, z: torch.Tensor, logits: torch.Tensor,
        probabilities: torch.Tensor, source_id: str, model_sha256: str,
        feature_schema_sha256: str, row_order_sha256: str,
        group_order_sha256: str, first_checkpoint_sha256: str,
        sampler_plan_sha256: str,
    ):
        if purpose not in PURPOSES or not source_id:
            raise ValueError("Invalid frozen purpose or source identity")
        digests = (model_sha256, feature_schema_sha256, row_order_sha256,
                   group_order_sha256, first_checkpoint_sha256, sampler_plan_sha256)
        if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in digests):
            raise ValueError("All frozen first-view identities must be SHA-256 digests")
        self.purpose, self.source_id = purpose, source_id
        self.model_sha256 = model_sha256
        self.feature_schema_sha256 = feature_schema_sha256
        self.row_order_sha256 = row_order_sha256
        self.group_order_sha256 = group_order_sha256
        self.first_checkpoint_sha256 = first_checkpoint_sha256
        self.sampler_plan_sha256 = sampler_plan_sha256
        self.z = torch.as_tensor(z, dtype=torch.float32).detach().cpu().clone().contiguous()
        self.logits = torch.as_tensor(logits, dtype=torch.float32).detach().cpu().clone().contiguous()
        self.probabilities = torch.as_tensor(probabilities, dtype=torch.float32).detach().cpu().clone().contiguous()
        classes = 2 if purpose == "income" else 6
        if (self.z.ndim != 2 or self.logits.shape != (len(self.z), classes)
                or self.probabilities.shape != self.logits.shape
                or not len(self.z) or not all(torch.isfinite(array).all()
                                               for array in (self.z, self.logits, self.probabilities))
                or not torch.allclose(self.probabilities, torch.softmax(self.logits, 1), atol=1e-6)):
            raise ValueError("Frozen first view must be a finite T=1 sampled-code release")
        digest = hashlib.sha256()
        identity = {"purpose": purpose, "source_id": source_id, "model_sha256": model_sha256,
                    "feature_schema_sha256": feature_schema_sha256,
                    "row_order_sha256": row_order_sha256,
                    "group_order_sha256": group_order_sha256,
                    "first_checkpoint_sha256": first_checkpoint_sha256,
                    "sampler_plan_sha256": sampler_plan_sha256}
        digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        for name, array in (("z", self.z), ("logits", self.logits), ("probabilities", self.probabilities)):
            _tensor_digest(digest, name, array)
        self.sha256 = digest.hexdigest()

    @classmethod
    def from_model(
        cls, model: BottleneckPair, data: FitData, *, purpose: str,
        source_id: str, first_checkpoint_sha256: str,
        sampler_plan_sha256: str, sampler_secret: bytes,
    ) -> FrozenPurposeView:
        """Make one HMAC-keyed draw per admitted private row in source order."""
        if (model.feature_names != data.feature_names
                or schema_fingerprint(model.feature_names) != data.feature_schema_sha256):
            raise ValueError("Frozen first-view model and FitData schemas differ")
        adapter = ReleaseAdapter(
            model, stochastic=True, sampler_secret=sampler_secret,
            temperatures={"income": 1.0, "occupation": 1.0},
            source_id=source_id,
            expected_model_sha256=model_fingerprint(model),
            expected_schema_sha256=schema_fingerprint(model.feature_names),
        )
        views = [adapter.release(
            purpose, person_key, data.x[index].numpy(), columns=model.feature_names
        ) for index, person_key in enumerate(data.row_ids)]
        return cls(
            purpose=purpose,
            z=torch.stack([torch.from_numpy(view["features"]) for view in views]),
            logits=torch.stack([torch.from_numpy(view["logits"]) for view in views]),
            probabilities=torch.stack([torch.from_numpy(view["probabilities"]) for view in views]),
            source_id=source_id, model_sha256=model_fingerprint(model),
            feature_schema_sha256=schema_fingerprint(model.feature_names),
            row_order_sha256=data.row_order_sha256,
            group_order_sha256=data.group_order_sha256,
            first_checkpoint_sha256=first_checkpoint_sha256,
            sampler_plan_sha256=sampler_plan_sha256,
        )

    def indexed(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the same stored first view on every stage-2 encounter."""
        return {"z": self.z[indices], "logits": self.logits[indices],
                "probabilities": self.probabilities[indices]}

    def save(self, path: str | Path) -> str:
        """Atomically store this person-level view in a private versioned file."""
        state = {
            "format": "TeamBFrozenFirstViewV1",
            "purpose": self.purpose, "source_id": self.source_id,
            "model_sha256": self.model_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "row_order_sha256": self.row_order_sha256,
            "group_order_sha256": self.group_order_sha256,
            "first_checkpoint_sha256": self.first_checkpoint_sha256,
            "sampler_plan_sha256": self.sampler_plan_sha256,
            "z": self.z, "logits": self.logits,
            "probabilities": self.probabilities,
            "view_sha256": self.sha256,
        }
        _atomic_torch_save(state, Path(path), prefix=".team-b-frozen-")
        return self.sha256

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str) -> FrozenPurposeView:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        if state.get("format") != "TeamBFrozenFirstViewV1":
            raise ValueError("Unsupported frozen first-view format")
        view = cls(
            purpose=state["purpose"], source_id=state["source_id"],
            model_sha256=state["model_sha256"],
            feature_schema_sha256=state["feature_schema_sha256"],
            row_order_sha256=state["row_order_sha256"],
            group_order_sha256=state["group_order_sha256"],
            first_checkpoint_sha256=state["first_checkpoint_sha256"],
            sampler_plan_sha256=state["sampler_plan_sha256"],
            z=state["z"], logits=state["logits"],
            probabilities=state["probabilities"],
        )
        if view.sha256 != expected_sha256 or view.sha256 != state["view_sha256"]:
            raise ValueError("Frozen first-view digest mismatch")
        return view


@dataclass(frozen=True)
class StageConfig:
    """Every numerical choice must be supplied by the sealed arm runner."""

    stage_id: str
    fit_role_id: str
    train_purposes: tuple[str, ...]
    frozen_purposes: tuple[str, ...]
    frozen_view_sha256: str | None
    first_checkpoint_sha256: str | None
    sampler_plan_sha256: str | None
    privacy_views: tuple[str, ...]
    stochastic: bool
    steps: int
    batch_size: int
    local_critic_updates: int
    pair_critic_updates: int
    weights: LossWeights
    learning_rate: float
    critic_learning_rate: float
    adam_betas: tuple[float, float]
    adam_eps: float
    weight_decay: float
    seed: int

    def __post_init__(self) -> None:
        if not self.stage_id or not self.fit_role_id:
            raise ValueError("A stage ID and sealed fitting role ID are required")
        if not self.train_purposes or len(set(self.train_purposes)) != len(self.train_purposes) or any(
            purpose not in PURPOSES for purpose in self.train_purposes
        ):
            raise ValueError("train_purposes must be a nonempty purpose subset")
        if len(set(self.frozen_purposes)) != len(self.frozen_purposes) or any(
            purpose not in PURPOSES or purpose in self.train_purposes for purpose in self.frozen_purposes
        ):
            raise ValueError("Frozen purposes must be distinct from trained purposes")
        if self.frozen_purposes:
            if len(self.frozen_purposes) != 1 or not all(
                isinstance(value, str) and _SHA256.fullmatch(value)
                for value in (self.frozen_view_sha256, self.first_checkpoint_sha256,
                              self.sampler_plan_sha256)
            ):
                raise ValueError("A frozen first view requires pinned view/checkpoint/sampler hashes")
        elif any(value is not None for value in (
            self.frozen_view_sha256, self.first_checkpoint_sha256, self.sampler_plan_sha256
        )):
            raise ValueError("Frozen first-view hashes are invalid without a frozen purpose")
        if "pair" in self.privacy_views and len(self.train_purposes) == 1 and not self.frozen_purposes:
            raise ValueError("Sequential pair critic needs a frozen first-stage view")
        if len(set(self.privacy_views)) != len(self.privacy_views) or any(
            view not in (*PURPOSES, "pair") for view in self.privacy_views
        ):
            raise ValueError("Unknown or duplicate privacy view")
        if any(view in PURPOSES and view not in self.train_purposes for view in self.privacy_views):
            raise ValueError("A local privacy view must belong to a trained purpose")
        if self.steps <= 0 or self.batch_size <= 0 or self.local_critic_updates < 0 or self.pair_critic_updates < 0:
            raise ValueError("Positive steps/batch and nonnegative critic update counts required")
        if (not self.privacy_views and (self.local_critic_updates or self.pair_critic_updates
                                        or self.weights.local_sex or self.weights.coalition_sex)):
            raise ValueError("Task-only stage cannot update or weight SEX critics")
        if any(view in PURPOSES for view in self.privacy_views) != bool(self.local_critic_updates):
            raise ValueError("Local critic update count must match active local views")
        if ("pair" in self.privacy_views) != bool(self.pair_critic_updates):
            raise ValueError("Pair critic update count must match the active pair view")
        if ("pair" in self.privacy_views) != bool(self.weights.coalition_sex):
            raise ValueError("Pair privacy weight must match the active pair view")
        if any(view in PURPOSES for view in self.privacy_views) != bool(self.weights.local_sex):
            raise ValueError("Local privacy weight must match active local views")
        if any(not math.isfinite(value) or value <= 0 for value in (
            self.learning_rate, self.critic_learning_rate, self.adam_eps
        )) or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Invalid Adam rates, epsilon or weight decay")
        if len(self.adam_betas) != 2 or not (0 <= self.adam_betas[0] < 1 and 0 <= self.adam_betas[1] < 1):
            raise ValueError("Invalid Adam betas")
        if not isinstance(self.seed, int):
            raise ValueError("An integer training seed is required")

    @property
    def sha256(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _code_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("bottleneck.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _named_seed(seed: int, stream: str, purpose: str) -> int:
    raw = json.dumps([seed, stream, purpose], separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _atomic_torch_save(state: dict, target: Path, *, prefix: str) -> None:
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent,
                                         prefix=prefix, delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class StageRunner:
    """One fixed stage; caller composes stages into registered control arms."""

    def __init__(
        self, model: BottleneckPair, critics: SexCritics, data: FitData,
        config: StageConfig, *, source_id: str, role_id: str,
        frozen_views: Mapping[str, FrozenPurposeView] | None = None,
        _resume: bool = False,
    ):
        if not source_id or not role_id:
            raise ValueError("Pinned source and role identities are required")
        if role_id != config.fit_role_id:
            raise ValueError("Fitting role differs from sealed stage role")
        if any(bool(torch.any(data.labels[purpose] < 0)) for purpose in config.train_purposes):
            raise ValueError("Active task labels must be complete on the fitting role")
        if model.class_counts != {"income": 2, "occupation": 6} or model.feature_names is None:
            raise ValueError("Stage model must use Adult task schema")
        if (model.feature_names != data.feature_names
                or schema_fingerprint(model.feature_names) != data.feature_schema_sha256):
            raise ValueError("Stage model and admitted FitData feature schemas differ")
        if model.latent_dim != critics.local["income"][0].in_features - 6:
            raise ValueError("Critic and encoder latent dimensions differ")
        if config.privacy_views and data.sex is None:
            raise ValueError("Protected labels are required for active fitting critics")
        self.frozen_views = dict(frozen_views or {})
        if set(self.frozen_views) != set(config.frozen_purposes):
            raise ValueError("Frozen purpose view set does not match stage config")
        self.stage_initial_model_sha256 = model_fingerprint(model)
        for purpose, view in self.frozen_views.items():
            if (view.purpose != purpose or view.sha256 != config.frozen_view_sha256
                    or view.source_id != source_id
                    or view.feature_schema_sha256 != schema_fingerprint(model.feature_names)
                    or view.row_order_sha256 != data.row_order_sha256
                    or view.group_order_sha256 != data.group_order_sha256
                    or view.first_checkpoint_sha256 != config.first_checkpoint_sha256
                    or view.sampler_plan_sha256 != config.sampler_plan_sha256
                    or len(view.z) != len(data.x)
                    or view.z.shape[1] != model.latent_dim
                    or (not _resume and view.model_sha256 != self.stage_initial_model_sha256)):
                raise ValueError("Frozen first-view identity or alignment mismatch")
        self.model, self.critics, self.data = model, critics, data
        self.config = config
        self.source_id, self.role_id = source_id, role_id
        self.code_sha256 = _code_sha256()
        for purpose in PURPOSES:
            active = purpose in config.train_purposes
            self.model.encoders[purpose].requires_grad_(active)
            self.model.heads[purpose].requires_grad_(active)
        self.model.train()
        self.critics.train()
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.model_optimizer = torch.optim.Adam(
            parameters, lr=config.learning_rate, betas=config.adam_betas,
            eps=config.adam_eps, weight_decay=config.weight_decay,
        )
        self.local_optimizers = {
            purpose: torch.optim.Adam(
                self.critics.local[purpose].parameters(), lr=config.critic_learning_rate,
                betas=config.adam_betas, eps=config.adam_eps, weight_decay=config.weight_decay,
            )
            for purpose in PURPOSES if purpose in config.privacy_views
        }
        self.pair_optimizer = (
            torch.optim.Adam(
                self.critics.coalition.parameters(), lr=config.critic_learning_rate,
                betas=config.adam_betas, eps=config.adam_eps, weight_decay=config.weight_decay,
            ) if "pair" in config.privacy_views else None
        )
        self.data_generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.encoder_generators = {
            purpose: torch.Generator(device="cpu").manual_seed(_named_seed(config.seed, "encoder", purpose))
            for purpose in PURPOSES
        }
        self.critic_generators = {
            view: {
                purpose: torch.Generator(device="cpu").manual_seed(
                    _named_seed(config.seed, f"critic:{view}", purpose)
                ) for purpose in PURPOSES
            } for view in config.privacy_views
        }
        self.order = torch.randperm(len(data.x), generator=self.data_generator)
        self.cursor = 0
        self.completed_steps = 0
        self.encoder_steps = {purpose: 0 for purpose in PURPOSES}
        self.local_critic_steps = {purpose: 0 for purpose in PURPOSES}
        self.pair_critic_steps = 0
        self.history: list[dict[str, float]] = []

    def _batch(self) -> tuple[torch.Tensor, torch.Tensor, dict, dict, torch.Tensor | None]:
        if self.cursor >= len(self.order):
            self.order = torch.randperm(len(self.data.x), generator=self.data_generator)
            self.cursor = 0
        end = min(self.cursor + self.config.batch_size, len(self.order))
        index = self.order[self.cursor:end]
        self.cursor = end
        return (index, self.data.x[index],
                {purpose: self.data.labels[purpose][index] for purpose in PURPOSES},
                {purpose: self.data.teacher_probabilities[purpose][index] for purpose in PURPOSES},
                self.data.sex[index] if self.data.sex is not None else None)

    def _outputs(self, x: torch.Tensor, indices: torch.Tensor, *, stream: str) -> dict:
        generators = (self.encoder_generators if stream == "encoder"
                      else self.critic_generators[stream])
        active = self.config.train_purposes
        outputs = self.model.training_outputs(
            x, stochastic=self.config.stochastic, generator=generators,
            purposes=active,
        )
        for purpose, view in self.frozen_views.items():
            outputs[purpose] = view.indexed(indices)
        return outputs

    def _critic_update(self, purpose: str, indices: torch.Tensor,
                       x: torch.Tensor, sex: torch.Tensor) -> None:
        with torch.no_grad():
            outputs = self._outputs(x, indices, stream=purpose)
        if purpose == "pair":
            _, pair_view = complete_view_tensor(outputs)
            optimizer = self.pair_optimizer
            logits = self.critics.coalition(pair_view)
        else:
            optimizer = self.local_optimizers[purpose]
            logits = self.critics.local[purpose](local_view_tensor(outputs[purpose]))
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(logits, sex).backward()
        optimizer.step()
        if purpose == "pair":
            self.pair_critic_steps += 1
        else:
            self.local_critic_steps[purpose] += 1

    def _encoder_update(self, indices: torch.Tensor, x: torch.Tensor, labels: dict, teachers: dict,
                        sex: torch.Tensor | None) -> dict[str, float]:
        for parameter in self.critics.parameters():
            parameter.requires_grad_(False)
        try:
            outputs = self._outputs(x, indices, stream="encoder")
            task_losses, teacher_losses, kl_losses = [], [], []
            for purpose in self.config.train_purposes:
                out = outputs[purpose]
                known = labels[purpose] >= 0
                task_losses.append(
                    F.cross_entropy(out["logits"][known], labels[purpose][known])
                    if bool(known.any()) else out["logits"].sum() * 0
                )
                teacher_losses.append(F.kl_div(
                    F.log_softmax(out["logits"], dim=1), teachers[purpose], reduction="batchmean"
                ))
                kl_losses.append(diagonal_gaussian_kl(out["mean"], out["log_variance"]))
            task_ce = torch.stack(task_losses).mean()
            distillation = torch.stack(teacher_losses).mean()
            bottleneck_kl = torch.stack(kl_losses).mean()
            zero = task_ce * 0
            local_sex_ce, coalition_sex_ce = zero, zero
            if self.config.privacy_views:
                selected_local = [purpose for purpose in PURPOSES if purpose in self.config.privacy_views]
                if selected_local:
                    local_sex_ce = torch.stack([
                        F.cross_entropy(self.critics.local[purpose](local_view_tensor(outputs[purpose])), sex)
                        for purpose in selected_local
                    ]).mean()
                if "pair" in self.config.privacy_views:
                    _, pair_view = complete_view_tensor(outputs)
                    coalition_sex_ce = F.cross_entropy(self.critics.coalition(pair_view), sex)
            terms = {
                "task_ce": task_ce, "distillation": distillation,
                "bottleneck_kl": bottleneck_kl, "local_sex_ce": local_sex_ce,
                "coalition_sex_ce": coalition_sex_ce,
            }
            objective = self.config.weights.encoder_objective(terms)
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError("Nonfinite Team B training objective")
            self.model_optimizer.zero_grad(set_to_none=True)
            objective.backward()
            self.model_optimizer.step()
            for purpose in self.config.train_purposes:
                self.encoder_steps[purpose] += 1
            return {name: float(value.detach()) for name, value in {**terms, "objective": objective}.items()}
        finally:
            for parameter in self.critics.parameters():
                parameter.requires_grad_(True)

    def run(self, *, steps: int | None = None) -> dict:
        remaining = self.config.steps - self.completed_steps
        if steps is not None and steps < 0:
            raise ValueError("Requested steps must be nonnegative")
        count = remaining if steps is None else min(steps, remaining)
        for _ in range(count):
            indices, x, labels, teachers, sex = self._batch()
            for purpose in PURPOSES:
                if purpose in self.local_optimizers:
                    for _ in range(self.config.local_critic_updates):
                        self._critic_update(purpose, indices, x, sex)
            if self.pair_optimizer is not None:
                for _ in range(self.config.pair_critic_updates):
                    self._critic_update("pair", indices, x, sex)
            metrics = self._encoder_update(indices, x, labels, teachers, sex)
            self.completed_steps += 1
            self.history.append(metrics)
        return self.receipt()

    def receipt(self) -> dict:
        return {
            "stage_id": self.config.stage_id,
            "completed_steps": self.completed_steps,
            "planned_steps": self.config.steps,
            "encoder_steps": dict(self.encoder_steps),
            "local_critic_steps": dict(self.local_critic_steps),
            "pair_critic_steps": self.pair_critic_steps,
            "source_id": self.source_id,
            "role_id": self.role_id,
            "data_sha256": self.data.sha256,
            "feature_schema_sha256": self.data.feature_schema_sha256,
            "row_order_sha256": self.data.row_order_sha256,
            "group_order_sha256": self.data.group_order_sha256,
            "teacher_identity": dict(self.data.teacher_identity),
            "config_sha256": self.config.sha256,
            "code_sha256": self.code_sha256,
            "stage_initial_model_sha256": self.stage_initial_model_sha256,
            "frozen_view_sha256": self.config.frozen_view_sha256,
            "first_checkpoint_sha256": self.config.first_checkpoint_sha256,
            "sampler_plan_sha256": self.config.sampler_plan_sha256,
            "model_sha256": model_fingerprint(self.model),
            "status": "COMPLETED" if self.completed_steps == self.config.steps else "RUNNING",
        }

    def save_checkpoint(self, path: str | Path) -> dict:
        """Write a versioned private checkpoint without replacing a previous one."""
        target = Path(path)
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format": "TeamBStageCheckpointV1",
            "source_id": self.source_id, "role_id": self.role_id,
            "data_sha256": self.data.sha256,
            "feature_schema_sha256": self.data.feature_schema_sha256,
            "row_order_sha256": self.data.row_order_sha256,
            "group_order_sha256": self.data.group_order_sha256,
            "teacher_identity": dict(self.data.teacher_identity),
            "config_sha256": self.config.sha256,
            "code_sha256": self.code_sha256,
            "stage_initial_model_sha256": self.stage_initial_model_sha256,
            "frozen_view_sha256": self.config.frozen_view_sha256,
            "first_checkpoint_sha256": self.config.first_checkpoint_sha256,
            "sampler_plan_sha256": self.config.sampler_plan_sha256,
            "model": self.model.state_dict(),
            "critics": self.critics.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "local_optimizers": {name: optimizer.state_dict()
                                 for name, optimizer in self.local_optimizers.items()},
            "pair_optimizer": self.pair_optimizer.state_dict() if self.pair_optimizer else None,
            "data_generator": self.data_generator.get_state(),
            "encoder_generators": {
                purpose: generator.get_state() for purpose, generator in self.encoder_generators.items()
            },
            "critic_generators": {
                view: {purpose: generator.get_state() for purpose, generator in per_purpose.items()}
                for view, per_purpose in self.critic_generators.items()
            },
            "order": self.order,
            "cursor": self.cursor,
            "completed_steps": self.completed_steps,
            "encoder_steps": self.encoder_steps,
            "local_critic_steps": self.local_critic_steps,
            "pair_critic_steps": self.pair_critic_steps,
            "history": self.history,
        }
        _atomic_torch_save(state, target, prefix=".team-b-stage-")
        return self.receipt()

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, model: BottleneckPair, critics: SexCritics,
        data: FitData, config: StageConfig, *, source_id: str, role_id: str,
        frozen_views: Mapping[str, FrozenPurposeView] | None = None,
    ) -> StageRunner:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        expected = {
            "format": "TeamBStageCheckpointV1",
            "source_id": source_id, "role_id": role_id,
            "data_sha256": data.sha256,
            "feature_schema_sha256": data.feature_schema_sha256,
            "row_order_sha256": data.row_order_sha256,
            "group_order_sha256": data.group_order_sha256,
            "teacher_identity": dict(data.teacher_identity),
            "config_sha256": config.sha256,
            "code_sha256": _code_sha256(),
            "frozen_view_sha256": config.frozen_view_sha256,
            "first_checkpoint_sha256": config.first_checkpoint_sha256,
            "sampler_plan_sha256": config.sampler_plan_sha256,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(f"Checkpoint {name.replace('_sha256', '')} identity mismatch")
        runner = cls(model, critics, data, config, source_id=source_id, role_id=role_id,
                     frozen_views=frozen_views, _resume=True)
        if frozen_views and any(view.model_sha256 != state["stage_initial_model_sha256"]
                                for view in frozen_views.values()):
            raise ValueError("Checkpoint first-stage model identity mismatch")
        runner.stage_initial_model_sha256 = state["stage_initial_model_sha256"]
        runner.model.load_state_dict(state["model"])
        runner.critics.load_state_dict(state["critics"])
        runner.model_optimizer.load_state_dict(state["model_optimizer"])
        for name, optimizer in runner.local_optimizers.items():
            optimizer.load_state_dict(state["local_optimizers"][name])
        if runner.pair_optimizer is not None:
            runner.pair_optimizer.load_state_dict(state["pair_optimizer"])
        runner.data_generator.set_state(state["data_generator"])
        for purpose, generator in runner.encoder_generators.items():
            generator.set_state(state["encoder_generators"][purpose])
        for view, per_purpose in runner.critic_generators.items():
            for purpose, generator in per_purpose.items():
                generator.set_state(state["critic_generators"][view][purpose])
        runner.order = state["order"]
        runner.cursor = state["cursor"]
        runner.completed_steps = state["completed_steps"]
        runner.encoder_steps = dict(state["encoder_steps"])
        runner.local_critic_steps = dict(state["local_critic_steps"])
        runner.pair_critic_steps = state["pair_critic_steps"]
        runner.history = list(state["history"])
        if not 0 <= runner.completed_steps <= config.steps:
            raise ValueError("Checkpoint step count exceeds sealed stage")
        return runner
