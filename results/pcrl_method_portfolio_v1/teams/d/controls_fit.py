"""Team D's matched feature controls on explicitly supplied allowed rows.

No source loader or private path is imported. A caller must use S2's admitted
role store and S3's global numerical queue. Continuous controls expose their
actual transformed feature and head output; hard-code controls expose IDs too.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import isfinite
from typing import Callable
import hashlib
import json
import re

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from prototype import PurposeNormalizer
from trainer import (DTrainingData, _Head, make_sign_tables, mmd_terms,
                     pooled_initializer, surrogate_select, _distances)


@dataclass(frozen=True)
class ControlFitConfig:
    codes: int = 1024
    hidden: int = 64
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 0.001
    reconstruction_weight: float = 0.25
    pair_features: int = 128

    def __post_init__(self) -> None:
        if (self.codes < 2 or self.codes & (self.codes - 1) or self.hidden < 1
                or self.epochs < 1 or self.batch_size < 2
                or not isfinite(self.learning_rate) or not isfinite(self.reconstruction_weight)
                or self.learning_rate <= 0 or self.reconstruction_weight < 0
                or self.pair_features < 1):
            raise ValueError("invalid D control configuration")


def _check_real_config(data: DTrainingData, config: ControlFitConfig) -> None:
    if data.role == "OSF_DEFENSE_FIT" and (data.h0.shape[1] != 16 or config != ControlFitConfig()):
        raise ValueError("Adult D controls require the frozen 16-D, 1024-code configuration")


def _release_dict(purpose: int, ids: np.ndarray, features: np.ndarray,
                  probabilities: np.ndarray, bits: int, kind: str) -> dict:
    return {
        "code_ids": ids.copy(),
        "decoded": features.copy(),
        "probabilities": probabilities.copy(),
        "decision": np.argmax(probabilities, axis=1).copy(),
        "metadata": {"purpose": purpose, "code_length_bits": bits,
                     "block_count": 1 if bits else 0, "fallback": False,
                     "stochastic": False, "kind": kind,
                     "calibration": "UNFITTED"},
    }


class TwoCodeRelease:
    """Two locally deployable 10-bit codebooks; public heads use only decoded codes."""

    def __init__(self, normalizer: PurposeNormalizer, codebooks: tuple[torch.Tensor, torch.Tensor],
                 heads: tuple[_Head, _Head], kind: str):
        if len(codebooks) != 2 or len(heads) != 2 or kind not in ("LOCAL", "SEQ12", "SEQ21"):
            raise ValueError("two matched codebooks/heads and a known kind are required")
        self.normalizer = normalizer
        self.codebooks = tuple(c.detach().clone().cpu().float() for c in codebooks)
        self.heads = tuple(copy.deepcopy(h).cpu().eval() for h in heads)
        self.kind = kind
        for h in self.heads:
            for p in h.parameters():
                p.requires_grad_(False)

    def recipient(self, purpose: int, latent: np.ndarray) -> dict:
        if purpose not in (0, 1):
            raise ValueError("purpose must be 0 or 1")
        x = torch.from_numpy(self.normalizer.transform(purpose, latent).astype(np.float32))
        with torch.no_grad():
            ids = torch.argmin(_distances(x, self.codebooks[purpose]), dim=1)
            z = self.codebooks[purpose][ids]
            p = torch.softmax(self.heads[purpose](z), dim=1)
        return _release_dict(purpose, ids[:, None].numpy(), z.numpy(), p.numpy(),
                             self.codebooks[purpose].shape[0].bit_length() - 1, self.kind)


class ContinuousRelease:
    """Frozen standardized latents, optionally with official fixed LEACE maps."""

    def __init__(self, normalizer: PurposeNormalizer, heads: tuple[_Head, _Head],
                 maps: tuple[object | None, object | None], kind: str):
        if len(heads) != 2 or len(maps) != 2 or kind not in ("UNCHANGED", "LEACE"):
            raise ValueError("two heads/maps and a known continuous control are required")
        self.normalizer = normalizer
        self.heads = tuple(copy.deepcopy(h).cpu().eval() for h in heads)
        self.maps = maps
        self.kind = kind
        for h in self.heads:
            for p in h.parameters():
                p.requires_grad_(False)

    def recipient(self, purpose: int, latent: np.ndarray) -> dict:
        if purpose not in (0, 1):
            raise ValueError("purpose must be 0 or 1")
        x = self.normalizer.transform(purpose, latent)
        z = x if self.maps[purpose] is None else self.maps[purpose].transform(x)
        z32 = torch.from_numpy(np.asarray(z, dtype=np.float32))
        with torch.no_grad():
            p = torch.softmax(self.heads[purpose](z32), dim=1).numpy()
        return _release_dict(purpose, np.empty((len(z), 0), dtype=np.int64),
                             np.asarray(z, dtype=np.float32), p, 0, self.kind)


@dataclass
class ControlFitResult:
    release: TwoCodeRelease | ContinuousRelease
    mode: str
    seed: int
    fit_rows: int
    history: list[dict]
    map_metadata: dict


def _batch_indices(n: int, seed: int, epoch: int, batch_size: int):
    order = np.random.default_rng(seed + 1_000_003 * epoch).permutation(n)
    for start in range(0, n, batch_size):
        subset = order[start:start + batch_size]
        if len(subset) < 2:
            raise ValueError("singleton fitting batch")
        yield torch.as_tensor(subset, dtype=torch.long)


def _task_loss(logits: torch.Tensor, labels: torch.Tensor, teacher: torch.Tensor,
               x: torch.Tensor, z: torch.Tensor, recon_weight: float) -> torch.Tensor:
    return (F.cross_entropy(logits, labels)
            + 0.5 * F.kl_div(F.log_softmax(logits, dim=1), teacher, reduction="batchmean")
            + recon_weight * ((x - z) ** 2).sum(dim=1).mean() / x.shape[1])


def _purpose_initializer(x: torch.Tensor, seed: int, purpose: int, codes: int) -> torch.Tensor:
    if len(x) < codes:
        raise ValueError("a codebook cannot have more centroids than fitting rows")
    km = KMeans(n_clusters=codes, random_state=int(2 * seed + purpose), n_init=1,
                max_iter=100, tol=1e-4, algorithm="lloyd")
    km.fit(x.numpy())
    return torch.from_numpy(np.asarray(km.cluster_centers_, dtype=np.float32).copy())


def purpose_initializers(
    data: DTrainingData, seed: int, config: ControlFitConfig,
    normalizer: PurposeNormalizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit exactly two reusable purpose codebook initializers before any arm.

    LOCAL, SEQ12 and SEQ21 must receive immutable copies of these tensors to
    count their seven total pooled/purpose initialization aliases honestly.
    """
    if normalizer.dim != data.h0.shape[1]:
        raise ValueError("control normalizer and frozen latents disagree")
    x = tuple(torch.from_numpy(normalizer.transform(i, h).astype(np.float32))
              for i, h in enumerate((data.h0, data.h1)))
    return (_purpose_initializer(x[0], seed, 0, config.codes),
            _purpose_initializer(x[1], seed, 1, config.codes))


@dataclass(frozen=True)
class DControlCache:
    normalizer: PurposeNormalizer
    pooled_centroids: np.ndarray
    purpose_centroids: tuple[torch.Tensor, torch.Tensor]
    receipt: dict

    def validate(self, data: DTrainingData, seed: int, config: ControlFitConfig,
                 source_model_sha256: tuple[str, str], row_order_sha256: str) -> None:
        expected = _cache_input_receipt(data, seed, config, source_model_sha256,
                                        row_order_sha256)
        for key, value in expected.items():
            if self.receipt.get(key) != value:
                raise ValueError(f"D control cache input identity mismatch: {key}")
        if self.receipt.get("pooled_initializer_sha256") != _sha_array(self.pooled_centroids):
            raise ValueError("D pooled initializer mutated after caching")
        if self.receipt.get("purpose_initializer_sha256") != [
                _sha_array(t.detach().cpu().numpy()) for t in self.purpose_centroids]:
            raise ValueError("D purpose initializer mutated after caching")
        if self.receipt.get("normalizer_sha256") != [
                _sha_array(self.normalizer.mean), _sha_array(self.normalizer.scale)]:
            raise ValueError("D purpose normalizer mutated after caching")


def _sha_array(a: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(value.shape).encode())
    h.update(str(value.dtype).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def _cache_input_receipt(data: DTrainingData, seed: int, config: ControlFitConfig,
                         source_model_sha256: tuple[str, str], row_order_sha256: str) -> dict:
    hashes = (*source_model_sha256, row_order_sha256)
    if len(source_model_sha256) != 2 or any(
            not isinstance(x, str) or re.fullmatch(r"[0-9a-f]{64}", x) is None for x in hashes):
        raise ValueError("two source hashes and admitted row/order hash must be lowercase SHA-256")
    return {
        "source_model_sha256": list(source_model_sha256),
        "seed": int(seed),
        "role": data.role,
        "admitted_fit_row_order_sha256": row_order_sha256,
        "internal_row_order_sha256": _sha_array(data.row_id0),
        "latent_sha256": [_sha_array(data.h0), _sha_array(data.h1)],
        "config": {"codes": config.codes, "hidden": config.hidden,
                   "epochs": config.epochs, "batch_size": config.batch_size,
                   "learning_rate": config.learning_rate,
                   "allowed_reconstruction_weights": [0.0, 0.25, 1.0],
                   "pair_features": config.pair_features},
    }


def prepare_control_cache(data: DTrainingData, seed: int,
                          config: ControlFitConfig = ControlFitConfig(), *,
                          source_model_sha256: tuple[str, str],
                          row_order_sha256: str) -> DControlCache:
    """Exactly one normalization and three KMeans fits for the D arm suite.

    Pass (normalizer, pooled_centroids) to every protected/task-only `train_d`
    call, and `normalizer` plus `purpose_centroids` to all three local/sequence
    calls. This is the executable alias boundary; a different source/split/seed
    must construct a different cache identity in S3's baseline registry.
    """
    _check_real_config(data, config)
    if data.sex is not None:
        raise ValueError("shared codebook initialization must use sex=None task-only data")
    receipt = _cache_input_receipt(data, seed, config, source_model_sha256,
                                   row_order_sha256)
    normalizer, pooled = pooled_initializer(data, seed, config.codes)
    purpose = purpose_initializers(data, seed, config, normalizer)
    pooled_copy = np.asarray(pooled, dtype=np.float32).copy()
    purpose_copy = tuple(t.detach().clone().cpu() for t in purpose)
    receipt["pooled_initializer_sha256"] = _sha_array(pooled_copy)
    receipt["purpose_initializer_sha256"] = [_sha_array(t.numpy()) for t in purpose_copy]
    receipt["normalizer_sha256"] = [_sha_array(normalizer.mean), _sha_array(normalizer.scale)]
    receipt["cache_id"] = hashlib.sha256(json.dumps(receipt, sort_keys=True,
                                                   separators=(",", ":")).encode()).hexdigest()
    return DControlCache(normalizer, pooled_copy, purpose_copy, receipt)


def fit_local_or_sequential(
    data: DTrainingData, seed: int, mode: str,
    config: ControlFitConfig = ControlFitConfig(),
    normalizer: PurposeNormalizer | None = None,
    initializers: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> ControlFitResult:
    """Fit LOCAL, SEQ12 or SEQ21 with identical per-purpose output capacity.

    Stage one has task + local hard-ID discrepancy. In a sequential stage two,
    the other codebook is frozen and the loss is task + mean(local, pair) hard-ID
    discrepancies on the aligned people. Each purpose receives 20 epochs.
    """
    if mode not in ("LOCAL", "SEQ12", "SEQ21"):
        raise ValueError("mode must be LOCAL, SEQ12 or SEQ21")
    if data.sex is None:
        raise ValueError("private local/sequential control fitting requires fitting-role SEX")
    _check_real_config(data, config)
    torch.set_num_threads(1)
    normalizer = normalizer if normalizer is not None else PurposeNormalizer.fit((data.h0, data.h1))
    if normalizer.dim != data.h0.shape[1]:
        raise ValueError("control normalizer and frozen latents disagree")
    x = tuple(torch.from_numpy(normalizer.transform(i, h).astype(np.float32))
              for i, h in enumerate((data.h0, data.h1)))
    y = (torch.from_numpy(data.y0), torch.from_numpy(data.y1))
    teacher = (torch.from_numpy(data.teacher0), torch.from_numpy(data.teacher1))
    torch.manual_seed(int(seed))
    heads = (_Head(x[0].shape[1], config.hidden, teacher[0].shape[1]),
             _Head(x[1].shape[1], config.hidden, teacher[1].shape[1]))
    initials = initializers if initializers is not None else purpose_initializers(data, seed, config, normalizer)
    if len(initials) != 2 or any(a.shape != (config.codes, x[0].shape[1])
                                 or not torch.isfinite(a).all() for a in initials):
        raise ValueError("two finite, fixed purpose initializer tensors are required")
    codebooks = [torch.nn.Parameter(initials[i].detach().clone().cpu().float()) for i in (0, 1)]
    signs = make_sign_tables(2 * seed + 1, config.codes, config.pair_features)
    order = (0, 1) if mode != "SEQ21" else (1, 0)
    history: list[dict] = []
    for stage, purpose in enumerate(order):
        optimizer = torch.optim.Adam([codebooks[purpose], *heads[purpose].parameters()],
                                     lr=config.learning_rate)
        for epoch in range(config.epochs):
            losses = np.zeros(3, dtype=np.float64)
            nbatches = 0
            for ix in _batch_indices(len(x[purpose]), seed + 100 * stage, epoch, config.batch_size):
                z, _, assignment = surrogate_select(x[purpose][ix], codebooks[purpose])
                task = _task_loss(heads[purpose](z), y[purpose][ix], teacher[purpose][ix],
                                  x[purpose][ix], z, config.reconstruction_weight)
                if stage == 0 or mode == "LOCAL":
                    group0, group1 = data.sex[ix.numpy()] == 0, data.sex[ix.numpy()] == 1
                    if not group0.any() or not group1.any():
                        raise ValueError("a control minibatch lacks one SEX group")
                    privacy = ((assignment[group0].mean(0) - assignment[group1].mean(0)) ** 2).sum()
                else:
                    first = order[0]
                    with torch.no_grad():
                        first_ids = torch.argmin(_distances(x[first][ix], codebooks[first]), dim=1)
                        first_onehot = F.one_hot(first_ids, config.codes).float()
                    if purpose == 0:
                        code0, code1 = assignment, first_onehot
                    else:
                        code0, code1 = first_onehot, assignment
                    local0, local1, pair = mmd_terms(code0, code1, torch.from_numpy(data.sex[ix.numpy()]), signs)
                    privacy = ((local0 if purpose == 0 else local1) + pair) / 2
                loss = task + privacy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses += (float(task.detach()), float(privacy.detach()), float(loss.detach()))
                nbatches += 1
            history.append({"stage": stage + 1, "purpose": purpose, "epoch": epoch + 1,
                            "batches": nbatches, "task": losses[0] / nbatches,
                            "code_id_discrepancy": losses[1] / nbatches,
                            "objective": losses[2] / nbatches})
    release = TwoCodeRelease(normalizer, (codebooks[0], codebooks[1]), heads, mode)
    return ControlFitResult(release, mode, int(seed), len(data.h0), history,
                            {"codebooks": 2, "codes_per_recipient": config.codes,
                             "local_epochs_per_recipient": config.epochs,
                             "pair_conditioned_order": order if mode != "LOCAL" else None,
                             "inference_requires_other_recipient": False})


def fit_continuous(
    data: DTrainingData, seed: int, kind: str,
    config: ControlFitConfig = ControlFitConfig(),
    normalizer: PurposeNormalizer | None = None,
) -> ControlFitResult:
    """Fit unchanged or official-LEACE frozen-latent control and matching heads."""
    if kind not in ("UNCHANGED", "LEACE"):
        raise ValueError("kind must be UNCHANGED or LEACE")
    if kind == "UNCHANGED" and data.sex is not None:
        raise ValueError("task-only unchanged control must receive sex=None")
    if kind == "LEACE" and data.sex is None:
        raise ValueError("LEACE fitting requires fitting-role SEX")
    _check_real_config(data, config)
    torch.set_num_threads(1)
    normalizer = normalizer if normalizer is not None else PurposeNormalizer.fit((data.h0, data.h1))
    if normalizer.dim != data.h0.shape[1]:
        raise ValueError("control normalizer and frozen latents disagree")
    x = tuple(normalizer.transform(i, h) for i, h in enumerate((data.h0, data.h1)))
    maps: tuple[object | None, object | None] = (None, None)
    if kind == "LEACE":
        # Official source wrapper checks the installed package/version/source hash.
        # Importing this module does not access any source data.
        from stored_model_eval.defenses import fit_leace
        if not np.issubdtype(np.asarray(data.row_id0).dtype, np.integer):
            raise ValueError("LEACE requires integer-compatible aligned row IDs for provenance")
        sex_onehot = np.eye(2, dtype=np.float64)[data.sex]
        maps = (fit_leace(x[0], sex_onehot, fit_row_ids=data.row_id0),
                fit_leace(x[1], sex_onehot, fit_row_ids=data.row_id1))
    z = tuple(x[i] if maps[i] is None else maps[i].transform(x[i]) for i in (0, 1))
    zt = tuple(torch.from_numpy(np.asarray(a, dtype=np.float32)) for a in z)
    y = (torch.from_numpy(data.y0), torch.from_numpy(data.y1))
    teacher = (torch.from_numpy(data.teacher0), torch.from_numpy(data.teacher1))
    torch.manual_seed(int(seed))
    heads = (_Head(zt[0].shape[1], config.hidden, teacher[0].shape[1]),
             _Head(zt[1].shape[1], config.hidden, teacher[1].shape[1]))
    history: list[dict] = []
    for purpose in (0, 1):
        optimizer = torch.optim.Adam(heads[purpose].parameters(), lr=config.learning_rate)
        for epoch in range(config.epochs):
            total = 0.0
            nbatches = 0
            for ix in _batch_indices(len(zt[purpose]), seed + 200 * purpose, epoch, config.batch_size):
                logits = heads[purpose](zt[purpose][ix])
                task = (F.cross_entropy(logits, y[purpose][ix])
                        + 0.5 * F.kl_div(F.log_softmax(logits, dim=1), teacher[purpose][ix],
                                         reduction="batchmean"))
                optimizer.zero_grad(set_to_none=True)
                task.backward()
                optimizer.step()
                total += float(task.detach())
                nbatches += 1
            history.append({"purpose": purpose, "epoch": epoch + 1, "batches": nbatches,
                            "task": total / nbatches})
    release = ContinuousRelease(normalizer, heads, maps, kind)
    meta = {"official_leace": kind == "LEACE", "map_fits": 2 if kind == "LEACE" else 0,
            "head_fits": 2, "head_epochs": config.epochs,
            "inference_uses_sex": False,
            "fit_row_count": len(data.row_id0)}
    if kind == "LEACE":
        meta["leace_map_provenance"] = [m.metadata.get("provenance") for m in maps]
        meta["leace_fit_row_id_sha256"] = [m.metadata.get("fit_row_ids_sha256") for m in maps]
    return ControlFitResult(release, kind, int(seed), len(data.h0), history, meta)


class RawJointReferenceRelease:
    """Admitted historical RAW-J encoder plus newly fitted, code-free purpose heads.

    The source encoder and its checkpoint SHA must be verified by S2/S4 before
    construction. The deployment interface takes the permitted 83-column X,
    never SEX or a true task label. No inherited scientific loader is called.
    """

    def __init__(self, encode: Callable[[int, np.ndarray], np.ndarray],
                 source_model_sha256: str, expected_sha256: str,
                 feature_release: ContinuousRelease):
        if source_model_sha256 != expected_sha256 or len(expected_sha256) != 64:
            raise ValueError("RAW-J source checkpoint hash was not admitted")
        self.encode = encode
        self.source_model_sha256 = source_model_sha256
        self.feature_release = feature_release

    def recipient(self, purpose: int, permitted_x: np.ndarray) -> dict:
        latent = np.asarray(self.encode(purpose, permitted_x), dtype=np.float32)
        out = self.feature_release.recipient(purpose, latent)
        out["metadata"]["kind"] = "RAW_J_ADMITTED"
        return out


def fit_raw_joint_reference(
    admitted_feature_data: DTrainingData,
    source_encode: Callable[[int, np.ndarray], np.ndarray],
    source_model_sha256: str,
    expected_source_sha256: str,
    seed: int,
    config: ControlFitConfig = ControlFitConfig(),
) -> tuple[RawJointReferenceRelease, ControlFitResult]:
    """Refit matched heads on role-only features of a hash-admitted RAW-J encoder.

    The caller must obtain features by forwarding only S2's allowed role files
    through the admitted historical model. The old all-role release is barred.
    This function does not load or retrain the historical encoder.
    """
    if source_model_sha256 != expected_source_sha256 or len(expected_source_sha256) != 64:
        raise ValueError("RAW-J checkpoint hash is not admitted")
    fitted = fit_continuous(admitted_feature_data, seed, "UNCHANGED", config=config)
    return (RawJointReferenceRelease(source_encode, source_model_sha256,
                                     expected_source_sha256, fitted.release), fitted)
