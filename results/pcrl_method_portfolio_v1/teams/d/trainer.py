"""Synthetic-ready trainer for Team D's fixed post-hoc hard VQ objective.

No loader, private path, assessment role, or inference-time SEX input exists in
this module. Adult training remains blocked until S2 admits a lawful paired
role-specific latent export. All numerical runs use S3's global semaphore.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from prototype import PurposeNormalizer, occupancy_report


@dataclass(frozen=True)
class DTrainConfig:
    codes: int = 1024
    hidden: int = 64
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 0.001
    pair_features: int = 128

    def __post_init__(self) -> None:
        if (self.codes < 2 or self.codes & (self.codes - 1) or self.hidden < 1
                or self.epochs < 1 or self.batch_size < 2 or self.learning_rate <= 0
                or self.pair_features < 1):
            raise ValueError("invalid fixed VQ training configuration")


@dataclass(frozen=True)
class DTrainingData:
    h0: np.ndarray
    h1: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    teacher0: np.ndarray
    teacher1: np.ndarray
    sex: np.ndarray
    row_id0: np.ndarray
    row_id1: np.ndarray
    role: str

    def __post_init__(self) -> None:
        if self.role not in ("SYNTHETIC", "OSF_DEFENSE_FIT"):
            raise ValueError("only synthetic or admitted defense-fit rows may be fitted")
        h0 = np.asarray(self.h0, dtype=np.float32)
        h1 = np.asarray(self.h1, dtype=np.float32)
        if h0.ndim != 2 or h0.shape != h1.shape or h0.shape[0] < 2 or not np.isfinite(h0).all() or not np.isfinite(h1).all():
            raise ValueError("paired finite frozen latents with equal shape required")
        n = h0.shape[0]
        row0, row1 = np.asarray(self.row_id0), np.asarray(self.row_id1)
        if row0.shape != (n,) or row1.shape != (n,) or not np.array_equal(row0, row1):
            raise ValueError("purpose row IDs must match in the same order")
        ys = []
        teachers = []
        for y_raw, p_raw in ((self.y0, self.teacher0), (self.y1, self.teacher1)):
            y = np.asarray(y_raw)
            p = np.asarray(p_raw, dtype=np.float32)
            if (y.shape != (n,) or not np.issubdtype(y.dtype, np.integer)
                    or p.ndim != 2 or p.shape[0] != n or p.shape[1] < 2
                    or np.any(y < 0) or np.any(y >= p.shape[1])
                    or not np.isfinite(p).all() or np.any(p < 0)
                    or not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
                raise ValueError("task labels and teacher probabilities are invalid")
            ys.append(y.astype(np.int64, copy=True))
            teachers.append(p.copy())
        sex = np.asarray(self.sex)
        if sex.shape != (n,) or not np.isin(sex, (0, 1)).all() or len(np.unique(sex)) != 2:
            raise ValueError("both binary SEX groups are required on fitting rows")
        for name, value in (("h0", h0.copy()), ("h1", h1.copy()),
                            ("y0", ys[0]), ("y1", ys[1]),
                            ("teacher0", teachers[0]), ("teacher1", teachers[1]),
                            ("sex", sex.astype(np.int64, copy=True)),
                            ("row_id0", row0.copy()), ("row_id1", row1.copy())):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SignTables:
    recipient0: torch.Tensor
    recipient1: torch.Tensor


def make_sign_tables(seed: int, codes: int, features: int) -> SignTables:
    """Independent public pair-code signs, each entry +/-1/sqrt(features)."""
    if codes < 2 or features < 1:
        raise ValueError("positive code and sign-feature counts required")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def draw() -> torch.Tensor:
        return (2 * torch.randint(0, 2, (codes, features), generator=generator) - 1).float() / sqrt(features)

    return SignTables(draw(), draw())


def _distances(h: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    if h.ndim != 2 or centroids.ndim != 2 or h.shape[1] != centroids.shape[1]:
        raise ValueError("latent and codebook dimensions disagree")
    return ((h[:, None, :] - centroids[None, :, :]) ** 2).sum(dim=2)


def surrogate_select(h: torch.Tensor, centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard IDs forward; tau=1 soft distance assignment only in backward."""
    distance = _distances(h, centroids)
    index = torch.argmin(distance, dim=1)
    hard = F.one_hot(index, centroids.shape[0]).to(centroids.dtype)
    # PyTorch softmax subtracts the row maximum internally; tau is fixed at 1.
    soft = torch.softmax(-distance, dim=1)
    assignment = hard + soft - soft.detach()
    decoded = assignment @ centroids
    return decoded, index, assignment


def mmd_terms(
    y0: torch.Tensor, y1: torch.Tensor, sex: torch.Tensor,
    tables: SignTables,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact local hard-ID frequency loss and sketched aligned pair-ID loss."""
    if y0.shape != y1.shape or sex.shape != (len(y0),):
        raise ValueError("paired code views and SEX labels must align")
    if tables.recipient0.shape != tables.recipient1.shape or tables.recipient0.shape[0] != y0.shape[1]:
        raise ValueError("sign-table and code dimensions disagree")
    group0, group1 = sex == 0, sex == 1
    if not bool(group0.any()) or not bool(group1.any()):
        raise ValueError("a paired minibatch lacks one SEX group")

    def mmd(phi: torch.Tensor) -> torch.Tensor:
        return ((phi[group0].mean(dim=0) - phi[group1].mean(dim=0)) ** 2).sum()

    phi0 = y0 @ tables.recipient0
    phi1 = y1 @ tables.recipient1
    pair = sqrt(tables.recipient0.shape[1]) * phi0 * phi1
    return mmd(y0), mmd(y1), mmd(pair)


def pooled_initializer(data: DTrainingData, seed: int, codes: int) -> tuple[PurposeNormalizer, np.ndarray]:
    """Same fit-only normalization and seeded pooled k-means for protected/control arms."""
    normalizer = PurposeNormalizer.fit((data.h0, data.h1))
    pooled = np.concatenate((normalizer.transform(0, data.h0),
                             normalizer.transform(1, data.h1)), axis=0)
    if len(pooled) < codes:
        raise ValueError("more pooled fit rows than centroids are required")
    km = KMeans(n_clusters=codes, random_state=int(seed), n_init=1,
                max_iter=100, tol=1e-4, algorithm="lloyd")
    km.fit(pooled)
    return normalizer, np.asarray(km.cluster_centers_, dtype=np.float32).copy()


class _Head(nn.Module):
    def __init__(self, dim: int, width: int, classes: int):
        super().__init__()
        self.l1 = nn.Linear(dim, width)
        self.l2 = nn.Linear(width, classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.l2(F.gelu(self.l1(z), approximate="none"))



class TorchDRelease:
    """Exact float32 replay of the hard code and protected heads used in fit."""

    def __init__(self, normalizer: PurposeNormalizer, codebook: torch.Tensor,
                 heads: tuple[_Head, _Head]):
        self.normalizer = normalizer
        self.codebook = codebook.detach().clone().cpu().float()
        self.heads = tuple(copy.deepcopy(head).cpu().eval() for head in heads)
        for head in self.heads:
            for parameter in head.parameters():
                parameter.requires_grad_(False)

    @property
    def code_length_bits(self) -> int:
        return self.codebook.shape[0].bit_length() - 1

    def recipient(self, purpose: int, latent: np.ndarray) -> dict:
        if purpose not in (0, 1):
            raise ValueError("purpose must be 0 or 1")
        # Match train_d: normalize in float64, then cast once to float32.
        standardized = self.normalizer.transform(purpose, latent).astype(np.float32)
        with torch.no_grad():
            x = torch.from_numpy(standardized)
            distance = _distances(x, self.codebook)
            ids = torch.argmin(distance, dim=1)
            decoded = self.codebook[ids]
            probabilities = torch.softmax(self.heads[purpose](decoded), dim=1)
            return {
                "code_ids": ids[:, None].numpy().copy(),
                "decoded": decoded.numpy().copy(),
                "probabilities": probabilities.numpy().copy(),
                "decision": torch.argmax(probabilities, dim=1).numpy().copy(),
                "metadata": {"purpose": purpose, "code_length_bits": self.code_length_bits,
                             "block_count": 1, "fallback": False, "stochastic": False},
            }


@dataclass
class DFitResult:
    release: TorchDRelease
    history: list[dict]
    role: str
    seed: int
    protected: bool
    fit_rows: int
    fit_code_change: dict


def code_change_receipt(
    h0: np.ndarray, h1: np.ndarray, normalizer: PurposeNormalizer,
    initial: np.ndarray, final: torch.Tensor, chunk_size: int = 256,
) -> dict:
    """Aggregate hard-ID churn, ties and occupancy; never persists row arrays."""
    if len(h0) != len(h1) or chunk_size < 1:
        raise ValueError("aligned latents and positive chunk size required")
    initial_t = torch.from_numpy(np.asarray(initial, dtype=np.float32))
    final_t = final.detach().cpu().float()
    views = []
    for purpose, latent in ((0, h0), (1, h1)):
        x = torch.from_numpy(normalizer.transform(purpose, latent).astype(np.float32))
        first_ids, last_ids = [], []
        first_ties = last_ties = 0
        with torch.no_grad():
            for start in range(0, len(x), chunk_size):
                batch = x[start:start + chunk_size]
                d_first, d_last = _distances(batch, initial_t), _distances(batch, final_t)
                first_ids.append(torch.argmin(d_first, dim=1).numpy())
                last_ids.append(torch.argmin(d_last, dim=1).numpy())
                first_ties += int(((d_first == d_first.min(dim=1, keepdim=True).values).sum(dim=1) > 1).sum())
                last_ties += int(((d_last == d_last.min(dim=1, keepdim=True).values).sum(dim=1) > 1).sum())
        before, after = np.concatenate(first_ids), np.concatenate(last_ids)
        views.append((before, after, first_ties, last_ties))
    changed0, changed1 = views[0][0] != views[0][1], views[1][0] != views[1][1]
    n = len(h0)
    return {
        "rows": n,
        "recipient_0": {"changed_rows": int(changed0.sum()), "changed_fraction": float(changed0.mean()),
                        "initial_tie_rows": views[0][2], "final_tie_rows": views[0][3]},
        "recipient_1": {"changed_rows": int(changed1.sum()), "changed_fraction": float(changed1.mean()),
                        "initial_tie_rows": views[1][2], "final_tie_rows": views[1][3]},
        "coalition_changed_rows": int(np.logical_or(changed0, changed1).sum()),
        "occupancy": occupancy_report(views[0][1][:, None], views[1][1][:, None]),
    }


def train_d(
    data: DTrainingData,
    seed: int,
    protected: bool,
    config: DTrainConfig = DTrainConfig(),
    initializer: tuple[PurposeNormalizer, np.ndarray] | None = None,
) -> DFitResult:
    """Fit a hard VQ on an explicitly supplied synthetic/admitted fitting role.

    The codebook, both heads, and public sign tables are fixed before the first step.
    Each minibatch uses one aligned index vector for the two purposes and SEX.
    The final iterate is returned without task/leakage-based epoch selection.
    """
    if data.role == "OSF_DEFENSE_FIT" and (data.h0.shape[1] != 16 or config != DTrainConfig()):
        raise ValueError("Adult candidate is fixed to 16-D and the registered configuration")
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    normalizer, initial = initializer if initializer is not None else pooled_initializer(data, seed, config.codes)
    initial = np.asarray(initial, dtype=np.float32)
    if initial.shape != (config.codes, data.h0.shape[1]) or not np.isfinite(initial).all():
        raise ValueError("initializer codebook shape or values are invalid")
    d = data.h0.shape[1]
    x0 = torch.from_numpy(normalizer.transform(0, data.h0).astype(np.float32))
    x1 = torch.from_numpy(normalizer.transform(1, data.h1).astype(np.float32))
    y0, y1 = torch.from_numpy(data.y0), torch.from_numpy(data.y1)
    u0, u1 = torch.from_numpy(data.teacher0), torch.from_numpy(data.teacher1)
    sex = torch.from_numpy(data.sex)
    codebook = nn.Parameter(torch.from_numpy(initial.copy()))
    head0 = _Head(d, config.hidden, u0.shape[1])
    head1 = _Head(d, config.hidden, u1.shape[1])
    optimizer = torch.optim.Adam([codebook, *head0.parameters(), *head1.parameters()],
                                 lr=config.learning_rate)
    sign_tables = make_sign_tables(seed * 2 + 1, config.codes, config.pair_features)
    permutation_rng = np.random.default_rng(seed)
    history: list[dict] = []
    for epoch in range(config.epochs):
        totals = np.zeros(4, dtype=np.float64)
        batches = 0
        for ix_np in np.array_split(permutation_rng.permutation(len(x0)),
                                    range(config.batch_size, len(x0), config.batch_size)):
            if len(ix_np) < 2:
                raise ValueError("a singleton minibatch is invalid")
            ix = torch.as_tensor(ix_np, dtype=torch.long)
            z0, _, code0 = surrogate_select(x0[ix], codebook)
            z1, _, code1 = surrogate_select(x1[ix], codebook)
            logits0, logits1 = head0(z0), head1(z1)
            task0 = (F.cross_entropy(logits0, y0[ix])
                     + 0.5 * F.kl_div(F.log_softmax(logits0, dim=1), u0[ix], reduction="batchmean")
                     + 0.25 * ((x0[ix] - z0) ** 2).sum(dim=1).mean() / d)
            task1 = (F.cross_entropy(logits1, y1[ix])
                     + 0.5 * F.kl_div(F.log_softmax(logits1, dim=1), u1[ix], reduction="batchmean")
                     + 0.25 * ((x1[ix] - z1) ** 2).sum(dim=1).mean() / d)
            task = (task0 + task1) / 2
            if protected:
                mmd0, mmd1, mmd_pair = mmd_terms(code0, code1, sex[ix], sign_tables)
                privacy = (mmd0 + mmd1 + mmd_pair) / 3
            else:
                privacy = task.new_zeros(())
            loss = task + privacy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals += [float(task0.detach()), float(task1.detach()),
                       float(privacy.detach()), float(loss.detach())]
            batches += 1
        history.append({"epoch": epoch + 1, "batches": batches,
                        "task0": totals[0] / batches, "task1": totals[1] / batches,
                        "code_id_mmd": totals[2] / batches, "objective": totals[3] / batches})
    release = TorchDRelease(normalizer, codebook, (head0, head1))
    fit_code_change = code_change_receipt(data.h0, data.h1, normalizer, initial, codebook)
    return DFitResult(release=release, history=history, role=data.role,
                      seed=int(seed), protected=bool(protected), fit_rows=len(x0),
                      fit_code_change=fit_code_change)
