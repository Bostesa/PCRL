"""HuggingFace ``LabHC/bias_in_bios`` → top-10 stratified subsample DataLoaders.

- Filters to the top-10 occupations by training-set frequency.
- Stratified-subsamples 50,000 bios from the train split, proportional to
  the (occupation × gender) bucket size.
- Tokenises with bert-base-uncased at ``max_length=128, truncation=True,
  padding="max_length"``.
- Evaluates on the full dev split (filtered to top-10).

Returned batches have keys: ``input_ids``, ``attention_mask``, ``occupation``
(int 0..9, local index into ``BIOS_TOP10``), ``gender`` (int 0/1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Profession id ↔ string mapping used by ``LabHC/bias_in_bios`` (alphabetic
# order, 28 classes). Source: De-Arteaga et al. NAACL 2019. Hardcoded here
# because the HF release stores ``profession`` as a bare int64 with no label
# names.
BIOS_LABELS: list[str] = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker",
    "interior_designer", "journalist", "model", "nurse", "painter",
    "paralegal", "pastor", "personal_trainer", "photographer", "physician",
    "poet", "professor", "psychologist", "rapper", "software_engineer",
    "surgeon", "teacher", "yoga_teacher",
]

# Top-10 occupations by training frequency, in user-specified display order.
# Verified against a train-split count: these IDs are the 10 most frequent.
BIOS_TOP10: list[str] = [
    "professor", "physician", "attorney", "photographer", "journalist",
    "nurse", "psychologist", "teacher", "dentist", "surgeon",
]
BIOS_TOP10_IDS: list[int] = [BIOS_LABELS.index(name) for name in BIOS_TOP10]
_LABEL_TO_LOCAL: dict[int, int] = {pid: i for i, pid in enumerate(BIOS_TOP10_IDS)}
_TOP10_SET: frozenset[int] = frozenset(BIOS_TOP10_IDS)


@dataclass
class TokenizedBios(Dataset):
    input_ids: torch.Tensor          # (N, 128) int64 — hard_text
    attention_mask: torch.Tensor     # (N, 128) int64 — hard_text
    occupation: torch.Tensor         # (N,) int64 in [0, 10)
    gender: torch.Tensor             # (N,) int64 in {0, 1}
    # Scrubbed (gender-neutralized) variant — present iff the loader was
    # built with ``include_scrubbed=True`` (Component 3 / pair-invariance).
    input_ids_scrubbed: torch.Tensor | None = None
    attention_mask_scrubbed: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        out = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "occupation": self.occupation[idx],
            "gender": self.gender[idx],
        }
        if self.input_ids_scrubbed is not None:
            out["input_ids_scrubbed"] = self.input_ids_scrubbed[idx]
            out["attention_mask_scrubbed"] = self.attention_mask_scrubbed[idx]
        return out


def _stratified_subsample(
    occ_local: np.ndarray, gen: np.ndarray, n: int, seed: int,
) -> np.ndarray:
    """Indices stratified by (occupation × gender), proportional to bucket size."""
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[int, int], np.ndarray] = {
        (o, g): np.where((occ_local == o) & (gen == g))[0]
        for o in range(10) for g in (0, 1)
    }
    sizes = {k: len(v) for k, v in buckets.items()}
    total = sum(sizes.values())
    if total == 0:
        raise RuntimeError("No rows found in any (occupation, gender) bucket")
    raw = {k: n * sz / total for k, sz in sizes.items()}
    take = {k: int(np.floor(v)) for k, v in raw.items()}
    deficit = n - sum(take.values())
    # Distribute the rounding leftover by largest fractional remainder.
    remainders = sorted(
        raw.items(), key=lambda kv: -(kv[1] - take[kv[0]]),
    )
    for k, _ in remainders[:deficit]:
        take[k] += 1
    out = []
    for k, want in take.items():
        idx = buckets[k]
        chosen = rng.choice(idx, size=min(want, len(idx)), replace=False)
        out.append(chosen)
    out_arr = np.concatenate(out)
    rng.shuffle(out_arr)
    return out_arr


def _tokenize_split(
    rows: dict, tokenizer, max_length: int = 128,
    *, include_scrubbed: bool = False,
) -> TokenizedBios:
    enc = tokenizer(
        rows["hard_text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    occ_local = torch.tensor(
        [_LABEL_TO_LOCAL[p] for p in rows["profession"]], dtype=torch.int64,
    )
    gen = torch.tensor(rows["gender"], dtype=torch.int64)
    ids_scrub = mask_scrub = None
    if include_scrubbed:
        # LabHC's HF release does not ship a scrubbed text field. Construct
        # the counterfactual x' by deterministic pronoun + honorific swap
        # (Zmigrod 2019). x' has the opposite gender of x; the pair-
        # invariance loss in cda_invariance.py pulls CLS(x) ≈ CLS(x').
        from .cda_invariance import swap_gender_pronouns
        cf_texts = [swap_gender_pronouns(t) for t in rows["hard_text"]]
        enc_scrub = tokenizer(
            cf_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        ids_scrub = enc_scrub["input_ids"].long()
        mask_scrub = enc_scrub["attention_mask"].long()
    return TokenizedBios(
        input_ids=enc["input_ids"].long(),
        attention_mask=enc["attention_mask"].long(),
        occupation=occ_local,
        gender=gen,
        input_ids_scrubbed=ids_scrub,
        attention_mask_scrubbed=mask_scrub,
    )


def _filter_top10(ds):
    """Filter a HuggingFace dataset to rows whose ``profession`` is in top-10."""
    return ds.filter(lambda r: r["profession"] in _TOP10_SET, num_proc=1)


def build_bios_loaders(
    *,
    n_train: int = 50_000,
    seed: int = 0,
    batch_size: int = 32,
    max_length: int = 128,
    num_workers: int = 0,
    tokenizer_name: str = "bert-base-uncased",
    cache_dir: str | None = None,
    include_scrubbed: bool = False,
) -> tuple[DataLoader, DataLoader, dict, "TokenizedBios", "TokenizedBios"]:
    """Build (train_loader, dev_loader, info, train_ds, dev_ds).

    ``train_ds`` and ``dev_ds`` are also returned so diagnostics can grab
    raw text / preprocessed tensors without re-tokenising.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    full_train = load_dataset(
        "LabHC/bias_in_bios", split="train", cache_dir=cache_dir,
    )
    train_filtered = _filter_top10(full_train)
    occ_int = np.array(train_filtered["profession"])
    gen_int = np.array(train_filtered["gender"])
    occ_local = np.array([_LABEL_TO_LOCAL[p] for p in occ_int])
    keep_idx = _stratified_subsample(occ_local, gen_int, n_train, seed)
    train_sub = train_filtered.select(keep_idx.tolist())
    train_ds = _tokenize_split(train_sub[:], tokenizer, max_length, include_scrubbed=include_scrubbed)

    full_dev = load_dataset(
        "LabHC/bias_in_bios", split="dev", cache_dir=cache_dir,
    )
    dev_filtered = _filter_top10(full_dev)
    dev_ds = _tokenize_split(dev_filtered[:], tokenizer, max_length, include_scrubbed=include_scrubbed)

    g_train = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g_train,
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Per-class counts for sanity (gender × occupation bucket sizes).
    bucket_counts: dict[str, int] = {}
    occ_train = train_ds.occupation.numpy()
    gen_train = train_ds.gender.numpy()
    for o in range(10):
        for g in (0, 1):
            n_cell = int(((occ_train == o) & (gen_train == g)).sum())
            bucket_counts[f"{BIOS_TOP10[o]}_g{g}"] = n_cell

    info = {
        "n_train": int(len(train_ds)),
        "n_dev": int(len(dev_ds)),
        "tokenizer": tokenizer_name,
        "max_length": max_length,
        "occupations": BIOS_TOP10,
        "occupation_ids_alphabetic": BIOS_TOP10_IDS,
        "train_bucket_counts": bucket_counts,
    }
    return train_loader, dev_loader, info, train_ds, dev_ds
