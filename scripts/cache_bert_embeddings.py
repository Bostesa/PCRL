"""Cache BERT-base layer-12 [CLS] embeddings on BIOS for the layer-12 PCRL run.

Output (per ``--output-dir``):
    cls_layer12_embeddings.npz   — train/dev cached features (standardized) + raw
    labels.npz                   — gender + per-purpose task labels (P1/P2/P3)
    standardize_stats.npz        — per-dim train mean/std (so any downstream
                                   pipeline can apply the same transform)
    cache_meta.json              — split sizes, model name, label maps

Frozen ``bert-base-uncased`` (no LoRA, no fine-tuning). We use the FULL 28-way
profession set (NOT the top-10 subset that ``pcrl/language/bios_dataset.py``
filters to), because P1's task is the full 28-way occupation classification.

Per-dimension train standardization is applied here so all downstream scripts
operate on the same scaled feature tensor; LEACE warm-start, R-LACE, MLP probes
all share that input.

Usage:
    python scripts/cache_bert_embeddings.py \\
        --output-dir results/bios_pcrl_layer12/cache \\
        --device cuda --batch-size 128
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


BIOS_LABELS_28: list[str] = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker",
    "interior_designer", "journalist", "model", "nurse", "painter",
    "paralegal", "pastor", "personal_trainer", "photographer", "physician",
    "poet", "professor", "psychologist", "rapper", "software_engineer",
    "surgeon", "teacher", "yoga_teacher",
]
LABEL_TO_IDX: dict[str, int] = {n: i for i, n in enumerate(BIOS_LABELS_28)}


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _build_super_map(p2_cfg: dict) -> dict[int, int]:
    """occupation_28_idx -> supercategory_5_idx."""
    order = p2_cfg["class_index_order"]  # length-5
    super_to_idx = {name: i for i, name in enumerate(order)}
    out: dict[int, int] = {}
    for super_name, occ_list in p2_cfg["supercategories"].items():
        super_idx = super_to_idx[super_name]
        for occ in occ_list:
            if occ not in LABEL_TO_IDX:
                raise ValueError(
                    f"Supercategory mapping references unknown occupation "
                    f"'{occ}' (not in BIOS_LABELS_28)"
                )
            out[LABEL_TO_IDX[occ]] = super_idx
    missing = set(range(28)) - set(out.keys())
    if missing:
        names = [BIOS_LABELS_28[i] for i in sorted(missing)]
        raise ValueError(
            f"P2 supercategory mapping is missing occupations: {names}. "
            f"Every BIOS occupation must be assigned a supercategory."
        )
    return out


def _build_medical_set(p3_cfg: dict) -> set[int]:
    medical_names = p3_cfg["medical"]
    out = set()
    for n in medical_names:
        if n not in LABEL_TO_IDX:
            raise ValueError(f"P3 medical list contains unknown occupation: {n}")
        out.add(LABEL_TO_IDX[n])
    return out


@torch.no_grad()
def _encode(
    model, tokenizer, texts: list[str], *, batch_size: int, max_length: int,
    device: torch.device, log_prefix: str,
) -> np.ndarray:
    out = np.empty((len(texts), 768), dtype=np.float32)
    n_batches = (len(texts) + batch_size - 1) // batch_size
    t0 = time.time()
    for bi, start in enumerate(range(0, len(texts), batch_size)):
        end = min(start + batch_size, len(texts))
        enc = tokenizer(
            texts[start:end],
            max_length=max_length, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        h = model(input_ids=ids, attention_mask=mask, return_dict=True)
        # last_hidden_state[:, 0, :] is the layer-12 [CLS] (encoder.layer[11] output
        # post-LayerNorm, BEFORE the BERT pooler's tanh projection).
        cls = h.last_hidden_state[:, 0, :].detach().to(torch.float32).cpu().numpy()
        out[start:end] = cls
        if (bi + 1) % 25 == 0 or end == len(texts):
            elapsed = time.time() - t0
            print(
                f"  [{log_prefix}] batch {bi+1}/{n_batches} "
                f"({end}/{len(texts)} examples)  elapsed={elapsed:.1f}s",
                flush=True,
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--config-dir", type=Path,
                    default=ROOT / "configs/bios_pcrl_layer12")
    ap.add_argument("--model-name", type=str, default="bert-base-uncased")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=-1,
                    help="If >0, take a stratified-ish prefix of the train set "
                    "for smoke runs. Defaults to full split.")
    ap.add_argument("--n-dev", type=int, default=-1,
                    help="If >0, take a prefix of the dev set for smoke.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Used only when subsampling with --n-train/--n-dev.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[cache] device={device}  model={args.model_name}", flush=True)

    p2_cfg = _load_yaml(args.config_dir / "P2.yaml")
    p3_cfg = _load_yaml(args.config_dir / "P3.yaml")
    super_map = _build_super_map(p2_cfg)
    medical_set = _build_medical_set(p3_cfg)
    print(f"[cache] super_map covers {len(super_map)} occupations; "
          f"medical_set has {len(medical_set)} occupations", flush=True)

    print("[cache] loading dataset LabHC/bias_in_bios ...", flush=True)
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    train_full = load_dataset("LabHC/bias_in_bios", split="train")
    dev_full = load_dataset("LabHC/bias_in_bios", split="dev")
    print(f"[cache] full train={len(train_full)}  full dev={len(dev_full)}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    if args.n_train > 0 and args.n_train < len(train_full):
        idx_train = rng.choice(len(train_full), size=args.n_train, replace=False)
        idx_train.sort()
        train = train_full.select(idx_train.tolist())
        print(f"[cache] subsampled train -> {len(train)} (seed={args.seed})",
              flush=True)
    else:
        train = train_full

    if args.n_dev > 0 and args.n_dev < len(dev_full):
        idx_dev = rng.choice(len(dev_full), size=args.n_dev, replace=False)
        idx_dev.sort()
        dev = dev_full.select(idx_dev.tolist())
        print(f"[cache] subsampled dev -> {len(dev)} (seed={args.seed})",
              flush=True)
    else:
        dev = dev_full

    print("[cache] loading tokenizer + model ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Encode train and dev.
    train_text = train["hard_text"]
    dev_text = dev["hard_text"]
    print(f"[cache] encoding train ({len(train_text)} examples)", flush=True)
    raw_train = _encode(
        model, tokenizer, train_text,
        batch_size=args.batch_size, max_length=args.max_length,
        device=device, log_prefix="train",
    )
    print(f"[cache] encoding dev ({len(dev_text)} examples)", flush=True)
    raw_dev = _encode(
        model, tokenizer, dev_text,
        batch_size=args.batch_size, max_length=args.max_length,
        device=device, log_prefix="dev",
    )

    # Per-dim standardization on TRAIN; apply same transform to DEV.
    mean = raw_train.mean(axis=0).astype(np.float32)
    std = raw_train.std(axis=0).astype(np.float32)
    std_safe = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    cls_train = ((raw_train - mean) / std_safe).astype(np.float32)
    cls_dev = ((raw_dev - mean) / std_safe).astype(np.float32)

    # Labels.
    occ_train = np.asarray(train["profession"], dtype=np.int64)
    occ_dev = np.asarray(dev["profession"], dtype=np.int64)
    gen_train = np.asarray(train["gender"], dtype=np.int64)
    gen_dev = np.asarray(dev["gender"], dtype=np.int64)

    super_train = np.asarray(
        [super_map[int(o)] for o in occ_train], dtype=np.int64,
    )
    super_dev = np.asarray(
        [super_map[int(o)] for o in occ_dev], dtype=np.int64,
    )
    medical_train = np.asarray(
        [(1 if int(o) in medical_set else 0) for o in occ_train], dtype=np.int64,
    )
    medical_dev = np.asarray(
        [(1 if int(o) in medical_set else 0) for o in occ_dev], dtype=np.int64,
    )

    # Save artifacts.
    emb_path = args.output_dir / "cls_layer12_embeddings.npz"
    labels_path = args.output_dir / "labels.npz"
    stats_path = args.output_dir / "standardize_stats.npz"
    meta_path = args.output_dir / "cache_meta.json"

    np.savez_compressed(
        emb_path,
        train_standardized=cls_train, dev_standardized=cls_dev,
        train_raw=raw_train.astype(np.float32),
        dev_raw=raw_dev.astype(np.float32),
    )
    np.savez_compressed(
        labels_path,
        gender_train=gen_train, gender_dev=gen_dev,
        occupation_train=occ_train, occupation_dev=occ_dev,
        supercategory_train=super_train, supercategory_dev=super_dev,
        medical_train=medical_train, medical_dev=medical_dev,
    )
    np.savez_compressed(stats_path, mean=mean, std=std_safe, raw_std=std)

    # Per-purpose label histograms (sanity).
    def _hist(y: np.ndarray, k: int) -> dict[int, int]:
        return {int(c): int((y == c).sum()) for c in range(k)}

    meta = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "n_train": int(len(cls_train)),
        "n_dev": int(len(cls_dev)),
        "feature_dim": int(cls_train.shape[1]),
        "num_classes": {"P1": 28, "P2": 5, "P3": 2},
        "P1_labels_alphabetic": BIOS_LABELS_28,
        "P2_class_index_order": p2_cfg["class_index_order"],
        "P3_class_index_order": p3_cfg["class_index_order"],
        "label_histograms_train": {
            "gender": _hist(gen_train, 2),
            "occupation": _hist(occ_train, 28),
            "supercategory": _hist(super_train, 5),
            "medical": _hist(medical_train, 2),
        },
        "label_histograms_dev": {
            "gender": _hist(gen_dev, 2),
            "occupation": _hist(occ_dev, 28),
            "supercategory": _hist(super_dev, 5),
            "medical": _hist(medical_dev, 2),
        },
        "subsample": {
            "n_train_arg": int(args.n_train),
            "n_dev_arg": int(args.n_dev),
            "seed": int(args.seed),
        },
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[cache] wrote {emb_path}  ({cls_train.shape[0]}+{cls_dev.shape[0]} rows)",
          flush=True)
    print(f"[cache] wrote {labels_path}", flush=True)
    print(f"[cache] wrote {stats_path}", flush=True)
    print(f"[cache] wrote {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
