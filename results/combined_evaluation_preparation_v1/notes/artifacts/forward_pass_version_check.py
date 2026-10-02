"""Frozen forward-pass agreement across PCRL code versions (no fitting, no gradients).

durable-guarantees (dg@956f5c8 utils/pcrl_io.py) imported the PCRL encoder classes
from whatever PCRL working tree PCRL_ROOT pointed at (no commit pin). pcrl/models/
{encoder,lora}.py changed after the Round-4 training (866ee7e, 910f4fe, 940912c,
c2d6e1a, 96c03d0, dbb24f4, 71d6a0f, 5f162ab). This script rebuilds the encoder
exactly as dg's _build_and_load_encoder does, under several PCRL snapshots, runs
the frozen forward pass on the PCRL train split (dg's rows), and reports
max |dH| between versions plus a sha256 of the float32 representation per
version. Nothing individual-level is written; only digests and summary stats.

Usage: python forward_pass_version_check.py --snapshots DIR1 DIR2 ... --data PCRL_DATA
        --ckpt-adult PATH --ckpt-hmda PATH --out JSON
Each DIR is a `git archive <ref> pcrl` extraction; the ref name is the dir basename.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch


def build_and_forward(snap: Path, dataset: str, ckpt_path: Path, data_root: Path):
    for m in [k for k in sys.modules if k == "pcrl" or k.startswith("pcrl.")]:
        del sys.modules[m]
    sys.path.insert(0, str(snap))
    try:
        from torch.utils.data import DataLoader
        base = importlib.import_module("pcrl.data.base")
        enc_mod = importlib.import_module("pcrl.models.encoder")
        lora_mod = importlib.import_module("pcrl.models.lora")
        if dataset == "adult":
            dm = importlib.import_module("pcrl.data.adult")
            purposes = dm.get_adult_purposes()
            ds = dm.AdultDataset(purposes=purposes, root=str(data_root), split="train", download=False)
        else:
            dm = importlib.import_module("pcrl.data.hmda")
            purposes = dm.get_hmda_purposes()
            ds = dm.HMDADataset(purposes=purposes, root=str(data_root), split="train")
        loader = DataLoader(ds, batch_size=512, shuffle=False,
                            collate_fn=base.collate_pcrl_batch, num_workers=0)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        target = "repr_proj_only"
        for k in ck["lora_adapters"]:
            if re.match(r"^\d+\.([12])\.", k):
                target = "all_linear"; break
        cfg = ck.get("config", {}) or {}
        rank = int(ck["lora_adapters"]["0.0.A.weight"].shape[0])
        alpha = float(cfg.get("lora_alpha", 16.0))
        backbone = enc_mod.StandardEncoder(input_dim=ds.info.num_features,
                                           hidden_dims=[128, 128], repr_dim=64, dropout=0.3)
        kw = dict(n_purposes=len(purposes), rank=rank, alpha=alpha, dropout=0.0)
        try:
            enc = lora_mod.PerPurposeLoRAEncoder(backbone, lora_target=target, **kw)
        except TypeError:
            enc = lora_mod.PerPurposeLoRAEncoder(backbone, **kw)
        missing = enc.backbone.load_state_dict(ck["backbone"], strict=False)
        enc.adapters.load_state_dict(ck["lora_adapters"])
        buf = ck.get("encoder_buffers", {}) or {}
        n_leace = 0
        for p in range(len(purposes)):
            if f"leace_P_p{p}" in buf and f"leace_mu_p{p}" in buf:
                enc.set_leace_projection(p, buf[f"leace_P_p{p}"], buf[f"leace_mu_p{p}"]); n_leace += 1
        enc.eval()
        with torch.no_grad():
            H = np.concatenate([enc(b["features"], 0).numpy() for b in loader]).astype(np.float32)
        return H, {"lora_target": target, "rank": rank, "alpha": alpha, "n_leace_set": n_leace,
                   "missing_keys": list(missing.missing_keys), "unexpected_keys": list(missing.unexpected_keys)}
    finally:
        sys.path.remove(str(snap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt-adult", required=True)
    ap.add_argument("--ckpt-hmda", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    out = {"device": "cpu", "purpose_idx": 0, "results": {}}
    for ds, ck in (("adult", a.ckpt_adult), ("hmda", a.ckpt_hmda)):
        Hs, res = {}, {}
        for s in a.snapshots:
            name = Path(s).name
            try:
                H, meta = build_and_forward(Path(s), ds, Path(ck), Path(a.data))
                Hs[name] = H
                res[name] = {**meta, "shape": list(H.shape),
                             "H_float32_sha256": hashlib.sha256(H.tobytes()).hexdigest(),
                             "mean_per_dim_std": float(H.std(0).mean())}
            except Exception as e:
                res[name] = {"error": f"{type(e).__name__}: {e}"}
        names = list(Hs)
        pair = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pair[f"{names[i]}__vs__{names[j]}"] = float(np.abs(Hs[names[i]] - Hs[names[j]]).max())
        out["results"][ds] = {"per_version": res, "max_abs_diff": pair}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
