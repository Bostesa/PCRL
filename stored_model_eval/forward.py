"""Frozen forward pass of a stored PCRL v2 checkpoint (torch imported lazily, eval mode, no grad).

Independent re-implementation of the architecture the Round-4/5 tabular checkpoints were trained with
(PCRL b96c412: pcrl/models/encoder.py::StandardEncoder + pcrl/models/lora.py::PerPurposeLoRAEncoder +
pcrl/models/task_head.py::TaskHead), rebuilt from tensor shapes and the embedded config only:

  for each hidden Linear l (backbone.network.{0,4}):
      z = x W_l^T + b_l + s * (x A_{p,l}^T) B_{p,l}^T + d_{p,l}        s = lora_alpha / rank
      x = Dropout(ReLU(BatchNorm_eval(z)))                               (Dropout = identity in eval)
  rep_p = x W_proj^T + b_proj + s * (x A_{p,2}^T) B_{p,2}^T + d_{p,2}
  logits_head = Linear(Dropout(ReLU(Linear(rep_p))))                    (task_heads.<name>.network.{0,3})

Refuses: hash mismatch; checkpoints carrying erase-layer buffers (encoder_buffers / leace_*) because this
re-implementation does not model them; any unexpected state-dict key (strict).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .admission import sha256_file
from .guards import require_outside_git


class HashMismatch(PermissionError):
    pass


def load_checkpoint(path: str | Path, expected_sha256: str, allow_pickle: bool = False):
    path = Path(path)
    got = sha256_file(path)
    if got != expected_sha256:
        raise HashMismatch(f"checkpoint {path} sha256 {got} != expected {expected_sha256}; refusing")
    import torch
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        mode = "weights_only=True"
    except Exception as e:  # noqa: BLE001
        if not allow_pickle:
            raise RuntimeError(f"weights_only load failed ({e}); pass allow_pickle to use the unsafe loader")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        mode = "weights_only=False (explicitly allowed)"
    return ck, {"path": str(path), "sha256": got, "load_mode": mode, "torch": torch.__version__}


class FrozenPCRLv2:
    BN_EPS = 1e-5  # torch.nn.BatchNorm1d default, as used by StandardEncoder

    def __init__(self, ck: dict):
        import torch
        self.torch = torch
        if ck.get("encoder_buffers") or any("leace" in k for k in ck.get("backbone", {})):
            raise NotImplementedError("checkpoint has erase-layer buffers; not modelled by this re-implementation")
        bb, ad, th = ck["backbone"], ck["lora_adapters"], ck.get("task_heads", {})
        lin = sorted({int(m.group(1)) for k in bb for m in [re.match(r"network\.(\d+)\.weight$", k)]
                      if m and bb[k].ndim == 2})
        self.hidden = []
        expected = set()
        for i in lin:
            bn = i + 1
            for k in (f"network.{i}.weight", f"network.{i}.bias", f"network.{bn}.weight", f"network.{bn}.bias",
                      f"network.{bn}.running_mean", f"network.{bn}.running_var", f"network.{bn}.num_batches_tracked"):
                if k not in bb:
                    raise KeyError(f"backbone missing {k}")
                expected.add(k)
            self.hidden.append({"W": bb[f"network.{i}.weight"], "b": bb[f"network.{i}.bias"],
                                "g": bb[f"network.{bn}.weight"], "beta": bb[f"network.{bn}.bias"],
                                "rm": bb[f"network.{bn}.running_mean"], "rv": bb[f"network.{bn}.running_var"]})
        expected |= {"repr_proj.weight", "repr_proj.bias"}
        extra = set(bb) - expected
        if extra:
            raise KeyError(f"unexpected backbone keys {sorted(extra)} (strict)")
        self.proj = {"W": bb["repr_proj.weight"], "b": bb["repr_proj.bias"]}
        self.n_layers = len(self.hidden) + 1
        pur = sorted({int(k.split(".")[0]) for k in ad})
        self.n_purposes = len(pur)
        self.rank = int(ad["0.0.A.weight"].shape[0])
        cfg = ck.get("config", {}) or {}
        self.alpha = float(cfg.get("lora_alpha", self.rank))
        if int(cfg.get("lora_rank", self.rank)) != self.rank:
            raise ValueError("config lora_rank disagrees with adapter tensor shape")
        self.scaling = self.alpha / self.rank
        self.adapters = [[{"A": ad[f"{p}.{l}.A.weight"], "B": ad[f"{p}.{l}.B.weight"], "d": ad[f"{p}.{l}.bias"]}
                          for l in range(self.n_layers)] for p in pur]
        if set(ad) != {f"{p}.{l}.{t}" for p in pur for l in range(self.n_layers) for t in ("A.weight", "B.weight", "bias")}:
            raise KeyError("adapter keys do not match backbone Linear layout (strict)")
        names = []
        for k in th:
            n = k.split(".network.")[0]
            if n not in names:
                names.append(n)
        self.head_names = names
        self.heads = {n: {"W0": th[f"{n}.network.0.weight"], "b0": th[f"{n}.network.0.bias"],
                          "W3": th[f"{n}.network.3.weight"], "b3": th[f"{n}.network.3.bias"]} for n in names}
        self.input_dim = int(self.hidden[0]["W"].shape[1])
        self.repr_dim = int(self.proj["W"].shape[0])

    def describe(self) -> dict:
        return {"input_dim": self.input_dim, "hidden_dims": [int(h["W"].shape[0]) for h in self.hidden],
                "repr_dim": self.repr_dim, "n_purposes": self.n_purposes, "lora_rank": self.rank,
                "lora_alpha": self.alpha, "heads_in_checkpoint_order": self.head_names}

    def _lin(self, x, W, b, a):
        return x @ W.T + b + self.scaling * ((x @ a["A"].T) @ a["B"].T) + a["d"]

    def encode(self, x, p: int):
        t = self.torch
        with t.no_grad():
            h = x
            for l, L in enumerate(self.hidden):
                z = self._lin(h, L["W"], L["b"], self.adapters[p][l])
                z = (z - L["rm"]) / t.sqrt(L["rv"] + self.BN_EPS) * L["g"] + L["beta"]
                h = t.relu(z)
            return self._lin(h, self.proj["W"], self.proj["b"], self.adapters[p][-1])

    def head(self, rep, name):
        t = self.torch
        H = self.heads[name]
        with t.no_grad():
            return t.relu(rep @ H["W0"].T + H["b0"]) @ H["W3"].T + H["b3"]


def run_forward(model: FrozenPCRLv2, X: np.ndarray, batch_size: int = 512, purpose_head: dict | None = None):
    """Returns {"rep_p{k}": (n, repr_dim), "logits_<head>": (n, C)}; head j is applied to purpose j's
    representation unless purpose_head maps head -> purpose index explicitly."""
    t = model.torch
    t.set_grad_enabled(False)
    Xt = t.as_tensor(np.asarray(X, dtype=np.float32))
    if Xt.shape[1] != model.input_dim:
        raise ValueError(f"features have {Xt.shape[1]} columns, checkpoint expects {model.input_dim}")
    out = {}
    ph = purpose_head or {n: j for j, n in enumerate(model.head_names)}
    for p in range(model.n_purposes):
        reps = [model.encode(Xt[i:i + batch_size], p) for i in range(0, len(Xt), batch_size)]
        out[f"rep_p{p}"] = t.cat(reps).numpy()
    for n, p in ph.items():
        out[f"logits_{n}"] = model.head(t.as_tensor(out[f"rep_p{p}"]), n).numpy()
    return out


def forward_to_cache(ckpt_path, expected_sha256, X, row_ids, cache_dir, tag: str, batch_size: int = 512,
                     allow_pickle: bool = False, dry_run: bool = False) -> dict:
    cache = require_outside_git(Path(cache_dir), "forward cache")
    ck, prov = load_checkpoint(ckpt_path, expected_sha256, allow_pickle)
    model = FrozenPCRLv2(ck)
    info = {"provenance": prov, "architecture": model.describe(), "n_rows": int(len(row_ids)),
            "checkpoint_state": {k: (v if not hasattr(v, "item") else v.item()) for k, v in ck.get("state", {}).items()}}
    if dry_run:
        info["dry_run"] = True
        return info
    out = run_forward(model, X, batch_size)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(cache / f"{tag}.npz", row_id=np.asarray(row_ids), **out)
    info["output"] = str(cache / f"{tag}.npz")
    info["output_sha256"] = sha256_file(cache / f"{tag}.npz")
    info["arrays"] = {k: list(v.shape) for k, v in out.items()}
    (cache / f"{tag}.provenance.json").write_text(json.dumps(info, indent=1, default=str))
    return info
