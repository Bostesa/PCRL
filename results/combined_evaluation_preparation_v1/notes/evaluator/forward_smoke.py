"""Frozen forward-pass smoke on the cached PCRL Round-4 Adult seed-0 checkpoint (no fitting of any model).

Steps
 1. Export PCRL b96c412 (the Round-4 training HEAD) read-only:  git archive b96c412 pcrl | tar -x -C <scratch>
    Used ONLY for (a) the Adult preprocessing pipeline (AdultDataset, norm stats from the train split, as
    run_v2_dataset.py@b96c412:93-98 does) and (b) an architecture cross-check against the original classes.
 2. Build the SMALL admitted subset: first N rows of the post-dropna Adult test split (row_id = position
    in that split). Features + ids are written to the private cache (outside git) with a manifest.
 3. `python -m stored_model_eval forward` (independent re-implementation, eval mode, no grad) twice with
    batch 512 and once with batch 32; compare bitwise / allclose.
 4. Cross-check against the archived PCRL StandardEncoder + PerPurposeLoRAEncoder + TaskHead (strict load).
Writes only aggregate numbers to notes/evaluator/forward_smoke_result.json (no row-level data in git).
Run: /Users/nathansamson/PCRL/.venv/bin/python forward_smoke.py
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WT = Path("/Users/nathansamson/PCRL/.worktrees/combined-evaluation-preparation-v1")
SCR = Path("/private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad")
CACHE = Path.home() / "PCRL_eval_cache_private"
CKPT = CACHE / "checkpoints" / "v2_adult_s0_final.pt"
CKPT_SHA = "1cfc2fefa8929bb4cb9e9680205987e2f3bf3cf37e17a1ba7cf7135cf494c061"
COMMIT = "b96c41256daeed6e644aba1443a47b16e28d089a"
N_ROWS = 256
PY = "/Users/nathansamson/PCRL/.venv/bin/python"
OUT = WT / "results/combined_evaluation_preparation_v1/notes/evaluator/forward_smoke_result.json"

sys.path.insert(0, str(WT))
from stored_model_eval.guards import install_network_guard  # noqa: E402

install_network_guard()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    t_all = time.perf_counter()
    res = {"checkpoint": str(CKPT), "checkpoint_sha256_expected": CKPT_SHA, "pcrl_commit": COMMIT}
    # 1. read-only export of the historical code
    exp = SCR / f"pcrl_{COMMIT[:7]}"
    if not (exp / "pcrl").exists():
        exp.mkdir(parents=True, exist_ok=True)
        subprocess.run(f"git -C /Users/nathansamson/PCRL archive {COMMIT} pcrl | tar -x -C {exp}", shell=True,
                       check=True)
    res["pcrl_export"] = {"dir": str(exp), "command": f"git -C /Users/nathansamson/PCRL archive {COMMIT} pcrl | tar -x -C <dir>"}
    sys.path.insert(0, str(exp))
    from pcrl.data.adult import AdultDataset, get_adult_purposes

    data_root = Path("/Users/nathansamson/PCRL/data")
    for f in ("adult.data", "adult.test"):
        if not (data_root / "adult" / f).exists():  # AdultDataset would silently generate synthetic data
            raise SystemExit(f"missing {f}; refusing (the historical loader falls back to synthetic data)")
    res["data_files_sha256"] = {f: sha(data_root / "adult" / f) for f in ("adult.data", "adult.test")}
    purposes = get_adult_purposes()
    res["purpose_order"] = [p.name for p in purposes]
    train = AdultDataset(purposes=purposes, root=str(data_root), split="train", download=False)
    test = AdultDataset(purposes=purposes, root=str(data_root), split="test", download=False,
                        norm_stats=train.norm_stats)
    X = test.features[:N_ROWS].numpy().astype(np.float32)
    ids = np.arange(N_ROWS, dtype=np.int64)
    res["subset"] = {"split": "test (post-dropna)", "n_rows": N_ROWS, "row_id": "position in post-dropna test split",
                     "n_features": int(X.shape[1]), "test_split_rows": int(len(test.features))}

    # 2. admitted subset in the private cache (outside git)
    sm = CACHE / "forward_smoke"
    sm.mkdir(parents=True, exist_ok=True)
    np.savez(sm / "adult_test_first256.npz", row_id=ids, features=X)
    man = {"schema": "stored_model_eval.manifest/v1", "synthetic": False, "required_arrays": ["features"],
           "files": {"checkpoint": {"path": str(CKPT), "sha256": CKPT_SHA},
                     "subset": {"path": str(sm / "adult_test_first256.npz"),
                                "sha256": sha(sm / "adult_test_first256.npz")}},
           "arrays": {"features": {"file": "subset", "key": "features", "ids": "row_id"}}}
    (sm / "forward_manifest.json").write_text(json.dumps(man, indent=1))

    # 3. CLI forward runs
    runs = {}
    for tag, bs in (("run_a", 512), ("run_b", 512), ("run_bs32", 32)):
        t0 = time.perf_counter()
        p = subprocess.run([PY, "-m", "stored_model_eval", "--out", str(sm / f"{tag}.json"), "forward",
                            "--manifest", str(sm / "forward_manifest.json"), "--cache-dir", str(sm / "cache"),
                            "--tag", tag, "--batch-size", str(bs), "--threads", "1"],
                           cwd=WT, capture_output=True, text=True)
        if p.returncode != 0:
            raise SystemExit(p.stderr[-2000:])
        runs[tag] = {"wall_s": round(time.perf_counter() - t0, 3), "info": json.loads((sm / f"{tag}.json").read_text())}
    za, zb, zc = (np.load(sm / "cache" / f"{t}.npz") for t in ("run_a", "run_b", "run_bs32"))
    keys = [k for k in za.files if k != "row_id"]
    res["outputs"] = {k: list(za[k].shape) for k in keys}
    res["deterministic_bitwise_repeat"] = all(np.array_equal(za[k], zb[k]) for k in keys)
    res["batch32_vs_batch512_max_abs_diff"] = float(max(np.abs(za[k] - zc[k]).max() for k in keys))
    res["row_ids_preserved"] = bool(np.array_equal(za["row_id"], ids))
    res["finite"] = all(np.isfinite(za[k]).all() for k in keys)
    res["runs"] = {k: {"wall_s": v["wall_s"], "load_mode": v["info"]["provenance"]["load_mode"],
                       "architecture": v["info"]["architecture"], "checkpoint_state": v["info"]["checkpoint_state"],
                       "output_sha256": v["info"]["output_sha256"]} for k, v in runs.items()}

    # refusal check: wrong expected hash must refuse
    bad = dict(man)
    bad["files"] = {**man["files"], "checkpoint": {"path": str(CKPT), "sha256": "0" * 64}}
    (sm / "bad_manifest.json").write_text(json.dumps(bad, indent=1))
    p = subprocess.run([PY, "-m", "stored_model_eval", "forward", "--manifest", str(sm / "bad_manifest.json"),
                        "--cache-dir", str(sm / "cache"), "--tag", "should_not_exist"], cwd=WT,
                       capture_output=True, text=True)
    res["hash_mismatch_refused"] = (p.returncode != 0 and not (sm / "cache" / "should_not_exist.npz").exists())
    # refusal check: cache inside git must refuse
    p = subprocess.run([PY, "-m", "stored_model_eval", "forward", "--manifest", str(sm / "forward_manifest.json"),
                        "--cache-dir", str(WT / "tmp_cache_should_not_exist"), "--tag", "x"], cwd=WT,
                       capture_output=True, text=True)
    res["cache_inside_git_refused"] = (p.returncode != 0 and not (WT / "tmp_cache_should_not_exist").exists())

    # 4. cross-check against the archived original classes (strict loads)
    import torch
    from pcrl.models.encoder import StandardEncoder
    from pcrl.models.lora import PerPurposeLoRAEncoder
    from pcrl.models.task_head import TaskHead
    torch.set_num_threads(1)
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    bb = StandardEncoder(input_dim=X.shape[1], hidden_dims=[128, 128], repr_dim=64, dropout=0.3)
    enc = PerPurposeLoRAEncoder(bb, n_purposes=3, rank=int(cfg["lora_rank"]), alpha=float(cfg["lora_alpha"]),
                                dropout=0.0)
    enc.backbone.load_state_dict(ck["backbone"], strict=True)
    enc.adapters.load_state_dict(ck["lora_adapters"], strict=True)
    enc.eval()
    heads = {}
    for name in res["purpose_order"]:
        hd = {k.split(".", 1)[1]: v for k, v in ck["task_heads"].items() if k.startswith(name + ".")}
        h = TaskHead(repr_dim=64, output_dim=hd["network.3.weight"].shape[0],
                     hidden_dim=hd["network.0.weight"].shape[0])
        h.load_state_dict(hd, strict=True)
        heads[name] = h.eval()
    diffs = {}
    with torch.no_grad():
        xt = torch.as_tensor(X)
        for p_idx, name in enumerate(res["purpose_order"]):
            r = enc(xt, p_idx).numpy()
            lg = heads[name](torch.as_tensor(r)).numpy()
            for key, orig in ((f"rep_p{p_idx}", r), (f"logits_{name}", lg)):
                ours = za[key]
                diffs[key] = {"max_abs": float(np.abs(orig - ours).max()),
                              "max_abs_value": float(np.abs(orig).max()),
                              "rel_to_max": float(np.abs(orig - ours).max() / np.abs(orig).max())}
    # float64 reference: both float32 implementations vs the same maths in float64
    enc64 = enc.double()
    heads64 = {n: h.double() for n, h in heads.items()}
    with torch.no_grad():
        x64 = torch.as_tensor(X.astype(np.float64))
        for p_idx, name in enumerate(res["purpose_order"]):
            r64 = enc64(x64, p_idx)
            l64 = heads64[name](r64)
            for key, ref in ((f"rep_p{p_idx}", r64.numpy()), (f"logits_{name}", l64.numpy())):
                diffs[key]["ours_f32_vs_f64_rel"] = float(np.abs(za[key] - ref).max() / np.abs(ref).max())
    res["head_order_matches_purpose_order"] = runs["run_a"]["info"]["architecture"]["heads_in_checkpoint_order"] == res["purpose_order"]
    res["max_abs_diff_vs_archived_pcrl_classes"] = diffs
    res["agrees_with_archived_classes_rel_1e-5"] = all(v["rel_to_max"] < 1e-5 for v in diffs.values())
    rep0 = za["rep_p0"]
    res["aggregate_rep_p0"] = {"mean_per_dim_std": float(rep0.std(0).mean()), "abs_mean": float(np.abs(rep0).mean())}
    res["total_wall_s"] = round(time.perf_counter() - t_all, 2)
    res["fits_performed"] = 0
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("deterministic_bitwise_repeat", "batch32_vs_batch512_max_abs_diff",
                                          "max_abs_diff_vs_archived_pcrl_classes", "hash_mismatch_refused",
                                          "cache_inside_git_refused", "outputs", "total_wall_s")}, indent=1))


if __name__ == "__main__":
    main()
