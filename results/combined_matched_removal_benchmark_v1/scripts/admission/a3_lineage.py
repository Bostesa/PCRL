"""A3: Round-4 lineage of each final.pt + a fit-free row-provenance fingerprint from best.pt. No fitting.

Lineage (per dataset, seed), against origin/main@55e4cb1d1 results/v2_<ds>_ROUND4/{per_seed_results,
dominant_axis_audit}.json:
  - state.epoch == per_seed.last_epoch == dominant_axis_audit.per_seed[s].epoch == 204 (Round 4 = 5 warmup + 200;
    Round 5 ends at 199);
  - saved dual state ck['lambdas'] == per_seed.lambdas_final (same keys, exact float equality);
  - 205 history entries per key; embedded config.checkpoint_dir ends with v2_<ds>_s<seed>; epochs 200, warmup 5;
  - global_step == 205 * ceil(n_train / 256) with n_train from the regenerated b96c412 train split;
  - checkpoint input_dim == regenerated feature width; no erase-layer buffers;
  - test-split class priors of every audited attribute == dominant_axis_audit priors (label-level check).
Row-provenance fingerprint (decides nothing about the encoder; feeds the defense_fit route decision):
  run_v2_dataset.py reloaded best.pt and computed repr_health (per-dim std, L2 norms, effective rank) and task
  accuracies on the TEST split. We repeat that frozen computation on the regenerated test rows with best.pt
  (eval, no grad). Agreement to float32 tolerance on all purposes and seeds means the regenerated test rows,
  their order and the train-derived normalisation equal what the Round-4 run used.
On success, admitted final.pt files are copied (hash-verified) into ~/PCRL_eval_cache_private/checkpoints/.
Writes notes/admission/lineage.json.
"""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from admission_common import (CKPT_DIR, DATASETS, INPUTS, NOTES, ORIGIN_MAIN, PCRL_REPO, PROV_CKPT, SEEDS,  # noqa
                              STAGING, WT, ensure_export, sha256_file, write_json)

sys.path.insert(0, str(WT))
from stored_model_eval.forward import FrozenPCRLv2, load_checkpoint, run_forward  # noqa: E402  (read-only import)


def git_json(path):
    return json.loads(subprocess.run(["git", "-C", str(PCRL_REPO), "show", f"{ORIGIN_MAIN}:{path}"], check=True,
                                     capture_output=True, text=True).stdout)


def effective_rank(reprs):  # verbatim from b96c412 experiments/run_v2_dataset.py
    if reprs.shape[0] < 2:
        return float("nan")
    centered = reprs - reprs.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    p = (s ** 2) / max((s ** 2).sum(), 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def repr_health(reprs):  # verbatim from b96c412 experiments/run_v2_dataset.py
    per_dim_std = reprs.std(axis=0)
    l2 = np.linalg.norm(reprs, axis=1)
    return {"per_dim_std_mean": float(per_dim_std.mean()), "per_dim_std_max": float(per_dim_std.max()),
            "per_dim_std_min": float(per_dim_std.min()), "l2_norm_mean": float(l2.mean()),
            "l2_norm_std": float(l2.std()), "effective_rank": effective_rank(reprs)}


def purposes_of(ds):
    ensure_export()
    if ds == "adult":
        from pcrl.data.adult import get_adult_purposes
        return get_adult_purposes()
    from pcrl.data.hmda import get_hmda_purposes
    return get_hmda_purposes()


def final_path(ds, s):
    """Staged extract if present, else the (previously admitted, hash-checked on load) private copy."""
    st = STAGING / "checkpoints" / f"v2_{ds}_s{s}" / "final.pt"
    return st if (s != 0 and st.exists()) else CKPT_DIR / f"v2_{ds}_s{s}_final.pt"


def main():
    ext = json.loads((NOTES / "checkpoint_extraction.json").read_text())["members"]
    out = {"reference": f"origin/main@{ORIGIN_MAIN}", "fits_performed": 0, "datasets": {}}
    for ds in DATASETS:
        ps = git_json(f"results/v2_{ds}_ROUND4/per_seed_results.json")
        da = git_json(f"results/v2_{ds}_ROUND4/dominant_axis_audit.json")
        P = purposes_of(ds)
        pnames = [p.name for p in P]
        pairs = [f"{p.name}__{a}" for p in P for a in p.disallowed_attrs]
        rows = np.load(INPUTS / f"{ds}_rows.npz")
        feats = np.load(INPUTS / f"{ds}_features.npz")
        te = rows["split"] == "test"
        assert np.array_equal(rows["row_id"], feats["row_id"])
        Xte = feats["features"][te]
        n_train = int((rows["split"] == "train").sum())
        task_of = {p.name: p.allowed_tasks[0] for p in P}
        dres = {"purposes_in_order": pnames, "pairs_in_order": pairs, "seeds": {}}
        for s in SEEDS:
            r = {}
            psr = next(x for x in ps["per_seed"] if x["seed"] == s)
            dar = da["per_seed"][str(s)]
            fp = final_path(ds, s)
            fsha = ext[f"checkpoints/v2_{ds}_s{s}/final.pt"]["sha256"]
            ck, prov = load_checkpoint(fp, fsha)
            st = ck["state"]
            lam = {k: float(v) for k, v in ck["lambdas"].items()}
            cfg = ck["config"]
            steps = math.ceil(n_train / cfg["batch_size"])
            model = FrozenPCRLv2(ck)
            checks = {
                "sha256_matches_inventory": prov["sha256"] == fsha,
                "epoch_204": st["epoch"] == 204,
                "epoch_eq_per_seed_last_epoch": st["epoch"] == psr["last_epoch"],
                "epoch_eq_dominant_axis_epoch": st["epoch"] == dar["epoch"],
                "lambda_keys_eq_per_seed": list(lam) == list(psr["lambdas_final"]),
                "lambdas_exactly_eq_per_seed": all(lam[k] == psr["lambdas_final"][k] for k in psr["lambdas_final"]),
                "lambda_key_order_eq_b96c412_purpose_pairs": list(lam) == pairs,
                "history_205_entries": all(len(v) == 205 for v in ck["history"].values()),
                "config_checkpoint_dir_is_round4_dir": cfg["checkpoint_dir"].rstrip("/").endswith(f"v2_{ds}_s{s}"),
                "config_epochs_200_warmup_5": cfg["epochs"] == 200 and cfg["warmup_epochs"] == 5,
                "global_step_eq_205_x_ceil(n_train/256)": st["global_step"] == 205 * steps,
                "input_dim_eq_regenerated": model.input_dim == Xte.shape[1],
                "head_order_eq_purposes": model.head_names == pnames,
                "no_erase_layer_buffers": not ck.get("encoder_buffers"),
                "da_rows_pairs_eq_b96c412": [f"{x['purpose']}__{x['attribute']}" for x in dar["rows"]] == pairs,
            }
            # label-level: test priors of every audited attribute == DA priors
            pri = {}
            for x in dar["rows"]:
                y = rows[x["attribute"]][te]
                emp = (np.bincount(y, minlength=x["num_classes"]) / len(y)).tolist()
                pri[f"{x['purpose']}__{x['attribute']}"] = float(np.max(np.abs(np.array(emp) - np.array(x["priors"]))))
            checks["test_priors_eq_dominant_axis_priors(max_abs<1e-12)"] = max(pri.values()) < 1e-12
            r["final"] = {"file": str(fp).replace(str(Path.home()), "~"), "sha256": fsha,
                          "state": {k: (None if isinstance(v, float) and math.isinf(v) else v) for k, v in st.items()},
                          "lambdas": lam, "checks": checks, "prior_max_abs_diff": pri,
                          "admitted": all(checks.values())}
            # ---- best.pt fingerprint over the regenerated test rows
            bsha = ext[f"checkpoints/v2_{ds}_s{s}/best.pt"]["sha256"]
            bck, _ = load_checkpoint(PROV_CKPT / f"v2_{ds}_s{s}_best.pt", bsha)
            bm = FrozenPCRLv2(bck)
            fw = run_forward(bm, Xte, batch_size=256)
            fpr = {}
            worst = 0.0
            for i, pn in enumerate(pnames):
                h = repr_health(fw[f"rep_p{i}"])
                ref = psr["per_purpose_health"][pn]
                rel = {k: abs(h[k] - ref[k]) / max(abs(ref[k]), 1e-12) for k in h}
                worst = max(worst, max(rel.values()))
                fpr[pn] = {"max_rel_diff": max(rel.values()), "shape_eq": list(ref["shape"]) == [int(te.sum()), 64]}
            acc = {}
            for pn in pnames:
                t = task_of[pn]
                pred = fw[f"logits_{pn}"].argmax(1)
                a = float((pred == rows[f"task_{t}"][te]).mean())
                acc[t] = {"recomputed": round(a, 6), "per_seed_results": psr["task_accuracies"][t],
                          "abs_diff": abs(round(a, 6) - psr["task_accuracies"][t])}
            r["best_fingerprint"] = {
                "best_state": {k: (None if isinstance(v, float) and math.isinf(v) else v)
                               for k, v in bck["state"].items()},
                "per_seed_best_epoch": psr["best_epoch"],
                "health_by_purpose": fpr, "health_worst_rel_diff": worst,
                "task_accuracy": acc,
                "health_match(rel<1e-4)": worst < 1e-4,
                "task_acc_match(abs<=1/n)": all(v["abs_diff"] <= 1.0 / int(te.sum()) + 1e-6 for v in acc.values()),
            }
            dres["seeds"][str(s)] = r
            print(ds, s, "admitted" if r["final"]["admitted"] else "NOT ADMITTED",
                  {k: v for k, v in checks.items() if not v}, "fingerprint worst rel", f"{worst:.2e}",
                  {t: v["abs_diff"] for t, v in acc.items()})
        out["datasets"][ds] = dres
    # copy admitted finals into the shared private checkpoint cache
    sums = CKPT_DIR / "SHA256SUMS"
    lines = sums.read_text().splitlines() if sums.exists() else []
    for ds in DATASETS:
        for s in SEEDS:
            r = out["datasets"][ds]["seeds"][str(s)]["final"]
            tgt = CKPT_DIR / f"v2_{ds}_s{s}_final.pt"
            if r["admitted"]:
                if not tgt.exists():
                    shutil.copyfile(final_path(ds, s), tgt)
                assert sha256_file(tgt) == r["sha256"], tgt
                line = f"{r['sha256']}  {tgt.name}"
                if line not in lines:
                    lines.append(line)
                r["admitted_copy"] = str(tgt).replace(str(Path.home()), "~")
    sums.write_text("\n".join(lines) + "\n")
    write_json(NOTES / "lineage.json", out)


if __name__ == "__main__":
    main()
