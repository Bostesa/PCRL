"""Build recount specs for the AAAI '59 of 67 approved configurations fail' audit, then run the package recount.

Configuration list (67) = durable-guarantees@956f5c8 results/honest_reaudit.json master (21 rows, adult/sex +
hmda/race) + results/expansion_reaudit_paperframing.json rows with approved_at_rest (46 rows).
Stored per-attacker AUCs: honest_reaudit master xgb_auc/mlp_auc; expansion_reaudit master xgb/mlp/lora_auc.
Probability arrays: drive members durable-guarantees/analysis/{tpr59_scores,tpr_ext_scores}/*.npz extracted to
session scratch; each file's sha256 is checked against the drive inventory before use.
File naming follows experiments/run_tpr_failing59.py / run_tpr_extension.py (slug = written file name).
Suites: (1) paper rule XGB+MLP; (2) + LoRA where the file stores it.
Run: python3 build_aaai67_spec.py   (system python3; no fitting)
"""
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SCR = Path("/private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad")
DG = SCR / "dg"
AN = SCR / "recon/verification/dg_drive/durable-guarantees/analysis"
INV = Path("/Users/nathansamson/storage-relocation-20260930/inventories/tree-durable-guarantees.json.gz")
HERE = Path(__file__).resolve().parent
WT = Path("/Users/nathansamson/PCRL/.worktrees/combined-evaluation-preparation-v1")

inv = json.load(gzip.open(INV))
hr = json.load(open(DG / "results/honest_reaudit.json"))
pf = json.load(open(DG / "results/expansion_reaudit_paperframing.json"))
er = json.load(open(DG / "results/expansion_reaudit.json"))
dg_head = subprocess.check_output(["git", "-C", str(DG), "rev-parse", "HEAD"], text=True).strip()


def existing_file(r):
    cell, m = r.get("cell", "adult/sex"), r["method"]
    if cell == "hmda/race":
        return "tpr59_scores/hmda_noise_s8.npz"
    if "noise" in m:
        s = float(m.replace("σ", "sigma").split("=")[-1])
        return f"tpr_ext_scores/hr_noise_s{s:g}.npz" if s in (4.0, 8.0) else f"tpr59_scores/adult_noise_s{s:g}.npz"
    fam, rank = m.rsplit(" r=", 1)
    return f"tpr59_scores/adult_{'mmd' if 'MMD' in fam else 'hsic'}_r{rank}.npz"


def expansion_file(cell, m):
    slug = cell.replace("/", "_") + "_" + m.replace(" ", "").replace("σ=", "s").replace("=", "")
    rel = f"tpr59_scores/exp59_{slug}.npz"
    return rel if (AN / rel).exists() else f"tpr_ext_scores/exp_{cell.replace('/', '_')}.npz"


configs = []
for r in hr["master"]:
    configs.append({"key": f"{r['experiment']}|{r.get('cell', 'adult/sex')}|{r['method']}", "rel": existing_file(r),
                    "stored": {"XGB": r["xgb_auc"], "MLP": r["mlp_auc"]}})
er_by = {(r["cell"], r["method"]): r for r in er["master"]}
for r in pf["rows"]:
    if not r.get("approved_at_rest"):
        continue
    e = er_by[(r["cell"], r["method"])]
    configs.append({"key": f"expansion|{r['cell']}|{r['method']}", "rel": expansion_file(r["cell"], r["method"]),
                    "stored": {"XGB": e["xgb_auc"], "MLP": e["mlp_auc"], "LoRA": e["lora_auc"]}})
assert len(configs) == 67, len(configs)

inv_check, specs = {}, {"paper_rule_xgb_mlp": [], "with_lora_where_stored": []}
for c in configs:
    p = AN / c["rel"]
    member = "durable-guarantees/analysis/" + c["rel"]
    exp = inv.get(member, {}).get("sha256")
    got = __import__("hashlib").sha256(p.read_bytes()).hexdigest() if p.exists() else None
    inv_check[member] = {"inventory_sha256": exp, "local_sha256": got, "match": exp is not None and exp == got}
    has_lora = p.exists() and any(k.startswith("LoRA_") for k in np.load(p).files)
    base = {"key": c["key"], "file": str(p), "file_sha256": exp}
    specs["paper_rule_xgb_mlp"].append({**base, "suite": ["XGB", "MLP"],
                                        "stored": {k: c["stored"][k] for k in ("XGB", "MLP")}})
    suite = ["XGB", "MLP", "LoRA"] if has_lora else ["XGB", "MLP"]
    specs["with_lora_where_stored"].append({**base, "suite": suite,
                                            "stored": {k: c["stored"][k] for k in suite if k in c["stored"]}})

summary = {"durable_guarantees_head": dg_head, "n_configs": len(configs),
           "n_distinct_files": len({c["rel"] for c in configs}),
           "inventory_check": {"n_files": len(inv_check), "n_match": sum(v["match"] for v in inv_check.values()),
                               "mismatch_or_missing": {k: v for k, v in inv_check.items() if not v["match"]}}}
for name, cl in specs.items():
    sp = {"bars": [0.52, 0.55, 0.60], "historical_min_support": 1, "supported_min_support": 100, "configs": cl}
    spath = HERE / f"aaai67_spec_{name}.json"
    spath.write_text(json.dumps(sp, indent=1))
    out = HERE / f"recount_aaai67_{name}.json"
    subprocess.run([sys.executable, "-m", "stored_model_eval", "--out", str(out), "recount", "probs", "--spec",
                    str(spath)], cwd=WT, check=True, capture_output=True)
    d = json.load(open(out))
    summary[name] = {k: d[k] for k in ("n_recounted", "n_pending", "problems", "fail_counts_recomputed",
                                       "fail_counts_from_stored_values", "n_distinct_measurements",
                                       "fail_counts_distinct_measurements", "shared_measurements",
                                       "max_abs_delta_vs_stored", "near_bar")}
(HERE / "recount_aaai67_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
