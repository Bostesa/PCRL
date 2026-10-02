"""D4: recount LAFTR-hard-R^2 (branch laftr-hard-r2-2026-05-17) from the full
per-seed outputs found only on the external drive
(wt-PCRL-superpowers-laftr-hard-r2-2026-05-17.tar), and compare to the PCRL
rows it was set against (best.pt per_seed_results = 54/60, not 56/60).
Run: /opt/homebrew/bin/python3 D_laftr_hard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D_common import SCRATCH, dump, file_bytes, git_json, inputs  # noqa: E402

AR = "wt-PCRL-superpowers-laftr-hard-r2-2026-05-17.tar::laftr-hard-r2-2026-05-17/results/"
BASE = SCRATCH / "laftr-hard-r2-2026-05-17" / "results"
PSR = {"adult": "results/v2_adult_ROUND5/per_seed_results.json",
       "hmda": "results/v2_hmda_ROUND5/per_seed_results.json",
       "diabetes": "results/v2_diabetes_ROUND7/per_seed_results.json"}
DA = {"adult": "results/v2_adult_ROUND5/dominant_axis_audit.json",
      "hmda": "results/v2_hmda_ROUND5/dominant_axis_audit.json",
      "diabetes": "results/v2_diabetes_ROUND7/dominant_axis_audit.json"}

out: dict = {"per_dataset": {}}
tot = {"lh_strict": 0, "lh_adj": 0, "lh_clean": 0, "n": 0, "pcrl_best": 0, "pcrl_final": 0}
for ds in ("adult", "hmda", "diabetes"):
    rel = f"laftr_hard_r2_{ds}_LAFTR_HARD_R2/per_seed_results.json"
    d = json.loads(file_bytes(BASE / rel, AR + rel))
    psr = {b["seed"]: b for b in git_json("origin/main", PSR[ds])["per_seed"]}
    da = git_json("origin/main", DA[ds])["per_seed"]
    rows, r2s, accs = [], [], {}
    pb = pf = 0
    for blk in d["per_seed"]:
        s = blk["seed"]
        h = blk["per_purpose_health"]
        for ar in blk["attribute_results"]:
            hh = h[ar["purpose"]]
            clean = (ar["linear_r2"] < 0.05 and hh["per_dim_std_mean"] >= 0.5
                     and hh["effective_rank"] >= 2.0)
            rows.append({"seed": s, "purpose": ar["purpose"], "attr": ar["attribute"],
                         "r2": ar["linear_r2"], "delta": ar["delta"], "adj_pass": ar["adj_pass"],
                         "clean": clean})
            r2s.append(ar["linear_r2"])
        for t, v in blk["task_accuracies"].items():
            accs.setdefault(t, []).append((v, psr[s]["task_accuracies"][t]))
        pb += sum(a["linear_r2"] < 0.05 for a in psr[s]["attribute_results"])
        pf += sum(r["r2_onehot"] < 0.05 for r in da[str(s)]["rows"])
    n = len(rows)
    res = {
        "n_rows": n,
        "seeds": [b["seed"] for b in d["per_seed"]],
        "best_epochs": [b.get("best_epoch") for b in d["per_seed"]],
        "cotter_selection": [b.get("cotter_selection", {}).get("kind") if isinstance(
            b.get("cotter_selection"), dict) else b.get("cotter_selection") for b in d["per_seed"]],
        "strict_r2_lt_0.05": sum(r["r2"] < 0.05 for r in rows),
        "adj_pass(stored: R2<0.05 and delta<0.02)": sum(r["adj_pass"] for r in rows),
        "clean(R2<0.05, std>=0.5, eff_rank>=2)": sum(r["clean"] for r in rows),
        "mean_r2": round(float(np.mean(r2s)), 4),
        "r2_range": [round(min(r2s), 4), round(max(r2s), 4)],
        "max_delta_aud": round(max(r["delta"] for r in rows), 4),
        "task_acc_mean_laftrhard_vs_pcrl_bestpt": {
            t: [round(float(np.mean([a for a, _ in v])), 4), round(float(np.mean([b for _, b in v])), 4)]
            for t, v in accs.items()},
        "pcrl_comparator_bestpt_strict": pb, "pcrl_finalpt_strict": pf,
        "stored_summary_STATUS": d["summary"]["STATUS"],
    }
    out["per_dataset"][ds] = res
    tot["lh_strict"] += res["strict_r2_lt_0.05"]
    tot["lh_adj"] += res["adj_pass(stored: R2<0.05 and delta<0.02)"]
    tot["lh_clean"] += res["clean(R2<0.05, std>=0.5, eff_rank>=2)"]
    tot["n"] += n
    tot["pcrl_best"] += pb
    tot["pcrl_final"] += pf
out["totals"] = tot
hl = (BASE / "laftr_hard_r2" / "HEADLINE.txt")
file_bytes(hl, AR + "laftr_hard_r2/HEADLINE.txt")
out["headline_txt"] = hl.read_text().splitlines()[3:7]
out["inputs"] = inputs()
p = dump("D_laftr_hard.json", out)
print(json.dumps({k: v for k, v in out.items() if k != "inputs"}, indent=1, default=float))
print("wrote", p)
