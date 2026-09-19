#!/usr/bin/env python3
"""Reproduce SELECTION_RESCORING_CHECK.json and DISPLACEMENT_CHECK.json from Study 4's stored fit records."""
import json, glob, collections, statistics, sys, os
import numpy as np, torch
EV = sys.argv[1] if len(sys.argv) > 1 else "/Users/nathansamson/PCRL-terminal-1-adversarial/results/pcrl_direct_adversarial_v1"
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
paths = sorted(glob.glob(os.path.join(EV, "seed_*/fits/*/fit_record.json")))
recs = [json.load(open(p)) for p in paths]
res = {}
for g in (0, 0.01, 0.1, 1):
    st = collections.Counter(); ch = 0
    for d in recs:
        ms = d["monitor_scores"]
        sc = [m["source"] + g * m["distortion"] + d["beta"] * m["penalty"] for m in ms]
        i = int(np.argmin(sc)); st[ms[i]["step"]] += 1; ch += i != d["selected_index"]
    res[str(g)] = {"step_distribution": dict(sorted(st.items())), "changed_vs_published": ch}
bad = sum(1 for d in recs for m in d["monitor_scores"]
          if abs(m["utility"] - (m["source"] + m["distortion"])) > 1e-6
          or abs(m["monitor_score"] - (m["utility"] + d["beta"] * m["penalty"])) > 1e-6)
mv = [d for d in recs if d["selected_step"] > 0]
dec = {k: float(np.mean([d["monitor_scores"][d["selected_index"]][k] - d["monitor_scores"][0][k] for d in mv])) for k in ("source", "distortion")}
dec["beta_pen"] = float(np.mean([d["beta"] * (d["monitor_scores"][d["selected_index"]]["penalty"] - d["monitor_scores"][0]["penalty"]) for d in mv]))
json.dump({"gamma_rescoring": res, "identity_failures": bad, "moved_n": len(mv), "moved_decomposition_mean": dec,
           "note": "counterfactual selection on fixed gamma=1 trajectories; not retraining"},
          open(os.path.join(OUT, "SELECTION_RESCORING_CHECK.json"), "w"), indent=1)
disp = []
for p, d in zip(paths, recs):
    if d["selected_step"] != 0: continue
    st = torch.load(p.replace("fit_record.json", "checkpoints.pt"), map_location="cpu", weights_only=False)["checkpoint_states"]
    vec = lambda s: torch.cat([v.flatten().double() for k, v in sorted(s.items()) if torch.is_tensor(v) and v.is_floating_point()])
    a, b = vec(st[0]), vec(st[-1]); disp.append(float((b - a).norm() / a.norm()))
json.dump({"step0_trajectories": len(disp), "moved": sum(x > 0 for x in disp), "median_relative_displacement": statistics.median(disp),
           "min": min(disp), "max": max(disp),
           "note": "final vs step-0 checkpoint, all floating tensors in checkpoint_states"},
          open(os.path.join(OUT, "DISPLACEMENT_CHECK.json"), "w"), indent=1)
print("ok", res["0"]["changed_vs_published"], len(disp))
