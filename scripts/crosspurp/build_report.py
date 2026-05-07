"""Build CROSSPURP_CONSTRAINT report: HEADLINE.txt, comparison_table.tex,
PAPER_PASTE.md."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

RES = ROOT / "results" / "v2_adult_CROSSPURP" / "results.json"
if not RES.exists():
    print("no results.json, exiting")
    sys.exit(0)

data = json.load(open(RES))
summary = data.get("summary", {})

# Baseline numbers (PCRL R5 / §5.4 — known from prior work)
baseline = {
    "per_pair_pass_audit": "23/24",
    "attack_flagged_above_1pp": "26/33",  # Criterion A
    "task_acc_income": 0.787,
}

out_dir = RES.parent
hl = []
hl.append("CROSSPURP_CONSTRAINT — Adult only, 3 seeds, 75 epochs")
hl.append("=" * 60)
hl.append(f"per-pair R² <= 0.05 (auditor):  {summary.get('per_pair_pass', '?/?')}")
hl.append(f"   baseline R5 audit:             {baseline['per_pair_pass_audit']}")
hl.append("")
hl.append("h_concat R² (test) per cross-purpose attr (mean over 3 seeds):")
for attr, m in summary.get("concat_r2_mean", {}).items():
    hl.append(f"   {attr:<15} mean={m['mean_r2']:.4f} max={m['max_r2']:.4f}  "
              f"pass(<=0.10): {m['n_pass']}/{m['n']}")
hl.append("")
hl.append(f"Cross-purpose attack flagged above 1pp:")
hl.append(f"   CROSSPURP_CONSTRAINT: {summary.get('attack_flagged_above_1pp', '?/?')}")
hl.append(f"   baseline R5/§5.4:     {baseline['attack_flagged_above_1pp']} (Criterion A)")
(out_dir / "HEADLINE.txt").write_text("\n".join(hl) + "\n")
print("\n".join(hl))

# LaTeX comparison
tex = []
tex.append("% CROSSPURP_CONSTRAINT vs baseline R5 — Adult, 3 seeds, 75 epochs.")
tex.append("\\begin{tabular}{lcc}")
tex.append("\\toprule")
tex.append("Metric & Baseline R5 & + cross-purpose constraint \\\\")
tex.append("\\midrule")
tex.append(f"Per-pair $R^2 \\le 0.05$ pass & {baseline['per_pair_pass_audit']} & {summary.get('per_pair_pass','?/?')} \\\\")
for attr, m in summary.get("concat_r2_mean", {}).items():
    tex.append(f"$h_{{\\rm concat}}$ $R^2$({attr.replace('_','\\_')}) mean & --- & "
               f"{m['mean_r2']:.4f} ({m['n_pass']}/{m['n']} $\\le$0.10) \\\\")
tex.append(f"Cross-purpose attack flagged ($>$1pp) & {baseline['attack_flagged_above_1pp']} & "
           f"{summary.get('attack_flagged_above_1pp','?/?')} \\\\")
tex.append("\\bottomrule")
tex.append("\\end{tabular}")
(out_dir / "comparison_table.tex").write_text("\n".join(tex) + "\n")

# Decide outcome class
new_pp = summary.get("per_pair_pass", "0/0")
new_attack = summary.get("attack_flagged_above_1pp", "0/0")
old_attack = baseline["attack_flagged_above_1pp"]
new_pp_n = int(new_pp.split("/")[0]) if "/" in new_pp else 0
new_pp_d = int(new_pp.split("/")[1]) if "/" in new_pp else 1
new_at_n = int(new_attack.split("/")[0]) if "/" in new_attack else 0
old_at_n = int(old_attack.split("/")[0])
attack_drop = old_at_n - new_at_n
per_pair_pass_rate = new_pp_n / max(new_pp_d, 1)

# Paper paste
pp = []
pp.append("# PAPER_PASTE — Cross-purpose constraint at training time (§5.4 extension)")
pp.append("")
pp.append("## Drop-in §5.4 paragraph (flowing prose)")
pp.append("")
pp.append(
    f"To test whether the cross-purpose leakage observed in §5.4 is addressable at "
    f"training time rather than only at audit time, we extended the proxy-Lagrangian "
    f"constraint set with three additional duals on $\\text{{linear-}}R^2(h_{{\\rm concat}}, A) "
    f"\\le 0.10$ for $A \\in \\{{\\text{{race}}, \\text{{sex}}, \\text{{age\\_group}}\\}}$, where "
    f"$h_{{\\rm concat}} = [h_{{p_1}}; h_{{p_2}}; h_{{p_3}}]$ is the concatenation of the "
    f"three per-purpose representations. We retrained Adult from scratch with the augmented "
    f"constraint set (75 epochs, three seeds, all other hyperparameters identical to "
    f"Round 5). Per-pair compliance moved from {baseline['per_pair_pass_audit']} to "
    f"{new_pp} on the auditor R² metric; the cross-purpose attack flag count under "
    f"Criterion A (concat acc > majority + 1pp) moved from {old_attack} to {new_attack}."
)
pp.append("")
pp.append("## Per-cell $h_{\\rm concat}$ R² (test split, mean over 3 seeds)")
pp.append("")
pp.append("| Attribute | $h_{concat}$ R² mean | $h_{concat}$ R² max | Pass $\\le$ 0.10 |")
pp.append("|-----------|---------------------:|--------------------:|:----------------|")
for attr, m in summary.get("concat_r2_mean", {}).items():
    pp.append(f"| {attr} | {m['mean_r2']:.4f} | {m['max_r2']:.4f} | {m['n_pass']}/{m['n']} |")
pp.append("")
pp.append("## Honest framing")
pp.append("")
if attack_drop >= 5 and per_pair_pass_rate >= 0.85:
    pp.append(
        "**Outcome (a): cross-purpose constraint reduces concat leakage without "
        "breaking per-purpose compliance.** The training-time constraint absorbs a "
        "meaningful fraction of the concatenation leakage that the post-hoc auditor "
        "previously found, while preserving the per-pair strict-pass rate. "
        f"Concretely: attack flag count fell {attack_drop} triples (from {old_attack} "
        f"to {new_attack}); per-pair pass rate stayed at {per_pair_pass_rate*100:.0f}%. "
        "Recommend for multi-purpose deployment when cross-purpose attack resistance "
        "matters."
    )
elif attack_drop >= 5 and per_pair_pass_rate < 0.85:
    pp.append(
        "**Outcome (b): cross-purpose constraint reduces concat leakage at the cost "
        "of per-purpose compliance.** The constraint suppresses concatenation-level "
        "predictability but degrades the per-pair strict-pass rate, indicating a real "
        "tradeoff between per-purpose and cross-purpose objectives at the scale of "
        f"these datasets. Attack flags fell {attack_drop} triples (from {old_attack} "
        f"to {new_attack}) while per-pair pass dropped from "
        f"{baseline['per_pair_pass_audit']} to {new_pp}."
    )
else:
    pp.append(
        "**Outcome (c): cross-purpose constraint has limited effect on concat leakage.** "
        "The training-time linear-$R^2$ constraint on $h_{\\rm concat}$ does not "
        "substantially reduce the post-hoc cross-purpose attack flag count "
        f"({old_attack} → {new_attack}), suggesting the concatenation leakage reflects "
        "nonlinear residual structure that linear constraints cannot reach. "
        f"Per-pair pass rate was {new_pp} (baseline {baseline['per_pair_pass_audit']}). "
        "Addressing cross-purpose leakage will likely require either nonlinear "
        "constraints (kernel-HSIC on $h_{\\rm concat}$, vCLUB MI bound) or attack-time "
        "defenses."
    )
pp.append("")
pp.append("## Files")
pp.append("- `results.json` — per-seed per-pair R², h_concat R², attack accuracies")
pp.append("- `comparison_table.tex` — paper-ready 3-column table")
pp.append("- `HEADLINE.txt` — 5-line summary")

(out_dir / "PAPER_PASTE.md").write_text("\n".join(pp) + "\n")
print("Wrote PAPER_PASTE.md")
