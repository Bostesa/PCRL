#!/usr/bin/env python3
"""Compensator-class analysis for §5.6 (BIOS BERT block-0 heads).

Source: /private/tmp/stage3_results/mech_interp_full_results.json
Output: results/compensator_class_analysis/

Question (per user's task brief):
  §5.6 names (0,6) as a "single counter-balance head writing opposite to the
  dominant (0,7), (0,10) gender pathway". Are there OTHER compensator heads in
  block 0 with the same opposing-direction signature?

Two independent criteria, both required for "directional compensator":
  C1 (signed OV projection): proj_along_LEACE_dir < 0 AND |proj| > 0.15
                             (i.e., head's mean OV write opposes the gender axis)
  C2 (ablation degradation): adding this head to the ablation set degrades
                             probe_acc by >= 5pp vs. the prior set.

The ablation table only contains top_2 / top_3 / top_5 ablations, so C2 can
be evaluated for (0,6) (added in top_3) and the union (0,4)+(1,0) (added in
top_5), but not for any other individual head — that limits how strong a
"class" claim can be made.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path("/private/tmp/stage3_results/mech_interp_full_results.json")
OUT = Path(__file__).resolve().parent.parent / "results" / "compensator_class_analysis"
OUT.mkdir(parents=True, exist_ok=True)

DOMINANT = {(0, 7), (0, 10)}
KNOWN_C = (0, 6)

PROJ_THRESH = 0.15
ABLATION_DEGRADE_PP = 5.0  # pp


def main():
    data = json.loads(SRC.read_text())["stage3"]
    per_head = data["per_head"]
    rd = data["residual_decomposition"]
    abl = data["ablation"]
    base = data["baseline_probe"]

    # ── per-head signed projections, all 144 cells ────────────────────
    rows = []
    for L in range(12):
        for h in range(12):
            key = f"L{L}_h{h}"
            ph = per_head[key]
            r = rd[key]
            rows.append({
                "block": L, "head": h,
                "delta_logit": ph["delta_logit"],
                "norm_delta": ph["norm_delta"],
                "flip_rate": ph["flip_rate"],
                "population_R2": ph["population_R2"],
                "ov_norm": r["ov_norm"],
                "proj_LEACE": r["proj_along_LEACE_dir"],
                "proj_classmean": r["proj_along_classmean_dir"],
            })

    block0 = [x for x in rows if x["block"] == 0]
    n_neg_proj_LEACE_global = sum(x["proj_LEACE"] < 0 for x in rows)
    n_neg_proj_class_global = sum(x["proj_classmean"] < 0 for x in rows)
    n_neg_dlgt_global = sum(x["delta_logit"] < 0 for x in rows)

    # ── ablation-derived per-head contribution ────────────────────────
    # adding (0,6) to {(0,7),(0,10)}: probe_acc 0.319 → 0.224  (Δ = −0.095)
    # adding (0,4)+(1,0) to {(0,7),(0,10),(0,6)}: 0.224 → 0.209 (Δ = −0.015 over 2 heads)
    base_top2_acc = abl["top_2"]["probe_acc_ablated"]
    base_top3_acc = abl["top_3"]["probe_acc_ablated"]
    base_top5_acc = abl["top_5"]["probe_acc_ablated"]

    delta_06_acc = base_top3_acc - base_top2_acc
    delta_04_10_acc_avg = (base_top5_acc - base_top3_acc) / 2

    base_top2_lgt = abl["top_2"]["mean_logit_ablated"]
    base_top3_lgt = abl["top_3"]["mean_logit_ablated"]
    base_top5_lgt = abl["top_5"]["mean_logit_ablated"]

    # ── Compensator classification per criteria ───────────────────────
    # C1 (directional): sign opposite to (0,7)/(0,10) AND |proj| > thresh.
    sign_dom = +1  # both (0,7) and (0,10) have positive proj
    for x in rows:
        c1 = (x["proj_LEACE"] * sign_dom < 0) and (abs(x["proj_LEACE"]) > PROJ_THRESH)
        c1_class = (x["proj_classmean"] * sign_dom < 0) and (
            abs(x["proj_classmean"]) > PROJ_THRESH
        )
        cls = "neutral"
        if (x["block"], x["head"]) in DOMINANT:
            cls = "dominant_writer"
        elif c1 or c1_class:
            cls = "directional_compensator"
        elif x["proj_LEACE"] > PROJ_THRESH:
            cls = "co_writer_block0" if x["block"] == 0 else "co_writer"
        x["criterion_C1_directional_compensator"] = bool(c1 or c1_class)
        x["classification"] = cls

    # C2 evidence is only available for (0,6).
    c2_evidence_06 = {
        "head": list(KNOWN_C),
        "from_set": "top_2 = [(0,7),(0,10)]",
        "to_set": "top_3 = [(0,7),(0,10),(0,6)]",
        "probe_acc_before": base_top2_acc,
        "probe_acc_after": base_top3_acc,
        "delta_pp": (base_top3_acc - base_top2_acc) * 100.0,
        "satisfies_C2": (base_top2_acc - base_top3_acc) * 100.0 >= ABLATION_DEGRADE_PP,
        "interpretation": (
            "Adding (0,6) to the ablation set DROPS probe accuracy from "
            f"{base_top2_acc*100:.1f}% to {base_top3_acc*100:.1f}% "
            f"({(base_top3_acc-base_top2_acc)*100:+.1f} pp). C2 satisfied."
        ),
    }
    c2_evidence_04_10 = {
        "heads_added": [[0, 4], [1, 0]],
        "from_set": "top_3",
        "to_set": "top_5",
        "probe_acc_before": base_top3_acc,
        "probe_acc_after": base_top5_acc,
        "delta_pp_avg": (base_top5_acc - base_top3_acc) * 100.0 / 2,
        "satisfies_C2_per_head_avg": (
            (base_top3_acc - base_top5_acc) * 100.0 / 2 >= ABLATION_DEGRADE_PP
        ),
        "interpretation": (
            "Adding (0,4) and (1,0) drops probe acc by "
            f"{(base_top5_acc-base_top3_acc)*100:+.1f} pp total ("
            f"{(base_top5_acc-base_top3_acc)*100/2:+.1f} pp/head). C2 NOT satisfied "
            "at the 5pp/head threshold."
        ),
    }

    # ── headline summary ──────────────────────────────────────────────
    block0_neg_proj = [x for x in block0 if x["proj_LEACE"] < 0]
    block0_compensator_C1 = [x for x in block0 if x["criterion_C1_directional_compensator"]]

    # (0,6) values for summary
    h06 = next(x for x in block0 if x["head"] == 6)
    h07 = next(x for x in block0 if x["head"] == 7)
    h10 = next(x for x in block0 if x["head"] == 10)

    summary = {
        "source_json": str(SRC),
        "n_cells_total": len(rows),
        "n_cells_with_negative_proj_LEACE": n_neg_proj_LEACE_global,
        "n_cells_with_negative_proj_classmean": n_neg_proj_class_global,
        "n_cells_with_negative_delta_logit": n_neg_dlgt_global,
        "block_0": {
            "n_heads": 12,
            "n_with_negative_proj_LEACE": len(block0_neg_proj),
            "n_satisfying_C1_directional_compensator": len(block0_compensator_C1),
            "min_proj_LEACE": min(x["proj_LEACE"] for x in block0),
            "max_proj_LEACE": max(x["proj_LEACE"] for x in block0),
            "mean_proj_LEACE": float(np.mean([x["proj_LEACE"] for x in block0])),
        },
        "key_heads": {
            "(0,6)": {
                "proj_LEACE": h06["proj_LEACE"],
                "proj_classmean": h06["proj_classmean"],
                "delta_logit": h06["delta_logit"],
                "is_largest_block0_proj_LEACE": (
                    h06["proj_LEACE"] == max(x["proj_LEACE"] for x in block0)
                ),
                "satisfies_C1_directional_compensator":
                    h06["criterion_C1_directional_compensator"],
                "classification": h06["classification"],
            },
            "(0,7)": {
                "proj_LEACE": h07["proj_LEACE"],
                "proj_classmean": h07["proj_classmean"],
                "delta_logit": h07["delta_logit"],
                "classification": h07["classification"],
            },
            "(0,10)": {
                "proj_LEACE": h10["proj_LEACE"],
                "proj_classmean": h10["proj_classmean"],
                "delta_logit": h10["delta_logit"],
                "classification": h10["classification"],
            },
        },
        "C2_evidence_for_06": c2_evidence_06,
        "C2_evidence_for_04_10": c2_evidence_04_10,
        "verdict": (
            "WALK BACK §5.6 directional-compensator framing for (0,6). "
            "ALL 12 block-0 heads — and ALL 144 cells globally — have POSITIVE "
            "OV projections along both LEACE and class-mean gender directions. "
            "(0,6) has the LARGEST positive proj_LEACE in block 0 (+0.388 vs. "
            "(0,7)=+0.054 and (0,10)=+0.105), so it writes in the SAME direction "
            "as the dominant heads, not opposite. The ablation-degradation signature "
            "for (0,6) (probe_acc 31.9%→22.4% when added to top-2 ablation) is real "
            "but does NOT establish opposing directionality; it is consistent with "
            "(0,6) being a 'pronoun-swap-robust' gender writer (carries gender from "
            "non-pronoun tokens) whose ablation removes a salvage pathway. "
            "No directional compensator class exists in this 144-cell sweep."
        ),
    }

    out = {
        "summary": summary,
        "per_head_144": rows,
        "ablation_table": {
            "baseline_corrupt_probe_acc": base["eval_corrupt_acc"],
            "baseline_clean_probe_acc": base["eval_clean_acc"],
            "top_2": abl["top_2"],
            "top_3": abl["top_3"],
            "top_5": abl["top_5"],
        },
    }
    (OUT / "compensator_analysis_results.json").write_text(json.dumps(out, indent=2))
    print(f"→ {OUT/'compensator_analysis_results.json'}")

    # ── plot: block-0 signed proj_LEACE bar chart ─────────────────────
    block0_sorted = sorted(block0, key=lambda x: x["proj_LEACE"], reverse=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    heads = [f"(0,{x['head']})" for x in block0_sorted]
    projs = [x["proj_LEACE"] for x in block0_sorted]
    colors = []
    for x in block0_sorted:
        if (x["block"], x["head"]) in DOMINANT:
            colors.append("#d62728")  # red — dominant writer
        elif (x["block"], x["head"]) == KNOWN_C:
            colors.append("#9467bd")  # purple — §5.6 named "compensator"
        else:
            colors.append("#7f7f7f")  # gray — other
    bars = ax.bar(heads, projs, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(PROJ_THRESH, color="green", linewidth=0.8, linestyle="--",
               label=f"|proj|={PROJ_THRESH} threshold")
    ax.axhline(-PROJ_THRESH, color="green", linewidth=0.8, linestyle="--")
    ax.set_ylabel(r"Signed $\langle\,W_O[h]\bar z_h,\;\hat g_{\rm LEACE}\rangle$")
    ax.set_xlabel("Block-0 head")
    ax.set_title(
        "Block-0 head OV writes along the LEACE rank-1 gender direction\n"
        "ALL 12 heads project positively — no directional compensators",
        fontsize=10,
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#d62728", label="dominant writer (0,7),(0,10)"),
        plt.Rectangle((0, 0), 1, 1, color="#9467bd", label="(0,6) — §5.6 'compensator'"),
        plt.Rectangle((0, 0), 1, 1, color="#7f7f7f", label="other block-0"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "block_0_writes_signed.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"→ {OUT/'block_0_writes_signed.pdf'}")

    # ── tex table ─────────────────────────────────────────────────────
    tex = [
        "% Auto-generated by scripts/compensator_class_analysis.py.",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"head & proj$_{\rm LEACE}$ & proj$_{\rm class}$ & "
        r"$\Delta_{\rm logit}$ & top-2 base & + this head ablated \\",
        r"\midrule",
    ]

    def row(L: int, h: int, abl_acc: float | None, note: str = ""):
        x = next(r for r in rows if r["block"] == L and r["head"] == h)
        a = f"{abl_acc*100:.1f}\\%" if abl_acc is not None else "--"
        return (
            f"({L},{h}) & {x['proj_LEACE']:+.3f} & {x['proj_classmean']:+.3f} "
            f"& {x['delta_logit']:+.3f} & "
            f"{base_top2_acc*100:.1f}\\% & {a} {note} \\\\"
        )

    # Dominant heads (skip ablation column — already in top-2 set)
    tex.append(row(0, 7, None, "[in top-2 set]"))
    tex.append(row(0, 10, None, "[in top-2 set]"))
    # (0,6) — known §5.6 'compensator'
    tex.append(row(0, 6, base_top3_acc, r"$\leftarrow$ §5.6 candidate"))
    # (0,4) added in top-5 — degradation per-head only
    tex.append(row(0, 4, base_top5_acc, r"[in top-5; per-head $\sim$0.7pp]"))
    tex.append(row(1, 0, base_top5_acc, r"[in top-5; per-head $\sim$0.7pp]"))
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    (OUT / "compensator_class_table.tex").write_text("\n".join(tex) + "\n")
    print(f"→ {OUT/'compensator_class_table.tex'}")

    # ── PAPER_PASTE.md ────────────────────────────────────────────────
    md = [
        "# PAPER_PASTE — block-0 compensator class analysis (§5.6)",
        "",
        "## Verdict (data-driven, brutally honest)",
        "",
        "**WALK BACK §5.6's directional-compensator framing for (0,6).** The ",
        "144-cell mech-interp sweep contains **zero** heads whose mean OV write ",
        "projects negatively along either the LEACE rank-1 gender direction or ",
        "the class-mean gender direction. (0,6) is **not** a counter-balance ",
        "head; it has the *largest* positive proj_LEACE in block 0 ",
        f"({h06['proj_LEACE']:+.3f}, vs. (0,7)={h07['proj_LEACE']:+.3f} and ",
        f"(0,10)={h10['proj_LEACE']:+.3f}), writing in the *same* direction as ",
        "the dominant heads.",
        "",
        "## What the ablation evidence does say",
        "",
        f"Adding (0,6) to the {{(0,7),(0,10)}} ablation set drops probe ",
        f"accuracy from {base_top2_acc*100:.1f}\\% (top-2) to ",
        f"{base_top3_acc*100:.1f}\\% (top-3) — a real ",
        f"{(base_top2_acc-base_top3_acc)*100:.1f} pp degradation. But this ",
        "degradation is not consistent with an *opposing-direction* writer; it ",
        "is consistent with (0,6) being a **pronoun-swap-robust gender carrier**: ",
        "after pronoun corruption, (0,6) still writes the original-gender ",
        "direction (because its OV pulls from non-pronoun tokens), so ablating ",
        "it on the corrupt run removes a salvage pathway for original-gender ",
        "information. The OV-projection sign (clean cache) and the ablation-",
        "degradation sign (corrupt cache) measure different things, and only ",
        "the former bears on the directional-compensator claim.",
        "",
        "## Recommended one-sentence framing for §5.6",
        "",
        "> Of the 12 block-0 heads, none meets a directional-compensator ",
        "> criterion (signed OV projection opposite to the dominant heads with ",
        f"> $|{{\\rm proj}}_{{\\rm LEACE}}|>{PROJ_THRESH}$); (0,6)'s ablation-degradation ",
        f"> signature ({base_top2_acc*100:.1f}\\%$\\to${base_top3_acc*100:.1f}\\%) ",
        "> arises not from opposing directionality but from its role as a ",
        "> pronoun-swap-robust gender carrier whose ablation removes a salvage ",
        "> pathway for original-gender information.",
        "",
        "## Don't claim",
        "",
        "- *\"Block-0 contains a class of compensator heads.\"* No directional ",
        "  compensators exist in the 144-cell sweep.",
        "- *\"(0,6) writes opposite to (0,7), (0,10).\"* (0,6) writes ",
        "  **most strongly in the same direction**; it is the most-aligned ",
        "  block-0 writer along the gender axis.",
        "",
        "## Why the data is unambiguous on this",
        "",
        f"- Min `proj_LEACE` over **all 144** (block, head) cells: ",
        f"  {min(x['proj_LEACE'] for x in rows):+.4f} (still positive).",
        f"- Min `proj_classmean` over **all 144** cells: ",
        f"  {min(x['proj_classmean'] for x in rows):+.4f} (still positive).",
        f"- Only **1 of 144** cells has negative $\\Delta_{{\\rm logit}}$: (1,8) ",
        "  at −0.072, but its OV projection is still positive (+0.066 ",
        "  classmean). (1,8) is a behavioral negative-$\\Delta$ head, not a ",
        "  directional compensator.",
        "",
        "## What's at stake",
        "",
        "If §5.6 currently relies on \"compensator class\" or \"counter-balance\" ",
        "language for (0,6), the language needs to be revised. The mechanistic ",
        "story that *is* supported is structurally weaker but more accurate: ",
        "block-0 attention has **diffuse positive contribution** to the gender ",
        "axis (12/12 heads positive, mean proj_LEACE = ",
        f"{summary['block_0']['mean_proj_LEACE']:+.4f}), with (0,7) and (0,10) ",
        "accounting for the largest *causal* effects (Δ_logit) but not the ",
        "largest *write magnitudes* (those are (0,6), (0,5), (0,4) — heads with ",
        "low flip rate but high OV-norm). The asymmetry between write-magnitude ",
        "ranking and causal-ranking is interesting in its own right and is ",
        "consistent with the Stage-3 'DIFFUSE MEDIATION' headline.",
        "",
        "## Numbers (block 0, sorted by proj_LEACE)",
        "",
        "| head | proj_LEACE | proj_class | Δ_logit | flip_rate | classification |",
        "|---|---|---|---|---|---|",
    ]
    for x in sorted(block0, key=lambda r: r["proj_LEACE"], reverse=True):
        md.append(
            f"| (0,{x['head']}) "
            f"| {x['proj_LEACE']:+.4f} "
            f"| {x['proj_classmean']:+.4f} "
            f"| {x['delta_logit']:+.4f} "
            f"| {x['flip_rate']:.3f} "
            f"| {x['classification']} |"
        )
    (OUT / "PAPER_PASTE.md").write_text("\n".join(md) + "\n")
    print(f"→ {OUT/'PAPER_PASTE.md'}")

    # ── HEADLINE.txt ──────────────────────────────────────────────────
    hl = [
        "Compensator-class analysis: NO directional compensators found.",
        f"  All 12/12 block-0 heads project POSITIVELY along gender axis "
        f"(min proj_LEACE = {summary['block_0']['min_proj_LEACE']:+.4f}).",
        f"  (0,6) has the LARGEST proj_LEACE (+{h06['proj_LEACE']:.3f}) — same "
        f"direction as (0,7),(0,10), not opposite.",
        f"  Ablation degradation 31.9%→22.4% on adding (0,6) is real but reflects "
        "a salvage-pathway role, not opposing directionality.",
        "Action: walk back §5.6 'counter-balance' framing for (0,6).",
    ]
    (OUT / "HEADLINE.txt").write_text("\n".join(hl) + "\n")
    print(f"→ {OUT/'HEADLINE.txt'}")


if __name__ == "__main__":
    main()
