"""Cross-purpose rebuttal aggregator — 3-dataset version.

Reads ``results/v2_<dataset>_<tag>/results.json`` per dataset (produced by
``run_eval_multi.py``) and writes a rebuttal comparison artifact set:

  results/rebuttal/cross_purpose/HEADLINE.txt
  results/rebuttal/cross_purpose/comparison.json
  results/rebuttal/cross_purpose/comparison_table.tex
  results/rebuttal/cross_purpose/PAPER_PASTE.md

The aggregator produces the *two* headline metrics the rebuttal answer to R3
requires:
  1. per-pair strict-R² compliance (count of pairs with R²(h_p, A) ≤ 0.05)
  2. cross-purpose attack-flag count (count of (arch, attr) cells with mean
     attack accuracy > majority + 1pp on h_concat) — the metric §5.5 reports
     as 22-26/33 in the submitted paper.

Optional comparison columns: pass ``--baseline-dataset <ds> <results.json>``
multiple times to compare against an erase-layer-only baseline run.

Usage:
  python scripts/crosspurp/build_report_multi.py \\
      --new adult    results/v2_adult_CROSSPURP_ERASE/results.json \\
      --new hmda     results/v2_hmda_CROSSPURP_ERASE/results.json \\
      --new diabetes results/v2_diabetes_CROSSPURP_ERASE/results.json \\
      [--baseline adult results/v2_adult_ERASE_PILOT/results.json] [...]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "results" / "rebuttal" / "cross_purpose"


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def count_per_pair_pass(rec: dict) -> tuple[int, int]:
    """Count cells with R²(h_p, A) ≤ τ across all seeds in the run."""
    n_pass = 0
    n_total = 0
    for s in rec["per_seed"]:
        for v in s["per_pair"].values():
            n_total += 1
            if v["pass"]:
                n_pass += 1
    return n_pass, n_total


def count_attack_flags_per_cell(rec: dict) -> tuple[int, int, dict]:
    """Mean delta_pp over seeds, per (arch, attr); flag = mean Δ > 1.0pp.

    Returns (n_flagged, n_total, per_cell_matrix) where per_cell_matrix[arch][attr]
    = {mean_delta_pp, flagged}.
    """
    attrs = rec.get("attrs") or []
    archs = ["LR", "MLP", "XGB"]
    per_cell = {arch: {} for arch in archs}
    for arch in archs:
        for attr in attrs:
            deltas = []
            for s in rec["per_seed"]:
                cell = s.get("attack", {}).get(arch, {}).get(attr)
                if cell is not None:
                    deltas.append(cell["delta_pp"])
            if deltas:
                mean_d = sum(deltas) / len(deltas)
                per_cell[arch][attr] = {
                    "mean_delta_pp": mean_d, "flagged": mean_d > 1.0,
                    "n_seeds": len(deltas),
                }
    n_total = sum(len(d) for d in per_cell.values())
    n_flagged = sum(1 for arch in per_cell for a in per_cell[arch]
                    if per_cell[arch][a]["flagged"])
    return n_flagged, n_total, per_cell


def count_concat_r2_pass(rec: dict, tau_cross: float = 0.10) -> tuple[int, int, dict]:
    """h_concat R² compliance across all (seed, cross_attr) cells."""
    cross_attrs = rec.get("cross_attrs") or []
    per_attr = {}
    n_pass = 0
    n_total = 0
    for attr in cross_attrs:
        r2s = [s["concat"][attr]["r2_onehot_test"]
               for s in rec["per_seed"] if attr in s["concat"]]
        if r2s:
            mean_r2 = sum(r2s) / len(r2s)
            max_r2 = max(r2s)
            passed = sum(1 for r in r2s if r <= tau_cross)
            per_attr[attr] = {"mean": mean_r2, "max": max_r2, "n_pass": passed, "n": len(r2s)}
            n_total += len(r2s)
            n_pass += passed
    return n_pass, n_total, per_attr


def summarise(rec: dict) -> dict:
    pp, ppt = count_per_pair_pass(rec)
    af, aft, attack_matrix = count_attack_flags_per_cell(rec)
    cp, cpt, concat_per_attr = count_concat_r2_pass(rec)
    return {
        "dataset": rec["dataset"], "tag": rec["tag"], "n_seeds": len(rec["per_seed"]),
        "per_pair_pass": pp, "per_pair_total": ppt,
        "attack_flagged": af, "attack_total": aft,
        "attack_per_cell": attack_matrix,
        "concat_r2_pass": cp, "concat_r2_total": cpt,
        "concat_per_attr": concat_per_attr,
    }


def fmt_count(n_pass: int, n_total: int) -> str:
    return f"{n_pass}/{n_total}" if n_total > 0 else "—"


def write_headline(new_summaries: dict, base_summaries: dict, out_path: Path) -> None:
    lines = ["CROSS-PURPOSE REBUTTAL — 3-dataset summary",
             "=" * 60, ""]
    total_pp_new = total_ppt_new = 0
    total_af_new = total_aft_new = 0
    total_pp_base = total_ppt_base = 0
    total_af_base = total_aft_base = 0
    for ds in ["adult", "hmda", "diabetes"]:
        n = new_summaries.get(ds)
        b = base_summaries.get(ds)
        lines.append(f"{ds.upper()}")
        if n:
            lines.append(f"  per-pair R² ≤ 0.05      : {fmt_count(n['per_pair_pass'], n['per_pair_total'])}"
                         + (f"   (baseline {fmt_count(b['per_pair_pass'], b['per_pair_total'])})" if b else ""))
            lines.append(f"  attack flags > +1pp     : {fmt_count(n['attack_flagged'], n['attack_total'])}"
                         + (f"   (baseline {fmt_count(b['attack_flagged'], b['attack_total'])})" if b else ""))
            lines.append(f"  h_concat R² ≤ 0.10      : {fmt_count(n['concat_r2_pass'], n['concat_r2_total'])}")
            total_pp_new += n["per_pair_pass"]; total_ppt_new += n["per_pair_total"]
            total_af_new += n["attack_flagged"]; total_aft_new += n["attack_total"]
            if b:
                total_pp_base += b["per_pair_pass"]; total_ppt_base += b["per_pair_total"]
                total_af_base += b["attack_flagged"]; total_aft_base += b["attack_total"]
        else:
            lines.append("  (no results)")
        lines.append("")
    lines.append("TOTAL")
    lines.append(f"  per-pair R² ≤ 0.05      : {fmt_count(total_pp_new, total_ppt_new)}"
                 + (f"   (baseline {fmt_count(total_pp_base, total_ppt_base)})" if total_ppt_base else ""))
    lines.append(f"  attack flags > +1pp     : {fmt_count(total_af_new, total_aft_new)}"
                 + (f"   (baseline {fmt_count(total_af_base, total_aft_base)})" if total_aft_base else ""))
    lines.append("")
    lines.append("(per-pair total = 3 seeds × pairs; attack total = 3 archs × |attrs| per dataset, summed)")
    out_path.write_text("\n".join(lines) + "\n")


def write_attack_matrix_md(new_summaries: dict, out_path: Path) -> None:
    """Per-cell attack flag matrix — one block per dataset."""
    lines = ["# Per-cell cross-purpose attack matrix",
             "",
             "FLAG = mean Δ_pp over seeds > +1.0pp on h_concat. Cells matching the §5.4 / §5.5 audit pool.",
             ""]
    for ds in ["adult", "hmda", "diabetes"]:
        n = new_summaries.get(ds)
        if not n:
            continue
        lines.append(f"## {ds.upper()} (tag={n['tag']}, n_seeds={n['n_seeds']})")
        archs = sorted(n["attack_per_cell"].keys())
        attrs = sorted({a for arch in n["attack_per_cell"] for a in n["attack_per_cell"][arch]})
        header = "| attr | " + " | ".join(archs) + " |"
        sep = "|" + "|".join(["---"] * (1 + len(archs))) + "|"
        lines += [header, sep]
        for attr in attrs:
            cells = []
            for arch in archs:
                c = n["attack_per_cell"][arch].get(attr)
                if c is None:
                    cells.append("—")
                else:
                    flag = "**FLAG**" if c["flagged"] else "ok"
                    cells.append(f"{c['mean_delta_pp']:+.2f}pp ({flag})")
            lines.append(f"| {attr} | " + " | ".join(cells) + " |")
        lines.append("")
    out_path.write_text("\n".join(lines))


def write_tex_table(new_summaries: dict, base_summaries: dict, out_path: Path) -> None:
    cols = "lrrr" if not base_summaries else "lrrrr"
    rows = []
    if base_summaries:
        rows.append(r"Dataset & per-pair R² ≤ 0.05 (new / baseline) & attack flags > 1pp (new / baseline) & $h_{\mathrm{concat}}$ R² ≤ 0.10 \\")
    else:
        rows.append(r"Dataset & per-pair R² ≤ 0.05 & attack flags > 1pp & $h_{\mathrm{concat}}$ R² ≤ 0.10 \\")
    for ds in ["adult", "hmda", "diabetes"]:
        n = new_summaries.get(ds)
        b = base_summaries.get(ds)
        if not n:
            continue
        pp_new = fmt_count(n["per_pair_pass"], n["per_pair_total"])
        af_new = fmt_count(n["attack_flagged"], n["attack_total"])
        cp_new = fmt_count(n["concat_r2_pass"], n["concat_r2_total"])
        if b:
            pp_str = f"{pp_new} / {fmt_count(b['per_pair_pass'], b['per_pair_total'])}"
            af_str = f"{af_new} / {fmt_count(b['attack_flagged'], b['attack_total'])}"
        else:
            pp_str = pp_new; af_str = af_new
        rows.append(f"{ds.capitalize()} & {pp_str} & {af_str} & {cp_new} \\\\")
    tex = "\n".join([
        r"\begin{tabular}{" + cols + "}",
        r"\toprule",
        rows[0],
        r"\midrule",
        *rows[1:],
        r"\bottomrule",
        r"\end{tabular}",
    ])
    out_path.write_text(tex + "\n")


def write_paper_paste(new_summaries: dict, base_summaries: dict, out_path: Path) -> None:
    def totals(summaries):
        return (sum(s["per_pair_pass"] for s in summaries.values()),
                sum(s["per_pair_total"] for s in summaries.values()),
                sum(s["attack_flagged"] for s in summaries.values()),
                sum(s["attack_total"] for s in summaries.values()))
    t_pp_n, t_ppt_n, t_af_n, t_aft_n = totals(new_summaries)
    t_pp_b, t_ppt_b, t_af_b, t_aft_b = (totals(base_summaries) if base_summaries else (0, 0, 0, 0))
    has_base = bool(base_summaries)
    md = ["# PAPER_PASTE — cross-purpose constraint at training time (§5.5 extension to 3 datasets)",
          "",
          "## Drop-in §5.5 paragraph (template — confirm honest framing applies)",
          ""]
    para = (
        "We extended the §5.5 cross-purpose linear-$R^2$ constraint on $h_{\\rm concat}$ from "
        "Adult-only to all three tabular datasets, retraining under the erase-layer "
        "architecture (the rebuttal headline). With three cross-purpose duals per dataset "
        "at threshold $\\tau_{\\rm concat}=0.10$ (Adult: race/sex/age\\_group; HMDA: "
        "ethnicity/race/sex; Diabetes: race/gender/age\\_bucket — the last triggers "
        "per-class OvR with 10 duals on the high-cardinality age\\_bucket), "
        f"per-pair strict-$R^2$ compliance is ${t_pp_n}/{t_ppt_n}$"
        + (f" versus the erase-layer-only baseline ${t_pp_b}/{t_ppt_b}$" if has_base else "")
        + f", and the cross-purpose attack flag count (Criterion A from §5.4: mean attack accuracy "
        f"over three seeds exceeds majority + 1pp) moves to ${t_af_n}/{t_aft_n}$"
        + (f" versus baseline ${t_af_b}/{t_aft_b}$" if has_base else "")
        + " (vs $22\\text{--}26/33$ at the submission)."
    )
    md.append(para)
    md += ["", "## Per-dataset table", "",
           "| Dataset | per-pair R² ≤ 0.05 | attack flags > 1pp | h_concat R² ≤ 0.10 |",
           "|---|---|---|---|"]
    for ds in ["adult", "hmda", "diabetes"]:
        n = new_summaries.get(ds)
        if not n: continue
        b = base_summaries.get(ds)
        pp_new = fmt_count(n["per_pair_pass"], n["per_pair_total"])
        af_new = fmt_count(n["attack_flagged"], n["attack_total"])
        cp_new = fmt_count(n["concat_r2_pass"], n["concat_r2_total"])
        pp_cell = f"{pp_new} (base {fmt_count(b['per_pair_pass'], b['per_pair_total'])})" if b else pp_new
        af_cell = f"{af_new} (base {fmt_count(b['attack_flagged'], b['attack_total'])})" if b else af_new
        md.append(f"| {ds.capitalize()} | {pp_cell} | {af_cell} | {cp_new} |")
    md.append("")
    md.append("## Honest framing checklist")
    md.append("")
    md.append("- Per-pair cells lost vs erase-layer baseline: report exact count and which cells.")
    md.append("- Attack flag count moved: report Δ in both directions (cells gained/lost).")
    md.append("- High-K duals (Diabetes age_bucket): report lambda_saturation if any.")
    md.append("- Per-attribute h_concat R² pass: separate the attrs the constraint binds on from the ones it doesn't.")
    out_path.write_text("\n".join(md) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", nargs=2, action="append", metavar=("DATASET", "RESULTS_JSON"),
                    required=True, help="New (erase + cross-purpose) results JSON. Repeatable.")
    ap.add_argument("--baseline", nargs=2, action="append", metavar=("DATASET", "RESULTS_JSON"),
                    default=[], help="Optional erase-only baseline. Repeatable.")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    new_summaries = {}
    for ds, p in args.new:
        rec = load(Path(p))
        new_summaries[ds] = summarise(rec)
    base_summaries = {}
    for ds, p in args.baseline:
        rec = load(Path(p))
        base_summaries[ds] = summarise(rec)

    write_headline(new_summaries, base_summaries, out_dir / "HEADLINE.txt")
    write_attack_matrix_md(new_summaries, out_dir / "attack_matrix.md")
    write_tex_table(new_summaries, base_summaries, out_dir / "comparison_table.tex")
    write_paper_paste(new_summaries, base_summaries, out_dir / "PAPER_PASTE.md")
    with open(out_dir / "comparison.json", "w") as f:
        json.dump({"new": new_summaries, "baseline": base_summaries}, f, indent=2)
    print(f"Wrote 5 artifacts under {out_dir}")


if __name__ == "__main__":
    main()
