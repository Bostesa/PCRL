"""Recovery-versus-utility plots, one figure per (dataset cell, release contract). Never one cross-dataset frontier.

    python plots.py --package-dir <results/combined_matched_removal_benchmark_v1> [--out-dir <package>/plots]

Reads only the aggregate report tables (RECOVERY.csv, UTILITY.csv, SEED_VARIATION.csv, SUPPORT_COVERAGE.csv,
SIGMA_STAR_SUMMARY.json). For each cell:
  C_rep                 : x = U2 accuracy (common LR probe on the released representation), y = C_rep NL macro AUC
  C_rep_plus_clean_out  : x = clean-output task accuracy (identical for every method by construction),
                          y = C_rep_plus_clean_out NL macro AUC (slate incl. ignore-rep / ignore-outputs candidates)
Points are seed means (encoder x release x attacker seeds within replicate); bars are exploratory 90% percentile
intervals on both axes; the noise sigma curve is a development description (sigma* chosen on attacker_val is ringed).
Annotations: supported class coverage and the encoder-seed / attacker-seed SDs (point estimates, not intervals).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ARM_COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}   # validated categorical slots 1-3 (all-pairs safe)
NOISE = "#52514e"                                               # secondary ink for the sigma curve (direct-labelled)
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABEL = {"A": "A untreated", "B": "B LEACE (target)", "C": "C LEACE (policy set)"}


def _rows(p: Path):
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _group(rows, cell, surface, recipe, metric):
    out = {}
    for r in rows:
        if r.get("cell") == cell and r.get("level") == "group" and r.get("surface") == surface and \
                r.get("recipe") == recipe and r.get("metric") == metric:
            out[r["arm_group"]] = (_f(r["point"]), _f(r["lower90"]), _f(r["upper90"]))
    return out


def plot_cell(cell, contract, rec, ut, sv, cov, sstar, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if contract == "C_rep":
        y = _group(rec, cell, "rep", "NL", "macro_auc")
        x = _group(ut, cell, "U2", "U2", "accuracy")
        xlabel = "U2 task accuracy (common LR probe on the released representation)"
        ylabel = "C_rep recovery: NL macro OvR AUC"
    else:
        y = _group(rec, cell, "rep+outputs", "NL", "macro_auc")
        clean = _group(ut, cell, "clean_output", "clean_output", "accuracy").get("CELL")
        x = {a: clean for a in y} if clean else {}
        xlabel = "clean-output task accuracy (unchanged across methods by construction)"
        ylabel = "C_rep_plus_clean_out recovery: NL macro OvR AUC"
    if not y or not x:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    noise = sorted((float(a.split("sigma")[1]), a) for a in y if a.startswith("D_sigma") and a in x)
    if noise:
        xs = [x[a][0] for _, a in noise]
        ys = [y[a][0] for _, a in noise]
        ax.plot(xs, ys, color=NOISE, lw=2, zorder=2, label="D noise, sigma grid (development description)")
        for (s, a), xx, yy in zip(noise, xs, ys):
            (_, xl, xh), (_, yl, yh) = x[a], y[a]
            ax.errorbar(xx, yy, xerr=[[xx - xl], [xh - xx]] if xl is not None else None,
                        yerr=[[yy - yl], [yh - yy]] if yl is not None else None, fmt="o", ms=5, color=NOISE,
                        ecolor=NOISE, elinewidth=1, capsize=2, zorder=3)
            ax.annotate(f"D σ={s:g}", (xx, yy), textcoords="offset points", xytext=(5, 4), fontsize=7, color=MUTED)
            if sstar is not None and abs(s - sstar) < 1e-12:
                ax.scatter([xx], [yy], s=160, facecolors="none", edgecolors=INK, linewidths=1.5, zorder=4,
                           label=f"σ* = {s:g} (chosen on attacker_val)")
    for a in ("A", "B", "C"):
        if a in y and a in x and y[a][0] is not None and x[a][0] is not None:
            (xx, xl, xh), (yy, yl, yh) = x[a], y[a]
            ax.errorbar(xx, yy, xerr=[[xx - xl], [xh - xx]] if xl is not None else None,
                        yerr=[[yy - yl], [yh - yy]] if yl is not None else None, fmt="o", ms=8, color=ARM_COLOR[a],
                        ecolor=ARM_COLOR[a], elinewidth=1.5, capsize=3, zorder=5, label=LABEL[a],
                        markeredgecolor=SURF, markeredgewidth=2)
            sd = [r for r in sv if r["cell"] == cell and r["arm_group"] == a and r["recipe"] == "NL" and
                  r["surface"] == ("rep" if contract == "C_rep" else "rep+outputs")]
            if sd:
                e, t = _f(sd[0]["encoder_seed_sd"]), _f(sd[0]["attacker_seed_sd_mean"])
                ax.annotate(f"{a}  SD enc {e:.3f} / att {t:.3f}" if e is not None and t is not None else a,
                            (xx, yy), textcoords="offset points", xytext=(8, -12), fontsize=7, color=INK)
    ax.axhline(0.55, color=MUTED, lw=1, ls="--", zorder=1)
    ax.annotate("diagnostic bar 0.55", (ax.get_xlim()[0], 0.55), textcoords="offset points", xytext=(4, 3),
                fontsize=7, color=MUTED)
    ax.grid(True, color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel(xlabel, fontsize=8, color=INK)
    ax.set_ylabel(ylabel, fontsize=8, color=INK)
    c = cov.get(cell, "?")
    ax.set_title(f"{cell} - {contract}\nsupported sensitive classes {c}; points = seed means; bars = exploratory 90% "
                 "intervals", fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{cell}__{contract}.png"
    fig.savefig(p, facecolor=SURF)
    plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    pkg = Path(a.package_dir)
    out = Path(a.out_dir) if a.out_dir else pkg / "plots"
    rec, ut = _rows(pkg / "RECOVERY.csv"), _rows(pkg / "UTILITY.csv")
    sv, sc = _rows(pkg / "SEED_VARIATION.csv"), _rows(pkg / "SUPPORT_COVERAGE.csv")
    cov = {}
    for r in sc:
        if r.get("what") == "sensitive" and r.get("class_coverage"):
            cov[r["cell"]] = r["class_coverage"]
    ss = json.loads((pkg / "SIGMA_STAR_SUMMARY.json").read_text()) if (pkg / "SIGMA_STAR_SUMMARY.json").exists() else {}
    cells = sorted({r["cell"] for r in rec if r.get("cell")})
    made = []
    for cell in cells:
        ds = cell.split("__")[0]
        sstar = (ss.get("values") or {}).get(ds)
        for contract in ("C_rep", "C_rep_plus_clean_out"):
            p = plot_cell(cell, contract, rec, ut, sv, cov, sstar, out)
            if p:
                made.append(str(p.relative_to(pkg)) if pkg in p.parents else str(p))
    out.mkdir(parents=True, exist_ok=True)
    (out / "PLOTS_INDEX.json").write_text(json.dumps({"plots": made, "note": "one figure per cell and contract; "
                                                      "never a cross-dataset frontier"}, indent=1))
    print(json.dumps({"plots": made}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
