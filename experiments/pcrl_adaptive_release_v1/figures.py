"""Aggregate-only 2018 outer task/recovery frontier figure.

Reads the locked selection mapping and already published aggregate reports.
It does not deserialize person records, contributions, or fitted predictors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import audit


METHODS = (
    ("H", "H only", "reference", "H"),
    ("U_candidate", "U candidate", "proposed", "17-token T0"),
    ("P_candidate", "Privacy-first", "proposed", "17-token T0"),
    ("D17", "D17", "matched_control", "17-token T0"),
    ("A_control_GRADIENT_001", "Gradient", "matched_control", "17-token T0"),
    ("A_control_historical_Q", "Historical Q", "matched_control", "17-token T0"),
    ("TaskOnly", "Task-only", "different_encoder_access", "17-token raw-input encoder"),
    ("J", "J (context)", "contextual_unmatched", "continuous J16"),
)
WEIGHTINGS = ("U", "PWGTP")
ANCHORS = (0, 1, 2)
ROLE_ORDER = ("utility:A/same_residence", "attack:A/SEX", "attack:A/RAC1P",
              "attack:AB/SEX", "attack:AB/RAC1P")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"nonfinite JSON constant {value}")))
    if not isinstance(value, dict):
        raise ValueError("aggregate source must be a JSON object")
    return value


def _score(report: dict, canonical: str, role: str, weighting: str) -> tuple[float, float]:
    if canonical == "H":
        baseline = [item["roles"][role]["H"][weighting]
                    for item in report["releases"].values()]
        if not baseline or max(baseline)-min(baseline) > 1e-10:
            raise ValueError("shared H score differs across release routes")
        score = h_score = float(baseline[0])
    else:
        item = report["releases"][canonical]["roles"][role]
        score = float(item["candidate"][weighting])
        h_score = float(item["H"][weighting])
    if not math.isfinite(score) or not math.isfinite(h_score):
        raise ValueError("nonfinite aggregate score")
    return score, h_score


def build_frontier(lock_path: Path, endpoint_path: Path, aggregate_dir: Path,
                   csv_path: Path) -> list[dict]:
    lock = _read(lock_path)
    endpoints = _read(endpoint_path)
    lock_sha = _sha(lock_path)
    if (lock.get("schema") != "pcrl-adaptive-selection-lock-v1" or
            lock.get("status") != "LOCKED" or
            lock.get("assessment_year") != 2018 or
            not lock.get("development_only") or
            endpoints.get("input_sha256", {}).get("selection_lock") != lock_sha or
            endpoints.get("outer_aggregate_crosscheck") != "PASS_ALL_THREE_ANCHORS"):
        raise ValueError("outer frontier sources do not match the locked 2018 study")
    if set(ROLE_ORDER) != set(audit.ROLES):
        raise ValueError("registered audit roles changed")
    reports = {}
    contextual = {}
    pins = endpoints["input_sha256"]["outer_aggregate_reports"]
    for anchor in ANCHORS:
        key = str(anchor)
        main_path = aggregate_dir / f"a{anchor}_OUTER_AUDIT.json"
        j_path = aggregate_dir / f"a{anchor}_CONTEXTUAL_J.json"
        if _sha(main_path) != pins[key]:
            raise ValueError(f"anchor {anchor} aggregate report differs from verified endpoint table")
        main = _read(main_path)
        j = _read(j_path)
        locked = lock["anchors"][key]
        for report, schema in ((main, "pcrl-adaptive-outer-audit-v1"),
                               (j, "pcrl-adaptive-continuous-outer-J-v1")):
            if (report.get("schema") != schema or report.get("anchor") != anchor or
                    report.get("assessment_year") != 2018 or
                    report.get("index_sha256") != lock["input_index_sha256"] or
                    report.get("selection_lock_sha256") != lock_sha or
                    report.get("no_outer_fit_or_selection") is not True):
                raise ValueError(f"anchor {anchor} outer report identity differs")
        if (main.get("inner_audit_sha256") != locked["inner_audit_sha256"] or
                main.get("inner_panel_complete_sha256") != locked["inner_panel_complete_sha256"] or
                sorted(main["releases"]) != sorted(locked["releases"]) or
                sorted(j["releases"]) != ["J"] or
                not j.get("contextual_not_matched_token_comparator") or
                j.get("inner_audit_sha256") != lock["external_context"]["J"][key]["inner_audit_sha256"] or
                j.get("inner_complete_sha256") != lock["external_context"]["J"][key]["inner_complete_sha256"]):
            raise ValueError(f"anchor {anchor} release or contextual J source differs")
        for role in ROLE_ORDER:
            for weighting in WEIGHTINGS:
                main_h = _score(main, "H", role, weighting)[0]
                j_h = _score(j, "H", role, weighting)[0]
                if abs(main_h-j_h) > 1e-10:
                    raise ValueError(f"anchor {anchor} contextual J H reference differs")
        reports[anchor] = main
        contextual[anchor] = j
    rows = []
    for method, label, category, wire in METHODS:
        for weighting in WEIGHTINGS:
            for role in ROLE_ORDER:
                losses, h_losses, canons = [], [], []
                source_shas = []
                for anchor in ANCHORS:
                    if method == "J":
                        report = contextual[anchor]
                        canonical = "J"
                        source_path = aggregate_dir / f"a{anchor}_CONTEXTUAL_J.json"
                    else:
                        report = reports[anchor]
                        canonical = ("H" if method == "H" else
                                     lock["anchors"][str(anchor)]["logical_to_canonical"][method])
                        source_path = aggregate_dir / f"a{anchor}_OUTER_AUDIT.json"
                    if canonical != "H" and canonical not in report["releases"]:
                        raise ValueError("locked logical release absent from outer report")
                    value, h_value = _score(report, canonical, role, weighting)
                    losses.append(value)
                    h_losses.append(h_value)
                    canons.append(canonical)
                    source_shas.append(_sha(source_path))
                # The registered aggregate is an equal mean of anchor-specific
                # weighted ratios; anchors share households, not populations.
                score = sum(losses) / len(ANCHORS)
                h_score = sum(h_losses) / len(ANCHORS)
                rows.append({
                    "method": method, "label": label, "category": category,
                    "wire": wire, "weighting": weighting, "role": role,
                    "loss_nats": score, "H_loss_nats": h_score,
                    "H_minus_method_nats": h_score-score,
                    "anchor_losses_json": json.dumps(losses, separators=(",", ":")),
                    "anchor_H_losses_json": json.dumps(h_losses, separators=(",", ":")),
                    "anchor_canonical_json": json.dumps(canons, separators=(",", ":")),
                    "anchor_aggregate_report_sha256_json": json.dumps(source_shas, separators=(",", ":")),
                    "selection_lock_sha256": lock_sha,
                    "development_only": "true",
                    "comparison_scope": ("contextual_unmatched" if method == "J" else
                                         "different_encoder_access" if method == "TaskOnly" else
                                         "matched_T0_17_token" if method != "H" else "H_only_reference"),
                })
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def draw_frontier(rows: list[dict], svg_path: Path, png_path: Path) -> None:
    plt.rcParams["svg.hashsalt"] = "pcrl-adaptive-outer-frontier-v1"
    colors = {"H": "#6b7280", "U_candidate": "#1d4ed8",
              "P_candidate": "#7c3aed", "D17": "#111827",
              "A_control_GRADIENT_001": "#0d9488",
              "A_control_historical_Q": "#a16207", "TaskOnly": "#c2410c",
              "J": "#dc2626"}
    markers = {"H": "s", "U_candidate": "o", "P_candidate": "D",
               "D17": "P", "A_control_GRADIENT_001": "v",
               "A_control_historical_Q": "^", "TaskOnly": "X", "J": "*"}
    lookup = {(row["method"], row["weighting"], row["role"]): row for row in rows}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex="col", sharey="row",
                             constrained_layout=True)
    for col, weighting in enumerate(WEIGHTINGS):
        for row_index, target in enumerate(("attack:AB/SEX", "attack:AB/RAC1P")):
            ax = axes[row_index, col]
            ax.axhline(0, color="#d1d5db", linewidth=.8)
            ax.axvline(0, color="#d1d5db", linewidth=.8)
            for method, label, category, _ in METHODS:
                task = lookup[(method, weighting, "utility:A/same_residence")]
                sensitive = lookup[(method, weighting, target)]
                x = task["H_minus_method_nats"]
                y = sensitive["H_minus_method_nats"]
                ax.scatter(x, y, marker=markers[method], s=105 if method == "J" else 63,
                           facecolors="none" if method == "J" else colors[method],
                           edgecolors=colors[method], linewidths=1.7,
                           label=label if row_index == 0 and col == 0 else None,
                           zorder=3)
            ax.set_title(f"{weighting} · coalition {target.split('/')[-1]}")
            ax.grid(color="#e5e7eb", linewidth=.6, zorder=0)
            if col == 0:
                ax.set_ylabel("Sensitive recovery over H (nats)\nlower is better")
            if row_index == 1:
                ax.set_xlabel("Residence task gain over H (nats) · higher is better")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(.5, 1.055), frameon=False, fontsize=9)
    fig.suptitle("2018 development outer assessment: task and coalition recovery",
                 fontsize=14, y=1.10)
    fig.text(.5, -.025,
             "Equal mean across three shared-household anchors; point estimates only. "
             "J is a contextual continuous release, not a matched 17-token control. "
             "Task-only uses different encoder access.",
             ha="center", fontsize=9, color="#4b5563")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, format="svg", bbox_inches="tight", metadata={"Date": None})
    # Matplotlib leaves spaces at SVG path line ends; normalize the exported
    # text so repository whitespace checks pass without changing geometry.
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/pcrl_adaptive_release_v1"))
    args = parser.parse_args()
    root = args.results_dir
    rows = build_frontier(root / "SELECTION_LOCK.json", root / "ENDPOINT_TABLE.json",
                          root / "private/aggregate_reports", root / "OUTER_FRONTIER.csv")
    draw_frontier(rows, root / "OUTER_FRONTIER.svg", root / "OUTER_FRONTIER.png")
    print(json.dumps({"rows": len(rows), "methods": len(METHODS),
                      "csv": str(root / "OUTER_FRONTIER.csv"),
                      "svg": str(root / "OUTER_FRONTIER.svg")}, sort_keys=True))


if __name__ == "__main__":
    main()
