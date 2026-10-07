"""[lra port of lcr/report.py at 091afc2: lcr->lra renames; later edits are listed in PORT_LOG.md]
Public report tables and figures for lra (lead).

The registered mechanism gate returned GATE_NOT_MET (FIXTURE_GATE.json, attempt 2). By the prompt's section 9 no
Adult fit was launched, so every Adult output is written as an explicit NOT_RUN record (never an empty or invented
table). The fixture-scope tables come from a deterministic replay of the LOCKED fixture engine. The replay's registered
fields must equal the registered FIXTURE_GATE.json exactly (timings excluded), or this module refuses to write.

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lra.sema --label A:report -- \\
        env OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lra.report all
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from lra import run as R

PKG = R.PKG
FIG = PKG / "figures"
GATE_LABEL = "MECHANISM_GATE_NOT_MET"
NOT_RUN_REASON = ("the registered mechanism gate returned GATE_NOT_MET (FIXTURE_GATE.json: NO_FIXTURE_TRIGGERED); "
                  "prompt section 9: do not launch Adult fits")
ADULT_STAGES = ["d1", "ctask", "fit", "inner", "inner_src", "controls", "select", "EVALUATION_LOCK", "assess", "infer"]
TIMING_KEYS = ("wall_s", "cpu_s")


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in TIMING_KEYS}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def _jsonable(x):
    return json.loads(json.dumps(R._finite(x), allow_nan=False))


def replay():
    """Re-run the locked fixture engine; refuse unless every registered field equals FIXTURE_GATE.json."""
    from lra import fixtures as FX
    gate = json.loads((PKG / "FIXTURE_GATE.json").read_text())
    laws = FX.load_laws()
    if gate["laws_sha256"] != laws["laws_sha256"]:
        raise SystemExit("REFUSED: FIXTURE_GATE.json binds different laws")
    out = []
    for fam, reg in zip(laws["families"], gate["fixtures"]):
        res, tables, arms = FX.run_fixture(fam)
        res["law_integrity"] = True
        mine = _strip(_jsonable({k: v for k, v in res.items() if k != "references"} | {"references": res["references"]}))
        if mine != _strip(reg):
            raise SystemExit(f"REFUSED: replay of {fam['id']} differs from the registered FIXTURE_GATE.json")
        out.append((fam, res, tables, arms))
    return gate, out


def _write_csv(path, rows, header=None):
    header = header or (list(rows[0]) if rows else ["status"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def decoder_only_ablation(out):
    """D0 versus D1 on IDENTICAL tokens (every fixed map of every fixture)."""
    rows = []
    for fam, res, _, arms in out:
        for cid, a in sorted(arms.items()):
            if "d0_of" not in a:
                continue
            b = arms[a["d0_of"]]
            same = all(np.array_equal(a["rel"][f"tok{i}"], b["rel"][f"tok{i}"]) for i in (1, 2))
            for i in (1, 2):
                ma, mb = a["metrics"], b["metrics"]
                rows.append({"scope": "fixture", "fixture": fam["id"], "d0_config": a["d0_of"], "d1_config": cid,
                             "recipient": i, "tokens_identical": same,
                             "L_D0": mb[f"L{i}"], "L_D1": ma[f"L{i}"], "dL_D1_minus_D0": ma[f"L{i}"] - mb[f"L{i}"],
                             "B_D0": mb[f"B{i}"], "B_D1": ma[f"B{i}"], "dB_D1_minus_D0": ma[f"B{i}"] - mb[f"B{i}"],
                             "I_i_D0": mb[f"I{i}"], "I_i_D1": ma[f"I{i}"], "I_i_identical": mb[f"I{i}"] == ma[f"I{i}"],
                             "I12_D0": mb["I12"], "I12_D1": ma["I12"], "I12_identical": mb["I12"] == ma["I12"],
                             "I_tok_q_minus_I_tok_D1": ma[f"I{i}_tok_q"] - ma[f"I{i}"],
                             "feasible_D0": b["feasible"], "feasible_D1": a["feasible"]})
    return rows


def budget_feasibility(out):
    from lra import fixtures as FX
    rows = []
    for fam, res, _, arms in out:
        ct = arms.get("U|C-TASK|i8o64|D1")
        U = res["U"]
        for cid, a in sorted(arms.items()):
            m = a["metrics"]
            p = R.parse_id(cid)
            r = {"scope": "fixture", "fixture": fam["id"], "config": cid, "arm": p["arm"], "decoder": p.get("decoder", "D0"),
                 "privacy_trained": bool(p.get("privacy_trained")), "feasible": a["feasible"],
                 "local_ok": bool(FX.local_ok(m, ct["metrics"])) if ct else None,
                 "mapper_status": a["record"]["status"] if "record" in a else None}
            for i in (1, 2):
                LU, BU = U[str(i)]["L"], U[str(i)]["B"]
                r[f"L{i}"], r[f"L{i}_U"], r[f"L{i}_limit"] = m[f"L{i}"], LU, LU + FX.BUDGET["ll"]
                r[f"L{i}_slack"] = r[f"L{i}_limit"] - m[f"L{i}"]
                r[f"B{i}"], r[f"B{i}_U"], r[f"B{i}_limit"] = m[f"B{i}"], BU, BU + FX.BUDGET["brier"]
                r[f"B{i}_slack"] = r[f"B{i}_limit"] - m[f"B{i}"]
                r[f"I{i}"] = m[f"I{i}"]
                r[f"tokens_per_class{i}"] = json.dumps(m[f"tokens_per_class{i}"])
            r.update({"I12": m["I12"], "T": m["T"], "Phi": m["Phi"], "acc1": m["acc1"], "acc2": m["acc2"]})
            rows.append(r)
    return rows


def decoder_certificates(out):
    from lra import decoder as DEC
    body = {"schema": "lra-decoder-certificates-v1", "scope": "fixture releases only (Adult D1 decoders were never fitted: "
            + NOT_RUN_REASON + ")", "tolerances": DEC.TOLERANCES, "fixtures": {}, "aggregate": {}}
    agg = {"releases": 0, "tables": 0, "supervised_tokens": 0, "fallback_tokens": 0, "all_converged": True,
           "max_stationarity_rel": 0.0, "max_dual_infeas_rel": 0.0, "max_projection_magnitude": 0.0,
           "max_sum_q_residual": 0.0, "min_margin": np.inf, "tokens_with_tie": 0}
    for fam, _, _, arms in out:
        f = {}
        for cid, a in sorted(arms.items()):
            if "decoder_body" not in a:
                continue
            _, d1, d2, sha = DEC.load_decoder_pair(json.loads(json.dumps(a["decoder_body"])), a["pair"], verify_solve=True)
            cs = {str(i): DEC.certificate_summary(d) for i, d in ((1, d1), (2, d2))}
            f[cid] = {"decoder_sha256": sha, "reloaded_with_bitwise_resolve": True, "certificates": cs}
            agg["releases"] += 1
            for c in cs.values():
                agg["tables"] += 1
                agg["supervised_tokens"] += c["supervised_tokens"]
                agg["fallback_tokens"] += c["fallback_tokens"]
                agg["all_converged"] = agg["all_converged"] and bool(c["all_converged"])
                for k in ("max_stationarity_rel", "max_dual_infeas_rel", "max_projection_magnitude", "max_sum_q_residual"):
                    agg[k] = max(agg[k], float(c[k]))
                agg["min_margin"] = min(agg["min_margin"], float(c["min_margin"]))
                agg["tokens_with_tie"] += int(c["tokens_with_tie"])
        body["fixtures"][fam["id"]] = f
    body["aggregate"] = agg
    return body


def optimization_receipts(out):
    body = {"schema": "lra-optimization-receipts-v1", "scope": "fixture mapper units only (" + NOT_RUN_REASON + ")",
            "fixtures": {}}
    for fam, res, _, arms in out:
        f = {}
        for cid, a in sorted(arms.items()):
            rec = a.get("record")
            if rec is None:
                continue
            starts = []
            for s in rec["starts"]:
                nm = s.get("name") or (s.get("start") or {}).get("name")
                st = s.get("status") or (s.get("stage1") or {}).get("status") or ("REFINED" if "final" in s else None)
                starts.append({"name": nm, "status": st, "eligible": s.get("eligible"), "sweeps": s.get("sweeps"),
                               "stop": s.get("stop"), "evals": s.get("evals")})
            f[cid] = {"status": rec["status"], "winner": rec["winner"], "work": rec["work"],
                      "eval_ceiling_hit": rec["work"]["evals"] >= rec["work"]["eval_ceiling"],
                      "deployed_feasible": rec["deployed"]["feasible"], "parity_ok": rec["deployed"]["parity_ok"],
                      "parity_abs_diff": rec["deployed"]["parity_abs_diff"], "rules_sha256": rec["rules_sha256"],
                      "sex_used_in_search": rec["sex_used_in_search"], "starts": starts,
                      "c6_label": res["checks"]["C6_HEURISTIC_LABELLING"]["labels"].get(cid)}
        body["fixtures"][fam["id"]] = f
    return body


def class_preservation(out):
    body = {"schema": "lra-class-preservation-v1", "scope": "fixture releases (Adult not run)", "fixtures": {}}
    for fam, res, _, arms in out:
        body["fixtures"][fam["id"]] = {
            "releases": len(arms),
            "all_preserved": all(a["metrics"]["keys_ok"] and a["metrics"]["decisions_ok1"] and a["metrics"]["decisions_ok2"]
                                 and a["metrics"]["q_ok1"] and a["metrics"]["q_ok2"] for a in arms.values()),
            "C4": res["checks"]["C4_DECISION_PRESERVATION"]}
    return body


def post_hoc_equal_leakage(out, gate):
    """UNREGISTERED, post-hoc descriptive comparison (found after the gate ran; never enters the gate or the label):
    at pair MI no larger than T*'s, the lowest task loss T among budget- and local-feasible privacy-trained D1 releases
    versus T*'s own T."""
    from lra import fixtures as FX
    body = {"schema": "lra-post-hoc-v1", "status": "UNREGISTERED_POST_HOC_DESCRIPTIVE",
            "found": "after the registered fixture stage, from the fixture trade-off figure",
            "cannot_change": "the gate verdict (GATE_NOT_MET), the registered trigger (pair-MI reduction only) or the label "
                             + GATE_LABEL, "fixtures": {}}
    for (fam, res, _, arms), reg in zip(out, gate["fixtures"]):
        t = reg["trigger"]
        if not t.get("T_star"):
            body["fixtures"][fam["id"]] = {"T_star": None, "reason": t.get("reason")}
            continue
        ts = arms[t["T_star"]]["metrics"]
        ct = arms["U|C-TASK|i8o64|D1"]["metrics"]
        cands = []
        for cid, a in arms.items():
            p = R.parse_id(cid)
            if not (p.get("privacy_trained") and p.get("decoder") == "D1"):
                continue
            m = a["metrics"]
            if a["feasible"] and FX.local_ok(m, ct) and m["I12"] <= ts["I12"] + 1e-12:
                cands.append((m["T"], cid, p["arm"]))
        cands.sort()
        best_by_arm = {}
        for T, cid, arm in cands:
            best_by_arm.setdefault(arm, {"config": cid, "T": T})
        body["fixtures"][fam["id"]] = {
            "T_star": t["T_star"], "T_star_I12": ts["I12"], "T_star_T": ts["T"],
            "C_TASK_T": ct["T"], "C_TASK_I12": ct["I12"],
            "n_privacy_D1_at_or_below_T_star_I12": len(cands),
            "best": None if not cands else {"config": cands[0][1], "arm": cands[0][2], "T": cands[0][0],
                                            "T_star_T_minus_best_T": ts["T"] - cands[0][0],
                                            "best_tie_set": sorted(c for T, c, _ in cands if T == cands[0][0]),
                                            "units": "T = L1 + L2 + 0.5 (B1 + B2): log loss (nats) plus half the Brier "
                                                     "score, a mixed-unit task objective"},
            "best_by_arm": best_by_arm}
    return body


def run_status(gate):
    ledger = [json.loads(l) for l in (R.RUN / "COMPUTE_LEDGER.jsonl").read_text().splitlines() if l.strip()]
    att = json.loads((PKG / "FIXTURE_ATTEMPTS.json").read_text())
    return {"schema": "lra-run-status-v1", "label": GATE_LABEL,
            "stages_run": [{"stage": e["stage"], "at": e["at"], "cpu_s": e["cpu_s"], "wall_s": e["wall_s"]} for e in ledger],
            "fixture_attempts": att["attempts"], "gate": {"verdict": gate["verdict"], "reasons": gate["reasons"]},
            "stages_not_run": {s: NOT_RUN_REASON for s in ADULT_STAGES},
            "locks": {"SOURCE_ADMISSION_LOCK": "pushed (4a91947)", "FIXTURE_LOCK": "pushed (9ac4cb7)",
                      "AMENDMENT_A1_FIXTURE_C5_SCOPE": "pushed (c9a7150)",
                      "SCIENCE_LOCK": "NOT WRITTEN (" + NOT_RUN_REASON + ")",
                      "EVALUATION_LOCK": "NOT WRITTEN (no Adult fit; assessment labels never unsealed)"},
            "adult_labels_used": ("none: source admission loaded the input but read no task label or SEX; no stage "
                                  "that reads task labels or SEX for fitting, audit or selection ran; lra never unsealed "
                                  "the assessment rows"),
            "claim_status_convention": ("NOT_RUN is a display convention for the gate-failure path, not a registered "
                                        "status; the 37 primary slots were never instantiated")}


def not_run_stubs():
    msg = {"status": "NOT_RUN", "label": GATE_LABEL, "reason": NOT_RUN_REASON}
    for name in ("SELECTION.json",):
        (PKG / name).write_text(json.dumps({"schema": "lra-not-run-v1", "file": name, **msg}, indent=1) + "\n")
    for name in ("INNER_SELECTION_TABLE.csv", "PRIMARY_ENDPOINTS.csv", "ALL_LEVELS.csv"):
        _write_csv(PKG / name, [{"file": name, **msg}], ["file", "status", "label", "reason"])


def figures(out, abl, bud):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    made = {}
    fams = [f["id"] for f, *_ in out]
    # 1: fixture trade-off (T vs pair MI) with the registered utility limits (feasible vs infeasible)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    for ax, fid in zip(axes, fams):
        rr = [r for r in bud if r["fixture"] == fid]
        for kind, mk in (("d1_fixed", "o"), ("weighted", "s"), ("constrained", "^"), ("ctask", "D")):
            for feas, col in ((True, "tab:blue"), (False, "tab:red")):
                pts = [(r["T"], r["I12"]) for r in rr if r["arm"] == kind and r["decoder"] == "D1" and r["feasible"] == feas]
                if pts:
                    ax.scatter(*zip(*pts), marker=mk, c=col, s=22, alpha=0.7,
                               label=f"{kind} {'feasible' if feas else 'infeasible'}")
        cls = [r for r in rr if r["config"] == "U|CLASS|i1o1|D1"]
        if cls:
            f = cls[0]["feasible"]
            ax.scatter([cls[0]["T"]], [cls[0]["I12"]], marker="*", s=170, facecolors="k" if f else "none",
                       edgecolors="k" if f else "tab:red", linewidths=1.2,
                       label=f"CLASS|D1 task-only ({'feasible' if f else 'infeasible'})")
        ax.set_title(fid, fontsize=9)
        ax.set_xlabel("T = L1 + L2 + 0.5 (B1 + B2)  [fitting]")
    axes[0].set_ylabel("pair MI I12 (nats, exact law)")
    axes[-1].legend(fontsize=6, loc="best")
    fig.suptitle("Fixture trade-off under the registered budgets (blue = budget-feasible, red = violates a budget)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_fixture_tradeoff.png", dpi=130)
    plt.close(fig)
    made["fig1_fixture_tradeoff.png"] = "inner trade-off analogue on the four fixtures (Adult inner audit not run)"
    # 3: fixed-token decoder ablation
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    for ax, fid in zip(axes, fams):
        rr = [r for r in abl if r["fixture"] == fid]
        x = np.arange(len(rr))
        ax.bar(x, [r["dL_D1_minus_D0"] for r in rr], color=["tab:green" if r["recipient"] == 1 else "tab:purple" for r in rr])
        ax.axhline(0, c="k", lw=0.6)
        ax.set_title(f"{fid}: D1 - D0 log loss on identical tokens", fontsize=8)
        ax.set_xticks([])
        ax.set_xlabel(f"{len(rr) // 2} fixed maps x 2 recipients (MI identical in all)")
    axes[0].set_ylabel("dL (nats)")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_fixed_token_decoder_ablation.png", dpi=130)
    plt.close(fig)
    made["fig3_fixed_token_decoder_ablation.png"] = "fixed-token decoder ablation on the fixtures"
    # 4: individual and pair information
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, fid in zip(axes, fams):
        rr = [r for r in bud if r["fixture"] == fid and r["decoder"] == "D1"]
        ax.scatter([r["I1"] for r in rr], [r["I12"] for r in rr], s=14, label="I1 vs I12", alpha=0.7)
        ax.scatter([r["I2"] for r in rr], [r["I12"] for r in rr], s=14, label="I2 vs I12", alpha=0.7)
        ax.set_title(fid, fontsize=9)
        ax.set_xlabel("individual MI (nats)")
    axes[0].set_ylabel("pair MI I12 (nats)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_fixture_individual_pair_information.png", dpi=130)
    plt.close(fig)
    made["fig4_fixture_individual_pair_information.png"] = ("individual / pair exact-law information on the fixtures "
                                                            "(Adult attacker recovery not run)")
    # 5: token counts
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, fid in zip(axes, fams):
        rr = [r for r in bud if r["fixture"] == fid and r["decoder"] == "D1"]
        t = [sum(json.loads(r["tokens_per_class1"])) + sum(json.loads(r["tokens_per_class2"])) for r in rr]
        ax.hist(t, bins=np.arange(min(t) - 0.5, max(t) + 1.5), color="tab:gray")
        ax.set_title(fid, fontsize=9)
        ax.set_xlabel("released tokens (both recipients)")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_fixture_token_counts.png", dpi=130)
    plt.close(fig)
    made["fig5_fixture_token_counts.png"] = "actual token counts of every fixture D1 release"
    idx = {"schema": "lra-figures-v1", "made": made,
           "not_made": {"fig2_locked_assessment_tradeoff": "NOT_RUN (" + NOT_RUN_REASON + ")"}}
    (FIG / "FIGURES.json").write_text(json.dumps(idx, indent=1) + "\n")
    return idx


def write_all():
    gate, out = replay()
    if gate["verdict"] != "GATE_NOT_MET":
        raise SystemExit("REFUSED: this report path is the registered gate-failure path")
    abl = decoder_only_ablation(out)
    bud = budget_feasibility(out)
    _write_csv(PKG / "DECODER_ONLY_ABLATION.csv", abl)
    _write_csv(PKG / "BUDGET_FEASIBILITY.csv", bud)
    for name, body in (("DECODER_CERTIFICATES.json", decoder_certificates(out)),
                       ("OPTIMIZATION_RECEIPTS.json", optimization_receipts(out)),
                       ("CLASS_PRESERVATION.json", class_preservation(out)),
                       ("RUN_STATUS.json", run_status(gate)),
                       ("POST_HOC_EQUAL_LEAKAGE_UTILITY.json", post_hoc_equal_leakage(out, gate))):
        (PKG / name).write_text(json.dumps(_jsonable(body), indent=1, allow_nan=False) + "\n")
    not_run_stubs()
    figs = figures(out, abl, bud)
    print(json.dumps({"replay_equal_to_registered": True, "ablation_rows": len(abl), "budget_rows": len(bud),
                      "figures": sorted(figs["made"])}, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] != ["all"]:
        raise SystemExit("usage: python -m lra.report all")
    write_all()
