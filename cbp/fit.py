"""Intermediate-lambda privacy bank for the confidence-budgeted privacy study (cbp; role B, optimizer engineer).

All mathematics is qpc's, imported UNCHANGED (qpc/compress.py, qpc/release.py and their dpc/qpc dependencies, pinned by
the qpc STAGE_B_LOCK; the pins are re-checked by ``code_version``). This module adds no optimiser change: no
convergence rescue, occupation-only objective, randomisation, feedback, coordinated-move solver, extra JOINT start or
continuation path. It only adds

  * the fixed bank definition (one rate i8o64, lambda grid, families, seeds, config IDs, reuse status, fit order);
  * ``fit_unit``: a pass-through to ``qpc.compress.fit_unit(family, fine_dict, T, tr, S_fit, 8, 64, lam, meta,
    witnesses=witnesses)`` whose record is augmented (key "cbp") with the prompt section 13 diagnostics, the final
    fitting statistics reconstructed FROM THE DEPLOYED RELEASE, and the JOINT-versus-sequential computational
    asymmetry receipt;
  * ``endpoint_parity``: re-verification of a REUSED qpc unit (lambda 0.01 / 0.1 privacy units, DIRECT-TASK,
    FINE-TASK, CLASS) without refitting it;
  * alias accounting and the synthetic full-bank timing (``python -m cbp.fit timing``; synthetic data only).

Bank (FIT_MANIFEST is the lead's; this is its code definition):
    rate (m1, m2) = (8, 64) states per teacher-predicted class; lambda in LAMS = (0.01, 0.025, 0.04, 0.06, 0.08, 0.1);
    families LOCAL, SEQ-12, SEQ-21, JOINT; seeds 0, 1, 2 -> 72 logical privacy mapping-pair units;
    REUSED (admitted after parity) = lambda in {0.01, 0.1} (24 units); NEW = lambda in {0.025, 0.04, 0.06, 0.08} (48);
    references (reused after parity): U|DIRECT-TASK|i8o64, U|FINE-TASK|i8o64, U|CLASS|i1o1 (9 units).
    Config IDs U|{FAM}|i8o64|l{lam:g}; unit names pol__s{k}__<config with "|" -> "_">.

    python -m cbp.fit bank                                    # print the bank (no data)
    python -m cbp.fit timing [--out <TIMING.json>] [--seeds 0,1,2]   # SYNTHETIC timing, under cbp.sema
"""
from __future__ import annotations

import hashlib
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from dpc.compress import mi_plugin
from dpc.partition import kl_rows
from qpc import compress as CP
from qpc import kmeans as KM
from qpc import partition as PT
from qpc import release as RL
from qpc import stagea as SA

WT = Path(__file__).resolve().parents[1]
QPC_PKG = "results/pcrl_confidence_capacity_v1"
PKG = "results/pcrl_confidence_budgeted_privacy_v1"

# ----------------------------------------------------------------------------------------------- the fixed bank
TEACHER = "U"
M1, M2 = 8, 64
LAMS = (0.01, 0.025, 0.04, 0.06, 0.08, 0.1)
REUSED_LAMS = (0.01, 0.1)
NEW_LAMS = (0.025, 0.04, 0.06, 0.08)
FAMILIES = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
SEEDS = (0, 1, 2)
REFERENCES = ("DIRECT-TASK", "FINE-TASK", "CLASS")
WITNESS_FAMILIES = ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")

# registered tolerances (fixed before any fit)
PARITY_RTOL = 1e-12        # |a - b| <= PARITY_RTOL * max(|a|, |b|) for D1, D2, I1, I2, I12 (and the F values)
MEAN_ATOL = 1e-12          # per-token mean probability (release rows vs stored statistics; summation order differs)

# qpc constants this study relies on (asserted unchanged at import; a changed qpc refuses to load cbp.fit)
QPC_CONSTANTS = {"TOL": 1e-12, "TIE_TOL": 1e-12, "SWEEPS": 5, "EPS": 1e-12,
                 "WITNESSES": ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21"),
                 "JOINT_START_ORDER": ("JOINT-GREEDY", "FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21"),
                 "PERM_SEED": 20261006, "N_PERM": 100, "OLD_RULE_DIAGNOSTIC": True}

# qpc code pins (sha256 of the files the qpc fits ran with; from results/pcrl_confidence_capacity_v1/STAGE_B_LOCK.json
# written 2026-10-06T04:43:38Z, and STAGE_A_LOCK.json for the Stage A DIRECT-TASK code)
QPC_STAGE_B_PINS = {
    "qpc/compress.py": "3495e2a8817135fdc8f29818303fd225b0704332f93203bf3caaa7a6ce871c8a",
    "qpc/release.py": "8131d085ff04ff6f55855b758173dd154bf2e2902fd3213420dcb8b677c12eb2",
    "qpc/partition.py": "a4396381979fc59b4484d72c44295b0032937b06f4eaa35a583dfb7230449001",
    "qpc/kmeans.py": "f126e6b387064f2846137ddfef1c158030b6bd213836bca12a2c014143b96ad9",
    "qpc/stagea.py": "2e0ca610d3d0a15feb0c7d4aba29b529de419bcf7e7b2dd9b3d769ccd5d01ea9",
    "dpc/compress.py": "96988629698b7aa31557e1acf47cf40a445249b22feef47ad7478ddf74834de2",
    "dpc/release.py": "0f4b7e26f70dd88d34782fd673ac5385e0e16b66434fca703053acacaf3112e5",
    "dpc/partition.py": "4b1d0febb7512646bd00664a36a35262f6bfebf86f943a448d63f46d3ea90cdd"}
STAGE_B_GATED = ("qpc/compress.py", "qpc/release.py")       # the lead's required pin; the closure is also checked
STAGE_A_GATED = ("qpc/stagea.py", "qpc/kmeans.py", "qpc/release.py")


def _check_qpc_constants():
    got = {"TOL": CP.TOL, "TIE_TOL": CP.TIE_TOL, "SWEEPS": CP.SWEEPS, "EPS": KM.EPS, "WITNESSES": CP.WITNESSES,
           "JOINT_START_ORDER": CP.JOINT_START_ORDER, "PERM_SEED": CP.PERM_SEED, "N_PERM": CP.N_PERM,
           "OLD_RULE_DIAGNOSTIC": CP.OLD_RULE_DIAGNOSTIC}
    bad = {k: (got[k], v) for k, v in QPC_CONSTANTS.items() if got[k] != v}
    if bad:
        raise ImportError(f"qpc constants differ from the registered values: {bad}")


_check_qpc_constants()
assert set(LAMS) == set(REUSED_LAMS) | set(NEW_LAMS) and not set(REUSED_LAMS) & set(NEW_LAMS)


def g(lam):
    return f"{float(lam):g}"


def config_id(family, lam=None):
    """Configuration IDs (qpc scheme): U|{FAM}|i8o64|l{lam:g}, U|FINE-TASK|i8o64, U|DIRECT-TASK|i8o64, U|CLASS|i1o1."""
    if family == "CLASS":
        return f"{TEACHER}|CLASS|i1o1"
    if family in ("FINE-TASK", "DIRECT-TASK"):
        if lam is not None:
            raise ValueError(f"{family} is task-only; lam must be None")
        return f"{TEACHER}|{family}|i{M1}o{M2}"
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family}")
    if lam is None:
        raise ValueError(f"{family} needs a lambda")
    cid = CP.config_id(TEACHER, family, M1, M2, float(lam))
    assert cid == f"{TEACHER}|{family}|i{M1}o{M2}|l{g(lam)}"
    return cid


def unit_name(seed, cid):
    return f"pol__s{int(seed)}__{cid.replace('|', '_')}"


def status_of(lam):
    lam = float(lam)
    if lam in REUSED_LAMS:
        return "REUSED"
    if lam in NEW_LAMS:
        return "NEW"
    raise ValueError(f"lambda {lam} is not on the registered grid {LAMS}")


def witness_configs(lam):
    """JOINT witnesses at the same lambda: FINE-TASK (lambda-free reference) and LOCAL, SEQ-12, SEQ-21 at lam."""
    return {f: (config_id("FINE-TASK") if f == "FINE-TASK" else config_id(f, lam)) for f in WITNESS_FAMILIES}


def bank():
    """The 72 logical privacy mapping-pair units, ordered (seed, lambda, family)."""
    out = []
    for k in SEEDS:
        for lam in LAMS:
            for fam in FAMILIES:
                cid = config_id(fam, lam)
                u = {"seed": k, "family": fam, "lam": lam, "config": cid, "unit": unit_name(k, cid),
                     "status": status_of(lam)}
                if fam == "JOINT":
                    u["witnesses"] = {f: {"config": c, "unit": unit_name(k, c)} for f, c in witness_configs(lam).items()}
                out.append(u)
    return out


def reference_units():
    """The 9 reused task-only references (3 per seed)."""
    return [{"seed": k, "family": f, "lam": None, "config": config_id(f), "unit": unit_name(k, config_id(f)),
             "status": "REUSED_REFERENCE"} for k in SEEDS for f in REFERENCES]


def reusable_ids():
    return [config_id(f, lam) for lam in REUSED_LAMS for f in FAMILIES] + [config_id(f) for f in REFERENCES]


def registered_ids():
    """Every mapping-pair configuration of the cbp bank (privacy grid + task-only references)."""
    return [config_id(f, lam) for lam in LAMS for f in FAMILIES] + [config_id(f) for f in REFERENCES]


def new_fit_order():
    """The 48 new fits in a valid order: per seed and new lambda, LOCAL, SEQ-12, SEQ-21, then JOINT (whose witnesses
    are the three just fitted plus the reused FINE-TASK)."""
    jobs = []
    for k in SEEDS:
        for lam in NEW_LAMS:
            for fam in FAMILIES:
                cid = config_id(fam, lam)
                j = {"seed": k, "family": fam, "lam": lam, "config": cid, "unit": unit_name(k, cid)}
                if fam == "JOINT":
                    j["witness_units"] = {f: unit_name(k, c) for f, c in witness_configs(lam).items()}
                jobs.append(j)
    return jobs


def bank_counts():
    b = bank()
    return {"logical_privacy_units": len(b), "reused_endpoint_units": sum(u["status"] == "REUSED" for u in b),
            "new_fits": sum(u["status"] == "NEW" for u in b), "reused_reference_units": len(reference_units()),
            "families": list(FAMILIES), "lams": list(LAMS), "reused_lams": list(REUSED_LAMS),
            "new_lams": list(NEW_LAMS), "seeds": list(SEEDS), "rate": [M1, M2],
            "note": "exact aliases (identical pair fingerprints) are counted after fitting with ``aliases``; an alias "
                    "is still one fitted (or reused) unit, never an extra model fit"}


# ----------------------------------------------------------------------------------------------- helpers
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _rel(a, b):
    a, b = float(a), float(b)
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b))


def compare_terms(recomputed, stored, keys, rtol=PARITY_RTOL):
    per = {}
    for k in keys:
        if k not in stored or k not in recomputed:
            per[k] = {"present": False}
            continue
        a, b = float(recomputed[k]), float(stored[k])
        per[k] = {"present": True, "recomputed": a, "stored": b, "abs_diff": abs(a - b), "rel_diff": _rel(a, b),
                  "bitwise_equal": bool(a == b)}
    present = [v for v in per.values() if v["present"]]
    ok = bool(len(present) == len(keys) and all(v["rel_diff"] <= rtol for v in present))
    return {"keys": list(keys), "rtol": rtol, "per_term": per, "within_rtol": ok,
            "max_abs_diff": max((v["abs_diff"] for v in present), default=None),
            "max_rel_diff": max((v["rel_diff"] for v in present), default=None),
            "all_bitwise_equal": bool(present and all(v["bitwise_equal"] for v in present))}


def term_keys(lam):
    return ("D1", "D2", "I1", "I2", "I12", "F_task") + (("F_local", "F_joint") if lam is not None else ())


# ----------------------------------------------------------------------------------------------- deployed release
def reconstruct_fit_statistics(pair, out, T, tr, S_fit, lam=None):
    """Final fitting statistics recomputed FROM THE DEPLOYED RELEASE arrays (``out``: row_id, tok_i, q_i, hard_i,
    alpha_i over ALL rows, as written to release.npz) on the fitting rows ``tr``: per recipient the token counts (exact
    against the policy's stored n_t), the per-token mean teacher probability (against S_t / n_t, |diff| <= MEAN_ATOL),
    the decoded vectors (bitwise against the policy prototypes), the decisions (exact against the teacher decision and
    the token class); then D_i = mean KL(p_i || q_i), I_i = plug-in I(S; token_i), I12 = plug-in I(S; (token_1, token_2))
    and the F values -- the qpc definitions (``qpc.compress._brute_terms``) applied to the stored release."""
    tr = np.asarray(tr)
    s = CP.check_sex(S_fit, tr.size)
    res = {"rows_fit": int(tr.size), "mean_atol": MEAN_ATOL}
    toks, structure_ok = {}, True
    terms = {}
    for i, pol in ((1, pair.p1), (2, pair.p2)):
        tok = np.asarray(out[f"tok{i}"])[tr].astype(np.int64)
        q = np.asarray(out[f"q{i}"], dtype=np.float64)[tr]
        hard = np.asarray(out[f"hard{i}"])[tr].astype(np.int64)
        P = KM.check_probs(np.asarray(T[f"p{i}"])[tr], pol.K, f"P{i}")
        d = np.asarray(T[f"d{i}"])[tr].astype(np.int64)
        alpha = int(np.asarray(out[f"alpha{i}"]))
        in_range = bool(tok.size == 0 or (tok.min() >= 0 and tok.max() < pol.T))
        r = {"alphabet_release": alpha, "alphabet_policy": pol.T, "alphabet_equal": alpha == pol.T,
             "tokens_in_alphabet": in_range}
        if in_range:
            n = np.bincount(tok, minlength=pol.T)
            Ssum = np.stack([np.bincount(tok, weights=P[:, k], minlength=pol.T) for k in range(pol.K)], 1)
            used = pol.token_n > 0
            r["token_counts_equal_policy"] = bool(np.array_equal(n, pol.token_n))
            if r["token_counts_equal_policy"] and used.any():
                diff = np.abs(Ssum[used] / n[used, None] - pol.token_S[used] / pol.token_n[used, None])
                r["token_mean_max_abs_diff"] = float(diff.max())
            else:
                r["token_mean_max_abs_diff"] = None if not r["token_counts_equal_policy"] else 0.0
            r["token_mean_within_atol"] = bool(r["token_mean_max_abs_diff"] is not None
                                              and r["token_mean_max_abs_diff"] <= MEAN_ATOL)
            r["decoded_equals_prototype_bitwise"] = bool(q.shape == (tok.size, pol.K)
                                                         and np.array_equal(q, pol.token_proto[tok]))
            r["decision_equals_token_class"] = bool(np.array_equal(hard, pol.token_class[tok]))
        else:
            r.update({"token_counts_equal_policy": False, "token_mean_max_abs_diff": None,
                      "token_mean_within_atol": False, "decoded_equals_prototype_bitwise": False,
                      "decision_equals_token_class": False})
        r["decisions_equal_teacher"] = bool(np.array_equal(hard, d))
        r["decoded_argmax_equals_decision"] = bool(q.shape[0] == 0 or np.array_equal(q.argmax(1), hard))
        ok_i = all(r[k] for k in ("alphabet_equal", "tokens_in_alphabet", "token_counts_equal_policy",
                                  "token_mean_within_atol", "decoded_equals_prototype_bitwise",
                                  "decision_equals_token_class", "decisions_equal_teacher",
                                  "decoded_argmax_equals_decision"))
        structure_ok = structure_ok and ok_i
        terms[f"D{i}"] = float(np.mean(kl_rows(P, q))) if tr.size else 0.0
        terms[f"I{i}"] = mi_plugin(s, tok)
        toks[i] = tok
        res[f"r{i}"] = r
    terms["I12"] = mi_plugin(s, toks[1] * pair.p2.T + toks[2])
    terms.update(CP.F_values(terms, lam))
    res["terms"] = terms
    res["structure_ok"] = bool(structure_ok)
    res["_tokens"] = toks
    return res


def _support(pol, tok):
    n = np.bincount(tok, minlength=pol.T) if tok.size else np.zeros(pol.T, dtype=np.int64)
    per = []
    for c in range(pol.K):
        idx = np.flatnonzero(pol.token_class == c)
        nn = n[idx]
        occ = nn[nn > 0]
        per.append({"class": c, "tokens": int(idx.size), "occupied_fit": int(occ.size),
                    "fallback": bool(pol.token_fallback[idx].any()), "rows_fit": int(nn.sum()),
                    "min_count": int(occ.min()) if occ.size else None,
                    "median_count": float(np.median(occ)) if occ.size else None,
                    "singletons": int(np.sum(nn == 1)), "lt5": int(np.sum((nn > 0) & (nn < 5)))})
    return per


def _sparsity_summary(t1, t2, pair):
    sp = CP.sparsity_receipt(t1, t2, pair)
    return {k: {x: v[x] for x in ("alphabet", "occupied", "unseen_fraction", "singleton_cells",
                                  "singleton_fraction_of_occupied", "lt5_cells", "entropy_nats")} for k, v in sp.items()}


def search_summary(rec):
    """Merge counts, positive merges, accepted exchanges, sweeps (cap 5), convergence and unresolved local optima from a
    qpc-format record (new cbp fit or reused qpc unit). Not defined for DIRECT-TASK (Stage A k-means)."""
    def _stage(v):
        merges, moves = v.get("merges", []) or [], v.get("moves", []) or []
        return {"merges_to_cap": int(v.get("merges_to_cap", 0) or 0),
                "extra_merges_greedy": int(v.get("extra_merges_greedy", 0) or 0),
                "merges_logged": len(merges),
                "positive_increment_merges": int(v.get("positive_increment_merges", 0) or 0),
                "accepted_exchanges": int(sum(1 for x in moves if x.get("kind") == "move")),
                "refine_merges": int(sum(1 for x in moves if x.get("kind") == "extra")),
                "sweeps": int(v.get("sweeps", 0) or 0), "converged": bool(v.get("converged", True))}
    if "starts" in rec:
        per = {}
        for name, v in rec["starts"].items():
            p = _stage(v)
            p.update({"initial_F_joint": v.get("initial_F_joint"), "refined_F_joint": v.get("refined_F_joint"),
                      "gap_to_winner": v.get("gap_to_winner"), "same_map_as_winner": v.get("same_map_as_winner"),
                      "source": v.get("source", "joint-greedy")})
            per[name] = p
        kind = "joint_starts"
        unresolved = len(rec.get("unresolved_local_optima", []))
        not_conv = list(rec.get("starts_not_converged", []))
        winner = rec.get("winner")
    elif "stages" in rec:
        per = {s.get("stage", str(j)): _stage(s) for j, s in enumerate(rec["stages"])}
        kind = "stages"
        unresolved = None
        not_conv = [n for n, v in per.items() if not v["converged"]]
        winner = None
    else:
        return {"defined": False, "reason": "Stage A DIRECT-TASK k-means unit (no Stage B search); its convergence "
                                            "receipts are the qpc dir__ units"}
    tot = {k: int(sum(v[k] for v in per.values())) for k in ("merges_to_cap", "extra_merges_greedy", "merges_logged",
                                                             "positive_increment_merges", "accepted_exchanges",
                                                             "refine_merges", "sweeps")}
    return {"defined": True, "kind": kind, "per": per, "totals": tot, "sweep_cap": CP.SWEEPS,
            "converged_all": bool(all(v["converged"] for v in per.values())), "not_converged": not_conv,
            "unresolved_local_optima": unresolved,
            "unresolved_local_optima_note": None if unresolved is not None else
            "single search path: unresolved local optima are defined only across JOINT's starts",
            "winner": winner, "qpc_summary": rec.get("summary")}


WORK_COVERAGE = {
    "JOINT": "qpc State.work summed over the five refined starts (JOINT-GREEDY + four witness refinements); the four "
             "witnesses' own fits are separate units and are NOT included",
    "SEQ-12": "qpc State.work of the stage-2 state only: stage 1 and the old-rule stage-1 diagnostic are not counted by "
              "qpc's counters (use the logged merges/moves and the unit CPU seconds)",
    "SEQ-21": "qpc State.work of the stage-2 state only: stage 1 and the old-rule stage-1 diagnostic are not counted by "
              "qpc's counters (use the logged merges/moves and the unit CPU seconds)",
    "LOCAL": "qpc State.work of the single state (both recipients' independent stages)",
    "FINE-TASK": "qpc State.work of the single state (both recipients' independent stages)",
    "CLASS": "no search", "DIRECT-TASK": "Stage A k-means (not a Stage B search)"}

ASYMMETRY_NOTE = ("Equal lambda access, data, state caps and fixed per-stage refinement limits (<= 5 sweeps) do not "
                  "imply identical realised compute. JOINT refines five starts and compares nine candidates (five "
                  "refined, four unchanged witnesses), and its witnesses are themselves complete fitted units of the "
                  "other families; a sequential arm runs one two-stage path (plus qpc's unselected old-rule stage-1 "
                  "diagnostic). The comparison is not an exact compute match, and the sequential adaptations are not "
                  "the official Taylor solver.")


def asymmetry_from_record(rec):
    fam = rec.get("family")
    s = search_summary(rec)
    a = {"family": fam, "cpu_seconds_unit": rec.get("cpu_seconds"), "wall_seconds_unit": rec.get("wall_seconds"),
         "qpc_work_counters": rec.get("work"), "work_counter_coverage": WORK_COVERAGE.get(fam)}
    if fam == "JOINT":
        a.update({"search_paths": len(CP.JOINT_START_ORDER), "candidates_compared": len(rec.get("candidates", [])),
                  "witnesses_consumed": len(CP.WITNESSES),
                  "witness_sources": {n: v.get("source") for n, v in rec.get("starts", {}).items() if n != "JOINT-GREEDY"},
                  "unselected_diagnostic_fits": 0})
    elif fam in ("SEQ-12", "SEQ-21"):
        a.update({"search_paths": 1, "stages_per_path": 2, "candidates_compared": 1, "witnesses_consumed": 0,
                  "unselected_diagnostic_fits": int("old_rule_stage1" in rec.get("baseline_correction", {}))})
    elif fam in ("LOCAL", "FINE-TASK"):
        a.update({"search_paths": 1, "stages_per_path": 2, "candidates_compared": 1, "witnesses_consumed": 0,
                  "unselected_diagnostic_fits": 0})
    else:
        a.update({"search_paths": 0, "candidates_compared": 1, "witnesses_consumed": 0, "unselected_diagnostic_fits": 0})
    if s.get("defined"):
        a.update({"sweeps_total": s["totals"]["sweeps"], "accepted_exchanges_total": s["totals"]["accepted_exchanges"],
                  "merges_logged_total": s["totals"]["merges_logged"],
                  "refine_merges_total": s["totals"]["refine_merges"]})
    a["note"] = ASYMMETRY_NOTE
    return a


def asymmetry_table(records):
    """records: {(seed, config_id): qpc-format record} (new cbp fits and reused qpc units alike). One row per
    (seed, lambda) with JOINT's own and witness-inclusive CPU against each sequential arm and LOCAL."""
    rows = []
    for k in SEEDS:
        for lam in LAMS:
            r = {"seed": k, "lam": lam}
            have = {f: records.get((k, config_id(f, lam))) for f in FAMILIES}
            ft = records.get((k, config_id("FINE-TASK")))
            if not all(have.values()) or ft is None:
                r["complete"] = False
                rows.append(r)
                continue
            cpu = {f: float(have[f].get("cpu_seconds", 0.0)) for f in FAMILIES}
            w = float(ft.get("cpu_seconds", 0.0)) + cpu["LOCAL"] + cpu["SEQ-12"] + cpu["SEQ-21"]
            r.update({"complete": True, "cpu_s": cpu, "fine_task_cpu_s": float(ft.get("cpu_seconds", 0.0)),
                      "joint_cpu_s_own": cpu["JOINT"], "joint_cpu_s_incl_witness_fits": cpu["JOINT"] + w,
                      "joint_over_seq12_own": cpu["JOINT"] / cpu["SEQ-12"] if cpu["SEQ-12"] > 0 else None,
                      "joint_over_seq21_own": cpu["JOINT"] / cpu["SEQ-21"] if cpu["SEQ-21"] > 0 else None,
                      "candidates": {f: asymmetry_from_record(have[f])["candidates_compared"] for f in FAMILIES},
                      "sweeps_total": {f: asymmetry_from_record(have[f]).get("sweeps_total") for f in FAMILIES},
                      "accepted_exchanges_total": {f: asymmetry_from_record(have[f]).get("accepted_exchanges_total")
                                                   for f in FAMILIES}})
            rows.append(r)
    return {"rows": rows, "note": ASYMMETRY_NOTE}


def section13(rec, pair, out, T, tr, S_fit, lam, recon=None):
    """Prompt section 13 diagnostics for one code, every statistic from the deployed release on the fitting rows."""
    if recon is None:
        recon = reconstruct_fit_statistics(pair, out, T, tr, S_fit, lam)
    t1, t2 = recon["_tokens"][1], recon["_tokens"][2]
    sp = _sparsity_summary(t1, t2, pair)
    out13 = {"alphabets": {"full": {"r1": pair.p1.T, "r2": pair.p2.T, "pair": pair.p1.T * pair.p2.T},
                           "occupied_fit": {k: sp[k]["occupied"] for k in ("r1", "r2", "pair")}},
             "states_per_predicted_class": {f"r{i}": {"tokens": pol.tokens_per_class(),
                                                       "occupied_fit": pol.effective_states_per_class()}
                                            for i, pol in ((1, pair.p1), (2, pair.p2))},
             "total_occupied_states_fit": int(sum(pair.p1.effective_states_per_class())
                                              + sum(pair.p2.effective_states_per_class())),
             "entropy_nats_fit": {k: sp[k]["entropy_nats"] for k in ("r1", "r2", "pair")},
             "singletons_fit": {k: sp[k]["singleton_cells"] for k in ("r1", "r2", "pair")},
             "lt5_fit": {k: sp[k]["lt5_cells"] for k in ("r1", "r2", "pair")},
             "unseen_fraction_fit": {k: sp[k]["unseen_fraction"] for k in ("r1", "r2", "pair")},
             "support_per_class_fit": {"r1": _support(pair.p1, t1), "r2": _support(pair.p2, t2)},
             "fallback_tokens": {"r1": [int(t) for t in np.flatnonzero(pair.p1.token_fallback)],
                                 "r2": [int(t) for t in np.flatnonzero(pair.p2.token_fallback)]},
             "objective_terms_deployed": recon["terms"],
             "lam": lam}
    if "sparsity_fit" in rec:
        q = {k: {x: rec["sparsity_fit"][k][x] for x in sp[k]} for k in sp}
        out13["sparsity_equals_qpc_receipt"] = bool(q == sp)
    if "final" in rec:
        out13["objective_terms_record"] = {k: rec["final"][k] for k in term_keys(lam) if k in rec["final"]}
        out13["deployed_vs_record_final"] = compare_terms(recon["terms"], rec["final"], term_keys(lam))
    elif "fit_distortion" in rec:
        out13["deployed_vs_record_fit_distortion"] = compare_terms(recon["terms"], rec["fit_distortion"], ("D1", "D2"))
    if "row_level_check" in rec:
        out13["deployed_vs_qpc_row_level_check"] = compare_terms(recon["terms"], rec["row_level_check"],
                                                                 ("D1", "D2", "I1", "I2", "I12"))
    if lam is not None and "final" in rec:
        f = rec["final"]
        out13["privacy_term"] = {"lam_weighted_local": lam * (f["I1"] + f["I2"]) / 2,
                                 "lam_weighted_joint": lam * ((f["I1"] + f["I2"]) / 2 + f["I12"]),
                                 "distortion": f["D1"] + f["D2"]}
    pn = rec.get("perm_null_mi_fit")
    if pn:
        out13["mi_vs_permutation_null"] = {
            k: {"fitted": recon["terms"][k], "null_mean": pn[k]["mean"], "null_q95": pn[k]["q95"],
                "null_max": pn[k]["max"], "fitted_minus_null_mean": recon["terms"][k] - pn[k]["mean"]}
            for k in ("I1", "I2", "I12")} | {"n_perm": pn["n_perm"], "seed": pn["seed"]}
    out13["reconstruction"] = {k: v for k, v in recon.items() if k not in ("_tokens", "terms")}
    out13["search"] = search_summary(rec)
    out13["note"] = ("Fitting-row statistics only (OSF_DEFENSE_FIT). True-label log loss and Brier versus teacher KL "
                     "come from the inner utility stage, not from fitting; fitted plug-in MI is a training criterion, "
                     "not a privacy bound.")
    return out13


# ----------------------------------------------------------------------------------------------- code version
def code_version(family):
    """Current hashes of the qpc/dpc code that produced and re-encodes the unit versus the qpc lock pins."""
    lockB = json.loads((WT / QPC_PKG / "STAGE_B_LOCK.json").read_text())["code_files"]
    lockA = json.loads((WT / QPC_PKG / "STAGE_A_LOCK.json").read_text())["code_files"]
    files = {}
    for f, pin in QPC_STAGE_B_PINS.items():
        cur = sha_file(WT / f)
        files[f] = {"current": cur, "stage_b_lock": lockB.get(f), "registered_pin": pin,
                    "equal": bool(cur == lockB.get(f) == pin)}
    gated = list(STAGE_B_GATED) + [f for f in QPC_STAGE_B_PINS if f not in STAGE_B_GATED]
    out = {"files": files, "gated_files": gated, "ok": bool(all(files[f]["equal"] for f in gated))}
    if family == "DIRECT-TASK":
        a = {f: {"current": sha_file(WT / f), "stage_a_lock": lockA.get(f)} for f in STAGE_A_GATED}
        for v in a.values():
            v["equal"] = bool(v["current"] == v["stage_a_lock"])
        out["stage_a"] = a
        out["ok"] = bool(out["ok"] and all(v["equal"] for v in a.values()))
    return out


def code_hashes():
    fs = ["cbp/fit.py"] + list(QPC_STAGE_B_PINS)
    return {f: sha_file(WT / f) for f in fs}


# ----------------------------------------------------------------------------------------------- fitting
def _policy_text(files):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "policy.json"
        files["policy.json"](p)
        return p.read_text()


def fit_unit(family, fine_dict, T, tr, S_fit, lam, meta, witnesses=None, *, refit_reason=None):
    """One NEW privacy mapping-pair unit at (8, 64): ``qpc.compress.fit_unit(family, fine_dict, T, tr, S_fit, 8, 64,
    lam, meta, witnesses=witnesses)`` unchanged; the returned (record, files) are qpc's, with one added record key
    "cbp" (section 13 diagnostics from the deployed release, the computational-asymmetry receipt, code hashes).

    Refusals: a family outside LOCAL/SEQ-12/SEQ-21/JOINT; a lambda off the grid; a REUSED endpoint lambda (0.01, 0.1)
    unless ``refit_reason`` documents why the admitted unit could not be reused (recorded as ENDPOINT_REFIT); JOINT
    without exactly the four witness policy.json dicts (FINE-TASK, LOCAL, SEQ-12, SEQ-21 at the same lambda; qpc
    validates their fine partitions, family, caps and lambda); witnesses for a non-JOINT family."""
    if family not in FAMILIES:
        raise ValueError(f"REFUSED: {family!r} is not a cbp privacy family {FAMILIES}; references are reused, not refit")
    if lam is None or float(lam) not in LAMS:
        raise ValueError(f"REFUSED: lambda {lam!r} is not on the registered grid {LAMS}")
    lam = float(lam)
    status = status_of(lam)
    if status == "REUSED":
        if not (isinstance(refit_reason, str) and refit_reason.strip()):
            raise ValueError(f"REFUSED: lambda {g(lam)} units are admitted qpc endpoints reused after parity; a refit "
                             "needs an explicit refit_reason (recorded as ENDPOINT_REFIT)")
        status = "ENDPOINT_REFIT"
    if family == "JOINT":
        if not isinstance(witnesses, dict) or set(witnesses) != set(WITNESS_FAMILIES) or \
                any(witnesses[f] is None for f in WITNESS_FAMILIES):
            raise ValueError(f"REFUSED: JOINT needs exactly the four witness policy.json dicts {WITNESS_FAMILIES}")
    elif witnesses:
        raise ValueError("REFUSED: witnesses are passed to JOINT only")
    cid = config_id(family, lam)
    meta = dict(meta or {})
    if meta.get("config") not in (None, cid):
        raise ValueError(f"meta config {meta.get('config')} differs from {cid}")
    if meta.get("teacher", TEACHER) != TEACHER:
        raise ValueError("REFUSED: the cbp bank uses the frozen U teacher only")
    rec, files = CP.fit_unit(family, fine_dict, T, tr, S_fit, M1, M2, lam, meta, witnesses=witnesses)
    c0, t0 = time.process_time(), time.perf_counter()
    pair = RL.PolicyPair.from_dict(json.loads(_policy_text(files)))
    if pair.fingerprint() != rec["pair_fingerprint"]:
        raise AssertionError("written policy.json differs from the fitted pair")
    out = files["release.npz"]
    recon = reconstruct_fit_statistics(pair, out, T, tr, S_fit, lam)
    if not recon["structure_ok"]:
        raise AssertionError("deployed-release reconstruction failed: " + json.dumps(
            {i: recon[f"r{i}"] for i in (1, 2)}, default=str))
    s13 = section13(rec, pair, out, T, tr, S_fit, lam, recon=recon)
    rec["cbp"] = {"schema": "cbp-fit-v1", "config": cid, "bank_status": status, "refit_reason": refit_reason,
                  "qpc_call": {"function": "qpc.compress.fit_unit", "m1": M1, "m2": M2, "lam": lam,
                               "witness_slots": sorted(witnesses) if witnesses else []},
                  "section13": s13, "computational_asymmetry": asymmetry_from_record(rec),
                  "code_sha256": code_hashes()}
    rec["cbp"]["augment_cpu_seconds"] = time.process_time() - c0
    rec["cbp"]["augment_wall_seconds"] = time.perf_counter() - t0
    KM.json_safe(rec)
    return rec, files


def aliases(policies):
    """{config_id: qpc.PolicyPair or policy.json dict} of ONE seed -> qpc.compress.find_aliases plus counts."""
    pp = {c: (p if isinstance(p, RL.PolicyPair) else RL.PolicyPair.from_dict(p)) for c, p in policies.items()}
    al = CP.find_aliases(pp)
    return {"aliases": al, "pair_alias_count": int(sum(1 for c, a in al["pair"].items() if a != c)),
            "pair_aliases": {c: a for c, a in al["pair"].items() if a != c}}


# ----------------------------------------------------------------------------------------------- parity (reuse)
PARITY_GATES = ("complete_json_hashes", "policy_integrity", "config_registered_reusable", "config_consistent",
                "fingerprint_matches_record", "row_id_equal", "release_bitwise", "class_preservation_all_rows",
                "fitting_statistics_exact", "fitting_terms_within_rtol", "fine_partition", "code_version")


def _bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())


def endpoint_parity(unit_dir, T, tr, S_fit, fine_dict, *, expected_config=None, meta=None, witness_records=None):
    """Parity record for a REUSED qpc unit (lambda 0.01/0.1 LOCAL, SEQ-12, SEQ-21, JOINT; DIRECT-TASK, FINE-TASK,
    CLASS). Nothing is refit and nothing is written. Gates (all must hold for "ok"):

      complete_json_hashes        every file matches the unit's COMPLETE.json (jcv.finalize.unit_complete)
      policy_integrity            policy.json loads as qpc.PolicyPair (prototypes recomputed, fingerprints checked)
      config_registered_reusable  the config is one of the 8 reusable privacy configs or the 3 references
      config_consistent           record config == policy config == config_id(family, lam) (== expected_config)
      fingerprint_matches_record  policy fingerprint == the record's pair fingerprint
      row_id_equal                stored row_id == T["row_id"]
      release_bitwise             qpc.stagea.encode_all(pair, T) equals the stored release.npz key by key (same key set,
                                  dtype, shape and bytes)
      class_preservation_all_rows stored and re-encoded decisions == teacher decisions on ALL rows; decoded argmax ==
                                  decision
      fitting_statistics_exact    deployed-release reconstruction on the fitting rows: alphabets, token counts == stored
                                  n_t, token means within MEAN_ATOL, decoded vectors bitwise == prototypes, decisions
      fitting_terms_within_rtol   D1, D2, I1, I2, I12 (+ F values) from the stored release vs the record's "final" terms
                                  (DIRECT-TASK: D1, D2 vs its "fit_distortion"), |diff| <= 1e-12 * max(|a|, |b|)
      fine_partition              Stage B units: both assignment partitions' fingerprints == the admitted fine__s{k}
                                  (fine_dict) and == the record's fine_fingerprints; DIRECT-TASK: not applicable (its own
                                  Stage A partition), instead the partitions match the record's receipts
      code_version                current qpc/compress.py and qpc/release.py (and the dpc/qpc closure) == the qpc
                                  STAGE_B_LOCK pins (DIRECT-TASK also the STAGE_A_LOCK pins)
    plus, when given: binding (meta's teacher_model_sha256 / feature_names_sha256 == the policy binding) and, for JOINT,
    witness_dominance (every final_minus_witness <= 0, every witness passed in, and -- with witness_records -- each
    unchanged witness F_joint matches that witness unit's final F_joint within PARITY_RTOL)."""
    from jcv.finalize import unit_complete
    unit_dir = Path(unit_dir)
    res = {"schema": "cbp-endpoint-parity-v1", "unit": unit_dir.name, "refit": False, "rtol": PARITY_RTOL,
           "mean_atol": MEAN_ATOL}
    checks = {}
    try:
        checks["complete_json_hashes"] = bool(unit_complete(unit_dir))
        rec = json.loads((unit_dir / "record.json").read_text())
        polz = json.loads((unit_dir / "policy.json").read_text())
        with np.load(unit_dir / "release.npz", allow_pickle=False) as z:
            stored = {k: z[k] for k in z.files}
    except Exception as e:  # noqa: BLE001 -- any unreadable file is a parity failure, reported with its cause
        res.update({"ok": False, "status": "PARITY_FAIL", "error": f"unreadable unit ({e.__class__.__name__}: {e})",
                    "checks": checks, "failed": ["unreadable"]})
        return res
    try:
        pair = RL.PolicyPair.from_dict(polz)
        checks["policy_integrity"] = True
    except Exception as e:  # noqa: BLE001 -- any load failure is an integrity failure
        checks["policy_integrity"] = False
        res.update({"ok": False, "status": "PARITY_FAIL", "error": f"policy integrity ({e})", "checks": checks,
                    "failed": [k for k, v in checks.items() if v is False]})
        return res
    fam, lam = pair.family, pair.config.get("lam")
    cid = rec.get("config")
    if isinstance(cid, dict):            # a Stage A pair_record before the runner's update carries the config dict
        cid = cid.get("config")
    res.update({"config": cid, "family": fam, "lam": lam})
    checks["config_registered_reusable"] = bool(cid in reusable_ids())
    try:
        own = config_id(fam, lam)
    except ValueError:
        own = None
    checks["config_consistent"] = bool(cid == pair.config.get("config") == own and
                                       (expected_config is None or cid == expected_config) and
                                       pair.config.get("m1") == (1 if fam == "CLASS" else M1) and
                                       pair.config.get("m2") == (1 if fam == "CLASS" else M2))
    fp_rec = rec.get("pair_fingerprint", rec.get("fingerprint"))
    fp_pr = rec.get("pair_record", {}).get("fingerprint", fp_rec)
    checks["fingerprint_matches_record"] = bool(pair.fingerprint() == fp_rec == fp_pr)
    res["pair_fingerprint"] = pair.fingerprint()
    checks["row_id_equal"] = bool(np.array_equal(stored.get("row_id"), np.asarray(T["row_id"])))
    # re-encode with qpc.release (encode_all raises on a class-preservation failure)
    try:
        reenc = SA.encode_all(pair, T)
        reenc_err = None
    except (AssertionError, ValueError) as e:
        reenc, reenc_err = None, str(e)
    if reenc is None:
        checks["release_bitwise"] = False
        res["release_bitwise_detail"] = {"error": reenc_err}
    else:
        det = {k: _bitwise(reenc[k], stored[k]) if k in stored else False for k in reenc}
        det["_same_key_set"] = bool(set(reenc) == set(stored))
        checks["release_bitwise"] = bool(all(det.values()))
        res["release_bitwise_detail"] = det
    d = {i: np.asarray(T[f"d{i}"]) for i in (1, 2)}
    cp = {}
    for i in (1, 2):
        h = np.asarray(stored.get(f"hard{i}"))
        q = np.asarray(stored.get(f"q{i}"))
        cp[f"r{i}"] = {"rows": int(d[i].shape[0]),
                       "stored_failures": int(np.sum(h != d[i])) if h.shape == d[i].shape else None,
                       "stored_argmax_failures": int(np.sum(q.argmax(1) != h)) if q.ndim == 2 and
                       q.shape[0] == h.shape[0] else None,
                       "reencoded_failures": None if reenc is None else int(np.sum(reenc[f"hard{i}"] != d[i]))}
    checks["class_preservation_all_rows"] = bool(reenc is not None and all(
        v["stored_failures"] == 0 and v["stored_argmax_failures"] == 0 and v["reencoded_failures"] == 0
        for v in cp.values()))
    res["class_preservation"] = cp
    # deployed fitting statistics and terms
    try:
        recon = reconstruct_fit_statistics(pair, stored, T, tr, S_fit, lam)
        checks["fitting_statistics_exact"] = bool(recon["structure_ok"])
        s13 = section13(rec, pair, stored, T, tr, S_fit, lam, recon=recon)
        if "deployed_vs_record_final" in s13:
            checks["fitting_terms_within_rtol"] = bool(s13["deployed_vs_record_final"]["within_rtol"])
        elif "deployed_vs_record_fit_distortion" in s13:
            checks["fitting_terms_within_rtol"] = bool(s13["deployed_vs_record_fit_distortion"]["within_rtol"])
        else:
            checks["fitting_terms_within_rtol"] = False
        res["section13"] = s13
    except (KeyError, ValueError, IndexError) as e:
        checks["fitting_statistics_exact"] = False
        checks["fitting_terms_within_rtol"] = False
        res["reconstruction_error"] = f"{e.__class__.__name__}: {e}"
    # fine partitions
    if fam == "DIRECT-TASK":
        fpr = [rec.get("r1", {}).get("assignment_partition_fingerprint"),
               rec.get("r2", {}).get("assignment_partition_fingerprint")]
        checks["fine_partition"] = bool(fpr == [pair.p1.fine.fingerprint(), pair.p2.fine.fingerprint()])
        res["fine_partition_detail"] = {"applicable": False, "reason": "DIRECT-TASK routes through its own Stage A "
                                        "k-means partition (qpc dir__s{k}__r{i}__m{m}), not fine__s{k}",
                                        "own_partitions_match_record": checks["fine_partition"]}
    else:
        f1, f2 = PT.load_fine(fine_dict)
        adm = [f1.fingerprint(), f2.fingerprint()]
        pol = [pair.p1.fine.fingerprint(), pair.p2.fine.fingerprint()]
        checks["fine_partition"] = bool(pol == adm and rec.get("fine_fingerprints") == adm)
        res["fine_partition_detail"] = {"applicable": True, "admitted": adm, "policy": pol,
                                        "record": rec.get("fine_fingerprints")}
    cv = code_version(fam)
    checks["code_version"] = bool(cv["ok"])
    res["code_version"] = cv
    if meta is not None:
        b = RL.binding(pair)
        checks["binding"] = bool(all(b.get(k) == meta.get(k) for k in RL.BINDING_KEYS))
    if fam == "JOINT":
        dom = rec.get("witness_dominance", {})
        okd = bool(set(dom) == set(CP.WITNESSES) and all(v["final_minus_witness"] <= 0 for v in dom.values()))
        src = {n: v.get("source") for n, v in rec.get("starts", {}).items() if n != "JOINT-GREEDY"}
        okd = okd and all(v == "passed_in" for v in src.values()) and set(src) == set(CP.WITNESSES)
        wd = {}
        if witness_records is not None:
            for fw in CP.WITNESSES:
                wr = witness_records.get(fw)
                if wr is None or fw not in dom:
                    wd[fw] = {"present": False}
                    okd = False
                    continue
                key = "F_joint"
                if fw == "FINE-TASK":
                    t = wr["final"]
                    wv = CP.F_values(t, lam)["F_joint"]
                else:
                    wv = wr["final"][key]
                rr = _rel(wv, dom[fw]["witness_F_joint"])
                wd[fw] = {"present": True, "witness_unit_F_joint": wv, "joint_record_witness_F_joint":
                          dom[fw]["witness_F_joint"], "rel_diff": rr, "within_rtol": rr <= PARITY_RTOL}
                okd = okd and rr <= PARITY_RTOL
        checks["joint_witness_dominance"] = bool(okd)
        res["joint_witnesses"] = {"dominance": dom, "sources": src, "witness_unit_cross_check": wd or None}
    res["checks"] = checks
    res["failed"] = [k for k, v in checks.items() if v is not True]
    res["ok"] = not res["failed"]
    res["status"] = "PARITY_PASS" if res["ok"] else "PARITY_FAIL"
    KM.json_safe(res)
    return res


# ----------------------------------------------------------------------------------------------- synthetic fixture
def synthetic_teacher(N=39170, n_fit=15434, seed=0, unseen_class=5, underflow=195):
    """Real-shaped SYNTHETIC teacher outputs (generator copied from qpc/tests/test_method.py ``synth`` at d0c8a45,
    itself from dpc): income K=2 with exact 0/1 underflow rows and class imbalance; occupation K=6 with class 5 never
    predicted; binary S correlated with both scores. N = 39,170 rows (the five OSF roles), fitting rows = a fixed
    random subset of 15,434. No real row, label or SEX is read. Returns (T, tr, S_fit)."""
    rng = np.random.default_rng(seed)
    S = rng.integers(0, 2, N)
    z = rng.normal(-1.4 + 0.9 * S, 1.6, N)
    p = 1 / (1 + np.exp(-z))
    P1 = np.stack([1 - p, p], 1)
    if underflow:
        u = rng.choice(N, underflow, replace=False)
        P1[u[: underflow // 2]] = [1.0, 0.0]
        P1[u[underflow // 2:]] = [0.0, 1.0]
    bias = np.array([0.0, 0.6, 0.1, -1.2, 1.0, 0.2])
    if unseen_class is not None:
        bias[unseen_class] = -14.0
    L = rng.normal(size=(N, 6)) * 1.2 + bias + np.outer(S - 0.5, [0.8, -0.6, 0.3, 0.0, -0.4, 0.0])
    E = np.exp(L - L.max(1, keepdims=True))
    P2 = E / E.sum(1, keepdims=True)
    tr = np.sort(np.random.default_rng([seed, 7]).choice(N, n_fit, replace=False))
    T = {"row_id": np.arange(N, dtype=np.int64) * 3 + 1, "p1": P1, "p2": P2, "d1": P1.argmax(1).astype(np.int64),
         "d2": P2.argmax(1).astype(np.int64)}
    return T, tr, S[tr].astype(np.int64)


def save_unit(unit_dir, files, record):
    """Atomic unit write in the qpc runner format (jcv.finalize.save_unit; npz dicts via np.savez_compressed)."""
    from jcv.finalize import save_unit as _save

    def writer(v, fname):
        if callable(v):
            return v
        if isinstance(v, dict) and fname.endswith(".npz"):
            return lambda p: np.savez_compressed(p, **{k: np.asarray(a) for k, a in v.items()})
        if isinstance(v, (dict, list)):
            return lambda p: p.write_text(json.dumps(v, allow_nan=False))
        return lambda p: p.write_text(str(v))
    _save(Path(unit_dir), {f: writer(v, f) for f, v in files.items()}, record)


def _dir_bytes(d):
    return int(sum(p.stat().st_size for p in Path(d).rglob("*") if p.is_file()))


def _maxrss_bytes():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def _timed(fn):
    c0, t0 = time.process_time(), time.perf_counter()
    out = fn()
    return out, time.process_time() - c0, time.perf_counter() - t0


def synthetic_seed_timing(seed, units_root, fine_caps=None):
    """Fit the synthetic analogue of the admitted qpc units (setup), the 48/3 = 16 NEW units of this seed and the
    parity of its 8 + 3 REUSED units. Returns per-item CPU/wall and the unit sizes."""
    fine_caps = dict(fine_caps or PT.CAPS)
    T, tr, S_fit = synthetic_teacher(seed=seed)
    meta0 = {"teacher": TEACHER, "seed": seed, "teacher_model_sha256": "a" * 64, "feature_names_sha256": "b" * 64}
    root = Path(units_root)
    out = {"seed": seed, "setup": {}, "new": {}, "parity": {}, "bytes": {}}
    # setup: fine partitions (admitted fine__s{k} analogue)
    (r_f, files_f), c, w = _timed(lambda: PT.fine_unit(T, tr, caps=fine_caps))
    with tempfile.TemporaryDirectory() as dd:
        files_f["fine.json"](Path(dd) / "fine.json")
        fine = json.loads((Path(dd) / "fine.json").read_text())
    out["setup"]["fine_partitions"] = {"cpu_s": c, "wall_s": w, "F": r_f["F"]}
    pols = {}

    def _store(cid, rec, files):
        rec.update({"seed": seed, "config": cid})             # as the qpc runner's record update
        u = root / unit_name(seed, cid)
        save_unit(u, files, rec)
        pols[cid] = json.loads((u / "policy.json").read_text())
        out["bytes"][cid] = _dir_bytes(u)
        return u
    # setup: DIRECT-TASK i8o64 (Stage A analogue)
    def _direct():
        p1, _ = SA.recipient_fit(T["p1"][tr], T["d1"][tr], 2, M1, 1)
        p2, _ = SA.recipient_fit(T["p2"][tr], T["d2"][tr], 6, M2, 2)
        return SA.pair_unit(p1, p2, T, {**meta0, "config": config_id("DIRECT-TASK")}, tr=tr)
    (rec, files), c, w = _timed(_direct)
    _store(config_id("DIRECT-TASK"), rec, files)
    out["setup"]["DIRECT-TASK"] = {"cpu_s": c, "wall_s": w}
    for fam in ("CLASS", "FINE-TASK"):
        cid = config_id(fam)
        (rec, files), c, w = _timed(lambda: CP.fit_unit(fam, fine, T, tr, S_fit, M1, M2, None, {**meta0, "config": cid}))
        _store(cid, rec, files)
        out["setup"][fam] = {"cpu_s": c, "wall_s": w}
    # setup: admitted endpoints (lambda 0.01, 0.1) with plain qpc, as the qpc runner did
    for lam in REUSED_LAMS:
        for fam in FAMILIES:
            cid = config_id(fam, lam)
            wit = {f: pols[c2] for f, c2 in witness_configs(lam).items()} if fam == "JOINT" else None
            (rec, files), c, w = _timed(lambda: CP.fit_unit(fam, fine, T, tr, S_fit, M1, M2, lam,
                                                            {**meta0, "config": cid}, witnesses=wit))
            _store(cid, rec, files)
            out["setup"][cid] = {"cpu_s": c, "wall_s": w}
    # mandatory: the 16 NEW fits of this seed through cbp.fit.fit_unit (incl. the unit write)
    for lam in NEW_LAMS:
        for fam in FAMILIES:
            cid = config_id(fam, lam)
            wit = {f: pols[c2] for f, c2 in witness_configs(lam).items()} if fam == "JOINT" else None

            def _job():
                rec, files = fit_unit(fam, fine, T, tr, S_fit, lam, {**meta0, "config": cid}, witnesses=wit)
                _store(cid, rec, files)
                return rec
            rec, c, w = _timed(_job)
            out["new"][cid] = {"cpu_s": c, "wall_s": w, "qpc_fit_cpu_s": rec["cpu_seconds"],
                               "augment_cpu_s": rec["cbp"]["augment_cpu_seconds"],
                               "search": {k: rec["cbp"]["section13"]["search"].get(k) for k in
                                          ("converged_all", "not_converged", "unresolved_local_optima", "winner")},
                               "deployed_vs_final_max_rel": rec["cbp"]["section13"]["deployed_vs_record_final"][
                                   "max_rel_diff"]}
    # mandatory: parity of the 8 reused privacy units and the 3 references (no refit)
    for cid in reusable_ids():
        pr, c, w = _timed(lambda: endpoint_parity(root / unit_name(seed, cid), T, tr, S_fit, fine,
                                                  expected_config=cid, meta=meta0))
        out["parity"][cid] = {"cpu_s": c, "wall_s": w, "ok": pr["ok"], "failed": pr["failed"],
                              "max_rel": (pr.get("section13", {}).get("deployed_vs_record_final")
                                          or pr.get("section13", {}).get("deployed_vs_record_fit_distortion")
                                          or {}).get("max_rel_diff")}
    out["aliases"] = aliases(pols)["pair_aliases"]
    return out


def _stats(xs):
    xs = [float(x) for x in xs]
    return {"units": len(xs), "cpu_total_s": round(sum(xs), 3), "cpu_mean_s": round(float(np.mean(xs)), 3),
            "cpu_max_s": round(max(xs), 3)} if xs else {"units": 0}


def timing(seeds=SEEDS, out_path=None):
    """Full registered bank on SYNTHETIC real-shaped data. Writes TIMING.json["fitting"] (other keys preserved)."""
    import platform
    t_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    c0, w0 = time.process_time(), time.perf_counter()
    per_seed = []
    with tempfile.TemporaryDirectory(prefix="cbp_fit_timing_") as d:
        for k in seeds:
            per_seed.append(synthetic_seed_timing(k, Path(d) / f"s{k}"))
    total_cpu, total_wall = time.process_time() - c0, time.perf_counter() - w0
    fam_of = lambda cid: cid.split("|")[1]  # noqa: E731
    new_by_fam = {f: _stats([v["cpu_s"] for s in per_seed for cid, v in s["new"].items() if fam_of(cid) == f])
                  for f in FAMILIES}
    par_by = {}
    for s in per_seed:
        for cid, v in s["parity"].items():
            par_by.setdefault(fam_of(cid), []).append(v["cpu_s"])
    parity_by_fam = {f: _stats(v) for f, v in sorted(par_by.items())}
    new_cpu = sum(v["cpu_s"] for s in per_seed for v in s["new"].values())
    par_cpu = sum(v["cpu_s"] for s in per_seed for v in s["parity"].values())
    setup_cpu = sum(v["cpu_s"] for s in per_seed for v in s["setup"].values())
    unit_bytes = [b for s in per_seed for b in s["bytes"].values()]
    mandatory = new_cpu + par_cpu
    n_seeds = len(list(seeds))
    scale = 3.0 / n_seeds
    rec = {
        "owner": "role B (optimizer engineer)",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "started_at": t_start,
        "data": "SYNTHETIC ONLY (cbp.fit.synthetic_teacher: 39,170 rows = the five OSF role sizes, a fixed random "
                "15,434 = fitting rows; income K=2 with 195 exact 0/1 rows and class imbalance; occupation K=6 with "
                "class 5 never predicted (5 predicted classes + 1 absent); binary S correlated with both scores). "
                "Fine partitions income 32 / occupation 128 per predicted class (qpc.partition.CAPS). No Adult row, "
                "task label or SEX was read.",
        "environment": {"threads": "OMP_NUM_THREADS=1 (+OPENBLAS/MKL=1), one process",
                        "semaphore": "run under cbp.sema (label B:fit-timing)",
                        "python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine()},
        "measure": "process CPU seconds (time.process_time) inside the worker; wall alongside; peak RSS of the worker",
        "seeds": list(seeds),
        "bank": bank_counts(),
        "mandatory_work": {
            "new_fits": {"units": sum(len(s["new"]) for s in per_seed), "cpu_total_s": round(new_cpu, 2),
                         "per_family": new_by_fam,
                         "note": "cbp.fit.fit_unit = qpc.compress.fit_unit (incl. its row-level recomputation, "
                                 "sparsity and 100-permutation MI null) + section 13 / deployed-release "
                                 "reconstruction + unit write"},
            "endpoint_parity": {"units": sum(len(s["parity"]) for s in per_seed), "cpu_total_s": round(par_cpu, 2),
                                "per_family": parity_by_fam, "all_pass": bool(all(v["ok"] for s in per_seed
                                                                                  for v in s["parity"].values())),
                                "note": "re-encode ALL rows from policy.json + teacher, bitwise vs release.npz; "
                                        "deployed-release fitting statistics; fine/code/binding checks; no refit"},
            "cpu_total_s": round(mandatory, 2)},
        "setup_not_part_of_cbp_work": {
            "what": "synthetic analogues of the ADMITTED qpc artifacts (fine partitions, DIRECT-TASK, FINE-TASK, CLASS, "
                    "lambda 0.01/0.1 privacy units), fitted with plain qpc so that the new JOINT units have witnesses "
                    "and the parity step has units to check; in the real study these are admitted, not refit",
            "cpu_total_s": round(setup_cpu, 2),
            "fine_partitions_cpu_s": [round(s["setup"]["fine_partitions"]["cpu_s"], 2) for s in per_seed],
            "fine_F": per_seed[0]["setup"]["fine_partitions"]["F"] if per_seed else None},
        "per_seed": [{"seed": s["seed"],
                      "new": {cid: {k: v[k] for k in ("cpu_s", "wall_s", "qpc_fit_cpu_s", "augment_cpu_s")}
                              for cid, v in s["new"].items()},
                      "new_search": {cid: v["search"] for cid, v in s["new"].items()},
                      "new_deployed_vs_final_max_rel": {cid: v["deployed_vs_final_max_rel"] for cid, v in
                                                        s["new"].items()},
                      "parity": s["parity"], "setup_cpu_s": {k: round(v["cpu_s"], 3) for k, v in s["setup"].items()},
                      "pair_aliases": s["aliases"]} for s in per_seed],
        "worker_cpu_total_s": round(total_cpu, 2), "worker_wall_total_s": round(total_wall, 2),
        "peak_rss_bytes": _maxrss_bytes(),
        "private_output_estimate": {"bytes_per_unit_mean": int(np.mean(unit_bytes)) if unit_bytes else None,
                                    "bytes_per_unit_max": int(max(unit_bytes)) if unit_bytes else None,
                                    "new_units_3_seeds_bytes": int(np.mean(unit_bytes) * 48) if unit_bytes else None,
                                    "note": "policy.json + release.npz (all rows) + record.json + COMPLETE.json"},
        "projection": {
            "mandatory_cpu_s_3_seeds_synthetic": round(mandatory * scale, 1),
            "real_over_synthetic_reference": "qpc: synthetic Stage B bank 83.8 CPU-s vs real partition + fit 85 CPU-s "
                                             "(COST_AND_CLOSEOUT.md: 12 + 73) -> ratio about 1.0",
            "projected_real_cpu_s": round(mandatory * scale, 1),
            "projected_real_cpu_s_conservative_x2": round(2 * mandatory * scale, 1),
            "projected_real_cpu_h_conservative_x2": round(2 * mandatory * scale / 3600, 4),
            "budget_context": "20 CPU-h study total, 4 CPU-h reserve; fitting is a negligible share",
            "scheduling": "one or two shards under cbp.sema; per seed: the 12 LOCAL/SEQ units of the four new lambdas "
                          "before their JOINT units (JOINT needs its three same-lambda witnesses plus the admitted "
                          "FINE-TASK); parity of the 33 reused units is independent and can run first"},
        "recommendation": {"bank": "FULL (72 privacy units: 24 reused after parity + 48 new; 9 references reused)",
                           "reduction_applied": False,
                           "reason": "measured mandatory fitting work is minutes of CPU, far below every budget line; "
                                     "no scientific choice depends on timing (no lambda pruning)"}}
    KM.json_safe(rec)
    if out_path is not None:
        p = Path(out_path)
        cur = json.loads(p.read_text()) if p.exists() else {}
        cur["fitting"] = rec
        tmp = p.with_name(p.name + ".tmp_fit")
        tmp.write_text(json.dumps(cur, indent=1, allow_nan=False) + "\n")
        tmp.replace(p)
    return rec


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m cbp.fit")
    ap.add_argument("cmd", choices=("bank", "timing"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(argv)
    if a.cmd == "bank":
        print(json.dumps({"counts": bank_counts(), "privacy_units": bank(), "references": reference_units(),
                          "new_fit_order": new_fit_order()}, indent=1, allow_nan=False))
        return 0
    import os
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1 (run under cbp.sema)")
    r = timing(tuple(int(x) for x in a.seeds.split(",")), a.out)
    print(json.dumps({k: r[k] for k in ("mandatory_work", "worker_cpu_total_s", "peak_rss_bytes", "projection")},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
