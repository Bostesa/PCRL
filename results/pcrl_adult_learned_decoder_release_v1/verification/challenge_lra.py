#!/usr/bin/env python3
"""Role E challenge harness (prompt section 18: "challenge the critical branches with your own tests").

This is NOT the independent verifier. It is an EXTERNAL process that imports the study's own lra modules (select,
family, assess, eval_lock) and feeds them the verifier's synthetic cases (JSON on stdin); it prints the study's answers
as JSON on stdout. replay_lra.py never imports lra: it launches this file as a subprocess and compares the answers with
its own independent implementation. Every write path of the study code is redirected to a temporary directory; no
private or public study file is written.

    <python> challenge_lra.py < cases.json > answers.json      (run by replay_lra.py inside its semaphore hold)
"""
import json
import os
import sys
import tempfile
from pathlib import Path

WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from lra import family as FAM  # noqa: E402
from lra import run as R  # noqa: E402
from lra import select as SEL  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="lra_challenge_"))
ORIG_PKG, ORIG_RUN = R.PKG, R.RUN


def _redirect():
    R.RUN = TMP
    R.PKG = TMP
    SEL.write_tables = lambda out: None
    R.event = lambda *a, **k: None
    SEL.inner_validation = lambda D: None
    SEL.same_map_fit_inner = lambda rows, p: None


def to_d_row(cid, spec):
    p = R.parse_id(cid)
    row = {"config": cid, "arm": SEL.arm(cid), "family": (p.get("base_family") or SEL.arm(cid)), "kind": p["kind"],
           "privacy_trained": SEL.privacy_trained(cid), "private_code": SEL.private_code(cid),
           "training": SEL.training(cid), "ok": bool(spec["ok"]),
           "technical_failure": [{"unit": "synthetic", "code": c, "detail": "challenge"} for c in spec.get("tech", [])],
           "fit_feasible": spec.get("fit_feasible"), "seeds": {}}
    for k, s in (spec.get("seeds") or {}).items():
        row["seeds"][int(k)] = {"auc": s["auc"], "ordinary": bool(s["ordinary"]),
                                "ordinary_shortfall": float(s["ordinary_shortfall"]),
                                "token_states": s.get("token_states"), "release_hash": s.get("release_hash"),
                                "canonical_hash": s.get("canonical_hash")}
    if spec["ok"]:
        row.update({k: spec[k] for k in ("mean_pair", "mean_v1", "mean_v2", "mean_sum_logloss", "mean_states",
                                         "ordinary_inner", "ordinary", "ordinary_shortfall")})
    return row


def run_selection(case):
    rows = {cid: to_d_row(cid, sp) for cid, sp in case["rows"].items()}
    ids = R.scored_ids()
    missing = sorted(set(ids) - set(rows))
    extra = sorted(set(rows) - set(ids))
    if missing:
        return {"error": "ids missing from the case", "missing": missing, "extra": extra}
    SEL.candidate_rows = lambda ids_, rp=None: {c: rows[c] for c in ids_}
    out = SEL.select_all(None)
    st = out["statuses"]
    keep = ("status", "config", "descriptive_config", "reason", "fallback_class", "fallback_rank_status", "winning",
            "identical_to_untrained")
    return {"ids": ids, "extra_ids_in_case": extra,
            "statuses": {x: {k: v.get(k) for k in keep} | {"aliases": {k: (v.get("aliases") or {}).get(k) for k in
                                                                      ("full", "representative",
                                                                       "representative_family",
                                                                       "representative_construction",
                                                                       "identical_to_untrained",
                                                                       "decided_by_config_id_tiebreak",
                                                                       "canonical_equivalent")}}
                         for x, v in st.items()},
            "resolved": out["resolved"], "role_aliases": out["role_aliases"],
            "role_canonical_equivalence": out["role_canonical_equivalence"]}


def run_fit_feasible(case):
    """F01: fit_feasible on synthetic fit records (R.done / R.rec / R.unit_for redirected to the case)."""
    cid = case["cid"]
    recs = {int(k): v for k, v in case["records"].items()}
    R.unit_for = lambda k, c: f"fit__s{k}__challenge"
    R.done = lambda u: recs.get(int(u.split("__")[1][1:]), {}).get("done", True)

    def rec(u):
        r_ = recs.get(int(u.split("__")[1][1:]), {})
        if r_.get("unreadable"):
            raise ValueError("unreadable record")
        return r_.get("record")
    R.rec = rec
    SEL.R = R
    v, why = SEL.fit_feasible(cid)
    return {"value": v, "detail": why}


def run_labels(case):
    lab, shown = FAM.overall_label(case["claims"], case["q"], case["technical_valid"], case.get("winning"),
                                   case["engineering_gate"], case.get("prefit_blocker"), case.get("disclosures"))
    return {"label": lab, "shown": shown}


def run_validity(case):
    from lra import assess as AS
    R.PKG, R.RUN = ORIG_PKG, ORIG_RUN               # verify_validity only READS the real gate result and locks
    try:
        r_ = AS.verify_validity(case["lock"])
        return {"refused": False, "receipt": r_}
    except SystemExit as e:
        return {"refused": True, "why": str(e)[:300]}
    finally:
        R.PKG, R.RUN = TMP, TMP


def main():
    _redirect()
    cases = json.load(sys.stdin)
    out = {}
    for name, c in cases.items():
        try:
            out[name] = {"selection": run_selection, "fit_feasible": run_fit_feasible, "label": run_labels,
                         "validity": run_validity}[c["kind"]](c)
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    json.dump(out, sys.stdout, default=str)


if __name__ == "__main__":
    main()
