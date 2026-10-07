"""[lra port of lcr/eval_lock.py at 091afc2: lcr->lra renames; later edits are listed in PORT_LOG.md and
REVIEW_FINDINGS_DISPOSITION.json (F04/F07/F10/F11/F13)]
EVALUATION_LOCK.json writer for lra (committed and pushed before the single assessment opening).

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m lra.eval_lock results/pcrl_adult_learned_decoder_release_v1/EVALUATION_LOCK.json

REFUSAL (F13, prompt sec. 13 stage 5): the lock is NOT written when technical_validity() fails -- unresolved technical
failures in the selection (any candidate's technical failure, a technically INVALID role), a failed or missing
real-data control verdict, failed inner validation, a missing or non-ADMITTED admission receipt, or an engineering gate
that is not ENGINEERING_READY and bound in the latest named lock (lra.run.engineering_ready; F10), or a
CORRECTNESS_LOCK / SCIENCE_LOCK that is missing or does not verify. There is no escape flag: any extra argument is
refused. Resolve the issue (engineering repair + a new pushed lock/amendment), then rebuild. A scientific ineligibility
(NO_ELIGIBLE_*, or INVALID only because a guard comparator has no nominee) is NOT a technical failure: such a bank is
locked with its fixed descriptive fallbacks, whose conjunctions can never pass.

Adapted from cbp/eval_lock.py at 7f3ec67. Freezes the inner-only statuses and resolved roles (P*, N*, J*, T*, C*,
C_pair*, Q), the scored list (prompt section 13), every scored unit with the sha256 of every unit file (release, decoder,
policy, inner), the composed source-reader bank (all 84 registered codes of each seed, containing every frozen composed
winner), the SEX prior hash, the assessment role manifest, attackers, lock/amendment hashes, locked code, the 37
endpoints, data-dependent role aliases and the global technical-validity inputs. Nothing here reads an assessment value.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from lra import family as FAM
from lra import lock as LK
from lra import run as R

ROLES = ("P*", "N*", "J*", "T*", "C*", "C_pair*", "Q")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prior_hash(D):
    tr = D["idx"]["DEFENSE_FIT"]
    c = np.bincount(np.asarray(D["sex"])[tr], minlength=2).astype(np.int64)
    return hashlib.sha256(c.tobytes()).hexdigest()


def _cfg(s):
    return (s or {}).get("config") or (s or {}).get("descriptive_config")


def scored_labels(S):
    """PROTOCOL.md section 13 (prompt section 13), fixed from the inner selection only: the seven roles (or their
    registered fallbacks, F04); U continuous and decisions alone (D0 CLASS) with its fixed D1 version U|CLASS|i1o1|D1
    (role F R-10: the cleanest same-token D0/D1 pair; DECISION_FLOOR_AND_FEASIBILITY.csv); the best D1 fixed-map privacy
    control and its paired
    D0 release; the best weighted privacy control; C-TASK; the five constrained arms; RAW-J, FARE, F0 and LEACE; plus
    the same-map D0 decoded versions of the nominated or fallback P* and of C-TASK (diagnostic releases for the
    decoder-only utility contrasts; selection.diagnostics.same_map_decoder_pairs). Duplicates (exact config aliases)
    are scored once; every role mapping is kept."""
    st, d = S["statuses"], S["diagnostics"]
    labels = [_cfg(st.get(x)) for x in ROLES]
    labels += ["SRC|U", R.d0_id("CLASS"), R.d1_id("CLASS")]       # decisions alone (D0 CLASS) + CLASS|D1 (F R-10)
    labels += [_cfg(d["best_d1_fixed_privacy"]), d["best_d1_fixed_privacy"].get("paired_d0")]
    labels += [_cfg(d["best_weighted_privacy"]), R.ctask_id()] + R.constrained_ids()
    labels += ["SRC|RAW-J_b0.3", "REF|F", "REF|F0", "REF|E"]
    for p in d.get("same_map_decoder_pairs") or []:
        labels += [p.get("d1"), p.get("d0")]
    return list(dict.fromkeys(x for x in labels if x))


def decoder_pair_labels(S):
    """Releases scored with the token-only / probability-only diagnostic families as well: both members of every
    registered same-map D1/D0 pair."""
    return {x for p in (S["diagnostics"].get("same_map_decoder_pairs") or []) for x in (p.get("d1"), p.get("d0")) if x}


def role_aliases(resolved, statuses=None):
    """Data-dependent aliases between roles (F07/F11): two roles alias iff they resolve to one configuration or to two
    configurations that are EXACT deployed-release aliases on all three seeds (the selection's per-role alias sets
    statuses[x]['aliases']['full'], computed on permitted fitting/inner rows for EVERY role including comparators and
    Q). Same-token/different-decoder releases are never aliases; canonical renamings are informational only."""
    full = {x: set((((statuses or {}).get(x) or {}).get("aliases") or {}).get("full") or ()) for x in ROLES}
    out = {}
    for i, a in enumerate(ROLES):
        for b in ROLES[i + 1:]:
            ca, cb = resolved.get(a), resolved.get(b)
            if not (ca and cb):
                continue
            if ca == cb:
                out[f"{a}=={b}"] = ca
            elif cb in full[a] or ca in full[b]:
                out[f"{a}=={b}"] = f"{ca} == {cb} (exact release alias)"
    return out


def alias_of_by_role(aliases):
    """Slot-level aliases implied by role aliases (the slots are kept; prose never counts one comparison twice)."""
    by = {}
    claims = {c: (n, m) for c, (n, m) in FAM.CLAIMS.items()}
    for e in FAM.PRIMARY:
        if e["claim"] == "Q":
            continue
        for e2 in FAM.PRIMARY:
            if e2["claim"] in ("Q", e["claim"]) or e2["id"] >= e["id"] or e2["kind"] != e["kind"]:
                continue
            if e.get("task") != e2.get("task") or e.get("view") != e2.get("view"):
                continue
            n1, m1 = claims[e["claim"]]
            n2, m2 = claims[e2["claim"]]
            same_nom = n1 == n2 or f"{n2}=={n1}" in aliases or f"{n1}=={n2}" in aliases
            same_ref = e["kind"] not in ("coalition", "local") or m1 == m2 or f"{m2}=={m1}" in aliases or \
                f"{m1}=={m2}" in aliases
            if same_nom and same_ref and e["id"] not in by:
                by[e["id"]] = e2["id"]
    return by


def _selection_technical(S):
    """Unresolved technical failures recorded by the inner selection (never a scientific ineligibility)."""
    out = []
    if S is None:
        return ["selection.json missing"]
    for c, f in sorted((S.get("technical_failures") or {}).items()):
        out.append({"candidate": c, "failures": f})
    for x in ROLES:
        s = (S.get("statuses") or {}).get(x) or {}
        if str(s.get("status", "")).startswith("INVALID_") and s.get("reason") != "MISSING_GUARD_COMPARATOR":
            out.append({"role": x, "status": s.get("status"), "reason": s.get("reason")})
    return out


def technical_validity(S=None):
    """Global technical validity (every failure blocks the lock, F13): real-data controls (AUDIT_PRELOCK_CHECKS.json),
    the NEW engineering gate (ENGINEERING_READY, bound in the latest named lock, on origin; F10), the CORRECTNESS_LOCK
    and SCIENCE_LOCK, inner validation, selection technical failures and admission."""
    from lra import lock as LK
    out = {"ok": True, "failures": []}
    p = R.PKG / "AUDIT_PRELOCK_CHECKS.json"
    if not p.exists():
        out["failures"].append("AUDIT_PRELOCK_CHECKS.json missing")
    else:
        v = json.loads(p.read_text()).get("verdict", {})
        out["controls_verdict_sha256"] = sha(p)
        if v.get("all_ok") is not True:
            out["failures"].append({"controls": v.get("failures") or "verdict.all_ok is not true"})
    ok, why = R.engineering_ready()
    gp = R.PKG / R.GATE_RESULT
    out["engineering_gate"] = {"verdict": json.loads(gp.read_text()).get("verdict") if gp.exists() else None,
                               "sha256": sha(gp) if gp.exists() else None, "ready": bool(ok), "detail": why}
    if not ok:
        out["failures"].append({"engineering_gate": why})
    locks = {}
    for n in ("CORRECTNESS_LOCK", "SCIENCE_LOCK"):
        lp = LK.PKG / f"{n}.json"
        locks[n] = sha(lp) if lp.exists() else None
        if not lp.exists():
            out["failures"].append(f"{n}.json missing")
    sl = LK.PKG / "SCIENCE_LOCK.json"
    if sl.exists():
        v = LK.verify_lock(sl, stage="select")
        out["science_lock_verify"] = {"ok": v["ok"], "mismatches": v["mismatches"][:10]}
        if not v["ok"]:
            out["failures"].append({"science_lock": v["mismatches"][:10]})
    out["locks_sha256"] = locks
    if S is None:
        sel = R.RUN / "selection.json"
        S = json.loads(sel.read_text()) if sel.exists() else None
    iv = (S or {}).get("inner_validation")
    out["inner_validation"] = iv
    if not (iv or {}).get("ok"):
        out["failures"].append({"inner_validation": iv})
    tf = _selection_technical(S)
    out["selection_technical_failures"] = tf
    if tf:
        out["failures"].append({"selection_technical_failures": tf[:10], "n": len(tf)})
    ad = R.PRIV / "admitted" / "ADMISSION_RECEIPT.json"
    if not ad.exists() or json.loads(ad.read_text()).get("verdict") != "ADMITTED":
        out["failures"].append("admission receipt missing or not ADMITTED")
    out["ok"] = not out["failures"]
    out["scope"] = "global: a failure here touches every claim (LABEL_TRUTH_TABLE.json failure_scope) and blocks the lock"
    return out


def refuse_unless_valid(L):
    """F13: no lock (and no assessment opening) on an unresolved technical failure; no escape flag."""
    tv = L.get("technical_validity") or {}
    if tv.get("ok") is not True:
        raise SystemExit("REFUSED: technical validity failed before the single assessment opening: "
                         f"{json.dumps(tv.get('failures'), default=str)[:2000]}; resolve it (engineering repair, a new "
                         "pushed lock or amendment) and rebuild; there is no override")
    return True


def build():
    from osf import data as OD
    from lra import data as DA
    S = json.loads((R.RUN / "selection.json").read_text())
    D = DA.load()
    labels = scored_labels(S)
    resolved = {x: _cfg(S["statuses"].get(x)) for x in ROLES}
    al = role_aliases(resolved, S["statuses"])
    if S.get("role_aliases") is not None and S["role_aliases"] != al:
        raise SystemExit("REFUSED: role aliases re-derived from the alias sets differ from selection.json")
    diag_pairs = decoder_pair_labels(S)
    comp = R.code_ids()
    lock = {"schema": "lra-evaluation-lock-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parent_commit": LK.git("rev-parse", "HEAD"),
            "locks_sha256": {p.name: sha(p) for p in sorted(LK.PKG.glob("*_LOCK.json"))
                             if p.name != "EVALUATION_LOCK.json"},
            "amendments_sha256": {p.name: sha(p) for p in sorted(LK.PKG.glob("AMENDMENT_A*.json"))},
            "locked_code_files": LK.locked_files(LK.latest()),
            "selection_sha256": sha(R.RUN / "selection.json"), "selection_public_sha256": sha(R.PKG / "SELECTION.json"),
            "statuses": {x: {kk: v.get(kk) for kk in ("status", "config", "descriptive_config", "descriptive_only",
                                                      "winning", "config_arm", "reason", "aliases",
                                                      "identical_to_untrained", "fallback_rank_status",
                                                      "fallback_class", "missing_guards")}
                         for x, v in S["statuses"].items()},
            "resolved": resolved, "role_aliases": al, "alias_of_by_role": alias_of_by_role(al),
            "role_canonical_equivalence": S.get("role_canonical_equivalence") or {},
            "release_identity_rows": S.get("release_identity_rows"),
            "decoder_ablation_pair": [_cfg(S["diagnostics"]["best_d1_fixed_privacy"]),
                                      S["diagnostics"]["best_d1_fixed_privacy"].get("paired_d0")],
            "same_map_decoder_pairs": S["diagnostics"].get("same_map_decoder_pairs") or [],
            "supplementary_z": FAM.Z_SUPPLEMENTARY,
            "scored_labels": labels, "technical_validity": technical_validity(S),
            "sex_prior_defense_fit_sha256": prior_hash(D),
            "assessment_role": {"name": "OSF_DEVELOPMENT_ASSESSMENT", **OD.manifest(D)["OSF_DEVELOPMENT_ASSESSMENT"]},
            "attackers": {"slate": "lra.audit: pinned FINAL slate + defence-aware readers on every primary contract; "
                                   "finite codes add cell-conditional readers on the exact token identity / tuple",
                          "fit": "AUDIT_FIT", "selection": "INNER_SELECTION (AUC primary, CE separate)",
                          "refits": [0, 1, 2], "coalition_bank": "pair + ignore-recipient-1 + ignore-recipient-2",
                          "composed_source_readers": "SRC|U composes, at the assessment, over every frozen source-winning "
                                                     f"composed reader of the inner bank (all {len(comp)} registered codes "
                                                     "per seed were composed at the inner stage) plus every scored code; "
                                                     "same-map D0 diagnostics are never composed; codes "
                                                     "that lost every inner comparison are not re-scored (exact: the final "
                                                     "audit re-selects on the same seed-0 inner fits)"},
            "endpoints": {"primary": [e["id"] for e in FAM.PRIMARY], "z": FAM.Z_PRIMARY, "B": FAM.B,
                          "boot_seed": FAM.BOOT_SEED, "size": FAM.PRIMARY_SIZE},
            "seeds": {}}
    abl_pair = diag_pairs | {x for x in (_cfg(S["diagnostics"]["best_d1_fixed_privacy"]),
                                         S["diagnostics"]["best_d1_fixed_privacy"].get("paired_d0")) if x}
    for k in R.SEEDS:
        score, files = {}, {}
        for c in labels:
            u = R.unit_for(k, c)
            score[c] = {"kind": R.parse_id(c)["kind"], "unit": u, "cid": c}
            if R.parse_id(c)["kind"] == "policy" and c not in abl_pair:
                score[c]["families"] = ["code"]        # complete interface only; token/prob diagnostics at the
                #                                        assessment only for the registered decoder-ablation pair
            names = (u,) if R.parse_id(c).get("diagnostic_only") else (u, R.inner_name(k, c))   # d0same: no inner unit
            for n in names:
                files[n] = json.loads((R.U(n) / "COMPLETE.json").read_text())["files"]
        for n in [f"tea__s{k}__U"] + [R.unit_for(k, c) for c in comp]:
            files.setdefault(n, json.loads((R.U(n) / "COMPLETE.json").read_text())["files"])
        from lra import audit as AU
        frozen = list(AU.composed_freeze_list(k, "U"))
        scored_codes = [c for c in labels if R.parse_id(c)["kind"] == "policy" and not R.parse_id(c).get("diagnostic_only")]
        comp_k = [c for c in comp if c in set(frozen) | set(scored_codes)]
        for c in comp_k:
            files.setdefault(R.unit_for(k, c), json.loads((R.U(R.unit_for(k, c)) / "COMPLETE.json").read_text())["files"])
        lock["seeds"][str(k)] = {"score": score, "unit_file_sha256": files, "u_label": "SRC|U",
                                 "composed_frozen_winners": frozen,
                                 "composed_policies": [R.unit_for(k, c) for c in comp_k]}
    return R._finite(lock)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0].startswith("-"):
        raise SystemExit("usage: python -m lra.eval_lock <EVALUATION_LOCK.json path> (no flags; there is no override "
                         "of a technical failure)")
    L = build()
    refuse_unless_valid(L)
    Path(argv[0]).write_text(json.dumps(L, indent=1, allow_nan=False) + "\n")
    return L


if __name__ == "__main__":
    L = main()
    print(json.dumps({"statuses": {x: [v["status"], v.get("config") or v.get("descriptive_config")]
                                   for x, v in L["statuses"].items()},
                      "n_labels": len(L["scored_labels"]), "role_aliases": L["role_aliases"],
                      "technical_validity": L["technical_validity"]["ok"]}, indent=1))
