"""Real-data controls of the held-out calibration study (hcal; role D, attack/control owner; prompt section 8 C).

An engineering check of the pipeline that produces the PRIMARY recovery: the COMMON bank (admitted legacy pipeline +
fresh pipeline; hcal.bank.common_record), and of the admitted continuous-U reader pipeline (ROT). A failed control
means TECHNICAL REVIEW of the audit (amendment), never "low leakage"; failures are recorded, never dropped.

COORDINATOR DECISION (role A, registered before any real control runs and before SCIENCE_LOCK): the REQUIRED verdicts
are the COMMON-bank verdicts (+ NULL_CALIBRATION on the common bank, + ROT on SRC|U); per-pipeline (legacy-only,
fresh-only) verdicts are reported as DIAGNOSTICS and never required.

PROVENANCE. The control RULES and LIMITS are the source's, UNCHANGED (lra.audit.CONTROL_LIMITS, itself cbp / qpc / smf):
NULL_Z 3.5, PLANT_MIN 0.75 (strictly greater), realised null threshold 0.5654143765984265 (within 1e-12, real split
only), CONF_ETA 0.05, ROT amplitude 1e-6, CONTROL_SEED from smf.audit, NULL_CALIBRATION = 5 permutations with seed
CONTROL_SEED + 100 + rep. IMPORTED UNCHANGED: smf.audit.null_split / frozen_permutation / null_threshold / noisy_sex,
lra.audit._split_audit / policy_views / check_release_keys (the LEGACY pipeline, fitted on AUDIT_FIT),
lra.audit.controls_for_release (ROT on SRC|U, rotate=True), dpc.audit.split_tokens / _roundtrip (the plant on an
original release, exactly as lra applies it), dpc.audit._source_bank / _select / decision_views,
lra.baselines.source_view_sets, hcal.bank.fresh_views. COPIED: lra.audit._split_audit with explicit fit rows ->
_split_audit (the FRESH pipeline, fitted on ATTACK_FIT_NEW); dpc.audit.split_tokens applied to a partition bank and
EVERY registered lookup table -> split_bank (incl. AMENDMENT_A1: a bit outside {0, 1} on an unlabelled row is set to
0); lra.audit.null_calibration / summarise_controls -> null_calibration / summarise_controls (common bank).

ONE S* FOR BOTH PIPELINES. INNER_SELECTION is split into exact-record-group halves A / B (smf.audit.null_split); S* =
SEX permuted within AUDIT_FIT, within A and within B (smf.audit.frozen_permutation exactly as lra.audit uses it,
CONTROL_SEED; its sha256 is recorded). The legacy pipeline fits on AUDIT_FIT; the fresh pipeline fits on ATTACK_FIT_NEW
using S* restricted to those rows (a permutation within AUDIT_FIT restricted to a fixed subset is still independent of
the features, so the null stays valid). Plants use the SAME noisy S* bits (20% random; b1 ~ Bernoulli(1/2) from
CONTROL_SEED + 7) for both pipelines. Labels read: SEX of AUDIT_FIT, ATTACK_FIT_NEW and INNER_SELECTION (hcal.data
procedure "attack"); the assessment is never indexed. The realised null threshold depends only on the SEX counts of half
B, so it must equal the source's on the real split (`realised_threshold=None` skips the receipt for synthetic splits).

ONE PLANT, EVERY BANK. For each code check the plant (token t -> (t, b)) is defined ONCE and applied (a) with
dpc.audit.split_tokens to each ORIGINAL release of the partition in registered legacy bank order (legacy partitions: D0
then D1; lra partitions: D1), audited by the LEGACY pipeline (lra.audit.policy_views + lra.audit._split_audit, fit
AUDIT_FIT); and (b) with split_bank to the frozen bank and every registered lookup table, audited by the FRESH pipeline
(hcal.bank.fresh_views, fit ATTACK_FIT_NEW). The planted tokens of every bank must agree (checked). Every planted
artifact goes through np.savez_compressed / np.load (serialisation exact). (c) COMMON verdict per check and view:
union over the banks in bank order [legacy D0, legacy D1 | legacy D1, fresh] with the hcal union winner rule applied on
HALF A (first bank whose A-selected AUC beats the running best by more than 1e-12), evaluated on half B:
  NULL       the real releases vs S*: every view common AUC_B <= thr = 0.5 + 3.5 sd0 (Mann-Whitney null sd on B).
  CONF_r{i}  b = noisy S*; the b = 1 copy decodes to (1 - 0.05) q_t + 0.05 e_d (same decision) in every release and
             every registered table: common AUC_B(v_i) > 0.75 and serialisation exact (decisions-only audit reported).
  COLL_r{i}  the same split with IDENTICAL vectors: common AUC_B(v_i) > 0.75 and serialisation exact (table-values-only
             audit reported).
  XOR        r1 split by b1, r2 by b2 = b1 XOR noisy S*, collision style: common AUC_B(pair) > 0.75 and the COMMON local
             winners AUC_B(v1) <= thr and AUC_B(v2) <= thr.
  NULL_CALIBRATION  5 permutations (seed CONTROL_SEED + 100 + rep, lra's permutation) on the common bank of
             U|DIRECT-TASK|i8o64 s0 (both pipelines per rep, the same permutation); pass iff no common exceedance.
  ROT_r{i}   (continuous SRC|U) lra.audit.controls_for_release(..., rotate=True), unchanged.
FRESH VIEW HYGIENE (role A decision, registered before SCIENCE_LOCK; hcal.bank.HYGIENE_RULE): every fresh-pipeline
view here (real or planted) is built by hcal.bank.fresh_views, which drops the columns constant on ATTACK_FIT_NEW and
then the columns bitwise identical on ATTACK_FIT_NEW to an earlier kept column (label-free, deterministic; the same
rule as the audit and every refit; the pair after the per-recipient hygiene). Legacy-pipeline views are the lra views
unchanged.
Planted releases and banks live in memory and in a temporary directory only; they are never saved as units.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np

from dpc import audit as DA
from dpc.audit import PRIMARY_VIEWS, TIE, auc1
from hcal import bank as B
from hcal import data as HD
from hcal import ids as I
from lra import audit as LA
from lra import baselines as BL
from smf import audit as SA

CONTROL_LIMITS = LA.CONTROL_LIMITS               # the source limits, unchanged
SOURCE_NULL_THRESHOLD = LA.SOURCE_NULL_THRESHOLD
CONTROL_SEED, NULL_Z, PLANT_MIN = LA.CONTROL_SEED, LA.NULL_Z, LA.PLANT_MIN
CONF_ETA, ROT_AMP = LA.CONF_ETA, LA.ROT_AMP
NULL_REPS = 5
THRESHOLD_TOL = 1e-12
CONTROL_CODES = ("U|DIRECT-TASK|i8o64", "U|JOINT|i8o64|l0.1", "U|CLASS|i1o1", "U|K-JOINT-PAIR|i8o64")
CODE_CHECKS = ("NULL", "CONF_r1", "CONF_r2", "COLL_r1", "COLL_r2", "XOR")
ROT_CHECKS = ("NULL", "ROT_r1", "ROT_r2")
FRESH = "fresh"
COMMON_RULE = ("union over the banks in bank order [legacy D0, legacy D1 | legacy D1, fresh]: winner = first bank "
               "whose half-A-selected AUC beats the running best by more than 1e-12; evaluated on half B")


# ------------------------------------------------------------------ plan
def control_plan():
    """The registered control plan (fixed by structure; never by results): the four codes at seed 0, each audited on the
    COMMON bank (legacy AUDIT_FIT pipeline on every original release + fresh ATTACK_FIT_NEW pipeline with the FULL
    registered table set, hcal.ids.decoders_of order) with NULL + CONF + COLL + XOR; NULL_CALIBRATION (5 permutations)
    on the common bank of U|DIRECT-TASK|i8o64 s0; ROT_r1 / ROT_r2 (+ its NULL) on SRC|U s0. Per-pipeline verdicts are
    diagnostic."""
    return {"codes": [[0, p] for p in CONTROL_CODES],
            "tables": {p: list(I.decoders_of(p)) for p in CONTROL_CODES},
            "legacy_releases": {p: [rid for rid, _ in B.legacy_units(0, p)] for p in CONTROL_CODES},
            "code_checks": list(CODE_CHECKS),
            "null_calibration": [0, CONTROL_CODES[0]], "null_reps": NULL_REPS,
            "rot": [[0, I.U_ID]], "rot_checks": list(ROT_CHECKS),
            "required": "COMMON-bank verdicts (legacy AUDIT_FIT pipeline + fresh ATTACK_FIT_NEW pipeline), "
                        "NULL_CALIBRATION on the common bank, ROT; per-pipeline verdicts diagnostic",
            "common_rule": COMMON_RULE,
            "fit_roles": {"legacy": "AUDIT_FIT (lra.audit._split_audit on lra.audit.policy_views of each original "
                                    "release)",
                          "fresh": "ATTACK_FIT_NEW (hcal.bank.fresh_views; S* restricted to these rows)",
                          "rot": "AUDIT_FIT (lra.audit.controls_for_release unchanged: the admitted U reader "
                                 "pipeline)"},
            "rule": "fixed by structure before any control runs: the task-only code, the JOINT code at the largest "
                    "registered lambda, the class-only code and the constrained code with the largest registered "
                    "neighbourhood, each with every original release and every registered lookup table"}


def control_jobs(plan):
    jobs = [("code", int(k), p) for k, p in plan["codes"]]
    if plan.get("null_calibration"):
        k, p = plan["null_calibration"]
        jobs.append(("calibration", int(k), p))
    jobs += [("rot", int(k), x) for k, x in plan.get("rot", [])]
    return jobs


# ------------------------------------------------------------------ roles, S*, pipelines
def control_roles(D):
    """(AUDIT_FIT rows, ATTACK_FIT_NEW rows, INNER_SELECTION rows) through the hcal allowlist, row-checked."""
    old = np.asarray(HD.labels_for(D, "attack", B.LEGACY_FIT_ROLE), dtype=np.int64)
    new = np.asarray(HD.labels_for(D, "attack", B.FRESH_FIT_ROLE), dtype=np.int64)
    sel = np.asarray(HD.labels_for(D, "attack", B.SEL_ROLE), dtype=np.int64)
    B.check_attack_rows(D, D["sex"], old, sel, B.LEGACY_FIT_ROLE)
    B.check_attack_rows(D, D["sex"], new, sel, B.FRESH_FIT_ROLE)
    return old, new, sel


def permuted_sex(D, halves, seed=CONTROL_SEED):
    """S* exactly as lra.audit draws it: smf.audit.frozen_permutation (within AUDIT_FIT, half A, half B).
    Returns (S*, sha256)."""
    return SA.frozen_permutation(np.asarray(D["sex"]), D, halves, seed=seed)


def _split_audit(views, Sp, D, fit_rows, halves, slate, finite=None):
    """FRESH pipeline: lra.audit._split_audit with explicit fit rows (ATTACK_FIT_NEW): fit the whole bank on fit_rows,
    select on half A, evaluate on half B."""
    sel_idx = np.asarray(D["idx"][B.SEL_ROLE])
    a, b = halves
    fin = DA._is_finite(views, finite)
    preds, _, cov, _ = DA._source_bank(views, Sp, np.asarray(fit_rows), sel_idx, a, slate, 0, fin, None)
    ya, yb = Sp[sel_idx][a], Sp[sel_idx][b]
    selA, banks = DA._select(preds, [""], ya, a)
    out = {}
    for w in PRIMARY_VIEWS:
        s = selA[w]["auc"]
        out[w] = {"selected_on_A": s["label"], "max_auc_A": s["inner_auc"],
                  "heldout_auc_B": auc1(yb, preds[s["pred_key"]][b]),
                  "per_attacker_auc_B": {DA._label(r): auc1(yb, preds[r["pred_key"]][b]) for r in banks[w]}}
    out["coverage"] = DA._coverage_summary(cov, np.arange(len(sel_idx)))
    return out


def _brief(r):
    return {w: {k: v for k, v in r[w].items() if k != "per_attacker_auc_B"} for w in PRIMARY_VIEWS}


def common_union(results):
    """COMMON split verdict values: results = ordered [(bank name, split-audit result)]; per view the winner is the
    first bank whose half-A selected AUC beats the running best by more than 1e-12; its half-B AUC is the common
    value."""
    if not results:
        raise ValueError("an empty control bank")
    out = {}
    for w in PRIMARY_VIEWS:
        j = 0
        for i, (_, r) in enumerate(results):
            if r[w]["max_auc_A"] > results[j][1][w]["max_auc_A"] + TIE:
                j = i
        name, r = results[j]
        out[w] = {"winner": name, "selected_on_A": f"{name}:{r[w]['selected_on_A']}", "max_auc_A": r[w]["max_auc_A"],
                  "heldout_auc_B": r[w]["heldout_auc_B"],
                  "bank_max_auc_A": {n: x[w]["max_auc_A"] for n, x in results},
                  "bank_heldout_auc_B": {n: x[w]["heldout_auc_B"] for n, x in results}}
    return out


def verdict(pname, target, vals, thr, exact=True):
    """Pass/fail of one check on per-view {heldout_auc_B} values (the source rules, unchanged)."""
    if pname == "NULL":
        return bool(all(vals[w]["heldout_auc_B"] <= thr for w in PRIMARY_VIEWS))
    detected = vals[target]["heldout_auc_B"] > PLANT_MIN
    if pname == "XOR":
        return bool(exact and detected and all(vals[w]["heldout_auc_B"] <= thr for w in ("v1", "v2")))
    return bool(exact and detected)


RULES = {"NULL": "AUC_B <= 0.5 + 3.5 sd0 on every view",
         "CONF": "AUC_B(v_i) > 0.75 (serialisation exact)", "COLL": "AUC_B(v_i) > 0.75 (serialisation exact)",
         "XOR": "AUC_B(pair) > 0.75 and AUC_B(v1), AUC_B(v2) <= null threshold (serialisation exact)"}


# ------------------------------------------------------------------ planted banks (memory / temporary directory only)
def split_bank(bank, tables, i, bit, collide=True, eta=CONF_ETA):
    """dpc.audit.split_tokens on a partition bank: recipient i's token t becomes 2 t + bit (alphabet doubles, the class
    table repeats); collide=True keeps both copies' vectors identical in EVERY lookup table; collide=False makes the
    bit = 1 copy decode to (1 - eta) table_v[t] + eta e_class(t) in every table v (more confident, same decision).
    AMENDMENT_A1: a bit outside {0, 1} (unlabelled rows) is set to 0. Returns (bank, tables)."""
    bk = {k: np.array(v) for k, v in bank.items()}
    t = bk[f"tok{i}"].astype(np.int64)
    a = int(np.asarray(bk[f"alpha{i}"]).ravel()[0])
    b = np.asarray(bit, dtype=np.int64)
    b = np.where((b == 0) | (b == 1), b, 0)
    cls = np.asarray(bk[f"class{i}"], dtype=np.int64)
    bk[f"tok{i}"] = 2 * t + b
    bk[f"alpha{i}"] = np.int64(2 * a)
    bk[f"class{i}"] = np.repeat(cls, 2)
    out = {}
    for nm, v in tables.items():
        pair = list(B._pair_tables(v, nm))
        T = pair[i - 1]
        T2 = np.repeat(T, 2, axis=0)
        if not collide:
            T2[1::2] = (1 - eta) * T + eta * DA.onehot(cls, T.shape[1])
        pair[i - 1] = T2
        out[nm] = tuple(pair)
    return bk, out


def _roundtrip(bank, tables, tmp, tag):
    names = list(tables)
    arrs = {f"bank__{k}": np.asarray(v) for k, v in bank.items()}
    for j, nm in enumerate(names):
        T1, T2 = B._pair_tables(tables[nm], nm)
        arrs[f"table{j}__r1"], arrs[f"table{j}__r2"] = T1, T2
    p = Path(tmp) / f"{tag}.npz"
    np.savez_compressed(p, **arrs)
    z = np.load(p, allow_pickle=False)
    # AMENDMENT_A1 (2026-10-07): serialisation exactness is BITWISE (dtype, shape and raw bytes, NaN payloads included).
    # np.array_equal treats NaN != NaN, and the real frozen banks carry NaN teacher means for reserved empty tokens.
    exact = sorted(z.files) == sorted(arrs) and all(
        z[k].dtype == arrs[k].dtype and z[k].shape == arrs[k].shape and
        np.ascontiguousarray(z[k]).tobytes() == np.ascontiguousarray(arrs[k]).tobytes() for k in arrs)
    bank2 = {k[len("bank__"):]: z[k] for k in z.files if k.startswith("bank__")}
    tables2 = {nm: (z[f"table{j}__r1"], z[f"table{j}__r2"]) for j, nm in enumerate(names)}
    return bank2, tables2, exact


def table_value_views(bank, tables):
    """Table-values-only views (every registered table's decoded vector, no token identity, no decision)."""
    X = {}
    for i in (1, 2):
        tok = np.asarray(bank[f"tok{i}"], dtype=np.int64)
        X[f"v{i}"] = np.hstack([B._pair_tables(tables[nm], nm)[i - 1][tok] for nm in tables])
    X["pair"] = np.hstack([X["v1"], X["v2"]])
    return {"family": "table_values_only", "X": X, "tokens": None}


TOKEN_KEYS = ("row_id", "tok1", "tok2", "hard1", "hard2", "alpha1", "alpha2")


def check_releases(k, p, bank, releases, D):
    """The original releases of p at seed k, in registered legacy bank order, each carrying EXACTLY the partition's
    tokens, decisions, alphabets and rows (so one plant applies to every bank). Returns [(rid, aligned release)]."""
    want = [rid for rid, _ in B.legacy_units(k, p)]
    if [rid for rid, _ in releases] != want:
        B._refuse(f"control releases {[rid for rid, _ in releases]} are not the registered {want}")
    out = []
    for rid, z in releases:
        LA.check_release_keys(z)
        zz = DA.align(z, D)
        bad = [x for x in TOKEN_KEYS if not np.array_equal(np.asarray(zz[x]).ravel(), np.asarray(bank[x]).ravel())]
        if bad:
            B._refuse(f"release {rid} does not carry the partition bank's {bad}")
        out.append((rid, zz))
    return out


def _plants(Sp):
    noisy = SA.noisy_sex(Sp, CONTROL_SEED)
    rng = np.random.default_rng(CONTROL_SEED + 7)
    b1 = rng.integers(0, 2, len(noisy))
    return {"NULL": ([], None),
            "CONF_r1": ([(1, noisy, False)], "v1"), "CONF_r2": ([(2, noisy, False)], "v2"),
            "COLL_r1": ([(1, noisy, True)], "v1"), "COLL_r2": ([(2, noisy, True)], "v2"),
            "XOR": ([(1, b1, True), (2, b1 ^ noisy, True)], "pair")}


def _bank_views(bank, tables, releases, D, ops, tmp, tag):
    """The planted (or real, ops = []) views of every bank in bank order: [(name, kind, views, serialisation exact)];
    planted tokens are checked to agree across the banks."""
    out, toks = [], []
    for j, (rid, z) in enumerate(releases):
        zp = dict(z)
        for i, bit, coll in ops:
            zp = DA.split_tokens(zp, i, bit, collide=coll)
        z2, exact = DA._roundtrip(zp, tmp, f"{tag}_legacy{j}") if ops else (zp, True)
        out.append((f"legacy:{rid}", "legacy", LA.policy_views(z2, D), bool(exact)))
        toks.append((z2["tok1"], z2["tok2"], z2["alpha1"], z2["alpha2"]))
    bk, tb = bank, tables
    for i, bit, coll in ops:
        bk, tb = split_bank(bk, tb, i, bit, collide=coll)
    if ops:
        bk, tb, exact = _roundtrip(bk, tb, tmp, f"{tag}_fresh")
    else:
        exact = True
    out.append((FRESH, "fresh", B.fresh_views(bk, tb, D), bool(exact)))
    ft = (bk["tok1"], bk["tok2"], bk["alpha1"], bk["alpha2"])
    if any(not all(np.array_equal(np.asarray(a).ravel(), np.asarray(b).ravel()) for a, b in zip(t, ft)) for t in toks):
        raise RuntimeError("TECHNICAL FAILURE: the plant produced different tokens in the legacy and fresh banks")
    return out, bk, tb


def _audit_banks(banks, Sp, D, fit_new, halves, slate):
    res = []
    for name, kind, V, _ in banks:
        r = LA._split_audit(V, Sp, D, halves, slate) if kind == "legacy" else \
            _split_audit(V, Sp, D, fit_new, halves, slate)
        res.append((name, r))
    return res


def _check_entry(pname, target, banks, results, thr):
    exact = all(e for *_, e in banks)
    com = common_union(results)
    key = pname.split("_")[0]
    entry = {"target_view": target, "serialisation_exact": exact, "rule": RULES[key], "common_rule": COMMON_RULE,
             "common": com, "ok": verdict(pname, target, com, thr, exact),
             "diagnostic": {n: {**_brief(r), "serialisation_exact": e,
                                "ok": verdict(pname, target, r, thr, e)}
                            for (n, _, _, e), (_, r) in zip(banks, results)}}
    if pname == "XOR":
        entry["common_locals_null_ok"] = all(com[w]["heldout_auc_B"] <= thr for w in ("v1", "v2"))
    return entry


# ------------------------------------------------------------------ one partition's common-bank controls
def controls_for_partition(label, k, p, bank, tables, releases, D, slate="final", workdir=None, reports=True,
                           expected=None):
    """NULL + CONF_r1/r2 + COLL_r1/r2 + XOR on the COMMON bank of partition p at seed k (module docstring rules).
    releases: [(release id, original release arrays)] in registered legacy bank order (B.legacy_units)."""
    t0, c0 = time.time(), time.process_time()
    if not D.get("sealed", False):
        raise SystemExit("REFUSED: controls run on the sealed D only")
    _, new, _ = control_roles(D)
    rels = check_releases(k, p, bank, releases, D)
    B.fresh_views(bank, tables, D, expected=expected)                          # registered tables, valid interface
    halves = SA.null_split(D)
    Sp, perm_sha = permuted_sex(D, halves)
    thr, sd = SA.null_threshold(Sp, D, halves)
    res = {"label": label, "partition": p, "seed": k, "tables": list(tables),
           "banks": [f"legacy:{rid}" for rid, _ in rels] + [FRESH], "permutation_sha256": perm_sha,
           "null_threshold": thr, "null_sd_B": sd, "plant_min": PLANT_MIN,
           "fit_roles": {"legacy": B.LEGACY_FIT_ROLE, FRESH: B.FRESH_FIT_ROLE}, "split": SA._split_receipt(D, halves),
           "required": "common", "checks": {}}
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        for pname, (ops, target) in _plants(Sp).items():
            banks, bk, tb = _bank_views(bank, tables, rels, D, ops, tmp, pname)
            results = _audit_banks(banks, Sp, D, new, halves, slate)
            entry = _check_entry(pname, target, banks, results, thr)
            entry["alphabets"] = [int(bk["alpha1"]), int(bk["alpha2"])]
            if reports and (pname.startswith("CONF") or pname == "XOR"):
                rd = LA._split_audit(DA.decision_views(bk["hard1"], bk["hard2"], D), Sp, D, halves, slate)
                entry["decisions_only_audit"] = _brief(rd)
                entry["decisions_only_misses_it"] = rd[target]["heldout_auc_B"] <= thr
            if reports and pname.startswith("COLL"):
                rq = LA._split_audit(table_value_views(bk, tb), Sp, D, halves, slate)
                entry["table_values_only_audit"] = _brief(rq)
                entry["table_values_only_misses_it"] = rq[target]["heldout_auc_B"] <= thr
            res["checks"][pname] = entry
    res["failures"] = [c for c, e in res["checks"].items() if not e["ok"]]
    res["diagnostic_failures"] = [f"{c}:{n}" for c, e in res["checks"].items() for n, d in e["diagnostic"].items()
                                  if not d["ok"]]
    res["all_ok"] = not res["failures"]
    res["wall_s"] = round(time.time() - t0, 1)
    res["cpu_s"] = round(time.process_time() - c0, 1)
    return res


def null_calibration(k, p, bank, tables, releases, D, reps=NULL_REPS, slate="final", seed0=CONTROL_SEED + 100,
                     expected=None):
    """lra.audit.null_calibration on the COMMON bank: per rep one permutation (lra's, seed seed0 + rep), both pipelines,
    common union on half A, evaluated on half B; every rep kept; per-bank values diagnostic."""
    _, new, _ = control_roles(D)
    rels = check_releases(k, p, bank, releases, D)
    B.fresh_views(bank, tables, D, expected=expected)
    halves = SA.null_split(D)
    with tempfile.TemporaryDirectory() as tmp:
        banks, _, _ = _bank_views(bank, tables, rels, D, [], tmp, "null")
    rows = []
    for r_ in range(reps):
        Sp, sha = permuted_sex(D, halves, seed=seed0 + r_)
        thr, sd = SA.null_threshold(Sp, D, halves)
        com = common_union(_audit_banks(banks, Sp, D, new, halves, slate))
        for w in PRIMARY_VIEWS:
            c = com[w]
            rows.append({"rep": r_, "seed": seed0 + r_, "view": w, "heldout_auc_B": c["heldout_auc_B"],
                         "max_auc_A": c["max_auc_A"], "winner": c["winner"], "selected_on_A": c["selected_on_A"],
                         "threshold": thr, "sd0": sd, "exceeds": c["heldout_auc_B"] > thr, "permutation_sha256": sha,
                         "bank_heldout_auc_B": c["bank_heldout_auc_B"],
                         "bank_exceeds": {n: v > thr for n, v in c["bank_heldout_auc_B"].items()}})
    hb = np.array([x["heldout_auc_B"] for x in rows])
    zz = (hb - 0.5) / np.array([x["sd0"] for x in rows])
    diag = {n: int(sum(x["bank_exceeds"][n] for x in rows)) for n, *_ in banks}
    return {"summary": {"reps": reps, "tests": len(rows), "exceedances": int(sum(x["exceeds"] for x in rows)),
                        "heldout_mean": float(hb.mean()), "heldout_max": float(hb.max()), "z_mean": float(zz.mean()),
                        "z_sd": float(zz.std(ddof=1)) if len(zz) > 1 else None, "null_z": NULL_Z,
                        "banks": [n for n, *_ in banks], "diagnostic_bank_exceedances": diag,
                        "rule": "no COMMON exceedance of 0.5 + 3.5 sd0 (common bank; " + COMMON_RULE + ")"},
            "rows": rows}


# ------------------------------------------------------------------ jobs
def _load_teacher_default(D):
    return lambda k: BL.load_teacher("U", k, D, I.ADM_UNITS)[0]


def _load_release_default(k, rid):
    return BL.load_unit_npz(I.lra_unit(k, rid), "release.npz", I.ADM_UNITS)[0]


def run_control_job(job, D, plan, load_bank_fn, tables_fn, load_release_fn=None, load_teacher_fn=None, slate="final",
                    workdir=None, reports=True):
    """One job of the plan -> {label: result} (code / rot) or {"NULL_CALIBRATION": ...}."""
    kind, k, x = job
    if kind in ("code", "calibration"):
        bank = load_bank_fn(k, x)
        tables = tables_fn(k, x)
        exp = (plan.get("tables") or {}).get(x)
        rel = load_release_fn or _load_release_default
        releases = [(rid, rel(k, rid)) for rid, _ in B.legacy_units(k, x)]
        if kind == "code":
            return {f"{x}|s{k}": controls_for_partition(f"{x}|s{k}", k, x, bank, tables, releases, D, slate=slate,
                                                        workdir=workdir, reports=reports, expected=exp)}
        return {"NULL_CALIBRATION": {"partition": f"{x}|s{k}", "bank": "common",
                                     **null_calibration(k, x, bank, tables, releases, D,
                                                        plan.get("null_reps", NULL_REPS), slate, expected=exp)}}
    if kind == "rot":
        if x != I.U_ID:
            raise ValueError(f"ROT controls are registered for {I.U_ID} only, not {x}")
        t = (load_teacher_fn or _load_teacher_default(D))(k)
        V = BL.source_view_sets(t, D, families=("interface",))["interface"]
        lab = f"{x}|s{k}|interface"
        return {lab: {"fit_role": "AUDIT_FIT (lra.audit.controls_for_release, unchanged)",
                      **LA.controls_for_release(lab, V, D, rotate=True, slate=slate, workdir=workdir)}}
    raise ValueError(f"unknown control job {job!r}")


def run_controls(D, plan=None, load_bank_fn=None, tables_fn=None, load_release_fn=None, load_teacher_fn=None,
                 slate="final", workdir=None, reports=True, jobs=None):
    """Run the plan's jobs (all, or the given subset) on the sealed D; returns the list of parts for summarise_controls.
    load_bank_fn(k, p) -> frozen partition bank (default: the hcal admitted bank npz); tables_fn(k, p) -> ordered
    {table name: (T_1, T_2)} with the partition's FULL registered table set (required); load_release_fn(k, release id)
    -> the original release arrays (default: the admitted <lra unit>/release.npz); load_teacher_fn(k) -> U teacher
    arrays (default: the admitted tea__s{k}__U)."""
    if not D.get("sealed", False):
        raise SystemExit("REFUSED: controls run on the sealed D only")
    if tables_fn is None:
        raise ValueError("tables_fn is required (the registered lookup tables of each control partition)")
    plan = plan or control_plan()
    if load_bank_fn is None:
        from hcal import admit as AD

        def load_bank_fn(k, p):
            z = np.load(AD.bank_path(k, p), allow_pickle=False)
            return {kk: z[kk] for kk in z.files}
    parts = []
    for job in (jobs if jobs is not None else control_jobs(plan)):
        parts.append(run_control_job(job, D, plan, load_bank_fn, tables_fn, load_release_fn, load_teacher_fn, slate,
                                     workdir, reports))
    return parts


def summarise_controls(parts, plan, slate="final", realised_threshold=SOURCE_NULL_THRESHOLD):
    """Verdict over every part: all_ok iff every COMMON check passes, ROT passes, NULL_CALIBRATION (common bank) is
    present with no exceedance, every planned control is present, and (real split) every realised null threshold
    equals the source's within 1e-12. Per-pipeline verdicts are reported as diagnostics and never enter all_ok."""
    releases = {k: v for p in parts for k, v in p.items() if k != "NULL_CALIBRATION"}
    calib = next((p["NULL_CALIBRATION"] for p in parts if "NULL_CALIBRATION" in p), None)
    failures = [f"{lab}:{f}" for lab, r in releases.items() for f in r["failures"]]
    calib_exc = (calib or {}).get("summary", {}).get("exceedances", 0)
    thr = sorted({float(r["null_threshold"]) for r in releases.values()} |
                 {float(x["threshold"]) for x in (calib or {}).get("rows", [])})
    if realised_threshold is not None:
        thr_ok = bool(thr) and all(abs(t - realised_threshold) <= THRESHOLD_TOL for t in thr)
        if not thr_ok:
            failures.append(f"REALISED_NULL_THRESHOLD {thr} != source {realised_threshold}")
    else:
        thr_ok = None
    planned = {f"{p}|s{k}" for k, p in plan.get("codes", [])} | {f"{x}|s{k}|interface" for k, x in plan.get("rot", [])}
    missing = sorted(planned - set(releases))
    if missing:
        failures.append(f"MISSING_CONTROLS {missing}")
    code_parts = {lab: r for lab, r in releases.items() if r.get("required") == "common"}
    verdict_ = {"all_ok": not failures and not calib_exc and calib is not None, "failures": failures,
                "required": "COMMON-bank checks, NULL_CALIBRATION on the common bank, ROT",
                "realised_null_thresholds": thr, "realised_threshold_matches_source": thr_ok,
                "null_calibration_exceedances": calib_exc, "null_calibration_present": calib is not None,
                "common_checks": {lab: {c: e["ok"] for c, e in r["checks"].items()} for lab, r in code_parts.items()},
                "rot_checks": {lab: {c: e["ok"] for c, e in r["checks"].items()} for lab, r in releases.items()
                               if lab not in code_parts},
                "diagnostic_per_pipeline": {lab: {c: {n: d["ok"] for n, d in e["diagnostic"].items()}
                                                  for c, e in r["checks"].items()} for lab, r in code_parts.items()},
                "diagnostic_failures": [f"{lab}:{f}" for lab, r in code_parts.items()
                                        for f in r.get("diagnostic_failures", [])],
                "diagnostic_null_calibration_bank_exceedances": (calib or {}).get("summary", {}).get(
                    "diagnostic_bank_exceedances"),
                "decisions_only_audit_misses_confidence_plant": {
                    lab: {c: e.get("decisions_only_misses_it") for c, e in r["checks"].items() if c.startswith("CONF")}
                    for lab, r in code_parts.items()},
                "on_failure": "a failed positive control triggers technical review and an amendment of the audit; it "
                              "is never interpreted as low leakage"}
    return DA._jsonable({
        "study": I.STUDY, "stage": "controls (inner roles only; sealed D)", "registered_limits": CONTROL_LIMITS,
        "roles": {"legacy_fit": B.LEGACY_FIT_ROLE, "fresh_fit": B.FRESH_FIT_ROLE, "rot_fit": "AUDIT_FIT (lra pipeline)",
                  "select_half_A_evaluate_half_B": B.SEL_ROLE, "assessment": "never indexed"},
        "slate": slate, "slate_members": [nm for nm, _ in DA.SLATES[slate]()], "plan": plan,
        "common_rule": COMMON_RULE,
        "label": "S* = SEX permuted within AUDIT_FIT, half A and half B (smf.audit.frozen_permutation, CONTROL_SEED, "
                 "as lra.audit); the fresh pipeline uses S* on ATTACK_FIT_NEW; plants encode noisy S* (20% random), "
                 "the same bits in every bank",
        "thresholds": {"null": "AUC_B <= 0.5 + 3.5 sd0", "plant_min": PLANT_MIN, "null_z": NULL_Z,
                       "rot_amplitude": ROT_AMP, "conf_eta": CONF_ETA,
                       "realised_threshold_required": realised_threshold},
        "releases": releases, "null_calibration": calib, "verdict": verdict_})


def public_json(summary):
    """Finite public JSON text of a control summary (refuses private paths)."""
    txt = json.dumps(DA._jsonable(summary), indent=1, allow_nan=False)
    for bad in (str(Path.home()), "PCRL_eval_cache_private", "/Users/", "/Volumes/", "/private/"):
        if bad in txt:
            raise SystemExit("REFUSED: a private path would enter the public control receipt")
    return txt
