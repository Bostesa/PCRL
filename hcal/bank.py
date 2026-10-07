"""Common attack bank of the held-out calibration study (hcal; role D, attack/control owner; prompt section 8).

Not a training critic and not a calibrator. Every number here is a SEX-recovery statistic of a frozen public release.

PROVENANCE. Attack machinery is IMPORTED UNCHANGED from the pinned modules (never edited, never monkeypatched):
  dpc.audit (0a7b05a5)  _source_bank (FINAL slate + CC / CCpair cell readers with the unseen-token prior and the
                        INNER-selected unseen-tuple fallback), _select (dual AUC / CE selection over the ordered banks,
                        first-in-bank-order ties at 1e-12, pair bank = coalition + ignore_recipient_2 +
                        ignore_recipient_1 = dpc.audit._bank_lists order), _summarise, _p1, auc1, ce1, cc_local,
                        cc_pair, canonical_columns, onehot, view_fingerprint, sha_arrays, public_record, SLATES
                        ("final" = smf.audit.final_slate(False): LR x5, MLP x4, HGB x4, DA_LR, DA_MLP), KS,
                        PRIMARY_VIEWS, TIE, ATT_SEEDS, CC_ALPHAS.
  lra.audit (9762025)   load_inner (hash-complete + schema lra-inner-v1), _cell_bank (token-only cell readers),
                        _derived, _is_cc, _pub, policy_views (legacy refit view), SCHEMA, CELL_READERS, synthetic_D /
                        synthetic_teacher (synthetic fixtures only).
  lra.baselines         source_view_sets (the SRC|U interface view of a refit).
  osf.audit             guard_inner (role structure) and smf.audit.check_labels (no sealed label on attacker rows).
COPIED (with explicit fit rows instead of D["idx"]["AUDIT_FIT"], otherwise line for line): lra.audit.inner_family ->
fresh_family; lra.audit.inner_family_cells -> fresh_token_family; lra.audit.composed_source_bank's winner loop ->
_union. Nothing else is new attack logic.

ROLES (hcal.data allowlist, procedure "attack").
  FRESH readers   FITTED on ATTACK_FIT_NEW only (the AUDIT_FIT groups NOT drawn into CALIBRATION_HELDOUT; 4,064 rows),
                  SELECTED and scored for the inner record on INNER_SELECTION (2,235 rows).
  LEGACY readers  the ADMITTED lra inner audits (no refit for the record). They were FITTED ON ALL HISTORICAL AUDIT_FIT
                  ROWS, INCLUDING THE ROWS NOW IN CALIBRATION_HELDOUT (disclosed; the legacy readers predate the
                  split and are admitted unchanged), selected on INNER_SELECTION; never fitted on INNER_SELECTION
                  or assessment labels.
  No reader here is ever fitted or selected on CALIBRATION_HELDOUT, CALIBRATION_TRAIN_MATCHED, HEAD_VALIDATION or the
  assessment. Score rows exist only in refit_selected and are passed in by hcal.assess; this module never names or
  indexes the assessment role itself, and refit_selected reads no label on score rows (labels are masked to fit +
  INNER rows).
  check_attack_rows enforces: fit rows == the allowlisted role rows exactly (ATTACK_FIT_NEW for fresh readers, AUDIT_FIT
  for legacy / lra-source refits), INNER rows == INNER_SELECTION exactly, no overlap with each other or the assessment
  (fresh: none with CALIBRATION_HELDOUT group rows, HEAD_VALIDATION, OSF_DEFENSE_FIT either), osf.audit.guard_inner on
  D, and no sealed (-1) label on fit / INNER rows.

THE COMMON ATTACK BANK of (partition p, source seed k), in this BANK ORDER:
  1. ADMITTED LEGACY BANKS (no refit): the lra inner record (schema lra-inner-v1; record["recovery"] = the PRIMARY
     complete-interface "code" family) of every ORIGINAL release of p: legacy partitions [aud__pol__s{k}__<D0>,
     aud__dec__s{k}__<D1>], lra partitions [aud__new__s{k}__<D1>] (hcal.ids.lra_inner). Used per view w in (v1, v2,
     pair): auc_seed0, auc (attacker-seed 0-2 mean), auc_per_seed, ce_seed0, ce, ce_per_seed, selected, ce_selected,
     selection. Schema, primary family "code", per-seed means, row hashes, slate and attacker seeds are checked.
  2. ONE FRESH BANK on the COMPLETE fresh interface (fresh_views): recipient i = [one-hot(tok_i over the full alphabet,
     dpc.audit.canonical_columns order), table_v[tok_i] for every registered public lookup table v of p in registered
     order, one-hot(hard_i)], then FRESH VIEW HYGIENE (below); pair = hygiene([X_1', X_2']); exact token identities
     declared (CC / CCpair). Full FINAL slate + cell readers, fitted on ATTACK_FIT_NEW, dual-selected on
     INNER_SELECTION, fixed orientation P(S = 1) (never flipped or clamped), AUC- and CE-selected attackers refit at
     attacker seeds 0, 1, 2 (cell readers are deterministic counts).
     Unseen local token -> the ATTACK_FIT_NEW SEX prior; unseen tuple -> the INNER-selected rule (local_1, local_2,
     prior).

FRESH VIEW HYGIENE (role A decision, option (a), registered before SCIENCE_LOCK; fresh_views / view_hygiene). Applied
to every fresh design matrix, identically in the audit (fresh_family), the controls' fresh pipeline and every refit (all
build their views through fresh_views): with F = the ATTACK_FIT_NEW positions (role membership only, no labels),
(1) drop every column constant on F (zero variance on the fit rows: e.g. one-hot columns of tokens absent from F; no
slate member can learn a weight for them from F), then (2) drop every column that is an exact bitwise duplicate on F of
an earlier kept column (first in column order kept); the pair hygiene runs on [X_1', X_2'] after the per-recipient
hygiene. Per view the receipt records the kept column indices, the dropped constant / duplicate counts and the sha256 of
the kept-index list; the view fingerprint is of the final matrices. Tokens and decisions are unchanged and the cell
readers (CC / CCpair, keyed on token identities) are unaffected. Legacy views stay exactly the admitted lra views.
Motivation: the pinned DA readers' SVD (smf.audit.Canon, LAPACK gesdd) failed to converge on a synthetic planted fresh
view with constant and duplicate fit-row columns; the rule keeps every pinned reader and loses nothing learnable from F.

UNION RULE (common_record; also compose_u): per view and criterion the banks are scanned in bank order; the winner is
the first bank whose SEED-0 selected value beats the running best by more than 1e-12 (AUC: larger; CE: smaller), so
exact ties keep the earlier bank; the reported value is that bank's attacker-seed 0-2 mean; recorded: the winner bank,
its selected label, its per-seed values, its selection detail (refit spec) and every bank's seed-0 value; worse_local,
mean_local and coalition_minus_best_local as lra.audit._derived. This ONE record is the primary recovery of EVERY
decoder variant of p (D0 / D1 / MEAN / H-TOKEN32 / H-GLOBAL-TEMP / H-CLASS-TEMP / T-TOKEN32 share the tokens and the
decisions; a decoder table changes only the decoded vectors, and the fresh view already carries every registered
table). check_common asserts identical numbers across the variants.

U COMPOSITION (compose_u). Continuous U's privacy bank = the admitted lra SRC|U record (aud__tea__s{k}__U: its composed
interface family, already the union of U's own interface bank and all 84 lra code banks) followed by the FRESH bank of
every audited partition of that seed in registered order (hcal.ids.partitions()). An attacker holding U's source
interface can compute every code token and every public decoder table, so every fresh reader composes; stored fresh
records are reused (no refit). Same winner rule. U's three decoder variants (identity, H-GLOBAL-TEMP, H-CLASS-TEMP)
share this one record. FIELD MAPPING of the admitted SRC|U record (records only; inspected on the lra store):
  seed-0 selection value  recovery.composed.auc_seed0[w] / .ce_seed0[w]  (== recovery.auc_seed0 / .ce_seed0)
  reported mean           recovery.composed.auc[w] / .ce[w]     (== recovery.auc / .ce == record.composed.auc / .ce)
  internal winner         recovery.composed.winner[w] / .ce_winner[w]  ("source" or an lra code id; == record.composed)
  winner label            recovery.composed.winner_label / .ce_winner_label (== recovery.selected / .ce_selected)
  per-seed values         winner "source": recovery.auc_per_seed[w] / .ce_per_seed[w] (lra.audit.inner_unit does NOT
                          overwrite these: they are the OWN bank's; their mean is recovery.own.auc / .ce);
                          winner code c: aud__<unit of c>/record.json recovery.auc_per_seed[w] / .ce_per_seed[w]
  selected attacker       "source": recovery.selection[w][crit] (own bank); code c: that record's recovery.selection
  stored INNER refits     "source": aud__tea__s{k}__U/inner_preds.npz SEL_interface_{crit}_{w};
                          code c: aud__<unit of c>/inner_preds.npz SEL_code_{crit}_{w}
  closure                 record.composed.closure.ok and record.composed.policies (the 84 lra codes, registered order)

TOKEN-ONLY DIAGNOSTIC (fresh_token_family; supplementary): the exact-token cell readers only (lra.audit._cell_bank)
fitted on ATTACK_FIT_NEW, same selection; never enters the common record.

ASSESSMENT REFIT HOOK (refit_selected): for a frozen winner (refit_spec of a common / U record) refit that attacker at
attacker seeds 0-2 on its OWN fit rows (fresh: ATTACK_FIT_NEW; legacy and lra source / code: AUDIT_FIT through the
original lra views), predict INNER and the caller's score rows, and REFUSE unless every seed's INNER prediction equals
the stored one bitwise.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from dpc import audit as DA
from dpc.audit import ATT_SEEDS, CC_ALPHAS, FALLBACK_RULES, KS, PRIMARY_VIEWS, TIE, auc1, ce1, sha_arrays
from hcal import data as HD
from hcal import ids as I
from lra import audit as LA
from osf.audit import ASSESS_NAMES, guard_inner
from smf import audit as SA

SCHEMA_FRESH = "hcal-fresh-v1"
SCHEMA_TOKEN = "hcal-fresh-token-v1"
SCHEMA_COMMON = "hcal-common-v1"
SCHEMA_U = "hcal-u-composed-v1"
LEGACY_SCHEMA = LA.SCHEMA                        # "lra-inner-v1"
FRESH_FIT_ROLE = "ATTACK_FIT_NEW"
LEGACY_FIT_ROLE = "AUDIT_FIT"
SEL_ROLE = "INNER_SELECTION"
CRITS = ("auc", "ce")
FIT_ROLE_OF = {"fresh": FRESH_FIT_ROLE, "legacy": LEGACY_FIT_ROLE, "lra_source": LEGACY_FIT_ROLE,
               "lra_code": LEGACY_FIT_ROLE}
STORED_PREFIX = {"fresh": "SEL_fresh", "legacy": "SEL_code", "lra_source": "SEL_interface", "lra_code": "SEL_code"}
TABLE_SUM_TOL = 1e-9
MEAN_TOL = 1e-12
UNION_RULE = ("per view and criterion: banks in bank order (admitted legacy banks in registered order, then the fresh "
              "bank; for U: the admitted lra SRC|U composed bank, then the fresh bank of every partition in registered "
              "order); winner = first bank whose seed-0 selected value beats the running best by more than 1e-12 (AUC: "
              "larger; CE: smaller); exact ties keep the earlier bank; reported value = the winner's attacker-seed 0-2 "
              "mean")


class BankRefused(ValueError):
    """A bank, record or row set violates the registered common-bank contract."""


class RefitMismatch(RuntimeError):
    """A refit attacker does not reproduce its stored INNER_SELECTION predictions bitwise."""


def _refuse(msg):
    raise BankRefused(f"REFUSED: {msg}")


def _f(x):
    return float(x)


# ------------------------------------------------------------------ roles
def role_hashes(D):
    """sha256 (dpc.audit.sha_arrays) of the row ids of INNER_SELECTION, AUDIT_FIT and ATTACK_FIT_NEW (when present)."""
    rid = np.asarray(D["row_id"])
    out = {SEL_ROLE: sha_arrays(rid[np.asarray(D["idx"][SEL_ROLE])]),
           LEGACY_FIT_ROLE: sha_arrays(rid[np.asarray(D["idx"][LEGACY_FIT_ROLE])])}
    if "hcal" in D:
        out[FRESH_FIT_ROLE] = sha_arrays(rid[HD.role_rows(D, FRESH_FIT_ROLE)])
    return out


def check_attack_rows(D, y, fit_rows, sel_rows, fit_role=FRESH_FIT_ROLE):
    """(fit_idx, sel_idx) after the registered row checks (module docstring, ROLES); refuses otherwise."""
    if fit_role not in (FRESH_FIT_ROLE, LEGACY_FIT_ROLE):
        _refuse(f"unknown attacker fit role {fit_role!r}")
    guard_inner(D)                                   # role structure of AUDIT_FIT / INNER_SELECTION (osf, unchanged)
    fit = np.asarray(fit_rows, dtype=np.int64).ravel()
    sel = np.asarray(sel_rows, dtype=np.int64).ravel()
    want_fit = np.asarray(HD.labels_for(D, "attack", fit_role), dtype=np.int64)
    want_sel = np.asarray(HD.labels_for(D, "attack", SEL_ROLE), dtype=np.int64)
    if not np.array_equal(fit, want_fit):
        _refuse(f"attacker fit rows are not exactly {fit_role}")
    if not np.array_equal(sel, want_sel):
        _refuse("attacker selection rows are not exactly INNER_SELECTION")
    if len(np.unique(fit)) != len(fit) or len(np.unique(sel)) != len(sel):
        _refuse("duplicate attacker rows")
    if np.intersect1d(fit, sel).size:
        _refuse("attacker fit rows overlap INNER_SELECTION")
    for a in ASSESS_NAMES:
        if a in D["idx"] and (np.intersect1d(fit, D["idx"][a]).size or np.intersect1d(sel, D["idx"][a]).size):
            _refuse(f"attacker rows overlap {a}")
    if fit_role == FRESH_FIT_ROLE:
        if np.setdiff1d(fit, np.asarray(D["idx"][LEGACY_FIT_ROLE])).size:
            _refuse("ATTACK_FIT_NEW rows outside AUDIT_FIT")
        hc = D.get("hcal", {}).get("roles", {})
        for r in ("CALIBRATION_HELDOUT", "CALIBRATION_HELDOUT_GROUP_ROWS", "CALIBRATION_TRAIN_MATCHED_GROUP_ROWS"):
            if r in hc and np.intersect1d(fit, hc[r]).size:
                _refuse(f"fresh attacker fit rows overlap {r}")
        for r in ("HEAD_VALIDATION", "OSF_DEFENSE_FIT"):
            if r in D["idx"] and np.intersect1d(fit, D["idx"][r]).size:
                _refuse(f"fresh attacker fit rows overlap {r}")
    try:
        SA.check_labels(np.asarray(y), fit, sel)
    except ValueError as e:
        _refuse(f"{e}")
    return fit, sel


# ------------------------------------------------------------------ fresh complete views
def _pair_tables(v, name):
    """(T_1, T_2) of one lookup table given as a pair sequence or {1|"r1": T_1, 2|"r2": T_2}."""
    if isinstance(v, dict):
        a = v.get(1, v.get("r1", v.get("1")))
        b = v.get(2, v.get("r2", v.get("2")))
    else:
        if len(v) != 2:
            _refuse(f"table {name!r} must give one array per recipient")
        a, b = v
    if a is None or b is None:
        _refuse(f"table {name!r} must give one array per recipient")
    return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)


def check_table(T, cls, K, name="table"):
    """Problems of one decoder lookup table (alphabet x K): finite, nonnegative, rows sum to 1 within 1e-9, and a STRICT
    argmax equal to the token class on every row (reserved / unoccupied tokens included)."""
    T = np.asarray(T, dtype=np.float64)
    cls = np.asarray(cls, dtype=np.int64)
    bad = []
    if T.ndim != 2 or T.shape != (len(cls), K):
        return [f"{name}: shape {T.shape} != ({len(cls)}, {K})"]
    if not np.all(np.isfinite(T)) or T.min() < 0:
        bad.append(f"{name}: entries not finite and nonnegative")
    if np.max(np.abs(T.sum(1) - 1.0)) > TABLE_SUM_TOL:
        bad.append(f"{name}: rows do not sum to one within {TABLE_SUM_TOL}")
    mx = T.max(1, keepdims=True)
    if not np.array_equal(T.argmax(1), cls) or np.any((T == mx).sum(1) != 1):
        bad.append(f"{name}: strict argmax differs from the token class")
    return bad


HYGIENE_RULE = ("FRESH VIEW HYGIENE (role A decision, registered before SCIENCE_LOCK): for each fresh design matrix "
                "X_w (v1, v2; pair computed on [X_1', X_2'] after the per-recipient hygiene), with F = the "
                "ATTACK_FIT_NEW positions (role membership only, no labels): (1) drop every column constant on F "
                "(zero variance on the fit rows), (2) then drop every column that is an exact bitwise duplicate on F "
                "of an earlier kept column (first in column order kept). Fresh views only; legacy views are the "
                "admitted lra views unchanged; cell readers key on token identities and are unaffected")


def hygiene_rows(D):
    """F = the fresh attacker fit rows (D["hcal"]["roles"]["ATTACK_FIT_NEW"]; role membership only, no label)."""
    roles = (D.get("hcal") or {}).get("roles") or {}
    if FRESH_FIT_ROLE not in roles:
        _refuse("fresh view hygiene needs D['hcal']['roles']['ATTACK_FIT_NEW']")
    return np.asarray(roles[FRESH_FIT_ROLE], dtype=np.int64)


def view_hygiene(X, F):
    """(X[:, kept], receipt) under FRESH VIEW HYGIENE: drop columns constant on rows F, then columns bitwise identical
    on F to an earlier kept column. Label-free and deterministic (a function of X[F] only)."""
    X = np.asarray(X, dtype=np.float64)
    Xf = X[np.asarray(F, dtype=np.int64)]
    const = np.all(Xf == Xf[:1], axis=0) if len(Xf) else np.ones(X.shape[1], dtype=bool)
    cols = np.ascontiguousarray(Xf.T)
    kept, seen, ndup = [], set(), 0
    for j in range(X.shape[1]):
        if const[j]:
            continue
        key = cols[j].tobytes()
        if key in seen:
            ndup += 1
            continue
        seen.add(key)
        kept.append(j)
    kept = np.asarray(kept, dtype=np.int64)
    rec = {"n_in": int(X.shape[1]), "n_kept": int(len(kept)), "dropped_constant": int(const.sum()),
           "dropped_duplicate": int(ndup), "kept": kept.tolist(), "kept_sha256": sha_arrays(kept)}
    return X[:, kept], rec


def fresh_design(bank, tables, D, expected=None):
    """The fresh design matrices BEFORE hygiene: recipient i = [onehot(canonical column of tok_i, alpha_i),
    table_v[tok_i] for v in tables, onehot(hard_i, K_i)]; refuses invalid tables or a decision != its token class.
    Returns (X {v1, v2}, tokens, interface receipt, table sha256)."""
    names = list(tables)
    if not names:
        _refuse("a fresh view needs at least one registered lookup table")
    if expected is not None and names != list(expected):
        _refuse(f"lookup tables {names} differ from the registered list {list(expected)}")
    if not np.array_equal(np.asarray(bank["row_id"]), np.asarray(D["row_id"])):
        _refuse("bank rows are not aligned with D (row_id order)")
    X, T, chk = {}, {}, {}
    pairs = {nm: _pair_tables(tables[nm], nm) for nm in names}
    for i, K in ((1, KS[0]), (2, KS[1])):
        tok = np.asarray(bank[f"tok{i}"], dtype=np.int64)
        hard = np.asarray(bank[f"hard{i}"], dtype=np.int64)
        cls = np.asarray(bank[f"class{i}"], dtype=np.int64)
        a = int(np.asarray(bank[f"alpha{i}"]).ravel()[0])
        if len(cls) != a or tok.min() < 0 or tok.max() >= a:
            _refuse(f"recipient {i}: tokens / class table inconsistent with alphabet {a}")
        if not np.array_equal(cls[tok], hard):
            _refuse(f"recipient {i}: a released decision differs from its token class")
        bad = [p for nm in names for p in check_table(pairs[nm][i - 1], cls, K, f"r{i}:{nm}")]
        if bad:
            _refuse(f"recipient {i}: {bad}")
        col = DA.canonical_columns(tok, D, a)
        blocks = [DA.onehot(col[tok], a)] + [pairs[nm][i - 1][tok] for nm in names] + [DA.onehot(hard, K)]
        X[f"v{i}"] = np.hstack(blocks)
        T[f"v{i}"] = tok
        chk[f"v{i}"] = {"alphabet": a, "states_used": int(len(np.unique(tok))), "tables": names,
                        "dims_before_hygiene": int(X[f"v{i}"].shape[1])}
    return X, T, chk, {nm: sha_arrays(*pairs[nm]) for nm in names}


def fresh_views(bank, tables, D, expected=None, meta=None):
    """COMPLETE fresh interface of a frozen partition bank (hcal.admit arrays tok_i, hard_i, class_i, alpha_i, row_id),
    after FRESH VIEW HYGIENE (HYGIENE_RULE; F read from D["hcal"]["roles"]["ATTACK_FIT_NEW"], refused if absent).

    tables: ordered {name: (T_1, T_2)} (T_i of shape (alpha_i, K_i)), every registered public lookup table of the
    partition in registered order (expected: that registered name list, checked when given). Recipient i (before
    hygiene): X_i = [onehot(canonical column of tok_i, alpha_i), table_v[tok_i] for v in tables, onehot(hard_i, K_i)];
    pair = hygiene([X_1', X_2']); tokens {v1: tok1, v2: tok2} (unchanged). The receipt (meta["hygiene"]) holds per view
    the kept column indices, the dropped constant / duplicate counts and the sha256 of the kept-index list; the view
    fingerprint is of the final matrices."""
    F = hygiene_rows(D)
    raw, T, chk, tsha = fresh_design(bank, tables, D, expected=expected)
    X, hyg = {}, {}
    for w in ("v1", "v2"):
        X[w], hyg[w] = view_hygiene(raw[w], F)
        chk[w]["dims"] = int(X[w].shape[1])
    X["pair"], hyg["pair"] = view_hygiene(np.hstack([X["v1"], X["v2"]]), F)
    hyg["fit_rows"] = {"role": FRESH_FIT_ROLE, "rows": int(len(F)), "row_index_sha256": sha_arrays(F)}
    hyg["rule"] = HYGIENE_RULE
    return {"family": "fresh", "X": X, "tokens": T,
            "meta": {**(meta or {}), "interface": chk, "tables": list(tables), "table_sha256": tsha, "hygiene": hyg,
                     "view": "hygiene([one-hot token (full alphabet, occurrence-ordered columns), table_v[tok_i] for "
                             "every registered lookup table v (registered order), one-hot decision_i]); pair = "
                             "hygiene([v1', v2'])"}}


def token_views(bank, D, meta=None):
    """Token-only diagnostic views (one-hot token over the full alphabet, canonical columns; exact token identities)."""
    if not np.array_equal(np.asarray(bank["row_id"]), np.asarray(D["row_id"])):
        _refuse("bank rows are not aligned with D (row_id order)")
    X, T = {}, {}
    for i in (1, 2):
        tok = np.asarray(bank[f"tok{i}"], dtype=np.int64)
        a = int(np.asarray(bank[f"alpha{i}"]).ravel()[0])
        X[f"v{i}"], T[f"v{i}"] = DA.onehot(DA.canonical_columns(tok, D, a)[tok], a), tok
    X["pair"] = np.hstack([X["v1"], X["v2"]])
    return {"family": "token", "X": X, "tokens": T, "meta": {**(meta or {}), "diagnostic": True,
                                                              "view": "token-only (exact token identities)"}}


# ------------------------------------------------------------------ fresh family (copy of lra.audit.inner_family)
def _row_hashes(D, fit_idx, sel_idx):
    rid = np.asarray(D["row_id"])
    return {"sel_row_id_sha256": sha_arrays(rid[sel_idx]), "fit_row_id_sha256": sha_arrays(rid[fit_idx])}


def _finish(out, per, seeds):
    out["auc_seed0"], out["ce_seed0"] = dict(out["auc"]), dict(out["ce"])
    out["auc"] = {w: float(np.mean(per["auc"][w]["auc_per_seed"])) for w in PRIMARY_VIEWS}
    out["ce"] = {w: float(np.mean(per["ce"][w]["ce_per_seed"])) for w in PRIMARY_VIEWS}
    out["auc_per_seed"] = {w: per["auc"][w]["auc_per_seed"] for w in PRIMARY_VIEWS}
    out["ce_per_seed"] = {w: per["ce"][w]["ce_per_seed"] for w in PRIMARY_VIEWS}
    out["ce_selected_auc_per_seed"] = {w: per["ce"][w]["auc_per_seed"] for w in PRIMARY_VIEWS}
    out["attacker_seeds"] = list(seeds)
    LA._derived(out)
    return out


def fresh_family(views, D, fit_rows, sel_rows, seeds=ATT_SEEDS, slate="final", y=None):
    """lra.audit.inner_family with EXPLICIT fit rows: the whole fresh bank (slate + cell readers) fitted at attacker
    seed 0 on ATTACK_FIT_NEW (fit_rows), dual-selected on INNER_SELECTION (sel_rows), then the AUC- and CE-selected
    attackers refit at every attacker seed on ATTACK_FIT_NEW and scored on INNER_SELECTION. y = SEX (default D["sex"],
    sealed -1 on assessment rows; only fit / INNER rows are read). Returns (record with private inner_predictions /
    sel_row_id, {f"{crit}_{w}": (len(seeds), n_INNER) P(S=1)})."""
    t0, c0 = time.time(), time.process_time()
    if views.get("family") != "fresh" or not views.get("tokens"):
        _refuse("fresh_family audits the complete fresh view (fresh_views) with declared token identities only")
    if tuple(seeds)[0] != 0:
        _refuse("the bank is fitted and selected at attacker seed 0; seeds must start with 0")
    yy = np.asarray(D["sex"] if y is None else y)
    fit_idx, sel_idx = check_attack_rows(D, yy, fit_rows, sel_rows, FRESH_FIT_ROLE)
    sel_pos = np.arange(len(sel_idx))
    preds, _, cov, secs = DA._source_bank(views, yy, fit_idx, sel_idx, sel_pos, slate, 0, True, None)
    ys = yy[sel_idx]
    sel, banks = DA._select(preds, [""], ys, sel_pos)
    rec = DA._summarise(sel, banks)
    facs = dict(DA.SLATES[slate]())
    cache, arrays, per = {}, {}, {"auc": {}, "ce": {}}
    for w in PRIMARY_VIEWS:
        for crit in CRITS:
            s = rec["selection"][w][crit]
            view, att, key = s["view"], s["attacker"], s["pred_key"]
            rows = []
            for sd in seeds:
                if sd == seeds[0] or LA._is_cc(att):
                    p1 = preds[key]
                else:
                    ck = (view, att, sd)
                    if ck not in cache:
                        X = np.asarray(views["X"][view], dtype=np.float64)
                        m = facs[att](sd).fit(X[fit_idx], yy[fit_idx])
                        cache[ck] = DA._p1(m, X[sel_idx])
                    p1 = cache[ck]
                rows.append(p1)
            A = np.stack(rows)
            arrays[f"{crit}_{w}"] = A
            per[crit][w] = {"auc_per_seed": [auc1(ys, p) for p in A], "ce_per_seed": [ce1(ys, p) for p in A]}
    keys = list(preds)
    out = dict(rec)
    out.update({
        "schema": SCHEMA_FRESH, "kind": "inner", "family": "fresh", "meta": DA._jsonable(views.get("meta", {})),
        "finite": True, "slate": slate, "slate_members": [nm for nm, _ in DA.SLATES[slate]()],
        "cc_alphas": list(CC_ALPHAS), "fallback_rules": list(FALLBACK_RULES),
        "roles": {"fit": FRESH_FIT_ROLE, "select_and_score": SEL_ROLE},
        "orientation": "P(S=1), fixed; never flipped or clamped",
        "unseen_token_rule": "ATTACK_FIT_NEW SEX prior; unseen tuple: INNER-selected rule (local_1, local_2, prior)",
        "coverage": DA._coverage_summary(cov, sel_pos), "n_fit": int(len(fit_idx)), "n_select": int(len(sel_idx)),
        **_row_hashes(D, fit_idx, sel_idx), "fit_seconds": secs, "view_fingerprint": DA.view_fingerprint(views),
        "inner_predictions": {"keys": keys, "P": np.stack([preds[k] for k in keys])},
        "sel_row_id": np.asarray(D["row_id"])[sel_idx], "refits": len(cache)})
    _finish(out, per, seeds)
    out["statistic"] = ("auc = mean over attacker seeds of the INNER_SELECTION AUC of the attacker selected (seed 0, "
                        "bank maximum) on INNER_SELECTION, readers fitted on ATTACK_FIT_NEW; auc_seed0 = the selection "
                        "value; selection-optimistic; not a null test")
    out["wall_s"] = round(time.time() - t0, 2)
    out["cpu_s"] = round(time.process_time() - c0, 2)
    return out, arrays


def fresh_token_family(views, D, fit_rows, sel_rows, seeds=ATT_SEEDS, y=None):
    """TOKEN-ONLY DIAGNOSTIC (lra.audit.inner_family_cells with explicit fit rows): CC / CCpair cell readers on the
    exact token identities, fitted on ATTACK_FIT_NEW, dual-selected on INNER_SELECTION. Supplementary; never in the
    common record. views: token_views(bank, D) (or any views declaring tokens)."""
    t0 = time.time()
    yy = np.asarray(D["sex"] if y is None else y)
    fit_idx, sel_idx = check_attack_rows(D, yy, fit_rows, sel_rows, FRESH_FIT_ROLE)
    sel_pos = np.arange(len(sel_idx))
    preds, cov = LA._cell_bank(views, yy, fit_idx, sel_idx, sel_pos)
    ys = yy[sel_idx]
    sel, banks = DA._select(preds, [""], ys, sel_pos)
    rec = DA._summarise(sel, banks)
    keys = list(preds)
    rec.update({
        "schema": SCHEMA_TOKEN, "kind": "inner", "family": views.get("family"),
        "meta": DA._jsonable(views.get("meta", {})), "finite": True, "slate": None, "slate_members": [],
        "readers": LA.TOKEN_FAMILY_READERS, "cell_readers": LA.CELL_READERS, "cc_alphas": list(CC_ALPHAS),
        "fallback_rules": list(FALLBACK_RULES), "roles": {"fit": FRESH_FIT_ROLE, "select_and_score": SEL_ROLE},
        "orientation": "P(S=1), fixed; never flipped or clamped", "coverage": DA._coverage_summary(cov, sel_pos),
        "n_fit": int(len(fit_idx)), "n_select": int(len(sel_idx)), **_row_hashes(D, fit_idx, sel_idx),
        "inner_predictions": {"keys": keys, "P": np.stack([preds[k] for k in keys])},
        "sel_row_id": np.asarray(D["row_id"])[sel_idx], "refits": 0, "wall_s": round(time.time() - t0, 2)})
    arrays, per = {}, {"auc": {}, "ce": {}}
    for w in PRIMARY_VIEWS:
        for crit in CRITS:
            A = np.stack([preds[rec["selection"][w][crit]["pred_key"]]] * len(seeds))
            arrays[f"{crit}_{w}"] = A
            per[crit][w] = {"auc_per_seed": [auc1(ys, p) for p in A], "ce_per_seed": [ce1(ys, p) for p in A]}
    _finish(rec, per, seeds)
    rec["statistic"] = "token-only diagnostic (cell readers on ATTACK_FIT_NEW); not part of the common record"
    return rec, arrays


def public_fresh(rec):
    """JSON-safe fresh record (private arrays replaced by their sha256 and key list; every float finite)."""
    out = LA._pub(rec)
    bad = LA._nonfinite_paths(out)
    if bad:
        raise RuntimeError(f"TECHNICAL FAILURE: nonfinite values in a fresh record: {bad[:5]}")
    json.dumps(out, allow_nan=False)
    return out


def fresh_unit_arrays(rec, arrays, prefix="fresh"):
    """The private inner_preds.npz layout of a fresh unit (lra layout): keys_<prefix>, P_<prefix>,
    SEL_<prefix>_<crit>_<w> (len(seeds), n_INNER) and sel_row_id."""
    out = {f"keys_{prefix}": np.asarray(rec["inner_predictions"]["keys"]), f"P_{prefix}": rec["inner_predictions"]["P"],
           "sel_row_id": np.asarray(rec["sel_row_id"])}
    out.update({f"SEL_{prefix}_{k}": np.asarray(v) for k, v in arrays.items()})
    return out


# ------------------------------------------------------------------ bank entries
def _entry(name, kind, r, unit=None, release=None, view_fingerprint=None):
    seeds = list(r["attacker_seeds"])
    e = {"name": name, "kind": kind, "fit_role": FIT_ROLE_OF[kind], "unit": unit, "release": release,
         "fit_row_id_sha256": r["fit_row_id_sha256"], "sel_row_id_sha256": r["sel_row_id_sha256"],
         "slate": r["slate"], "attacker_seeds": seeds, "view_fingerprint": view_fingerprint,
         "stored_prefix": STORED_PREFIX[kind]}
    for key in ("auc", "ce", "auc_seed0", "ce_seed0"):
        e[key] = {w: _f(r[key][w]) for w in PRIMARY_VIEWS}
    for key in ("auc_per_seed", "ce_per_seed"):
        e[key] = {w: [_f(x) for x in r[key][w]] for w in PRIMARY_VIEWS}
    e["selected"] = {w: r["selected"][w] for w in PRIMARY_VIEWS}
    e["ce_selected"] = {w: r["ce_selected"][w] for w in PRIMARY_VIEWS}
    e["selection"] = {w: {c: {k: v for k, v in r["selection"][w][c].items()} for c in CRITS} for w in PRIMARY_VIEWS}
    _check_entry(e)
    return e


def _check_entry(e):
    """Per-seed lengths, means and the seed-0 selection value of one bank entry (refuses otherwise)."""
    n = len(e["attacker_seeds"])
    if not n or e["attacker_seeds"][0] != 0:
        _refuse(f"{e['name']}: attacker seeds {e['attacker_seeds']} do not start at seed 0")
    for w in PRIMARY_VIEWS:
        for crit in CRITS:
            ps = e[f"{crit}_per_seed"][w]
            vals = [e[crit][w], e[f"{crit}_seed0"][w], *ps]
            if not all(math.isfinite(v) for v in vals):
                _refuse(f"{e['name']}/{w}/{crit}: nonfinite value")
            if len(ps) != n:
                _refuse(f"{e['name']}/{w}/{crit}: {len(ps)} per-seed values for {n} attacker seeds")
            if abs(float(np.mean(ps)) - e[crit][w]) > MEAN_TOL:
                _refuse(f"{e['name']}/{w}/{crit}: reported value is not the attacker-seed mean")
            if abs(ps[0] - e[f"{crit}_seed0"][w]) > MEAN_TOL:
                _refuse(f"{e['name']}/{w}/{crit}: seed-0 per-seed value differs from the selection value")
            s = e["selection"][w][crit]
            if abs(_f(s["inner_auc" if crit == "auc" else "inner_ce"]) - e[f"{crit}_seed0"][w]) > MEAN_TOL:
                _refuse(f"{e['name']}/{w}/{crit}: selection detail differs from the seed-0 selection value")


def legacy_entry(name, rec):
    """Bank entry of one ADMITTED lra inner record (schema lra-inner-v1; primary family = complete code interface)."""
    if rec.get("schema") != LEGACY_SCHEMA:
        _refuse(f"{name}: schema {rec.get('schema')!r} is not {LEGACY_SCHEMA!r}")
    if rec.get("kind", "policy") != "policy" or rec.get("primary_family") != LA.PRIMARY_CODE_FAMILY:
        _refuse(f"{name}: not a code record whose primary family is the complete code interface")
    r = rec["recovery"]
    if r.get("family") != LA.PRIMARY_CODE_FAMILY:
        _refuse(f"{name}: recovery family {r.get('family')!r} is not the complete code interface")
    return _entry(name, "legacy", r, unit=name, release=rec.get("cid"), view_fingerprint=rec.get("view_fingerprint"))


def fresh_entry(rec, name="fresh", unit=None):
    """Bank entry of a fresh_family record (private or public form)."""
    if rec.get("schema") != SCHEMA_FRESH or rec.get("family") != "fresh":
        _refuse(f"{name}: not a fresh complete-interface record")
    if (rec.get("roles") or {}).get("fit") != FRESH_FIT_ROLE:
        _refuse(f"{name}: fresh readers were not fitted on {FRESH_FIT_ROLE}")
    return _entry(name, "fresh", rec, unit=unit or name, view_fingerprint=rec.get("view_fingerprint"))


def load_legacy(k, p, units_dir=None):
    """[(unit name, record, private arrays)] of the admitted original-release inner audits of partition p, seed k, in
    registered bank order (lra.audit.load_inner: hash-complete AND schema lra-inner-v1). Default store: hcal admitted
    copies."""
    root = Path(I.ADM_UNITS if units_dir is None else units_dir)
    out = []
    for rid, name in legacy_units(k, p):
        rec, arr = LA.load_inner(name, units_dir=root)
        if rec.get("cid") != rid or int(rec.get("seed", -1)) != int(k):
            _refuse(f"{name} holds {rec.get('cid')} seed {rec.get('seed')}, not {rid} seed {k}")
        out.append((name, rec, arr))
    return out


def legacy_units(k, p):
    """[(release id, lra inner unit)] of the ORIGINAL releases of p (legacy: D0, D1; lra: D1), registered bank order."""
    decs = ("D0", "D1") if I.is_legacy(p) else ("D1",)
    return [(I.release_id(p, d), I.lra_inner(k, I.release_id(p, d))) for d in decs]


# ------------------------------------------------------------------ union (copy of lra composed_source_bank's loop)
def _detail(e, w, crit):
    s = dict(e["selection"][w][crit])
    kind = s.pop("kind", e["kind"])
    d = {"bank": e["name"], "kind": kind, "fit_role": FIT_ROLE_OF[kind], "unit": s.pop("unit", e["unit"]),
         "release": s.pop("release", e["release"]), "stored_key": s.pop("stored_key", None),
         "view_fingerprint": s.pop("view_fingerprint", e["view_fingerprint"]),
         "fit_row_id_sha256": e["fit_row_id_sha256"], "sel_row_id_sha256": e["sel_row_id_sha256"],
         "slate": e["slate"], "attacker_seeds": list(e["attacker_seeds"])}
    if d["stored_key"] is None:
        d["stored_key"] = f"{STORED_PREFIX[kind]}_{crit}_{w}"
    for k in ("view", "attacker", "pred_key", "candidate", "label", "inner_auc", "inner_ce", "bank_index"):
        if k in s:
            d[k] = s[k]
    return d


def _consistency(entries, hashes=None):
    names = [e["name"] for e in entries]
    if len(set(names)) != len(names):
        _refuse(f"duplicate bank names {names}")
    for key in ("sel_row_id_sha256", "slate"):
        if len({e[key] for e in entries}) != 1:
            _refuse(f"banks disagree on {key}: { {e['name']: e[key] for e in entries} }")
    if len({tuple(e["attacker_seeds"]) for e in entries}) != 1:
        _refuse("banks disagree on attacker seeds")
    fits = {}
    for e in entries:
        fits.setdefault(e["fit_role"], set()).add(e["fit_row_id_sha256"])
    if any(len(v) != 1 for v in fits.values()):
        _refuse(f"banks of one fit role disagree on fit rows: {fits}")
    if len(fits) > 1 and len({next(iter(v)) for v in fits.values()}) != len(fits):
        _refuse("fresh readers carry the AUDIT_FIT row hash (they must be fitted on ATTACK_FIT_NEW only)")
    if hashes is not None:
        for e in entries:
            if e["sel_row_id_sha256"] != hashes[SEL_ROLE]:
                _refuse(f"{e['name']}: INNER_SELECTION rows differ from D's")
            if e["fit_row_id_sha256"] != hashes[e["fit_role"]]:
                _refuse(f"{e['name']}: fit rows differ from D's {e['fit_role']}")


def _union(entries, schema, hashes=None, **head):
    if not entries:
        _refuse("an empty bank")
    _consistency(entries, hashes)
    rec = {"schema": schema, **head, "rule": UNION_RULE, "tie": TIE,
           "banks": [{k: e[k] for k in ("name", "kind", "fit_role", "unit", "release", "fit_row_id_sha256")}
                     for e in entries],
           "sel_row_id_sha256": entries[0]["sel_row_id_sha256"], "slate": entries[0]["slate"],
           "attacker_seeds": list(entries[0]["attacker_seeds"]),
           "auc": {}, "ce": {}, "auc_seed0": {}, "ce_seed0": {}, "auc_per_seed": {}, "ce_per_seed": {},
           "winner": {}, "ce_winner": {}, "winner_label": {}, "ce_winner_label": {}, "selected": {}, "ce_selected": {},
           "winner_detail": {}, "bank_seed0": {"auc": {}, "ce": {}}, "bank_mean": {"auc": {}, "ce": {}}}
    for w in PRIMARY_VIEWS:
        ia = ic = 0
        for j, e in enumerate(entries):
            if e["auc_seed0"][w] > entries[ia]["auc_seed0"][w] + TIE:
                ia = j
            if e["ce_seed0"][w] < entries[ic]["ce_seed0"][w] - TIE:
                ic = j
        rec["winner_detail"][w] = {}
        for crit, j in (("auc", ia), ("ce", ic)):
            e = entries[j]
            wk = "winner" if crit == "auc" else "ce_winner"
            sk = "selected" if crit == "auc" else "ce_selected"
            rec[crit][w] = e[crit][w]
            rec[f"{crit}_seed0"][w] = e[f"{crit}_seed0"][w]
            rec[f"{crit}_per_seed"][w] = list(e[f"{crit}_per_seed"][w])
            rec[wk][w] = e["name"]
            rec[f"{wk}_label"][w] = e[sk][w]
            rec[sk][w] = f"{e['name']}:{e[sk][w]}"
            rec["winner_detail"][w][crit] = _detail(e, w, crit)
            rec["bank_seed0"][crit][w] = {x["name"]: x[f"{crit}_seed0"][w] for x in entries}
            rec["bank_mean"][crit][w] = {x["name"]: x[crit][w] for x in entries}
    LA._derived(rec)
    return rec


def common_record(legacy_records, fresh_record, partition=None, seed=None, hashes=None, fresh_name="fresh",
                  fresh_unit=None):
    """THE primary recovery record of every decoder variant of partition p at seed k.

    legacy_records: [(unit name, lra inner record[, arrays])] of the original releases of p in registered order
    (load_legacy); fresh_record: the fresh_family record of p (fresh complete view, ATTACK_FIT_NEW). Union rule: module
    docstring. With partition and seed the legacy unit names must be exactly legacy_units(seed, partition); with hashes
    (role_hashes(D)) every bank's row hashes are checked against D's roles."""
    entries = [legacy_entry(x[0], x[1]) for x in legacy_records]
    if not entries:
        _refuse("a partition's common bank needs its admitted legacy banks")
    if partition is not None and seed is not None:
        want = [nm for _, nm in legacy_units(seed, partition)]
        if [e["name"] for e in entries] != want:
            _refuse(f"legacy banks {[e['name'] for e in entries]} are not the registered {want}")
        rids = [rid for rid, _ in legacy_units(seed, partition)]
        if [e["release"] for e in entries] != rids:
            _refuse(f"legacy records hold {[e['release'] for e in entries]}, not {rids}")
    entries.append(fresh_entry(fresh_record, name=fresh_name, unit=fresh_unit))
    rec = _union(entries, SCHEMA_COMMON, hashes=hashes, partition=partition, seed=seed)
    rec["decoder_variants"] = (list(I.decoders_of(partition)) if partition in I.partitions() else None)
    rec["shared_by"] = ("every decoder variant of the partition (identical tokens and decisions; the fresh view "
                        "carries every registered lookup table)")
    rec["legacy_disclosure"] = ("legacy readers were fitted on all historical AUDIT_FIT rows, including rows now in "
                                "CALIBRATION_HELDOUT, and selected on INNER_SELECTION; never fitted on INNER_SELECTION "
                                "or assessment labels")
    return rec


def _canon(o):
    return json.dumps(DA._jsonable(o), sort_keys=True, allow_nan=False)


def record_sha256(rec):
    return sha_arrays(np.frombuffer(json.dumps(DA._jsonable(rec), sort_keys=True, allow_nan=False).encode(),
                                    dtype=np.uint8))


def assign_variants(k, p, common):
    """{release id: common record} for every registered decoder variant of p (one shared record)."""
    if common.get("schema") != SCHEMA_COMMON:
        _refuse("not a common record")
    if common.get("partition") not in (None, p) or common.get("seed") not in (None, k):
        _refuse(f"common record of {common.get('partition')} s{common.get('seed')} assigned to {p} s{k}")
    return {I.release_id(p, d): common for d in I.decoders_of(p)}


NUMBERS = ("auc", "ce", "auc_seed0", "ce_seed0", "auc_per_seed", "ce_per_seed", "winner", "ce_winner", "winner_label",
           "ce_winner_label", "worse_local", "mean_local", "coalition_minus_best_local")


def check_common(variant_records, common=None):
    """Assert that every decoder variant carries the IDENTICAL primary recovery (numbers, winners and the full record);
    raises AssertionError otherwise. Returns a receipt."""
    items = list(variant_records.items())
    if not items:
        raise AssertionError("no decoder variants")
    ref = common if common is not None else items[0][1]
    h = record_sha256(ref)
    bad = []
    for rid, r in items:
        if any(_canon(r.get(k)) != _canon(ref.get(k)) for k in NUMBERS):
            bad.append(f"{rid}: primary numbers differ")
        elif record_sha256(r) != h:
            bad.append(f"{rid}: record differs")
    if bad:
        raise AssertionError("decoder variants do not share one primary recovery: " + "; ".join(bad))
    return {"ok": True, "variants": [rid for rid, _ in items], "record_sha256": h,
            "rule": "one common attack bank per (partition, seed); every decoder variant receives the same selected "
                    "attack predictions, hence identical primary AUC / CE"}


# ------------------------------------------------------------------ U composition
def u_entry(u_record, code_records=None, strict=True):
    """Bank entry of the admitted lra SRC|U composed record (field mapping in the module docstring). code_records:
    {lra code id: its lra inner record} for every code that won a composed (view, criterion) (per-seed values and
    selection detail live there)."""
    code_records = code_records or {}
    if u_record.get("schema") != LEGACY_SCHEMA or u_record.get("kind") != "source" or \
            u_record.get("cid") != I.U_ID or u_record.get("primary_family") != "interface":
        _refuse("not the admitted lra SRC|U inner record (schema lra-inner-v1, kind source, primary interface)")
    k = int(u_record["seed"])
    r = u_record["recovery"]
    c = r["composed"]
    top = u_record.get("composed") or {}
    for key in ("auc", "ce", "auc_seed0", "ce_seed0"):
        if any(_f(r[key][w]) != _f(c[key][w]) for w in PRIMARY_VIEWS):
            _refuse(f"SRC|U recovery.{key} is not its composed value")
    for key in ("auc", "ce"):
        if any(_f(top[key][w]) != _f(c[key][w]) for w in PRIMARY_VIEWS):
            _refuse(f"SRC|U record.composed.{key} differs from recovery.composed.{key}")
    if strict:
        if not (top.get("closure") or {}).get("ok"):
            _refuse("SRC|U composed bank closure is not ok")
        if list(c.get("policies") or []) != LA.registered_composition_bank():
            _refuse("SRC|U composed bank is not the registered 84-code bank")
    e = {"name": u_record.get("inner_unit") or f"aud__tea__s{k}__U", "kind": "lra_source", "fit_role": LEGACY_FIT_ROLE,
         "unit": u_record.get("inner_unit") or f"aud__tea__s{k}__U", "release": I.U_ID,
         "fit_row_id_sha256": r["fit_row_id_sha256"], "sel_row_id_sha256": r["sel_row_id_sha256"], "slate": r["slate"],
         "attacker_seeds": list(r["attacker_seeds"]), "view_fingerprint": None, "stored_prefix": "SEL_interface",
         "auc": {}, "ce": {}, "auc_seed0": {}, "ce_seed0": {}, "auc_per_seed": {}, "ce_per_seed": {},
         "selected": {}, "ce_selected": {}, "selection": {w: {} for w in PRIMARY_VIEWS}, "inner_winner": {}}
    for w in PRIMARY_VIEWS:
        e["inner_winner"][w] = {}
        for crit in CRITS:
            win = c["winner" if crit == "auc" else "ce_winner"][w]
            e[crit][w], e[f"{crit}_seed0"][w] = _f(c[crit][w]), _f(c[f"{crit}_seed0"][w])
            e["selected" if crit == "auc" else "ce_selected"][w] = c["winner_label" if crit == "auc" else
                                                                     "ce_winner_label"][w]
            e["inner_winner"][w][crit] = win
            if win == "source":
                ps = [_f(x) for x in r[f"{crit}_per_seed"][w]]
                if abs(float(np.mean(ps)) - _f(r["own"][crit][w])) > MEAN_TOL:
                    _refuse(f"SRC|U recovery.{crit}_per_seed is not the own bank's (mapping changed?)")
                s = {**r["selection"][w][crit], "kind": "lra_source", "unit": e["unit"], "release": I.U_ID,
                     "stored_key": f"SEL_interface_{crit}_{w}"}
            else:
                if win not in code_records:
                    _refuse(f"SRC|U composed winner {win} ({w}/{crit}): its lra inner record is required")
                cr = code_records[win]
                if cr.get("cid") != win or cr.get("schema") != LEGACY_SCHEMA or \
                        cr.get("primary_family") != LA.PRIMARY_CODE_FAMILY or int(cr.get("seed", -1)) != k:
                    _refuse(f"record given for {win} is not its lra inner record at seed {k}")
                rr = cr["recovery"]
                for key in ("fit_row_id_sha256", "sel_row_id_sha256", "slate"):
                    if rr[key] != r[key]:
                        _refuse(f"{win} differs from SRC|U on {key}")
                if list(rr["attacker_seeds"]) != list(r["attacker_seeds"]):
                    _refuse(f"{win} differs from SRC|U on attacker seeds")
                if _f(rr[f"{crit}_seed0"][w]) != e[f"{crit}_seed0"][w] or _f(rr[crit][w]) != e[crit][w]:
                    _refuse(f"{win} record values differ from the SRC|U composed values ({w}/{crit})")
                ps = [_f(x) for x in rr[f"{crit}_per_seed"][w]]
                unit = cr.get("inner_unit") or I.lra_inner(k, win)
                s = {**rr["selection"][w][crit], "kind": "lra_code", "unit": unit, "release": win,
                     "view_fingerprint": cr.get("view_fingerprint"), "stored_key": f"SEL_code_{crit}_{w}"}
            e[f"{crit}_per_seed"][w] = ps
            e["selection"][w][crit] = s
    _check_entry(e)
    return e


def load_u(k, units_dir=None):
    """(SRC|U inner record, {winning code id: its inner record}) from the admitted store (records only)."""
    root = Path(I.ADM_UNITS if units_dir is None else units_dir)
    u, _ = LA.load_inner(f"aud__tea__s{k}__U", units_dir=root)
    c = u["recovery"]["composed"]
    wins = sorted({x for crit in ("winner", "ce_winner") for x in c[crit].values() if x != "source"})
    codes = {}
    for cid in wins:
        codes[cid], _ = LA.load_inner(I.lra_inner(k, cid), units_dir=root)
    return u, codes


def compose_u(lra_src_composed, fresh_records, seed=None, hashes=None, complete=True):
    """U's privacy record: the admitted lra SRC|U composed bank, then the fresh bank of every audited partition of that
    seed in registered order (stored fresh records; no refit). lra_src_composed: u_entry(...) or (u_record,
    code_records). fresh_records: ordered [(partition, fresh record[, fresh unit name])] or {partition: fresh record};
    complete=True requires all registered partitions."""
    first = lra_src_composed if isinstance(lra_src_composed, dict) and lra_src_composed.get("kind") == "lra_source" \
        else u_entry(*lra_src_composed)
    items = list(fresh_records.items()) if isinstance(fresh_records, dict) else list(fresh_records)
    items = [(x[0], x[1], x[2] if len(x) > 2 else f"fresh:{x[0]}") for x in items]   # (partition, record, unit)
    order = I.partitions()
    ps = [x[0] for x in items]
    if any(p not in order for p in ps):
        _refuse(f"unknown partitions {[p for p in ps if p not in order]}")
    pos = [order.index(p) for p in ps]
    if pos != sorted(pos) or len(set(pos)) != len(pos):
        _refuse("fresh banks are not in registered partition order")
    if complete and ps != order:
        _refuse(f"U composition needs the fresh bank of every partition ({len(ps)} of {len(order)} given)")
    entries = [first] + [fresh_entry(fr, name=f"fresh:{p}", unit=unit) for p, fr, unit in items]
    rec = _union(entries, SCHEMA_U, hashes=hashes, release=I.U_ID, seed=seed)
    rec["u_variants"] = I.u_release_ids()
    rec["lra_internal_winner"] = DA._jsonable(first["inner_winner"])
    rec["composition"] = ("an attacker holding U's source interface computes every code token and every public decoder "
                          "table, so every fresh reader composes; the lra SRC|U composed bank already holds U's own "
                          "interface bank and all 84 lra code banks")
    return rec


# ------------------------------------------------------------------ assessment refit hook
def refit_spec(record, w, crit, stored):
    """The frozen winner of (view w, criterion crit) of a common / U record, with its stored INNER predictions
    ((len(seeds), n_INNER), e.g. stored_predictions(detail))."""
    d = dict(record["winner_detail"][w][crit])
    d["stored"] = np.asarray(stored, dtype=np.float64)
    return d


def spec_for(entry, w, crit, stored):
    """Refit spec of an arbitrary bank entry's selected attacker (tests / replay)."""
    d = _detail(entry, w, crit)
    d["stored"] = np.asarray(stored, dtype=np.float64)
    return d


def stored_predictions(detail, units_dir=None):
    """Stored INNER refits of a winner from its hash-complete unit (default store: hcal run units for fresh banks,
    hcal admitted copies otherwise)."""
    from jcv.finalize import unit_complete
    root = Path(units_dir) if units_dir is not None else (I.UNITS if detail["kind"] == "fresh" else I.ADM_UNITS)
    d = root / detail["unit"]
    if not unit_complete(d):
        _refuse(f"{detail['unit']} is missing or not hash-complete")
    z = np.load(d / "inner_preds.npz", allow_pickle=False)
    return np.asarray(z[detail["stored_key"]])


def views_loader(detail, D, units_dir=None, fresh=None):
    """Zero-argument loader of the views a winner was audited on: legacy / lra code -> lra.audit.policy_views of the
    original release (admitted <release unit>/release.npz); lra source -> lra.baselines.source_view_sets(U teacher)
    ["interface"]; fresh -> fresh() (the caller's fresh_views(bank, tables, D) of the partition)."""
    kind = detail["kind"]
    root = Path(I.ADM_UNITS if units_dir is None else units_dir)
    if kind == "fresh":
        if fresh is None:
            _refuse("a fresh winner needs the partition's fresh_views loader")
        return fresh
    unit = str(detail["unit"])
    if not unit.startswith("aud__"):
        _refuse(f"{unit} is not an lra inner unit")
    rel_unit = unit[len("aud__"):]
    if kind in ("legacy", "lra_code"):
        def load():
            from lra import baselines as BL
            z, _ = BL.load_unit_npz(rel_unit, "release.npz", root)
            return LA.policy_views(z, D)
        return load
    if kind == "lra_source":
        def load_src():
            from lra import baselines as BL
            k = int(rel_unit.split("__")[1][1:])
            t, _ = BL.load_teacher("U", k, D, root)
            return BL.source_view_sets(t, D, families=("interface",))["interface"]
        return load_src
    _refuse(f"unknown bank kind {kind!r}")


def refit_selected(spec, views_or_loader, D, y, fit_rows, sel_rows, score_rows, seeds=ATT_SEEDS):
    """Refit the frozen winner `spec` at attacker seeds on its OWN fit rows (fresh: ATTACK_FIT_NEW; legacy / lra
    source / lra code: AUDIT_FIT through the original lra views), predict INNER_SELECTION and the caller's score rows.
    REFUSES (RefitMismatch) unless every seed's INNER prediction equals spec["stored"] bitwise. Returns (P_sel
    (len(seeds), n_sel) P(S=1), P_score (len(seeds), n_score, 2) = [1 - p, p]). No label on score rows is read."""
    kind = spec["kind"]
    if kind not in FIT_ROLE_OF:
        _refuse(f"unknown bank kind {kind!r}")
    if spec.get("fit_role", FIT_ROLE_OF[kind]) != FIT_ROLE_OF[kind]:
        _refuse(f"{kind} readers are fitted on {FIT_ROLE_OF[kind]}, not {spec.get('fit_role')}")
    seeds = tuple(seeds)
    if list(seeds) != list(spec.get("attacker_seeds", seeds))[:len(seeds)]:
        _refuse(f"refit seeds {list(seeds)} differ from the frozen attacker seeds {spec.get('attacker_seeds')}")
    yy = np.asarray(y)
    fit_idx, sel_idx = check_attack_rows(D, yy, fit_rows, sel_rows, FIT_ROLE_OF[kind])
    score = np.asarray(score_rows, dtype=np.int64).ravel()
    if len(np.unique(score)) != len(score) or np.intersect1d(score, fit_idx).size or \
            np.intersect1d(score, sel_idx).size:
        _refuse("score rows must be distinct and disjoint from the attacker fit and selection rows")
    if score.size and (score.min() < 0 or score.max() >= len(D["row_id"])):
        _refuse("score rows outside D")
    h = _row_hashes(D, fit_idx, sel_idx)
    for key in ("fit_row_id_sha256", "sel_row_id_sha256"):
        if spec.get(key) is not None and spec[key] != h[key]:
            _refuse(f"{key} differs from the frozen winner's")
    stored = np.asarray(spec["stored"], dtype=np.float64)
    if stored.shape != (len(seeds), len(sel_idx)):
        _refuse(f"stored predictions have shape {stored.shape}, expected {(len(seeds), len(sel_idx))}")
    views = views_or_loader() if callable(views_or_loader) else views_or_loader
    if spec.get("view_fingerprint") is not None and DA.view_fingerprint(views) != spec["view_fingerprint"]:
        _refuse("refit views differ from the audited views (view fingerprint)")
    ylab = np.zeros(len(yy), dtype=np.int64)                    # labels of fit + INNER rows only
    ylab[fit_idx], ylab[sel_idx] = yy[fit_idx], yy[sel_idx]
    view, att = spec["view"], spec["attacker"]
    P_sel, P_sc = [], []
    if LA._is_cc(att):
        toks = views.get("tokens") or {}
        if att.startswith("CCpair"):
            Pi, c = DA.cc_pair(toks["v1"], toks["v2"], ylab, fit_idx, sel_idx, np.arange(len(sel_idx)))
            Ps, _ = DA.cc_pair(toks["v1"], toks["v2"], ylab, fit_idx, score, np.arange(0, dtype=np.int64),
                               rules=c["fallback_rule"])
        else:
            Pi, _ = DA.cc_local(toks[view], ylab, fit_idx, sel_idx)
            Ps, _ = DA.cc_local(toks[view], ylab, fit_idx, score)
        P_sel, P_sc = [Pi[att]] * len(seeds), [Ps[att]] * len(seeds)
    else:
        fac = dict(DA.SLATES[spec.get("slate") or "final"]())[att]
        X = np.asarray(views["X"][view], dtype=np.float64)
        for sd in seeds:
            m = fac(sd).fit(X[fit_idx], ylab[fit_idx])
            P_sel.append(DA._p1(m, X[sel_idx]))
            P_sc.append(DA._p1(m, X[score]) if score.size else np.zeros(0))
    P_sel = np.stack(P_sel)
    bad = [sd for j, sd in enumerate(seeds) if not np.array_equal(P_sel[j], stored[j])]
    if bad:
        raise RefitMismatch(f"REFUSED: refit of {spec.get('bank')}:{spec.get('label', att)} does not reproduce the "
                            f"stored INNER_SELECTION predictions bitwise at attacker seeds {bad}")
    P_score = np.stack([np.stack([1.0 - p, p], 1) if p.size else np.zeros((0, 2)) for p in P_sc])
    return P_sel, P_score


# ------------------------------------------------------------------ synthetic fixtures and timing (no real data)
SYN_TABLES = ("D0", "D1", "H-TOKEN32", "H-GLOBAL-TEMP", "H-CLASS-TEMP", "T-TOKEN32")


def synthetic_D(seed=0, n_new=4064, group_pairs=True, **sizes):
    """lra.audit.synthetic_D (study role sizes by default) + D["hcal"]["roles"]: AUDIT_FIT rows paired into two-row
    exact-record groups, ATTACK_FIT_NEW = every row of a random group-closed subset with n_new rows,
    CALIBRATION_HELDOUT = one representative (smallest row id) of every other AUDIT_FIT group. Synthetic only."""
    D = LA.synthetic_D(seed=seed, **sizes)
    rng = np.random.default_rng(seed + 991)
    af = np.asarray(D["idx"][LEGACY_FIT_ROLE], dtype=np.int64)
    unit = np.asarray(D["unit"], dtype=np.int64).copy()
    if group_pairs:
        unit[af[1::2]] = unit[af[0::2][:len(af[1::2])]]
    D["unit"] = unit
    groups, sizes_ = np.unique(unit[af], return_counts=True)
    order = rng.permutation(len(groups))
    order = np.concatenate([order[sizes_[order] > 1], order[sizes_[order] == 1]])    # singletons last (exact count)
    chosen, rows = [], 0
    for j in order:
        if rows + int(sizes_[j]) <= n_new:
            chosen.append(groups[j])
            rows += int(sizes_[j])
        if rows == n_new:
            break
    if rows != n_new:
        raise ValueError(f"cannot draw a group-closed ATTACK_FIT_NEW of {n_new} rows")
    new = np.sort(af[np.isin(unit[af], chosen)])
    held_rows = np.sort(af[~np.isin(unit[af], chosen)])
    reps = np.sort([int(held_rows[unit[held_rows] == g].min()) for g in np.unique(unit[held_rows])])
    D["hcal"] = {"roles": {"ATTACK_FIT_NEW": new, "CALIBRATION_HELDOUT": np.asarray(reps, dtype=np.int64),
                           "CALIBRATION_HELDOUT_GROUP_ROWS": held_rows}}
    return D


def _smooth(mu, c, K, eps=1e-6):
    e = np.zeros(K)
    e[c] = 1.0
    return (mu + eps + eps * e) / (1 + (K + 1) * eps)


def _softmax_rows(L):
    P = np.exp(L - L.max(1, keepdims=True))
    return P / P.sum(1, keepdims=True)


def synthetic_bank(D, t, m=(8, 64), n_tables=5, sex_tilt=0.0, seed=0):
    """Synthetic frozen partition bank (hcal.admit layout) of teacher t: per present predicted class m_i tokens by
    within-class quantiles of the top probability on OSF_DEFENSE_FIT, one reserved token per absent class; and n_tables
    class-preserving lookup tables (D0-like smoothed token means, D1-like class mix, token-varying mix, global and
    per-class temperature, ...). sex_tilt > 0 moves SEX = 1 rows to the next token (a synthetic leak). Synthetic only.
    Returns (bank, ordered tables {name: (T_1, T_2)})."""
    rng = np.random.default_rng(seed)
    fit = np.asarray(D["idx"]["OSF_DEFENSE_FIT"], dtype=np.int64)
    S = np.asarray(D["sex"])
    bank = {"row_id": np.asarray(D["row_id"])}
    names = list(SYN_TABLES[:n_tables])
    tabs = {nm: [None, None] for nm in names}
    for i, (K, mi) in enumerate(zip(KS, m), start=1):
        p, d = np.asarray(t[f"p{i}"], dtype=np.float64), np.asarray(t[f"d{i}"], dtype=np.int64)
        present = sorted(set(np.unique(d[fit]).tolist()))
        base, cls, nxt = {}, [], 0
        for c in present:
            base[c] = nxt
            nxt += mi
            cls += [c] * mi
        reserve = {}
        for c in range(K):
            if c not in present:
                reserve[c] = nxt
                nxt += 1
                cls.append(c)
        cls = np.asarray(cls, dtype=np.int64)
        tok = np.zeros(len(d), dtype=np.int64)
        for c in range(K):
            rows = np.flatnonzero(d == c)
            if c not in present:
                tok[rows] = reserve[c]
                continue
            fr = rows[np.isin(rows, fit)]
            edges = np.quantile(p[fr, c], np.linspace(0, 1, mi + 1)[1:-1]) if mi > 1 else np.zeros(0)
            j = np.searchsorted(edges, p[rows, c], side="right")
            if sex_tilt:
                j = np.where((S[rows] == 1) & (rng.random(len(rows)) < sex_tilt), np.minimum(j + 1, mi - 1), j)
            tok[rows] = base[c] + j
        T0 = np.zeros((nxt, K))
        for tt in range(nxt):
            mm = fit[tok[fit] == tt]
            mu = p[mm].mean(0) if len(mm) else np.full(K, 1.0 / K)
            T0[tt] = _smooth(mu, cls[tt], K)
        E = DA.onehot(cls, K)
        mix = rng.uniform(0.0, 0.5, nxt)[:, None]
        tau_c = rng.uniform(0.5, 1.5, K)
        cand = {"D0": T0, "D1": 0.7 * T0 + 0.3 * E, "H-TOKEN32": (1 - mix) * T0 + mix * E,
                "H-GLOBAL-TEMP": _softmax_rows(np.log(T0) / 0.7),
                "H-CLASS-TEMP": _softmax_rows(np.log(T0) / tau_c[cls][:, None]),
                "T-TOKEN32": 0.9 * T0 + 0.1 * E}
        for nm in names:
            tabs[nm][i - 1] = cand[nm]
        bank.update({f"tok{i}": tok, f"hard{i}": d.copy(), f"class{i}": cls, f"alpha{i}": np.int64(nxt)})
        assert np.array_equal(cls[tok], d), "synthetic bank is not class preserving"
    return bank, {nm: tuple(v) for nm, v in tabs.items()}


def timing(seed=0, slate="final", n_new=4064, m=(8, 64), n_tables=5, token_family=True):
    """CPU seconds (time.process_time) of ONE fresh_family on a synthetic i8o64 partition with n_tables lookup tables at
    the real role sizes (ATTACK_FIT_NEW 4,064 fit rows, INNER_SELECTION 2,235; lra.audit.synthetic_D). Synthetic
    only."""
    import platform
    import resource
    D = synthetic_D(seed=seed, n_new=n_new)
    t = LA.synthetic_teacher(D, seed=seed)
    bank, tables = synthetic_bank(D, t, m=m, n_tables=n_tables, sex_tilt=0.1, seed=seed)
    fit = HD.role_rows(D, FRESH_FIT_ROLE)
    sel = np.asarray(D["idx"][SEL_ROLE])
    c0 = time.process_time()
    V = fresh_views(bank, tables, D)
    views_cpu = time.process_time() - c0
    c0, w0 = time.process_time(), time.time()
    rec, arr = fresh_family(V, D, fit, sel, slate=slate)
    fam_cpu, fam_wall = time.process_time() - c0, time.time() - w0
    out = {"kind": "SYNTHETIC timing (no real data, no labels); one process, OMP_NUM_THREADS=1",
           "fixture": {"rows": int(len(D["row_id"])), "ATTACK_FIT_NEW": int(len(fit)), "INNER_SELECTION": int(len(sel)),
                       "alphabets": [int(bank["alpha1"]), int(bank["alpha2"])], "lookup_tables": list(tables),
                       "dims": {w: int(V["X"][w].shape[1]) for w in PRIMARY_VIEWS}, "slate": slate,
                       "slate_members": [nm for nm, _ in DA.SLATES[slate]()], "attacker_seeds": list(ATT_SEEDS)},
           "fresh_views_cpu_s": round(views_cpu, 2), "fresh_family_cpu_s": round(fam_cpu, 2),
           "fresh_family_wall_s": round(fam_wall, 2), "refits": rec["refits"],
           "selected": rec["selected"], "auc": rec["auc"]}
    if token_family:
        c0 = time.process_time()
        fresh_token_family(token_views(bank, D), D, fit, sel)
        out["token_family_cpu_s"] = round(time.process_time() - c0, 2)
    n_units = len(I.partitions()) * len(I.SEEDS)
    out["estimate"] = {"fresh_units": n_units,
                       "fresh_cpu_h_all_units_at_this_cost": round(n_units * (fam_cpu + views_cpu) / 3600, 3),
                       "note": "upper-ish: the i1o1 CLASS partition is cheaper; real views may converge differently "
                               "(MLP early stopping)"}
    out["machine"] = {"python": platform.python_version(), "platform": platform.machine(),
                      "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    return out


def main(argv=None):
    """python -m hcal.bank timing [--slate final] [--tables 5]   (synthetic only; run under hcal.sema)"""
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("timing",))
    ap.add_argument("--slate", default="final")
    ap.add_argument("--tables", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    print(json.dumps(DA._jsonable(timing(seed=a.seed, slate=a.slate, n_tables=a.tables)), indent=1))


if __name__ == "__main__":
    main()
