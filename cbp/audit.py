"""Equal-strength attacker banks, composed source readers, coverage / MI receipts and real-data controls for the
confidence-budgeted privacy study (cbp; role D; prompt sections 10, 12, 14). Not a training critic.

PROVENANCE. A COPY of qpc/audit.py at the source handoff tip d0c8a45c879d01fb8b736ccc091ec3e2c3e9b351 (evidence
9dd06da6), whose attack machinery is dpc/audit.py at 0a7b05a5, IMPORTED UNCHANGED. qpc/ and dpc/ are never edited and
nothing in them is monkeypatched at runtime. The source FINAL slate and readers are unchanged:
smf.audit.final_slate(False) (LR x5, MLP x4, HGB x4, DA_LR, DA_MLP; pinned grids), the cell-conditional readers
CC_alpha{0.5, 1, 5} / CCpair_alpha{0.5, 1, 5} (prior-weighted Dirichlet), occurrence-ordered categorical one-hot token
columns, dual (AUC / CE) selection with first-in-bank-order ties (1e-12), fixed orientation P(S = 1) (never flipped
or clamped), the unseen-token prior and the INNER-selected unseen-pair fallback, full categorical identities + decoded
probabilities + decisions, pair tuples and both ignore-the-other-recipient banks, fitting on AUDIT_FIT and selection on
INNER_SELECTION, and dpc.audit.final_audit for the locked assessment.

DOCUMENTED DIFF against qpc/audit.py (everything not listed is the qpc logic, byte for byte):
  D1  docstrings and the study name (pcrl_confidence_budgeted_privacy_v1), SCHEMA "cbp-inner-v1".
  D2  binding: unit store, ids and scored lists through cbp.run (UNITS = <PRIVATE_CACHE>/cbp_v1/run/units,
      parse_id, unit_for, config_id, code_ids, composition_ids, COMPOSED_EXTRA_IDS, PKG) and cbp.baselines; SEX
      label rows through cbp.data.labels_for (which is qpc.data.labels_for).
  D3  REGISTERED COMPOSITION BANK (prompt sec. 10, lead decision 2026-10-06 "option b"): SRC|U composes, per seed,
      over the ENTIRE cbp code bank (27 codes = DIRECT-TASK i8o64, FINE-TASK i8o64, CLASS-ONLY, then the 24 privacy
      codes in lambda order (0.01, 0.025, 0.04, 0.06, 0.08, 0.1) x family order (LOCAL, SEQ-12, SEQ-21, JOINT)), THEN
      the 11 admitted qpc public maps outside that bank (COMPOSED_EXTRA: DIRECT-TASK i4o8, i4o16, i4o32, i4o64,
      i8o8, i8o16, i8o32 and LOCAL / SEQ-12 / SEQ-21 / JOINT at lambda 1), which are composition-only (never
      selection candidates, never scored as releases). The registration lives HERE (REG_*, COMPOSED_EXTRA) and must
      equal cbp.run.code_ids() / COMPOSED_EXTRA_IDS exactly, or the composed bank is REFUSED. The closure check
      expects exactly 27 + 11 completed U policy release units per seed on disk (any missing or unexpected unit ->
      REFUSED). RAW-J has no codes: its composed bank is its own bank.
  D4  inner records: identical fields (recovery.auc / .ce, utility, utility_gate, preserved, token_states,
      composed, coverage_receipt, mi_diagnostic, ...); token_states is null (None) for continuous releases and never
      Infinity; EVERY float in a record must be finite (TECHNICAL FAILURE otherwise, so a primary quantity can never
      be silently nulled by a later finite-JSON pass); one added diagnostic, cbp_headroom (log-loss excess <= 0.006,
      Brier excess <= 0.0035 per task at this seed) -- selection recomputes it and never uses qpc.utility's 0.0075
      "headroom" flag inside utility_gate.
  D5  validate_inner / validate_composed: independent post-hoc checks of a saved inner record against its stored
      predictions (best reader selected, fixed orientation, token identities present, aligned pair view, no extra
      release keys, every composed winner frozen) -- the deliberate-defect guards of prompt sec. 14.
  D6  controls: the cbp control plan (same structure as qpc's; the JOINT code is the one at the LARGEST registered
      lambda, 0.1), the registered pass limits CONTROL_LIMITS (below; written before any fit) and a realised-threshold
      receipt against the source's AUDIT_PRELOCK_CHECKS.json; finite public JSON (allow_nan=False).
  D7  synthetic timing and the cbp bank estimate (CBP_BANK).

REGISTERED CONTROL PASS LIMITS (prompt sec. 10; the SOURCE limits, unchanged; registered here and in AUDIT_COMPUTE.json
before any fit; controls are an engineering check, a failure is never evidence of removal):
  NULL      every view: held-out AUC_B <= 0.5 + 3.5 sd0 (sd0 = Mann-Whitney null sd on half B; the source's realised
            threshold on these roles was 0.5654143765984265, sd0 0.018689821885264708; a different realised value is
            reported as a technical failure because the split and permutation are fixed by seed).
  CONF_r{i} AUC_B(v_i) > 0.75 (serialisation exact).        COLL_r{i}  AUC_B(v_i) > 0.75 (serialisation exact).
  XOR       AUC_B(pair) > 0.75 AND AUC_B(v1) <= null threshold AND AUC_B(v2) <= null threshold.
  ROT_r{i}  AUC_B(v_i) > 0.75 AND AUC_B(pair) > 0.75 (serialisation exact).
  NULL_CALIBRATION  5 permutations; pass iff no exceedance of the null threshold.
Positive-control plants are built in memory / a temporary directory only; they are never saved as units, so they can
never enter a candidate or composed bank (the closure check would refuse them).

ROLES. Every attacker is FITTED on AUDIT_FIT (6,065 rows) and SELECTED on INNER_SELECTION (2,235 rows). Every inner
entry point runs osf.audit.guard_inner (attacker roles carry exactly those roles, no overlap with the assessment, no
sealed SEX labels). SEX is read on AUDIT_FIT / INNER_SELECTION (attacks), on OSF_DEFENSE_FIT only for the MI
diagnostic (cbp.data procedure "permutation_diagnostic"), and on assessment rows only inside cbp.assess. Token
occupancy of assessment rows (a label-free function of the release) is counted for the coverage receipt.

VIEWS (the release interface is audited whole; integer token IDs are never an ordinal feature).
  policy (pol__s{k}__<cid>/release.npz with EXACTLY row_id, tok1, q1, hard1, alpha1, tok2, q2, hard2, alpha2; any other
  key -- continuous probabilities, fine IDs, residuals -- is REFUSED, so protected-code attackers can never read the
  original continuous probabilities): recipient view v_i = [one-hot(token over the full alphabet alpha_i, columns
  ordered by first occurrence), decoded q_i, one-hot(decision_i)], pair = [v1, v2]; finite, tokens = tok_i.
  source (tea__s{k}__<teacher>/teacher.npz): interface [c_i, p_i] (PRIMARY), complete [r_i, c_i], scores c_i, probs p_i,
  decisions one-hot(d_i) (finite).  reference (ref__s{k}__<label>/reference.npz): cbp.baselines.reference_view_sets.

BANKS. Per target view, in bank order: v_i own:<v_i>:<attacker> (slate + CC); pair coalition:pair:<attacker> (slate on
[v1, v2] + CCpair on the exact tuple), ignore_recipient_2:v1:<attacker> (every v1 candidate), ignore_recipient_1:v2:
<attacker> (every v2 candidate). Unseen local token -> AUDIT_FIT SEX prior (the attacker-fitting prior). Unseen tuple ->
the rule R_a in {local_1, local_2, prior} with the lowest INNER_SELECTION cross-entropy over the INNER_SELECTION rows
whose tuple is unseen (ties in that order; none unseen -> prior), frozen before any scoring.

SELECTION AND SEEDS. The whole bank is fitted at attacker seed 0 and the AUC-selected (highest INNER_SELECTION AUC) and
CE-selected (lowest INNER_SELECTION cross-entropy, 1e-12 clip) attackers are chosen separately on those values. Each
selected attacker is then refit at attacker seeds 0, 1, 2 on AUDIT_FIT and scored on INNER_SELECTION (cell readers are
deterministic counts; their refit is identical). Record: recovery.auc[w] = mean over seeds 0-2 of the AUC-selected
attacker's INNER_SELECTION AUC; recovery.auc_seed0[w] = the selection value (bank maximum); recovery.ce[w] / ce_seed0
likewise for the CE-selected attacker. Inner AUCs are selection-optimistic statistics, identical for every release; not
null tests.

COMPOSED SOURCE BANK (section 10). The code maps are public and recipient i's map reads only p_i, which every source
family except decisions determines exactly; so an attacker holding the source output can apply ANY fitted code of the
same teacher and seed. `composed_source_bank` therefore forms, per source family and view, the union of the source's
own bank and the code-reader banks of EVERY policy of that teacher and seed in the registered composition bank
(D3: the 27 cbp codes in cbp.run.code_ids() order, then the 11 COMPOSED_EXTRA qpc maps), reusing each policy's stored
inner records
(no refit; the composed attacker A(g(p)) equals the policy attacker A on the same rows and AUDIT_FIT fit). The decisions
family composes only with the class-only code (its tokens are functions of d_i). Composed selection uses the same rule
at the bank level: per view the winning bank is the first (own, then policies in registered order) whose seed-0
selected value beats the running best by more than 1e-12; the winner is that bank's own selected attacker, and the
composed recovery is its seed 0-2 mean. CLOSURE: every expected policy inner unit must be complete, every completed
pol__s{k}__<teacher>_* release unit on disk must be expected, and all records must share the INNER_SELECTION / AUDIT_FIT
row hashes, slate and attacker seeds -- otherwise REFUSED. The source inner record stores the composed winners
(`composed.winner`, `composed.ce_winner`, per family) and `composed.freeze` = every policy whose reader won any (family,
view, criterion); the evaluation lock must list those policies as composed readers of the final source audit.
Protected-code attackers never receive source outputs.

COVERAGE AND MI (section 9; diagnostics only). `coverage_receipt`: per recipient and for the pair, the declared
alphabet, the cells actually occupied on OSF_DEFENSE_FIT / AUDIT_FIT / INNER_SELECTION / assessment rows, the AUDIT_FIT
count histogram, singleton-cell fractions (of occupied cells and of rows), and the rows falling back (unseen in
AUDIT_FIT) on fit (0 by construction), selection and assessment rows. `mi_diagnostic`: plug-in MI (nats) of SEX and
tok1, tok2, (tok1, tok2) on OSF_DEFENSE_FIT, and a permutation-null MI at the same alphabet from a FIXED list of
MI_PERMS permutations of DEFENSE_FIT SEX (seed MI_PERM_SEED; the list hash is recorded). It never alters objectives,
lambdas or nominees, and an MI decrease is never read as protection.

REAL-DATA CONTROLS (`stage_controls`; the LEAD runs them after AUDIT_AND_SELECTION_LOCK; sealed D; inner roles only;
pass limits = REGISTERED CONTROL PASS LIMITS above).
INNER_SELECTION is split by exact-record group into halves A / B (smf.audit.null_split); SEX is permuted within
AUDIT_FIT, within A and within B (smf.audit.frozen_permutation, drawn and hashed before any fit) to S*. Each check fits
on AUDIT_FIT, SELECTS on A over the whole bank and EVALUATES that attacker on held-out B. Plants encode noisy S* (20% of
rows replaced by a random class; a perfect reader has AUC 0.9). Pass rules (thresholds smf.audit's, unchanged:
NULL_Z 3.5, PLANT_MIN 0.75; thr = 0.5 + 3.5 sd0 on B):
  (a) NULL       real release vs S*: every view AUC_B <= thr.
  (b) CONF_r{i}  token t -> (t, b), b = noisy S*, the b = 1 copy decodes to (1 - 0.05) q_t + 0.05 e_d (same decision):
                 AUC_B(v_i) > 0.75. The decisions-only audit of the planted release is reported (expected <= thr).
  (c) COLL_r{i}  the same split with IDENTICAL decoded vectors (two IDs, one decoded vector, different information):
                 AUC_B(v_i) > 0.75. The decoded-probability-only audit is reported (it cannot see the plant).
  (d) XOR        recipient 1 split by b1 ~ Bernoulli(1/2), recipient 2 by b2 = b1 XOR noisy S*, collision-style:
                 AUC_B(pair) > 0.75 AND AUC_B(v1) <= thr AND AUC_B(v2) <= thr (not recoverable from either alone).
  (e) ROT_r{i}   continuous views (U, RAW-J, E interface): v_i -> [v_i, 1e-6 (noisy S* - 0.5)] Q, Q a seeded Haar-random
                 orthogonal matrix (the clue is spread over every coordinate at ~1e-6 of its scale), written with
                 np.savez_compressed and re-read: arrays bit-exact AND AUC_B(v_i) > 0.75 AND AUC_B(pair) > 0.75. The
                 per-attacker held-out table shows which (scale-sensitive) readers miss it.
  NULL_CALIBRATION  repeated permutations (seed CONTROL_SEED + 100 + rep) on one release; every rep kept; pass iff no
                 exceedance.
A failed positive control means TECHNICAL REVIEW of the audit (amendment), never "low leakage". Failures are recorded,
never dropped. The plan (CONTROL_PLAN rule) is fixed by structure before any control runs.

UNIT CONTRACT (cbp.run stage_inner / stage_inner_src): `inner_unit(kind, k, cid, D)` returns the JSON-safe record
  recovery {auc, ce, auc_seed0, ce_seed0, auc_per_seed, ce_per_seed, selected, ce_selected, tables, coverage, ...}
  (primary family; for SRC|<teacher> already the COMPOSED bank), families {secondary families, same structure},
  composed {auc, ce, winner, ce_winner, freeze, per_family, policies} (sources), utility (qpc.utility.
  release_inner_utility: {"income", "occupation"}), utility_gate (gate_record vs U of the same seed), preserved
  {"1", "2"}, decision_preservation, token_states (alpha1 + alpha2; None = JSON null for continuous releases, never
  Infinity), token_states_detail, coverage_receipt and mi_diagnostic (finite codes), cbp_headroom (D4), provenance
and "_files" = {"inner_preds.npz": {sel_row_id, keys_<fam>, P_<fam> (n_candidates, n_INNER) seed-0 P(S=1),
SEL_<fam>_<crit>_<view> (3, n_INNER) selected attacker per seed}}.
"""
from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np

from sklearn.metrics import roc_auc_score

from cbp import baselines as BL                  # D2
from dpc import audit as DA
from dpc.audit import (ATT_SEEDS, CC_ALPHAS, FALLBACK_RULES, FIT_ROLE, KS, PRIMARY_VIEWS, SEL_ROLE, TIE,  # noqa: F401
                       auc1, ce1, public_record, sha_arrays)
from osf.audit import guard_inner
from qpc import utility as UT
from smf import audit as SA

RELEASE_KEYS = frozenset({"row_id", "tok1", "q1", "hard1", "alpha1", "tok2", "q2", "hard2", "alpha2"})
FIT_NAME = "OSF_DEFENSE_FIT"
ASSESS = "OSF_DEVELOPMENT_ASSESSMENT"
MI_PERM_SEED = 20261011
MI_PERMS = 50
CONTROL_SEED, NULL_Z, PLANT_MIN = SA.CONTROL_SEED, SA.NULL_Z, SA.PLANT_MIN
ROT_AMP = SA.PLANT_TINY_AMP
CONF_ETA = DA.CONF_ETA
SCHEMA = "cbp-inner-v1"                          # D1
STUDY = "pcrl_confidence_budgeted_privacy_v1"

# D3: the registered composition bank (prompt secs. 7, 10; lead decision "option b"). Must equal cbp.run exactly.
REG_RATE = (8, 64)
REG_LAMS = (0.01, 0.025, 0.04, 0.06, 0.08, 0.1)
REG_PRIVACY = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
REG_TASK = ("U|DIRECT-TASK|i8o64", "U|FINE-TASK|i8o64", "U|CLASS|i1o1")
COMPOSED_EXTRA = ("U|DIRECT-TASK|i4o8", "U|DIRECT-TASK|i4o16", "U|DIRECT-TASK|i4o32", "U|DIRECT-TASK|i4o64",
                  "U|DIRECT-TASK|i8o8", "U|DIRECT-TASK|i8o16", "U|DIRECT-TASK|i8o32",
                  "U|LOCAL|i8o64|l1", "U|SEQ-12|i8o64|l1", "U|SEQ-21|i8o64|l1", "U|JOINT|i8o64|l1")
N_CODES, N_EXTRA = 27, 11

# D6: registered control pass limits (the source's; see the module docstring). Written before any fit.
SOURCE_NULL_THRESHOLD = 0.5654143765984265       # results/pcrl_confidence_capacity_v1/AUDIT_PRELOCK_CHECKS.json
SOURCE_NULL_SD0 = 0.018689821885264708
CONTROL_LIMITS = {
    "null": "every view: held-out AUC_B <= 0.5 + 3.5 sd0 (sd0 = Mann-Whitney null sd on half B)",
    "null_z": NULL_Z, "plant_min": PLANT_MIN, "plant_rule": "strictly greater than plant_min",
    "CONF_r1_CONF_r2": "AUC_B(v_i) > 0.75 and serialisation exact",
    "COLL_r1_COLL_r2": "AUC_B(v_i) > 0.75 and serialisation exact",
    "XOR": "AUC_B(pair) > 0.75 and AUC_B(v1) <= null threshold and AUC_B(v2) <= null threshold",
    "ROT_r1_ROT_r2": "AUC_B(v_i) > 0.75 and AUC_B(pair) > 0.75 and serialisation exact",
    "null_calibration": "5 permutations (seed CONTROL_SEED + 100 + rep); pass iff no exceedance of 0.5 + 3.5 sd0",
    "source_realised_null_threshold": SOURCE_NULL_THRESHOLD, "source_realised_sd0": SOURCE_NULL_SD0,
    "realised_threshold_rule": "the realised null threshold must equal the source's within 1e-12 (fixed split and "
                               "permutation seeds on the same roles); otherwise a technical failure",
    "control_seed": CONTROL_SEED, "rot_amplitude": ROT_AMP, "conf_eta": CONF_ETA,
    "plants_in_candidate_bank": "never (in-memory / temporary-directory releases, never saved as units)",
    "source": "results/pcrl_confidence_capacity_v1/AUDIT_PRELOCK_CHECKS.json thresholds (smf.audit NULL_Z, "
              "PLANT_MIN), unchanged"}


# ------------------------------------------------------------------ ids, units
def _R():
    from cbp import run as R                     # D2
    return R


def _g(x):
    return f"{x:g}"


def registered_code_bank():
    """The 27 registered cbp codes (D3), in cbp.run.code_ids() order: task-only references, then privacy codes."""
    return list(REG_TASK) + [f"U|{f}|i{REG_RATE[0]}o{REG_RATE[1]}|l{_g(lam)}" for lam in REG_LAMS for f in REG_PRIVACY]


def registered_composition_bank():
    """The 38 public code maps SRC|U composes with (D3): the 27 cbp codes, then the 11 qpc extras."""
    return registered_code_bank() + list(COMPOSED_EXTRA)


def registration_check(R=None):
    """{ok, ...}: cbp.run's code_ids / COMPOSED_EXTRA_IDS / composition_ids equal this registration exactly."""
    R = R or _R()
    got = {"code_ids": list(R.code_ids()), "COMPOSED_EXTRA_IDS": list(R.COMPOSED_EXTRA_IDS),
           "composition_ids": list(R.composition_ids())}
    want = {"code_ids": registered_code_bank(), "COMPOSED_EXTRA_IDS": list(COMPOSED_EXTRA),
            "composition_ids": registered_composition_bank()}
    bad = [k for k in want if got[k] != want[k]]
    scored_codes = [c for c in R.scored_ids() if R.parse_id(c)["kind"] == "policy"]
    if scored_codes != registered_code_bank():
        bad.append("scored_ids (policy part) != the 27 registered codes")
    if set(COMPOSED_EXTRA) & set(R.scored_ids()):
        bad.append("a composition-only extra is a selection candidate")
    if len(registered_code_bank()) != N_CODES or len(COMPOSED_EXTRA) != N_EXTRA:
        bad.append("registered bank sizes are not 27 + 11")
    return {"ok": not bad, "mismatches": bad, "n_codes": N_CODES, "n_extra": N_EXTRA}


def parse_cid(cid):
    return _R().parse_id(cid)


def unit_of(k, cid):
    return _R().unit_for(k, cid)


def _units(units_dir=None):
    return BL.units_dir(units_dir)


def _complete(name, units_dir=None):
    from jcv.finalize import unit_complete
    return unit_complete(_units(units_dir) / name)


def is_class_only(cid):
    c = parse_cid(cid)
    return c["kind"] == "policy" and c["family"] == "CLASS"


# ------------------------------------------------------------------ views
def check_release_keys(z):
    keys = set(z.files if hasattr(z, "files") else z)
    extra, missing = sorted(keys - RELEASE_KEYS), sorted(RELEASE_KEYS - keys)
    if extra or missing:
        raise ValueError(f"REFUSED: release keys must be exactly {sorted(RELEASE_KEYS)}; extra {extra}, missing "
                         f"{missing} (no continuous probabilities, fine IDs or residuals may accompany a code)")


def policy_views(z, D, meta=None):
    check_release_keys(z)
    return DA.policy_views(z, D, meta=meta)


class _LazyCodeX(dict):
    """X mapping of a code release that builds v1 / v2 / pair on access and keeps nothing (bit-identical to
    dpc.audit.policy_views; used for the many composed readers of a final source audit to bound memory)."""

    def __init__(self, z, D):
        super().__init__()
        self._z, self._D = z, D

    def _v(self, i):
        z = self._z
        return DA.code_view(z[f"tok{i}"].astype(np.int64), z[f"q{i}"], z[f"hard{i}"],
                            int(np.asarray(z[f"alpha{i}"]).ravel()[0]), KS[i - 1], self._D)

    def __getitem__(self, w):
        if w == "pair":
            return np.hstack([self._v(1), self._v(2)])
        return self._v(int(w[1]))

    def __contains__(self, w):
        return w in PRIMARY_VIEWS

    def keys(self):
        return list(PRIMARY_VIEWS)


def lazy_policy_views(z, D, meta=None):
    """Validated code views whose design matrices are rebuilt on every access (same values as policy_views)."""
    check_release_keys(z)
    zz = DA.align(z, D)
    chk, T = {}, {}
    for i in (1, 2):
        c = DA.token_decoder_check(zz[f"tok{i}"].astype(np.int64), zz[f"q{i}"], zz[f"hard{i}"])
        if not c["ok"]:
            raise ValueError(f"REFUSED: recipient {i} interface invalid: {c['problems']}")
        chk[f"v{i}"] = {**c, "alphabet": int(np.asarray(zz[f"alpha{i}"]).ravel()[0])}
        T[f"v{i}"] = zz[f"tok{i}"].astype(np.int64)
    return {"family": "code", "X": _LazyCodeX(zz, D), "tokens": T,
            "meta": {**(meta or {}), "interface": chk, "lazy": True,
                     "view": "[one-hot token (full alphabet, occurrence-ordered columns), decoded q_i, "
                             "one-hot decision_i]"}}


def view_sets(kind, k, cid, D, units_dir=None):
    """(primary family, {family: views}, outputs {p1, p2, hard1, hard2}, provenance, release arrays or None)."""
    c = parse_cid(cid)
    if kind != c["kind"]:
        raise ValueError(f"kind {kind!r} does not match config id {cid!r}")
    unit = unit_of(k, cid)
    if kind == "policy":
        z, sha = BL.load_unit_npz(unit, "release.npz", units_dir)
        meta = {"kind": "policy", "teacher": c["teacher"], "seed": k, "unit": unit, "config": cid,
                "family_method": c["family"], "m1": c["m1"], "m2": c["m2"], "lam": c["lam"],
                "m": 1 if c["family"] == "CLASS" else None}
        V = policy_views(z, D, meta=meta)
        out = {"p1": z["q1"], "p2": z["q2"], "hard1": z["hard1"], "hard2": z["hard2"]}
        return "code", {"code": V}, out, {"unit": unit, "complete_sha256": sha}, z
    if kind == "source":
        t, prov = BL.load_teacher(c["teacher"], k, D, units_dir)
        meta = {"kind": "source", "teacher": c["teacher"], "seed": k, "unit": unit, "label": cid}
        return "interface", BL.source_view_sets(t, D, meta), BL.source_outputs(t), prov, None
    sets, out, prov = BL.reference_view_sets(c["label"], k, D, units_dir=units_dir, with_outputs=True)
    return "interface", sets, out, prov, None


# ------------------------------------------------------------------ inner audit of one view set (+ seed refits)
def _is_cc(att):
    return str(att).startswith("CC")


def inner_family(views, D, slate="final", seeds=ATT_SEEDS):
    """dpc.audit.inner_audit (seed 0, full bank, dual selection) + refits of the AUC- and CE-selected attackers at
    every attacker seed, scored on INNER_SELECTION. Returns (record with private arrays, {array name: array})."""
    rec = DA.inner_audit(views, D, slate=slate)
    yy = np.asarray(D["sex"])
    fit_idx, sel_idx = DA.roles(D)
    ys = yy[sel_idx]
    P0 = dict(zip(rec["inner_predictions"]["keys"], rec["inner_predictions"]["P"]))
    facs = dict(DA.SLATES[slate]())
    cache, arrays = {}, {}
    per = {"auc": {}, "ce": {}}
    for w in PRIMARY_VIEWS:
        for crit in ("auc", "ce"):
            s = rec["selection"][w][crit]
            view, att, key = s["view"], s["attacker"], s["pred_key"]
            rows = []
            for sd in seeds:
                if sd == seeds[0] or _is_cc(att):
                    p1 = P0[key]
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
            per[crit][w] = {"label": s["label"], "auc_per_seed": [auc1(ys, p) for p in A],
                            "ce_per_seed": [ce1(ys, p) for p in A]}
    out = dict(rec)
    out["auc_seed0"], out["ce_seed0"] = dict(rec["auc"]), dict(rec["ce"])
    out["auc"] = {w: float(np.mean(per["auc"][w]["auc_per_seed"])) for w in PRIMARY_VIEWS}
    out["ce"] = {w: float(np.mean(per["ce"][w]["ce_per_seed"])) for w in PRIMARY_VIEWS}
    out["auc_per_seed"] = {w: per["auc"][w]["auc_per_seed"] for w in PRIMARY_VIEWS}
    out["ce_per_seed"] = {w: per["ce"][w]["ce_per_seed"] for w in PRIMARY_VIEWS}
    out["ce_selected_auc_per_seed"] = {w: per["ce"][w]["auc_per_seed"] for w in PRIMARY_VIEWS}
    out["attacker_seeds"] = list(seeds)
    out["refits"] = len(cache)
    _derived(out)
    out["statistic"] = ("auc = mean over attacker seeds of the INNER_SELECTION AUC of the attacker selected (seed 0, "
                        "bank maximum) on INNER_SELECTION; auc_seed0 = the selection value; selection-optimistic, "
                        "identical for every release; not a null test")
    return out, arrays


def _derived(r):
    r["worse_local"] = max(r["auc"]["v1"], r["auc"]["v2"])
    r["mean_local"] = (r["auc"]["v1"] + r["auc"]["v2"]) / 2
    r["coalition_minus_best_local"] = r["auc"]["pair"] - r["worse_local"]


def _pub(rec):
    out = public_record(rec)
    return DA._jsonable(out)


# ------------------------------------------------------------------ composed source bank
def expected_policies(teacher, protocol=None):
    """D3: every public code map of `teacher` in the REGISTERED composition order (U: the 27 cbp codes, then the 11
    COMPOSED_EXTRA qpc maps; RAW-J: none). REFUSED unless cbp.run agrees exactly with the registration."""
    chk = registration_check()
    if not chk["ok"]:
        raise SystemExit(f"REFUSED: cbp.run disagrees with the registered composition bank: {chk['mismatches']}")
    return [c for c in registered_composition_bank() if parse_cid(c)["teacher"] == teacher]


def load_inner(name, units_dir=None):
    """(record, private arrays) of a completed inner__<unit> (hash-complete)."""
    nm = name if name.startswith("inner__") else f"inner__{name}"
    d = _units(units_dir) / nm
    if not _complete(nm, units_dir):
        raise SystemExit(f"REFUSED: {nm} is missing or not hash-complete")
    rec = json.loads((d / "record.json").read_text())
    arr = {}
    if (d / "inner_preds.npz").exists():
        z = np.load(d / "inner_preds.npz", allow_pickle=False)
        arr = {k: z[k] for k in z.files}
    return rec, arr


def _closure(k, teacher, expected, units_dir=None):
    pre = f"pol__s{k}__{teacher}_"
    on_disk = sorted(p.name for p in _units(units_dir).glob(pre + "*")
                     if p.is_dir() and not p.name.endswith((".tmp", ".quarantined")) and _complete(p.name, units_dir))
    exp_units = {unit_of(k, c) for c in expected}
    unexpected = [u for u in on_disk if u not in exp_units]
    missing_release = [c for c in expected if unit_of(k, c) not in on_disk]          # D3: exact on-disk bank
    missing_inner = [c for c in expected if not _complete(f"inner__{unit_of(k, c)}", units_dir)]
    return {"expected": len(expected), "release_units_on_disk": len(on_disk), "unexpected_release_units": unexpected,
            "missing_release_units": missing_release, "missing_inner_units": missing_inner,
            "ok": not unexpected and not missing_release and not missing_inner and len(on_disk) == len(expected)}


def composed_source_bank(k, teacher, own, D, policy_cids=None, units_dir=None, protocol=None):
    """Composed bank of source SRC|<teacher> at seed k. own = {family: inner_family record} of the source.
    policy_cids default: expected_policies(teacher). Returns {family: composed record} plus the summary under "_all".
    REFUSES on any closure failure (missing policy inner unit, unexpected code unit, row / slate / seed mismatch)."""
    registered = policy_cids is None
    expected = expected_policies(teacher, protocol) if registered else list(policy_cids)
    clo = _closure(k, teacher, expected, units_dir)
    if not clo["ok"]:
        raise SystemExit(f"REFUSED: composed source bank of {teacher} s{k} is not closed: {clo}")
    pols = []
    for c in expected:
        r, _ = load_inner(unit_of(k, c), units_dir)
        rr = r["recovery"]
        for key in ("sel_row_id_sha256", "fit_row_id_sha256", "slate", "attacker_seeds"):
            ref = own["interface"].get(key) if key in own["interface"] else None
            if rr.get(key) != ref:
                raise SystemExit(f"REFUSED: {c} inner record differs from the source on {key}")
        if r.get("seed") is not None and int(r["seed"]) != int(k):
            raise SystemExit(f"REFUSED: {c} inner record is for seed {r.get('seed')}, not {k}")
        pols.append((c, unit_of(k, c), rr))
    out, freeze = {}, set()
    for fam, src in own.items():
        banks = [("source", None, src)] + [(c, u, rr) for c, u, rr in pols if fam != "decisions" or is_class_only(c)]
        rec = {"auc": {}, "ce": {}, "auc_seed0": {}, "ce_seed0": {}, "winner": {}, "ce_winner": {},
               "winner_label": {}, "ce_winner_label": {}, "own_auc": dict(src["auc"]), "own_ce": dict(src["ce"]),
               "n_banks": len(banks), "policies": [c for c, _, _ in banks[1:]]}
        for w in PRIMARY_VIEWS:
            ia = ic = 0
            for j, (_, _, b) in enumerate(banks):
                if b["auc_seed0"][w] > banks[ia][2]["auc_seed0"][w] + TIE:
                    ia = j
                if b["ce_seed0"][w] < banks[ic][2]["ce_seed0"][w] - TIE:
                    ic = j
            for crit, j in (("auc", ia), ("ce", ic)):
                c, u, b = banks[j]
                rec[crit][w] = float(b[crit][w])
                rec[f"{crit}_seed0"][w] = float(b[f"{crit}_seed0"][w])
                lab = b["selected" if crit == "auc" else "ce_selected"][w]
                rec["winner" if crit == "auc" else "ce_winner"][w] = c
                rec["winner_label" if crit == "auc" else "ce_winner_label"][w] = lab if c == "source" else \
                    f"composed[{u}]:{lab}"
                if c != "source":
                    freeze.add(c)
        _derived(rec)
        out[fam] = rec
    out["_all"] = {"freeze": sorted(freeze, key=expected.index), "closure": clo, "teacher": teacher, "seed": k,
                   "policies": expected,
                   "policy_list_source": "registered (D3: 27 cbp codes, then the 11 COMPOSED_EXTRA qpc maps)"
                   if registered else "EXPLICIT list (not the registered bank; tests / replay only)",
                   "composition_only": [c for c in expected if c in COMPOSED_EXTRA],
                   "rule": "per view: first bank (own, then policies in registered order) whose seed-0 selected value "
                           "beats the running best by > 1e-12; reported value = that bank's seed 0-2 mean; decisions "
                           "family composes with the class-only code only",
                   "reuse": "policy candidates enter with their stored inner records (same AUDIT_FIT fit, same "
                            "INNER_SELECTION rows); no refit"}
    return out


def composed_freeze_list(k, teacher, units_dir=None):
    """Policies whose composed readers won any (family, view, criterion) of SRC|<teacher> at seed k: they must join
    the final source audit as composed readers (frozen in the evaluation lock)."""
    r, _ = load_inner(unit_of(k, f"SRC|{teacher}"), units_dir)
    return list((r.get("composed") or {}).get("freeze", []))


# ------------------------------------------------------------------ coverage and MI receipts
def _idx(D, role):
    if role in D["idx"]:
        return np.asarray(D["idx"][role])
    alias = {"OSF_DEFENSE_FIT": "DEFENSE_FIT", "DEFENSE_FIT": "OSF_DEFENSE_FIT"}.get(role)
    return np.asarray(D["idx"][alias]) if alias in D["idx"] else np.zeros(0, dtype=np.int64)


def _hist(counts):
    u, c = np.unique(counts, return_counts=True)
    return {str(int(a)): int(b) for a, b in zip(u, c)}


def _cov_one(keys, D, alphabet=None):
    fit = np.asarray(keys)[_idx(D, FIT_ROLE)]
    u, cnt = np.unique(fit, return_counts=True)
    out = {"alphabet": int(alphabet) if alphabet is not None else None,
           "occupied": {r: int(len(np.unique(np.asarray(keys)[_idx(D, r)]))) for r in
                        (FIT_NAME, FIT_ROLE, SEL_ROLE, ASSESS)},
           "audit_fit_count_histogram": _hist(cnt),
           "audit_fit_singleton_cells": int((cnt == 1).sum()),
           "singleton_fraction_of_occupied": float((cnt == 1).mean()) if len(cnt) else None,
           "singleton_fraction_of_rows": float(cnt[cnt == 1].sum() / cnt.sum()) if cnt.sum() else None,
           "audit_fit_max_count": int(cnt.max()) if len(cnt) else 0}
    fb = {}
    for r in (FIT_NAME, FIT_ROLE, SEL_ROLE, ASSESS):
        kk = np.asarray(keys)[_idx(D, r)]
        miss = ~np.isin(kk, u)
        fb[r] = {"rows": int(len(kk)), "fallback_rows": int(miss.sum()),
                 "fallback_fraction": float(miss.mean()) if len(kk) else None,
                 "distinct_unseen_cells": int(len(np.unique(kk[miss])))}
    out["fallback_use"] = fb
    return out


def coverage_receipt(t1, t2, D, alpha1=None, alpha2=None):
    """Occupied local and pair alphabets, AUDIT_FIT counts, singleton fractions and fallback use (label-free)."""
    pk = DA._pair_keys(t1, t2)
    out = {"v1": _cov_one(t1, D, alpha1), "v2": _cov_one(t2, D, alpha2), "pair": _cov_one(pk, D, None)}
    out["pair"]["declared_tuple_space"] = int(alpha1) * int(alpha2) if alpha1 is not None and alpha2 is not None \
        else None
    out["note"] = ("fit fallback is 0 by construction (readers are fitted on AUDIT_FIT); unseen local tokens use the "
                   "AUDIT_FIT SEX prior; unseen tuples use the INNER-selected rule recorded in recovery.coverage.pair")
    return out


def _mi(t, s):
    """Plug-in MI (nats) of integer keys t and binary s."""
    t = np.asarray(t)
    s = np.asarray(s, dtype=np.int64)
    _, inv = np.unique(t, return_inverse=True)
    inv = inv.ravel()
    n = len(s)
    J = np.zeros((inv.max() + 1 if n else 0, 2))
    np.add.at(J, (inv, s), 1.0)
    J /= max(n, 1)
    pt, ps = J.sum(1, keepdims=True), J.sum(0, keepdims=True)
    m = J > 0
    return float(np.sum(J[m] * np.log(J[m] / (pt @ ps)[m])))


def sex_rows(D, procedure, role):
    """Rows of `role` whose SEX `procedure` may read (cbp.data.labels_for = qpc.data.labels_for; D2) and the labels."""
    try:
        from cbp import data as CD
        rows = np.asarray(CD.labels_for(D, procedure, role))
    except ModuleNotFoundError as e:
        if e.name != "cbp.data":
            raise
        rows = _idx(D, role)
    S = np.asarray(D["sex"])[rows]
    if S.size and S.min() < 0:
        raise PermissionError("REFUSED: sealed SEX labels")
    return rows, S


def permutation_list(n, perms=MI_PERMS, seed=MI_PERM_SEED):
    rng = np.random.default_rng(seed)
    P = np.stack([rng.permutation(n) for _ in range(perms)]) if perms else np.zeros((0, n), dtype=np.int64)
    return P, sha_arrays(P.astype(np.int64))


def mi_diagnostic(t1, t2, D, perms=MI_PERMS, seed=MI_PERM_SEED):
    """Plug-in fitted MI of DEFENSE_FIT SEX with tok1, tok2 and the tuple, and the permutation-null MI at the same
    alphabet (fixed permutation list). Diagnostic only (never an objective, lambda or nominee input)."""
    rows, S = sex_rows(D, "permutation_diagnostic", FIT_NAME)
    P, psha = permutation_list(len(rows), perms, seed)
    keys = {"v1": np.asarray(t1)[rows], "v2": np.asarray(t2)[rows], "pair": DA._pair_keys(t1, t2)[rows]}
    out = {"role": FIT_NAME, "n": int(len(rows)), "perms": int(perms), "perm_seed": int(seed),
           "perm_list_sha256": psha, "units": "nats", "views": {}}
    for w, kk in keys.items():
        mi = _mi(kk, S)
        null = np.array([_mi(kk, S[p]) for p in P])
        e = {"occupied": int(len(np.unique(kk))), "fitted_mi": mi}
        if len(null):
            sd = float(null.std(ddof=1)) if len(null) > 1 else None
            e.update({"null_mean": float(null.mean()), "null_sd": sd, "null_q95": float(np.quantile(null, 0.95)),
                      "null_max": float(null.max()), "excess_over_null_mean": float(mi - null.mean()),
                      "z": float((mi - null.mean()) / sd) if sd else None,
                      "fraction_null_ge_fitted": float((null >= mi).mean())})
        out["views"][w] = e
    out["note"] = ("plug-in MI is biased upward for sparse alphabets; the permutation null estimates that bias at the "
                   "same alphabet; a fitted-MI decrease is not protection, a population bound or a ranking")
    return out


# ------------------------------------------------------------------ the inner unit (lead's runner contract)
def inner_unit(kind, k, cid, D, units_dir=None, slate="final", seeds=ATT_SEEDS, policy_cids=None, protocol=None):
    """ONE inner audit unit (cbp.run saves the record as inner__<unit>; "_files" holds the private predictions)."""
    t0, c0 = time.time(), time.process_time()
    guard_inner(D)
    primary, sets, out, prov, z = view_sets(kind, k, cid, D, units_dir)
    recs, arrays = {}, {}
    for fam, V in sets.items():
        r, a = inner_family(V, D, slate=slate, seeds=seeds)
        recs[fam] = r
        arrays[f"keys_{fam}"] = np.asarray(r["inner_predictions"]["keys"])
        arrays[f"P_{fam}"] = r["inner_predictions"]["P"]
        for nm, A in a.items():
            arrays[f"SEL_{fam}_{nm}"] = A
    arrays["sel_row_id"] = recs[primary]["sel_row_id"]
    fam_pub = {fam: _pub(r) for fam, r in recs.items()}
    composed = None
    if kind == "source":
        teacher = parse_cid(cid)["teacher"]
        comp = composed_source_bank(k, teacher, recs, D, policy_cids=policy_cids, units_dir=units_dir,
                                    protocol=protocol)
        allc = comp.pop("_all")
        for fam, cr in comp.items():
            p = fam_pub[fam]
            p["own"] = {key: p[key] for key in ("auc", "ce", "auc_seed0", "ce_seed0", "selected", "ce_selected",
                                                "worse_local", "mean_local", "coalition_minus_best_local")}
            for key in ("auc", "ce", "auc_seed0", "ce_seed0", "worse_local", "mean_local",
                        "coalition_minus_best_local"):
                p[key] = cr[key]
            p["selected"], p["ce_selected"] = dict(cr["winner_label"]), dict(cr["ce_winner_label"])
            p["composed"] = DA._jsonable(cr)
        composed = {"auc": comp[primary]["auc"], "ce": comp[primary]["ce"], "winner": comp[primary]["winner"],
                    "ce_winner": comp[primary]["ce_winner"], "winner_label": comp[primary]["winner_label"],
                    "ce_winner_label": comp[primary]["ce_winner_label"],
                    "per_family": {f: {"winner": v["winner"], "ce_winner": v["ce_winner"]} for f, v in comp.items()},
                    **DA._jsonable(allc)}
    u_t, u_prov = BL.load_teacher("U", k, D, units_dir)
    probs = {1: out["p1"], 2: out["p2"]}
    hard = {1: out["hard1"], 2: out["hard2"]}
    util = UT.release_inner_utility(probs, hard, D)
    U_u = UT.release_inner_utility({1: u_t["p1"], 2: u_t["p2"]}, {1: u_t["d1"], 2: u_t["d2"]}, D)
    if kind == "policy":
        own_t = u_t if parse_cid(cid)["teacher"] == "U" else BL.load_teacher(parse_cid(cid)["teacher"], k, D,
                                                                               units_dir)[0]
        dp = UT.decision_preservation(out, own_t, D)
        basis = "released hard_i == the code's teacher d_i on every D row"
    else:
        dp = UT.decision_preservation(out, out, D)
        basis = "continuous source / reference: the release's decisions are its own (identity, still computed)"
    preserved = {"1": bool(dp["tasks"]["0"]["ok"]), "2": bool(dp["tasks"]["1"]["ok"])}
    g = UT.gate_record(util, U_u, {1: preserved["1"], 2: preserved["2"]})
    res = {"schema": SCHEMA, "kind": kind, "cid": cid, "seed": k, "unit_of": prov["unit"],
           "of_complete_sha256": prov.get("complete_sha256"), "u_teacher_complete_sha256": u_prov["complete_sha256"],
           "primary_family": primary, "recovery": fam_pub[primary],
           "families": {f: v for f, v in fam_pub.items() if f != primary}, "composed": composed,
           "utility": DA._jsonable(util), "utility_gate": DA._jsonable(g), "preserved": preserved,
           "decision_preservation": DA._jsonable({**dp, "basis": basis}), "slate": slate,
           "slate_members": [nm for nm, _ in DA.SLATES[slate]()], "attacker_seeds": list(seeds),
           "token_states": None,                 # D4: JSON null for continuous releases, never Infinity
           "token_states_rule": "null: continuous release (no finite token alphabet); never serialised as Infinity",
           "cbp_headroom": cbp_headroom(g),
           "private_file": "inner_preds.npz (keys_<fam>, P_<fam> seed-0 candidates; SEL_<fam>_<crit>_<view> (seeds, "
                           "n_INNER) selected attackers; sel_row_id)"}
    if kind == "policy":
        a1, a2 = (int(np.asarray(z[f"alpha{i}"]).ravel()[0]) for i in (1, 2))
        fit = _idx(D, FIT_NAME)
        res["token_states"] = a1 + a2
        res["token_states_rule"] = "alpha1 + alpha2 of the deployed policy (declared alphabets incl. reserved tokens)"
        res["token_states_detail"] = {f"recipient_{i}": {"alphabet": a, "defense_fit_occupied": int(
            len(np.unique(np.asarray(z[f"tok{i}"])[fit]))), "all_rows_occupied": int(len(np.unique(z[f"tok{i}"])))}
            for i, a in ((1, a1), (2, a2))}
        res["coverage_receipt"] = coverage_receipt(z["tok1"], z["tok2"], D, a1, a2)
        res["mi_diagnostic"] = mi_diagnostic(z["tok1"], z["tok2"], D)
        res["view_fingerprint"] = DA.view_fingerprint(sets["code"])
    elif kind == "reference" and "cells" in sets:
        T = sets["cells"]["tokens"]
        res["coverage_receipt"] = coverage_receipt(T["v1"], T["v2"], D)
        res["mi_diagnostic"] = mi_diagnostic(T["v1"], T["v2"], D)
    res["wall_s"] = round(time.time() - t0, 2)
    res["cpu_s"] = round(time.process_time() - c0, 2)
    res = DA._jsonable(res)
    _check_finite(res)
    res["_files"] = {"inner_preds.npz": arrays}
    return res


CBP_HEAD_LL, CBP_HEAD_BRIER = 0.006, 0.0035


def cbp_headroom(gate_rec):
    """D4 diagnostic: the cbp nominee headroom rule at ONE seed from a qpc.utility.gate_record (log-loss excess <=
    0.006 and Brier excess <= 0.0035 on each task, inclusive). NOT qpc.utility's 0.0075 "headroom" flag. Selection
    (cbp.select) recomputes this from ll_excess / brier_excess itself; this field is a cross-check only."""
    out = {}
    for t in ("income", "occupation"):
        ll, br = float(gate_rec[t]["ll_excess"]), float(gate_rec[t]["brier_excess"])
        out[t] = {"ll_excess": ll, "brier_excess": br, "ll_ok": ll <= CBP_HEAD_LL, "brier_ok": br <= CBP_HEAD_BRIER,
                  "ll_shortfall": max(0.0, (ll - CBP_HEAD_LL) / CBP_HEAD_LL),
                  "brier_shortfall": max(0.0, (br - CBP_HEAD_BRIER) / CBP_HEAD_BRIER)}
    out["pass"] = all(out[t]["ll_ok"] and out[t]["brier_ok"] for t in ("income", "occupation"))
    out["rule"] = ("cbp headroom (prompt sec. 9): log-loss excess <= 0.006 nats and Brier excess <= 0.0035, each task, "
                   "THIS seed; diagnostic only (selection recomputes it on every seed)")
    return out


def _nonfinite_paths(o, path="$"):
    if isinstance(o, dict):
        return [p for k, v in o.items() for p in _nonfinite_paths(v, f"{path}.{k}")]
    if isinstance(o, (list, tuple)):
        return [p for j, v in enumerate(o) for p in _nonfinite_paths(v, f"{path}[{j}]")]
    if isinstance(o, (float, np.floating)) and not math.isfinite(float(o)):
        return [path]
    return []


def _check_finite(res):
    for w in PRIMARY_VIEWS:
        for key in ("auc", "ce"):
            v = res["recovery"][key][w]
            if v is None or not math.isfinite(float(v)):
                raise RuntimeError(f"TECHNICAL FAILURE: nonfinite recovery.{key}.{w} for {res['cid']}")
    bad = _nonfinite_paths({k: v for k, v in res.items() if k != "_files"})        # D4: every float finite
    if bad:
        raise RuntimeError(f"TECHNICAL FAILURE: nonfinite values in the inner record of {res.get('cid')}: {bad[:5]}")
    json.dumps({k: v for k, v in res.items() if k != "_files"}, allow_nan=False)


# ------------------------------------------------------------------ D5: independent post-hoc record checks
class AuditDefect(RuntimeError):
    """A saved inner record violates the registered audit contract (prompt sec. 14 deliberate-defect guards)."""


def _auc_fixed(y, p):
    """Fixed-orientation AUC of the score P(S = 1), computed with sklearn directly (independent of dpc.audit.auc1)."""
    return float(roc_auc_score(np.asarray(y) == 1, np.asarray(p, dtype=np.float64)))


def _ce_fixed(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    return float(-np.mean(np.log(np.clip(np.where(y == 1, p, 1.0 - p), 1e-12, 1.0))))


def _first_best(vals, maximize):
    """The registered tie rule: first bank row, replaced only by a later row better by more than 1e-12."""
    b = 0
    for j in range(1, len(vals)):
        if (vals[j] > vals[b] + TIE) if maximize else (vals[j] < vals[b] - TIE):
            b = j
    return b


def validate_composed(rec, expected=None):
    """Problems (list of str) of a SOURCE inner record's composition: every composed winner of every (family, view,
    criterion) is frozen, every frozen policy won something, the composed value is never below the own bank's, the
    decisions family composed only with the class-only code, closure ok and (when given) the bank equals `expected`."""
    bad = []
    comp = rec.get("composed") or {}
    freeze = list(comp.get("freeze") or [])
    if not (comp.get("closure") or {}).get("ok"):
        bad.append("composition closure is not ok")
    if expected is not None and list(comp.get("policies") or []) != list(expected):
        bad.append("composed bank differs from the registered composition bank")
    fams = {rec["primary_family"]: rec["recovery"], **(rec.get("families") or {})}
    won = set()
    for fam, r in fams.items():
        cr = r.get("composed")
        if cr is None:
            bad.append(f"{fam}: no composed record")
            continue
        if fam == "decisions" and any(not is_class_only(c) for c in cr.get("policies", [])):
            bad.append("decisions family composed with a code that is not a function of the decision")
        for crit in ("winner", "ce_winner"):
            for w, c in cr[crit].items():
                if c == "source":
                    continue
                won.add(c)
                if c not in freeze:
                    bad.append(f"composed winner {c} ({fam}/{w}/{crit}) is missing from composed.freeze")
        own = r.get("own") or {}
        for w in PRIMARY_VIEWS:
            if own and cr["auc_seed0"][w] < own["auc_seed0"][w] - TIE:
                bad.append(f"{fam}/{w}: composed selection value below the own bank's")
            if own and cr["ce_seed0"][w] > own["ce_seed0"][w] + TIE:
                bad.append(f"{fam}/{w}: composed CE selection value above the own bank's")
    bad += [f"frozen policy {c} won no (family, view, criterion)" for c in freeze if c not in won]
    return bad


def validate_inner(rec, arrays, D, release=None, registered=True, tol=1e-12):
    """D5. Check one SAVED inner record against its stored INNER_SELECTION predictions; raises AuditDefect listing every
    violation, else returns a receipt. Uses an independent fixed-orientation AUC / CE (sklearn) and the registered
    first-best tie rule:
      orientation  every candidate's inner_auc / inner_ce equals the fixed-orientation statistic of its stored P(S=1)
                   (a flipped score or max(AUC, 1 - AUC) is caught);
      best reader  the AUC-selected candidate is the first bank row with the maximum inner AUC (CE: minimum) and
                   auc_seed0 / ce_seed0 equal it (a reversed best/worst selection is caught);
      seed refits  SEL_<fam>_<crit>_<view>[0] is the selected candidate's seed-0 prediction; per-seed statistics are
                   the fixed-orientation values of the stored refits and the own recovery is their mean;
      code views   (policies) finite bank with CC readers for v1 / v2 and CCpair on the exact tuple, coalition and both
                   ignore-the-other-recipient banks in the pair table; with `release`: exactly the registered release
                   keys (appended clean probabilities refused) and a view_fingerprint equal to that of the views
                   rebuilt from the release (q-only views or a misaligned pair view change it); token_states =
                   alpha1 + alpha2;
      sources      token_states null; validate_composed (every composed winner frozen; registered bank when
                   `registered`);
      JSON         every float finite."""
    bad = []
    _, sel_idx = DA.roles(D)
    ys = np.asarray(D["sex"])[sel_idx]
    if ys.size and ys.min() < 0:
        raise PermissionError("REFUSED: sealed SEX labels")
    if "sel_row_id" in arrays and not np.array_equal(np.asarray(arrays["sel_row_id"]),
                                                     np.asarray(D["row_id"])[sel_idx]):
        bad.append("stored INNER_SELECTION rows differ from D's")
    fams = {rec["primary_family"]: rec["recovery"], **(rec.get("families") or {})}
    for fam, r in fams.items():
        own = r.get("own") or r
        if f"keys_{fam}" not in arrays or f"P_{fam}" not in arrays:
            bad.append(f"{fam}: stored predictions missing")
            continue
        P = np.asarray(arrays[f"P_{fam}"])
        if r.get("inner_predictions_sha256") != sha_arrays(P):
            bad.append(f"{fam}: stored predictions differ from the recorded hash")
        Pk = dict(zip([str(x) for x in arrays[f"keys_{fam}"]], P))
        for w in PRIMARY_VIEWS:
            tab = r["tables"][w]
            if any(row["pred_key"] not in Pk for row in tab) or not tab:
                bad.append(f"{fam}/{w}: bank rows without stored predictions")
                continue
            A = [_auc_fixed(ys, Pk[row["pred_key"]]) for row in tab]
            C = [_ce_fixed(ys, Pk[row["pred_key"]]) for row in tab]
            for row, a, c in zip(tab, A, C):
                if abs(a - row["inner_auc"]) > tol or abs(c - row["inner_ce"]) > 1e-9:
                    bad.append(f"{fam}/{w}/{row['pred_key']}: stored inner AUC/CE {row['inner_auc']:.6f}/"
                               f"{row['inner_ce']:.6f} != fixed-orientation {a:.6f}/{c:.6f} (orientation)")
                    break
            for crit, vals, mx, sk, tag in (("auc", A, True, "selected", "auc_seed0"),
                                            ("ce", C, False, "ce_selected", "ce_seed0")):
                j = _first_best(vals, mx)
                sel = r["selection"][w][crit]
                lab = f"{tab[j]['candidate']}:{tab[j]['view']}:{tab[j]['attacker']}"
                if sel.get("bank_index") != j or sel.get("pred_key") != tab[j]["pred_key"] or own[sk][w] != lab:
                    bad.append(f"{fam}/{w}: the {crit}-selected reader {own[sk][w]} is not the registered best "
                               f"{lab} (best/worst reversal or wrong tie rule)")
                if abs(float(own[tag][w]) - vals[j]) > tol:
                    bad.append(f"{fam}/{w}: {tag} {own[tag][w]} != best bank value {vals[j]}")
                key = f"SEL_{fam}_{crit}_{w}"
                if key not in arrays:
                    bad.append(f"{fam}/{w}: seed refits {key} missing")
                    continue
                S3 = np.asarray(arrays[key])
                if not np.array_equal(S3[0], Pk[sel["pred_key"]]):
                    bad.append(f"{fam}/{w}: seed-0 refit of the {crit}-selected reader is not its bank prediction")
                stat = _auc_fixed if crit == "auc" else _ce_fixed
                per = [stat(ys, x) for x in S3]
                if max(abs(a - b) for a, b in zip(per, r[f"{crit}_per_seed"][w])) > 1e-9 or \
                        abs(float(np.mean(per)) - float(own[crit][w])) > 1e-9:
                    bad.append(f"{fam}/{w}: reported {crit} is not the seed mean of the stored refits")
    if rec.get("kind") == "policy":
        r = rec["recovery"]
        keys = set(r.get("inner_prediction_keys") or [])
        need = {f"{v}:CC_alpha{a}" for v in ("v1", "v2") for a in CC_ALPHAS} | {f"pair:CCpair_alpha{a}" for a in
                                                                                  CC_ALPHAS}
        if not r.get("finite") or not need <= keys:
            bad.append("code audited without token-identity readers (finite bank / CC readers missing)")
        cands = {row["candidate"] for row in r["tables"]["pair"]}
        if not {"coalition", "ignore_recipient_2", "ignore_recipient_1"} <= cands:
            bad.append("pair bank lacks the coalition or an ignore-the-other-recipient bank")
        if release is not None:
            try:
                check_release_keys(release)
                fp = DA.view_fingerprint(policy_views(release, D))
                if fp != rec.get("view_fingerprint"):
                    bad.append("audited views differ from the release's registered code views (token identities, "
                               "appended probabilities or pair alignment)")
                a12 = sum(int(np.asarray(release[f"alpha{i}"]).ravel()[0]) for i in (1, 2))
                if rec.get("token_states") != a12:
                    bad.append("token_states != alpha1 + alpha2")
            except ValueError as e:
                bad.append(f"release refused: {e}")
    elif rec.get("kind") in ("source", "reference"):
        if rec.get("token_states") is not None:
            bad.append("token_states of a continuous release must be null")
    if rec.get("kind") == "source":
        exp = None
        if registered:
            t = parse_cid(rec["cid"])["teacher"]
            exp = [c for c in registered_composition_bank() if parse_cid(c)["teacher"] == t]
        bad += validate_composed(rec, exp)
    nf = _nonfinite_paths(rec)
    if nf:
        bad.append(f"nonfinite values {nf[:3]}")
    if bad:
        raise AuditDefect(f"{rec.get('cid')} s{rec.get('seed')}: " + "; ".join(bad))
    return {"ok": True, "cid": rec.get("cid"), "seed": rec.get("seed"), "families": sorted(fams)}


def validate_all(D, units_dir=None, seeds=(0, 1, 2)):
    """D5 over the whole saved inner bank (lead / verifier; inner roles only): every registered code and composition-
    only map (with its release), both sources (registered composition) and the three references, every seed. Returns
    {"ok", "checked", "defects": [...], "missing": [...]}; never raises on a defect (the caller decides)."""
    guard_inner(D)
    ids = registered_composition_bank() + ["SRC|U", "SRC|RAW-J_b0.3", "REF|E", "REF|F", "REF|F0"]
    out = {"checked": 0, "defects": [], "missing": []}
    for k in seeds:
        for cid in ids:
            name = f"inner__{unit_of(k, cid)}"
            if not _complete(name, units_dir):
                out["missing"].append(name)
                continue
            rec, arr = load_inner(name, units_dir)
            rel = BL.load_unit_npz(unit_of(k, cid), "release.npz", units_dir)[0] if \
                parse_cid(cid)["kind"] == "policy" else None
            try:
                validate_inner(rec, arr, D, release=rel, registered=True)
            except AuditDefect as e:
                out["defects"].append(str(e))
            out["checked"] += 1
    out["ok"] = not out["defects"] and not out["missing"]
    return out


# ------------------------------------------------------------------ real-data controls (lead runs; inner roles only)
def _split_audit(views, Sp, D, halves, slate, finite=None):
    """Fit AUDIT_FIT, select on half A over the whole bank, evaluate on held-out half B (+ per-attacker B table)."""
    fit_idx, sel_idx = DA.roles(D)
    a, b = halves
    fin = DA._is_finite(views, finite)
    preds, _, cov, _ = DA._source_bank(views, Sp, fit_idx, sel_idx, a, slate, 0, fin, None)
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


def rotated_views(views, Sp, recipient, amp=ROT_AMP, seed=CONTROL_SEED):
    """Continuous view copy whose v_i is [v_i, amp (noisy S* - 0.5)] Q (Haar-random Q); the pair uses the planted
    v_i."""
    noisy = SA.noisy_sex(np.asarray(Sp), seed)
    w = f"v{recipient}"
    X = np.asarray(views["X"][w], dtype=np.float64)
    Q = SA.random_rotation(X.shape[1] + 1, seed + 2 + recipient)
    Xp = np.hstack([X, amp * (noisy[:, None] - 0.5)]) @ Q
    Xs = dict(views["X"])
    Xs[w] = Xp
    Xs["pair"] = np.hstack([Xs["v1"], Xs["v2"]])
    meta = {"recipient": recipient, "amplitude": amp, "rotation_seed": seed + 2 + recipient,
            "rotation_dim": int(X.shape[1] + 1), "max_coordinate_share_of_clue_direction": float(np.abs(Q[-1]).max())}
    return {"family": f"{views.get('family')}+ROT", "X": Xs, "tokens": None, "meta": meta}


def _roundtrip_views(V, tmp, tag):
    p = Path(tmp) / f"{tag}.npz"
    np.savez_compressed(p, **{w: V["X"][w] for w in PRIMARY_VIEWS})
    z = np.load(p)
    exact = all(np.array_equal(z[w], V["X"][w]) for w in PRIMARY_VIEWS)
    return {**V, "X": {w: z[w] for w in PRIMARY_VIEWS}}, exact


def controls_for_release(label, views, D, z_policy=None, rotate=False, slate="final", workdir=None):
    """Null (a) on one release; plants (b)-(d) when z_policy is a code release; rotated plant (e) when rotate=True
    (continuous views). Inner roles, permuted labels; explicit pass rules (module docstring)."""
    guard_inner(D)
    sex_rows(D, "controls", FIT_ROLE)
    sex_rows(D, "controls", SEL_ROLE)
    S = np.asarray(D["sex"])
    halves = SA.null_split(D)
    Sp, perm_sha = SA.frozen_permutation(S, D, halves)
    thr, sd = SA.null_threshold(Sp, D, halves)
    res = {"label": label, "family": views.get("family"), "permutation_sha256": perm_sha, "null_threshold": thr,
           "null_sd_B": sd, "plant_min": PLANT_MIN, "checks": {}}
    t0 = time.time()
    nul = _split_audit(views, Sp, D, halves, slate)
    res["checks"]["NULL"] = {**{w: {**_brief(nul)[w], "ok": nul[w]["heldout_auc_B"] <= thr} for w in PRIMARY_VIEWS},
                             "rule": "AUC_B <= 0.5 + 3.5 sd0 on every view"}
    res["checks"]["NULL"]["ok"] = all(res["checks"]["NULL"][w]["ok"] for w in PRIMARY_VIEWS)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        if z_policy is not None:
            zz = DA.align(z_policy, D)
            check_release_keys(zz)
            noisy = SA.noisy_sex(Sp, CONTROL_SEED)
            rng = np.random.default_rng(CONTROL_SEED + 7)
            b1 = rng.integers(0, 2, len(noisy))
            plants = {"CONF_r1": ([(1, noisy, False)], "v1"), "CONF_r2": ([(2, noisy, False)], "v2"),
                      "COLL_r1": ([(1, noisy, True)], "v1"), "COLL_r2": ([(2, noisy, True)], "v2"),
                      "XOR": ([(1, b1, True), (2, b1 ^ noisy, True)], "pair")}
            for pname, (ops, target) in plants.items():
                zp = dict(zz)
                for i, bit, coll in ops:
                    zp = DA.split_tokens(zp, i, bit, collide=coll)
                z2, exact = DA._roundtrip(zp, tmp, pname)
                V = policy_views(z2, D, meta={"plant": pname})
                r = _split_audit(V, Sp, D, halves, slate)
                entry = {"target_view": target, "serialisation_exact": exact, **_brief(r), "coverage": r["coverage"]}
                if pname.startswith("CONF") or pname == "XOR":
                    rd = _split_audit(DA.decision_views(z2["hard1"], z2["hard2"], D), Sp, D, halves, slate)
                    entry["decisions_only_audit"] = _brief(rd)
                    entry["decisions_only_misses_it"] = rd[target]["heldout_auc_B"] <= thr
                if pname.startswith("COLL"):
                    qv = {"family": "decoded_q_only", "tokens": None,
                          "X": {"v1": z2["q1"], "v2": z2["q2"], "pair": np.hstack([z2["q1"], z2["q2"]])}}
                    rq = _split_audit(qv, Sp, D, halves, slate)
                    entry["decoded_probability_only_audit"] = _brief(rq)
                    entry["decoded_probability_only_misses_it"] = rq[target]["heldout_auc_B"] <= thr
                detected = r[target]["heldout_auc_B"] > PLANT_MIN
                if pname == "XOR":
                    entry["locals_null_ok"] = all(r[w]["heldout_auc_B"] <= thr for w in ("v1", "v2"))
                    entry["rule"] = "AUC_B(pair) > 0.75 and AUC_B(v1), AUC_B(v2) <= null threshold"
                    entry["ok"] = bool(exact and detected and entry["locals_null_ok"])
                else:
                    entry["rule"] = f"AUC_B({target}) > 0.75 (serialisation exact)"
                    entry["ok"] = bool(exact and detected)
                res["checks"][pname] = entry
        if rotate:
            for i in (1, 2):
                Vr, exact = _roundtrip_views(rotated_views(views, Sp, i), tmp, f"ROT_r{i}")
                r = _split_audit(Vr, Sp, D, halves, slate, finite=False)
                w = f"v{i}"
                entry = {"target_view": w, "serialisation_exact": exact, "plant": Vr["meta"], **_brief(r),
                         "per_attacker_auc_B": {x: r[x]["per_attacker_auc_B"] for x in (w, "pair")},
                         "canon_receipt": {x: SA.Canon().fit(Vr["X"][x][DA.roles(D)[0]]).receipt()
                                           for x in (w, "pair")},
                         "rule": f"serialisation exact and AUC_B({w}) > 0.75 and AUC_B(pair) > 0.75"}
                entry["ok"] = bool(exact and r[w]["heldout_auc_B"] > PLANT_MIN and
                                   r["pair"]["heldout_auc_B"] > PLANT_MIN)
                res["checks"][f"ROT_r{i}"] = entry
    res["failures"] = [p for p, e in res["checks"].items() if not e["ok"]]
    res["all_ok"] = not res["failures"]
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def null_calibration(views, D, reps=5, slate="final", seed0=CONTROL_SEED + 100):
    """Repeated shuffled-label nulls on one release (select half A, evaluate half B); every rep kept."""
    guard_inner(D)
    halves = SA.null_split(D)
    rows = []
    for k in range(reps):
        Sp, sha = SA.frozen_permutation(np.asarray(D["sex"]), D, halves, seed=seed0 + k)
        thr, sd = SA.null_threshold(Sp, D, halves)
        r = _split_audit(views, Sp, D, halves, slate)
        for w in PRIMARY_VIEWS:
            rows.append({"rep": k, "view": w, "heldout_auc_B": r[w]["heldout_auc_B"], "max_auc_A": r[w]["max_auc_A"],
                         "selected_on_A": r[w]["selected_on_A"], "threshold": thr, "sd0": sd,
                         "exceeds": r[w]["heldout_auc_B"] > thr, "permutation_sha256": sha})
    hb = np.array([x["heldout_auc_B"] for x in rows])
    zz = (hb - 0.5) / np.array([x["sd0"] for x in rows])
    return {"summary": {"reps": reps, "tests": len(rows), "exceedances": int(sum(x["exceeds"] for x in rows)),
                        "heldout_mean": float(hb.mean()), "heldout_max": float(hb.max()), "z_mean": float(zz.mean()),
                        "z_sd": float(zz.std(ddof=1)) if len(zz) > 1 else None, "null_z": NULL_Z,
                        "rule": "no exceedance of 0.5 + 3.5 sd0"}, "rows": rows}


# D6: fixed control plan (registered before any real-data control runs; chosen by structure, never by results; the
# same structure as the qpc plan that ran):
#   codes (null + CONF_r1/r2 + COLL_r1/r2 + XOR), seed 0: U|DIRECT-TASK|i8o64 (the task-only code at the one rate),
#   the JOINT code at the one rate and the LARGEST registered lambda (U|JOINT|i8o64|l0.1), and U|CLASS|i1o1;
#   sources (null on all five families + ROT_r1/ROT_r2 on the interface family), seed 0: U and RAW-J_b0.3;
#   references (null on every family; ROT_r1/ROT_r2 on E's interface family), seed 0: E, F, F0;
#   null calibration: 5 permutations on the first code.
# The composition-only extras are not control targets (they are never selection candidates).
def control_plan(protocol=None):
    pols = [REG_TASK[0], f"U|JOINT|i{REG_RATE[0]}o{REG_RATE[1]}|l{_g(max(REG_LAMS))}", REG_TASK[2]]
    return {"policies": [(0, c) for c in pols], "sources": [("U", 0), ("RAW-J_b0.3", 0)],
            "references": [("E", 0), ("F", 0), ("F0", 0)], "rotate_references": ["E"], "null_policy": (0, pols[0]),
            "null_reps": 5}


def _control_jobs(plan):
    jobs = [("policy", k, c) for k, c in plan["policies"]] + [("source", k, t) for t, k in plan["sources"]] + \
           [("reference", k, lab) for lab, k in plan["references"]]
    if plan.get("null_policy"):
        jobs.append(("calibration", plan["null_policy"][0], plan["null_policy"][1]))
    return jobs


def run_control_job(job, D, plan, units_dir=None, slate="final", workdir=None):
    kind, k, x = job
    if kind == "policy":
        u = unit_of(k, x)
        z, sha = BL.load_unit_npz(u, "release.npz", units_dir)
        V = policy_views(z, D, meta={"kind": "policy", "seed": k, "unit": u, "config": x})
        return {f"{x}|s{k}": {"unit": u, "complete_sha256": sha,
                              **controls_for_release(x, V, D, z_policy=z, slate=slate, workdir=workdir)}}
    if kind == "source":
        t, prov = BL.load_teacher(x, k, D, units_dir)
        out = {}
        for fam, V in BL.source_view_sets(t, D).items():
            lab = f"SRC|{x}|s{k}|{fam}"
            out[lab] = {"complete_sha256": prov["complete_sha256"],
                        **controls_for_release(lab, V, D, rotate=(fam == "interface"), slate=slate, workdir=workdir)}
        return out
    if kind == "reference":
        sets, _, prov = BL.reference_view_sets(x, k, D, units_dir=units_dir, with_outputs=True)
        out = {}
        for fam, V in sets.items():
            lab = f"REF|{x}|s{k}|{fam}"
            rot = fam == "interface" and x in plan.get("rotate_references", [])
            out[lab] = {"complete_sha256": prov.get("complete_sha256"),
                        **controls_for_release(lab, V, D, rotate=rot, slate=slate, workdir=workdir)}
        return out
    z, _ = BL.load_unit_npz(unit_of(k, x), "release.npz", units_dir)
    return {"NULL_CALIBRATION": {"release": f"{x}|s{k}",
                                  **null_calibration(policy_views(z, D), D, plan["null_reps"], slate)}}


def summarise_controls(parts, plan, slate="final", source_threshold=SOURCE_NULL_THRESHOLD):
    releases = {k: v for p in parts for k, v in p.items() if k != "NULL_CALIBRATION"}
    calib = next((p["NULL_CALIBRATION"] for p in parts if "NULL_CALIBRATION" in p), None)
    failures = [f"{lab}:{f}" for lab, r in releases.items() for f in r["failures"]]
    calib_exc = (calib or {}).get("summary", {}).get("exceedances", 0)
    thr = sorted({float(r["null_threshold"]) for r in releases.values()} |
                 {float(x["threshold"]) for x in (calib or {}).get("rows", [])})
    if source_threshold is not None:            # D6: realised threshold receipt against the source's
        thr_ok = bool(thr) and all(abs(t - source_threshold) <= 1e-12 for t in thr)
        if not thr_ok:
            failures.append(f"REALISED_NULL_THRESHOLD {thr} != source {source_threshold}")
    else:
        thr_ok = None
    verdict = {"all_ok": not failures and not calib_exc and calib is not None, "failures": failures,
               "realised_null_thresholds": thr, "realised_threshold_matches_source": thr_ok,
               "null_calibration_exceedances": calib_exc, "null_calibration_present": calib is not None,
               "positive_controls": {lab: {p: e["ok"] for p, e in r["checks"].items() if p != "NULL"}
                                     for lab, r in releases.items() if len(r["checks"]) > 1},
               "decisions_only_audit_misses_confidence_plant": {
                   lab: {p: e.get("decisions_only_misses_it") for p, e in r["checks"].items() if p.startswith("CONF")}
                   for lab, r in releases.items() if any(p.startswith("CONF") for p in r["checks"])},
               "on_failure": "a failed positive control triggers technical review and an amendment of the audit; it "
                             "is never interpreted as low leakage"}
    return {"study": STUDY, "stage": "controls (inner roles only; sealed D)", "registered_limits": CONTROL_LIMITS,
            "roles": {"fit": FIT_ROLE, "select_half_A_evaluate_half_B": SEL_ROLE, "assessment": "never indexed"},
            "slate": slate, "slate_members": [nm for nm, _ in DA.SLATES[slate]()], "cc_alphas": list(CC_ALPHAS),
            "plan": DA._jsonable(plan),
            "label": "S* = SEX permuted within AUDIT_FIT, half A and half B (smf.audit.frozen_permutation, "
                     "CONTROL_SEED); plants encode noisy S* (20% random)",
            "thresholds": {"null": "AUC_B <= 0.5 + 3.5 sd0", "plant_min": PLANT_MIN, "null_z": NULL_Z,
                           "rot_amplitude": ROT_AMP, "conf_eta": CONF_ETA},
            "releases": releases, "null_calibration": calib, "verdict": verdict}


def stage_controls(D, shard_spec=None, units_dir=None, out_path=None, slate="final", plan=None, workdir=None,
                   source_threshold=SOURCE_NULL_THRESHOLD):
    """cbp.run LATE "controls" entry: runs this shard's jobs of the fixed plan (private partial receipts); when every
    job is done, writes the public AUDIT_PRELOCK_CHECKS.json (finite JSON; refuses private paths) and returns its
    verdict (pass limits: CONTROL_LIMITS)."""
    import platform
    import resource
    if not D.get("sealed", False):
        raise SystemExit("REFUSED: controls run on the sealed D only")
    guard_inner(D)
    plan = plan or control_plan()
    jobs = _control_jobs(plan)
    i, n = (0, 1) if not shard_spec else (int(x) for x in str(shard_spec).split("/"))
    part_dir = _units(units_dir).parent / "controls"
    part_dir.mkdir(parents=True, exist_ok=True)
    t0, c0 = time.time(), time.process_time()
    for j, job in enumerate(jobs):
        p = part_dir / f"job{j:02d}.json"
        if j % n != i or p.exists():
            continue
        r = run_control_job(job, D, plan, units_dir, slate, workdir)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(DA._jsonable({"job": list(job), "result": r,
                                                "cpu_s": round(time.process_time() - c0, 1)})))
        tmp.rename(p)
    done = [part_dir / f"job{j:02d}.json" for j in range(len(jobs))]
    if not all(p.exists() for p in done):
        return {"status": "PARTIAL", "jobs_done": sum(p.exists() for p in done), "jobs": len(jobs)}
    parts = [json.loads(p.read_text())["result"] for p in done]
    out = summarise_controls(parts, plan, slate, source_threshold=source_threshold)
    out["compute"] = {"this_shard_wall_s": round(time.time() - t0, 1),
                      "this_shard_cpu_s": round(time.process_time() - c0, 1),
                      "job_cpu_s": [json.loads(p.read_text()).get("cpu_s") for p in done],
                      "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "omp_threads": 1,
                      "python": platform.python_version()}
    out_path = out_path or (_R().PKG / "AUDIT_PRELOCK_CHECKS.json")
    txt = json.dumps(DA._jsonable(out), indent=1, allow_nan=False)        # D6: finite public JSON
    for bad in (str(Path.home()), "PCRL_eval_cache_private", "/Users/", "/Volumes/"):
        if bad in txt:
            raise SystemExit("REFUSED: a private path would enter the public control receipt")
    Path(out_path).write_text(txt + "\n")
    return out["verdict"]


# ------------------------------------------------------------------ synthetic fixtures and timing (no real data)
def synthetic_D(seed=0, **sizes):
    """Study-shaped synthetic D with every role, task labels and SEX (P(S=1) = 0.68); assessment labels sealed."""
    return UT.synthetic_task_D(seed=seed, **sizes)


def synthetic_teacher(D, seed=0, absent_class=5):
    """U-shaped synthetic teacher: occupation never predicts `absent_class` (as U's class 5)."""
    t = UT.synthetic_teacher(D, seed=seed)
    n = len(D["row_id"])
    rng = np.random.default_rng(seed + 7)
    L = np.log(np.clip(t["p2"], 1e-300, 1))
    if absent_class is not None:
        L[:, absent_class] -= 30.0
    P = np.exp(L - L.max(1, keepdims=True))
    t["p2"] = P / P.sum(1, keepdims=True)
    t["d2"] = t["p2"].argmax(1)
    for i, K in ((1, 2), (2, 6)):
        c = np.log(np.clip(t[f"p{i}"], 1e-300, 1))
        t[f"c{i}"] = c - c.mean(1, keepdims=True)
        t[f"r{i}"] = np.hstack([t[f"c{i}"], rng.normal(size=(n, 4))])
    return t


def synthetic_release(D, t, m1, m2, eps=1e-12, sex_tilt=0.0, seed=0):
    """Class-preserving finite code of teacher t: per predicted class, m tokens by within-class quantiles of the top
    probability (label-free); q = smoothed OSF_DEFENSE_FIT token means; a reserved fallback token per absent class.
    sex_tilt > 0 (synthetic only) moves rows to the next token by SEX to create a leak."""
    rng = np.random.default_rng(seed)
    fit = _idx(D, FIT_NAME)
    z = {"row_id": np.asarray(D["row_id"])}
    for i, (K, m) in enumerate(((2, m1), (6, m2)), start=1):
        p, d = np.asarray(t[f"p{i}"]), np.asarray(t[f"d{i}"])
        present = sorted(set(np.unique(d[fit]).tolist()))
        base, tok = {}, np.zeros(len(d), dtype=np.int64)
        nxt = 0
        for c in range(K):
            if c in present:
                base[c] = nxt
                nxt += m
        reserve = {c: nxt + j for j, c in enumerate([c for c in range(K) if c not in present])}
        alpha = nxt + len(reserve)
        Q = np.zeros((alpha, K))
        for c in range(K):
            rows = np.flatnonzero(d == c)
            if c not in present:
                tok[rows] = reserve[c]
                Q[reserve[c]] = (np.eye(K)[c] + eps + eps * np.eye(K)[c]) / (1 + (K + 1) * eps)
                continue
            fr = rows[np.isin(rows, fit)]
            edges = np.quantile(p[fr, c], np.linspace(0, 1, m + 1)[1:-1]) if m > 1 else np.zeros(0)
            j = np.searchsorted(edges, p[rows, c], side="right")
            if sex_tilt:
                j = np.where((np.asarray(D["sex"])[rows] == 1) & (rng.random(len(rows)) < sex_tilt),
                             np.minimum(j + 1, m - 1), j)
            tok[rows] = base[c] + j
            for jj in range(m):
                mm = fr[(tok[fr] == base[c] + jj)]
                mu = p[mm].mean(0) if len(mm) else np.eye(K)[c]
                Q[base[c] + jj] = (mu + eps + eps * np.eye(K)[c]) / (1 + (K + 1) * eps)
        q = Q[tok]
        assert np.array_equal(q.argmax(1), d), "synthetic code is not class preserving"
        z.update({f"tok{i}": tok, f"q{i}": q, f"hard{i}": d.copy(), f"alpha{i}": np.asarray(alpha)})
    return z


def synthetic_reference_cells(D, t, ncell=(40, 60), seed=0):
    """F-like reference: one-hot cell code r_i, cell-constant probabilities and decisions."""
    rng = np.random.default_rng(seed)
    z = {"row_id": np.asarray(D["row_id"])}
    n = len(D["row_id"])
    for i, (K, nc) in enumerate(((2, ncell[0]), (6, ncell[1])), start=1):
        cells = rng.integers(0, nc, n)
        P = rng.dirichlet(np.ones(K), nc)
        z[f"cells{i}"], z[f"r{i}"] = cells, np.eye(nc)[cells]
        z[f"p{i}"], z[f"d{i}"] = P[cells], P[cells].argmax(1)
        c = np.log(P[cells])
        z[f"c{i}"] = c - c.mean(1, keepdims=True)
    return z


CBP_BANK = {
    "seeds": 3,
    "codes_per_seed": {"i8o64": 26, "i1o1": 1},
    "codes_note": "27 registered codes per seed: DIRECT-TASK i8o64, FINE-TASK i8o64, CLASS-ONLY i1o1, 24 privacy codes "
                  "(6 lambdas x LOCAL / SEQ-12 / SEQ-21 / JOINT) at i8o64",
    "composition_only_per_seed": {"i4o8": 1, "i4o16": 1, "i4o32": 1, "i4o64": 1, "i8o8": 1, "i8o16": 1, "i8o32": 1,
                                  "i8o64": 4},
    "composition_only_note": "the 11 COMPOSED_EXTRA qpc public maps (inner-audited, never selection candidates); "
                             "i4oX is costed at the measured i8oX (one more income state per class; slightly "
                             "conservative)",
    "sources": 2, "references": {"continuous_E": 1, "cells_F_F0": 2},
    "composed_codes_per_source_seed": {"SRC|U": 38, "SRC|RAW-J_b0.3": 0},
    "inner_units_registered": 96, "inner_units_composition_only": 33,
    "control_plan": {"codes": 3, "source_families": 10, "reference_families": 15, "rot_views": 3,
                     "null_calibration_reps": 5},
    "assessment": {"labels_per_seed": 20, "codes_per_seed": 15, "sources_per_seed": 2, "references_per_seed": 3,
                   "src_u_composed_codes": 38,
                   "note": "about 20 locked labels per seed (lead): ~15 codes (P*, J*, T*, C_rate, C_global, Q, CLASS, "
                           "source JOINT / SEQ-12 / SEQ-21 0.1, no-headroom winner, per-family headroom winners; "
                           "aliases deduplicated), SRC|U, SRC|RAW-J, E, F, F0. SRC|U is costed with ALL 38 composition "
                           "codes as composed readers (upper bound; the freeze list alone is smaller)"}}
LEAD_BANK = CBP_BANK                             # name kept for callers of the qpc API
# qpc's measured real study-worker CPU (results/pcrl_confidence_capacity_v1/COST_AND_CLOSEOUT.md) over its synthetic
# estimate (its AUDIT_COMPUTE.json): inner + inner_src 2,611 s vs 0.565 h; controls 714 s vs 0.318 h; assessment
# 3,588 s vs 1.144 h. Used only to report a calibrated planning figure next to the raw synthetic estimate.
QPC_CALIBRATION = {"real_over_synthetic": {"inner": round(2611 / (0.565 * 3600), 3),
                                           "controls": round(714 / (0.318 * 3600), 3),
                                           "assessment": round(3588 / (1.144 * 3600), 3)},
                   "source": "results/pcrl_confidence_capacity_v1/COST_AND_CLOSEOUT.md (study workers) and "
                             "AUDIT_COMPUTE.json (estimates)"}


def _cpu(fn, *a, **kw):
    c0, t0 = time.process_time(), time.time()
    r = fn(*a, **kw)
    return r, round(time.process_time() - c0, 2), round(time.time() - t0, 2)


def estimates(meas, bank=None):
    """D7: CPU estimates of the cbp inner bank (96 registered + 33 composition-only units), controls and the single
    assessment, from the synthetic measurements."""
    bank = bank or CBP_BANK
    pin = meas["policy_inner"]
    unit = lambda rate: pin[{"i4o8": "i8o8", "i4o16": "i8o16", "i4o32": "i8o32", "i4o64": "i8o64"}.get(rate, rate)][
        "unit_cpu_s"]  # noqa: E731
    codes = sum(n * unit(r) for r, n in bank["codes_per_seed"].items())
    extra = sum(n * unit(r) for r, n in bank["composition_only_per_seed"].items())
    src = meas["source_unit"]["SRC|U_cpu_s"] + meas["source_unit"]["SRC|RAW-J_cpu_s"]
    refs = meas["reference_unit"]["E_cpu_s"] + bank["references"]["cells_F_F0"] * meas["reference_unit"]["F_cpu_s"]
    per_seed = codes + extra + src + refs
    est = {"inner_units": bank["inner_units_registered"] + bank["inner_units_composition_only"],
           "inner_cpu_s_per_seed": round(per_seed, 1),
           "inner_cpu_s_per_seed_parts": {"registered_codes": round(codes, 1), "composition_only": round(extra, 1),
                                          "sources_incl_composition": round(src, 1), "references": round(refs, 1)},
           "inner_cpu_h": round(bank["seeds"] * per_seed / 3600, 3),
           "per_unit_cpu_s": {r: v["unit_cpu_s"] for r, v in pin.items()},
           "composed_assembly_cpu_s_38_codes": meas["source_unit"].get("composed_assembly_cpu_s")}
    if meas.get("controls"):
        cp, c = bank["control_plan"], meas["controls"]
        fam_null = meas["source_unit"]["SRC|RAW-J_cpu_s"] / 5          # one family's null ~ one family's inner audit
        ctl = (cp["codes"] * c["code_null_and_5_plants_cpu_s"] + (cp["source_families"] + cp["reference_families"]) *
               fam_null + cp["rot_views"] * c["source_interface_null_and_2_rot_cpu_s"] +
               cp["null_calibration_reps"] * c["code_null_only_cpu_s"])
        est["controls_cpu_h"] = round(ctl / 3600, 3)
    if meas.get("final"):
        a, f = bank["assessment"], meas["final"]
        code_f = f["code_i8o64_one_family_cpu_s"]
        ratio = {r: unit(r) / unit("i8o64") for r in list(bank["codes_per_seed"]) +
                 list(bank["composition_only_per_seed"])}
        comp = (sum(n * code_f * ratio[r] for r, n in bank["codes_per_seed"].items()) +
                sum(n * code_f * ratio[r] for r, n in bank["composition_only_per_seed"].items()))
        per = (a["codes_per_seed"] * code_f + 2 * f["source_own_five_families_cpu_s"] + comp +
               a["references_per_seed"] * f["source_own_five_families_cpu_s"] + 2 * code_f)
        est["assessment_cpu_h"] = round(bank["seeds"] * per / 3600, 3)
        est["assessment_basis"] = (
            f"per seed: {a['codes_per_seed']} codes x one family at the i8o64 cost; both sources' five own families; "
            f"SRC|U composed with all {a['src_u_composed_codes']} composition codes (each composed code's bank fitted "
            "once per unit and shared across the four composing families by the content memo; cost scaled by its "
            "inner cost); E / F / F0 at the five-family source cost plus one cells family each for F / F0")
    total = est["inner_cpu_h"] + est.get("controls_cpu_h", 0) + est.get("assessment_cpu_h", 0)
    est["total_cpu_h"] = round(total, 3)
    cal = QPC_CALIBRATION["real_over_synthetic"]
    ct = (est["inner_cpu_h"] * cal["inner"] + est.get("controls_cpu_h", 0) * cal["controls"] +
          est.get("assessment_cpu_h", 0) * cal["assessment"])
    est["qpc_calibrated_total_cpu_h"] = round(ct, 3)
    est["qpc_calibration"] = QPC_CALIBRATION
    est["budget_with_x2_margin_cpu_h"] = round(2 * total, 2)
    est["wall_h_two_workers"] = round(total / 2, 2)
    est["safety_note"] = ("synthetic MLP early stopping converges fast; qpc's synthetic estimate (4.05 CPU-h with x2) "
                          "against ~2 CPU-h measured for its real study workers suggests the raw estimate is already "
                          "conservative; the x2 margin is the scheduling budget")
    return est


def reestimate(path, bank=None):
    """Recompute the estimates of an existing AUDIT_COMPUTE.json from its measurements (no new timing)."""
    d = json.loads(Path(path).read_text())
    d["bank"] = bank or CBP_BANK
    d["estimates"] = estimates(d["measured"], d["bank"])
    Path(path).write_text(json.dumps(DA._jsonable(d), indent=1, allow_nan=False) + "\n")
    return d["estimates"]


def _timing_store(root, D, t, rels, slate, meas):
    """Synthetic cbp-named store: teachers, references, the 38 composition codes (measured units for the distinct
    alphabets; the other names reuse the measured record of the same alphabet -- composition reads records only)."""
    from jcv.finalize import save_unit

    def _save(name, files, rec):
        save_unit(root / name, {f: (lambda p, a=a: np.savez_compressed(p, **a)) for f, a in files.items()}, rec)
    _save("tea__s0__U", {"teacher.npz": t}, {"t": "U"})
    _save("tea__s0__RAW-J_b0.3", {"teacher.npz": synthetic_teacher(D, seed=9)}, {"t": "RAW-J"})
    _save("ref__s0__E", {"reference.npz": {k: v for k, v in synthetic_teacher(D, seed=5).items()}}, {"t": "E"})
    _save("ref__s0__F", {"reference.npz": synthetic_reference_cells(D, t, ncell=(82, 97))}, {"t": "F"})
    rate_of = lambda c: "i1o1" if is_class_only(c) else "i{m1}o{m2}".format(**parse_cid(c))  # noqa: E731
    measured = {}
    for cid in registered_composition_bank():
        r = rate_of(cid)
        key = {"i4o8": "i8o8", "i4o16": "i8o16", "i4o32": "i8o32", "i4o64": "i8o64"}.get(r, r)
        _save(unit_of(0, cid), {"release.npz": rels[key]}, {"synthetic": True})
        if key not in measured:
            rec, cpu, wall = _cpu(inner_unit, "policy", 0, cid, D, units_dir=root, slate=slate)
            files = rec.pop("_files")
            measured[key] = (rec, files)
            meas["policy_inner"][key]["unit_cpu_s"] = cpu
            meas["policy_inner"][key]["unit_wall_s"] = wall
        rec, files = measured[key]
        _save(f"inner__{unit_of(0, cid)}", files, {**rec, "cid": cid, "unit_of": unit_of(0, cid)})


def timing(m2s=(8, 16, 32, 64), slate="final", seed=0, out_path=None, bank=None, controls=True, final=True,
           workdir=None):
    """D7: CPU seconds on study-shaped synthetic data at the study's real row counts and alphabets (occupation m2 per
    class x 5 predicted classes + 1 reserved token, income 8 per class x 2): full inner units (attack bank + seed refits
    + utility + coverage + MI) of every distinct code alphabet, both sources with the REGISTERED 38-code composition
    (real closure and registration checks), E- and F-like references, controls and final audits. Writes
    AUDIT_COMPUTE.json (finite JSON, with the registered control limits)."""
    import platform
    import resource
    bank = bank or CBP_BANK
    D = synthetic_D(seed=seed)
    t = synthetic_teacher(D, seed=seed)
    meas = {"policy_inner": {}, "source_unit": {}, "reference_unit": {}, "controls": {}, "final": {}}
    rels = {}
    for m1, m2 in [(1, 1)] + [(8, m) for m in m2s]:
        z = synthetic_release(D, t, m1, m2, sex_tilt=0.1)
        rels[f"i{m1}o{m2}"] = z
        V = policy_views(z, D)
        meas["policy_inner"][f"i{m1}o{m2}"] = {
            "alphabets": [int(z["alpha1"]), int(z["alpha2"])],
            "dims": {w: int(V["X"][w].shape[1]) for w in PRIMARY_VIEWS},
            "pair_tuples_occupied_audit_fit": int(len(np.unique(DA._pair_keys(z["tok1"], z["tok2"])[
                _idx(D, FIT_ROLE)])))}
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        root = Path(tmp) / "units"
        root.mkdir()
        _timing_store(root, D, t, rels, slate, meas)
        for lab, cid in (("SRC|U", "SRC|U"), ("SRC|RAW-J", "SRC|RAW-J_b0.3")):
            rec, cpu, wall = _cpu(inner_unit, "source", 0, cid, D, units_dir=root, slate=slate)
            meas["source_unit"][f"{lab}_cpu_s"] = cpu
            meas["source_unit"][f"{lab}_wall_s"] = wall
            if cid == "SRC|U":
                meas["source_unit"]["SRC|U_composed_banks"] = rec["composed"]["closure"]["expected"]
                fams = {rec["primary_family"]: rec["recovery"], **rec["families"]}
                own = {f: {**v, **v["own"]} for f, v in fams.items()}      # the own banks, from the saved record
                _, cpu, _ = _cpu(composed_source_bank, 0, "U", own, D, units_dir=root)
                meas["source_unit"]["composed_assembly_cpu_s"] = cpu
        for lab in ("E", "F"):
            _, cpu, wall = _cpu(inner_unit, "reference", 0, f"REF|{lab}", D, units_dir=root, slate=slate)
            meas["reference_unit"][f"{lab}_cpu_s"] = cpu
    zb = rels["i8o64"]
    if controls:
        _, cpu, _ = _cpu(controls_for_release, "code", policy_views(zb, D), D, z_policy=zb, slate=slate)
        meas["controls"]["code_null_and_5_plants_cpu_s"] = cpu
        _, cpu, _ = _cpu(controls_for_release, "code", policy_views(zb, D), D, slate=slate)
        meas["controls"]["code_null_only_cpu_s"] = cpu
        Vi = BL.source_view_sets(t, D, families=("interface",))["interface"]
        _, cpu, _ = _cpu(controls_for_release, "src", Vi, D, rotate=True, slate=slate)
        meas["controls"]["source_interface_null_and_2_rot_cpu_s"] = cpu
    if final:
        Du = dict(D)
        a = _idx(D, ASSESS)
        Vb = policy_views(zb, D)
        _, cpu, _ = _cpu(DA.final_audit, Vb, Du, a, slate=slate)
        meas["final"]["code_i8o64_one_family_cpu_s"] = cpu
        own = 0.0
        for fam, V in BL.source_view_sets(t, D).items():
            _, c1, _ = _cpu(DA.final_audit, V, Du, a, slate=slate)
            own += c1
        meas["final"]["source_own_five_families_cpu_s"] = round(own, 2)
        Vi = BL.source_view_sets(t, D, families=("interface",))["interface"]
        _, cpu, _ = _cpu(DA.final_audit, Vi, Du, a, composed=[("c1", Vb), ("c2", policy_views(rels["i8o32"], D))],
                         slate=slate)
        meas["final"]["source_interface_with_2_composed_cpu_s"] = cpu
        meas["final"]["score_rows"] = int(len(a))
    est = estimates(meas, bank)
    out = {"study": STUDY, "owner": "D (attacker engineer)",
           "kind": "SYNTHETIC timing estimates (no real data, no labels); real views may converge differently (MLP "
                   "early stopping); one process under cbp.sema, OMP_NUM_THREADS=1; cpu_s = process CPU time of the "
                   "measuring process",
           "fixture": "cbp.audit.synthetic_D (OSF_DEFENSE_FIT 15,434 / AUDIT_FIT 6,065 / INNER_SELECTION 2,235 / "
                      "assessment 13,936 rows, P(S=1) = 0.68) + synthetic_release (income 8 per class x 2, "
                      "occupation m2 per class x 5 predicted classes + 1 reserved token; SEX-tilted 10%); FINAL slate "
                      "(LR x5, MLP x4, HGB x4, DA_LR, DA_MLP) + cell readers {0.5, 1, 5}; the AUC- and CE-selected "
                      "attackers refit at attacker seeds 0-2; a temporary cbp-named unit store with the registered "
                      "38-code composition bank (real registration and closure checks)",
           "registered_control_limits": CONTROL_LIMITS,
           "registered_composition_bank": {"codes": registered_code_bank(), "composition_only": list(COMPOSED_EXTRA)},
           "slate": slate, "slate_members": [nm for nm, _ in DA.SLATES[slate]()], "cc_alphas": list(CC_ALPHAS),
           "bank": bank, "measured": meas, "estimates": est,
           "machine": {"python": platform.python_version(), "omp_threads": 1, "platform": platform.machine(),
                       "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
           "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = DA._jsonable(out)
    if out_path:
        txt = json.dumps(out, indent=1, allow_nan=False)
        for bad in (str(Path.home()), "PCRL_eval_cache_private", "/Users/", "/Volumes/", "/private/"):
            if bad in txt:
                raise SystemExit("REFUSED: a private path would enter AUDIT_COMPUTE.json")
        Path(out_path).write_text(txt + "\n")
    return out


def write_timing_key(audit_compute, timing_path):
    """Add ONLY the "audit" key to TIMING.json (owned by role B; read-modify-write, other keys untouched)."""
    d = json.loads(Path(audit_compute).read_text())
    tp = Path(timing_path)
    T = json.loads(tp.read_text()) if tp.exists() else {}
    e = d["estimates"]
    T["audit"] = {"owner": "role D (attacker engineer)", "written_at": d.get("written_at"),
                  "data": "SYNTHETIC ONLY (cbp.audit.synthetic_D at the real role sizes); no Adult row, task label or "
                          "SEX was read",
                  "environment": {"threads": "OMP_NUM_THREADS=1, one process", "semaphore": "cbp.sema label "
                                  "D:audit-timing", "python": d["machine"]["python"]},
                  "measure": "process CPU seconds (time.process_time) inside the measuring process",
                  "per_unit_cpu_s": e["per_unit_cpu_s"], "source_unit": d["measured"]["source_unit"],
                  "reference_unit": d["measured"]["reference_unit"], "controls": d["measured"]["controls"],
                  "final": d["measured"]["final"],
                  "estimates_cpu_h": {"inner": e["inner_cpu_h"], "controls": e.get("controls_cpu_h"),
                                      "assessment": e.get("assessment_cpu_h"), "total": e["total_cpu_h"],
                                      "with_x2_margin": e["budget_with_x2_margin_cpu_h"],
                                      "wall_h_two_workers": e["wall_h_two_workers"]},
                  "inner_units": e["inner_units"], "details": "results/pcrl_confidence_budgeted_privacy_v1/"
                                                              "AUDIT_COMPUTE.json"}
    tmp = tp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(T, indent=1, allow_nan=False) + "\n")
    tmp.replace(tp)
    return T["audit"]


def main(argv=None):
    """python -m cbp.audit timing [--out results/pcrl_confidence_budgeted_privacy_v1/AUDIT_COMPUTE.json]
                                  [--timing-json results/pcrl_confidence_budgeted_privacy_v1/TIMING.json]
    python -m cbp.audit estimate --out <AUDIT_COMPUTE.json>                         (synthetic only; no real data)"""
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("timing", "estimate"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--timing-json", default=None)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    if a.cmd == "estimate":
        print(json.dumps(reestimate(a.out), indent=1))
        return
    r = timing(out_path=a.out, workdir=a.workdir)
    if a.out and a.timing_json:
        write_timing_key(a.out, a.timing_json)
    print(json.dumps(r["estimates"], indent=1))


if __name__ == "__main__":
    main()
