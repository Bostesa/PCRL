"""Equal-strength attacker banks, composed source readers, coverage / MI receipts and real-data controls for the
learned-decoder constrained-release study (lcr; role D; prompt sections 10, 11, 12, 14). Not a training critic.

PROVENANCE. A COPY of cbp/audit.py at the cbp final commit 7f3ec67b2ecd86d474e2ff27167091af9923f572 (itself a copy of
qpc/audit.py at d0c8a45, whose attack machinery is dpc/audit.py at 0a7b05a5, IMPORTED UNCHANGED). cbp/, qpc/ and dpc/
are never edited and nothing in them is monkeypatched at runtime. The source FINAL slate and readers are unchanged:
smf.audit.final_slate(False) (LR x5, MLP x4, HGB x4, DA_LR, DA_MLP; pinned grids), the cell-conditional readers
CC_alpha{0.5, 1, 5} / CCpair_alpha{0.5, 1, 5} (prior-weighted Dirichlet), occurrence-ordered categorical one-hot token
columns, dual (AUC / CE) selection with first-in-bank-order ties (1e-12), fixed orientation P(S = 1) (never flipped
or clamped), the unseen-token prior and the INNER-selected unseen-pair fallback, full categorical identities + decoded
probabilities + decisions, pair tuples and both ignore-the-other-recipient banks, fitting on AUDIT_FIT and selection on
INNER_SELECTION, the AUC- and CE-selected attackers refit at attacker seeds 0, 1, 2, and dpc.audit.final_audit for the
locked assessment.

DOCUMENTED DIFF against cbp/audit.py (everything not listed is the cbp logic, unchanged):
  L1  docstrings and the study name (pcrl_learned_decoder_constrained_release_v1), SCHEMA "lcr-inner-v1".
  L2  binding: unit store, ids, unit names and scored lists through lcr.run (UNITS, parse_id, unit_for, inner_name,
      code_ids, scored_ids) and lcr.baselines; SEX label rows through lcr.data.labels_for (= qpc.data.labels_for).
      lcr inner audits live in the aud__<release unit> namespace (lcr.run.inner_name). The admitted cbp inner__*
      audits are custody only: every lcr reader of an inner unit REQUIRES schema "lcr-inner-v1" (load_inner), so a
      cbp record can never pass as an lcr record.
  L3  REGISTERED COMPOSITION BANK (prompt sec. 11, lead decision 2026-10-07): SRC|U composes, per seed, over the
      COMPLETE registered lcr code bank = exactly lcr.run.code_ids(), 83 codes in this order:
        D0 admitted (27)   U|DIRECT-TASK|i8o64, U|FINE-TASK|i8o64, U|CLASS|i1o1, then the 24 privacy maps in lambda
                           order (0.01, 0.025, 0.04, 0.06, 0.08, 0.1) x family order (LOCAL, SEQ-12, SEQ-21, JOINT);
        D1 fixed-map (26)  the same DIRECT-TASK, FINE-TASK and 24 privacy maps with suffix |D1 (same tokens as their
                           D0 map, different decoded q: a DISTINCT public map, kept as such);
        new fits (30)      U|C-TASK|i8o64|D1, U|W-{family}|i8o64|l{lam}|D1 (same lambda x family order),
                           U|K-{LOCAL,SEQ-12,SEQ-21,JOINT-SINGLE,JOINT-PAIR}|i8o64|D1.
      The registration lives HERE (REG_*) and must equal lcr.run.code_ids() and the policy part of scored_ids()
      exactly, or the bank is REFUSED. cbp's 11 qpc composition-only extras (COMPOSED_EXTRA) are NOT part of the lcr
      bank: they are not admitted to the lcr store and none of them was a cbp composed winner (cbp
      INDEPENDENT_VERIFICATION freeze_own lists, seeds 0-2); COMPOSED_EXTRA = () here. CLOSURE: every one of the 83
      release units (pol__ / dec__ / new__ s{k}__U_*) must be complete on disk with its complete lcr inner unit
      (aud__..., schema lcr-inner-v1); any completed release unit or lcr inner unit of that teacher and seed outside
      the 83 is a STRAY and is REFUSED, as is any missing one. RAW-J has no codes: its composed bank is its own bank.
  L4  CODE VIEW FAMILIES (prompt sec. 11). Every code release (D0, D1 fixed-map, new fit; identical release.npz keys)
      is audited with THREE families, each with AUC/CE selection and seed refits (full FINAL slate + cell readers,
      except the token-only family: cell readers only, L4b):
        code   PRIMARY = the complete interface [one-hot token (full alphabet), decoded q_i, one-hot decision_i]; pair =
               [v1, v2] with the exact token tuple and both ignore-recipient banks (cbp's single family, unchanged).
               Nomination, composition and every primary result use ONLY this family.
        token  DIAGNOSTIC token-only: the exact token identities tok_i and the exact tuple, audited with the cell
               readers only (L4b; the view also carries the one-hot token over the full alphabet in the same
               occurrence-ordered columns, which fixes its fingerprint).
        prob   DIAGNOSTIC probability-only: decoded q_i alone; finite by VALUE (cell readers keyed on the exact
               float64 value of q_i / the tuple of values), so two tokens that share a decoded vector are ONE
               probability cell here -- by design of this diagnostic, never in the complete family, whose pair
               identities are the exact token tuples and are never collapsed because probabilities coincide.
      Diagnostic families are reported (record["families"]) and never enter nomination or composition.
  L4b TOKEN-ONLY READERS (lead decision 2026-10-07, registered before SCIENCE_LOCK): the token-only diagnostic family
      is audited with the exact token-identity cell readers only -- CC_alpha{0.5, 1, 5} on tok_i, CCpair_alpha{0.5,
      1, 5} on the exact tuple (INNER-selected unseen-tuple fallback), both ignore-the-other-recipient banks, the same
      dual AUC / CE selection (dpc.audit._select) -- in the inner audit (inner_family_cells) and in the assessment
      (final_audit_cells). Rationale: for a pure categorical input the per-cell readers are the direct finite-sample
      Bayes readers of the token; the FINAL slate on a one-hot of tokens is a regularised approximation of them; and
      the family is diagnostic only. The complete-interface family (primary, nominated and composed) and the
      probability-only family keep the full FINAL slate + cell readers, unchanged. (Cost: the full-slate token family
      measured 27.9 CPU-s per i8o64 unit, about as much as the complete family; the cells-only family costs < 1 s.)
  L5  REUSE (mathematically identical, verified; lead decision 2026-10-07). EXACT ALIAS CACHE: a code family whose
      view fingerprint (sha256 of the v1 / v2 / pair design matrices; it fixes the token or value partitions the
      cell readers key on) equals that of an already COMPLETE lcr inner unit of the same seed, with the same schema,
      slate, attacker seeds and AUDIT_FIT / INNER_SELECTION row hashes, is copied from it (record + stored
      predictions; the match is re-checked on the hash-complete record and its prediction sha256, the per-unit
      sidecar view_fingerprints.json is only an index). Search order: for a D1 fixed-map code its D0 map first (same
      tokens, so the token-only family always aliases), then every other code inner unit of the seed by name. No
      waiting: a unit not yet complete (e.g. on the other shard) means the family is computed. The path is recorded
      per family (family_paths, reused_families, token_family_path); results are bitwise identical to recomputing
      (tested). A D1 fixed-map release must carry exactly its D0 map's tokens (tok / hard / alpha / row_id, else
      REFUSED). The admitted cbp inner audits are NOT reused (lead decision 2026-10-07: their family set lacks the
      token-only and probability-only diagnostics, so the exact parity rule fails by construction); all 81 D0 code
      and 9 reference inner units are recomputed, and cbp_parity(k, cid) compares the recomputed lcr primary family
      with the admitted cbp record (records only).
  L6  inner records: the cbp fields (recovery.auc / .ce, utility, utility_gate, preserved, token_states, composed,
      coverage_receipt, mi_diagnostic, view_fingerprint, ...) with schema lcr-inner-v1, plus arm / decoder /
      base_family / privacy_trained (lcr.run.parse_id), view_fingerprints per family, family_cpu_s, family_paths /
      reused_families (L5), and (D1) the d0_token_parity receipt; code units also carry the view_fingerprints.json
      sidecar (L5 index). DROPPED: cbp_headroom (cbp's 0.006 / 0.0035 every-seed buffer is not part of the lcr
      selection, prompt sec. 10; the field is removed so no consumer can use it). EVERY float finite (else
      TECHNICAL FAILURE).
  L7  validate_inner / validate_composed / validate_all over the lcr bank, plus: the diagnostic families are
      present, finite and bound to the release (token / prob fingerprints rebuilt from release.npz), and the schema
      is lcr-inner-v1.
  L8  controls: the cbp control rules and the registered CONTROL_LIMITS unchanged; the plan (fixed by structure) adds
      one D1 fixed-map code (U|JOINT|i8o64|l0.1|D1, the D1 version of cbp's JOINT control code) and one new
      constrained code (U|K-JOINT-PAIR|i8o64|D1, the largest registered neighbourhood) to cbp's three codes.
  L9  lcr stage helpers (inner_jobs, source_jobs, inner_job, stage_inner, stage_inner_src), the synthetic D1 release
      fixture, synthetic timing over the lcr bank (LCR_BANK) and the atomic TIMING.json["audit"] merge.

REGISTERED CONTROL PASS LIMITS (the SOURCE limits, unchanged; registered here and in AUDIT_COMPUTE.json before any
fit; controls are an engineering check, a failure is never evidence of removal):
  NULL      every view: held-out AUC_B <= 0.5 + 3.5 sd0 (sd0 = Mann-Whitney null sd on half B; the source's realised
            threshold on these roles was 0.5654143765984265, sd0 0.018689821885264708; a different realised value is
            reported as a technical failure because the split and permutation are fixed by seed).
  CONF_r{i} AUC_B(v_i) > 0.75 (serialisation exact).        COLL_r{i}  AUC_B(v_i) > 0.75 (serialisation exact).
  XOR       AUC_B(pair) > 0.75 AND AUC_B(v1) <= null threshold AND AUC_B(v2) <= null threshold.
  ROT_r{i}  AUC_B(v_i) > 0.75 AND AUC_B(pair) > 0.75 (serialisation exact).
  NULL_CALIBRATION  5 permutations; pass iff no exceedance of the null threshold.
Positive-control plants are built in memory / a temporary directory only; they are never saved as units, so they can
never enter a candidate or composed bank (the closure check would refuse them as strays).

ROLES. Every attacker is FITTED on AUDIT_FIT (6,065 rows) and SELECTED on INNER_SELECTION (2,235 rows). Every inner
entry point runs osf.audit.guard_inner (attacker roles carry exactly those roles, no overlap with the assessment, no
sealed SEX labels). SEX is read on AUDIT_FIT / INNER_SELECTION (attacks), on OSF_DEFENSE_FIT only for the MI
diagnostic (lcr.data procedure "permutation_diagnostic"), and on assessment rows only inside lcr.assess. Token
occupancy of assessment rows (a label-free function of the release) is counted for the coverage receipt.

VIEWS (the release interface is audited whole; integer token IDs are never an ordinal feature).
  code releases (pol__ D0 / dec__ D1 / new__ s{k}__<cid>/release.npz with EXACTLY row_id, tok1, q1, hard1, alpha1, tok2,
  q2, hard2, alpha2; any other key -- continuous probabilities, fine IDs, residuals -- is REFUSED, so protected-code
  attackers can never read the original continuous probabilities): families code / token / prob (L4).
  source (tea__s{k}__<teacher>/teacher.npz): interface [c_i, p_i] (PRIMARY), complete [r_i, c_i], scores c_i, probs p_i,
  decisions one-hot(d_i) (finite).  reference (ref__s{k}__<label>/reference.npz): lcr.baselines.reference_view_sets.

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

COMPOSED SOURCE BANK (section 11). U continuous can compute every public map AND every public decoder: recipient i's
map reads only p_i, which every source family except decisions determines exactly, and the decoder is a public
function of the token. So an attacker holding the source output can apply ANY registered code of the same teacher and
seed (its complete code interface). `composed_source_bank` forms, per source family and view, the union of the
source's own bank and the PRIMARY (complete code) reader banks of EVERY code of that teacher and seed in the
registered bank (L3: the 83 lcr codes in lcr.run.code_ids() order), reusing each code's stored inner records (no
refit; the composed attacker A(g(p)) equals the code attacker A on the same rows and AUDIT_FIT fit). The decisions
family composes only with the class-only code (its tokens are functions of d_i). Composed selection uses the same rule
at the bank level: per view the winning bank is the first (own, then codes in registered order) whose seed-0 selected
value beats the running best by more than 1e-12; the winner is that bank's own selected attacker, and the composed
recovery is its seed 0-2 mean. This happens BEFORE strongest-control and nominee selection (lcr.select reads the
composed source record). CLOSURE: L3 -- otherwise REFUSED; all records must also share the INNER_SELECTION / AUDIT_FIT
row hashes, slate and attacker seeds. The source inner record stores the composed winners (`composed.winner`,
`composed.ce_winner`, per family) and `composed.freeze` = every code whose reader won any (family, view, criterion);
the evaluation lock must list those codes as composed readers of the final source audit. Protected-code attackers
never receive source outputs.

COVERAGE AND MI (diagnostics only). `coverage_receipt`: per recipient and for the pair, the declared alphabet, the
cells actually occupied on OSF_DEFENSE_FIT / AUDIT_FIT / INNER_SELECTION / assessment rows, the AUDIT_FIT count
histogram, singleton-cell fractions (of occupied cells and of rows), and the rows falling back (unseen in AUDIT_FIT) on
fit (0 by construction), selection and assessment rows. `mi_diagnostic`: plug-in MI (nats) of SEX and tok1, tok2,
(tok1, tok2) on OSF_DEFENSE_FIT, and a permutation-null MI at the same alphabet from a FIXED list of MI_PERMS
permutations of DEFENSE_FIT SEX (seed MI_PERM_SEED; the list hash is recorded). It never alters objectives, lambdas or
nominees, and an MI decrease is never read as protection. D0 and D1 of one map have identical token MI by construction.

REAL-DATA CONTROLS (`stage_controls`; the LEAD runs them under SCIENCE_LOCK; sealed D; inner roles only; pass limits =
REGISTERED CONTROL PASS LIMITS above). INNER_SELECTION is split by exact-record group into halves A / B
(smf.audit.null_split); SEX is permuted within AUDIT_FIT, within A and within B (smf.audit.frozen_permutation, drawn and
hashed before any fit) to S*. Each check fits on AUDIT_FIT, SELECTS on A over the whole bank and EVALUATES that attacker
on held-out B. Plants encode noisy S* (20% of rows replaced by a random class; a perfect reader has AUC 0.9). Pass rules
(thresholds smf.audit's, unchanged: NULL_Z 3.5, PLANT_MIN 0.75; thr = 0.5 + 3.5 sd0 on B):
  (a) NULL       (shuffled SEX) real release vs S*: every view AUC_B <= thr.
  (b) CONF_r{i}  (confidence signal within predicted class) token t -> (t, b), b = noisy S*, the b = 1 copy decodes to
                 (1 - 0.05) q_t + 0.05 e_d (same decision): AUC_B(v_i) > 0.75. The decisions-only audit of the planted
                 release is reported (expected <= thr).
  (c) COLL_r{i}  (distinct-token / same-probability leak) the same split with IDENTICAL decoded vectors (two IDs, one
                 decoded vector, different information): AUC_B(v_i) > 0.75. The decoded-probability-only audit is
                 reported (it cannot see the plant).
  (d) XOR        (coalition XOR) recipient 1 split by b1 ~ Bernoulli(1/2), recipient 2 by b2 = b1 XOR noisy S*,
                 collision-style: AUC_B(pair) > 0.75 AND AUC_B(v1) <= thr AND AUC_B(v2) <= thr.
  (e) ROT_r{i}   (source rotated signal) continuous views (U, RAW-J, E interface): v_i -> [v_i, 1e-6 (noisy S*
                 - 0.5)] Q, Q a seeded Haar-random orthogonal matrix, written with np.savez_compressed and re-read:
                 arrays bit-exact AND AUC_B(v_i) > 0.75 AND AUC_B(pair) > 0.75.
  NULL_CALIBRATION  repeated permutations (seed CONTROL_SEED + 100 + rep) on one release; every rep kept; pass iff no
                 exceedance.
A failed positive control means TECHNICAL REVIEW of the audit (amendment), never "low leakage". Failures are recorded,
never dropped. The plan (control_plan) is fixed by structure before any control runs.

UNIT CONTRACT (lcr.run stage_inner / stage_inner_src; or the helpers here): `inner_unit(kind, k, cid, D)` returns the
JSON-safe record
  recovery {auc, ce, auc_seed0, ce_seed0, auc_per_seed, ce_per_seed, selected, ce_selected, tables, coverage, ...}
  (primary family; for SRC|<teacher> already the COMPOSED bank), families {secondary / diagnostic families, same
  structure}, composed {auc, ce, winner, ce_winner, freeze, per_family, policies, closure} (sources), utility
  (qpc.utility.release_inner_utility: {"income", "occupation"} each with acc, logloss, brier, const_acc, ...),
  utility_gate (gate_record vs U of the same seed), preserved {"1", "2"}, decision_preservation, token_states
  (alpha1 + alpha2; None = JSON null for continuous releases, never Infinity), token_states_detail, coverage_receipt
  and mi_diagnostic (codes), arm / decoder / base_family / privacy_trained, view_fingerprint(s), provenance and
  "_files" = {"inner_preds.npz": {sel_row_id, keys_<fam>, P_<fam> (n_candidates, n_INNER) seed-0 P(S=1),
  SEL_<fam>_<crit>_<view> (3, n_INNER) selected attacker per seed}}. Saved as lcr.run.inner_name(k, cid) = aud__<unit>.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from sklearn.metrics import roc_auc_score

from lcr import baselines as BL                  # L2
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
SCHEMA = "lcr-inner-v1"                          # L1
STUDY = "pcrl_learned_decoder_constrained_release_v1"
CODE_FAMILIES = ("code", "token", "prob")        # L4: primary first
PRIMARY_CODE_FAMILY = "code"
DIAGNOSTIC_FAMILIES = ("token", "prob")

# L3: the registered composition bank (prompt sec. 11; lead decision 2026-10-07). Must equal lcr.run exactly.
REG_RATE = (8, 64)
REG_LAMS = (0.01, 0.025, 0.04, 0.06, 0.08, 0.1)
REG_PRIVACY = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
REG_CONSTRAINED = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT-SINGLE", "JOINT-PAIR")
REG_TASK = ("U|DIRECT-TASK|i8o64", "U|FINE-TASK|i8o64", "U|CLASS|i1o1")
COMPOSED_EXTRA = ()                              # cbp's 11 qpc extras are not part of the lcr bank (L3)
N_D0, N_D1, N_NEW = 27, 26, 30
N_CODES = N_D0 + N_D1 + N_NEW                    # 83

# Registered control pass limits (the source's; see the module docstring). Written before any fit.
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
              "PLANT_MIN), unchanged (cbp CONTROL_LIMITS)"}


# ------------------------------------------------------------------ ids, units
def _R():
    from lcr import run as R                     # L2
    return R


def _g(x):
    return f"{x:g}"


def registered_d0():
    return list(REG_TASK) + [f"U|{f}|i{REG_RATE[0]}o{REG_RATE[1]}|l{_g(lam)}" for lam in REG_LAMS for f in REG_PRIVACY]


def registered_d1():
    return [c + "|D1" for c in registered_d0() if c != "U|CLASS|i1o1"]


def registered_new():
    r = f"i{REG_RATE[0]}o{REG_RATE[1]}"
    return ([f"U|C-TASK|{r}|D1"] + [f"U|W-{f}|{r}|l{_g(lam)}|D1" for lam in REG_LAMS for f in REG_PRIVACY] +
            [f"U|K-{a}|{r}|D1" for a in REG_CONSTRAINED])


def registered_code_bank():
    """The 83 registered lcr codes (L3), in lcr.run.code_ids() order: D0 (27), D1 fixed-map (26), new fits (30)."""
    return registered_d0() + registered_d1() + registered_new()


def registered_composition_bank():
    """The public code maps SRC|U composes with (L3): exactly the 83 registered codes (no composition-only extras)."""
    return registered_code_bank() + list(COMPOSED_EXTRA)


def registration_check(R=None):
    """{ok, ...}: lcr.run's code_ids / scored_ids equal this registration exactly."""
    R = R or _R()
    bad = []
    if list(R.code_ids()) != registered_code_bank():
        bad.append("code_ids != the 83 registered codes (order included)")
    for nm, want in (("d0_ids", registered_d0()), ("d1_fixed_ids", registered_d1()), ("new_fit_ids", registered_new())):
        if hasattr(R, nm) and list(getattr(R, nm)()) != want:
            bad.append(f"{nm} != the registered {nm}")
    scored_codes = [c for c in R.scored_ids() if R.parse_id(c)["kind"] == "policy"]
    if scored_codes != registered_code_bank():
        bad.append("scored_ids (policy part) != the 83 registered codes")
    if len(registered_code_bank()) != N_CODES or len(set(registered_code_bank())) != N_CODES:
        bad.append("registered bank size is not 83 distinct codes")
    if (len(registered_d0()), len(registered_d1()), len(registered_new())) != (N_D0, N_D1, N_NEW):
        bad.append("registered bank parts are not 27 + 26 + 30")
    units = {R.unit_for(0, c) for c in registered_code_bank()}
    if len(units) != N_CODES:
        bad.append("two registered codes share a release unit name")
    return {"ok": not bad, "mismatches": bad, "n_codes": N_CODES, "parts": {"d0": N_D0, "d1_fixed": N_D1,
                                                                            "new": N_NEW}}


def parse_cid(cid):
    return _R().parse_id(cid)


def unit_of(k, cid):
    return _R().unit_for(k, cid)


def inner_of(k, cid):
    """lcr inner-audit unit name of a release (lcr.run.inner_name: aud__<release unit>)."""
    return _R().inner_name(k, cid)


def _units(units_dir=None):
    return BL.units_dir(units_dir)


def _complete(name, units_dir=None):
    from jcv.finalize import unit_complete
    return unit_complete(_units(units_dir) / name)


def is_class_only(cid):
    c = parse_cid(cid)
    return c["kind"] == "policy" and c["family"] == "CLASS"


def d0_of(cid):
    """The D0 map of a D1 fixed-map code (same tokens), else None."""
    c = parse_cid(cid)
    if c["kind"] != "policy" or c["arm"] != "d1_fixed":
        return None
    assert cid.endswith("|D1")
    return cid[:-len("|D1")]


# ------------------------------------------------------------------ views
def check_release_keys(z):
    keys = set(z.files if hasattr(z, "files") else z)
    extra, missing = sorted(keys - RELEASE_KEYS), sorted(RELEASE_KEYS - keys)
    if extra or missing:
        raise ValueError(f"REFUSED: release keys must be exactly {sorted(RELEASE_KEYS)}; extra {extra}, missing "
                         f"{missing} (no continuous probabilities, fine IDs or residuals may accompany a code)")


def policy_views(z, D, meta=None):
    """PRIMARY complete code interface (dpc.audit.policy_views, unchanged) after the release-key contract."""
    check_release_keys(z)
    return DA.policy_views(z, D, meta=meta)


def _checked(z, D):
    check_release_keys(z)
    zz = DA.align(z, D)
    chk = {}
    for i in (1, 2):
        c = DA.token_decoder_check(zz[f"tok{i}"].astype(np.int64), zz[f"q{i}"], zz[f"hard{i}"])
        if not c["ok"]:
            raise ValueError(f"REFUSED: recipient {i} interface invalid: {c['problems']}")
        chk[f"v{i}"] = {**c, "alphabet": int(np.asarray(zz[f"alpha{i}"]).ravel()[0])}
    return zz, chk


def token_views(z, D, meta=None):
    """L4 DIAGNOSTIC token-only family: one-hot token over the FULL alphabet (dpc.audit canonical occurrence-ordered
    columns, as in the complete view), cell readers on tok_i and the exact tuple."""
    zz, chk = _checked(z, D)
    X, T = {}, {}
    for i in (1, 2):
        tok = zz[f"tok{i}"].astype(np.int64)
        a = chk[f"v{i}"]["alphabet"]
        col = DA.canonical_columns(tok, D, a)
        X[f"v{i}"], T[f"v{i}"] = DA.onehot(col[tok], a), tok
    X["pair"] = np.hstack([X["v1"], X["v2"]])
    return {"family": "token", "X": X, "tokens": T, "meta": {**(meta or {}), "interface": chk, "diagnostic": True,
            "view": "token-only: [one-hot token (full alphabet, occurrence-ordered columns)]"}}


def prob_views(z, D, meta=None):
    """L4 DIAGNOSTIC probability-only family: decoded q_i alone, finite by exact float64 VALUE (tokens sharing a decoded
    vector are one cell here, by design of this diagnostic)."""
    zz, chk = _checked(z, D)
    X = {f"v{i}": np.asarray(zz[f"q{i}"], dtype=np.float64) for i in (1, 2)}
    X["pair"] = np.hstack([X["v1"], X["v2"]])
    V = {"family": "prob", "X": X, "tokens": None, "meta": {**(meta or {}), "interface": chk, "diagnostic": True,
         "view": "probability-only: [decoded q_i]; cell readers keyed on the exact value of q_i"}}
    V = DA.finite_by_value(V)
    V["family"] = "prob"
    return V


def code_view_sets(z, D, meta=None):
    """{family: views} of one code release: code (PRIMARY, complete), token, prob (diagnostics)."""
    return {"code": policy_views(z, D, meta=meta), "token": token_views(z, D, meta=meta),
            "prob": prob_views(z, D, meta=meta)}


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
    """Validated complete code views whose design matrices are rebuilt on every access (same values as policy_views)."""
    zz, chk = _checked(z, D)
    T = {f"v{i}": zz[f"tok{i}"].astype(np.int64) for i in (1, 2)}
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
                "family_method": c["family"], "arm": c["arm"], "decoder": c["decoder"], "m1": c["m1"], "m2": c["m2"],
                "lam": c["lam"], "m": 1 if c["family"] == "CLASS" else None}
        sets = code_view_sets(z, D, meta=meta)
        out = {"p1": z["q1"], "p2": z["q2"], "hard1": z["hard1"], "hard2": z["hard2"]}
        return "code", sets, out, {"unit": unit, "complete_sha256": sha}, z
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


# L4b (lead decision 2026-10-07, pre-lock): the token-only DIAGNOSTIC family is audited with the exact
# token-identity cell readers only -- CC_alpha{0.5, 1, 5} on tok_i and CCpair_alpha{0.5, 1, 5} on the exact tuple
# (INNER-selected unseen-tuple fallback), with both ignore-the-other-recipient banks in the pair table and the same
# dual AUC / CE selection (dpc.audit._select, first-in-bank-order ties). For a pure categorical input the per-cell
# readers are the direct finite-sample Bayes readers of the token; the slate on a one-hot of tokens is a regularised
# approximation of them; and the family is diagnostic only. The complete-interface family (primary, nominated and
# composed) and the probability-only family keep the full FINAL slate, unchanged.
TOKEN_FAMILY_READERS = "cells_only"
CELL_READERS = [f"CC_alpha{a}" for a in CC_ALPHAS] + [f"CCpair_alpha{a}" for a in CC_ALPHAS]


def _cell_bank(views, yy, fit_idx, pred_idx, sel_pos):
    """{pred_key: P(S=1) on pred_idx} of the cell readers only (same keys and order as dpc.audit._source_bank)."""
    toks = views.get("tokens")
    if not toks:
        raise ValueError("REFUSED: a cells-only family needs declared token identities")
    preds, cov = {}, {}
    for w in ("v1", "v2"):
        Pc, c = DA.cc_local(toks[w], yy, fit_idx, pred_idx)
        preds.update({f"{w}:{nm}": p for nm, p in Pc.items()})
        cov[w] = {"tokens_in_fit": c["tokens_in_fit"], "_seen": c["seen"]}
    Pc, c = DA.cc_pair(toks["v1"], toks["v2"], yy, fit_idx, pred_idx, sel_pos)
    preds.update({f"pair:{nm}": p for nm, p in Pc.items()})
    cov["pair"] = {"pairs_in_fit": c["pairs_in_fit"], "fallback_rule": c["fallback_rule"],
                   "fallback_ce_on_unseen_inner_rows": c["fallback_ce_on_unseen_inner_rows"], "_seen": c["seen"]}
    return preds, cov


def inner_family_cells(views, D, slate="final", seeds=ATT_SEEDS):
    """L4b inner audit of a token-only view set with the cell readers only; same record / array contract as
    inner_family (the readers are deterministic counts, so every attacker-seed refit equals seed 0)."""
    t0 = time.time()
    yy = np.asarray(D["sex"])
    guard_inner(D, yy)
    fit_idx, sel_idx = DA.roles(D)
    sel_pos = np.arange(len(sel_idx))
    preds, cov = _cell_bank(views, yy, fit_idx, sel_idx, sel_pos)
    ys = yy[sel_idx]
    sel, banks = DA._select(preds, [""], ys, sel_pos)
    rec = DA._summarise(sel, banks)
    keys = list(preds)
    rec.update({
        "kind": "inner", "family": views.get("family"), "meta": views.get("meta", {}), "finite": True,
        "slate": slate, "slate_members": [], "readers": TOKEN_FAMILY_READERS, "cell_readers": CELL_READERS,
        "cc_alphas": list(CC_ALPHAS), "fallback_rules": list(FALLBACK_RULES),
        "roles": {"fit": FIT_ROLE, "select_and_score": SEL_ROLE},
        "orientation": "P(S=1), fixed; never flipped or clamped",
        "coverage": DA._coverage_summary(cov, sel_pos), "composed_coverage": {}, "composed_units": [],
        "n_fit": int(len(fit_idx)), "n_select": int(len(sel_idx)), **_row_hashes(D), "fit_seconds": {},
        "inner_predictions": {"keys": keys, "P": np.stack([preds[k] for k in keys])},
        "sel_row_id": np.asarray(D["row_id"])[sel_idx], "wall_s": round(time.time() - t0, 2)})
    arrays, per = {}, {"auc": {}, "ce": {}}
    for w in PRIMARY_VIEWS:
        for crit in ("auc", "ce"):
            A = np.stack([preds[rec["selection"][w][crit]["pred_key"]]] * len(seeds))
            arrays[f"{crit}_{w}"] = A
            per[crit][w] = {"auc_per_seed": [auc1(ys, p) for p in A], "ce_per_seed": [ce1(ys, p) for p in A]}
    out = dict(rec)
    out["auc_seed0"], out["ce_seed0"] = dict(rec["auc"]), dict(rec["ce"])
    out["auc"] = {w: float(np.mean(per["auc"][w]["auc_per_seed"])) for w in PRIMARY_VIEWS}
    out["ce"] = {w: float(np.mean(per["ce"][w]["ce_per_seed"])) for w in PRIMARY_VIEWS}
    out["auc_per_seed"] = {w: per["auc"][w]["auc_per_seed"] for w in PRIMARY_VIEWS}
    out["ce_per_seed"] = {w: per["ce"][w]["ce_per_seed"] for w in PRIMARY_VIEWS}
    out["ce_selected_auc_per_seed"] = {w: per["ce"][w]["auc_per_seed"] for w in PRIMARY_VIEWS}
    out["attacker_seeds"] = list(seeds)
    out["refits"] = 0
    _derived(out)
    out["statistic"] = ("L4b token-only diagnostic: cell readers only (CC / CCpair on the exact token identities); "
                        "auc = the selected reader's INNER_SELECTION AUC (deterministic, identical at every attacker "
                        "seed); selection-optimistic; not a null test")
    return out, arrays


L4B_RULE = {"id": "L4b", "decided": "lead, 2026-10-07, registered before SCIENCE_LOCK",
            "rule": "the token-only DIAGNOSTIC family is audited with the exact token-identity cell readers only: "
                    "CC_alpha{0.5, 1, 5} on tok_i and CCpair_alpha{0.5, 1, 5} on the exact tuple (INNER-selected "
                    "unseen-tuple fallback), both ignore-the-other-recipient banks, the same dual AUC / CE selection "
                    "(dpc.audit._select); inner: lcr.audit.inner_family_cells; assessment: lcr.audit.final_audit_cells",
            "rationale": "for a pure categorical input the per-cell readers are the direct finite-sample Bayes readers "
                         "of the token; the FINAL slate on a one-hot of tokens is a regularised approximation of "
                         "them; the family is diagnostic only",
            "unchanged": "the complete-interface family (primary, nominated and composed) and the probability-only "
                         "family keep the full FINAL slate + cell readers",
            "cost_before": "full-slate token family 27.85 CPU-s per synthetic i8o64 unit (first timing run)"}


def final_audit_cells(views, D, score_rows, seeds=ATT_SEEDS, y=None):
    """L4b final audit of a token-only view set (lcr.assess): cell readers fitted on AUDIT_FIT, dual-selected on
    INNER_SELECTION, scored on score_rows (disjoint, unsealed); the dpc.audit.final_audit record / probs contract."""
    t0 = time.time()
    yy = np.asarray(D["sex"] if y is None else y)
    guard_inner(D, yy)
    fit_idx, sel_idx = DA.roles(D)
    score_rows = DA._check_score_rows(D, score_rows, yy)
    pred_idx = np.concatenate([sel_idx, score_rows])
    sel_pos, score_pos = np.arange(len(sel_idx)), np.arange(len(sel_idx), len(pred_idx))
    preds, cov = _cell_bank(views, yy, fit_idx, pred_idx, sel_pos)
    ys = yy[sel_idx]
    sel, banks = DA._select(preds, [""], ys, sel_pos)
    probs, scored = {}, {}
    y_sc = yy[score_rows]
    for w in PRIMARY_VIEWS:
        scored[w] = {}
        for crit in ("auc", "ce"):
            s = sel[w][crit]
            p1 = preds[s["pred_key"]][score_pos]
            P = np.stack([np.stack([1.0 - p1, p1], 1)] * len(seeds))
            probs[f"{crit}_{w}"] = P
            scored[w][crit] = {"label": s["label"], "candidate": s["candidate"], "source_view": s["view"],
                               "attacker": s["attacker"], "inner_auc": s["inner_auc"], "inner_ce": s["inner_ce"],
                               "seeds": list(seeds), "auc_per_seed": [auc1(y_sc, p[:, 1]) for p in P],
                               "ce_per_seed": [ce1(y_sc, p[:, 1]) for p in P]}
            scored[w][crit]["auc_mean"] = float(np.mean(scored[w][crit]["auc_per_seed"]))
            scored[w][crit]["ce_mean"] = float(np.mean(scored[w][crit]["ce_per_seed"]))
    rec = DA._summarise(sel, banks)
    rec.update({"kind": "final", "family": views.get("family"), "meta": views.get("meta", {}), "finite": True,
                "slate": None, "slate_members": [], "readers": TOKEN_FAMILY_READERS, "cell_readers": CELL_READERS,
                "cc_alphas": list(CC_ALPHAS), "composed_units": [],
                "coverage": {"own": DA._coverage_summary(cov, sel_pos, score_pos)}, "scored": scored,
                "n_fit": int(len(fit_idx)), "n_select": int(len(sel_idx)), "n_score": int(len(score_rows)),
                "orientation": "P(S=1) column, fixed; never flipped or clamped on scored rows",
                "wall_s": round(time.time() - t0, 2)})
    return rec, probs


def family_auditor(kind, fam):
    """L4b: the inner auditor of one family (cells-only for the token-only code diagnostic, else the full slate)."""
    return inner_family_cells if kind == "policy" and fam == "token" else inner_family


def _derived(r):
    r["worse_local"] = max(r["auc"]["v1"], r["auc"]["v2"])
    r["mean_local"] = (r["auc"]["v1"] + r["auc"]["v2"]) / 2
    r["coalition_minus_best_local"] = r["auc"]["pair"] - r["worse_local"]


def _pub(rec):
    out = public_record(rec)
    return DA._jsonable(out)


# ------------------------------------------------------------------ composed source bank
def expected_policies(teacher, protocol=None):
    """L3: every public code map of `teacher` in the REGISTERED composition order (U: the 83 lcr codes; RAW-J: none).
    REFUSED unless lcr.run agrees exactly with the registration."""
    chk = registration_check()
    if not chk["ok"]:
        raise SystemExit(f"REFUSED: lcr.run disagrees with the registered composition bank: {chk['mismatches']}")
    return [c for c in registered_composition_bank() if parse_cid(c)["teacher"] == teacher]


def load_inner(name, units_dir=None):
    """(record, private arrays) of a completed lcr inner unit (hash-complete AND schema lcr-inner-v1; L2). `name` is an
    aud__ unit name or a release unit name (mapped to aud__<unit>)."""
    nm = name if name.startswith("aud__") else f"aud__{name}"
    d = _units(units_dir) / nm
    if not _complete(nm, units_dir):
        raise SystemExit(f"REFUSED: {nm} is missing or not hash-complete")
    rec = json.loads((d / "record.json").read_text())
    if rec.get("schema") != SCHEMA:
        raise SystemExit(f"REFUSED: {nm} has schema {rec.get('schema')!r}, not {SCHEMA!r} (a cbp record is never an "
                         "lcr inner record)")
    arr = {}
    if (d / "inner_preds.npz").exists():
        z = np.load(d / "inner_preds.npz", allow_pickle=False)
        arr = {k: z[k] for k in z.files}
    return rec, arr


def _lcr_inner_ok(name, units_dir=None):
    if not _complete(name, units_dir):
        return False
    try:
        return json.loads((_units(units_dir) / name / "record.json").read_text()).get("schema") == SCHEMA
    except (OSError, ValueError):
        return False


CODE_PREFIXES = ("pol", "dec", "new")


def _closure(k, teacher, expected, units_dir=None):
    root = _units(units_dir)

    def good(p):
        return p.is_dir() and not p.name.endswith((".tmp", ".quarantined")) and _complete(p.name, units_dir)
    on_disk = sorted(p.name for pre in CODE_PREFIXES for p in root.glob(f"{pre}__s{k}__{teacher}_*") if good(p))
    inner_disk = sorted(p.name for pre in CODE_PREFIXES for p in root.glob(f"aud__{pre}__s{k}__{teacher}_*")
                        if good(p))
    exp_units = {unit_of(k, c) for c in expected}
    exp_inner = {inner_of(k, c) for c in expected}
    unexpected = [u for u in on_disk if u not in exp_units]
    unexpected_inner = [u for u in inner_disk if u not in exp_inner]
    missing_release = [c for c in expected if unit_of(k, c) not in on_disk]          # L3: exact on-disk bank
    missing_inner = [c for c in expected if not _lcr_inner_ok(inner_of(k, c), units_dir)]
    return {"expected": len(expected), "release_units_on_disk": len(on_disk), "inner_units_on_disk": len(inner_disk),
            "unexpected_release_units": unexpected, "unexpected_inner_units": unexpected_inner,
            "missing_release_units": missing_release, "missing_inner_units": missing_inner,
            "ok": not unexpected and not unexpected_inner and not missing_release and not missing_inner and
            len(on_disk) == len(expected)}


def composed_source_bank(k, teacher, own, D, policy_cids=None, units_dir=None, protocol=None):
    """Composed bank of source SRC|<teacher> at seed k. own = {family: inner_family record} of the source.
    policy_cids default: expected_policies(teacher). Returns {family: composed record} plus the summary under "_all".
    REFUSES on any closure failure (missing / stray code release or lcr inner unit, row / slate / seed mismatch)."""
    registered = policy_cids is None
    expected = expected_policies(teacher, protocol) if registered else list(policy_cids)
    clo = _closure(k, teacher, expected, units_dir)
    if not clo["ok"]:
        raise SystemExit(f"REFUSED: composed source bank of {teacher} s{k} is not closed: {clo}")
    pols = []
    for c in expected:
        r, _ = load_inner(inner_of(k, c), units_dir)
        rr = r["recovery"]
        if r.get("primary_family") != PRIMARY_CODE_FAMILY or rr.get("family") != "code":
            raise SystemExit(f"REFUSED: {c} inner record's primary family is not the complete code interface")
        for key in ("sel_row_id_sha256", "fit_row_id_sha256", "slate", "attacker_seeds"):
            ref = own["interface"].get(key) if key in own["interface"] else None
            if rr.get(key) != ref:
                raise SystemExit(f"REFUSED: {c} inner record differs from the source on {key}")
        if r.get("seed") is not None and int(r["seed"]) != int(k):
            raise SystemExit(f"REFUSED: {c} inner record is for seed {r.get('seed')}, not {k}")
        if r.get("cid") not in (None, c):
            raise SystemExit(f"REFUSED: {inner_of(k, c)} holds the record of {r.get('cid')}, not {c}")
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
                   "policy_list_source": "registered (L3: the 83 lcr codes = lcr.run.code_ids())"
                   if registered else "EXPLICIT list (not the registered bank; tests / replay only)",
                   "composition_only": [c for c in expected if c in COMPOSED_EXTRA],
                   "rule": "per view: first bank (own, then codes in registered order) whose seed-0 selected value "
                           "beats the running best by > 1e-12; reported value = that bank's seed 0-2 mean; decisions "
                           "family composes with the class-only code only; codes enter with their PRIMARY (complete "
                           "code interface) family",
                   "reuse": "code candidates enter with their stored lcr inner records (same AUDIT_FIT fit, same "
                            "INNER_SELECTION rows); no refit"}
    return out


def composed_freeze_list(k, teacher, units_dir=None):
    """Codes whose composed readers won any (family, view, criterion) of SRC|<teacher> at seed k: they must join the
    final source audit as composed readers (frozen in the evaluation lock)."""
    r, _ = load_inner(inner_of(k, f"SRC|{teacher}"), units_dir)
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
    """Rows of `role` whose SEX `procedure` may read (lcr.data.labels_for = qpc.data.labels_for; L2) and the labels."""
    try:
        from lcr import data as LD
        rows = np.asarray(LD.labels_for(D, procedure, role))
    except ModuleNotFoundError as e:
        if e.name != "lcr.data":
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
D1_TOKEN_KEYS = ("row_id", "tok1", "tok2", "hard1", "hard2", "alpha1", "alpha2")


def d0_token_parity(k, cid, z, units_dir=None):
    """L5: a D1 fixed-map release must carry EXACTLY its D0 map's tokens, decisions and alphabets (only q may differ).
    Returns the receipt; REFUSES (SystemExit) otherwise."""
    d0 = d0_of(cid)
    z0, sha0 = BL.load_unit_npz(unit_of(k, d0), "release.npz", units_dir)
    eq = {x: bool(np.array_equal(np.asarray(z[x]), np.asarray(z0[x]))) for x in D1_TOKEN_KEYS}
    q_equal = {f"q{i}": bool(np.array_equal(np.asarray(z[f"q{i}"]), np.asarray(z0[f"q{i}"]))) for i in (1, 2)}
    if not all(eq.values()):
        raise SystemExit(f"REFUSED: D1 fixed-map release {cid} s{k} does not keep its D0 map's tokens exactly: "
                         f"{[x for x, ok in eq.items() if not ok]} (a calibration-only control may change only q)")
    return {"d0_config": d0, "d0_unit": unit_of(k, d0), "d0_complete_sha256": sha0, "equal": eq,
            "q_identical_to_d0": q_equal, "ok": True,
            "rule": "tok / hard / alpha / row_id equal to the D0 map (exact); q may differ"}


SIDECAR = "view_fingerprints.json"                # L5: tiny per-unit index of the audited view fingerprints


def _row_hashes(D):
    fit_idx, sel_idx = DA.roles(D)
    return {"sel_row_id_sha256": sha_arrays(np.asarray(D["row_id"])[sel_idx]),
            "fit_row_id_sha256": sha_arrays(np.asarray(D["row_id"])[fit_idx])}


def _sidecar(k, cid, fps, slate, seeds, D):
    return {"schema": SCHEMA, "kind": "policy", "cid": cid, "seed": int(k), "slate": slate,
            "attacker_seeds": list(seeds), **_row_hashes(D), "view_fingerprints": dict(fps)}


def _alias_candidates(k, cid, units_dir=None):
    """lcr code inner units of seed k searched for an exact alias, in a deterministic order: for a D1 fixed-map code
    its D0 map first (same partition), then every other code inner unit of the seed by name."""
    root = _units(units_dir)
    me = inner_of(k, cid)
    first = [inner_of(k, d0_of(cid))] if d0_of(cid) else []
    rest = sorted(p.name for pre in CODE_PREFIXES for p in root.glob(f"aud__{pre}__s{k}__*")
                  if p.is_dir() and not p.name.endswith((".tmp", ".quarantined")) and p.name != me and
                  p.name not in first)
    return first + rest


def _find_alias(k, cid, fam, fp, side, units_dir=None):
    """L5 exact alias cache: (public family record, its stored arrays, info) of the first completed lcr code inner unit
    of the same seed whose `fam` view fingerprint equals `fp` (the design matrices of v1, v2 and pair are then
    bit-identical, and so are the token / value partitions the cell readers key on) and whose slate, attacker seeds,
    schema and AUDIT_FIT / INNER_SELECTION row hashes equal this audit's; None otherwise (compute, never wait). The
    sidecar is only an index: the match is re-checked on the hash-complete record itself."""
    root = _units(units_dir)
    for name in _alias_candidates(k, cid, units_dir):
        sp = root / name / SIDECAR
        if not sp.exists():
            continue
        try:
            sc = json.loads(sp.read_text())
        except (OSError, ValueError):
            continue
        if (sc.get("view_fingerprints") or {}).get(fam) != fp or any(sc.get(x) != side[x] for x in (
                "schema", "seed", "slate", "attacker_seeds", "sel_row_id_sha256", "fit_row_id_sha256")):
            continue
        if not _lcr_inner_ok(name, units_dir):
            continue
        rec, arr = load_inner(name, units_dir)
        fr = rec["recovery"] if fam == rec.get("primary_family") else (rec.get("families") or {}).get(fam)
        if fr is None or (rec.get("view_fingerprints") or {}).get(fam) != fp or fr.get("slate") != side["slate"] or \
                fr.get("attacker_seeds") != side["attacker_seeds"] or \
                fr.get("sel_row_id_sha256") != side["sel_row_id_sha256"] or \
                fr.get("fit_row_id_sha256") != side["fit_row_id_sha256"]:
            continue
        keys = [x for x in arr if x in (f"keys_{fam}", f"P_{fam}") or x.startswith(f"SEL_{fam}_")]
        if f"keys_{fam}" not in keys or f"P_{fam}" not in keys or len(keys) != 2 + 2 * len(PRIMARY_VIEWS):
            continue
        if fr.get("inner_predictions_sha256") != sha_arrays(np.asarray(arr[f"P_{fam}"])):
            continue
        return json.loads(json.dumps(fr)), {x: arr[x] for x in keys}, {"from_unit": name, "from_config": rec.get("cid"),
                                                                       "view_fingerprint": fp}
    return None


def inner_unit(kind, k, cid, D, units_dir=None, slate="final", seeds=ATT_SEEDS, policy_cids=None, protocol=None,
               reuse=True):
    """ONE inner audit unit (saved by the runner as lcr.run.inner_name(k, cid) = aud__<unit>; "_files" holds the private
    predictions)."""
    t0, c0 = time.time(), time.process_time()
    guard_inner(D)
    pc = parse_cid(cid)
    primary, sets, out, prov, z = view_sets(kind, k, cid, D, units_dir)
    d1p = d0_token_parity(k, cid, z, units_dir) if kind == "policy" and pc["arm"] == "d1_fixed" else None
    fps = {fam: DA.view_fingerprint(V) for fam, V in sets.items()} if kind == "policy" else None
    recs, arrays, fam_pub, fam_cpu, reused, paths = {}, {}, {}, {}, {}, {}
    side = _sidecar(k, cid, fps, slate, seeds, D) if kind == "policy" else None
    for fam, V in sets.items():
        cf = time.process_time()
        hit = _find_alias(k, cid, fam, fps[fam], side, units_dir) if reuse and kind == "policy" else None
        if hit is not None:
            pub, arr, info = hit
            pub["meta"] = {**DA._jsonable(V.get("meta", {})), "reused_from": info["from_unit"]}
            fam_pub[fam] = pub
            arrays.update(arr)
            reused[fam] = info
            paths[fam] = f"reused from {info['from_unit']} (identical view fingerprint, slate, seeds and rows)"
        else:
            paths[fam] = ("computed" if kind != "policy" else "computed (reuse disabled)" if not reuse else
                          "computed (no completed lcr inner unit of this seed had an identical view; no waiting)")
            r, a = family_auditor(kind, fam)(V, D, slate=slate, seeds=seeds)
            recs[fam] = r
            arrays[f"keys_{fam}"] = np.asarray(r["inner_predictions"]["keys"])
            arrays[f"P_{fam}"] = r["inner_predictions"]["P"]
            for nm, A in a.items():
                arrays[f"SEL_{fam}_{nm}"] = A
            fam_pub[fam] = _pub(r)
        fam_cpu[fam] = round(time.process_time() - cf, 2)
    if kind != "policy" and primary not in recs:
        raise RuntimeError("only code families are ever reused")
    arrays["sel_row_id"] = np.asarray(D["row_id"])[DA.roles(D)[1]]
    composed = None
    if kind == "source":
        teacher = pc["teacher"]
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
        own_t = u_t if pc["teacher"] == "U" else BL.load_teacher(pc["teacher"], k, D, units_dir)[0]
        dp = UT.decision_preservation(out, own_t, D)
        basis = "released hard_i == the code's teacher d_i on every D row"
    else:
        dp = UT.decision_preservation(out, out, D)
        basis = "continuous source / reference: the release's decisions are its own (identity, still computed)"
    preserved = {"1": bool(dp["tasks"]["0"]["ok"]), "2": bool(dp["tasks"]["1"]["ok"])}
    g = UT.gate_record(util, U_u, {1: preserved["1"], 2: preserved["2"]})
    res = {"schema": SCHEMA, "kind": kind, "cid": cid, "seed": k, "unit_of": prov["unit"],
           "inner_unit": inner_of(k, cid),
           "of_complete_sha256": prov.get("complete_sha256"), "u_teacher_complete_sha256": u_prov["complete_sha256"],
           "primary_family": primary, "recovery": fam_pub[primary],
           "families": {f: v for f, v in fam_pub.items() if f != primary}, "composed": composed,
           "utility": DA._jsonable(util), "utility_gate": DA._jsonable(g), "preserved": preserved,
           "decision_preservation": DA._jsonable({**dp, "basis": basis}), "slate": slate,
           "slate_members": [nm for nm, _ in DA.SLATES[slate]()], "attacker_seeds": list(seeds),
           "token_states": None,                 # JSON null for continuous releases, never Infinity
           "token_states_rule": "null: continuous release (no finite token alphabet); never serialised as Infinity",
           "family_cpu_s": fam_cpu, "reused_families": reused, "family_paths": paths,
           "private_file": "inner_preds.npz (keys_<fam>, P_<fam> seed-0 candidates; SEL_<fam>_<crit>_<view> (seeds, "
                           "n_INNER) selected attackers; sel_row_id)"}
    if kind == "policy":
        res.update({"arm": pc["arm"], "decoder": pc["decoder"], "base_family": pc["base_family"],
                    "privacy_trained": bool(pc["privacy_trained"]), "lam": pc["lam"],
                    "diagnostic_families": list(DIAGNOSTIC_FAMILIES),
                    "primary_rule": "nomination, composition and primary results use the complete code interface "
                                    "(family 'code'); token-only and probability-only families are diagnostics",
                    "d0_token_parity": d1p, "token_family_path": paths["token"]})
        a1, a2 = (int(np.asarray(z[f"alpha{i}"]).ravel()[0]) for i in (1, 2))
        fit = _idx(D, FIT_NAME)
        res["token_states"] = a1 + a2
        res["token_states_rule"] = "alpha1 + alpha2 of the deployed release (declared alphabets incl. reserved tokens)"
        res["token_states_detail"] = {f"recipient_{i}": {"alphabet": a, "defense_fit_occupied": int(
            len(np.unique(np.asarray(z[f"tok{i}"])[fit]))), "all_rows_occupied": int(len(np.unique(z[f"tok{i}"])))}
            for i, a in ((1, a1), (2, a2))}
        res["coverage_receipt"] = coverage_receipt(z["tok1"], z["tok2"], D, a1, a2)
        res["mi_diagnostic"] = mi_diagnostic(z["tok1"], z["tok2"], D)
        res["view_fingerprint"] = fps["code"]
        res["view_fingerprints"] = fps
    elif kind == "reference" and "cells" in sets:
        T = sets["cells"]["tokens"]
        res["coverage_receipt"] = coverage_receipt(T["v1"], T["v2"], D)
        res["mi_diagnostic"] = mi_diagnostic(T["v1"], T["v2"], D)
    res["wall_s"] = round(time.time() - t0, 2)
    res["cpu_s"] = round(time.process_time() - c0, 2)
    res = DA._jsonable(res)
    _check_finite(res)
    res["_files"] = {"inner_preds.npz": arrays}
    if kind == "policy":
        res["_files"][SIDECAR] = side
    return res


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
    bad = _nonfinite_paths({k: v for k, v in res.items() if k != "_files"})        # every float finite
    if bad:
        raise RuntimeError(f"TECHNICAL FAILURE: nonfinite values in the inner record of {res.get('cid')}: {bad[:5]}")
    json.dumps({k: v for k, v in res.items() if k != "_files"}, allow_nan=False)


# ------------------------------------------------------------------ L5: recomputed D0 audits vs the admitted cbp audits
PARITY_KEYS = ("inner_predictions_sha256", "selected", "ce_selected", "auc_seed0", "ce_seed0", "auc_per_seed",
               "ce_per_seed", "auc", "ce", "sel_row_id_sha256", "fit_row_id_sha256", "slate_members", "attacker_seeds",
               "inner_prediction_keys")


def cbp_parity(k, cid, units_dir=None, tol=0.0):
    """Records-only receipt: the recomputed lcr inner audit (aud__<unit>) of an admitted D0 code or reference equals the
    admitted cbp inner audit (inner__<unit>, custody) on every family cbp audited (primary code family; every
    reference family) -- predictions hash, selected readers, per-seed AUC / CE -- and on utility, preservation and
    token_states. Expected bitwise (same slate, seeds, roles and views). No data is loaded."""
    c = parse_cid(cid)
    if c["kind"] == "policy" and c["arm"] != "d0":
        raise ValueError(f"{cid} is not an admitted D0 code")
    if c["kind"] == "source":
        raise ValueError("composed sources are rebuilt over the lcr bank; there is no cbp parity for them")
    unit = unit_of(k, cid)
    root = _units(units_dir)
    cb_name = f"inner__{unit}"
    if not _complete(cb_name, units_dir):
        raise SystemExit(f"REFUSED: admitted cbp audit {cb_name} is missing or not hash-complete")
    cb = json.loads((root / cb_name / "record.json").read_text())
    lc, _ = load_inner(inner_of(k, cid), units_dir)
    mm = []
    cfams = {cb["primary_family"]: cb["recovery"], **(cb.get("families") or {})}
    lfams = {lc["primary_family"]: lc["recovery"], **(lc.get("families") or {})}
    if cb.get("primary_family") != lc.get("primary_family"):
        mm.append("primary family differs")
    for fam, r in cfams.items():
        if fam not in lfams:
            mm.append(f"{fam}: absent from the lcr record")
            continue
        for key in PARITY_KEYS:
            a, b = r.get(key), lfams[fam].get(key)
            if not _close(a, b, tol):
                mm.append(f"{fam}.{key} differs")
    for key in ("utility", "preserved", "token_states", "slate_members", "attacker_seeds"):
        if not _close(cb.get(key), lc.get(key), tol):
            mm.append(f"{key} differs")
    return {"ok": not mm, "mismatches": mm, "cid": cid, "seed": k, "cbp_unit": cb_name, "lcr_unit": inner_of(k, cid),
            "families_compared": sorted(cfams), "lcr_only_families": sorted(set(lfams) - set(cfams)),
            "tolerance": tol}


def _close(a, b, tol):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[x], b[x], tol) for x in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, (float, int)) or isinstance(b, float) and isinstance(a, (float, int)):
        return abs(float(a) - float(b)) <= tol
    return a == b


# ------------------------------------------------------------------ post-hoc record checks (sec. 14 defect guards)
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
    criterion) is frozen, every frozen code won something, the composed value is never below the own bank's, the
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
    bad += [f"frozen code {c} won no (family, view, criterion)" for c in freeze if c not in won]
    return bad


def _need_cc(r, views=("v1", "v2")):
    keys = set(r.get("inner_prediction_keys") or [])
    need = {f"{v}:CC_alpha{a}" for v in views for a in CC_ALPHAS} | {f"pair:CCpair_alpha{a}" for a in CC_ALPHAS}
    return bool(r.get("finite")) and need <= keys


def validate_inner(rec, arrays, D, release=None, registered=True, tol=1e-12):
    """L7. Check one SAVED inner record against its stored INNER_SELECTION predictions; raises AuditDefect listing every
    violation, else returns a receipt. Uses an independent fixed-orientation AUC / CE (sklearn) and the registered
    first-best tie rule:
      schema       lcr-inner-v1;
      orientation  every candidate's inner_auc / inner_ce equals the fixed-orientation statistic of its stored P(S=1)
                   (a flipped score or max(AUC, 1 - AUC) is caught);
      best reader  the AUC-selected candidate is the first bank row with the maximum inner AUC (CE: minimum) and
                   auc_seed0 / ce_seed0 equal it (a reversed best/worst selection is caught);
      seed refits  SEL_<fam>_<crit>_<view>[0] is the selected candidate's seed-0 prediction; per-seed statistics are
                   the fixed-orientation values of the stored refits and the own recovery is their mean;
      code views   (codes) primary family "code" = finite bank with CC readers for v1 / v2 and CCpair on the exact
                   tuple, coalition and both ignore-the-other-recipient banks in the pair table; diagnostic families
                   token (finite, CC readers) and prob (finite by value) present; with `release`: exactly the
                   registered release keys (appended clean probabilities refused) and view fingerprints of all three
                   families equal to those rebuilt from the release (omitted token identities, appended
                   probabilities or a misaligned pair view change them); token_states = alpha1 + alpha2;
      sources      token_states null; validate_composed (every composed winner frozen; registered 83-code bank when
                   `registered`);
      JSON         every float finite."""
    bad = []
    if rec.get("schema") != SCHEMA:
        bad.append(f"schema {rec.get('schema')!r} is not {SCHEMA}")
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
        if rec.get("primary_family") != PRIMARY_CODE_FAMILY or r.get("family") != "code":
            bad.append("the primary family of a code is not the complete code interface")
        if not _need_cc(r):
            bad.append("code audited without token-identity readers (finite bank / CC readers missing)")
        cands = {row["candidate"] for row in r["tables"]["pair"]}
        if not {"coalition", "ignore_recipient_2", "ignore_recipient_1"} <= cands:
            bad.append("pair bank lacks the coalition or an ignore-the-other-recipient bank")
        diag = rec.get("families") or {}
        for fam in DIAGNOSTIC_FAMILIES:
            if fam not in diag:
                bad.append(f"diagnostic family {fam} missing")
            elif not _need_cc(diag[fam]):
                bad.append(f"diagnostic family {fam} audited without its cell readers")
        if release is not None:
            try:
                check_release_keys(release)
                fps = {fam: DA.view_fingerprint(V) for fam, V in code_view_sets(release, D).items()}
                if fps["code"] != rec.get("view_fingerprint") or fps["code"] != (rec.get("view_fingerprints") or {}
                                                                                 ).get("code"):
                    bad.append("audited views differ from the release's registered code views (token identities, "
                               "appended probabilities or pair alignment)")
                for fam in DIAGNOSTIC_FAMILIES:
                    if fps[fam] != (rec.get("view_fingerprints") or {}).get(fam):
                        bad.append(f"audited {fam} views differ from the release's {fam}-only views")
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
    """L7 over the whole saved lcr inner bank (lead / verifier; inner roles only): every registered code (with its
    release), both sources (registered composition) and the three references, every seed. Returns {"ok", "checked",
    "defects": [...], "missing": [...]}; never raises on a defect (the caller decides)."""
    guard_inner(D)
    ids = registered_composition_bank() + ["SRC|U", "SRC|RAW-J_b0.3", "REF|E", "REF|F", "REF|F0"]
    out = {"checked": 0, "defects": [], "missing": []}
    for k in seeds:
        for cid in ids:
            name = inner_of(k, cid)
            if not _complete(name, units_dir):
                out["missing"].append(name)
                continue
            try:
                rec, arr = load_inner(name, units_dir)
                rel = BL.load_unit_npz(unit_of(k, cid), "release.npz", units_dir)[0] if \
                    parse_cid(cid)["kind"] == "policy" else None
                validate_inner(rec, arr, D, release=rel, registered=True)
            except (AuditDefect, SystemExit) as e:
                out["defects"].append(f"{name}: {e}")
            out["checked"] += 1
    out["ok"] = not out["defects"] and not out["missing"]
    return out


# ------------------------------------------------------------------ lcr.run stage helpers (L9)
def inner_jobs(seeds=(0, 1, 2), kinds=("policy", "reference")):
    """Ordered (seed, cid) jobs of the inner stage: every D0 code (all seeds), then every D1 fixed-map code (so the
    token-only reuse of L5 normally finds its D0 audit), then the new fits, then the references."""
    R = _R()
    groups = []
    if "policy" in kinds:
        groups += [R.d0_ids(), R.d1_fixed_ids(), R.new_fit_ids()]
    if "reference" in kinds:
        groups += [[f"REF|{r}" for r in R.REFS]]
    return [(k, c) for grp in groups for c in grp for k in seeds]


def source_jobs(seeds=(0, 1, 2)):
    R = _R()
    return [(k, f"SRC|{t}") for k in seeds for t in R.TEACHERS]


def inner_job(k, cid, D, units_dir=None, slate="final"):
    """(unit name, files, record) of one inner job, ready for lcr.run.save(name, files, record)."""
    t0, c0 = time.time(), time.process_time()
    r = inner_unit(parse_cid(cid)["kind"], k, cid, D, units_dir=units_dir, slate=slate)
    files = r.pop("_files", {})
    r.update({"of": unit_of(k, cid), "config": cid, "seed": k, "job_wall_s": round(time.time() - t0, 2),
              "job_cpu_s": round(time.process_time() - c0, 2)})
    return inner_of(k, cid), files, r


def _lcr_done(name, units_dir=None):
    """An lcr inner unit is done iff it is hash-complete with schema lcr-inner-v1 (else it is redone)."""
    return _lcr_inner_ok(name, units_dir)


def stage_inner(D, shard_spec=None, units_dir=None, save=None, slate="final"):
    """Inner stage: this shard's inner_jobs() (round-robin i/n over the ordered list), resumable."""
    R = _R()
    save = save or R.save
    for k, cid in R.shard(inner_jobs(), shard_spec):
        if _lcr_done(inner_of(k, cid), units_dir):
            continue
        save(*inner_job(k, cid, D, units_dir, slate))


def stage_inner_src(D, shard_spec=None, units_dir=None, save=None, slate="final"):
    """Composed-source stage: REFUSES unless every seed's 83-code bank is closed (releases + lcr inner units)."""
    R = _R()
    save = save or R.save
    for k in (0, 1, 2):
        clo = _closure(k, "U", expected_policies("U"), units_dir)
        if not clo["ok"]:
            raise SystemExit(f"REFUSED: composed source banks need the closed 83-code bank first (s{k}): "
                             f"{ {x: v for x, v in clo.items() if x != 'ok'} }")
    for k, cid in R.shard(source_jobs(), shard_spec):
        if _lcr_done(inner_of(k, cid), units_dir):
            continue
        save(*inner_job(k, cid, D, units_dir, slate))


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


# L8: fixed control plan (registered before any real-data control runs; chosen by structure, never by results):
#   codes (null + CONF_r1/r2 + COLL_r1/r2 + XOR), seed 0: cbp's three -- U|DIRECT-TASK|i8o64 (the task-only code at
#   the one rate), the JOINT code at the LARGEST registered lambda (U|JOINT|i8o64|l0.1), U|CLASS|i1o1 -- plus the D1
#   fixed-map version of that JOINT code (U|JOINT|i8o64|l0.1|D1; same tokens, learned decoder) and the new constrained
#   code with the largest registered neighbourhood (U|K-JOINT-PAIR|i8o64|D1);
#   sources (null on all five families + ROT_r1/ROT_r2 on the interface family), seed 0: U and RAW-J_b0.3;
#   references (null on every family; ROT_r1/ROT_r2 on E's interface family), seed 0: E, F, F0;
#   null calibration: 5 permutations on the first code.
def control_plan(protocol=None):
    joint = f"U|JOINT|i{REG_RATE[0]}o{REG_RATE[1]}|l{_g(max(REG_LAMS))}"
    pols = [REG_TASK[0], joint, REG_TASK[2], joint + "|D1", f"U|K-JOINT-PAIR|i{REG_RATE[0]}o{REG_RATE[1]}|D1"]
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
        return {f"{x}|s{k}": {"unit": u, "complete_sha256": sha, "arm": parse_cid(x)["arm"],
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
    if source_threshold is not None:            # realised threshold receipt against the source's
        thr_ok = bool(thr) and all(abs(t - source_threshold) <= 1e-12 for t in thr)
        if not thr_ok:
            failures.append(f"REALISED_NULL_THRESHOLD {thr} != source {source_threshold}")
    else:
        thr_ok = None
    plan_codes = [c for _, c in plan.get("policies", [])]
    arms = sorted({parse_cid(c)["arm"] for c in plan_codes})
    verdict = {"all_ok": not failures and not calib_exc and calib is not None, "failures": failures,
               "realised_null_thresholds": thr, "realised_threshold_matches_source": thr_ok,
               "null_calibration_exceedances": calib_exc, "null_calibration_present": calib is not None,
               "code_arms_covered": arms,
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
    """lcr.run LATE "controls" entry: runs this shard's jobs of the fixed plan (private partial receipts); when every
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
                                                "cpu_s": round(time.process_time() - c0, 1)}), allow_nan=False))
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
    txt = json.dumps(DA._jsonable(out), indent=1, allow_nan=False)        # finite public JSON
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
    probability (label-free); q = smoothed OSF_DEFENSE_FIT token means (a D0-style decoder); a reserved fallback token
    per absent class. sex_tilt > 0 (synthetic only) moves rows to the next token by SEX to create a leak."""
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


def synthetic_d1(z, mix=0.3, eps=1e-12):
    """D1-shaped synthetic release of a code z: the SAME tokens, decisions and alphabets; per token the decoded vector
    is moved toward its class, u = (1 - mix) q_t + mix e_d, released as (u + eps 1 + eps e_d) / (1 + (K + 1) eps)
    (class preserving, token constant). A stand-in for lcr.decoder output (synthetic shapes only)."""
    zp = {k: np.array(v) for k, v in z.items()}
    for i in (1, 2):
        q = np.asarray(z[f"q{i}"], dtype=np.float64)
        K = q.shape[1]
        e = DA.onehot(z[f"hard{i}"], K)
        u = (1 - mix) * q + mix * e
        zp[f"q{i}"] = (u + eps + eps * e) / (1 + (K + 1) * eps)
        assert np.array_equal(zp[f"q{i}"].argmax(1), np.asarray(z[f"hard{i}"]))
    return zp


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


LCR_BANK = {
    "seeds": 3,
    "codes_per_seed": {"d0_i8o64": 26, "d0_i1o1": 1, "d1_i8o64": 26, "new_i8o64": 30},
    "codes_note": "83 registered codes per seed (lcr.run.code_ids()): D0 27 (DIRECT-TASK, FINE-TASK, CLASS-ONLY i1o1, "
                  "24 privacy maps), D1 fixed-map 26 (same tokens as D0, learned q), new fits 30 (C-TASK, 24 weighted, "
                  "5 constrained); every code at most 8 / 64 states per predicted class (i8o64 shapes). Each code "
                  "inner unit audits three families (code primary + token-only + probability-only diagnostics); a D1 "
                  "unit reuses its D0 map's token family (L5)",
    "sources": 2, "references": {"continuous_E": 1, "cells_F_F0": 2},
    "composed_codes_per_source_seed": {"SRC|U": 83, "SRC|RAW-J_b0.3": 0},
    "inner_units": {"codes": 249, "of_which_d0_recomputed": 81, "sources": 6, "references": 9},
    "control_plan": {"codes": 5, "source_families": 10, "reference_families": 15, "rot_views": 3,
                     "null_calibration_reps": 5},
    "assessment": {"labels_per_seed": 20, "codes_per_seed": 15, "codes_with_diagnostics_per_seed": 2,
                   "sources_per_seed": 2, "references_per_seed": 3,
                   "src_u_composed_codes_freeze_plus_scored": 20, "src_u_composed_codes_upper": 83,
                   "note": "about 20 locked labels per seed (lead): ~15 codes (P*, N*, J*, T*, C*, C_pair*, Q, best "
                           "D1 control + its D0, best weighted, C-TASK, 5 constrained arms; aliases deduplicated), "
                           "SRC|U, SRC|RAW-J, E, F, F0. SRC|U is costed (a) with the freeze list + the scored codes "
                           "(~20; exact, see recommendation) and (b) with all 83 codes (upper bound). Token-only / "
                           "probability-only diagnostics are scored only for the registered decoder-ablation pair (the "
                           "best D1 fixed-map control and its D0 map; lead's lcr.eval_lock); every other code scores "
                           "the primary family only"}}
# cbp's measured real study-worker CPU over its synthetic estimate (results/pcrl_confidence_budgeted_privacy_v1/
# COST_AND_CLOSEOUT.md lead stages: inner + inner_src 79.9 CPU-min vs its estimate 0.951 CPU-h; assessment 90.3 CPU-min
# vs 1.608 CPU-h); controls use qpc's ratio (cbp did not separate them). Planning figures only.
CBP_CALIBRATION = {"real_over_synthetic": {"inner": round(79.9 / 60 / 0.951, 3),
                                           "controls": round(714 / (0.318 * 3600), 3),
                                           "assessment": round(90.3 / 60 / 1.608, 3)},
                   "source": "results/pcrl_confidence_budgeted_privacy_v1/COST_AND_CLOSEOUT.md (lead stages) and "
                             "AUDIT_COMPUTE.json (estimates); controls: results/pcrl_confidence_capacity_v1"}


def _cpu(fn, *a, **kw):
    c0, t0 = time.process_time(), time.time()
    r = fn(*a, **kw)
    return r, round(time.process_time() - c0, 2), round(time.time() - t0, 2)


def estimates(meas, bank=None):
    """L9: CPU estimates of the lcr inner bank, controls and the single assessment from the synthetic measurements."""
    bank = bank or LCR_BANK
    pin = meas["policy_inner"]
    cps = bank["codes_per_seed"]
    full = pin["d0_i8o64"]["unit_cpu_s"]
    fam = pin["d0_i8o64"]["family_cpu_s"]
    d1 = pin["d1_i8o64_token_reused"]["unit_cpu_s"]
    cls = pin["d0_i1o1"]["unit_cpu_s"]
    codes = cps["d0_i8o64"] * full + cps["d0_i1o1"] * cls + cps["d1_i8o64"] * d1 + cps["new_i8o64"] * full
    codes_no_reuse = codes + cps["d1_i8o64"] * fam["token"]     # the D1 token family computed instead of reused
    src = meas["source_unit"]["SRC|U_cpu_s"] + meas["source_unit"]["SRC|RAW-J_cpu_s"]
    refs = meas["reference_unit"]["E_cpu_s"] + bank["references"]["cells_F_F0"] * meas["reference_unit"]["F_cpu_s"]
    per_seed = codes + src + refs
    d0_extra = (cps["d0_i8o64"] * fam["code"] + cps["d0_i1o1"] * pin["d0_i1o1"]["family_cpu_s"]["code"] + refs)
    est = {"inner_units": sum(bank["inner_units"].values()) - bank["inner_units"]["of_which_d0_recomputed"],
           "inner_cpu_s_per_seed": round(per_seed, 1),
           "inner_cpu_s_per_seed_parts": {"codes": round(codes, 1), "sources_incl_composition": round(src, 1),
                                          "references": round(refs, 1)},
           "inner_cpu_h": round(bank["seeds"] * per_seed / 3600, 3),
           "inner_cpu_h_without_token_reuse": round(bank["seeds"] * (codes_no_reuse + src + refs) / 3600, 3),
           "inner_cpu_h_primary_family_only": round(bank["seeds"] * (
               (cps["d0_i8o64"] + cps["d1_i8o64"] + cps["new_i8o64"]) * fam["code"] +
               pin["d0_i1o1"]["family_cpu_s"]["code"] + src + refs) / 3600, 3),
           "d0_recompute_extra_cpu_h": round(bank["seeds"] * d0_extra / 3600, 3),
           "d0_recompute_extra_basis": "what reusing the admitted cbp audits would have saved: the primary code family "
                                       "of the 27 D0 codes and the 3 references per seed (the token-only and "
                                       "probability-only diagnostics are needed under either choice)",
           "per_unit_cpu_s": {k: v["unit_cpu_s"] for k, v in pin.items()},
           "per_family_cpu_s_i8o64": fam,
           "composed_assembly_cpu_s_83_codes": meas["source_unit"].get("composed_assembly_cpu_s")}
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
        diag_f = f["code_i8o64_token_family_cpu_s"] + f["code_i8o64_prob_family_cpu_s"]
        src_own = f["source_own_five_families_cpu_s"]
        refs_f = a["references_per_seed"] * src_own + 2 * code_f
        base = a["codes_per_seed"] * code_f + 2 * src_own + refs_f
        lo = base + a["src_u_composed_codes_freeze_plus_scored"] * code_f
        hi = base + a["src_u_composed_codes_upper"] * code_f
        est["assessment_cpu_h"] = round(bank["seeds"] * lo / 3600, 3)
        est["assessment_cpu_h_with_code_diagnostics"] = round(bank["seeds"] * (
            lo + a["codes_with_diagnostics_per_seed"] * diag_f) / 3600, 3)
        est["assessment_cpu_h_diagnostics_on_every_code"] = round(bank["seeds"] * (lo + a["codes_per_seed"] * diag_f)
                                                                  / 3600, 3)
        est["assessment_cpu_h_upper_all_83_composed"] = round(bank["seeds"] * (hi + a["codes_per_seed"] * diag_f) /
                                                              3600, 3)
        est["assessment_basis"] = (
            f"per seed: {a['codes_per_seed']} codes x the primary family at the i8o64 cost (+ token (cells only, "
            f"L4b) / prob diagnostics for {a['codes_with_diagnostics_per_seed']} codes in the _with_code_diagnostics "
            "figure, for all codes in _diagnostics_on_every_code); both sources' five own families; SRC|U composed "
            f"with {a['src_u_composed_codes_freeze_plus_scored']} codes (freeze list + scored codes; the composed code "
            "bank is fitted once per unit and shared across the four composing families by the content memo) or all "
            f"{a['src_u_composed_codes_upper']} (upper); E / F / F0 at the five-family source cost plus one cells "
            "family each for F / F0")
    total = est["inner_cpu_h"] + est.get("controls_cpu_h", 0) + est.get("assessment_cpu_h_with_code_diagnostics", 0)
    est["total_cpu_h"] = round(total, 3)
    est["total_cpu_h_upper"] = round(est["inner_cpu_h_without_token_reuse"] + est.get("controls_cpu_h", 0) +
                                     est.get("assessment_cpu_h_upper_all_83_composed", 0), 3)
    cal = CBP_CALIBRATION["real_over_synthetic"]
    ct = (est["inner_cpu_h"] * cal["inner"] + est.get("controls_cpu_h", 0) * cal["controls"] +
          est.get("assessment_cpu_h_with_code_diagnostics", 0) * cal["assessment"])
    est["cbp_calibrated_total_cpu_h"] = round(ct, 3)
    est["cbp_calibration"] = CBP_CALIBRATION
    est["budget_with_x2_margin_cpu_h"] = round(2 * total, 2)
    est["wall_h_two_workers"] = round(total / 2, 2)
    est["safety_note"] = ("synthetic MLP early stopping converges fast; cbp's synthetic estimate (2.89 CPU-h) against "
                          "about 2.8 CPU-h measured for its lead's audit stages; the x2 margin is the scheduling "
                          "budget")
    return est


def registration_block():
    """Registered (pre-fit) rules written into AUDIT_COMPUTE.json by `timing` and refreshed by `estimate`."""
    return {"registered_control_limits": CONTROL_LIMITS, "control_plan": DA._jsonable(control_plan()),
            "registered_composition_bank": {"codes": registered_code_bank(), "composition_only": list(COMPOSED_EXTRA),
                                            "n": N_CODES},
            "code_families": {"primary": PRIMARY_CODE_FAMILY, "diagnostic": list(DIAGNOSTIC_FAMILIES),
                              "token_family_readers": TOKEN_FAMILY_READERS, "L4b": L4B_RULE},
            "alias_cache": "L5: exact view-fingerprint alias reuse within a seed (path recorded per family; bitwise "
                           "identical to recomputation); opportunistic, no saving assumed in the estimates beyond "
                           "the D1 token family"}


def reestimate(path, bank=None):
    """Recompute the estimates (and refresh the registration block) of an existing AUDIT_COMPUTE.json from its
    measurements (no new timing)."""
    d = json.loads(Path(path).read_text())
    d.update(registration_block())
    d["bank"] = bank or LCR_BANK
    d["estimates"] = estimates(d["measured"], d["bank"])
    Path(path).write_text(json.dumps(DA._jsonable(d), indent=1, allow_nan=False) + "\n")
    return d["estimates"]


def _save_unit(root, name, files, rec):
    """jcv.finalize.save_unit with lcr.run.save's writers (.npz: dict of arrays; .json: finite JSON)."""
    from jcv.finalize import save_unit

    def w(f, a):
        if f.endswith(".json"):
            return lambda p: p.write_text(json.dumps(DA._jsonable(a), allow_nan=False))
        return lambda p: np.savez_compressed(p, **a)
    save_unit(root / name, {f: w(f, a) for f, a in files.items()}, rec)


def _timing_store(root, D, t, rels, slate, meas):
    """Synthetic lcr-named store: teachers, references, the 83 registered codes. Measured inner units: one D0 i8o64
    code (all three families), its D1 version (token family reused, L5), the class-only code; the other registered
    names reuse the measured record of their kind (composition reads records only; the closure is real)."""
    _save_unit(root, "tea__s0__U", {"teacher.npz": t}, {"t": "U"})
    _save_unit(root, "tea__s0__RAW-J_b0.3", {"teacher.npz": synthetic_teacher(D, seed=9)}, {"t": "RAW-J"})
    _save_unit(root, "ref__s0__E", {"reference.npz": {k: v for k, v in synthetic_teacher(D, seed=5).items()}},
               {"t": "E"})
    _save_unit(root, "ref__s0__F", {"reference.npz": synthetic_reference_cells(D, t, ncell=(82, 97))}, {"t": "F"})
    def kind_of(c):
        if is_class_only(c):
            return "d0_i1o1"
        return {"d0": "d0_i8o64", "d1_fixed": "d1_i8o64"}.get(parse_cid(c)["arm"], "new_i8o64")
    for cid in registered_code_bank():
        _save_unit(root, unit_of(0, cid), {"release.npz": rels[kind_of(cid)]}, {"synthetic": True})
    measured = {}
    order = ["U|JOINT|i8o64|l0.1", "U|JOINT|i8o64|l0.1|D1", "U|CLASS|i1o1"]
    keys = {"U|JOINT|i8o64|l0.1": "d0_i8o64", "U|JOINT|i8o64|l0.1|D1": "d1_i8o64_token_reused",
            "U|CLASS|i1o1": "d0_i1o1"}
    for cid in order:
        rec, cpu, wall = _cpu(inner_unit, "policy", 0, cid, D, units_dir=root, slate=slate)
        files = rec.pop("_files")
        measured[cid] = (rec, files)
        m = meas["policy_inner"].setdefault(keys[cid], {})
        m.update({"unit_cpu_s": cpu, "unit_wall_s": wall, "family_cpu_s": rec["family_cpu_s"],
                  "reused_families": sorted(rec["reused_families"])})
        _save_unit(root, inner_of(0, cid), files, rec)
    src = {"d0_i8o64": "U|JOINT|i8o64|l0.1", "d1_i8o64": "U|JOINT|i8o64|l0.1|D1", "new_i8o64": "U|JOINT|i8o64|l0.1",
           "d0_i1o1": "U|CLASS|i1o1"}
    for cid in registered_code_bank():
        if cid in order:
            continue
        rec, files = measured[src[kind_of(cid)]]
        _save_unit(root, inner_of(0, cid), files, {**rec, "cid": cid, "unit_of": unit_of(0, cid),
                                                    "inner_unit": inner_of(0, cid)})


def timing(slate="final", seed=0, out_path=None, bank=None, controls=True, final=True, workdir=None):
    """L9: CPU seconds on study-shaped synthetic data at the study's real row counts and alphabets (income 8 per class x
    2, occupation 64 per class x 5 predicted classes + 1 reserved token): full inner units (three families, attack
    bank + seed refits + utility + coverage + MI) of a D0 code, its D1 version (token family reused) and the
    class-only code; both sources with the REGISTERED 83-code composition (real closure and registration checks); E-
    and F-like references; controls; final audits. Writes AUDIT_COMPUTE.json (finite JSON, registered control
    limits)."""
    import platform
    import resource
    bank = bank or LCR_BANK
    D = synthetic_D(seed=seed)
    t = synthetic_teacher(D, seed=seed)
    meas = {"policy_inner": {}, "source_unit": {}, "reference_unit": {}, "controls": {}, "final": {}}
    z0 = synthetic_release(D, t, 8, 64, sex_tilt=0.1)
    rels = {"d0_i8o64": z0, "d1_i8o64": synthetic_d1(z0), "new_i8o64": z0, "d0_i1o1": synthetic_release(D, t, 1, 1)}
    shapes = {}
    for nm, z in (("i8o64", z0), ("i1o1", rels["d0_i1o1"])):
        S = code_view_sets(z, D)
        shapes[nm] = {"alphabets": [int(z["alpha1"]), int(z["alpha2"])],
                      "dims": {f: {w: int(S[f]["X"][w].shape[1]) for w in PRIMARY_VIEWS} for f in S},
                      "pair_tuples_occupied_audit_fit": int(len(np.unique(DA._pair_keys(z["tok1"], z["tok2"])[
                          _idx(D, FIT_ROLE)])))}
    meas["shapes"] = shapes
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
                meas["source_unit"]["SRC|U_closure_ok"] = rec["composed"]["closure"]["ok"]
                fams = {rec["primary_family"]: rec["recovery"], **rec["families"]}
                own = {f: {**v, **v["own"]} for f, v in fams.items()}      # the own banks, from the saved record
                _, cpu, _ = _cpu(composed_source_bank, 0, "U", own, D, units_dir=root)
                meas["source_unit"]["composed_assembly_cpu_s"] = cpu
        for lab in ("E", "F"):
            _, cpu, wall = _cpu(inner_unit, "reference", 0, f"REF|{lab}", D, units_dir=root, slate=slate)
            meas["reference_unit"][f"{lab}_cpu_s"] = cpu
    zb = z0
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
        S = code_view_sets(zb, D)
        for f, key in (("code", "code_i8o64_one_family_cpu_s"), ("prob", "code_i8o64_prob_family_cpu_s")):
            _, cpu, _ = _cpu(DA.final_audit, S[f], Du, a, slate=slate)
            meas["final"][key] = cpu
        _, cpu, _ = _cpu(final_audit_cells, S["token"], Du, a)                 # L4b (as lcr.assess scores it)
        meas["final"]["code_i8o64_token_family_cpu_s"] = cpu
        own = 0.0
        for fam, V in BL.source_view_sets(t, D).items():
            _, c1, _ = _cpu(DA.final_audit, V, Du, a, slate=slate)
            own += c1
        meas["final"]["source_own_five_families_cpu_s"] = round(own, 2)
        Vi = BL.source_view_sets(t, D, families=("interface",))["interface"]
        _, cpu, _ = _cpu(DA.final_audit, Vi, Du, a, composed=[("c1", S["code"]),
                                                               ("c2", policy_views(rels["d1_i8o64"], D))],
                         slate=slate)
        meas["final"]["source_interface_with_2_composed_cpu_s"] = cpu
        meas["final"]["score_rows"] = int(len(a))
    est = estimates(meas, bank)
    out = {"study": STUDY, "owner": "D (attacker / statistics engineer)",
           "kind": "SYNTHETIC timing estimates (no real data, no labels); real views may converge differently (MLP "
                   "early stopping); one process under lcr.sema, OMP_NUM_THREADS=1; cpu_s = process CPU time of the "
                   "measuring process",
           "fixture": "lcr.audit.synthetic_D (OSF_DEFENSE_FIT 15,434 / AUDIT_FIT 6,065 / INNER_SELECTION 2,235 / "
                      "assessment 13,936 rows, P(S=1) = 0.68) + synthetic_release i8o64 (income 8 per class x 2, "
                      "occupation 64 per class x 5 predicted classes + 1 reserved token; SEX-tilted 10%) and its "
                      "synthetic_d1 (same tokens, learned-style q); FINAL slate (LR x5, MLP x4, HGB x4, DA_LR, DA_MLP) "
                      "+ cell readers {0.5, 1, 5}; the AUC- and CE-selected attackers refit at attacker seeds 0-2; "
                      "three code families (code / token / prob); a temporary lcr-named unit store with the "
                      "registered 83-code composition bank (real registration and closure checks); token-only "
                      "family with the cell readers only (L4b)",
           **registration_block(),
           "d0_reuse_decision": "RECOMPUTE: the admitted cbp inner audits lack the token-only and probability-only "
                                "families, so the exact parity rule fails; all 81 D0 code and 9 reference inner units "
                                "are recomputed (aud__ namespace); cbp_parity compares the recomputed primary family "
                                "with the admitted cbp record",
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
    """Merge ONLY the "audit" key into TIMING.json (C owns "fitting"): read-modify-write under an exclusive flock on
    TIMING.json.lock, temporary file + os.replace; every other key untouched."""
    d = json.loads(Path(audit_compute).read_text())
    tp = Path(timing_path)
    e = d["estimates"]
    entry = {"owner": "role D (attacker / statistics engineer)", "written_at": d.get("written_at"),
             "data": "SYNTHETIC ONLY (lcr.audit.synthetic_D at the real role sizes); no Adult row, task label or SEX "
                     "was read",
             "environment": {"threads": "OMP_NUM_THREADS=1, one process", "semaphore": "lcr.sema label D:audit-timing",
                             "python": d["machine"]["python"]},
             "measure": "process CPU seconds (time.process_time) inside the measuring process",
             "per_unit_cpu_s": e["per_unit_cpu_s"], "per_family_cpu_s_i8o64": e["per_family_cpu_s_i8o64"],
             "source_unit": d["measured"]["source_unit"], "reference_unit": d["measured"]["reference_unit"],
             "controls": d["measured"]["controls"], "final": d["measured"]["final"],
             "estimates_cpu_h": {"inner": e["inner_cpu_h"], "inner_without_token_reuse":
                                 e["inner_cpu_h_without_token_reuse"], "d0_recompute_extra":
                                 e["d0_recompute_extra_cpu_h"], "controls": e.get("controls_cpu_h"),
                                 "assessment": e.get("assessment_cpu_h"),
                                 "assessment_with_code_diagnostics": e.get("assessment_cpu_h_with_code_diagnostics"),
                                 "assessment_upper_all_83_composed": e.get("assessment_cpu_h_upper_all_83_composed"),
                                 "total": e["total_cpu_h"], "total_upper": e["total_cpu_h_upper"],
                                 "cbp_calibrated_total": e["cbp_calibrated_total_cpu_h"],
                                 "with_x2_margin": e["budget_with_x2_margin_cpu_h"],
                                 "wall_h_two_workers": e["wall_h_two_workers"]},
             "inner_units": e["inner_units"],
             "details": "results/pcrl_learned_decoder_constrained_release_v1/AUDIT_COMPUTE.json"}
    lockp = tp.with_name(tp.name + ".lock")
    with open(lockp, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        T = json.loads(tp.read_text()) if tp.exists() else {}
        T["audit"] = entry
        tmp = tp.with_name(tp.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(DA._jsonable(T), indent=1, allow_nan=False) + "\n")
        os.replace(tmp, tp)
        fcntl.flock(lf, fcntl.LOCK_UN)
    return entry


def main(argv=None):
    """python -m lcr.audit timing [--out results/pcrl_learned_decoder_constrained_release_v1/AUDIT_COMPUTE.json]
                                  [--timing-json results/pcrl_learned_decoder_constrained_release_v1/TIMING.json]
    python -m lcr.audit estimate --out <AUDIT_COMPUTE.json>                         (synthetic only; no real data)"""
    import argparse
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
        if a.timing_json:
            write_timing_key(a.out, a.timing_json)
        return
    r = timing(out_path=a.out, workdir=a.workdir)
    if a.out and a.timing_json:
        write_timing_key(a.out, a.timing_json)
    print(json.dumps(r["estimates"], indent=1))


if __name__ == "__main__":
    main()
