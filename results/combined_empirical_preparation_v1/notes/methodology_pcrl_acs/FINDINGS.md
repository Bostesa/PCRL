# FINDINGS — methodology audit of the PCRL ACS release-study lineage

(Saved by the owner from the role's final report; the role's own write was refused by the harness.)
Lineage withdrawn 2026-09-26; these are repairs for any future reuse.

Sources: M = `research/pcrl-submission-finish-v1`@55c0c5a35 `papers/pcrl_satml_final_v1/main.tex`;
TD = task-directed f4bdf4cd5; FP = final-prospective 5e154e5c4; OD = objective-diagnosis 3e67c5247;
GR = guarantee-review 1dfb17f2c. Numbers recounted in recounts.json (39/39 match stored values).

**F1 HIGH — H-relative contract not met in reporting.** M:26-28, 76-78, 187 promise H-relative and
absolute recovery; Tables II–IV are J-relative only. TD HEADLINE_EVIDENCE
`/nominees/utility_first/sensitive/*/*/recovery_over_H` 0.00086–0.00614 (8/8 positive); FP INFERENCE_2016
`/descriptive[recovery_over_H]` Q 6/8 and D17 8/8 intervals exclude 0; GR PROSPECTIVE_INTERPRETATION.md:46-55.
Repair: H-relative column + abstract sentence. Verify: printed = stored to 5 dp.

**F2 MED — P-1 wording at M:365** ("capping added recovery"; clause is CE_J−CE_M ≤ 0.001). GR:90-105.
Present at 55c0c5a3, fixed only in the PDF. Repair: apply GR fragment P-1. Verify: `git grep "capping added
recovery"` empty.

**F3 MED — Manuscript provenance gap.** Downloads PDF differs from main.tex in title, Fig. 1 notes, the
P-1 fix and the sentence "point estimates meet the 0.003 target" (true for both Q and D17); `git log --all
-S` finds no source. Repair: commit source + hashes. Verify: a ref compiles to the PDF text.

**F4 MED — 2016 "previously unused" / "no earlier stage had touched" (M:360).** Admission
(evidence-paper 0d8f4b67b, DATA_2016_ADMISSION.md:3-9) ran schema, code-set and missing checks on label
columns over all rows and stored fit/validation RAC1P counts (FP DATA_INDEPENDENCE.md:26-40). Lock set
before final labels were read (EVALUATION_LOCK.json 00:49Z); scoring 00:54Z 2026-09-23. 2016 now spent;
later 2018 designs cite the 2016 outcome (shared-context PROTOCOL.md:29). Repair: "not used for any
fitting, selection or scoring before the lock".

**F5 MED — Study 6 audit-slate defect** (disclosed v6 D8; unrepaired; not SaTML evidence). Wire is
[H_A, Z_J, R] (utility-extension METHOD.md:8); METHOD:20-21 promised in-slate J predictors; instead H-only
ancestors were routed (ablations-facct `acs_spectral_audits.py:172-183`) and ref_J run as a separate
condition. Repair: add column-projected [H_A, Z_J] candidates, reselect, rescore (needs archives). Verify:
`ancestor_J__*` candidates in every A/AB role.

**F6 MED — Catch-up gives unequal slates** (Sept 8–10 redesign; disclosed). Catch-up candidates exist
only where a saved observer exists (`acs_coalition_audits.py:297-307`); E has none (coalition TABLE.md:61).
Catch-up wins 52/54 race endpoints; in the source-guard study the coalition ordering reverses between
scopes. Repair: standard_independent as primary scope for cross-system contrasts. Verify: recompute from
stored PAIRED.csv.

**F7 LOW/MED — H-only floor vs "unclipped".** No numerical clipping (`uncertainty.py:6`), but when the H
route wins on validation the increment is exactly 0 and releases tie (privacy-first PRELOCK_NOTES.md:21-36,
RESEARCH_DECISION.md:25,105; TD T0_code A/RAC1P = −0.0). Repair: report selected route per role plus stored
`independent_selection` increments.

**F8 MED — C3/R1 cites selection-pool intervals.** M:350-352 and ledger R1 call them adjusted bounds;
replacement e3415b94d `replacement.py:122-180` selects probes on downstream_validation and scores the same
pool (ancestor_inclusive takes the min on the same rows); PROTOCOL.md:41, STATISTICAL_PLAN.md:107 say so;
AMENDMENT_1 M3 voids the family of 68; effect ~5.5 SE. Repair: "screening contrast", drop "68".

**F9 MED — Held-out CMI and solver status.** Coarse held-out CMI 0.0127–0.0247; four-cell 0.0164–0.0209
(OD OBJECTIVE_DIAGNOSIS.md:38-39); D17 violates all 4 of Q's constraints (:25); solver certified to
1.27e-7 in floating point (:40), so M:240-241 outdated. Repair: add, with plug-in-bias caveat.

**F10 LOW — Conditioning-scope wording.** Abstract (M:29) omits "coarse"; M:213-215 "joint-view budgets"
applies to (Z, coarse C), not (Z, H).

**F11 LOW — Table III caption** says "unweighted" but worst-sensitive column is max over both weightings
(`build_final_assets.py:86-87`); Q's −0.00246 is person-weighted, unweighted max −0.00394.

**F12 LOW/MED — "Matched" RR75/W75** are fixed ρ=0.75 mixtures of D17's map (FP common.py PANEL; TD
`mechanisms.py:259-298`) — matched on interface only; development nominees were Ttask_rr_0.75 and
T0_withhold_0.75_a33.

**F13 LOW/MED — Inference scope.** PWGTP ratio estimates with household bootstrap, no replicate weights;
3 development anchors split one cohort (test pools of 2,015 households each, 5,473 in union; a household
tested in one anchor is fitted in another); all 2016 anchors score the same 23,684 people; bounds are
estimate ± z·SE conditional on fitted models. Repair: state scope.

**F14 LOW — Null-token noise floor.** Independent random token (true increment 0) measures
[−0.00036, +0.00077], comparable to the +0.001 cap and to Q's smallest increment 0.00086; constant_best
exactly 0. Repair: report this calibration.

**F15 LOW — Bound-statement wording.** OD:31 calls the p_t-weighted bound "distribution-free" (only the
radius/capacity form is); GR's "fresh randomness" = independent of data, not a new draw per query.

**F16 INFO — Data years.** CA 2018 exhausted; 2017 spent (origin 349efa454; local ad2c08872 lacks 5
commits); 2016 spent. TX planned only; no NY. No income>50k task on the PublicCoverage low-income cohort.

**F17 INFO** — Precision-claim correction chain verified.

**F18 INFO — Verified correct:** nested H reference; full single-recipient attackers in coalition slates;
separate selection and scoring; token-law scoring with persistent tokens; J as comparator (C9).

**F19 INFO** — Stochastic-channel Stage B label leak and same-row minimum withdrawn (C3; replacement
PAPER_ADDENDUM:31).
