# Statistical reviewer, R1 card check and claims-review wave 1 (verbatim; delivered to the coordinator at about 22:28Z; harness indentation removed)

# RFA1 statistical reviewer: R1 card check and claims review wave 1 (v1)

Written 2026-10-25, 22:21–22:27Z. Everything read was a record or code: card `RF_RFIT_R1.md` (058176cc…), `claims/RFA_v1/DECISIONS.json` (884075bd…), `claims/PLAN_RFA_v1.json`, `rfa/rfclaims.py`, the relevant parts of `rfa/claims.py`, and the fit records `fam/FIT_RF_V{0,1}.json` (key names only). Nothing under `.private/` was read; that includes the grids and BOOT files. I read no label or row-level array, made no git change and spawned no agents. CPU this round was about 3 CPU-s; about 39 CPU-s across all waves.

## Integrity checks (all pass)

- The frozen rules files appear in git only in the registration commit 43830ae19 (21:51:57Z). `rfa/rules.py` at the decision commit 586ed2560 and at HEAD is 76baa04a…, and the record's `rules_sha256` is 4e03270c… (RULES.json).
- Registration order:
  - card committed 21:51:57Z;
  - the fit code (`rfclaims.py` with the falsifier code) committed 21:56:02Z;
  - plan written 21:59:21Z and committed 21:59:32Z;
  - first MODEL_SELECT read 21:59:53Z.

  So the reading is correctly the registered one. The F1, F5 and F6 thresholds (0.05; > 0.5; 0.02, 0.01 and 10×) were frozen before any MODEL_SELECT read.
- The record has every key the rules need: START, CONT, the three paths, `lo_not_saved`, `f0`, the ALTCRIT lag (0.109 / 0.210, so not a near alias), selection data `MODEL_FIT_OOF`, and grid, cost and `nolookup` entries for all ids and the nine references. The learning-curve half entries cover the gating points and H_LJ, with paired SEs (`PAIRED_BOOTSTRAP`). `cand_meta` records that no SEX-recovery diagnostic was used. Grid validity reads `READER_SEEDS_EFFECTIVE` (strict NRM1 effective; 16 equalities explained).

---

## (1) R1 card check: the frozen rules read R1 unchanged; no rule change is needed

`decisions()` reads a record with:
- revision 1;
- `card_committed_before_any_model_select_outcome` False (card first committed 22:23:27Z, after the 21:59:53Z read; `rfa.claims.registration_status` produces exactly this);
- family RFA or FAM;
- R1's own ids (RF1_FULL / RF1_TASK / RF1_STOP_LO, RF1_STOP_HI / RF1_ALTC_HI / RF1_RAWJ_HI), `lo_not_saved` [ALTCRIT, RAWJ], R1's `f0` and lag.

It reads that record as **ADAPTIVE_R1**: every label is prefixed and `campaign_summary` counts K = 2 looks. R1's STOPFIT (the same episodes and three families, R1's re-calibrated lambda, the fit path detached) is its isolating control.

I wrote **`roles/statistical_reviewer/RULES_ADDENDUM_1_R1.md`** (d7726c9d…). It changes no label, gate or class. It covers four things.

**(a) A beside family-change reading FC_R1** (RF1_FULL vs RF_FULL), frozen before the R1 fit:
- inputs: `NR.comparison` on the R1-bank entry, `NR._primary` cost d = cost(RF1_FULL) − cost(RF_FULL), and the 8 secondary metrics;
- spend states: `R1_SPENDS_NO_MORE`, `R1_SPENDS_MORE_UNRESOLVED`, `R1_SPENDS_MORE_RESOLVED`;
- labels, prefixed `ADAPTIVE_R1__`: `FAMILY_CHANGE_RESOLVED`, `_WITH_HIGHER_SPEND`, `_RECIPIENT_GATE_FAILED`, `_POINT_ONLY`, `_NOT_POSITIVE`;
- it never gates and is never an attribution to SOFTBIN: R1 also changes lambda and k*.

**(b) REQUIRED for the R1 assembly: bitwise-equal points.** If RF1_TASK or RF1_RAWJ_HI is bitwise equal to a v1 point, keep the R1 ids in `grid`, `costs_vs` and `nolookup` and disclose the identity in the plan. Never put it in `alias_of`, which maps a path's ids to START, CONT or an earlier id of the same reading.

**(c) REQUIRED: two defects found in v1, to fix in v1's assembly and R1's.** The learning curve's own grid validity, and the falsifier detail (items 2 and 3 below).

**(d) Wording.** R1 was designed after v1's MODEL_SELECT reader winners were seen (F3). Its SOFTBIN family targets the recipes that won there (CATI1, ET25, the trees). Any R1 fall in those recipes' AUC on the same SCORE households is adaptively targeted: not information removal, and not protection against other readers. Report R1 beside v1, with K = 2 looks and no correction. A positive R1 result would need new households before it could be confirmed.

---

## (2) Claims review wave 1 of the registered reading RFA_v1

**Every expected label follows from the record under the frozen rules:**

| result | label | why it follows |
|---|---|---|
| D1 | TASK_FEASIBLE | all 28 conditions pass at the point; separately: CONFIDENCE_UNRESOLVED (v1 occupation-vs-Ucal NLL excess 0.00737 ± 0.00133 against 0.010; v1 men's guard 0.00442 ± 0.00074 against 0.005), and not reserve-qualified (binding: v1 men's guard) |
| D2 | PARTIAL_START_ONLY | START +0.0127 (se_agg 0.0059), lower bound +0.0008, both recipient gates pass. CONT +0.0086 (0.0092) point only. STOPFIT at Q* RF_STOP_HI +0.0028 (0.0045). ALTCRIT at Q* RF_ALTC_HI +0.0005 (0.0010; the floor binds). RAWJ at Q* RF_RAWJ_HI −0.0016 (0.0084), income recipient gate failed, targeting failed. |
| D3 | NOT_SUPERIOR | passes NEURAL_RAW, W, TR100, TT100. TWJ is resolved (+0.0266) but fails a recipient gate. B2TWIN, H_LJ and SUCC are point only (+0.011 / +0.017 / +0.004) with failed gates. RETIRED_B3CTL −0.004. All nine are READER_BUDGET_COMPARABLE. |
| D4 | NOT_MET | target screen not met against NEURAL_RAW (+0.0127) and H_LJ (+0.0168, not resolved even above 0) |
| LC | LEARNABILITY_LIMITATION | point limitations against RF_TASK (Δ −0.0011, se 0.0111), RF_ALTC_HI (−0.0001, 0.0037) and H_LJ (−0.0045, 0.0132); none resolved. No shrinkage against START (+0.0038), RF_STOP_HI (+0.0029) or RF_RAWJ_HI (+0.0027). |
| falsifiers | F2, F3, F4 triggered; F0, F1, F5, F6 not | F0 orthogonal share 0.102 / 0.112 ≥ 0.05 (active) |
| outcome | BOUNDED_NEGATIVE_EXPERIMENT | partial PARTIAL_START_ONLY |

Development bounds (upper = mean + 2·se_agg): STOPFIT 0.0117; ALTCRIT 0.0025, set by the SE floor; RAWJ 0.0153; CONT 0.0270.

Flags as recorded:
- CAND_OPERATING_POINT_DIFFERS_BY_VERSION (k* = S20 / S70)
- INNER_EPISODE_BUDGET_AT_LEAST_EVALUATION_FIT (4,096 ≥ FIT ≈ 4,002)
- CONTROL_GRID_COARSE_STOPFIT / _ALTCRIT / _RAWJ (version 0 coarse below, RAWJ coarse both; version 1 fine)
- LO_NOT_SAVED_SPEC_BUDGET_ALTCRIT / _RAWJ
- READER_SENSITIVE_ENHANCED_RF_RAWJ_HI
- POINT_FEASIBLE_NOT_CONFIDENCE_RESOLVED and POINT_FEASIBLE_NOT_RESERVE_QUALIFIED

D2R reads NOT_RUN for now.

### Record findings

1. **The labels follow (verified).** D2's numbers reproduce the rules' arithmetic. On MODEL_SELECT each Q* spent 0.0003–0.0005 nats more total NLL than CAND (STOPFIT −0.00039 ± 0.00009; ALTCRIT −0.00041 ± 0.00010; RAWJ −0.00028 ± 0.00056). The residual mismatch is small and favours the controls.
2. **REQUIRED: the learning curve's own grid validity.** `rfclaims.run` passes the main bank's validity as `learning_curve.grid_valid`. RULES.md §9 requires the half-budget (`@0.5`) cells' own `grid_validity` (same seeds, bootstrap seeds, B = 400 and minimum set).
   - Make sure `CL.seed_checks` actually covers the `@0.5` cells: `NR.parse_cell` ignores `@` names.
   - Compute it and rebuild the record. If it is effective, LC and F4 stand; if not, LC reads NOT_COMPUTED_GRID_INVALID and F4 is not assessable.
   - Until then, any LC or F4 sentence must say "pending the half-budget grid-validity check".
3. **REQUIRED: save `falsifier_detail`.** `build()` computes it but DECISIONS.json does not save it.
   - Without it, F3's trigger cannot be checked from the record: the per-recipe gains against RF_STOP_HI, the tree-table mean and the LOG_LR gain.
   - Note that the R1 card quotes per-recipe falls against START, which is a different contrast from F3's.
   - Save it beside DECISIONS.json and cite those numbers for F3.
4. **ADVISORY: falsifier operationalisation, to disclose.**
   - F3 implements the card's "CATI" as CATI1 only.
   - F5 implements only the "mostly negative held-out orientation" disjunct. The "unchanged oriented fresh AUC" disjunct is not computed, but it is not met here: the START comparison is resolved.
   - F6's 0.02 / 0.01 thresholds are the coordinator's operationalisation, frozen at 21:56Z before any MODEL_SELECT read.
5. **ADVISORY: report version means and the failing views.**
   - The D2 and D3 records keep only aggregate means and SEs. Report the version means (from the grid aggregates) for START and the references.
   - State which single view fails each D3 and D4 recipient gate. Because the design is occupation-only, the income view stays NEURAL_RAW's bitwise, so the failures against B2TWIN, H_LJ, RETIRED_B3CTL, SUCC and TWJ are probably on the income view. Verify that from the grid before saying it.

### Wording to use

- **D2:** "the candidate's linked-pair recovery by the declared reader procedure was resolved below the unchanged start NEURAL_RAW (+0.0127, se 0.0059; both recipient gates), and was not separated from any mechanism-specific control: continued task training (+0.0086, 0.0092), the stop-fit control at matched spend (+0.0028, 0.0045), the alternating critic (+0.0005, 0.0010 at the SE floor) or NEURAL_RAW's own recipe (−0.0016, 0.0084)."
- Then: "PARTIAL_START_ONLY is not a component gain; the fit path (the ingredient tested) is not isolated; F2, the registered falsifier, is triggered" (registered prediction: tie, 0.75).
- **Bounds:** "development bounds: on this grid a gain beyond the stop-fit control larger than about 0.012 is not supported (beyond the alternating critic about 0.0025, set by the SE floor); conditional on the fitted releases, for this procedure on reused people; not population bounds and not adjusted for the adaptive search."
- **Learning curve:** "point estimates of the benefit against RF_TASK, RF_ALTC_HI and H_LJ were smaller with readers fitted on the full FIT than on its half (none resolved); against START the benefit was larger at the full budget; the inner episodes used about the evaluation's own FIT budget (4,096 vs about 4,002), the setting most favourable to this mechanism; budgets above FIT were not tested."
- **D1:** "point-feasible in both versions; not confidence-resolved; not reserve-qualified."
- **D3 / D4:** "NOT_SUPERIOR (passes NEURAL_RAW, W, TR100, TT100; fails B2TWIN, H_LJ, RETIRED_B3CTL, SUCC, TWJ); the material target is not met."
- **Disclosures:**
  - ALTCRIT is not a near alias (lag 0.11 / 0.21), yet its recovery at its bracket point is within 0.0005 of CAND's;
  - CAND moved little in version 0 (k* = S20; MODEL_FIT out-of-fold spend 0.0008 nats);
  - the F0 gate is a SEX-based MODEL_FIT fit gate only.
- **Every headline carries:** "D2R NOT_RUN (second split pending)", the learnability caveat, "development evidence on adaptively reused MODEL_SELECT people", "D6 NOT_CLAIMED", "D5 not applicable (no local twin)".

### Wording to avoid

- "the objective reduced / protected SEX recovery" as an attribution to the mechanism;
- "privacy guarantee", or "information removed";
- "equal to", "no effect" or "tie proven": NOT_SEPARATED is the default reading, not equality;
- "beats four references" or "competitive";
- "robust" before D2R;
- "the benefit shrinks with more auxiliary data" stated as established (it is point only);
- "close to the material target";
- any claim of improvement from W, TR100 or TT100 alone.
