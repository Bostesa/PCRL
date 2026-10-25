# Decision rules of the direct-recovery frontier campaign (DRF1; statistical reviewer; schema DRF1-rules-v1)

**Status.** Writing started 2026-10-25T02:38:01Z (`date -u`); this file was written from 03:00:53Z; the completion time is in
section 14. Written before any outcome of this campaign existed: I read no label, ran no real-data computation, fitted nothing and
called no loader that reads labels. Every number I computed came from invented data (`python3 -m drf.test_rules`, an in-memory
mutation check and a normal-approximation power calculation run from the worktree, one numerical thread).
- Machine-readable constants: `RULES.json` (schema `DRF1-rules-v1`). Implementation: `drf/rules.py` (pure functions; imports
  `nrm.rules` unchanged as NR; `load_rules()` refuses a RULES.json that disagrees with the module, a changed NRM1 RULES.json or a
  changed `nrm/rules.py`). Invented-data tests: `python3 -m drf.test_rules` (prints JSON; exit 0 iff all pass).
- Protocol review: `REPORT_PREREGISTRATION.md` (same directory), numbered findings marked REQUIRED or ADVISORY. The harness
  refused my direct write of that report file (subagents return reports as text), so it is returned to the coordinator in the
  hand-back message, to be saved verbatim with attribution (as the SAR1 report was).
- **Inputs read** (code and records only): `roles/coordinator/DESIGN_BRIEF.md`; the SAR1 `RULES.md` / `RULES.json`; `sar/rules.py`
  and `sar/test_rules.py` (through their mechanical copies in `drf/`, confirmed identical up to the renames); `nrm/rules.py`;
  `nrm/test_rules.py` (invented-data helpers); the list of `drf/` modules that call the rule API; for consistency only, sections
  5-8 of the coordinator's draft `PROTOCOL.md` (about 03:05Z) and the label-free `READER_SPLITS.json` seed / tag fields. I read no
  investigator file, no card, nothing under `.private/`, no label file, nothing of MODEL_SELECT.

**What every rule here is.** A development rule on previously and adaptively used records (MODEL_FIT and MODEL_SELECT were used by
earlier campaigns, and the DR budgets were set from published MODEL_SELECT aggregates of the twin): nothing here is confirmation.
No rule changes an original condition. G1 gates; the confidence conjunction and the two-SE reserve are reported beside and never
gate; the optimistic diagnostic and the failure stage never gate. No absolute AUC threshold is installed anywhere (in particular
not 0.58, a planning consequence of estimated variability in an earlier report); every criterion is comparator-based. Every
interval is a conditional development summary that does not correct the accumulated adaptive search (section 10).

**Sign conventions** (NR's, unchanged): pair benefit b = comparator AUC - DR AUC (positive favours DR); recipient difference
r = DR AUC - comparator AUC (positive = DR's single view is more recoverable); paired costs loss(DR) - loss(comparator) (negative
favours DR; accuracy enters as a loss).

## 0. Inherited unchanged

From NRM1 (`nrm.rules`, imported unchanged; RULES.json 84f18776..., nrm/rules.py d5df8686..., both checked by `load_rules()`; never
monkeypatched, test `t_nrm_rules_unchanged_not_monkeypatched`):
- **G1** (`point_eligibility`): all 28 original usefulness conditions at the point: per version 12 task clauses (income and
  occupation x U0 and Ucal x accuracy loss <= 0.010, NLL excess <= 0.010 nats, summed Brier excess <= 0.005) and 2 within-SEX
  occupation guards (men's and women's occupation summed Brier excess vs U0 <= 0.005); two versions, 28 in total (not 28 per version).
- The two-SE reserve (`reserve_status`, k = 2, H_LJ floor SE): reported beside, never gating.
- The **aggregate paired screen** (`aggregate`, `benefit_screen`): 8 registered cells = 2 versions x 2 repetitions (R1, R2) x 2
  reader seeds (k0, k1); blocks = version x repetition, block means over reader seeds; se_agg = max(sqrt(se_hh^2 + se_rs^2 +
  se_mc^2), 0.001); resolved iff mean - 2 se_agg > threshold and both version means > threshold; an exact alias is `ALIAS_ZERO`,
  never positive.
- The **recipient condition** (`recipient_screen`): neither single view more recoverable than the comparator's by more than 0.010
  AUC per version at the aggregate point; `comparison` = pair benefit + both recipient gates.
- The **frontier rule** (`frontier_comparison`): bracket Q* = the cheapest path point whose pooled MODEL_SELECT paired total-NLL cost
  is >= DR's; AGG_POSITIVE vs Q*; no secondary metric (the 8 per-task metrics incl. the men's and women's occupation Brier)
  resolved higher vs Q*.
- `robustness`; claims G4 (D3), G5 (D4, the 0.020 material target), G6 (D5); the nine-reference benchmark.
From SAR1 (`sar/rules.py`, adapted, not imported): grid validity with the extra-trees support and few-valued explanations, refit
effectiveness, the reader-budget credit rule, the alias collapse, the non-table winner's version means (reader targeting), the
confidence conjunction, the cheaper-point domination conjunct and the recipient gate at Q*, the D5 collapse rules, D6.

### 0.1 Changes from SAR1 (all registered before any outcome)

1. The candidate is **DR**, one constrained exact-AUC fit per version (no strength path): UM, strength selection, CONT, ALT,
   CALIBRATION bracket data and `strongest_nonzero` are removed.
2. **D2 gates on the two family controls**: the CTRL frontier (origin TWIN) and OBJ_MI (one point, the single-point
   spend-resolution rule, section 4.2). Met label `COMPONENT_GAIN_BEYOND_FAMILY_CONTROLS`.
3. START = SUCC moves out of D2 into the separately reported **family change FC**; **TWIN** is reported beside, never sufficient.
4. New: the **optimistic MODEL_FIT diagnostic** and the **failure-stage classification** (section 5), reported, never gating.
5. D2R: decisive set {Q*, OBJ_MI}; **failed robustness recipient conditions** are carried into the headline as well as sign changes.
6. Grid validity: the **finite-pair extension** of the few-valued explanation (section 9); the SAR1 reading is reported beside.
7. Registration: family `DR`, revisions 0-2 with prospective notes; registered iff the card precedes any MODEL_SELECT outcome; the
   headline is the latest pre-outcome card.
8. Reader grid: offset +1100, offsets 300, 400, ..., 1000 forbidden (160 seeds), bootstrap base 2026102500, split tag
   `DRF1_devgrid_R{r}`.

## 1. The reading

One reading = one DR card (revision 0, 1 or 2): the mapping DR fitted on all MODEL_FIT under the estimated law, per version,
deployed from the six fields; read at that release only (never at an internal iterate, a restart, an optimistic construction or a
fold fit). Family name `DR` only (anything else, revision 3 or more: refused). `cand.alias_of_twin` (exact float64 wire equality
with TWIN in both versions) must be recorded. Records are listed in section 11. Comparator keys: `comparator_ids(rec)` reads the
twin's id from `twin.id` (default `TWIN`; the protocol's `TWB_C64`) and OBJ_MI's from `obj_mi.config_id` (default `OBJ_MI`; the
protocol's `DR_MI`); the candidate's id is `cand.config_id` (the protocol's `DR_AUC`); CTRL points keep their saved ids
(`DR_CTRL_LO`, `DR_CTRL_HI`). SUCC and the references are keyed by their reference names.

## 2. D1 task feasibility (`d1`, `confidence_conjunction`)

- **Gating (G1)**: all 28 conditions at the point at DR: `TASK_FEASIBLE`, `CAND_INFEASIBLE`, `INCOMPLETE`.
- **Confidence conjunction (reported, never gating)** for DR, SUCC, TWIN and OBJ_MI: per condition from its own household-clustered
  SE (no floor): PASS iff value + 2 se <= cap; VIOLATION iff value - 2 se > cap; UNRESOLVED otherwise (neither a pass nor a
  violation). `CONFIDENCE_ALL_PASS` / `CONFIDENCE_VIOLATION` / `CONFIDENCE_UNRESOLVED` / `INCOMPLETE`. These are 28 one-sided
  per-condition bounds, not a simultaneous statement; no multiplicity correction.
- **Reserve (reported, never gating)**: NR's two-SE reserve of each.
- Flags `START_POINT_INELIGIBLE`, `TWIN_POINT_INELIGIBLE`, `OBJ_MI_POINT_INELIGIBLE`: a point-ineligible control stays a control
  (its spend is read by the spend rules), `POINT_FEASIBLE_NOT_CONFIDENCE_RESOLVED`, `POINT_FEASIBLE_NOT_RESERVE_QUALIFIED`.
- **Usefulness failure** may stop the competitive reader stage (`reader_stage` = `STOPPED_USEFULNESS_FAILURE`, allowed only when DR
  fails G1; recorded while DR passes G1 it makes D2 INCOMPLETE): the fitted release is preserved, D2-D5 read
  `NOT_ASSESSED_TASK_INFEASIBLE` (never negative, never protection), the class is `TASK_INFEASIBLE` with flag
  `READER_STAGE_STOPPED_RELEASE_PRESERVED`, and the failure is classified by section 5 (stage 1 or 2).

## 3. Comparators and their roles

| Comparator | What it is | Role |
|---|---|---|
| START = SUCC | the unchanged strong starting release (outside the finite family; in-sample on MODEL_FIT) | family change FC (separate); a D3 reference |
| TWIN | the family origin TWB_C64 (B1 tables; weak task-only twin) | CTRL path origin; reported beside, never sufficient alone |
| CTRL | the SEX-blind task-only coarsening path from TWIN (greedy atom merges by least total-NLL increase); two points saved before any MODEL_SELECT read on MODEL_FIT nested held-out total-NLL spend | D2 gate (frontier rule) |
| OBJ_MI | the previous dependence objective (pi-weighted plug-in MI of the pair and both views) in the same family, identical constraints, moves, data exposure and optimizer budget; one point per version | D2 gate (objective alignment, section 4.2) |
| nine references | NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ, W, TR100, TT100 (U0 / Ucal diagnostics) | D3; D4 against NEURAL_RAW and H_LJ |

A family change (finite family vs SUCC) is reported separately from the objective comparison: FC attributes nothing to the
exact-AUC objective, and D2 never contains SUCC. An attribution against TWIN alone is insufficient (flag
`TWIN_ONLY_ATTRIBUTION_INSUFFICIENT` when DR beats TWIN without D2).

## 4. D2 component gain (`d2`, `obj_spend_state`, `brackets`, `grid_fineness`, `family_change`, `twin_beside`)

### 4.1 Labels in order of precedence

1. `INCOMPLETE` (DR status or utility); 2. `NOT_ASSESSED_EXACT_ALIAS` (DR = TWIN: the optimizer made no move; no protection
experiment); 3. `NOT_ASSESSED_TASK_INFEASIBLE`; 4. `NOT_COMPUTED_GRID_VALIDITY_UNKNOWN` / `NOT_COMPUTED_GRID_INVALID`; 5.
`NOT_ASSESSED_CONTROLS_NOT_RUN` (CTRL or OBJ recorded as not run for budget; recorded, never inferred); 6. `INCOMPLETE` (CTRL
selection data not recorded or not `MODEL_FIT_NESTED`: a re-selection is a POST HOC repair; any needed input missing: never skipped);
7. the conjuncts:
- **(c) CTRL_FRONTIER**: path = [TWIN, the distinct saved CTRL points] (aliases once; an alias of TWIN collapses into it); (c1)
  `NR.frontier_comparison` = `DOMINATES_CONTROL_FRONTIER` (Q* by NR's rule on MODEL_SELECT; AGG_POSITIVE vs Q*; no secondary
  metric resolved higher vs Q*); (c2) the recipient gate vs Q*; (c3) no cheaper path point (TWIN included) resolved less
  recoverable than DR (aggregate upper bound < 0 and both version means < 0);
- **(o) OBJ**: AGG_POSITIVE pair benefit vs OBJ_MI, the recipient gate vs OBJ_MI, no secondary metric resolved higher vs OBJ_MI,
  and OBJ_MI not resolved cheaper than DR (section 4.2);
- **(d) reader targeting**: the non-table winner's point pair benefit > 0 in both version means vs Q* and vs OBJ_MI.
8. `COMPONENT_GAIN_BEYOND_FAMILY_CONTROLS` iff (c), (o) and (d); `READER_TARGETED` iff (c) and (o) pass and (d) fails;
`NOT_SEPARATED` iff neither (c) nor (o) passes; otherwise `PARTIAL_CTRL_FRONTIER_ONLY` (the gain may come from SEX-aware routing in
the finite family rather than the changed objective) or `PARTIAL_OBJ_ONLY` (the gain may be explained by task spending). **No
partial outcome is a claim**; `NOT_SEPARATED` is the default reading, never "equal". Scope sentence of the met label: a resolved
linked-pair benefit with both recipient gates beyond the SEX-blind coarsening path at matched MODEL_SELECT total-NLL spend and beyond
the previous dependence objective with identical constraints at no resolved lower OBJ_MI spend, for the declared development
readers on reused people; not superiority over SUCC or the references (D3), not the family change, not the material target (D4).

### 4.2 OBJ_MI: how a single control point gates, and how its spend mismatch is reported

The inherited frontier rule chooses, among several path points, one that spends at least as much as the candidate. A single
control point cannot be chosen, and DR and OBJ_MI face identical budgets, so both will often stop near the same binding
constraint, where the point rule "OBJ_MI spends at least as much" would pass about half the time on noise alone. The single-point
rule therefore tests whether the spend difference is resolved (the same guard NR applies to secondary metrics). With
d = pooled MODEL_SELECT paired total-NLL cost(DR) - cost(OBJ_MI) (NR._primary: mean of versions, SE = mean of version SEs):
- `OBJ_MI_SPENDS_AT_LEAST_AS_MUCH` (d <= 0): spend-matched or favourable;
- `OBJ_MI_SPENDS_LESS_UNRESOLVED` (0 < d and d - 2 SE <= 0): credited, with flag `OBJ_MI_SPENDS_LESS_UNRESOLVED` and a statement;
- `OBJ_MI_SPENDS_LESS_RESOLVED` (d - 2 SE > 0): the OBJ comparison reads `NOT_SEPARATED`, reason `NOT_SEPARATED_FROM_SPEND`: at
  identical constraints DR used more of the budget; **the exact-AUC objective is not credited for a spend change**; the benefit is
  reported as a budget-use difference (`passed_if_spend_matched`), never as objective alignment.
Always reported: d (pooled, SE, by version), all 8 secondary paired differences vs OBJ_MI, both MODEL_FIT nested spends (plan), the
strict NRM1 one-point frontier reading beside (`inherited_point_rule_beside`), and `CAND_ALIAS_OF_OBJ_MI` (reason
`DR_EQUALS_OBJ_MI`) when the two objectives reach the same release. The residual mismatch never gates. The OBJ comparison isolates
the objective only together with (c): (c) excludes SEX-blind spending as the explanation at DR's own spend; (o) excludes the old
dependence objective at no resolved lower spend; neither excludes a better optimizer for the old objective.

### 4.3 The CTRL frontier, no bracket, residual mismatch

- The two CTRL points are saved before any MODEL_SELECT read on MODEL_FIT nested held-out paired total-NLL spend over TWIN: the
  largest path point with spend <= DR's (TWIN counts with spend 0) and the smallest with spend >= DR's + 2 SE (SE = the
  household-clustered SE, on the same nested held-out data, of that point's paired difference against DR). On MODEL_SELECT, NR's
  rule picks Q* on the path. CALIBRATION is not read by this campaign; MODEL_SELECT is never selection data.
- **No bracket** (every path point spends less MODEL_SELECT total NLL than DR): (c) reads `NOT_SEPARATED`, reason `NO_BRACKET`,
  flag `NO_BRACKET_CTRL`: lower recovery cannot be separated from task spending; a limit of the control path, never evidence of
  equality or of no effect.
- Bracket at TWIN (DR spends no more than TWIN): flag `CTRL_BRACKET_IS_ORIGIN`; an empty saved list: `CTRL_AT_ORIGIN`.
- **Residual mismatch** (reported, never gating): cost(DR) - cost(Q*) (pooled, SE, by version), the smallest overspend vs a cheaper
  point, NR's interpolation. Flags `RECIPIENT_GATE_FAILED_VS_<id>`, `DOMINATED_BY_CHEAPER_<point>`, `READER_TARGETED_<c>`,
  `READER_SENSITIVE_ENHANCED_<point>` (reported only).

### 4.4 Fineness of the saved CTRL points (`grid_fineness`; reported, never gating)

Per version on MODEL_FIT_NESTED with s = DR's nested spend over TWIN and SE from the same data: `FINE` iff lower >= 0.75 s (and <= s)
and upper >= s + 2 SE and <= s + 2 SE + 0.25 s; `COARSE_BELOW` / `COARSE_ABOVE` / `COARSE_BOTH`; `CONTROL_PATH_MAY_NOT_BRACKET`
(largest path spend < s + 2 SE: extend the path before reading); `SAVED_POINT_RULE_NOT_FOLLOWED` (with the full path recorded, the
saved points are not the largest / smallest qualifying points); `NOT_APPLICABLE_CAND_NOT_ABOVE_TWIN` (s <= 0);
`SELECTION_DATA_NOT_ALLOWED` (anything but MODEL_FIT_NESTED); `INCOMPLETE`; `NOT_GIVEN`. Computed before any MODEL_SELECT read.

### 4.5 FC and TWIN (reported apart)

- **FC** (`family_change`): DR vs SUCC: `FAMILY_CHANGE_BELOW_START_RESOLVED` iff AGG_POSITIVE pair benefit and both recipient gates,
  else `FAMILY_CHANGE_NOT_RESOLVED` (D2's gate labels otherwise); the paired spend vs SUCC is reported.
- **TWIN** (`twin_beside`): DR vs TWIN: `BELOW_TWIN_RESOLVED` / `NOT_RESOLVED_VS_TWIN`; never sufficient alone.

### 4.6 Power and multiplicity (normal approximation; invented settings)

One comparison resolves a true benefit delta with probability Phi(delta / se_agg - 2): 50% at 2 se_agg, 80% at 2.84 se_agg. With
se_agg 0.008 (CTRL at Q*) and 0.007 (OBJ_MI), correlation 0.5, D2 resolves a true benefit of 0.010 / 0.015 / 0.020 / 0.025 / 0.030
against both with probability about 0.12 / 0.33 / 0.61 / 0.84 / 0.95; when OBJ_MI reproduces half of the gain, 0.25 / 0.40 / 0.55 at
0.020 / 0.025 / 0.030 (targeting (d) lowers both slightly). A D2 look is an intersection-union conjunction of two comparisons: its
false-pass rate is at most that of its weaker member (one AGG_POSITIVE comparison passes a true null in about 0.07-2% of replicates
under six moderate NRM1 known laws and about 8% when one version or split dominates, NRM1 RULES.md 5.5); with K D2 looks the chance
that at least one passes by chance is at most K times the per-look rate. No multiplicity correction is applied or claimed.

## 5. The optimistic diagnostic and the failure stage (`useful_optimistic`, `failure_stage`; reported, never gating)

### 5.1 Useful optimistic point (prospective, comparator-based, MODEL_FIT only)

OPT_S (fine cells split by true SEX; illegal at release) and OPT_L (legal cells, true-SEX empirical law, in-sample) are evaluated on
MODEL_FIT in two modes: `in_sample` (exact optimal-reader AUC of the fitted finite law) and `nested` (the 5-fold household nested
held-out values). The SEX-blind CTRL path (TWIN first, spend 0) is evaluated on the same law and mode. A construction X is
**USEFUL** in a mode iff in both versions:
1. X is within all 8 DR budgets (its in-sample MODEL_FIT spends; recorded `within_budget`);
2. against B = the cheapest CTRL path point whose MODEL_FIT total-NLL spend over the twin (same mode) is >= X's (NR's bracket
   choice; ties in path order): AUC*_pair(B) - AUC*_pair(X) >= 0.020 (the material target's size, used as a comparator margin);
3. neither single view's AUC* exceeds B's by more than 0.010 (the recipient allowance).
`USEFUL_OPTIMISTIC_POINT` iff OPT_S or OPT_L is useful in the in-sample mode; `NO_USEFUL_OPTIMISTIC_POINT` iff both are evaluable
there (complete, same law, bracketed) and neither is useful; otherwise `UNDETERMINED` (a construction beyond every CTRL path spend is
`NOT_EVALUABLE_NO_BRACKET`; a comparator on another law is incomplete). The nested mode is reported beside (`held_out_survives`).
Exact plug-in values, no SE: a diagnostic, not inference. A common shift of every AUC changes no label (test). No preliminary
significance, optimistic forecast or novelty gate may prevent an exploratory fit: DR, OBJ_MI and CTRL are fitted and read whatever
this diagnostic says. Neither success nor failure proves population feasibility or impossibility.

### 5.2 Failure stage

Legal stage from DR's decisions: **L2** (DR fails G1 on MODEL_SELECT, or DR = TWIN), **L3** (G1 kept, neither D2 nor D3 met), **L4**
(D2 met, D3 not), **L5** (D3 met). Stage:

| Stage | Condition | Meaning (brief) |
|---|---|---|
| `STAGE_1_NO_USEFUL_OPTIMISTIC_POINT` | L2 or L3, and NO_USEFUL_OPTIMISTIC_POINT | failure within the chosen finite model, estimated law and optimizer; not a general impossibility |
| `STAGE_2_OPTIMISTIC_SUCCEEDS_LEGAL_MAPPING_FAILS` | L2 and USEFUL_OPTIMISTIC_POINT | a possible estimation, representation, optimization or generalization limitation |
| `STAGE_3_KEEPS_UTILITY_TIES_CONTROLS` | L3 and USEFUL_OPTIMISTIC_POINT | no demonstrated mechanism advantage |
| `STAGE_4_BEATS_CONTROLS_MISSES_STRONG_REFERENCES` | L4 | at most a scoped component result if robustness supports it: flag `SCOPED_COMPONENT_RESULT_ROBUSTNESS_SUPPORTED` iff D2R `ROBUSTNESS_CONFIRMED` with no sign change and no robustness recipient failure, else `COMPONENT_RESULT_NOT_ROBUSTNESS_SUPPORTED` |
| `STAGE_5_COMPETITIVE_CONDITIONS_PASS` | L5 (G1 + every eligible reference, recipient gates) | exploratory development evidence requiring fresh confirmation (flag `D3_WITHOUT_D2` if D2 is not met) |
| `UNDETERMINED_OPTIMISTIC_DIAGNOSTIC` | L2 or L3, and UNDETERMINED | stage 1 cannot be told from stage 2 / 3 |
| `INCOMPLETE` | D1, D2 or D3 undecided | |

The legal stage is always reported with the stage. At L4 / L5 a failed optimistic diagnostic is flagged
(`NO_USEFUL_OPTIMISTIC_POINT_BUT_LEGAL_MAPPING_PROGRESSED`) and changes nothing: the diagnostic only separates stage 1 from 2 and 3.
"Ties its controls" means D2 not met (partials included; the stage-3 statement lists what passed).

## 6. D3, D4, D5, D6 (`_d3`, `_d4`, `_d5`, `reference_credit`)

Read at DR whenever DR is fitted and G1-feasible and the grid is valid (also when D2 is INCOMPLETE or controls were not run);
otherwise they carry D2's gate label or INCOMPLETE. **D3** = NRM1 G4: per point-eligible reference, AGG_POSITIVE pair benefit with
the recipient gates; `SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES` only if every one passes; passed and failed listed. **D4** = NRM1 G5:
`MATERIAL_0020_RESOLVED` iff the 0.020 target screen is resolved against both NEURAL_RAW and H_LJ with the recipient gates; smaller
gains keep their level (`MATERIAL_0020_POINT_ONLY`, `POSITIVE_BELOW_MATERIAL_TARGET`, `NOT_MET`); never weakened. Both carry the
reader-budget credit rule (a READER_BUDGET_ASYMMETRIC reference is credited only if the non-table winner's point benefit is
positive in both versions; missing input is INCOMPLETE, never "not beaten"). **D5** = NRM1 G6 only when a card claims joint design
with the local twin and both sequential orders built, else `NOT_APPLICABLE_NOT_JOINT`. **D6** = `NOT_CLAIMED` (this design overlaps
privacy funnels and the earlier SCM / FRC finite routing); never inferred from performance.

## 7. Outcome class, the three outcomes, registration and the campaign summary (`decisions`, `adaptive_tag`, `campaign_summary`)

- **Outcome class**, first applicable: `INCOMPLETE` (D1 incomplete); `NO_OUTPUT_CHANGE` (DR = TWIN); `TASK_INFEASIBLE` (DR fails G1);
  `INCOMPLETE` (D2 incomplete or not computed); `PROTECTION_NOT_ASSESSED` (a family control not run); `INCOMPLETE` (D3 undecided);
  `METHOD_CANDIDATE` (D2 and D3 met); `COMPONENT_GAIN_NOT_COMPETITIVE` (D2 met, D3 not); `REFERENCE_ADVANTAGE_NOT_ATTRIBUTED` (D3
  met, D2 not); `FEASIBLE_NO_RELIABLE_GAIN`. D4, D5, FC, TWIN, D2R and the stage never change the class. Flags `D3_WITHOUT_D2`,
  `D4_WITHOUT_D2`, `TWIN_ONLY_ATTRIBUTION_INSUFFICIENT`, `D2R_SIGN_CHANGE_CARRIED_INTO_HEADLINE`,
  `D2R_RECIPIENT_FAILURE_CARRIED_INTO_HEADLINE`, `REVISION_NOTE_MISSING` and the D1 / D2 / D5 / stage flags.
- **Three separately reported outcomes** (`outcomes`, one statement quoted whole): (1) component gain = D2 with D2R's label; (2)
  strong references = D3 with passed / failed lists; (3) material target = D4; beside them FC, TWIN and the failure stage, and the D2R
  headline caveat when present. None implies another.
- **Registration.** A reading is registered iff its card was committed before any MODEL_SELECT outcome of this campaign (any
  MODEL_SELECT-derived number of any configuration: utility, reader AUC or paired cost of DR, OBJ_MI, a CTRL point or a re-read
  reference; the coordinator records the time of the first such read). Revision 0, or a revision 1 / 2 whose prospective note names
  the observed failure, the actual changed operation and a MODEL_FIT or SYNTHETIC trigger. At most two evidence-driven revisions
  (revision 3+ refused). Not registered: `ADAPTIVE_R0_LATE_CARD`; `ADAPTIVE_R<n>` (committed after an outcome, or triggered by a
  MODEL_SELECT outcome: adaptive as a whole); `ADAPTIVE_R<n>_NO_NOTE` (no complete note). Every label of a non-registered reading
  (D1, D2, FC, TWIN, D3-D6, the stage, the class) is prefixed `<tag>__`.
- **Campaign summary**: the headline = the registered reading with the highest revision (the latest pre-outcome card); earlier
  registered readings are listed as superseded (`MULTIPLE_REGISTERED_READINGS`); every other reading with its tag; K = the number
  of D2 looks; the readings with D2, D3, D4 met; flag `D2_MET_ONLY_IN_NON_REGISTERED_LOOKS`; and the statement every campaign-level
  sentence carries: "D2 was met in m0 of k0 registered headline readings and in m1 of k1 other readings (K D2 looks in total); D3 was
  met in .. and D4 in .. of n readings; one AGG_POSITIVE comparison passes a true null in about 0.07-2% ... no multiplicity
  correction is applied or claimed."

## 8. Robustness D2R and refit effectiveness (`d2r`, `refit_effectiveness`)

- `refit_effectiveness`: at least one of R3 / R4 run (flag `ONE_ROBUSTNESS_SPLIT_ONLY`), assignment hashes distinct from each other
  and from R1 / R2, every SCORE overlap involving them recorded, finite and in [0, 1), grid validity on all four reader seeds (k0-k3)
  and every repetition run; otherwise `REFIT_NOT_GENUINELY_EFFECTIVE` and the failed part's cells count once.
- **D2R** (descriptive; never changes D2): read at DR with the registered decisive set {the registered CTRL bracket Q*, OBJ_MI}
  (recorded in `comparators`; never re-chosen on robustness cells; no retuning of DR, the brackets or any selection). Per decisive
  comparator `ROBUSTNESS_CONFIRMED` iff `NR.robustness` confirms (AGG_POSITIVE on the robustness cells and on all cells), both
  single-view recipient gates hold on the robustness cells (`NR.recipient_screen(..., cells='robustness')`), the registered cells
  are present and the refit is `REFIT_GENUINELY_EFFECTIVE`. Beside: SUCC, TWIN, the references present (at least NEURAL_RAW and
  H_LJ). **A sign change** between the registered-cell and the robustness-cell aggregate mean, and **a failed robustness recipient
  gate**, on any comparator, decisive or beside, is flagged and its `headline_caveat` appears in the headline. R3 / R4 are new
  household splits of the same people: robustness to classifier refitting, never replication.

## 9. Reader-grid validity of the registered DRF1 grid (`grid_validity`; prospective)

Reader seeds S<v>k<k> recorded as distinct integers equal to MODEL_SEED_BASE + 1100 + 10 v + k (2026102305-2026102308,
2026102315-2026102318), none equal to MODEL_SEED_BASE + {300, 400, ..., 1000} + 10 v + k (k = 0..9; 160 seeds; +1000 is SAR1's
realized grid); bootstrap seeds S<v><R> = 2026102500 + 100 v + repetition index (disclosed: the v = 0 seeds equal SAR1's v = 1 seeds,
applied there to other SCORE household sets; harmless). Split tag `DRF1_devgrid_R{r}`. **Minimum set** = DR, TWIN, the saved CTRL
points, OBJ_MI and SUCC (for the refit: the D2R release and its decisive comparators); references off it. Per seed pair, every
seed-dependent recipe is checked and every seed-dependent equality explained:
- off the minimum set by NRM1 A-SC1 (iii) (another recipe differs on the same wire / view / block; the same recipe differs on
  another view of the same wire; the distinct count recorded);
- on the minimum set by the inherited extra-trees support explanation (ET5 / ET25, (iii) (a)-(c), distinct inputs <= 1600 / 320);
- anywhere by the **few-valued explanation** (SAR1, unchanged): the view's recorded distinct-input count is at most 320
  (= 8001 // 25: at least 25 MODEL_SELECT people per value on average) and the same recipe differs on another wire or view in the
  same version, repetition and seed pair. Every income (<= 16 values) and occupation (<= 64) view of a finite-family release is
  within it by construction;
- anywhere by the **finite-pair extension** (DRF1): a pair view of a finite release can exceed 320 (at most 16 x 64 = 1024 distinct
  vectors). A pair-view equality with 320 < n is explained iff the same wire's recorded income and occupation counts in that version
  are both <= 320 and n <= their product <= 1024, **and** the same recipe differs on the **pair view of another wire** in the same
  version, repetition and seed pair (the pair-view code path demonstrably receives the seed). Flag `FEW_VALUED_PAIR_ABOVE_320`. With
  more than 320 values there are fewer than 25 people per value on average, so the count alone does not make seed invariance likely:
  this is a necessary condition of the mechanism, not a sufficient one, and it is not extended to single views or to wires whose
  single views are many-valued.
Explained equalities are counted once (n_eff per wire / view / recipe / seed pair: `n_eff_seed_cells_if_this_recipe_wins`) and never
invalidate the comparison; se_agg is unaffected (se_rs uses block means over reader seeds, so an explained duplicate reduces n_eff
without biasing the spread). Any other seed-dependent equality invalidates the grid. The strict NRM1 reading and the SAR1 reading
(320 on every view) are reported beside. Claims need `grid_valid` recorded and exactly True; otherwise D2-D5, FC, TWIN and D2R read
`NOT_COMPUTED_GRID_*`; any later repair is POST HOC. Distinct-input counts are counted on the MODEL_SELECT pool (8001 people) per
wire, version and view (report finding R4).

## 10. Estimator justification: what each uncertainty statement covers

Three sources are kept apart, and no statement claims more than its source.
1. **Household sampling uncertainty.** Each utility condition carries a household-clustered SE over MODEL_SELECT households; each
   reader cell carries the SE of a household bootstrap of its SCORE households (tenths 7-9) with the release, the readers and their
   selection held fixed. These describe the sampling of households into MODEL_SELECT / SCORE, conditional on everything fitted;
   they say nothing about refitting the release or the readers. se_hh in the aggregate is the RMS of the cell SEs (an upper bound
   under any cross-cell correlation of the shared households).
2. **Fitting variation.** (a) The two versions are procedure repeats that share every MODEL_FIT row and the MODEL_SELECT people
   (different admitted teacher versions and optimizer seeds): their spread is procedure (seed / teacher) variation only, not
   sampling variation of the training data, and "both versions" is a robustness requirement, never replication. (b) The repetitions
   R1 / R2 (R3 / R4) are household re-splits of the same MODEL_SELECT people into FIT / VALIDATION / SCORE; se_rs, the spread of the
   four version x repetition block means (divisor n_b, deliberately conservative), absorbs version, split, interaction and reader
   parts with essentially one degree of freedom each. (c) **The nested MODEL_FIT estimator** (5 outer household folds, release
   optimization and tables refitted on the outer-training households, spends and exact recovery on the held-out fold, teachers fixed
   at their admitted out-of-fold statistics) estimates the expected held-out behaviour of the *procedure* trained on about 80% of
   MODEL_FIT (16,000 people), averaged over training sets. It does not estimate the conditional performance of the deployed release,
   which is a single fit on all MODEL_FIT (not a fold bag); it is mildly pessimistic for the larger training size; and because the
   teachers are fixed at their admitted out-of-fold statistics (each person's teacher value comes from models trained on other
   MODEL_FIT households, held-out-fold households included for the outer-training people), it is not nested for the teachers. The spread of the five fold values mixes held-out household noise with fit-to-fit variation of
   training sets that overlap by 75%; such fold estimates are positively correlated, so their naive SE (SD / sqrt 5) understates the
   variance of the cross-validation mean, and no unbiased variance estimator exists for K-fold cross-validation in general. The
   spread is therefore never used as an uncertainty estimate for the deployed release or as a confidence interval: nested values
   serve selection (the CTRL bracket points, the budget check, the optimistic diagnostic) and are reported as development
   diagnostics. The SE in "DR's spend + 2 SE" is the household-clustered SE of the pooled held-out paired difference given the fold
   fits: a selection device that covers held-out household noise, not fit variability.
3. **Numerical randomness.** The DRF optimizer is deterministic given its seed and data (seeded sweep order, bounded restarts,
   restart choice by the MODEL_FIT objective with a deterministic tie-break): DR, OBJ_MI and the CTRL path carry no MC term (se_mc =
   0; a recorded se_mc for them is a protocol error, not a correction), and the optimizer's seed dependence is not covered by any SE
   beyond what the two versions show. Reader seeds k0 / k1 (k2 / k3) enter through the block means; seed equalities on few-valued
   views reduce n_eff (section 9).
The exact optimal-reader AUC inside the optimizer is computed under the estimated law (pi-weighted); it is a model-based quantity,
not an estimate of MODEL_SELECT reader AUC, and it is optimistic on MODEL_FIT in-sample. Recovery claims are read only on
MODEL_SELECT with fitted readers. **All intervals are conditional development summaries**: conditional on the released
configurations, on selection made on MODEL_FIT, on budgets set from MODEL_SELECT aggregates and on the adaptive history of
MODEL_SELECT across earlier campaigns; none corrects the accumulated adaptive search.

## 11. Required records (the coordinator's inputs per reading)

1. `family` = `DR`, `card`, `revision`, `card_committed_before_any_model_select_outcome`, `revision_note` (revision >= 1), the time of
   the first MODEL_SELECT read; `cand` = {status FITTED, config_id, evals, alias_of_twin}; `reader_stage`.
2. `start`, `twin`, `obj_mi` evals; `paths['CTRL']` (saved ids), `alias_of`, `selection_data['CTRL']` = MODEL_FIT_NESTED,
   `controls_not_run` (budget only); `fineness` from `grid_fineness` with the full path recorded.
3. Grid entries (pair and both single views) of DR vs SUCC, TWIN, OBJ_MI, every CTRL point and the nine references; paired costs of
   DR vs TWIN, OBJ_MI, every CTRL point and SUCC (`total|nll` + the 8 secondary metrics); `nolookup` per-cell benefits vs every CTRL
   point, OBJ_MI and the references; `enhanced` (reported); the reader-budget flags.
4. `optimistic`: OPT_S and OPT_L per version and mode (law, spend, AUC* of the three views, in-sample `within_budget`) and the CTRL
   path on the same law and mode (TWIN first).
5. Grid validity inputs (seeds S<v>k<k>, bootstrap seeds S<v><R>, seed checks with `k_pair`, distinct-input counts of all three views
   per wire and version on the MODEL_SELECT pool, the minimum set); for D2R the merged registered + robustness entries with all three
   views, the decisive `comparators`, the split records.
6. `novelty` from a separate review (else NOT_CLAIMED); `joint` only if joint design is claimed.

## 12. Limitations

- Version- or split-dominated laws give about 8% false pass per AGG_POSITIVE comparison (NRM1 5.5); two versions and two registered
  splits rest on one degree of freedom each.
- The OBJ rule credits an unresolved spend mismatch (disclosed); it cannot exclude that a better optimizer for the MI objective would
  close the gap; the CTRL frontier is read on total NLL, and a guard-bound DR may be favoured or disfavoured by that (the secondary
  metrics are the guard).
- The recipient allowance is a point rule: a true single-view worsening of exactly 0.010 passes it about a third of the time.
- The few-valued and finite-pair explanations are necessary conditions of their mechanisms, not sufficient ones.
- The DR budgets come from MODEL_SELECT aggregates of the twin (adaptive, disclosed): DR's G1 pass on MODEL_SELECT is not an
  independent test of the constraint transfer.
- The useful-optimistic margin (0.020 against the SEX-blind path) is a heuristic on exact plug-in values; it separates stage 1 from
  stages 2-3 only and is never evidence of population feasibility.
- All readings use the same adaptively used MODEL_SELECT people; registration status and the look count are disclosed instead of a
  correction.

## 13. Changes requested of the protocol

The numbered findings are in `REPORT_PREREGISTRATION.md`. One of them (R12) concerns the draft `PROTOCOL.md` (read at about
03:05Z for consistency only, sections 5-8): its section 8 describes D2 as a gain "beyond START, the CTRL frontier and OBJ_MI"; these
rules keep START = SUCC out of D2 (the family change FC is reported separately and SUCC is a D3 reference), because the brief
requires the family change to be reported separately from the objective comparison and stage 4 ("beats its controls") refers to
the family controls. Either PROTOCOL section 8 is aligned, or I re-freeze these rules with START gating, before registration.
The protocol's reader seeds, bootstrap rule and split tags agree with section 9 (READER_SPLITS.json checked label-free). The rules above implement every REQUIRED finding that concerns a decision;
the others concern records the coordinator must produce. If the coordinator declines one, the corresponding rule must be changed
(and this file re-frozen) before registration, never after an outcome.

## 14. Completion record

- First freeze 2026-10-25T03:04:59Z; revised before registration and completed 2026-10-25T03:08:00Z (`date -u`): configurable comparator
  ids for the protocol's names (`comparator_ids`, test `t_protocol_ids`) and the PROTOCOL alignment note (section 13). Started
  2026-10-25T02:38:01Z. The hand-back message records the SHA-256 of
  RULES.json, drf/rules.py, drf/test_rules.py and this file at completion (this file cannot contain its own hash).
- `python3 -m drf.test_rules`: 24 of 24 PASS, exit 0 (invented data only; about 0.30 CPU-s per run; one numerical thread). An
  in-memory mutation check (37 single-point seeded bugs of `drf/rules.py`, among them: the OBJ resolved-spend rule dropped or
  replaced by the point rule, OBJ not gating, TWIN or FC made part of D2, the CTRL origin set to SUCC, a missing bracket passing,
  cheaper-point domination or targeting ignored, the finite-pair extension without its product bound, its 1024 cap, its single-view
  check or its pair-view seed evidence, the few-valued explanation without seed evidence, offset 1000 not forbidden, an absolute
  0.58 AUC threshold in the optimistic diagnostic, the diagnostic ignoring the budget or the single views or read on the nested
  mode, stage 1 / 3 swapped, the diagnostic gating stage 4, robustness recipient failures or beside-comparator sign changes ignored,
  a MODEL_SELECT-triggered revision registered, a revision without a note registered, revision 3 accepted, the headline taken from
  the lowest revision, a reader-stage stop accepted while G1 passes, D3-without-D2 renamed METHOD_CANDIDATE, the confidence
  conjunction made gating, MODEL_SELECT bracket data accepted, DR = TWIN not gated, the OBJ_MI id hard-coded) was caught 37 of 37
  times; no repository file was changed by it.
- Inherited files verified unchanged: NRM1 RULES.json 84f18776..., nrm/rules.py d5df8686... (load_rules checks both).
- No label was read and no real-data computation was run.
