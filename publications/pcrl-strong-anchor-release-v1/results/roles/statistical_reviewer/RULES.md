# Decision rules of the strong-anchor release campaign (SAR1; statistical reviewer; schema SAR1-rules-v1)

**Status.** Writing started 2026-10-24T04:28:04Z (`date -u`); this file was written from 04:53Z; the completion time is in section 13.
Written before any outcome of this campaign existed: I read no label, ran no real-data computation, fitted nothing and called no
loader that reads labels. Every number I computed came from invented data (`python3 -m sar.test_rules`, an in-memory mutation check
and a normal-approximation power calculation in my scratch directory).
- Machine-readable constants: `RULES.json` (schema `SAR1-rules-v1`). Implementation: `sar/rules.py` (pure functions; imports
  `nrm.rules` unchanged as NR; `load_rules()` refuses a RULES.json that disagrees with the module, a changed NRM1 RULES.json or a
  changed `nrm/rules.py`). Invented-data tests: `python3 -m sar.test_rules` (prints JSON; exit 0 iff all pass).
- Protocol review: `REPORT_PREREGISTRATION.md` (same directory), numbered findings marked REQUIRED or ADVISORY. The harness refused
  my direct write of that report file (subagents return reports as text), so it was returned to the coordinator in the hand-back
  message, to be saved verbatim with attribution (as the IPR1, DDR1 and FRC1 reports were).
- **Inputs read** (code and records only): `roles/coordinator/REVIEWER_BRIEF.md`, `roles/coordinator/DESIGN_BRIEF.md`,
  `anchors/ANCHOR_SELECTION.md`, `anchors/ANCHOR_TABLE.md`, the coordinator's draft `PROTOCOL.md` (read at 04:31Z, SHA-256
  ee5a2a15...; changed at 04:32Z to db87f8b8..., one byte shorter, sections 5-8 unchanged on re-check), the label-free
  `READER_SPLITS.json`; the IPR1 `RULES.md` / `RULES.json`; `ipr/rules.py` via its mechanical copy `sar/rules.py`; `nrm/rules.py`;
  the NRM1 `RULES.md` section 5.5; `ohn/rules.py` (conditions, reserve); `sar/evaluate.py` (`paired_costs`) and `sar/devreaders.py`
  (`paired`, seed constants). I read no investigator file, no card, nothing under `.private/`, no label file, nothing of MODEL_SELECT.

**What every rule here is.** A development rule on previously and adaptively used records (MODEL_FIT and MODEL_SELECT were used by
earlier campaigns, and both anchors and their binding conditions were identified on MODEL_SELECT records): nothing here is
confirmation. No rule changes an original limit (PROTOCOL 5). Point eligibility (G1) gates; the confidence conjunction and the
two-SE reserve are reported separately and never gate. Every uncertainty statement is asymptotic and model-based, and no interval
corrects the adaptive search. Versions 0 and 1 share every training row and the MODEL_SELECT people: "both versions" is a
procedure-repeat requirement, never replication.

**Sign conventions** (NR's, unchanged): pair benefit b = comparator AUC - candidate AUC (positive favours the candidate);
recipient difference r = candidate AUC - comparator AUC (positive = the candidate's single view is more recoverable); paired costs
loss(candidate) - loss(comparator) (negative favours the candidate; accuracy enters as a loss). A condition difference CAND - START
is the paired cost on the matching metric: the reference (U0 or Ucal) cancels on the same people, so `guard|sex0` (men) ->
`occupation|men_brier`, `guard|sex1` (women) -> `occupation|women_brier`, `<task>|<ref>|accuracy_loss / nll_excess / brier_excess`
-> `<task>|acc_loss / nll / brier` (`condition_metric`).

## 0. Inherited unchanged

From NRM1 (`nrm.rules`, imported unchanged; RULES.json 84f18776..., nrm/rules.py d5df8686..., both checked by `load_rules()`;
never monkeypatched, test `t_nrm_rules_unchanged_not_monkeypatched`): G1 = 24 clauses + 4 guards at the point (`point_eligibility`);
the reserve (`reserve_status`, k = 2, H_LJ floor SE, MC SE in quadrature); randomized releases (`mc_cell`, se_mc in se_agg and in
paired costs; `MC_SE_NOT_ESTIMABLE` travels with the reading); the aggregate paired screen (8 registered cells, blocks = version x
repetition, block means over reader seeds, se_agg = max(sqrt(se_hh^2 + se_rs^2 + se_mc^2), 0.001), resolved iff mean - 2 se_agg >
threshold and both version means > threshold; `ALIAS_ZERO` never positive); `recipient_screen` (gate r_v <= 0.010 per version at
the aggregate point); `comparison`; the frontier rule (`frontier_comparison`: bracket Q* = the cheapest path point whose pooled
MODEL_SELECT paired total-NLL cost is >= the candidate's; AGG_POSITIVE vs Q*; no secondary metric resolved higher vs Q*);
`robustness`; claims G4 / G5 / G6 and the nine-reference benchmark. From IPR1 (`ipr/rules.py`, copied, not imported): grid
validity with the prospective extra-trees explanation, refit effectiveness, the reader-budget credit rule, the alias collapse, the
non-table winner's version means, D5's collapse rules, D6, `strongest_nonzero`, the registered / adaptive labelling.

### 0.1 Changes from IPR1 (all registered before any outcome)

1. Comparison structure: TWIN / SBC / SPEND / RO are replaced by **START** (the family's frozen starting release), **CONT** (gating
   iff the card declares it needed) and the **CTRL frontier** (section 4). D2's met label is
   `COMPONENT_GAIN_BEYOND_START_AND_CONTROLS`; reader targeting (d) is kept.
2. D1 reports the **confidence conjunction** (PASS / UNRESOLVED / VIOLATION per condition) beside G1 and the reserve (section 2).
3. **UM** (new; section 3): usefulness margin against START's binding condition; preparatory evidence only.
4. D2R is read at CAND with the registered decisive set; sign changes are carried into the headline; at least one of R3 / R4.
5. Outcome classes renamed so that no partial result reads as the desired method; the three outcomes are reported separately.
6. Grid validity: the **few-valued explanation** (section 9) counts seed equalities on few-valued views once instead of
   invalidating the grid (brief: "count actual diversity (n_eff) rather than invalidating the comparison").
7. Families `FA`, `FB`, `FS_<letters/digits>`; FA START = SUCC, FB START = NEURAL_RAW; registered only revision 0 of FA / FB.
8. Reader grid: offset +1000, offsets 300, 400, 500, 600, 700, 800, 900 forbidden (140 seeds), bootstrap base 2026102400, split tag
   `SAR1_devgrid_R{r}`. (The mechanical copy's tests still asserted the IPR1 seeds; they are rewritten.)
9. (Section 14, coordinator's request before registration.) An optional card-declared **ALT** frontier path (the starting
   release's own protection operation continued to matched spend), and the **declared selection data** of every bracket.

## 1. The family reading

One reading = one card (revision 0, a revision, or the replacement family) at its CAND:
- **CAND**: the operation at the strength chosen by the card's MODEL_FIT-only rule (`status` FITTED, per-version `strength`), or none
  (`NO_FEASIBLE_STRENGTH`). D1, UM and D2-D5 are read at CAND only. A strength is a number, a component vector or a per-version dict of
  either; nonzero in a version iff every component is finite and >= 0 and at least one is > 0.
- **START**: FA SUCC, FB NEURAL_RAW (frozen, both versions; `start.id` must equal the family's anchor, else D2 / UM read
  INCOMPLETE); an FS family declares a benchmark reference. START is keyed in `grid`, `costs_vs` and `nolookup` by its reference name
  (one entry serves D2 and D3). `start.binding` = the declared binding condition per version (section 3).
- **CONT**: an equally exposed task-only or zero-protection continuation; `cont_declared` (boolean, card) is required; gating iff
  true; a CONT entry given without a declaration is reported beside, never gating.
- **CTRL path**: `ctrl_origin` (card: `START`, or `CONT` when the untargeted operation's zero strength is the continuation) followed
  by the distinct saved CTRL points (`paths['CTRL']`; aliases once via `alias_of`, whose targets may be `START` / `CONT`; an alias
  of the origin collapses into it). An empty list is compared at the origin (flag `CTRL_AT_ORIGIN`). A path not given is
  INCOMPLETE unless `controls_not_run['CTRL']` is True (budget; recorded, never inferred).
- **ALT path** (optional; `alt_declared` boolean from the card, absent = false and then nothing changes): "the starting release's
  own protection operation continued to matched spend" (for FB, the RAW-J critic recipe continued from NEURAL_RAW): origin START
  (always, whatever `ctrl_origin` is) followed by the distinct saved ALT points (`paths['ALT']`); the same bracket rule and frontier
  conjuncts as CTRL. Gating iff declared; a declared ALT not run for budget gives `NOT_ASSESSED_CONTROLS_NOT_RUN`; an ALT path given
  without a declaration is reported beside, never gating.
- **Selection data** (`selection_data[CTRL]`, `selection_data[ALT]`, recorded in the plan): the data on which CAND's spend over START,
  the saved points' spends and the SE of the bracket rule were measured: `MODEL_FIT_OOF` (MODEL_FIT out-of-fold paired costs) or
  `CALIBRATION` (CALIBRATION paired costs, task labels only, never SEX) when START is in-sample on MODEL_FIT
  (`start_in_sample_on_model_fit` recorded True); never MODEL_SELECT. FA declares CALIBRATION for its CTRL; FB MODEL_FIT_OOF for CTRL
  and ALT. CAND's own strength selection uses the card's declared selection data under the same restriction (within-SEX guard parts
  MODEL_FIT out-of-fold), recorded in the plan and checked by the verifier.
- **References** (D3 / D4): NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ, W, TR100, TT100 (frozen, point-eligible), re-read on
  the SAR1 grid with the same reader fitting and selection opportunity; U0 / Ucal are utility diagnostics; DDR_PTC_R1 is beside.

## 2. D1 task feasibility (`d1`, `confidence_conjunction`)

- **Gating (G1)**: all 28 original conditions at the point at CAND: `TASK_FEASIBLE`, `CAND_INFEASIBLE`, `NO_FEASIBLE_STRENGTH`,
  `INCOMPLETE`. Each fitted nonzero strength's point label is recorded (for `SELECTED_STRENGTH_INFEASIBLE`).
- **Confidence conjunction (reported, never gating)**: for CAND, START and CONT, each condition from its own household-clustered SE
  (sqrt(se^2 + se_mc^2) for a randomized release; no floor): **PASS** iff value + 2 se <= cap; **VIOLATION** iff value - 2 se > cap;
  **UNRESOLVED** otherwise (neither a pass nor a violation). Release label `CONFIDENCE_ALL_PASS` (all 28 pass),
  `CONFIDENCE_VIOLATION` (any violation), `CONFIDENCE_UNRESOLVED`, `INCOMPLETE`. These are 28 one-sided per-condition bounds, not a
  simultaneous statement; no multiplicity correction. A point failure is never a PASS; a point pass is never a VIOLATION. SUCC's
  men's guard (+0.0044, se 0.0018) reads UNRESOLVED: SUCC is point-eligible, not confidence-resolved.
- **Reserve (reported, never gating)**: NR's two-SE reserve of CAND, START and CONT.
- The phrase "task feasible" means G1; "confidence-resolved" may be used only with `CONFIDENCE_ALL_PASS`; neither is a
  population guarantee. Flags `START_POINT_INELIGIBLE` (disagrees with the frozen anchor record), `CONT_POINT_INELIGIBLE`,
  `POINT_FEASIBLE_NOT_CONFIDENCE_RESOLVED`, `POINT_FEASIBLE_NOT_RESERVE_QUALIFIED`.

## 3. UM usefulness margin against START (`um`; defined for every reading, meaningful for FA)

1. **Binding condition** per version = the condition with the smallest two-SE reserve ratio of START's MODEL_SELECT record
   (`binding_condition`: NR.reserve_status read one version at a time). It must equal `start.binding` (declared in the plan from
   the frozen anchor table) and, for FA / FB, the frozen values: SUCC `guard|sex0` / `guard|sex0`; NEURAL_RAW `guard|sex0` /
   `income|U0|brier_excess` (flag `BINDING_CONDITION_DIFFERS_BY_VERSION`). A disagreement is INCOMPLETE (a record error).
2. **Resolved improvement**: per version d_v = paired cost CAND - START on `condition_metric(binding_v)` with its household-clustered
   SE (MC in quadrature); pooled by NR's rule (mean of versions, SE = mean of version SEs, a correlation bound); resolved iff
   pooled + 2 SE < 0 **and** both d_v < 0.
3. **Recovery profile retained** (`recovery_profile`, needs a valid grid): the pair and both single views of CAND not more
   recoverable than START's by more than 0.010 AUC per version at the aggregate point, and no resolved pair worsening (aggregate upper
   bound of the pair benefit vs START below 0).
4. **Whole release**: CAND's whole-release point margin (min over the 14 conditions of (cap - value) / cap) not below START's in
   either version, so that the margin is not shifted to another condition (`WHOLE_RELEASE_POINT_MARGIN_BELOW_START`). Reported
   beside: all 8 paired condition differences with resolved better / worse (`CONDITION_RESOLVED_WORSE_<metric>`), CAND's and START's
   reserve and its change per version, the confidence state of the binding condition at CAND and START.
5. **Labels** in order: `NOT_ASSESSED_NO_FEASIBLE_STRENGTH`; `NOT_ASSESSED_EXACT_ALIAS`; `INCOMPLETE` (CAND); `NOT_ASSESSED_TASK_INFEASIBLE`
   (CAND fails G1: "with G1 kept" fails); `INCOMPLETE` (START record, binding, paired costs); `MARGIN_NOT_RESOLVED` (needs no grid);
   `NOT_COMPUTED_GRID_*`; `INCOMPLETE` (grid entry vs START); `MARGIN_GAIN_PROFILE_NOT_RETAINED`; `MARGIN_GAIN_SHIFTED_ELSEWHERE`;
   `MARGIN_RECOVERED` (2, 3 and 4 hold, CAND G1).
6. **Scope**: a margin gain is preparatory evidence only, never a protection, component-gain or competitive claim; START's binding
   condition was identified on the same MODEL_SELECT people (selection on its level, not on the paired difference).

## 4. D2 component gain (`d2`, `brackets`, `grid_fineness`)

### 4.1 Labels in order of precedence

1. `NOT_ASSESSED_NO_FEASIBLE_STRENGTH`; 2. `NOT_ASSESSED_EXACT_ALIAS` (every nonzero strength transmits START's vectors: the family
stops); 3. `INCOMPLETE` (CAND status or utility); 4. `NOT_ASSESSED_TASK_INFEASIBLE` (never negative, never protection); 5.
`NOT_COMPUTED_GRID_VALIDITY_UNKNOWN` / `NOT_COMPUTED_GRID_INVALID`; 6. `INCOMPLETE` (START not the family's anchor; `cont_declared`
not boolean; `alt_declared` recorded but not boolean; `ctrl_origin` not START / CONT, or CONT without a declared CONT); 7.
`NOT_ASSESSED_CONTROLS_NOT_RUN` (a gating control recorded as not run for budget); 8. `INCOMPLETE` (a gating frontier path's
selection data not recorded, or not allowed: flag `CONTROL_SELECTION_DATA_NOT_ALLOWED_<key>`, the control must be re-selected, a
POST HOC repair; any needed input missing: never skipped); 9. the conjuncts:
- **(a)** G1 at CAND (step 4);
- **(b) START**: `NR.comparison` passes (AGG_POSITIVE pair benefit and the recipient gate, both views, both versions);
- **(b') CONT** (when declared): the same;
- **(c) CTRL_FRONTIER** (origin + saved points): (c1) `NR.frontier_comparison` = `DOMINATES_CONTROL_FRONTIER`; (c2) the recipient
  gate vs Q*; (c3) no cheaper path point resolved less recoverable than CAND (aggregate upper bound < 0 and both version means < 0);
- **(c') ALT_FRONTIER** (when declared; START + saved ALT points): the same (c1)-(c3) at the ALT bracket;
- **(d) reader targeting**: the non-table winner's point pair benefit > 0 in both version means vs START, CONT (when gating) and
  every gating Q* (CTRL's and, when declared, ALT's).
10. `COMPONENT_GAIN_BEYOND_START_AND_CONTROLS` iff (b), (b') when declared, (c), (c') when declared and (d); `READER_TARGETED` iff
every gating comparison passes and (d) fails; `NOT_SEPARATED` iff no gating comparison passes; otherwise `PARTIAL_<passed>_ONLY`
(passed comparisons joined by `+` in the order START, CONT, CTRL_FRONTIER, ALT_FRONTIER), with a statement naming what passed and
what did not: without CTRL_FRONTIER the gain may be explained by task spending, without ALT_FRONTIER by running the starting
release's own protection operation further (spending more of its margin), without CONT by retraining alone, without START it is not
lower than the start. **No partial outcome is a claim.** With ALT declared the met label's scope adds "beyond the starting release's
own protection operation continued to matched MODEL_SELECT total-NLL spend". `NOT_SEPARATED` is the default reading, never "equal". Scope sentence of the met label:
a resolved linked-pair benefit with both recipient gates beyond START, CONT (when declared) and the untargeted control at matched
MODEL_SELECT total-NLL spend, for the declared development readers on reused people; not superiority over the strong references
(D3), not the material target (D4).

### 4.2 The CTRL and ALT frontiers, no bracket, residual mismatch

- For each frontier path, two points are saved before any MODEL_SELECT read, on the card's declared selection data (MODEL_FIT
  out-of-fold paired costs, or CALIBRATION paired costs when START is in-sample on MODEL_FIT; never MODEL_SELECT): the largest grid
  point whose paired total-NLL spend over START is <= CAND's (the origin counts with spend 0) and the smallest whose spend is >=
  CAND's + 2 SE (SE = household-clustered SE, on the same data, of that point's paired difference against CAND). On MODEL_SELECT,
  NR's rule picks Q* on the path. CALIBRATION brackets carry the flag `SELECTION_DATA_CALIBRATION_<key>`.
- **No bracket** (every path point spends less MODEL_SELECT total NLL than CAND): the comparison reads `NOT_SEPARATED`, reason
  `NO_BRACKET`, flag `NO_BRACKET_CTRL` / `NO_BRACKET_ALT`. Meaning: lower recovery cannot be separated from task spending; it is a limit of the control
  path, never evidence that CAND equals the control or that the mechanism has no effect; the mechanism's contribution is never
  called isolated.
- **Bracket at the origin** (CAND spends no more than START): flag `CTRL_BRACKET_IS_ORIGIN`; the CTRL comparison then repeats the
  START comparison (it is not a second independent pass).
- **Residual mismatch** (reported, never gating): cost(CAND) - cost(Q*) (pooled, SE, by version), the smallest overspend against a
  cheaper point, NR's interpolation. Flags `RECIPIENT_GATE_FAILED_VS_<Q*>`, `DOMINATED_BY_CHEAPER_<point>`, `READER_TARGETED_<c>`,
  `READER_SENSITIVE_ENHANCED_<point>` (reported only), `CAND_ALIAS_OF_START` / `_CONT`, `CONT_REPORTED_BESIDE_NOT_DECLARED`,
  `CONT_NOT_SEPARATED_REPORTED_BESIDE`.
- The bracket is read on total NLL; a guard- or accuracy-bound CAND may be favoured or disfavoured by it; (c1)'s secondary metrics
  (including the men's and women's occupation Brier) are the guard.

### 4.3 Power and multiplicity (normal approximation; invented settings)

One comparison resolves a true benefit delta with probability Phi(delta / se_agg - 2): 50% at 2 se_agg, 80% at 2.84 se_agg
(0.011-0.016 / 0.015-0.023 at the 0.0053-0.0081 realized against a close comparator; 0.015-0.028 / 0.021-0.039 between strong
references). D2 with START (se 0.006) and CTRL (0.008), correlation 0.5: 0.15 / 0.38 / 0.66 / 0.86 at delta 0.010 / 0.015 / 0.020 /
0.025 against both; with CONT (0.007) as well 0.09 / 0.30 / 0.59 / 0.83; when the CTRL reproduces half the gain 0.22 at 0.020 and
0.33 at 0.025. With a declared ALT (0.008): START + CTRL + ALT (FB-like) 0.26 / 0.53 / 0.78 / 0.93 at 0.015 / 0.020 / 0.025 / 0.030;
all four 0.22 / 0.49 / 0.76 / 0.92; when the ALT operation reproduces half the gain 0.20 at 0.020 and 0.32 at 0.025. D2 is an
intersection-union conjunction: its per-look false-pass rate is at most that of its weakest member.

### 4.4 Fineness of the saved CTRL / ALT points on the declared selection data (`grid_fineness`; reported, never gating)

Per path (CTRL, ALT) and version, on the path's declared selection data with s = CAND's spend over START and the SE from the same
data: `FINE` iff lower >= 0.75 s (and <= s) and upper >= s + 2 SE and <= s + 2 SE + 0.25 s; `COARSE_BELOW` / `COARSE_ABOVE` /
`COARSE_BOTH`; `CONTROL_PATH_MAY_NOT_BRACKET` (largest grid spend < s + 2 SE); `SAVED_POINT_RULE_NOT_FOLLOWED` (lower > s, upper < s +
2 SE although reachable, or, with the full grid recorded, the saved points are not the largest / smallest qualifying grid points);
`NOT_APPLICABLE_CAND_NOT_ABOVE_START` (s <= 0: the origin is the bracket); `SELECTION_DATA_NOT_ALLOWED` (MODEL_SELECT or another
source, or CALIBRATION while START is not recorded in-sample on MODEL_FIT; flag `CONTROL_SELECTION_DATA_NOT_ALLOWED_<key>`);
`INCOMPLETE` (selection data not recorded: flag `CONTROL_SELECTION_DATA_NOT_RECORDED_<key>`; or an input missing); `NOT_GIVEN`.
Computed before any MODEL_SELECT read so that a coarse path gets more grid points first; flags copied into D2's flags; never a
reason to set a reading aside (D2 itself enforces the selection-data restriction, section 4.1 step 8).

## 5. D3 and D4 (`_d3`, `_d4`, `reference_credit`)

Read at CAND whenever CAND is fitted and G1-feasible and the grid is valid (also when D2 is INCOMPLETE or NOT_ASSESSED_CONTROLS_NOT_RUN);
otherwise they carry D2's gate label or INCOMPLETE. **D3** = NRM1 G4: per reference, resolved pair benefit (AGG_POSITIVE) with the
recipient gates; `SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES` only if every point-eligible reference passes; the references passed and
failed are listed (`passed`, `failing`, `per_reference`). START is one of the nine: D3's START entry is D2's START entry. **D4** =
NRM1 G5: `MATERIAL_0020_RESOLVED` iff the 0.020 target screen is resolved against both NEURAL_RAW and H_LJ with the recipient gates;
smaller gains keep their level (`MATERIAL_0020_POINT_ONLY`, `POSITIVE_BELOW_MATERIAL_TARGET`, `NOT_MET`); never weakened. Both carry
the reader-budget credit rule (a READER_BUDGET_ASYMMETRIC reference is credited only if the non-table winner's point benefit is
positive in both versions; missing input is INCOMPLETE, never "not beaten").

## 6. D5 and D6 (`_d5`)

D5 = NRM1 G6 unchanged, only when a card claims joint design (local twin and both sequential orders built); otherwise
`NOT_APPLICABLE_NOT_JOINT` (not claimed). A declared design with a missing grid entry or cost reads INCOMPLETE; collapsed sequential
orders as IPR1 (to the separate design: the reduced test; to the joint design: `NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL`). Gated like
D3 / D4; never changes the class. D6 = `NOT_CLAIMED` unless a separate prior-art review supplies `NOVELTY_REVIEW_PENDING`,
`NOVEL_PER_SEPARATE_REVIEW` or `PRIOR_ART_FOUND`; never inferred from performance.

## 7. The three outcomes, the outcome class, registration and the campaign summary (`decisions`, `campaign_summary`)

- **Three separately reported outcomes** (`outcomes`, one statement, quoted whole): (1) component gain = D2 (with D2R's label and
  any sign-change caveat); (2) strong references = D3 (passed / failed lists); (3) material target = D4. UM is reported beside as
  preparatory. None implies another.
- **Outcome class**, first applicable: `TASK_INFEASIBLE` (no feasible strength, or CAND and every other fitted nonzero strength fail
  G1); `SELECTED_STRENGTH_INFEASIBLE` (CAND fails G1, another fitted strength passes: never promoted); `INCOMPLETE` (D1 incomplete, D2
  incomplete or not computed); `NO_OUTPUT_CHANGE` (exact alias of START: established directly, not a protection experiment);
  `PROTECTION_NOT_ASSESSED` (a gating control not run); `INCOMPLETE` (D3 undecided); `METHOD_CANDIDATE` (D2 met and D3 met; D4 is
  reported separately and never implied); `COMPONENT_GAIN_NOT_COMPETITIVE` (D2 met, D3 not); `REFERENCE_ADVANTAGE_NOT_ATTRIBUTED`
  (D3 met, D2 not: lower recovery than every reference, START included, but not separated from CONT / CTRL or reader-targeted);
  `USEFULNESS_MARGIN_ONLY` (UM `MARGIN_RECOVERED`, D2 and D3 not met: preparatory); `INCOMPLETE` (UM undecided);
  `FEASIBLE_NO_RELIABLE_GAIN`. D4, D5 and D2R never change the class. Flags: `D3_WITHOUT_D2`, `D4_WITHOUT_D2`,
  `POINT_FEASIBLE_NOT_RESERVE_QUALIFIED`, `POINT_FEASIBLE_NOT_CONFIDENCE_RESOLVED`, `FAMILY_STOPS_EXACT_ALIAS`,
  `NO_STRENGTH_MEETS_MODEL_FIT_BUDGET`, `CAND_STRENGTH_DIFFERS_BY_VERSION`, `D2R_SIGN_CHANGE_CARRIED_INTO_HEADLINE`, the D1, UM, D2,
  D5 and MC flags.
- **Registration**: a reading is registered iff it is revision 0 of FA or FB and its card was committed before any MODEL_SELECT
  outcome of this campaign; at most one per family. Every other reading is tagged `<KIND>_<parts>` (`REVISED` if its card preceded
  every MODEL_SELECT outcome, else `ADAPTIVE`; parts `FS`, `R<n>`, `R0_LATE_CARD`) and every label (D1, UM, D2-D6, the class) is
  prefixed `<tag>__`. A revision triggered by a MODEL_SELECT outcome is adaptive as a whole. A family other than FA, FB or
  `FS_<letters/digits>` (CTL included) is refused (ValueError).
- **Campaign summary**: the registered reading per family, every other reading with its tag, K = the number of D2 looks (readings
  whose D2 reached the conjuncts), the readings with D2, D3, D4 met and UM recovered, flag `D2_MET_ONLY_IN_NON_REGISTERED_LOOKS`, and
  the statement every campaign-level sentence must carry: "D2 was met in m0 of k0 registered family readings and in m1 of k1
  non-registered readings (K D2 looks in total); D3 was met in .. and D4 in .. of n readings; UM MARGIN_RECOVERED (preparatory only)
  in ..; one AGG_POSITIVE comparison passes a true null in about 0.07-2% of replicates under six moderate NRM1 known laws and about
  8% when one version or split dominates (NRM1 RULES.md 5.5); a D2 look is an intersection-union conjunction of two to four such
  comparisons (START, CONT when declared, the CTRL frontier at its bracket, the ALT frontier at its bracket when declared), so its
  false-pass rate is at most that of its weakest member; with K D2 looks the chance that at least one passes by chance is at most K times the per-look rate; no multiplicity
  correction is applied or claimed."

## 8. Strongest feasible nonzero release and D2R (`strongest_nonzero`, `refit_effectiveness`, `d2r`)

- `strongest_nonzero`: as IPR1 with START in place of TWIN (`pair_vs_start`; flag `MIXED_STARTS`); winner's curse; descriptive.
- `refit_effectiveness`: R1, R2 plus the robustness repetitions run (at least one of R3 / R4; flag `ONE_ROBUSTNESS_SPLIT_ONLY`): their
  assignment hashes distinct from each other and from R1 / R2, every SCORE overlap involving them recorded, finite and in [0, 1);
  grid validity on all four reader seeds and every repetition run; otherwise `REFIT_NOT_GENUINELY_EFFECTIVE` and the failed part's
  cells count once.
- **D2R** (`d2r`; descriptive, never changes D2): read at CAND (the D2 configuration) with the registered decisive set {START, CONT
  when gating, the registered CTRL bracket Q*, the registered ALT bracket when declared} recorded in `comparators` (an absent key is
  INCOMPLETE, and `ALT` is required iff `alt_declared`; `None` means no bracket / no CONT). Per comparator `ROBUSTNESS_CONFIRMED` iff `NR.robustness` confirms (AGG_POSITIVE on the robustness cells and on all cells),
  the registered cells are present and the refit is `REFIT_GENUINELY_EFFECTIVE`; family label over the decisive set; the other
  references present (at least NEURAL_RAW and H_LJ) are reported beside. **A sign change** between the registered-cell and the
  robustness-cell aggregate mean, on a decisive comparator or a reference beside, is flagged (`SIGN_CHANGE_<comparator>`) and its
  `headline_caveat` must appear in the headline (`outcomes.statement` carries it). No retuning on the robustness cells: CAND, the
  brackets and every selection are the registered ones. R3 / R4 are new splits of the same people: robustness to classifier
  refitting, never replication.

## 9. Reader-grid validity of the registered SAR1 grid (`grid_validity`; prospective)

Reader seeds S<v>k<k> recorded as distinct integers equal to MODEL_SEED_BASE + 1000 + 10 v + k (READER_SPLITS.json keys V<v>k<k>;
2026102205-2026102208, 2026102215-2026102218), none equal to MODEL_SEED_BASE + {300, ..., 900} + 10 v + k (k = 0..9; 140 seeds; +900
is IPR1's realized grid); bootstrap seeds S<v><R> recorded and equal to 2026102400 + 100 v + repetition index. Per seed pair, every
seed-dependent recipe checked and every seed-dependent equality explained:
- off the minimum set by NRM1 A-SC1 (iii) ((a) another seed-dependent recipe differs in the same wire, view, version and
  repetition; (b) the same recipe differs on another view of the same wire; (c) the distinct-input count is recorded);
- on the minimum set by the inherited extra-trees support explanation (ET5 / ET25, (a) + (b) + (c), distinct inputs <= 8001 //
  min_leaf: 1600 / 320);
- anywhere by the **few-valued explanation**: the view's recorded distinct-input count is at most 320 (= 8001 // 25) and the same
  recipe differs (a recorded check with `equal_k0_k1` false) on another wire or view in the same version, repetition and seed pair
  (the seed is shown to reach the recipe). On a few-valued view any reader can reduce to a seed-invariant ranking of the few values;
  such equalities are counted once (n_eff, reported) and never invalidate the comparison.
Any other seed-dependent equality invalidates the grid. **Minimum set** = a D2 reading's CAND, START, CONT (when declared), the
saved CTRL points and the saved ALT points (when declared); for the refit, the D2R release and its decisive comparators; references, ablations and D5 comparators are off
it. The strict NRM1 reading is reported beside. Claims need `grid_valid` recorded and exactly True; otherwise UM (when its profile
is needed), D2-D5 and D2R read `NOT_COMPUTED_GRID_*`; any later repair is POST HOC. Note: the aggregate's se_rs uses block means
over reader seeds, so an explained seed equality reduces n_eff without biasing se_agg.

## 10. Required records (the coordinator's inputs; the plan file of each reading, see the report)

1. `family`, `card`, `revision`, `card_committed_before_any_model_select_outcome`; CAND's status, config_id, per-version `strength`,
   `every_nonzero_strength_alias_of_start`; the utility records of START, CAND, CONT and every saved nonzero strength.
2. `start` = {id, evals, binding}; `cont_declared`, `cont` (config_id, evals) when declared; `ctrl_origin`; `alt_declared`;
   `paths['CTRL']`, `paths['ALT']` (when declared), `alias_of`, `controls_not_run` (budget only); `selection_data` per gating frontier
   path and `start_in_sample_on_model_fit`; `fineness` from `grid_fineness` on the declared selection data (recommended, with the
   full grid).
3. Grid entries (all three views) of CAND vs START, CONT, every CTRL and ALT point, the nine references and the D5 comparators;
   paired costs vs START, CONT and every CTRL and ALT point (`total|nll` + the 8 secondary metrics, `se_mc` for randomized releases);
   `nolookup` per-cell benefits vs START, CONT, every CTRL and ALT point and the nine references; `enhanced` (reported); the
   reader-budget flags.
4. `joint` for a jointly designed family; `novelty` from a separate review.
5. Grid validity inputs (seeds as S<v>k<k>, bootstrap seeds, seed checks with `k_pair`, distinct-input counts per wire, version and
   view for every minimum-set wire, the minimum set).
6. For D2R: merged registered + robustness entries, the decisive `comparators`, the split records.

## 11. Limitations and points not settled here

- Version- or split-dominated laws give about 8% false pass per AGG_POSITIVE comparison (NRM1 5.5); a reading whose block means show
  one dominant version or split carries that caveat. Two versions and two registered splits rest on one degree of freedom each.
- Selection of CAND and of the CTRL / ALT points on MODEL_FIT or CALIBRATION treats refitted pipelines as fixed; a selection
  device, not inference. On CALIBRATION, components in-sample there (FA: H3 income and Ucal) bias the measured spend (the income side
  in SUCC's favour, so CAND's spend over SUCC tends to look larger); this affects only which points are saved and whether they
  bracket, never the MODEL_SELECT reading of Q*, which uses MODEL_SELECT paired costs by NR's rule. A biased bracket shows up as
  NO_BRACKET or a coarse fineness label, never as a pass.
- The ALT path is a targeted control (the start's own protection operation): passing it says the gain is not reproduced by running
  that operation to matched spend, not that no targeted operation could reproduce it.
- The anchors and START's binding condition were chosen on MODEL_SELECT: START's level on the binding condition is selected
  upward-biased (regression to the mean favours an apparent margin recovery measured against the cap); UM therefore uses the paired
  difference, which removes the shared sampling error but not the selection of which condition to work on.
- The recipient allowance and the recovery-profile retention are point rules: under the NRM1 calibration a true single-view
  worsening of exactly 0.010 passes the point gate about 32-34% of the time.
- The CONT and CTRL-origin declarations, the "untargeted" nature of the CTRL operation and its equal exposure are card statements the
  rules cannot verify; they are checked by the challenger and the verifier (an exact-alias wire check where the comparators exist).
- The few-valued and extra-trees explanations are necessary conditions of their mechanisms, not sufficient ones.
- All readings use the same adaptively used MODEL_SELECT people; registration status and the look count are disclosed instead of a
  correction.

## 12. Changes requested of the protocol

The numbered findings with exact wording are in `REPORT_PREREGISTRATION.md`. The rules above implement every REQUIRED finding that
concerns a decision. If the coordinator declines one, the corresponding rule must be changed (and this file re-frozen) before
registration, never after an outcome.

## 13. Completion record

- First freeze completed 2026-10-24T04:55:49Z (`date -u`); revised before registration at the coordinator's request (section 14;
  the counts below are those of the first freeze, section 14 gives the revised ones). Started 04:28:04Z. The hand-back message records the SHA-256 of RULES.json, sar/rules.py and sar/test_rules.py at completion (this
  file cannot contain its own hash).
- `python3 -m sar.test_rules`: 23 of 23 PASS, exit 0 (invented data only; about 0.22 CPU-s per run; one numerical thread). A scratch
  in-memory mutation check (24 single-line seeded bugs of `sar/rules.py`, among them: offset 900 not forbidden, UM's both-version
  condition, the whole-release point margin or the resolved-pair-worsening check dropped, CONT never gating, a missing bracket
  passing, targeting ignored, the confidence PASS read at the point, MC SE ignored, the few-valued explanation without its seed
  evidence or its 320 bound, a NRM1 (iii) explanation accepted on the minimum set, USEFULNESS_MARGIN_ONLY unreachable, D3 crediting
  a not-credited reference, sign changes beside ignored, START not checked against the family, the declared binding not checked,
  D2R without a robustness split, D3-without-D2 renamed METHOD_CANDIDATE, a missing CTRL path treated as empty, a CONT origin
  without a declared CONT, an undeclared CONT made gating, the confidence flag dropped) was caught 24 of 24 times; no repository
  file was changed by it.
- Inherited files verified unchanged: NRM1 RULES.json 84f18776..., nrm/rules.py d5df8686... (load_rules checks both).
- No label was read and no real-data computation was run.

## 14. Changelog (before registration; no outcome of this campaign exists)

- **2026-10-24T05:05:54Z** (`date -u`). Revision at the coordinator's request (messages at about 04:56Z and 05:00Z, based on the two
  investigators' designs; I did not read the proposals, so the rules stay family-neutral):
  1. **ALT path** (sections 1, 4.1, 4.2, 4.4, 8, 9, 10): optional, card-declared `alt_declared`; "the starting release's own
     protection operation continued to matched spend"; origin START; the same two-point bracket rule, conjuncts (c1)-(c3) and
     targeting (d) against its Q*; gating iff declared; partial order START, CONT, CTRL_FRONTIER, ALT_FRONTIER; an undeclared ALT
     path is reported beside and changes no label; D2R's decisive set adds ALT's registered bracket when declared; the multiplicity
     statement counts two to four comparisons per D2 look. FA declares no ALT.
  2. **Declared selection data** (sections 1, 4.1 step 8, 4.2, 4.4): CAND's spend, the saved points' spends and the bracket SE come
     from the card's declared data per path (`MODEL_FIT_OOF`, or `CALIBRATION` when START is in-sample on MODEL_FIT; never
     MODEL_SELECT), recorded in the plan; not recorded or not allowed -> D2 INCOMPLETE; CALIBRATION disclosed by a flag; the
     in-sample bias on CALIBRATION is a stated limitation (section 11). FB declares MODEL_FIT_OOF for CTRL and ALT; FA declares
     CALIBRATION for its CTRL.
  3. The coordinator's answers recorded: the UM whole-release point-margin clause ADOPTED; the targeting conjunct (d) KEPT as a
     gate; the few-valued bound 320 ADOPTED; initial budgets FA 8 (CAND 2 + CTRL 4 + CONT 2) and FB 12 (CAND 2 + CONT 2 + CTRL 4 +
     ALT 4), 20 of 36 in total.
  - `python3 -m sar.test_rules`: 25 of 25 PASS, exit 0 (invented data; about 0.26 CPU-s). The in-memory mutation check, extended
    by 9 ALT / selection-data mutations (CONT or ALT dropped from the gating set, ALT not evaluated as gating, MODEL_SELECT bracket
    data accepted, CALIBRATION accepted without START in-sample, ALT's origin following CTRL's, ALT ignored in D2R, an undeclared ALT
    made gating, the ALT scope dropped, the fineness data check removed), was caught 33 of 33 times; no repository file was changed
    by it. The SHA-256 of RULES.json, sar/rules.py and sar/test_rules.py after this revision are in the hand-back message.
