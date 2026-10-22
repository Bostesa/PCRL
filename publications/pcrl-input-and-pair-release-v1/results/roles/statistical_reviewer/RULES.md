# Decision rules of the input-and-pair release campaign (IPR1; statistical reviewer; schema IPR1-rules-v1)

**Status.** Writing started 2026-10-22T01:40:07Z (`date -u`). The completion time is recorded in section 13. Written before any
outcome of this campaign existed: I read no label, ran no real-data computation, fitted nothing and called no loader that reads
labels. Every number I computed came from invented data (`python3 -m ipr.test_rules`; an in-memory mutation check; a normal-
approximation power calculation in my scratch directory).
- Machine-readable constants: `RULES.json` (schema `IPR1-rules-v1`). Implementation: `ipr/rules.py` (pure functions; imports
  `nrm.rules` unchanged as NR; `load_rules()` refuses a RULES.json that disagrees with the module, a changed NRM1 RULES.json or a
  changed `nrm/rules.py`). Invented-data tests: `python3 -m ipr.test_rules` (prints JSON; exit 0 iff all pass).
- Protocol review: `REPORT_PREREGISTRATION.md` (same directory), numbered findings marked REQUIRED or ADVISORY. The harness refused
  my direct write of that report file (subagents return reports as text), so it was returned to the coordinator in the hand-back
  message, to be saved verbatim with attribution (as the DDR1 and FRC1 reports were).
- **Inputs read** (code and public records only): the coordinator's draft `PROTOCOL.md` (SHA-256 efa40008..., last modified
  01:39:38Z, unchanged while I worked), `FAILURE_MAP.md`, `EXPOSURE_MAP.md`, `roles/coordinator/DESIGN_BRIEF.md`, `cards/README.md`
  and the label-free `READER_SPLITS.json` of this campaign; the DDR1 `RULES.md`, `RULES.json`, `REPORT_PREREGISTRATION.md` and
  `CLOSING_STATEMENT.md` (sections 1-2); `ddr/rules.py` (via its mechanical copy `ipr/rules.py`, diffed against it),
  `ddr/test_rules.py` (via `ipr/test_rules.py`); `nrm/rules.py` (frontier rule, claims G4-G6) and the helpers of `nrm/test_rules.py`.
  I did not read `roles/designer_a/`, `roles/designer_b/`, any card other than `cards/README.md`, anything under `.private/`,
  any label file, anything of MODEL_SELECT or any reader cache.
- **Coordinator note (disclosed).** At about 02:00Z, while I worked, the coordinator asked (without proposal content) that the
  rules give a defined label to an operation that reads no SEX and is a function of the twin's released vectors plus a
  SEX-independent key, and to D5 when the sequential orders collapse by construction to the joint or the separate design. Both are
  implemented as general rules (sections 3.3 and 5); they apply to any family that declares those properties in its card.

**What every rule here is.** A development rule on previously and adaptively used records (MODEL_FIT and MODEL_SELECT were used by
earlier campaigns): nothing here is confirmation. No rule changes an original limit (PROTOCOL 3; NRM1 RULES.md 0). The two-SE
reserve is reported, never gating. Point feasibility is never called confidence preservation. Every uncertainty statement is
asymptotic and model-based. Versions 0 and 1 share every training row and the MODEL_SELECT people: "both versions" is a
procedure-repeat requirement, never replication. Sign conventions are NR's: pair benefit b = comparator AUC - candidate AUC
(positive favours the candidate); recipient difference r = candidate AUC - comparator AUC; paired costs loss(candidate) -
loss(comparator) (negative favours the candidate; accuracy enters as a loss).

## 0. Inherited unchanged

From NRM1 (`nrm.rules`, imported unchanged; RULES.json 84f18776..., nrm/rules.py d5df8686..., both checked by `load_rules()`;
nothing is monkeypatched, test `t_nrm_rules_unchanged_not_monkeypatched`): G1 = 24 clauses + 4 guards at the point; the reserve;
randomized releases (`mc_cell`, se_mc in se_agg and in paired costs; `MC_SE_NOT_ESTIMABLE` travels with the reading); the aggregate
paired screen (8 registered cells, blocks = version x repetition, se_agg = max(sqrt(se_hh^2 + se_rs^2 + se_mc^2), 0.001), both
version means must agree; `ALIAS_ZERO` never positive), `comparison` (AGG_POSITIVE + recipient gate r_v <= 0.010 in both views and
versions), the frontier rule (`frontier_comparison`), `robustness`, claims G4 / G5 / G6 and the nine-reference benchmark. The NRM1
known-law limitation applies (false pass of one AGG_POSITIVE comparison about 0.07-2% under six moderate laws, about 8% when one
version or split dominates).

From DDR1 (`ddr/rules.py`; copied, not imported) every rule not listed in 0.1 is kept: D1; D2's conjuncts (b), (c1)-(c3), (d) and
its gates; D3 / D4 with the reader-budget credit rule; D6; the outcome classes; registered / adaptive labelling and the campaign
summary; `strongest_nonzero`; `refit_effectiveness`; grid validity with the prospective extra-trees (ER) explanation.

### 0.1 Changes from DDR1 (all registered before any outcome)

1. D2 has a third gating control path, RO (release-only), with a declared kind and its own met label for a release-only operation
   (section 3).
2. A declared SEX-blind operation has degenerate SBC and RO comparisons; its D2 gates on TWIN and SPEND only, with its own met label
   (section 3.3).
3. D5 = NR G6 passed through (its NOT_APPLICABLE_* labels included), INCOMPLETE for missing D5 records, gated like D3 / D4, and
   defined for sequential orders that collapse by construction (section 5).
4. The MODEL_FIT control-grid fineness check `grid_fineness` (reported, section 3.4) and the flags
   `CAND_STRENGTH_DIFFERS_BY_VERSION` and `SBC_CONSTRUCTION_DIFFERS_BY_VERSION`, computed from the record (DDR1 closing statement
   1.2 items 2 and the SBC caveat of 1.3).
5. Strengths may be component vectors (DDR1 read [0.35, 0.65] as "not nonzero": its deviation 8 and closing statement 1.2.1).
6. D2R: three paths; an unrecorded bracket gives INCOMPLETE; the release id is read from `selected` or `config_id` (DDR1 deviation
   12 (c)).
7. Families IA, IB, FS_<name> only; registered readings only revision 0 of IA or IB.
8. Reader grid: offset +900, forbidden offsets 300-800 (120 seeds), bootstrap base 2026102300, split tag `IPR1_devgrid_R{r}`.

## 1. The family reading (inputs of `d1`, `d2`, `decisions`)

One reading = one card (revision 0, a revision, or the replacement family) at its CAND:
- **CAND**: the operation at the strength chosen by the card's MODEL_FIT-only rule (`status` FITTED; `strength` the per-version
  selection), or none (`NO_FEASIBLE_STRENGTH`). D1-D5 are read at CAND only.
- **TWIN**: the equally trained task-only model of the family.
- **Control paths** SBC, SPEND, RO: each = **TWIN followed by the distinct saved control points** (`control_path`; an exact alias of
  an earlier point, or of TWIN, counted once). An empty list is compared at [TWIN] (flag `<key>_AT_IDENTITY`). A path not given is
  `INCOMPLETE` unless `controls_not_run[key]` is True (budget; recorded, never inferred) or it is degenerate by declaration (3.3).
- **Card declarations copied into the record**: `ro` = {`kind`: `SUBSTITUTED` | `GENERIC_ZTILT`, `operation_release_only`: bool}
  (required whenever the RO path is given); `operation_sex_blind` (bool, default false); `sbc_construction` per version (reported);
  `joint` (D5; section 5).
- Grid entries (`paired()` format) of CAND against TWIN, every path point, the nine references and the D5 comparators; MODEL_SELECT
  paired costs (`total|nll` + the 8 secondary metrics) against TWIN and every path point; `nolookup` per-cell pair benefits against
  TWIN, every path point and the references; optionally `enhanced` (reported only); reader-budget flags; `grid_valid`; `revision`,
  `family`, `card_committed_before_any_model_select_outcome`; optionally `fineness` = `grid_fineness(...)`.
- **Strength format**: a number, a component vector (for example recipient-split strengths) or a per-version dict of either.
  Nonzero in a version iff every component is finite and >= 0 and at least one is > 0. The tie-break scalar is the sum of every
  component over both versions (any positive monotone summary selects the same release unless |dx| <= 1e-12).

## 2. D1 task feasibility (`d1`)

Unchanged from DDR1: `NO_FEASIBLE_STRENGTH`, `INCOMPLETE`, `CAND_INFEASIBLE`, `TASK_FEASIBLE`, `CAND_FEASIBLE_TWIN_INFEASIBLE`; the
reserve of both reported beside; each saved nonzero strength's point label recorded (component vectors are read).

## 3. D2 improvement beyond matched controls (`d2`, `brackets`, `grid_fineness`)

### 3.1 Labels in order of precedence

1. `NOT_ASSESSED_NO_FEASIBLE_STRENGTH`; 2. `NOT_ASSESSED_EXACT_ALIAS` (the family stops); 3. `INCOMPLETE` (CAND status or utility);
4. `NOT_ASSESSED_TASK_INFEASIBLE` (CAND fails G1: not run, never negative, never protection); 5. `NOT_COMPUTED_GRID_VALIDITY_UNKNOWN`
/ `NOT_COMPUTED_GRID_INVALID`; 6. `INCOMPLETE` (`operation_sex_blind` not a boolean; an SBC path given for a SEX-blind operation);
7. `NOT_ASSESSED_CONTROLS_NOT_RUN` (a **gating** path recorded as not run for budget); 8. `INCOMPLETE` (the RO declaration missing or
inconsistent; a SUBSTITUTED RO path for a SEX-blind operation; any needed input missing: never skipped); 9. the conjuncts:
- **(a) G1** at CAND (step 4);
- **(b) vs TWIN**: `NR.comparison` passes (AGG_POSITIVE pair benefit and the recipient gate, both views, both versions);
- **(c) vs each gating control path** (`SBC_FRONTIER`, `SPEND_FRONTIER`, `RO_FRONTIER`; for a SEX-blind operation `SPEND_FRONTIER`
  only), each separately, all three of: (c1) `NR.frontier_comparison` = `DOMINATES_CONTROL_FRONTIER` (bracket Q* = the cheapest path
  point whose pooled MODEL_SELECT paired total-NLL cost is >= CAND's; AGG_POSITIVE vs Q*; no secondary metric resolved higher vs
  Q*); (c2) the recipient gate vs Q*; (c3) no cheaper path point resolved less recoverable than CAND (aggregate upper bound < 0 and
  both version means < 0);
- **(d) reader targeting**: the non-table winner's point pair benefit > 0 in both version means vs TWIN and every gating Q*.
10. Result: the **met label** if (b), every gating (c) and (d) hold; `READER_TARGETED` if (b) and every gating (c) hold and (d)
fails; `NOT_SEPARATED` if no gating comparison passes; otherwise `PARTIAL_<passed>_ONLY` (passed comparisons joined by `+` in the
order TWIN, SBC_FRONTIER, SPEND_FRONTIER, RO_FRONTIER; for example `PARTIAL_TWIN+SBC_FRONTIER+SPEND_FRONTIER_ONLY`). **No partial
outcome is a claim.** `NOT_SEPARATED` is the default reading, never "equal".

| Met label (all count as "D2 met") | When | Scope sentence (D2 field `scope`; quoted with the label) |
|---|---|---|
| `IMPROVEMENT_BEYOND_MATCHED_CONTROLS` | RO kind SUBSTITUTED | the gain needs SEX information beyond what the twin's release carries (for the declared readers) |
| `IMPROVEMENT_BEYOND_MATCHED_CONTROLS` | RO kind GENERIC_ZTILT, operation not release-only (fallback) | the gain is not reproduced by the generic release-only tilt at matched spend; the operation driven only by the twin's release was not built |
| `RELEASE_ONLY_MAP_BEYOND_MATCHED_CONTROLS` | the operation is itself release-only (RO = GENERIC_ZTILT) | a better release-only map than the generic tilt at matched spend; never information beyond the twin's release |
| `SEX_BLIND_IMPROVEMENT_BEYOND_TWIN_AND_SPEND` | declared SEX-blind operation (3.3) | SBC and RO degenerate by construction; assessed against TWIN and SPEND only; no SEX-targeting claim |

Flags: `CAND_ALIAS_OF_TWIN`, `<key>_AT_IDENTITY`, `NO_BRACKET_<key>` (a design limit of the path, never evidence of equality),
`RECIPIENT_GATE_FAILED_VS_<Q*>`, `DOMINATED_BY_CHEAPER_<point>`, `READER_TARGETED_<comparison>`, `READER_SENSITIVE_ENHANCED_<point>`
(reported only), `RO_SUBSTITUTED` / `RO_GENERIC_ZTILT`, `RELEASE_ONLY_OPERATION`, `SEX_BLIND_OPERATION`,
`SBC_DEGENERATE_BY_CONSTRUCTION`, `RO_DEGENERATE_BY_CONSTRUCTION`, `RO_GENERIC_ZTILT_REPORTED_BESIDE`,
`SBC_CONSTRUCTION_DIFFERS_BY_VERSION` (reported), and the fineness flags of 3.4 (reported). If CAND is cheaper than TWIN, TWIN is Q*
on every path. `brackets()` returns each path's Q* by NR's rule (test `t_brackets_equal_nrm_rule`, all three paths).

### 3.2 The RO path

- **SUBSTITUTED** (default for an operation with a SEX-dependent fitted input that is not already a function of TWIN's release):
  CAND's operation with the MODEL_FIT SEX column replaced, in every SEX-dependent fitted input, by a release-only SEX surrogate:
  P(SEX | TWIN's released log-probabilities of both recipients), fitted on MODEL_FIT and cross-fitted on the household folds (as a
  soft label where the fit accepts weights, otherwise one fixed-seed draw, declared in the card). It answers whether the gain needs
  SEX information beyond what the twin's release carries.
- **GENERIC_ZTILT**: the generic release-only tilt of TWIN (`ipr.ctlkit`). Required when the operation is itself release-only
  (every SEX-dependent fitted input already a function of TWIN's released vectors: the substitution would return the operation
  itself; a SUBSTITUTED declaration is then INCOMPLETE). Allowed for another operation only as the card's declared fallback.
- The RO path fits the frontier machinery unchanged: TWIN + two saved points, Q* read on MODEL_SELECT total NLL by NR's rule.
- **Power and multiplicity.** D2 is an intersection-union conjunction: each comparison must pass on its own, so adding RO cannot
  raise the per-look false-pass rate above that of its weakest member (no adjustment within a look). It lowers power, most when the
  release-only control reproduces part of the gain. Normal approximation (invented, ignoring the version-mean condition): one
  comparison resolves a true benefit delta with probability Phi(delta / se_agg - 2): 0.50 at delta = 2 se_agg, 0.80 at 2.84 se_agg.
  With se_agg 0.010 / 0.012 / 0.015 / 0.009 (TWIN / SBC / SPEND / RO) and correlation 0.5, D2's power is about 0.24, 0.56 and 0.90
  at delta = 0.025, 0.035 and 0.050 against every comparator, and about 0.29 when delta is 0.035 against TWIN, SBC and SPEND but
  0.015 against RO (0.16 for a DDR-like 0.025 / 0.013).

### 3.3 A SEX-blind operation (declared `operation_sex_blind`)

An operation with no SEX-dependent fitted input (SEX enters only through the MODEL_FIT strength-selection guards; for example a
function of the twin's released vectors and a SEX-independent key) has an SBC path and a substituted RO path that would each be the
operation itself: both comparisons are **degenerate by construction** (`DEGENERATE_BY_CONSTRUCTION`, flags
`SBC_DEGENERATE_BY_CONSTRUCTION`, `RO_DEGENERATE_BY_CONSTRUCTION`). No SBC path is built (an SBC path given is INCOMPLETE); a
SUBSTITUTED RO path given is INCOMPLETE. D2 gates on TWIN and `SPEND_FRONTIER` (targeting vs TWIN and the SPEND Q*); its met label is
`SEX_BLIND_IMPROVEMENT_BEYOND_TWIN_AND_SPEND`, quoted with its scope: no SEX-targeting claim and no claim about SEX information beyond
the twin's release is possible. A GENERIC_ZTILT RO path, if built, is computed and reported beside (`passed_if_it_were_gating`),
never gating, never a targeting point, and its missing inputs never make D2 incomplete. A SEX-blind operation's SBC and RO are never
"not run" (they are not part of its D2). The conjunction has two members instead of four, so its power is higher; its per-look
false-pass rate is still bounded by one comparison's.

### 3.4 MODEL_FIT control-grid fineness (`grid_fineness`; reported, never gating)

Per path and version, with s = CAND's MODEL_FIT out-of-fold paired total-NLL spend over TWIN and SE the household-clustered SE of the
paired difference (upper point - CAND): `FINE` iff the saved lower point spends >= 0.75 s (and <= s) and the saved upper point spends
>= s + 2 SE and <= s + 2 SE + 0.25 s; `COARSE_BELOW` / `COARSE_ABOVE` / `COARSE_BOTH` otherwise; `CONTROL_PATH_MAY_NOT_BRACKET` if the
path's largest grid spend is below s + 2 SE; `SAVED_POINT_RULE_NOT_FOLLOWED` if the lower point spends more than s or the upper less
than s + 2 SE although the path reaches it; `NOT_APPLICABLE_CAND_NOT_ABOVE_TWIN` if s <= 0 (TWIN is then Q* on every path);
`INCOMPLETE` if an input is missing; `NOT_GIVEN` for a path not built. Computed on MODEL_FIT before any MODEL_SELECT read so that a
coarse path gets added MODEL_FIT grid points first; its flags (`CONTROL_GRID_COARSE_<key>`, `CONTROL_PATH_MAY_NOT_BRACKET_<key>`,
`CONTROL_POINT_RULE_NOT_FOLLOWED_<key>`, `CONTROL_GRID_FINENESS_INCOMPLETE_<key>`) are copied into D2's flags; they never change a
label and are never a reason to set a reading aside. (The DDR1 release-only grid, 0.90x below and 2.08x above, reads `COARSE_ABOVE`.)

## 4. D3 and D4 (`_d3`, `_d4`, `reference_credit`)

Unchanged from DDR1: read at CAND whenever CAND is fitted and G1-feasible and the grid is valid (also when D2 is INCOMPLETE or
NOT_ASSESSED_CONTROLS_NOT_RUN); otherwise they carry D2's gate label or INCOMPLETE. D3 = NRM1 G4 (nine references, intersection-
union) and D4 = NRM1 G5 (0.020 vs NEURAL_RAW and H_LJ), both with the reader-budget credit rule (three-valued). The dual-design
releases are reported beside, never members.

## 5. D5 and D6 (`_d5`)

- **D5 = NRM1 G6 unchanged** on `joint` = {`outputs_changed`, `jointly_designed`, `separate`, `income_first`, `occupation_first`,
  `costs_vs`}: `NOT_APPLICABLE_NOT_JOINT` (no joint design declared, or `joint` absent), `NOT_APPLICABLE_ONE_OUTPUT_CHANGED`,
  `NOT_ASSESSED_MISSING_CONTROLS` (a comparator not named), `JOINT_CONTRIBUTION_SUPPORTED` (AGG_POSITIVE + recipient gate vs each
  comparator and CAND's MODEL_SELECT primary cost not resolved higher than any of theirs) or `NOT_SUPPORTED`.
- **Missing records**: a declared joint design that names all three comparators but lacks a grid entry (pair or a recipient view)
  or a paired cost of one reads `INCOMPLETE` (NR would read the first as missing controls and the second as NOT_SUPPORTED; a missing
  record is never a negative result).
- **Sequential orders that collapse by construction** (`joint.collapsed` = {order: `SEPARATE` | `JOINT`}, declared in the card
  before its first fit with the construction argument): an order equal to the **joint** design gives
  `NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL` (the joint design is that sequential design: no joint-design advantage over sequential
  design can exist; no claim); an order equal to the **separate** design is replaced by the separate comparator and NR G6 is read
  unchanged on that record (the **reduced test**; NR's label; flag `D5_REDUCED_<ORDER>_EQUALS_SEPARATE`; the statement names the
  reduction). Naming another comparator for a collapsed order, or a malformed declaration, is INCOMPLETE. Without a declaration an
  order that was not built is not named, and NR gives `NOT_ASSESSED_MISSING_CONTROLS`.
- **Gating**: D5 carries D2's `NOT_ASSESSED_*` / `NOT_COMPUTED_*` gate label (PROTOCOL 6.2 (a): "D3-D5 carry the same label") and
  is INCOMPLETE when CAND is not fitted or not G1-eligible. D5 never changes the outcome class. D5 comparators are off the
  grid-validity minimum set (as references are).
- **D6** = `NOT_CLAIMED` unless a separate prior-art review supplies `NOVELTY_REVIEW_PENDING`, `NOVEL_PER_SEPARATE_REVIEW` or
  `PRIOR_ART_FOUND`; never inferred from performance.

## 6. Outcome class per family reading (`decisions`)

As DDR1, first applicable: `TASK_INFEASIBLE`; `SELECTED_STRENGTH_INFEASIBLE`; `INCOMPLETE`; `PROTECTION_NOT_ASSESSED`; `CANDIDATE`
(D2 met and D3 met); `COMPONENT` (D2 met, D3 not); `TASK_ONLY_FEASIBLE_NO_PROTECTION`. "D2 met" is any of the three met labels; the
class carries `ro_scope` (= D2's scope sentence), which every sentence quoting the class repeats. Flags: `D3_WITHOUT_D2`,
`TWIN_INFEASIBLE`, `POINT_FEASIBLE_NOT_RESERVE_QUALIFIED`, `FAMILY_STOPS_EXACT_ALIAS`, `NO_STRENGTH_MEETS_MODEL_FIT_BUDGET`,
`CAND_STRENGTH_DIFFERS_BY_VERSION` (from `cand.strength`), the D2 and D5 flags and the MC flags. D4 and D5 never change the class.

## 7. Registered and non-registered readings; campaign summary

- A family reading's `family` must be `IA`, `IB` or `FS_<letters/digits>`; anything else (`CTL` included: shared controls are not a
  reading) is refused (ValueError).
- **Registered reading**: revision 0 of IA or IB whose card was committed before any MODEL_SELECT outcome of this campaign; at most
  one per family; the family headline is its registered reading.
- Every other reading is tagged `<KIND>_<parts>` as in DDR1 (`REVISED` / `ADAPTIVE`; `FS`, `R<n>`, `R0_LATE_CARD`) and every D label
  and the class are prefixed `<tag>__`. A revision triggered by a MODEL_SELECT outcome is adaptive as a whole.
- **`campaign_summary`**: the registered reading per family, every other reading with its tag, K = the number of D2 looks (readings
  whose D2 reached the conjuncts), the readings with D2 met, flag `D2_MET_ONLY_IN_NON_REGISTERED_LOOKS`, and the statement every
  campaign-level sentence must carry: "D2 was met in m0 of k0 registered family readings and in m1 of k1 non-registered readings (K
  D2 looks in total); one AGG_POSITIVE comparison passes a true null in about 0.07-2% of replicates under six moderate NRM1 known laws
  and about 8% when one version or split dominates (NRM1 RULES.md 5.5); a D2 look is an intersection-union conjunction of four such
  comparisons (TWIN, SBC, SPEND, RO), so its false-pass rate is at most that of its weakest member; with K D2 looks the chance that at
  least one passes by chance is at most K times the per-look rate."

## 8. Strongest feasible nonzero release, D2R and the ablation

- `strongest_nonzero`: as DDR1 (largest registered-cell aggregate pair benefit vs TWIN among the family's own operation
  configurations with a nonzero strength in every version, G1, not an alias; ties to the smaller strength, then the lower revision,
  then config_id; winner's curse; descriptive). Component vectors are read; the selected id is returned as `selected` and
  `config_id`.
- **Its bracketing controls** are recomputed by `brackets()` on all three paths (for a SEX-blind operation: SPEND, and a reported
  ZTILT bracket is not decisive).
- `refit_effectiveness`: as DDR1 (R3 / R4 hashes distinct, SCORE overlaps in [0, 1), grid validity on the eight seeds with both seed
  pairs judged separately and the bootstrap seeds of R1-R4; minimum set = the strongest release, its TWIN and its bracketing
  controls).
- **D2R** (`d2r`; descriptive, never changes D2): decisive set {TWIN, the release's bracketing controls of SBC, SPEND and RO};
  `ROBUSTNESS_CONFIRMED` per comparator iff `NR.robustness` confirms on the merged registered + robustness cells (24 robustness
  cells) and the refit is `REFIT_GENUINELY_EFFECTIVE`; family label CONFIRMED iff all decisive comparators are; `NOT_RUN` if any
  entry is missing; **`INCOMPLETE` if a path's bracket is not recorded** (key absent, or a brackets() record with status INCOMPLETE;
  an explicit None means no bracket). NEURAL_RAW and H_LJ reported beside. R3 / R4 are new splits of the same people: robustness to
  classifier refitting, never replication.
- **Ablation**: descriptive; never upgrades D2.

## 9. Reader-grid validity of the registered IPR1 grid (`grid_validity`; prospective)

As DDR1 with this grid's constants: reader seeds S<v>k<k> recorded as distinct integers equal to MODEL_SEED_BASE + 900 + 10 v + k
(READER_SPLITS.json keys V<v>k<k>; 2026102105-2026102118), none equal to MODEL_SEED_BASE + {300, 400, 500, 600, 700, 800} + 10 v + k
(k = 0..9; 120 seeds; +800 is DDR1's realized grid); bootstrap seeds S<v><R> recorded and equal to 2026102300 + 100 v + repetition
index (required inputs); per seed pair, every seed-dependent recipe checked, NRM1 A-SC1 (iii) off the minimum set, and on the
minimum set only the extra-trees explanation (ET5 / ET25, (a) + (b) + (c), distinct-input count <= 8001 // min_leaf: 1600 / 320).
**Minimum set** = a D2 reading's CAND, TWIN and the saved points of its gating paths (SBC, SPEND, RO; a reported-beside ZTILT path
is off it); for the refit, the strongest release, its TWIN and its bracketing controls; references, ablations and D5 comparators
are off it. Claims need `grid_valid` recorded and exactly True; otherwise D2-D5 (and D2R) read `NOT_COMPUTED_GRID_*`; any later
repair is POST HOC.

## 10. Required records (the coordinator's inputs)

1. Per reading: `family`, `card`, `revision`, `card_committed_before_any_model_select_outcome`, CAND's status, config_id and
   per-version `strength`, `every_nonzero_strength_alias_of_twin`, the utility records of TWIN, CAND and every saved nonzero strength.
2. The control paths as saved ids per path (`SBC`, `SPEND`, `RO`), `alias_of`, `controls_not_run` (budget only), and from the card:
   `ro` (kind, operation_release_only), `operation_sex_blind`, `sbc_construction` per version; `fineness` from `grid_fineness` on
   MODEL_FIT spends (recommended).
3. Grid entries (all three views) of CAND vs TWIN, every path point, the nine references and the D5 comparators; paired costs vs
   TWIN and every path point (`total|nll` + 8 secondary metrics, `se_mc` for randomized releases); `nolookup` per-cell benefits vs
   TWIN, every path point and the nine references; `enhanced` (reported); the reader-budget flags.
4. `joint` for a jointly designed family (comparator names, `costs_vs`, and `collapsed` when the card declares a collapse).
5. Grid validity inputs as DDR1 (reader seeds as S<v>k<k>, bootstrap seeds, seed checks, distinct-input counts, minimum set).
6. For D2R: merged registered + robustness entries, the release's brackets on all three paths, the split records.

## 11. Limitations and points not settled here

- Version- or split-dominated laws give about 8% false pass per AGG_POSITIVE comparison (NRM1 5.5); such a reading carries the
  caveat.
- MODEL_FIT selection of CAND and of the control points treats refitted pipelines as fixed; it is a selection device, not inference.
- The bracket is read on total NLL; a guard- or accuracy-bound CAND may be favoured or disfavoured by it; (c1)'s secondary metrics
  are the guard.
- All readings use the same adaptively used MODEL_SELECT people; registration status and the look count are disclosed instead of a
  correction.
- The SUBSTITUTED RO path is only as release-only as its surrogate: the surrogate must be cross-fitted, otherwise it can memorize
  SEX labels of the MODEL_FIT people and carry more than the release. A weaker surrogate may also need larger strengths to reach
  CAND's spend (`CONTROL_PATH_MAY_NOT_BRACKET`).
- The SEX-blind and collapsed-order declarations are card statements the rules cannot verify; they must be made before the first fit
  and are checked by the challenger and the verifier (an exact-alias wire check where the comparators exist).
- The extra-trees support bound is a necessary condition of the mechanism, not a sufficient one.

## 12. Changes requested of the protocol

The numbered findings with exact wording are in `REPORT_PREREGISTRATION.md`. The rules above implement every REQUIRED finding that
concerns a decision. If the coordinator declines one, the corresponding rule must be changed (and this file re-frozen) before
registration, never after an outcome.

## 13. Completion record

- Completed 2026-10-22T02:08:15Z (`date -u`, read immediately before this line was written; no later time is claimed). Started
  01:40:07Z.
- `python3 -m ipr.test_rules`: 28 of 28 PASS, exit 0 (invented data only; about 0.63 CPU-s per run). A scratch in-memory mutation
  check (22 single-line seeded bugs of `ipr/rules.py`, among them: RO mapped to the SPEND path, the release-only or SEX-blind met
  label dropped, a release-only operation accepted with a substituted RO path, a SEX-blind operation gated on all four paths or
  given an SBC or substituted RO path, targeting extended to a beside path, the D5 collapse rules or missing-record rule removed, D5
  not gated, component strengths requiring every component > 0, a non-IA / IB family registered or accepted, an unrecorded D2R
  bracket accepted, the fineness upper bound ignored, +800 not forbidden, paths inferred as not run, the recipient gate or the
  cheaper-control check at the bracket ignored, fineness flags dropped) was caught 22 of 22 times; no file in the repository was
  changed by it.
- SHA-256 at completion: RULES.json 4c235046c57d8175..., ipr/rules.py 4ef5fa35e68676a8..., ipr/test_rules.py a5e11220eee4bc87...
  (this file cannot contain its own hash). Reviewed PROTOCOL.md efa4000830f08ba0... (unchanged since 01:39:38Z).
- Inherited files verified unchanged at completion: NRM1 RULES.json 84f18776..., nrm/rules.py d5df8686..., nrm/test_rules.py
  d03bbed2... (load_rules checks the first two).
- No label was read and no real-data computation was run.
