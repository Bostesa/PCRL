# Decision rules of the dual-design release campaign (DDR1; statistical reviewer; schema DDR1-rules-v1)

**Status.** Writing started 2026-10-21T16:52:55Z (`date -u`). The completion time is recorded at the end of this file. Written
before any outcome of this campaign existed: I read no label, ran no real-data computation, fitted nothing and called no loader
that reads labels. Every number I computed came from invented data (`python3 -m ddr.test_rules`).
- Machine-readable constants: `RULES.json` (schema `DDR1-rules-v1`). Implementation: `ddr/rules.py` (pure functions; imports
  `nrm.rules` unchanged as NR; `load_rules()` refuses a RULES.json that disagrees with the module, a changed NRM1 RULES.json or a
  changed `nrm/rules.py`). Invented-data tests: `python3 -m ddr.test_rules` (prints JSON; exit 0 iff all pass).
- Protocol review: `REPORT_PREREGISTRATION.md` (same directory), numbered findings marked REQUIRED or ADVISORY. The harness refused
  my direct write of that report file (subagents return reports as text), so it was returned to the coordinator in the hand-back
  message, to be saved verbatim with attribution (as the FRC1 report was).
- **Inputs read** (code and public records only): the coordinator's draft `PROTOCOL.md` (sections 0-12, writing started 16:52Z) and
  `roles/coordinator/DESIGN_BRIEF.md`; `READER_SPLITS.json`, `DEVGRID_CODE_LOCK.json` and `roles/coordinator/checks/DEVGRID_SELFTEST.json`
  of this campaign; `nrm/rules.py`, `nrm/test_rules.py` and the NRM1 `RULES.md` / `RULES.json`; `frc/rules.py`, `frc/test_rules.py`,
  the FRC1 `RULES.md`, `RULES.json`, `RULES_ER1.md`, `RULES_ER1.json` and `REPORT_PREREGISTRATION.md`; the header and the
  reader-validity functions of `fsa/rules.py`, the FSA1 `READER_SPLITS.json` (reader seeds and bootstrap only) and its
  `RESEARCH_DECISION.md` (questions 4-7); the constants, manifest and selftest code of `ddr/devreaders.py`; the docstrings of
  `ddr/readerbudget.py`, `ddr/readerextra.py`, `ddr/evaluate.py`, `ddr/release.py`, `ddr/jobqueue.py`; `ddr/refitcheck.py`;
  `ddr/common.py` (constants and the registration file list); `frc/claims.py` (how the FRC1 rules were called); `nrm/fa/prc.py`
  (`select_path`); `ffv/readers.py` (extra-trees settings). I read no designer proposal and no card (none existed; the designer and
  challenger directories held no proposal when I listed them), nothing under `admitted/exports`, nothing of MODEL_SELECT and no
  reader cache.
- **Disclosure (known before freezing).** (1) The published records listed in PROTOCOL section 0 and the registered-grid table of
  the design brief. (2) The reader-seed base MODEL_SEED_BASE + 800 and the bootstrap base 2026102200 of this grid are exactly the
  ones FSA1 registered (FSA1 `READER_SPLITS.json`: V0k0 = 2026102005, bootstrap "2026102200 + 100 x version + repetition index").
  FSA1 never built a grid and read no MODEL_SELECT record (its RESEARCH_DECISION, question 7), so no realized reader fit used them.
  They are not forbidden here; the protocol's word "new" is corrected in REPORT_PREREGISTRATION.md (finding 1).

**What every rule here is.** A development rule on previously and adaptively used records (MODEL_FIT and MODEL_SELECT were used by
earlier campaigns): nothing here is confirmation. No rule changes an original limit (PROTOCOL 3; NRM1 RULES.md 0). The two-SE
reserve is reported, never gating. Point feasibility is never called confidence preservation. Every uncertainty statement is
asymptotic and model-based. Versions 0 and 1 share every training row and the MODEL_SELECT people: "both versions" is a
procedure-repeat requirement, never replication. Sign conventions are NR's: pair benefit b = comparator AUC - candidate AUC
(positive favours the candidate); recipient difference r = candidate AUC - comparator AUC; paired costs loss(candidate) -
loss(comparator) on the same people (negative favours the candidate; accuracy enters as a loss).

## 0. Inherited unchanged

From NRM1 (`nrm.rules`, imported unchanged; RULES.json 84f18776..., nrm/rules.py d5df8686..., both checked by `load_rules()`; nothing
is monkeypatched, test `t_nrm_rules_unchanged_not_monkeypatched`): the original limits and G1 = 24 clauses + 4 guards at the point
(`point_eligibility`); the reserve (`reserve_status`, MC SE in quadrature); randomized releases (`mc_cell`; `cell_se_mc` enters
se_agg as its own component, paired-cost `se_mc` enters in quadrature; `MC_SE_NOT_ESTIMABLE` travels with the reading); the
aggregate paired screen (8 registered cells, blocks = version x repetition, se_agg = max(sqrt(se_hh^2 + se_rs^2 + se_mc^2), 0.001),
both version means must agree; `ALIAS_ZERO` never positive), its benefit, 0.020 and recipient labels (`comparison`: AGG_POSITIVE +
recipient gate r_v <= 0.010 in both views and versions); the frontier rule (`frontier_comparison`: bracket Q* = the cheapest control
point whose pooled MODEL_SELECT paired total-NLL cost is >= the candidate's, AGG_POSITIVE vs Q*, no secondary metric resolved
higher vs Q*); `robustness`; claims G4 / G5 / G6 (`claims`); the nine-reference benchmark with its published point labels and alias
structure; the scope sentences. The known-law limitation of NRM1 RULES.md 5.5 applies unchanged (false pass of one AGG_POSITIVE
comparison about 0.07-2% under six moderate laws, about 8% when one version or split dominates).

Copied from FRC1 (`frc.rules`; frc is not imported): the reader-budget credit rule for references (D3, D4), the control-path alias
collapse, the version means of the non-table winner, `strongest_nonzero`, `refit_effectiveness` and the seed part of grid validity;
FRC1's post hoc ER1 explanation is adopted here PROSPECTIVELY in a narrower form (section 9).

## 1. The family reading (inputs of `d1`, `d2`, `decisions`)

One reading = one card (the registered card, revision 0, or a revision / the replacement family) at its CAND:
- **CAND**: the family's operation at the strength chosen by the card's MODEL_FIT-only rule (`status` FITTED), or none
  (`NO_FEASIBLE_STRENGTH`: no nonzero grid strength met the MODEL_FIT budget; no fallback). D1-D4 are read at CAND only, never at
  another strength, the strongest release or the ablation.
- **TWIN**: the equally trained task-only model of the family.
- **Control paths**: SBC and SPEND, each = **TWIN (strength 0 of every SEX-blind path) followed by the distinct saved control
  points** (`control_path`; an exact alias of an earlier point, or of TWIN, is counted once). A path given as an empty list is
  compared at [TWIN] (flag `<key>_AT_IDENTITY`). A path not given is `INCOMPLETE`, unless the record says it was not run for
  budget (`controls_not_run[key] = True`; never inferred).
- **strengths**: every saved nonzero-strength configuration of the operation under this card, CAND included (needed only when
  CAND fails G1, to tell `TASK_INFEASIBLE` from `SELECTED_STRENGTH_INFEASIBLE`).
- Grid entries (`paired()` format) of CAND against TWIN, every control-path point and the nine references; MODEL_SELECT paired costs
  (`nrm.evaluate` / `ddr.evaluate` `paired_costs`, with `total|nll` and the 8 secondary metrics) against TWIN and every path point;
  the non-table winner's per-cell pair benefit (`nolookup`) against TWIN, every path point and the nine references; optionally the
  enhanced winner's (`enhanced`, reported only); the reader-budget flags of the references; `grid_valid` (section 9); `revision`,
  `family` and `card_committed_before_any_model_select_outcome` (section 7).

## 2. D1 task feasibility (`d1`)

G1 (`NR.point_eligibility`, original limits, inclusive) for TWIN and CAND; the reserve of both reported beside.

| Label | Condition (first applicable) |
|---|---|
| `NO_FEASIBLE_STRENGTH` | CAND status NO_FEASIBLE_STRENGTH (no CAND exists) |
| `INCOMPLETE` | CAND status missing, or CAND's utility incomplete |
| `CAND_INFEASIBLE` | CAND fails G1 (whatever TWIN; TWIN's label reported) |
| `INCOMPLETE` | CAND passes G1, TWIN's utility incomplete |
| `TASK_FEASIBLE` | CAND and TWIN pass G1 |
| `CAND_FEASIBLE_TWIN_INFEASIBLE` | CAND passes, TWIN fails (flag `TWIN_INFEASIBLE` on the outcome class) |

Each saved nonzero strength's own point label is recorded (`fitted_strengths`); strength-0 entries are ignored and listed.

## 3. D2 improvement beyond matched controls (`d2`, `brackets`)

Labels in order of precedence:
1. `NOT_ASSESSED_NO_FEASIBLE_STRENGTH` (no CAND);
2. `NOT_ASSESSED_EXACT_ALIAS` (every allowed nonzero strength transmits exactly the twin's vectors; the family stops);
3. `INCOMPLETE` (CAND status or utility missing);
4. **`NOT_ASSESSED_TASK_INFEASIBLE`** (CAND fails G1): the protection comparison is reported as not run (PROTOCOL 7), never as a
   negative result and never as protection; it precedes the grid gate because it is known from utility alone;
5. `NOT_COMPUTED_GRID_VALIDITY_UNKNOWN` / `NOT_COMPUTED_GRID_INVALID` (`grid_valid` not exactly `True`);
6. `NOT_ASSESSED_CONTROLS_NOT_RUN` (a control path recorded as not run for budget);
7. `INCOMPLETE` (any needed input missing: grid entry vs TWIN, Q* or a cheaper path point; recipient views vs Q*; a paired cost or a
   secondary metric of NR's frontier rule; a non-table targeting input vs TWIN or a Q*). Never skipped.
8. Then the conjuncts:
   - **(a) G1** at CAND (already established by step 4);
   - **(b) vs TWIN**: `NR.comparison` passes (pair benefit `AGG_POSITIVE` and the recipient gate, both views, both versions);
   - **(c) vs the SBC path and vs the SPEND path**, each separately (`SBC_FRONTIER`, `SPEND_FRONTIER`), all three of:
     - (c1) `NR.frontier_comparison` = `DOMINATES_CONTROL_FRONTIER` against the path (bracket Q*, AGG_POSITIVE vs Q*, no secondary
       metric resolved higher vs Q*);
     - (c2) **the recipient gate against Q*** (both views, both versions; `NR.comparison(grid[Q*])`). A comparison against a
       matched control carries the original per-comparator recipient allowance, as NRM1 G2 did for both matched controls;
       otherwise leakage could move to one recipient relative to the control (measured failure 3) while the pair improves;
     - (c3) **no cheaper path point is resolved less recoverable than CAND**: for every path point whose pooled paired total-NLL
       cost is below CAND's (d = cost(CAND) - cost(point) > 0), the pair benefit is not resolved negative (aggregate mean +
       2 se_agg < 0 and both version means < 0). The bracket rule alone looks only upward; on a non-monotone control path (for
       example coarsening that helps table readers, measured failure 4) a cheaper control can leak less than CAND, and then CAND is
       not on the control frontier. A point-only or one-version difference never triggers (c3); an exact alias is not "less
       recoverable".
   - **(d) reader targeting**: against TWIN and every bracketing point Q* that (c1) selected, the non-table winner's point pair
     benefit (`nolookup`, 8 registered cells, block means then repetition means) is > 0 in both version means.
9. Result: `READER_TARGETED` if (b), SBC_FRONTIER and SPEND_FRONTIER all pass and (d) fails; **`IMPROVEMENT_BEYOND_MATCHED_CONTROLS`**
   (the only D2 "met") if they all pass and (d) holds; `NOT_SEPARATED` if none passes (the default reading); otherwise
   `PARTIAL_<passed>_ONLY`, passed comparisons joined by `+` in the order TWIN, SBC_FRONTIER, SPEND_FRONTIER (for example
   `PARTIAL_TWIN_ONLY`). **No partial outcome is a claim.**

Flags: `CAND_ALIAS_OF_TWIN` (ALIAS_ZERO vs TWIN; (b) fails), `SBC_AT_IDENTITY` / `SPEND_AT_IDENTITY`, `NO_BRACKET_<key>` (every path
point is cheaper than CAND: reason `CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT`; a design limit of the control path, never
evidence of equality), `RECIPIENT_GATE_FAILED_VS_<Q*>` (reason `RECIPIENT_GATE_FAILED_AT_BRACKETING_CONTROL`),
`DOMINATED_BY_CHEAPER_<point>` (reason `DOMINATED_BY_CHEAPER_CONTROL`), `READER_TARGETED_<comparison>` (a passed comparison whose
targeting fails), `READER_SENSITIVE_ENHANCED_<point>` (the enhanced / TABLE_CV winner's point benefit is <= 0 in some version
mean; **reported only**, never gating).

If CAND is cheaper than TWIN, TWIN is the closest control from above on both paths (Q* = TWIN): no SEX-blind path point can spend
less than its own strength 0, so the matched-spend comparison is the TWIN comparison. `brackets(costs_vs, paths, alias_of)` returns
each path's Q* by NR's rule (test `t_brackets_equal_nrm_rule` checks equality with `NR.frontier_comparison`).

**Power.** Per comparison, 2 se_agg (about 0.010 at se_agg 0.005, the FRC1 grid's typical value) is the resolution threshold; about
0.015 is resolved with roughly 80% power (NRM1 5.5: 0.45-0.51 at 0.010, 0.96-0.99 at 0.020 under moderate laws). D2 needs three
resolved comparisons plus targeting, so its power at a given true effect is lower than any one comparison's. "Not separated" is the
default reading; a smaller effect is never "equal".

## 4. D3 and D4 (`_d3`, `_d4`, `reference_credit`)

Read at CAND, the D2 configuration, whenever CAND is fitted and G1-feasible and the grid is valid (also when D2 is INCOMPLETE or
NOT_ASSESSED_CONTROLS_NOT_RUN); otherwise they carry D2's gate label (`NOT_ASSESSED_*`, `NOT_COMPUTED_*`) or `INCOMPLETE`.
- **D3** = NRM1 G4: all nine point-eligible references, intersection-union, AGG_POSITIVE and the recipient gate each.
- **D4** = NRM1 G5: 0.020 vs NEURAL_RAW and H_LJ, levels kept; met iff `MATERIAL_0020_RESOLVED`. `POSITIVE_BELOW_MATERIAL_TARGET`
  is a smaller gain, never the criterion.
- **Reader-budget credit rule** (FRC1, unchanged): a reference flagged `READER_BUDGET_ASYMMETRIC` (`ddr.readerbudget.contrast`, pair
  view) is credited iff the non-table winner's point benefit against it is > 0 in both version means; a missing input gives
  `INCOMPLETE` for that reference, never "not beaten". Three-valued: NOT_CREDITED turns a met D3 into `NOT_SUPERIOR` and any D4
  level into `NOT_MET`; INCOMPLETE turns a met D3, or a D4 level above NOT_MET, into `INCOMPLETE`; a label already failing on its
  own results stays failing (incomplete references listed).

## 5. D5 and D6

- **D5** = `NOT_ASSESSED_MISSING_CONTROLS` unless the separately designed recipients and both sequential orders exist (`joint` with
  `separate`, `income_first`, `occupation_first`); then NRM1 G6 unchanged (`JOINT_CONTRIBUTION_SUPPORTED` / `NOT_SUPPORTED` / ...).
- **D6** = `NOT_CLAIMED` unless a separate prior-art review supplies `NOVELTY_REVIEW_PENDING`, `NOVEL_PER_SEPARATE_REVIEW` or
  `PRIOR_ART_FOUND`; any other value reads `NOT_CLAIMED`; never inferred from performance.

## 6. Outcome class per family reading (`decisions`)

First applicable:

| Class | Condition |
|---|---|
| `TASK_INFEASIBLE` | D1 `NO_FEASIBLE_STRENGTH` (flag `NO_STRENGTH_MEETS_MODEL_FIT_BUDGET`), or CAND and every other saved nonzero strength fail G1 |
| `SELECTED_STRENGTH_INFEASIBLE` | CAND fails G1 and another saved nonzero strength passes it (reported; never promoted to CAND) |
| `INCOMPLETE` | CAND fails G1 with the saved strengths unrecorded or incomplete; D1 INCOMPLETE; D2 INCOMPLETE or NOT_COMPUTED_*; D2 met and D3 INCOMPLETE or NOT_COMPUTED_* |
| `PROTECTION_NOT_ASSESSED` | D2 `NOT_ASSESSED_CONTROLS_NOT_RUN` (budget forbade a matched control) |
| `CANDIDATE` | D2 met and D3 met |
| `COMPONENT` | D2 met, D3 not met |
| `TASK_ONLY_FEASIBLE_NO_PROTECTION` | CAND passes G1 and D2 is not met (NOT_ASSESSED_EXACT_ALIAS, READER_TARGETED, PARTIAL_*, NOT_SEPARATED); TWIN's failure is a flag, not a class |

Flags: `D3_WITHOUT_D2` (references beaten without separation from the matched controls: not a candidate), `TWIN_INFEASIBLE`,
`POINT_FEASIBLE_NOT_RESERVE_QUALIFIED` (near-limit; never called robust), `FAMILY_STOPS_EXACT_ALIAS`, the D2 flags and the MC
flags. D2 not met with D3 incomplete is still `TASK_ONLY_FEASIBLE_NO_PROTECTION` (D3 cannot change it). D4 never changes the class;
it is reported beside it.

## 7. Registered and non-registered readings; campaign summary (`adaptive_tag`, `decisions`, `campaign_summary`)

- **Registered reading**: revision 0 of DA or DB whose card was committed before any MODEL_SELECT outcome of this campaign (record
  `card_committed_before_any_model_select_outcome = true`). There is at most one per family; the family headline is its registered
  reading.
- **Every other reading** gets a tag `<KIND>_<parts>`: KIND = `REVISED` if its card was committed before any MODEL_SELECT outcome,
  else `ADAPTIVE`; parts = `FS` (replacement family), `R<n>` (revision n >= 1), `R0_LATE_CARD` (a revision-0 card committed after an
  outcome, or the flag not recorded). Examples: `ADAPTIVE_R1`, `REVISED_R1`, `ADAPTIVE_FS`, `REVISED_FS`, `ADAPTIVE_FS_R1`,
  `ADAPTIVE_R0_LATE_CARD`. **Every D label and the outcome class of such a reading is prefixed `<tag>__`** (for example
  `ADAPTIVE_R1__COMPONENT`); `label_unprefixed` keeps the raw label and the `met` booleans are kept for counting. A non-registered
  reading is reported beside its family's registered reading and is never presented as a registered result.
- A revision triggered by a MODEL_SELECT outcome is adaptive as a whole, even though its parameters are chosen on MODEL_FIT; its G1
  on the same MODEL_SELECT people is selection-dependent.
- **`campaign_summary`**: the registered reading per family (a second registered reading of one family is refused), every other
  reading with its tag, K = the number of D2 looks (readings whose D2 reached the conjuncts), the readings with D2 met, flag
  `D2_MET_ONLY_IN_NON_REGISTERED_LOOKS`, and the statement every campaign-level sentence must carry: "D2 was met in m0 of k0
  registered family readings and in m1 of k1 non-registered readings (K D2 looks in total); one AGG_POSITIVE comparison passes a
  true null in about 0.07-2% of replicates under six moderate NRM1 known laws and about 8% when one version or split dominates;
  with K D2 looks the chance that at least one passes by chance is at most K times the per-look rate."
- Within a reading, D2, D3 and D4 are intersection-union conjunctions (each comparison must pass on its own): no multiplicity
  adjustment inside a look. Across looks there is none either: the count and the registration status are disclosed instead.

## 8. Strongest feasible nonzero release, D2R and the ablation (`strongest_nonzero`, `refit_effectiveness`, `d2r`)

- **Strongest feasible nonzero release** (for D2R and the ablation): among the saved configurations of the family's OPERATION
  (controls and ablations excluded; every card, each tagged by revision) with a nonzero strength in every version, G1, not an exact
  alias of an earlier configuration and not ALIAS_ZERO against its TWIN, with a complete registered grid: the largest registered-cell
  aggregate pair benefit x against its TWIN (it may be negative). Ties (|dx| <= 1e-12) go to the smaller strength (a per-version
  strength is summed), then the lower revision, then config_id. None: `NO_FEASIBLE_NONZERO_RELEASE` (no refit, no ablation). Flag
  `MIXED_TWINS` if eligible configurations have different twins. It is chosen on registered MODEL_SELECT cells (winner's curse):
  descriptive; it never changes D1-D6.
- **Its bracketing controls** are recomputed by the same rule from its own MODEL_SELECT paired costs (`brackets`); when it is CAND
  they are D2's Q*.
- **Genuinely effective refit** (`refit_effectiveness`): `REFIT_GENUINELY_EFFECTIVE` iff (a) R3 and R4 use different household
  splits (assignment hashes recorded and distinct from each other and from R1 / R2; every SCORE household overlap involving R3 or R4
  recorded, finite and in [0, 1)) and (b) the grid validity of section 9 holds on the eight seeds k0-k3, with both seed pairs (k0 / k1
  and k2 / k3) judged separately, the bootstrap seeds of R1-R4, and the minimum set = the strongest release, its TWIN and its
  bracketing controls. The extra-trees explanation applies here too (prospectively; FRC1 kept the strict rule for its refit).
  Otherwise `REFIT_NOT_GENUINELY_EFFECTIVE` and the failed part's cells ("robustness repetitions", "robustness reader seeds") are
  counted once: no replication credit. Seed-invariant winners on few-valued views give identical k-cells, which NR's aggregate
  already counts once (n_eff is reported per comparator).
- **D2R** (`d2r`; descriptive, never changes D2): per comparator in the decisive set {TWIN, the release's bracketing controls},
  `ROBUSTNESS_CONFIRMED` iff `NR.robustness` = ROBUSTNESS_CONFIRMED (AGG_POSITIVE on the robustness cells alone and on all cells) on
  an entry that carries the 8 registered cells and the robustness cells merged (R3, R4 with k0-k3; R1, R2 with k2, k3: 24 cells),
  AND the refit is `REFIT_GENUINELY_EFFECTIVE`; `NOT_RUN` if the entry is missing; otherwise `ROBUSTNESS_NOT_CONFIRMED` with reasons
  (`NOT_AGG_POSITIVE_ON_ROBUSTNESS_OR_ALL_CELLS`, `REFIT_NOT_GENUINELY_EFFECTIVE`, `REGISTERED_CELLS_ABSENT`). Family label:
  CONFIRMED iff confirmed against every decisive comparator; NOT_RUN if any is not run; else NOT_CONFIRMED. NEURAL_RAW and H_LJ are
  reported beside. Gates: `NOT_COMPUTED_GRID_*` as in section 9. R3 / R4 are new splits of the same MODEL_SELECT people and k2 / k3
  new reader seeds: robustness to classifier refitting, never replication.
- **Ablation** (descriptive): `NR.comparison` and `NR.robustness` of the strongest release against its ablation and of the ablation
  against TWIN; it never upgrades D2 and inherits the strongest release's winner's curse.

## 9. Reader-grid validity of the registered DDR1 grid (`grid_validity`; prospective)

`READER_SEEDS_EFFECTIVE` iff all of:
1. **Reader seeds**: the required keys S<v>k<k> (S0k0, S0k1, S1k0, S1k1; k2 / k3 added for the refit) recorded as distinct integers;
   every key of the form S<v>k<k> and equal to the registered MODEL_SEED_BASE + 800 + 10 v + k (2026101205 + 800 + ...); none equal
   to MODEL_SEED_BASE + {300, 400, 500, 600, 700} + 10 v + k (k = 0..9; 100 seeds; the realized OHN1/CTS1, CTS, TSZ1, NRM1 and FRC1
   grids; a local equivalent of `nrm.rules.forbidden_seeds`, which reads its own (300, 500)). Requiring the registered values
   prevents a re-run with fresh seeds after an outcome.
2. **Bootstrap seeds**: S<v><R> recorded for both versions and every required repetition (R1, R2; R1-R4 for the refit), each equal
   to 2026102200 + 100 v + repetition index (R1 0, R2 1, R3 2, R4 3); no other key form. A re-run with another bootstrap seed would
   change every cell SE.
3. **Recipe equalities, per seed pair** (k0 / k1; for the refit also k2 / k3, judged separately because NRM1's "differs" key has no
   seed pair, and pooling would let a difference in one pair explain an equality in the other): `nrm.rules.grid_validity` is called
   unchanged per pair (every seed-dependent recipe ET5, ET25, MLP64, MLP6432, OUT_MLP, LOG_MLP must be checked; seed-invariant recipes
   that differ are flagged `NONDETERMINISTIC_RECIPE`, not fatal).
   - Off the minimum set, NRM1 A-SC1 (iii) unchanged: an equality is explained by (a) another seed-dependent recipe differing in the
     same wire, view, version and repetition, (b) the same recipe differing on another view of the same wire, and (c) the view's
     distinct-input count recorded label-free. "Differs" = a recorded check with `equal_k0_k1` false.
   - **On the minimum set** (for a D2 reading: its CAND, its TWIN and every saved point of its two control paths; for the refit:
     the strongest release, its TWIN and its bracketing controls; references and ablations are off it; the seed and bootstrap parts
     are common to every reading), an equality is explained only if **the recipe is an
     extra-trees recipe (ET5 or ET25), (a) + (b) + (c) hold, and the recorded distinct-input count of the view is at most
     8001 // min_leaf (ET5: 1600; ET25: 320)**. This is FRC1's ER1 (post hoc there) adopted before any grid output, restricted to its
     documented mechanism: an extra-trees forest (300 trees, max_features 1.0, min leaf 5 / 25) on a view with few distinct
     transmitted vectors isolates each value in its own leaf and becomes a seed-invariant frequency table; that needs at least
     min_leaf reader-FIT people per value, impossible with more than 8001 // min_leaf distinct values in MODEL_SELECT. An MLP has no
     such mechanism: an MLP equality on the minimum set (or an extra-trees equality on a large-support view) is a seed-implementation
     error and invalidates the grid.
   - Explained minimum-set equalities are reported (`explained_minimum_set_equalities`: count, items, the n_eff they imply per wire /
     view / recipe / seed pair); their duplicate cells are counted once by NR's aggregate. The strict NRM1 reading (any minimum-set
     equality invalidates) is reported beside (`strict_nrm1`).
4. Claims need `grid_valid` recorded and exactly `True`: a missing value gives `NOT_COMPUTED_GRID_VALIDITY_UNKNOWN` and any other
   value `NOT_COMPUTED_GRID_INVALID` for D2-D4 (and D2R). A later repair would be POST HOC and labelled so.

## 10. Required records (the coordinator's inputs)

1. Per reading: `family`, `card`, `revision`, `card_committed_before_any_model_select_outcome` (from the git log of the card and the
   first MODEL_SELECT read in the access log), CAND's status and config_id, `every_nonzero_strength_alias_of_twin` (wire hashes),
   the utility records (`ddr.evaluate`) of TWIN, CAND and every saved nonzero strength.
2. The control paths as saved ids per path, `alias_of` (wire hashes), `controls_not_run` (only when the budget forbade a path).
3. Grid entries (`paired()` format, all three views) of CAND against TWIN, every path point and the nine references; paired costs
   against TWIN and every path point (`total|nll` and the 8 secondary metrics, `se_mc` for randomized releases); `nolookup` per-cell
   pair benefits against TWIN, every path point and the nine references (`frc.claims.nolookup_benefit` logic); `enhanced` likewise
   (reported); the pair-view reader-budget flag per reference.
4. Grid validity inputs: the seeds (S<v>k<k> from READER_SPLITS.json), the bootstrap seeds (S<v><R>; **required**: without them
   the grid is invalid by rule), the recorded seed checks (from the extras' score fingerprints; `k_pair` for the refit), the
   label-free distinct-input counts per wire, version and view over the MODEL_SELECT people, and the reading's minimum set (its
   CAND, TWIN and saved control points). `grid_validity` and `refit_effectiveness` take no `er1` argument: the DDR1 explanation is
   the registered rule.
5. For D2R: the strongest release's merged registered + robustness entries, its paired costs against the path points (for its
   brackets), the split records (assignment hashes and SCORE overlap shares of R1-R4).

## 11. Limitations and points not settled here

- With two versions and two registered splits, a version- or split-dominated law gives a false-pass rate of about 8% per
  AGG_POSITIVE comparison (NRM1 5.5). Any D2 whose block means show one dominant version or split carries that caveat.
- The MODEL_FIT selection of CAND and of the control points (nested held-out estimates with household-clustered SEs) treats the
  refitted pipelines as fixed; it is a selection device, not inference. The rules take the saved points as given and do not
  re-derive them.
- The bracket is read on MODEL_SELECT paired total NLL; if CAND's binding MODEL_FIT constraint was a guard or an accuracy budget, the
  total-NLL bracket can favour either side. The secondary-metric check (c1) and the full paired-cost vector (reported) are the guard;
  no exact spend match is required.
- G1, the bracket, D2 and the strongest-release choice are all read on the same adaptively used MODEL_SELECT people; selection over
  readings inflates the best one (winner's curse). Registration status and the look count are disclosed instead of a correction.
- (c3) protects against a cheaper control that is resolved better; a cheaper control that is better only at the point is reported
  (the interpolation and per-point benefits) but does not fail D2.
- The extra-trees support bound is a necessary condition of the mechanism, not a sufficient one; the exact condition (every distinct
  value has at least min_leaf reader-FIT people) is visible in `ddr.readerbudget` records and may be reported beside.

## 12. Changes requested of the protocol

The numbered findings, with exact wording, are in `REPORT_PREREGISTRATION.md`. The rules above already implement the REQUIRED ones
that concern decisions (control paths starting at TWIN; (c2) and (c3); NOT_ASSESSED_TASK_INFEASIBLE; the two added outcome classes;
the registered / adaptive labelling and the campaign summary; the D2R and refit definitions; grid validity with bootstrap seeds and
the prospective extra-trees explanation). If the coordinator declines one of them, the corresponding rule must be changed (and this
file re-frozen) before registration, never after an outcome.

## 13. Completion record

- Completed 2026-10-21T17:26:03Z (`date -u`, read in the same shell command that wrote this line). Started 16:52:55Z.
- Correction (disclosed): the first version of this line stated "Completed 2026-10-21T17:30:53Z", a value ahead of the clock
  that was not a `date -u` reading (the coordinator's `date -u` read 17:23:55Z when checking). It was replaced by the reading
  above at the coordinator's request; no rule changed with it.
- `python3 -m ddr.test_rules`: 22 of 22 PASS, exit 0 (invented data only; about 0.2 CPU-s per run). A scratch in-memory mutation
  check (34 single-line seeded bugs of `ddr/rules.py`, among them: the extra-trees restriction or the support bound dropped, seed
  pairs pooled, bootstrap seeds ignored, the recipient gate at the bracket or the cheaper-control check ignored, targeting in one
  version only or at >= 0, an infeasible candidate assessed, unknown or truthy grid validity accepted, controls inferred as not run,
  TWIN not prepended, a late revision-0 card registered, the adaptive prefix dropped, every reading counted as a look, ties to the
  larger strength, a missing reference input counted as not beaten, the refit ignored by D2R) was caught 34 of 34 times; no file in
  the repository was changed by it.
- SHA-256 at completion: given in the hand-back report (RULES.json, ddr/rules.py and ddr/test_rules.py are hashed after this line
  was written, and this file cannot contain its own hash).
- Inherited files verified unchanged at completion: NRM1 RULES.json 84f18776..., nrm/rules.py d5df8686..., nrm/test_rules.py
  d03bbed2..., ohn/rules.py 8686a78f....
- No label was read and no real-data computation was run.
