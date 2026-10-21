# Statistical rules of the finite-solver adequacy campaign (FSA1; statistical and challenger reviewer; schema FSA1-rules-v1)

**Status.** Writing started 2026-10-21T02:41:05Z (`date -u`); the completion time is in section 12. Written before the registration
and before any outcome of this campaign. I read no label, opened nothing under `.private/`, ran no real-data computation and fitted
nothing.
- Machine-readable constants: `RULES.json` (schema `FSA1-rules-v1`). Implementation: `fsa/rules.py` (pure functions; `nrm.rules` as
  NR and `frc.rules` as FR imported unchanged, never monkeypatched; `load_rules()` refuses a RULES.json that disagrees with the
  module or a changed NRM1 / FRC1 rule file). Invented-data tests: `python3 -m fsa.test_rules` (exit 0 iff all pass).
- Inputs read: the coordinator's AGENT_CONTEXT.md; FRC1 `RESEARCH_DECISION.md`, `TABLES.md`, the reviewer's `RULES.md`,
  `RULES_ER1.md`, `CLOSING_STATEMENT.md`; `frc/rules.py`, `frc/test_rules.py`, `nrm/rules.py`, the helpers of `nrm/test_rules.py`;
  card FA_PRC_v1; `nrm/fa/prc.py` (`route`, `update_book`, `set_tables`, `objective`, `lloyd`, `_chain`, `_protected_start`,
  `select_path`, `spend`); `frc/fa/route.py` (`tables`, `nested_c`, `fit_pipeline_c`, `nested_lookup`, `brackets_for`,
  `stage_paths`); the key structure (not the values) of FRC1 `fa/SELECTION_V0.json`; card FA_FSA_v2 (commit 91ba9a27e).
- **Known before freezing (disclosed).** (1) Every FRC1 published result (04ac9161f), including one protected partition per version,
  PRC_S2 0.6485 against HQ_S2 0.6489 and the twin 0.6555. (2) The coordinator's 02:48Z note: on an invented-data timing trace that is
  not a registered law, both solvers stalled after the first strengths and SV1 reached a strength-8 objective lower by about 0.0012
  on 6,000 invented rows. This is a surrogate-objective difference only. Under these rules it is evidence for nothing, and that law
  can never be a registered law (section 3.1). (3) The fsa copy of `frc/fa/route.py` changes `UNT_TAG` to `FSA_UNT_S{v}`, so a new
  UNT control is not FRC1's UNT_BR (section 7.2).

**What every rule here is.** A development rule on previously and adaptively used records. Nothing here is confirmation. No rule
changes an original limit. Versions 0 and 1 share every training row and the MODEL_SELECT people, so "both versions" is a
procedure-repeat requirement, never a replication. No FSA1 rule relabels an FRC1 record: FRC1's strict `INCOMPLETE` and post hoc
ER1 `ENGINEERING_REPAIR` both stand. **Sign convention:** every difference is revised minus original (or candidate minus
comparator) of a loss or an AUC, so a negative value favours the revised solver. Accuracy enters as a loss. Pair benefits from
`nrm.rules` keep their own convention (comparator AUC minus candidate AUC, positive favours the candidate).

## 0. Inherited unchanged

The original limits, the aggregate screen, recipient gates, point eligibility (G1), the reserve, the frontier rule and claims G4-G6
come from `nrm.rules` (RULES.json 84f18776..., `nrm/rules.py` d5df8686...). F4, reader targeting, the reader-budget reference rule
(`_d3`, `_d4`), `strongest_nonzero` and alias-path handling come from `frc.rules` (registered RULES.json a0dc0fc7...; `frc/rules.py`
11d3df7f..., the post-ER1 file). FRC1's `grid_validity` and `refit_effectiveness` are **not** reused: their +700 seed rule belongs
to FRC1 (section 6 replaces them prospectively). The test `t_inherited_unchanged_not_monkeypatched` checks all of this.

## 1. Tolerances (fixed now; each is used everywhere below)

| Name | Value | Meaning | Rationale |
|---|---|---|---|
| `OBJ_TIE` | 1e-9 nats | \|J_a − J_b\| ≤ 1e-9 is an exact objective tie | Coordinator's proposal, adopted. J averages ≤ 20,000 O(1) terms, so two implementations differ by about 1e-12, and the code's own acceptance uses 1e-12. This is a numerical tie band, **not** a materiality threshold. Materiality comes only from the usefulness / recovery evidence of sections 3-4 |
| `AUC_MATERIAL` | 0.001 | \|ΔAUC\| < 0.001: no material change. ΔAUC ≤ −0.001: material reduction | Coordinator's proposal, adopted. It is 1/20 of the 0.020 target and 1/15 of FRC1's minimum detectable pair benefit (about 0.015), so a smaller exact difference could matter to no later screen |
| `TASK_NO_WORSE` | 0.0005 | A task-loss increase below 0.0005 is "no worse". It applies to each of eight losses: income / occupation NLL, accuracy loss and summed Brier, and the men's and women's occupation Brier | Coordinator's proposal, adopted and **extended to the two within-SEX occupation Brier losses**: the four subgroup guards are original limits and the objective's guard targets them |
| `TASK_MATERIAL` | 0.001 nats | A total-NLL (income + occupation) reduction of at least 0.001 is a material task gain | New. The task-side alternative needs it. It is twice the no-worse band and 1/8 of the S2 occupation-NLL budget |
| `RECIPIENT_LIMIT` | 0.010 | A single-view AUC may rise by at most 0.010 | The inherited NRM1 recipient allowance |
| `Z` | 2 | Resolution multiplier | Inherited |
| `EPS` | 1e-12 | Numerical guard: a difference equal to a stated threshold up to rounding counts as equal to it | 0.5005 − 0.5 evaluates to 0.00049999999999994; without the guard, decimal thresholds would depend on representation |

Thresholds are inclusive where a rule says "at least" or "at most" and exclusive where it says "below".

## 2. Route D: an independently verified implementation defect

**Definition.** A defect is code behaviour that (i) contradicts a committed specification and (ii) changes a saved release or a
decision input of the real pipeline (`nrm/fa/prc.py` as imported by `frc/fa/route.py`, or `frc/fa/route.py` itself). The committed
specifications are card FA_PRC_v1 section 1, cards FA_FRC_v1 / FA_FRC_R1, and a function's own documented contract, for example
`lloyd`'s "monotone in the exact objective and rule-consistent", `_chain`'s "never worse than the twin at this strength" and
`protected_path`'s "each recorded state equals protected_book(target)".

**Not defects (documented intentional approximations, fixed now):**
- `SHRUNK_ROUTING_TABLES`: routing on 20-pseudo-count tables while acceptance uses the unshrunk plug-in information;
- `CELL_MEAN_CODEWORD_REFRESH`: codewords refreshed as cell means, although the guard also depends on the occupation codewords;
- `B1_EMISSION_DIFFERS_FROM_SEARCH_PROTOTYPES`;
- `SEVEN_TILT_FIRST_IMPROVEMENT_SEARCH`.

Changing one of these is a solver revision (routes E / S), never a defect.

**Per-claim ladder** (`defect_label`; first applicable):
1. `NOT_A_DEFECT_DOCUMENTED_APPROXIMATION`;
2. `NOT_A_DEFECT_NO_COMMITTED_SPECIFICATION`: the source is not one of the four above, or there is no quoted committed text;
3. `NOT_A_DEFECT_NO_CONTRADICTION`: no reproducible input shows the contradiction (invented data suffices);
4. `NOT_A_DEFECT_NOT_IN_REAL_PIPELINE`;
5. `NOT_A_DEFECT_NO_EFFECT`: the correction changes no MODEL_FIT person's partition, no emitted vector beyond 1e-12 and no decision
   input beyond 1e-12;
6. `DEFECT_UNVERIFIED`: the verifier has not independently reproduced both the contradiction and the change;
7. `DEFECT_VERIFIED_MATERIAL`: the partition of at least one MODEL_FIT person changes, or a discrete decision input changes (a
   selection, bracket, F4 pass, alias status), or an AUC input moves by ≥ 0.001, or a task input by ≥ 0.0005;
8. otherwise `DEFECT_VERIFIED_IMMATERIAL`.

**Route label** (`route_d`): `D_ADVANCES` iff some claim is `DEFECT_VERIFIED_MATERIAL`. `INCOMPLETE` iff none is and some claim is
unverified. `D_NO_MATERIAL_DEFECT` otherwise; an empty claim list means no defect was claimed.

**Consequences.** A verified defect never edits an earlier record. Corrected values go in a dated note that says whether any FRC1
label would change. **Rationale:** the list of approximations is fixed now so that "defect" cannot be stretched after a result,
and an immaterial defect is corrected and reported but does not justify a pilot.

## 3. Route E: exact-reference comparison on registered invented laws

**3.1 Registration.** The invented laws, the registered strengths (the grid 0.25-8) and the revised solver's code (card FA_FSA_v2)
are committed before any solver runs on any registered law. Every registered (law, strength) cell is reported exactly once, and an
unreported registered cell makes the route `E_INVALID`. Any law already run by a solver is excluded, including the 02:48Z
6,000-row timing trace.

**3.2 Validity of a cell (same class).** Both solvers use the same objective (`nrm.fa.prc.objective` unchanged), the same routing
family (`route`: constants free, tilt 0 ≤ t ≤ λ), the same capacity, teachers and start (the twin at λ = 0, then the registered
continuation). Deployability is checked by the unchanged `route` on every support point of the law. The original point must be
deployable, and every metric is evaluated exactly at the law level. If any of this is missing, the route is `E_INVALID`.

**3.3 Applicable reference.** The best deployable same-class point found by any registered method in that cell: the revision, or
an exhaustive or exact same-class reference. The original solver **matches** the reference iff J_O ≤ J_best + 1e-9. **Unrestricted
partition enumeration** (points outside the rule family) is reported as a gap and never advances. It is also never the applicable
reference: beating a restricted deployable model with an unrestricted one says nothing about the solver.

**3.4 Exact usefulness** (`usefulness`; Δ = revised − original under the law):
- RECOVERY: exact Bayes pair AUC Δ ≤ −0.001, each single-view Δ ≤ +0.010, and every one of the eight task losses Δ < +0.0005;
- or TASK: total NLL Δ ≤ −0.001, every task loss Δ < +0.0005, pair AUC Δ < +0.001 and single views ≤ +0.010.

The exact Bayes AUC of the emitted-value partition is the largest AUC any reader of that release can reach, so it is
reader-independent within functions of the release. Flag `OBJECTIVE_RECOVERY_MISALIGNED` when J falls but the pair AUC rises by
≥ 0.001.

**3.5 Label of one revision** (`route_e`; first applicable):

| Label | Condition |
|---|---|
| `E_INVALID` | 3.1 or 3.2 fails, or an improving cell lacks exact metrics |
| `E_REVISED_NOT_DEPLOYABLE` | the revision's point is not deployable in some cell |
| `E_REVISED_WORSE_SOMEWHERE` | J_R > J_O + 1e-9 in some registered cell |
| `E_ADVANCES` | some cell has J_R < J_O − 1e-9 with 3.4 usefulness, the revision is runnable on the source, and it gives a distinct deployable nonzero point (≠ the original's, ≠ the twin) on the cached MODEL_FIT statistics in both versions |
| `E_GAP_WITHOUT_RUNNABLE_REVISION` | as above but not runnable, or the reference beats the original and the revision does not |
| `E_NO_DISTINCT_SOURCE_POINT` | useful on the laws, but it reproduces the original (or the twin) on the source in some version |
| `E_OLD_MATCHES_REFERENCE` | the original matches the applicable reference in every cell |
| `E_PROXY_ONLY` | J improves in some cell, but nowhere with 3.4 usefulness |

**Rationale.** Exact evaluation has no sampling error, so significance is not required, as the brief says. The remaining risks are
selection over laws and adaptivity, and registration (3.1) controls them. A revision that is worse than the original anywhere is
not a better optimizer of the same objective: a revision warm-started at the original's state never is.

## 4. Route S: source-specific evidence (cached MODEL_FIT statistics and nested MODEL_FIT)

**4.1 Inputs and their status.** The full-pool cached MODEL_FIT out-of-fold statistics: the NRM1 task teachers and the P1
SEX-side teachers. **They are derived from MODEL_FIT training labels (income, occupation, SEX).** Every record must say so, and
never "label-free" or "untouched" (`label_derived_statistics_disclosed`). The route is `S_INVALID` if any of these fails:
- the card was committed before the run;
- MODEL_SELECT stayed unopened;
- the nested run's original-solver entries equal the replay's per-person outputs (card FA_FSA_v2 section 3).

**4.2 Objective (S-a).** At some registered strength, in both versions, J_R < J_O − 1e-9 on the full-pool fitted objective, with the
revised point deployable and distinct from both the original's and the twin's partitions.
- Not deployable at some strength: `S_REVISED_NOT_DEPLOYABLE`.
- Worse beyond 1e-9 at some strength: `S_REVISED_WORSE_SOMEWHERE`.
- No such strength: `S_OLD_MATCHES_REFERENCE`.

**4.3 Usefulness / recovery rationale (S-b), MODEL_FIT nested held-out, at the S2 selection.** The revised point R is the revised
solver's `select_path` choice at S2. No S2 choice in some version gives `S_NO_FEASIBLE_REVISED_POINT`. Per version:
- **Bracket** B = the cheapest original path point whose nested total-NLL spend over the twin is ≥ R's spend − 0.0005, or the
  most-spending original point if there is none.
- **Recovery:** in both versions, the nested held-out pair LOOKUP AUC Δ(R − B) ≤ −0.001, the total-NLL spend Δ(R − B) < +0.0005,
  and the income / occupation LOOKUP AUC Δ ≤ +0.010. Pooled over versions, mean ΔAUC + 2 × mean SE < 0, with the paired
  household-bootstrap SE (B = 400, `SeedSequence([2026102200, v, 4])`, the same weights for both solvers).
- **or Task**, against the original S2 point: in both versions the spend Δ ≤ −0.001, the pair AUC Δ < +0.001 and the single views
  ≤ +0.010. Pooled, mean Δspend + 2 × mean household-clustered SE < 0.

If R is an alias of the original S2 point in some version, neither rationale holds (flag `ALIAS_OF_ORIGINAL_AT_S2_V<v>`). The
mean of the version SEs is the pooled SE that is valid under any correlation of the versions (they share rows).

**4.4 Label:** `S_ADVANCES` iff S-a and S-b hold; `S_PROXY_ONLY` iff S-a holds and S-b fails. This is the coordinator's case: the
objective falls, but the nested held-out recovery and task metrics do not improve, and **the route closes**
(`t_route_s` checks this with the 0.0012 trace numbers).

**Rationale.** Unlike invented laws, these estimates are noisy. The rule has the structure of the inherited aggregate screen: a
materiality threshold that must hold in both versions, plus one resolution condition on the version-pooled mean. Point thresholds
alone would let noise advance a null solver. Separate 2-SE resolution in each version would add little evidence, because the
versions share rows.

## 5. The advancement decision and the closing rule (`advancement`)

5.1 The decision is taken only on a complete record: while any attempted route is `INCOMPLETE` or `*_INVALID`, the label is
**INCOMPLETE**. Then **ADVANCE** iff some attempted route advances. **OPTIMIZER_REPAIR_CLOSED** iff none advances, at least one
route was attempted, and every attempted route reached a closing label (D: `D_NO_MATERIAL_DEFECT`; E and S: every label in sections 3.5 and 4 except
`*_ADVANCES`, `*_INVALID` and `INCOMPLETE`). The closing statement: "the optimizer-repair hypothesis closes: the original solver
matches the applicable reference, or a revised solver improves only the surrogate objective without a usefulness / recovery
rationale". A route can be withdrawn only by a dated record after an invalid evaluation; it then counts as not attempted and is
flagged. Statistical significance on invented laws is never required.

5.2 **Source evidence supersedes invented laws for the same revision.** If route S is attempted, then:
- an `E_ADVANCES` of a revision whose S label is closing becomes `E_SUPERSEDED_BY_S`;
- if that revision's S label is incomplete, it becomes `INCOMPLETE`.

Invented laws stand in for the source, and once the source has been measured on MODEL_FIT, the measurement governs. Route D
advances independently of E and S.

5.3 **Equal-effort control.** SV1_first (card FA_FSA_v2) is evaluated with the same functions and reported. It **never** advances,
even if its own label would. `equal_effort_attribution` reports, per strength and version, whether the finer candidate set alone
helps (`control < original`) and whether best-improvement selection adds to it (`SV1 < SV1_first`). This attribution is
descriptive and never a decision input.

5.4 **Revisions.** At most two registered variants (SV1, and SV2 only by a dated amendment card with its own control). Every
registered variant is evaluated on every attempted route, and nothing else. The pilot solver is the first registered variant
that advances by E or S, with every verified material defect corrected. If only D advances, it is the corrected original solver.

## 6. Reader validity of any new (FSA1) reader grid (`reader_validity`; prospective, never applied to FRC1)

`GRID_VALID` iff all of the following hold (claims need `grid_valid` recorded and true):
- (a) **Seeds.** Recorded distinct integers S<v>k<k> = MODEL_SEED_BASE + **800** + 10v + k (2026101205 + 800 + ...). None equals an
  earlier grid's seed, MODEL_SEED_BASE + {300, 400, 500, 600, 700} + 10v + k with k = 0..9 (100 forbidden seeds). Requiring the
  registered values, not just new ones, prevents re-running with fresh seeds after an outcome.
- (b) **Splits.** Each repetition has:
  - the tag `FSA1_devgrid_R<r>`;
  - an assignment hash recorded and distinct from every other repetition's (household fitting splits vary);
  - pairwise SCORE household overlaps recorded in [0, 1);
  - a FIT / SCORE household overlap of 0;
  - the bootstrap seed 2026102200 + 100v + repetition index (R1 = 0, R2 = 1, ...).
- (c) **Recorded checks.** Schema, access and score checks are recorded, and each has an empty error list. An unrecorded check fails.
- (d) **Stochastic recipes.** A view is *large-support* iff its recorded label-free distinct-input count is ≥ **2,048**: above the
  1,024 joint values a 16 × 64 release can take, and below the thousands of distinct vectors of a continuous release.
  - A **seed-implementation error** is a seed-dependent recipe (ET5, ET25, MLP64, MLP6432, OUT_MLP, LOG_MLP) that is equal between
    its two seeds on a large-support view, or that has no large-support check in some (version, repetition) block (untestable).
  - An equality on a small-support (finite) view is **expected seed invariance** (the ER1 mechanism). It never invalidates, and
    its duplicate cells count once: `nrm.rules.aggregate` already computes n_eff over distinct cell values.
  - A missing distinct-input count fails.
  - The record lists, per recipe and support class, how many checks were made and how many differed (`stochastic_recipes`).

A deterministic (seed-invariant) evaluation classifier is a legitimate reader. One that differs between seeds is only flagged
`NONDETERMINISTIC_RECIPE`. **Rationale:** FRC1's strict rule failed on exactly this expected invariance, and the repair (ER1) was
post hoc. This rule fixes the treatment in advance, and still fails genuine seed, split, schema, access and score errors. The
refit diagnostic (`refit_effective`) applies the same rule to repetitions R1-R4 and seeds k0-k3.

## 7. The pilot (only after ADVANCE; `pilot` returns `PILOT_NOT_AUTHORIZED` otherwise; its own card is committed before its fits)

7.1 **Operating path** (`operating_path`). Per version and solver, the path is the identity (TWIN) plus the distinct `select_path`
selections at S1, S2 and S3 on MODEL_FIT nested estimates: at most three nonzero points. Selections with identical MODEL_FIT outputs
count once, and one identical to TWIN is an alias of the identity. The primary point is the S2 selection. Points chosen on
anything but MODEL_FIT nested estimates are `INVALID`.

7.2 **Comparators.**
- The original solver's PRC and HQ: the saved frc PRC_S2 / HQ_S2 may be reused.
- The revised solver's PRC and HQ, with the same variant, the TWB_C64 initialization, the same settings and the same effort rules.
- TWB_C64.
- One untargeted or ordinary-compression control path, registered in the pilot card with its MODEL_FIT bracket rule. A new UNT
  control uses `FSA_UNT_S{v}` and is not FRC1's UNT_BR.
- The nine admitted references.

7.3 **Task feasibility (C2).** `nrm.rules.point_eligibility` under the original limits: accuracy loss ≤ 0.010, NLL excess ≤ 0.010,
summed Brier excess ≤ 0.005, all 24 clauses and the 4 occupation subgroup guards. The two-SE reserve is reported, never gating
(flag `POINT_FEASIBLE_NOT_RESERVE_QUALIFIED`). The 0.004 capacity gate is not reopened and no function computes it.

7.4 **Comparisons.**
- All comparisons are paired on MODEL_SELECT with household-clustered uncertainty (the inherited aggregate screen) and use measured
  frontiers at matched spend (`nrm.rules.frontier_comparison`; the bracket is the cheapest control that spends at least as much).
- Differing task costs are reported directly.
- `revised_vs_original` compares the revised primary point with the original path [TWIN, original PRC points]:
  - `RECOVERY_GAIN_AT_MATCHED_SPEND` iff it dominates that frontier;
  - else `TASK_GAIN_AT_NO_WORSE_RECOVERY` iff, against the original primary point, the pooled paired total-NLL cost is resolved
    negative (value + 2 se < 0), the pair benefit is > −0.001 and the recipient gate passes;
  - else `NO_MEASURED_GAIN_OVER_ORIGINAL`.

7.5 **Claims (separate labels; none implies another).**

| Claim | Rule |
|---|---|
| C1 better optimization | `BETTER_OPTIMIZATION` iff in both versions J_R < J_O − 1e-9 at the revised primary point's strength (full-pool cached MODEL_FIT statistics) and J_R ≤ J_O + 1e-9 at every registered strength; `MIXED` if only the first holds; else `NOT_BETTER`. It concerns the surrogate on label-derived training statistics, never usefulness |
| C2 useful task performance | `TASK_FEASIBLE` (7.3) |
| C3 protection beyond matched controls | revised PRC against TWIN, the equally optimized revised HQ path (`HQ_FRONTIER`) and the control path (`CONTROL_FRONTIER`). Conjuncts: (a) G1; (b) TWIN `nrm.rules.comparison` passes; (c) `DOMINATES_CONTROL_FRONTIER` on both paths (an empty path is compared at TWIN, flag `<path>_AT_IDENTITY`); (d) F4 of the revised point (`frc.rules.f4`, both versions); (e) reader targeting against TWIN and every bracket (`frc.rules.targeting`). Labels in order: `NOT_COMPUTED_GRID_*`, `NOT_ASSESSED_NO_FEASIBLE_POINT`, `NOT_ASSESSED_EXACT_ALIAS` (= twin), `NOT_ASSESSED_ALIAS_OF_ORIGINAL`, `NOT_ASSESSED_UNEQUAL_EFFORT` (the HQ settings hash differs, or HQ stopped on a cap while PRC converged), `INCOMPLETE`, `NOT_MET_F4`, `READER_TARGETED`, `PROTECTION_BEYOND_MATCHED_CONTROLS` (met) / `MET_WITHOUT_TASK_FEASIBILITY`, `NO_REDUCTION_BEYOND_ANY_CONTROL`, `PARTIAL_<passed>_ONLY`. The record gives the minimum detectable benefit (2 × the largest bracket se_agg) |
| C4 strong references | NRM1 G4 (all nine point-eligible references, recipient gates) with the FRC1 reader-budget rule (`frc.rules._d3`) |
| C5 0.020 margin | NRM1 G5 with the same rule (`frc.rules._d4`); flag `MATERIAL_0020_MARGIN_MET` |
| C6 joint design | `NOT_ASSESSED_MISSING_CONTROLS` unless the separate design and both sequential orders exist (then NRM1 G6) |
| C7 novelty | `NOT_CLAIMED`: a better line search is not a novel privacy algorithm, and the objective has named prior art (privacy funnel, IBSI, FARE) |

7.6 **Outcome class** (`pilot`; first applicable):
1. `INCOMPLETE`: C2, C3 or C4 incomplete or not computed for grid validity;
2. `METHOD_CANDIDATE`: C2, C3 and C4 met;
3. `COMPONENT`: C2 and C3 met, C4 not;
4. `ENGINEERING_RESULT`: C3 not met, C1 and C2 met, and `revised_vs_original` shows a measured gain (recovery at matched spend, or
   task at no worse recovery). A lower objective without a measured gain over the original solver is **not** an engineering
   result;
5. `INCOMPLETE`: C1 or the revised-versus-original comparison incomplete;
6. `BOUNDED_NEGATIVE_RESULT` otherwise, stated with its bound (the minimum detectable benefit).

Flags:
- `NOT_ATTRIBUTABLE_TO_BETTER_OPTIMIZATION` (C3 met, C1 not);
- `C4_WITHOUT_C3`;
- `MATERIAL_0020_MARGIN_MET`;
- `POINT_FEASIBLE_NOT_RESERVE_QUALIFIED`.

7.7 **Two versions.** Every "both versions" condition is a procedure repeat on shared rows, never independent replication. The
scope sentences of NRM1, plus three FSA1 sentences, go in every record.

7.8 **Diagnostics, never decision inputs** (`diagnostics`).
- For the strongest feasible nonzero point (`frc.rules.strongest_nonzero`, unchanged): one component ablation (registered in the
  pilot card; for SV1 the natural one is SV1_first) and the effective comparator refits (`refit_effective`).
- They run even if the first intervals cross zero.
- `pilot` never reads them, and a test shows that adding them changes no label.

## 8. P1 replay acceptance (`p1_replay`)

The coordinator reruns FRC1's nested P1 estimator unchanged and saves per-person held-out outputs; the verifier recomputes from
them.
- **Estimator identity.** `nrm/fa/prc.py` must be byte-identical (1192f34d...), and the P1 functions (`nested_c`, `fit_pipeline_c`,
  `chain_inputs`, `tables`, `apply_c`, `emit`, `route_any`, `entry_embedding`, `shuf_embedding`, `_plan`, `total_spend`,
  `brackets_for`, `_wauc_draws`, `nested_lookup` and the selection loop of `stage_paths`) must be textually identical to
  `frc/fa/route.py` 933d0d6f..., with the recorded plan, construction, capacity, T_fixed, folds and the old bootstrap seed
  2026102100. The verifier checks this. Otherwise the label is `P1_ESTIMATOR_CHANGED`.
- **Reproduction** (coordinator against the recorded `SELECTION_V<v>.json`). Every number in `nested_means`,
  `audit_spend_over_twin`, `brackets`, `nested_lookup_pair`, `F4` and `selection` must agree within **1e-12** absolute, and every
  discrete output (selections, brackets, F4 pass and entry) must be equal. Text fields are not compared. **Rationale:** the code is
  deterministic on one thread; any person routed differently moves a mean by far more than 1e-12. The coordinator's proposal is
  adopted.
- **Verification** (verifier against the replay). It covers at least the spends and clustered SEs, the selection, F4 and the
  nested LOOKUP AUCs and bootstrap SEs, with the bootstrap redrawn from the registered seed. Numbers must agree within **1e-9**
  absolute: independent code changes the summation order (about 1e-12), and 1e-9 is still far below any decision-relevant
  difference (≥ 1e-4). A discrete difference is a disagreement unless a recomputed quantity lies within 1e-9 of its decision
  threshold (estimate + 2 SE against a budget, diff − 2 SE against 0, a spend against the bracket target). Then it is reported as a
  `BOUNDARY_TIE`, and the frozen code's value stands.
- **Labels** (worst over versions):
  - `P1_REPLAY_VERIFIED`: the replayed outputs may serve as the original-solver baseline, and FRC1's verifier-scope gap closes for
    those quantities;
  - `P1_NOT_REPRODUCED`: the replayed outputs are not decision inputs; route S then uses the nested stage's own original entries
    (same folds and teachers); a dated diagnosis precedes any route-S decision; and FRC1's P1 record is reported as not reproduced;
  - `P1_VERIFICATION_DISAGREES`: the outputs are not decision inputs until a dated diagnosis;
  - `INCOMPLETE`.

**A mismatch never changes an FRC1 label.**

## 9. Challenge: threats and what the protocol must state

1. **Multiplicity.**
   - The decision space spans three routes, two variants plus a control, six strengths, two versions, several laws, three budgets
     and seven claims.
   - The protocol must say that no family-wise error control is claimed, and that every registered cell, strength and variant is
     reported (route E: 3.1; route S: all strengths; section 5.4).
   - Each pilot label inherits 2-SE screens without adjustment, and the record states the minimum detectable benefit.
   - No route, law or budget may be chosen after its result.
2. **Adaptivity.**
   - SV1 was designed after FRC1's saturation (one partition per version) and revised (FA_FSA_v1 to v2) after an invented-data
     trace.
   - The protocol must date every solver change. It must state that SV1's proposal class was chosen with knowledge of FRC1 results
     and of one invented trace, and that MODEL_FIT and MODEL_SELECT were already used adaptively by every earlier campaign.
   - Any change after a registered-law, source or nested result is a new variant (amendment card, counted against the limit of
     two, disclosed as post hoc).
3. **Surrogate against recovery.** J uses the fitted propensity's plug-in information on label-derived training statistics. It is a
   log-likelihood surrogate of a LOOKUP reader, not AUC, not held-out and not the grid readers. The protocol must:
   - never report a lower J as protection or usefulness;
   - report for every point the fitted π information next to the held-out label LOOKUP AUC (the divergence);
   - state the coordinator's example plainly: a strength-8 objective 0.0012 lower with unchanged recovery is a proxy-only result,
     and under 4.4 / 5.1 it closes.
4. **Unequal effort.**
   - SV1 evaluates more candidates than the original. The protocol must record per chain the CPU time, route and objective
     evaluations, accepted steps and stopping reason.
   - It must report SV1_first as the equal-effort decomposition (5.3).
   - It must require revised PRC and revised HQ to share variant, initialization, settings and stopping rules (C3 otherwise
     `NOT_ASSESSED_UNEQUAL_EFFORT`).
   - A gain that SV1_first already achieves is a candidate-resolution gain, not a selection gain. Either way it is an optimizer
     gain, never a privacy mechanism.
5. **HQ restrictions.**
   - HQ's routing replaces π, qO0 and qO1 by their twin-cell means (`restrict_embedding`, inside `route`), but its tables
     (`set_tables`) use the person-level π, and its acceptance objective (`objective`) uses the person-level π, qO0 and qO1:
     `_prepare` and `frc.fa.route.chain_inputs` pass the unrestricted embedding to `lloyd`.
   - HQ is therefore the same objective searched over a restricted, q-measurable rule family. It is neither "the IBSI objective" nor
     "all of h(q)".
   - The protocol must describe HQ exactly so, apply SV1 to HQ identically, and never call a PRC-HQ difference "information
     propensity adds".
6. **Expressivity.**
   - If the task scores q(x) = (qI, qO) identify the support states (distinct six-field states have distinct q), every partition
     PRC can produce, a function of (q, π, qO0, qO1), is also a function of q. Card FA_PRC_v1's claim that "h(q) cannot reach
     partitions that separate equal-q people" is then vacuous.
   - The protocol must record, label-free on MODEL_FIT features, the number of distinct support states and the share of people
     whose q is shared with another support state.
   - It must state that an unrestricted q-mapping could match any PRC partition on that support. A C3 advantage over HQ is then
     about rule parametrization and search inside two restricted deployable families, not about the information in propensity.
7. **Invented laws.**
   - Laws written by the team can favour the variant. Route E therefore needs registration before any run (3.1), and route S
     supersedes it for the same variant (5.2).
   - The exact Bayes AUC is reader-optimal only within functions of the release. On the source, the readers are trained and
     data-limited (FRC1 TABLES section 9), so measured recoveries are lower bounds.
8. **Saturation and matched spend.**
   - In FRC1 the S2 budget did not bind and every strength 0.25-8 gave one partition, so comparisons at "the same budget" were
     comparisons at the top strength.
   - Route S therefore brackets by measured nested spend (4.3), and the pilot uses frontiers at matched measured spend (7.4).
   - The protocol must report realized spend, never budget labels, as the matching variable.
9. **Uncertainty under refitting.** Household-clustered SEs treat the refitted pipelines as fixed, and the LOOKUP bootstrap is
   conditional on the fitted tables. The protocol must state that both understate uncertainty, as FRC1's rules section 11 did.
10. **Winner's curse.** The strongest nonzero point and the pilot variant are chosen by rule. Diagnostics on them are descriptive
    (7.8), and the protocol must not present the selected point's benefit as unbiased.

## 10. Required records (the coordinator's inputs; field names as in the `fsa/rules.py` docstrings)

| Rule | Function | Record |
|---|---|---|
| D | `route_d` | defect claims, each with location, specification source and quote, approximation name or None, demonstration, effect counts and the verifier's confirmation |
| E | `route_e` | registration flags and cells per law and strength: same-class flags, original / revised / reference / unrestricted points with exact AUCs and eight task losses, runnable flag, source-distinct flags |
| S | `route_s` | disclosure and registration flags; fitted J per version and strength with deployability and distinctness; nested S2 selections, the original path with spends, paired differences (AUC of three views, spend) with SEs |
| Advancement | `advancement` | registered variants, attempted / withdrawn routes, the SV1_first control records and the source objectives of original / SV1 / SV1_first |
| Reader validity | `reader_validity` | seeds, seed checks, distinct-input counts, splits (tags, hashes, overlaps), bootstrap seeds, schema / access / score checks |
| Pilot | `pilot` | as listed in its docstring, including the effort records |
| P1 replay | `p1_replay` | recorded SELECTION_V<v>.json, the replay in the same structure with `estimator_unchanged` and `per_person_outputs_saved`, and the verifier's recomputation |

## 11. Not settled here

- Which invented laws, and how many, are registered: the protocol's choice, fixed before any run (3.1). These rules only require
  that they exist, are reported in full and exclude the trace.
- The pilot's single control path (UNT or ORD) and its bracket rule: the pilot card's choice, made on MODEL_FIT only.
- The large-support threshold 2,048 assumes the 16 × 64 protection capacity. A pilot at 24 × 128 (up to 3,072 joint values) would
  class its pair view as large-support. That is conservative (an extra-trees equality there would fail), and an amendment would
  be dated.
- With two shared-row versions, the pooled 2-SE condition of 4.3 is a heuristic guard, not an error-rate guarantee.
- I did not review the solver code (`fsa/fa/solver.py`, `solve.py`) or the protocol draft. The verifier must check SV1's claimed
  rule-consistency.

## 12. Completion record

- Started 2026-10-21T02:41:05Z (`date -u`). The completion time is the `date -u` reading in the hand-back report, taken after the
  last file was written and the CPU charge recorded.
- `python3 -m fsa.test_rules`: the result and the SHA-256 of RULES.md, RULES.json, `fsa/rules.py` and `fsa/test_rules.py` are in the
  hand-back report (this file cannot contain its own hash).
- Inherited files unchanged at writing: NRM1 RULES.json 84f18776..., FRC1 RULES.json a0dc0fc7..., `nrm/rules.py` d5df8686...,
  `frc/rules.py` 11d3df7f..., `ohn/rules.py` 8686a78f..., `nrm/fa/prc.py` 1192f34d..., `frc/fa/route.py` 933d0d6f....
- No label was read and no real-data computation was run.
