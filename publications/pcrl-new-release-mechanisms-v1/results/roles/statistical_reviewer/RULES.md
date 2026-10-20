# Generic claims and selection rules of the new-release-mechanisms campaign (NRM1; statistical reviewer)

**Status.** Writing started 2026-10-20T14:45:00Z (`date -u`). The completion time is recorded at the end of this file. Written
before any candidate outcome of this campaign: I read no label, ran no real-data computation, fitted nothing and called no loader
that reads labels.
- Machine-readable constants: `RULES.json` (schema `NRM1-rules-v1`). Implementation: `nrm/rules.py` (pure functions; reads only
  this RULES.json and, through the imported `ohn.rules` / `tsz.rules`, their unchanged rule files). Invented-data tests and the
  known-law screen checks: `python3 -m nrm.test_rules`, which writes `SCREEN_CHECKS.json` beside this file.
- Inputs read: the coordinator's AGENT_CONTEXT.md; the TSZ1 RULES.md / RULES.json / ADJUDICATION_SEEDCHECK.md; `tsz/rules.py`,
  `tsz/test_rules.py`, `tsz/claims.py`, `tsz/devreaders.py`; `ohn/rules.py` and the OHN1 SELECTION_RULE.md / RULES.json; the
  docstrings and interfaces of `nrm/devreaders.py` and `nrm/evaluate.py` (no outputs); the published TSZ1 records
  `conditional/COND_UTILITY.json` and `size/SIZE_RESULTS.json` (point and reserve status of the references only).
- **Disclosure.** The published reference statuses (section 3) and the last grid's pair AUC means (AGENT_CONTEXT) were known when
  these rules were written. No outcome of this campaign existed.

**What every rule here is.** A development rule on previously and adaptively used records. MODEL_SELECT was used by every earlier
campaign, so nothing here is confirmation. No rule changes an original limit. The two-SE reserve is a development heuristic, not a
cap or a population guarantee. Every uncertainty statement is asymptotic and model-based (normal approximations with conservative
variance bounds), never a finite-sample or distribution-free guarantee. "Recovery" always means SEX recovery by the declared
development readers. Lower recovery for these readers is never protection against every reader.

## 0. Original limits (unchanged)

| Condition | Limit | Count |
|---|---|---|
| Accuracy loss, NLL excess (nats), summed Brier excess vs U0_S<v> and Ucal_S<v>, income and occupation | 0.010, 0.010, 0.005 (inclusive) | 24 clauses = 2 tasks x 2 references x 3 metrics x 2 versions |
| Within-SEX occupation summed-Brier excess vs U0_S<v>, men and women | 0.005 each (inclusive) | 4 guards |
| Recipient non-worsening (OHN1 `original_limits.recipient`; TSZ1 section 0): each single output's SEX-recovery AUC minus the comparator's, income view and occupation view | <= 0.010 | per comparator, view and version |
| Material target (pair benefit vs NEURAL_RAW and H_LJ) | 0.020 AUC | separate from smaller positive improvements |

A version v = training seeds and internal folds of version v, evaluated against U0_S<v> / Ucal_S<v>. **The two versions share every
training row and the MODEL_SELECT people. They are procedure repeats, never independent replications.**

## 1. Generic claims (kept distinct; `nrm.rules.claims`)

No claim implies another. Each is computed and reported for every configuration.

| Claim | Label if met | Exact condition |
|---|---|---|
| **G1** task feasibility | `TASK_FEASIBLE` | The complete release, on its actual emitted outputs, passes all 24 clauses and 4 guards at the point. Randomized releases use draw-averaged values (section 9). The reserve is reported beside G1 and is never part of it. |
| **G2** incremental recovery reduction beyond matched controls | `INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS` | G1 holds, and against BOTH matched controls the pair benefit is `AGG_POSITIVE` and the recipient gate passes (both views, both versions; section 5). The two matched controls are (a) the family's equally trained task-only twin and (b) the simplest alternative explanation declared in the mechanism card before the fit. |
| **G3** stable utility-recovery frontier improvement | `STABLE_FRONTIER_IMPROVEMENT` | G1 holds, and the actual selected output is `DOMINATES_CONTROL_FRONTIER` against the family's SEX-blind control path (section 7). |
| **G4** original-benchmark superiority | `SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES` | G1 holds, and against EVERY point-eligible reference of section 3 the pair benefit is `AGG_POSITIVE` and the recipient gate passes. Intersection-union: each comparison must pass on its own, so no multiplicity adjustment is needed for the conjunction. The old every-cell profile is reported beside it (`old_rule_profile`). |
| **G5** the 0.020 material criterion | `MATERIAL_0020_RESOLVED` | G1 holds; `TARGET_AGG_RESOLVED` against NEURAL_RAW and against H_LJ; the recipient gate passes against both. The other labels are: `TARGET_RESOLVED_RECIPIENT_GATE_FAILED` (target resolved against both, but a recipient gate against one of them fails); `MATERIAL_0020_POINT_ONLY` (aggregate mean > 0.020 against both, not resolved); **`POSITIVE_BELOW_MATERIAL_TARGET`** (AGG_POSITIVE against both, target not met), reported as a smaller positive improvement; and `NOT_MET`. Without G1, any level above NOT_MET becomes `MET_WITHOUT_TASK_FEASIBILITY` (the level is kept in `level`). G5 is never nested in or implied by G4; G4 is shown beside it. |
| **G6** joint-design contribution | `JOINT_CONTRIBUTION_SUPPORTED` | Only if both outputs change and are jointly designed. It also requires the matched separate design and both sequential orders (income first, occupation first). Against each of these three: AGG_POSITIVE benefit, recipient gate, and a primary cost (section 7) that is not resolved higher. Checked in order: (1) if fewer than two distinct outputs change, `NOT_APPLICABLE_ONE_OUTPUT_CHANGED`, with the sentence "only the <output> output changes; no joint-design claim"; (2) if both change but are not declared jointly designed, `NOT_APPLICABLE_NOT_JOINT` (also used when no joint record exists); (3) missing controls give `NOT_ASSESSED_MISSING_CONTROLS`; (4) otherwise `JOINT_CONTRIBUTION_SUPPORTED` or `NOT_SUPPORTED`. |
| **G7** novelty | from a separate prior-art review | Copied verbatim from that review (`NOT_CLAIMED` by default; also `NOVELTY_REVIEW_PENDING`, `NOVEL_PER_SEPARATE_REVIEW`, `PRIOR_ART_FOUND`). **Never inferred from any performance label**; any other value is read as `NOT_CLAIMED`. |

- **Useful component (always visible).** `USEFUL_COMPONENT_NOT_COMPETITIVE` iff G2 holds and G4 does not. A smaller positive
  development result is reported with this label, never as a competitive result.
- **Protection without task feasibility.** `MET_WITHOUT_TASK_FEASIBILITY` applies to G2-G6 whenever the comparisons pass but G1
  fails. It is never claimable.
- Failing labels name their reason: `NOT_MET`, `NO_MATCHED_CONTROLS`, `NOT_SUPERIOR` (failing references listed), `INCOMPLETE`
  (a required comparison missing), and the section 7 labels for G3 (`NOT_SEPARATED_FROM_TASK_SPENDING` with its reason,
  `SEPARATED_ON_PRIMARY_COST_ONLY`, `NOT_ASSESSED`).

## 2. Scope sentences (attached to every claim record)

- "Lower recovery for the declared development readers is not protection against every reader."
- "MODEL_SELECT is adaptively used: development evidence, never confirmation."
- "Versions share training rows; repetitions share households; reader recipes and overlapping splits are not independent datasets."
- "A reserve-frontier result is not superiority over every historical method."
- "Novelty is never inferred from performance."

## 3. Strongest admitted reference set (frozen; RULES.json `references`)

**Eligibility rule.** A reference is eligible iff all 24 clauses and 4 guards pass at the point on both versions, from the
published MODEL_SELECT values (OHN1 / TSZ1 rule). Reserve, recovery and any outcome of this campaign play no role.
- `nrm.evaluate` re-evaluates the frozen wires. The result is reported beside the published label, and a disagreement is flagged
  `REFERENCE_POINT_STATUS_DISAGREES`.
- The published label governs eligibility. It is never changed after an outcome.

| Reference | What it is | Published point label | Published whole-release reserve (min ratio v0 / v1) |
|---|---|---|---|
| NEURAL_RAW | neural release | POINT_ELIGIBLE | RESERVE_NOT_MET (1.947 / 1.558) |
| H_LJ | LOCAL income + JOINT_TREE occupation | POINT_ELIGIBLE | RESERVE_NOT_MET (0.321 / 0.624) |
| B2TWIN | 6-level LOCAL income code + JOINT_TREE | POINT_ELIGIBLE | RESERVE_NOT_MET (0.321 / 0.624) |
| RETIRED_B3CTL | JOINT occupation + 6-level income code | POINT_ELIGIBLE | RESERVE_NOT_MET (0.321 / 0.349) |
| SUCC | H3 income + JOINT_TREE | POINT_ELIGIBLE | RESERVE_NOT_MET (0.321 / 0.624) |
| TWJ | HC_STRONG income + JOINT_TREE | POINT_ELIGIBLE | RESERVE_NOT_MET (0.321 / 0.624) |
| W | A1_PO_16 (100%) + H3 income | POINT_ELIGIBLE | RESERVE_MET (3.511 / 2.862) |
| TR100 | TASK_REFIT (100%) + H3 income | POINT_ELIGIBLE | RESERVE_MET (3.412 / 2.862) |
| TT100 | T_TREE (100%) + H3 income | POINT_ELIGIBLE | RESERVE_MET (3.326 / 2.600) |

- **All nine are point-eligible, so all nine form the original benchmark of G4.**
- The six inherited references fail the reserve heuristic and stay in the benchmark anyway.
- Target references for G5: NEURAL_RAW and H_LJ (unchanged).

**Diagnostic or ineligible (never in the original benchmark).**
- PUB_IBSI: `DIAGNOSTIC_POINT_INELIGIBLE` (published seed-1 men's guard 0.0050142 > 0.005).
- U0 and Ucal: `UTILITY_REFERENCE_ONLY` (they define the clauses and are not recovery comparators).

**Known alias structure (stated before any outcome; verified by wire hash, `alias_groups`).**
- H_LJ, B2TWIN, SUCC and TWJ share the JOINT_TREE occupation wire.
- SUCC, W, TR100 and TT100 share the H3 income wire.
- Each group is ONE single-view comparison: reported once and counted once in every count (`distinct_comparators`).
- A candidate carrying H3 income is an exact alias of SUCC / W / TR100 / TT100 on the income view. Its income recipient
  difference is then exactly 0 (`ALIAS_ZERO`: passes non-worsening, never a benefit).

## 4. Original point eligibility and the reserve-qualified frontier

**Point eligibility** (`point_eligibility`):
- 12 clauses + 2 guards per version, inclusive, values from `ohn.rules.conditions` (refuses re-capped or malformed input), both
  versions, for 24 + 4.
- `POINT_ELIGIBLE` / `POINT_INELIGIBLE` (failing conditions listed) / `INCOMPLETE`.
- Randomized releases: draw-averaged values. Each draw's own point status is reported as `worst_draw`, never gating.
- Recipient eligibility uses the original 0.010 allowance through the aggregate gate of section 5. The old every-cell gate is shown
  beside it.

**Reserve** (`reserve_status`):
- `ohn.rules.reserve`, imported unchanged: k = 2; se_used = max(own SE, H_LJ floor SE x sqrt(5306 / G)); margin >= 2 se_used,
  inclusive; 28 conditions.
- It is applied IDENTICALLY to candidates, controls and references.
- For randomized releases the own SE is sqrt(se^2 + se_mc^2) before flooring. The reserve without the MC term is reported beside.
- Qualified = `RESERVE_MET` on both versions.

**Reserve-qualified frontier** (`reserve_frontier`, separate from the original benchmark):
- Points: every configuration and every reference, each with its reserve qualification and pooled registered-cell pair AUC.
- The Pareto frontier of qualified points uses lower pair AUC and higher minimum ratio.
- A candidate is `RESERVE_FRONTIER_IMPROVEMENT` iff it is qualified and its pair benefit is `AGG_POSITIVE` against every qualified
  control and reference.
- Every point-eligible reference stays in the original benchmark whether or not it meets the reserve. Failing references are listed
  (`references_failing_reserve`), never dropped.
- **A reserve-frontier result is not superiority over every historical method.**

## 5. The aggregate paired screen (prospective; `aggregate`, `benefit_screen`, `target_screen`, `recipient_screen`)

### 5.1 Cells and blocks

- **Inputs.** The grid's `paired()` output per (candidate, comparator, view): cell values (pair benefit b = comparator AUC -
  candidate AUC; recipient difference r = candidate - comparator) and paired household-bootstrap cell SEs (B = 400; within a
  (version, repetition) every release shares the same weights). Randomized releases also carry `cell_se_mc`.
- **Decision cells.** The 8 registered cells `S<v>R<r>k<k>` (v in {0, 1}, r in {1, 2}, k in {0, 1}). Every configuration has them.
  - Robustness cells (R3, R4, k2, k3) give a separate label, `ROBUSTNESS_CONFIRMED` / `ROBUSTNESS_NOT_CONFIRMED` / `NOT_RUN`
    (AGG_POSITIVE on the robustness cells alone and on all cells). It never changes a decision or a ranking.
  - Learning-curve cells (`@`) never enter a screen.
- **Blocks.** b = (version, repetition). m_b = mean of the block's reader-seed cells. Aggregate x = mean of the block means.
  Per-version mean = mean of that version's block means.

### 5.2 Uncertainty (it respects shared households and shared readers)

se_agg = max(sqrt(se_hh^2 + se_rs^2 + se_mc^2), 0.001), except for an exact alias. Its three components:

- **se_hh** = sqrt(sum_c w_c s_c^2), where w_c is the cell's weight in x (the RMS of the paired bootstrap SEs).
  - Because sqrt(sum w s^2) >= sum w s >= SD(sum w e), this is an upper bound on the SCORE-household sampling SE of x for ANY
    correlation between cells.
  - That covers cells sharing households across versions, reader seeds and overlapping repetitions; the per-block bootstrap weights
    cannot represent those correlations.
- **se_rs** = sqrt(sum_b (m_b - x)^2 / n_blocks): the version, split, interaction and reader-seed part.
  - Model the block means as version + split + interaction + noise, with the split shared by both versions because the split tag
    does not depend on the version.
  - Then E[se_rs^2] equals the version and split contributions to Var(x) when each factor has 2 levels, and exceeds the interaction
    and within-block contributions (3/4 against 1/4 with 4 blocks).
  - So the estimator is unbiased or conservative for every variance component, but rests on few degrees of freedom (see 5.6).
- **se_mc**: the same RMS bound over cell MC SEs (section 9).
- **The 0.001 AUC floor** binds only for near-identical releases, whose paired SEs shrink towards 0. This is the same reason the
  reserve floors its SE. With the floor, a near alias needs a benefit above 0.002 to resolve. 0.001 is one tenth of the recipient
  allowance.
- **Exact alias**: every cell value and SE is exactly 0. The comparison is `ALIAS_ZERO`: one comparison, counted once, never
  positive.
- **Shared readers.** The cache key is the wire hash, so an identical view reuses identical readers. Comparator views with
  identical wires form one alias group. A candidate's comparisons against different references share its readers, which is why G4
  is an intersection-union conjunction and no claim averages across references.

### 5.3 Labels

- **Pair benefit** (`benefit_screen`, threshold 0):
  - `AGG_POSITIVE` iff x - 2 se_agg > 0 AND both version means > 0 (the direction must hold in both procedure repeats);
  - `AGG_POSITIVE_POINT_ONLY` iff x > 0 otherwise;
  - `AGG_NOT_POSITIVE` iff x <= 0;
  - `ALIAS_ZERO`; `INCOMPLETE` (a registered cell, SE or finite value missing).
- **0.020 criterion** (`target_screen`, same rule at threshold 0.020): `TARGET_AGG_RESOLVED` / `TARGET_AGG_POINT_ONLY` /
  `TARGET_AGG_NOT_MET` (an alias or incomplete entry is NOT_MET).
- **Recipient view** (`recipient_screen`, per version, because the original allowance is per version). r_v = mean of version v's
  block means, and se_v is computed from v's cells only. Labels, first applicable:
  - `ALIAS_ZERO` (identical view; passes);
  - `RECIPIENT_AGG_RESOLVED_WORSE` (some r_v - 2 se_v > 0.010);
  - `RECIPIENT_AGG_POINT_FAILED` (some r_v > 0.010);
  - `RECIPIENT_AGG_RESOLVED` (r_v + 2 se_v <= 0.010 for both versions);
  - `RECIPIENT_AGG_EDGE` (point pass, not resolved).
  - **The gate** (eligibility; the original allowance applied at the aggregate point) passes iff the label is RESOLVED, EDGE or
    ALIAS_ZERO, that is, r_v <= 0.010 for both versions (inclusive). The cells above 0.010 are listed.
- **Sign reversals** (every comparison): distinct cells with the opposite sign to x, blocks with the opposite sign, versions with the
  opposite sign.
- **n_eff and seed effectiveness.** n_eff counts distinct cells (seed-deterministic k-duplicates once). `k_differs` is given per
  block, with the qualifier `READER_SEEDS_EFFECTIVE_ALL_BLOCKS` / `READER_SEEDS_PARTLY_EFFECTIVE` /
  `READER_SEED_REPLICATION_ABSENT`.

### 5.4 Old every-cell rule (kept beside it as a diagnostic; earlier campaigns are never relabelled)

- On the 8 registered cells: `tsz.rules.stability_label` (STABLE_POSITIVE: every cell > 0 and d_v - 2 sd_total > 0), and
  `tsz.rules.target_label` at 0.020.
- `tsz.rules.recipient_gate` (every cell <= 0.010).
- Whenever the old and the new decision disagree, the record carries `EVERY_CELL_RULE_DIVERGES`. The divergence is disclosed with the
  claim.
- **One noisy cell does not veto a candidate in this campaign.** The aggregate decides; the uncertainty analysis of 5.2 is the
  reviewed analysis.
- Verdicts of earlier campaigns stay exactly as published.

### 5.5 Known-law behaviour (`SCREEN_CHECKS.json`; invented data only; not a universal error guarantee)

**Simulation model.** 20,000 replicates per setting, rng seed 20261020. The 8 registered cells follow
x_vrk = mu + a_v + b_r + g_vr + d_vrk + e_vrk, with:
- a: version effect;
- b: split effect, shared by both versions;
- g: version-specific readers on a split;
- d: reader seed;
- e: SCORE-household error, with correlation 0.8 within a block, 0.5 for the same split in the other version, and 0.15 across
  splits.

Reported SEs carry the B = 400 bootstrap noise. Other settings: deterministic winners duplicate k0 into k1 in half the blocks; one
cell is contaminated by +/- 0.015 with probability 0.2. The scalar functions agreed with the vectorized copy on 8,800 replicates
across the 8 laws.

Pass rates, new aggregate screen / old every-cell rule:

| Law (SDs in AUC) | Benefit, true 0 (false pass) | 0.005 | 0.010 | 0.020 | Target, true 0.020 (false pass) |
|---|---|---|---|---|---|
| sampling-dominated (s .004, version/split .001, seed .002) | 0.0007 / 0.0003 | 0.053 / 0.016 | 0.485 / 0.260 | 0.996 / 0.977 | 0.0008 / 0.0001 |
| reader/split-dominated (s .003, V .002, R .003, seed .004) | 0.0088 / 0.0011 | 0.118 / 0.026 | 0.491 / 0.208 | 0.974 / 0.874 | 0.0090 / 0.0008 |
| version-dominated (V .005) | 0.0197 / 0.0070 | 0.161 / 0.084 | 0.506 / 0.367 | 0.961 / 0.944 | 0.0204 / 0.0071 |
| deterministic winners (half the blocks) | 0.0008 / 0.0002 | 0.052 / 0.018 | 0.455 / 0.249 | 0.994 / 0.958 | 0.0002 / 0.0000 |
| one contaminated cell (p 0.2, +/- 0.015) | 0.0008 / 0.0001 | 0.054 / 0.013 | 0.447 / 0.213 | 0.991 / 0.926 | 0.0006 / 0.0001 |
| large noise (s .008, V .003, R .003, seed .005) | 0.0011 / 0.0001 | 0.011 / 0.002 | 0.061 / 0.015 | 0.449 / 0.226 | 0.0011 / 0.0001 |
| version-extreme (V .008, others <= .002) | **0.078** / 0.064 | 0.236 / 0.217 | 0.476 / 0.484 | 0.857 / 0.911 | (in JSON) |
| split-extreme (R .008, others <= .002) | **0.080** / 0.054 | 0.245 / 0.182 | 0.484 / 0.394 | 0.861 / 0.789 | (in JSON) |

Recipient view (true worsening mu; rates are point gate / RESOLVED / old every-cell gate):

| Law | mu = 0 | 0.005 | 0.010 (at the allowance) | 0.015 | 0.020 |
|---|---|---|---|---|---|
| sampling-dominated | 0.997 / 0.372 / 0.890 | 0.881 / 0.035 / 0.450 | 0.336 / 0.000 / 0.059 | 0.019 / 0 / 0.001 | 0.000 / 0 / 0 |
| reader/split-dominated | 0.979 / 0.406 / 0.677 | 0.784 / 0.090 / 0.269 | 0.315 / 0.006 / 0.039 | 0.036 / 0.000 / 0.002 | 0.001 / 0 / 0 |
| large noise | 0.840 / 0.041 / 0.384 | 0.619 / 0.007 / 0.177 | 0.324 / 0.001 / 0.052 | 0.114 / 0 / 0.011 | 0.026 / 0 / 0.001 |

**Reading.**
1. Under the null the new benefit screen falsely passes 0.07-2.0% of the time under six of the eight laws. That is below the
   one-sided normal 2.3% that z = 2 suggests, because se_hh is a worst-case bound and both version means must agree.
2. **Limitation.** When the version or the split component dominates (SD 0.008 against <= 0.002 for the rest), the false-pass rate
   is about 8%. The old rule gives 5-6% under the same laws. With only 2 versions and 2 registered splits those components rest on
   one degree of freedom each, and no estimator that keeps power can repair that. The robustness repetitions add split levels;
   versions stay at 2. Any claim whose block means show one dominant version or split is reported with that caveat.
3. Under the five moderate laws the new screen roughly doubles the old rule's power at a true benefit of 0.010 (0.45-0.51 against
   0.21-0.37) and is barely affected by one contaminated cell (0.447 against 0.485). The old rule's power drops more (0.213 against
   0.260).
4. The recipient gate passes truly non-worsening outputs (mu = 0) in 79-99.7% of replicates over the eight laws. The old every-cell
   gate passes them in only 38-91%: one noisy cell vetoed 9-62% of genuinely non-worsening outputs.
5. The gate is the original **point** allowance applied to the aggregate, so a release exactly at the allowance (0.010) passes
   about one third of the time (0.25-0.44; both version means must be <= 0.010). At 1.5 times the allowance (0.015) it passes 2-4%
   under the moderate laws and up to 16% under the large-noise and extreme laws. The `RECIPIENT_AGG_RESOLVED` label (upper bound
   within the allowance; it passes almost nothing at or above 0.010) is reported beside it for every comparison.
6. Exact aliases: pass rate 0 (deterministic).

## 6. Reader-stability summaries (`reader_stability`) and grid validity (`grid_validity`)

**Per (candidate, comparator, view)** the summary reports:
- per-cell differences, with sign changes counted over distinct cells;
- `cells_beyond_threshold` (pair: <= 0; single views: > 0.010);
- each release's winners per cell, the distinct winners (`winner_changes`), and the cells whose winner is seed-dependent (ET5,
  ET25, MLP64, MLP6432, OUT_MLP, LOG_MLP);
- `deterministic_cells` (both winners seed-invariant, so the cell cannot change with the reader seed);
- VALIDATION near ties (< 0.010) per release;
- the number of cells in which the two releases' winners differ;
- `k_differs` per block, n_eff and the reader-seed qualifier.

Effective-seed counting:
- Seed-deterministic k-duplicates and exact aliases count once. An identical view of the two releases is reported as
  `EXACT_ALIAS_COUNTED_ONCE` (zero difference), not as 8 agreeing cells.
- "Across reader seeds" may be written only when the qualifier is `READER_SEEDS_EFFECTIVE_ALL_BLOCKS` and the grid is valid.
  Otherwise the text states on how many blocks the reader seed changed the result.

**Grid validity** (the A-SC1 criterion of TSZ1, adopted prospectively here, before any outcome):

`READER_SEEDS_EFFECTIVE` iff all of the following hold:
1. The reader seeds are recorded, distinct integers, none equal to MODEL_SEED_BASE + 300 + 10 v + k or + 500 + 10 v + k
   (k = 0..9; the OHN1/CTS1 and TSZ1 grids). The four registered seeds S0k0, S0k1, S1k0, S1k1 must all be recorded. This
   campaign uses + 600.
2. On the minimum set (the selected candidate and its matched controls), no seed-dependent recipe has bitwise-equal k0 / k1 SCORE
   scores.
3. Every other such equality is explained by all three of:
   - (a) another seed-dependent recipe differs in the same wire, view, version and repetition;
   - (b) the same recipe differs on another view of the same wire;
   - (c) the view's distinct-input count is recorded label-free.

   "Differs" means a RECORDED check with `equal_k0_k1` false. A recipe or view that was never checked is not evidence.

Seed-invariant recipes that differ are flagged `NONDETERMINISTIC_RECIPE` (recorded, not fatal).

**No claim label without a recorded valid grid.** `claims` needs `grid_valid` recorded and true. A missing value gives
`NOT_COMPUTED_GRID_VALIDITY_UNKNOWN` for G2-G6, and a false value gives `NOT_COMPUTED_GRID_INVALID`. G1 is still computed, there
is no tier, and the configuration cannot be selected until an engineering repair is recorded.

## 7. Frontier against the SEX-blind control path (`frontier_comparison`; no exact cost match required)

**Cost.**
- Primary cost = total NLL spending of the complete release: income NLL + occupation NLL (nats per person, MODEL_SELECT,
  draw-averaged).
- It is compared through PAIRED per-person differences d = cost(point) - cost(control), with household-clustered SEs from
  `nrm.evaluate` (`total|nll`).
- If only the per-task differences are given, SE(sum) <= se_income + se_occupation (flag `COST_SE_BOUND`).
- Across versions: d = mean of the two version differences; SE = mean of the two SEs (correlation bound: the versions share people).
  MC SEs are added in quadrature per version.

**Rule** (the actual selected output only; every path point is reported):
1. **Bracket.** Q* = the control-path point with the smallest pooled cost among those whose pooled point cost is >= the point's
   (d <= 0). Q* spends at least as much at the point estimate, so it should leak less. Comparing against it favours the control.
2. **Recovery.** The pair benefit of the point against Q* must be `AGG_POSITIVE` (section 5).
3. **Secondary spending.** For income and occupation accuracy loss, Brier and NLL, and the men's and women's occupation Brier: if
   any pooled paired difference is resolved higher (d_m - 2 se_m > 0), the label is `SEPARATED_ON_PRIMARY_COST_ONLY`, with the
   metrics listed.

**Labels.**
- `DOMINATES_CONTROL_FRONTIER`: steps 1-3 all hold.
- `NOT_SEPARATED_FROM_TASK_SPENDING`, with reason `CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT` (no bracket: the control path
  must reach the candidate's spending to test it), `NO_RESOLVED_BENEFIT_AT_BRACKETING_CONTROL`, or `CONTROL_PATH_MISSING`.
- `INCOMPLETE`: paired costs or the grid entry against a needed control are missing. That includes any of the 8 secondary metrics
  against Q* being missing or non-finite: step 3 is never skipped.

**Interpolation (reported, never gating).** Linear interpolation between the cheaper neighbour and Q* at the point's cost.
- lambda = d_lo / (d_lo - d_hi).
- Benefit = (1 - lambda) b_lo + lambda b_hi, with SE <= (1 - lambda) se_lo + lambda se_hi.
- The actual utility vectors of every path point and of the point are always reported.

## 8. Candidate selection (at most ONE development candidate; `select_candidate`)

**Universe.** The complete ledger: every configuration of every family and every challenger control saved through
`nrm.release.save` and evaluated in both versions. Control-labelled configurations compete on equal terms and are never excluded
after results; a chosen control carries `CONTROL_DESIGNATED_CANDIDATE`.

**Not selectable:**
- `NOT_SELECTABLE_OUTPUT_MISMATCH`: the wire hashes recorded with the utility evaluation and with the grid differ (any version,
  task, or draw of a randomized release). No nearby strength, seed or draw is ever substituted.
- `ALIAS_OF_<REFERENCE>` or `ALIAS_OF_<earlier config_id>`: an exact alias. The earliest ledger entry keeps the result; later
  aliases are counted once. Reference aliases are detected two ways:
  - by wire hash: the wires of all 9 references are a required input, and selection refuses to run without them;
  - by an `ALIAS_ZERO` pair comparison against any reference.
- `BELOW_MINIMUM_TIER`: the configuration does not reach T2.
- `NOT_SELECTABLE_INCOMPLETE_REFERENCE_GRID`: a grid entry against one of the 9 references is missing or incomplete, so the
  score cannot be computed (every selectable configuration is read against all 9).

**Tiers** (the claim ladder, highest first):
- T5 = G1 G2 G3 G4 and G5 resolved;
- T4 = G1 G2 G3 G4;
- T3 = G1 G2 G3;
- T2 = G1 G2 (minimum).

**Choice.**
- Only the highest occupied tier is considered.
- Score = the minimum, over the 9 point-eligible references, of the registered-cell aggregate pair benefit x (the OHN1 score on
  the aggregate).
- Near-tie set: score >= best - 0.005.
- Deterministic tie-breaks, in order:
  1. reserve-qualified on both versions first (implied by tie-break 2 whenever both versions are evaluated, since qualified means
     every ratio >= 2; kept explicit);
  2. leximin whole-release reserve ratio vector (larger minimum first, 1e-9 ties pass on);
  3. fewer fitted components: a number, or a per-version record summed over both versions. 0 counts as 0; a missing value or a
     missing version counts as infinitely many;
  4. earliest ledger entry (a missing entry sorts last);
  5. config_id (lexicographic).
- No configuration at T2 or above gives `NO_DEVELOPMENT_CANDIDATE`.

**Never used for selection:** G6, G7, robustness cells, the old every-cell rule.

**The record shows** for the chosen configuration and for every other: G1-G7, the useful-component label, tier, score, reserve
status and the near-tie set.

Selection over many configurations on adaptively used MODEL_SELECT inflates the chosen score (winner's curse). The selected
candidate is development evidence only.

## 9. Randomized releases

- **Utility** (as `nrm.evaluate` computes it): each metric on each emitted draw (vector and its own argmax), averaged over R
  predeclared one-draw-per-person repetitions.
  - Household SE = mean per-draw SE. MC SE = SD over draws / sqrt(R).
  - Every draw's own point status is reported (`worst_draw`). An averaged vector is never used unless it is the actual release.
- **Grid** (`mc_cell`): the cell value is the mean over the cell's evaluated draws, each draw being a separate one-draw release
  with its own readers.
  - Household SE = RMS of the draws' paired bootstrap SEs (a bound, because draws share SCORE households and weights).
  - MC SE = SD over draws / sqrt(D), with D >= 2. D = 1 gives `MC_SE_NOT_ESTIMABLE`, and that flag travels with every claim of that
    configuration.
- **Decisions**: the MC SE enters se_agg as its own component and the reserve's own SE in quadrature. Household-bootstrap,
  reader/split and MC components are always reported separately.

## 10. What is NOT claimable

- Protection against arbitrary readers.
- Confirmation (MODEL_SELECT is adaptively used); anything about DEV_REPORT or the attack roles (closed).
- Independent replication: versions share rows; repetitions share households.
- That the reserve is a cap or a population guarantee.
- Superiority over every historical method from a reserve-frontier result.
- A joint-design contribution when only one output changes, or without the separate design and both sequential orders.
- Novelty from performance.
- Any protection label without G1.
- A competitive claim from a useful component.
- Any claim from a substituted output.

## 11. Constants (RULES.json)

| Constant | Value |
|---|---|
| Caps; guard; recipient allowance; target | 0.010 / 0.010 / 0.005; 0.005; 0.010 per comparator, view, version; 0.020 vs NEURAL_RAW and H_LJ |
| Reserve | ohn.rules.reserve unchanged, k = 2, H_LJ floor; MC SE in quadrature for randomized releases |
| Aggregate screen | z = 2; se_agg floor 0.001 AUC; both version means must agree in sign (benefit, target); recipient per version |
| Decision cells | the 8 registered S<v>R<r>k<k>; R3 / R4 / k2 / k3 robustness only |
| Frontier | primary cost total NLL (paired); bracket = cheapest control spending >= the point; z = 2 for secondary spending |
| Selection | tiers T5 > T4 > T3 > T2 (minimum T2); score = min over 9 references of x; near tie 0.005; 5 tie-breaks |
| Reader seeds | forbidden offsets 300 and 500 (k = 0..9); this campaign 600 |

## 12. Inputs the coordinator's records must carry (open points)

1. Per configuration and version: `wire_sha256_eval` and `wire_sha256_grid` ({income, occupation}; every draw for randomized
   releases), `n_fitted_components`, `entry` (ledger order), `kind` and `grid_valid` (the section 6 check). The wires of all 9
   references are passed to the selection.
2. In each mechanism card, before the fit: the family's task-only twin and its simplest alternative explanation (G2). For a joint
   design, also the separate design and both sequential orders (G6). For G3, the SEX-blind control path.
3. Grid entries in the `paired()` format for: every configuration against its matched controls, against the 9 references and (for
   G3) against the bracketing control points; `cell_se_mc` for randomized releases.
4. Paired costs from `nrm.evaluate` with a `total|nll` entry: the SE of the per-person summed NLL difference. Without it, the
   conservative sum-of-SEs bound applies.
5. The novelty label comes from a separate prior-art review, if any.

## 13. Change record (before registration; no outcome existed)

**Self-review.** Corrections made before freezing:
- G5 is not nested in G4 (it is a separate claim).
- `NOT_SELECTABLE_INCOMPLETE_REFERENCE_GRID` replaces a misleading `BELOW_MINIMUM_TIER` label for configurations with an incomplete
  reference grid.
- The invalid-grid rule is enforced in code, not only stated.
- Quoted simulation ranges were corrected to cover all eight laws.
- Two extreme laws were added so that the limitation in 5.5 is disclosed, not hidden. The screen itself was NOT changed after
  seeing the simulated rates.

**Independent code review.** A review subagent, launched by me and reading only the code and these files, reported seven
correctness defects at 15:08Z. All seven were reproduced and fixed, and each now has a test that fails on the earlier code:
1. G3 could pass with secondary spending metrics missing or non-finite. These are now `INCOMPLETE`.
2. The fitted-components tie-break turned 0 into infinity and failed on per-version records and on a missing ledger entry.
3. Grid validity counted an unchecked recipe or view as "differs", and accepted a single recorded seed.
4. Claims proceeded when `grid_valid` was missing, or false as a non-Python boolean.
5. Reference-alias detection was off when no reference wires were passed. The wires are now required, and an `ALIAS_ZERO` pair
   comparison also marks an alias.
6. G6 gave `NOT_APPLICABLE_ONE_OUTPUT_CHANGED`, with a false sentence, when both outputs changed but were not jointly designed, and
   counted a duplicated output twice.
7. G5 emitted positive levels without G1, and mislabelled "target resolved, recipient gate failed" as point-only.

Also from the review:
- The RULES.json interpolation text no longer claims a cost-slope term that the code does not compute.
- Tie-break 1 is implied by the leximin tie-break; it is kept explicit, and leximin now has its own test.

No threshold, z, floor, cell set, reference, label ladder or tier changed. All of this happened before any outcome of this
campaign existed.

**Process note.** One shell call at about 14:50Z contained an empty `python3 - <<'EOF'` heredoc. It executed nothing, but the
brief forbids heredocs. It is recorded here.

## 14. Completion record

- Completed 2026-10-20T15:11:09Z (`date -u`, read before this section was written).
- `python3 -m nrm.test_rules`: 18 of 18 PASS (invented data only). The 8 known laws were simulated with 20,000 replicates per
  setting; 8,800 scalar-versus-vectorized agreement checks; about 1 CPU-s.
- SHA-256 at completion:
  - RULES.json 84f187768817e79ef9f2bb36151bea81f00b8f62d9179c8cd523069550674652
  - nrm/rules.py d5df868678d483a3452e251d8f24575932d23f0e5d22e3ef1ca3627a35030df1
  - nrm/test_rules.py d03bbed2f2a4b94c85f48b24926274567849a80e65ff816ee8217c4b668f60d5
  - SCREEN_CHECKS.json fc37fc64d6f750f9c69594600aed708347e82120c585ebb27343200d35713d81 (deterministic, rng seed 20261020)
- No label was read and no real-data computation was run.
