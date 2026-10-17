# Protocol: SCM engineering benchmark (frozen before any real fit)

**Status.** This is an engineering benchmark of an existing method, Shared-Cell Merging (SCM). SCM applies established merging methods (the privacy funnel, agglomerative information bottleneck, IBSI). This benchmark makes no novelty claim.

**Scope.** Everything here is exploratory development evidence on repeatedly used Florida ACS 2019 development people. The sealed assessment stays closed. The service, adversary and composition statement are `results/pcrl_structural_mechanism_pilot_v1/SERVICE_AND_THREAT_MODEL.md` (v1, with corrections), reused unchanged.

**Question.** Can this specific bounded shared-partition recipe preserve held-out task quality and reduce measured SEX recovery, relative to its matched task-only and published-objective controls and the eligible neural benchmark? The question is **not** whether any shared partition can preserve utility.

**Source.** The stopped structural campaign, `research/pcrl-structural-mechanism-pilot-v1` at `ec20c8bdf` (public package `9b0f51c4e`). Designer 1's proposal (`6bc0617b…`) is the recipe, with the corrections listed in section 12. That campaign's decision and locks are unchanged.

## 1. Roles and access

| Role | Use in this benchmark |
|---|---|
| MODEL_FIT | Teachers (features, task labels). SEX propensity and within-SEX guard (SEX as a training signal only). Fine partition and merge statistics |
| CALIBRATION | Not used by SCM: its heads are pooled teacher probabilities with no temperature. U0/Ucal are the frozen references |
| MODEL_SELECT | Held-out utility clauses, cross-fitted privacy proxies, and all selection and nomination (features, labels, SEX) |
| ATTACK_FIT / ATTACK_VALIDATION | Final-bank readers only (fit, then orient and select) |
| DEV_REPORT | Features for releases. Truth is re-read once, logged, after the committed REPORTING_LOCK |
| SEALED_ASSESSMENT | Closed |

## 2. Teachers (per model seed s in {0, 1})

**Folds.** Five household folds on MODEL_FIT. The unique household ids are permuted with seed `seed_for('SCM1|folds', s)` and then dealt in rotation.

**Task teachers.** For each fold f, one teacher per task (income, occupation) is trained on the other four folds.
- Registry: the T_TREE settings (CatBoost 1.2.8, depth 4, learning rate 0.05, at most 800 trees, l2 3, one thread).
- Tree count: the first argmin of a 15% household internal validation loss (`acs_replication.data.internal_split`, seed `100 s + f`), then a refit on the four folds.
- Seed: `seed_for('SCM1teacher', 100 s + f, task, 4)`.
- Export: each teacher becomes a numpy `TreeTeacher`, with routing parity checked on its training rows.

**Out-of-fold (OOF) probabilities.** Each MODEL_FIT person's income vector t1 and occupation vector t2 come from the fold teacher that excluded their household.

**Runtime map.** For every other role, t̄(x) is the arithmetic mean of the five fold teachers' probability vectors, per task.
- **Mismatch.** OOF vectors come from one model that did not see the person. Runtime vectors average five models, so they are smoother. Cells and tables are estimated on OOF vectors and applied to runtime vectors, so the two are not identical.
- **Diagnostics reported:**
  - total-variation distance between MODEL_FIT OOF cell occupancy and MODEL_SELECT runtime occupancy;
  - the share of MODEL_FIT people whose runtime cell equals their OOF cell. This is descriptive: on MODEL_FIT, four of the five runtime models saw the person.

**SEX propensity π̂.** Same folds, the CatBoost binary registry, seed `seed_for('SCM1propensity', 100 s + f, 'sex', 4)`. It is OOF for MODEL_FIT people, used for training only, and never deployed.

## 3. Fine partition C0

**Points.** The OOF pairs (t1, t2) of the 20,000 MODEL_FIT people.

**KL-Bregman k-means**, nominal M0 = 512 cells.
- Distortion: KL(t1 ‖ μ1) + KL(t2 ‖ μ2).
- Centroid: the member mean, which is the Bregman centroid for this KL direction.
- Assignment: argmax over c of Σ_task Σ_k t_k log μ_{c,k}; ties go to the lowest index.
- Initialisation: k-means++ with KL distortion, seed `seed_for('SCM1kmeans', s)`.
- Iterations: at most 50 Lloyd iterations, stopping when no assignment changes.
- Empty cells: cells empty at the end are dropped, with no reseeding. The occupied count is reported.
- Numerics: log μ is clipped at 1e-300.

**Runtime assignment.** The same rule applied to t̄(x) over the occupied cells.

## 4. Merge statistics and plug-in costs (MODEL_FIT; teacher OOF law as truth)

**Per-cell statistics.**
- Counts n_c.
- Sums of t1, t2 and ‖t1‖²; sums of Σ_k t1 log t1 and of max t1.
- The sum of π̂.
- Per SEX s: n_{c,s} and the sum of one-hot occupation labels.

**Occupation, cell level.** Merging a and b into m changes the plug-in metrics exactly:

| Metric | Exact change |
|---|---|
| Brier | n_a n_b / n_m ‖μ_a − μ_b‖² / N (Ward) |
| NLL | [n_a KL(μ_a ‖ μ_m) + n_b KL(μ_b ‖ μ_m)] / N |
| Accuracy loss | [n_a max μ_a + n_b max μ_b − n_m max μ_m] / N |

**Income, grouped.** The released income vector is that of one of 32 groups.
- Grouping: final cells are sorted by pooled P(income = 1), ascending, with ties by cell index. Cell c goes to group g = min(31, floor(32 (M_before + n_c/2) / N)), where M_before is the MODEL_FIT mass of the cells before it.
- Group vector: the pooled t1 over the group's MODEL_FIT members. One vector is stored per group and indexed, so every member of a group receives bit-identical values.
- Costs: income Brier, NLL and accuracy are computed **exactly on the grouped release**, from the per-cell sums. Every budget check uses these grouped costs.

**Within-SEX guard** (occupation Brier, per SEX).
- Shrunk labels: ỹ_{f,s} = (Σ_{i in f, s} e_{y_i} + 20 μ_f) / (n_{f,s} + 20), with κ = 20. This is computed **once at the fine cells f**, with μ_f the fine cell's pooled teacher vector, and every person inherits their fine cell's value. The guard's plug-in law is therefore fixed, and merge costs are exact and additive.
- Stored sums: Ỹ_{c,s} = Σ_{f in c} n_{f,s} ỹ_{f,s}.
- Cost of a merge: Δ_s = [n_{m,s}‖μ_m‖² − n_{a,s}‖μ_a‖² − n_{b,s}‖μ_b‖² − 2(μ_m · (Ỹ_a + Ỹ_b) − μ_a · Ỹ_a − μ_b · Ỹ_b)] / N_s. This can be negative.

**SEX gain.** G = n_a n_b / n_m (π_a − π_b)² / N, where π_c is the cell mean of π̂.

**What these costs are.** They are exact under the fitted plug-in law. They are **not** held-out or population utility costs, and pooled teacher probabilities are **not** automatically calibrated on new people. The binding utility check is the held-out MODEL_SELECT clauses (section 9).

## 5. Arms per seed (all share the teachers, fine partition, routing, income-grouping rule and released format)

Budgets apply to every capped arm, relative to C0's release (C0 with income grouping). They scale with β except accuracy:

| Constraint | Budget |
|---|---|
| NLL, each task | β·0.010 |
| Summed Brier, each task | β·0.005 |
| Accuracy, each task | 0.005 |
| Within-SEX occupation Brier, each SEX | β·0.005 |

**Exact greedy (every capped arm).**
- At each step every live pair is scored. Candidates are tried in priority order, ties going to the lowest (a, b).
- A merge is accepted when every cumulative cost stays within its budget: cell-level occupation and guard costs, plus the exact grouped income costs after the merge.
- Remaining budgets are R_k = B_k − spent_k. A negative guard change increases R.
- This replaces the stopped campaign's lazy heap, which the verifier found could differ from exact greedy.

| Arm | Definition |
|---|---|
| C0 | Unmerged occupied fine cells plus income grouping (reference) |
| SCM_b (β in {0.5, 1.0, 1.5}) | Priority P = G / (Σ_k max(Δ_k, 0)/R_k + 1e-12), using cell-level Δ_k for every metric, income included. Stops at the first step with no admissible pair; final count K_β |
| TWIN_b | The task-only merge twin. Priority −Σ_k max(Δ_k, 0)/R_k (cheapest normalised merge). Same budgets, **including the within-SEX guard: guard-only SEX access, not SEX-blind**. Stops at K_β. If it cannot reach K_β, it stops at the first step with no admissible pair; that is recorded as unmatched capacity, a coverage gap for the recipe comparison |
| IBSI_b_g (γ in {0.5, 1, 2}) | **PUB_IBSI**: the agglomerative IBSI objective (Chechik and Tishby, NIPS 2002, `FIELD_MATRIX.md` row P3), solved by AIB-style greedy merging (Slonim and Tishby 1999). Merge the pair minimising ΔI(T; Y⁺) − γ ΔI(T; Y⁻), with ΔI(T; Y⁺) = ΔNLL_income + ΔNLL_occupation (plug-in task information lost, nats, cell level) and ΔI(T; Y⁻) = n_a KL_B(π_a ‖ π_m) + n_b KL_B(π_b ‖ π_m) over N (plug-in SEX information removed). No per-clause caps. Stops at K_β. γ values frozen before fitting (section 12) |
| SHUF_b_r (r = 0..4) | SCM with the **initial-cell SEX rates permuted across initial cells** (masses fixed; permutation seed `seed_for('SCM1shuffle', s, r)`). Everything else is SCM's, including the within-SEX guard with **true** labels (disclosed). Stops at K_β or earlier (a coverage gap). A mechanism-attribution control, not a valid permutation test |

That gives 31 releases per seed. **Capacity reported for every arm:**
- nominal cells;
- occupied cells per role;
- distinct transmitted income, occupation and pair vectors per role;
- income groups in use.

## 6. Released wire and routing

- R1 receives (v1[g(x)], argmax) and R2 receives (v2[c(x)], argmax): float64 values copied from stored tables.
- R1's output is routed only to R1 and R2's only to R2; the coalition is the aligned pair.
- Nothing else is transmitted.
- Deployment takes only the six raw fields through the frozen 145-column encoder and refuses any other schema, SEX included. It also refuses requests for hidden outputs (cell or group ids, teacher probabilities, π̂).

## 7. Selection and nomination (MODEL_SELECT only; frozen)

1. **Utility eligibility.** All 24 held-out clauses on MODEL_SELECT, against U0 and Ucal, on both seeds:
   - accuracy loss ≤ 0.010;
   - NLL excess ≤ 0.010 nats;
   - summed Brier excess ≤ 0.005.

   These are point feasibility only. Confidence status is reported separately (section 9). The source's subgroup guards are reported separately and do not gate.
2. **Privacy proxy, cross-fitted and household-separated.**
   - Split: MODEL_SELECT is split into two household halves, A and B (seed `seed_for('SCM1proxysplit')`, the same split for every arm).
   - Readers: OUT_LR, OUT_HGB and the frozen lookup reader, each on the pair wire.
   - Directions: fit on A and score B, and fit on B and score A. Orientation is fixed at +1 for every reader, because each reader is fit to estimate P(female); no held-out label orients anything.
   - The proxy for one reader is the mean AUC over the two directions. The arm's proxy is the maximum over the three readers.
3. **SCM nominee.** Among the β values eligible on both seeds, pick the one minimising the mean over seeds of the arm proxy. Ties go to the smaller β.
4. **Ineligible diagnostic fallback.** If no SCM setting qualifies, pick exactly one β, minimising the worst normalised violation, max over the 24 MODEL_SELECT clauses (12 per seed) of value/cap. Ties go to the smaller β. Its matched twin is evaluated diagnostically. Neither becomes competitively eligible.
5. **Twin.** The nominee's matched twin (same β), included whatever its eligibility. Whether it reached K_β on each seed is recorded as `twin_matched_capacity`.
6. **PUB_IBSI nominee.** At the nominee's β, among the eligible γ values, pick the one minimising the mean proxy. If none is eligible, pick the γ with the smallest worst normalised violation; it is labelled ineligible and shown descriptively.
7. **Shuffled nominee.** At the nominee's β, among the shuffled controls that are eligible **and reached the matched cell count K_β on both seeds**, pick the one with the minimum mean proxy, which is the most protective. A shuffled run that stopped above K_β is never a nominee. If none is eligible, the final bank has no shuffled control, and that comparison is recorded as a coverage gap.

## 8. Final recovery bank (per seed; 7 releases)

The bank holds:
- SCM nominee (or the ineligible fallback);
- TWIN;
- IBSI;
- SHUF (if any);
- NEURAL_RAW (frozen ρ1 E2, eligible);
- H_LJ (frozen; the eligible tree or hybrid release with the lowest published pair recovery);
- C0 (reference, descriptive).

Frozen comparators are admitted bitwise from the hash-checked frozen-hybrids bank. Utility-ineligible releases are shown descriptively and never counted as beaten.

**Readers.** The frozen 8-reader bank per view: OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP, DEC_LR, CONF_HGB and LOOKUP. For the pair, 8 pair readers plus 16 ignore-one aliases.
- Fit on ATTACK_FIT (8,001); orient and select on ATTACK_VALIDATION (3,000); ties go to the earlier candidate.
- Model seed `MODEL_SEED_BASE + s`.
- **LOOKUP**: the key is the exact float64 bytes; the score is (n_female + 2 π)/(n + 2); an unseen key scores π.
- Controls on the nominee's wire: a shuffled-SEX null (OUT_HGB) and a planted SEX bit (OUT_HGB), each on three views.
- No learning-curve sweep.
- Reported: coverage and occupancy on ATTACK_FIT and ATTACK_VALIDATION, as distinct vectors, people per vector, and the share of validation vectors seen in fitting. About 15 examples per cell is a diagnostic heuristic only.

**After the final readers.** The verifier independently refits a predefined subset: the pair winner for SCM, TWIN and NEURAL_RAW, on both seeds.

## 9. Reporting (after the committed REPORTING_LOCK; one logged DEV_REPORT truth re-read)

**Inference.**
- One household bootstrap over DEV_REPORT households (B = 2048, seed 2026101705), shared across releases, endpoints and seeds.
- Wald bounds: point ± t(G − 1) × bootstrap SE.
- Claim clauses: one-sided at α/3, α = 0.025.
- Primary family: two-sided 95% Bonferroni-t over K_primary = 42.
- Conditioning: on the fitted models and readers. The two seeds share people and are not independent populations.

**Primary roster (K = 42).** Benefit = comparator AUC − SCM AUC; worsening = SCM AUC − TWIN AUC.

| Endpoint | Count | Margin |
|---|---|---|
| RECIPE coalition benefit vs TWIN, s0 and s1 | 2 | 0.020 |
| RECIPE income and occupation worsening vs TWIN, s0 and s1 | 4 | ≤ 0.010 |
| ATTRIBUTION coalition benefit vs IBSI | 2 | 0 |
| ATTRIBUTION coalition benefit vs SHUF (MISSING if no shuffled control) | 2 | 0 |
| COMPETITIVE coalition benefit vs NEURAL_RAW (nested 0.020) | 2 | 0 |
| COMPETITIVE coalition benefit vs H_LJ | 2 | 0 |
| UTILITY clauses of the SCM release on DEV_REPORT | 24 | caps |
| SUBGROUP within-SEX occupation Brier excess vs U0, both groups, both seeds | 4 | ≤ 0.005 |

**Secondary, descriptive (K = 10):**
- SCM vs C0 and TWIN vs C0, coalition, both seeds (4);
- single-recipient AUC differences SCM − NEURAL_RAW (4);
- SCM − TWIN pair AUC using the inherited 7-reader winner (2).

**Claims, kept separate.** There are three α/3 slices: A (with attribution tested in sequence inside it), B and C.
- **A, recipe:** coalition benefit vs TWIN clause > 0.020 on both seeds; worsening clauses ≤ 0.010; DEV 24-point gate; MODEL_SELECT eligibility. If TWIN did not reach K_β on either seed, A is `UNMATCHED_CAPACITY`, which means not established.
- **Attribution**, tested in fixed sequence inside A's α/3 slice and only if A is established; otherwise it is descriptive (`NOT_TESTED_RECIPE_NOT_ESTABLISHED`). It requires benefit vs IBSI and vs SHUF > 0 on both seeds. If the IBSI nominee is utility-ineligible, the IBSI part is `COMPARATOR_INELIGIBLE` and its endpoints are descriptive; an ineligible comparator is never counted as beaten. If SHUF is absent, its part is `MISSING`.
- **B, competitive:** benefit vs NEURAL_RAW and vs H_LJ > 0 on both seeds, with nested 0.020.
- **C, subgroup.**
- **D, joint:** not claimed.
- **E, novelty:** not claimed.

An ineligible fallback makes A and B `NOMINEE_INELIGIBLE`; its recovery comparisons are descriptive. A missing comparator is reported as a coverage gap, not a privacy result. MISSING endpoints stay in K_primary = 42, which is conservative.

## 10. Units and budget (precommitted; 8 CPU-h and 6 h elapsed from 02:38:58Z, two slots, one thread)

| Unit | Content | Estimate (CPU-h) |
|---|---|---|
| SCM_FIT_S0, SCM_FIT_S1 | 15 CatBoost fits; k-means; 31 merge paths; outputs on 4 roles | 0.5 each |
| SCM_SELECT | 31 arms × 2 seeds × 3 proxy readers × 2 directions; clauses | 0.3 |
| SCM_BANK | final bank assembly and hashes | 0.02 |
| SCM_READERS_S0, SCM_READERS_S1 | final-bank readers plus controls | 0.3 each |
| SCM_REPORT | bootstrap endpoints | 0.05 |
| SCM_DEPLOY | packages and raw-field reproduction | 0.05 |
| Verifier refits and recomputation | | 0.4 |
| Agents and light scripts (charged) | | 0.6 |
| Margin for failed attempts | | 1.0 |
| **Total** | | **about 4.0** |

## 11. Stopping and repairs

**No early stop.** Nothing stops the bank early on C0's utility, a forecast or novelty.

**Engineering repairs** (code defects only) get a dated note and a new versioned unit id, and preserve the failed attempt. They never change thresholds, settings or the selection rule after results are seen.

## 12. Deviations from the proposal and from the stopped campaign (declared before fitting)

1. **Exact greedy.** Every live pair is re-scored at every step, instead of a lazy heap.
2. **Income grouping costs.** They are included exactly in every budget check.
3. **The twin keeps the within-SEX guard** (guard-only SEX access). The proposal's twin used no SEX.
4. **Selection proxy.** It is replaced by the cross-fitted MODEL_SELECT proxy, per the source reviews and this prompt.
5. **Gates dropped.** The proposal's G0–G3 gates are dropped as stopping rules. Its falsifier F1 is reported as a prediction check on the cross-fitted MODEL_SELECT proxy: TWIN − SCM ≥ 0.020 on both seeds, meaning SCM's proxy is at least 0.020 below its matched twin's. Otherwise F1 is reported as refuted.
6. **PUB_IBSI added**, as required by the closest-work review. It is IBSI's β → ∞ hard (fixed-cardinality) limit, solved with the AIB greedy solver:
   - the two task terms are summed, following IBSI Eq. 6 with two relevant variables;
   - both information terms are plug-in estimates under the fitted teacher and propensity laws;
   - the paper's own experiments used a different, sequential, solver.

   γ = 1 is the published value (the paper reports γ = 0 and γ = 1, §6.2); 0.5 and 2 are data-free factor-2 brackets. The paper's "equal weight" reading of γ = 1 assumes the two variables share an alphabet, which they do not here, so that reading is not used.
7. **Five shuffled controls** permute initial-cell SEX rates, as the adversary reviewer recommended.
8. **Correction note on the historical proposal.** SCM §5's sentence that NEURAL_RAW "merges nothing beyond the encoder's OTHER pooling" is an unsupported injectivity-type claim. Whether any fitted model is injective on the population is unknown (`CORRECTIONS.md` §1(e), §2). The proposal file is kept unchanged with this note.
9. **Priority conventions, declared:**
   - negative per-merge costs are clipped to 0 in the priority, and remaining budgets are floored at 1e-15;
   - the priority uses cell-level income costs, while admissibility uses the exact grouped income costs, and the two can differ in either direction (a merge inside one income group costs nothing on the grouped release);
   - grouped income spending and guard spending can be negative;
   - TWIN's priority includes both within-SEX guards.
10. **Known proxy bias.** The proxy takes the maximum over three readers, which biases it upward by a similar amount for every arm. LOOKUP, at 10–40 people per cell per half, under-reads finer partitions, which favours smaller β in selection.
