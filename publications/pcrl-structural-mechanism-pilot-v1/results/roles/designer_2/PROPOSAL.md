# SCAR: shared-cell agglomerative release (designer 2)

**Status.** A data-free design, dated 2026-10-16. No real-person array was opened and nothing was fitted on real data. One exact synthetic toy of my own was run (§13); its numbers describe invented laws only. No feasibility, competitive or novelty claim is made.

## 1. Name and summary

**SCAR (shared-cell agglomerative release)** fits fresh task-only CatBoost teachers on MODEL_FIT, quantises the joint task-posterior space into 8,192 fine cells, and merges them greedily into K coarse cells. Each merge is scored by its exact plug-in cost to the six task metrics (each divided by its cap) minus λ times its exact plug-in reduction of I(S; cell). The cell SEX rate comes from a cross-fitted SEX propensity used only in training, and no merge may push any plug-in clause cost past its cap. Both releases are functions of the same coarse cell: the occupation recipient gets the cell's pooled occupation posterior, and the income recipient gets the pooled income posterior of one of 64 fixed income bins. Inputs in one coarse cell get bit-identical outputs, so a merged distinction is gone for every reader, and linking the recipients cannot restore it. Local continuous corrections did recombine: LOCAL_TREE read 0.582 from occupation alone and 0.657 from the pair (seed 0). The twin is the same pipeline with λ = 0 at the same K. The algorithmic core is established (§10). I expect the strongest falsifier (§9) more likely than not to trigger.

## 2. Mapping and optimisation problem

| Role | Use | Task labels | SEX |
|---|---|---|---|
| MODEL_FIT | teachers; fine cells; SEX propensity π̂ (2-fold household cross-fitting, out-of-fold predictions only); λ scale; agglomeration; pooled heads; income bins | teachers only | training signal via out-of-fold π̂ only; the twin uses none |
| CALIBRATION | unused (stays the Ucal role, as for T_TREE and TASK_REFIT) | — | — |
| MODEL_SELECT | frozen selection rule and gates G0–G4 (§9); nothing fitted | clauses | proxy readers, as in the frozen rule |
| ATTACK roles, DEV_REPORT | the prepared pilot's readers and endpoints, unchanged | — | readers only |

**Deployed map.** It runs on numpy only and refuses any schema other than the 145-column encoding of the six fields (`TreeTeacher.check_inputs`).
1. Encode AGEP, WKHP, COW, SCHL, MAR and POBP with the existing MODEL_FIT encoder.
2. Compute q(x) = (T1(x), T2(x)) ∈ R⁷ from two task teachers. They use T_TREE's registry (CatBoost depth 4, learning rate 0.05, ≤ 800 trees, l2 3, 15% household early stopping, refit) and are exported as `TreeTeacher`s with bitwise-verified routing.
3. Fine cell a(x) = argmin_j ‖q(x) − m_j‖² over 8,192 k-means centroids (ties to the lowest index).
4. Coarse cell c(x) = M[a(x)].
5. f2(x) = H2[c(x)] and f1(x) = B1[c(x)], stored float64 vectors (one per coarse cell or income bin). d_i = argmax f_i, ties to the first class. (f1, d1) goes only to R1 and (f2, d2) only to R2.

**Objective.** Over merge tables M with K classes, minimise J_λ(M) = T(M) + λ·Î(S; C_M).
- **Task cost.** T(M) = (1/N) Σ_{i=1,2} [ΔBrier_i/0.005 + ΔNLL_i/0.010 + ΔAcc_i/0.010], the plug-in excess of the pooled release over the teacher posteriors on MODEL_FIT.
- **SEX information.** Î is the plug-in mutual information with π̄_c = the mean out-of-fold π̂ over the cell's MODEL_FIT members.
- **Constraint.** Each of the six cumulative plug-in excesses, fine quantisation included, stays ≤ its cap relative to the teacher. The teacher's headroom over U0 and Ucal is the generalisation margin.
- For a fixed partition the pooled posterior is optimal under every proper score, so only the partition is searched.

**Algorithm.** Greedy, with no optimality claim.
- **Merge identities.** For clusters with counts n, mean teacher posteriors μ₁ and μ₂, and SEX rate π̄, merging c and c′ into m changes these exactly:
  - ΔBrier_i = n_c n_c′/n_m · ‖μ_ic − μ_ic′‖² (Ward);
  - ΔNLL_i = n_c KL(μ_ic‖μ_im) + n_c′ KL(μ_ic′‖μ_im);
  - ΔAcc_i = n_c max μ_ic + n_c′ max μ_ic′ − n_m max μ_im;
  - SEX gain ΔI = n_c KL_B(π̄_c‖π̄_m) + n_c′ KL_B(π̄_c′‖π̄_m).
- **Merge order.** With ΔT the task-cost change, merge the admissible pair with the smallest ΔT − λ·ΔI/N until K clusters remain. If none is admissible first, stop and record the count.
- **Candidate pairs.** A symmetric 16-nearest-neighbour graph of the centroids, united with its minimum spanning tree so it stays connected. Merged clusters inherit edges; a lazy heap holds the scores.
- **K (task-only rule, fixed now).** The smallest count on the unbudgeted twin path at which all six plug-in costs are ≤ their fine-level value + 0.25 × cap. Every arm of a seed uses it.
- **λ.** λ = r·λ₀, with λ₀ = (median positive ΔT)/(median positive ΔI/N) over the initial edges. **Candidate settings: r ∈ {0.5, 1, 2}**, common to both seeds.
- **Heads.** H2[c] = μ₂c. The income bins are 64 equal-mass bins of the pooled class-1 income probability over MODEL_FIT people; the binning rule reads no SEX. B1[c] is the mean teacher income posterior over MODEL_FIT people whose cells share c's bin.
- **No checkpoints.** The output is the partition at exactly K.

## 3. Which input distinctions it merges

**Both arms** merge every distinction within a fine cell (inputs whose seven teacher probabilities share a Voronoi cell) and the task-cheapest merges down to K.

**SCAR additionally** merges neighbouring cells whose SEX composition differs more than their task posteriors do. R1 sees only the 64 income bins.

**Expected extra merges** (a hypothesis): profiles reaching similar income and occupation posteriors by routes with different sex mixes.
- **WKHP:** 30–39-hour profiles with 40-hour profiles of slightly lower schooling.
- **COW:** nonprofit (2) and local government (3) with for-profit (1) and state government (4).
- **MAR:** widowed, divorced or separated (2–4) with never married (5) or married (1), at matched age and hours.
- **SCHL:** associate's (20) with some college (18–19).
- **POBP:** birthplaces with similar task profiles but different sex ratios.
- **AGEP:** few.

**Measurement.** For MODEL_SELECT input pairs merged by SCAR but not by the twin, the pilot reports how often each field differs (label-free).

## 4. Why these distinctions should matter less for the tasks than for SEX

A merge's task cost is exactly the divergence between the two cells' task posteriors, whatever their SEX rates. Its SEX gain is exactly the divergence between their SEX rates. Merges are therefore cheap for the tasks and costly for SEX inference exactly where input regions reach nearly the same task posteriors with different sex mixes.

**Reasons to expect such regions.**
1. With η = (1 − π)q⁰ + πq¹ (CORRECTIONS §4; SCAR fits no SEX-conditional model), SEX moves a task posterior only by π(q¹ − q⁰). Where that within-SEX difference is small, sex composition varies while the posterior barely moves.
2. Hours, schooling, sector and marital status trade off against one another in the task posteriors, but not in the same way for SEX.

**Where this could be wrong.** If E[S | task posteriors] changes steeply mainly along directions the tasks need, merges cost as much as any other change. A back-of-envelope (logistic-normal reader score; gradient g of the SEX posterior per unit of the posterior vector):
- Moving the pair from about 0.66 to 0.63 removes about 0.007 of Var(E[S | V]) and costs about 0.007/g² summed Brier: 0.03 at g = 0.5, 0.005 at g = 1.2.
- JOINT_TREE lowered the pair by 0.061 (0.664 → 0.603) at about 0.016 summed Brier above the teacher (DEV_REPORT seed 0, both tasks). Exact removal matches that only if g ≳ 0.9.
- If g is smaller, continuous corrections gain from reader limitations what removal cannot, and SCAR is dominated on the declared metric even if it works as designed.

**One lever SCAR does have.** JOINT_TREE's cost fell on income (+0.0103 against about 0.0099 available), while occupation (+0.0060) left about 0.004 unused. SCAR's per-clause budgets stop income spending at its cap and push merges to the occupation side.

## 5. Why the neural comparator may retain them

**It cannot merge.** NEURAL_RAW ρ1 E2 is U0 plus 80 adversarial steps at learning rate 1e-4: a continuous function of the encoded input, with no mechanism designed to give different inputs bit-identical outputs (whether it is injective on the population is unknown, CORRECTIONS §2). The task loss rewards any resolution, so it keeps every distinction SCAR merges, available to a stronger reader.

**Its stronger settings failed income clauses**, mostly summed Brier (EVIDENCE_PACK B). Shifting an unmerged output pays in its calibration term while resolution, and the SEX information, stay. A SCAR merge is calibrated on its cell and pays only resolution. The caps charge both terms; this efficiency difference is an untested hypothesis.

**Stage-1 diagnostic: NR_PROJECTED.** NEURAL_RAW's frozen pair wire, averaged within SCAR's cells and re-read. Its drop measures how much of the comparator's recoverable SEX lies in distinctions SCAR merges: the "information the neural control demonstrably retains" that `CLOSED_LANES.md` asks for.

## 6. How utility is maintained

**Per-clause budgets.** Plug-in costs are exact for all six metrics, and each clause is capped. Pooled heads carry no within-cell calibration error relative to the teacher.

**Margin.** The teacher's MODEL_SELECT headroom over U0 is held back for generalisation: summed Brier 0.0041/0.0045 for income and 0.0051/0.0036 for occupation. All six metrics have positive headroom; the thinnest is income accuracy, at 0.0012/0.0006.

**Selection.** The frozen rule over r ∈ {0.5, 1, 2}: all 24 MODEL_SELECT point clauses on both seeds first, then the minimum mean frozen OUT_LR coalition proxy. If none is feasible, the result is an ineligible, descriptive fallback.

**Reported, not gating.**
- Plug-in against realised MODEL_SELECT costs. Fitting-pool guards proved optimistic in the closed lanes, and this budget rule is one.
- Within-SEX occupation Brier excess against U0. Merging sex-heterogeneous cells pulls each group's pooled posterior towards the other's; MMP's prototype hit 2.2–2.6× the guard for men.

## 7. What prevents critic-specific masking

- **No critic is trained.** The privacy term is a plug-in functional of cell SEX rates.
- **Merges are exact.** Co-cell inputs get bit-identical outputs; distinct transmitted pairs number at most the coarse cells hit. No reader of any class can use a merged distinction.
- **Readers and controls.** The v1 lookup reader attacks the cells directly. Learning curves on 25/50/100% of ATTACK_FIT should flatten under removal but not under masking. Shuffled-SEX and planted-bit controls check the pipeline. A diagnostic reports the AUC of cell-averaged out-of-fold π̂ on MODEL_SELECT.
- **Gates.** G3 compares SCAR with SCAR_PERM (SCAR driven by a permuted π̂), separating SEX-specific merges from merely departing from task-only merging; in my toy that departure alone explained 0.005–0.022 of the gap. G4 requires the benefit under nonlinear and lookup proxies.

## 8. The transmitted output

- **R1** receives f1(x) ∈ Δ² (one of at most 64 stored float64 vectors) and d1.
- **R2** receives f2(x) ∈ Δ⁵ (one of K stored vectors) and d2.
- **Nothing else.** No cell or bin index, teacher output, SEX proxy, state or randomness is sent.
- **What each view identifies.** A vector identifies its cell to whoever sees it (SERVICE_AND_THREAT_MODEL v1 §1): R2 and the pair identify the coarse cell, R1 only its bin. Merged inputs are bit-identical; distinct cells with near-equal vectors stay distinguishable.
- **Contract.** Inside the primary contract (CORRECTIONS row 6).

## 9. Strongest falsifying result (pre-registered)

The gates run on MODEL_FIT and MODEL_SELECT before any ATTACK role is touched. Each must hold on both seeds; any failure stops the pilot as NOT_SUPPORTED_AT_GATE.

| Gate | Requirement |
|---|---|
| G0 | SCAR_TWIN passes all 24 MODEL_SELECT clauses |
| G1 | at least one r passes all 24 |
| **G2** | selected SCAR's OUT_LR coalition proxy ≤ SCAR_TWIN's − 0.020 |
| G3 | selected SCAR's OUT_LR proxy ≤ SCAR_PERM's − 0.010 |
| G4 | SCAR_TWIN − SCAR ≥ 0.010 under the OUT_HGB and lookup proxies |

**Falsifier.** G0 and G1 pass, but G2 fails. At matched capacity and within the caps, SEX-aware exact merging of task-similar shared cells then fails to lower even the linear coalition proxy by 0.020 below task-only merging. That refutes the hypothesis for this population, K and λ, and points to the §4 geometry. If G0 fails instead, a shared finite partition cannot meet the caps even without privacy (FARE-like).

**Downstream falsifier.** A DEV_REPORT coalition AUC not below NEURAL_RAW's 0.640 / 0.638.

**My rough priors.** G0 passes 0.6; G2 passes given G0 0.3; established plain superiority over NEURAL_RAW ≤ 0.2.

## 10. Difference from closed lanes and prior methods

**Closed lanes.** SCAR has no tree or hybrid composition or leaf-score correction (the teacher's leaves are not the release partition), no adapter, and no critic to schedule or refresh. It has no decoder, calibrator or mixture on frozen outputs: the partition is learned here, with SEX, and the heads are pooled posteriors on it. It is not a penalty sweep: r is the permitted setting; the change is the merge operator. The Adult latent codes (decision-preserving, i8o64, 10-bit with MMD) were critic- or MMD-trained with trained decoders, per recipient as far as their summaries state. SCAR is a related family, so the prior-work reviewer must judge the overlap. Feature masks delete fields; SCAR merges only task-similar regions.

**Not MMP or RCCS.** No SEX-conditional teachers or mixture (MMP). No propensity or shift at deployment (RCCS); π̂ only orders merges during training.

**FARE** grows one fair CART-style tree per task: at most 100 cells, a demographic-parity criterion, a logistic head. It was utility-infeasible here even without fairness (Z1 failed 12/12). SCAR's cells instead:
- live in task-posterior space, so hundreds to thousands of them can track the posterior;
- are shared by both recipients, which meets FARE's coverage condition "both outputs functions of the same cells" (a FARE-style certificate would be vacuous at K ≈ 10³; none is claimed);
- come from an information criterion.

**Other methods.** FAGTB is continuous adversarial boosting of one output. Adversarial representations give protection relative to the critic class. The privacy funnel (Makhdoumi et al. 2014) merges greedily for min I(S; Y) s.t. I(X; Y) ≥ R under a known law; SCAR's utility is instead task-posterior resolution (IB with side information, Chechik and Tishby, NIPS 2002), estimated from teachers, with two capped tasks and a twin.

**Established technique.** k-means, Ward linkage, KL and Bregman merge identities (Banerjee et al., JMLR 2005), agglomerative IB (Slonim and Tishby, NIPS 1999) and the privacy funnel. I did not reopen these papers in this session. The prior-work reviewer should check the citations, and whether a coalition-closed shared-partition release exists.

**Joint component.** The shared partition is a representation choice. No claim is made that joint design beats sequential design (§9.1), and no claim-5 endpoint is registered. The justification is statistical and computational:
- the coalition holds one K-cell partition, whose SEX rates are each estimable from about N/K people, whereas separate partitions join into up to K1·K2 cells;
- all costs are closed forms in one set of statistics.

## 11. The task-only twin

**SCAR_TWIN** shares the teachers, fine cells, K, graph, agglomeration code, and head and income-bin rules, with λ = 0. It never fits π̂ or reads SEX. It is matched in capacity (K cells), data exposure (MODEL_FIT features and task labels) and steps, minus the π̂ fits, which are the privacy component.

**SCAR_PERM** is a diagnostic, not the twin: SCAR with π̂ permuted across MODEL_FIT people (fixed seed) and λ₀ recomputed, at the selected r.

## 12. Implementation sketch and compute

**Reused code.** `acs_replication.trees` (`make_classifier`, `from_catboost`, `TreeTeacher`, `verify_against_catboost`), `acs_replication.data` (`internal_split`, `clauses`, `task_vectors`) and `selection.proxy_auc`. Also the stages, readers and endpoints of `mechanism_design/pilot/pipeline.py`, with SCAR and SCAR_TWIN in place of MMP and its twin, plus the lookup reader, NR_PROJECTED and SCAR_PERM.

**New code.** `partition.py` (k-means, graph, budgeted agglomeration, K rule), `release.py` (map, schema guard, routing) and `checks.py` (§13).

**Compute per seed, single-threaded.**

| Step | Time |
|---|---|
| 4 CatBoost fits (2 teachers on 20,000; 2 π̂ halves on 10,000) | ≤ 20 min |
| k-means (8,192 × 7) | ≈ 2 min |
| 5 agglomerations | ≈ 10 min |
| MODEL_SELECT clauses and 3 proxies for 5 arms | ≈ 5 min |

About 37 minutes per seed, so roughly 75 CPU-minutes for both seeds, within the 2 CPU-hour budget. Stages 1–2 follow the prepared pilot.

## 13. Minimal finite-law check

**Specification.** Exact laws, no fitting, on a 10 × 10 grid x = (a, b) with uniform mass: a is task-relevant, b is a "route". Y1 is binary and Y2 has three classes, both with exact posteriors; S is binary.
- **ROUTE:** tasks on a, SEX on b.
- **PARTIAL:** tasks mostly on a; SEX mostly on b, partly on a.
- **CONFLICT:** tasks and SEX on a only.

Reported for the twin and each r: exact pair and single-recipient AUC\* keyed on bit-identical transmitted vectors, per-task Brier, NLL and accuracy excess, and resolution kept.

**Checks.**
- **(a)** Inputs that M puts in one coarse cell get bit-identical outputs; distinct pairs are counted against cells.
- **(b)** Repeated and co-cell inputs are bit-identical (on real data: label-free on MODEL_SELECT, before G0).
- **(c)** A task-feasible nontrivial reference exists: the twin at K ≥ the number of distinct posteriors has zero excess (on real data: G0).
- **(d)** No label leakage: deployment takes no SEX, and SCAR built from a permuted SEX posterior loses most of the benefit.
- **(e)** No omitted wire or useless task: wherever protection is claimed, both releases keep their resolution, and a constant release (AUC\* 0.5) fails the caps.
- **(f)** In CONFLICT, no setting within the caps lowers AUC\* by more than 0.01.

**My toy run.** A numpy script (not delivered, since one file was requested, and not independently verified). It uses the §2 formulas, the λ₀ rule, an 8-NN plus minimum-spanning-tree graph and grid cells as fine cells, with neither k-means nor budgets.
- **ROUTE** (AUC\*(S; X) = 0.703). At K = 10 the twin and SCAR both reach 0.500 at zero cost: the twin removes free information too. At K = 50 the twin keeps numerically equal posteriors in separate cells, whose float64 pooled vectors differ in the last bit (13 distinct vectors for 10 real values; AUC\* 0.547). SCAR merges them explicitly (0.500). Merges must be explicit.
- **PARTIAL** (0.711), K = 25. Twin 0.681; SCAR 0.627 / 0.612 / 0.592 at r = 0.5 / 1 / 2, with excess ≤ 0.0006 summed Brier and ≤ 0.0010 NLL, no accuracy loss and ≥ 99.6% of resolution kept. Permuted-SEX SCAR (r = 1, 5 permutations): 0.659–0.676.
- **CONFLICT** (0.703), K = 25. r = 0.5 changes nothing; r = 1 gives 0.695 at Brier 0.0033/0.0043. r = 2 reaches 0.587 only at a summed Brier of about 0.10 (20× the cap), which feasibility rejects. At K = 5 the twin itself exceeds the Brier cap (0.0053/0.0059): the G0 failure mode.

These laws were built for the checks. They show the mechanics, not that ACS has a PARTIAL-like structure.
