# Designer 1 proposal: Shared-Cell Merging (SCM)

Status: data-free proposal. No real-person array was opened (nothing under `.private/`, no `.npz`), nothing was fitted on real data, and `roles/designer_2/` was not opened. The only computation was the exact finite-law pre-check in section 13, on invented laws.

## 1. Name and summary

**Shared-Cell Merging (SCM)** releases both recipients' probabilities from **one learned many-to-one partition** of the six-field input space.
- **Cells.** A fine partition (512 cells) of the space of cross-fitted CatBoost task posteriors is merged greedily. Each merge pools two cells into one calibrated cell posterior.
- **Merge choice.** Each step takes the pair that removes the most between-cell SEX-rate variance per unit of exactly computed utility cost (NLL, Brier and accuracy for both tasks, plus the within-SEX occupation guard), each cost under its own registered budget.
- **Releases.** Recipient 2 gets its cell's pooled occupation posterior. Recipient 1 gets the pooled income posterior of one of 32 equal-mass groups of cells.
- **Coalition bound.** Both outputs are functions of the same final cell C, so AUC\*(S; V) ≤ AUC\*(S; C): a finite table that can be estimated directly, without a critic.
- **What is established.** The ingredients: privacy-funnel and agglomerative-IB merging, Bregman k-means, cross-fitting. No algorithmic novelty is claimed, and nothing yet shows SCM is competitive.

**Why this family (a known-law fact; standard calibration–refinement reasoning).**
- For any deterministic release (f1, f2), let C be the partition its pair induces, and replace each f_i by P(Y_i | C).
- This weakly lowers expected NLL and Brier, weakly raises Bayes accuracy, and cannot raise AUC\*(S; V), because the new pair is a function of C.
- So against a Bayes attacker, "partition + cell posteriors" weakly dominates every deterministic release. This says nothing about finite samples or fitted readers (W2).

## 2. Mapping and optimisation problem

| Step | Role | Labels | SEX |
|---|---|---|---|
| S1. Cross-fitted teachers: T_TREE registry (depth 4, lr 0.05, ≤ 800 trees, l2 3), 5 household folds, each with internal 15% household early stopping. Out-of-fold (OOF) posteriors t1_j ∈ Δ², t2_j ∈ Δ⁵ | MODEL_FIT | Y1, Y2 | no |
| S2. Fine shared partition: KL Bregman k-means on (t1_j, t2_j), distortion KL(t1‖μ1) + KL(t2‖μ2), M0 = 512, k-means++, 50 Lloyd steps. Centroids are member means, so the objective is the NLL quantization cost | MODEL_FIT | none | no |
| S3. Cross-fitted SEX propensity π̂_j (T_TREE binary registry, same folds) | MODEL_FIT | — | training signal only |
| S4. Exact greedy merging under budgets. The guard uses SEX-specific occupation label means per cell, shrunk toward the cell mean (κ = 20) | MODEL_FIT | Y2 (guard) | training signal only |
| S5. Table: v2[c] = mean OOF t2 over the cell's members; final cells sorted by mean OOF t1 and cut into 32 contiguous equal-mass groups; v1[g] = the group's pooled t1 | MODEL_FIT | none | no |
| Selection (frozen rule); CALIBRATION unused | MODEL_SELECT | yes | proxy AUC only |

**Objective.** The coalition sees exactly C, so the Bayes attacker's SEX-Brier advantage is L(C) = Σ_c (n_c/N)(π_c − π̄)², where π_c is the mean of π̂ over the cell. SCM minimises L over coarsenings C of C0, subject to Δ_k(C) ≤ B_k:

| Constrained quantity | Budget B_k |
|---|---|
| NLL, each task | β·0.010 |
| Summed Brier, each task | β·0.005 |
| Accuracy, each task | 0.005 in every setting (income has no teacher accuracy headroom) |
| Within-SEX occupation Brier, each SEX | β·0.005 |

Δ is the exact plug-in increase relative to C0, taking the OOF posteriors as truth.

**Exact merge accounting** (checked against brute force in section 13). Pooling cells A and B into U, with masses n, means μ and SEX rates π:

| Quantity | Exact change |
|---|---|
| Brier, task i | (n_A n_B/n_U)‖μ_{i,A} − μ_{i,B}‖²/N |
| NLL, task i | [n_A KL(μ_{i,A}‖μ_{i,U}) + n_B KL(μ_{i,B}‖μ_{i,U})]/N |
| Accuracy loss, task i | [n_A(max μ_{i,A} − μ_{i,A}[argmax μ_{i,U}]) + n_B(…)]/N |
| SEX gain | G = (n_A n_B/n_U)(π_A − π_B)²/N |

**Algorithm and checkpoint rule.**
- Each step merges the admissible pair (Δ_k ≤ remaining R_k for every k) with the largest G / Σ_k Δ_k/R_k. All pairs are evaluated, with row updates after each merge; ties go to the lowest (a, b).
- The path stops at the first step with no admissible pair. There is no stopping on leakage.
- Each β runs its own deterministic path. The paths are not nested, because the priority normalises by the remaining budget.

**Deployed map (six raw fields only).**
1. The frozen 145-column encoder, which refuses SEX and every other forbidden field.
2. t̄(x): the average of the 5 fold teachers, as explicit numpy trees.
3. c0 = argmin_c KL(t̄1‖μ1_c) + KL(t̄2‖μ2_c).
4. c = merge_map[c0] and g = group[c].
5. R1 = (v1[g], argmax) and R2 = (v2[c], argmax). Stored float64 vectors are copied, never recomputed; ties go to the lowest index or class.

## 3. Which distinctions it merges

**S2 (no SEX used)** merges every input distinction that leaves the pair of task posteriors inside one cell. Expected:
- most POBP levels beyond the regional and occupational effects the teachers use;
- MAR widowed versus divorced versus separated where the task effects match;
- adjacent AGEP years;
- WKHP values inside a cell (40 versus 42).

Some of these carry SEX in X (widowhood, sex-skewed migration origins), and S2 removes them at quantization cost only.

**S4 (SEX used as a training signal)** pools neighbouring posterior cells with opposite SEX tilt. Expected examples, which are hypotheses:
- WKHP 40 versus 45–60 at the same education and COW;
- COW private for-profit versus non-profit versus local government at the same SCHL and WKHP, where occupation is mostly group 0 either way;
- SCHL master's versus professional versus doctorate inside group-0 cells;
- MAR married versus widowed or divorced at matched hours.

**Reported answer: a label-free merge ledger on MODEL_SELECT inputs.** For each field and each pair of its levels, it gives the share of people whose final cell is unchanged when that field is switched with the other five held fixed. It is reported for SCM, the twin and C0.

## 4. Why those distinctions should matter less for the tasks than for SEX

- **The task price.** For a calibrated release, the task value of keeping A and B apart is exactly the pooling cost: quadratic in the posterior gap for Brier, a KL divergence for NLL.
- **The SEX price.** The value to a Bayes attacker is quadratic in the SEX-rate gap.
- **The ratio.** Merge efficiency is (π_A − π_B)²/‖μ_A − μ_B‖². The mass factor cancels, and S4 spends the budget in decreasing order of this ratio.
- **Where cheap merges exist.** If E[S | posteriors] is L-Lipschitz, each unit of Brier buys at most L² units of SEX variance. Cheap merges exist only where SEX varies faster than the task posteriors.

**Where this could be wrong.** If the posteriors' own level sets carry most of the SEX signal (spec §9.5; task-only occupation alone reads 0.650), S4 has little room. Law C in section 13 shows such a floor.

## 5. Why the neural comparator may retain them

- **No merging.** NEURAL_RAW ρ1 E2 is U0 (a ReLU network on 145 columns) after 80 Adam steps at lr 1e-4. It merges nothing beyond the encoder's OTHER pooling, so every distinction its outputs keep stays available to a stronger reader.
- **Displacement, not pooling.** It moves outputs rather than pooling them. By section 1, its own recalibration would weakly dominate it against a Bayes attacker, so it may pay a calibration cost on top of the resolution cost. That is consistent with every stronger grid point failing, mostly on income summed Brier (41 clause failures).
- **Measured signs.** Its pair still exceeds its best single recipient by 0.018 / 0.015 (triangulation remains). Its occupation recipient (0.622 / 0.623) keeps most of U0's leakage (0.641 / 0.638).

This is consistency, not proof. I cannot show which distinctions its fitted readers use.

## 6. How utility is maintained

- **Exact, separate budgets.** Every merge's cost is exact under the plug-in law, with separate budgets per task, metric and SEX.
- **Calibrated values.** Released values are pooled OOF posteriors, calibrated by construction up to teacher calibration. There is no temperature step, so CALIBRATION is unused.
- **The β settings.** β ∈ {0.5, 1.0, 1.5} is set from the published MODEL_SELECT teacher headroom over U0: income NLL 0.0066/0.0068, Brier 0.0041/0.0045; occupation NLL 0.0121/0.0082, Brier 0.0051/0.0036.
- **Selection (frozen rule).** All 24 MODEL_SELECT clauses first, then the minimum mean OUT_LR coalition proxy, with one common β across seeds. If no setting is feasible, the frozen fallback applies and the candidate is ineligible.
- **Diagnostics.** The guard and the income-grouping cost are reported on MODEL_SELECT.

## 7. What prevents apparent success from being critic-specific masking

- **No critic, structural bound.** Nothing is trained against a critic. The release takes at most 512 values, so any reader's population AUC on V is ≤ AUC\*(S; C).
- **Pre-registered checks.**

| Check | What it is | What it guards against |
|---|---|---|
| Cell-table reader | SEX rates keyed on the exact released pair; fitted on ATTACK_FIT, Beta(1,1)-smoothed, scored on DEV_REPORT. If it beats the slate winner by more than 0.010, the competitive claim is reported NOT_ESTABLISHED | A reader slate that under-reads a finite-cell release |
| Pair learning curves | 25 / 50 / 100% of ATTACK_FIT | Leakage the readers have not yet learned (a rising curve) |
| Shuffled-SEX path | π̂ permuted, stopped at the same final cell count; must not match SCM | Protection from coarsening alone |
| Reader controls | Shuffled-null and planted-bit controls on the SCM wire | Broken reader pipelines |

## 8. The actual transmitted output

- **Content.** Recipient 1 gets (v1[g], argmax) in Δ². Recipient 2 gets (v2[c], argmax) in Δ⁵. Float64, deterministic, one release per person.
- **Nothing else.** No cell or group id, identifier, teacher output, propensity, seed or state.
- **Disclosed and accounted for.** The vectors identify the group or cell. The coalition table is C, the occupation recipient alone resolves C, and the income recipient resolves only the 32 groups.

## 9. Strongest falsifying result (registered before any data)

**F1.** At the selected β, on both seeds, SCM's coalition cell-table AUC on MODEL_SELECT (the table built from MODEL_FIT wires and MODEL_FIT SEX) is **not at least 0.020 below that of its matched task-only twin**. That would mean the SEX-aware choice of merged distinctions buys under 0.02 at feasible utility, refuting the hypothesis here.

**Stage-0 gates** (MODEL_FIT and MODEL_SELECT only; no reader is fitted before they pass; any failure is reported NOT_SUPPORTED_AT_GATE):

| Gate | Condition | If it fails |
|---|---|---|
| G0, family feasibility | The unmerged C0 release, with income grouping, passes all 24 clauses | STOP: the family is infeasible at this resolution (FARE Z1's 100-leaf per-task cells failed 12/12) |
| G1, utility | At least one β is feasible | STOP |
| G2 | F1 | STOP |
| G3, competitive premise | SCM's OUT_HGB coalition proxy ≤ NEURAL_RAW E2's (recomputed from the frozen bank) − 0.012, both seeds | STOP |

**Endpoints** (unchanged; exploratory, because MODEL_SELECT and DEV_REPORT are reused):
- recipe benefit versus the matched twin, lower bound > 0.020 on both seeds;
- recipient worsening ≤ 0.010;
- competitive versus NEURAL_RAW, lower bound > 0, nested 0.020;
- the 24 DEV clauses with the confidence field;
- the subgroup guards.

**Registered bet.** P(G0) ≈ 0.5; P(G2 | G0, G1) ≈ 0.25; P(competitive established) ≈ 0.05. Modal outcome: NOT_SUPPORTED_AT_GATE.

## 10. Difference from the closed lanes and from prior work

**Closed lanes.**
- No frozen output is composed or decoded: teachers, partition and values are refitted inside the candidate.
- There is no adapter and no critic.
- β is a utility budget, not a penalty weight. The structural change is the partition.

**Real overlap: frozen-bank channels** (re-routing frozen codes, Adult best 0.006) **and latent codes** (finest code +0.020 nats occupation log loss).
- SCM *is* a deterministic channel on its own teacher posteriors.
- It differs in: a fine (512-cell) start built for both tasks jointly; one cell shared by both recipients; pooled-posterior heads instead of trained heads; an exact SEX criterion on cells.
- ACS occupation has 5 classes (a 4-dimensional posterior), the only reason to hope the code-lane log-loss bottleneck is milder here. That is unverified (G0).

**Prior work.**
- **FARE.** A fair-Gini/DP tree in raw space with a DP certificate for its cells. Here it was fit per task, so the certificate does not cover the pair. F1 and Z1 both failed 12/12. SCM shares one partition (the case FARE's logic covers), builds cells in posterior space, and merges by exact ratio.
- **FAGTB.** One boosted output against a logistic adversary; no merging.
- **Privacy funnel** (Makhdoumi et al. 2014). Greedy merging for min I(S; Z) subject to I(X; Z) ≥ R under a known law. SCM changes the utility side to two-task NLL, Brier and accuracy plus the guard, uses plug-in cross-fitted estimates, and uses SEX variance in place of mutual information.
- **Other established pieces.** Agglomerative IB (Slonim and Tishby 1999), which is the twin; Bregman clustering (Banerjee et al., JMLR 2005); fair clustering (Chierichetti et al. 2017), here deployed as a function of x only.
- **Adversarial representations, LEACE/INLP.** Critics or linear erasure; not used.

**No joint-component claim (claim 5).** A sequential designer can produce the same release (spec §9.1). The shared cell is chosen for estimability: one table of at most 512 cells instead of a join of up to M1·M2.

## 11. The task-only twin

- **Identical to SCM in** S1, S2 and S5: the same fits, budgets and all-pairs loop.
- **Privacy component disabled.** S3 is not used, and the priority is −Σ_k Δ_k/R_k, the cheapest normalised merge (agglomerative IB). No SEX enters at all.
- **Matched capacity.** For each β the twin stops at SCM's final cell count; any shortfall is reported.
- **Result.** Capacity, rows, roles, teachers and loop are matched; only which distinctions are merged differs.
- **Descriptive extras** (not endpoints): C0 and a budget-matched twin.

## 12. Implementation sketch and compute

- **Code.** Reuses the pipeline shape of `mechanism_design/pilot/pipeline.py`: `fit_catboost`, `utility_clauses`, `proxy_auc`, `merge_diagnostic`, the stage 0/1/2 structure and `endpoint`. About 250 new lines of numpy: Bregman k-means, merge loop, table, ledger and cell-table reader.

| Compute item | Estimate |
|---|---|
| Per seed: 15 CatBoost fits (10 teachers + 5 propensity; ≤ 800 trees, about 13.6k rows, single thread) | about 0.5–1.5 min each |
| Per seed: k-means (20k × 7, 512 centroids) | about 30 s |
| Per seed: seven merge paths (3 β × SCM and twin, plus the shuffled control) | about 2 min |
| Per seed: releases and proxies | about 2 min |
| **Total, both seeds** | **0.5–1 CPU-h, under 1.5 with margin** |

- **Settings.** Three fixed settings per seed: β ∈ {0.5, 1.0, 1.5}. Everything else is fixed (M0 = 512, 5 folds, 32 groups, κ = 20, accuracy budget 0.005, seeds `seed_for('SCM', s, …)`). The checkpoint rule is in section 2.
- **Stage 1.** The frozen reader slate plus the cell-table reader, outside the fitting budget.

## 13. Minimal finite-law check

**Pre-check: exact enumeration, run in my scratch directory.** The script is not in the repo; the laws are specified below for re-implementation.
- **Fields.** H ∈ {0,1,2} (P = .25/.55/.20), E ∈ {0,1} (P(E=1) = .4), Z ∈ {0,1} (P(Z=1) = .1, a pure proxy).
- **SEX.** logit P(S=1|x) = (.9, 0, −.9)[H] + 1.5Z.
- **Income.** P(Y1|x) = σ(−1.6 + 1.2H + 1.3E).
- **Occupation, 3 classes.** Logits (−.5 + 2E, .6 − .6H, −.8 + .7H − .8E).
- **Law F** adds U (P = .5; small task effects, SEX-neutral) and V (P(V=1) = .3; SEX +1.2, small task effects).
- **Reference and caps.** The reference is the exact posterior. Caps (accuracy / NLL / Brier, each task): .03/.03/.015 for laws A, B and D; .01/.01/.005 for laws C and F.

| Law | What it shows | Exact result |
|---|---|---|
| A, proxy + conflict | S2 merges distinct inputs exactly when the posteriors coincide (12 inputs → 6 cells). Outputs within a cell are bitwise identical. The Bayes AUC on the full wire (both vectors and both decisions) equals the cell AUC, so no wire is omitted | AUC\*(S; X) 0.6755 → 0.6421 at zero cost → 0.6348 after one merge. **The twin makes the same merge: the free S2 gain is shared with the twin** |
| B, pure proxy only | Useful deterministic removal (as in check B) | 0.5574 → 0.5000 at zero cost |
| C, conflict only (H drives SEX and both tasks) | Sensitive and necessary task information conflict | 0.6467 → 0.6392 (−0.0075) at income NLL +0.0058, Brier +0.0038; then the budget binds |
| D, useless occupation task | Protection inflates when one task is useless; the gain-over-constant check flags it | 0.6105, with occupation gain over constant exactly 0 |
| F, rich law | Recipe effect at matched capacity; shuffled-SEX control | SCM, 7 cells: 0.6391 (income +.0022 NLL / +.0019 Brier; occupation +.0021 / +.0013). Twin, 7 cells: 0.6539. Shuffled-signal controls at 7 cells: 0.6487, 0.6503, 0.6407 |

- **Exactness.** At every merge, the closed-form costs equal brute force to 1e-12.
- **No label leakage.** The release is a function of x alone; SEX enters only through the merge priority.
- **A task-feasible, nontrivial reference exists in every law.** The stage-2 release meets the caps with a positive gain over constant on both tasks, except the deliberately useless task in law D.
- **Before any real data.** The pilot repeats these checks on a synthetic stand-in with the real 145-column shape (a planted pure proxy and a planted near-proxy), plus a label-free within-cell identity check on MODEL_SELECT inputs and a wire-completeness check.

## Weaknesses: where the rationale could be wrong

- **W1. Required disclosure** (law C) may leave S4 little room. S2's gains are shared with the twin, so they cannot create a recipe effect.
- **W2. No learnability shield.** A release with at most 512 values is fully learnable, so fitted readers will measure SCM near its Bayes cell value, while continuous comparators benefit from reader limits. Precedent: the teacher's income output reads 0.569, but FARE Z1's 100-cell income output reads 0.646. SCM can be more honest and still lose the head-to-head.
- **W3. Noise and calibration.** About 39 people per initial cell, so greedy selection on noisy π_c regresses to the mean on new people. The OOF teacher versus bag-average mismatch can miscalibrate cell values.
- **W4. Subgroup guard.** SEX-aware pooling pushes cost onto the minority-SEX members of merged cells (MMP failed this guard at 2.2–2.6×), and the guard estimate is noisy.
- **W5. Partial proxy.** SEX variance is not AUC.
- **W6. Closed-lane overlap** is real (section 10).
