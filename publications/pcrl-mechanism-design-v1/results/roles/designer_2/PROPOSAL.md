# Designer 2 proposal: release-conditional counter-shift (RCCS)

## 0. Header
- **Designer:** 2. **Decision:** `PROPOSAL`. **Finished:** 2026-10-16T05:32:11Z.
- **Inputs read:** ROLES_AND_OWNERSHIP, PROBLEM_SPECIFICATION, EVIDENCE_PACK, CLOSED_LANES (read before §3.7, §9 and the decision), the four `acs_replication` modules, SCIENCE_LOCK, SELECTION_DETAIL, the source `RESEARCH_DECISION.md` / `DEV_REPORT_RESULTS.json` reader-slate matrix, the frozen-hybrids `START_HERE.md`, and the minimal-adapter `RESEARCH_DECISION.md` (summary only). Nothing under `roles/designer_1/`. No person-level file.
- **Units (all synthetic):** `D2_GAUSS_V1`, `D2_FINITE_V2`, `D2_FINITE_V3`, `D2_FINITE_V4` (V4 adds the coin control). `D2_FINITE_V1` is superseded: float rounding in its toy law split cells with equal posteriors. V2–V4 use exact fractions.

## 1. Plain-language summary
Each recipient's output can carry SEX information on its own. The linked pair also reads the *contrast* between the two outputs, meaning the part of one output that the other does not explain. Because the outputs are strongly correlated (R² ≈ 0.82), SEX signal in that contrast is magnified. The published reader AUCs show the effect. Read as Gaussian-equivalent signal, the pair's excess over occupation alone is 5–7.6× the income-only signal in every service whose occupation output was protected: LOCAL, JOINT, H_LJ and H_JL. It is about 1× in the task-only services. Protecting each output on its own therefore leaves the residual contrast between the outputs exposed, and pushing the income output harder hits the binding income Brier clause first. RCCS leaves income exactly at its task teacher, so it spends no income budget. It then moves only the occupation output, where the budget has slack, so that occupation adds no linear or quadratic SEX information *beyond what the income release already gives*. The occupation output keeps the SEX signal that income already reveals (redundancy) and loses the signal unique to occupation. The shift is driven by an explicit, cross-fitted sex-propensity score of the six fields and solved as a small deterministic program, with no adversary and no SEX at inference. A Gaussian model calibrated to the published AUCs predicts pair AUC ≈ 0.60–0.61 at about LOCAL_TREE's occupation spend, with income untouched. That prediction is uncertain and is falsified by the stated tests.

## 2. Release contract
This is the **primary contract, unchanged**. Recipient 1 receives the T_TREE income probability vector and its argmax, both deterministic. Recipient 2 receives `softmax(z2(x) + A ψ(x))` and its argmax, a deterministic function of the six fields only. ψ uses five SEX-propensity tree models fitted on MODEL_FIT and evaluated on the six fields. There is no shared state, randomness, metadata, abstention or interaction. All comparators already live under this contract.

## 3. The eight questions
**3.1 What is removed or changed.** The release-conditional SEX information of the occupation output, meaning the information it adds given the income output:
- the partial covariance Cov(φ2(f2), S | φ1(f1)) of its wire features (probabilities, log-probabilities, decision);
- the S-dependence of their second moments.

Not removed:
- anything in the income release;
- the part of occupation's SEX covariance that is predicted by its covariance with income. That part is deliberately kept.

**3.2 Why it is unnecessary at the required utility.**
- The occupation clauses score only how well f2 predicts Y2. The shift moves f2 along a propensity residual whose effect on S-covariance is first order, while its proper-score cost E‖δ‖² is second order.
- Income utility is untouched. T_TREE passes every income clause with margin: Brier −0.0041 / −0.0045 vs U0 on MODEL_SELECT.
- The occupation budget is slack in every protected ACS service (MODEL_SELECT, vs U0, s0/s1):
  - NEURAL_RAW occupation NLL +0.0004 / +0.0013 (cap 0.010), Brier +0.0003 / +0.0003 (cap 0.005);
  - JOINT_TREE NLL +0.0025 / +0.0029;
  - LOCAL_TREE NLL −0.0030 / −0.0020.
- LOCAL_TREE cut occupation-only Gaussian-equivalent signal by 71% (AUC 0.650 → 0.582) for 0.0035 / 0.0027 summed Brier and 0.009 / 0.006 nats from T_TREE. The budget from T_TREE is 0.010 / 0.009 Brier and 0.022 / 0.018 nats.

**3.3 Why NEURAL_RAW retains it** (code and published aggregates).
- (a) Its objective is risk(income) + risk(occupation) + risk(pair) (`neural_raw.py`, RAW-J). The pair-term gradient moves the income output, whose summed Brier is the binding clause (the most frequent failed clause, 41 failures across the grid). Only the earliest checkpoint (ρ1 E2) is feasible, and it leaves about 90% of the occupation budget unspent.
- (b) In Gaussian-equivalent units (d′² = 2Φ⁻¹(AUC)², DEV_REPORT s0), its pair (0.257) decomposes as follows:
  - occupation alone: 0.193;
  - a partial-income term: 0.064, which is 2.4× its income-only 0.027.
  Equivalently, W_pair = W_inc (0.027) + W_occ|inc (0.230). So 90% of the pair's signal is the occupation output's information given income, which is exactly RCCS's target, and it sits in the channel whose budget is unspent.
- (c) Its own-view occupation term pushes occupation's marginal SEX covariance toward zero, which opposes the redundant solution. In the exact finite law (§11), the own+pair objective needs 27% more budget than RCCS for the same pair AUC.

**3.4 Computation and principle.** Two exact facts about linear information, both verified in `D2_GAUSS_V1`. Write c = Cov(F, S) = Cov(F, π) for π = P(S=1 | X), and Σ for the covariance of the release F = (f1, f2).
- **(i) Amplification law.** W_pair = W_occ + (c1 − b′c2)² / Var(f1 | f2), where W = c′Σ⁻¹c is the linear information.
  - Local protection drives c2 → 0, which turns income's covariance into a partial covariance amplified by 1/(1−R²). Published amplifications: LOCAL 5.67, JOINT 5.88, H_LJ 5.06, H_JL 7.60 (5.58 for LOCAL's linear readers alone), against TASK_REFIT 0.94, T_TREE 1.02, U0 1.16 and NEURAL_RAW 2.39. That implies R²(f1 | f2) ≈ 0.82.
  - The pair-optimal occupation target with income fixed is **c2 = Σ21 Σ11⁻¹ c1**, i.e. zero partial covariance. It is not c2 = 0.
- **(ii) Counter-shift cost (linear-Gaussian model).** A driver correlated with π that is *not* a function of F removes all linear information at calibrated cost ‖c‖²/Var(π). Deterministic post-processing of F costs ‖c‖²/Var(E[π | F]). Invertible linear post-processing leaves W unchanged, and any invertible post-processing leaves the Bayes information unchanged.
  - The driver is the cross-fitted propensity π̂. The shift A is chosen to minimise the closed-form partial criterion of §4.

**3.5 Why it does not merely fool its own critic.**
- No critic is trained. The criterion is the exact best ridge critic, refit in closed form at every evaluation on out-of-fold MODEL_FIT data, so there is no critic lag to exploit.
- Evaluation uses the fresh 7-reader slate on the ATTACK roles.
- The criterion includes quadratic features, and a registered OUT_HGB MODEL_SELECT gate stops the pilot if the nonlinear proxy exceeds the linear proxy by > 0.03 (§12).
- A placebo driver (shuffled SEX) and a post-processing arm separate real removal from feature artefacts.
- Honest limit: the release is injective on the input support, so its population Bayes AUC*(S; V) equals AUC*(S; X). That is equally true of NEURAL_RAW and the trees. Exact check: the injective shift INJ attains AUC*(S; X) in every finite law. All claims are fitted-reader claims.

**3.6 Why this moves the frontier.**
- NEURAL_RAW ρ1 and JOINT_TREE trade income Brier for pair AUC at about 0.12–0.16 summed Brier per unit of proxy AUC (MODEL_SELECT). RCCS sits off that curve: its income Brier is constant at the teacher.
- At equal occupation spend, the calibrated model predicts:
  - RCCS pair 0.60–0.61;
  - the same machinery with the local target (MIN), 0.62–0.71, i.e. a LOCAL-like pair with no gain (calibration replay at d′²_X = 0.40: MIN reproduces LOCAL's 0.657 as 0.656).
- The test compares targets under identical machinery and budget (§12). It does not compare checkpoints.

**3.7 Difference from the closed lanes.**

| Lane | Difference |
|---|---|
| Critic refresh | No critic: a closed-form ridge criterion, recomputed exactly. |
| Strength / allocation / schedule | The objective is a different functional (the partial information of f2 given the actual f1), not a re-weighting of RAW-J terms. Income gets **no** update. Holding it fixed is a choice specific to ACS FL, justified by measured income-binding and occupation slack (on Adult the binding clause was occupation). The arm NR_OCC (RAW-J with income frozen, same budget) is the control for "this is only allocation". |
| Conditional kernels | Those conditioned on task *labels*, which are unreleased (§9.3). RCCS conditions on the other recipient's *released output*, which is what the coalition sees. |
| Utility guards | Not a guard. The lesson that fit-pool checks fail held out is addressed by cross-fitting, a 30-parameter correction and κ < 1 budget fractions selected on MODEL_SELECT. |
| Small adapters | The 325-parameter head-row adapter and the two-recipient adapter had no SEX-propensity driver. In the exact law that makes them post-processing (COND_PP: 2× cost). Post-processing can lower W only by discarding task information (§3.4 ii). The minimal adapter moved occupation-only AUC by only 0.010, so it never tested the conditional target at strength. |
| Masks / codes / VIB | RCCS adds a counter-shift; it removes no inputs and quantises nothing. |
| Frozen-bank / LP kernels / decoders | Post-processing of frozen outputs, which is exactly the COND_PP control. Calibration is not the protection source. |
| Protected trees | L_TREE uses the local target and, by its measured 5.67 amplification, creates the partial term. JOINT_TREE's local terms oppose redundancy and its pair term moved income. |
| Frozen hybrids | H_LJ's JOINT occupation was conditioned on a different income (its amplification is 5.06). RCCS conditions on the income it is actually paired with. |
| Joint vs local / sequential | No joint claim. RCCS is sequential (income first, fixed) with full look-ahead. Its claim is about the *target* and the *driver*. |

**3.8 Falsifiers** (any one rejects the explanation).
1. At matched κ, RCCS's MODEL_SELECT pair proxy is not ≥ 0.015 below MIN's, or RCCS's "pair minus best single" is not < 0.02. The predicted signature is that RCCS's occupation-only AUC is *higher* than MIN's (≈ 0.60 vs 0.58) while its pair is lower.
2. RCCS ≈ COND_PP or ≈ COND_SHUF (difference < 0.01): the driver is not the ingredient.
3. RCCS does not beat NR_OCC by ≥ 0.01 at matched κ: any gain over NEURAL_RAW is allocation, not mechanism.
4. The nonlinear-proxy gap is > 0.03, or no κ is MODEL_SELECT-feasible.
5. Precondition (MODEL_FIT, before any fit of A): the out-of-fold linear decomposition of T_TREE's wires shows R²(φ1 | φ2) < 0.6, or W_occ|inc < 50% of W_pair, or the propensity residual has V̂/Var(π̂) < 0.05. Any of these means the amplification account or the driver premise is wrong for this data.

## 4. Formal objective
Notation follows PROBLEM_SPECIFICATION:
- T1, T2: the T_TREE releases; z2: centred occupation raw scores.
- φ1 = (p1, logit p1, d1) and φ2 = (p2, log p2, e_{d2}): the reader wire features.
- q = (p1·p2, upper(p2 p2ᵀ)): quadratic block.
- π̂: cross-fitted SEX propensity.
- r(x) = logit π̂(x) − ℓ̂(φ1(T1), φ2(T2)): OLS residual on MODEL_FIT, coefficients then frozen.
- ψ = (r, r·g) with g = (1, p1, p2,1..5), so J = 7.

Release: f1 = T1; f2^A = softmax(z2 + Aψ), where A ∈ ℝ^{5×7} has centred columns (30 free parameters).

minimise over A: Q(A) = R²_cv(S ~ φ1, φ2(f2^A), q(f1, f2^A)) − R²_cv(S ~ φ1) + γ‖A‖²_F

subject to, out-of-fold on MODEL_FIT:
- occupation NLL(f2^A) − NLL(T2) ≤ κ b_nll;
- summed Brier difference ≤ κ b_br;
- accuracy loss ≤ κ b_acc.

Definitions:
- R²_cv is the household-grouped 5-fold ridge R² of MODEL_FIT SEX, closed form per fold, with decisions replaced by a temperature-0.05 softmax surrogate during optimisation.
- b are T_TREE's occupation budgets (headroom plus cap, the minimum over U0 and Ucal on MODEL_FIT).
- κ ∈ {0.25, 0.5, 0.75, 1}; γ = 10⁻⁴ is fixed.
- In its population version, Q = 0 if and only if occupation's wire and quadratic features add no linear predictive information about S beyond income's.

Solver: SLSQP on 30 parameters, deterministic.

## 5. Training (access by role)
1. **MODEL_FIT** (features, Y2, SEX):
   - T_TREE is reused (frozen).
   - π̂: 5-fold household-grouped CatBoost (the T_TREE registry, target SEX). MODEL_FIT rows get out-of-fold π̂. Deployment uses the mean of the fold-model logits.
   - Fit ℓ̂, then solve §4 for each κ and seed.
2. **CALIBRATION:** unused; Ucal stays fixed.
3. **MODEL_SELECT:** the frozen rule (24-clause point feasibility, then the minimum mean OUT_LR pair proxy, with κ common to both seeds). The nonlinear gate is reported and acts only as a stopping rule.
4. **ATTACK_FIT / ATTACK_VALIDATION:** readers only.
5. **DEV_REPORT:** exploratory only. SEX is used only on MODEL_FIT.

## 6. Inference
Six fields → the 145-column encoding (MODEL_FIT moments and levels) → T1 (income release and decision). Then z2 and the five SEX fold models → π̂ → r → ψ → f2 = softmax(z2 + Aψ) and d2 = argmax f2. No SEX, no other prohibited input, no state.

## 7. Complexity
- **Training:** 5 SEX CatBoost fits (about 5× one T_TREE fit, minutes of CPU). Each criterion evaluation is O(n d²) with n = 20,000 and d ≈ 40, and SLSQP needs ≲ 500 evaluations. Under 5 CPU-minutes per κ and seed; the whole grid takes under 1 CPU-hour.
- **Inference:** 7 tree ensembles (≤ 800 depth-4 trees each) plus O(35) arithmetic per person.

## 8. Expected failure modes
1. **Nonlinear readers:** the shift creates non-Gaussian class shapes that MLP/HGB readers find. The quadratic terms and the gate mitigate this but do not remove it.
2. **Weak driver:** little readable SEX information exists in X beyond the task pair (FARE Z1 pair 0.667 vs T_TREE 0.664; the calibration replay favours V/Var(π) ≈ 0.1). RCCS then degenerates toward post-processing, which costs about 2× more (exact negative law).
3. **Estimation noise in π̂:** its part orthogonal to F inflates cost (efficiency V²/(V + Var e)).
4. **The d′² reading is wrong:** it is a Gaussian-equivalent reading of selected-reader AUCs, not a covariance measurement. The amplification regularity could then be coincidental.
5. **Subgroup guard:** the shift is SEX-correlated, and the within-SEX occupation Brier could breach 0.005 (LOCAL is already unresolved for men).
6. **Occupation accuracy:** decisions flip, against a 0.010 cap.
7. **Held-out transfer** of out-of-fold moments.
8. **Floor:** the income-only floor (T_TREE 0.569) limits the pair from below.

## 9. Claims (kept separate)
- **Competitive (vs NEURAL_RAW; eligible set {NEURAL_RAW}):**
  - the coalition-benefit lower bound must be > 0 on both seeds, with the nested material margin 0.020;
  - RCCS must meet the same utility requirements (24/24 MODEL_SELECT and the 24-clause DEV point gate).
  - The model's point prediction is a benefit of ≈ 0.03. The material margin (≈ 0.032 needed at DEV size) is therefore **borderline**, and plain superiority is the realistic target.
- **Mechanism (diagnostic family, separate):** RCCS vs MIN, COND_PP, COND_SHUF and NR_OCC at matched κ (§3.8).
- **Joint component: none claimed.** RCCS is sequential with income fixed. By §9.1, joint design adds no expressiveness.
- **Not claimed:**
  - novelty;
  - any population floor, or a reduction of AUC*(S; V) below AUC*(S; X);
  - protection of attributes other than SEX;
  - individual fairness (the shift uses a SEX-propensity score of non-task inputs);
  - that Gaussian-equivalent AUC readings measure covariance;
  - a recipe effect vs TASK_REFIT beyond what a pilot shows.

## 10. Feasibility
**The model.** The Gaussian model (`D2_GAUSS_V1`) has income frozen at T_TREE (d′² 0.060), occupation d′² 0.297 and pair d′² 0.359. It was swept over R²(f1 | f2) ∈ {0.70, 0.82, 0.90}, α (the share of income's covariance mirrored by occupation) ∈ {0.4, 0.6} and d′²_X ∈ {0.40, 0.49, 0.64} (15 valid settings). The cost unit is the occupation spend at which MIN brings occupation-only AUC to LOCAL_TREE's 0.582. That ties the unit to LOCAL's *measured* spend: 0.0035 / 0.0027 summed Brier and 0.009 / 0.006 nats.

**Results.**
- RCCS reaches pair 0.606 at **0.89–1.25 units** in 14 of 15 settings, and at 1.78 units in the weakest-driver setting (V/Var(π) = 0.02).
- At 1 unit, RCCS gives 0.601–0.612 (0.630 in the weakest setting); MIN gives 0.616–0.714.
- The occupation budget from T_TREE is ≈ 2.4–3.2 units, so 1–1.8 units fit. Income spend is zero.

**Calibration check.** With income at LOCAL's 0.558 and R² = 0.82, MIN reproduces LOCAL's pair: 0.656 against the published 0.657 (d′²_X = 0.40).

**What would make it impossible:**
- V ≈ 0, meaning no readable SEX information beyond the task pair (the exact negative law);
- a mostly nonlinear coalition signal;
- π̂ too noisy.

**Status.** These are model predictions from fitted-reader aggregates, not evidence of effect. No threshold or comparator is changed.

## 11. Data-free prototype (specified, run, exact)
**Laws.** `D2_FINITE_V4` uses 8 cells x = (a, b, c), each binary with mass 1/8:
- P(Y1 = 1) = 0.3 + 0.4a (income discloses a);
- Y2 = 0.30 + 0.05a + 0.30b;
- c is SEX-only.

**Arms.** The occupation release is the calibrated posterior of a partition, with exhaustive search over all 4,140 partitions:
- TASK;
- MIN (occupation-only Bayes AUC);
- COND (pair Bayes AUC, income fixed);
- COND_PP (cannot split cells with identical base releases, i.e. no access to c);
- JOINTSUM (occupation-only + pair, the strongest RAW-J-structure analogue);
- INJ (unpooled shift).

**Metrics.** Exact Bayes AUC from the released view, and exact summed Brier, NLL and accuracy.

| Law | AUC*X / income-only / TASK pair | Result |
|---|---|---|
| Feasible (π = 0.5 − 0.15a − 0.15b + 0.15c) | 0.644 / 0.577 / 0.615 | **MIN worsens the pair** (0.644 at B 0.0006) while its occupation-only AUC falls to 0.577. **COND** reaches 0.596 at 0.0112 and the income-only floor 0.577 at 0.0225. COND_PP needs 0.0225 and 0.045 for the same two points (**driver halves the cost**). JOINTSUM: 0.596 at 0.0142, and 0.601 at B = 0.020 (worse than COND's 0.596). INJ = AUC*X. |
| Negative / required disclosure (π = 0.5 − 0.15a − 0.30b; c independent of everything) | 0.735 / 0.594 / 0.735 | Nothing helps at B ≤ 0.010. At B 0.015, COND reaches 0.688 only by using c as a deterministic *coin* for partial pooling. From B 0.0225 COND = COND_PP = 0.641, which requires destroying occupation information. |
| Coin control (feasible law, but c S-irrelevant) | 0.624 / 0.582 / 0.624 | COND −0.010 at B 0.015 and −0.021 at 0.0225 (= COND_PP). With an S-informative c (feasible law) the reductions are −0.019 at 0.0112 and −0.038 at 0.0225. The driver's SEX content, not just its splitting power, carries the gain; COND_SHUF is the pilot analogue. |
| Redundant only (π depends on a) | 0.633 / 0.633 / 0.633 | No release lowers the pair. MIN spends 0.0013 to make occupation-only AUC 0.500, for zero pair benefit. |

`D2_GAUSS_V1` verifies the following:
- the amplification identity (max relative error 4e−15);
- the counter-shift cost ‖c‖²/Var(π) against projection ‖c‖²/W (closed form = formula; Monte Carlo n = 4·10⁵ within 0.5%; gain factor 1 + V/W).

No complementary-clue joint case is given because no joint claim is made. The verifier should recompute the finite tables with separate code.

## 12. Real-data pilot sketch (not run)
**Arms** (2 seeds; same frozen T_TREE; κ grid {0.25, 0.5, 0.75, 1}):
- RCCS;
- MIN;
- COND_PP (ψ built from base-pair features only);
- COND_SHUF (π̂ fitted on household-permuted SEX);
- NR_OCC: the strongest existing-machinery analogue on the same base. It is `leafcorr.py`'s J objective (own-view + pair critics) restricted to occupation, with income frozen at T_TREE (the S12 rule with the income phase skipped). Checkpoints E2–E20 are nominated by the frozen rule;
- RCCS_U0 (base U0 for both recipients) and its matched NR_OCC_U0 (`neural_raw.py` with λ_income = 0 from U0). This pair separates teacher headroom from mechanism.

Fixed comparators: NEURAL_RAW ρ1 E2, TASK_REFIT, T_TREE, LOCAL_TREE, JOINT_TREE (diagnostic), H_LJ.

**Equal exposure.** Every arm uses MODEL_FIT only with the same folds. Nominees come from the frozen MODEL_SELECT rule. The readers are the frozen 7 × 3 slate (ATTACK_FIT/VALIDATION).

**Units and runtime.**
- 7 arms × 4 κ (or checkpoints) × 2 seeds = 56 fits (under 2 CPU-hours including NR_OCC and NR_OCC_U0);
- plus 7 nominees × 42 reader fits (about 1 CPU-hour).

**Stage 0** (MODEL_FIT only; no fit of A, no reader). Compute the precondition quantities of falsifier 5 from out-of-fold T_TREE wires, MODEL_FIT SEX and π̂. Stop if any fails.

**Stopping rule** (MODEL_SELECT only, before any reader). Stop with `NOT_SUPPORTED` if any of the following holds:
- no κ is feasible for RCCS;
- the RCCS proxy is not ≥ 0.015 below MIN at matched κ;
- the OUT_HGB-minus-OUT_LR proxy gap is > 0.03;
- RCCS is not ≥ 0.01 below NR_OCC.

**Status of the evidence.** DEV_REPORT is spent, so pilot results are exploratory development evidence. A confirmatory claim needs a freshly registered state-year, for example the next state in the admission order. The Florida sealed set stays closed.
