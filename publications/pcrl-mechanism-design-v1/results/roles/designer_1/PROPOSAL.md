# Designer 1 proposal: SEX-mixture marginalisation of the task posteriors (MMP)

## 0. Header

- **Designer:** 1
- **Decision:** `PROPOSAL`. Confidence is low to moderate. A cheap premise gate that uses only MODEL_SELECT (§12) can kill it before any attacker is fitted.
- **Finished (UTC):** 2026-10-16T05:26Z
- **Check run:** unit `D1_MMP_LAWS_V2`. Script `d1_mmp_finite_laws.py`; output `D1_MMP_FINITE_LAWS_RESULTS_V2.json`, both in this directory. It is data-free and uses the coordinator's exact finite-law toolkit.
- **Disclosed post-hoc change.** V1 (`…_RESULTS.json`) had a weaker negative law N: π ∈ {.17, .37, .57, .77}; income (male / female) .55 / .30; occupation (.40, .15, .45) / (.45, .47, .08).
  - In V1, feasible merges reached 0.651 from AUC*(S;X) = 0.751, so V1 was not a clean required-disclosure case. MMP was infeasible there as well.
  - After seeing V1, I strengthened only law N's gaps. Laws F and R are unchanged.
- **Read:** the problem specification, the evidence pack, `CLOSED_LANES.md` (before §3.7, §9 and this decision), the listed code, and published aggregates only.
  - Source study: `SELECTION_DETAIL.json`, `STAGE_A_HEADROOM.json`, `SUBGROUP_TABLE.csv`, and the slate matrices in `DEV_REPORT_RESULTS.json`.
  - Frozen hybrids: `DEV_REPORT_RESULTS.json` and `READER_ACCOUNTING.csv`.

## 1. Plain-language summary

A model that predicts income or occupation without seeing SEX must guess SEX from the six fields, because SEX predicts both tasks (occupation strongly). The law of total probability makes this explicit. The SEX-blind posterior is the two within-SEX posteriors mixed with the person's *inferred* probability of being female. So every released vector contains an inferred-SEX term: (π(x) − π̄) times the female-minus-male gap.

MMP fits task models that see SEX (on MODEL_FIT only). At release it averages them over SEX with **fixed population weights**, which deletes that term while keeping every field's within-SEX effect on the task.

Its utility price has exact forms: the Brier price is an identity and the NLL price has a bound. Both are checkable on MODEL_SELECT before any attacker. There is no critic, and the primary contract is unchanged.

Whether the deleted term carries enough of the coalition's SEX signal to beat NEURAL_RAW by the 0.020 margin is **not** shown by any published number. The pilot's first stage tests exactly that.

## 2. Release contract

**Primary contract, unchanged.**
- Each recipient gets its own probability vector and argmax decision, computed deterministically from the six raw fields.
- There is no shared state, randomness, metadata or abstention.
- The deployed model's internal SEX slot is evaluated at the constants 0 and 1, and the two outputs are mixed with fixed weights. **No person's SEX is read**, and the package refuses SEX and PUMA columns, as the existing packages do.
- All comparators (NEURAL_RAW ρ1 E2, TASK_REFIT, T_TREE, U0/Ucal) already use this contract.

## 3. The eight questions

**Notation.** S = 1 is female. Let π(x) = P(S = 1 | x) and π̄ = P(S = 1). Let q_i^s(x) = P(Y_i | x, S = s), and η_i(x) = P(Y_i | x) = (1 − π) q_i^0 + π q_i^1.

**3.1 What is removed?** The inferred-SEX component

Δ_i(x) = η_i(x) − m_i(x) = (π(x) − π̄)(q_i^1(x) − q_i^0(x)),  where  m_i = (1 − π̄) q_i^0 + π̄ q_i^1.

The decomposition η_i = m_i + Δ_i is an identity. Δ_i is the only route by which a SEX-blind posterior converts the fields' SEX-predictiveness into task predictions. It is the reader's target π(x), centred at π̄, times a gap direction common to both groups. m_i is the Bayes posterior under the counterfactual law P(x)·P(s)·P(y | x, s), in which the fields carry no SEX information.

**3.2 Why might it be unnecessary at the required utility?**
- **Summed Brier (exact identity):** Brier(m_i) − Brier(η_i) = E[(π − π̄)² ‖q_i^1 − q_i^0‖²].
- **NLL (bound):** NLL(m_i) − NLL(η_i) = E[KL(η_i ‖ m_i)] ≤ E[KL(π(X) ‖ π̄)] = I(S; X). This follows from joint convexity of KL: the two mixtures share their components.

The price is Var(π) times the squared within-SEX gap. That is small for income (gap about 0.1 at fixed fields) and moderate for occupation (§10).

The available budget is the cap plus the SEX-aware tree's headroom over U0. On MODEL_SELECT, T_TREE's headroom was:
- occupation: NLL .0121 / .0082, Brier .0051 / .0036, accuracy +.64 / +.61 pp;
- income: NLL .0066 / .0068, Brier .0041 / .0045.

In every finite law the Brier identity holds to 6e-17 and the NLL bound holds (§11).

**3.3 Why does NEURAL_RAW ρ1 E2 retain it?** The evidence is quantitative but indirect.
1. **Training targets.** U0's objective (cross-entropy without SEX) targets η_i, which contains Δ_i in full. Identifying m_i needs the components q_i^s, i.e. SEX at fit time. NEURAL_RAW does use SEX at fit time, but only through critic gradients on its outputs. Those reward fooling the current critics, not reproducing m_i.
2. **NEURAL_RAW E2 is a small move from U0.** It is 80 Adam steps at learning rate 1e-4. All 24 of its MODEL_SELECT clauses are at most 0.57 of their caps. Its DEV pair AUC is 0.640 / 0.638 against U0's 0.654 / 0.649, keeping 91% / 93% of U0's above-chance coalition AUC.
3. **Its leakage sits where Δ moves mass.** For the pair on DEV, DEC_LR gives 0.568 / 0.566, CONF_HGB 0.574 / 0.566 and LOG_MLP 0.640 / 0.638. For the occupation recipient on ATTACK_VALIDATION, DEC_LR gives 0.513 / 0.522, CONF_HGB 0.523 / 0.549 and LOG_MLP 0.624 / 0.627. So the signal lies in the *shape* of the vectors, not in decisions or confidence. Δ_2 shifts mass along the gendered direction between non-argmax classes, which is a shape change that leaves most argmaxes alone.
4. **Its change is not the signature of removing Δ.**
   - Removing Δ from a calibrated posterior raises **both** groups' within-SEX Brier. For a constant gap g, with v = Var π and κ₃ its third central moment, men's increase is ‖g‖²[v(3π̄ − 1) + κ₃]/(1 − π̄) and women's is ‖g‖²[v(2 − 3π̄) − κ₃]/π̄. Both are positive at π̄ ≈ 0.47 when −0.41v < κ₃ < 0.59v.
   - NEURAL_RAW instead moved occupation loss from women to men relative to U0 (`SUBGROUP_TABLE.csv`). Men: NLL +.0047 / +.0070, Brier +.0013 / +.0022. Women: NLL −.0050 / −.0029, Brier −.0018 / −.0007.
   - That is a directional shift. U0's own within-SEX miscalibration is a confounder, so this point is supporting, not decisive.

**Not established:** the share of NEURAL_RAW's 0.640 that Δ carries. Gate F1 (§12) measures it.

**3.4 Principle.** Exact marginalisation, with no optimisation against privacy.
- Fit q̂_i^s by a SEX-aware CatBoost teacher on MODEL_FIT (the T_TREE registry with SEX appended).
- Release m̂_i = (1 − π̄) q̂_i^0 + π̄ q̂_i^1, mixed in probability space; the decision is the argmax.
- To the extent the SEX-aware trees do not split on fields that matter only through SEX, m̂_i is constant along them. Those cells genuinely merge, so Bayes recovery falls along those directions, not just fitted-reader recovery.

**3.5 What prevents fooling a training critic?**
- There is no critic, adversary or penalty. The release is fixed by an identity.
- Protection is judged by the fresh defense-aware slate, refitted on ATTACK_FIT under MMP.
- The honest limit applies to every comparator alike. A generic smooth (or very finely partitioned) deterministic release is essentially injective on the populated support, so AUC*(S; V) ≈ AUC*(S; X) for NEURAL_RAW, the trees and MMP. Fitted-recovery differences are differences in how learnable the signal is. MMP's population-level effect is limited to merged directions.

**3.6 Why might this move the frontier rather than pick another point?**
- The old frontier perturbs a SEX-blind posterior against critics. Under a cap, the cheapest critic-fooling perturbation need not delete Δ. It can *mask* Δ with distortions that are predictable from the fields.
- The coalition can undo such masking, because the other output reveals those fields. The measured pattern is consistent with this:
  - LOCAL_TREE's occupation fell 0.650 → 0.582 while its pair stayed at 0.657;
  - H_LJ's occupation was 0.560 / 0.583 with a pair of 0.641 / 0.643.
- MMP deletes the common latent π(x) from **both** outputs at once (both use the constant π̄). No π-term remains in either output for the other to decode.
- Within-SEX task value is kept. So is any SEX signal carried by within-SEX effects of task-relevant fields (failure mode 1).
- **Finite-law support:** MMP's partition is the exact frontier optimum in law R at both budgets and in law F with headroom. In law F at the strict cap, MMP fails accuracy and partial merges do better (§11).
- **The real-data question is open:** whether Δ is a large share of the learnable pair signal (F1, F2).

**3.7 Difference from the closed lanes** (`CLOSED_LANES.md`, row by row)

| Closed lane | Difference |
|---|---|
| Critic refresh; penalty strength, allocation and schedule | No critic, coefficient or schedule |
| Conditional kernels (MMD given task labels) | No conditioning on Y, no kernel or penalty. It removes the P(S\|X)-mediated term by an identity |
| Utility guards | No guard pool. The price has a closed form, checked on held-out MODEL_SELECT rather than on the fitting pool (lesson 5) |
| Small adapters | Not an adapter on frozen outputs. It refits SEX-aware models |
| **Feature masks** | A mask deletes a field's within-SEX task value *and* its proxy value. On Adult 0 of 5 masks were feasible, and the retrained full-input head already had NLL excess .045–.060. MMP deletes only the proxy route. The two coincide only for pure proxies (laws F, R) |
| Latent bottlenecks and codes | Full-precision outputs, no capacity knob. Merging happens only along directions the SEX-aware model ignores |
| Frozen-bank channels; decoder/calibration rescue | No frozen bank and no post-hoc decoder |
| Protected trees | Same model family. No leaf-score correction or critic; SEX enters at fit time and is marginalised at release |
| Frozen hybrids; stochastic channels | No composition; deterministic |
| ACS-2018 "invariant baselines" | **Not verified by me** beyond the helper's one line ("invariant objective"). MMP uses no objective; the prior-work reviewer should check |

Lesson 3 (joint never beat local or sequential) does not apply: MMP is local and makes no joint claim.

**3.8 Falsifiers**

| Id | Observation | What it falsifies |
|---|---|---|
| F1 (premise) | On MODEL_SELECT, the frozen OUT_LR proxy of the twin (α = 1) minus MMP's is < 0.02 on either seed | Δ is not a major learnable carrier |
| F2 (carry-over) | MMP's pair AUC is ≥ NEURAL_RAW's, or MMP's pair-minus-best-single exceeds NEURAL_RAW's (.018 / .015) by > 0.02 | Removal carries into the coalition |
| F3 (utility) | Any of the 24 clauses fails on MODEL_SELECT | Removal is affordable |
| F4 (prototype) | Identity error > 1e-12; MMP feasible in law N; MMP off the exact frontier in law R, or in law F with headroom | The explanation itself |

## 4. Formal objective

**Target** (the specification's problem): over deterministic maps (f_1, f_2), minimise the coalition reader AUC on V, subject to the 24 clauses against U0 and Ucal. The subgroup guard is reported separately.

**MMP** is a closed-form candidate:

f_i^MMP = argmin over g of E_P̃[ℓ(g(X), Y_i)],  with P̃ = P(x) P(s) P(y | x, s).

Here ℓ is NLL or Brier, both proper, so the minimiser is m_i. Its excess over η_i has the forms in §3.2.

**Fitting:** q̂_i^s minimises cross-entropy in the T_TREE class with SEX as an extra input, on MODEL_FIT; π̄ is the MODEL_FIT female share.

**Twin (α = 1):** m̂_i^(1) = (1 − π̂(x)) q̂_i^0 + π̂(x) q̂_i^1, where π̂ is a CatBoost SEX classifier on X fitted on MODEL_FIT. It uses the same components and differs only in the weights, so it isolates the marginalisation step. T_TREE is a secondary SEX-blind reference.

## 5. Training procedure (access by role)

| Role | Use |
|---|---|
| MODEL_FIT | Features, Y_1, Y_2 and SEX. Per seed: two SEX-aware CatBoost teachers (T_TREE registry: depth 4, learning rate 0.05, ≤ 800 trees, l2 3, 15% household internal stop, refit, numpy export verified bitwise), plus π̄ and the twin's π̂. SEX is used only here |
| CALIBRATION | Not used: Ucal only |
| MODEL_SELECT | Frozen rule: 24-clause point feasibility, then the mean OUT_LR coalition proxy; plus the registered premise gate F1, computed with the same proxy |
| ATTACK_FIT / ATTACK_VALIDATION | Readers only |
| DEV_REPORT | Exploratory, already opened |
| SEALED_ASSESSMENT | Closed |

There is no grid; the nominee is MMP by registration.

## 6. Inference

1. Encode the six raw fields with MODEL_FIT moments and levels (width 145).
2. For each task, evaluate the SEX-aware teacher at x ⊕ 0 and x ⊕ 1.
3. Compute m̂_i = (1 − π̄) p̂_i(x, 0) + π̄ p̂_i(x, 1).
4. Release R_i = (m̂_i, argmax m̂_i), taking the first class on ties.

No SEX is read, and no other recipient's output is used.

## 7. Complexity

- **Training:** about T_TREE's cost per task and seed, plus one classifier for the twin. There is no adversarial loop.
- **Inference:** two oblivious-tree passes per task, O(2·T·depth), with T ≤ 800 and depth 4.

## 8. Expected failure modes

1. **Residual task-relevant proxies (most likely).** Hours, class of worker and schooling predict SEX *and* the tasks within SEX. MMP keeps their within-SEX effects, so the coalition can still triangulate them. Law R shows the residual exactly (0.607 vs 0.678). A real residual above about 0.61 rules out the material target.
2. **Decision flips.** Removing Δ_2 moves argmaxes away from gendered classes. In law F this cost 1.33 pp of accuracy: infeasible at the cap, feasible with headroom. Partial merges reached 0.585 at the cap, which all-or-nothing MMP misses.
3. **Occupation NLL or Brier price.** It may exceed the cap plus headroom if Var(π) or the gaps are larger than assumed.
4. **Subgroup guard.** Both groups' occupation Brier rises (§3.3). The men's seed-1 headroom is thin: T_TREE is −.0002 against U0.
5. **Extrapolation.** The rarer-SEX component q̂^s(x) carries weight π̄ or 1 − π̄ everywhere, so loss may concentrate in sex-atypical niches.
6. **Prior art.** I believe the device is known in anti-discrimination profiling (for example Pope and Sydnor, 2011, averaging an attribute-aware model over the attribute's distribution). At best this is an engineering combination; the reviewer judges.

## 9. Claims, kept separate

- **Competitive method vs NEURAL_RAW ρ1 E2:** the coalition benefit, with lower bound > 0 on both seeds and the nested 0.020 material margin.
  - Material superiority needs a point benefit of about 0.032 (pair ≈ .608 / .606). Starting from a twin near T_TREE's 0.664, that is a drop of about 0.056, so Δ must carry about a third of the twin's above-chance pair AUC.
  - My subjective forecasts:
    - F1 passes: .45;
    - 24-clause feasibility: .55;
    - point pair below NEURAL_RAW on both seeds: .30;
    - material superiority established: .07.
- **Recipe effect vs the twin** (> 0.020 on both seeds, recipient worsening ≤ 0.010): reported separately. Forecast .30.
- **Joint component:** none claimed. MMP is local, with a fixed shared constant and no shared state or randomness.
- **Not claimed:** novelty; a population privacy floor; any AUC*(S; V) reduction beyond merged directions; subgroup suitability; any attribute other than SEX; fitted AUC as a floor.
- **Consistency with specification §9:**
  - MMP changes decisions, so the decision-only floor of §9.2 does not bind it; its accuracy cost is failure mode 2.
  - No true-label quantity is used as a floor (§9.3).
  - Nothing relies on joint-versus-sequential expressiveness (§9.1).

## 10. Feasibility

**Utility: plausible, not shown.**
- Brier price: E[(π − π̄)²‖g_i‖²], exactly. NLL price: about ½E[(π − π̄)² χ²(q_i^1, q_i^0)], and at most I(S; X).
- Illustrative assumptions (not data): Var π ≈ .02–.03; income gap about .1; occupation gaps .1–.4 per class.
  - Income: ≤ .001 nats and ≤ .001 Brier.
  - Occupation: about .004–.010 nats and .002–.004 Brier.
- Budget (cap plus headroom): about .017 nats and .009 Brier per task.
- F3 decides it.

**Recovery: possible, not contradicted, unproven.**
- NEURAL_RAW's decision channel (.567) lies below the target.
- JOINT_TREE reached .603 / .606 in this population. Its income Brier spend (+.0094 from the teacher) is about ten times MMP's predicted income price.
- Nothing published gives Δ's share of the signal.
- **It becomes impossible if** the within-SEX effects of task-relevant fields alone give a pair above about .61 (the law-R mechanism, which is plausible here). Any release that keeps those effects would then fail.
- No threshold or comparator is changed.

## 11. Data-free prototype (executed check and plan)

**Setup (exact).**
- P(S = 1) ≈ .47; Y_1 binary and Y_2 with three classes, conditionally independent given (X, S).
- Metrics: exact Bayes AUC from the released vectors plus decisions (ties counted ½), and exact accuracy, NLL and summed Brier.
- Reference: the exact SEX-blind posterior η, the analogue of a perfect U0.
- Budgets: cap (.010 / .010 / .005) and cap + headroom (.016 / .020 / .009).
- Comparators:
  - the **global frontier** over all pairs of set partitions of the cells, calibrated per block. It weakly dominates every deterministic release, so it is the strongest analogue of NEURAL_RAW or any method. A smooth injective analogue would show no Bayes reduction at all.
  - the best local design, and both sequential orders S12 and S21.

| Law | Construction | AUC*(S;X) | MMP pair | MMP excess vs η (income; occupation) | Frontier min (cap / +headroom) | Outcome |
|---|---|---|---|---|---|---|
| **N**, required disclosure | Tasks depend on X only via S; π ∈ {.10, .37, .63, .90} | .8325 | .5 (constant) | accuracy .068; .064. NLL .022; .069 | .799 / .765 | MMP infeasible, as it should be. Nothing feasible goes below .765. Local is .799 at both budgets; with headroom S21 = global .765 |
| **F**, pure proxy | Z (3 levels) moves π only; W (2 levels) is a task feature, W ⊥ S | .6517 | **.500** (exact ties) | income NLL .0006, Brier .0005; occupation NLL .0052, Brier .0034, **accuracy .0133** | .585 / **.500** | With headroom, MMP = global = local = S12 = S21. At the cap, MMP fails accuracy (failure mode 2) |
| **R**, task-relevant proxy | Z pure proxy; H moves π *and* the tasks | .6784 | **.607** = AUC*(S;H) | .0009 / .0007; .0071 / .0045 (NLL / Brier). Block-calibrated: .0038 / .0023 | .607 / .607 | Feasible at the cap. MMP's partition **is** the frontier; the residual is required disclosure through H |

The Brier identity holds to ≤ 6e-17 and the NLL bound holds (I(S;X) = .201, .039, .051 nats).

**Specified, not run (secondary metrics):**
1. Continuous-field laws with population-optimal *restricted* readers (logistic on log-probabilities, by numerical integration). Expected: MMP removes the π-aligned share and leaves the H-type residual.
2. A two-task masking law. A local field-predictable distortion beats single-recipient restricted readers but not the pair reader, with MMP, local and S12/S21 controls at equal budget.

## 12. Real-data pilot sketch (not to be run here)

- **Arms** (both seeds; MODEL_FIT access only, the same for all):
  - MMP (α = 0) and its twin (α = 1);
  - frozen NEURAL_RAW ρ1 E2, TASK_REFIT and T_TREE;
  - U0 / Ucal as references.
- **Stage 0, premise gate (MODEL_SELECT only, no attack data).**
  - Compute the 24 clauses and the frozen OUT_LR proxy for MMP and the twin.
  - Stop on F3 (any clause fails) or F1 (proxy difference < .02 on either seed).
  - Cost: 6 CatBoost fits and 4 proxy fits, about 0.3 CPU-hours.
- **Stage 1** (only if Stage 0 passes).
  - The 7-reader slate × 3 views × {MMP, twin} × 2 seeds = 84 fresh fits, about 1 CPU-hour.
  - Score with the source report code, household bootstrap B = 2048.
  - Endpoints: the competitive claim, the recipe effect vs the twin, subgroup guards, and F2.
- **Stopping:** any of F1–F3 stops the pilot and is reported. There is no rescue, retune or new nominee.
- **Data use:** DEV_REPORT use is exploratory. A confirmatory claim needs a fresh population, and SEALED_ASSESSMENT stays closed.
