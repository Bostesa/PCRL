# Designer A proposal: SEX-standardized task posteriors (STD)

Campaign `pcrl_dual_design_release_v1`, Designer A. Written 2026-10-21, 17:12-17:20Z (label-free work from 16:50Z).

- **Read:** the design brief; `pcrl_new_release_mechanisms_v1` FAILURE_MAP and RESEARCH_DECISION; `pcrl_finite_routing_comparison_v1`
  RESEARCH_DECISION, TABLES and `dev/UTILITY_LEDGER.csv`; `pcrl_finite_solver_adequacy_v1` RESEARCH_DECISION; the MMP records of
  `pcrl_mechanism_design_v1`; `nrm/fa/prc.py`; the `nrm/devreaders.py` header; the two caches named in the brief.
- **Not opened:** any label file; anything of MODEL_SELECT; any reader cache; `roles/designer_b/`.
- **CPU:** 66.9 s measured, one thread. Itemized in the last section.
- **Files** (this directory):
  - `mech_sketch.py`: fit, release, selection rule, controls, and revisions R1 (`estimator='ipw'`) and R2 (`decisions='twin'`);
    `selftest()` passes on invented data.
  - `feasibility.py`, `feasibility2.py` and `feasibility3.py` write `FEASIBILITY.json`, `FEASIBILITY2.json` and `FEASIBILITY3.json`.
    All three are label-free plug-in estimates.
  - `inspect_caches.py`: lists the structure of the two caches.

## 1. The operation

Notation:
- Z = `ohn.fam_a.compact_po(X)`; s = 1 for women.
- π(x) = P(SEX = 1 | x); η = P(y | x) is the SEX-blind posterior.
- q_s(x) = P(y | x, SEX = s), and Δ(x) = q_1(x) − q_0(x) is the within-x gap between women and men.

| Part | Definition | Fitted on (role, labels) | Settings |
|---|---|---|---|
| Twin tI, tO | HGB on Z | MODEL_FIT income and occupation labels | A1_PO_16 learner: lr 0.025, 16 leaves, minimum leaf 100, l2 3, at most 1000 rounds. T* = first argmin of the pooled inner 5-fold household-CV NLL. Same construction as `nrm.fa.prc.fit_teachers` (settings, fold tags, seeds), so the cached TASK teachers can be reused once the coordinator confirms them by hash |
| SEX-aware qI_s, qO_s | The same HGB on [Z, SEX], evaluated at s = 0 and s = 1 | MODEL_FIT task labels; SEX is a training input column | Same settings, folds, seeds and T* rule. qO_s is the cached SEXSIDE `occ_s` teacher; qI_s is the only new teacher |
| Weight w (one scalar, shared by both recipients) | The value on {0.30, 0.31, …, 0.60} that minimizes max(men's, women's) out-of-fold occupation summed-Brier excess of the λ = 1 release over the twin. Ties go to the value nearest the SEX share | MODEL_FIT occupation and SEX labels, 5 household folds | Plug-in estimate about 0.45; the SEX share is 0.49 |

**Release.**
- P_inc = (1 − λ)·tI + λ·[(1 − w)·qI_0 + w·qI_1]
- P_occ = (1 − λ)·tO + λ·[(1 − w)·qO_0 + w·qO_1]
- Decisions are the argmax. The output is float64 and deterministic.

**Why this removes SEX information.**
- Identity: η = Σ_s π_s·q_s = m_w + (π(x) − w)·Δ(x), where m_w = Σ_s w_s·q_s.
- The second term is where the person's SEX propensity enters a SEX-blind posterior. The twin targets η; at λ = 1, STD targets m_w.
- No propensity model is fitted or deployed.
- λ = 0 is the twin.

**Strength grid and selection** (MODEL_FIT only):
- Grid: Λ = {0.5, 0.75, 1.0}.
- Out-of-fold releases come from the 5 household folds of `nrm.fa.prc`, with T fixed at the full-pool T*.
- Six spend clauses over the twin: occupation NLL; income NLL; occupation accuracy loss; income accuracy loss; men's occupation summed
  Brier; women's occupation summed Brier.
- λ* = the largest λ whose every clause meets estimate + 2 household-clustered SE ≤ the existing budget S3 of `nrm.fa.prc`
  (0.012, 0.006, 0.008, 0.006, 0.0045, 0.0045).
- If no λ qualifies, nothing is released: R1 applies if its trigger holds; otherwise stop.
- CALIBRATION is not used.

**Gate G1** (MODEL_FIT only, before any MODEL_SELECT read):
- Readers: household-cross-fitted LOG_LR and HGB15 on the out-of-fold releases, with MODEL_FIT SEX labels.
- Pass: STD_λ*'s pair AUC is at least 0.005 below both the twin's and the spend-matched UND point's, in both versions.
- Otherwise stop. The result is reported as negative and no MODEL_SELECT grid is run.

## 2. Closest previous implementation and what differs

**Closest: SEX-mixture marginalisation (MMP).** `results/pcrl_mechanism_design_v1/roles/designer_1/PROPOSAL.md`, campaign
`pcrl_mechanism_design_v1`, 2026-10-16.
- **Device:** CatBoost SEX-aware teachers averaged with weights π̄.
- **Evidence:** data-free finite laws and a prototype only. The real-data pilot never ran; it was listed as awaiting the advisor's
  target decision.
- **Prototype:** contradicted MMP's learnability leg in a one-to-one law (readers 0.0004 apart).
- **Verifier's 10-field probe:** MMP minus twin was −0.033, +0.018 and +0.019 across three draws, so the sign was not definite.
- **In code:** `nrm/fa/prc.py` already fits qOs, but only as PRC's guard statistic; it was never released.

**What is the same:** the λ = 1 device is MMP's.

**What is substantively different:**
1. **Real-teacher evidence for MMP's open premise.** The premise is that the Δ-term carries much of the learnable SEX signal. It was
   tested on the cached ACS FL teachers (section 4).
2. **A strength path with a MODEL_FIT rule.** MMP was all-or-nothing, and in its law F it failed accuracy at full strength.
3. **A guard-balanced weight instead of π̄.** This targets MMP's failure mode 4, the subgroup guard. Plug-in men's Brier spend fell from
   0.0052 to 0.0014 at the same plug-in recovery.
4. **SEX-blind spend-matched controls.** UND and CRS, plus the COMP and STD_OCC ablations. MMP was compared only with its
   propensity-weighted twin and frozen references.
5. **A predeclared IPW estimator (R1)** for the attribution and extrapolation failure that the sketch exhibits (section 4).
6. **A MODEL_FIT gate** instead of a MODEL_SELECT premise gate.

**Closed families:** none is reused. There are no cells or routing, no noise channel, no decoder or calibrator, no controller and no
leaf-score correction.

## 3. Which measured failure it addresses

**(1) Spend explained every drop.**
- Earlier results: FB2 vs MINF +0.0027 / +0.0019 (se 0.0089). PRC_S2 vs HQ_S2 +0.0003, vs UNT_BR +0.0031, vs ORD_S2 +0.0053.
- STD's price is second order: Brier price = E[(π − w)²‖Δ‖²], an identity.
- The price falls only on the π-driven term. That term is about 1.4% of the twin occupation output's variation in the Fisher metric
  (plug-in), so a SEX-blind operation of equal spend mostly removes task variation instead.
- FB2 added noise along one person-specific SEX direction. In-sample linear discriminability fell by 36-40%, yet readers did no better
  than under SEX-blind noise, consistent with readers using SEX information outside that direction.
- STD is deterministic and deletes the propensity term itself rather than masking it. Its linear dependence on logit π̂ is about 0.42
  of the twin's (section 4), and the remaining dependence is the within-SEX effects it keeps by design.

**(2) Pair super-additivity** (H_LJ: 0.5504 and 0.5727 alone, 0.6252 linked).
- One w removes the shared latent π(x) from both recipients at once.
- No π-term is left for a linked reader to triangulate.

**(3) Leakage moves between recipients** (H_JJ; task-conditioned income +0.063).
- Both recipients are treated alike, and both views are predicted to fall.
- STD_OCC measures any transfer.

**(5) Protecting income cost income accuracy** (H_JJ: 0.0140).
- STD changes income only by (π − w) times the income gap. Assuming a within-x gap of about 0.1 (not measured; no P(income | x, SEX)
  is cached), the price is about 0.001 nats.

**(4) LOOKUP readers favour finite releases.** STD is continuous.

**What it does not address:** the within-SEX effects of SEX-predictive task fields (hours, class of worker, schooling). They are kept
by design. Most of the above-chance AUC remains after STD (section 4).

## 4. Usefulness and label-free feasibility

**Plug-in design** (`FEASIBILITY*.json`):
- **Inputs:** cached MODEL_FIT out-of-fold predictions only. Pseudo-SEX is drawn as Bernoulli(π̂(x)) from the cached propensity.
- **Readers:** fitted on 4,000 people (about the grid's reader-FIT size). Scored on the other 16,000 by expected (soft) AUC.
- **Scope:** occupation only; income stays at the twin, because P(income | x, SEX) is not cached.
- **Spend:**
  - T1 = E KL(twin ‖ release), taking the twin as the truth.
  - T2 = the same against the SEX-aware composite Σ_s π̂_s·q_s.
  - The real spend should lie between them.
- **Guards:** within-SEX Brier relative to the twin, with q_s as the within-SEX truth.
- Cells give version 0 ; version 1 where both were run.

| Occupation release | Spend T1 / T2 | Accuracy loss (T1) | Men / women Brier | Occupation view LR / HGB | Pair LR / HGB |
|---|---|---|---|---|---|
| TWIN | 0 | 0 | 0 / 0 | .623/.613 ; .625/.617 | .640/.625 ; .641/.628 |
| STD λ 0.5, w .49 | .0022 / −.0008 | .0012 | .0016 / −.0024 | .607/.603 ; .608/.608 | .632/.618 ; .632/.622 |
| STD λ 0.75, w .49 | .0049 / .0004 | .0028 | .0031 / −.0030 | .597/.598 | .627/.614 |
| STD λ 1, w .49 | .0087 / .0027 ; .0084 / .0029 | .0047 | .0052 / −.0034 | .587/.593 ; .587/.598 | .624/.612 ; .624/.616 |
| STD λ 1, w .45 | .0083 / .0025 | .0039 | .0014 / .0002 | .587/.594 | .624/.612 |
| COARSE K128 (SEX-blind) | .0084 / .0070 | .0046 | .0025 / .0037 | .618/.613 ; .620/.617 | .639/.621 ; .639/.625 |
| NOISE σ .17 (SEX-blind) | .0087 / .0087 | .0053 | .0039 / .0040 | .620/.605 ; .620/.609 | .638/.621 ; .639/.623 |
| FLAT (SEX-blind), v0 | .0087 / .0073 | .0015 | .0020 / .0026 | .620/.613 | .640/.625 |
| TEMP (SEX-blind), v0 | .0087 / .0074 | .0000 | −.0019 / .0082 | .623/.613 | .640/.625 |
| COMP (Σ π̂_s q_s) | .0036 / −.0036 | .0015 | −.0005 / −.0022 | .631/.621 ; .632/.626 | .646/.631 ; .648/.636 |

**Reading:**
- Bayes AUC of π̂ itself: .681 / .683.
- At equal T1 spend, STD lowers plug-in pair AUC by .012-.017 and the occupation view by .018-.038.
- The information-removing SEX-blind controls lower the pair by .001-.005 and the occupation view by at most .008.
- Near-invertible maps (FLAT, TEMP) lower the pair by at most .001.
- COMP raises the pair by .006-.008.
- A data-rich reader (HGB, 16,000 for fitting) still sees the effect: occupation .622 to .601, pair .634 to .623.
- I(occupation; SEX | x) is about .062 nats; the cached inner-CV occupation NLLs are 1.2114 (SEX-blind) and 1.1476 (SEX-aware).
- Linear R² of logit π̂ on the occupation log-probabilities (`FEASIBILITY3.json`): twin .463 / .466, COMP .526 / .534, STD λ 1
  .195 / .199. STD keeps about 0.42 of the twin's linear π-dependence.
- Person-level clause SEs under T1, at λ 1 with w .45: occupation accuracy loss .0039 (SE .0016) and occupation NLL .0083 (SE .0009).
  At λ .75: .0023 (.0014) and .0048 (.0007).

**Utility room** (MODEL_SELECT, A1_PO_16 occupation; W row of `frc/dev/UTILITY_LEDGER.csv`, versions 0 / 1, against U0):
- Occupation: NLL −.0161 / −.0128, accuracy −.0099 / −.0097, Brier −.0067 / −.0055.
- Guards: men −.0039 / −.0023, women −.0097 / −.0087, against a cap of +.005.
- Income (TWB_C64, from the same HGB teacher): NLL −.0076 / −.0077, accuracy −.0034 / −.0008.
- At λ = 1 every plug-in spend fits within this room. S3 is the binding filter: occupation accuracy .0039 + 2 × .0016 = .0071,
  against a budget of .008. Household clustering, which no cache allows label-free, would raise the SE somewhat. With the SEX-share
  weight (.49) the same clause is .0079, at the edge.

**Caveats:**
- Plug-in AUCs run .01-.025 below the real readers (W real: pair .6506, occupation .6480).
- Pseudo-SEX carries only π̂'s information.
- Two reader recipes were used, not sixteen.
- Income is not tested.
- **Estimator caveat** (from `mech_sketch.py` on invented data): boosted SEX-aware trees attribute part of the SEX effect to
  SEX-correlated fields.
  - Under a pure-proxy shift, the standardized output kept 0.34-0.43 of the twin's movement at realistic overlap (3 seeds), and 0.89
    under extreme separation.
  - The real-teacher numbers above already include this effect, net.

## 5. Twin and controls (each configuration in both versions)

| Configuration | Definition | Role |
|---|---|---|
| DA_TWIN | λ = 0: the TASK teachers, full-pool fits on MODEL_FIT | Equally trained task-only twin: same learner, people, folds, seeds and T* rule; never reads SEX |
| DA_UND ×3 | The twin's own boosting path stopped at a per-task fraction of T*. Fractions are chosen on MODEL_FIT out-of-fold predictions so the per-task NLL spend is about 0.5×, 1× and 2× that of STD_λ* | **Simplest matched control.** Same model, people and spend, no SEX. It tests whether a weaker task model simply leaks less |
| DA_CRS ×2 | KL k-means coarsening of the twin with out-of-fold teacher-mean (B1) tables. Capacities are chosen on MODEL_FIT to bracket STD_λ*'s spend | Information-removing (merging) SEX-blind control; the ORD analogue |
| DA_COMP | Σ_s π̂_s(x)·q_s(x), using the cached SEXSIDE SEX model | Structural ablation: same components with person-specific weights. STD vs COMP isolates fixing the weights |
| DA_STD_OCC | Occupation STD at λ*; income is the twin | Per-recipient attribution and leakage transfer |

**Readings:**
- **Primary:** STD_λ* against the UND frontier. UND brackets STD's MODEL_SELECT spend and is interpolated there (`nrm.rules`
  frontier logic).
- **Secondary:** against CRS.
- **Descriptive:** TWIN, COMP, STD_OCC and the nine references.

## 6. Bounded revision plan

Each revision is triggered on MODEL_FIT, before any MODEL_SELECT read of an STD release.

**R1, IPW-STD (estimator).**
- **Trigger (either):**
  - (a) At λ = 1, a task's out-of-fold NLL spend exceeds twice its out-of-fold plug-in T1 value. That means the counterfactual
    components are poorly estimated.
  - (b) G1 fails, and for either recipient the out-of-fold linear R² of logit π̂ on the λ = 1 release's log-probabilities is more
    than half the twin's. That means the SEX effect was attributed to proxy fields.
  - For occupation, (b) is label-free and already computed on the cached teachers: ratio 0.42 / 0.43, so not met. Income is unknown.
- **Revision:** one SEX-blind HGB per task on Z, trained with weights w_s / π̂_s(x).
  - π̂ is cross-fitted on household folds and clipped to [0.05, 0.95]; the weights are normalized to mean 1.
  - Same settings; T* uses the weighted CV NLL.
  - Release (1 − λ)·twin + λ·IPW model, with the same grid, selection rule and gate.
- **Why it can help:** the Bayes target is the same m_w, but there is no SEX input and no counterfactual evaluation. π̂ is a training
  weight only and is not deployed.
- **Sketch check:** effective sample-size share 0.74-0.78 at realistic overlap. The IPW release lay nearer the plug-in target than the
  twin did (KL .008-.009 vs .012-.019).

**R2, decision-preserving STD.**
- **Trigger:** λ* < 1, and the only failing clause at the next grid value is an accuracy clause.
- **Revision:** in the released vector, the twin's decision class is pooled with every class that exceeds it. The common value is
  their mean, and the twin's class gets +1e-9, so decisions equal the twin's. The selection is then re-run on the same grid.
- **Cost:** the decisions carry the twin's propensity-driven argmax.
- **In the sketch:** `keep_decisions`. The selftest checks that both recipients' decisions equal the twin's and that rows whose argmax
  did not move are unchanged.

**Otherwise:** if λ* is None or G1 fails without a trigger, stop and report a negative result.

## 7. Registered predictions, configurations, CPU

Predictions are for the registered grid of real readers (8 cells; mean AUC). Utility is on MODEL_SELECT against U0. Intervals are
subjective 80% ranges.

| Quantity | Prediction |
|---|---|
| Selection | λ* = 1.0 in at least one version (p .6), else 0.75; w ≈ .45 [.40, .48] |
| G1 | Passes (p .75) |
| TWIN: pair / income view / occupation view | .658 [.648, .668] / .567 [.555, .580] / .647 [.638, .656] |
| STD_λ*: pair / income view / occupation view | .638 [.625, .651] / .555 [.542, .568] / .622 [.605, .637] |
| Pair benefit vs TWIN | +.020 [+.008, +.032]; both recipient views fall (recipient gate passes) |
| UND at STD's spend | Pair .652 [.642, .662]. STD's benefit +.014 [+.002, +.026], resolved (> 2 se) with p .35 |
| CRS bracket | STD's benefit +.012 [.000, +.025] |
| COMP; STD_OCC | COMP pair ≥ TWIN − .003. STD_OCC pair .644 [.630, .656], about 2/3 of STD's benefit |
| Against NEURAL_RAW .6309 / H_LJ .6252 | Worse by .007 / .013 [−.020, +.006]. **The 0.020 material margin is not met (p .97)** |
| Utility at λ* | All 24 clauses and 4 guards pass (p .8). Occupation NLL vs U0 ≈ −.010; men's guard ≈ −.001 [−.004, +.003]; income accuracy loss ≈ .000 [−.004, +.005] |

**Configurations: 11**, each run in both versions.
- DA_TWIN; DA_STD_050, DA_STD_075 and DA_STD_100; DA_UND ×3; DA_CRS ×2; DA_COMP; DA_STD_OCC.
- If the budget is tight, the minimum set is 8: drop CRS ×2 and COMP.
- R1 and R2 replace the STD path; they add no configurations.

**CPU estimate:**
- **New fits, per version (about 350 s):**
  - SEX-aware income teacher: inner CV, full fit and 5 out-of-fold fits, about 11 HGB fits, ≈ 170 s.
  - Twin fold refits for the UND staged path: 10 fits, ≈ 100 s.
  - CRS k-means and B1 tables: ≈ 20 s.
  - G1 readers: ≈ 60 s.
- **New fits, both versions:** about 0.2 CPU-h. Add about 0.3 CPU-h if the cached teachers cannot be reused.
- **Reader grid:** 11 configurations × 8 cells × 15.7 s ≈ 1,400 s.
- **Total:** about 0.6 CPU-h, or about 0.9 CPU-h with teacher refits. R1 adds about 0.2 CPU-h.

## 8. Novelty status

**The device is established.** Closest published methods:
- Pope and Sydnor (AEJ: Economic Policy, 2011): fit with the protected attribute, then predict with it averaged at population values.
  Checked only at abstract level, per this repository's prior-work review.
- Žliobaitė and Custers (AI & Law, 2016), and Žliobaitė (2017).
- Pearl's adjustment formula (Biometrika, 1995).
- Ghili, Kazemi and Karbasi, "train then mask" (AAAI 2019). Recalled from memory, not re-verified here.
- Lipton, Chouldechova and McAuley (NeurIPS 2018) call training-only use of the attribute a disparate learning process.

**R1** is standard inverse-probability weighting of a standardized quantity. Kamiran and Calders' (2012) reweighing uses (S, Y) cell
weights, which is a different target.

**Classification:**
- Using the device as a defense against SEX inference from released probability vectors is an engineering combination, as
  `pcrl_mechanism_design_v1` ruled for MMP.
- The strength path and the guard-balanced weight are engineering choices.
- **No novelty is claimed.** STD must not be called fair: it is not counterfactually fair (Kusner et al., 2017).

**CPU used** (process time, one thread; 66.9 s in all, plus under 1 s of untimed JSON and CSV printing):

| Item | CPU-s |
|---|---|
| Cache inspection | 0.8 |
| Soft-AUC check | 0.9 |
| `feasibility.py --rich` | 22.6 |
| `feasibility2.py` | 19.6 |
| `feasibility3.py` | 0.8 |
| Sketch selftests (four runs; the first failed and led to the fixture change) | 3.2 + 3.6 + 3.6 + 3.7 |
| Two invented-law diagnostics | 1.1 + 6.9 |
