# Designer B proposal: PTC, propensity-tilt cancellation (campaign `pcrl_dual_design_release_v1`)

Written 2026-10-21, about 17:45Z. Design space B: fixed, equally trained task-only models; the mechanism only changes how the
released vectors are formed. Label-free: nothing under `admitted/exports/*/{income,occupation,sex}.npz`, nothing of MODEL_SELECT, no
reader cache and no raw features were opened. Inputs read: the brief, the records it names, `results/pcrl_new_release_mechanisms_v1/`
(designer B closing report, FB2 code), and `results/pcrl_mechanism_design_v1/` (RCCS, MMP). CPU used: about 220 s (one thread).
This version applies the coordinator's protocol rules of 17:35Z (PROTOCOL.md section 4).

## 1. The operation

Inputs at runtime are the six fields x only. Every fitted part comes from MODEL_FIT.

| Fitted part | Role, labels | Size |
|---|---|---|
| p_I(x), p_O(x): the cached HGB task teachers (`TASK.pkl`, A1_PO_16 settings), unchanged | the twin; task labels only | fixed |
| q(x) = P(SEX = 1 \| x): the cached SEX-side HGB (`SEXSIDE.pkl` 'sex') | SEX label (training only; never a runtime input or output) | fixed |
| w_I (2), W_O[k] (5 x 5), qbar, qbar_k: closed form from the out-of-fold statistics and the SEX label | SEX label | 27 numbers |

With c(p) = log p minus its mean over classes, and k(x) = argmax p_O(x) (the twin's own occupation decision):

    log r_I(x) = log p_I(x) - alpha * (q(x) - qbar)   * w_I       + const      (income: one global class-weight vector)
    log r_O(x) = log p_O(x) - alpha * (q(x) - qbar_k) * W_O[k]    + const      (occupation: one vector per decision stratum)
    w_I    = Cov(c(p_I), s) / Cov(q, s);     W_O[k] = Cov(c(p_O), s | k) / Cov(q, s | k);     qbar_k = mean(q | k)

Each recipient receives only its own tilted vector and its argmax. The operation is deterministic, with no randomness and no runtime
SEX. At alpha = 1 the released centred log-probabilities have zero covariance with SEX on MODEL_FIT out-of-fold: overall for income,
and within each occupation-decision stratum for occupation. Equivalently, men and women have equal mean released log-ratio vectors.

This works without SEX at runtime because of the tower identity: Cov(f(x), s) = Cov(f(x), P(s | x)) for any release f(x). A runtime
propensity can therefore cancel a covariance with the unseen SEX column. The weights use the SEX label (Cov(q, s), not Var(q)), so the
cancellation targets the true covariance even when q is miscalibrated. A stratum with fewer than 200 people, or Cov(q, s | k) <= 0,
falls back to the global occupation vector (none did in the cache: strata 11.5k / 2.8k / 4.4k / 0.62k / 0.70k).

**Strength path and selection (MODEL_FIT only; no privacy gate).**
- Strength grid: alpha in {0.2, 0.35, 0.5, 0.65, 0.8, 1.0}.
- Selection uses 5 outer household folds. In each fold, the weights are refit on the other four folds' cached out-of-fold statistics
  with their SEX labels and applied to the held-out fold.
- Per-person spend over the twin is computed with labels: income and occupation NLL, summed Brier and accuracy, plus the men's and
  women's occupation summed Brier (SEX labels).
- alpha_S2 / alpha_S3 is the largest alpha whose estimate + 2 household-clustered SE meets every component of the budget:
  - S2: occupation NLL 0.008, income NLL 0.004, each task's Brier 0.003, men's and women's occupation Brier 0.003, occupation accuracy
    0.006, income accuracy 0.004;
  - S3: the same components at 0.012 / 0.006 / 0.0045 / 0.0045 / 0.008 / 0.006.
- These are `nrm.fa.prc.BUDGETS` plus the two Brier components.
- If alpha_S2 = alpha_S3, there is one candidate.
- Disclosure: the cached teachers' cross-fitting folds are not the outer folds, so the estimate is nested for the weights, not for
  the teachers (the B1 / FB2 practice).

## 2. Closest previous implementation and what is different

**RCCS** (`results/pcrl_mechanism_design_v1/roles/designer_2/PROPOSAL.md`; data-free, rejected as presented, never run) is the
closest. It applied a deterministic counter-shift to the occupation logits, A·psi(x), driven by five cross-fitted SEX-propensity
models. A (30 parameters) was chosen by optimising a ridge criterion with quadratic features, conditional on the frozen income
output.

PTC differs in four ways:
- It shifts both recipients, with one scalar driver.
- It is closed form, with no criterion or critic class.
- Its target is exact first-moment cancellation: unconditional for income, within the released occupation decision for occupation.
- It comes with label-free evidence on the real cached statistics and with the decisive controls.

The rejection grounds and how PTC stands on them:
- **S5 (feasible case broke binding clauses).** The plug-in spends at the S2 and S3 points are inside budget (section 4), and the
  rule enforces the binding income Brier and men's guard.
- **S6 (obfuscation: it injects propensity content).** Conceded in part and measured (section 4, learnability).

Other near designs:
- **FB2** (`nrm/fb/channel.py`): Gaussian noise along an LDA direction. PTC is deterministic, cancels the whole mean vector, and the
  person's own propensity sets the amplitude. FB2's mean-vector counterpart was a function of the release only and was never fitted.
- **MMP**: fits SEX-conditional task models. PTC keeps the task-only teachers.
- **Leaf corrections**: critic-trained per-leaf scores. PTC is output-level and closed form.

## 3. Which measured failure it addresses

- **Failure 1, "a SEX-blind control at matched spend did as well".** For any release whose shift amplitude is unrelated to the
  person's SEX, the covariance with SEX stays at the twin's, because Cov(t, s) = Cov(z, s); such a release only adds variance. PTC
  zeroes that covariance at the same spend. In the plug-in world, at alpha = 0.5, PTC's pair is 0.024 below the matched SHUF control,
  while FB2 was +0.002 / +0.003 and PRC was +0.0003 vs HQ.
- **Failure 2, super-additive pair.** Both recipients are cancelled jointly.
- **Failure 3, leakage moving between recipients.** In the plug-in world both views fall: income 0.556 to 0.524, occupation 0.628 to
  0.596.
- **Failure 4, table readers.** The release is continuous, with no finite cells.
- **Failure 5, income accuracy.** Plug-in income accuracy loss is 0.0016 at the S2 point and 0.0033 at the S3 point.
- **nrm's remainder finding.** nrm located the occupation view's SEX information in the non-argmax remainder (0.643 to 0.560 when
  flattened), but flattening broke utility. PTC removes only the SEX-covariant part of the remainder, within each decision.

## 4. Why it might preserve usefulness: label-free feasibility (plug-in world)

**How the plug-in world works** (`feasibility.py`; `FEASIBILITY.json`; V0 / V1).
- The cached propensity q plays the truth.
- Readers are HGB15 and LR on the benchmark's rich features, trained on 4,000 pseudo-labelled people.
- AUC is the expected AUC on the other 16,000 people.
- Spend is KL / Brier / accuracy against the twin posterior; the men's guard uses `occ_s[0]` as men's truth.
- These are simulations, not evidence about the benchmark readers.

| Release (V0 / V1) | Pair AUC | Income | Occupation | KL inc / occ | Income acc loss | Income Brier | Men's occ Brier |
|---|---|---|---|---|---|---|---|
| Twin (BASE) | .6420 / .6413 | .556 / .556 | .629 / .626 | 0 | 0 | 0 | 0 |
| PTC alpha 0.35 (plug-in S2) | .6219 / .6211 | .534 / .534 | .609 / .605 | .0023 / .0013 | .0016 | .0018 | .0015 |
| SHUF, matched to it | .6382 / .6368 | .556 / .555 | .626 / .623 | same total | .0015 | .0017 | .0002 |
| PTC alpha 0.5 (plug-in S3) | .6102 / .6095 | .522 / .525 | .599 / .594 | .0046 / .0027 | .0033 | .0037 | .0023 |
| SHUF, matched to it | .6347 / .6333 | .556 / .554 | .624 / .620 | same total | .0030 | .0034 | .0006 |
| Global occupation tilt, alpha 0.5 (ablation) | .6100 / .6097 | .522 / .525 | .604 / .601 | .0046 / .0044 | .0033 | .0037 | .0043 |
| PTC with estimated propensity, alpha 0.5 | .6190 / .6197 | .529 / .524 | .603 / .601 | .0066 / .0041 | .0045 | .0052 | .0031 |
| ... its SHUF | .6327 / .6301 | | | same total | | | |
| ZTILT relabeling, matched to PTC alpha 0.5 | .6335 / .6326 | .535 / .531 | .616 / .611 | same total | .0031 | .0035 | .0027 |
| Decisions only (floor, reference) | .5775 / .5776 | | | | | | |

Reading the table:
- **Spend.** Occupation accuracy loss is at most 0.0003 and the women's guard does not worsen. The tilt is bounded: at alpha = 0.5,
  no class log-probability moves by more than about 0.69 (largest |W| 2.72, in a stratum with qbar_k ≈ 0.49).
- **Stratification.** It halves the men's-guard cost against the global tilt at the same pair.
- **Estimated propensity.** A propensity re-estimated with correlation 0.84 to the truth shrinks the advantage over SHUF from 0.024 to
  0.012.
- **Relabeling.** The relabeling ablation (amplitude from P(SEX | the twin's own release)) does nothing for the pair. The gain needs the
  six-field propensity's information beyond the release.
- **Learnability** (RCCS's S6; `LEARNING_CURVE.json`, V0, alpha 0.5).
  - The pair benefit over SHUF is 0.025 / 0.025 / 0.022 / 0.018 with 2k / 4k / 8k / 16k reader-FIT people, and 0.010 for a data-rich
    denoised reader. About 60% erodes; it stays positive.
  - The benefit over the twin goes from 0.032 to 0.018.
  - So part of the gain is a reader-budget effect.
- **Superseded design** (`superseded_fisher_shift.py`). A person-specific Fisher-shaped direction re-encoded the propensity: its gain
  over SHUF was only 0.002 to 0.007.

## 5. Twin and controls

Bracket points are chosen on MODEL_FIT nested total NLL spend: one just below and one just above PTC_S3's.

| Config | Role | Content |
|---|---|---|
| TW_B | equally trained task-only twin | the `TASK.pkl` teachers released unchanged (alpha = 0); never reads SEX |
| SBC_LO / SBC_HI | protocol same-form SEX-blind control | PTC refit with a fixed-seed permutation of the MODEL_FIT SEX column (seed 20261021): the propensity HGB is refit on the permuted column with the same settings and folds, and the weights by the same rule. **Named fallback:** if Cov(q, s) <= 1e-4 on out-of-fold, or alpha <= 50 cannot reach PTC_S3's spend, the target becomes a keyed Bernoulli(0.5) label per (SCHL, COW) cell (`fallback_target`: SEX-independent, learnable, binary). On invented data the permuted column was degenerate and the fallback ran. |
| SPEND_LO / SPEND_HI | protocol spend-only, information-removing control | SEX-blind KL k-means quantization of each recipient's twin vector, releasing out-of-fold cell means (`nrm.fa.prc` twin_book / B1 tables). Codebook ladder: income {64, 32, 16, 12}, occupation {256, 128, 64, 48}. Non-invertible. |
| SHUF_S3 (further) | person-alignment control | PTC's own weights and strata; the person's q is replaced by a keyed six-field-hash donor's q; scale matched on spend |
| ZTILT_S3 (further) | relabeling control | the amplitude is P(SEX \| the twin's released log-probabilities), fitted with SEX on MODEL_FIT; own weights; spend-matched |

## 6. Bounded revision plan (at most two; candidate plus its SBC point each)

- **R1, recipient-split strengths.**
  - *Trigger.* Observed at the MODEL_FIT selection in both versions: the binding component at S3 belongs to one recipient while every
    component of the other recipient has at least 40% budget slack. The plug-in expects income to bind at alpha 0.5 while occupation
    has slack.
  - *Revision.* Select (alpha_I, alpha_O) on the product grid by the same rule (maximise alpha_I + alpha_O; ties go to the smaller
    max).
  - *Plug-in support (global tilt).* (0.35, 0.65) gives pair 0.609 at income Brier 0.0018 (`REVISION_PROBE.json`).
- **R2, joint-decision strata.**
  - *Trigger.* Observed after the grid in both versions: the pair benefit over SBC is below half the larger single-view benefit, so
    the linked pair re-assembles SEX.
  - *Revision.* Stratify the occupation weights by (income decision x occupation decision), with a fallback to the occupation
    stratum below 200 people, and the income weights by the occupation decision. The same rule then cancels the first moments
    within the decision cells that the linked reader sees.
  - Not probed.

## 7. Registered predictions, configurations, CPU

Predictions (registered grid; benefit = control AUC - PTC AUC):

| # | Prediction | Probability |
|---|---|---|
| P1 | TW_B meets the original 24 clauses and 4 guards in both versions, with pair 0.645-0.665 | 0.8 |
| P2 | alpha_S2 in {0.2, 0.35} and alpha_S3 in {0.35, 0.5}; the binding components are income Brier/NLL, then the men's guard | 0.7 |
| P3 | PTC_S3 pair is 0.012-0.030 below TW_B in both versions | 0.65 |
| P4 | Decisive. PTC_S3 beats the SBC bracket by 0.008-0.025 (positive in both versions p 0.75; resolved at 2 SE p 0.5) and the SPEND bracket by 0.010-0.030 (resolved p 0.55) | as stated |
| P5 | The income view is 0.012-0.030 below SBC (resolved p 0.6); the occupation view is 0.006-0.020 below (resolved p 0.4) | as stated |
| P5b | Neither view is worse than the twin's | 0.9 |
| P6 | ZTILT_S3 pair within ±0.007 of SHUF_S3; PTC_S3 beats SHUF_S3 by 0.008-0.020 | 0.7 / 0.45 |
| P7 | The 0.020 margin below NEURAL_RAW (0.6309) and H_LJ (0.6252) is NOT met; PTC_S3 pair 0.620-0.645 | 0.9 |
| P8 | PTC_S2 meets the original limits in both versions; PTC_S3 does too | 0.75 / 0.55 |
| P9 | Descriptive. The benefit is smaller on the robustness readers, and the FIT-fraction curve is roughly flat | — |

Falsifiers:
- **F1.** The benefit over SBC is not positive in both versions, or the aggregate is below 1 SE. The family closes; no revision.
- **F2.** ZTILT is within 0.005 of PTC: a relabeling explains the gain.
- **F3.** Recipient non-worsening fails.

**Configurations.**
- 7 required: TW_B, PTC_S2, PTC_S3, SBC_LO, SBC_HI, SPEND_LO, SPEND_HI. That is 6 if alpha_S2 = alpha_S3.
- 2 further, in priority order: SHUF_S3, then ZTILT_S3.
- Revisions add at most 4.
- Maximum 13.

**CPU, mechanism side, both versions.**
- Weights and selection: arithmetic on cached statistics, under 1 min.
- SBC propensity refits: 6 HGB fits, about 1-3 min.
- SPEND quantizers: about 4 min.
- ZTILT propensity: about 1 min.
- Releases: about 10 s per configuration.
- Total under 15 CPU-min. The reader grid dominates, at the campaign's per-release cost.

## 8. Novelty status

Not claimed. The components are published:
- the tower-identity use of a protected-attribute proxy: multiaccuracy (Kim, Ghorbani & Zou 2019), multiaccurate proxies (Diana et
  al. 2022), proxy-based disparity under unawareness (Chen et al. 2019);
- zero cross-covariance erasure: LEACE (Belrose et al. 2023), a projection;
- covariance or mean-difference constraints: Zafar et al. 2017 and Calders et al. 2013, both at training time;
- exponential-tilt posterior adjustment: Saerens et al. 2002;
- attribute averaging: Pope & Sydnor 2011, i.e. MMP;
- perturbing released vectors against attribute inference: AttriGuard (Jia & Gong 2018).

The closest in-repository design is RCCS, data-free. I did no literature search in this session, so "not found" means "not found in
the records read".

**Limits.**
- All section 4 numbers are plug-in simulations on adaptively used MODEL_FIT statistics.
- PTC injects propensity content and is injective on the support. Its benefit is a declared-reader-budget result, partly eroding with
  reader data. Optimal-attacker recovery is not claimed to fall.

Record: `mech_sketch.py` (selftest passes), `feasibility.py` (records `FEASIBILITY.json` and `FEASIBILITY_GLOBAL_TILT.json`), `learning_curve.py`, `revision_probe.py`, `superseded_fisher_shift.py`, and the early JSON files.
