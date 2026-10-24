# FA_TFRO_R1: occupation-only SUCC-anchored teacher-feature residual (revision R1 of FA_TFR_v1; card draft, to be committed as `cards/FA_TFRO_R1.md` before its fit)

> **SAR1 declarations (coordinator; part of this card):** family FA, revision 1 (reading FA_R1); START = SUCC;
> `cont_declared` = true (CONT = FA_TFRO_R1_CONT); `ctrl_origin` = START; CTRL = FA_TFRO_R1_TASK (the occupation-only lambda = 0
> interpolation path), saved points FA_TFRO_R1_TASK_LO / FA_TFRO_R1_TASK_HI (an endpoint alias of START or CONT is not saved twice);
> `alt_declared` = false; `selection_data` CTRL = CALIBRATION; `start_in_sample_on_model_fit` = true (H3 income and Ucal are in-sample on
> CALIBRATION; in this revision every income output is SUCC's own, so no income spend enters the rule); joint design not claimed; novelty
> not claimed; no mechanism-specific reader declared; at most 8 new specifications. The budgets are SUCC's own MODEL_SELECT margins to
> the caps from the frozen anchor table (MODEL_SELECT aggregates of the frozen anchor, known before this campaign; disclosed as such).
> **Adaptive status:** designed from FA_TFR_v1's MODEL_FIT / CALIBRATION records and the committed anchor table only; no MODEL_SELECT
> outcome of FA or FB had been read. The mechanical rule (deviation 5) tags this reading ADAPTIVE_, and that tag is kept. The
> investigator's text follows unchanged.


Investigator A, 2026-10-24 05:33Z (`date -u`).
- **Status:** REVISED_. It was designed from the committed FA_TFR_v1 MODEL_FIT / CALIBRATION fit records only; no MODEL_SELECT read has
  happened for FA.
- **Code:** `sar/fa/tfro.py`, a new module. It reuses `sar/fa/tfr.py` helpers unchanged: teachers, features, pins, bracket, preflight.
- **Deployment:** `sar/fa/__init__.py` routes `FA_TFRO_R1*` ids to `sar.fa.tfro.predict`. Every other FA id still goes to
  `sar.fa.tfr.predict`, unchanged.
- **Interpreters:** fitting runs in the torch venv on one thread; deployment needs numpy and sklearn only.
- **Evidence status:** development evidence on previously used records.

## 0. Diagnosis (from `results/pcrl_strong_anchor_release_v1/fa/FIT_V0.json` and `FIT_V1.json` only)

Both versions read NO_FEASIBLE_STRENGTH: ρ = 0.3 failed both parts of the v1 rule. The figures below are v0 / v1, estimates with
household-clustered SE.

| | What the records show | Reading |
|---|---|---|
| **D1: part (a) failed on components CALIBRATION cannot measure at v1's budgets** | Every point estimate is favourable or near 0: occupation NLL −0.0160 / −0.0140, occupation Brier −0.0063 / −0.0053, occupation accuracy −0.0091 / −0.0050, income accuracy −0.0081 / −0.0054, income Brier +0.0007 / +0.0019, income NLL +0.0004 / +0.0022. The failures are income Brier (bounds 0.0043 / 0.0056 > 0.0030), income NLL in v1 (0.0072 > 0.0060) and occupation accuracy in v1 (0.0056 > 0.0050) | The v1 budgets (0.4-0.6 of SUCC's margins) sat inside CALIBRATION's 2-SE noise (2 SE: income Brier 0.0036, occupation accuracy about 0.011). Worse, the income-side CALIBRATION spends are biased in SUCC's favour, because H3's STRONG stack, β budget and Platt map were fitted on all of CALIBRATION. Moving the income output makes the income components unmeasurable without that bias |
| **D2: part (b) is not a valid estimator of the cap-relevant quantity** | On CALIBRATION, where the occupation side is out of sample, CAND's occupation summed Brier is −0.0063 / −0.0053 better than SUCC's (resolved). On MODEL_FIT, where SUCC is in-sample, the same CAND reads +0.0148 / +0.0165 worse (descriptive out-of-fold): a sign flip of about 0.02. The MODEL_FIT difference against the λ = 0 LO point (s = 0.5 / 0.4) is +0.0073 / +0.0144 for men and +0.0139 / +0.0110 for women (SE about 0.0006) | This difference is of the same order as SUCC's in-sample offset and cannot be separated from it on MODEL_FIT. Two releases that depart from SUCC differently forfeit different amounts of its in-sample fit, so the first-order cancellation assumed by the v1 rule does not hold. LO's CALIBRATION occupation Brier was not recorded; its total NLL is at least as good as CAND's by construction |
| **D3: within one release the SEX contrast is little affected by that offset** | For the λ = 0 LO point both groups read about +0.004 over SUCC on MODEL_FIT (the shared in-sample term), with a men-minus-women contrast of only −0.0005 / +0.0007. CAND's contrast is −0.0072 / +0.0042 at ρ = 0.3. The descriptive path turns men-heavy with ρ: +0.0042 / +0.0061 at ρ = 3, +0.0156 / +0.0201 at ρ = 10 (exact values: `r1_contrasts.json`) | The critic-shaped movement puts its cost increasingly on men, which is the binding guard. The contrast is the part of the guard that only MODEL_FIT can supply |

**Also:**
- At ρ = 0.3 CAND recovered on CALIBRATION most of the occupation quality that separates SUCC from the HGB teacher. On MODEL_SELECT that
  gap is 0.0186 / 0.0187 nats, an anchor-table difference, not comparable sample by sample. The weakest grid point is therefore already
  a strong move toward the teacher.
- The v1 card's declared revisions are not used. R1 (pair-only critics) has a trigger that refers to the λ = 0 path's CALIBRATION
  spends, which the records do not hold when no candidate exists, and it would not address D2. R2 (SBC) needs a reading of CAND.

## 1. Operation (mechanism change: the residual may move the occupation output only)

**Runtime inputs:** the six permitted fields only. SEX is never an input or output.

**Release:**
- **income** = SUCC's H3 income vector, **bitwise**;
- **occupation** = softmax(centre(z_S + s·r_O(φ))). r_O is a network 29 → 64 → 64 → 5 (ReLU) whose zero-initialized output layer makes
  θ = 0 START. s = 1 for CAND; s = 0 is START exactly.
- φ is unchanged from v1: the standardized compact fields; COW and MAR one-hot; SUCC's logits; the HGB teachers' logits (out-of-fold
  on MODEL_FIT training rows, deployed elsewhere).

**Training, as FA_TFR_v1 except on the occupation output only:**
- the task term is cross-entropy of the occupation vector to the HGB occupation teacher's out-of-fold probabilities, so the fit reads
  no task label;
- plus λ · [risk(income) + risk(occupation) + risk(pair)] with `acs_replication.leafcorr.Critics` unchanged. risk(income) is constant
  because income is frozen; the pair critic sees H3's income with the moved occupation.
- λ comes from the same 16-draw calibration rule;
- AdamW lr 1e-3, weight decay 0.01, clip 5; 20 epochs × 40 steps × 256 rows; E20 kept.

**Why this is structural.**
- Every component that CALIBRATION cannot measure without SUCC's in-sample bias (all income components, D1) is removed from the
  operation, so the income clauses are SUCC's exactly.
- Every fitted layer that the task part of the rule measures is out of sample on CALIBRATION: JOINT_TREE, the HGB occupation teacher
  and the residual.
- The binding condition, the men's occupation guard, is the only part left to estimate. Part (b) estimates it from data that can see it
  (section 6).

**Frozen inputs:** SUCC's 7 frozen files per version (pinned; `preflight` passes for both versions as of 05:31Z) and nrm TASK.pkl
(pinned in this campaign's `SOURCE_ADMISSION.json`). Hashes are recorded in every model file, and `predict` refuses changed files.

## 2. Closest previous implementation and the difference

**FA_TFR_v1** (both outputs moved; the v1 rule). R1 differs in three ways:
- one output only;
- a guard estimator that does not rely on the in-sample cancellation;
- budgets equal to SUCC's margins to the caps.

**Occupation-headroom P2 / P2A** (an occupation-only leaf-score correction against the pair critic with H3 frozen, from a task-only
witness base). TFRO differs in two ways:
- it is not a leaf-score correction: a network over teacher logits and compact fields;
- it starts at the protected release and moves toward the teacher, rather than adding protection to a task-only base.

**IA_OSG** (occupation-only input generalization, income passed through). The same recipient restriction with a different operation.

## 3. Measured failure addressed

**SUCC's men's within-SEX occupation summed-Brier guard:** +0.0044174 / +0.0039406 against 0.005 (margin 0.00058 / 0.00106).

**The v1 diagnosis D1-D3:** a residual that recovers task quality cannot be selected while the income output moves and while the guard
is estimated by a MODEL_FIT difference that does not cancel SUCC's in-sample fit.

## 4. Why usefulness may be preserved

**On occupation**, the CALIBRATION records at ρ = 0.3 show resolved gains over SUCC: NLL −0.016 / −0.014, Brier −0.0063 / −0.0053.

**On income**, there is nothing to preserve: it is SUCC's bitwise, so the 12 income clauses are SUCC's own (income reserve 3.51 / 2.86).

**The usefulness result reported** is MODEL_SELECT with the original, unchanged caps (24 clauses, 4 guards).

## 5. Twin / CONT

**FA_TFRO_R1_CONT:** the same occupation-only network, features, rows, folds, optimizer and schedule with λ = 0 (no critics, no SEX), at
s = 1. Its income is SUCC's and its occupation is the distilled HGB teacher, so it is about W (pair 0.6459 on IPR1_MAIN). **Gating.**

## 6. Strength grid and selection rule (prospective; SAR1 frozen rules otherwise)

**Grid:** ρ ∈ {0.3, 0.6, 1, 2, 3, 6, 10}. Each ρ is fitted in increasing order (5 household out-of-fold fits on SAR_xf_FA_S<v>, the v1
folds, plus 1 full fit), and the scan stops at the first ρ that fails. ρ\* is the largest ρ such that it and every smaller ρ pass both
parts below. If ρ = 0.3 fails, nothing is released in that version (NO_FEASIBLE_STRENGTH).

**Budgets are SUCC's own margins to the caps, not internal allowances.** They are read in code from the committed
`anchors/ANCHOR_TABLE.json` (`succ_margins`): cap minus the larger of SUCC's U0 / Ucal clause values per version, and cap minus SUCC's
guard value for the guards.

| Component | v0 | v1 |
|---|---|---|
| occupation NLL | 0.006594 | 0.007053 |
| occupation summed Brier | 0.004877 | 0.004506 |
| occupation accuracy loss | 0.012000 | 0.014000 |
| men's guard | 0.000583 | 0.001059 |
| women's guard | 0.009938 | 0.008039 |

**Part (a):** the CALIBRATION paired occupation spends of the full-MODEL_FIT fit over SUCC (NLL, summed Brier, accuracy loss), each at
estimate + 2 household-clustered SE, are within the margins. The income spends are exactly 0, and the code asserts it.

**Part (b), the guard estimate** (replaces v1's MODEL_FIT difference against a λ = 0 point):
- L = the CALIBRATION paired occupation summed-Brier spend over SUCC. It is the population-average change, measured where every layer
  it involves is out of sample.
- C = the MODEL_FIT household out-of-fold men-minus-women occupation summed-Brier spend contrast of the same release over SUCC, with a
  linearized, household-clustered SE. SEX exists only on MODEL_FIT.
- p = the MODEL_FIT share of men.
- Δ_men = L + (1 − p)·C and Δ_women = L − p·C, with SE² = SE_L² + weight²·SE_C² (CALIBRATION and MODEL_FIT are disjoint people).
- Pass if Δ_men + 2 SE ≤ the men's margin and Δ_women + 2 SE ≤ the women's margin.

**Why this measures the cap-relevant quantity.**
- The cap is each group's summed-Brier excess against U0 on new people, i.e. SUCC's excess (which sets the margin) plus the release's
  change for that group.
- Any group change decomposes exactly as the population-average change plus the share-weighted between-group contrast, so Δ_men and
  Δ_women are the identities applied to the two estimable pieces.
- The level comes from the only SEX-free data on which SUCC's occupation side is out of sample (CALIBRATION). The contrast comes from
  the only SEX-labelled data (MODEL_FIT), where SUCC's in-sample term enters both groups of one release.
- The records support that this term is close to SEX-symmetric: the λ = 0 LO point's contrast is −0.0005 / +0.0007 while both its
  groups carry about +0.004.

**Biases that remain (stated before the fit):**
1. C still contains any SEX-asymmetric part of SUCC's in-sample term forfeited by the release. Its size for the critic-shaped move is
   not known.
   - The λ = 0 evidence suggests it is small and, if anything, men-unfavourable (conservative): the λ = 0 point's MODEL_FIT contrast is
     less men-favourable than half of the teachers' MODEL_SELECT contrast (anchor table: men −0.0083 / −0.0062 against women −0.0048 /
     −0.0057 at s = 1).
   - This is a design-time reading of committed numbers, not a calibration.
2. CALIBRATION's SEX composition is unknown and assumed equal to MODEL_FIT's p. The two roles come from one random split.
3. The rule controls sampling noise in L and C (2 SE). It does not cover the MODEL_SELECT evaluation noise of the guard (SUCC's own guard
   SE is 0.0018 / 0.0017), so point feasibility on MODEL_SELECT is predicted, not guaranteed.
4. The margins are MODEL_SELECT quantities taken from the committed anchor table, as the brief requires. The rule spends up to the full
   point margin, with the 2-SE cushion on the CALIBRATION / MODEL_FIT estimates.

**Label use:**
- CALIBRATION income and occupation labels (never SEX): only part (a), the level L and the CTRL bracket. Nothing is fitted on
  CALIBRATION, and the controls' exposure is identical.
- MODEL_FIT SEX: the critics and the contrast C. MODEL_FIT task labels: C only.

## 7. Matched controls (frozen SAR1 rules)

**CTRL:** the occupation-only λ = 0 path with origin START (s = 0 is SUCC exactly), on S_PATH = {0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
0.4, …, 1}. The declared selection data are the CALIBRATION paired total-NLL spend over SUCC, which equals the occupation NLL spend
because income is SUCC's. `sar.fa.tfr.bracket` is used unchanged:
- **FA_TFRO_R1_TASK_LO:** the point with the largest spend ≤ CAND's;
- **FA_TFRO_R1_TASK_HI:** the point with the smallest spend ≥ CAND's spend + 2 SE (SE of that point's paired difference against CAND);
- flags CONTROL_PATH_MAY_NOT_BRACKET and CONTROL_PATH_NO_LO_POINT;
- comparisons on spend values only;
- endpoints at s = 0 (START) or s = 1 (FA_TFRO_R1_CONT) are recorded as aliases, not saved twice.

**CONT:** FA_TFRO_R1_CONT (λ = 0, s = 1), gating.

**Ablation:** CTRL is the critic ablation of CAND.

**Not built:** SBC, the DQ spend path and the release-only tilt. D5 is not claimed.

## 8. Revisions

This is FA's first revision slot. No further revision is declared; a second one would need a new card.

## 9. Registered predictions, specification count, CPU, fit record

**Predictions** (per version unless stated; registered-cell means for AUCs):

| Quantity | Prediction |
|---|---|
| ρ\* | P(0.3) 0.15, P(0.6) 0.20, P(1) 0.25, P(2) 0.15, P(3) 0.08, P(6) 0.04, P(10) 0.03, P(None) 0.10 |
| the scan stops on part (b), the men's guard estimate (not on part (a)), in both versions | 0.65 |
| CALIBRATION occupation NLL spend of CAND over SUCC < 0 in both versions | 0.85 |
| MODEL_SELECT men's guard of CAND below SUCC's in both versions | 0.80; men's reserve ratio ≥ 2 in both: 0.35 |
| all 28 conditions met at the point, both versions | 0.75 |
| pair AUC | 0.620 (80%: 0.600-0.638); occupation view 0.610 (0.585-0.632); income view = SUCC's 0.531 up to reader refitting |
| pair benefit over SUCC | −0.013 (80%: −0.030 to +0.005); P(> 0) 0.20; P(resolved) 0.04 |
| pair benefit over FA_TFRO_R1_TASK_LO | +0.010 (80%: −0.005 to +0.025) |
| pair benefit over FA_TFRO_R1_TASK_HI | +0.003 (80%: −0.012 to +0.018) |
| resolved over both CTRL points | 0.12 |
| FA_TFRO_R1_CONT pair AUC | 0.645 (0.638-0.652); CAND resolved below it: 0.75 |
| CONTROL_PATH_MAY_NOT_BRACKET in at least one version | 0.30; at least one alias: 0.45 |
| D3 / D4 | 0.02 / 0.02 |

**How a failure would be read:**
- **Men's guard improves but the pair is not below SUCC:** usefulness margin recovered, profile not kept, in this function class. The
  teacher's men-favourable gain is SEX-readable.
- **CAND not separated from CTRL:** the critics do not shape the movement toward the teacher.
- **ρ = 0.3 fails on part (b):** the critic cost lands on men faster than the teacher's gain recovers them.

**Specifications:** at most 8 = CAND 2 (FA_TFRO_R1 × 2 versions) + CTRL 4 (FA_TFRO_R1_TASK_LO and _HI × 2) + CONT 2 (FA_TFRO_R1_CONT ×
2). There are fewer with aliases, a missing bracket side or NO_FEASIBLE_STRENGTH.

**CPU:**
- per ρ, 6 critic fits at about 9.5 CPU-s each (measured label-free for v1);
- the scan is expected to stop after 3-4 values, so about 4-6 CPU-min per version;
- the worst case is about 9 CPU-min (all 7 values plus the λ = 0 path, releases and MODEL_FIT readers).

**Fit record** (`results/pcrl_strong_anchor_release_v1/fa/FIT_R1_V<v>.json`), per version:
- CAND: its config, ρ\*, status and its CALIBRATION total-NLL spend over SUCC (value, SE, all components);
- the per-ρ table: part (a) spends and pass flags, the guard estimate (L, C, p, Δ_men, Δ_women with SEs) and pass flags, the margins, λ,
  and the biased MODEL_FIT spends;
- CTRL: the origin, the selection data, every S_PATH point's spend over SUCC and the SE of its paired difference against CAND, the LO /
  HI ids or aliases with their spends, and the flags;
- CONT: its config (gating);
- the frozen-file and TASK.pkl hashes;
- the descriptive MODEL_FIT readers.

## 10. Novelty

Not claimed. The closest published methods are adversarial debiasing (Zhang, Lemoine & Mitchell 2018), knowledge distillation (Hinton,
Vinyals & Dean 2015), and stacking and residual learning on frozen teachers. No literature search was run.
