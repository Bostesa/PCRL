# Family B mechanism card v3 (addendum to FB_FSC_v2): decisive-phase MEAN and the training perturbation FB2P

Designer B, 2026-10-20.
- It includes the statistical reviewer's ruling `roles/statistical_reviewer/RULING_FB_V3.md` (APPROVE WITH CHANGES, clock 15:59:33Z),
  with its changes written in exactly.
- It was committed before any output of the registered grid NRM1_MAIN was read, and before any fit it governs.
- Everything in FB_FSC_v2 stands unless changed here.
- Code: `nrm/fb/posthoc.py` and `nrm/fb/release.py` at commit 2fa4e64ac. Invented-data selftest: 30/30 under python3 and under the torch
  venv.
- The MEAN and FB2P fits run only on the coordinator's GO, after the grid results have applied the rules below.

**Label attached to every result governed here:** "FB2 was designed after family-B screening on MODEL_SELECT (targeting signal, F3); it is
evaluated on the same people; development evidence only."

**Scope sentence:** one released instance per person; repeated-query protection not claimed.

## 1. Decisive-phase MEAN (reviewer C5)

**Configuration:** FB2_SEX_S<j>_MEAN in both versions (+2), kind 'ablation'.

**Operation:**
- The exact deterministic mean vector E_ε[emitted] of the FITTED FB2_SEX_S<j> channel.
- It uses the same frozen geometry (δ, Σ_w), per-task temperatures, cap and strength, read from that channel's write-once
  `CHANNEL_FB2_SEX_S<j>.json`, whose SHA-256 is recorded.
- It is evaluated by 48-node Gauss-Hermite quadrature on the frozen TWA base.
- Decisions are the argmax of the mean vector. It is deterministic: R = 1, and keys are refused at deployment.

**Fitting:**
- `python3 -m nrm.fb.posthoc fit-mean --version <v> --of FB2_SEX_S<j> --card results/pcrl_new_release_mechanisms_v1/cards/FB_FSC_v3.md`.
- **No label is read.** No new geometry, temperature or critic is fitted (n_calibrator_fits = n_critic_fits = 0). The v2 base gate
  (bitwise equality with ohn's hash-checked stored releases) applies.
- It refuses when the source channel is not fitted.

**Strength (C5):**
- The highest FB2_SEX strength that is TASK_FEASIBLE in both versions AND DOMINATES_CONTROL_FRONTIER against both declared SEX-blind
  paths (MINF and CTL_REM on FB2_BASE).
- If no strength qualifies, MEAN is not fitted.

**F2, exact definition:**
- F2 HOLDS iff `nrm.rules.benefit_screen` on the pair view is NOT `AGG_POSITIVE`.
  - Candidate FB2_SEX_S<j>, comparator FB2_SEX_S<j>_MEAN, registered cells.
- In words: randomization counts as the active ingredient only if the randomized release is resolved as less recoverable than its own
  mean vector.
- Reported beside it:
  - the aggregate and both version means;
  - MEAN's own G1;
  - the single-view (income and occupation recipient) differences.

**MEAN is not selectable:** it has no declared G2 alternative, so it is `NO_MATCHED_CONTROLS` (DECLARATIONS_V1, D1).

## 2. Training perturbation FB2P (brief section 12: "repeat a promising comparison under a genuine training perturbation, with equally perturbed controls")

**Configurations:** FB2P_SEX_S<j> and FB2P_MINF_S<j>, each in both versions (+4), kind 'perturbation'. Revision tag FB2P.

**What is re-estimated** (everything the mechanism itself fits; the frozen base is an admitted input and is unchanged):
- **SEX geometry (δ, Σ_w):**
  - From a **household bootstrap resample of MODEL_FIT**: 13,337 households drawn with replacement, all members of each drawn
    household.
  - Seed [7300, v, 0].
  - Same geometry logits as v2: A1_PO_16 out-of-fold occupation, HC_STRONG in-sample income.
- **Per-task CALIBRATION temperatures:**
  - From a **household bootstrap resample of CALIBRATION**, with seed [7300, v, 1].
  - **The same resample for FB2P_SEX and FB2P_MINF** (equally perturbed). MINF has no SEX geometry, so its fitted parameters are only the
    temperatures.
- **Release key:** a **fresh write-once root key** per (FB2P, version), at `runs/FB/MODELS/FB2P/V<v>/RELEASE_ROOT.bin`. The emitted draws
  are therefore fresh as well.
- **Recorded in every record and CHANNEL file:** the seeds, the number of households, the households drawn and distinct, and the people in
  the resample and distinct.

**Limitation sentence (travels with every FB2P result):** "the frozen base models are not refitted; the perturbation covers the channel's
fitted parameters and the release key only".

**Fitting:**
- `python3 -m nrm.jobqueue run --unit FB2P_FIT_V<v> --stage fit --rss-gib 2 -- python3 -m nrm.fb.posthoc fit --version <v> --card results/pcrl_new_release_mechanisms_v1/cards/FB_FSC_v3.md --configs FB2P_SEX_S<j>,FB2P_MINF_S<j>`
- One unit per version, one revision per unit (enforced).
- Label reads as in v2: MODEL_FIT SEX ('fit') and CALIBRATION income and occupation ('calibrate').

**(a) Strength rule (declared before NRM1_MAIN is read):**
- FB2P uses the highest FB2_SEX strength that meets both conditions:
  - TASK_FEASIBLE in both versions;
  - its registered-cell pair benefit against its same-strength FB2_MINF is `AGG_POSITIVE` on NRM1_MAIN.
- If no strength qualifies, FB2P is not fitted (label NOT_RUN), and its 4 configurations are not spent.
- FB2_MINF_S5's v1 utility failure does not block S5. The control is still perturbed at the same strength, and its own G1 is reported.

**(b) Only the equal pair is read.**
- FB2_BASE and the CTL_REM points are not refitted. If they carry fitted parameters (temperatures, remainder shares), they are not
  equally perturbed.
- So the consistency reading uses only FB2P_SEX_S<j> against FB2P_MINF_S<j>.
- Comparisons against unperturbed points are descriptive.
- No frontier label is computed on a path that mixes perturbed and unperturbed points.

**(c) Paired read.**
- FB2P is read on the registered cells with the declared instance map (R1 → instance 0, R2 → instance 1, k0 and k1 sharing it).
- It is read in a bank that also contains FB2_SEX_S<j> and FB2_MINF_S<j>. Their readers are cache hits, so nothing is refit, and the
  perturbed-minus-original contrast shares the bootstrap weights.
- Monte Carlo replicates are not required.

**(d) Utility and spend reported:**
- the G1 of both FB2P releases;
- the `nrm.evaluate` paired costs of FB2P_SEX against FB2P_MINF: `total|nll` and the 8 secondary metrics.

## 3. Perturbation reading (descriptive only; defined exactly as in RULING_FB_V3 section 3)

All screens use NRM1 functions on the registered cells.

**Inputs.**
- B0 = `benefit_screen` (pair view) of FB2_SEX_S<j> against FB2_MINF_S<j>, with aggregate x0 and standard error s0.
- B1 = the same for FB2P_SEX_S<j> against FB2P_MINF_S<j>, with x1 and s1.
- Δ = x1 − x0.
- se_Δ = `se_agg` of the per-cell difference entry, with cells = b1_c − b0_c and cell_se = the SD of the paired draw differences from the
  shared bank (`nrm.rules.aggregate`). Without a shared bank, use the bound se_Δ = s0 + s1.
- c1 = pooled paired total-NLL cost of FB2P_SEX against FB2P_MINF (`nrm.rules._primary`), with value and SE.

**Labels.**
- `NOT_RUN`: FB2P was not fitted (rule 2a) or not read on the registered cells.
- `NOT_APPLICABLE_ORIGINAL_NOT_POSITIVE`: B0 is not AGG_POSITIVE. The numbers are still reported. Under rule 2a this cannot occur for a
  fitted FB2P.
- `PERTURBATION_CONSISTENT`: all three of the following hold:
  - B1 is AGG_POSITIVE;
  - |Δ| ≤ 2·se_Δ;
  - SEX does not spend resolved-more than MINF after the perturbation, that is, NOT (c1.value − 2·c1.se > 0).
- `PERTURBATION_NOT_CONSISTENT`: otherwise. The failing parts are named: "perturbed benefit not resolved", "benefit changed by more than
  2·se_Δ (direction)", or "SEX now spends more than MINF".

**Status.**
- The label is a qualifier reported beside FB2's "SEX beyond MINF" statement. It never changes a decision, a tier, a selection or a
  falsifier.
- The FB2P releases are kind 'perturbation'. They are never pooled with FB2, never matched controls, and never selectable (no declared G2
  alternative).
- The limitation sentence travels with every FB2P result.

## 4. Budget and order

- **Configurations:** at most 6 more: MEAN 2 (conditional on C5) and FB2P 4 (conditional on rule 2a). Family B then holds at most 34.
- **CPU:**
  - FB2P is about 20 CPU-s per version unit, measured on the FB2 units.
  - MEAN is about 15 CPU-s per version.
  - Request 0.05 CPU-h.
- **Order:**
  1. The registered grid NRM1_MAIN.
  2. The coordinator applies C5 and rule 2a to the grid and utility records, and records the outcome.
  3. On GO: fit FB2P (both versions) if rule 2a qualifies a strength, and MEAN (both versions) if C5 qualifies a strength.
  4. Read as in sections 1-3.

## 5. Registered predictions (before any NRM1_MAIN output was read and before any fit governed here)

- **P6b:** if MEAN is fitted, F2 does not hold. FB2_SEX_S<j> is AGG_POSITIVE against its own mean vector (p ≈ 0.6). MEAN recovers within
  0.003 of FB2_BASE (p ≈ 0.75).
- **P7b:** if FB2P is fitted, the reading is PERTURBATION_CONSISTENT (p ≈ 0.55). |Δ| is less than 0.005 in pair AUC.
