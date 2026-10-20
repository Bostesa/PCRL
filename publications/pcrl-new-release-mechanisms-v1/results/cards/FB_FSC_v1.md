# Family B mechanism card v1: FSC-CA (channel-aware joint training with a SEX-targeted, Fisher-shaped, coupled Gaussian logit channel)

Designer B, 2026-10-20. This card governs every family-B fit.

- It includes the coordinator's decisions in `roles/coordinator/CARD_DECISIONS_V1.md` (committed 15:19:15Z) and the challenger's
  review in `roles/challenger/REVIEW_PROPOSALS.md`.
- The protocol receipt was verified at 15:14:43Z. The card was committed before any family-B fit or label read.
- Code: `nrm/fb/` at commit e4da5a8fc (`nrm.fb.channel`, `nrm.fb.learner`, `nrm.fb.release`). The invented-data selftests pass:
  channel 27/27, learner 20/20.
- Scope: exploratory development on previously used public-use ACS 2019 1-Year PUMS Florida records. All MODEL_SELECT evidence is
  development evidence, never confirmation.
- "Recovery" means the declared development readers' AUC for the recorded SEX column. Lower recovery for these readers is never
  protection against every reader.

## 1. Operation

**Learner.**
- Two task networks, one per task (income K = 2, occupation K = 5).
- Each network is `Linear(145,128)-ReLU-Linear(128,128)-ReLU-Linear(128,32)-Linear(32,K)`: the U0 Recipe A shape at twice the width.
- Both networks are trained jointly from scratch under the configuration's own channel.
  - The loss is the summed income and occupation cross-entropy of the EMITTED vectors.
  - Each training example gets one draw per step, reparameterized: the gradient flows through the deterministic logits.
  - The per-person noise operator is recomputed from the detached logits.
- The release maps the six raw fields, through the admitted 145-column encoding X, to the two vectors.
- Each recipient receives only its own float64 probability vector and its argmax decision. One vector and one decision per
  recipient per person.

**Coordinates.**
- z = [H2 l_inc ; H5 l_occ] in R^5, where H_k are orthonormal Helmert rows orthogonal to the all-ones vector.
- These are linked centred logits: 1 income coordinate and 4 occupation coordinates. Mapping back leaves the softmax unchanged.

**Channel law** (person i, release instance r).
- F_i is the per-person Fisher of the summed task log loss in z. It is block-diagonal and computed from the deterministic
  probabilities, so it needs no label.
- Directions, one per mode:
  - **'sex' (candidate):**
    - d_i = (F_i + s^2 Sigma_w^-1)^-1 Sigma_w^-1 delta;
    - v_i = d_i / sqrt(d_i' F_i d_i), so v_i has unit Fisher norm.
    - The geometry is frozen per configuration from MODEL_FIT, on the final learner's in-sample logits:
      delta = mean z(women) - mean z(men), and Sigma_w = pooled within-SEX covariance (ridge 1e-6 times the mean variance).
    - d_i is the rank-1 minimizer of the linked linear-Gaussian SEX discriminability delta' (Sigma_w + Sigma_n)^-1 delta at
      second-order Fisher cost s^2 / 2.
  - **'minf' (SEX-blind control path, G3):**
    - v_i = e_i / sqrt(lambda_min,i), where e_i is the eigenvector of F_i with the smallest eigenvalue ("noise where the task is
      cheapest").
    - It has unit Fisher norm. Sign convention: the largest-|.| component is positive.
    - Because F_i is block-diagonal, this direction generically lies in ONE recipient's block. MINF's spend can therefore fall on one
      task. On the invented selftest data it fell entirely on occupation.
  - **'iso' (strong SEX-blind control):**
    - z~_i = z_i / T + c_i F_i^-1/2 xi_{i,r} / sqrt(5), with xi_{i,r} ~ N(0, I_5).
    - These are five independent draws, so ISO is UNCOUPLED.
    - c_i = min(s, 1.5 / sqrt(tr F_i^-1 / 5)).
- Rank-1 modes ('sex', 'minf'):
  - c_i = min(s, 1.5 / |v_i|). The predeclared cap is 1.5 centred-logit units per unit draw.
  - z~_i = z_i / T + c_i eps_{i,r} v_i, where T holds the per-task CALIBRATION temperatures and the geometry is rescaled exactly to
    match.
  - eps_{i,r} ~ N(0, 1) is ONE scalar per person per instance.
- Emission:
  - p~_inc = softmax(H2' z~_inc) and p~_occ = softmax(H5' z~_occ);
  - each decision is the argmax of the emitted vector.
- Cost: to second order, the expected NLL cost of the noise is s^2 / 2 nats per person (linked total) in every mode. For calibrated
  vectors, the summed-Brier excess is at most that cost, to second order.
- Where the cap binds, the realized cost is lower. This applies mostly to MINF when lambda_min < (s / 1.5)^2.
- **Per-task spend is read beside the linked cost.** The direction decides how the linked s^2 / 2 splits between income and
  occupation, so it can fall on occupation alone.

**Noise coupling.**
- SEX and MINF use the same coupled rank-1 construction: one shared scalar per person drives both recipients' perturbation along
  the person's own direction. Their comparison therefore isolates targeting.
- ISO is uncoupled.
- **No coupling or joint-design claim is made.** A coupling contribution would need an uncoupled version of the SEX channel, which is
  not funded.

**Seed handling (frozen).**
- Draws are generated as follows:
  - eps_{i,r}, and xi_{i,r} for ISO, are Box-Muller transforms of keyed BLAKE2b;
  - key = key_{config,v} = BLAKE2b(key = root_v, message = "NRM-FB|<config>|V<v>");
  - message = "NRM-FB|<role>|r<r>|<row_id>".
- A draw depends only on (key, role, r, row_id). It does not depend on row order, batch or n.
- Instances r = 0..7 are independent. Within an instance, a re-query returns the same draw.
- Other streams, each separate from the release keys:
  - CALIBRATION temperature fitting uses its own key ("CALIBRATION-DRAWS"), with common random numbers across configurations;
  - early stopping uses fixed fit-time-only validation draws;
  - training noise and permutations come from numpy PCG64 streams seeded by (version, phase, epoch), identical for every learner of a
    version, so learners are paired.

**Custody of the private root key.**
- root_v is 32 bytes from `os.urandom` at fit time. It is written once to `.private/<campaign>/runs/FB/MODELS/V<v>/RELEASE_ROOT.bin`
  (mode 0600).
- It is never published or committed.
- With it, the draws can be regenerated and the channel inverted, so it is a custody secret.
- Replay happens only from the private tree or its backup. The independent verifier may replay from the private tree.

**Deployed randomness for people outside the admitted rows.**
- A deployed service draws per-request keyed draws from a service CSPRNG key, used as the root, with a per-request identifier as
  row_id.
- Each person gets one released instance. **Repeated-query protection is not claimed.**
- Claims are limited to one released instance per person. No reader may ever see two instances of the same person.

**R = 8.**
- Each randomized configuration stores 8 independent one-draw-per-person release instances as (8, n, k) stacks for all six roles.
- Utility (via `nrm.evaluate`) is computed on each emitted vector and its emitted argmax, then averaged over the 8 instances.
- The Monte Carlo SE (sd over instances / sqrt 8) is reported separately from household-sampling and reader-fitting variation.
- An averaged vector is never substituted: FB_SEX_S2_MEAN is its own deterministic release.

**Reader sampling access.**
- Readers train and score under the same channel law.
- In a reader cell every person contributes exactly ONE draw, from the same instance r:
  - auxiliary people (reader FIT and VALIDATION) contribute one draw each;
  - SCORE people contribute one draw each.
- No multi-draw variant is requested, and a target person never supplies more than one draw.
- Linked readers pool the two recipients' draws of the same instance, which is the actual joint release.

**Deployment.**
- `nrm.fb.release.predict(config_id, version, X, feature_names, draw=0, keys=None)` reads only `model_paths(config_id, version)`:
  - the learner weights;
  - `CHANNEL_<config>.json` (frozen geometry, temperatures, strength, cap, feature names);
  - for randomized configurations only, the root key file.
- Randomized configurations need out-of-band keys {'role', 'row_id'}. Deterministic configurations refuse keys and any draw other
  than 0.
- predict with the role's own keys and draw 0 reproduces RELEASE.npz draw 0 bitwise. This needs the torch interpreter.

## 2. Closest prior PCRL attempts; what is different; which bottleneck

**Closest attempt:** `results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/`, score channel "A" (Adult).
- It mapped frozen U0 scores to a frozen T* bank (16 / 321 canonical vectors).
- A gate chose among at most 16 same-class joint actions, with one joint categorical draw per person, R = 3.
- 14 of 16 FIT searches had no feasible policy, and no sampled release was attacked.
- The measured bottleneck was the bank: the identity action itself exceeded the FIT occupation U0 NLL cap (+0.0043 / +0.0054;
  `design/A_FIT_FEASIBILITY_SUMMARY.md`).

**Related attempts:**
- `pcrl_conditional_budgeted_release_v1`: LP mixtures toward frozen donors. Pair reduction 0.0020 at 0.0098 of the 0.010 NLL cap;
  "no randomness advantage".
- `pcrl_bank_anchored_release_v1` and `pcrl_incumbent_view_risk_v1`: re-routing among existing codes, 0.002-0.006.
- `pcrl_method_portfolio_v1` B-JVIB: a jointly trained sampled latent. It released the latent, and it failed seed-0 occupation NLL.
- `pcrl_decision_preserving_compression_v1`: every finite code cost at least about 0.02 nats of occupation NLL.

**What is different here** (confirmed by the challenger as real operational differences):
- (a) Continuous noise on the learner's own logits, with no bank. The identity is the learner's own release, so there is no
  quantization cost.
- (b) The learner is trained through the actual channel from scratch.
- (c) Temperatures are fitted on emitted draws.

**Design choices** of one Gaussian output channel: the Fisher shaping, the 1.5 cap and the shared-eps coupling.

**Bottleneck addressed:** the quantization cost of bank and code channels, and the post-hoc miscalibration of noise added to a frozen
predictor.

## 3. Why it could improve protection without repeating the utility cost

**The cost:**
- The Jensen gap is real: zero-mean logit noise costs about 1/2 n' F n nats, and Brier pays sum_k Var(p~_k).
- With unit-Fisher-norm directions, every person pays s^2 / 2 nats, so only placement matters.

**Why placement can pay:**
- Deterministic releases lower a re-trained flexible reader's recovery mainly by merging people. Injective changes are inverted by
  readers trained on them, and merging pays refinement loss.
  - The challenger notes that the "invertible or merge" premise is not exhaustive. A deterministic remainder flattening (CTL_REM) is
    therefore run as a challenger control on FB_TWIN.
- Noise buys partial ambiguity continuously, at second-order cost.
- The reduction in linear discriminability per nat at small s is 2 a' F^-1 a, with a = Sigma_w^-1 delta. This is large exactly when
  SEX information lies in low-Fisher directions: log-odds among outcomes the predictor already considers unlikely.
- The SEX direction is chosen for SEX discriminability per unit Fisher cost. MINF spends the same cost where the task is cheapest,
  regardless of SEX.
- Training through the channel, together with temperatures fitted on emitted draws, removes the avoidable miscalibration part of
  post-hoc noise.

**When it is not favourable:**
- If the SEX information sits in task-dominant, high-Fisher directions, the reduction at s^2 / 2 <= 0.0072 nats is a few
  thousandths of AUC.
- Small logit noise leaves decisions almost unchanged. Expect any effect in smooth readers; decision and LOOKUP readers keep their
  recovery.
- A fit-time diagnostic is recorded per configuration: the in-sample one-draw linked Delta^2 on MODEL_FIT, emitted against identity.

**Expectation:**
- The twin is a U0-class neural release, with little NLL slack relative to U0.
- The 0.020 target against NEURAL_RAW (0.6245) and H_LJ (0.6183) is not expected to be met.

## 4. Task-only twin

- FB_TWIN uses the same architecture, initialisation, household split, permutations, training-noise stream (unused), early-stopping
  rule and CALIBRATION temperature procedure, under the identity channel. This is equal data and equal tuning effort.
- It never uses SEX.
- Capacity is the U0 shape at twice the width. The twin must pass the original limits: 24 clauses + 4 guards in both versions.
  Otherwise revision R1 applies.

## 5. Declared twin and simplest alternatives

- **Twin:** FB_TWIN.
- **Simplest alternatives (declared before the fit):**
  - **MINF (FB_MINF_S2, FB_MINF_S3):** the SEX-blind control path (G3).
    - Same coupled rank-1 construction, s, cap, R, key scheme and channel-aware training.
    - The direction is each person's minimum-Fisher eigenvector: untargeted noise of the same form.
  - **ISO (FB_ISO_S2):** the strong SEX-blind control. Full-rank, Fisher-isotropic, uncoupled, at matched second-order cost, with the
    same training.
  - **ID (FB_SEX_S2_ID):** the identity channel of the S2 candidate learner. It tests whether the trained predictor, rather than the
    noise, changed recovery.
  - **MEAN (FB_SEX_S2_MEAN):** the deterministic construction, the exact mean vector E[p~] of the S2 channel (48-node Gauss-Hermite).
    A drop for flexible readers would flag reader sensitivity, not protection.
  - **POSTHOC (FB_POSTHOC_S2):** the S2 SEX-targeted channel on the twin learner, with geometry from the twin and its own
    temperatures. It tests joint adaptation.
- **Challenger controls on FB_TWIN (not family-B configurations):**
  - CTL_REM: deterministic SEX-blind remainder flattening, with strength bracketing FB_SEX_S2's realized MODEL_SELECT spend.
  - Contingent: CTL_BLIND_OCC.
- **SEX-labelled exposure:** delta and Sigma_w are a closed-form LDA-type critic using MODEL_FIT SEX on in-sample logits.
  - It is counted as critic fits in the records: every epoch estimate of the sex learners, plus each frozen release geometry.
  - The SEX-blind channels (MINF, ISO) and the twin never read SEX in their channel. They share the training schedule.

## 6. Falsifying outcomes

- **F1 (targeting, restated):** the targeting rationale fails if, at matched realized MODEL_SELECT cost, the SEX path's linked
  recovery is not below the MINF path at S2 and S3. ISO is reported beside.
- **F2 (randomness):** FB_SEX_S2_ID or FB_SEX_S2_MEAN recovers no more than FB_SEX_S2. Randomization is then not the active
  ingredient.
- **F3 (joint adaptation):** FB_POSTHOC_S2 matches FB_SEX_S2 on both recovery and utility cost. Joint training then adds nothing.
- **F4 (reader mismatch):** the fit-time Delta^2 falls by at least 50% at S2, but the screen pair AUC falls by less than 0.003.
- **Margins:** frontier margins include Monte Carlo (R = 8), household and reader-seed variance.

## 7. Fitting plan

- **MODEL_FIT** (20,000 people / 13,337 households), operation 'fit' with this card:
  - Labels: income and occupation are the task losses. SEX is used only for the closed-form geometry of the sex modes, and through
    the channel-aware loss it reaches the networks of the sex learners.
  - Folds: a household-grouped 15% internal split (BLAKE2b of household id keyed by version) for early stopping only, on the summed
    emitted NLL with fixed draws.
  - The final refit is from the same initialisation on all 20,000 people for the selected epoch count. The scheme is non-nested.
  - Actual training size of the deployed parameters: 20,000 people / 13,337 households.
  - The release geometry uses in-sample logits on all of MODEL_FIT (non-nested, disclosed).
- **CALIBRATION** (2,982 people), operation 'calibrate': labels income and occupation, used only for two temperatures per
  configuration.
  - Golden-section search on log T in [log 0.5, log 3], with 2 coordinate sweeps.
  - Randomized configurations use the mean emitted NLL over 8 keyed calibration draws; MEAN uses its exact NLL.
- **MODEL_SELECT:** never read by family B. Feedback comes only from `nrm.evaluate` and `nrm.screen` outputs.
- **Seeds per version:**
  - initialisation 7100 + 10 v + task index (the same for all learners);
  - training streams 7200 + v (phase, epoch);
  - split keyed by v;
  - both versions fitted.
- **Training recipe:** Adam lr 1e-3, batch 512, at most 40 epochs, patience 5, minimum delta 1e-4. One thread, torch 2.10.0.
- **Predeclared strength path:** s in {S2 = 0.09, S3 = 0.12}, giving second-order linked cost {0.0041, 0.0072} nats.
  - The twin is the s = 0 anchor.
  - MINF uses the same s; ISO uses S2.
  - No other strengths, penalty weights, coupling variants or temperature tuning.
- **Queue:** one unit per version, through the shared queue:
  `python3 -m nrm.jobqueue run --unit FB_FIT_V<v> --stage fit --rss-gib 2 -- <repository>/.venv/bin/python -m nrm.fb.learner fit --version <v> --card results/pcrl_new_release_mechanisms_v1/cards/FB_FSC_v1.md`
  A retry needs new configuration ids, because the root-key file and the releases are write-once.

## 8. Readers needed

- Randomized configurations need one-draw channel-aware readers, as in section 1.
- Cell-to-instance map: cell (version v, repetition j, reader seed k) uses instance r = j for both seeds. This separates seed
  variation from Monte Carlo variation.
- Monte Carlo variation: refit one reader recipe per cell on instances j+2 and j+4, with the same split and seed, for FB_SEX_S2 and
  FB_MINF_S2 only.
- Report Monte Carlo variation separately from household-split and reader-seed variation.
- Deterministic configurations use the standard readers (R = 1).

## 9. Budget

- **Configurations:** 9 per version × 2 versions = 18, as allocated:
  - FB_TWIN (twin);
  - FB_SEX_S2 and FB_SEX_S3 (candidate);
  - FB_MINF_S2, FB_MINF_S3 and FB_ISO_S2 (control);
  - FB_SEX_S2_ID, FB_SEX_S2_MEAN and FB_POSTHOC_S2 (ablation).
- **Fits per version:** 6 learners × 2 model fits each (search + refit), 9 calibrator fits, and critic fits as counted above.
- **CPU** (torch, one thread):
  - Invented-data timing at the real size gives about 0.1 CPU-s per training epoch per learner.
  - Estimate: <= 4 CPU-min per version.
  - Request: 0.25 CPU-h for fits, including one retry.
- **Memory:** < 1 GiB (queue reservation 2 GiB).

## 10. Registered predictions and revision plan

**Predictions** (registered before any fit):
- P1: FB_TWIN passes all 24 clauses + 4 guards in both versions (p ≈ 0.7).
- P2: FB_SEX_S2 lowers pair AUC against FB_TWIN by 0.002-0.012 (central 0.006). FB_MINF_S2 and FB_ISO_S2 lower it by <= 0.004.
- P3: the SEX path lies below the MINF path at matched realized cost at S2 and S3 (p ≈ 0.5). MINF also places noise in low-Fisher
  directions, so the targeting margin may be small.
- P4: FB_SEX_S2_ID and FB_SEX_S2_MEAN are within 0.003 of FB_TWIN's recovery (p ≈ 0.75).
- P5: FB_POSTHOC_S2 is about equal to FB_SEX_S2 in recovery. Channel-aware training saves less than 0.001 nats (p ≈ 0.65).
- P6: the 0.020 target against NEURAL_RAW and H_LJ is NOT met (p > 0.9).

**Revisions** (each one a new card with recorded approval before its fit; none is pre-funded):
- **R1 (utility):** if FB_TWIN fails any original clause or guard in either version, make one capacity revision: d-256-256-64-K with
  Adam weight decay 1e-4, same channels, new configuration ids. If it still fails, the outcome is UTILITY_INFEASIBLE and the family
  closes.
- **R2 (strength):** if FB_SEX_S3 passes everything with at least 0.004 nats of occupation-NLL slack in both versions, add S4 = 0.15
  for SEX and MINF.
- **R3 (F4):** use decision-cell-conditional geometry: delta_c and Sigma_w,c per linked decision cell (10 cells), on S2 and S3 with
  the MINF match.
- **Stop:**
  - If F1 holds after R3, or F2 holds at any point, the family is CLOSED, with no strength, penalty or coupling tuning.
  - If only F3 holds, report that joint adaptation adds nothing.
