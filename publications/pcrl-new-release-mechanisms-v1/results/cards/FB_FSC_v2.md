# Family B mechanism card v2 (revision FB2): the post-hoc SEX-targeted coupled logit channel on a frozen task-only base

Designer B, 2026-10-20. This card governs every FB2 fit. Inputs: the coordinator's proposal (`roles/coordinator/REVISION_FB_R1_PROPOSAL.md`, 15:31Z)
and instructions (15:36Z, corrected 15:39Z), and the statistical reviewer's FINAL ruling (`roles/statistical_reviewer/RULING_FB_R1_FINAL.md`,
C1-C7, which supersedes `RULING_FB_R1.md` where they differ). Code: `nrm/fb/posthoc.py` (with `nrm/fb/release.py`, `nrm/fb/channel.py`) at
commit 0a8b2de9d. Invented-data selftest: 21/21 under python3 and under the torch venv. The card was committed before any FB2 fit and before
any FB2 read.

**Label attached to every FB2 result:** "FB2 was designed after family-B screening on MODEL_SELECT (targeting signal, F3); it is
evaluated on the same people; development evidence only."

**Scope sentence:** one released instance per person; repeated-query protection not claimed.

Recovery means the declared development readers' AUC for the recorded SEX column. Lower recovery for these readers is never protection
against every reader.

## 0. Revision record

**Evidence for the revision** (MODEL_SELECT, through `nrm.evaluate` / `nrm.screen`; adaptive development evidence):
1. **P1 is false: the card-v1 task-only learner fails the original limits.**
   - FB_TWIN v1 fails the men's occupation Brier guard (0.00854 > 0.005) and has an occupation NLL excess of +0.0067 vs U0.
   - FB_TWIN v0 passes only at the edge (men's guard 0.00460).
   - Every FB_* configuration fails v1 (G1 = TASK_INFEASIBLE for all 18 FB_*).
2. **F3 holds in screening: joint adaptation adds nothing.** Pair AUC (max of 4 recipes, screening halves): FB_POSTHOC_S2
   0.6292 / 0.6383 against FB_SEX_S2 0.6325 / 0.6391.
3. **A targeting signal at matched second-order cost (screening point values, no uncertainty):**
   - pair AUC: SEX_S2 0.6325 / 0.6391 against MINF_S2 0.6389 / 0.6413 and ISO_S2 0.6391 / 0.6413;
   - pair AUC: SEX_S3 0.6284 / 0.6361 against MINF_S3 0.6391 / 0.6398;
   - ID 0.6403 / 0.6454 and MEAN 0.6410 / 0.6457 are about equal to the twin;
   - realized total-NLL excess at S2: SEX 0.0081 / 0.0103 against MINF 0.0093 / 0.0111; at S3: 0.0122 / 0.0153 against
     0.0128 / 0.0151.
   - F2 and F1 of card v1 were not triggered.

**Protocol deviation (C1):**
- Card v1 registered R1 as "a wider MLP, otherwise UTILITY_INFEASIBLE and close".
- It is replaced, after MODEL_SELECT outcomes, by this revision: a stronger frozen base, with channel training dropped.
- This is recorded in PROTOCOL_DEVIATIONS.md by the coordinator. The FB_* results stay, are reported as family-B v1 outcomes, are never
  pooled with FB2 and never act as FB2 matched controls.

**Changes from card v1:**
- (a) **Base.** The frozen task-only base TWA_v (below) replaces the trained networks.
- (b) **No training through the channel.** The channel law is applied post hoc; F3 retired that ingredient.
- (c) **Strengths and controls.** Strengths {0.12, 0.21}. The PCA and ISO controls are not used. MINF is the G3 path, and CTL_REM is a
  second G3 path.
- (d) **Geometry.** δ and Σ_w are estimated on out-of-fold occupation logits (recommended by the reviewer; see section 1).
- (e) **FB2_BASE** is the frozen base emitted unchanged.
- No claim of joint training, joint design or coupling.

## 1. Operation

**Base (frozen; never refitted here): TWA_v.**
- **Income:** HC_STRONG S<v>, computed by `ihs.fam_h.predict(v, 'HC_STRONG', X, names)['income']`. This is the income-headroom campaign's
  task-only strong-income twin.
  - Per its manifest (RELEASE.json `roles_read`), it read MODEL_FIT income (Step-1 HGB fit and CV) and **CALIBRATION income (stack,
    budget cross-fitting, Platt map)**.
  - Its predict also reads stored T_TREE / LOCAL_TREE / JOINT_TREE parameters. The income offset is 0, and the JOINT_TREE correction
    (replication, MODEL_FIT SEX) does not enter its income.
- **Occupation:** A1_PO_16 S<v>, computed by `ohn.fam_a.predict(v, 'A1_PO_16', X, names)`. HGB on the POBP-ordinal coding, fitted on all
  MODEL_FIT occupation.
- **Gate before anything is written:** the fit computes the base from features for all six roles and requires bitwise equality with ohn's
  hash-checked stored releases (`ohn.devrun.anchor_income('TW', v, role)`, `ohn.devrun.load_occupation('A', 'A1_PO_16', v)`).
- Recorded utility of the base (earlier campaigns, MODEL_SELECT):
  - income NLL excess vs U0 −0.0084 / −0.0084;
  - occupation NLL excess vs U0 −0.0161 / −0.0128;
  - men's guard −0.0039 / −0.0023.

**Coordinates.**
- z = [H2 log p_inc ; H5 log p_occ] in R^5 (centred log-probabilities, Helmert rows).
- F_i is the per-person Fisher of the summed task log loss, computed from the deployed base probabilities, so it needs no label.

**Channel law** (person i, release instance r; card v1's law, not trained through):
- **'sex' (candidates FB2_SEX_S3, FB2_SEX_S5):**
  - d_i = (F_i + s² Σ_w⁻¹)⁻¹ Σ_w⁻¹ δ; v_i = d_i / sqrt(d_iᵀ F_i d_i).
  - δ = mean z(women) − mean z(men) and Σ_w = pooled within-SEX covariance, from MODEL_FIT SEX, computed on geometry logits:
    - occupation: **A1_PO_16's admitted out-of-fold MODEL_FIT bank**, CROSSFIT.npz `MODEL_FIT__oof` (5-fold household cross-fitting,
      hash-checked by ohn's loader; its decision agreement with the deployed A1 is recorded);
    - income: **HC_STRONG in-sample**, because HC_STRONG has no admitted out-of-fold MODEL_FIT bank (its STEP1.json holds CV
      metrics only). Disclosed.
- **'minf' (G3 control path FB2_MINF_S3, FB2_MINF_S5; contingent FB2_MINF_S6):**
  - v_i = the eigenvector of F_i with the smallest eigenvalue, divided by sqrt(λ_min).
  - Sign convention: the largest-|.| component is positive.
  - SEX-blind. It generically lies in one recipient's block.
- **Both modes:**
  - c_i = min(s, 1.5 / |v_i|);
  - z̃_i = z_i / T + c_i ε_{i,r} v_i, where ε_{i,r} ~ N(0, 1) is ONE scalar per person per instance;
  - emitted vectors are the softmax of each block; decision = argmax of each emitted vector;
  - each recipient receives only its own vector and decision;
  - second-order linked cost s²/2 nats per person (lower where the cap binds);
  - per-task spend is read beside the linked cost.
- **Temperatures:** per task, fitted on CALIBRATION on the emitted draws (golden section on log T in [log 0.5, log 3], two sweeps, 8 keyed
  calibration draws with common random numbers across configurations).
  - Used by the noisy channels only, matched between SEX and MINF.
  - This is a **second use of CALIBRATION income** after HC_STRONG's stack and Platt map. Disclosed.
- **FB2_BASE (twin):** the frozen base emitted unchanged.
  - No noise, no temperature, no CALIBRATION read, no log/softmax round trip.
  - Its income view is an exact alias of TWJ's income, and its occupation view of W's occupation (C3; recorded once).
- **Mean-vector construction (decisive phase only):** the exact E_ε[emitted] (48-node Gauss-Hermite) of one FB2_SEX channel. Deterministic.

**Noise coupling.**
- SEX and MINF use the same coupled rank-1 construction, so their comparison isolates targeting.
- **No coupling or joint-design claim.**

**Seed handling (frozen).**
- ε_{i,r} = Box-Muller of keyed BLAKE2b, with:
  - key = BLAKE2b(key = root, message = "NRM-FB|<config>|V<v>");
  - message = "NRM-FB|<role>|r<r>|<row_id>".
- There is one private 32-byte root per (revision, version) (ENG1-08):
  - it is `runs/FB/MODELS/FB2/V<v>/RELEASE_ROOT.bin`, created from `os.urandom` at the first FB2 fit of version v, mode 0600;
  - contingent FB2 configurations reuse their version's root;
  - draws depend only on (key, role, r, row_id).
- **Custody:** the root is never published or committed. With it, the draws can be regenerated and the channel inverted. Replay happens
  only from the private tree or its backup. The independent verifier may replay from the private tree.
- **Deployed randomness for people outside the admitted rows:** per-request keyed draws from a service CSPRNG key, used as the root, with
  a per-request identifier as row_id. One released instance per person. Repeated-query protection is not claimed.

**R = 8.**
- Each randomized configuration stores 8 independent one-draw-per-person instances as (8, n, k) stacks.
- Utility is computed on each emitted vector and its argmax, averaged over the 8 instances. Monte Carlo SE is reported separately.

**Instance map (C2, as declared by the reviewer):**
- R1 → 0, R2 → 1, R3 → 6, R4 → 7, with the reader seeds k2 and k3 using their block's instance.
- Monte Carlo replicate members at offsets +2 and +4, run through the full 16-recipe procedure.
  - Required for FB2_SEX_S5 and FB2_MINF_S5.
  - Declared for all four randomized FB2 configurations (FB2_SEX_S3, FB2_SEX_S5, FB2_MINF_S3, FB2_MINF_S5).
- **Index arithmetic, for the coordinator's reader code:**
  - With R = 8, the +2 / +4 members of the R1 and R2 blocks are instances {2, 4} and {3, 5}. Together with the main instances {0, 1, 6, 7}
    they use each of the 8 instances exactly once.
  - The R3 and R4 blocks (6, 7) have no in-range +2 / +4 members.
  - The committed `nrm.devreaders` REP_INDEX (R3 → 2, R4 → 3) and its range check must follow this map before the FB2 grid.

**Reader sampling access.**
- Readers train and score under the same channel law, with exactly one draw per person from the cell's instance. This holds for auxiliary
  people (reader FIT and VALIDATION) and for SCORE people.
- No multi-draw access. Linked readers pool both recipients' draws of the same instance.

**Deployment.**
- `nrm.fb.release.predict(config_id, version, X, feature_names, draw=0, keys=None)` dispatches FB2_* by revision to
  `nrm.fb.posthoc.predict2`, which runs on numpy / scikit-learn under python3.
- Randomized configurations need out-of-band keys {'role', 'row_id'}. Deterministic configurations refuse keys and any draw other than 0.
- With the role's own keys and draw 0, it reproduces RELEASE.npz draw 0 bitwise.
- `model_paths` lists:
  - `CHANNEL_<config>.json` (with the revision tag);
  - the root key, for randomized configurations only;
  - **the base sources:** HC_STRONG's STATE / RELEASE.json / RELEASE.npz and Step-1 files, the admitted stage files read by
    `ihs.fam_h.predict`, the A1_PO_16 run directory (OCC.npz / OCC.json / CROSSFIT.npz / model files), and the two governance
    admissions.
- These base files lie outside this campaign's runs folder, so `nrm.deploy.build` must accept them (it currently keys dependencies
  relative to `runs/`). `CODE_MODULES` pins `ihs.fam_h`, `ihs.bases`, `ihs.common`, `ohn.fam_a`, `ohn.common`, `ohn.devrun` and
  `acs_replication.trees`.

## 2. Closest prior PCRL attempts; what is different; which bottleneck

Without channel training, FB2 is post-hoc noise on a frozen predictor. Its closest prior work is now:

**The closed ACS-2018 stochastic-channel and replacement-channel studies** (local git branches):
- `research/pcrl-stochastic-channel-v1` and `research/pcrl-stochastic-replacement-overnight-v1`. Both closed before any stochastic
  constraint ran. The replacement coder's quantization cost was +0.021 nats.
- `research/pcrl-adaptive-release-v1`: the stochastic Q showed no advantage. The privacy-first channel's residence loss (+0.0095) broke
  its +0.001 guard.
- A synthetic certified model on those branches found zero randomization advantage under local-only constraints, and a positive one only
  under coalition constraints.

**The frozen-bank channels:**
- `pcrl_latent_and_sampled_methods_v1` score channel A: a frozen T* code bank. Its identity action alone exceeded the FIT occupation U0
  NLL cap (+0.0043 / +0.0054). 14 of 16 searches were infeasible, and it was never attacked.
- `pcrl_conditional_budgeted_release_v1`: LP mixtures toward donors. Pair reduction 0.0020 at 0.0098 of the NLL cap; "no randomness
  advantage".
- `pcrl_bank_anchored_release_v1` and `pcrl_incumbent_view_risk_v1`: re-routing among codes, about 0.002-0.006.

**Card v1 (FB_FSC_v1):** the same channel trained through; screening above.

**What is different:**
- Continuous noise on the frozen base's own log-probabilities, with no code bank and no replacement coder. The identity is the base
  itself, so there is no quantization cost.
- The noise direction is the Fisher-optimal linear-Gaussian SEX direction.
- The base has 0.013-0.026 nats of measured slack, where earlier channels had none.

**Bottleneck addressed:**
- the quantization / bank cost of earlier stochastic channels;
- the missing task slack of the card-v1 learner.

## 3. Why it could improve protection without repeating the utility cost

**The cost:**
- The Jensen gap is real: noise costs about ½ nᵀFn nats, and Brier pays Σ Var.
- With unit-Fisher-norm directions every person pays s²/2 (0.0072 / 0.0221 nats at S3 / S5, linked), which is within the base's slack.

**Why the targeting can pay:**
- The SEX direction chooses where to spend: maximal linear SEX discriminability removed per unit Fisher cost.
- MINF spends the same second-order cost where the task is cheapest, regardless of SEX. CTL_REM spends it deterministically and
  SEX-blind.
- Screening of card v1 suggests SEX < MINF at matched cost. FB2 tests this on a base that can afford the cost.

**When it is not favourable:**
- If readers use decision-level or nonlinear structure that small logit noise does not touch.
- If SEX information sits in high-Fisher directions.
- The 0.020 material target against NEURAL_RAW / H_LJ is not expected.

## 4. Twin

- **FB2_BASE** = TWA_v emitted unchanged.
- **C3:** it must pass all 24 clauses + 4 guards in both versions through `nrm.evaluate` on hash-checked wires **before any protected FB2
  release is read on MODEL_SELECT**. If it fails, stop before reading any protected FB2 release.

## 5. Declared alternatives and paths (G2, G3)

- **G2 matched alternative per candidate:**
  - FB2_MINF_S3 for FB2_SEX_S3;
  - FB2_MINF_S5 for FB2_SEX_S5.
- **FB2_MINF_S3 / FB2_MINF_S5 / FB2_MINF_S6** are the G3 control path only: NO_MATCHED_CONTROLS and **not selectable** (C4).
- **G3 SEX-blind paths:**
  - {FB2_BASE, FB2_MINF_S3, FB2_MINF_S5} (plus FB2_MINF_S6 if fitted);
  - **the CTL_REM path on FB2_BASE (C7):** deterministic SEX-blind remainder flattening. It is funded now at up to 2 levels × 2 versions =
    4 configurations, chosen by measured MODEL_SELECT spend only to bracket FB2_SEX_S3 and FB2_SEX_S5. It is fitted by the challenger
    under a CTL v2 card committed before that fit, and never on FB_TWIN.
- **Strengths:** fixed {0.12, 0.21} for SEX and MINF, with no other strength.
- **Contingent MINF bracketing level FB2_MINF_S6 (s = 0.26, +2):**
  - It is chosen by spend only.
  - It is fitted only if `nrm.rules.frontier_comparison` returns CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT for an FB2_SEX point.
  - No extension after the fact.
- **G6** = NOT_ASSESSED_MISSING_CONTROLS, with `outputs_changed` set mechanically. A coupled channel on both outputs has no separate design
  and no sequential orders.
- **G7** = NOT_CLAIMED.

## 6. Falsification

- **F1 (C6):** F1 holds unless `frontier_comparison` gives **DOMINATES_CONTROL_FRONTIER at BOTH S3 and S5** against the declared paths.
  - If F1 holds, the family closes.
  - An INCOMPLETE frontier record (missing paired costs or grid entries) must be completed. It is never read as F1 and never as a pass.
- **F2 (decisive phase):** the deterministic MEAN construction reaches SEX's recovery.
- **MEAN (C5, +2):**
  - It is fitted at the highest FB2_SEX strength that is TASK_FEASIBLE in both versions AND DOMINATES_CONTROL_FRONTIER against both
    declared paths.
  - If no strength qualifies, MEAN is not fitted.
- **Utility:** an FB2 configuration that fails the original limits is reported and never claimed.

## 7. Fitting plan

- **MODEL_FIT** (20,000 people / 13,337 households), operation 'fit' with this card: **SEX only**, for the closed-form geometry of the sex
  configurations.
  - This is SEX-labelled exposure: an LDA-type critic on out-of-fold occupation logits and in-sample income logits.
  - No MODEL_FIT task label is read, because the base is frozen.
- **CALIBRATION** (2,982 people), operation 'calibrate': income and occupation, for the two temperatures of each noisy configuration.
  FB2_BASE reads none.
- **MODEL_SELECT:** never read by family B. Feedback comes from `nrm.evaluate` / `nrm.screen` outputs only.
- **No model is trained.**
  - Training size of this revision's own parameters: geometry on MODEL_FIT (20,000), temperatures on CALIBRATION (2,982).
  - The base was trained by its source campaigns: A1_PO_16 on all MODEL_FIT; HC_STRONG on MODEL_FIT + CALIBRATION.
- **Version v:** base S<v>, root and keys per (FB2, v). Both versions are fitted.
- **Queue:** one unit per version:
  `python3 -m nrm.jobqueue run --unit FB2_FIT_V<v> --stage fit --rss-gib 2 -- python3 -m nrm.fb.posthoc fit --version <v> --card results/pcrl_new_release_mechanisms_v1/cards/FB_FSC_v2.md`
  This fits the 5 firm configurations. Contingent ones use `--configs`, under the conditions above.

## 8. Readers needed

- Randomized configurations: one-draw channel-aware readers, under the instance map and Monte Carlo members of section 1.
- FB2_BASE: standard readers (R = 1).

## 9. Budget

**Configurations:**
- Firm: 5 per version × 2 = 10 (FB2_BASE, FB2_SEX_S3, FB2_SEX_S5, FB2_MINF_S3, FB2_MINF_S5).
- Contingent: FB2_MINF_S6 +2; MEAN +2.
- CTL_REM on FB2_BASE: up to 4, counted under the challenger's CTL v2 card.

**CPU:**
- Base computation for six roles plus emission and temperature fits: about 1 CPU-min per version (no training).
- Request: 0.1 CPU-h, including one retry.

## 10. Registered predictions (before any FB2 fit) and stop rules

- **P1b:** FB2_BASE passes all 24 + 4 in both versions (p ≈ 0.9; aliases of TWJ income and W occupation).
- **P2b:** FB2_SEX_S3 / FB2_SEX_S5 lower the pair AUC against FB2_BASE by about 0.006 / 0.015 (descriptive central values; base about
  0.645-0.65).
- **P3b:** SEX DOMINATES_CONTROL_FRONTIER against the MINF path at both S3 and S5 (p ≈ 0.45). Against the CTL_REM path as well
  (p ≈ 0.35).
- **P4b:** all FB2 noisy configurations at S5 remain TASK_FEASIBLE in both versions (p ≈ 0.6). The men's guard and the occupation NLL
  bind first.
- **P5b:** the 0.020 material target against NEURAL_RAW (0.6245) and H_LJ (0.6183) is NOT met (p > 0.9).

**Stop rules:**
- If FB2_BASE fails C3: stop, and read no protected release.
- If F1 holds: the family closes, with no strength, coupling or penalty tuning.
- Contingent fits occur only under their declared conditions.
