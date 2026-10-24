# Card FB_PFD_v1: pair Fisher-discriminant training of the U0 task networks (family FB)

> **SAR1 declarations (coordinator, from the adjudication of 2026-10-24; part of this card):** family FB; START = NEURAL_RAW;
> `cont_declared` = true (CONT = FB_PFD_TASK); `ctrl_origin` = START; CTRL = FB_PFD_DQ (SEX-blind dithered quantization of
> NEURAL_RAW), saved points FB_PFD_DQ_LO / FB_PFD_DQ_HI; `alt_declared` = true; ALT = FB_PFD_RAWJ (NEURAL_RAW's own RAW-J recipe
> continued to matched spend), saved points FB_PFD_RAWJ_LO / FB_PFD_RAWJ_HI; `selection_data` CTRL = MODEL_FIT_OOF, ALT = MODEL_FIT_OOF
> (spends over the cross-fitted RAW-J E2, NEURAL_RAW's out-of-fold equivalent); `start_in_sample_on_model_fit` = true; joint design not
> claimed; novelty not claimed; no mechanism-specific reader declared; at most 12 specifications. The investigator's text follows
> unchanged.


Draft by investigator B, 2026-10-24 (written 04:22Z-04:58Z; revised 04:58Z-05:10Z after the coordinator's adjudication "PROCEED, with
required changes", `date -u`), for the coordinator to commit as `results/pcrl_strong_anchor_release_v1/cards/FB_PFD_v1.md` before the
first fit it governs (never edited afterwards). Governs, in versions 0 and 1, CAND FB_PFD, CONT FB_PFD_TASK, CTRL FB_PFD_DQ_LO /
FB_PFD_DQ_HI and ALT FB_PFD_RAWJ_LO / FB_PFD_RAWJ_HI: at most 12 specifications (CAND 2 + CONT 2 + DQ 4 + RAWJ 4; a bracket point that
is the origin or coincides with the other side is not saved, item 7). Code: `sar/fb/pfd.py`, committed with this card
(`<repository>/.venv/bin/python -m sar.fb.pfd fit --version V --card results/pcrl_strong_anchor_release_v1/cards/FB_PFD_v1.md`,
run inside a queue unit; `selftest` and `selftest_stage` pass on invented data). Deployment: `sar.fb.predict` / `sar.fb.model_paths`
(numpy only; runs under python3). Rationale and evidence: `roles/investigator_b/PROPOSAL.md`.

1. **Operation.**
   - Runtime input: the admitted 145-column encoding X of the six permitted fields only. Released: income P_inc (n, 2) and occupation
     P_occ (n, 5), float64 softmax of the centred logits of two U0-architecture networks (d-64-ReLU-64-ReLU-16-K), each recipient its
     own vector and argmax decision; deterministic; computed by a float64 numpy forward pass of the saved float32 weights.
   - Start: U0_S<v> (`acs_replication` STAGE_A_<v>/U0_MODEL.pt; MODEL_FIT income / occupation labels, inherited). Inside cross-fitting
     fold k: the U0 recipe refitted from scratch on the fold's training rows (Recipe A: Adam 1e-3, batch 512, the admitted epoch counts
     7 / 14 (v0) and 9 / 13 (v1), Stage A's seed `trees.seed_for('ACSR1neural', v, task)`).
   - Training (MODEL_FIT income, occupation and SEX of the training people; SEX only in P): Adam lr 1e-4, betas (0.9, 0.999), eps 1e-8,
     no weight decay, gradient-norm clip 5 per network; per step one 256-row task batch per recipient from NEURAL_RAW's task stream
     (`seed_for('THP1task', v)`, 40 steps per epoch) and one 2,048-row privacy batch (`random.Random(seed_for('SAR1pfd-privacy', v))`);
     gradient of CE_inc + CE_occ + lambda P with
     P = sum over v in (income view, occupation view, pair) of [Phi(D_v / sqrt 2) - 1/2], D_v^2 = Delta_v' (S_v + eps_v I)^-1 Delta_v,
     phi_v = degree-2 polynomial features of the batch-standardized logit differences (income z1 - z0: 2; occupation z_k - z0, k = 1..4:
     14; pair: 20), Delta_v = women's minus men's mean of phi_v, S_v = pooled within-SEX covariance, eps_v = 1e-3 tr(S_v) / p_v.
     lambda = rho x median ||grad (CE_inc, CE_occ)|| / median ||grad P|| over 16 fixed draws at the start (`seed_for('SAR1pfd-calibration',
     v)`), one coefficient for both networks, rho = 1. Checkpoints E0..E10.
   - Released configurations: CAND FB_PFD = PFD at E_k* (item 6); CONT, CTRL and ALT (items 5, 7).
   - Pins: the fit records the SHA-256 of every `acs_replication` file it reads (STAGE_A_<v>/U0_MODEL.pt, STAGE_A_<v>/STAGE_A_SEED.json,
     NEURAL_RAW_<v>/NEURAL_RAW__rho1.0__E2.pt) in the fit record (`pins_sha256`), in every MODEL.pkl and in every release record.
2. **Closest previous implementation.** NEURAL_RAW (`acs_replication/neural_raw.py` with `leafcorr.Critics`, `leafcorr.objective`;
   protected-tree replication) and JOINT_TREE's pair critic (`acs_replication/leafcorr.py`). Substantive differences: the reader in the
   objective is the exact in-batch best response of a fixed quadratic-discriminant class (no online critics, no adversarial lag, no
   min-with-entropy clamp), full within-SEX covariance whitening (partial, incremental weights for a recipient's coordinates instead of
   the marginal weights of early-stopped, diagonally normalized critics), an AUC-scale coalition objective, and one coalition strength for
   both recipients instead of NEURAL_RAW's per-recipient lambda_J (income about 2x occupation). No routing, no post-processing of a
   frozen vector (P is invariant to temperature and bias moves), no noise, merging, code, input generalization, standardization,
   reweighting or controller.
3. **Measured failure addressed** (records in PROPOSAL.md section 1). NEURAL_RAW's pair is its occupation view (0.6253 = 0.6253), and its
   own stronger points fail income clauses first (`pcrl_acs_protected_tree_replication_v1/SELECTION_DETAIL.json`: rho 1 E5 fails only
   version-1 income Brier, max 1.059 of cap, with occupation NLL +0.0048 / +0.0045), while its income-view critics read almost nothing
   (logged risk(income) 0.000-0.0045) and its income network's weighted privacy gradient is 1.1-2.1x its task gradient (`LOGS.json`).
   The income log-odds is 68-74% predictable from the occupation log-ratios (label-free R^2), and along the path the income output moves
   as much as or more than the occupation output (`roles/investigator_b/check_release_geometry.json`).
4. **Why usefulness may be preserved.** The task loss is each recipient's full-vector cross-entropy from U0; the rule budgets all eight
   MODEL_FIT components (both within-SEX occupation guards included) at about 0.5-0.6 of the caps; NEURAL_RAW's rho 1 E5 occupation cost
   (+0.0048 / +0.0045 nats, Brier +0.0018 / +0.0013) fits inside the occupation allowances, and the operation is designed to spend less
   income utility than NEURAL_RAW's path at equal recovery. Label-free feasibility numbers (disclosed as such): the invented-data
   attribution check (whitened income-gradient share 0.040 vs marginal-weight share 0.460 when the income coordinate only proxies an
   occupation coordinate); D_pair >= D_view; affine invariance; no MODEL_SELECT number was computed.
5. **Twin (CONT, gating).** FB_PFD_TASK: the identical PFD loop, starts, task rows, optimizer, folds and checkpoints with lambda = 0 (SEX
   not read by the update), released at the candidate's E_k*: the equally exposed zero-protection continuation. Declared gating.
6. **Strength grid and selection rule** (MODEL_FIT only; no CALIBRATION label). Path E1..E10 at rho = 1. Out-of-fold releases on the
   famrun household folds `SAR_xf_FB_S<v>` (5 folds; each fold restarts from its own U0-recipe start; all three trained arms (PFD, TASK,
   RAWJ) on identical task rows). Paired spend of each checkpoint over the fold start (famkit.spend, household-clustered SE; accuracy as
   a loss; men's and women's occupation Brier as linearised subgroup means). k* = the largest E_k such that E_1..E_k all keep estimate +
   2 SE <= BUDGET on every component (famkit.select_strength): occupation NLL 0.0060, income NLL 0.0050, occupation Brier 0.0025, income
   Brier 0.0025, men's occupation Brier 0.0028, women's occupation Brier 0.0028, occupation accuracy 0.0060, income accuracy 0.0060. If
   E1 fails, status NO_FEASIBLE_STRENGTH: nothing is released and the family's initial comparison ends. The deployed CAND is the
   all-MODEL_FIT run from U0 at E_k* (status FITTED).
7. **Matched controls** (coordinator adjudication, 2026-10-24). Both bracket paths have origin START = NEURAL_RAW: out of fold, the
   cross-fitted RAWJ E2 (NEURAL_RAW's out-of-fold equivalent); deployed, NEURAL_RAW rho 1 E2. All bracket selections use MODEL_FIT
   out-of-fold total-NLL spends over START (selection data MODEL_FIT_OOF), saved before any MODEL_SELECT read. CAND's spend over START is
   its out-of-fold paired spend over the cross-fitted RAWJ E2 (value and household-clustered SE recorded).
   - **Bracket rule** (`sar.fb.pfd.bracket`, both paths): LO = the grid point with the largest spend over START that is <= CAND's; HI =
     the grid point with the smallest spend >= CAND's + 2 SE, SE = the household-clustered SE of that point's paired MODEL_FIT difference
     against CAND (ties: the first in grid order). If no point reaches it, HI = the largest-spend point with flag
     CONTROL_PATH_MAY_NOT_BRACKET. If no point is <= CAND's, LO is the origin and is not saved (flag LO_IS_ORIGIN); if LO and HI coincide,
     only HI is saved (flag LO_EQUALS_HI). If CAND's spend over START is <= 0, the origin is the bracket and no point of either path is
     saved (flag CTRL_BRACKET_IS_ORIGIN).
   - **CTRL = FB_PFD_DQ_LO / _HI** (the required untargeted control): SEX-blind subtractive-dithered quantization of NEURAL_RAW's release
     (`sar.ctlkit.dq_fit` / `dq_release`, steps `sar.ctlkit.DQ_GRID_FINE`, keyed dither tag `SAR_DQ_FB_S<v>` on the six compact fields).
     The basis is fitted on the cross-fitted RAWJ-E2 MODEL_FIT outputs; the deployed point is DQ applied to deployed NEURAL_RAW (its
     pinned E2 weights under this family's float64 forward law) with the same basis, step and keyed dither. SEX is never read.
   - **ALT = FB_PFD_RAWJ_LO / _HI** (the starting release's own protection operation continued to matched spend, for the optional gating
     ALT_FRONTIER): NEURAL_RAW's own recipe (online SEX critics, per-recipient lambda_J at rho 1, `run_arm('RAW_J')` step for step with a
     checkpoint every epoch) from the same starts, task rows, optimizer, folds and SEX exposure; grid E3..E10 (E2 is the origin and is not
     saved again).
   - **Recorded checks:** the all-MODEL_FIT RAWJ run's E2 weights equal `NEURAL_RAW__rho1.0__E2.pt` bitwise, and the U0 recipe on all
     MODEL_FIT reproduces U0 bitwise.
   - **Fit record** `results/pcrl_strong_anchor_release_v1/fb/FIT_PFD_V<v>.json`: CAND's config, strength k*, status and spend over START
     (value, SE); for CTRL and ALT the origin, the selection data (MODEL_FIT_OOF), every grid point's spend over START and the SE of its
     paired difference against CAND, the saved LO / HI config ids with their spends, and the flags; CONT's config; the pins.
   - **Not built, with reasons:** SBC (the same PFD training with the PSEUDO2 SEX-independent attribute in place of SEX) and RO (the same
     training with SEX replaced by a release-only surrogate, P(SEX | NEURAL_RAW's released vectors)). Both are constructible but are not
     part of the initial comparison: each would add up to 4 specifications and 15 trained runs per version, and the adjudicated
     untargeted control is CTRL (SEX-blind DQ of the starting release), with CONT as the protection-off ablation. A generic release-only
     tilt of NEURAL_RAW is not used as a stand-in, because it is the closed release-only-tilt family itself. Joint design: not claimed
     (D5 NOT_APPLICABLE_NOT_DECLARED). The ablation of the strongest feasible release is CONT (protection off).
8. **Revisions** (at most two; triggers read MODEL_FIT records only; one fit each, same rule and budgets, then stop; each with its own
   CONT and CTRL bracket, at most 2 + 2 + 4 = 8 specifications; the ALT bracket is re-selected on the already trained RAWJ path by the
   item 7 rule and only points not already saved are saved):
   - R1 (best-response evasion): in both versions the candidate's MODEL_FIT HGB15 pair AUC falls (vs the fold start) by less than half of
     its LOG_LR pair drop. Change: degree-3 features (income 3, occupation 34, pair 55) with a 4,096-row privacy batch.
   - R2 (budget never reached): k* = E10 in both versions with every component below 70% of its budget at E10. Change: rho = 3.
9. **Registered predictions, configuration count, CPU.** Predictions as in PROPOSAL.md section 9 (registered 04:58Z, copied here for
   the record): bitwise checks hold 0.85; k* in E2..E6 in both versions 0.60; MF binding component on the occupation side 0.65 (men's
   guard 0.45); candidate's income share of the NLL spend below the control's 0.70; all 28 conditions at the MS point 0.75; candidate
   pair median 0.610 (80% interval 0.592-0.628); benefit vs NEURAL_RAW > 0 at the point 0.75, resolved 0.40; vs the matched control > 0
   at the point 0.65, resolved (D2) 0.15; vs H_LJ > 0 at the point 0.55; D4 material target 0.04; income view <= NEURAL_RAW's + 0.010
   in both versions 0.70; pair - occupation view <= 0.010 0.75; continuation's pair within 0.005 of U0's 0.80; most likely outcome
   (0.45): feasible, pair 0.600-0.618, point benefit +0.007 to +0.025 over NEURAL_RAW, not separated from its matched control.

   **Dated note, 2026-10-24 05:04Z (after the adjudication; the numbers above are unchanged).** "The matched control" in the predictions
   above is now the ALT bracket: the income-share prediction is read against both saved ALT points, and "vs the matched control" against
   the ALT HI point (and LO where saved). Added predictions for the new controls: the CTRL (DQ) path brackets CAND without
   CONTROL_PATH_MAY_NOT_BRACKET in both versions 0.80; ALT brackets CAND inside E3..E10 in both versions 0.75; CTRL_BRACKET_IS_ORIGIN
   (CAND spends no more than NEURAL_RAW over START) in either version 0.10; CAND's pair below the DQ_HI point at the point 0.75, resolved
   (both versions > 0, mean - 2 se_agg > 0) 0.30; CAND separated from every gating control (CTRL, CONT and, if gating, ALT) 0.12.

   Configurations: at most 12 (CAND 2 + CONT 2 + DQ 4 + RAWJ 4). CPU: about 4-6 CPU-min per version (upper estimate 12: three trained
   arms on 5 folds plus the deployed fits, the 40-step DQ path on MODEL_FIT, releases for six roles, MODEL_FIT readers on up to 8 views),
   one thread, torch venv `<repository>/.venv/bin/python`; deployment checks run under python3.
10. **Novelty status.** Not claimed. Relatives: adversarial fairness penalties with online critics (NEURAL_RAW and its sources),
    discrepancy penalties (MMD, HSIC), Fisher / Mahalanobis separability measures, closed-form linear concept erasure (LEACE). No
    literature search was run.
