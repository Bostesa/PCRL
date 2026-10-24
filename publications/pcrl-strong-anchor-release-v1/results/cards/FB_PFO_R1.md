# Card FB_PFO_R1: the coalition objective carried by the occupation network alone, income frozen at START (family FB, revision R1)

> **SAR1 declarations (coordinator; part of this card):** family FB, revision 1 (reading FB_R1); START = NEURAL_RAW;
> `cont_declared` = true (CONT = FB_PFO_TASK); `ctrl_origin` = START; CTRL = FB_PFO_DQ (SEX-blind dithered quantization of NEURAL_RAW's
> occupation output, income passed through), saved point FB_PFO_DQ_HI only (LO recorded, not saved: specification budget; the origin
> serves as the lower point, fineness COARSE_BELOW expected); `alt_declared` = true; ALT = FB_PFO_RAWJ (NEURAL_RAW's own RAW-J recipe
> continued, 10-step checkpoints), saved point FB_PFO_RAWJ_HI only (an exact alias of an already saved FB_PFD_RAWJ point is reused, not
> saved again); `selection_data` CTRL = MODEL_FIT_OOF, ALT = MODEL_FIT_OOF; `start_in_sample_on_model_fit` = true; joint design not
> claimed; novelty not claimed; no mechanism-specific reader declared; at most 8 new specifications.
> **Adaptive status:** designed from FB_PFD_v1's MODEL_FIT records only; no MODEL_SELECT outcome of FA or FB had been read. The only
> MODEL_SELECT reads before this card were the reference utility and the reference reader grid units (05:13Z onward), whose outputs no
> one had opened; the mechanical rule (first MODEL_SELECT read before the card's commit) nevertheless tags this reading ADAPTIVE_, and
> that tag is kept. The investigator's text follows unchanged.


Draft by investigator B, 2026-10-24 (written 05:15Z-05:35Z, `date -u`), for the coordinator to commit as
`results/pcrl_strong_anchor_release_v1/cards/FB_PFO_R1.md` before the first fit it governs (never edited afterwards).
- **Basis:** designed from the registered FB_PFD_v1 MODEL_FIT records only. No MODEL_SELECT outcome of FB exists, so the reading is
  REVISED_, not ADAPTIVE_.
- **Code:** `sar/fb/pfo.py`, a new module. FB_PFD_v1's committed code path `sar/fb/pfd.py` is unchanged and is imported read-only for its
  objective functions and helpers. `sar/fb/__init__.py` dispatches FB_PFO* configurations to `sar.fb.pfo.predict`; FB_PFD* configurations
  keep `sar.fb.pfd.predict`.
- **Command:** `<repository>/.venv/bin/python -m sar.fb.pfo fit --version V --card results/pcrl_strong_anchor_release_v1/cards/FB_PFO_R1.md`,
  run inside a queue unit. `selftest` and `selftest_stage` pass on invented data.

**SAR1 declarations (proposed for the coordinator):**
- family FB, revision R1, label prefix REVISED_; START = NEURAL_RAW;
- `cont_declared` = true (CONT = FB_PFO_TASK); `ctrl_origin` = START; CTRL = FB_PFO_DQ (SEX-blind dithered quantization of NEURAL_RAW's
  occupation output, income passed through bitwise), saved point FB_PFO_DQ_HI;
- `alt_declared` = true; ALT = NEURAL_RAW's own RAW-J recipe continued, re-selected on that path, saved point FB_PFO_RAWJ_HI or an alias
  of an already saved FB_PFD_RAWJ point;
- `selection_data` CTRL = MODEL_FIT_OOF, ALT = MODEL_FIT_OOF (spends over the cross-fitted RAW-J E2, NEURAL_RAW's out-of-fold
  equivalent); `start_in_sample_on_model_fit` = true;
- joint design not claimed; novelty not claimed; no mechanism-specific reader; at most 8 new specifications.

1. **Operation.**
   - **Runtime input and release.** The admitted 145-column encoding X of the six permitted fields only. Income P_inc (n, 2) and
     occupation P_occ (n, 5), each the float64 softmax of the centred logits of a U0-architecture network; each recipient gets its own
     vector and argmax decision; deterministic.
   - **Income (not trained).** START's released income vector: NEURAL_RAW rho 1 E2's income network, bitwise
     (`NEURAL_RAW_<v>/NEURAL_RAW__rho1.0__E2.pt`). Inside cross-fitting fold k it is the fold's cross-fitted RAW-J E2 income network.
     CAND's and CONT's income vectors equal START's bitwise; the fit asserts this on every fold and checkpoint and records the check.
   - **Occupation (trained).** Starts from START's occupation network (deployed: NEURAL_RAW E2; fold k: the fold's RAW-J E2). It is trained
     on MODEL_FIT occupation labels, with SEX only inside P_O, using
     `CE_occ + lambda * P_O`, where `P_O = [Phi(D_occ / sqrt 2) - 1/2] + [Phi(D_pair / sqrt 2) - 1/2]`.
     - D_occ and D_pair are FB_PFD_v1's closed-form whitened quadratic best response (degree-2 polynomial features of the
       batch-standardized logit differences, pooled within-SEX covariance, ridge 1e-3 tr/p).
     - D_pair is computed on the pair (frozen income logit difference as released, a constant input; trainable occupation logit
       differences). The income-view term is dropped because it is constant.
     - lambda = rho x median ||grad CE_occ|| / median ||grad P_O|| over 16 fixed draws at START (`seed_for('SAR1pfo-calibration', v)`),
       occupation network only, rho = 1.
     - Optimizer: Adam 1e-4, betas (0.9, 0.999), eps 1e-8, no weight decay, gradient-norm clip 5.
     - Batches: one 256-row task batch per step from NEURAL_RAW's task stream (`seed_for('THP1task', v)`); one 2,048-row privacy batch
       per step (`random.Random(seed_for('SAR1pfo-privacy', v))`).
   - **Path.** Checkpoints every 10 steps, S0..S400.
   - **Released configurations.** CAND FB_PFO = the occupation network at S_k* with START's income network (item 6). CONT, CTRL and ALT
     are in items 5 and 7.
   - **Pins.** The SHA-256 of every `acs_replication` file read (STAGE_A_<v>/U0_MODEL.pt, STAGE_A_<v>/STAGE_A_SEED.json,
     NEURAL_RAW_<v>/NEURAL_RAW__rho1.0__E2.pt) is recorded in the fit record, every MODEL.pkl and every release record.
2. **Closest previous implementations and the structural change.**
   - **FB_PFD_v1 (`sar/fb/pfd.py`).** It used the same objective functions, but both networks were trained from U0, the income-view term
     was included, and the checkpoints were per epoch. R1:
     - removes the income recipient from training (its output is START's, bitwise);
     - makes the occupation network carry the coalition term, with the released income as a fixed input of D_pair;
     - starts from START rather than U0, so CAND at S0 is START;
     - uses a 10-step path.
   - **The tree arm's sequential schedules S12 / S21 (`acs_replication/leafcorr.py`).** One recipient is frozen while the other trains,
     but these are leaf-score corrections with online critics (closed). R1 is neither: no routing, no critics.
   - **IA_OSG (input-and-pair campaign).** Its occupation-only change was an input generalization with the twin's income, and its pair
     did not follow the occupation view. R1's occupation network is trained against the pair formed with the deployed income vector
     itself, so the pair term already sees the partner it ships with.
   - **Not closed families.** R1 is no scalar leaf correction, post-processing of a frozen vector (P_O is invariant to temperature and
     bias moves), noise, merge, code, input generalization, standardization, reweighting or controller.
3. **Measured failure addressed** (MODEL_FIT records only: `results/pcrl_strong_anchor_release_v1/fb/FIT_PFD_V0.json` / `_V1.json`,
   extracted by `roles/investigator_b/r1_diagnosis_extract.py`; spends over the fold start, estimate (SE)).
   - **The binding condition is income utility at the first checkpoints. The occupation side stayed inside its budget.**
     - v0: k* = E1. E2 fails income Brier +0.0053 (0.0009) and income NLL +0.0066 (0.0011). At E2 occupation NLL is +0.0024 (0.0007),
       the men's guard +0.0000 (0.0004) and the women's +0.0013 (0.0004), all within budget at +2 SE.
     - v1: NO_FEASIBLE_STRENGTH. E1 fails income Brier +0.0026 (0.0006), while occupation NLL is -0.0006, the men's guard -0.0013 and
       the women's +0.0001. At E2: income Brier +0.0091, income NLL +0.0112, occupation NLL +0.0017.
   - **The income network absorbed the objective.**
     - The income share of CAND's NLL spend is 1.02 over the fold start (4.60 over START), against 0.40 (ALT HI) and 0.49-0.57 (CTRL)
       for the controls. Contradiction 1 of FB_PFD_v1 holds, and its attribution hypothesis is withdrawn.
     - In the deployed v0 run's logged privacy batches over the first 40 steps:

       | In-batch statistic | Step 0 | Step 40 | Change |
       |---|---|---|---|
       | Income-view D² | 0.0418 | 0.0049 | -88% |
       | Occupation-view D² | 0.2863 | 0.2616 | -9% |
       | Pair D² | 0.3486 | 0.3319 | -5% |

       Both networks received a privacy push equal to their task gradient (step 0: income 0.332 vs 0.337; occupation 0.369 vs 0.373).
     - **Reading (hypotheses, not results).** (a) In AUC units the view term's gradient per unit D² scales as 1/D, so the near-chance
       income view (D about 0.2) is weighted about 3x the pair (D about 0.59). (b) The one-coordinate income output is cheap to move in D
       but costly in income utility. (c) The coordinator's alternative: the pair's incremental SEX signal sits in the income coordinate
       given occupation, so the exact best response targets income. Under every reading, any coalition objective that can move the
       income network spends income first. R1 does not let it move.
   - **The first checkpoint was too coarse.** The descriptive MODEL_FIT readers (v0) put CAND at E1 at START's recovery: LOG_LR pair
     0.6229 vs 0.6234, HGB15 0.6268 vs 0.6300. FOLD_START is 0.6388 / 0.6424, and the RAW-J E3 point 0.6174 / 0.6232.
   - **Why the income components are not in R1's rule (the diagnosis argues it).** START's own out-of-fold income spend over the fold
     start is +0.0020 (0.0007) NLL and +0.0017 (0.0005) Brier in v1, i.e. +0.0027 at +2 SE, above the registered 0.0025 income-Brier
     budget. Any release that keeps START's income would fail the registered income rule in v1 for a reason no strength can change.
     R1 does not touch the income output, so its income components over START are exactly 0 (asserted). Its income values over the
     fold start equal START's and are recorded.
4. **Why usefulness may be preserved.**
   - The income output is NEURAL_RAW's, so its income clauses are NEURAL_RAW's own, and the anchor is point-eligible.
   - The occupation network keeps its full-vector cross-entropy from START, and the rule budgets the five occupation components,
     both within-SEX guards included, with FB_PFD_v1's limits.
   - The RAW-J path's MODEL_FIT occupation spends (ALT) show the room the occupation side has: v0 E5 +0.0038, E10 +0.0096; v1 E5 +0.0023,
     E10 +0.0070 nats.
   - Label-free numbers from the invented-data selftest (disclosed as such):
     - CAND lowers the pair separation (in-sample D²_pair 0.060) against CONT (0.258) and START (0.217) while the income layers stay
       bitwise START's.
     - The fine RAW-J loop equals FB_PFD_v1's `run_rawj` at every epoch boundary, bitwise.
5. **Twin (CONT, gating).** FB_PFO_TASK is the identical occupation-only loop, with the same start, task rows, optimizer, folds and
   checkpoints, and lambda = 0. SEX is never read. The income is frozen at START's. It is released at CAND's S_k*.
6. **Strength grid and selection rule** (MODEL_FIT only; no CALIBRATION label).
   - **Path.** S10..S400, every 10 steps, at rho = 1. The 10-step path is part of the structural change: the whole push lands on the
     occupation network from step 0, and FB_PFD_v1's 40-step first checkpoint was too coarse.
   - **Cross-fitting.** Out-of-fold releases on the famrun household folds `SAR_xf_FB_S<v>` (5 folds). Each fold runs: the U0 recipe on
     its training rows (the fold start), then the RAW-J path from it (fold START = its S80 = E2), then CAND and CONT from fold START.
   - **Rule.** k* = the largest S_k such that S10..S_k all keep estimate + 2 household-clustered SE <= BUDGET_OCC on the five occupation
     components of the out-of-fold paired spend over the fold start (`famkit.spend`, `famkit.select_strength`): occupation NLL 0.0060,
     occupation Brier 0.0025, men's occupation Brier 0.0028, women's occupation Brier 0.0028, occupation accuracy 0.0060.
   - **Outcomes.** If S10 fails, status is NO_FEASIBLE_STRENGTH and nothing is released. Otherwise the deployed CAND is NEURAL_RAW's
     income network plus the occupation network trained on all MODEL_FIT from NEURAL_RAW E2's occupation network to S_k* (status FITTED).
7. **Matched controls.** Origin START = NEURAL_RAW for both paths: out of fold the cross-fitted RAW-J S80 (E2), deployed NEURAL_RAW rho
   1 E2. Selection data are MODEL_FIT_OOF, saved before any MODEL_SELECT read. CAND's spend over START is recorded as value, SE and income
   part (the income part is 0).
   - **Bracket rule.** The same `sar.fb.pfd.bracket` as FB_PFD_v1: HI = the point with the smallest spend >= CAND's + 2 SE; if none
     reaches it, the largest-spend point with flag CONTROL_PATH_MAY_NOT_BRACKET; CTRL_BRACKET_IS_ORIGIN if CAND's spend <= 0.
     - Only HI is saved (specification budget). LO, the largest spend <= CAND's, is computed and recorded but not saved
       (flag LO_NOT_SAVED_SPEC_BUDGET).
     - Under the frontier rule Q* is then HI, a control spending more than CAND. This is conservative toward the control.
   - **CTRL = FB_PFO_DQ_HI** (untargeted, matched to the mechanism): SEX-blind subtractive-dithered quantization of NEURAL_RAW's
     occupation output, with the income passed through bitwise.
     - It uses FB_PFD_v1's DQ basis (`sar.ctlkit.dq_fit` on the cross-fitted RAW-J E2 outputs), steps `DQ_GRID_FINE` and dither tag
       `SAR_DQ_FB_S<v>`. Its occupation coordinates therefore equal FB_PFD_v1's CTRL path at the same step.
     - Deployed: the same law on deployed NEURAL_RAW.
   - **ALT = FB_PFO_RAWJ_HI**: NEURAL_RAW's own RAW-J recipe continued beyond E2, re-selected on the same path with checkpoints every 10
     steps (S90..S400; the training is step for step `run_arm('RAW_J')`).
     - If the selected checkpoint equals an already saved FB_PFD_RAWJ point bitwise (v0: FB_PFD_RAWJ_HI = E3 = S120), that config is
       used as an alias (`alias_of`) and no new specification is saved.
   - **Recorded checks.** The all-MODEL_FIT RAW-J S80 weights equal `NEURAL_RAW__rho1.0__E2.pt` bitwise in both versions; the U0 recipe
     reproduces U0; CAND's income layers are START's.
   - **Fit record** (`results/pcrl_strong_anchor_release_v1/fb/FIT_PFO_V<v>.json`):
     - CAND's config, strength (steps), status and spend over START;
     - per path: origin, selection data, every grid point's spend over START and the SE of its paired difference against CAND, the
       LO / HI points, saved config ids, alias and flags;
     - CONT's config, the pins and the checks.
   - **Not built.** SBC (PFO with the PSEUDO2 attribute in place of SEX) and RO (PFO with a release-only SEX surrogate). Both are
     constructible and are left out for the specification budget. Joint design: not claimed (NOT_APPLICABLE_NOT_DECLARED).
8. **Revisions.** This is FB's first revision. No further revision is declared by this card.
9. **Registered predictions** (probabilities, written before the fit). Configurations and CPU follow the table.

   | # | Prediction | P |
   |---|---|---|
   | 1 | k* >= S30 in both versions | 0.75 |
   | 2 | NO_FEASIBLE_STRENGTH in either version | 0.08 |
   | 3 | k* in S40..S200 in both versions | 0.55 |
   | 4 | Binding MODEL_FIT component (first failure after k*): women's guard 0.40, occupation NLL 0.35, men's guard 0.10, occupation accuracy or Brier 0.10, none (S400 feasible) 0.05 | - |
   | 5 | CAND's spend over START: median +0.0035 nats, 80% interval [+0.0010, +0.0070]; income part exactly 0 | - |
   | 6 | MODEL_FIT LOG_LR pair of CAND at least 0.010 below START's in both versions | 0.45 |
   | 7 | MODEL_FIT LOG_LR pair of CAND below ALT HI's | 0.50 |
   | 8 | All 28 conditions at the MODEL_SELECT point | 0.70 (the risk is the occupation guards' transfer; a guard binds the point 0.5) |
   | 9 | Pair benefit vs START (NEURAL_RAW) > 0 at the point / resolved | 0.75 / 0.35 |
   | 10 | vs CTRL (FB_PFO_DQ_HI) > 0 at the point / resolved | 0.75 / 0.30 |
   | 11 | vs ALT (FB_PFO_RAWJ_HI or its alias) > 0 at the point / resolved | 0.55 / 0.15 |
   | 12 | vs CONT > 0 at the point / resolved | 0.85 / 0.45 |
   | 13 | D2 met (START, CONT, CTRL frontier and ALT frontier) | 0.10 |
   | 14 | D4 material target | 0.03 |
   | 15 | Income view passes the recipient condition (the vectors are NEURAL_RAW's) | 0.99 |
   | 16 | Occupation view <= NEURAL_RAW's + 0.010 | 0.97 |
   | 17 | Pair - occupation view <= 0.010 | 0.55 |
   | 18 | Most likely outcome: feasible at S60-S160; pair about 0.605-0.620 on MODEL_SELECT; point benefit +0.005 to +0.020 over NEURAL_RAW; not separated from ALT; above CONT | 0.40 |

   **What would contradict the rationale.**
   - (a) The occupation-only rule stops at S10-S30 with no MODEL_FIT pair drop: the occupation side has no cheap SEX signal either.
   - (b) CAND's MODEL_FIT pair falls by less than half of its occupation-view drop: the frozen income carries the coalition's signal (the
     coordinator's reading), which caps any occupation-only design.
   - (c) CTRL or ALT at matched spend reproduce CAND's pair drop.

   **Configurations.** At most 8 new: CAND 2 + CONT 2 + CTRL HI 2 + ALT HI 2. That falls to 7 when ALT aliases FB_PFD_RAWJ_HI in v0,
   and to 4 in a version where CTRL_BRACKET_IS_ORIGIN.

   **CPU.** About 2-3 CPU-min per version (upper estimate 6). Per fold: the U0 recipe, a 400-step RAW-J path, two 400-step occupation
   paths and out-of-fold releases at 41 checkpoints per path. Then the deployed fits, the 40-step occupation DQ path, the releases and the
   MODEL_FIT readers on up to 6 views. One thread, torch venv; deployment runs under python3.
10. **Novelty status.** Not claimed. Relatives: sequential / one-recipient schedules of joint protection, discrepancy and Fisher /
    Mahalanobis separability penalties, closed-form linear erasure (LEACE). No literature search was run.
