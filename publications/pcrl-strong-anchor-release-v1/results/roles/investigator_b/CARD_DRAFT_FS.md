# Card FS_CPI_v1: conditional pair-increment residual on SUCC's income, occupation frozen (replacement family FS)

Draft by investigator B, 2026-10-24 (written 05:54Z-06:15Z, `date -u`), for the coordinator to commit as
`results/pcrl_strong_anchor_release_v1/cards/FS_CPI_v1.md` before the first fit it governs (never edited afterwards).
- **Status:** the campaign's one replacement family. It is ADAPTIVE as a whole, designed after the FA and FB MODEL_SELECT readings
  (`claims/FA_R1`, `claims/FB_R1`).
- **Code:** `sar/fs/cpi.py`, with deployment routed through `sar/fs/__init__.py`. It reuses `sar.fa.tfr`'s frozen-input loaders and
  feature map and `sar.fb.pfd`'s closed-form objective functions, read-only; neither module is changed.
- **Command:** `<repository>/.venv/bin/python -m sar.fs.cpi fit --version V --card results/pcrl_strong_anchor_release_v1/cards/FS_CPI_v1.md`,
  run inside a queue unit. `selftest` and `selftest_stage` pass.

**SAR1 declarations (proposed for the coordinator):**
- family `FS_CPI` (release files under `FS/`, card prefix `FS_`); label prefix ADAPTIVE_;
- START = SUCC (the declared benchmark reference, both versions);
- `cont_declared` = false: the lambda = 0 continuation is START bitwise (item 5);
- `ctrl_origin` = START; CTRL = FS_CPI_DQ (SEX-blind dithered quantization of SUCC's income, occupation passed through bitwise), saved
  point FS_CPI_DQ_HI;
- `alt_declared` = true; ALT = SUCC's own protection operation continued (H3's LJDIR transplant with beta beyond SUCC's), saved points
  FS_CPI_BETA_LO / FS_CPI_BETA_HI;
- `selection_data` CTRL = CALIBRATION, ALT = CALIBRATION; `start_in_sample_on_model_fit` = true;
- joint design not claimed: sequential by construction, occupation first (D5 NOT_APPLICABLE_NOT_DECLARED); novelty not claimed; no
  mechanism-specific reader; at most 8 new specifications.

1. **Operation.**
   - **Runtime input and release.** The admitted 145-column encoding X of the six permitted fields only. Each recipient gets its own
     float64 vector and argmax decision; deterministic.
   - **Occupation (not trained).** SUCC's JOINT_TREE E20 occupation vector, bitwise (`ihs.fam_h.predict`, frozen hash-pinned files).
     Every occupation clause and both guards, the binding men's guard included, are SUCC's own.
   - **Income.** `p = softmax(-(d_S + g)/2, (d_S + g)/2)`.
     - d_S is SUCC's H3 income log-odds.
     - g = g_theta(phi(x)) is a 64-64 ReLU residual, float64, with a zero-initialized output layer, so theta = 0 is SUCC.
     - phi is `sar.fa.tfr`'s label-free feature map: the standardized compact fields (AGEP, WKHP, SCHL, POBP code; COW and MAR one-hot),
       SUCC's logits (d_S, the JOINT occupation logits) and the admitted HGB teachers' logits (nrm fa_cache TASK.pkl: household
       out-of-fold values on MODEL_FIT, the deployed models elsewhere).
   - **Training.** On all 20,000 MODEL_FIT people. MODEL_FIT SEX is used only inside P_FS, and no task label is read. The loss is
     `KL(sigma(d_S) || sigma(d_S + g)) + lambda * P_FS`, where
     `P_FS = [Phi(D_pair / sqrt 2) - 1/2] + relu([Phi(D_inc / sqrt 2) - 1/2] - [Phi(D_inc,START / sqrt 2) - 1/2])`.
     - D_v² is FB_PFD_v1's closed-form whitened quadratic best response on the privacy batch (`sar.fb.pfd.poly2` / `fisher_d2`, ridge
       1e-3 tr/p).
     - D_pair is computed on (trainable income log-odds, the frozen occupation log-ratios as released).
     - The income-view term is a hinge at START's in-batch value: it acts only if the income view would become MORE separable than
       START's.
     - Optimizer: AdamW lr 1e-3, weight decay 0.01, clip 5. 400 steps; anchor batch 256 (`seed_for('SAR1fs-anchor', v)`); privacy batch
       2,048 (`seed_for('SAR1fs-privacy', v)`).
   - **Why this targets the increment.** With occupation frozen, `D_pair² = D_occ² + D²_{income | occupation}`, and only the second term
     can move. Under the full within-SEX covariance whitening, the income coordinate is weighted by its PARTIAL SEX signal given the
     released occupation coordinates: exactly SUCC's 0.040 super-additive increment, as far as a quadratic discriminant sees it.
   - **Pins.** The SHA-256 of every frozen file read (`sar.fa.tfr.frozen_inputs`: H3 STATE / STEP1 / HGB, the T_TREE teachers, LOCAL /
     JOINT E20; plus TASK.pkl) is recorded in the fit record, every MODEL.pkl and every release record. `predict` refuses if they change.
2. **Closest work and the structural difference.**
   - **H3 / LJDIR (`ihs/fam_h.py`).** Income is `Platt(s_B + D_L + beta * D_J)`, where D_J was trained by the replication jointly with
     JOINT_TREE's OWN income wire, then transplanted onto the H3 base at a budgeted beta (0.25 / 0.20). It was never trained against the
     pair it ships in (H3 income, JOINT occupation), and that pair is super-additive by 0.040. FS_CPI's residual is trained against the
     released pair itself, with the frozen partner as a constant input.
   - **ihs family P, PAIO (`ihs/fam_p.py`).** Occupation frozen; income = T_TREE routing plus learned leaf-score corrections with online
     SEX critics (income and pair wires) and a projection budget. The income-headroom successor's decision records that training against
     the frozen occupation "added only about 0.001" beyond the transplanted direction. FS_CPI differs in four ways:
     - (i) function class: a residual network over H3, the compact fields and teacher logits, not leaf scores (closed) on the task-only
       routing;
     - (ii) objective: the closed-form whitened best response, whose partial weights isolate the income increment given occupation, where
       early-stopped online critics weight correlated coordinates by their marginal mean difference (FB_PFD_v1's diagnosis);
     - (iii) anchor: KL to START, with no label cross-entropy on people where H3 is in-sample;
     - (iv) base: the already protected H3, not the task-only teacher.
   - **B3CTL / B2TWIN codes (amd).** 6-level income codes (closed). FS_CPI keeps continuous probabilities.
   - **FA_TFR_v1 / FA_TFRO_R1 (`sar/fa/`).** Same feature map and residual shape, but online critics with HGB distillation on the
     occupation output (R1) or both outputs (v1). FS_CPI moves the income only, against the frozen occupation, with the closed-form pair
     objective and the START anchor.
3. **Measured facts addressed** (committed records).
   - **SUCC's pair is super-additive.** On the SAR1 grid its pair is 0.6166, occupation view 0.5768, income view 0.5291: the pair sits
     0.040 above its occupation view.
   - **The occupation side is exhausted.** FA_TFRO_R1 moved the occupation output with H3 income frozen. Its feasible version 0 reached
     pair 0.6027 with occupation view 0.5845 (still 0.018 above), and version 1 broke the men's guard.
   - **FB's records point to the income coordinate.** The whitened coalition objective spent the income coordinate first (FB_PFD_v1:
     in-batch income-view D² down 88% in 40 steps), and the pair's incremental SEX signal sits in income given occupation. FB_R1, the
     occupation alone, was not separated from its ALT.
   - **SUCC's income has margin** (MODEL_SELECT records, anchor table): income NLL -0.0047 / -0.0052 vs U0 and -0.0036 vs Ucal (v0);
     Brier -0.0028 / -0.0033 vs U0 and -0.0023 vs Ucal (v0); accuracy loss +0.0001 / +0.0017; income reserve 3.51 / 2.86.
4. **Why usefulness may be preserved.**
   - Occupation is SUCC's bitwise (checked on every saved release).
   - The income moves only as far as lambda buys protection against the KL anchor to START.
   - The rule budgets the three income components on CALIBRATION at about half of SUCC's smallest recorded income margin.
   - Label-free checks (selftests):
     - a zero residual releases SUCC (max |diff| 2e-16);
     - lambda = 0 keeps the residual exactly 0;
     - the ALT law at SUCC's beta and maps reproduces SUCC's income exactly;
     - in an invented world with a conditional income increment, lambda = 1 cuts the increment D²_pair - D²_occ from 0.331 to 0.016
       without raising the income view (0.717 to 0.332).
5. **Twin (CONT).** Not declared. The anchor's gradient is zero at g = 0 and the output layer starts at zero, so lambda = 0 IS START,
   bitwise: the START comparison is the zero-protection comparison.
   - A CONT that differs from START would need a task term that moves the income at lambda = 0, e.g. distillation toward the HGB teacher.
   - That variant was rejected for FA's diagnosis D1: every such path point departs from START, and CALIBRATION's 2-SE noise of
     income Brier (about 0.0036) exceeds the budgets at every strength.
6. **Strength grid and selection rule.**
   - **Grid.** lambda in (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6), in nats of anchor KL per unit of AUC proxy. There is no gradient
     calibration, because the anchor has no gradient at START. One 400-step fit per lambda on all MODEL_FIT.
   - **Rule.** lambda* = the largest grid value such that it and every smaller one keep estimate + 2 household-clustered SE <=
     BUDGET_CAL on the CALIBRATION paired income spend over START_CF: income NLL 0.0065, income Brier 0.0035, income accuracy loss 0.0040.
     If the first fails, status is NO_FEASIBLE_STRENGTH.
   - **Budgets.** About 0.5 of SUCC's smallest recorded MODEL_SELECT margins over versions and references (income NLL 0.0136, Brier
     0.0073, accuracy 0.0083). This is an adaptive, disclosed choice; the factor 0.5 also absorbs a MODEL_FIT/CALIBRATION-to-MODEL_SELECT
     transfer of up to 2x (FB_R1: 1.95x).
   - **Occupation spends** are exactly 0.
   - **Selection data (honest statement).**
     - START_CF is H3 with its stack and Platt maps household-cross-fitted on CALIBRATION (folds `IHS_H_budget_S<v>`, H3's own budget
       estimator).
     - The residual, the HGB teachers and H3's trees and HGB learner are out of sample on CALIBRATION.
     - Remaining in-sample element: beta itself was chosen by H3's CALIBRATION rule (one parameter). Its bias is second order and
       disclosed.
     - MODEL_FIT is not used for spends: H3 and JOINT are in-sample there, and FA's D2 showed MODEL_FIT offsets of about 0.02.
     - CAND's CALIBRATION release is START_CF + g, with START_CF's own log-odds as the residual's d_S input (the out-of-fold equivalent).
7. **Matched controls.** Origin START. Selection data CALIBRATION, saved before any MODEL_SELECT read. CAND's spend over START (value,
   SE) is its paired CALIBRATION total-NLL spend over START_CF; only income moves, so it is the income NLL spend.
   - **CTRL = FS_CPI_DQ_HI** (untargeted): SEX-blind subtractive-dithered quantization of SUCC's income log-odds, with occupation passed
     through bitwise.
     - It uses `sar.ctlkit.dq_fit` on SUCC's MODEL_FIT outputs (label-free), `dq_release` over DQ_GRID_FINE and tag `SAR_DQ_FS_S<v>`.
       On CALIBRATION it is applied to START_CF's income; deployed, to SUCC's.
     - One point is saved: the smallest spend >= max(CAND's + 2 SE, 1.5 x CAND's). The 1.5x hedge is prospective, against the
       NO_BRACKET seen in FB_R1. If no point reaches it, the largest-spend point is saved with flag CONTROL_PATH_MAY_NOT_BRACKET.
   - **ALT = FS_CPI_BETA_LO / _HI**: SUCC's own operation continued: `a (s_B + D_L + beta D_J) + b` with beta on H3's grid above SUCC's
     (0.30 / 0.25 ... 2.0).
     - Spends use H3's household-cross-fitted CALIBRATION maps.
     - The deployed points refit the stack and Platt maps on all CALIBRATION for that beta, exactly `ihs.fam_h`'s recipe.
     - Bracket: LO = the largest spend <= CAND's, HI = the smallest spend >= CAND's + 2 SE (`sar.fb.pfd.bracket`).
   - **Flags.** CTRL_BRACKET_IS_ORIGIN when CAND's spend <= 0 (nothing saved); LO_IS_ORIGIN / LO_EQUALS_HI as in the shared rule.
   - **Recorded checks.**
     - the ALT recipe at SUCC's beta reproduces SUCC's income on CALIBRATION (expected <= 1e-12 with the real labels);
     - START_CF's mean NLL against H3's own recorded CALIBRATION cost;
     - occupation SUCC's bitwise in every saved release;
     - the pins.
   - **Fit record** (`results/pcrl_strong_anchor_release_v1/fs/FIT_CPI_V<v>.json`):
     - CAND's config, strength (lambda*), status and spend over START;
     - per path: origin, selection data, every grid point's spend over START and the SE of its paired difference against CAND, the
       saved LO / HI ids and flags;
     - the CONT declaration, the per-lambda CALIBRATION spends, the training logs, the pins and the checks.
   - **Not built.** SBC (CPI with the PSEUDO2 attribute in place of SEX) and RO (CPI with a release-only SEX surrogate). Both are
     constructible and are left out for the specification budget.
8. **Revisions.** None declared by this card.
9. **Registered predictions** (probabilities, written before the fit). Configurations and CPU follow the table.

   | # | Prediction | P |
   |---|---|---|
   | 1 | lambda* >= 0.1 in both versions | 0.65 |
   | 2 | NO_FEASIBLE_STRENGTH in either version | 0.10 |
   | 3 | Binding CALIBRATION component: income Brier 0.45, NLL 0.30, accuracy 0.25 | - |
   | 4 | CAND's CALIBRATION spend over START: median +0.003 nats, 80% interval [+0.001, +0.006] | - |
   | 5 | All 28 conditions at the MODEL_SELECT point (the occupation is SUCC's) | 0.85 |
   | 6 | Income view <= every comparator's + 0.010 | 0.90 |
   | 7 | MODEL_SELECT pair: median 0.603, 80% interval [0.590, 0.615]; occupation view = SUCC's (up to reader noise) | - |
   | 8 | Pair - occupation view: median +0.025 (from +0.040); <= +0.015 | 0.30 |
   | 9 | Benefit vs START (SUCC) > 0 at the point / resolved | 0.80 / 0.45 |
   | 10 | vs CTRL > 0 at the point / resolved | 0.80 / 0.40 |
   | 11 | NO_BRACKET_CTRL | 0.15 |
   | 12 | vs ALT > 0 at the point / resolved | 0.60 / 0.20 |
   | 13 | NO_BRACKET_ALT | 0.20 |
   | 14 | D2 met (START, CTRL frontier, ALT frontier) | 0.15 |
   | 15 | D3 | 0.10 |
   | 16 | D4 (resolved 0.020 vs NEURAL_RAW and H_LJ) | 0.12 |
   | 17 | Most likely outcome: feasible; pair 0.598-0.610; point benefit +0.006 to +0.018 over SUCC; separated from START and CTRL in at most one version; not separated from ALT | 0.40 |

   **What would contradict the rationale.**
   - (a) The CALIBRATION rule stops at lambda <= 0.05: the conditional increment cannot be bought with income utility.
   - (b) The MODEL_FIT descriptive pair-minus-occupation-view gap does not shrink while the income view falls: the objective removes
     marginal, not conditional, signal on real data.
   - (c) ALT at matched spend reproduces CAND's pair drop: more of H3's own transplant is as good.
   - (d) CTRL at matched spend reproduces it: information removal, not targeting.

   **Configurations.** At most 8 new: CAND 2 + CTRL 2 + ALT 4 (fewer when a bracket sits at the origin).

   **CPU.** About 1.5-2 CPU-min per version (upper estimate 4): seven residual fits (about 0.5 s each), SUCC / H3 components on
   MODEL_FIT and CALIBRATION, the beta path on cross-fitted CALIBRATION maps, the 40-step DQ path, releases for up to 4 configurations x 6
   roles, and the descriptive MODEL_FIT readers. One thread, torch venv; deployment runs under python3 (checked).
10. **Novelty status.** Not claimed. Relatives: sequential (occupation-first) protection; adversarial and conditional fairness penalties;
    Fisher / Mahalanobis separability; closed-form erasure (LEACE). No literature search was run.
