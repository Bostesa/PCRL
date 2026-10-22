# Designer A proposal: shared SEX-aware field generalization (SFG)

Campaign `pcrl_input_and_pair_release_v1`, Designer A (before the release). Written 2026-10-22, 01:52-01:54Z (`date -u`); label-free
work from 01:32Z.

- **Read:** the design brief; both FAILURE_MAPs; the dual-design RESEARCH_DECISION, ADVISOR_BRIEF and challenger review; `nrm/fa/prc.py`,
  `ohn/fam_a.py` (compact coding), the `ddr/ctlkit.py` index. **Not opened:** any label file, anything of MODEL_SELECT, any reader
  cache, `roles/designer_b/`. Both caches' row hashes equal the admitted MODEL_FIT `row_id` hash (`nrm.fa.prc.array_sha`), V0 and V1.
- **Files here:** `mech_sketch.py` (fit, release, ladder, selection rule; `selftest()` passes on invented data, 11.4 CPU-s);
  `feasibility.py`, `feasibility2.py`, `feasibility3.py` -> `FEASIBILITY*.json` (label-free plug-in estimates); `inspect_inputs.py`. Known flaw: `feasibility.py` orders all 12 steps in one list, so its
  last SEX-aware point (k=6) lets a later moderate step replace a strong one; that point is not cited. `mech_sketch.build_ladder`
  only moves fields to coarser levels.
- **Bottom line first.** My label-free evidence says that **no operation in my space plausibly beats its matched controls by
  0.02 pair AUC**. SFG is the best-justified one: it is the only device I found with a SEX-specific plug-in effect (about 0.012 over
  its SEX-blind same-form control at equal spend), and, unlike the tilt and standardization family of row 14, it *removes*
  information: the linked pair is a function of h(x), so a reader-independent ceiling exists. But a release-only map of the twin reproduces most of its plug-in gain at no more spend
  (failure 6 is predicted to repeat), and its feasible plug-in benefit over the twin is only about 0.005-0.013. I recommend a lean run (10-16
  configurations) as the input-side closure test, or giving the budget to Designer B.

## 1. The operation

- **Menu (fixed, label-free; two coarser levels per field; `mech_sketch.MENU`).** Ages are the 40 admitted levels (25-64); hours
  are recovered from the standardized WKHP (modal value = 40 h, minimum 1 h; check: maximum 99).

| Field | Moderate step | Strong step |
|---|---|---|
| AGEP | AGE5: 5-year bands (8) | AGE10: 10-year bands (4) |
| WKHP | WK6: 1-19, 20-34, 35-39, 40, 41-49, 50+ | WK3: <35, 35-44, 45+ |
| SCHL | SCH5: <=15, 16-17, 18-20, 21, 22-24 | SCH3: <=17, 18-20, >=21 |
| COW | COW5: for-profit, nonprofit, government, self-employed, unpaid | COW3: private, government, self-employed |
| MAR | MAR3: married, previously married, never married | MAR1: suppressed |
| POBP | POB4: Florida, other US, territories, foreign/other | POB1: suppressed |

- **Ladder (MODEL_FIT; task labels and SEX as a training label; 5 household folds).** For each step s applied alone: the task
  spend dT_s = out-of-fold increase of income NLL + occupation NLL over the twin, and the SEX information removed dI_s = out-of-fold
  increase of the log-loss of an HGB SEX model on h_s(Z) over one on Z. Greedy: each field moves only to a coarser level; take the
  step with the largest incremental dI / dT (`build_ladder`). h_k = the first k steps (a field keeps its coarsest step).
- **Task models.** HGB, A1_PO_16 settings (lr 0.025, 16 leaves, minimum leaf 100, l2 3), COW / MAR native categories, iterations by
  the same inner 5-fold household-CV T* rule as the twin (`nrm.fa.prc.tstar`), fitted on h_k(Z) of MODEL_FIT with income and
  occupation labels.
- **Release.** P_inc = f_inc(h_k(x)), P_occ = f_occ(h_k(x)); argmax decisions; float64, deterministic. No SEX and no propensity at
  runtime. k = 0 is the twin.
- **Strength grid and selection.** k in {2, 3, 4}. k* = the largest k whose every MODEL_FIT out-of-fold clause over the twin meets
  estimate + 2 household-clustered SE <= budget: NLL 0.012 per task (the twin is about 0.010-0.012 per task better than U0), accuracy loss
  0.008 per task, men's and women's occupation summed Brier 0.0045. CALIBRATION is not used. If no k passes, nothing is released.
- **Gate G1 (MODEL_FIT, before any MODEL_SELECT read).** Household-cross-fitted LOG_LR and HGB15 pair readers on out-of-fold releases
  with MODEL_FIT SEX: SFG_k* must be at least 0.005 below the matched BLIND point in both versions; otherwise stop and report.

## 2. Closest previous implementation and what differs

- **Grouped feature selection** (row 8; `results/pcrl_capacity_matched_feature_selection_v1/`, Adult): per-recipient deletion of whole
  raw groups. It failed because one-recipient drops vanish in the pair (the union covers all columns). **SFG** merges levels inside
  fields, on ACS FL, with **the same h for both recipients**, so the union failure cannot occur.
- **SCM** (row 5; `results/pcrl_scm_engineering_benchmark_v1/`, `pcrl_scm_utility_matched_v1/`): post-hoc merges of posterior cells.
  SFG merges raw-field levels before training, refits the task models, and releases continuous vectors.
- **FRC ORD** (`results/pcrl_finite_routing_comparison_v1/`): SEX-blind coarsening of the twin's outputs (release side).
- Not a closed family: no tilt, standardization, reweighting, cells in posterior space, noise, decoder, controller, critic or leaf
  correction. SEX only orders a fixed menu.

## 3. Failure addressed

Measured failures (2) super-additive pair (TWJ: 0.560 occupation view, 0.648 pair) and (3) leakage moving between recipients: both
outputs are functions of h(x), so the pair can never carry more than h(X) does, for any reader (data-processing inequality). This also
answers the removal-versus-obfuscation critique of PTC (injective, propensity written into the vector). It does **not** credibly
address (1) and (6), and it meets (5) head-on: income pays most of the spend (below).

## 4. Usefulness: label-free feasibility (plug-in: cached pi and eta as the truth)

Leakage = soft-label AUC of a cross-fitted regression of pi on the release (a denoised, data-rich reader); spend = KL from the cached
eta, net of the smoother's own (0.008 nats). Calibration: plug-in twin pair / income / occupation 0.654 / 0.554 / 0.644 (V0), against the
realized MODEL_SELECT 0.6506 / 0.5606 / 0.6450. Full-x ceiling 0.681 / 0.682 (V0 / V1).

| Release (V0; V1 in brackets) | Spend, nats (income + occupation) | Pair change | Ceiling AUC*(h) | Plug-in income accuracy loss |
|---|---|---|---|---|
| SFG k=2 (MAR1) | 0.0085 (0.0056 + 0.0028) [0.0081] | -0.0053 [-0.0058] | 0.666 | 0.007 |
| **SFG k=3 (MAR1 + WK3)** | **0.0181 (0.0108 + 0.0073) [0.0178]** | **-0.0127 [-0.0134]** | **0.656** | 0.011 (0.009 net) |
| SFG k=4 (+ COW3) | 0.0346 [0.0339] | -0.0243 [-0.0246] | 0.644 | 0.015, women's Brier +0.0095: infeasible |
| BLIND (cheapest first), 5 steps | 0.0203 | -0.0006 | 0.674 | 0.007 |
| BLIND, 6 steps | 0.0380 | -0.0093 | 0.658 | 0.018 |
| Interaction constraints: MAR main effect only / fully additive | 0.0113 / 0.0223 | -0.0013 / -0.0014 [-0.0006] | n/a | 0.004 / 0.007 |
| PROJ = E[SFG k=3 release given the twin's release] | -0.0006 | -0.0078 [-0.0077] | n/a | |
| Release-only tilt of the twin against E[pi given the twin's release], scale 0.5 / 1.0 | 0.0128 / 0.051 (vs the raw twin) | -0.0065 / -0.037 | n/a | |

Readings: SEX information sits cheaply in MAR (married / never / previously married) and in hours detail; the cheapest task steps
(age bands, POBP, SCHL detail) carry almost none. So SFG beats its SEX-blind same-form control by about 0.012 at equal spend. But
(i) field-structured models do nothing; (ii) the release-only projection keeps about 60% of the gain at no extra spend; (iii) the
release-only tilt, interpolated to SFG's spend, is about as low or lower (about -0.012 at SFG's net spend, -0.019 at
its gross spend 0.026, against SFG's -0.010 to -0.013). The ceiling at k=3 (0.656) is not below
the twin's realized pair, so the removal property has no bite at feasible strength. Expected k* = 2 (p 0.6) or 3 (p 0.4): at k = 3 the
net plug-in income accuracy loss (0.009) already exceeds the 0.008 budget, so the income clauses bind (failure 5) and R2 is the
likely revision.

## 5. Twin and matched controls

- **Twin:** DA_TWIN, the cached TASK teachers on the full Z (shared; inherited).
- **Simplest matched control, BLIND:** the same menu and ladder code with `order='blind'` (smallest task spend first, no target);
  two MODEL_FIT points bracketing SFG_k*'s spend. Same form, same spend, no SEX information.
- **SEX-independent attribute, ATTR:** `order='attr'` with PSEUDO2 (`ddr/ctlkit.pseudo2_select`), first rung reaching SFG_k*'s spend.
- **Release-only:** PROJ (E[SFG_k* release given the twin's release], fitted on MODEL_FIT out-of-fold releases, so no more spend by
  Jensen) and ZTILT at SFG_k*'s MODEL_FIT spend (ctlkit); **SEX-blind spending on the twin:** DQ at SFG_k*'s spend (ctlkit).

## 6. Revisions (at most two)

- **R1, slack-weighted ladder.** Trigger: on MODEL_FIT, an income clause is the only failing clause at k*+1 while every occupation
  clause keeps at least 50% slack. Change: dT = dNLL_inc / b_inc + dNLL_occ / b_occ in the ladder ratio. Same menu and rule.
- **R2, split the overshooting step.** Trigger: on MODEL_FIT, the next ladder step exceeds the remaining budget by more than 50%.
  Change: replace it by its two predeclared halves (WK3: merge only <35 / only 45+; COW3: merge only government / only self-employed;
  MAR1: merge never with previously married first). Same rule.

## 7. Registered predictions, configurations and CPU

Pair benefit = comparator AUC - SFG AUC on MODEL_SELECT, declared readers.

| Comparison | Predicted benefit | P(resolved, > 2 se) |
|---|---|---|
| vs TWIN (0.6506) | +0.005 at k*=2 (pair about 0.645); +0.010 at k*=3 (about 0.640) | 0.25 |
| vs BLIND bracket | +0.005 (k*=2) to +0.010 (k*=3) | 0.20 |
| vs ATTR | +0.007 | 0.20 |
| vs PROJ | +0.004 | 0.10 |
| vs ZTILT (matched spend) | -0.005 (ZTILT lower) | 0.05 |
| vs DQ (matched spend) | +0.007 | 0.20 |
| vs NEURAL_RAW / H_LJ | -0.003 / -0.019 | < 0.02 |

Views at k*=3 (k*=2): income about 0.553 (0.558; twin 0.5606), occupation about 0.636 (0.640; twin 0.6450). Per view against the
controls: within +-0.005 of BLIND, PROJ and ZTILT on the occupation view; on the income view about 0.02 above ZTILT (the tilt
flattens income: plug-in income view 0.526 at scale 0.5). Utility: point-feasible at k* (p 0.8); binding: income
accuracy or NLL, then women's occupation Brier. D2 NOT_SEPARATED (p 0.85), D4 NOT_MET (p 0.98). Data-rich readers: the benefit over
the twin persists (at least 0.008), with no removal-versus-tilt separation (benefit over ZTILT <= 0).

**Configurations** (twin inherited): core 10 = SFG_k* (2), BLIND bracket (4), PROJ (2), ZTILT (2); optional 6 = the other SFG rung
(2; the strength path and one-field ablation), ATTR (2), DQ (2). Total 16 at most.
**CPU:** ordering 0.7 CPU-h (12 steps x 3 HGB x 5 folds per version, plus the ATTR target), equally trained configuration fits
0.8, selection 0.1, readers about 1.0: **about 2.6 CPU-h**.

## 8. Novelty

NOT_CLAIMED. Field generalization and suppression (Samarati and Sweeney 1998; Sweeney 2002; Mondrian, LeFevre et al. 2006); the
privacy funnel (Makhdoumi et al. 2014) and agglomerative information bottleneck (Slonim and Tishby 1999); optimized pre-processing
(Calmon et al. 2017); information-theoretic privacy via data processing (du Pin Calmon and Fawaz 2012); interaction constraints in
gradient boosting. Training-target and row operations were not proposed: any SEX-dependent target or row weight yields a posterior
that is a functional of (pi, P(y | x, s)), so it re-parametrizes the closed STD / IPW / tilt family.

## CPU used (process CPU, one thread)

`inspect_inputs.py` 0.8 s; `feasibility.py` 69.5 s; `feasibility2.py` 31.6 s; `feasibility3.py` 21.8 s; `mech_sketch.selftest()`
11.4 s plus about 5 s in a failed first run; row-hash check 0.9 s; timing probe and start-ups about 4 s. **Total about 146 CPU-s.**
