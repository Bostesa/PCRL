# Research decision: minimal task-wire adapter

**Completion label:** COMPLETE: execution, independent verification, deployment, same-device backup and restore, and publication. The off-device copy is pending (no writable external drive). Independent verification (INDEPENDENT_VERIFICATION.json) passed all 43 checks. It re-solved all 16 J_TRUST counterfactuals with its own gradients and solver, matching the stored displacements to 2.2e-18; the privacy term changed the displacement by at least 6.0e-04 and changed the probe wire. In those 16 states no budget halfspace was active, so the re-solves exercise the unconstrained and norm-ball branches only.
**Scientific label:** `POINT_FEASIBLE_NO_COMPETITIVE_CRITERION`.
**Recipe status:** CLOSED. This exact adapter recipe will not be re-tuned on these selection or INNER outcomes.

## The question and the answer
The question was whether a small, constrained correction to the occupation predictions (325 parameters in the head's row space, over a frozen trunk) can cut sensitive recovery below fully projected RAW_PROJ while keeping the original useful predictions and confidence.

**Utility: yes.** The nominee J_TRUST (epoch 10) passes all 24 original point clauses on FIT2 and on INNER in both seeds.

**Coalition protection: no.** Fresh readers recover the aligned pair (income plus occupation) only 0.0049 AUC less than from RAW_PROJ. The simultaneous interval is [−0.0017, +0.0115], entirely below the required 0.020. Every one of the five primary comparisons is an *established shortfall*, not a near miss.

What the adapter did earn is narrower. It lowered **occupation-only** recovery by about 0.010 AUC versus RAW_PROJ (diagnostic D07, 0.0100 [0.0025, 0.0174]) and by 0.008 versus the task-only control (D15, 0.0082 [0.0008, 0.0155]). Both are established in the separate diagnostic family (K=45). This is not specific to the coalition objective: L_TRUST got the same reduction (D09, 0.0100 [0.0023, 0.0178]) and J_FREE 0.0121. The reduction does not survive into the coalition view, because the frozen income recipient plus the occupation predictions still reveal SEX.

## Primary family (K = 39, fixed at SELECTION_LOCK before INNER)
The candidate is J_TRUST epoch 10. All five comparators were point-eligible at FIT2: RAW_PROJ, TASK_TRUST (e10), L_TRUST (e10), L_FREE (e5) and J_FREE (e10). Intervals are paired over both seeds, with a duplicate-group bootstrap (2,048 draws) and a Student-t Bonferroni critical value of 3.224.

| comparison | pair benefit (needs lower > 0.020) | occupation no-excess bound (upper ≤ 0.010; not a protection criterion) | income no-excess bound (upper ≤ 0.010) |
|---|---|---|---|
| vs RAW_PROJ | +0.0049 [−0.0017, +0.0115] **established shortfall** | −0.0100 [−0.0173, −0.0026] PASS | 0 (identical income wire) PASS |
| vs TASK_TRUST | +0.0051 [−0.0014, +0.0115] **established shortfall** | −0.0082 [−0.0154, −0.0009] PASS | 0 PASS |
| vs L_TRUST | −0.0002 [−0.0060, +0.0056] **established shortfall** | +0.0001 [−0.0050, +0.0052] PASS | 0 PASS |
| vs L_FREE | +0.0032 [−0.0031, +0.0095] **established shortfall** | −0.0054 [−0.0115, +0.0007] PASS | 0 PASS |
| vs J_FREE | +0.0022 [−0.0040, +0.0085] **established shortfall** | +0.0021 [−0.0038, +0.0081] PASS | 0 PASS |

The JSON and CSV files show "established shortfall" as status `ESTABLISHED_VIOLATION`: the whole interval lies below the required benefit.

**J_TRUST INNER utility, 24 clauses:**
- 5 PASS, all seed-1 occupation clauses: accuracy vs U0 and Ucal, NLL vs U0, Brier vs U0 and Ucal.
- 19 UNRESOLVED and 0 established violations. The unresolved clauses are all 12 income clauses, all 6 seed-0 occupation clauses, and seed-1 occupation NLL vs Ucal.
- Every point estimate passes.
- The 12 income clauses are the frozen income recipient's inherited uncertainty. The largest upper bound is income NLL vs Ucal in seed 1, at 0.0223 against a cap of 0.010. No occupation adapter can resolve these.

## Direct answers
1. **Did the adapter change the actual predictions because of the privacy term?** Yes.
   - For the first 8 accepted J_TRUST updates in each seed, re-solving each update at its saved state without the privacy gradient gave a different actual displacement and a different deployed probe wire beyond the frozen tolerance. The verifier independently re-solved all 16 and confirmed both.
   - By epoch 10, J_TRUST had moved the probe's occupation logits by up to 0.21–0.22 and changed 21–24 of 512 probe decisions. TASK_TRUST moved them by 0.03–0.05.
2. **Did those changes keep useful confidence outside the fitting pool?** Yes, at the point level.
   - On FIT2, J_TRUST's occupation NLL rose 0.0028 (S0) and 0.0015 (S1) above the start. On INNER, the occupation NLL excess vs U0 is 0.0055 (S0) and 0.0027 (S1), within the 0.010 cap.
   - The fitting-vs-FIT2 plot shows no material overfit for the TRUST arms. The FREE arms drifted past +0.004 on FIT2 in seed 0.
   - Simultaneous utility bounds remain partly unresolved (above).
3. **Did fresh readers recover less than from RAW_PROJ and the strongest point-eligible matched control?**
   - **Occupation alone:** yes. It is 0.010 lower vs RAW_PROJ (D07, established) and 0.008 lower vs TASK_TRUST (D15, established). L_TRUST achieved the same occupation reduction, so the coalition term is not responsible.
   - **Coalition pair:** not meaningfully. It is 0.0049 lower vs RAW_PROJ. In the primary family this is ESTABLISHED_VIOLATION against 0.020; in the diagnostic family (D07) the reduction is UNRESOLVED, [−0.0018, +0.0115]. Against the strongest pair-protecting point-eligible control, L_TRUST, the change is −0.0002.
4. **Did trust help, and did coalition coupling add anything?**
   - **Trust helped utility.** J_TRUST had lower held-out occupation NLL than J_FREE in both seeds on FIT2 and INNER. On INNER the excess vs U0 is 0.0055 vs 0.0070 (S0) and 0.0027 vs 0.0038 (S1), descriptively.
   - **Trust did not change recovery.** The J_TRUST − J_FREE difference is UNRESOLVED in every view.
   - **Coupling added nothing measurable.** J_TRUST vs L_TRUST: pair −0.0002 [−0.0060, +0.0056], occupation −0.0001.
5. **Did the previous separate and global guards buy protection, or only weaken their own critics?** Their fresh-reader recovery really fell against their unchanged start (RAW_HYBRID). This is diagnostic family K = 45; each value below is the drop in AUC:

   | guard | pair | occupation view |
   |---|---|---|
   | SEPARATE_GUARD e5 | 0.0151 [0.0044, 0.0257] | 0.0256 [0.0122, 0.0390] |
   | GLOBAL_UPDATE_GUARD e5 | 0.0116 [0.0025, 0.0207] | 0.0164 [0.0046, 0.0282] |

   So fresh readers recover less from the guard checkpoints than from their start; the reduction is not just weaker training critics. These are whole-checkpoint comparisons with no task-only control, so the drop cannot be attributed to the privacy-bearing updates alone. Both guards fail the original utility caps: occupation NLL excess vs U0 is 0.014–0.021 on FIT2 and 0.011–0.0175 on INNER. This is diagnosis, not a competitive success.

   The registered contrasts decompose the **own-projected** global variant, not the hybrid guard's own drop:
   - **D04, changed predictions under the head-row-space projection contract** (own-projected vs RAW_PROJ): pair 0.0209 [0.0108, 0.0310] and occupation 0.0350, both established.
   - **D03, residual removal at fixed changed predictions** (own-projected vs hybrid guard): occupation 0.0285, established; pair 0.0111 [−0.0003, 0.0226], unresolved.
   - **D05, combined**: pair 0.0228 [0.0120, 0.0335].

   The hybrid GLOBAL guard keeps its residual. Its drop relative to RAW_HYBRID (D02, pair 0.0116) satisfies D02 = D04 + D06 − D03 ≈ 0.0209 + 0.0019 − 0.0111. No registered contrast isolates changed predictions under the hybrid contract, so the cause of D02 itself is unresolved.
6. **What comes from already-known compression?** Not established. RAW_PROJ vs RAW_HYBRID gives pair 0.0019 [−0.0060, 0.0097] and occupation 0.0098 [−0.0016, 0.0212], both unresolved. The occupation point estimate is comparable in size to the adapter's own occupation effect. The adapter's established occupation reduction is measured against RAW_PROJ, so compression is already netted out of it.
7. **What original utility uncertainty remains because income is frozen?** All 12 income clauses are UNRESOLVED on INNER, for every service, with upper bounds up to 0.022. No occupation update can resolve them.

## Accounting note
The science lock planned 1,188 direct reader fits, mis-stating the control banks as 216 per seed. The actual control banks were 108 fits per seed (2 banks × 18 readers × 3 views), 216 in total, giving 972 direct fits.

## Label rules
The completion-label rules are those of handoff section 11. SCIENCE_LOCK registers the five labels and the primary-family rules (roster, K, endpoint thresholds) that they depend on.

## Strongest favorable and adverse matched comparisons
- **Most favorable:** J_TRUST vs RAW_PROJ, occupation view, recovery −0.0100 [−0.0173, −0.0026]. In the pair view, J_TRUST vs TASK_TRUST is +0.0051 [−0.0014, +0.0115].
- **Most adverse:** J_TRUST vs L_TRUST, pair −0.0002 [−0.0060, +0.0056]. The coalition objective bought nothing over the local one.
- **Gap to the failed clause:** the pair benefit vs RAW_PROJ needs a lower bound above 0.020. The point estimate is 0.0151 short, and even the upper bound is 0.0085 short.

## Forecasts (registered in SCIENCE_LOCK before fitting; not revised)
| forecast | probability | outcome |
|---|---|---|
| F1 a nonzero J_TRUST checkpoint passes both-seed original point utility (FIT2) | 0.55 | **true** (epoch 10) |
| F2 trust improves held-out utility relative to J_FREE | 0.75 | **true** descriptively (lower NLL/Brier excess, both seeds, both roles) |
| F3 J_TRUST beats RAW_PROJ by the coalition margin | 0.04 | **false** (established shortfall) |
| F4 J_TRUST beats the strongest eligible local control | 0.05 | **false** |
| F5 old GLOBAL_UPDATE_GUARD lowers independent recovery vs RAW_HYBRID | 0.30 | **true** (pair 0.0116, lower bound 0.0025 > 0; diagnostic family) |

## What this does and does not show
- **Earned:**
  - A runnable, point-feasible occupation adapter whose privacy term changes the transmitted predictions (16/16 counterfactuals, independently re-solved).
  - An established reduction in occupation-only fresh recovery versus RAW_PROJ and the matched task-only control. The local objective achieves it equally.
  - Fresh-reader evidence that the earlier guard checkpoints lowered recovery relative to their start, at an unacceptable utility cost. Without a task-only control, this cannot be attributed to their privacy-bearing updates alone.
- **Not earned:**
  - No coalition privacy criterion.
  - No benefit from trust or coupling on recovery.
  - No resolution of frozen income's utility uncertainty.
  - No population guarantee, novelty claim, or joint design of both recipients.
- **Caveats:**
  - Everything here is exploratory development on exposed people.
  - The two model seeds reuse the same people.
  - The intervals are conditional on the fitted models and readers.

## Next decision
This recipe is closed. Since occupation-only gains vanish in the coalition view while income stays frozen, the next study should decide whether to **allow a constrained change to the income recipient's released predictions, or move to a coalition-level release design**, rather than adding capacity to the occupation adapter.
