# Research decision: two-recipient task-wire adapter

**Completion label:** COMPLETE: execution, independent verification (8/8 checks PASS, `INDEPENDENT_VERIFICATION.json`), deployment, same-host backup and restore, and publication. The off-device copy is pending.
**Scientific label:** `POINT_FEASIBLE_NO_COMPETITIVE_CRITERION` (recomputed independently with the same result)
**Recipe status:** CLOSED. This fixed recipe will not be re-tuned on these selection or INNER outcomes.

## The question and the answer
The question was whether adapting **both** recipients' transmitted predictions (65 income parameters plus 325 occupation parameters, over frozen trunks and heads) yields a useful coalition improvement: a fresh-reader pair-recovery reduction above 0.020 AUC with simultaneous bounds, where occupation-only adaptation had reached 0.0049.

**No.** The primary candidate J_BOTH (epoch 10) reduces aligned-pair recovery relative to RAW_PROJ by a point estimate of **0.0036**, interval [−0.0025, +0.0097]. That is smaller than the previous occupation-only 0.0049, and the whole interval lies below 0.020: an *established shortfall*. All five registered comparisons are established shortfalls.

## Primary family (K = 39; roster fixed at SCIENCE_LOCK, eligibility at SELECTION_LOCK)
The candidate is J_BOTH E10, which is point-feasible at FIT2 in both seeds. All five comparators were point-eligible, so no slot is incomplete. Intervals are paired over both seeds with a duplicate-group bootstrap (2,048 draws) and a Student-t Bonferroni critical value of 3.224.

| J_BOTH vs | pair benefit (needs lower > 0.020) | income excess (upper ≤ 0.010) | occupation excess (upper ≤ 0.010) |
|---|---|---|---|
| RAW_PROJ | +0.0036 [−0.0025, +0.0097] **shortfall** | −0.0037 [−0.0089, +0.0016] PASS | −0.0055 [−0.0128, +0.0018] PASS |
| T_BOTH | +0.0027 [−0.0041, +0.0095] **shortfall** | −0.0045 [−0.0092, +0.0001] PASS | −0.0058 [−0.0125, +0.0009] PASS |
| L_BOTH | −0.0007 [−0.0067, +0.0052] **shortfall** | −0.0002 [−0.0051, +0.0048] PASS | +0.0044 [−0.0035, +0.0124] UNRESOLVED |
| S12 | −0.0009 [−0.0064, +0.0046] **shortfall** | −0.0008 [−0.0051, +0.0036] PASS | +0.0025 [−0.0041, +0.0090] PASS |
| S21 | −0.0000 [−0.0058, +0.0058] **shortfall** | −0.0011 [−0.0056, +0.0034] PASS | +0.0042 [−0.0032, +0.0116] UNRESOLVED |

"Shortfall" is status `ESTABLISHED_VIOLATION`: the whole interval lies below the required benefit. The income and occupation excess rows are no-excess bounds, not protection criteria.

**J_BOTH INNER utility (24 clauses):** 5 PASS, 19 UNRESOLVED, 0 established violations, and every point estimate passes.
- **PASS:** seed-1 income accuracy vs U0 and vs Ucal; seed-1 occupation NLL vs U0, and Brier vs U0 and vs Ucal.
- **Largest upper bound:** seed-1 income NLL vs Ucal, at 0.0226 against a cap of 0.010.

## Direct answers
1. **Did changing income help?** Not for the coalition.
   - **Income alone:** income-only adaptation gives a pair reduction of −0.0004 relative to RAW_PROJ. That means pair recovery was 0.0004 *higher*, unresolved (D08 [−0.0060, +0.0051]).
   - **Income added to changed occupation:** substituting J_BOTH's changed income wire for the raw one, with the occupation wire already changed, gives a pair reduction of −0.0024. Recovery was 0.0024 higher, unresolved (D14 [−0.0073, +0.0025]).
   - **Income wire alone:** the changed income wire lowered income-only recovery by about 0.003–0.004 (D01/D08/D11 r1), unresolved.
   - **Task loss:** the income change did not improve task loss either. J_BOTH's INNER income NLL excess vs U0 is 0.0075 / 0.0062 (seed 0 / seed 1), against 0.0059 / 0.0061 for RAW_PROJ.
   - **What this does not show:** these results do not show that frozen income caused the previous shortfall, or that it did not; they only show this recipe's income change bought no coalition reduction.
2. **Did both useful tasks survive?** At the point level, yes; at the simultaneous level, less well than RAW_PROJ.
   - **Point level:** all 24 point estimates pass on FIT2 and INNER.
   - **Simultaneous bounds:** J_BOTH resolves 5 of 24 clauses, against 10 of 24 for RAW_PROJ.

     | service | occupation clauses PASS | max occupation upper bound | income clauses PASS |
     |---|---|---|---|
     | RAW_PROJ | 10/12 | 0.0129 | 0/12 |
     | T_BOTH (task only) | 8/12 | 0.0134 | 2/12 |
     | J_BOTH | 3/12 | 0.0162 | 2/12 |

   - **Held-out drift elsewhere in the bank:** three controls that passed FIT2 exceed the INNER point cap on income NLL vs Ucal. These are I_ONLY seed 1 (0.0101), S12 seed 1 (0.0101) and S21 (0.0102 / 0.0114); they are diagnostic only.
3. **Did each recipient and their combination reveal less?**
   - **Occupation alone:** J_BOTH 0.0055 lower than RAW_PROJ, unresolved (D01 r2). L_BOTH's occupation-only reduction of 0.0099 is established (D09), as is O_PREVIOUS's 0.0100 (D15, replicating the previous study).
   - **Income alone:** about 0.0037 lower, unresolved.
   - **Pair:** 0.0036, unresolved, and below the required margin.
   - **Complementarity:** the descriptive interaction is −0.0021, with an unadjusted 95% interval of [−0.0063, +0.0020] that includes zero. There is no evidence that combining the changed wires reduces recovery beyond the sum of the substitutions.
   - **Against the previous study:** the direct contrast with the previous occupation-only component (D06) is pair −0.0012 [−0.0065, +0.0041] and occupation view −0.0045 [−0.0105, +0.0015], both unresolved. Adapting both recipients did not beat adapting occupation alone.
4. **Did joint training beat local training and both sequential orders?** No. On the pair view, J_BOTH vs L_BOTH is −0.0007, vs S12 −0.0009 and vs S21 −0.0000, all unresolved in the diagnostic family. On occupation, L_BOTH and both sequential arms reveal slightly *less* than J_BOTH (point −0.0025 to −0.0044, unresolved).
5. **Did income uncertainty improve or remain unresolved?** It remained essentially unresolved.
   - **Income:** J_BOTH passes 2 of 12 clauses (seed-1 accuracy vs U0 and Ucal); 10 remain unresolved, with upper bounds up to 0.0226. RAW_PROJ passes 0 of 12, with a max upper bound of 0.0223.
   - **Why the 2 passes are not a privacy benefit:** the task-only T_BOTH gets the same 2 passes, so they come from small accuracy shifts.
   - **Occupation got worse:** occupation uncertainty widened, from 10/12 PASS for RAW_PROJ to 3/12 for J_BOTH (answer 2).
6. **What earned evidence, and what failed?**
   - **Earned:**
     - a runnable, point-feasible two-recipient adapter whose privacy term changes both transmitted wires (8/8 counterfactuals per recipient and seed; all 64 saved J_BOTH states independently re-solved, matching to ≤1.7e-18; income never had an active budget case, so its active-constraint path was checked only on 100 data-free fixtures);
     - an exact, matched six-arm bank with sequential phase freezing;
     - an established occupation-only reduction for the local arm L_BOTH (0.0099), and a replication of the previous occupation-only component (D15, 0.0100). Both are non-candidate, single-wire results.
   - **Failed:**
     - the coalition criterion (an established shortfall against every comparator);
     - any advantage of joint over local or sequential training;
     - any coalition benefit from changing income;
     - resolution of income utility uncertainty;
     - preservation of RAW_PROJ's resolved occupation clauses (10/12 resolved for RAW_PROJ, 3/12 for J_BOTH).

## Strongest favorable and adverse comparisons
- **Most favorable (primary):** J_BOTH vs RAW_PROJ, pair +0.0036 [−0.0025, +0.0097]. Across the diagnostic family, the strongest established reductions are both single-wire, non-candidate results:
  - O_PREVIOUS vs RAW_PROJ, occupation view: 0.0100 [0.0025, 0.0174] (D15, the inherited component);
  - L_BOTH vs RAW_PROJ, occupation view: 0.0099 [0.0012, 0.0186] (D09, the local arm).
- **Most adverse:** J_BOTH vs HYB_RI_JO (D14), pair reduction −0.0024 [−0.0073, +0.0025], unresolved. Adding the changed income wire to the changed occupation wire raised pair recovery slightly. In the primary family, the worst comparison is J_BOTH vs S12, pair −0.0009.
- **Gap to the failed clause:** the pair benefit vs RAW_PROJ is 0.0164 short at the point estimate, and its upper bound is still 0.0103 short.

## Forecasts (registered in SCIENCE_LOCK before fitting; not revised)
| forecast | p | outcome |
|---|---|---|
| F1 J_BOTH has a point-feasible FIT2 nominee | 0.55 | **true** (E10) |
| F2 J_BOTH pair benefit vs RAW_PROJ lower bound > 0.020 | 0.04 | **false** |
| F3 J_BOTH beats L_BOTH, S12 and S21 on pair (all established) | 0.03 | **false** |
| F4 I_ONLY establishes a pair reduction vs RAW_PROJ | 0.20 | **false** (−0.0004) |
| F5 all 12 J_BOTH income clauses PASS on INNER | 0.05 | **false** (2/12) |
| F6 D14 establishes a pair reduction | 0.20 | **false** (−0.0024) |

## Limits
- **Exposure:** this is exploratory development on exposed people; the design came after the previous results.
- **Conditioning:** intervals are conditional on the fitted models and readers, and the two seeds reuse the same people.
- **Hybrids:** they are fixed wire substitutions, not trained arms or information-theoretic proofs.
- **O_PREVIOUS:** an inherited component, not an equally trained control.
- **Sequential arms:** S12/S21 are learned sequential controls, not the published finite-law algorithm.
- **Novelty:** a separate question; nothing here establishes a population privacy guarantee.

## Status ledger
- **Point feasible:** J_BOTH and every control at FIT2. On INNER, all J_BOTH point estimates pass.
- **Established benefit:** none for the candidate. In the diagnostic family, L_BOTH and O_PREVIOUS show established occupation-only reductions.
- **Established shortfall:** all five primary pair-benefit endpoints.
- **Unresolved:** 19 of 24 J_BOTH utility clauses; two occupation no-excess bounds (vs L_BOTH and vs S21); 43 of 45 diagnostic endpoints.
- **Unavailable:** none. All five comparators were eligible and every endpoint was estimable.

## Backup
The same-host snapshot was restored from the backup alone, but that is not off-device custody. The off-device copy is **PENDING** (`BACKUP_VERIFICATION.json`), and the predecessor's off-device copy is also still pending.

## Decision
Close this fixed recipe. Adding a 65-parameter rank-1 income residual (over a frozen income head) to the occupation residual did not reduce coalition recovery by a resolvable amount relative to RAW_PROJ, any matched control, or the previous occupation-only adapter. This held whether training was joint, local or sequential, and income-only did no better. It also cost resolvable occupation utility.

The income KL budget was never active in J_BOTH (final KL 0.00074 / 0.00036), so the trust region on income was not what limited it. These results speak only to this recipe; they do not show that coalition signal is irremovable by other designs. No nearby parameter adjustment is scheduled.
