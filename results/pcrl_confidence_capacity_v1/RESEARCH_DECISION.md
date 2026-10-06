# Research decision: confidence capacity and privacy (qpc)

**Label: CONFIDENCE_FEASIBILITY_ESTABLISHED.** The same label results under the STAGE_A_LOCK rule and under the amended rule (PROTOCOL §11).

This is an exploratory, locked development comparison on 13,936 previously used Adult assessment rows (`EXPOSURE_LEDGER.md`). It is not fresh confirmation and not a population privacy guarantee.

## Plain-language answer

**Did more capacity or convergence fix confidence? Capacity did; convergence did not.**
- **Convergence alone:** at the old rate (8 states per class for each task), letting k-means converge (≤ 200 rounds, every class reaching a fixed point) changed occupation fit distortion by at most 0.0003 nats. Inner log-loss excess did not improve (seed 0: +0.0131 → +0.0133 nats).
- **More occupation capacity:** with 64 states per predicted class, the task-only code is eligible on every inner seed. On the locked assessment it passes all four registered confidence bounds:
  - occupation log loss +0.00295 nats [−0.0002, 0.0061];
  - occupation Brier +0.0017 [0.0007, 0.0027];
  - income log loss +0.0008 [−0.0007, 0.0023];
  - income Brier +0.0005 [0.0000, 0.0011].

  Every decision is preserved on every row.

**Did privacy training add a useful advantage? Not under the registered criterion.**
- **Recovery:** at the same capacity, the inner-selected privacy-trained code P* (JOINT, λ 0.1) lowers SEX recovery from both outputs together (pair AUC) by 0.034 [0.029, 0.039] relative to the task-only code T* (FINE-TASK). That is above the 0.02 margin, and neither recipient's own AUC rises.
- **Confidence cost:** P* costs +0.0081 nats of occupation log loss, with upper bound 0.0121 against the 0.01 allowance. Claim C therefore passes 10 of 11 clauses and is NOT_ESTABLISHED.

**Did joint design beat sequential design? No.**
- No JOINT code passed its guards, so J* has no eligible nominee; the strongest non-joint control is the sequential SEQ-21 λ 0.1 code.
- **Descriptive comparison:** the JOINT fallback (the same code as P*) leaks 0.003 [0.0001, 0.006] less on the pair than SEQ-21, but more to the occupation recipient (+0.011).

**What can be run** (`MODEL_MANIFEST.json`; tested `python -m qpc.deploy` from the 83-column input):

| Release | Status |
|---|---|
| U's continuous output | Eligible |
| Q\* (U\|DIRECT-TASK\|i8o64) | Confidence-feasible, but it protects little: pair SEX AUC 0.849 vs 0.858 for U |
| P\* (U\|JOINT\|i8o64\|l0.1) | Inner-eligible, with confidence preservation NOT established on the assessment |

## Status by category (prompt §17)

| Category | What it covers |
|---|---|
| Feasible on inner selection | Q* = U\|DIRECT-TASK\|i8o64 (the only eligible Stage A rate). Stage B: FINE-TASK i8o64; all four λ 0.01 privacy codes; SEQ-12, SEQ-21 and JOINT at λ 0.1. LOCAL λ 0.1 is eligible on 2/3 seeds only; every λ 1 code is ineligible |
| Confidence preservation established on the locked reused assessment | Q* only (P34–P37 all PASS). Not P* (P29 upper bound 0.0121) |
| Registered comparative criterion met | None. Claim C: 10/11 clauses. Claims A and B: no eligible J* |
| Descriptive favourable results | P* vs T* pair −0.034 (passes the margin clause). JOINT λ 0.1 lies below the task-only capacity curve |
| Missing comparator | None. C_rate = C_global = SEQ-21 λ 0.1 and T* = FINE-TASK i8o64 were all eligible NOMINEEs |
| Technically invalid or untriggered work | None invalid. Untriggered: the second Stage B rate (only one rate passed the gate), so 66 of the prompt's 105 nominal units exist. λ 0.01 and λ 1 privacy codes were inner-only (not in the scored list) |

## Claims (37-slot family; z = 3.2048452050105634; B = 1,999 exact-record-group bootstrap; seed 20261007)

| Claim | Nominee vs comparator | Status | Numeric clauses passing | Key clauses |
|---|---|---|---|---|
| A | J* (no eligible nominee; fallback JOINT λ 0.1) vs C_rate (SEQ-21 λ 0.1) | NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE | 8/11 (descriptive) | pair 0.0031 [0.0001, 0.0062]; v2 +0.0112 (guard breached); occupation LL +0.0081 [0.0040, 0.0121] |
| B | J* vs C_global (SEQ-21 λ 0.1) | NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE | 8/11 (descriptive) | same as A (C_global = C_rate) |
| C | P* (JOINT λ 0.1) vs T* (FINE-TASK i8o64) | NOT_ESTABLISHED | 10/11 | pair 0.0336 [0.0286, 0.0387] PASS; v1 −0.0010, v2 −0.0487 PASS; occupation LL +0.0081, upper bound 0.0121, NOT met |
| Q* | U\|DIRECT-TASK\|i8o64 | PASS | 4/4 | occupation LL +0.00295 [−0.0002, 0.0061]; occupation Brier +0.0017 [0.0007, 0.0027] |

## Assessment results (means over seeds 0–2; FINAL attack slate; lower AUC = less SEX recovery)

Accuracy of every code equals U's: 0.844 income and 0.475 occupation, by class preservation.

| Release | Pair AUC | Income-recipient AUC | Occupation-recipient AUC | Occupation LL excess | Occupation Brier excess | Income LL excess |
|---|---:|---:|---:|---:|---:|---:|
| U continuous (released interface; composed readers over all 22 codes) | 0.858 | 0.696 | 0.856 | 0 | 0 | 0 |
| Q* DIRECT-TASK i8o64 | 0.849 | 0.696 | 0.834 | +0.0030 | +0.0017 | +0.0008 |
| T* FINE-TASK i8o64 | 0.846 | 0.696 | 0.831 | +0.0031 | +0.0014 | +0.0011 |
| LOCAL i8o64 λ 0.1 | 0.834 | 0.697 | 0.800 | +0.0048 | +0.0019 | +0.0007 |
| SEQ-12 i8o64 λ 0.1 | 0.816 | 0.696 | 0.782 | +0.0082 | +0.0032 | +0.0011 |
| SEQ-21 i8o64 λ 0.1 (C_rate = C_global) | 0.816 | 0.695 | 0.771 | +0.0084 | +0.0033 | +0.0018 |
| **JOINT i8o64 λ 0.1 (P\*; J\* fallback)** | **0.813** | 0.695 | 0.782 | +0.0081 | +0.0029 | +0.0014 |
| DIRECT-TASK i8o32 | 0.842 | 0.696 | 0.826 | +0.0057 | +0.0026 | +0.0008 |
| DIRECT-TASK i8o16 | 0.834 | 0.696 | 0.807 | +0.0107 | +0.0044 | +0.0008 |
| DIRECT-TASK i8o8 | 0.820 | 0.696 | 0.787 | +0.0210 | +0.0077 | +0.0008 |
| CLASS-ONLY (decisions alone) | 0.739 | 0.587 | 0.687 | +0.1102 | +0.0411 | +0.0846 |
| RAW-J β 0.3 continuous | 0.788 | 0.685 | 0.774 | +0.0134 | +0.0065 | −0.0034 |
| FARE (official) | 0.704 | 0.685 | 0.636 | +0.0460 | +0.0233 | +0.0133 |
| F0 (no-fairness FARE) | 0.865 | 0.803 | 0.851 | +0.0299 | +0.0122 | −0.0115 |
| LEACE (official) | 0.808 | 0.554 | 0.801 | +0.0518 | +0.0274 | +0.1155 |

**Accuracy exceptions.** RAW-J reaches occupation accuracy 0.466, FARE 0.451 and F0 0.460. LEACE reaches income accuracy 0.794.

**Sources.** Full levels with SEs: `ALL_LEVELS.csv` and `COALITION_AND_RATE_RESULTS.csv`. Figures: `figures/fig_tradeoff.pdf` and `figures/fig_capacity.pdf`.

## Explanations (prompt §14)

**1. Convergence at a fixed rate (A1; `CONVERGENCE_DIAGNOSTIC.csv`).**
- **Reproduction:** the source 20-round DIRECT-TASK (8, 8) code is reproduced exactly on all three seeds (identical token IDs, bitwise decoded vectors).
- **Before and after:**
  - With the 20-round cap, 0–1 of 2 income classes and 2–4 of 5 occupation classes stopped at a fixed point; the rest hit the cap.
  - With ≤ 200 rounds, every class reached a fixed point, using at most 78 rounds.
- **Occupation fit KL:** 0.03083 → 0.03076 (seed 0), 0.033856 → 0.033856 (seed 1), 0.03272 → 0.03240 (seed 2).
- **Inner occupation log-loss excess:** +0.0131 → +0.0133, +0.0083 → +0.0083, +0.0134 → +0.0142.

Finishing the clustering is not what was missing; the old code was not "incorrect" for stopping at its registered cap. The three-start A2 (8, 8) code has worst-seed inner excess +0.0138 and assessment excess +0.0210, the same failure as dpc.

**2. Occupation capacity** (`CAPACITY_CURVE.csv`; income cap 8).

| Occupation states per class | Mean fit KL | Assessment LL excess | Assessment Brier excess |
|---:|---:|---:|---:|
| 8 | 0.0318 | +0.0210 | +0.0077 |
| 16 | 0.0197 | +0.0107 | +0.0044 |
| 32 | 0.0124 | +0.0057 | +0.0026 |
| 64 | 0.0078 | +0.0030 | +0.0017 |

Pair recovery rises with capacity: 0.820, 0.834, 0.842 and 0.849.

**3. Income versus occupation costs.**
- Income at 8 states per class costs +0.0008 nats on the assessment.
- Income at 4 states costs +0.0051 nats and fails the inner Brier allowance (+0.0053 worst seed).
- Occupation is the binding task at every rate below 64.

**4. Token caps versus emitted states and entropy.**
- At cap 64, all 320 occupation cells (5 predicted classes × 64) are emitted on the fitting rows, with fitting-row token entropy 5.31 nats.
- At caps 32, 16 and 8 the entropy is 4.70, 4.04 and 3.37 nats.
- Occupation class 5 is never predicted by the teacher and keeps a reserved fallback token, so the alphabet is 321.
- The privacy objective uses fewer states than its cap. On the fitting rows, JOINT λ 0.1 emits 269, 298 and 293 occupation tokens by seed (alphabets 270, 299 and 294, including the fallback), and JOINT λ 1 emits 99, 114 and 93.

**5. True-label confidence versus teacher KL.** Fit KL (distortion against the teacher) and true-label log-loss excess move together across the capacity curve. The excess is roughly 0.4–0.7 × the fit KL on the assessment. On the 2,235 inner rows, per-seed excess is noisy: at 32 states, one seed reached +0.0102 while the assessment mean is +0.0057.

**6. Ordinary versus privacy-trained compression** (same capacity i8o64; assessment).
- Pair recovery: task-only FINE-TASK 0.846 → LOCAL 0.834 → SEQ-12 0.816 / SEQ-21 0.816 → JOINT 0.813.
- Occupation log-loss cost: +0.0031 → +0.0048 → +0.0082 / +0.0084 → +0.0081.
- **Frontier:** privacy training buys about 0.03 pair AUC for about +0.005 nats. That is a better recovery-for-confidence trade than lowering capacity: DIRECT-TASK i8o16 has pair 0.834 at +0.0107 nats. But it is still above the registered upper bound.
- **Fitted MI** (pair, OSF_DEFENSE_FIT): JOINT λ 0.1 0.207 nats vs FINE-TASK 0.295. Their permutation nulls are 0.063 and 0.085, so the excess over null is 0.144 vs 0.210. The gain mostly does not track alphabet size (mean occupation alphabet 288 vs 321).
- **Not a guarantee:** fitted MI is a training criterion, biased at these alphabets, and not a privacy guarantee.

**7. Joint versus local and both sequential orders.**
- **At λ 0.1 on the assessment:** JOINT 0.813, SEQ-21 0.816, SEQ-12 0.816, LOCAL 0.834. Joint is within 0.003 of both sequential orders.
- **Inner selection:** JOINT λ 0.1 breaches the registered guard against SEQ-21 on the occupation recipient (seed means: inner v2 0.777 vs 0.768 + 0.005; the registered guard is per seed, and the largest breach is seed 1: 0.7708 vs 0.7598 + 0.005), so J* has no eligible nominee.
- **Corrected sequential baseline:** in all 18 sequential units the stage-one map was fitted against the other recipient's class-only release under the actual F_joint. The old D + 1.5λI rule would have chosen a different stage-one map in all 18 (independent verifier).
- **Optimiser:** JOINT is a local search; it misses one coordinated XOR move on a tiny exhaustive fixture (math review §3). No dominance over DIRECT-TASK is claimed.

**8. Absolute recovery, including decision leakage.**
- The decision vector alone gives pair AUC 0.739 (CLASS-ONLY).
- Every decision-preserving code here sits between 0.739 and the continuous 0.858.
- That floor is measured recovery by the declared attackers. The structural fact is only that each recipient can read its decision, not a population certificate.

**9. Do the inner gate and the assessment bounds agree?** Largely, at the extremes:
- **Agreement:** i8o64 passes both. Rates 8 and 16 fail the inner gate, and their assessment excess (+0.0210, +0.0107) would also fail the allowance.
- **Disagreement:** rate 32 failed the inner gate on one seed (+0.0102 on 2,235 rows) but has assessment mean +0.0057.
- **P\*:** inner-eligible (worst-seed occupation excess +0.0083) but its assessment upper bound is 0.0121.
- **Reading:** the inner gate is a per-seed point check on a small set, and the assessment bound adds z × SE.

## Strongest favourable and strongest adverse comparisons

**Strongest favourable.**
- Privacy-trained JOINT λ 0.1 vs task-only FINE-TASK at the same capacity: pair SEX AUC −0.0336 [−0.0387, −0.0286], with no increase for either recipient and identical decisions.
- Q* establishes that a decision-preserving finite code can keep both tasks' confidence within the registered allowances on this benchmark.

**Strongest adverse.**
- The code that keeps confidence (Q*) barely reduces recovery: pair 0.849 vs 0.858 for U's continuous scores.
- The code that reduces recovery (P*) does not keep occupation confidence within its registered bound (0.0121 > 0.01).
- Joint optimisation adds nothing measurable over sequential optimisation.
- FARE leaks least (0.704) but loses 2.4 occupation-accuracy points and 0.046 nats.

## Registered predictions (`PREDICTIONS.json`, committed 03ffe1c before any fit)

| ID | Prediction | Registered probability | Outcome |
|---|---|---:|---|
| QP1 | Converging (8, 8) alone passes | 0.07 | **No** |
| QP2 | An expanded-capacity code passes on every seed | 0.65 | **Yes** (i8o64 only) |
| QP3 | Q* passes all four assessment bounds | 0.40 | **Yes** |
| QP4 | P* beats T* by > 0.02 (lower bound) AND claim C passes | 0.12 | **No.** The margin clause passes (0.0286), but claim C fails on occupation log loss |
| QP5 | JOINT beats the strongest sequential arm by > 0.02 at useful confidence | 0.04 | **No** (0.003) |
| QP6 | No overall method advantage | 0.90 | **Yes** |
| QP7 | Label | CONFIDENCE_FEASIBILITY_ESTABLISHED 0.30 | **CONFIDENCE_FEASIBILITY_ESTABLISHED** |

## Process and validity

| Step | Commit and time |
|---|---|
| Predictions | 03ffe1c (04:05:54Z, before any fit) |
| SOURCE_ADMISSION_LOCK | 59cac84; teachers bitwise equal to the admitted dpc outputs |
| STAGE_A_LOCK | 2fd6b24 (04:20:25Z) |
| Gate | Stage A 04:20Z; gate 04:21Z |
| STAGE_B_LOCK | 03fe9dd (04:43Z) |
| AUDIT_AND_SELECTION_LOCK | 1c83e1b (04:51Z) |
| AMENDMENT_A1 | 520f93c: semaphore signal handling, engineering only |
| EVALUATION_LOCK | 4e92ce4 (05:27:48Z); the assessment opened 05:27:55Z, once |

**Protocol text changes after STAGE_A_LOCK** (before any recovery data, but after the Stage A gate was known; recorded in STAGE_B_LOCK):
- the guard-blocked nominee status (SEL-R1);
- a validly passing claim keeps its favourable label (SEL-C2);
- the unresolved-Q* wording (SEL-R2).

The label is reported under both rules; they agree here.

**Reviews and controls.**
- Math review: 0 open REQUIRED across Stage A, Stage B, selection/labels and attacks.
- Real-data controls: all pass, with 0 null exceedances. Confidence, collision, XOR and rotated-signal plants are detected, and a decisions-only audit misses the confidence plant.

**Process deviations (disclosed).**
- A math-review test file was edited after AUDIT_AND_SELECTION_LOCK. The controls stage refused it at 05:16:07Z, before loading data. The locked file was restored, and the edits moved to an unlocked post-lock test file.
- A queued review process was killed without a semaphore release record. The reviewer confirmed it ran about 40 CPU-s with no orphan. The semaphore was hardened (AMENDMENT_A1).
- Light review checks (about 62 CPU-s, each under the one-CPU-minute "light" threshold) ran next to two heavy holders.
- The semaphore log never shows more than two holds.

**Verification:** `VALIDATION.md` and `INDEPENDENT_VERIFICATION.json`.

## Prior art and claims

Every ingredient is prior work (`PRIOR_ART_AND_BASELINE_GAPS.md`): clustering and the information bottleneck, output compression, privacy-funnel objectives, data processing, and sequential releases under collusion. The official PURIFIER repository is still empty, and no Taylor et al. solver code was found (re-checked 2026-10-06). The sequential arms are matched adaptations, not the Taylor solver. No novelty is claimed, and a rate increase is not an algorithmic contribution.

## Decision and next step

1. **Record the result:** confidence feasibility is established for a task-only, decision-preserving code at 8 + 64 states per class, on this reused benchmark.
2. **Document the empirical confidence/recovery trade-off** (prompt §18, "confidence passes but protection disappears"):
   - the confidence-feasible task-only code protects little (−0.009 pair AUC vs continuous scores);
   - the privacy-trained codes buy about 0.03 pair AUC at about +0.005 nats, which is just outside the registered bound.
3. **Do not attribute any gain to joint optimisation:** joint ties sequential.
4. **Any follow-up** must be newly registered on inner data before any population is spent: for example, a privacy weight between 0.01 and 0.1 at i8o64, or a privacy term restricted to occupation. Neither may be selected or scored from this assessment. No confirmation population is opened by this study.
