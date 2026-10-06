STATUS (2026-10-06 17:18Z, before FIT_LOCK): **0 REQUIRED**, so nothing from this review blocks the Stage 1 freeze.
There are 4 RECOMMENDED findings and 11 NOTES:
- E-R1 is RESOLVED (`cbp/run.py` 961679acced8, verified 17:13Z).
- E-R2 is adopted for PROTOCOL section 6 and METHOD_CARD section 5, and still applies to the final reports.
- E-R3 and E-R4 are open. Neither blocks the freeze.

| Review | REQUIRED | RECOMMENDED | NOTES |
|---|---|---|---|
| Structural proofs and scope (sections 1-5) | 0 | 1 (E-R2, wording) | 5 |
| `cbp/fit.py`, `cbp/deploy.py`, METHOD_CARD.md (section 6) | 0 | 1 (E-R1, in the lead's `cbp/run.py` call; resolved) | 4 |
| `cbp/audit.py`, composition and controls only (section 7) | 0 | 1 (E-R3, reporting) | 2 |
| PROTOCOL.md claim scope (section 6.4) | 0 | 1 (E-R4, Taylor sentence) | 0 |

# Math and claims review: confidence-budgeted privacy compression (cbp)

**Role and files.** Role E (math and claims reviewer). This role owns three files:
- this file;
- `PRIOR_ART_AND_CLAIM_SCOPE.md`;
- `cbp/review_tests/test_math_review.py` (never locked; `cbp/review_tests/` is not globbed by any lock).

It edited no other file, committed nothing and pushed nothing.

**Finding classes.**
- REQUIRED: a demonstrated defect, with a failing fixture and a proposed patch. It blocks the Stage 1 freeze.
- RECOMMENDED: a change the reviewer advises.
- NOTE: a limit or scope statement, kept for the record.

**Command** (about 25 s on one thread; every run went through the semaphore):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m cbp.sema \
        --label E:tests -- ~/PCRL/.venv/bin/python -m pytest cbp/review_tests/test_math_review.py -q

**Independence.** The REFERENCE section of the test file imports nothing from `qpc`, `dpc` or `cbp`:
- first-index argmax in pure Python;
- the smoothing formula;
- scalar `math.log` KL;
- dict-based plug-in MI and conditional MI.

`qpc`, `dpc` and `cbp` entry points are called only to produce the objects under test.

**Real data.** None. The reviewer read no Adult row, SEX value, task label, unit array or receipt of this study's
real fits, and ran no real-data code.

**Reviewed files** (sha256 prefixes at review time; the cbp files are working-tree drafts):

| File | sha256 prefix | Note |
|---|---|---|
| `cbp/fit.py` | 0cf288c0fed2 | role B, draft of about 17:00Z |
| `cbp/deploy.py` | d30f5a17fab3 | role B |
| `cbp/audit.py` | c91ce76b4a15 | role D; composition, MI diagnostic and controls only |
| `cbp/run.py` | 961679acced8 | lead; `stage_fit` only (it calls `cbp.fit`); includes the E-R1 resolution |
| `METHOD_CARD.md` | dc366978a337 | role B; section 6.3 |
| `PROTOCOL.md` | 27e24d6542bc | lead (commit 807fa89); claim-scope sections only (section 6.4) |
| `qpc/compress.py` | 3495e2a88171 | equals the qpc STAGE_B_LOCK pin |
| `qpc/release.py` | 8131d085ff04 | equals the pin |
| `qpc/kmeans.py` | f126e6b38706 | equals the pin |
| `dpc/release.py`, `dpc/partition.py` | 0f4b7e26f70d, 4b1d0febb751 | equal the pins |
| `qpc/deploy.py` | dd683fb94d36 | `cbp.deploy` wraps it |

The source study's reviews are not repeated. `results/pcrl_confidence_capacity_v1/MATH_REVIEW.md` covers:
- the k-means, release, compression, selection, family and inference code;
- 74 of 74 non-equivalent mutants caught.

cbp reuses `qpc/compress.py` and `qpc/release.py` unchanged (section 6). This review therefore concentrates on:
- the structural statements;
- the objectives at the new λ grid;
- the sequential correction and JOINT dominance;
- the new cbp layers.

## 1. Decision containment: an explicit proof

### 1.1 Setting and assumptions

Recipient i has K classes (income K = 2, occupation K = 6). A fitted policy has:
- a fine partition with cells f, a class map κ(f), strictly positive routing centroids μ_f and statistics (n_f, S_f);
- a token map τ from cells to tokens.

Derived quantities:
- the token class γ(t);
- the prototype q_t = smooth(S_t / n_t, γ(t)) when n_t > 0, and smooth(uniform, γ(t)) for a reserved fallback token.

Here smooth(m, c) = (m + ε·1 + ε·e_c) / (1 + (K+1)ε), with ε = 1e-12, in float64.

| # | Assumption | Where the code enforces it |
|---|---|---|
| A1 | **Accepted input.** p is a float64 K-vector, finite, p_k ≥ 0 and \|Σp − 1\| ≤ 1e-9. Any other input is refused, not released. | `dpc.partition.check_probs` |
| A2 | **Teacher decision.** d(p) = min{k : p_k = max_j p_j}, numpy's first-index argmax. A supplied decision array must equal it. | `check_decisions` |
| A3 | **Cover.** Every predicted class has at least one cell. A class with no fitting row has exactly one reserved fallback cell with n = 0. | `FinePartition.validate` |
| A4 | **Within-class tokens.** All member cells of a token share one class, so γ(τ(f)) = κ(f). | `dpc.release.token_tables` refuses a class-mixing token; `compress.State` refuses class-mixing coarse cells |
| A5 | **Strict prototypes.** q_t[γ(t)] > q_t[k] for every k ≠ γ(t). | `check_prototypes` at Policy construction; on load (stored prototypes must equal recomputation bitwise); per row in `qpc.release.encode` |
| A6 | **Deployment rule.** The release of p is (t, q_t, γ(t)), with t = τ(f*) and f* = argmin over cells f with κ(f) = d(p) of KL(p ‖ μ_f), ties to the lowest index. If class d(p) has one cell, f* is that cell. | `dpc.partition.assign_fine`, called by `qpc.release.encode`, called by `qpc.deploy.release` and `cbp.deploy.release` |
| A7 | **Arithmetic.** IEEE-754 binary64 with round-to-nearest. The registered sums are left folds in one fixed row order: `np.bincount`, then the `token_tables` loop over member cells in increasing index. | numpy 2.4.2 as pinned |

### 1.2 Theorem 1: exact decision preservation and containment

Under A1-A6, take any accepted p: a fitting row, an inner row, an assessment row or any future row, whether or not its
vector occurred in fitting. Let encode(p) = (t, q, δ). Then:
- **(a)** δ = d(p);
- **(b)** q has the STRICT argmax δ, so every tie convention a consumer applies to q returns δ;
- **(c)** δ = γ(t) is a function of the token alone, so the release contains the decision;
- **(d)** no guard in `encode` can fire.

Under A7 in addition, A5 holds automatically for every legitimately fitted statistic (Lemma 3). So the construction
never refuses a valid fit.

**Lemma 1 (routing stays in the predicted class).**
- By A3 the candidate set C_d = {f : κ(f) = d} is non-empty.
- If |C_d| = 1, f* is that cell.
- Otherwise each KL(p ‖ μ_f) is a finite real, because μ_f > 0 (`kl_matrix` refuses non-positive references) and p is
  finite with 0 log 0 = 0. The argmin of a finite non-empty set lies in that set.
- So κ(f*) = d.
- Rounding in the KL arithmetic can change WHICH cell of class d is chosen, never its class.

**Lemma 2 (token class).** By A4, γ(τ(f*)) = κ(f*) = d. So δ = γ(t) = d. This is integer equality, with no floating
point.

**Proof of the theorem.**
- (a) is Lemma 2.
- (b) is A5 applied to t.
- (c) holds by construction.
- (d): `encode` raises only if δ ≠ d, if argmax q ≠ d, or if q differs from the registered formula recomputed from the
  token's statistics. The first two are excluded by (a) and (b). The third is excluded because q_t is computed by that
  formula from those statistics. ∎

**Lemma 3 (A5 holds for fitted statistics, under A7).**

Setup:
- Take a token t of class c.
- Its statistics come from fitting rows that deployment routed to its member cells: `kmeans._check_deployment` asserts
  that the stored statistics equal the deployed ones bitwise.
- By Lemma 1, every such row x has d(x) = c. So x_c ≥ x_k for all k, strictly for k < c.

(i) **The weak order survives summation.**
- For every k, S_t[k] is the same left fold s ← fl(s + x_k) over the same rows in the same order.
- fl is monotone, and addition is monotone in each argument. By induction, S_t[c] ≥ S_t[k].
- Division by the same n_t > 0 is monotone, so m_c ≥ m_k for m = fl(S_t / n_t).

(ii) **Smoothing makes the order strict.**
- The registered code computes q_k = fl(a_k / D), where:
  - a_k = fl(m_k + ε) for k ≠ c;
  - a_c = fl(fl(m_c + ε) + ε);
  - D = fl(1 + fl((K+1)ε)) > 0.
- By monotonicity, a_k ≤ a'_c := fl(m_c + ε).
- Rounding gives a_c ≥ a'_c + ε − ½ ulp(a_c).
- For a mean of probability rows, m_c ≤ (1 + 1e-9)(1 + n·2^-53) < 2 for any n below 10^15. So ulp(a_c) ≤ 2^-51 ≈ 4.4e-16
  and a_c − a_k ≥ ε − 2.2e-16.
- After division, q_c − q_k ≥ (ε − 2.2e-16)/(1 + 7e-12) − 4.4e-16 > 9e-13 > 0.
- A fallback token has m = uniform, so every coordinate is tied and the same argument applies.

So A5 holds, with a margin above 9e-13, whatever the ties. ∎

**The role of ε·e_c.** It is what makes ties strict.
- Suppose every member of a cell ties its class c with a LATER index k. Then m_c = m_k exactly.
- Without the e_c term, q_c = q_k. A last-index consumer would then read class k.
- Under continuous score distributions such ties have probability zero. Sampled checks therefore never reveal the
  difference (section 1.3).

**Scope of the proof.**
- It uses only A1-A7. The objective, λ, optimiser, starts, sweeps and witnesses never enter.
- It therefore covers every family: CLASS, DIRECT-TASK, FINE-TASK, LOCAL, SEQ-12, SEQ-21 and JOINT.
- It covers every λ of the grid, all 48 new fits and the 24 reused ones, with no new argument.
- It covers the deployed path. `qpc.deploy.release` calls `qpc.release.encode` on the teacher's float64 p, and
  `cbp.deploy.release` adds only a registration check before calling it.
- "Decision preserved" means equal to the first-index argmax of the p that the deployment computes. Equality with the
  ADMITTED teacher decisions relies on the bitwise teacher parity established at source admission, not on this theorem.

Tests:
- `test_lemma_strict_smoothing_margin_for_any_weakly_ordered_mean`: 17 magnitudes from 0 and denormals to 1.999; all
  coordinates tied; random weak orders. Margin > 9e-13.
- `test_lemma_row_order_sums_and_means_keep_the_weak_order`: later-index exact ties, one-ulp tie breaks, 1e-300
  entries, 1 to 2,000 rows, token accumulation over member cells.
- `test_routing_never_leaves_the_predicted_class`.
- `test_guards_refuse_class_mixing_nonstrict_prototypes_and_tampering`.
- `test_decision_preserved_on_adversarial_inputs_for_every_family_and_lambda`: CLASS, FINE-TASK, LOCAL, SEQ-12, SEQ-21
  and JOINT at λ 0.01, 0.06 and 50. The inputs are fitting rows plus unseen ones:
  - every one-hot, including the absent class;
  - every exact two-way tie and the K-way tie;
  - one-ulp tie breaks;
  - denormals and −0.0;
  - accepted sum errors of ±9e-10;
  - ties involving the absent class;
  - spiky and near-uniform rows far from every centroid.

  Over 20,000 row checks, the released decision equals the first-index argmax, the token class equals the decision,
  and the decoded vector is the prototype with STRICT argmax under both first- and last-index conventions.
- `test_decision_preserved_for_arbitrary_within_class_groupings`: 200 random within-class groupings, from the identity
  to one token per class.
- `test_input_boundary_accepted_rows_preserve_and_others_are_refused`.

### 1.3 Why passing millions of row checks is not itself a proof

1. **The checks run only on rows that exist.**
   - The theorem is about every accepted input, including future rows.
   - The decisive cases occur with probability about zero under continuous teacher scores, and may be absent from all
     five roles. These are exact ties with a later index, absent-class rows, denormals, and single-cell classes on
     unseen rows.
   - `test_sampled_row_checks_cannot_distinguish_a_construction_without_the_eps_ec_term` shows a smoothing rule without
     ε·e_c passing all 200,000 random checks and failing on one exact tie.
2. **The count is not independent evidence.** `encode` raises on any violation. "0 failed checks" in a completed run
   is therefore implied by the run completing. The count says only that no exception occurred on those rows. The
   universal statement needs Theorem 1's argument that the exception path is unreachable.
3. **Row checks cannot cover code paths the data does not exercise.**

Row-check and failure counts (CLASS_PRESERVATION.json) are implementation receipts, as the prompt requires. The
guarantee rests on Theorem 1.

### 1.4 What Theorem 1 does not repair or imply

- **Teacher errors are copied exactly.**
  - Accuracy, the confusion matrix and every per-class recall equal the teacher's under every code.
  - Occupation class 5 is never predicted on the fitting rows. Its recall is 0 under U and stays 0 under every code;
    class-preserving compression cannot improve it.
  - If the teacher predicts class 5 on a later row, the release says 5 with the fallback vector smooth(uniform). That
    vector's argmax margin is only about 1e-12, so it carries no confidence.
  - Test: `test_absent_class_zero_recall_is_inherited_not_repaired`, which checks identical confusion matrices and
    class-5 recall 0.
- **Confidence is not preserved.** Log loss and Brier change, and are measured on true labels.
- **Inputs outside A1 are refused, never silently released.**
- **The decision is a disclosure floor.** For every class-preserving code, I(S; C) = I(S; d) + I(S; C | d) ≥ I(S; d)
  (section 2). No code here can reduce the SEX signal carried by the decisions.
- **It is not a privacy property** of any kind.

## 2. Standard data-processing facts and their assumptions

These are standard facts, not new theorems. Test: `test_data_processing_and_containment_on_a_fitted_code`.

- **DP1 (post-processing by a fixed public map).**
  - Assumptions:
    - C_i = g_i(p_i) with g_i deterministic and frozen;
    - g_i was fitted on OSF_DEFENSE_FIT only, so for an AUDIT_FIT, INNER_SELECTION or assessment row the map does not
      depend on that row.
  - Statement: conditional on the fitted maps, S → (p1, p2) → (C1, C2) is a Markov chain under any law of a held-out
    row. Hence I(S; C1, C2) ≤ I(S; p1, p2) and I(S; C_i) ≤ I(S; p_i).
  - Fitting rows differ:
    - The privacy-trained maps depend on those rows' S, so the chain fails unconditionally.
    - The inequality then holds only for the fitting rows' empirical law, with the fitted maps treated as fixed.
- **DP2 (merging).** For C' = h(C), I(S; C') ≤ I(S; C) on any law, including the empirical one. Merging tokens never
  increases plug-in MI.
- **DP3 (containment chain rule).** Because d_i is a function of C_i (Theorem 1(c)):
  I(S; C1, C2) = I(S; d1, d2) + I(S; C1, C2 | d1, d2) ≥ I(S; d1, d2), and likewise per recipient.
- **DP4 (simulation).** Whoever holds p_i can compute g_i(p_i). Any reader of a code is therefore a reader of p_i
  composed with g_i. This is why the U continuous source bank composes with every public code map before selection
  (prompt section 10; section 7 below).
- **DP5 (Bayes-optimal readers).**
  - Assumptions: the true population law is known, and randomised likelihood-ratio tests are allowed.
  - Statement: the ROC of a garbling lies weakly below that of the original (Blackwell). So the optimal population AUC
    satisfies AUC*(d) ≤ AUC*(C) ≤ AUC*(p).

**What these facts do NOT imply.**

- **No SEX AUC bound for any fitted attacker, the declared slate included.**
  - Data processing orders information under the true law. It does not order the held-out AUC of a learner fitted on
    6,065 rows.
  - Coarsening can make a finite-sample reader BETTER. In `test_data_processing_does_not_bound_a_fitted_readers_auc`, a
    coarse code with strictly less population information gives an exact-code reader a held-out AUC higher by more
    than 0.01 than the fine code it came from, in at least 9 of 10 draws.
  - Measured AUCs are estimates of one selected reader's recovery, not bounds.
- **No population MI bound.**
  - The bound is I(S; p), which is unknown and large: U's pair AUC was 0.858 in the source.
  - Fitted plug-in values describe the empirical law of 15,434 rows. They are biased upward by the alphabet and downward
    by selection (section 3).
- **No training-data privacy.** See also E-N2.
  - Maps are fitted on OSF_DEFENSE_FIT rows, and the privacy families also use those rows' SEX.
  - A policy file stores per-cell counts and sums. A singleton cell's sum is one person's probability vector.
  - The grouping of a privacy-trained map is a function of the fitting rows' SEX.
  - Nothing here limits membership or attribute inference about the fitting rows.
- **No comparison between maps.** Data processing does not say that a privacy-trained map leaks less than a task-only
  map of the same capacity. That is an empirical, held-out question (claim C).

## 3. The privacy objectives

### 3.1 Definitions

On the N = 15,434 OSF_DEFENSE_FIT rows (natural logs; source definitions, unchanged):
- D_i = (1/N) Σ_rows KL(p_i ‖ q_{C_i}): the mean fitting KL from the teacher vector to the decoded prototype.
- I_i = Î(S; C_i): plug-in MI of the exact count table of SEX and the FULL categorical token.
- I12 = Î(S; C1, C2): the same for the aligned token pair.
- F_task = D1 + D2.
- F_local = D1 + D2 + λ(I1 + I2)/2.
- F_joint = D1 + D2 + λ[(I1 + I2)/2 + I12].

`test_objectives_are_decision_floor_plus_within_class_terms` pins the weights:
- W_joint has wI = λ/2 and w12 = λ;
- W_local has wI = λ/2 and w12 = 0;
- `F_values` agrees with these on the registered grid.

### 3.2 The objectives cannot touch the decision floor

DP3 applies to every class-preserving map on the same fitting rows, so:

    F_local = K_L + D1 + D2 + (λ/2)[I(S;C1|d1) + I(S;C2|d2)]
    F_joint = K_J + D1 + D2 + (λ/2)[I(S;C1|d1) + I(S;C2|d2)] + λ I(S;C1,C2|d1,d2)

The constants are K_L = (λ/2)[I(S;d1) + I(S;d2)] and K_J = K_L + λ I(S;d1,d2). Both are the same for every map, since
the decisions are fixed by the teacher. Consequences:
- The privacy term prices only within-class residual information.
- No λ can trade against the decision floor.
- The test checks that the floor is identical across 25 random maps and that both decompositions hold to 1e-12.

### 3.3 Plug-in bias at these alphabets and the permutation null

Under a true null (S independent of the code) the plug-in MI is positive. To first order it is about (L − 1)/(2N) for
L occupied cells (Miller-Madow). It grows when cells become sparse. Synthetic null on N = 15,434, P(S = 1) = 0.33,
uniform occupancy (`test_plugin_mi_null_bias_at_the_i8o64_alphabets`; five draws):

| Code | Occupied cells | Null plug-in MI (nats) | (L − 1)/(2N) |
|---|---:|---:|---:|
| income, 2 classes × 8 | 16 | 0.00055 | 0.00049 |
| occupation, 5 classes × 64 | 320 | 0.0104 | 0.0103 |
| pair, 16 × 320 | 4,877 of 5,120 | 0.198 | 0.158 (first order underestimates at about 3 rows per cell) |

Real codes occupy far fewer pair cells. The source measured permutation-null means for I12 of 0.063 (JOINT λ 0.1) and
0.085 (FINE-TASK) (`results/pcrl_confidence_capacity_v1/RESEARCH_DECISION.md`, section 6). The λ-weighted null level
of the pair term is then:

| λ | 0.01 | 0.025 | 0.04 | 0.06 | 0.08 | 0.1 |
|---|---:|---:|---:|---:|---:|---:|
| λ × (0.063 to 0.085) nats | 0.0006-0.0009 | 0.0016-0.0021 | 0.0025-0.0034 | 0.0038-0.0051 | 0.0050-0.0068 | 0.0063-0.0085 |

**Reading.**
- Over the upper half of the grid, the λ-weighted null level is the same order as the 0.006-nat headroom rule and the
  source codes' distortion.
- The null level is not constant across maps: merging lowers it. So λ·I12 partly prices occupied alphabet size, and an
  objective decrease can be pure bias reduction.
- The permutation-null receipts at the same code (prompt section 13) estimate this alphabet bias. Read every fitted MI
  beside them.

### 3.4 What fitted MI means and does not mean

**What it means.**
- It is the MI of the empirical joint law of (SEX, token) on the 15,434 fitting rows.
- It is an exact function of the counts.
- It is the training criterion being minimised.

**What it does not mean.**
- **Not a population quantity and not a bound.**
  - The alphabet bias pushes it up.
  - Selection pushes it down: a privacy-trained map is chosen to make exactly these counts balanced.
  - In `test_fitted_mi_below_its_permutation_null_is_selection_not_protection`, under a TRUE null:
    - JOINT at λ 5 reaches a fitted I12 below half of the permutation-null mean at its own code;
    - LOCAL at λ 1 reaches I2 below 0.6 times its null;
    - on fresh null rows the same JOINT code shows more than twice its fitted value.
  - Exploration on three seeds: JOINT λ 5 fitted I12 0.010 against a null mean of 0.045; 0.012 against 0.041; 0.006
    against 0.032.
- **A negative "excess over null" is selection, not protection.**
- **The permutation null at a fixed code does not measure the selection effect.** Permuting S without refitting the map
  estimates the alphabet bias only. Measuring the selection effect would need permuted-S refits, which were not run
  and are not required.
- **Not an attacker bound, and not a ranking of held-out recovery.** It is also not comparable across alphabets
  without the null. Held-out matched audits govern every claim.

## 4. The corrected sequential design

**Definition.** SEQ-ab runs in two stages, each of which is local search (greedy to the caps, objective-improving extra
merges, at most 5 exchange sweeps):
- Stage one minimises F_joint(C_a, d_b) over the class-preserving maps C_a of recipient a, with recipient b held at its
  CLASS-ONLY release d_b. C_a is then frozen.
- Stage two minimises F_joint(C_a*, C_b) over C_b, starting from b's fine cells.

**The stage-one objective, written out.**

    F_joint(C_a, d_b) = D_a + D_b^class + λ[(I_a + I(S;d_b))/2 + I(S;C_a,d_b)]
                      = [D_a + 1.5 λ I_a] + λ I(S; d_b | C_a) + const
                      = D_a + λ I_a/2 + λ I(S; C_a | d_b) + const'

- The source dpc surrogate D_a + 1.5 λ I_a is F_joint with b replaced by a CONSTANT. That view releases nothing, and no
  admissible b does: every class-preserving C_b determines d_b.
- The correction is therefore exactly λ I(S; d_b | C_a) ≥ 0, up to map-independent constants.
- It penalises first-map clues that are COMPLEMENTARY to the disclosed decision of b, and penalises less those that are
  redundant with it.
- Test: `test_sequential_correction_term_is_lambda_times_conditional_mi_of_the_other_decision`. The constant is the same
  for 30 random first maps to 1e-12.

**Why CLASS-ONLY is the right counterpart.** It is the coarsest release b can ever make under the contract. Every
admissible C_b refines d_b, so I(S; C_a, C_b) ≥ I(S; C_a, d_b). Two things follow:
- Stage one's pair term is a lower bound on the final pair term, whatever stage two does.
- That bound counts exactly the information the contract guarantees b will disclose.

A constant counterpart understates the pair term by λ I(S; d_b | C_a) for every candidate. It therefore misranks maps
whose clues complement d_b.

**Checks.** `test_sequential_stage_one_uses_the_class_only_counterpart_never_a_constant` runs on two fixtures at every
registered λ and at λ 1:
- the receipt's stage-one value equals my row recomputation with b at one token per predicted class;
- the counterpart's own term I(S; d_b) is present and non-zero (above 0.1 nats where d_2 carries S), so the
  counterpart is not constant;
- both stages run under W_joint;
- the first map is frozen.

Also:
- `test_cbp_new_units_sequential_counterpart_and_joint_dominance_from_the_deployed_release` repeats the check at the
  real rate (8, 64), on B's synthetic generator, from the STORED release arrays, through `cbp.fit.fit_unit`.
- On the designed fixture the correction changes the stage-one map, and the corrected map is never worse on F_joint.

**What it does not give.**
- Not optimality: stage one is local, and stage two is conditional on the frozen map.
- Not order-independence. SEQ-12 and SEQ-21 are different arms.
- Not the Taylor, Vippathalla and Coon algorithm. Theirs uses randomised channels, Blahut-Arimoto-style updates and
  later requests unknown when the first release is designed (`PRIOR_ART_AND_CLAIM_SCOPE.md`).
- No held-out statement.
- The source dpc surrogate is still fitted inside every SEQ unit as an unselected DIAGNOSTIC
  (`OLD_RULE_DIAGNOSTIC = True`, asserted by `cbp.fit`). It is never released or selected.

## 5. JOINT

### 5.1 Theorem 2: witness dominance on the fitting objective

**Setting.**
- W = {FINE-TASK, LOCAL_λ, SEQ-12_λ, SEQ-21_λ}: witness pairs fitted at the same caps (8, 64), on the same fine
  partitions and the same fitting rows and SEX.
- qpc refuses a witness whose fine fingerprints, family, caps or λ differ.
- JOINT's candidate set is Ω = {5 refined starts: JOINT-GREEDY and the four witnesses, each with at most 5 joint sweeps}
  ∪ W, with the witnesses UNCHANGED.
- The output is the candidate with the lowest from-scratch F̂_joint, with ties in a fixed order.

**Statement.** F̂_joint(JOINT) ≤ F̂_joint(w) for every w ∈ W.

**Proof.**
- W ⊆ Ω, and a minimum over a set is at most each of its elements.
- The final value is recomputed by the same deterministic code from the same labels, so it equals the candidate's
  value bitwise.
- qpc asserts this (`AssertionError` otherwise), and also asserts that no refinement raises the stage objective.
- The row-level recomputation from the deployed release agrees within 1e-9; that is asserted, and about 1e-15 is
  observed.
- So dominance also holds on the deployed release, up to that tolerance. ∎

**Corollary.** At each λ, JOINT ≤ SEQ-12, SEQ-21, LOCAL and FINE-TASK on the fitting F_joint. This holds BY
CONSTRUCTION. It is not an empirical finding, and not evidence for joint design.

**Tests.**
- `test_witness_dominance_holds_on_the_fitting_F_joint_over_the_registered_grid` covers all six registered λ. It
  recomputes from the released rows with my own KL and MI, and checks:
  - 9 candidates, 4 of them unchanged;
  - passed-in and recomputed witnesses give the same JOINT fingerprint.
- `test_cbp_new_units_...` does the same at (8, 64) through `cbp.fit`.

### 5.2 What dominance does NOT give

Each item is demonstrated on synthetic data in `test_dominance_is_only_on_the_fitting_objective_not_components_heldout_or_direct_task`
unless noted:

- **(a) Not the components.**
  - A lower F_joint can come with higher distortion D, or with a higher I1, I2 or I12, than a witness.
  - So JOINT may be WORSE for confidence or for one recipient's leakage.
- **(b) Not held-out rows.**
  - On held-out rows, JOINT's F_joint exceeds a witness's at some registered λ.
  - Exploration (6 seeds × 7 λ values) gave 63 held-out reversals against LOCAL/SEQ witnesses.
  - Nothing follows about held-out recovery (AUC).
- **(c) Not global optimality.** The registered search misses coordinated two-cell moves. This is the source review's
  XOR fixture, section 3.2.
- **(d) Not DIRECT-TASK.**
  - DIRECT-TASK is outside Ω and outside the fine-state family: its cells are k-means cells on rows, not unions of fine
    cells.
  - On the synthetic fixture, DIRECT-TASK at the same caps has a LOWER fitting F_joint than JOINT at every registered
    λ. Exploration: 41 of 42 (seed, λ) settings.
  - "JOINT dominates (or beats) DIRECT-TASK on the objective" must never be written.
- **(e) Not other λ.** There is no relation between JOINT at one λ and any arm at another.
- **(f) Not confidence.** D is teacher KL, not true-label loss. See source MATH_REVIEW section 1.5.

### 5.3 Computational asymmetry versus the sequential arms

**What each arm does.**

| Arm | Search |
|---|---|
| JOINT | Refines 5 starts with at most 5 joint sweeps each, and compares 9 candidates. It consumes 4 complete fitted units (FINE-TASK, LOCAL, SEQ-12, SEQ-21 at the same λ) |
| SEQ | One two-stage path (each stage greedy plus at most 5 sweeps), plus the unselected old-rule diagnostic |
| LOCAL | One path |

**Consequences.**
- Equal λ access, data, caps and the 5-sweep limit do NOT give equal realised compute.
- Ω ⊇ W, so JOINT's fitting-objective advantage is guaranteed.
- Any held-out JOINT advantage could reflect extra search rather than joint design. A compute-matched check would be
  needed to attribute it.
- Conversely, a JOINT tie with SEQ despite more search is informative.

**Work counters.**
- qpc's work counters are NOT comparable across families:
  - for SEQ they cover stage two only;
  - for JOINT they sum the five starts and exclude the witnesses' own fits.
- `cbp.fit.WORK_COVERAGE` states this, correctly.
- Use the unit CPU and the witness-inclusive JOINT CPU from `cbp.fit.asymmetry_table`.
- Test: `test_joint_search_has_more_starts_than_the_sequential_arms`.
- No compute-matched JOINT is in the registered bank, and none is required. The prompt only asks that the asymmetry be
  recorded.

## 6. Review of role B's `cbp/fit.py` and `cbp/deploy.py`

### 6.1 Checked

**No optimiser change.**
- `fit_unit` is a pass-through to `qpc.compress.fit_unit(family, fine, T, tr, S_fit, 8, 64, λ, meta, witnesses)`.
- `test_cbp_fit_unit_is_the_unchanged_qpc_unit_plus_receipts` checks, against direct qpc calls (LOCAL λ 0.04 and SEQ-21
  λ 0.06, at real caps on B's generator), that the following are identical:
  - the policy JSON;
  - the release arrays, bitwise;
  - the whole qpc record, apart from the added "cbp" key and the timing fields.
- No cbp module assigns to a qpc or dpc attribute; I grepped for assignment, `setattr` and `reload`.
- `cbp.fit` asserts the qpc constants at import: TOL, TIE_TOL, SWEEPS = 5, EPS, the witnesses, the JOINT start order, the
  permutation seed and count, and OLD_RULE_DIAGNOSTIC. `test_registered_search_constants_unchanged` pins them too.

**Refusals** (`test_cbp_fit_refuses_off_grid_reused_and_witnessless_jobs`):
- off-grid λ;
- a reused endpoint λ without a refit reason (recorded as ENDPOINT_REFIT otherwise, never silently);
- JOINT without its four witness policies;
- references (FINE-TASK);
- witnesses for a non-JOINT family.

**Reuse parity correctness** (`endpoint_parity`; `test_endpoint_parity_accepts_an_exact_replica_and_refuses_defects`).
An exact replica passes all 12 gates plus the binding gate. Each defect is caught by the right gate:

| Defect | Gate that catches it |
|---|---|
| Other SEX on 40 fitting rows | `fitting_terms_within_rtol` |
| One-ulp change of one decoded value, re-saved with a valid COMPLETE.json | `release_bitwise` |
| Fine partitions fitted at other caps | `fine_partition` |
| Wrong expected configuration | `config_consistent` |
| Wrong teacher binding | `binding` |
| A JOINT unit whose witness record's F_joint differs by 1e-9 | `joint_witness_dominance` |
| A JOINT unit fitted with RECOMPUTED rather than passed-in witnesses | `joint_witness_dominance` |

The gates as a whole are mathematically sufficient for "reusable without refit":
- code pins equal to STAGE_B_LOCK;
- identical fine partitions;
- identical fitting rows and SEX, implied by the objective terms within 1e-12 relative;
- bitwise deployment on all rows.

A deterministic refit from these inputs would reproduce the unit, so refitting would only manufacture a receipt, which
the prompt forbids.

**Deployed-release reconstruction** (`reconstruct_fit_statistics`):
- D uses `kl_rows` on the stored decoded vectors, and I uses `mi_plugin` on the stored tokens. These are the qpc
  definitions applied to the release, not to the engine state.
- Token counts must equal the stored n_t.
- Token means agree within MEAN_ATOL = 1e-12. The summation orders differ (rows versus cells), so bitwise equality is
  not expected, and the tolerance is right.

**Sequential counterpart and JOINT dominance at (8, 64)** through `cbp.fit`: section 4 and section 5.1. The
asymmetry receipt records 5 search paths, 9 candidates and 4 passed-in witnesses for JOINT, against 1 and 1 for SEQ.

**Deployment.**
- `cbp.deploy.release` = registration check + `qpc.deploy.release` → `qpc.release.encode` on float64 teacher
  probabilities. This is exactly the path of Theorem 1.
- Output is only token, decoded vector and decision per recipient.

### 6.2 Findings on `cbp/fit.py` and its call site

**E-R1 (RECOMMENDED; `cbp/run.py` `stage_fit`, about line 244; the lead's file).**

*What happens.*
- The reuse parity call is `FT.endpoint_parity(U(n), T, tr, S_fit, fine)`, without `witness_records` and without
  `meta`.
- For the 6 reused JOINT units (λ 0.01 and 0.1 × 3 seeds), parity therefore checks only the JOINT record's internal
  dominance: sources "passed_in" and every final_minus_witness ≤ 0.
- It does not check that the dominated witnesses are the admitted witness units of the same seed and λ.
- The binding gate never runs.

*Why it matters.* The prompt says to "assert dominance only on the fitting objective over those unchanged witnesses".
For reused units, "those witnesses" should be verified to be the admitted ones. B's gate supports this, and my test
shows it catches a 1e-9 discrepancy.

*Classification.* This is a verification gap, not a demonstrated defect, so it does not block the freeze.

*Resolution (verified 17:13Z).* `cbp/run.py` 961679acced8, committed by the lead in 75dbc30, now calls
`endpoint_parity(..., expected_config=cid, meta=bind_meta(k, cid, D), witness_records=wr)`:
- `wr` holds the FINE-TASK, LOCAL, SEQ-12 and SEQ-21 unit records of the same seed and λ for every reused JOINT unit;
- `bind_meta` carries both sha256 binding keys.

My first read of `stage_fit` predated that commit.

*The patch as proposed:*

    wrec = None
    if parse_id(cid)["family"] == "JOINT":
        lj = parse_id(cid)["lam"]
        wrec = {f: json.loads((U(unit_for(k, config_id(f, *RATE, None if f == "FINE-TASK" else lj)))
                               / "record.json").read_text()) for f in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")}
    r = FT.endpoint_parity(U(n), T, tr, S_fit, fine, expected_config=cid, meta=bind_meta(k, cid, D),
                           witness_records=wrec)

Pass `meta` only if `bind_meta` returns the two sha256 binding keys.

**E-N1 (NOTE; parity tolerance near zero).** `PARITY_RTOL` is purely relative. A fitted MI term very close to 0 could
fail it on summation-order roundoff although the absolute difference is about 1e-16.
- This is not observed in B's synthetic timing or in my fixtures.
- A failure would be loud (`PARITY_FAIL`), never silent.
- If it occurs, diagnose it as an engineering parity issue with the absolute difference reported. Do not refit to
  manufacture a receipt.

**E-N2 (NOTE; training-data scope of the deployable map).**
- `policy.json` carries every fine cell's n, S and A on the fitting rows. A singleton cell's S is one person's exact
  probability vector.
- A privacy-trained grouping is a function of the fitting rows' SEX.
- The policies are kept private (`MODEL_MANIFEST` uses `<PRIVATE_CACHE>`), which is correct.
- The audit's assumption that maps are public is a conservative attacker model for the recipients. It is not a licence
  to publish the maps.
- "Deployable" must not be read as "publishable without training-data risk".

### 6.3 METHOD_CARD.md (dc366978a337, reviewed 17:15Z)

**It agrees with the code and with sections 1-5.** Specifically:
- the bank: the λ grid, 24 reused units, 48 new units, 9 references, config IDs and unit names;
- the pass-through and the import-time constant check;
- the objectives, matching section 3.1;
- the search rules, the 5-sweep cap and "not rescued";
- the corrected sequential design: CLASS-ONLY counterpart, never a constant, old surrogate as an unselected diagnostic;
- JOINT: five starts, nine candidates, dominance on the fitting objective only, no DIRECT-TASK containment;
- the asymmetry table and the work-counter scopes, matching section 5.3;
- the parity gates and registered tolerances, matching section 6.1;
- the release interface;
- decision containment under its assumptions (i)-(iii). These are A3-A5 here; A1, A2 and A6 appear in its deployment
  paragraph;
- the "Not claimed" list.

**No REQUIRED or RECOMMENDED finding.** Two optional wording NOTEs:

**E-N10 (NOTE; METHOD_CARD section 2).** "Prototypes and partitions use only OSF_DEFENSE_FIT teacher probabilities and
SEX" is slightly loose.
- The fine partitions and prototypes use teacher probabilities only.
- The coarse GROUPINGS of the privacy families also use SEX.

**E-N11 (NOTE; METHOD_CARD section 4, JOINT).** The dominance caveat could add "nor component-wise (D1, D2, I1, I2, I12)
dominance" (section 5.2(a) here).

### 6.4 PROTOCOL.md claim scope (27e24d6542bc)

**These sections match the prompt:**
- section 1: output-release scope, not representation learning;
- section 2: the exposure statement verbatim;
- section 5: dominance on the fitting objective only, and the asymmetry "not described as matched compute";
- section 6: the non-implications (no SEX AUC bound, no population MI bound, no training-data privacy) and the pointer
  to this file for the containment argument;
- section 10: what a C pass does not mean.

**E-R4 (RECOMMENDED; PROTOCOL section 5).**
- Prompt section 7: "Do not call the sequential adaptations the official Taylor solver."
- PROTOCOL does not state this. METHOD_CARD sections 4.1 and 10 do, and METHOD_CARD is locked with the protocol
  documents.
- Suggested sentence for the section 5 "Compute asymmetry" bullet: "The sequential arms are matched adaptations of the
  source's corrected design, not the official Taylor, Vippathalla and Coon solver (PRIOR_ART_AND_CLAIM_SCOPE.md)."
- Optionally, in section 10: "A local or sequential winner is reported as such; no joint contribution is earned unless
  A and B pass."
- Not blocking: METHOD_CARD already carries the statement.

## 7. Review of role D's `cbp/audit.py`: mathematical soundness only

### 7.1 Composition

The rule is per source family and view:
- the bank list is the source's own bank, then every registered code's bank in registered order (27 cbp codes, then 11
  composition-only qpc maps);
- the decisions family uses only the CLASS-ONLY code, whose tokens are functions of d_i;
- the winner is the FIRST bank whose seed-0 selection value beats the running best by more than 1e-12;
- the reported value is that bank's seed 0-2 mean;
- every winning policy is frozen.

Soundness:
- Composition is justified by DP4: each code view is a function of p_i.
- The score views determine p_i through the public deployed head (dpc audit documentation).
- Reusing each policy's stored inner predictions is exact reuse, not an approximation: same AUDIT_FIT fit, same
  INNER_SELECTION rows, hash-checked.
- `test_composed_source_bank_rule_and_envelope` matches the code against my transcription on 30 synthetic banks with
  exact seed-0 ties. It checks winners, values, the decisions restriction and the freeze set.

**E-N3 (NOTE; the envelope holds for the selection statistic only).**
- By construction, U's composed seed-0 selection value is ≥ every code bank's, for every view.
- Consequence: if T* = U, the P* guards (individual AUC ≤ T* + 0.005) are nearly automatic at the selection statistic.
  The pair clause then asks for a margin below U's upper envelope.
- The REPORTED statistic is the seed 0-2 mean of the winning bank. It can fall BELOW the source's own reported mean;
  this happened in the test.
- So no guard or clause is automatically satisfied on reported values.
- The composed AUC is selection-optimistic: a maximum over 39 banks (source review AU-N2).

### 7.2 MI diagnostic

`_mi` equals the plug-in definition (`test_audit_mi_diagnostic_formula`), and the permutation list is a fixed function
of its seed. It is a diagnostic only.

**E-R3 (RECOMMENDED; reporting).** Two null receipts exist for the same estimand:
- the qpc fit receipt: 100 permutations, seed 20261006, inside each unit record;
- `cbp.audit.mi_diagnostic`: 50 permutations, seed 20261011.

Every report that shows a fitted MI must:
- name which null it uses;
- show the excess over that null;
- never read a negative excess as protection (section 3.4).

### 7.3 Controls

All are inner roles only, with labels permuted within AUDIT_FIT, within half A and within half B.

| Control | Assessment |
|---|---|
| NULL | Selecting on half A and evaluating on held-out half B gives an unbiased null AUC for the selected reader. The Mann-Whitney sd0 = sqrt((n0 + n1 + 1)/(12 n0 n1)) ignores ties. Ties reduce the true variance, so 0.5 + 3.5 sd0 is conservative for tied, discrete code readers. The realised threshold depends only on n0 and n1 of half B, so requiring equality with the source's 0.5654143765984265 is consistent |
| CONF | The b = 1 copy decodes to 0.95 q_t + 0.05 e_d. The argmax stays strictly d, so the plant preserves decisions. A perfect reader has AUC 0.5 + 0.5(0.9 − 0.1) = 0.9 |
| COLL | Two token IDs share one decoded vector. A probability-only audit cannot see the plant; the token identity can |
| XOR | b1 ~ Bernoulli(1/2) is independent of S*. So b2 = b1 XOR S' is marginally independent of S*, each local view is null, and the pair recovers S' |
| ROT | A clue of amplitude 1e-6 is spread by a Haar-random rotation. Scale-sensitive readers may miss it; failure is a technical-review trigger, as registered |

Plants never enter a bank: they are in memory or a temporary directory, never saved as units, and closure refuses
unexpected units.

**E-N4 (NOTE; multiplicity).**
- About 30 view-level null tests at z = 3.5 give roughly a 1% chance of one false exceedance in total (source AU-N5).
- A single exceedance is an investigation, not evidence of leakage.
- A passing null is not evidence of removal.

No REQUIRED or RECOMMENDED finding on the composition or controls mathematics.

## 8. Findings register

| ID | Class | Where | Summary | Status |
|---|---|---|---|---|
| E-R1 | RECOMMENDED | `cbp/run.py` `stage_fit` (lead) | Pass `witness_records` and `meta` to `endpoint_parity` for reused JOINT units | RESOLVED (75dbc30; verified 17:13Z) |
| E-R2 | RECOMMENDED | PROTOCOL / METHOD_CARD / reports (lead, B) | State decision preservation as Theorem 1 with A1-A7. Present row-check counts as receipts. Use the section 1.4 and section 2 non-implication wording | adopted in PROTOCOL section 6 and METHOD_CARD section 5; open for the final reports |
| E-R3 | RECOMMENDED | reports (lead, D) | Name the MI null used; show the excess; a negative excess is not protection | open (reports) |
| E-R4 | RECOMMENDED | PROTOCOL section 5 (lead) | Add "matched adaptations, not the official Taylor solver" | open; not blocking (METHOD_CARD carries it) |
| E-N1 | NOTE | `cbp/fit.py` | Relative-only parity tolerance near zero; a failure would be loud | - |
| E-N2 | NOTE | policies | Maps carry fitting-row statistics and SEX-dependent groupings; keep them private | - |
| E-N3 | NOTE | `cbp/audit.py` | The composition envelope holds for the seed-0 selection value, not the reported mean | - |
| E-N4 | NOTE | controls | Multiplicity of about 30 null tests | - |
| E-N5 | NOTE | objectives | λ·I12 null level is comparable to the headroom over the upper half of the grid (section 3.3) | - |
| E-N6 | NOTE | JOINT | Dominance by construction; demonstrated non-dominance in components, on held-out rows and against DIRECT-TASK | - |
| E-N7 | NOTE | SEQ | Correction = λ I(S; d_b \| C_a); the old surrogate remains a diagnostic only | - |
| E-N8 | NOTE | absent class | Class-5 recall 0 is inherited; the fallback carries no confidence | - |
| E-N9 | NOTE | prior art | See `PRIOR_ART_AND_CLAIM_SCOPE.md`; sequential arms are matched adaptations, not the Taylor solver | - |
| E-N10 | NOTE | METHOD_CARD section 2 | Only the privacy groupings use SEX; prototypes do not | - |
| E-N11 | NOTE | METHOD_CARD section 4 | Could add "no component-wise dominance" | - |

## 9. Test inventory (`cbp/review_tests/test_math_review.py`; 28 tests, all passing)

| Area | Tests |
|---|---|
| S1 decision containment | `test_lemma_strict_smoothing_margin_for_any_weakly_ordered_mean` (×2), `test_lemma_row_order_sums_and_means_keep_the_weak_order`, `test_decision_preserved_on_adversarial_inputs_for_every_family_and_lambda`, `test_decision_preserved_for_arbitrary_within_class_groupings`, `test_routing_never_leaves_the_predicted_class`, `test_guards_refuse_class_mixing_nonstrict_prototypes_and_tampering`, `test_input_boundary_accepted_rows_preserve_and_others_are_refused`, `test_absent_class_zero_recall_is_inherited_not_repaired`, `test_sampled_row_checks_cannot_distinguish_a_construction_without_the_eps_ec_term` |
| S2 data processing | `test_data_processing_and_containment_on_a_fitted_code`, `test_data_processing_does_not_bound_a_fitted_readers_auc` |
| S3 objectives and MI | `test_objectives_are_decision_floor_plus_within_class_terms`, `test_plugin_mi_null_bias_at_the_i8o64_alphabets`, `test_fitted_mi_below_its_permutation_null_is_selection_not_protection` |
| S4 sequential | `test_sequential_stage_one_uses_the_class_only_counterpart_never_a_constant` (×2 fixtures), `test_sequential_correction_term_is_lambda_times_conditional_mi_of_the_other_decision` |
| S5 JOINT | `test_witness_dominance_holds_on_the_fitting_F_joint_over_the_registered_grid`, `test_dominance_is_only_on_the_fitting_objective_not_components_heldout_or_direct_task`, `test_joint_search_has_more_starts_than_the_sequential_arms`, `test_registered_search_constants_unchanged` |
| S6 `cbp.fit` | `test_cbp_fit_unit_is_the_unchanged_qpc_unit_plus_receipts`, `test_cbp_fit_refuses_off_grid_reused_and_witnessless_jobs`, `test_cbp_new_units_sequential_counterpart_and_joint_dominance_from_the_deployed_release`, `test_endpoint_parity_accepts_an_exact_replica_and_refuses_defects` |
| S7 `cbp.audit` | `test_composed_source_bank_rule_and_envelope`, `test_audit_mi_diagnostic_formula` |

The tests import `cbp` modules lazily and skip when a module is absent. Because this directory is never locked, the
tests can be strengthened after the freeze without touching a locked file.

## 10. Real-data contact and resource use by the reviewer

**Real data.** None.

**Semaphore runs** (labels `E:*`; one thread each; ledger `<PRIVATE_CACHE>/cbp_v1/run/SEMA_LOG.jsonl`):

| Run | What | CPU |
|---|---|---|
| E:scratch-explore1, E:scratch-explore2 | Synthetic exploration for the demonstration fixtures | about 4 s and about 60 s |
| E:tests-draft1 to E:tests-draft3, E:tests-final | Test runs (final: 28 passed) | about 0.1 s, 13 s, 24 s and 23 s |
| E:bias-numbers | The section 3.3 null table | about 2 s |

Total: about 2 CPU-minutes. Literature checks (GitHub API, arXiv, web search) ran without the semaphore; they are not
numerical.
