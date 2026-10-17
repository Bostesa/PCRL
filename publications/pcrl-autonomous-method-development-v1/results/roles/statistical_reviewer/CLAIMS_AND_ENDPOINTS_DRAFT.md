# Claims and endpoints: DRAFT (frozen only if a finalist is promoted)

**Status.** Statistical reviewer's draft, written 2026-10-17 at about 17:50Z, before any finalist freeze, final reader fit, or ATTACK or DEV_REPORT label read.
- It becomes binding only when copied, with every placeholder resolved, into the campaign's finalist and reporting locks. That must happen before any final reader is fitted.
- The report code must assert the counts below.
- If no finalist is promoted, nothing here is used.

**Instantiation.** The worked instance is the registered-text finalist `B3CTL_UNCOUPLED_MIX50_C6A`, adjudicated in `DEVELOPMENT_REVIEW.md` §0. The other families appear as placeholders.

**Census context.**
- Public-use Florida ACS 2019 person records; DEV_REPORT has 5,943 persons in 3,955 households.
- Each recipient receives its own float64 probability vector and argmax decision: income (2 classes) or occupation (5 classes), computed from AGEP, WKHP, COW, SCHL, MAR and POBP.
- Readers infer the recorded SEX column from each output alone and from the linked pair.

---

## 0. Notation and family placeholders

- F is the finalist; s ∈ {0, 1} is the seed; r ∈ {income, occupation} is the recipient.
- REF = {NEURAL_RAW, H_LJ}, the eligible references. Each is MODEL_SELECT point-eligible on both seeds; NEURAL_RAW is 24/24 on DEV_REPORT in the source.
- TO is the matched task-only control: same input access, task-training exposure, capacity and optimization effort, with the privacy-specific element removed.
- SS is the strongest simple control of F's family, as named on its mechanism card.
- LOC is the local or sequential control that J needs.

| Slot | A (CCLC) | B, m = 1 (pure coupling: income = h(occupation vector)) | B, m > 1 coupled (income = h(occupation vector, c), e.g. B2) | B, m > 1 uncoupled (income = calibrated levels of c, e.g. **B3CTL**) | C (MIX) |
|---|---|---|---|---|---|
| **TO** | TASKREFIT80 with F's exact controller dictionary, privacy off: same 80 % of people, epochs, optimizer; rollbacks logged | same h on TASK_REFIT E20 bases | same h and c construction on TASK_REFIT E20 bases (B2CTL_TASKONLY_TT, on T_TREE bases, is acceptable only with "teacher-base construction" wording) | six decision-aligned, CALIBRATION-calibrated levels of the TASK_REFIT E20 income log-odds, plus the TASK_REFIT E20 occupation output (**to be fitted**: 2 configurations) | the SEX-blind T_TREE (bitwise identical-recipe refit) |
| **SS** | the best MODEL_SELECT-eligible stored JOINT_TREE checkpoint (fixed coefficient, stopping time only); if none is eligible, H_LJ (alias) | the uncoupled income of the card (H_LJ: same occupation output, LOCAL income) → alias | the uncoupled same-c twin. Its pair carries the same information, so this is a **utility leg** (income NLL), not a pair leg | H_LJ, the card's uncoupled LOCAL income with the same occupation output → **alias** of claim B's H_LJ keys | the matched-cost SEX-free temperature control (C1_CTRL_TEMP type) |
| **LOC** (J) | the same controller with the local objective, only if fitted and banked; otherwise NOT_CLAIMED | NOT_CLAIMED | NOT_CLAIMED (coupling is not a joint objective) | **B2CTL_UNCOUPLED_C6A**: the same m, bins, calibration and occupation output, with the LOCAL-only statistic (α = 0) | NOT_CLAIMED |
| Occupation output | its own | JOINT_TREE E20 if reused | JOINT_TREE E20 (= H_LJ's, bitwise) | JOINT_TREE E20 (= H_LJ's, bitwise) | its own (C1) or H_LJ's (C2) |

**When the occupation output is H_LJ's, bitwise (families B and C2):**
- the occupation-recipient endpoint against H_LJ is flagged `IDENTICAL_RELEASE` (point = SE = 0);
- the occupation-side DEV_REPORT utility, guards and recovery are flagged `REPLAY_OF_PUBLISHED`.

---

## 1. Signs and estimands (DEV_REPORT, per seed)

**AUC source.** Each AUC comes from the winner selected on ATTACK_VALIDATION, per release and view, from the frozen 8-reader bank. The bank includes the corrected LOOKUP reader and single-output alternatives for the pair readers. Winners are fitted on ATTACK_FIT and scored on DEV_REPORT.

| Estimand | Definition | Direction |
|---|---|---|
| `benefit_X` | AUC_pair(X) − AUC_pair(F) | positive favours F |
| `worsening_X,r` | AUC_r(F) − AUC_r(X) | non-worsening iff ≤ 0.010 |
| `utility` | accuracy_loss = acc(ref) − acc(F); nll_excess = NLL(F) − NLL(ref); brier_excess = Brier(F) − Brier(ref), for ref ∈ {U0, Ucal}, with NLL_i = −log max(p_{i,y_i}, 1e−12), Brier_i = Σ_k (p_ik − 1{y_i = k})² and accuracy from the argmax (first index on ties) | caps 0.010 / 0.010 / 0.005, inclusive |
| `subgroup` | occupation Brier(F) − Brier(U0), within recorded SEX g ∈ {0 = men, 1 = women} | cap 0.005, inclusive |
| `nll_diff` | mean_i [NLL_i(F) − NLL_i(X)] | positive means F spends more |
| `utility-leg benefit` (B coupled only) | NLL_income(SS) − NLL_income(F) | positive means coupling buys income NLL |
| `SEC` view and release differences | plus − minus, as named | — |

---

## 2. Claims: separate conclusions with explicit conjunctions

### Common conditions

- **CV.** The planted-SEX and shuffled-label controls are valid on every view and seed, inside bands frozen with the reader bank.
- **UG.** All 24 utility points are ≤ cap.
- **SG.** All 4 subgroup points are ≤ 0.005.
- **RG.** All 8 recipient points against NEURAL_RAW and H_LJ are ≤ 0.010.
- **E_POINT** = UG ∧ SG ∧ RG.
- **AV_X.** X is in the final bank on both seeds.
  - SS and LOC must also be MODEL_SELECT point-eligible (12 clauses and 2 guards, both seeds). An advantage over an infeasible alternative is not attributed.
  - TO needs only to be present. Its MODEL_SELECT utility and screens are printed.
- **BEN_X(s, m).** The clause bound L1 = point − c1·SE > m.
- **NW_X(s, r).** The clause bound U1 = point + c1·SE ≤ 0.010.
- **Alias.** When a slot is an eligible reference (an alias), its BEN/NW conditions are read from claim B's keys at the same c1.

### E: eligibility (gate; spends no α)

- **E label.** `DEV_POINT_ELIGIBLE` iff E_POINT. Otherwise `DEV_POINT_INELIGIBLE`, with the failed keys reported in the order UG → SG → RG.
- **Separate fields, always reported:**
  1. MODEL_SELECT development eligibility and the selection label (`EDGE_SELECTED` or `ROBUST_SELECTED`);
  2. DEV_REPORT point status: 24 + 4 + 8;
  3. DEV_REPORT simultaneous utility confidence on the primary family, over the 28 utility and subgroup endpoints: `CONFIDENCE_FEASIBLE` (all PASS), `ESTABLISHED_UTILITY_VIOLATION` (some family L > cap) or `CONFIDENCE_UNRESOLVED`, with counts;
  4. recipient confidence from claim B's 8 clause bounds, as PASS / UNRESOLVED / ESTABLISHED_VIOLATION counts;
  5. the `REPLAY_OF_PUBLISHED` and `IDENTICAL_RELEASE` flags.

  A point pass is not established population preservation.

### B: competitive against every eligible reference (α / n_claims)

- **Conjunction.** `SUPERIOR_TO_ALL_ELIGIBLE` = CV ∧ E_POINT ∧ ⋀_{X ∈ REF, s} BEN_X(s, 0) ∧ ⋀_{X ∈ REF, s, r} NW_X(s, r).
- **Material target.** `MATERIAL_SUPERIORITY_ESTABLISHED` adds ⋀_{X, s} BEN_X(s, 0.020). It uses the same bound, so no extra α is spent. It is reported separately as "positive" or "beyond the original 0.020 target".
- **Failure label.** The first failing item in this order: `CONTROL_FAILURE` → `FINALIST_FAILS_DEV_UTILITY_GATE` → `FINALIST_FAILS_DEV_SUBGROUP_POINT_GUARD` → `FINALIST_FAILS_DEV_RECIPIENT_POINT_GATE` → `NOT_SUPERIOR_TO_ALL_ELIGIBLE`. The full condition vector is reported, with categories (§5).
- **Per-reference legs** are condition values, not separate conclusions.

### M: mechanism attribution (α / n_claims; strict fixed sequence TO → SS inside the slice)

- **Step 1.** `M_TO` = CV ∧ E_POINT ∧ AV_TO ∧ ⋀_s BEN_TO(s, 0) ∧ ⋀_{s, r} NW_TO(s, r).
- **Step 2,** tested only if M_TO holds. `M_SS` = AV_SS ∧ ⋀_s BEN_SS(s, 0) ∧ ⋀_{s, r} NW_SS(s, r).
  - For B, m > 1 coupled, BEN_SS is the income-NLL utility leg. The pair difference against the twin is secondary, expected 0.
  - For aliases, the conditions are read from B's keys.
- **Labels:**
  - `MECHANISM_ATTRIBUTION_ESTABLISHED` (M_TO ∧ M_SS);
  - `TASK_ONLY_ATTRIBUTION_ONLY` (M_TO only; M_SS's failure category is printed);
  - `NOT_ESTABLISHED` (M_TO fails, and SS is `NOT_TESTED_SEQUENCE_STOPPED`).
- **Mandatory qualifiers:**
  - the F − TO and F − SS income and occupation NLL differences;
  - for families B and C2: "the privacy-trained components are the replication's stored Stage B corrections".
- **Wording.** "On DEV_REPORT, on both seeds, the declared readers recover SEX from F's linked pair less well than from [TO]'s." It is never "the mechanism removes SEX information".

### J: joint-design advantage (α / n_claims, only if AV_LOC holds at the freeze; otherwise `NOT_CLAIMED` and no keys)

- **Conjunction.** `JOINT_ADVANTAGE_ESTABLISHED` = CV ∧ E_POINT ∧ AV_LOC ∧ ⋀_s BEN_LOC(s, 0) ∧ ⋀_{s, r} NW_LOC(s, r).
- **Mandatory qualifier.** The F − LOC income NLL and Brier differences: the cost of the joint component. On MODEL_SELECT, B3CTL spends about 0.004 more income Brier than B2CTL_UNCO.
- **B3CTL wording.** "At the same six-level coarsening and calibration, replacing half of the LOCAL_TREE income statistic by the replication's pair-aware JOINT_TREE statistic lowers the declared readers' pair recovery." The joint objective is inherited, not designed here.

### N: novelty, `NOT_CLAIMED` (every family)

Each family's ingredients are known:
- output coarsening and histogram calibration (B uncoupled);
- stacking (B coupled);
- marginalising the protected characteristic (Pope & Sydnor 2011; C);
- adversarial or constrained training (Zhang et al. 2018; Cotter et al. 2019; FAGTB; A).

The privacy-trained components of B and C2 are inherited.

### IBSI comparison: `DESCRIPTIVE` (my call)

It is reported as a secondary key with a simultaneous interval and spends no α. Reasons:
1. **IBSI_NLLM is not an eligible comparator.**
   - On MODEL_SELECT it fails the seed-1 men's guard (0.00501 > 0.005) and the occupation recipient screen against H_LJ (+0.039 / +0.018).
   - The adapted PUB_IBSI γ = 2 failed 3 DEV clauses on seed 1 in the source.
   - A benefit over an ineligible release is not a competitiveness result.
2. **It is matched to SCM, not to F.** It was matched to SCM's MODEL_SELECT occupation NLL, so a difference is not cost-matched.
3. **Reader asymmetry.** It is a discrete release (about 470 distinct pairs) that faces a nearly Bayes-optimal LOOKUP, while F's pair is continuous. A pair difference therefore mixes reader classes.

### Subgroup

- There is no separate claim.
- E carries the 4 guard points and their family statuses.
- Within-SEX NLL is descriptive. For the JOINT_TREE occupation output, the MODEL_SELECT seed-0 men's occupation NLL excess against U0 is +0.017.

---

## 3. Endpoint roster

### Primary family

Per seed. The clause bound is c1; the family interval is c_P.

| Key (suffix `\|s{s}`) | Kind | Margin | Claim | Count |
|---|---|---|---|---|
| `UTILITY\|{income,occupation}\|{U0,Ucal}\|{accuracy_loss,nll_excess,brier_excess}` | no_excess | 0.010 / 0.010 / 0.005 | E gate | 12 |
| `SUBGROUP\|occupation_brier_excess_vs_U0\|sex{0,1}` | no_excess | 0.005 | E gate | 2 |
| `COMPETITIVE\|pair_benefit_vs_{NEURAL_RAW,H_LJ}` | benefit | 0; reference 0.020 | B | 2 |
| `COMPETITIVE\|{income,occupation}_worsening_vs_{NEURAL_RAW,H_LJ}` | no_excess | 0.010 | B (points: E) | 4 |
| `MECH_TO\|pair_benefit_vs_TO` and `MECH_TO\|{income,occupation}_worsening_vs_TO` | benefit / no_excess | 0 / 0.010 | M, step 1 | 3 |
| `MECH_SS\|pair_benefit_vs_SS` (B coupled: `income_nll_benefit_vs_SS`) and `MECH_SS\|{income,occupation}_worsening_vs_SS` | benefit / no_excess | 0 / 0.010 | M, step 2 | 3 × σ |
| `JOINT\|pair_benefit_vs_LOC` and `JOINT\|{income,occupation}_worsening_vs_LOC` | benefit / no_excess | 0 / 0.010 | J | 3 × ι |

Here σ = 1 if SS is a distinct release (0 if it aliases a reference), and ι = 1 if J is claimed.

**K_P = 2 × (23 + 3σ + 3ι):**

| Finalist type | σ | ι | K_P |
|---|---|---|---|
| **B3CTL** (SS = H_LJ alias; J against B2CTL_UNCO) | 0 | 1 | **52** |
| B, m > 1 coupled (SS = twin utility leg; J not claimed) | 1 | 0 | 52 |
| C | 1 | 0 | 52 |
| B, m = 1 | 0 | 0 | 46 |
| A, with a distinct SS and a banked LOC | 1 | 1 | 58 |
| A, without LOC | 1 | 0 | 52 |

### Secondary family

Per seed, 10 keys of kind `difference`, giving **K_S = 20**:
1. `SEC|F_pair_minus_F_occupation`: the residual coalition synergy. For B, m = 1 the population value is 0.
2. `SEC|F_pair_minus_F_income`.
3. `SEC|IBSI_NLLM_minus_F|pair`: the IBSI comparison (descriptive, flagged COMPARATOR_INELIGIBLE).
4. `SEC|income_nll_diff_vs_NEURAL_RAW` and `SEC|income_nll_diff_vs_H_LJ` (2 keys).
5. `SEC|occupation_nll_diff_vs_NEURAL_RAW` and `SEC|occupation_nll_diff_vs_H_LJ` (2 keys). The H_LJ key is `IDENTICAL_RELEASE` when the occupation output is reused.
6. `SEC|income_nll_diff_vs_TO`.
7. `SEC|income_nll_diff_vs_LOC`. If J is not claimed, this is `income_brier_diff_vs_SS` instead.
8. `SEC|FAMILY_SLOT`:
   - A and B3CTL: `H_JJ_minus_F|pair`, the share of the fully pair-aware reference's protection that is retained (H_JJ is COMPARATOR_INELIGIBLE but computed);
   - B coupled: `TWIN_minus_F|pair`, flagged INFORMATION_IDENTICAL, a representation check expected to be 0;
   - C: `SEXBLIND_INCOME_HYBRID_minus_F|pair` if banked, otherwise MISSING.

### Outside both families (`MARGINAL_DESCRIPTIVE`, marginal 95 %)

These carry no status and take no part in any claim:
- F − each banked comparator for accuracy, NLL and Brier, both tasks;
- within-SEX occupation Brier and NLL;
- the remaining per-recipient AUC differences;
- per-reader AUCs and single-output alternatives;
- distinct-output counts, classwise NLL and predicted-class frequencies.

### Bank and fixed counts

- **B3CTL bank, per seed (7 effective releases, at most 8):** F, NEURAL_RAW, H_LJ, TO, B2CTL_UNCO, IBSI_NLLM, H_JJ.
- **Fixed K.** Every key is always reported and stays in its K, whatever the coverage.

---

## 4. Inference and multiplicity

**Bootstrap.**
- Weights: `task_headroom.inference.group_weights(DEV_REPORT unit, seed = 2026101705, draws = 2048)`, shared across releases, endpoints and seeds.
- Halt unless G = 3955 households and n = 5943 persons.
- **Seed.** 2026101705 is the SCM follow-up's registered seed, reused because it is fixed before any final fit and is never re-drawn.
  - It makes a verifier check possible. NEURAL_RAW's and H_LJ's frozen outputs give utility and subgroup endpoints that must reproduce the published values and SEs exactly.
  - If their reader scores replay bitwise, their AUCs and SEs must reproduce too.
- **AUC replicates.** `task_conditioned_representation.inference.bootstrap_auc(sex, score, W)`. An AUC difference is the difference of the two weighted AUCs within each replicate.
- **Loss replicates.** A loss difference is Σ W_b,i d_i / Σ W_b,i. For subgroups, the weights are restricted to that subgroup's persons.
- **Bounds.** SE = replicate SD (ddof 1); Wald bounds point ± c·SE; df = G − 1 = 3954 throughout, including subgroups.

**α allocation.**
- One-sided α = 0.025, split equally across n_claims slices. n_claims is the number of claims among B, M and J that spend α, fixed at the freeze: 3 if J is claimed (B3CTL, or A with LOC), otherwise 2.
- B and J are intersection-union tests. M is a strict fixed sequence TO → SS.
- E, the UTILITY and SUBGROUP statuses, the secondary family and the nested 0.020 target spend no claim α.

**Critical values** (reproduced; the source checks 2.4988491, 3.3346075, 3.0801279 and 1.9605641 match):

| Rule | n_claims or K | Value |
|---|---|---|
| clause c1 = t(1 − 0.025/n) | n = 3 | 2.3949991 |
| | n = 2 | 2.2422567 |
| primary c_P = t(1 − 0.05/(2K_P)) | K_P = 52 | 3.3040310 |
| | K_P = 46 | 3.2694077 |
| | K_P = 58 | 3.3346075 |
| secondary c_S | K_S = 20 | 3.0252810 |
| marginal descriptive | — | 1.9605641 |

**Scope.**
- Results are conditional on the fitted releases and readers.
- They are not ACS design-based: no survey or replicate weights.
- They exclude the adaptive development search.
- The two seeds share people and bootstrap draws.

---

## 5. Status and label vocabulary

**Endpoint statuses** (family interval [L, U] at c_P or c_S):

| Kind | Statuses |
|---|---|
| benefit | ESTABLISHED (L > m), ESTABLISHED_SHORTFALL (U < m), UNRESOLVED |
| no_excess | PASS (U ≤ m), ESTABLISHED_VIOLATION (L > m), UNRESOLVED |
| difference | ABOVE_ZERO, BELOW_ZERO, UNRESOLVED |

- **Clause fields.** L1 and U1 at c1 exist only for AVAILABLE benefit and no_excess keys in the primary family.
- **Availability.** AVAILABLE, COMPARATOR_INELIGIBLE or MISSING.
- **Flags.** IDENTICAL_RELEASE, ALIAS_OF_COMPETITIVE, REPLAY_OF_PUBLISHED, INFORMATION_IDENTICAL. Only AVAILABLE keys can satisfy a condition.

**Condition failure categories.** Each false condition gets exactly one:
- **UNAVAILABLE.** The comparator is missing or ineligible, `NOT_TESTED_SEQUENCE_STOPPED` or `NOT_CLAIMED`. This says nothing about F.
- **POINT_FAILURE.** A gate (UG, SG or RG) fails at the point. E fails whatever the intervals show.
- **Failed benefit clause (L1 ≤ m):**
  - ESTABLISHED_SHORTFALL if the family U < m (the data exclude the benefit);
  - otherwise POINT_SHORTFALL if point ≤ m;
  - otherwise UNRESOLVED.
- **Failed no-excess clause (U1 > m):**
  - ESTABLISHED_VIOLATION if the family L > m;
  - otherwise POINT_EXCESS if point > m;
  - otherwise UNRESOLVED.

**Merging and wording rules.**
- Categories are never merged in text.
- UNRESOLVED is never written as "equivalent", "no difference", "no worsening" or "no benefit".
- L1 > 0 but not > 0.020 is written "positive, below the original 0.020 target".

**Mandatory qualifiers on every claim:**
- the selection label (`EDGE_SELECTED` for B3CTL);
- the E fields;
- REPLAY and IDENTICAL flags;
- each comparator's DEV 12-clause point status;
- "exploratory development evidence on previously used records".

---

## 6. Must not be claimed

- **Reader results are not guarantees.** AUCs are evidence against the declared 8-reader bank under the 8,001-person ATTACK_FIT budget only. Pair AUCs near 0.60 mean SEX is still inferable.
- **No injectivity from determinism.** Deterministic outputs are not assumed one-to-one in the inputs.
  - Information-identity statements are made only where the construction makes them exact. For B with m = 1, the pair and the occupation output determine each other.
  - Where the identity depends on distinct calibrated levels, those must be checked in the release record.
  - "log m" bounds are vacuous for m ≥ 2, since I(SEX; ·) ≤ log 2, and are not cited.
- **No equivalence from UNRESOLVED,** and no "matched" comparison without a registered matching gate.
- **LOOKUP is not population-optimal.**
  - It is degenerate (≈ 0.50) on continuous pairs.
  - Its strength on coarse outputs makes the single-output comparisons conservative for F, not optimal.
- **Two seeds are the same people.** "Both seeds" means robustness to the model seed on the same people with the same bootstrap draws, not replication.
- **Intervals exclude the adaptive search:**
  - 42+ development evaluations, revisions and controls;
  - the promotion and its adjudication;
  - α = 0.5 and m = 6 chosen with a MODEL_SELECT-derived offset;
  - reader selection;
  - the earlier campaigns' use of MODEL_SELECT and DEV_REPORT.

  DEV_REPORT is not historically unseen. F's occupation output and H_LJ's DEV results were published beforehand.
- **No MODEL_SELECT margin is quoted as an effect size** (winner's curse on one fixed split).
- **No coupling claim** for any B finalist, and no claim that this campaign designed the privacy-trained components it reuses.
- **No subgroup claim** beyond the registered occupation Brier guard. **No fairness claim.** **No novelty claim.**
- **No sealed-assessment opening** follows from any outcome.

---

## 7. To resolve at the freeze (blocking)

1. **Finalist.** By the registered §6 text (B3CTL), or a declared deviation with both finalists printed.
2. **Controls.** TO fitted and banked (2 configurations). SS, LOC, IBSI_NLLM and H_JJ banked, with at most 8 effective releases per seed after alias deduplication.
3. **Counts.** σ, ι, n_claims, K_P and K_S fixed and asserted by the code.
4. **Injectivity.** For every INFORMATION_IDENTICAL statement: distinct calibrated levels, and distinct bin coefficients of h.
5. **Bets.** Registered before the REPORTING_LOCK, with a probability each: E point on DEV (the income Brier margin is 0.87–0.88 of cap), B, M_TO and J.
6. **Reader bank.** 8 readers including the corrected LOOKUP, single-output alternatives, and the planted / shuffled control bands, all frozen before any ATTACK fit.
