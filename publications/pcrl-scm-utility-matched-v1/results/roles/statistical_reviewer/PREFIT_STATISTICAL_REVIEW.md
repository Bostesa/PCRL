# Pre-fit statistical review: SCM utility-matched follow-up

**Role.** Statistical-design reviewer. This document freezes the signs, comparison counts and multiplicity allocation before the new fitting stage. Written 2026-10-17, before any new trajectory, MODEL_SELECT label read or reader fit.

**This version** includes my decisions on the coordinator's note about the math reviewer's `PREFIT_REVIEW.md` (§0). Those decisions change the frozen counts from the draft's 64 / 16 to **K_primary = 58 and K_secondary = 24**.

**What I reviewed.**
- `results/pcrl_scm_utility_matched_v1/PROTOCOL.md` (SHA-256 `28657d84da44924edfb4526c4e9bac47a61e2c21f7162a5fedceb82c511b3a0a`), mainly §10, §11 and §13, plus the statistical parts of §5 to §8.
- The draft code that implements it, as it stood at 01:07 local time: `scm_um/make_lock.py` (rosters and inventory), `scm_um/report.py` (endpoints, statuses, labels) and `scm_um/select.py` (role-status vocabulary only).
- The math reviewer's `roles/math_reviewer/PREFIT_REVIEW.md`.
- The source conventions: `results/pcrl_scm_engineering_benchmark_v1/PROTOCOL.md` §9 and its post-run clarifications, `scm_bench/report.py`, `scm_bench/make_lock.py`, `task_headroom.inference.group_weights`, `task_conditioned_representation.inference.bootstrap_auc`, and the source's public aggregate results (`report/DEV_REPORT_RESULTS.json` header, `PRIMARY_ENDPOINTS.csv`, `SECONDARY_ENDPOINTS.csv`).

**What I did not do.**
- I ran no numerical job and loaded no person-level array.
- I did not read `.private/` and made no commit.
- I wrote only into this folder.
- The only computations were closed-form t quantiles, using G = 3955 from the source's public results. As a check they reproduce the source's recorded critical values exactly (2.3949991, 3.2435114, 2.8086105).

**Verdict.** The structure is sound, but seven must-fix items block the freeze.
- **Sound:**
  - the intersection-union conjunctions;
  - four α/4 claim slices;
  - nested margins;
  - the source bootstrap and paired conventions;
  - point gates with separate simultaneous statuses;
  - fixed K, with unavailable endpoints still counted.
- **Must-fix items:**
  - MF-1 is a validity defect. As drafted in `report.py`, the two matched components are tested in parallel, not in sequence.
  - MF-2 to MF-7 add precision that the commissioning requirements need:
    - the match semantics;
    - the availability vocabulary;
    - the failure categories;
    - separate eligibility fields;
    - the 0.020 target in the decision;
    - the secondary roster.

## 0. Decisions on the coordinator's note (math reviewer's proposals)

### (a) One-sided DEV carry-over gate (math reviewer M1)

**Decision.**
- **Not accepted** for the labels.
- **The coordinator's plan is accepted**, with the definitions below. I see no statistical problem with it.

**Why the one-sided gate is not adopted for labels.** The level is the same either way. Any gate added to an intersection-union conjunction only shrinks the rejection region, so P(label established | any null) ≤ α/4 under both the one-sided and the two-sided gate. The choice is therefore about what "matched" means, and the commissioning text fixes that:
- a drift outside the tolerance "did not carry over";
- the tolerance is never widened.

A one-sided gate widens the tolerance to infinity on the comparator-spent-more side, so the M label keeps the registered two-sided gate.

**Why the "conservative for SCM" reading is not taken up.** It rests on an unverified assumption: that further ordinary merging lowers fresh-reader AUC monotonically. Fitted readers on 8,001 ATTACK_FIT people need not behave that way (see the math reviewer's own S4 on finite-reader effects). The sub-status below therefore states the direction of spending and does not assert conservativeness.

**Frozen rule.** For each matched role R ∈ {ORD_NLLM, IBSI_NLLM, SHUF_NLLM} and seed s whose match endpoint is AVAILABLE, let d_s be the DEV_REPORT point of SCM − R occupation NLL.
- **Label gate (unchanged, two-sided).** MATCH_DID_NOT_CARRY_OVER iff some seed has |d_s| > 0.001.
- **Descriptive sub-status, always reported for every seed and every direction:**
  - `CARRIED_OVER` if |d_s| ≤ 0.001;
  - `SCM_SPENT_MORE` if d_s > +0.001 (the confounded direction);
  - `COMPARATOR_SPENT_MORE` if d_s < −0.001.
- **Reporting when the gate fails.** The benefit and worsening clauses are still reported as "comparison-only, not cost-matched on DEV_REPORT", together with the sub-status vector. They spend no α. They are never called matched or established. The research decision stays COMPARISON_CLOSED_INCONCLUSIVE_AT_REGISTERED_TOLERANCE, with the sub-status as a qualifier.
- **Components.** The same applies to the IBSI and SHUF components.
- **P4** is scored as registered (two-sided).

**Note.** The strongest-proxy rule tends to pick ORD near the high-spending edge of the window, so MATCH_DID_NOT_CARRY_OVER is a likely registered outcome (P4 = 0.35). That is a known consequence of the commissioned design and is not repaired.

### (b) Move the six occupation-NLL match endpoints to the secondary family; add the income-NLL difference vs ORD_NLLM

**Decision: accepted. Final counts: K_primary = 58, K_secondary = 24.**

**Why moving them is right.** The labels use only the points of the match endpoints, which do not depend on the family. Their intervals serve only the descriptive equivalence status, which is effectively unreachable (N1).
- Keeping them in the primary family would only widen the simultaneous intervals of the endpoints that do matter for claims and statuses: benefit, worsening, utility and subgroup.
- With 58 in the primary family, c_P falls from 3.3620 to 3.3346.
- In the secondary family (K = 24, c_S = 3.0801), the commissioner's required paired interval is still reported simultaneously.

**The income-NLL difference.** SCM − ORD_NLLM income NLL per seed (2 endpoints, kind `difference`) is accepted as secondary. It makes the math reviewer's S3 point (the confound can move to income) visible with a simultaneous interval.

**Per seed:**
- primary: 3 MATCH + 2 MATCH_COMPONENT + 4 CAPACITY + 6 COMPETITIVE + 12 UTILITY + 2 SUBGROUP = 29;
- secondary: 8 + 3 match + 1 income-NLL = 12.

### (c) NOMINEE_IS_C0 (math reviewer M4)

**Decision: accepted.**

**Rule.** NOMINEE_IS_C0 holds iff, in some seed, the nominee's effective release equals C0 (bit-identical per-initial-cell tables).
- **When it is fixed.** It is decided from SELECTION_LOCK before DEV_REPORT.
- **Where it sits.** In the label order of M, K and B, it comes immediately after NOMINEE_INELIGIBLE. As a pre-DEV gate it only shrinks rejection regions, so the levels are unchanged.
- **Consequences:**
  - components and IBSI_CNT are NOT_TESTED;
  - claim C is still reported, qualified as being about the C0 release;
  - the research decision is COMPARISON_CLOSED_NOMINEE_IS_C0.

### Other math-reviewer items with statistical content

- **M3 (availability per seed for ORD, per role for IBSI/SHUF):** consistent with MF-3 below and adopted.
- **S3 (the `INCOME_EXPENDITURE_NOT_MATCHED` qualifier):** acceptable as a descriptive qualifier. It spends no α and does not change any label.
- **S4 (learning curve):** acceptable only as descriptive output. It must stay outside both rosters, all labels and the research decision.

## 1. Checks that pass

1. **The draft's counts are correct by enumeration.**
   - Primary: per seed, MATCH 4, MATCH_COMPONENT 4, CAPACITY 4, COMPETITIVE 6, UTILITY 12 and SUBGROUP 2 give 32, so 64.
   - Secondary: 8 per seed, so 16.
   - `scm_um/make_lock.py::rosters()` produces exactly those keys and asserts 64/16.
   - After decision (b) the frozen counts are 58 / 24, enumerated key by key in the FROZEN DESIGN. The code's assertion must change to 58 / 24.
2. **Intersection-union logic.** Each label is established only if every component one-sided test rejects at α/4. That is a level-α/4 intersection-union test of the union null, with no adjustment inside the conjunction.
   - Point gates (UG, SG, NOMINEE_IS_C0, DEV point match) are extra necessary conditions. They only shrink the rejection region.
   - Requiring both seeds is also an intersection-union condition. It does not need the seeds to be independent, and they are not: same people, same bootstrap draws.
3. **Nested 0.020.** "L1 > 0, then L1 > 0.020" reads one one-sided confidence bound at two margins. Rejecting at 0.020 implies rejecting at 0, so no extra α is spent.
4. **Separate α slices.** M, K, B and C each get one-sided α/4 = 0.00625, so the familywise one-sided error across claims is at most 0.025 (Bonferroni). It is conservative under the positive dependence between M and K.
5. **The K-internal sequence is valid.** IBSI_CNT is tested only when K is established, so it is a two-step fixed sequence.
6. **Source conventions are followed.**
   - Bootstrap: household bootstrap B = 2048 with seed 2026101705, from the same `group_weights` function, shared across releases, endpoints and seeds.
   - Bounds: Wald bounds with t(G − 1).
   - Families: two-sided Bonferroni-t families.
   - Claim clauses: one-sided at α / n_claims.
   - Confidence status: defined on the family interval.
   - Subgroup claim: SUBGROUP_QUALIFIED on clause bounds and SUBGROUP_HARM on family bounds.
   - `report.py` implements all of these as in the source.
7. **Point gates with separate confidence statuses follow the source.** Claims A and B were gated on the DEV 24-point clauses (B per the frozen code; source clarification 1), and confidence was reported on the family interval.
   - The SG gate and the conservative nomination are a declared deviation (§17 item 6), in the conservative direction, and they are what the commissioner asked for.
   - An established violation (family L > cap) implies a point failure, because L ≤ point. The point gates therefore already exclude every established utility or subgroup violation.
8. **Unavailable endpoints stay in K.** This is conservative: K is fixed whatever the coverage. In `report.py`, the clause checks `ok_L` and `ok_U` require availability `AVAILABLE`, so a non-matched endpoint can never satisfy a claim condition.
9. **Reusing the source bootstrap seed is endorsed** (see N5). Any other seed would be an unregistered choice.
10. **Signs are consistent** with the source and the commissioning requirements:
    - benefit = comparator AUC − SCM AUC;
    - worsening = SCM AUC − comparator AUC, with non-worsening iff ≤ 0.010;
    - NLL difference = SCM − comparator.

**Critical values (frozen).** G = 3955 DEV_REPORT households and 5,943 persons, as in the source; df = 3954.

| Rule | Formula | Value |
|---|---|---|
| Claim clause, one-sided α/4 | t_{G−1}(1 − 0.025/4) | 2.4988491 |
| Primary family, two-sided 95 %, K = 58 | t_{G−1}(1 − 0.05/116) | 3.3346075 |
| Secondary family, two-sided 95 %, K = 24 | t_{G−1}(1 − 0.05/48) | 3.0801279 |
| Marginal 95 %, descriptive only | t_{G−1}(0.975) | 1.9605641 |

## 2. MUST-FIX (blocking)

### MF-1. Matched components must form a strict fixed sequence; the draft code tests them in parallel

**The problem.** `report.py::component()` tests IBSI and SHUF independently whenever M is established. SHUF can therefore be declared PASS after IBSI has failed or was not testable.
- Two hypotheses tested at α/4 each after one gatekeeper is not a fixed sequence.
- When M truly holds and both component nulls are true, the slice's familywise error can reach α/2.
- §11 calls this a "fixed sequence" but never states the stop rule.

**Proposed §11 text.** It replaces the "Matched components" paragraph:

> **Matched components (inside M's α/4 slice; strict fixed sequence M → IBSI → SHUF).**
> - The IBSI component is tested only if M is MATCHED_ADVANTAGE_ESTABLISHED. Otherwise both components are NOT_TESTED_M_NOT_ESTABLISHED, and their endpoints are descriptive.
> - IBSI takes one of four statuses:
>   - UNAVAILABLE: IBSI_NLLM is not MATCHED on both seeds;
>   - MATCH_DID_NOT_CARRY_OVER: some seed's DEV_REPORT point |SCM − IBSI_NLLM occupation-NLL difference| > 0.001, reported with the per-seed carry-over sub-status of §0(a);
>   - PASS: the benefit clause L1 > 0 on both seeds;
>   - FAIL.
> - The SHUF component is tested only if IBSI is PASS. Otherwise SHUF is NOT_TESTED_SEQUENCE_STOPPED, and its endpoints are descriptive. When tested, SHUF takes the same four statuses.
> - Overall: COMPONENTS_ESTABLISHED iff IBSI = PASS and SHUF = PASS. Otherwise COMPONENTS_NOT_ESTABLISHED, naming the first component that did not pass.
> - Shuffled controls are component comparisons, not permutation tests.
> - A component PASS is a statement about the linked pair at the achieved match. It is never described as a recipient-level or competitive result.

**Code change.** In `component()` for SHUF, return `'NOT_TESTED_SEQUENCE_STOPPED'` unless the IBSI status is `'PASS'`, and add `MC['decision']`.

**Why "stop on anything but PASS".** Skipping a component that was UNAVAILABLE at MODEL_SELECT would also be valid, because that event is fixed before DEV_REPORT. Skipping one that failed MATCH_DID_NOT_CARRY_OVER would not, because that event depends on DEV_REPORT data. The single strict rule is adopted to avoid a two-case rule.

### MF-2. Define the `match` endpoint kind exactly (now in the secondary family)

§11 calls an interval "outside ±0.001" and gives no estimator. Proposed text, replacing the "match" clause of "Endpoint statuses" and extending §10:

> **Occupation-NLL match endpoints (kind `match`; secondary family).**
> - **Estimand.**
>   - Per person: d_i = NLL_i(SCM) − NLL_i(comparator), with NLL_i = −log max(p_{i,y_i}, 1e-12) on the released occupation vector, over all DEV_REPORT persons.
>   - Point: mean_i d_i.
>   - Replicate b: Σ_i W_bi d_i / Σ_i W_bi, using the shared household weights.
>   - SE: the replicate standard deviation (ddof 1).
>   - Family interval: [L, U] = point ∓ t_{G−1}(1 − 0.05/48) × SE (K_secondary = 24).
> - **Point status.** POINT_MATCH iff |point| ≤ 0.001 (inclusive, float64, no slack); otherwise POINT_MISMATCH. The per-seed carry-over sub-status is that of §0(a).
> - **Interval status.**
>   - EQUIVALENCE_ESTABLISHED iff L ≥ −0.001 and U ≤ 0.001.
>   - MISMATCH_ESTABLISHED iff L > 0.001 or U < −0.001.
>   - Otherwise EQUIVALENCE_UNRESOLVED.
> - **No clause bound.** A match endpoint has no clause bound, so its clause fields are null.
> - **Labels use only the point.** The M and component gates read only the point of this secondary endpoint, and they spend no α.
> - **Achieved match.** Beside each endpoint the report prints the MODEL_SELECT mismatch actually achieved, e_s(SCM) − e_s(comparator), from SELECTION_LOCK (`delta`).
> - **Wording.** "Matched" in any label means the DEV_REPORT point match only. POINT_MATCH with EQUIVALENCE_UNRESOLVED is written "point-matched; equivalence not established".

**Code status.**
- `status_at('match', …)` already implements these interval statuses.
- `endpoint()` must set `clause_L1`/`clause_U1` to null for `match` and `difference` kinds.
- The `at` statuses for a match endpoint currently reuse the `difference` names (`ABOVE_ZERO` at ±0.001). Drop them, or name them `ABOVE_REF`/`BELOW_REF`.
- The `carry` and `component()` gates must read these endpoints from the secondary record.

### MF-3. One availability vocabulary per endpoint, fixed before DEV_REPORT

**The problem.**
- §11 says "MISSING (no matched release)", which is inaccurate. When there is no match but a nearest diagnostic exists, the endpoint is DIAGNOSTIC_NOT_MATCHED, not MISSING.
- Endpoints against an ineligible comparator currently read `AVAILABLE`. `report.py::availability()` maps an IBSI_CNT `SAME_COUNT_INELIGIBLE_FALLBACK` and an ORD_CNT that fails its clauses to `AVAILABLE`, so an endpoint table can show a bare ESTABLISHED.

**Proposed text.** It replaces the §11 sentence that begins "An endpoint whose comparator role is unavailable…":

> Every one of the 58 primary and 24 secondary keys is always reported and stays in its K. Each endpoint has one availability, fixed before DEV_REPORT from SELECTION_LOCK and the bank manifest:
> - **AVAILABLE.**
> - **DIAGNOSTIC_NOT_MATCHED:** the role holds its nearest diagnostic. This is decided per seed for ORD_NLLM, and for both seeds together for the common-γ and common-r roles IBSI_NLLM and SHUF_NLLM.
> - **COMPARATOR_INELIGIBLE:** ORD_CNT fails that seed's 12 MODEL_SELECT point clauses, or IBSI_CNT's γ is the ineligible fallback.
> - **MISSING:** the role is empty, because no candidate passes its seed's clauses or ORD never reaches K*_s.
>
> Endpoints that are not AVAILABLE are still computed whenever a release exists. They:
> - carry their availability as a status prefix;
> - have null clause fields;
> - never satisfy a claim condition or a gate.
>
> A secondary endpoint takes the least available of its two roles' availabilities.

This matches the role statuses `select.py` writes: MATCHED, NEAREST_DIAGNOSTIC_NOT_MATCHED, MISSING, SAME_COUNT, SAME_COUNT_INELIGIBLE_FALLBACK and UNMATCHED_CAPACITY.

### MF-4. Define the failure categories the commissioner requires

§11 says point failures, established shortfalls, unresolved intervals and unavailable comparisons are "reported as distinct categories" but does not define them. Proposed text, added after "Endpoint statuses":

> Each claim reports every condition of its conjunction with its truth value. A false condition gets exactly one category:
> - **UNAVAILABLE:** MISSING, DIAGNOSTIC_NOT_MATCHED, COMPARATOR_INELIGIBLE, UNMATCHED_CAPACITY, MATCH_UNAVAILABLE, NOMINEE_INELIGIBLE, NOMINEE_IS_C0 or NOT_TESTED_*.
> - **POINT_FAILURE:** UG or SG fails at the point, or the DEV_REPORT match point fails (with its carry-over sub-status).
> - **For a failed benefit clause (L1 ≤ m):**
>   - ESTABLISHED_SHORTFALL if the family U < m;
>   - otherwise POINT_SHORTFALL if point ≤ m;
>   - otherwise UNRESOLVED.
> - **For a failed no-excess clause (U1 > m):**
>   - ESTABLISHED_VIOLATION if the family L > m;
>   - otherwise POINT_EXCESS if point > m;
>   - otherwise UNRESOLVED.
>
> Text never merges these categories. An UNRESOLVED worsening clause is not evidence of worsening, and an UNRESOLVED benefit clause is not evidence of no benefit.

`report.py` already records the full condition vector for M, K and B. It needs a `failure_category` for each false condition.

### MF-5. Report original eligibility separately from conservative nomination eligibility

The commissioner asked for "original point utility eligibility and simultaneous confidence status". `report.py` reports only NOMINATION_ELIGIBLE / INELIGIBLE_FALLBACK for MODEL_SELECT, and the fallback may still pass the original 24 clauses. Proposed text, replacing §11 "Utility status":

> **Utility status (separate fields, always all reported):**
> 1. MODEL_SELECT original point eligibility of the bank's SCM release: ORIGINAL_POINT_FEASIBLE_24, or the failed keys. These are the 24 original clauses at caps 0.010 / 0.010 / 0.005, over both seeds.
> 2. Conservative nomination eligibility: NOMINATION_ELIGIBLE, or INELIGIBLE_FALLBACK with the failed nomination conditions; plus NOMINEE_IS_C0 when it applies.
> 3. DEV_REPORT point: POINT_FEASIBLE_24, or the failed keys.
> 4. DEV_REPORT simultaneous confidence on the K = 58 primary family: CONFIDENCE_FEASIBLE_24 / ESTABLISHED_UTILITY_VIOLATION / CONFIDENCE_UNRESOLVED, with counts.
> 5. Subgroup point status (four guards) and subgroup simultaneous statuses (family PASS / ESTABLISHED_VIOLATION / UNRESOLVED for each guard), beside the claim C label.
>
> For every bank role, the report also gives the MODEL_SELECT 12-clause point status and the DEV_REPORT 12-clause point status per seed.

### MF-6. Name the 0.020 target in every decision, and qualify the research decision

**The problem.** The source's recipe claim had 0.020 as its primary margin. Here M's primary margin is 0, with 0.020 nested. That is acceptable, because the commissioner asks for both. However:
- `research_decision` maps MATCHED_ADVANTAGE_ESTABLISHED to `ENGINEERING_REFERENCE_FROZEN` whether or not the original target was met;
- the decision text carries none of the unfavourable qualifiers.

**Proposed text.** It replaces the "Research decision (fixed mapping)" paragraph:

> **Per-seed 0.020 reporting.** Per seed, every benefit endpoint of M, K and B reports:
> - the point;
> - the clause bound L1 (one-sided α/4);
> - the family interval;
> - whether L1 > 0 and whether L1 > 0.020 (the original per-seed target).
>
> **Research decision (fixed mapping):**
> - MATERIAL_ADVANTAGE_ESTABLISHED → ENGINEERING_REFERENCE_FROZEN_AT_ORIGINAL_TARGET.
> - MATCHED_ADVANTAGE_ESTABLISHED but not material → ENGINEERING_REFERENCE_FROZEN_POSITIVE_BELOW_ORIGINAL_TARGET. The text states that the original 0.020 per-seed target was not established.
> - MATCH_UNAVAILABLE → FRONTIER_DELIVERED_MATCH_UNAVAILABLE. Missing coverage is neither a demographic-inference result nor a method win.
> - MATCH_DID_NOT_CARRY_OVER → COMPARISON_CLOSED_INCONCLUSIVE_AT_REGISTERED_TOLERANCE, with the per-seed carry-over sub-status of §0(a).
> - NOMINEE_IS_C0 → COMPARISON_CLOSED_NOMINEE_IS_C0.
> - CONTROL_FAILURE → COMPARISON_CLOSED_CONTROL_FAILURE. No recovery comparison is interpretable.
> - Otherwise → COMPARISON_CLOSED_<label>.
>
> **Mandatory qualifiers.** Every decision is printed with four qualifiers that are never omitted:
> 1. the DEV_REPORT utility confidence status (MF-5, item 4);
> 2. the claim C label;
> 3. the DEV_REPORT equivalence status and carry-over sub-status of the ORD_NLLM match on each seed;
> 4. claim B's label.
>
> A release is called competitive only under SUPERIOR_TO_ALL_ELIGIBLE. A frozen engineering reference never authorises opening the sealed assessment.

### MF-7. Replace the secondary-roster sentence (ambiguous, and changed by decision (b))

**The problem.** The §11 wording "C0 − SCM, C0 − ORD_NLLM, HIST_SCM − SCM, ORD_CNT − ORD_NLLM and ORD_NLLM − SCM on the inherited 7-reader slate" can be read as applying the slate to all five differences. `make_lock.py` applies it only to the last, which mirrors the source's `TWIN_minus_SCM_slate7`.

**Proposed text.** It replaces the "Secondary" sentence and the MATCH / MATCH_COMPONENT rows of the primary table:

> **Secondary (K = 24; per seed 12):**
> - pair-AUC differences C0 − SCM, C0 − ORD_NLLM, HIST_SCM − SCM and ORD_CNT − ORD_NLLM, on the full 8-reader bank (24 pair candidates);
> - the pair-AUC difference ORD_NLLM − SCM, on the inherited 7-reader slate (7 pair readers and 14 ignore-one candidates, winner on ATTACK_VALIDATION);
> - occupation-NLL differences SCM − ORD_CNT and SCM − IBSI_CNT;
> - SCM pair AUC − SCM occupation-recipient AUC (full bank);
> - the occupation-NLL match endpoints SCM − ORD_NLLM, SCM − IBSI_NLLM and SCM − SHUF_NLLM (kind `match`; their points are the claim gates);
> - income-NLL difference SCM − ORD_NLLM.
>
> Kind `difference` has statuses ABOVE_ZERO / BELOW_ZERO / UNRESOLVED on the K = 24 family interval. Kind `match` follows MF-2. Neither kind has clause fields.
>
> **Primary MATCH and MATCH_COMPONENT rows:**
> - MATCH (claim M): coalition benefit vs ORD_NLLM; income and occupation worsening vs ORD_NLLM.
> - MATCH_COMPONENT (inside M): coalition benefit vs IBSI_NLLM and vs SHUF_NLLM.
>
> The primary family has 29 endpoints per seed (K_primary = 58).

## 3. SHOULD-FIX

- **S1. Pairing wording (§11).** Replace "paired differences are computed person-level within each replicate" with:

  > every replicate applies one household-weight vector to every release and endpoint; an AUC difference is the difference of the two weighted AUCs within a replicate; loss and subgroup differences are weighted means of person-level differences within a replicate.
- **S2. Bootstrap identity.**
  - Name the functions: `task_headroom.inference.group_weights` and `task_conditioned_representation.inference.bootstrap_auc`.
  - Assert G = 3955 households and 5,943 persons; on a mismatch the report halts.
  - The verifier checks that the bootstrap weights reproduce the source's. For every release whose DEV_REPORT scores replay the source bitwise (the label-free replay of HIST_SCM, C0, NEURAL_RAW and H_LJ), its AUC point and SE must equal the source `DEV_REPORT_RESULTS.json` values exactly.
- **S3. Identical-release flag.** When both releases of a difference are one effective release in a seed (`service_of_role` equal), flag the endpoint `IDENTICAL_RELEASE`. Its point and SE are exactly 0: a benefit clause cannot pass, but a match endpoint would show a trivially true EQUIVALENCE_ESTABLISHED. This is most likely for HIST_SCM − SCM if t = 0.010 were nominated; the C0 case is covered by NOMINEE_IS_C0.
- **S4. Comparator DEV feasibility as a qualifier.** Comparator eligibility stays MODEL_SELECT-based (source convention). Each comparison prints the comparator's DEV 12-clause point status per seed. If a comparator fails a DEV point clause, the label carries "comparator fails DEV point clause(s): <keys>" and is not otherwise changed.
- **S5. "All other task and subgroup results" for each comparison.** These cover ORD_NLLM, IBSI_NLLM, SHUF_NLLM, ORD_CNT, IBSI_CNT, NEURAL_RAW and H_LJ:
  - What is reported: paired SCM − comparator differences for income and occupation accuracy, NLL and Brier; within-SEX occupation Brier and NLL; and the per-recipient AUC differences that are not already primary.
  - Form: point and SE, with a marginal 95 % interval (t(G−1), 0.975) labelled `MARGINAL_DESCRIPTIVE`.
  - These carry no status, take no part in any claim, and are outside both K.
  - This also covers the math reviewer's S3 for MODEL_SELECT and DEV_REPORT.
- **S6. Naming in tables (source MF-3 and D4 lessons).**
  - Every table header names its bound: "clause bound (one-sided α/4)" or "family interval (K = 58 or K = 24)".
  - Clause fields are null for `match` and `difference` kinds and for endpoints that are not AVAILABLE.
- **S7. Bets (§13). Settle these before the lock, or not at all.**
  - **Scoring rules.**
    - P3 is conditional: it is VOID if P2 fails.
    - P4 and P5 are unconditional: they are NOT_HELD unless ORD_NLLM is MATCHED on both seeds.
    - P4 stays two-sided.
    - Each bet is HELD, NOT_HELD or VOID, and the mean Brier score over non-void bets is reported.
  - **P8 coherence.** Its necessary conditions P4 (0.35), P6 (0.75) and P7 (0.50) multiply to about 0.13 under independence. That is before the two benefit clauses and four worsening clauses.
    - With source recipient SEs of 0.0049 to 0.0073, each non-worsening clause needs SCM's recipient AUC to read below ORD_NLLM's by about 0.002 to 0.008 at the point.
    - So 0.20 implies strong positive dependence. Record that assumption, or revise the bet.
    - Bets never decide anything.
- **S8. Claim C wording.** When the SCM release is the ineligible fallback or C0, the claim C label is reported as being about that release.

## 4. NOTES

- **N1. Equivalence at ±0.001 is effectively unreachable, and that is by design.**
  - EQUIVALENCE_ESTABLISHED needs SE ≤ 0.001 / 3.0801 ≈ 0.00032 nats, even in the K = 24 family.
  - The source's occupation-NLL excess SEs (SCM against the references) were about 0.0033. A paired SCM − ORD_NLLM SE will be smaller but is unlikely to be below 0.0005.
  - So EQUIVALENCE_UNRESOLVED is expected. The labels key on the DEV_REPORT point match, and the wording rules in MF-2 apply.
- **N2. What the margins demand at plausible SEs.** Source pair-difference SEs were 0.0033 to 0.0059, and recipient-difference SEs 0.0049 to 0.0073.

  | Clause | Point needed | At SE 0.0045 to 0.0075 |
  |---|---|---|
  | Benefit L1 > 0 | > 2.4988 × SE | 0.011 to 0.019 |
  | Benefit L1 > 0.020 | > 0.020 + 2.4988 × SE | 0.031 to 0.039 |
  | Non-worsening U1 ≤ 0.010 | ≤ 0.010 − 2.4988 × SE | −0.001 to −0.009 |

  At recipient-sized SEs, the non-worsening clause effectively asks for demonstrated improvement. Low power on these clauses is expected; see MF-4 on how such failures are worded.
- **N3. The level guarantee is unconditional.** P(label established | any null) ≤ α/4. It is not a guarantee conditional on the DEV_REPORT match having carried over, so text must not say "given the match, the advantage is established at α/4".
- **N4. "Both seeds" is not replication.** It means robustness to the model seed on the same people, with the same bootstrap draws, so the seed endpoints are positively correlated. The intersection-union test does not need independence.
- **N5. Reusing the source bootstrap seed.**
  - The bootstrap weights are identical to the source's, so replayed releases reproduce the source numbers, and no seed choice is made after the history is known.
  - The relative Monte Carlo error of each SE at B = 2048 is about 1.6 %.
  - A borderline decision is deterministic given the frozen seed and is never re-drawn.
- **N6. Bonferroni across the four slices is conservative.** M and K test related hypotheses. This is accepted.
- **N7. Subgroup df.** Subgroup endpoints use t(G − 1) with the all-household G (source convention). The difference from a subgroup-specific G is negligible at about 2,700 households.
- **N8. Known history.** For the historical nominee, the public recipient differences would fail claim B's new worsening clauses: income vs NEURAL_RAW +0.018 / +0.015, and occupation vs H_LJ +0.041 / +0.030. Adding those clauses can only make claim B harder, so it cannot favour a label.
- **N9. P1 is effectively a reproduction check.** N0.010 is TERM, which is HIST (see the math reviewer's M2 on the "unreached" flag). The public MODEL_SELECT seed-1 men's guard is 0.0059 > 0.005.
- **N10. Selection is outside the intervals.**
  - ORD_NLLM is the most protective matched ordinary control on the proxy, which is conservative for SCM on the proxy.
  - The intervals do not include the selection of t, γ, r, the comparator or the readers, nor the development history on these records.

---

## FROZEN DESIGN

This block is authoritative where the draft is ambiguous. MF-1 to MF-7 and decisions §0(a) to (c) must be applied to PROTOCOL.md, `UNIT_INVENTORY.json`/`make_lock.py` (assert 58 / 24) and `report.py` before `ANALYSIS_LOCK.json` is written. Their keys, counts, signs and α must match this block.

**Signs** (s ∈ {0, 1}; AUCs come from the ATTACK_VALIDATION-selected winner for each view):
- `benefit` = AUC(comparator, pair) − AUC(SCM, pair). Positive favours SCM.
- `worsening` = AUC(SCM, r) − AUC(comparator, r), for r ∈ {income, occupation}. Non-worsening iff ≤ 0.010.
- `nll_diff` (occupation or income) = mean_i [NLL_i(SCM) − NLL_i(comparator)], with NLL_i = −log max(p_{i,y_i}, 1e-12). Positive means SCM spends more.
- `utility`:
  - accuracy_loss = acc(ref) − acc(SCM);
  - nll_excess = NLL(SCM) − NLL(ref);
  - brier_excess = Brier(SCM) − Brier(ref);
  - ref ∈ {U0, Ucal}.
- `subgroup` = occupation Brier(SCM) − Brier(U0) within recorded SEX g ∈ {0, 1}.
- `SEC` AUC differences = plus − minus, as named. `SEC` NLL differences are SCM − comparator.

**Primary roster.** 29 keys per seed; K_primary = 58. The claim tag of each key is in brackets.

| Key pattern (per seed `s{s}`) | Kind | Margin / refs | Claim |
|---|---|---|---|
| `MATCH\|coalition_benefit_vs_ORD_NLLM\|s{s}` | benefit | 0.0; refs 0.0, 0.020 | M |
| `MATCH\|{income,occupation}_worsening_vs_ORD_NLLM\|s{s}` (2) | no_excess | 0.010 | M |
| `MATCH_COMPONENT\|coalition_benefit_vs_IBSI_NLLM\|s{s}` | benefit | 0.0 | MC, sequence step 1 |
| `MATCH_COMPONENT\|coalition_benefit_vs_SHUF_NLLM\|s{s}` | benefit | 0.0 | MC, sequence step 2 |
| `CAPACITY\|coalition_benefit_vs_ORD_CNT\|s{s}` | benefit | 0.0; refs 0.0, 0.020 | K |
| `CAPACITY\|{income,occupation}_worsening_vs_ORD_CNT\|s{s}` (2) | no_excess | 0.010 | K |
| `CAPACITY\|coalition_benefit_vs_IBSI_CNT\|s{s}` | benefit | 0.0 | KC, sequence after K |
| `COMPETITIVE\|coalition_benefit_vs_{NEURAL_RAW,H_LJ}\|s{s}` (2) | benefit | 0.0; refs 0.0, 0.020 | B |
| `COMPETITIVE\|{income,occupation}_worsening_vs_{NEURAL_RAW,H_LJ}\|s{s}` (4) | no_excess | 0.010 | B |
| `UTILITY\|{income,occupation}\|{U0,Ucal}\|{accuracy_loss,nll_excess,brier_excess}\|s{s}` (12) | no_excess | caps 0.010 / 0.010 / 0.005 | U (point gate, no α) |
| `SUBGROUP\|occupation_brier_excess_vs_U0\|sex{0,1}\|s{s}` (2) | no_excess | 0.005 | C (and SG point gate) |

Per seed: 3 + 2 + 4 + 6 + 12 + 2 = 29, so **K_primary = 58**.

**Secondary roster.** 12 keys per seed; K_secondary = 24.

| Key pattern (per seed `s{s}`) | Kind | Margin / refs | Use |
|---|---|---|---|
| `SEC\|C0_minus_SCM\|pair\|s{s}` | difference | 0 | full 8-reader bank |
| `SEC\|C0_minus_ORD_NLLM\|pair\|s{s}` | difference | 0 | full bank |
| `SEC\|HIST_SCM_minus_SCM\|pair\|s{s}` | difference | 0 | full bank |
| `SEC\|ORD_CNT_minus_ORD_NLLM\|pair\|s{s}` | difference | 0 | full bank |
| `SEC\|ORD_NLLM_minus_SCM_slate7\|pair\|s{s}` | difference | 0 | inherited 7-reader slate |
| `SEC\|occupation_nll_difference_vs_ORD_CNT\|s{s}` | difference | 0 | SCM − ORD_CNT |
| `SEC\|occupation_nll_difference_vs_IBSI_CNT\|s{s}` | difference | 0 | SCM − IBSI_CNT |
| `SEC\|SCM_pair_minus_SCM_occupation\|s{s}` | difference | 0 | cross-view, full bank |
| `SEC\|occupation_nll_difference_vs_ORD_NLLM\|s{s}` | match | ±0.001 | point = claim M gate (no α) |
| `SEC\|occupation_nll_difference_vs_IBSI_NLLM\|s{s}` | match | ±0.001 | point = IBSI component gate |
| `SEC\|occupation_nll_difference_vs_SHUF_NLLM\|s{s}` | match | ±0.001 | point = SHUF component gate |
| `SEC\|income_nll_difference_vs_ORD_NLLM\|s{s}` | difference | 0 | SCM − ORD_NLLM income NLL |

Per seed: 8 + 3 + 1 = 12, so **K_secondary = 24**.

**Inference.**
- **Bootstrap:** `task_headroom.inference.group_weights` over DEV_REPORT households, B = 2048, seed 2026101705. It is shared across releases, endpoints and seeds. Assert G = 3955 and 5,943 persons.
- **AUC replicates:** `bootstrap_auc` with the same weights.
- **SE:** the replicate standard deviation (ddof 1).
- **Bounds:** Wald, point ± c × SE.
- **Scope:** conditional on the fitted releases and readers; not ACS design variance; excludes selection and development history.

**α allocation.** One-sided α = 0.025 is split into four claim slices of α/4 = 0.00625 each.
- **M:** the M label (margin 0; nested 0.020 by the same bound), then the IBSI component, then the SHUF component, as a strict fixed sequence. Stop at the first non-PASS.
- **K:** the K label (margin 0; nested 0.020), then IBSI_CNT, as a fixed sequence.
- **B:** intersection-union over 4 benefit clauses and 8 worsening clauses (margin 0; nested 0.020).
- **C:** intersection-union over 4 guard clauses.

UTILITY clauses, all gates (including the match points) and the whole secondary family spend no claim α.

**Critical-value rules** (computed by the frozen code from the observed G; expected values at G = 3955):
- Clause: c1 = t_{G−1}(1 − 0.025/4) = **2.4988491**.
  - Benefit clause passes iff L1 = point − c1·SE > m.
  - No-excess clause passes iff U1 = point + c1·SE ≤ m.
- Primary family: c_P = t_{G−1}(1 − 0.05/(2·58)) = **3.3346075**.
- Secondary family: c_S = t_{G−1}(1 − 0.05/(2·24)) = **3.0801279**.
- Marginal descriptive: t_{G−1}(0.975) = **1.9605641**.

**Statuses** (family interval [L, U] of the endpoint's own family):
- **Benefit:** ESTABLISHED (L > m), ESTABLISHED_SHORTFALL (U < m), otherwise UNRESOLVED.
- **No-excess:** PASS (U ≤ m), ESTABLISHED_VIOLATION (L > m), otherwise UNRESOLVED.
- **Match:**
  - point: POINT_MATCH iff |point| ≤ 0.001;
  - per-seed sub-status: CARRIED_OVER, SCM_SPENT_MORE (> +0.001) or COMPARATOR_SPENT_MORE (< −0.001);
  - interval: EQUIVALENCE_ESTABLISHED (L ≥ −0.001 and U ≤ 0.001), MISMATCH_ESTABLISHED (L > 0.001 or U < −0.001), otherwise EQUIVALENCE_UNRESOLVED.
- **Difference:** ABOVE_ZERO, BELOW_ZERO or UNRESOLVED.

**Availability (MF-3).** AVAILABLE, DIAGNOSTIC_NOT_MATCHED, COMPARATOR_INELIGIBLE or MISSING, plus the flag IDENTICAL_RELEASE. All 82 keys are always reported and counted. Only AVAILABLE endpoints can satisfy a claim condition or a gate.

**Common conditions.**
- CV: controls valid on all six views.
- NOM: SCM is the nominee, not the fallback.
- NC0: the nominee's effective release differs from C0 in both seeds.
- UG: all 24 DEV_REPORT point clauses ≤ cap.
- SG: all 4 DEV_REPORT within-SEX occupation Brier points ≤ 0.005.
- AV_R(s): role R is AVAILABLE in seed s.
- PM_R(s): |DEV_REPORT point of SCM − R occupation NLL| ≤ 0.001 in seed s (two-sided).
- BEN_R(s, m): benefit clause L1 > m.
- NW_R(s, r): worsening clause U1 ≤ 0.010.

**Labels as explicit conjunctions.** Labels for failures report the first failing condition in this order:
- M: CONTROL_FAILURE → NOMINEE_INELIGIBLE → NOMINEE_IS_C0 → MATCH_UNAVAILABLE → MATCH_DID_NOT_CARRY_OVER → NOMINEE_FAILS_DEV_UTILITY_GATE → NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD → NOT_ESTABLISHED.
- K and B: the same order, with K's UNMATCHED_CAPACITY → COMPARATOR_INELIGIBLE in place of the match conditions, and B having none.

Each failure label also reports the full condition vector with the MF-4 categories.
- **MATCHED_ADVANTAGE_ESTABLISHED** = CV ∧ NOM ∧ NC0 ∧ AV_ORD_NLLM(0) ∧ AV_ORD_NLLM(1) ∧ PM_ORD_NLLM(0) ∧ PM_ORD_NLLM(1) ∧ UG ∧ SG ∧ BEN_ORD_NLLM(0, 0) ∧ BEN_ORD_NLLM(1, 0) ∧ ⋀_{s,r} NW_ORD_NLLM(s, r).
  - MATERIAL_ADVANTAGE_ESTABLISHED adds ⋀_s BEN_ORD_NLLM(s, 0.020).
- **IBSI component PASS** = M established ∧ AV_IBSI_NLLM(both seeds) ∧ ⋀_s PM_IBSI_NLLM(s) ∧ ⋀_s BEN_IBSI_NLLM(s, 0).
  - SHUF component PASS = IBSI PASS ∧ AV_SHUF_NLLM(both seeds) ∧ ⋀_s PM_SHUF_NLLM(s) ∧ ⋀_s BEN_SHUF_NLLM(s, 0).
  - COMPONENTS_ESTABLISHED = IBSI PASS ∧ SHUF PASS.
- **SAME_CAPACITY_ADVANTAGE_ESTABLISHED** = CV ∧ NOM ∧ NC0 ∧ ORD_CNT reached K*_s on both seeds ∧ ORD_CNT passes its 12 MODEL_SELECT clauses on both seeds ∧ UG ∧ SG ∧ ⋀_s BEN_ORD_CNT(s, 0) ∧ ⋀_{s,r} NW_ORD_CNT(s, r).
  - Material adds ⋀_s BEN_ORD_CNT(s, 0.020).
  - IBSI_CNT PASS = K established ∧ IBSI_CNT's γ is eligible ∧ ⋀_s BEN_IBSI_CNT(s, 0).
- **SUPERIOR_TO_ALL_ELIGIBLE** = CV ∧ NOM ∧ NC0 ∧ UG ∧ SG ∧ ⋀_{X ∈ {NEURAL_RAW, H_LJ}, s} BEN_X(s, 0) ∧ ⋀_{X, s, r} NW_X(s, r).
  - MATERIAL_SUPERIORITY adds ⋀_{X, s} BEN_X(s, 0.020).
- **Claim C:**
  - SUBGROUP_QUALIFIED = ⋀_{g,s} (guard clause U1 ≤ 0.005).
  - SUBGROUP_HARM = ⋁_{g,s} (family L > 0.005).
  - Otherwise SUBGROUP_UNRESOLVED.
- **Separate fields (MF-5)** and the **research decision with its 0.020 status and four qualifiers (MF-6)** are always reported.
