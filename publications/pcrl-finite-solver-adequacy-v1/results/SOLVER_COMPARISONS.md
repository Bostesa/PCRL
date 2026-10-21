# Solver comparisons on the source (cached MODEL_FIT statistics and nested MODEL_FIT)

Campaign `pcrl_finite_solver_adequacy_v1`. The numbers are in `tables/SOLVER_COMPARISON_TABLES.md`, generated from
`source/SOURCE_CHECK_V<v>.json`, `nested/NESTED_SOLVERS_V<v>.json`, `replay/P1_REPLAY_V<v>.json` and `decisions/*.json`.
- The full-pool statistics are the previous campaign's cached out-of-fold teacher predictions on MODEL_FIT, derived from MODEL_FIT
  training labels.
- The nested runs refit the teachers in 5 outer household folds, as the original P1 estimator did.
- MODEL_SELECT was not opened.

## Solvers (card FA_FSA_v2)

| Solver | Lloyd step | Role |
|---|---|---|
| original | tilts λ, λ/2, …, λ/64 on the refreshed book; first improvement | the frozen original (`nrm.fa.prc.lloyd`) |
| SV1 | exact tilt line searches t ∈ [0, λ] on the refreshed book and the current book; best improvement; accepted only if `nrm`'s own objective on `nrm`'s own routing improves | the revision |
| SV1_first | the SV1 candidates with first improvement, in the original order | equal-effort-per-step control (reported only): the same candidate set and evaluation per step, not the same effort per chain |

The objective, routing family, teachers, capacity, twin start, strength continuation and emitted B1 tables are unchanged, so the
contrast is solver-only. PRC and HQ get the same variants.

## Results

1. **The replay closes the previous verification gap.** The previous campaign's nested P1 estimator, rerun unchanged, reproduces
   all of these with difference 0.0:
   - all 413 recorded nested means per version;
   - the selection and the brackets;
   - F4: 0.00715 (se 0.00225) / 0.01113 (se 0.00231).

   The nested stage's original-solver entries equal the replay in all 117 per-person arrays. The verifier's independent
   recomputation is in VERIFICATION.md. The frozen replay rule gives `P1_REPLAY_VERIFIED` under the statistical reviewer's
   estimator-identity ruling; its strict text gives `P1_ESTIMATOR_CHANGED`, because the replay re-implements the selection loop with
   renamed local variables (RESEARCH_DECISION.md, "Rulings and disclosures").
2. **Better optimization of the stated objective (question A).**
   - SV1's fitted objective is below the original's at every registered strength in both versions for PRC: by 3e-5 to 0.0056 in
     version 0 and 3e-6 to 0.0105 in version 1.
   - Every SV1 state is rule-consistent and distinct from both the original's and the twin's partition.
   - For HQ, SV1 is lower in 11 of 12 cells and higher in version 0 at strength 8 (+1.4e-4).
   - **SV1 is not optimal.** In version 1 at strength 8, the audit exhibited a deployable partition on the original's final
     refreshed-book line with objective 2.5732, in a tilt window 4.8e-6 wide near 0.1875. SV1's chain ends at 2.5797: better than
     the original's 2.5902, but above that partition, which lies off SV1's own continuation path.
   - **The equal-effort attribution is confounded by the cap.**
     - SV1_first accepts the first improving candidate at each step. It reached the registered 30-step path cap, still improving, in 23 of
       24 family-version-strength chains. HQ version 0 at strength 8 stopped after 20 steps with no improving proposal. The records
       do not say which line each accepted step came from.
     - Under that cap it ended above the original at every strength (PRC and HQ, both versions) and above SV1 in all 12 cells per
       family.
     - This does not show that finer resolution cannot help. On the exact laws, SV1_first equals SV1.
   - **Effort.**
     - SV1 chains took 24.5-32.9 CPU-s on the full pool, and the originals 4.1-4.8 CPU-s.
     - SV1_first took 436.5-510.8 CPU-s and 13-18 times SV1's Lloyd steps (170-180 against 10-13).
     - So it matches SV1's candidate set per step, not its effort per chain.
3. **What the better optimum looks like.**
   - SV1's strength-8 PRC state re-routes 23.6% / 21.2% of people relative to the twin (original 15.6% / 13.8%).
   - Its fitted π-weighted pair information falls by only 0.0002 nats (0.03300 / 0.03252 against 0.03322 / 0.03273).
   - The emitted tables' training cross-entropy rises by 0.0008.
   - The pair term accounts for about -0.0018 of the -0.0056 objective change at strength 8 in version 0 (8 × -0.0002). The rest
     comes from the other terms (task cross-entropy to the codewords, per-recipient information, guard), which were not decomposed
     here.
4. **Usefulness and recovery on MODEL_FIT nested held-out data (question B).**
   - SV1's S2 selection is strength 8 in both versions. Its nested total-NLL spend over the twin is 0.00153 / 0.00121, against
     0.00214 / 0.00194 for the original's S2 point.
   - **The bracket** is the cheapest original path point whose nested spend is at least SV1's spend − 0.0005 (RULES 4.3). That is
     PRC@2.0 in both versions.
     - In version 0 it behaves as the same partition as PRC@8.0: identical nested spend and identical paired differences.
     - In version 1 it spends 0.00088, which is 0.00034 less than SV1's point. It is matched only within the 0.0005 tolerance.
   - **Against the bracket**, SV1's held-out pair LOOKUP AUC is lower by 0.0018 (se 0.0020) and 0.0038 (se 0.0017). The spend
     difference is -0.00061 (se 0.00060) in version 0 and +0.00034 (se 0.00052) in version 1.
     - The per-version point conditions held.
     - Pooled, -0.0028 with mean se 0.0019 (2-SE interval -0.0065 to +0.0009): the pooled 2-SE condition fails.
   - **Against the original S2 point**, the pair difference is -0.0018 / -0.0003, at spend -0.0006 / -0.0007.
     - In version 1 the single views move in opposite directions: occupation +0.0023 (se 0.0022) and income -0.0062 (se 0.0026).
     - The task alternative fails: spend -0.0006 / -0.0007 against the required -0.001.
   - **Route S: `S_PROXY_ONLY`.**
5. **HQ under the same solver (question C, descriptive only).**
   - Revised HQ at strength 8 has nested pair LOOKUP AUC 0.6225 / 0.6251 at spend 0.0029 / 0.0063 (original HQ: 0.6262 / 0.6276 at
     0.0012 / 0.0041).
   - Revised PRC at strength 8 is 0.6209 / 0.6203 at 0.0015 / 0.0012, that is 0.0016 / 0.0048 below revised HQ at lower spend.
   - Under the original solver the same strength-8 gap was 0.0035 / 0.0070. In the previous campaign that larger nested gap did not
     carry to the MODEL_SELECT reader grid (PRC_S2 against HQ_S2: benefit +0.0003, se 0.0051).
   - There are no SEs and no registered comparison, and MODEL_SELECT was not examined. LOOKUP is the reader type the objective
     targets, so by card FA_PRC_v1 an advantage seen only by it is reader-targeted. Question C is unanswered.
   - Task scores identify the support states for 98.7-99.0% of people. Such a difference would therefore concern rule parametrization
     and search inside two restricted families, not the information propensity adds.

## Decision

The frozen advancement rule returns `OPTIMIZER_REPAIR_CLOSED` (D `D_NO_MATERIAL_DEFECT`, E `E_SUPERSEDED_BY_S`, S `S_PROXY_ONLY`).
The conditional pilot did not run.
