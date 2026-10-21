# Objective audit: the routing proposal, the acceptance objective and the emitted release

Campaign `pcrl_finite_solver_adequacy_v1`. The numbers are in `tables/OBJECTIVE_AUDIT_TABLES.md`, generated from
`audit/OBJECTIVE_AUDIT_V<v>.json`. The audit uses the cached MODEL_FIT out-of-fold statistics of the previous campaign's releases:
the NRM1 task teachers and the P1 SEX-side teachers. These statistics are derived from MODEL_FIT training labels; no label file was
read and MODEL_SELECT was not opened. The exact-reference investigator checked the same code independently on invented laws
(`EXACT_REFERENCE_CASES.md`).

## What is compared

| Quantity | Where it lives | What it is |
|---|---|---|
| Routing proposal score | `nrm.fa.prc.route` | Per person and joint cell: task cross-entropy to the codewords + tilt × (dependence on 20-pseudo-count shrunk cell SEX rates + κ × guard). It is a linearization around the current tables |
| Acceptance objective | `nrm.fa.prc.objective` | Mean task cross-entropy to the codewords + λ × (unshrunk π-weighted plug-in information of the pair and of each recipient's cell) + λκ × the per-SEX guard |
| Emitted tables | `frc.fa.route.tables` (B1) | Out-of-fold teacher means of each configuration's own final cells |
| Emitted task losses | the evaluation records | MODEL_FIT nested held-out losses (`nested/`); MODEL_SELECT losses of the saved releases from the previous campaign. Not re-read here |

## Findings

1. **No implementation defect** (frozen rule `route_d`: `D_NO_MATERIAL_DEFECT`).
   - The investigator's independent code reproduces `objective` within 1.2e-13, `route` with 0 mismatches, `update_book` /
     `set_tables` within 1e-14, and `lloyd`, `_chain` and `protected_path` bitwise on every registered law.
   - The audit's traced copy of `lloyd` reproduces the saved PRC_S2 and HQ_S2 partitions in both versions.
   - Every behaviour examined matches card FA_PRC_v1 section 1.
2. **The four documented approximations, measured.**
   - **Shrunk, linearized routing against the exact objective.**
     - On the final refreshed book's tilt line at strength 8 (68,921-91,490 distinct deployable partitions), the routing's dependence
       score and the exact dependence term are anti-ranked: Spearman -0.995 / -0.996 for PRC, -1.000 / -0.999 for HQ.
     - This was measured on one line per family and version, at strength 8 only, and no causal test was run. It is consistent with
       the original chains rejecting large tilts (PRC accepted 1 of 7 Lloyd steps). The exact acceptance test keeps every accepted
       step valid.
   - **Cell-mean codeword refresh against the guard.** On the real chains the refresh never raised the objective (largest change
     -9.1e-7 / -7.1e-7). In law B the stale codewords left the objective 1.3e-3 to 1.1e-2 above the codeword-refreshed value without
     changing the release.
   - **B1 emission against search prototypes.**
     - The largest absolute difference is 0.0095 / 0.0066 for PRC_S2, 0.032 / 0.054 for HQ_S2 and 0.0023 / 0.0019 for the twin.
     - The emitted tables' training cross-entropy is lower than or equal to the prototypes' (equal to six decimals for income in
       PRC_S2 version 0). The objective's task term uses the prototypes, so it slightly overstates the emitted release's training task
       loss.
     - Partitions are accepted by the objective; strengths are selected on the emitted release's nested held-out spend.
     - The guard computed with emitted tables is within 3.2e-4 of the prototype guard.
   - **Seven tilts, first improvement.** See `EXACT_REFERENCE_CASES.md` and `SOLVER_COMPARISONS.md`. Improving tilt windows without a
     power of two are never tried, in laws D and E and in version 1 at strength 8 on the real statistics (an improving partition
     near t = 0.19).
3. **Monotone acceptance.**
   - The original chains' accepted objective decreases strictly: 7 Lloyd steps with 1 accepted for PRC, 9 / 8 steps with 3 / 2
     accepted for HQ.
   - Most strengths accept nothing: PRC accepts only at strength 0.25, and HQ also at one later strength.
   - The refreshed book's line was enumerated at strength 8 only. There, in PRC version 0, every partition on it is worse than the
     current state (best 2.622865 against 2.607569). In PRC version 1 the improving window falls between tried tilts (2.573211 in a
     window 4.8e-6 wide near t = 0.1875, against 2.590162).
   - Monotonicity is not an optimality certificate.
4. **Runtime consistency.**
   - Every accepted state is (book, route(book)), so the deployed six-field rule reproduces the fitted partition. All 72 saved books
     re-route to their saved partitions bitwise, also from the backup alone.
   - A 1e-9 relative tilt change moves no one, so no accepted tilt is on a knife edge.
   - Routed with the deployed full-pool teachers, 48-53% of the training rows land in the same joint cell as their out-of-fold
     routing, so about half move. This is because full-pool predictions on training rows differ from out-of-fold ones.
   - The B1 tables use the out-of-fold cells, which is what new people resemble. This was not examined further and is not a defect
     under the rules.
5. **Expressivity.**
   - MODEL_FIT has 17,744 distinct six-field support states among 20,000 people. The deployed task scores take 17,670 / 17,652
     distinct values, and only 1.0% / 1.3% of people share their task scores with another support state.
   - An unrestricted task-score mapping could therefore reproduce any partition PRC can produce on this support, except on the
     1.0-1.3% of people whose task scores are shared with another support state.
   - HQ is restricted (routing on twin-cell means of π, qO0 and qO1, while its tables and acceptance use person-level values). A PRC
     against HQ difference is about rule parametrization and search, not information that propensity adds.

## Classification

Route D classifies these differences; it does not measure their effect on decisions. The last column gives what was measured.

| Difference | Classification | Measured effect |
|---|---|---|
| Shrunk, linearized routing against unshrunk plug-in acceptance | documented approximation (`SHRUNK_ROUTING_TABLES`) | anti-ranked along the strength-8 line (Spearman -0.995 to -1.000); consistent with the rejection of large tilts |
| Cell-mean refresh against the guard | documented approximation (`CELL_MEAN_CODEWORD_REFRESH`) | never raised the objective on the real chains (largest change -9.1e-7) |
| B1 emission against prototypes | documented approximation (`B1_EMISSION_DIFFERS_FROM_SEARCH_PROTOTYPES`) | emitted training cross-entropy at most the prototypes'; the objective slightly overstates the emitted training task loss |
| Seven-tilt first-improvement search | documented approximation (`SEVEN_TILT_FIRST_IMPROVEMENT_SEARCH`) | it changes the solutions: improving windows are missed (laws D and E; version 1 at strength 8). SV1 addresses this with a solver-only change |
