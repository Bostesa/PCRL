# Threat-model and contract check of the two final proposals

Recipient-use and adversary reviewer, 2026-10-16. I did not edit either proposal and opened no `.private/` or `.npz` file.

**What I read.**
- `roles/designer_1/PROPOSAL.md` (SCM) and `roles/designer_2/PROPOSAL.md` (SCAR).
- `SERVICE_AND_THREAT_MODEL.md` v1, §1–3.
- Code: `acs_replication/deploy.py:44-60` and `acs_replication/trees.py:77-83`.
- `roles/coordinator/SUPPORT_FACTS.json`.

The two designs belong to the same family. Each builds a shared finite partition of task-posterior space, merges cells greedily with a SEX term, and releases pooled heads. Their shared issues come first.

## Common to both

**Contract (holds for both, with one gap).**
- Both deployed maps take only the six fields, through the existing encoder.
- The refusal of SEX holds by exact schema: `deploy.read_raw_csv` raises `PermissionError` on any FORBIDDEN column and requires the exact permitted header (`deploy.py:48-52`), and `TreeTeacher.check_inputs` requires the registered 145 names and width (`trees.py:77-83`).
- Released values are copies of stored float64 table vectors. No teacher output, centroid distance or precision beyond the table can leak, and the decisions are functions of the vectors.
- **The gap:** the existing `deploy.release` returns both tasks to one caller (`deploy.py:58ff`), so separate routing must be built in the new release path.

**What each view identifies (both).**
- R2's vector identifies the coarse cell, and R1's vector is a function of that cell. Whenever R2's vectors are distinct, the linked pair therefore carries nothing beyond R2.
- Linkage adds nothing by construction. The IGNORE_income alias is the optimal pair reader in population, and the coalition and occupation-recipient endpoints measure the same quantity.
- All protection must come from R2's view, which T_TREE reads at 0.650 alone. Both proposals should state this.
- Neither proposal claims it, but it bounds what the "coalition" framing adds.

**X1 (SERIOUS, both). The gate proxies are in-sample for the SEX-aware arm.**
- The proxy readers are fitted on MODEL_FIT wires with MODEL_FIT SEX, the same SEX that drove π̂ and therefore the merges. This covers SCM's F1 cell table and SCAR's G2–G4 proxies, which follow the frozen rule.
- π̂ for one fold is fitted on the other folds' SEX, and the gate table uses every fold's SEX. The merges equalise the cells' MODEL_FIT SEX rates, including noise, with a winner's curse (SCM W3). A table fitted on MODEL_FIT therefore under-reads the SEX-aware arm more than the SEX-free twin.
- The bias favours the candidate, by an unknown amount, against margins of 0.010–0.020. Fitting-pool guards were optimistic in the closed lanes.
- **Fix.** The gates are new, so fit their readers on one household half of MODEL_SELECT and score on the other half, in both directions. Alternatively, fit them on CALIBRATION, which neither design uses. Keep the frozen selection rule, but disclose that its OUT_LR proxy shares the bias.

**X2 (SERIOUS, both). The permuted-SEX control is too noisy to separate SEX-specific merging from merely leaving the twin's path.**
- SCM runs one shuffled path with no numeric criterion ("must not match SCM").
- SCAR_PERM uses one permutation against G3's margin of 0.010. SCAR's own toy shows a spread of 0.659–0.676 over 5 permutations (0.017), and departure alone explained 0.005–0.022.
- **Fix.** Use at least 5 fixed-seed permutations, and require the candidate to beat the *best* permuted path by the margin.
- Permute cell-level SEX rates across fine cells rather than π̂ across people. Person-level permutation leaves near-constant cell rates, which λ₀ then rescales into white noise. Cell-level permutation keeps the signal's granularity and magnitude and destroys only its alignment with x.

**Minor, both.**
- **Label-free checks before any reader:**
  - distinct-value counts per view on every role (R1 ≤ the number of groups or bins; R2 and the pair ≤ the final cell count);
  - every pair of co-cell inputs gives bit-identical outputs;
  - candidate, twin and permuted arms use the same canonicalisation: each table vector is stored once and indexed. Otherwise, last-bit float differences in one arm become extra keys and show up as a spurious recipe effect. SCAR's toy twin produced 13 distinct vectors for 10 real values (§13).
- **Reader bank.**
  - **Lookup reader.** The frozen lookup is the Bayes plug-in for these releases. It is adequate when cells hold enough ATTACK_FIT people.
  - **Planted cell-split control.** Add a positive control on the candidate's own wire. A SEX-dependent change of one unit in the last place, in 10% of cells on all roles, should be detected by the lookup and may be missed by HGB. This calibrates the lookup's power at the realised cell count.
  - **Learning curves.** Report them per reader for candidate and twin, with a reading fixed in advance: a candidate lookup curve that still rises at 100% more steeply than the twin's marks the recipe effect as partly reader-limited.
  - **Optional.** A descriptive lookup reader shrunk toward OUT_HGB's prediction, with its pseudo-count chosen on ATTACK_VALIDATION.
- **Capacity matching.** A matched cell count is not matched capacity, because SEX-aware merges can make cell masses more unequal. Report the effective number of cells, exp H(C), for every arm.
- **No critic-masking, label-leak, omitted-wire or useless-task route found:**
  - neither design trains a critic;
  - each release is a function of x alone (SEX enters only the merge order, and SCM's Y2 enters only its guard);
  - the proxies read the full wire;
  - the caps plus the gain-over-constant checks (SCM law D, SCAR check e) cover a useless task.
  - The masking risk therefore lies in *measurement* (X1, X2, S1), not in the release.

## SCM (designer 1)

1. **Contract.** As above. R1 = v1[g] is indexed by group, so its storage is explicit, and R2 = v2[c]. There is no id, propensity or state. **MINOR:** the deployed map (§2, step 5) never states routing to separate recipients.
2. **What each view identifies, and reader adequacy.**
   - R2 and the pair identify the final cell (≤ 512, fewer after merging); R1 identifies one of 32 groups.
   - ATTACK_FIT has 8,001 people, so cells average at least about 16 people each. That is adequate for the lookup, and HGB covers neighbouring vectors.
   - **MINOR:** name the estimator for AUC\*(S; C). An in-sample plug-in is biased upward; MODEL_FIT-fitted tables are biased toward the candidate (X1).
3. **Masking, leakage, controls.** The bound AUC\*(S; V) ≤ AUC\*(S; C) is a correct population statement for any reader. Measurement risks are X1 and X2. **MINOR:** keep the budget-matched twin in the report alongside the count-matched one.
4. **Utility risks that bear on the threat model.**
   - **Partition mismatch (MINOR, but report it).** Cells, values and π_c are built from out-of-fold posteriors (S1–S5), but deployment assigns cells by the average of the 5 fold teachers. The optimised partition is therefore not the deployed one. Report the share of MODEL_FIT people whose deployed cell equals their fitting cell, and the shift in v2 when values are recomputed on deployed membership.
   - **Income grouping (MINOR).** Grouping income into 32 groups (S5) happens after the budgeted merges, so its income cost lies outside the merge budgets; only G0 and selection catch it.
   - **Within-SEX guard (MINOR).** SCM budgets the within-SEX occupation guard, which is good. The estimate, however, is an in-sample plug-in on roughly 20 people per SEX per initial cell, shrunk with κ = 20, so it is optimistic. MMP reached 2.2–2.6× this guard. Make its MODEL_SELECT value, for both SEX groups, a stage-0 gate rather than a diagnostic.

**SERIOUS for SCM:** X1 and X2.

## SCAR (designer 2)

1. **Contract.** As above. Routing (§2, step 5) and the schema guard are stated.
   - **MINOR hidden-wire risk.** B1 is described as "stored float64 vectors (one per coarse cell or income bin)". If B1 is materialised per coarse cell, with each entry recomputed as a mean, last-bit differences would let R1 identify the coarse cell (K values) instead of its bin (64). This is exactly SCAR's own toy effect.
   - **Fix.** Store 64 bin vectors once and index them by bin[c]. Check that R1 has at most 64 distinct values on every role.
2. **What each view identifies, and reader adequacy (SERIOUS, S1).**
   - R2 and the pair identify the coarse cell, and §10 expects K ≈ 10³. That leaves about 8 ATTACK_FIT people per cell on average, so the lookup is noisy and some cells go unseen.
   - SEX-aware merges preferentially pool neighbouring cells with opposite tilt. They remove the smooth, linearly readable part of the signal and can leave a rougher residual, which OUT_LR and HGB read worse from 8,001 labels. A measured recipe effect can therefore exceed the information actually removed.
   - The headline falsifier G2 uses OUT_LR, the proxy most exposed to this. G4 asks only 0.010 under HGB and the lookup.
   - **Fix.** Evaluate G2 on each arm's maximum over the {OUT_LR, OUT_HGB, lookup} proxies, cross-fitted as in X1. Then either bound K so that the median ATTACK_FIT count per cell is at least about 15, or register now that recoveries at a K where the lookup curve still rises are reported as reader-limited.
3. **Masking, leakage, controls.** X2 applies, and G3's margin is below the toy's permutation spread. NR_PROJECTED is a useful diagnostic, provided its readers are refitted on the projected wire.
4. **Utility risks that bear on the threat model.**
   - **Within-SEX guard (SERIOUS, S2).** The within-SEX occupation Brier guard is "reported, not gating" (§6), and the six budgeted clauses are pooled over SEX. Merging cells with opposite SEX tilt is the mechanism that moved MMP to 2.2–2.6× the guard. Nothing stops the selected r from concentrating cost on one SEX, which would leave an ESTABLISHED_SUBGROUP_HARM risk at DEV_REPORT and protection partly bought by worse predictions for one sex.
   - **Fix.** Add the two within-SEX excesses as budgeted plug-in clauses with shrinkage, and gate on their MODEL_SELECT values.
   - **In-sample teachers (MINOR).** The teachers are refitted on all of MODEL_FIT, and the cells, heads, K rule, λ₀ and plug-in costs all use the same people's in-sample posteriors. Held-out costs will be higher. MODEL_SELECT feasibility catches this, so the consequence is a likelier G0 or G1 failure, not a false claim.
   - **Income accuracy (MINOR).** Income accuracy has the thinnest headroom (0.0012 / 0.0006).

**SERIOUS for SCAR:** X1, X2, S1 and S2.

## Which variant is safer to pilot

**SCM, after X1 and X2 are fixed.**
- Its headline falsifier, F1, already uses the reader whose structure matches the release (the cell table).
- It budgets the within-SEX guard.
- Its cells are larger (at most 512, so on average at least about 16 ATTACK_FIT people per cell).
- Its table indexing is explicit.

SCAR is a close variant with stronger routing statements and the NR_PROJECTED diagnostic, which SCM should borrow. SCAR becomes comparable once G2 uses the maximum over proxies, the guard is budgeted and K is bounded. For either design, cross-fitted gate proxies and a multi-permutation control are prerequisites for a gate pass to mean anything.
