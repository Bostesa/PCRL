# Independent verifier brief (privacy-first selector v1)

The verifier works with fresh context and writes **separate code** in `experiments/pcrl_privacy_first_selector_v1/verify_independent.py`.

- **Allowed:** numpy, scipy, json and hashlib.
- **Not allowed:** importing `solve.py`, `outer.py`, `lockgate.py` or `host.py`.
- **Also not allowed:** the SC or TAC inference code (`inference.py`, `paired_household_bounds`, `evaluate_family`).
- **What counts as the specification:** the file formats below and `PROTOCOL.md`. The implementation under test does not.

## A. Solves (from the frozen basis files)

Inputs, for each anchor a:

- `private/prior_units/a{a}_NM4_U/closing/TASK_BLOCKS.npz`, with arrays U_B, U_A, W_B and W_A. The B arrays are 32×17 and the A arrays 4×5.
- `closing/CALIBRATED_BLOCKS.npz`, with arrays `cut_{i:04d}_B` and `cut_{i:04d}_A`.
- `closing/CALIBRATED_BANK.json`, whose `cuts` entries i carry id, role, weighting, rho and floor.
- D17: the 32×17 one-hot map. It is recovered as `B = D17` from `private/units/a{a}_TASK_SEL4`? No. Use the D17 map file in the pinned index (`results/pcrl_task_aligned_cuts_v1/REUSABLE_INPUTS_PINNED.json` → anchor → map D17 → path); its SHA-256 is in `BASIS_MANIFEST.json` as `d17_map_array_sha256`, which is the law identity of the float64 array.

Checks:

1. **Objective.**
   - t(q) = min over AB/SEX cuts of (L_c(q) − rho_c).
   - Feasibility: every other cut satisfies L_c(q) ≥ floor_c (floor_c = rho_c − .001), and task_v(q) ≤ task_v(D17) + .001 for v ∈ {U, W}.
   - Tolerance 1e-7.
2. **D4 and D1.** Recompute by your own enumeration.
   - q is one-hot per context: B = 0, A one-hot rows.
   - For K=1, sum the A blocks over the 4 contexts.
   - Tie rule: max t, then (within 1e-12) lower 0.5(task_U + task_W), then fewer non-D17 contexts, then lexicographic index.
3. **R4 and R1.** Re-solve with a formulation different from the implementation: for example, highs-ipm, or the dual LP. Compare t with `SOLVE_REPORTS.json` to 1e-6.
4. **TASK_SEL4.** The argmin task, with no constraint.
5. **DET_SEL4.** The lowest balanced task among assignments with every cut ≥ floor, with **no task cap**.
6. **Parameters.** Compare with `private/units/a{a}_{X}/PARAMS.npz`. D4, D1 and TASK_SEL4 must match exactly. Report the R-law value differences.

## B. Outer point estimates and decisions (after assessment)

Inputs:

- `SELECTION_LOCK.json`: endpoints in `family_manifest`, `secondary_manifest` and `capability_manifest`, with `plus`/`minus` release names, role, weighting and threshold;
- `anchors[a].logical_to_canonical`;
- for each anchor, `private/outer/a{a}/OUTER_AUDIT.json` (`releases[canonical].private_contribution_prefix`) and `OUTER_CONTRIBUTIONS.npz`.

Per canonical release, prefix p and role r, with stem `{p}__{r with ':' and '/' replaced by '_'}`, the archive holds `{stem}_households`, `{stem}_weights` (PWGTP), `{stem}_candidate_loss` and `{stem}_H_loss`, aligned per person. "H" is the H-only ancestor (any release's `H_loss`; they are identical).

The estimand, for endpoint e and anchor a:

- D_i = loss_plus_i − loss_minus_i, with persons aligned by position and ids asserted equal.
- The weight a_i is 1 for U, or PWGTP.
- Anchor estimate = Σ a_i D_i / Σ a_i.
- The estimate is the equal mean over the 3 anchors.

Checks:

1. **Point estimates.** Recompute every estimate and compare with `ENDPOINT_TABLE.json` to 1e-12.
2. **Bootstrap.** Write your own household bootstrap:
   - one multinomial resample over the **union** of household IDs across anchors;
   - per anchor, the weighted ratio from household sums;
   - the equal mean over anchors;
   - SE with ddof = 1;
   - 10,000 draws, with **your own seed** (not 20260926).
3. **Bounds.** Recompute the bounds at the locked z of each family.
4. **Decisions.** Compare every `passed_upper_bound` decision and the four registered labels (`INFERENCE.json` → `decision_labels`). Report any flip and its margin.

## C. Custody

- The lock commit is on origin, and its `git show` bytes have the locked SHA-256.
- The lock commit time precedes the `OUTER_UNLOCK.json` `verified_utc`.
- That precedes `ORIGINAL_RESTORE.json` utc.
- That precedes every `outer/a*/COMPLETE.json` `completed_utc`.
- No inner panel or unit file changed after the lock: recompute the SHA-256 of `inner_panels/a*/COMPLETE.json` and `INNER_AUDIT.json` against the lock pins.

## Output

`INDEPENDENT_VERIFICATION.json` holds per-check pass/fail, maximum differences and flips. `REVIEW.md` holds short caveats.
