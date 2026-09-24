# Independent prefit review — 2026-09-24 UTC

Scope: read-only review of the adaptive-release source, registered protocol, global role loader, finite reference/solver, Branch A center, Branch B primitives, independent audit and endpoint family. No ACS candidate fit or outcome was run by this reviewer.

**Current decision (03:02 UTC):** The registered A center and inner-audit code are conditionally ready for the first ACS fit after the coordinator commits and pins the exact source, verifies the sanitized 2018 input receipt on the execution host, and records the staged unit/command before launch. Branch B is not cleared until its complete fit/serialization/restore path lands and passes a synthetic end-to-end check. The findings below retain the original pre-correction failures and their subsequent verified resolutions.

Material blockers:

1. **Cleared at 03:01 UTC:** `evaluate.audit_panel` lacked a sanitized 2018 receipt precondition and could trigger the inherited raw-joblib fallback. The negative fixture failed at that call; the added gate now passes before deserialization, as `fit_a.run_center` already required.
2. **Cleared at 03:02 UTC:** `fit_a.run_center_from_roles` could select an earlier Q feasible only for its own bank. It now replays each round against the final rebased union bank and excludes infeasible channels. A new counterexample fixture verifies this.
3. **Open:** Branch B currently has split-search, child aggregation, dual-pricing and nuisance primitives, but no complete saved, hash-replayed center fit/round receipt in `fit_b.py`. This is a dispatch-readiness blocker for a B center, not a mathematical objection to the primitives.

Verified or conditionally sound:

- `roles.pooled_role` uses the registered first-64-bit SHA-256 household assignment, pools only the four inner historical sources and refuses prelock `outer_assessment`. The A `run_center` explicitly requires `SANITIZATION.json` and checks the outer label field is absent.
- The task and sensitive coefficient calculations enumerate token log loss per original person, separately normalize U and PWGTP on eligible rows, and aggregate each person once. No second state-mass multiplier appears in the reviewed path. A's fixed bank includes H, D17, historical Q and coverage sources. Coalition attack coefficients include A same-token and B H-only ancestors.
- `reference.calibrate_reference` forms `rho=min_a L_a(D17)` on the same coefficient people, class order and weighting, then places every retained floor at `rho−delta`. The D17 witness is independently replayed. The finite LP dual and deterministic MILP are same-bank statements, not an oracle or population guarantee.
- The locked endpoint code uses task CE(candidate)−CE(control) and sensitive recovery CE(control)−CE(candidate). U requires task ≤−.003 in both weightings with eight sensitive guards ≤+.001. P requires task ≤+.001 and AB/SEX recovery ≤−.002 in both weightings, with the other sensitive guards ≤+.001. Endpoint IDs are generated from locked slots/aliases. The coordinator must still verify that the selection lock names every mandatory comparator; the generic enumerator does not impose that list.
- The inner audit is structured for seven shared H-only slates, five fresh slates per release, validation-only route selection, legal A/B coalition ancestors, exact token expectation and household contributions. It requires the sanitized-input fix before execution.

Verification run from the adaptive worktree:

```sh
/Users/nathansamson/PCRL/.venv/bin/python -m pytest -q tests/pcrl_adaptive_release_v1 tests/pcrl_task_aligned_cuts_v1/test_audit_inference.py tests/pcrl_task_aligned_cuts_v1/test_solver_review.py
```

Result: **92 passed, 1 failed**, with the single failure the sanitized-receipt negative fixture above. Earlier focused controls/audit/solver and A/B primitive suites passed before that fixture was added. No favorable ACS outcome was inspected.

## Verified correction at 03:01 UTC

The coordinator added the sanitized-receipt precondition before `evaluate.audit_panel` deserializes a prepared object. `data._sanitized_record` verifies the receipt, index, source and copy hash; an absent copy now raises. I reran the negative fixture and focused evaluate/A/reference/roles/inference suite: **24/24 passed**. Blocker 1 is cleared without changing scientific outcomes. The multi-round final-bank selection issue and missing B center remain open.

I also rehashed the eight files named in `PROTOCOL_LOCK.json`: all match. An independent one-state sign calculation with D17 attack loss 0.600, allowance 0.001 and action-1 attack loss 0.300 yields floor 0.599; the LP moves 0.003333 of its row to action 1, leaves sensitive loss exactly 0.599, and has a numerical fixed-bank gap near 1.1e−16. This checks the direction and scale of the registered loss floor without ACS rows.

## Verified correction at 03:02 UTC

Branch A now replays every round channel against the final rebased union bank and chooses only channels feasible under that bank. Its new counterexample test rejects an attractive earlier round that violates a later cut. The full focused adaptive plus inherited audit/solver suite passes **97/97**. Blocker 2 is cleared for channel feasibility. If an earlier feasible channel is selected, its old round objective/dual cannot be compared to a final-bank MILP bound; a final-cost replay is required for such a certificate. The published top-line prefit decision is therefore updated: the registered multi-round A center is conditionally ready once its source, protocol and sanitized input pins are committed and verified on the execution host. Branch B remains pending a complete fit path.

At 03:04 UTC I ran a fresh public synthetic A center through two rounds (`max_rounds=1`, H/D17 sources, A/SEX target). It completed with two round records, selected round 1, and wrote a completion receipt. This exercised the new final-bank selection path in the complete fit orchestration without ACS rows.

## Branch B executable review after first landing

`fit_b.py` now contains a full center path, but the first landing repeats the former A selection error: it chooses the lowest inner-selection loss across rounds without replaying earlier channels under the final rebased bank. Its return value is `current_q` (last round) even if the receipt names an earlier selected round. Both are prefit blockers for B; they were reported to the coordinator and Branch B owner for correction. Existing B fixtures test split, coefficient, dual, runtime and parity primitives, but none yet runs a two-round saved B center through selected-channel restore. No B ACS fit was launched by this reviewer.

## Branch B amendment review at 03:08 UTC

The `AMENDMENT_01.md` bytes match the SHA-256 in `PROTOCOL_AMENDMENTS.json` (`a77279820a0996159b99ea56c9a97eabf51179a0c5c722f5a9283ed2f569e0a8`). The amendment replaces `inner_check` as the split-stability resource with household-disjoint `inner_selection`, fixes the stability gain at at least 25% of the fitting gain, and retains the 100-unique/100-effective fitting-household support floor. It is explicitly dated before any Branch B ACS fit or outcome.

`fit_b.select_nested_partition` ranks fixed-price gains on `coefficient_split`, computes stability gains on `inner_selection` using its own U/PWGTP normalizers, and rejects undefined or subthreshold gain. A synthetic counterexample with fitting gain 4 and selection gain 0.8 rejects the split at ratio 0.20; selection gain 1.2 accepts it at ratio 0.30. The newly added `select_b_final_bank` replays every round against the final rebased union bank, excludes infeasible earlier releases, and returns the selected release rather than the last iterate. Focused B/refinement/reference fixtures pass **27/27**.

**B dispatch remains on hold:** `run_center_from_roles` still calls `_score_inner` on `child_roles["inner_check"]` each round and writes `inner_check_fixed_decoder_task` into the round receipt before the common independent audit. Current selection does not use that field, but fitting reads and exposes the reserved audit outcome. Defer this score to the common audit and add a negative test that B fitting does not access `inner_check` labels. Also pin the amendment SHA in the B dispatch/input receipt (or a validated immutable parent queue receipt) so a resumed B unit cannot silently cross the protocol version. The latter is a provenance condition; the direct `inner_check` read is the data-role blocker.

### Correction verified after the 03:08 amendment

Branch B now excludes `inner_check` from private child routing, never scores its labels in the fit, and pins the exact amendment SHA plus B/refinement/nuisance source bytes and source commit in `INPUTS.json` and `COMPLETE.json`. A synthetic full saved support-limited B fit and immutable resume succeeded with an object that raises if `inner_check` labels are accessed. The full adaptive plus inherited audit/solver suite passed **101/101** with stable pre/post source hashes. An earlier concurrent test run failed in a transient, simultaneously edited A parallel-slate fixture; that test and helper had disappeared by the stable rerun, so this is not counted as a persistent B failure. The original direct-read blocker and provenance condition are cleared. A non-alias refined-channel end-to-end synthetic fit is still desirable; its absence is implementation coverage risk, not a known semantic fault. Conditional B first-fit GO requires coordinator commit/pin of exact source and unit before dispatch; Linux x86 runtime parity remains required before a refined release is packaged as runnable.

## Matched control and task-only audit integration

The new fixed-bank control runner was independently reviewed by the Branch A owner. Its A/B loaders verify the selected center channel against the final frozen bank; the deterministic MILP and exact-token gradient use the same cost/cut matrices. A prefit discrepancy was corrected: constant replacement now uses registered existing token 0 even when another action is cheaper. The corrective synthetic fixture sets action 1 cheapest, failed against the former cost-selected token and passed after correction. This runner's `D_task` is only the final frozen-decoder rowwise optimum. The gradient grid is a fixed-bank comparator pending equal fresh attack/decoder opportunities. Because an earlier selected adaptive Q need not minimize the final bank, a separate same-final-cost LP solve and matched MILP lower bound are required before any integrality-gap statement.

The task-only baseline is a frozen nuisance-role residence predictor and 17-token quantizer on allowed X_A/H_A. Its exact audit law can differ among people sharing the same T32 code. Initial integration lacked a sanitized gate in the partition-control runner, a task-only archive completion receipt, and an audit adapter for this original-person law. The baseline owner corrected the first two: `_sanitized_record` is checked before prepared deserialization and `COMPLETE.json` inventories the private fitted model for `archive_unit`. The audit owner added a pinned model-backed exact person×17 law path to the common audit with token-0 replacement and existing-token randomized response. Synthetic direct and full-panel tests show no H_B/labels/ID access in the law and no law rows in the report. All adaptive plus inherited focused tests pass **129/129** after those corrections; no ACS job was launched by this reviewer.
