# Code checks: predecessor split search (commit 2ce5d171)

Scope: the code is read-only. Paths are relative to the worktree root unless stated otherwise. The synthetic demonstrations are in
`code_checks_demo.py`, with output in `code_checks_demo.out`. Run them with `PYTHONPATH=. /Users/nathansamson/PCRL/.venv/bin/python <script>`.
They use no ACS rows. The predecessor suite `tests/pcrl_adaptive_release_v1/` passes **181/181** locally on this worktree with the `.venv` Python.
No private or outer data were opened.

Abbreviations: R = `experiments/pcrl_adaptive_release_v1/refinement.py`, FB = `.../fit_b.py`, TB = `.../task_baselines.py`.

## A1. `fixed_price_gain()` is nonnegative by construction — CONFIRMED

- R:262-263 computes `g.sum(0).min() - g[left].sum(0).min() - g[~left].sum(0).min()`. This is
  `min_z Σ_parent g(z) − min_z Σ_left g(z) − min_z Σ_right g(z)`.
- Proof: let z* be the parent argmin. Then Σ_P g(z*) = Σ_L g(z*) + Σ_R g(z*) ≥ min_z Σ_L g(z) + min_z Σ_R g(z). The difference is therefore ≥ 0.
  R:264-265 turns a negative value into an assertion failure ("wrong sign"). R:266 then clamps with `max(gain, 0.)`.
- Demo: over 20,000 random splits, the smallest raw value was −5.3e-15, which is floating-point noise.
- Consequence: the score is an *opportunity*: the in-sample value of re-choosing tokens after the split. It cannot report that a split hurts.

## A2. The checking score re-optimizes on checking rows — CONFIRMED; the ratio test is lenient

- R:343-362: for each candidate, the checking rows use the *training-derived threshold*. The code then calls
  `fixed_price_gain(check_g[c], c_left[c])` at R:362. That call takes new argmins on the checking rows for the parent, left child and right child.
  No action fitted on the training rows is carried over. So the checking score is a reoptimized empirical opportunity, not the
  out-of-sample value of the training-fitted action change. It is ≥ 0 by A1.
- The checking side has **no support check**. R:361 requires only that each checking child is nonempty.
- `select_nested_partition()` (FB:504-608) calls `rank_splits(..., checking=...)` at FB:546-555. It keeps candidates with `gain > 1e-12` (FB:556).
  It rejects a candidate unless `checking_gain + 1e-12 ≥ 0.25 × fitting_gain` (FB:571-573, `MIN_CHECK_GAIN_RATIO = 0.25` at FB:26).
  AMENDMENT_01.md describes this rule and calls it "an empirical stability filter, not … held-out performance". That wording is accurate. The docstring's "separate checking scores" is looser.
- Pools and normalization: fitting uses `coefficient_split` and checking uses `inner_selection` (FB:750-753).
  On anchor 0 these hold 5,257 and 2,215 people (3,506 and 1,468 households; `DATA_ROLE_COUNTS.json`).
  `priced_original_person_rows` normalizes each pool separately: task weights `0.5/n + 0.5·w/Σw` (FB:389), attack weights `1/n` or `w/Σw` (FB:405-408).
  Both gains are therefore in per-population-average units and comparable in scale. However, the upward bias of a reoptimized
  min-gap grows as the pool shrinks, so the smaller checking pool inflates checking gains.
- Demo, pure-noise null with true split gain exactly 0 and pool sizes in the anchor-0 ratio:
  - mean fitting gain 0.031 versus mean re-optimized checking gain 0.054; the checking gain is never negative;
  - **73% of pure-noise splits pass the 0.25 ratio test**;
  - a frozen-action diagnostic (choose parent/child argmins on fitting rows, score them on checking rows) has mean −0.011 and is negative in 54% of trials, as it should be.
- Same pattern in the matched partition controls: TB:522-557 uses the same `rank_splits` checking score and `CHECK_GAIN_RATIO`.
  Because TB does not filter out zero gain, a zero-gain candidate satisfies `0 ≥ 0.25·0`. This affects the task_only and joint_risk policies (TB:553-557).

## A3. An empty positive-gain list is labeled `support_limited` whenever `min_households > 1` — CONFIRMED

Exact path:
1. `rank_splits` silently drops support-failing candidates with `continue` at R:339-340. They are not recorded anywhere.
2. FB:556 then drops support-passing candidates with zero gain.
3. `candidate_history` records only the survivors (FB:559-565).
4. If nothing survives, FB:566-567 sets
   `terminal_status = "support_limited" if min_households > 1 else "no_positive_fixed_price_gain"`.
5. The production call passes `min_households=100` (FB:776). The label is therefore always `support_limited` in production.
   FB:801-802 then maps it to `SUPPORT_LIMITED_ALIAS_A`.

Demo: 800 people and households, with 400 unique households in each child. `rank_splits` returns two support-passing candidates with gain 0.0.
`select_nested_partition` then returns `status='support_limited'` and `candidate_history=[[]]`.
The unit test `tests/.../test_fit_b.py:70-81` conflates the two causes: its rows have zero gain *and* 8 households.

TB behaves differently. It keeps zero-gain candidates, so its `support_limited` (TB:549-551) means "no support-passing candidate, or an empty checking child". That meaning is closer to the label.

## A4. Reproducing the support diagnosis — PARTLY reproduced

Reproduced from code:
- Each child needs at least 100 unique households and at least 100 Kish-effective households: R:336-340, together with FB:776-777 (`min_households=100, min_effective_households=100`) and the receipt fields at FB:711-712.
- Kish effective households are computed after aggregating each child's person weights by household (R:280-288).
- Thresholds are only the training quantiles 0.25, 0.5 and 0.75 of the active parent rows (R:330).

Committed record of "198":
- `results/pcrl_adaptive_release_v1/RUN_STATE.json` → `a0_B_center`: `"max_parent_unique_coefficient_households": 198`, `"parents_with_at_least_200_unique_households": 0`.
- `EVENT_LOG.jsonl` line 18 (`a0_B_support_limited_alias`): `max_parent_unique_households: 198`, `no_parent_can_support_two_children: true`.
- The prose restatement is in `STATE_REFINEMENT.md:3`.

Not reproduced:
- No code in `experiments/` computes a maximum per-parent household count; a grep for `max_parent`/`parents_with_at_least` finds nothing. The figure is a coordinator-computed summary.
- The per-parent counts need the private `coefficient_split` rows and T32 codes. They are not in any committed file.
- `SPLIT_SEARCH.json`, stored in the private B archive, cannot supply them either: its `candidate_history` holds only support-passing, positive-gain candidates (FB:559-565).

Anchors 1 and 2: **no equivalent receipt exists**. `RUN_STATE.json` `a1_B_center`/`a2_B_center` and `EVENT_LOG.jsonl` lines 26-27 record only the status, the 100/100 floors, `realized_states: 32` and the parity counts (5,317 and 5,334 people).
There is no per-parent household count and no "no parent ≥ 200" flag. The sentence "the other two receipts also report no admissible split" is supported only by the status string, and by A3 that string cannot distinguish support failure from zero gain.

Context check, not a reproduction: anchor 0's `coefficient_split` has 3,506 households across 32 parents, a mean of about 110. A maximum of 198 is plausible.

**Correction to the "cannot" argument.** Children are disjoint in *persons*, not households (R:332-333). Unique households are counted per child (R:336).
A household whose members fall on both sides of a person-level feature counts in both children. The features `residual`, `task_posterior` and the risk features are person-level, and there are about 1.5 people per household.
So "198 unique households ⇒ two children cannot each have 100" is a heuristic, not a proof.
Demo: a parent with 198 two-person households, split on a feature that differs within each household, yields children with (198, 198) unique households and (198, 198) Kish households. It passes the floor.
Conversely, 200 households is not sufficient either: the quantile-only thresholds and PWGTP-dispersed Kish counts can each still fail.
The actual binding cause in each anchor is **not determinable** from committed artifacts plus code.

## A5. Scope — CONFIRMED: the old verdict is unchanged

- All three B units report 32 realized states and zero rules, with Q bitwise identical to A (`RUN_STATE.json`, `STATE_REFINEMENT.md`).
- In code, the status `support_limited` at 32 states means the *first* iteration's ranked list was empty (FB:566-568, 598-599, 801). So the lenient ratio test (A2) was never reached.
- A frozen-action or support-correct re-labeling cannot create accepted splits there.
- `selection_lock.py:144-145` drops B as an exact A alias. No inference endpoint involves B.
- The findings in A1-A4 correct the *meaning* of the labels and scores and the "198 ⇒ impossible" argument. They do not correct the result that zero splits were accepted.

Guidance for the new study:
- count support-rejected, zero-gain, checking-rejected and accepted proposals separately;
- replace the reoptimized checking score with a frozen-action score that can be negative;
- apply support checks on the checking side too;
- record per-parent unique and Kish household counts in the committed aggregate receipt for every anchor.
