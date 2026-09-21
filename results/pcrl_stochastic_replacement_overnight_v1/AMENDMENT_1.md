# AMENDMENT 1 — responses to Terminal 2's pre-outcome review

**Date: 2026-09-21, 09:30Z.** Terminal 2's review
(`REVIEW_REPLACEMENT_PROTOCOL.md`, read at commit `0af175c35`) raised three items. All three are
**accepted**. Timing is disclosed per item: M1 and M2 concern stages that **never ran**, so they
constrain no published number; **M3 does** touch a published number and is handled in full below.

The registration, the `.001` allowance and every other scientific criterion are unchanged.

---

## M1 — solver honesty for `δ > 0`: accepted, registered, never bound

**The documentation gap is real.** `METHOD.md` §2 registered the independent-recomputation
requirement only for the `δ = 0` linear path. It should have been registered for every budget.

**Registered now**, for any constrained fit in this study or a successor: per fit, record solver name
and status; primal residuals `max_t |Σ_z Q[t,z] − 1|` and `min_{t,z} Q[t,z]`; the optimality gap
where the solver reports one; an **objective reconstruction** recomputing `Σ p(t) Q[t,z] D[t,z]` from
the returned `Q`; and an **independent CMI recomputation** from the returned `Q` and `p̂_r`, for
**every** `δ`. Tolerances `1e-8` on residuals and `1e-6` on constraint violation; a fit failing any
of them is refit on the registered alternate solver schedule and recorded as such.

**Status:** no constrained fit ever ran, so no published number rests on solver arithmetic. The
*implementation* already satisfied the requirement — `channel._solve_verified` recomputes `cmi`
independently of the solver's reporting and retries on a second solver when the recomputation fails
the budget, and `_report` emits `verified_cmi`, `constraint_violation`, `row_sum_residual`,
`negativity_residual` and `optimality_gap`. The only evidence of it operating is
`synthetic/DETERMINISTIC_COMPARISON.json`, where every cell's recomputed violation is `≤ 1e-6`, and
`test_synthetic.py::test_recorded_comparison_is_internally_consistent` asserts it.

## M2 — audit attackers receive unrounded continuous `H`: accepted, stated, never ran

**Registered:** the audit slate is fitted on the **original continuous** `H_A` / `H_B`. Conditioning
cell indices are optimizer artifacts of the fitted finite model and **never appear on any audited
wire**. A full-view attacker receives the unrounded probability vectors. Asserted in code at audit
time.

**Status:** never exercised — no audit ran. Terminal 2 is right that the distinction carries the
whole weight of a protection result, which is exactly why the precursor's binning fixture (a full
`log 2` of measured leakage dropping to zero under coarsening) is retained in the test suite.

## M3 — the correction family of 68: accepted, and it touches a published number

**Terminal 2 could not reconstruct 68 because the arithmetic does not survive being written out.**
What `RUN_MATRIX.json` encoded was `4 × 2 × 2 × 4 + 4 = 68`: four primary sensitive endpoints, two
weightings, two routes, four nominees, plus four external-baseline comparisons. Written out, that
count **double-counts each endpoint across the two routes** (the same endpoint test appears in both)
and **omits the residence tests** each route requires. It is not a defensible family.

**It is not silently corrected**, because a published number used it: the Stage-R adjusted bounds
were computed at `z = 3.18` from this divisor.

**How the published number is affected — fully, rather than by re-deriving a divisor that suits it.**
The Stage-R contrasts are **screening** quantities. `PROTOCOL.md` §3 and `STATISTICAL_PLAN.md` §4
registered screens as **point-estimate scheduling rules, explicitly not confidence claims**, and the
frozen family was declared for **finalist** comparisons that never ran. Applying any correction to
Stage R was therefore conservatism beyond the registration, not the registered family being spent.

Rather than pick a new divisor, the honest report is the **sensitivity**:

| contrast | estimate | SE | `z` needed to reach zero | family size that would void it |
|---|---|---|---|---|
| `H+raw` − `J`, unweighted | −0.01533 | 0.00279 | 5.50 | **≈ 2.6 × 10⁶** |
| `H+raw` − `J`, person-weighted | −0.01850 | 0.00332 | 5.57 | **≈ 3.9 × 10⁶** |
| `H+T(64)` − `H`, unweighted | −0.01679 | 0.00374 | 4.48 | ≈ 1.4 × 10⁴ |
| `H+T(64)` − `H`, person-weighted | −0.01680 | 0.00463 | 3.63 | **≈ 351** |

| divisor | `z` | `H+raw` − `J` adjusted upper |
|---|---|---|
| 68 (as published) | 3.180 | −0.00646 |
| 100 | 3.291 | −0.00615 |
| 200 | 3.481 | −0.00562 |
| 1000 | 3.891 | −0.00448 |

**Conclusion, and one downgrade.** The **headline** result — the unquantized A-side view beating
same-host `J` — is insensitive to the divisor: it would take a family of roughly **2.6 million** tests
to void it, so the defective count does not threaten it and no re-derivation could rescue or destroy
it. **But `H+T(64)` − `H` under person weighting is downgraded**: at ≈ 351 it is *not* robust to a
plausibly larger family, so it is reported from here on as a point estimate with its interval, and
**not** as a corrected significant finding. `RESEARCH_DECISION.md` §3 is amended accordingly.

**Not fixed by picking a new number.** A defensible family can only be written out against a
finalist set that does not exist, since selection never ran. Any successor study must record the
arithmetic, not the total.

## Optional items

* **Realised support of `Q`** — accepted as a good idea; not applicable, no `Q` was fitted on ACS.
* **`argmax`-rounded ablation** — already registered as an ablation and never a candidate
  (`RUN_MATRIX.json` `controls_...`); restated here because rounding changes the released object and
  can violate the constraints.

## Terminal 2's two closeout requests

Both done before this amendment: the branch is pushed
(`35bb9dc3932e9eafab10490f073433ce57f4499c`, remote SHA verified by `git ls-remote`), and
`terminal_1/STATUS.json` is refreshed to the closed state with live unit counts.
