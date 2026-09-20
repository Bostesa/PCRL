# STAGE A — repairs and diagnostics, before any new study is registered

Scope: fix what `VERIFICATION.md` confirmed, add the review's fixtures, and expose the
constant-channel penalty. **No historical result is recomputed, no release is refitted, no decision
is revised.** The pilot's verdict (`pcrl_utility_extension_v1`, Tier 2 FAIL) stands unchanged.

Environment used for these fixtures: Python 3.13.7, torch 2.14.0, numpy 2.5.3 — **not** the study
environment (torch 2.10.0, numpy 2.4.2). Every test added here is data-free and establishes a
mechanism or an invariant, never a study number, so the version difference is immaterial. Any
reproduction of study numbers must use the environment pinned in
`results/pcrl_utility_extension_v1/REPRODUCE.md`.

## 1. In-slate untouched-J predictors — `experiments/pcrl_stochastic_channel_v1/slate.py`

The confirmed defect (VERIFICATION §A): the extension's slate contained no predictor that reads
`[H_A, Z_J]` while ignoring `R`, although METHOD §1 justified the promise of one by the fact that a
larger input can make a *finite* learner worse.

Added:

* `route_j_anchor(view, width_a, columns)` — the columns of an extended wire that reproduce the
  untouched-J wire. `A → 0..19`; `AB → 0..19` plus `H_B` at `width_a, width_a+1`, because
  `dax.build_wires` appends `B` last so `H_B` does not sit at a fixed offset. `B` is refused: the
  `B` wire is `H_B` in every system and canonical `B` candidates are shared unchanged.
* `j_anchor_metadata(...)` — marks the candidate `source_condition='ref_J'`, `j_anchor=True`,
  `ignores_extension=True`, `anchor_ancestor=False`. This is the **separate-comparator vs
  in-slate-predictor** distinction the pilot's prose blurred, now explicit in the record.
* `build_j_anchor_candidates(j_audits, width_a)` / `add_j_anchors(...)` — the supplement, covering
  A and AB roles for both budgets. Skips the source's own H ancestors (the extension routes those
  itself) and its inherited singletons (`inherit_singletons` regenerates them here).

Hook: `acs_spectral_audits.build_audits` gains a keyword-only `extra_candidates=None`. It merges
after the ancestor block and before `inherit_singletons`, refuses `B` roles, and refuses id
collisions. **With the default `None` the function behaves exactly as before** — the historical
tests (`tests/test_acs_spectral_audits.py`, `tests/test_acs_fixed_predictions.py`,
`tests/test_acs_preservation_audits.py`) pass unchanged.

The routing validates itself. `build_audits` already re-scores every candidate that carries
`projection_columns` and asserts the recomputed validation metrics equal the recorded ones; the
restore path additionally re-checks `fit_hashes` and `validation_hashes` on the routed columns.
A J anchor can only pass those if its selected columns reproduce the J wire bit for bit, which is
guaranteed by `ext.assert_parity` holding `H_A` and `Z_J` byte-identical.

**Registered direction, before any run.** Selection is `argmin` validation log loss, so this
supplement can only lower the selected attacker loss — raising measured recovery and raising the
increment over J. It makes the protection screen strictly *harder*. It is a correctness fix, not a
repair, and nothing in the pilot is re-derived from it.

## 2. Constant-channel diagnostic — `experiments/pcrl_stochastic_channel_v1/diagnostics.py`

`constant_channel_gain(...)` runs the **shipped** `extension.role_gains` (pinned by
`test_diagnostic_uses_the_shipped_role_gains_not_a_reimplementation`) against an extension whose
column standard deviation is exactly zero. Numbers and the baseline-quality ladder are in
`VERIFICATION.md` §B and `CONSTANT_CHANNEL_DIAGNOSTIC.json`.

`null_corrected_gain(...)` reports the raw gain, the matched constant-channel null at identical
seeds and step budget, and their difference. It is documented in the source as a **calibration of
scale, not an unbiased estimator** of `I(S;R | view)`: the two runs share an initialisation and a
step budget but not a loss surface.

What this does and does not license:

* It **does** show the reported training gain is not zero under the null, and that the statistic is
  one-sided (`clamp(min=0)` on an `argmax`) so slack can only inflate it.
* It **does not** quantify the effect in the study. The magnitude is baseline-dependent and the
  real `p0J` is far better fitted than any rung of the ladder.
* It **does not** touch the pilot's audited leakage, which is recomputed by freshly fitted
  attackers in `run_dev_2018`.

## 3. Review fixtures vendored — `experiments/pcrl_stochastic_channel_v1/fixtures/`

`stochastic_channel_fixtures.py` is vendored byte-for-byte from the 2026-09-20 review bundle;
sha256 `998573c81c4b5558ea524a09eaddbc893a19ddd9f7f065949685336c4092c085`, pinned by test and
cross-checked against the value the script records inside its own JSON output. The suite
re-executes it in a temporary directory and asserts the regenerated JSON is identical, then
independently recomputes the exact-independence claim in `fractions` rather than trusting the
artifact.

## 4. Deferred to the stage that builds the mechanism

"Test that the released object is the sampled token, not its probability vector" cannot be written
before the mechanism exists. It is registered as a **gate** in `REGISTRATION.md`, not left to
discretion: releasing `Q[t,:]`, its logits, or an expected prototype is a different mechanism and
may disclose `T`. The review's own fixture already records the leakage of the probability-vector
release for both toys (`releasing_probability_vector_sensitive_information_nats`: 0.4748 and
0.2499 nats, against 0 for the sampled token).

## 5. Test inventory

`tests/pcrl_stochastic_channel_v1/` — 29 tests, all passing:

| file | tests | covers |
|---|---|---|
| `test_slate_routing.py` | 14 | column routing, `H_B` placement, projection composition, `B` refusal, anchor metadata, supplement shape, opt-in hook, selection monotonicity |
| `test_constant_channel.py` | 7 | positive gain on a constant channel, one-sidedness, zero at init, null correction both ways, baseline-quality decay, shipped-code guard, constant detector |
| `test_review_fixtures.py` | 8 | vendored sha, self-reported sha, byte-identical regeneration, deterministic/stochastic separation, exact independence recomputed, both coarsening counterexamples, teacher-fidelity counterexample |

Full repository suite after these changes: **772 passed**, 0 failures.
