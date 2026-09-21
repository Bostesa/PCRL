# PRECURSOR_CORRECTIONS — interpretation errors in `pcrl_stochastic_channel_v1`

**Date: 2026-09-21.** Written before any new outcome of this study was inspected.

Subject: the closed precursor, pinned at `cd895e4291b3dc2c0ed5e9c5482415d30f4e9f2a`
(branch `research/pcrl-stochastic-channel-v1`).

**What is preserved and not touched.** The precursor's records stay exactly as published: its
`REGISTRATION.md`, `RESEARCH_DECISION.md`, `AMENDMENT_1.md`, `gates/G0.json`, `gates/G2.json`,
`STAGE_B_RESULTS.json` and its **operational FAIL at G2**, together with the fact that its stages C
and D were never run. Nothing here reverses that operational decision, and nothing here justifies
repeating its experiment. These are corrections to *interpretation and naming*, plus two genuine
implementation defects in an auxiliary diagnostic.

Each item was independently verified against the pinned source, not against a summary.

---

## C1 — "J" in the precursor is a 20-column view, not the service predictions

**Verified.** `gates/G2.json` records `ref_J_probe_input_dim: 20`. With
`dax.H_A_WIDTH = 4` and `dax.A0_WIDTH = 16`, the baseline the precursor called "same-host J" is
`[H_A, Z_J]` — the four service-probability coordinates **plus a sixteen-coordinate auxiliary
channel**.

**What was wrong.** The precursor's `RESEARCH_DECISION.md` §1 wrote that the code's residence signal
"is already subsumed by the released service predictions `J`". That sentence conflates two different
objects. The comparison actually run was against `H_A` *and* a 16-dim auxiliary channel. A statement
about "the released service predictions" is a statement about `H_A`, four columns, and was never
measured as the baseline.

**Consequence.** For a *replacement* design the relevant reference is `H_A` alone. The precursor
never measured that contrast. Its headline claim is therefore not supported by its own baseline, and
the primary comparison of the present study is `[H_A, Z]` against `[H_A, Z_J]`, with `H`-only also
reported.

## C2 — Two finite probes failing is not subsumption, independence, or exhaustion

**Verified.** The measurement was fresh `logistic` + `mlp` probes (`stage_b.probe_residence`,
`fit_candidates(..., families=('logistic','mlp'), budget={'mlp':{'epochs':40}})`) on
`[H_A, Z_J, onehot(T)]` versus `[H_A, Z_J]`, selected by validation log loss.

**What was wrong.** Four claims in the precursor exceed that evidence:

1. "already subsumed" — an information-theoretic claim. Not established by two fitted predictors.
2. Conditional independence of residence from `T` given the baseline — never tested.
3. "exhaustion of the input space" — the A-side inputs were probed through **one** quantizer at two
   resolutions, not exhausted.
4. "a mechanism-level explanation for why this line of work has repeatedly gone negative" — a
   causal explanation of *other* studies, asserted from one screen.

**The correct statement.** At the two tested resolutions, with this quantizer and this probe family,
on the 2018 development pools, no added validation utility was found over `[H_A, Z_J]`. That is an
operational screening result about predictors, not about information.

## C3 — The exactly-zero inclusion-respecting advantage is an identity, not a measurement

**Verified.** `stage_b.run_seed` computes
`best_code = min(unconstrained['log_loss'], ref_j['log_loss'])` and
`adv = ref_j['log_loss'] - best_code`. Whenever the code probe is not better, this is identically
`0.0` by construction.

**What was wrong.** The precursor reported "exactly `0.00000` in 3/3 anchors" and again "in all six
cells" as though the repetition across anchors were corroborating evidence. It is a tautology: the
`min` selects the ancestor, and the difference of a quantity with itself is zero. **Ancestor fallback
selecting the same J predictor explains exactly zero observed improvement — and equally, explains
exactly zero observed disclosure change.** It carries no information about whether other predictors
or other information gains exist.

**Consequence.** This statistic is retained only as a bookkeeping check that the ancestor is
available in the slate. It is never again reported as a finding, and its zero variance is never
reported as precision.

## C4 — `_baseline_score` is label-dependent; the "probability-decile partition" is withdrawn

**Verified, and this is a real implementation defect.**

```python
def _baseline_score(probe: dict, x) -> np.ndarray:
    """The baseline probe's per-row loss, used only as a 1-d partition coordinate for B1."""
    return np.asarray(probe['per_row_loss'], np.float64)

def _per_row(candidate, x_val, y_val):
    p = np.asarray(candidate.predict_proba(x_val), np.float64)
    y = np.asarray(y_val, np.int64)
    return -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12))
```

`per_row_loss` is `-log p(y_true)`. It is a function of the **true residence label**, not a predicted
probability. The precursor's `run_seed` nonetheless built
`decile_cells` from it and labelled the result `probability_decile_partition`.

**Two errors.** The name is wrong — it is a *loss* decile, not a probability decile. And the object
is invalid as a conditioning partition: conditioning coordinates must depend only on permitted
features and predictions, never on the label. A label-dependent partition can manufacture or destroy
apparent conditional signal.

**Withdrawn.** The `probability_decile_partition` numbers in `gates/G2.json`
(`-0.06931 / -0.08077 / -0.08806`) are withdrawn as meaningless, not merely as noisy. The precursor
already described them as a variance artifact; they are worse than that.

**This did not drive the stopping decision.** `G2` was decided by the B2 deployable screen. The
label-dependent partition entered only the auxiliary B1 diagnostic, which the precursor itself
reported as "not reliably estimable". The operational FAIL therefore stands on B2 alone.

## C5 — A cross-fitted smoothed-table loss difference is not a ceiling

**Verified.** `stage_b.ceiling_from_code` returns a key literally named
`ceiling_reduction_nats`, and its docstring asserts "A positive value is an upper bound on what any
function of `T` can add **in this fitted model**".

**What was wrong.** No such bound was derived. The computed object is the difference in
cross-fitted, pseudocount-smoothed *table-predictor* log loss between two cell partitions. That is a
property of one estimator at one smoothing setting and one cell count; it is neither an upper bound
over functions of `T`, nor a plug-in estimate of a mutual information. The clearest proof that it is
not a bound is that it came out **negative** — an information quantity cannot be, so a quantity that
can be negative is not bounding one. Negative estimates are evidence about the estimator's variance,
never evidence for a negative information quantity.

**Renamed.** In this study the analogous quantity is called
`table_predictor_loss_difference`, with the estimator, cell count, rows-per-cell and pseudocount
reported alongside. The word "ceiling" is not used unless a bound is actually derived for the
computed object, and no such derivation is attempted here.

## C6 — Precision: the condition, and two over-readings

The correct condition is
`estimate + critical_value * SE <= margin`, evaluated for the actual candidate/comparator pair.

Already corrected in the precursor's `AMENDMENT_1.md`, and re-affirmed here. Two further
over-readings are corrected now:

* **A negative estimate is not universally required.** A small positive estimate with a small enough
  `SE` also satisfies the condition. `AMENDMENT_1.md` §4 item 4 wrote that a pass "requires a
  candidate that *reduces* additional sensitive recovery"; that is true only at the historical
  half-widths, not as a general requirement. Corrected.
* **Unchanged sensitive predictions do not imply unchanged useful capability.** Two releases can
  yield identical measured sensitive recovery and differ in residence utility. The precursor's
  reasoning slid between "no measured disclosure change" and "no change", which does not follow.

Historical half-widths continue not to constrain a new pair's `SE` (`AMENDMENT_1.md` §1, error 2).

## C7 — For a J-retaining extension, strictly-lower disclosure than J is not an achievable target

**Verified as a structural fact, and it reframes the selection criterion.**

If a release retains `J` — as `[H_A, Z_J, R]` does — then any attacker can ignore the appended block
and reproduce the `J`-only attack. So optimal sensitive recovery from the extension is **at least**
that of `J`: appending cannot remove information already available in `J`. Symmetrically, an
ancestor-inclusive validation maximum over a slate containing the `J` predictor cannot score *worse*
than that predictor, so it cannot show improved privacy below the same ancestor by construction.

**What was wrong.** The predecessor programme made a *negative incremental disclosure* relative to
`J` part of the selection target for a `J`-retaining extension. In the ideal limit that target is
unreachable: the best attainable is equality. Chasing it can only select releases that are
numerically indistinguishable from `J`, which is what
`pcrl_utility_extension_v1`'s `RESEARCH_DECISION.md` described as "the selection preferred
nearly-untouched J".

**Consequence.** A strict disclosure *improvement* is available only to a design that **removes**
`J` from the release — a replacement. This is the central reason the present study tests a
replacement contract rather than another extension, and it is registered before any outcome.

---

## What follows, and what does not

**Follows.** The precursor asked whether `T` *adds* utility to a 20-column baseline that already
contains a 16-dim auxiliary channel. For a replacement design that is the wrong question, and C7
shows the extension framing cannot deliver a strict disclosure improvement even in principle.

**Does not follow.** No result here says a replacement *will* work. The precursor's operational FAIL
at G2 is preserved. The corrections are not a retrospective rescue and do not entitle this study to
re-run the precursor's experiment; the present study runs a different, separately registered
comparison.

**Fixtures.** `tests/pcrl_stochastic_replacement_overnight_v1/test_precursor_corrections.py` pins
C1 (widths), C3 (the `min` identity), C4 (label dependence of the withdrawn partition), C5 (the
quantity can be negative, so it is not a bound), C7 (an ancestor-inclusive maximum cannot beat its
ancestor), and the §3 replacement counterexample.
