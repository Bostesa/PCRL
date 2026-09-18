# SELECTION_DIAGNOSIS — what the previous run's selection objective discarded

Generated 2026-09-18T21:56:22.691540+00:00 from `CHECKPOINT_LEDGER.json`
(`sha256 e16ed048f287ff5f3b45397b7028ae414798f84ec827d13df78305ec16a3eeda`), which is the machine-readable backing for every
number below. Source study: the completed direct-adversarial run at evidence commit
`69e790af36c5ca53203dab17b757a8e3415ee934`, read **read-only**.

**Scope.** stored records only: no refit, no re-audit, no 2018 or 2017 outcome opened, no residence or commute label read. This stage refits nothing and re-audits nothing:
it is arithmetic on the monitor records and parameter states the previous run saved.

---

## 0. The prerequisite question

> Did previous training generate plausible alternatives that its selection objective
> discarded, or were the trajectories themselves unpromising?

**Answer: it generated them, and selection discarded them.** The evidence is in §2
and §3. This is a statement about *selection*, not about whether the discarded
checkpoints would have audited well — nothing here has been audited.

## 1. The reconstruction is exact

* `114` fitted trajectories recovered, 7 checkpoints each.
* `utility == source + w*distortion` and `monitor_score == utility + beta*penalty`
  hold on every stored record of all three slates: `0` identity failures.
* Re-deriving `argmin_t (C_t + 1.0*D_t + beta*P_t)` from the components reproduces the
  published selected index in **114 of 114** trajectories, with no mismatches.

So the published selection rule, the stored components and this study's arithmetic
are the same object. Every counterfactual below inherits that.

### 1.1 Gradients and loss scales agree with the implementation

* Mapper gradient norms over `6954` logged steps: all finite, `0` exact zeros,
  range `[0.2028, 1.3679]`, median `0.3479`. The mapper step
  was doing work at every logged step; nothing was silently detached.
* Dose-response against the contemporaneous training slate, averaged over arms:

| policy / beta | penalty first | penalty last | delta | distortion last |
|---|---|---|---|---|
| `C1 / beta=0` | 0.0558 | 0.2120 | +0.1562 | 0.0030 |
| `C1 / beta=0.1` | 0.0558 | 0.1904 | +0.1346 | 0.0048 |
| `C1 / beta=0.3` | 0.0556 | 0.1289 | +0.0733 | 0.0151 |
| `C1 / beta=1` | 0.0556 | 0.0701 | +0.0145 | 0.0419 |
| `C1 / beta=3` | 0.0558 | 0.0529 | -0.0029 | 0.0743 |
| `L1 / beta=0.1` | 0.0280 | 0.1026 | +0.0746 | 0.0033 |
| `L1 / beta=0.3` | 0.0280 | 0.0904 | +0.0624 | 0.0066 |
| `L1 / beta=1` | 0.0280 | 0.0499 | +0.0219 | 0.0244 |
| `L1 / beta=3` | 0.0280 | 0.0290 | +0.0010 | 0.0521 |
| `L2 / beta=0.1` | 0.0560 | 0.1941 | +0.1381 | 0.0044 |
| `L2 / beta=0.3` | 0.0556 | 0.1269 | +0.0713 | 0.0165 |
| `L2 / beta=1` | 0.0556 | 0.0677 | +0.0120 | 0.0426 |
| `L2 / beta=3` | 0.0560 | 0.0473 | -0.0087 | 0.0702 |

The penalty rise is monotonically suppressed as `beta` grows and the final distortion
rises monotonically with `beta`. The declared sign convention therefore holds and the
protection gradient does real work. **But** the contemporaneous slate gains 32x its starting budget along a trajectory, so a penalty that merely stops rising at high beta means the channel kept pace with a strengthening attacker. It is not evidence that absolute recovery fell, and the 2018 audit is the only thing that speaks to that.

## 2. 39/72 unchanged main channels was a SELECTION outcome

In the registered main block alone (width x policy x beta x seed, no repeats or
ablations), **39 of 72** trajectories published step 0 — the
previously reported 39/72. Across all fitted trajectories:

* **63 of 114** trajectories published their
  **step-0** checkpoint, i.e. the unmoved starting channel.
* **63 of those 63** had in fact moved: their final checkpoint sits at a
  median relative parameter displacement of **0.0991** from step 0
  (range `[0.0753, 0.1419]`), with median final teacher
  distortion **0.0119**.

**No trajectory was stationary.** The correct reading of the previous run's 39/72
unchanged main channels is: optimisation moved the channel, and the registered
selection objective preferred the starting point anyway. It is *not* evidence that
optimisation never moved, and it is *not* evidence that teacher distortion alone
caused the failure — §3 shows the objective as a whole, not the distortion term
alone, is what selected.

### 2.1 Step 0 usually won by a real margin, not a near tie

Margin from step 0 to the runner-up checkpoint, across the 63 step-0
selections, on the registered fresh-probe score:

| percentile | margin |
|---|---|
| p0 | 0.00175 |
| p10 | 0.00368 |
| p25 | 0.00592 |
| p50 | 0.01001 |
| p75 | 0.01114 |
| p90 | 0.01338 |
| p100 | 0.01399 |

At a near-tie threshold of `0.005`, only
**12 of 63** step-0 selections were
near ties. The rest were decisive. Selection was not balanced on a knife edge that a
small implementation change would have tipped.

## 3. What the moving arms actually paid, and for what

Across the **51** trajectories that did select a moved checkpoint, the
selected-minus-initial decomposition of the score (mean over arms):

| term | mean | median |
|---|---|---|
| `source` | +0.01405 | +0.01368 |
| `distortion` | +0.02791 | +0.02541 |
| `beta_times_penalty` | -0.08354 | -0.06360 |
| `monitor_score` | -0.04158 | -0.02771 |

Reading: a moved checkpoint is bought with **both** a higher source loss and a higher
teacher distortion, paid for by a lower protection penalty. The distortion cost is
about **2.0x** the source cost in
these units. That makes the teacher term a large share of what the selection rule was
charging for movement — **but the source term is not zero either**, so this is not a
one-factor story, and the coefficient `1.0` on distortion was specified, never shown
optimal.

## 4. Counterfactual rescoring: selection only

> **This is not a utility ablation.** The trajectories rescored below were produced
> under `gamma = 1`. Rescoring them at another `gamma` changes which of *those*
> checkpoints is selected. It says nothing about what retraining at another `gamma`
> would produce, because a different `gamma` would have generated a different
> trajectory. Track N answers that question by actually retraining; this section
> cannot and does not.

`argmin_t (C_t + gamma*D_t + beta*P_t)` on the same stored records:

| gamma | step 0 | 100 | 200 | 300 | 400 | 500 | 600 | selections changed vs gamma=1 |
|---|---|---|---|---|---|---|---|---|
| `0` | 45 | 11 | 6 | 32 | 10 | 2 | 8 | 50 |
| `0.01` | 45 | 12 | 6 | 31 | 10 | 2 | 8 | 49 |
| `0.1` | 48 | 11 | 6 | 32 | 9 | 1 | 7 | 43 |
| `1` | 63 | 23 | 15 | 10 | 2 | 0 | 1 | 0 |

Removing the teacher term entirely moves **50
of 114** selections and drops the step-0 count from
63 to 45. By policy:

| policy | selections changed at gamma=0 | arms |
|---|---|---|
| `C1` | 23 | 48 |
| `L1` | 8 | 24 |
| `L2` | 19 | 42 |

The `gamma` values are **not** filtered by residence or commute outcome, and none was
removed for being inconvenient: the registered grid `{0, .01, .1, 1}` is reported in
full, including `gamma = .01`, whose effect is nearly indistinguishable from `gamma = 0`
at this resolution.

## 5. The zero-gain option was **not** active too often

`G_j = max(0, max_k g_jk)`. A raw count of zeros in the stored records reads
**23.33%**, which invites the
conclusion that the baseline is beating every attacker. That count is wrong, and the
reason is structural rather than empirical:

* a role with no attacker slot reports a zero by the policy definition, not by measurement: L1/L2 spend their two coalition-equivalent slots on local replicas and never fit an AB attacker, so every AB entry in a local arm is structural.

Separating the two:

| slate | attacker budget | measured zero-option rate |
|---|---|---|
| fresh probe (the selection yardstick) | 300, equal per checkpoint | 0.23% |
| contemporaneous training slate | 100 rising to 3220 | 0.43% |
| final slate | equal budget, wrong target | 1.85% |

Per role, on the fresh probe, restricted to roles that actually had an attacker slot:

| role | measured zero rate | structural (no slot) rate |
|---|---|---|
| `A/RAC1P` | 0.00% | 0.00% |
| `A/SEX` | 0.38% | 0.00% |
| `A/public_coverage` | 0.00% | 0.00% |
| `AB/RAC1P` | 0.30% | 57.89% |
| `AB/SEX` | 0.89% | 57.89% |

**Conclusion, with its residual uncertainty.** Where an attacker existed, it found a
positive gain essentially always. The three candidate explanations offered for a
frequently-active zero option resolve as follows:

* *the correction attacker is underfit at the probe budget* — **ruled out as the
  driver**. A 10x larger contemporaneous budget gives essentially the same measured
  rate, so the 300-update probe is not manufacturing zeros.
* *the service-only baseline is strong* and *the achievable gain is genuinely near
  zero* — **not separated, and not separable from these records**, because both
  predict the same small positive margins. They are also barely relevant here: the
  measured rate is small enough that the clamp is not what shaped selection.

This removes one hypothesis from the list of explanations for the previous run's
failure. It does **not** establish that the attackers were strong in absolute terms —
300 fresh updates on three differentiable families is a weak slate by design, and
only the audit speaks to absolute recoverability.

## 6. Comparisons that are invalid because the budgets differ

Along one trajectory the contemporaneous slate grows from
`100` to `3220`
attacker updates — a factor of `32.2`.

* **Valid for cross-checkpoint comparison:** `monitor_scores`.
* **Invalid:**
  * contemporaneous_monitor_scores (budget rises with the step by construction; the measured penalty therefore rises with training even when the channel does not move)
  * final_slate_monitor_scores (equal budget but a single slate specialised to the final channel, which understates early-checkpoint recoverability)

Both invalid slates are biased in the **same direction**, toward never moving the
channel. The previous run reached the same conclusion prospectively and switched to
the equal-budget probe before opening any outcome; this diagnosis confirms the
switch was necessary and reproduces its arithmetic exactly.

## 7. What this does and does not license

Established:

1. The previous run's selection arithmetic is exactly reproducible.
2. Every trajectory moved; unchanged published channels are a selection outcome.
3. Step 0 usually won decisively, not by a near tie.
4. Removing the teacher term changes roughly half of all selections.
5. The zero-gain clamp was essentially inactive wherever an attacker existed.

**Not** established, and not to be inferred:

* that a discarded checkpoint would have audited better — none of them has been
  audited, and the monitor score is a weak selection yardstick, not a protection
  measurement;
* that `gamma < 1` is better — §4 changes selection on fixed trajectories only;
* that teacher distortion alone caused the previous failure — §3 shows the source
  term moves too;
* that J-initialised fine-tuning behaves like this at all — the previous study
  initialised from `A0` in every arm and never ran it.
