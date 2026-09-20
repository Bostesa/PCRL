# RESEARCH_DECISION — `pcrl_utility_extension_v1`

All numbers are **2018 development** on repeatedly used pools. Residence is reserved-from-training
capability, not a blind endpoint. The tier reached is **Tier 2 (pilot), FAIL**; Tiers 3 and 4 were
not started, by the protocol rule that a failed pilot does not open a larger grid.

## 1. Verdict

**Useful capability improved; measured protection did not hold at the declared operating point.**
A small appended channel R reliably adds capability on the declared proxy (10-45% reduction in
residual teacher-reconstruction MSE over the frozen J release, every configuration, all three
anchors, validation split), and every configuration passed the historical source allowances. No
configuration kept the additional measured sensitive recovery over J within the declared .001-nat
screen. The best protection achieved was +0.0016 nats (worst endpoint, `X_r2_L2_b030`), at the
lowest capability gain in the grid (0.111).

| decision level | result |
|---|---|
| Mechanism works as specified | **Yes**: H and Z_J byte-identical on every pool and unit; R depends only on A-side inputs; 42/42 units fitted; 42 audited (1 exact duplicate); 0 quarantines; 0 probability-integrity failures |
| Additional useful capability | **Yes, on the proxy** (reconstruction), not demonstrated on residence |
| Bounded disclosure at the declared screen | **No** |
| Coalition-specific benefit | **Not established** (C1 no better than matched L2 at matched beta) |
| Candidate for confirmation | **Not reached** |

## 2. The exploratory reanalysis of the completed 372-slot study

Under a NEW utility-first criterion (residence-loss reduction >= .003 nats with an adjusted interval
below zero, and one-sided adjusted upper bounds <= .001 on each of the four sensitive endpoints,
both weightings), re-adjusted for the new search family (m = 2480; the old m = 3900 adjustment was
not reused): **0 of 124 searched configurations pass against J, 0 against `leace_A0`, 0 against
both.** 55 of 124 have a residence point estimate better than J under both weightings, and the
closest are all A0-channel releases whose sensitive upper bounds are an order of magnitude outside
the tolerance (e.g. `E_A0_C_k2`: residence -0.0094 / -0.0100, worst sensitive upper +0.039 / +0.040).
The previous study's negative verdict and its original criterion are unchanged; this reanalysis adds
that its evidence contains no utility-first candidate either.

## 3. Descriptive test-split frontier (exploratory, post-hoc, no claim)

Reported because the pilot is closed and the releases were audited; it was NOT used for any gate or
selection. Seed means, unweighted; residence is a loss difference versus J (negative = better
residence capability); recovery increments are over the cloud re-audit of J.

| unit | residence - J | A/SEX | AB/SEX | A/RAC1P | AB/RAC1P |
|---|---|---|---|---|---|
| `X_r2_C1_b030` | -0.0023 | +0.0011 | +0.0035 | -0.0001 | +0.0000 |
| `X_r2_L2_b030` | +0.0003 | +0.0030 | -0.0015 | +0.0003 | +0.0017 |
| `X_r2_L1_b030` | -0.0021 | -0.0030 | +0.0046 | -0.0008 | +0.0069 |
| `X_r2_C1_b010` | -0.0003 | +0.0089 | +0.0037 | +0.0032 | +0.0076 |
| `U_r2` | -0.0021 | +0.0270 | +0.0195 | +0.0351 | +0.0306 |
| `P_r2` | +0.0006 | +0.0225 | +0.0131 | +0.0147 | +0.0193 |
| `ref_leace_A0` | -0.0004 | +0.0013 | -0.0043 | +0.0095 | +0.0095 |
| `ref_A0` | -0.0102 | +0.0292 | +0.0214 | +0.0524 | +0.0469 |

The two strongly protected arms (`X_r2_C1_b030`, `X_r2_L2_b030`) sit within about .003 nats of J on
residence and within about .003 on the sensitive endpoints: at this beta the release is close to J in
every direction, which is what "the selection preferred nearly-untouched J" looks like. The
unprotected control shows what the channel can carry: residence -0.0021 with +0.027 to +0.035 of
additional sensitive recovery. That contrast is the study's clearest finding: **on this channel the
extra residence-relevant capability and the extra sensitive recovery were not separable by the
adversarial term at any beta tried.**

## 4. Attack strength

Eight prespecified attacker recipes were calibrated on the 2018 fitting/validation pools against H,
untouched A0, J and LEACE-on-A0. **None was stronger than the standard slate** on the frozen rule
(0 of 8 improved >= 3 of 4 A0 endpoints; best mean A0 validation loss 0.9417 versus the standard
slate's selected candidates). The standard slate was retained and **added stress strength is NOT
established** - the same outcome as the predecessor study, reached by a different instrument. The
positive control passed: the standard slate recovers A0's known sensitive signal (+0.032 A/SEX,
+0.051 A/RAC1P over H, seed means, both weightings).

## 5. Limitations

* Development on repeatedly used 2018 pools; no confirmation is claimed and none is possible here.
* Screening thresholds are point estimates, not confidence statements; no adjusted intervals were
  computed for the pilot because no configuration was nominated (Tier 4 was not reached).
* The capability proxy is teacher reconstruction, not residence. A better reconstruction is not
  automatically better transfer, and the descriptive residence numbers above are not a substitute.
* Appending R cannot reduce optimal utility or optimal sensitive recovery (METHOD section 1); the
  measured increments are finite-learner quantities, and a near-zero increment would not show that
  no information remains.
* r = 2 only, one architecture, one budget, three anchors. r in {4, 8} was planned for Tier 3 and was
  not run: the pilot gate failed and the protocol forbids expanding the grid after that.
* 2016 remains sealed and unused.

## 6. What would be worth doing next (not done here)

The frontier is monotone in beta and the gap at beta = 30 is small (+0.0016 against a .001 screen)
while capability is nearly exhausted (0.111). A larger beta, or a narrower R, would be an
outcome-driven change to this design and is therefore a **separate prospective study**, not an
extension of this one. The honest reading of this pilot is that the declared operating point was not
reached by this family, not that it is one grid point away.
