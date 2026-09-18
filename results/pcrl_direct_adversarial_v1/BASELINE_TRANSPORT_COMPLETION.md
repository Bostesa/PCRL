# BASELINE_TRANSPORT_COMPLETION — the 15 missing 2017 interfaces

The `leace_A0`, `splince_A0` and `optnet16_{L1,L2,C1}` adaptations of commit
`73903b7f28df68284285f0610a4036beb32b208f` were never evaluated on 2017, so the
previous study's strongest finding — that a closed-form 2023 linear eraser matches
`J` and beats four studies of developed mechanism — had been seen on **one year
only**. This completes those **15 interfaces** (5 arms x 3 seeds).

**These are EXPLORATORY CROSS-YEAR DEVELOPMENT numbers.** The 2017
`final_evaluation` partition is spent. The original frozen 2017 transport result
keeps its historical status and is neither restated nor overwritten. 2017 is not a
new test merely because these methods had not been scored there.

## Transport rule

The **2018-fitted transformation is reused** on 2017. No new 2017 eraser or
encoder is fitted and then called transported.

* `leace_A0` / `splince_A0`: the frozen `A0` inference pipeline is run on the 2017
  features through the transport study's own `FrozenSeed.interface`, and the
  **saved affine map is applied to that auxiliary channel alone**. `H_A` and `H_B`
  are appended unchanged.
* `optnet16_*`: the saved encoders act on the frozen whitened features, exactly as
  on 2018.

## Identity proofs

Neither the erasure affine maps nor the OptNet encoders were persisted by the
invariant study's production run. Both stages are deterministic, so both are
**reconstructed** — and neither is permitted to inherit the old identity until it
is proved. The proof used is the strongest available: the reconstruction rebuilds
the **2018** release and must match the stored 2018 `releases.npz` **bitwise on**
**all seven pools**. A mismatch would be recorded as a new fit under a new name.

| seed | arm | status | max abs difference vs stored 2018 | identity proved |
|---|---|---|---|---|
| 0 | `leace_A0` | IDENTITY PROVED | 0.000e+00 | yes |
| 0 | `optnet16_C1` | IDENTITY PROVED | 0.000e+00 | yes |
| 0 | `optnet16_L1` | IDENTITY PROVED | 0.000e+00 | yes |
| 0 | `optnet16_L2` | IDENTITY PROVED | 0.000e+00 | yes |
| 0 | `splince_A0` | IDENTITY PROVED | 0.000e+00 | yes |
| 1 | `leace_A0` | IDENTITY PROVED | 0.000e+00 | yes |
| 1 | `optnet16_C1` | IDENTITY PROVED | 0.000e+00 | yes |
| 1 | `optnet16_L1` | IDENTITY PROVED | 0.000e+00 | yes |
| 1 | `optnet16_L2` | IDENTITY PROVED | 0.000e+00 | yes |
| 1 | `splince_A0` | IDENTITY PROVED | 0.000e+00 | yes |
| 2 | `leace_A0` | IDENTITY PROVED | 0.000e+00 | yes |
| 2 | `optnet16_C1` | IDENTITY PROVED | 0.000e+00 | yes |
| 2 | `optnet16_L1` | IDENTITY PROVED | 0.000e+00 | yes |
| 2 | `optnet16_L2` | IDENTITY PROVED | 0.000e+00 | yes |
| 2 | `splince_A0` | IDENTITY PROVED | 0.000e+00 | yes |

