# DEVELOPMENT_2018 — the primary comparison

Split `test`, scope `kernel_expanded_catchup`, budget 360,
seeds 0/1/2. **These are development numbers**: the 2018 pools are the original
development resource of the residual spectral study and have been used repeatedly.

## Main table (seed means, unweighted)

| condition | residence gain over H | additional A/SEX | additional A/RAC1P | additional AB/SEX | additional AB/RAC1P |
|---|---|---|---|---|---|
| `spectral_lin16_C1` | 0.0265 | 0.0135 | 0.0274 | 0.0098 | 0.0231 |
| `spectral_lin16_L1` | 0.0288 | 0.0150 | 0.0381 | 0.0149 | 0.0374 |
| `spectral_lin16_L2` | 0.0285 | 0.0141 | 0.0328 | 0.0101 | 0.0237 |
| `spectral_nlr16_C1` | 0.0277 | 0.0071 | 0.0202 | 0.0099 | 0.0115 |
| `spectral_nlr16_L1` | 0.0288 | 0.0135 | 0.0389 | 0.0135 | 0.0340 |
| `spectral_nlr16_L2` | 0.0251 | 0.0082 | 0.0157 | 0.0059 | 0.0141 |
| `spectral_lin8_C1` | 0.0261 | 0.0125 | 0.0058 | 0.0091 | 0.0082 |
| `spectral_lin8_L1` | 0.0270 | 0.0131 | 0.0209 | 0.0100 | 0.0193 |
| `spectral_lin8_L2` | 0.0255 | 0.0121 | 0.0089 | 0.0015 | 0.0101 |
| `spectral_nlr8_C1` | 0.0259 | 0.0059 | 0.0055 | 0.0036 | 0.0040 |
| `spectral_nlr8_L1` | 0.0260 | 0.0114 | 0.0295 | 0.0106 | 0.0227 |
| `spectral_nlr8_L2` | 0.0253 | 0.0036 | 0.0109 | 0.0079 | 0.0127 |
| `J` | 0.0215 | -0.0047 | 0.0071 | 0.0011 | 0.0057 |
| `H` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `E` | 0.0253 | 0.0145 | 0.0593 | 0.0128 | 0.0488 |
| `A0` | 0.0318 | 0.0261 | 0.0614 | 0.0214 | 0.0436 |
| `spectral_S0` | 0.0319 | 0.0279 | 0.0655 | 0.0237 | 0.0500 |

Additional recovery is the increment over `H`'s own selected attack, matched on
seed, scope, budget, view, target and weighting. Absolute recovery and `H`'s own
absolute recovery are both in `PER_SEED.csv`; they are not replaced by the
increment. Signed increments are preserved: a negative value is a measurement, not
negative information, and it does not erase what an `H`-only attack can reach.

## Reused criteria (labelled as reused, not re-justified)

| condition | weight | residence gain mean | min | .01 reference | half-headroom (pass/fail/undef) | source allowance (pass/fail/undef) |
|---|---|---|---|---|---|---|
| `A0` | person_weighted | 0.0283 | 0.0249 | pass | 0/3/0 | 3/0/0 |
| `A0` | unweighted | 0.0318 | 0.0277 | pass | 2/1/0 | 3/0/0 |
| `E` | person_weighted | 0.0233 | 0.0196 | pass | 0/3/0 | 3/0/0 |
| `E` | unweighted | 0.0253 | 0.0188 | pass | 0/3/0 | 3/0/0 |
| `H` | person_weighted | 0.0000 | 0.0000 | fail | 0/3/0 | 3/0/0 |
| `H` | unweighted | 0.0000 | 0.0000 | fail | 0/3/0 | 3/0/0 |
| `J` | person_weighted | 0.0195 | 0.0165 | pass | 0/3/0 | 3/0/0 |
| `J` | unweighted | 0.0215 | 0.0175 | pass | 0/3/0 | 3/0/0 |
| `spectral_S0` | person_weighted | 0.0284 | 0.0259 | pass | 0/3/0 | 1/2/0 |
| `spectral_S0` | unweighted | 0.0319 | 0.0264 | pass | 3/0/0 | 0/3/0 |
| `spectral_lin16_C1` | person_weighted | 0.0242 | 0.0209 | pass | 0/3/0 | 1/2/0 |
| `spectral_lin16_C1` | unweighted | 0.0265 | 0.0221 | pass | 0/3/0 | 2/1/0 |
| `spectral_lin16_L1` | person_weighted | 0.0260 | 0.0228 | pass | 0/3/0 | 2/1/0 |
| `spectral_lin16_L1` | unweighted | 0.0288 | 0.0273 | pass | 1/2/0 | 2/1/0 |
| `spectral_lin16_L2` | person_weighted | 0.0260 | 0.0249 | pass | 0/3/0 | 1/2/0 |
| `spectral_lin16_L2` | unweighted | 0.0285 | 0.0235 | pass | 1/2/0 | 1/2/0 |
| `spectral_lin8_C1` | person_weighted | 0.0256 | 0.0221 | pass | 1/2/0 | 2/1/0 |
| `spectral_lin8_C1` | unweighted | 0.0261 | 0.0213 | pass | 0/3/0 | 2/1/0 |
| `spectral_lin8_L1` | person_weighted | 0.0248 | 0.0217 | pass | 0/3/0 | 2/1/0 |
| `spectral_lin8_L1` | unweighted | 0.0270 | 0.0223 | pass | 1/2/0 | 2/1/0 |
| `spectral_lin8_L2` | person_weighted | 0.0257 | 0.0244 | pass | 1/2/0 | 2/1/0 |
| `spectral_lin8_L2` | unweighted | 0.0255 | 0.0229 | pass | 0/3/0 | 2/1/0 |
| `spectral_nlr16_C1` | person_weighted | 0.0254 | 0.0227 | pass | 0/3/0 | 1/2/0 |
| `spectral_nlr16_C1` | unweighted | 0.0277 | 0.0227 | pass | 0/3/0 | 1/2/0 |
| `spectral_nlr16_L1` | person_weighted | 0.0256 | 0.0234 | pass | 0/3/0 | 1/2/0 |
| `spectral_nlr16_L1` | unweighted | 0.0288 | 0.0259 | pass | 2/1/0 | 2/1/0 |
| `spectral_nlr16_L2` | person_weighted | 0.0243 | 0.0202 | pass | 0/3/0 | 1/2/0 |
| `spectral_nlr16_L2` | unweighted | 0.0251 | 0.0209 | pass | 0/3/0 | 2/1/0 |
| `spectral_nlr8_C1` | person_weighted | 0.0238 | 0.0188 | pass | 0/3/0 | 1/2/0 |
| `spectral_nlr8_C1` | unweighted | 0.0259 | 0.0184 | pass | 1/2/0 | 1/2/0 |
| `spectral_nlr8_L1` | person_weighted | 0.0234 | 0.0189 | pass | 0/3/0 | 1/2/0 |
| `spectral_nlr8_L1` | unweighted | 0.0260 | 0.0208 | pass | 0/3/0 | 2/1/0 |
| `spectral_nlr8_L2` | person_weighted | 0.0228 | 0.0182 | pass | 0/3/0 | 2/1/0 |
| `spectral_nlr8_L2` | unweighted | 0.0253 | 0.0190 | pass | 1/2/0 | 3/0/0 |

The half-headroom and source-allowance criteria are **tri-state**: the
historical helpers return "undefined" when the headroom ratio or a source loss is
not defined on that seed. Undefined is reported as its own column rather than
collapsed into a failure.

## Registered decisions

* `coordination_original_rank16_C1_vs_local`: **NOT SUPPORTED** (advantage on all comparators: False; residence within .001: False; A/RAC1P significantly worse anywhere: False)
* `coordination_original_rank16_C1_vs_J`: **NOT SUPPORTED** (advantage on all comparators: False; residence within .001: True; A/RAC1P significantly worse anywhere: True)
* `coordination_original_rank8_C1_vs_local`: **NOT SUPPORTED** (advantage on all comparators: False; residence within .001: True; A/RAC1P significantly worse anywhere: False)
* `coordination_original_rank8_C1_vs_J`: **NOT SUPPORTED** (advantage on all comparators: False; residence within .001: True; A/RAC1P significantly worse anywhere: False)
* `coordination_nonlinear_rank16_C1_vs_local`: **NOT SUPPORTED** (advantage on all comparators: False; residence within .001: False; A/RAC1P significantly worse anywhere: False)
* `coordination_nonlinear_rank16_C1_vs_J`: **NOT SUPPORTED** (advantage on all comparators: False; residence within .001: True; A/RAC1P significantly worse anywhere: True)
* `coordination_nonlinear_rank8_C1_vs_local`: **NOT SUPPORTED** (advantage on all comparators: False; residence within .001: True; A/RAC1P significantly worse anywhere: False)
* `coordination_nonlinear_rank8_C1_vs_J`: **NOT SUPPORTED** (advantage on all comparators: False; residence within .001: True; A/RAC1P significantly worse anywhere: False)

The `.001` residence condition is a point-difference rule reused for
comparability. A pass under it is **not** an equivalence result, and the
simultaneous intervals do not establish equivalence within that band. Separate
noninferiority / equivalence interval outcomes are recorded per contrast in
`DEVELOPMENT_2018.json` under `decisions[...]["noninferiority"]`.

## Rotation decomposition (the surrogate test that needs no attacker)

| condition | rotation-only share of training gain (per seed) | mean |
|---|---|---|
| `spectral_nlr16_C1` | 0.625, 0.717, 0.657 | 0.666 |
| `spectral_nlr16_L1` | 0.907, 0.807, 0.793 | 0.836 |
| `spectral_nlr16_L2` | 0.645, 0.656, 0.582 | 0.628 |
| `spectral_nlr8_C1` | 0.514, 0.549, 0.495 | 0.519 |
| `spectral_nlr8_L1` | 0.590, 0.768, 0.689 | 0.682 |
| `spectral_nlr8_L2` | 0.367, 0.478, 0.457 | 0.434 |

Utility, the original linear penalty and the information content of the
release are all invariant under `W -> W Q` for orthogonal `Q`; the nonlinear
penalty is not. So any training-objective gain reachable by rotating within the
original subspace is surrogate movement that provably changes nothing an
attacker can recover from the release.

