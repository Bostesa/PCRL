# VALIDATION — ten checks aimed at material risks

Not an exhaustive test suite. Each check targets a specific way this study could
silently be wrong.

| check | result |
|---|---|
| gradient signs: stronger recovery raises the penalty, the mapper step reduces it | PASS |
| nested baseline: the zero correction reproduces `p0_j` exactly | PASS |
| label exclusion: residence and commute unreachable from the representation label path | PASS |
| household boundaries: internal folds partition `representation_fit`, no household straddles | PASS |
| release parity: `H_A` and `H_B` bitwise preserved in every wire | PASS |
| map serialisation: a reloaded channel reproduces its release bitwise | PASS |
| resumed-unit identity: recorded hashes still match their files | PASS |
| role masks: the audited forbidden registry is the full historical eleven | PASS |
| end-to-end: a bitwise-identical released channel reproduces its comparator's endpoints exactly under the matched-exposure scope | not applicable |
| score aggregation: stored endpoints recomputable from stored predictions | not applicable |
| simultaneous construction: the candidate-wide family covers every searched contrast and dominates | not applicable |

## The sign fixture, in numbers

Zero-correction gain **+0.000000** (exactly the service-only predictor, by construction); trained attacker gain **+0.6931**; after 50 mapper steps under `U + 1.0 * penalty` the penalty falls to **+0.0000**. The two players are not reversed and the service baseline is not being optimised in place of the channel.

## Backing data

* `VALIDATION.json`

