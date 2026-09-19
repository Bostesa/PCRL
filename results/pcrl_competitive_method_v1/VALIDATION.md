# VALIDATION

Backing: `VALIDATION.json` (`ba529009c102b7d7d546cd89a9a598ecde97f7fd33f93771187828b1ccb92b8b`), regenerated `REGEN_FOR_2017.json`.

| check | result |
|---|---|
| mathematical fixtures | 16/16 pass (`tests/pcrl_competitive_method_v1`) |
| historical gamma=1 A0 units reproduced bit-exactly | 18/18 |
| Track N selection reconstructed from stored components | 168/168 |
| serialisation replay (every family, every anchor) | 33 replays, max abs diff 0.0e+00 |
| matched audit exposure (one audit-count profile) | 1 profile over 378 unit-anchors |
| independent replay scorer vs audit (4 sensitive + residence, both weightings) | 3780 checks, max abs diff 4.4e-16 |
| probability validation / quarantines | 0 quarantined attempts |
| **ref_A0 == historical A0 audit** | **FAILED**: max abs diff 4.52e-04 (1 cell) — cause established, amendment 4 |
| **ref_J == historical J audit** | **FAILED**: max abs diff 3.09e-03 (2 cells) — cause established, amendment 4 |
| regenerated 2018 audits for 2017 (incident 1) | 18/18 metrics identical and retained predictions bitwise equal |

The reference-identity failures are not numerical faults: the historical A0/J audits carry
`derived__*` candidates that no new release has (2/66 and 4/66 selected cells). Seed 0
matched exactly. `ref_J` is used as the exposure-matched J comparator; verdicts are
reported under both.
