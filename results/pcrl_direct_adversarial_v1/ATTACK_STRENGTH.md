# ATTACK_STRENGTH — what the attacks actually found

A low measured recovery means the **declared finite attack family** did not find
one inside its budget. It is not independence, not differential privacy, not a
bound on `I(S;Z|H)` and not protection against arbitrary attackers.

## Training attacker versus fresh auditor

Two different quantities on two different pools: the training gain is measured on
the internal `monitor` fold against a 300-update differentiable probe; the audit
recovery is measured on the test pool against the full independent slate, which
includes boosted trees and kernel ridge that were **never in the training
gradient at all**. A large gap means the mapper was fooling a family the auditor
does not belong to.

Rows: 900. Mean training fresh-probe gain **+0.0196**; mean audit additional recovery over `H` **+0.0233**. Association across arms: Pearson 0.5772334769861324, Spearman 0.43591620895005223 (n = 900).

_An observational association across fitted arms. It does not identify a
mechanism and no causal reading is offered._

## Attacker refresh

1350 refresh decisions; the refreshed attacker was kept in **84** of them. Mean monitor cross-entropy recovery -0.01304, maximum +0.01012.

A positive recovery means the incumbent had been **fooled by the moving
channel** rather than genuinely defeated — which is exactly the failure mode
the refresh exists to detect.

## Extended catch-up: budget 120 against budget 360

How much more the longer attack recovers on the same scope. A large value
means the shorter budget understated what was reachable.

| condition | mean catch-up gain | max | rows |
|---|---|---|---|
| `leace_A0` | +0.0003 | +0.0030 | 24 |
| `leace_dax16_none` | +0.0003 | +0.0030 | 24 |
| `L025` | +0.0001 | +0.0018 | 24 |
| `L20` | +0.0001 | +0.0027 | 24 |
| `dax8_L1_b100` | +0.0001 | +0.0027 | 24 |
| `dax8_L1_b300` | +0.0001 | +0.0027 | 24 |
| `dax8_L2_b100` | +0.0001 | +0.0027 | 24 |
| `dax8_L2_b300` | +0.0001 | +0.0027 | 24 |
| `dax8_C1_b100` | +0.0001 | +0.0027 | 24 |
| `dax8_C1_b300` | +0.0001 | +0.0027 | 24 |
| `splince_A0` | +0.0001 | +0.0027 | 24 |
| `splince_dax16_none` | +0.0001 | +0.0027 | 24 |
| `leace_dax8_none` | +0.0001 | +0.0027 | 24 |
| `E` | +0.0000 | +0.0000 | 24 |
| `A0` | +0.0000 | +0.0000 | 24 |
| `spectral_S0` | +0.0000 | +0.0000 | 24 |
| `spectral_M025` | +0.0000 | +0.0000 | 24 |
| `spectral_M1` | +0.0000 | +0.0000 | 24 |
| `spectral_L025` | +0.0000 | +0.0000 | 24 |
| `spectral_L1` | +0.0000 | +0.0000 | 24 |
| `spectral_C025` | +0.0000 | +0.0000 | 24 |
| `spectral_C1` | +0.0000 | +0.0000 | 24 |
| `spectral_L2` | +0.0000 | +0.0000 | 24 |
| `optnet16_L1` | +0.0000 | +0.0000 | 24 |
| `optnet16_L2` | +0.0000 | +0.0000 | 24 |

_Full table in `MECH_CATCHUP.csv`._

## Stress attack suite — two fresh initialisations and a 720-epoch continuation

Prespecified in `PROTOCOL.md` §9 before any outcome, on the four family sensitive
roles. The set contains **both sides**: the competitors `J` and `leace_A0` and this
study's own width-16 `C1` and `L2` arms at `beta` 0.3 and 1.0. Strengthening the
attack on a competitor alone would be the obvious way to manufacture a win, and is
not done. Selection is on attacker validation only, written to disk before the test
pool is read.

`validation improvement` is the default suite's selected validation log loss minus
the stress suite's. Positive means the stress suite found a **stronger** attack.

| condition | role-cells | mean validation improvement | max | 720-epoch chosen |
|---|---|---|---|---|
| `J` | 12 | -0.00770 | +0.00231 | 0/12 |
| `dax16_C1_b030` | 12 | +0.00028 | +0.00555 | 0/12 |
| `dax16_C1_b100` | 12 | +0.00074 | +0.00634 | 0/12 |
| `dax16_L2_b030` | 12 | +0.00028 | +0.00555 | 0/12 |
| `dax16_L2_b100` | 12 | +0.00208 | +0.00969 | 0/12 |
| `leace_A0` | 12 | -0.00491 | +0.00329 | 0/12 |

A 720-epoch trajectory winning selection would mean the default 360-epoch budget
had been understating what is recoverable. Where the 360-epoch checkpoint still
wins, the extra budget bought nothing and the longer trajectory overfits its own
validation pool.

## Backing data

* `MECH_TRAINING_VS_AUDIT.csv`
* `MECH_REFRESH.csv`
* `MECH_CATCHUP.csv`
* `MECHANISM.json`
* `STRESS_SUMMARY.json`
* `STRESS_COMPARISON.csv`

