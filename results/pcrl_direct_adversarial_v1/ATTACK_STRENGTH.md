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

## Backing data

* `MECH_TRAINING_VS_AUDIT.csv`
* `MECH_REFRESH.csv`
* `MECH_CATCHUP.csv`
* `MECHANISM.json`

