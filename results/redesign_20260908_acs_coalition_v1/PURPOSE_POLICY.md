# Declared purposes, forbidden targets and decisions

This study tests a stipulated ACS benchmark policy. It does not assert that the two task families identify real institutions or that a recipient must receive features instead of a prediction service. The [protocol](PROTOCOL.md) fixes the policy before any new fitting.

| Access | Authorized source tasks | Reserved authorized task | Forbidden targets | Forbidden targets used during representation training |
| --- | --- | --- | --- | --- |
| A | Income >50k; civilian employed and at work | Same residence | Public coverage; commute >20 minutes; SEX; RAC1P | Public coverage; SEX; RAC1P |
| B | Public coverage | Commute >20 minutes | Income >50k; civilian employed and at work; same residence; SEX; RAC1P | Income >50k; civilian at work; SEX; RAC1P |
| AB, both outputs for the same person | Union of the three source tasks | Residence and commute | SEX; RAC1P | SEX; RAC1P |

The five task identities, eligibility masks and household pools are unchanged. Residence and commute remain reserved labels: they enter independent utility heads and opposing-purpose attackers only after all six final paired systems for that seed are frozen. There are nine training-observer roles and eleven post-freeze audit roles. There is no reserved-task observer during representation fitting.

Permitting AB to perform the union of authorized tasks avoids demanding that the coalition forget a task that one recipient is allowed to perform. It does not make suppressing SEX/RAC1P free: authorized tasks can correlate with prohibited attributes. A policy that authorizes exact U and V while forbidding exact recovery of their deterministic product S=UV is contradictory. The earlier [exact calculation](../redesign_20260908_acs_restricted_inputs_v1/PURPOSE_COORDINATION_EXACT.md) illustrates that distinction; it is not a certificate for these ACS labels.

## Explicit amendment to the prior proposal

The earlier [four-condition design](../redesign_20260908_acs_restricted_inputs_v1/PURPOSE_COORDINATION_DESIGN.md) remains historical. The current six-condition protocol makes three changes before observing new outcomes:

1. Add Iplus to distinguish a coalition objective from stronger local sensitive-attribute pressure. Iplus preserves the ordinary opposing-source penalties and adds sensitive-only local terms; it does not indiscriminately double all individual penalties.
2. Include existing singleton and public-head attacks in the coalition's candidate pools. A coalition can ignore one output, and an F recipient can compute the fixed public source heads from its feature coordinates. Omitting those attacks would artificially weaken the modeled recipient.
3. Decide residence and commute feature capability separately. Prior ACS evidence supplies residence headroom but no comparable commute advantage. Failure on one task cannot erase a measured result on the other, and residence cannot stand in for commute.

These amendments do not change the sampled population, reserved-label boundary, parent reference, support rules or existing descriptive margins.

## Six fixed conditions and what each contrast means

All conditions receive the original PCA32 input, use the same assigned source labels, have 6,355 total forward-model parameters and start from matched raw-PCA16 mapper/source-head tensors. F exposes 16 feature coordinates per recipient. P exposes exactly two A probabilities and one B probability. Equal fitting capacity and input/label access do not mean equal transmitted information; that width difference is part of the interface comparison.

Let `L_ind=(L_A+L_B)/2`, where A averages its three training-observer normalized losses and B averages its four. Let `M_A` and `M_B` each average the local SEX/RAC1P normalized losses, and `M_AB` average the two coalition losses. All target entropies come from the same fixed representation-fitting priors.

| Regime, within either F or P | Forward objective |
| --- | --- |
| I | `L_source − .1 L_ind` |
| Iplus | `L_source − .1 L_ind − .1 (M_A + M_B)` |
| J | `L_source − .1 L_ind − .1 M_AB` |

`M_A+M_B` is a **sum**. The added nominal coefficient is .05 per sensitive target and affected branch in both Iplus and J. This is coefficient matching, not equality of gradient norms, losses, observer information or optimization difficulty. The ordinary opposing-source coefficients remain unchanged. The source objective gives A and B equal purpose weight, rather than taking a three-task mean.

J−I asks whether adding the declared coalition objective helps relative to ordinary local protection. J−Iplus asks whether the result survives the targeted stronger-local alternative. Iplus−I measures that alternative directly. A favorable J−I result alone does not establish that joint observations were necessary. F−P within each regime compares the actual feature and probability interfaces; coordinated F must be compared with coordinated P.

## Separate research decisions

The reporting matrix keeps the following conclusions distinct:

- Coalition SEX and RAC1P recovery under J versus I, and separately J versus Iplus, alongside authorized utility and individual forbidden-target recovery.
- A's residence capability for F-J versus P-J and direct E, with the predeclared .01-nat feature-loss advantage reference.
- B's commute capability under the same comparisons and reference, without borrowing residence's result or inventing a commute-retention denominator.
- Original-PCA32 source retention, residential retention, attribute-gain references, stronger-audit changes and category-support limitations.

Original PCA32 remains the parent for each source loss allowance of .01 nats, half of positive residential headroom over the better unprotected rich bank, and halving positive parent attribute gain where defined. The feature-control comparison uses .01-nat utility advantage and .005-nat sensitive-gain excess as descriptive references. Report every failure, undefined denominator and signed gain. There is no inherited numerical privacy threshold for an opposing source or reserved task; report those losses/gains and their support without inventing a pass criterion.

Unweighted validation chooses predictions once; PWGTP scores those same predictions beside unweighted results. The three seeds share a cohort, so their SDs are descriptive. Original test households remain **DEVELOPMENT EVALUATION**. All nine RAC1P categories remain visible, including absent fitting/validation support. No finite numerical comparison or unsuccessful attacker supplies a full race, all-target, population or privacy certificate.
