# D first-pair development result: seed 0

**Status: technically replayed, point-utility ineligible.** R04's independent receipt is `TECHNICAL_REPLAY_PASS` for the first predeclared D-K versus task-only T pair. This is one model seed on 2,235 reused INNER_SELECTION people, with fresh readers using the 4,064-person ATTACK_FIT_NEW role. It is not the complete two-seed control comparison, a simultaneous confidence result or confirmation.

## Actual release and replay scope

Each recipient received its ten-bit hard public code, code-only calibrated probabilities and decision, and fixed public metadata. The code permits at most 1,024 states per recipient. On INNER, D-K occupied 14 income and 166 occupation code states; T occupied 11 and 168. Occupied states are observed use, not a reduction of the declared code capacity.

R04 independently reproduced raw-input-to-wire probabilities with maximum difference **0**, reconstituted every recipient and aligned-pair complete view exactly on ATTACK_FIT_NEW and INNER, checked the FIT-only scaler on 15,434 rows, restored fixed reader predictions, and verified null, shuffled-label, planted-leak and ignore-recipient controls. The receipt verifies this technical scope; R18's separate deployment acceptance and the remaining D controls are outside this first-pair verdict.

## Utility against the locked references

Caps are accuracy drop <= **0.010**, NLL excess <= **0.010**, and summed Brier excess <= **0.005**, on **each** task against **both** U0 and Ucal. Numbers below are point costs; negative costs favor the release.

| Arm | Task | Reference | Accuracy drop | NLL excess | Brier excess | Point result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| T | Income | U0 | -0.002237 | -0.006829 | -0.000016 | Pass |
| T | Income | Ucal | -0.002237 | -0.004469 | -0.000826 | Pass |
| T | Occupation | U0 | 0.000895 | **0.025228** | **0.010329** | Miss NLL/Brier |
| T | Occupation | Ucal | 0.000895 | **0.026591** | **0.009682** | Miss NLL/Brier |
| D-K | Income | U0 | 0.002237 | 0.003916 | **0.006970** | Miss Brier |
| D-K | Income | Ucal | 0.002237 | 0.006276 | **0.006160** | Miss Brier |
| D-K | Occupation | U0 | **0.016555** | **0.031445** | **0.013906** | Miss all three |
| D-K | Occupation | Ucal | **0.016555** | **0.032808** | **0.013259** | Miss all three |

Both arms predict above their task-majority accuracies. D-K accuracy is 0.849664 for income and 0.474273 for occupation, gains of 0.088143 and 0.191051 over the majority class; T accuracy is 0.854139 and 0.489933, gains of 0.092617 and 0.206711. This establishes nonconstant task use, while the locked accuracy/confidence conjunction fails at the point estimate. No simultaneous bounds are in this receipt, so these are **point shortfalls**, not interval-established cost violations.

## Complete-view SEX recovery

| Arm | Recipient 1 AUC | Recipient 2 AUC | Aligned pair AUC |
| --- | ---: | ---: | ---: |
| T, task-only compressor | 0.710718 | 0.795267 | 0.832582 |
| D-K, kernel repair | 0.671178 | 0.753706 | 0.805905 |
| T minus D-K | 0.039540 | 0.041562 | **0.026676** |

The pair difference clears the **0.020 point target by 0.006676**, and neither recipient shows a local AUC increase in this seed. It is a descriptive matched structural contrast: **both releases fail the required occupation NLL/Brier point caps, and D-K also misses income Brier and occupation accuracy**. T is not a utility-eligible competitive control in this seed, and D-K itself is not an eligible method release. No adjusted interval, second paired seed, full local/sequential/adversarial control slate or nominated result is present here. Lower measured recovery cannot earn a competitive or kernel-objective advantage at this utility cost.

D-K has worse task metrics than T in both tasks: income accuracy -0.004474, NLL +0.010745 and Brier +0.006986; occupation accuracy -0.015660, NLL +0.006217 and Brier +0.003577. The favorable recovery contrast and adverse utility contrast must be reported together. Joint-design attribution remains untested until matched local and both freeze-first sequential designs are completed; randomness attribution does not apply to this deterministic hard-code release.

## Evidence pins and limits

- R00's campaign-level interpretation is `results/pcrl_latent_and_sampled_methods_v1/D_FIRST_COMPARISON.md`; this R19 note expands its point clauses and component attribution without changing that diagnostic status. R00 also records the failed manifest-path attempt and dated engineering amendment.
- R04: `results/pcrl_latent_and_sampled_methods_v1/r04/FIRST_PAIR_INDEPENDENT_REPLAY.json`, SHA-256 `6ebf1a518ce23ef32835c3f39cdac4c5663e56a14df51f4c8b38866a9e68bfad`, schema `r04-first-real-d-pair-replay-v1`, status `TECHNICAL_REPLAY_PASS`.
- R00 first-seed fit/release receipt: `results/pcrl_latent_and_sampled_methods_v1/D_FIRST_S0_R1_RECEIPT.json`, seed 0 and fit unit `D_FIRST_T_DK_S0_R1`; package manifest hashes D-K `8b6fb67d75d90b111a86e07f71f413e047ce71b23b412a29db66aa7094360e91`, T `eb9e27ce6f1dc1bb139c3766479d4f3dd42eb9e137ed2a2632da617c6c7f4094`.
- Effective lock SHA-256 `a75db45770d2ecb1385cf0bbbcbfec86a217b244ebf898ce61acd569609f5a48`; saved endpoint SHA-256 `caee0f3383ac1b2a2ebdc730511cd14622d689c2b1c3a1b7b742a4a79a63113d`.

The old assessment was not opened for this comparison. Ucal's calibration family had earlier INNER exposure, and this campaign's INNER outcomes remain exploratory development evidence. R19 does not edit R00's shared decision or lock.
