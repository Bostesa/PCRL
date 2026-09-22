# Research decision

No adjusted competitive pass is established among the supplied frozen routes. Evaluated failures and blocked comparisons are separated below. This is not an impossibility result for other encoders, distributions or privacy–utility trade-offs.

## Frozen adjusted claims

| Group | Route | Claim | Status | EVIDENCE.json pointer |
| --- | --- | --- | --- | --- |
| main | protection_first | competitive | did_not_pass | /selected_inference/main/protection_first/competitive |
| main | protection_first | historical_J | did_not_pass | /selected_inference/main/protection_first/historical_J |
| main | utility_first | competitive | did_not_pass | /selected_inference/main/utility_first/competitive |
| main | utility_first | historical_J | did_not_pass | /selected_inference/main/utility_first/historical_J |
| attribution | protection_first | coalition | did_not_pass | /selected_inference/attribution/protection_first/coalition |
| attribution | protection_first | randomization | did_not_pass | /selected_inference/attribution/protection_first/randomization |
| attribution | protection_first | task_refinement | did_not_pass | /selected_inference/attribution/protection_first/task_refinement |
| attribution | utility_first | randomization | did_not_pass | /selected_inference/attribution/utility_first/randomization |
| diagnostic | protection_first | competitive_all_families | blocked | /selected_inference/diagnostic/protection_first/competitive_all_families |
| diagnostic | protection_first | randomization_all_controls | blocked | /selected_inference/diagnostic/protection_first/randomization_all_controls |
| diagnostic | utility_first | competitive_all_families | blocked | /selected_inference/diagnostic/utility_first/competitive_all_families |
| diagnostic | utility_first | randomization_all_controls | did_not_pass | /selected_inference/diagnostic/utility_first/randomization_all_controls |

Main competitive, historical-J, attribution and stricter diagnostic formulas are separate. The adjusted family is not enlarged by descriptive full-grid tables. The generator checks formula-tree consistency and endpoint presence; accepted upstream claim evaluation must recompute the registered bound checks.

## Selected residence and disclosure results

| Configuration | Weight | Fixed CE | Independent CE | Selected CE | CE − H | CE − J |
| --- | --- | --- | --- | --- | --- | --- |
| T0_L_0.01_a17 | unweighted | 0.482987108 | 0.483683397 | 0.483197663 | -0.0419539949 | -0.0162508904 |
| T0_L_0.01_a17 | PWGTP | 0.459490345 | 0.461252968 | 0.460460631 | -0.0386084644 | -0.0163935697 |
| Ttask_C_0.01_a17 | unweighted | 0.488928559 | 0.489180088 | 0.489180088 | -0.03597157 | -0.0102684655 |
| Ttask_C_0.01_a17 | PWGTP | 0.464888483 | 0.466222528 | 0.466222528 | -0.0328465665 | -0.0106316718 |

CE is in nats; lower is better for prediction. CE − H and CE − J are signed loss differences: negative is improved utility.

| Configuration | Role | Weight | Selected CE | Recovery over H | Recovery increment over J |
| --- | --- | --- | --- | --- | --- |
| T0_L_0.01_a17 | A/RAC1P | unweighted | 1.26988107 | 0.000858287196 | -0.00566725989 |
| T0_L_0.01_a17 | A/RAC1P | PWGTP | 1.26327207 | 0.00146757257 | -0.00481770237 |
| T0_L_0.01_a17 | A/SEX | unweighted | 0.681757991 | 0.00613566926 | -0.00961180724 |
| T0_L_0.01_a17 | A/SEX | PWGTP | 0.680708758 | 0.00577569463 | -0.00246055858 |
| T0_L_0.01_a17 | AB/RAC1P | unweighted | 1.25603669 | 0.00156831171 | -0.00393608256 |
| T0_L_0.01_a17 | AB/RAC1P | PWGTP | 1.25055464 | 0.00179759324 | -0.0046264099 |
| T0_L_0.01_a17 | AB/SEX | unweighted | 0.679798724 | 0.00541821079 | -0.0110296428 |
| T0_L_0.01_a17 | AB/SEX | PWGTP | 0.679972968 | 0.00438815785 | -0.00619987928 |
| Ttask_C_0.01_a17 | A/RAC1P | unweighted | 1.27146213 | -0.000722771413 | -0.0072483185 |
| Ttask_C_0.01_a17 | A/RAC1P | PWGTP | 1.2638192 | 0.000920441337 | -0.00536483361 |
| Ttask_C_0.01_a17 | A/SEX | unweighted | 0.682579203 | 0.00531445743 | -0.0104330191 |
| Ttask_C_0.01_a17 | A/SEX | PWGTP | 0.683783153 | 0.00270129902 | -0.00553495419 |
| Ttask_C_0.01_a17 | AB/RAC1P | unweighted | 1.25600592 | 0.0015990908 | -0.00390530348 |
| Ttask_C_0.01_a17 | AB/RAC1P | PWGTP | 1.25070754 | 0.0016446874 | -0.00477931573 |
| Ttask_C_0.01_a17 | AB/SEX | unweighted | 0.67912892 | 0.00608801469 | -0.0103598389 |
| Ttask_C_0.01_a17 | AB/SEX | PWGTP | 0.681331489 | 0.00302963697 | -0.00755840016 |

Recovery over H is CE(H) − CE(candidate), so positive means additional measured disclosure. Recovery increment over J is CE(J) − CE(candidate), so positive means more recovery than J. These are full-H predictive comparisons, not modeled CMI. Means require all three anchors; missing configurations are not silently averaged.

## Descriptive range and subjective bets

| Descriptive extremum | Value | Configurations |
| --- | --- | --- |
| Lowest balanced selected CE − J | -0.020134446 | continuous_task |
| Highest balanced selected CE − J | 0.0239589996 | H, constant_best, constant_best_a33 |

These extrema are drawn only from complete supplied configurations, use equal U/PWGTP weighting, and carry no new significance or selection claim. Full-grid point estimates remain descriptive.

| Bet | Prospective probability | Assessment | Complete pertinent coverage | Source pointer |
| --- | --- | --- | --- | --- |
| P1 | 0.85 | supported | True | /predictions/P1 |
| P2 | 0.6 | refuted | True | /predictions/P2 |
| P3 | 0.6 | supported | True | /predictions/P3 |
| P4 | 0.4 | refuted | True | /predictions/P4 |
| P5 | 0.65 | refuted | True | /predictions/P5 |
| P6 | 0.25 | refuted | True | /predictions/P6 |
| P7 | 0.7 | supported | True | /predictions/P7 |

P1–P7 preserve their prospective probabilities and machine assessments. A supported existential bet may have incomplete coverage; refutation requires complete pertinent attempts. Unassessed is not refuted. These bets are not scientific gates.

## Interpretation and handoff

This is a supervised residence release study. The teacher and finite-channel costs use residence labels; the risk refinement additionally fits protected-label predictors. J is an inherited representation without that residence supervision. A gain over J alone therefore does not isolate a channel-design benefit. Current label-matched controls, including the original teacher-pool and the mechanism40/union88 supervised LEACE and SPLINCE supplements, determine the registered competitive comparison. The union pool describes available fitting rows, not a proved superset of every method's actual protected-label observations.

Ttask and Trisk both refine the 32-cell T0 code and use at most 64 input cells. Coarse row copying proves family inclusion under the same action dictionary, empirical laws, costs and budgets; it does not establish a held-out improvement. Equal-size task/risk refinements are not nested. Attribute a risk-specific effect only to the separate registered comparison against both T0 and Ttask with its recovery caps. The learned teacher and quantized code are not asserted to be exact sufficient statistics. The exact theoretical fixture and empirical teacher/quantization effects answer different questions.

The modeled quantities are CMI under fitted finite conditional laws, with A and AB and both unweighted and PWGTP measures retained. These constraints are not guarantees conditional on the recipient's full H. Full-H attacker CE improvement over the same H is measured predictive recovery; it is neither a CMI estimate nor a CMI lower or upper bound for these restricted model slates. A zero fitted budget allows pre-existing disclosure through H. It is not differential privacy, worst-case protection, joint intersectional protection, or a population guarantee. Marginal SEX/RAC1P constraints remain separate.

One token Z is drawn from Q(.|T). The fixed decoder is sigmoid(logit(b(H_A)) + a_Z). Report expected cross-entropy across tokens, not the cross-entropy of a mixture prediction; publishing a Q row or a risk score would change the release. Fixed decoder, independent deployment learner and validation-selected deployment learner losses remain separate. The same selected predictions are scored under both weightings.

The three anchor pools are disjoint within each anchor but overlap heavily across anchors. The evaluation seal is anchor-specific: these are not globally unseen people or globally untouched labels. The late [DATED_SPLIT_CLARIFICATION.md](DATED_SPLIT_CLARIFICATION.md) records this existing design and does not create a retrospective independent holdout. Shared-household adjustment handles the registered overlap structure for conditional development diagnostics; model-fitting and selection dependencies remain. Fresh-year confirmation remains prospective and unexecuted here.

The executable contract and limitations are in [PROTOCOL.md](PROTOCOL.md), [METHOD.md](METHOD.md), [RELEASE_CONTRACT.md](RELEASE_CONTRACT.md), [INTERPRETATION_ADDENDUM_2026-09-22.md](INTERPRETATION_ADDENDUM_2026-09-22.md), and [DATED_SPLIT_CLARIFICATION.md](DATED_SPLIT_CLARIFICATION.md). [OWNERSHIP_CORRECTION.md](OWNERSHIP_CORRECTION.md) locates the predecessor's planned coverage-containing local conditioner; it does not claim that an offending ACS fit was executed.

[NOVELTY_AND_ASSUMPTIONS.md](NOVELTY_AND_ASSUMPTIONS.md) and [LITERATURE_SOURCES.json](LITERATURE_SOURCES.json) give primary-source section/theorem references for perfect privacy, privacy funnels, randomized preprocessing, LEACE and SPLINCE. No novelty is asserted for convexity, nullspaces, quantization, randomization, multi-view constraints or nested refinements. A task-posterior sufficiency separation can be useful without being a new general privacy-funnel theorem.

[HISTORICAL_SCORE_INDEX.json](HISTORICAL_SCORE_INDEX.json) and its [reading guide](HISTORICAL_SCORE_INDEX.md) preserve published H/J, LEACE-on-A0, SPLINCE-on-A0 and executed OptNet scores with exact source commits, paths and hashes. Their supervision, fitting/slate and development data use differ from the current-host matched audits. In particular, historical expanded/kernel-expanded audit scores are separate from the current common audit slate; they are not the same attacker benchmark. Those historical scores are a separate evidence stratum and are not substituted into current matched comparisons or current inference. The optional neural-adversarial baseline was not executed; no comparison with it is established.

The inherited work's service/release ownership framing, explicit recipient views, byte-preserving H baseline and reproducible empirical comparisons remain useful contributions to examine on their own terms. This extension does not silently replace the original manuscript or establish all of its claims. [REVIEW_INDEX.md](REVIEW_INDEX.md) is an internal evidence-navigation aid: actual venue reviews have not been adjudicated by this generator. Any final response to actual reviews must quote and answer those reviews faithfully and separately; current venue prior-review disclosure requirements must be satisfied in any later authorized submission. No manuscript edit, registration, paper submission or contact is performed by this reporting step.

[ORIGINAL_WORK_HANDOFF.md](ORIGINAL_WORK_HANDOFF.md) locates the original dominant-axis/rare-class-sensitive audit, correctly scoped composition correction and historical rebuttal measurements at the full pinned review commit, with exact paths, lines and hashes. Preserve their demonstrated observations and the coverage of any actual verification; a source index does not newly verify historical outcomes. Do not revive the invalid R²-to-classification-accuracy guarantee or describe shared-union erasure as retaining conflicting purposes. The pinned Gaussian comparison distinguishes successful per-purpose retention from destruction by the shared union. Historical rebuttal measurements also do not prove an architecture-wide impossibility theorem.
