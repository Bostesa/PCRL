# R19 evidence and claim template

Status: `TEMPLATE_ONLY`. R19 records evidence and claim limits; R00 owns the campaign protocol, locks, selection and final decision. This file does not establish a scientific result.

## Fixed interpretation rules

- This is exploratory development on repeatedly used Adult people. The old assessment remains embargoed; more agents, seeds or locks do not create confirmation.
- Keep the D feature-code service and A sampled-score service separate. Each reported recovery value must identify the recipient's actual transmitted view or the aligned pair.
- Report each paired model seed and the paired aggregate. The same people across seeds are one development population.
- A useful task result needs both tasks and both U0 and Ucal references: accuracy loss at most 0.010, NLL excess at most 0.010 nats, and summed Brier excess at most 0.005. Report point feasibility separately from simultaneous interval status.
- Competitive recovery needs coalition SEX AUC improvement at least 0.020 over every fixed eligible matched control, and no recipient AUC worsening above 0.010. Name the strongest eligible control and its capacity/interface/exposure match.
- For a positive clause, record its paired point difference, simultaneous interval, direction and boundary rule. Mark `PASS`, `UNRESOLVED`, `VIOLATION` or `UNMEASURED` clause by clause. A point advantage alone is not a passed statistical claim.
- Distinguish a method advantage from attribution to privacy training, joint design, kernel objective or stochasticity. The relevant ablation and both freeze-first sequential orders must support the narrower claim.
- A complete negative comparison is a result. An unfitted, invalid or missing-control arm is `INCOMPLETE` or `UNTESTED`, never a measured loss.

## Evidence packet required for each arm

| Field | Entry to fill from immutable evidence |
| --- | --- |
| Arm and family | Exact registry ID; D or A; objective/ablation; seed IDs |
| Source | Full source commit, code/checkpoint hashes, local seal and dated amendments |
| Exposure | Frozen encoder or score source training rows; FIT/CAL/ATTACK/INNER direct row, duplicate and group overlap; prior INNER exposure |
| Admission | Role-only export hash, allowed labels/procedure, old assessment and HEAD_VALIDATION exclusions |
| Service | Complete wire fields, metadata, hard-code/bank capacity, public parameters, query and randomness contract |
| Fit | Task/protection/attack unit IDs, queue receipts, CPU and memory, checkpoint selection and calibration |
| Utility | Per seed/task/reference accuracy, natural-log NLL, summed Brier, recall and constant/majority gain; point and simultaneous interval statuses |
| Recovery | Per seed local 1, local 2 and aligned-pair SEX AUC; fixed reader winner and defense-aware/ignore-recipient coverage |
| Matched controls | Exact IDs, eligible/ineligible reasons, capacity and input/output match, aliases and strongest-control rule |
| Comparison | Paired control-minus-candidate coalition AUC and candidate-minus-control local AUC, intervals and all required clauses |
| Replay | R04 independent endpoint/release/selection result, actual deployment test, unresolved defects |
| Custody | Private output hashes, local snapshot/restore, remote and off-device status |

## Claim register to fill after R00 results

| Claim ID | Question | Necessary comparison | Current status | Evidence and limitation |
| --- | --- | --- | --- | --- |
| D-SERVICE | Did D transmit a real hard public code with code-only task output? | Saved release, representative deployment, independent restore | `TECHNICAL_REPLAY_PASS_BOTH_SEEDS; DEPLOYMENT_PENDING` | R04 raw-to-wire and complete views exact on both paired seeds; R18 separate acceptance pending. |
| D-USEFUL | Was D point and interval useful on both tasks against U0 and Ucal? | Per seed, task and reference utility clauses | `POINT_SHORTFALL_BOTH_SEEDS` | D-K misses occupation accuracy/NLL/Brier in both seeds and seed-0 income Brier; no simultaneous bounds. |
| D-COMPRESSION | Did protected D improve beyond equally trained task-only hard-code T? | Same initialization, capacity, rows, training allowance and complete views | `DESCRIPTIVE_RECOVERY_REDUCTION_WITH_UTILITY_COST` | Pair T-minus-D-K AUC 0.026676/0.042181 by seed; both arms fail occupation confidence caps. |
| D-COMPETITIVE | Did D beat every eligible relevant control without a local shift? | T, local repair, both freeze-first sequential orders and ordinary adversarial comparator | `NO_TWO_SEED_POINT_ELIGIBILITY; K567_PENDING` | R04 all-seven point attack replay passed, but every ten-bit arm fails the task point gate; the matched eligible set is empty. |
| D-KERNEL | Did the fixed-kernel objective add benefit? | D-K versus matched T and any admitted D-C objective variant | `NO_ELIGIBLE_COMPONENT_GAIN_YET` | Lower AUC accompanies worse utility; MMD is no privacy certificate. |
| D-JOINT | Did joint D design add benefit? | Both local and both freeze-first sequential orders under same service | `NOT_EARNED_AT_POINT_GATE` | L/S12/S21 all fail utility; no eligible matched joint comparison. Do not generalize to all sequential laws. |
| F-POINT | Does unchanged continuous latent F retain the locked task utility? | Two seeds, both tasks and U0/Ucal, independently replayed source releases | `POINT_FEASIBLE_24_OF_24` | F exposes 16 continuous values per recipient; pair AUC 0.869335/0.862196; simultaneous intervals pending. |
| A-SERVICE | Was an actual sampled confidence output transmitted and scored? | Fixed draw realization, public bank/code/metadata and replay | `PENDING` | Expected utility and sampled utility are distinct. |
| A-USEFUL | Did A meet both task and confidence limits? | Expected-law arithmetic and the precommitted fixed sampled realizations | `PENDING` | No averaged-probability substitution or best-draw selection. |
| A-RANDOMNESS | Did randomness add benefit? | Strong same-bank deterministic and task-only stochastic controls | `PENDING` | A deterministic tie removes randomness attribution. |
| A-COMPETITIVE | Did A beat every eligible score-track control without a local shift? | Product, local, both freeze-first sequential designs and compatible incumbents | `PENDING` | Do not rank as feature sharing. |
| A-JOINT | Did central joint design add benefit? | Both freeze-first design orders; exact sampling factorizations recorded as aliases | `PENDING` | A joint law can be sequentially sampled; no general expressivity claim. |
| CONFIRMATION | Is there independent confirmation? | New authorized, genuinely unexposed population and prospective protocol | `NO` | This campaign only uses reused development data. |

## Evidence-linked result sentence template

`[Arm ID]` transmitted `[exact fields/capacity]` to recipient 1 and `[exact fields/capacity]` to recipient 2. On the same `[N]` INNER people in paired seeds `[IDs]`, its task utility was `[per-clause point and interval status]` against both U0 and Ucal. SEX AUC was `[local 1]`, `[local 2]` and `[pair]`; the strongest eligible `[control ID]` had `[three AUCs]`. Coalition benefit (control minus candidate) was `[point, simultaneous interval]`, while local worsening was `[two points and intervals]`. `[Component]` attribution is `[earned/not earned/incomplete]` because `[matched ablation and sequential evidence]`. Independent replay `[scope/result]`. These are selection-influenced development results, with `[remaining gaps]`.

## Required closeout checks

R19 will not upgrade a claim until the exact arm registry, unit ledger, endpoints, all mandatory controls and R04 replay agree. The final human handoff must give the strongest favorable and adverse matched comparisons, identify unrun/failed units, and state what is runnable. Numerical overlap between screens and any pilot must be disclosed.

### Historical anchors, not new evidence

- Reference repair at local commit `ce1308a52fb97b4673d26f44c1466ce3b0cea379`: 24/24 F point utility costs met, but 14/24 simultaneous utility clauses unresolved; joint mask equalled task-only in both seeds; coalition AUC 0.8972768264 versus S21 0.8959778086. Source: `results/pcrl_utility_reference_repair_v1/RESEARCH_DECISION.md` and `HANDOFF.json`.
- Prior portfolio at local commit `38d641189d3e253ddff0ca372d295a034752017f`: B was real-data tested without a competitive criterion pass; A protected channel and D latent repair were not Adult method screens. Source: `results/pcrl_method_portfolio_v1/RESEARCH_DECISION.md` and `HANDOFF.json`.
- Capacity-matched mask predecessor at local commit `9e6bfea62e82297d024fd9588a5f4bfb30e0a9de`: stopped at its utility gate with no fresh privacy attack, so it cannot supply a D/A privacy result. Source: `results/pcrl_capacity_matched_feature_selection_v1/RESEARCH_DECISION.md` and `HANDOFF.json`.

All three pins resolve as local Git commits. This template makes no GitHub or off-device verification claim.
