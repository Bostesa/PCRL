# Coalition audit

The audit tests recovery from each allowed interface and from the same person's paired outputs. The three attack pools are reported separately. The union of singleton attack candidates is retained in every sensitive-attribute coalition pool; consequently its minimum **validation** loss cannot exceed any included singleton candidate's validation loss. Development loss need not be monotone, because the chosen checkpoint and candidate are selected using validation data.

All three seeds completed the full prescribed matrix. Catch-up is decisive for race: it wins 52 of 54 eligible race endpoints at 360 epochs, compared with 21 of 54 SEX endpoints. The saved-start exposure therefore remains a material qualification to the apparent protection measured by fresh attacks. The complete [independent model/score replay](SCORE_REPLAY.json) passed for every seed, candidate, and legal projection path.

## What changed with audit budget

There are 231 role/seed endpoints per scope: seven conditions × eleven roles × three seeds. “Changed” means a different selected candidate or selected model state, rather than the automatically different metadata describing a longer trajectory. The table uses development log loss on the selected predictions; lower loss means more recovered target information.

| Scope | Changed selected candidate/state, 120→360 | Candidate ID changed | Development loss lower / higher among changed endpoints |
|---|---:|---:|---:|
| Standard independent | 22 / 231 | 5 | 7 / 15 |
| Expanded independent | 25 / 231 | 8 | 7 / 18 |
| Expanded with catch-up | 17 / 231 | 4 | 10 / 7 |

All F selections in the expanded catch-up pool retain the same selected model at both budgets. The P and direct-E changes are recorded alongside the F fresh-composition changes in [AUDIT_BUDGET.csv](AUDIT_BUDGET.csv), with both person-weighted and unweighted scores. A longer validation-selected trajectory does not guarantee lower development loss.

A concrete counterexample is seed 0, F_J, B/RAC1P: expanded-independent selection moves from epoch 65 to epoch 255 of a public-probability MLP. Validation loss improves, while development loss rises by `0.020299` nats (`0.014952` with person weighting). AB inherits this same candidate and change. This reverses the three-seed mean F_J−F_I race-loss difference in that scope: AB changes from `−0.001126` to `+0.005640` nats; B changes from `−0.003807` to `+0.002959` (weighted: `−0.000242` to `+0.004742`). The apparent reduction in recovery reflects a worse-generalizing selected attacker, not an additional protection update. The pooled F selections do not change.

Six feature/bank race-inequality records change at the longer budget, all for seed 0's expanded-independent F_J versus P_J comparisons and repeated across the two scoring weights. Residence with AB changes from failing to satisfying the **numerical** joint feature/bank inequalities; its joint assessment remains unassessable because race coverage is incomplete. Two commute comparisons change their race inequality but still fail the utility margin. No fixed-parent policy, attribute-halving inequality, or assessed joint conclusion changes. Small P rank reversals also occur, with mean differences on the order of `1e-5` to `1e-4` nats; these are not evidence of a stable ordering. See [feature_capability.json](feature_capability.json), [criteria.json](criteria.json), and [the full paired contrasts](PAIRED.csv).

## Saved-start versus fresh attacks

Catch-up wins 97 of 162 learned endpoints that have an applicable observer. These are endpoint counts, so a selected A/B observer inherited by AB can appear again; they are not counts of new trajectories. All 162 own catch-up trajectories were fitted regardless of which one ultimately won.

| Target | Catch-up winners / eligible endpoints at 360 | Winners selecting epoch 0 |
|---|---:|---:|
| SEX | 21 / 54 | 2 |
| RAC1P | 52 / 54 | 13 |
| Public coverage, on A | 16 / 18 | 3 |
| Income, on B | 6 / 18 | 1 |
| Civilian employment, on B | 2 / 18 | 0 |
| Total | 97 / 162 | 19 |

Nineteen winners therefore reproduce the unchanged saved observer. Among the later selections, seed 0 P_I's B/SEX catch-up selects epoch 330, and P_Iplus's B/income catch-up selects epoch 210. Their longer-budget development changes are small and mixed across scoring weights. Other fresh selections also occur late in the trajectory. The saved learning curves support continued uncertainty about audit saturation; the authorized study stops at 360 epochs and adds no further fits.

## What the attacker receives

| Interface | Purpose A | Purpose B | Coalition AB | Additional public composition |
|---|---:|---:|---:|---|
| F | 16 mapper coordinates | 16 mapper coordinates | 32 concatenated coordinates | Assigned frozen source heads produce 2, 1, and 3 probabilities respectively |
| P | 2 source probabilities | 1 source probability | 3 concatenated probabilities | Identical to the wire; deduplicated without another fit |
| Direct E | Frozen E, 16 coordinates | The same frozen E, 16 coordinates | Two identical copies, 32 coordinates | No native source head or saved observer |

A's forbidden targets are public coverage, commute over 20 minutes, SEX, and RAC1P. B's are income over $50,000, civilian employment at work, same residence, SEX, and RAC1P. AB is audited for SEX and RAC1P. These eleven roles use the same target-specific attacker rows across views and conditions. Every target is binary except RAC1P, whose full nine-class schema is retained.

The feature and prediction conditions have all nine genuine training observers: three on A, four on B, and two on AB. Residence and commute have no training observers and no catch-up candidate. The direct E control has none.

## Fixed candidates and selection

Each fresh role fits logistic regression, two independently initialized 64/32 ReLU MLPs, and the two fixed boosted trees (minimum leaf sizes 20 and 5). The MLPs run for 360 epochs on one continuous path, retaining the earliest best validation checkpoint within the first 120 epochs and within all 360. Logistic and tree candidates are shared across these two budgets. No family, coefficient, input dimension, or fitting row is selected by development performance.

Every genuine final observer gets its own direct-coordinate catch-up trajectory with Adam reset to its declared defaults. The saved observer is checked before the first update; its original coordinates and class alignment remain intact. The unchanged observer is exported as a diagnostic, excluded from candidate pools. The catch-up candidate may select epoch 0, so a catch-up winner does not necessarily establish that additional optimization helped.

| Selection scope | Eligible candidates |
|---|---|
| `standard_independent` | Five fresh wire candidates; AB also includes every A/B sensitive wire candidate through exact projection |
| `expanded_independent` | Standard pool plus the five fresh public-source-probability compositions for F, including all corresponding A/B projections into AB |
| `expanded_catchup` | Expanded independent pool plus the own saved-start catch-up candidate and every applicable A/B catch-up projection |

The selector minimizes unweighted attacker-validation log loss, breaking ties by candidate ID. The same selected predictions receive both primary unweighted and PWGTP sensitivity scores. No separate weighted selection is made. All candidates and all selection memberships are retained in each condition's `audits/audit_selection.json`; raw development scores and probabilities are in the corresponding metrics and local prediction files.

The exact original direct-E A/SEX and A/RAC1P nested fits and the original SEX/race exposed controls are reused. Additional direct-E roles and the five missing binary exposed controls are fitted under the predeclared role seeds. This is a fitting-budget extension of fixed families, without a family or data search. The original PCA32 policy parent retains its historical 120-epoch audit evidence; comparisons against that fixed parent are explicitly mixed-budget descriptive references, not newly matched 360-epoch parent audits.

## Exposure, depth, and preprocessing

The paired forward model contains 6,355 parameters: two 3,152-parameter mappers plus 34 A-source-head and 17 B-source-head parameters. There is no reconstruction decoder. Each mapper has two affine layers and one ReLU; each source head has one affine layer followed by a sigmoid. Only assigned heads are part of each native prediction interface.

All learned releases inherit the prescribed source and protection training. The 80 continuation epochs also route the prescribed forbidden-source and attribute protection losses through the forward mapper. F source heads receive their ordinary source gradients; P heads are part of the released wire and also receive protection gradients. The source-presentation count below describes the ordinary source objective, not the total label exposure of the learned system. Each assigned source head and mapper receives 60 base epochs plus 80 continuation epochs of source training on the original representation-fitting pool, or **140 source-label presentations per eligible fitting person per assigned target**. The public-head composition is therefore a fixed, source-trained function of the F output; it is not a fresh attacker trained solely on the attacker pool. Its attacker component is nevertheless fitted only on the designated attacker rows. This exposure and additional depth are disclosed separately from the standard wire pool.

The recorded fitting exposure is:

| Seed | Representation-fitting people | Source presentations per assigned target (140 epochs) | Presentations per observer role (260 passes) | RAC1P code 4 in representation fitting |
|---|---:|---:|---:|---:|
| 0 | 10,513 | 1,471,820 | 2,733,380 | 1 |
| 1 | 10,428 | 1,459,920 | 2,711,280 | 1 |
| 2 | 10,551 | 1,477,140 | 2,743,260 | 0 |

All three source labels and both attribute labels are eligible for every representation-fitting person in these fixed pools. The class-4 count describes one category, not missing target labels.

Each saved observer inherits **260 representation-fitting passes**: 20 observer-warmup epochs plus three detached updates per minibatch during 80 continuation epochs. Its catch-up path adds 120 or 360 passes through the disjoint capped attacker pool, starting from that genuine final observer with Adam reset. The exact original label and raw-row hashes, known-label counts, and presentations are recorded per observer. Catch-up and fresh candidate comparisons therefore have different inherited exposure.

A fresh MLP has three affine layers, two ReLUs, and `64 × input_width + 2144 + 33 × number_of_classes` parameters. A public-head composition adds the applicable frozen 34, 17, or 51 head parameters and one sigmoid stage to the fresh auditor's path from the F wire. A lifted singleton reuses that exact fitted network and preprocessing; projecting from AB adds no parameters or fitting exposure. Logistic and tree configurations are unchanged; actual saved models remain available locally.

Fresh candidate preprocessing is fitted on the prescribed attacker-fitting examples only. Catch-up uses identity preprocessing on its original wire coordinates. Both purpose mappers use the original representation-fitting PCA32 statistics. Public source probabilities are computed directly from F outputs and frozen public heads. No raw PCA32 input or hidden P feature is passed to an auditor.

## Controls and finite scope

The prior is the original per-class pseudocount-one fitting prior. Exposed controls use one-hot labels with the same candidate families and nested MLP budget. An absent category cannot be validated merely because overall control accuracy or AUROC is high: all class supports, undefined statistics, and failures remain reported. In particular, RAC1P code 4 has no independent attacker-fitting or attacker-validation examples in any seed; the original representation-fitting observer exposure is a separate pool and must not be conflated with independent support. At development evaluation, code 4 is absent in seeds 0 and 1 and occurs once in seed 2 (PWGTP 79). The selected race exposed control in seed 2 has overall accuracy `0.999664`, yet code-4 recall is zero despite one-vs-rest AUROC 1.0. The six binary exposed controls reach selected development losses no higher than `0.0000131` nats. Those successful supported-schema controls do not resolve the missing-category assessment or prove attack saturation.

For direct E, A and B contain exactly the same information. Different seeded fresh fits or input duplication can change numerical predictions, but AB receives no second information source. Inheriting every singleton candidate guards the coalition validation comparison against weaker fitting on duplicated coordinates.

The k = 1, 2, and 4 check repeats the same deterministic output. Both the generated arrays and selected 360-epoch auditors through first-copy projections are replayed exactly. This is a **duplication/invariance check**, with no fresh versions, noise, or additional released values. A finite predictive audit, a failed attacker, or deterministic repetition does not establish privacy, purpose compliance, or a guarantee against future attacks.
