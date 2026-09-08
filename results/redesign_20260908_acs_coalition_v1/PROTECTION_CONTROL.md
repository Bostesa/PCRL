# Coalition protection versus stronger local sensitive protection

Iplus changes only the local SEX/RAC1P penalty. It is the control needed to distinguish a coalition-input intervention from simply increasing the local sensitive-attribute gradient. It does not double the opposing-source penalty, add observers, or add fitting labels.

Every normalized observer loss uses its target's fixed, unsmoothed empirical entropy from representation-fitting rows. The same entropy is used for that target in every view. Eligibility masks retain each target's eligible-row denominator; an all-missing minibatch contributes differentiable zero, with the declared outer means unchanged. The source objective is the equal-purpose mean `(mean(income,employment)+public_coverage)/2`, giving each A task weight .25 and B's task weight .5.

Let `ell(v,t)` be that normalized observer CE. `L_A` averages A's coverage/SEX/race losses; `L_B` averages B's income/employment/SEX/race losses. `L_ind=(L_A+L_B)/2`. `M_A`, `M_B`, and `M_AB` each average the two sensitive losses on their respective view.

| Regime | Forward-model objective |
| --- | --- |
| I | `L_source − .1 L_ind` |
| Iplus | `L_source − .1 L_ind − .1 (M_A + M_B)` |
| J | `L_source − .1 L_ind − .1 M_AB` |

`M_A+M_B` is a **sum**. Averaging those two terms again would halve the intended extra local control. The following magnitudes multiply individual normalized CEs; all protection signs are negative in the minimized forward objective.

| Role | I | Iplus | J |
| --- | ---: | ---: | ---: |
| A: opposing public coverage | 1/60 | 1/60 | 1/60 |
| B: each opposing income/employment task | 1/80 | 1/80 | 1/80 |
| A: each local sensitive target | 1/60 | 1/60 + .05 | 1/60 |
| B: each local sensitive target | 1/80 | 1/80 + .05 | 1/80 |
| AB: each coalition sensitive target | 0 | 0 | .05 |

Each branch therefore receives the nominal extra coefficient .05 per sensitive target in Iplus and J. Their gradients are not equal: J's observer receives the complete pair, whereas the two Iplus observers receive separate singletons. The observer functions, CE values, conditioning, gradient directions and resulting optimizer steps can differ. There is no adaptive coefficient or norm matching.

All six conditions train the same nine observer roles, including coalition observers in I/Iplus. The observer optimizer minimizes the fixed mean of nine ordinary masked CEs; entropy normalization is used only for the reversed forward objective. Every role has its own 64/32 ReLU network and deterministic seed `20260911 + 100*study_seed + 10*view_index + target_index`. View indices are A=0, B=1, AB=2, with the fixed target order in [PROTOCOL.md](PROTOCOL.md). Dimensions differ between F and P and are explicitly recorded; observer parameter totals are not claimed equal across interfaces.

| Interface | A observer parameters, 3 roles | B observer parameters, 4 roles | AB observer parameters, 2 roles | All observers |
| --- | ---: | ---: | ---: | ---: |
| F | 9,933 | 13,167 | 8,747 | 31,847 |
| P | 7,245 | 9,327 | 5,035 | 21,607 |

Both interfaces have the same 6,355 forward parameters. Their observer counts reflect 16/16/32 versus 2/1/3 input widths, with matching hidden widths and target schemas.

The two-purpose source warmup is fitted once per seed. After its 60 epochs, F and P receive identical forward tensors and complete Adam state. Each gets 20 observer-only epochs, during which forward tensors and forward Adam remain unchanged. Within each interface, I/Iplus/J clone every forward/observer tensor, both complete Adam states, and the fully specified continuation schedule. The 80 continuation epochs use exactly three detached observer updates per forward-model update. No reserved label, teacher, preservation term, decoder, reconstruction path, or validation-selected forward checkpoint exists in this trainer.

The completed records have the following counts in **every final condition**. They include the shared prefixes in each final model's history; the common source warmup was executed only once per seed. Each source task received 140 row passes, and each of the nine observer roles received 260. All five training labels were known on these fitting rows, so the presentation counts below apply to every respective source task or observer role.

| Seed | Fitting rows | Forward Adam steps | Observer Adam steps per role | Label presentations per source task | Label presentations per observer role |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 10,513 | 5,880 | 10,920 | 1,471,820 | 2,733,380 |
| 1 | 10,428 | 5,740 | 10,660 | 1,459,920 | 2,711,280 |
| 2 | 10,551 | 5,880 | 10,920 | 1,477,140 | 2,743,260 |

Known labels do not imply support for every category: the absent race category remains absent. The complete per-role, per-stage evidence is in the [seed 0](seed_0/training/training.json), [seed 1](seed_1/training/training.json), and [seed 2](seed_2/training/training.json) training records.

The same first source-warmup minibatch supplies all diagnostics at I, W, interface forks and finals. [Gradient records](GRADIENT_DIAGNOSTICS.csv) separate raw and applied source, ordinary-individual, extra-local and coalition gradients for each branch mapper and its native heads, plus their norms, dots and defined cosines. Extra-local and ordinary-local A-to-B/B-to-A gradients are checked to be zero. Coalition gradients reach both branch mappers when nondegenerate. F protection leaves source heads outside its gradient path; P protection differentiates through the released sigmoid probabilities and their source heads. This is part of the declared interface intervention. Initialized observers used at I/W are explicitly hypothetical; no protection update occurs in source warmup.

At the exact common interface fork, the extra applied mapper-gradient norms differ despite the matched nominal coefficients. These are three-seed means on the predeclared diagnostic minibatch, before Iplus/J continuations diverge; they are not population estimates or privacy measurements.

| Interface / branch | Iplus extra-local norm | J coalition norm | Seeds with Iplus norm greater than J |
| --- | ---: | ---: | --- |
| F / A | .027841 | .019255 | 0, 1, 2 |
| F / B | .035850 | .018851 | 0, 1, 2 |
| P / A | .005042 | .004727 | 1 |
| P / B | .006194 | .005884 | 1, 2 |

Thus Iplus supplies a larger extra norm on both F branches at this point, while the P ordering varies by branch and seed. The recorded combined-protection norm measures the vector sum of applied components, not the sum of their norms; cancellation can change its ordering. No coefficient was adjusted to these observations. Across all 54 diagnostic points, cross-branch local/extra norms and F protection-to-head norms are exactly zero. P's raw protection-to-head norms and both branches' raw coalition mapper norms are nonzero at every checked point. These checks establish the intended paths, not a causal explanation of final recovery or utility.

Checkpoint schedule state records all phase seeds/hashes, the phase/epoch boundary, next minibatch index, row/batch sizes, and torch RNG state. All permutations were materialized before fitting. Diagnostics take no optimizer step and preserve tensors, existing parameter gradients and RNG state. [Focused training tests](../../tests/test_acs_coalition_training.py) independently reconstruct the targeted loss algebra, reproduce all six miniature fork gradients exactly, and verify access paths, matching initial heads, masks, state forks and exposure counts. [Independent training replay](../../scripts/verify_acs_coalition_training.py) checks completed scientific checkpoints without refitting them.

[Final training replay](TRAINING_REPLAY.json) passed all three seeds, exact model/Adam/schedule forks and all **54 gradient points with maximum scalar discrepancy 0**. It also verified all 4,512 artifacts preserved through the explicitly recorded operational loader recovery. Replay took 2.498 seconds and performed no fitting; the training and audit modules retained their original frozen identities.

Scientific interpretation must report J−I and J−Iplus separately, alongside Iplus−I. A J−I recovery reduction alone does not establish a benefit from the coalition's extra information over the stronger local control. Nor does a locally opposing gradient establish the cause of a final utility or audit result. Fixed authorized-task losses, all individual forbidden-task audits, inherited and public-composition audit scopes, and both scoring weights determine the measured tradeoff.
