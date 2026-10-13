# Actual own-head geometry and disclosure

Ordinary head-row-space projection preserves the current useful heads on every audited row. It changes the released feature information, and lowers measured local income SEX recovery substantially. Strong occupation and aligned-pair recovery persists. This is a tested compression baseline, with no new defense training and no new privacy-method claim.

## Definition and arithmetic

For the current affine head, class-center its weight matrix and bias to obtain A and c, so centered logits are Az+c. Construct an orthonormal row-space basis B in float64 under the frozen SVD rule, with P=BᵀB=A⁺A. The admitted algebra check is AP=A. The actual disclosure uses q=zBᵀ and then Pz=qB. It never adds the discarded residual back through a rounded full projector. Actual logits, probabilities and decisions are recomputed from that released representation under the current own head.

The released wires contain Pz16, centered own logits, own probabilities and the actual own decision. Raw z, its residual, teacher scores, raw83, task/protected labels, IDs and hidden metadata are absent. Reader banks may use the public fixed head geometry. The rank rule, ambiguity handling, tie convention and tolerances were frozen in [HEAD_GEOMETRY_SPEC.md](r2/HEAD_GEOMETRY_SPEC.md).

Measured ranks are1 for income and5 for occupation for all six U0/RAW/JOINT source-seed anchor packages. A projected wire still occupies16 representation columns, but those columns have only1 or5 effective degrees of freedom. The aligned representations therefore have6 effective dimensions instead of32. This is a narrower disclosure contract and does not demonstrate equal-capacity representation superiority or usefulness for future heads.

In exact arithmetic Pz can be reconstructed from centered logits and public head parameters. An invertible expression of those same scores removes no information. Any measured recovery difference between score and projection readers reflects their finite models, preprocessing and numerical realization, rather than a proved information-theoretic advantage for re-expression.

## Real-row checks and independent replay

The runner performed36 recipient/role audits: three anchors, two seeds, two heads and three authorized roles (FIT, ATT, label-free INNER forwarding). All passed AP, logits, probabilities and exact decisions; decision mismatch count was0. Maximum absolute logit discrepancy was2.486900e−14 and probability discrepancy2.498002e−15, below the declared1e−10 tolerance. No row had a top-two logit gap within the frozen1e−10 near-tie threshold, so the real-row evidence does not itself exercise an actual near-tie case.

[HEAD_GEOMETRY.json](HEAD_GEOMETRY.json) preserves those pre-INNER checks. The independent audit separately recomputed the projector and real-row arithmetic. Its later [all-release forward replay](r7/INDEPENDENT_ALL_RELEASE_FORWARDS_V2.json) covers72 full/projected packages across216 role forwards and432 encoder/head evaluations, using original artifacts and independent forward math. The initial adapter mismatch and its corrected replay remain preserved rather than hidden.

Identical decisions must give identical frozen decision-only attacks. The report explicitly checked15 decision score curves per matched full/projected anchor/seed/role:12 matched role identities in total. They are exact aliases, counted as no additional scientific evidence. See [DECISION_READER_ALIASES.json](r8/report/DECISION_READER_ALIASES.json).

## Recovery and usefulness

The supplemental complete-view family includes all18 applicable readers for each local view and54 for each pair, including exact local-ignore aliases. Paired INNER AUCs are:

| Anchor | Full income / occupation / pair | Projected income / occupation / pair |
|---|---|---|
| U0 | 0.821195 / 0.858140 / 0.870143 | 0.710008 / 0.853580 / 0.860816 |
| RAW .3 epoch 5 | 0.751910 / 0.838102 / 0.856824 | 0.689458 / 0.828298 / 0.846138 |
| JOINT .3 epoch 5 | 0.815163 / 0.846895 / 0.861891 | 0.699037 / 0.830861 / 0.848239 |

Projected probabilities preserve accuracy/NLL/Brier apart from rounding. RAW occupation accuracy/NLL/summed Brier remain0.495302/1.269066/0.648705; JOINT remains0.493512/1.268751/0.648444. Both projected configurations retain24/24 point utility clauses. Simultaneous K864 utility remains5 PASS/19 UNRESOLVED per configuration; projection does not establish an otherwise unresolved source utility conjunction.

In K18, RAW projection versus full RAW has pair benefit0.010686 [0.000388,0.020984], UNRESOLVED against the required0.020. JOINT projection versus full RAW has0.008585 [−0.001563,0.018732], SHORTFALL. These do not establish a competitive repair. Both projected anchors establish their two local nonworsening guards against full RAW within that family.

The original K108 base9 family remains separately archived. The stronger supplementary envelope raises U0-projection pair AUC from0.860185 to0.860816 and JOINT-projection income AUC from0.698026 to0.699037; ignoring applicable score/decision/confidence attacks would understate complete-view recovery. See [COMPLETE_VIEW_AUDIT.json](r8/complete_view/COMPLETE_VIEW_AUDIT.json) and [RELEASE_SURFACE_COMPARISONS.csv](r8/report/RELEASE_SURFACE_COMPARISONS.csv).

The supported interpretation is measured extra-feature leakage alongside substantial prediction leakage. The remaining opportunity requires modifying the actual occupation task computation or disclosure contract with explicit usefulness tests. Ordinary projection is a necessary baseline and receives credit for its ordinary compression gain.
