# R3 utility and inference contract (prefit proposal)

This file defines the R3 implementation boundary for R0's campaign lock. It is not a shared lock and does not authorize a real-data read by itself.

## Exact output service

For each person, a fitted policy gives a normalized mass over up to 16 ordered *pairs* of canonical global W_TSTAR public bank IDs. Invalid slots carry zero mass. Each recipient's fixed bank table maps its own emitted ID to one probability vector and one first-tie argmax decision. All valid actions retain that recipient's B0 decision class. The exact B0 pair is in every support, with the B0 identity policy assigning it mass one. Duplicate pair actions carry aggregate probability equal to the sum of their masses; their existence does not create a new public state. No fine source token, action position, teacher score, latent value or private random seed is transmitted.

The source A protocol already chose exact expected loss over the actual sampled public outputs. For each task, row and policy, R3 scores expected accuracy as the law-weighted probability of a correct table decision, expected natural-log NLL as the law-weighted NLL of each table output, and expected summed multiclass Brier as the law-weighted Brier of each table output. The NLL probability floor is `1e-12`; Brier sums across all 2 income or 6 occupation classes. **Loss of the averaged probability vector is a different service and is not used.** Deterministic policies' expected and realized task costs coincide.

## Three separate cost roles

On `OSF_DEFENSE_FIT`, the new optimization proxy compares each task/seed policy's expected accuracy, NLL and Brier with its *own frozen B0* on the same rows. Inclusive limits are accuracy loss ≤0.002, NLL excess ≤0.001 nats, Brier excess ≤0.0005. Exact B0 has zero differences. These fitting constraints are not evidence about INNER or population utility. The old U0/Ucal-relative FIT costs are recomputed as diagnostic columns; the source audit's W_TSTAR occupation U0 NLL excess was 0.0143120106 at seed 0 and 0.0154205557 at seed 1, so that retired gate genuinely rejected identity.

On `INNER_SELECTION`, primary point eligibility compares expected policy costs with the unchanged U0 **and** Ucal on the *same people*, each seed and task. Inclusive limits are candidate minus reference accuracy ≥−0.010, NLL excess ≤0.010 nats, Brier excess ≤0.005. All 24 seed/task/reference/metric clauses must pass for a policy to enter the fresh SEX attack stage. J is nominated only if J itself passes all 24. B0 remains a reference even if other arms fail. An old FIT proxy failure does not bar INNER eligibility under the new design.

Cached one-draw-per-person releases are the actual recipient service for recovery attacks. For stochastic policies, three precommitted complete paired release realizations per source seed are scored separately as secondary realized utility diagnostics. They are not concatenated into a recipient view and the favorable draw is never selected. Because the primary utility estimand integrates the exact fitted law, sampling variance does not enter its point cost; mechanism variance enters the realized diagnostics and privacy summaries. No all-label or population utility guarantee follows.

## Fixed comparison and interval proposal

The prospective ordered arm registry is `B0,T,D,P,L,J,S12,S21,W_Q,P_STAR`. If W_Q or P_STAR cannot be admitted, its preregistered clauses remain null. The fixed family comprises:

| Clause kind | Dimensions | Count |
| --- | --- | ---: |
| Exact-law primary utility | 10 arms × 3 summaries (`s0,s1,mean`) × 2 tasks × 2 references (`U0,Ucal`) × 3 metrics | 360 |
| Complete-view recovery | 45 unordered arm pairs × 3 views (`r1,r2,pair`) × 9 summaries (six seed/draw, two seed means, grand mean) | 1,215 |
| **Total** | | **1,575** |

The required primary seed-specific utility gates use `s0` and `s1`; the equal-seed `mean` is descriptive and covered for inference. Every potential selected comparison belongs to the family, including comparisons with B0 and the conditional P_STAR. The strongest eligible competitor for J is the point-eligible non-J arm with the **lowest** equal-seed-mean pair AUC, ties broken by the declared arm order. This is chosen from `B0,T,D,P,L,S12,S21,W_Q,P_STAR` after admitted service and utility eligibility are resolved. Raw pair benefit is selected-control AUC minus J AUC; local worsening is J AUC minus comparator AUC. R3 proposes the no-local-shift guard against both B0 and the selected strongest competitor; R0 must seal its precise comparator before fitting. Report every seed/draw and aggregate result, and do not call selection on reused data independent confirmation.

For duplicate-group uncertainty, draw 2,234 INNER group IDs with replacement per replicate and include all member rows with multiplicity. Use the same group draw for every seed, task, arm, view and contrast. A fixed cached-release clause holds its realization fixed and has group uncertainty only. For each seed-mean recovery replicate, draw a three-ID tuple from `{0,1,2}` with replacement **once for the whole seed and all arms/views**, average those three complete releases, and then average source seeds for the grand mean. This crosses people and mechanism realization uncertainty without giving a recipient repeated draws. Use 2,048 replicates as in source A. When some fixed slots remain null, use two-sided Bonferroni at fixed K=1,575, Student df=2,233 for group-only clauses, df=2 for three-release mean recovery clauses. Exact-law utility uses group-only uncertainty. Boundary equality passes: accuracy lower bound ≥−0.010; NLL/Brier upper bounds ≤0.010/0.005; pair-benefit lower bound ≥0.020; local-worsening upper bound ≤0.010. If a bound crosses, status is `UNRESOLVED`; lower-bound cost violation is `VIOLATION`. These are approximate simultaneous development intervals conditional on fitted policies and readers, with coarse release precision from only three realizations.

The R3 pure scorer is `bank_anchored/r3_utility.py`. R5 must independently implement endpoint math rather than import it. R4 owns actual recovery readers, and R0 seals exact seeds, ordered clause IDs and interval implementation before the relevant fits.
