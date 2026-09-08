# Fixed coalition contrasts

All six conditions and three seeds are complete.

The comparisons below were declared before fitting. They are descriptive differences on reused development households, not significance tests or privacy guarantees. Utility differences are left loss minus right loss; negative is better. Attack differences are left signed gain minus right signed gain; negative means less measured recovery. Cells show mean ± sample SD, unweighted / PWGTP, on the same selected predictions.

## Coalition objective and stronger local pressure

J−I and J−Iplus answer different questions. Iplus changes only the additional local SEX/RAC1P terms. Nominal per-target/branch coefficients match J, while realized gradients and observer information need not. Opposing-source gradients are unchanged. [Frozen policy](PURPOSE_POLICY.md).

| Contrast | A residence loss Δ U / W | B commute loss Δ U / W | AB SEX gain Δ U / W | AB race gain Δ U / W |
| --- | --- | --- | --- | --- |
| F_J − F_I | 0.005794 ± 0.004640 / 0.004370 ± 0.006184 | 0.001243 ± 0.001745 / 0.000442 ± 0.001403 | -0.013469 ± 0.007060 / -0.011413 ± 0.006498 | -0.006554 ± 0.008404 / -0.006309 ± 0.010359 |
| F_J − F_Iplus | 0.003820 ± 0.004486 / 0.001279 ± 0.003797 | 0.001082 ± 0.002222 / 0.000935 ± 0.002013 | -0.005827 ± 0.003787 / -0.002704 ± 0.004336 | -0.003364 ± 0.012472 / -0.004679 ± 0.011083 |
| F_Iplus − F_I | 0.001974 ± 0.008053 / 0.003091 ± 0.009554 | 0.000160 ± 0.001162 / -0.000493 ± 0.001264 | -0.007642 ± 0.003889 / -0.008709 ± 0.003052 | -0.003189 ± 0.004696 / -0.001630 ± 0.004071 |
| P_J − P_I | 0.000399 ± 0.000191 / 0.000218 ± 0.000135 | -0.000241 ± 0.000294 / -0.000389 ± 0.000555 | -0.001176 ± 0.000106 / -0.000687 ± 0.001010 | -0.000259 ± 0.002784 / -0.000256 ± 0.002554 |
| P_J − P_Iplus | 0.000110 ± 0.000431 / -0.000086 ± 0.000297 | -0.000246 ± 0.000313 / -0.000203 ± 0.000212 | -0.000175 ± 0.000165 / -0.000403 ± 0.000579 | 0.000567 ± 0.000126 / 0.000905 ± 0.000653 |
| P_Iplus − P_I | 0.000290 ± 0.000252 / 0.000304 ± 0.000209 | 0.000005 ± 0.000368 / -0.000186 ± 0.000562 | -0.001001 ± 0.000262 / -0.000284 ± 0.001447 | -0.000826 ± 0.002661 / -0.001162 ± 0.002221 |

Attack cells above use expanded_catchup360. [Every paired role/scope/budget/seed](PAIRED.csv) and [aggregates](PAIRED_AGGREGATE.csv) retain standard independent and public-head-expanded conclusions separately. A gain that exists only before inherited/public/saved-start candidates enter is not robust to that legal access.

## Feature capability is task-specific

Residence belongs to A and commute to B. The .01-nat feature-advantage reference is evaluated separately. The .005-nat sensitive-gain excess references are evaluated for the task recipient and AB. Neither a residence result nor a global conjunction substitutes for a commute result.

| Comparison | Residence loss Δ U / W | Commute loss Δ U / W | AB SEX gain Δ U / W | AB race gain Δ U / W |
| --- | --- | --- | --- | --- |
| F_I − P_I | -0.019044 ± 0.009780 / -0.015572 ± 0.012053 | -0.002484 ± 0.001078 / -0.001652 ± 0.001642 | 0.014504 ± 0.011810 / 0.013516 ± 0.010793 | 0.018995 ± 0.004724 / 0.017616 ± 0.004539 |
| F_Iplus − P_Iplus | -0.017360 ± 0.001563 / -0.012785 ± 0.004432 | -0.002329 ± 0.000699 / -0.001959 ± 0.001055 | 0.007863 ± 0.007990 / 0.005091 ± 0.008728 | 0.016631 ± 0.004844 / 0.017147 ± 0.002863 |
| F_J − P_J | -0.013650 ± 0.006464 / -0.011420 ± 0.007752 | -0.001000 ± 0.002022 / -0.000821 ± 0.002513 | 0.002211 ± 0.004981 / 0.002790 ± 0.004195 | 0.012700 ± 0.007959 / 0.011563 ± 0.008743 |
| F_J − E | 0.009041 ± 0.008505 / 0.007258 ± 0.011032 | 0.004493 ± 0.002911 / 0.001635 ± 0.002270 | -0.002796 ± 0.005728 / -0.000341 ± 0.002825 | -0.014438 ± 0.009432 / -0.008402 ± 0.015485 |

| Scoring | Reserved task | F-J comparator | .01 utility inequality /3 | Utility + both AB gain inequalities /3 | Complete sensitive support /3 |
| --- | --- | --- | --- | --- | --- |
| U | same_residence | P_J | 2 | 0 | 0 |
| U | same_residence | E | 0 | 0 | 0 |
| U | commute_over20 | P_J | 0 | 0 | 0 |
| U | commute_over20 | E | 0 | 0 | 0 |
| W | same_residence | P_J | 1 | 0 | 0 |
| W | same_residence | E | 0 | 0 | 0 |
| W | commute_over20 | P_J | 0 | 0 | 0 |
| W | commute_over20 | E | 0 | 0 | 0 |

All task-specific own-view and coalition checks, including numerical versus support-qualified status, are in [feature_capability.json](feature_capability.json). Opposing-task recovery has no invented pass threshold.

## Audit scope, budget and evidence limits

AB includes every legal singleton sensitive candidate, not only each singleton winner. Its selected validation loss cannot exceed any included candidate on the same rows. The validation winner can have worse development loss than a singleton development winner. [Coalition-minus-singleton evidence](COALITION_MINUS_SINGLETON.csv) retains that distinction.

Public native heads add fit budget and a composed function family for F. P reuses the identical wire candidates; its hidden features remain unavailable. [Scope expansions](AUDIT_SCOPE.csv), [budget differences](AUDIT_BUDGET.csv), [candidate lineage](CANDIDATE_LINEAGE.json.gz), [all learning curves](AUDIT_CURVES.csv.gz) and [support](SUPPORT.csv) expose these differences.

The standalone saved-observer candidate is diagnostic and excluded from selection. Epoch zero remains eligible inside its catch-up trajectory, so an expanded-catchup advantage can reflect inherited fitting exposure without beneficial additional optimization. [Audit construction and exposure](COALITION_AUDIT.md).

Nested120/360 selected losses differ in 64 seed/condition/view/target/scope cases (256 scoring cells). A zero change means the selected trajectory checkpoint stayed competitive; it does not establish a complete attacker class or privacy.

RAC1P retains the nine-class schema. Missing independent attacker-fitting/validation support and exposed-control failures remain visible rather than repaired by changing the population. Full race and all-target protection are not certified. Representation-fitting observer support is a separate exposure pool. The deterministic repeated-output check duplicates the same version; it is not the fresh-noise experiment in the historical exact note.

The final [research decision](RESEARCH_DECISION.md) interprets coalition benefit over I, survival against Iplus, residence and commute capability, source retention and audit reversals as separate findings.
