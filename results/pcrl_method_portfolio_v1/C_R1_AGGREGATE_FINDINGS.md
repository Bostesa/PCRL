# Team C R1: aggregate development screen

**Status:** independent S4 R3 replay **PASS, 17/17 units**. The immutable Team C finalizer and the linked CSV extraction were originally marked `AGGREGATED_PENDING_S4_REPLAY`; that historical status is retained in those files. S4's replay index SHA-256 is `c406b2a1da773d53e893a38e06809f265b9d6b5121769e58f26b1501f301486a`; its maximum endpoint/clause difference was `2.22e-16`, and it independently found no exact nonalias comparator. These remain development aggregates on reused `INNER_SELECTION` people, not confirmation. The finalizer SHA-256 is `f8d5d894091e54a6993d16c4100d2e24c3f1723b6d3e70cf9397f7850c641c0e`; S3's finalizer receipt SHA-256 is `c17197f7ec28d45908ac0f7e7ab1a5c595a51f7d83766097bf1511e3689253e0`. All 16 arm-seed score-receipt hashes were checked against the finalizer before the CSV extraction. No person-level arrays were read for this report.

The active study is eight **standard grouped-mask and retraining** arms, at two paired model seeds. The joint grouped-selector prototype failed its prefit planted-XOR premise and was not fitted on Adult. It supplies no joint-design evidence.

## Every arm and seed

All numbers below are aggregate `INNER_SELECTION` scores. `Encoded` gives transmitted selected raw-column counts for income/occupation/coalition. Task triples are accuracy/NLL/multiclass Brier. AUC triples are recipient 1/recipient 2/aligned coalition complete-view SEX recovery under the fixed six-reader slate and ignore-recipient aliases. Values are rounded to four decimals; the public [screen rows](C_R1_SCREEN_RESULTS.csv) preserve full precision.

The [per-class aggregate table](C_R1_PER_CLASS_RESULTS.csv) gives each task's class support and recall for all 16 rows. The sixth occupation class has support **one** on `INNER_SELECTION`; its recall is suppressed in the public table as `SUPPRESSED_SMALL_CELL`. Its private aggregate remains available for audit, but cannot support a stable class-specific claim.

| Arm (`C-SIMPLE-` prefix) | Seed | Encoded | Income acc/NLL/Brier | Occupation acc/NLL/Brier | SEX AUC r1/r2/pair |
| --- | ---: | --- | --- | --- | --- |
| FULL | 0 | 83/83/83 | 0.8519/0.3158/0.2015 | 0.4957/1.2684/0.6511 | 0.8968/0.8968/0.8968 |
| FULL | 1 | 83/83/83 | 0.8559/0.3155/0.2005 | 0.4908/1.2663/0.6504 | 0.8968/0.8968/0.8969 |
| REL0 | 0 | 77/77/77 | 0.8510/0.3150/0.2003 | 0.4837/1.2782/0.6571 | 0.8013/0.8013/0.8013 |
| REL0 | 1 | 77/77/77 | 0.8515/0.3179/0.2012 | 0.4810/1.2801/0.6588 | 0.8013/0.8013/0.8013 |
| TASK-SHARED1 | 0 | 42/42/42 | 0.8559/0.3152/0.1996 | 0.4881/1.2632/0.6476 | 0.8979/0.8995/0.9005 |
| TASK-SHARED1 | 1 | 42/42/42 | 0.8515/0.3176/0.2011 | 0.4922/1.2654/0.6497 | 0.8982/0.9000/0.9007 |
| INDEP1 | 0 | 77/76/83 | 0.8510/0.3150/0.2003 | 0.4895/1.2647/0.6493 | 0.8013/0.8948/0.8970 |
| INDEP1 | 1 | 77/76/83 | 0.8515/0.3179/0.2012 | 0.5020/1.2622/0.6479 | 0.8013/0.8948/0.8968 |
| COMMON1 | 0 | 76/76/76 | 0.8515/0.3170/0.2009 | 0.4895/1.2647/0.6493 | 0.8948/0.8948/0.8955 |
| COMMON1 | 1 | 76/76/76 | 0.8546/0.3171/0.2010 | 0.5020/1.2622/0.6479 | 0.8948/0.8948/0.8950 |
| SEQ-IO1 | 0 | 77/76/83 | 0.8510/0.3150/0.2003 | 0.4895/1.2647/0.6493 | 0.8013/0.8948/0.8970 |
| SEQ-IO1 | 1 | 77/76/83 | 0.8515/0.3179/0.2012 | 0.5020/1.2622/0.6479 | 0.8013/0.8948/0.8968 |
| SEQ-OI1 | 0 | 76/76/76 | 0.8515/0.3170/0.2009 | 0.4895/1.2647/0.6493 | 0.8948/0.8948/0.8955 |
| SEQ-OI1 | 1 | 76/76/76 | 0.8546/0.3171/0.2010 | 0.5020/1.2622/0.6479 | 0.8948/0.8948/0.8950 |
| TASK-PAIR1 | 0 | 42/82/83 | 0.8559/0.3152/0.1996 | 0.4917/1.2659/0.6493 | 0.8979/0.8979/0.8979 |
| TASK-PAIR1 | 1 | 76/42/83 | 0.8546/0.3171/0.2010 | 0.4922/1.2654/0.6497 | 0.8948/0.9000/0.9000 |

## Utility and comparison decision

Against **both** frozen U0 and Ucal references, seven arms other than `REL0` meet all 12 mean accuracy/NLL/Brier caps at their two-seed point estimates. This is point feasibility only. Their simultaneous intervals leave at least six utility clauses unresolved; none establishes the complete common-criterion conjunction. `REL0` has eight of 12 point utility margins nonnegative. Its occupation NLL and Brier exceed the cap at the point estimate against both references:

| REL0 occupation excess | Point | Simultaneous 95% interval | Cap | Clause |
| --- | ---: | --- | ---: | --- |
| NLL vs U0 | 0.012699 | [-0.006663, 0.032061] | 0.010 | UNRESOLVED |
| NLL vs Ucal | 0.014151 | [-0.001556, 0.029859] | 0.010 | UNRESOLVED |
| Brier vs U0 | 0.010516 | [0.001084, 0.019948] | 0.005 | UNRESOLVED |
| Brier vs Ucal | 0.009960 | [0.002022, 0.017898] | 0.005 | UNRESOLVED |

The favorable descriptive contrast is `REL0` versus `FULL`: two-seed mean coalition AUC is **0.801252 versus 0.896861**, a reduction of **0.095610**. The simultaneous interval for that reduction is **[0.067824, 0.123396]**, conditional on these fitted models and reused people. Its local AUCs are also lower (both 0.801252 versus 0.896785). This compares **77-column** and **83-column** coalition disclosures, and `REL0` misses occupation mean utility caps at the point estimate. It therefore does not satisfy the competitive method criterion.

The adverse comparison is `INDEP1` and `SEQ-IO1`: recipient 1 AUC is **0.801252**, but their coalition AUC is **0.896868**, essentially the `FULL` coalition's **0.896861**. The two selected raw subsets unite to all **83** permitted columns, so the coalition sees the full raw input even though one recipient sees less. Among the seven point-feasible arms, the lowest coalition AUC is `COMMON1`/`SEQ-OI1` at **0.895252**, only **0.001609** below `FULL`, far short of the 0.020 target. `TASK-SHARED1` releases 42 columns to each recipient yet has coalition AUC **0.900606**, illustrating that column compression by itself is not a privacy result.

The finalizer reports **no distinct, exact semantic-and-encoded-capacity, nonalias Team C comparator for any of the eight arms** in both seeds. The Team C internal strongest-control clause is therefore `INCOMPLETE_NO_EXACT_C_CAPACITY_CONTROL` for every arm. In particular, `REL0`'s 77/77/77 encoded capacity comes from the unique six-column relationship atom; `INDEP1`/`SEQ-IO1` and `COMMON1`/`SEQ-OI1` reach the same reported frontier points; `TASK-PAIR1` has seed-dependent recipient encoded widths. S1 must still apply the portfolio-wide eligibility rule if another feature-track arm has the same actual disclosure and capacity. These data provide no Team C method win and no joint-design claim.

The simultaneous intervals use 2,048 paired-group bootstrap replicates over 2,234 `INNER_SELECTION` groups, a family maximum studentized critical value of 3.347, and one fixed release per person. Seeds are paired models, not independent populations. The intervals remain exploratory and conditional on selection and fitted models. S4 independently reloaded the saved models and replayed endpoints and clauses in its R3 verification; the replay agrees with the reported values to numerical precision. It does not make the reused development population independent confirmation.
