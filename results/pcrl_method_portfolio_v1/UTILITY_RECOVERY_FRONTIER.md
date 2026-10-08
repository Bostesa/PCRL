# Adult development utility and recovery frontier

The figures are paired means across two fitted seeds on the same 2,235 INNER_SELECTION people. SEX AUC is the fixed complete-view reader envelope for recipient 1, recipient 2 and their aligned coalition. Smaller AUC means less measured recovery. Point utility means both task accuracy, natural-log NLL and summed multiclass Brier meet the frozen caps against **both** U0 and Ucal in **each** seed. It is not a simultaneous-interval pass. Different feature capacities and service contracts are not interchangeable controls.

## Team B: retrained feature representations

S4 numerically replayed all 16 arm-seed outputs and the two shared raw83 anchors. B utility and strongest-control comparison are in `B_SCREEN_AGGREGATES.csv` and `B_CONTROL_SELECTION.json`.

| Arm | Paired AUC r1 / r2 / coalition | Point utility both seeds | Interpretation |
|---|---|---|---|
| B-DVIB | 0.854166 / 0.851060 / 0.868611 | Yes | Deterministic bottleneck; coalition recovery exceeds RAWJ. |
| B-JVIB | 0.680765 / 0.787864 / 0.799879 | No | Lowest B coalition AUC; seed-0 occupation NLL/Brier miss. |
| B-LOCAL | 0.691032 / 0.829265 / 0.843151 | No | Local protection misses occupation utility. |
| B-RAW-TASK | 0.801213 / 0.873273 / 0.878615 | Yes | Raw task-only control. |
| B-RAWJ | 0.842926 / 0.847468 / 0.863083 | Yes | Strong raw joint control; strongest eligible comparator for B-DVIB. |
| B-SEQ12 | 0.694726 / 0.807512 / 0.823777 | No | First sequential order misses utility. |
| B-SEQ21 | 0.697580 / 0.834564 / 0.846922 | No | Reverse order misses utility. |
| B-TASK | 0.699500 / 0.847747 / 0.853680 | No | Identical-architecture task-only bottleneck misses occupation utility. |

The contextual raw83/Ucal anchor has r1/r2/coalition AUC 0.896971 / 0.896785 / 0.896971. It is not an alias of any B or C arm. B point-feasible arms do not gain the required 0.020 coalition AUC against their strongest eligible matched alternative. B simultaneous intervals were not computed.

## Team C: disclosed selected raw features

Team C completed all 135 corrected queue units; its independent S4 saved-model replay remains pending. Capacities, per-seed utility, per-class recall and simultaneous intervals are in `C_R1_AGGREGATE_FINDINGS.md` and its companion CSVs. The coalition receives the union of the selected raw feature values.

| Arm | Encoded columns r1 / r2 / union | Paired AUC r1 / r2 / coalition | Point utility both seeds |
|---|---|---|---|
| C-SIMPLE-FULL | 83 / 83 / 83 | 0.896785 / 0.896785 / 0.896861 | Yes |
| C-SIMPLE-REL0 | 77 / 77 / 77 | 0.801252 / 0.801252 / 0.801252 | No: occupation NLL/Brier |
| C-SIMPLE-TASK-SHARED1 | 42 / 42 / 42 | 0.898033 / 0.899751 / 0.900606 | Yes |
| C-SIMPLE-INDEP1 | 77 / 76 / 83 | 0.801252 / 0.894818 / 0.896868 | Yes |
| C-SIMPLE-COMMON1 | 76 / 76 / 76 | 0.894818 / 0.894818 / 0.895252 | Yes |
| C-SIMPLE-SEQ-IO1 | 77 / 76 / 83 | 0.801252 / 0.894818 / 0.896868 | Yes |
| C-SIMPLE-SEQ-OI1 | 76 / 76 / 76 | 0.894818 / 0.894818 / 0.895252 | Yes |
| C-SIMPLE-TASK-PAIR1 | seed-dependent / seed-dependent / 83 | 0.896361 / 0.898911 / 0.898928 | Yes |

For TASK-PAIR1, the exact encoded widths are 42/82/83 in seed 0 and 76/42/83 in seed 1. None of the eight C arms has a distinct exact semantic-and-encoded-capacity nonalias comparator in both seeds. REL0 versus FULL is a descriptive, width-unmatched 0.095610 coalition AUC reduction with an occupation utility point shortfall. Among seven point-utility-feasible arms, the lowest coalition AUC is only 0.001609 below FULL. The C common-criterion result remains incomplete until S4 numerical replay and matched-capacity comparison.

Team A's score-only channel and Team D's latent repair have no Adult method-fit/attack endpoints. Their missing rows are not negative results. No finalist was selected from this frontier.
