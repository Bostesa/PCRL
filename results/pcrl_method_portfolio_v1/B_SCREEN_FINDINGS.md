# Team B development screen

Team B retrained task-sufficient bottlenecks and heads on the permitted raw inputs. Its joint stochastic arm (B-JVIB) was compared with a deterministic bottleneck, task-only bottlenecks, separately protected recipients, both sequential orders, and matched raw joint/task controls. Each recipient disclosed its protected features and head output; attackers saw each complete local view and their coalition. Two paired seeds used the permitted Adult development roles. This is an exploratory comparison on reused outcomes.

All 16 arm-by-seed screen rows, two unprotected feature anchors, both useful references and all three SEX attack views are in `B_SCREEN_AGGREGATES.csv`. The public `B_CONTROL_SELECTION.json` records the frozen comparator rule and point decisions. S4 independently replayed the 18 rows and audited 822 aggregate numbers (maximum absolute difference 4.44e-16). All required B V2 leakage controls passed. The source stage, replay and audit hashes are bound in `B_AGGREGATE_PROVENANCE.json`.

| Arm | Coalition AUC, paired mean | Both-reference utility point eligible in both seeds? | Interpretation |
|---|---:|---|---|
| B-JVIB | 0.799879 | No | Lowest recovery, but seed-0 occupation loss/Brier fails the frozen mean-utility limits. |
| B-DVIB | 0.868611 | Yes | Worse coalition AUC than eligible raw joint control by 0.005529; recipient 1 AUC also worsens by 0.011239. |
| B-TASK | 0.853680 | No | Seed-0 occupation Brier exceeds original U0 allowance. |
| B-RAW-TASK | 0.878615 | Yes | Worse coalition AUC than raw joint by 0.015532. |
| B-LOCAL | 0.843151 | No | Both seeds fail occupation utility. |
| B-SEQ12 | 0.823777 | No | Seed 0 fails occupation utility versus Ucal. |
| B-SEQ21 | 0.846922 | No | Both seeds fail occupation utility. |
| B-RAWJ | 0.863083 | Yes | Versus strongest eligible nonself control B-DVIB, coalition benefit is 0.005529, below 0.020. |

The eligible set is B-DVIB, B-RAW-TASK and B-RAWJ under either audited interpretation of the local control cap. None meets the 0.020 coalition improvement target against its strongest eligible control. The stochastic joint arm's favorable recovery comes with a measured utility shortfall, so it is not a competitive or joint-design result. No simultaneous uncertainty intervals were computed for this stage; the reported comparisons are development point estimates, not common-criterion passes or independent confirmation.
