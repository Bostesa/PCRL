# Team C skeptical review — grouped feature selection

Status: pre-fit premise review. This review is independent of Team C implementation and does not approve a real-data fit, a result, or a novelty claim. The release boundary is coherent in synthetic checks. Real-data fit readiness still depends on S2's safe role-specific loader, S3's complete timing receipt, and S1's campaign lock. Recheck the final spec and adapter before that lock.

## Main premise and disclosure boundary

The information-changing operation is a **global, purpose-specific mask on permitted raw feature groups**, followed by retraining each task head on exactly its selected inputs. The recipient receives its selected **values**, feature names/mask, head probability vector and decision, and all transmitted metadata. The aligned coalition receives both recipients' full views. An attacker must be able to use the raw selected values directly and to ignore either recipient. A head fitted from the full 83-column input, or a clean original-teacher output appended to the release, would bypass the mask.

Let the two selected group sets be M1 and M2. Since the coalition can reconstruct the raw values in M1 union M2, any privacy gain attributable to feature withholding requires that union to omit informative distinctions. If the union covers all 83 permitted columns, the coalition has the full baseline predictor vector; weaker measured recovery would indicate attack or fitting limitations, not information erasure. A recipient's measured AUC above the coalition's is likewise a warning to strengthen the coalition reader, because it can ignore the other recipient. This statement concerns the disclosed raw features, not an information-theoretic proof that any particular reduced union is private.

The source's registered 83-column schema comprises five numeric and five categorical source fields. `education-num` recodes `education`; they must be one indivisible selection atom or an apparent removal of education leaves its content. Categorical one-hot blocks must be selected and disclosed together, including the fixed all-zero convention for an unseen category. Masks are global, not chosen per person. Record both semantic-group count and expanded-column count for each recipient and the union.

Historical `jcv` admission aggregates show `relationship` equals Husband or Wife for 45.887% of its Adult rows; `share_male` is 0.99994 for Husband and 0.00055 for Wife (`results/pcrl_joint_complete_view_method_v1/DATA_ADMISSION.json`). These are already published development aggregates, not a new role-specific estimate. A candidate that retains `relationship` in either mask should expect a direct coalition shortcut. The Team C designer has stated that C-JOINT-GROUP will exclude `relationship` from both masks before fitting. This is a testable hypothesis and may fail utility. A **common shared-removal control** with the same exclusion and comparable mask budgets is essential: if it matches C-JOINT-GROUP, the result supports simple common removal rather than a joint selection principle.

## Data and target leakage gates

The exact ordered 83-column schema must be hash-checked. Reject predictors named `sex`, `race`, `income`, `occupation`, `occupation_group`, `fnlwgt`, task-label aliases, row IDs or record keys. The source rule removed SEX, race, both task-label sources, sampling weight and IDs, then kept age, workclass, education, education-num, marital-status, relationship, capital-gain, capital-loss, hours-per-week and native-country (`jcv/data.py`; hcal `PROVENANCE_REPORT.md`). Any new feature derivation needs the same dependency audit. Training SEX may guide a global mask; no inference path may read SEX.

The inherited `osf.data.load` materializes all label arrays before it masks old assessment rows. Calling it and discarding columns afterward still makes forbidden arrays transiently accessible. Team C real-data work must use S2's independently guarded role-specific input path, and that guard must be tested **before** a fit. Do not open `HEAD_VALIDATION` labels or old assessment person-level arrays. Only the specified fitting, calibration, attack-fitting and inner-selection roles may feed their corresponding operations. The inherited loader defect is a data-access blocker, not a measured Team C failure.

## Matched comparison needed

The following controls must receive the same 83-column source schema and role partition, task-head family and training/early-stopping budget, calibration rule, attack slate and complete feature-track interface:

1. Full-feature task heads and **unconstrained** utility-only masks at matched recipient budgets; the latter may select `relationship` and is required as a strong task-only control. Also include a relationship-excluded utility-only arm to isolate selection after the fixed exclusion. The full-feature arm does not replace either matched-budget arm.
2. Separately privacy-selected masks, attacked jointly.
3. Sequential mask design in both orders, with a faithful feature-selection implementation rather than a borrowed score-release number.
4. Common shared-mask removal, including the predeclared `relationship` exclusion when comparing with C-JOINT-GROUP.
5. A compatible existing coalition-aware selector if its setting and code actually match.

Freeze a **budget vector**: selected semantic groups and expanded columns per recipient, plus unique groups and columns in the union. Shared masks cannot generally match all four recipient/union counts simultaneously; report any unmatched comparison as a capacity limitation and compare at attainable matched points. For an unconstrained task-only mask, matching recipient counts is possible even if its union differs; disclose that difference. Count mask search, head fits, calibrators, attack fits, forward passes and preprocessing. Provide exact baseline IDs and aliases. A similar score or mask is not an alias. Strongest eligible privacy control has the **lowest** complete-view SEX AUC among eligible matched alternatives, with a frozen tie rule; selecting the highest AUC would manufacture an easier benefit. The historical hcal `T*` selection orders eligible task-only arms by pair AUC ascending.

The attacker should include linear and nonlinear raw-value readers, feature-name/mask and output readers, and a defense-aware reader on the aligned coalition view. A raw-feature-only reader gives a useful lower bound on what the coalition can recover. Include a planted-leak engineering test, shuffled-SEX and constant/null controls. Scores alone are not a complete-view audit. Check attacks by view and seed; never choose a favourable seed or omit a failed configuration.

## Novelty and interpretation

Privacy-aware feature selection and subset-level dependence scores are established ideas: see [Khodadadian et al., *Information Theoretic Measures for Fairness-aware Feature Selection*](https://arxiv.org/abs/2106.00772) and [Xing et al., *Fairness-Aware Unsupervised Feature Selection*](https://arxiv.org/abs/2106.02216). Their fairness objectives/settings are not automatically this campaign's SEX-recovery objective, but they preclude claiming feature selection itself as new. [Taylor et al.](https://arxiv.org/abs/2601.21859) study sequential releases with collusion under information and distortion measures; their stated setting does not make a simple grouped mask a reproduction of their algorithm. Claim only the operation and empirical comparison actually tested.

A candidate may be a useful transparent feature-track method if it keeps both tasks' accuracy, mean NLL and Brier against both U0 and Ucal while reducing fresh SEX recovery beyond the strongest matched feature-track control. A **joint-design** claim additionally requires advantage over both sequential orders and the best non-joint design. If the shared-removal or utility-only mask explains the gain, report that favourable simple result without a joint novelty claim. Wide intervals remain unresolved; point-estimate feasibility can justify only exploratory promotion.

The draft `C-JOINT-GROUP` scorecard is a sum of **single-atom leave-one-out** SEX AUC changes, with an intersection bonus. It prelocks logistic and depth-2-tree readers and takes the larger internal-validation AUC per mask. This can detect a simple XOR clue, but the additive scorecard remains an overlap heuristic, not a direct estimate of the privacy of each candidate coalition union. Higher-order interactions and reader misspecification can still defeat it. The required complementary-clue toy law should test the stated capability. If it fails that law, do not describe the selector as solving arbitrary coalition complementarity or silently revise it after seeing results.

## Pre-fit blocking checks

- Source schema and role-specific loader are independently guarded; no forbidden array is materialized.
- Final method/controls declare exact global masks, selection atoms, head inputs, public parameters and every transmitted component.
- The relationship exclusion, education tie, candidate IDs, mask budgets, search limits and both sequential orders are locked before outcome inspection.
- The matched-budget utility-only comparator is unconstrained by the candidate's relationship exclusion; a restricted version is separately labeled as an attribution control. The strongest-control rule minimizes eligible complete-view SEX AUC.
- Complete-view attack code receives exact selected raw values and both head outputs for the coalition; the ignore-other check is present.
- Toy law includes a complementary clue (e.g. two individually uninformative pieces whose union determines SEX) and a no-complementarity case. These show implementation capability, not Adult prevalence.

If any check fails, affected real-data conclusions should be marked incomplete or invalid; retain the engineering work and resolve the premise before fitting.

## Review of current engineering draft

At this review, `prototype.py` validates the ordered source schema, coherent atoms, selected-value disclosure and recipient purpose/class widths. Its synthetic interface tests passed: `10 passed in 0.03s` with the repository environment, no real data opened. This tests the release boundary only. It does not validate mask search, training, calibration, attack strength, role-safe loading or endpoint inference. `PLANNED_UNITS.json` now charges 18 unique scorecard masks per seed (72 proxy fits per seed; 144 over two screen seeds) and 11 final-head arms; S3 must still time the full attack/calibration/verification slate before S1 locks a feasible campaign.
