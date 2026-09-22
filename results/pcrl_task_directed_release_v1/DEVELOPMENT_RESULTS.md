# Development evaluation tables

Source: EVIDENCE.json; native accepted summaries: EVALUATION_GRID.json. Selection SHA-256: `061467a38e7e40721777fb0c25f7348eb0a510541c6a6525bb8eb4f9cb2bdf70`. Grid digest: `4259cdd23b36254f2d5b2e81dfd625ef66ec6ebacc91f5ca0346e77e8fa2b143`. Every complete configuration stores its three summary/receipt hashes.

Machine-rendered descriptive aggregates. Means require all three anchors; unweighted and PWGTP metrics use the same frozen predictors. Missing diagnostics are unavailable, never imputed. These are dependent analyses of the same inherited 2018 cohort; people are not globally unseen across anchors. The anchor-specific test seal does not imply globally untouched labels. Household bounds are conditional on fitted models; they do not include retraining or validation-selection uncertainty. See DATED_SPLIT_CLARIFICATION.md.

This is historically used 2018 development evaluation. Recovery is attacker log-loss reduction, not mutual information. Adjusted selected, attribution, and stricter diagnostic verdicts remain separate in EVIDENCE.json and CLAIM_RESULTS.json.

## Execution counts

| Count | Value |
| --- | --- |
| nominal_release_anchor_units | 333 |
| nominal_map_units | 162 |
| nominal_role_audit_units | 5328 |
| scheduled_release_anchor_units | 297 |
| scheduled_map_units | 126 |
| scheduled_role_audit_units | 4752 |
| resource_unscheduled_release_anchor_units | 36 |
| accepted_audit_units | 297 |
| accepted_map_units | 126 |
| accepted_evaluation_units | 297 |
| scheduled_evaluation_units | 297 |
| incomplete_audit_units | 0 |
| incomplete_evaluation_units | 0 |
| extra_accepted_audit_units | 0 |
| accepted_resource_unscheduled_audit_units | 0 |
| registered_numerical_retry_slots | 4 |
| completed_numerical_retry_attempts | 4 |
| installed_numerical_replacements | 4 |
| retained_original_after_rejected_retry | 0 |
| preserved_original_map_versions | 4 |
| preserved_original_audit_versions | 4 |
| repeated_audit_units | 4 |
| total_accepted_map_versions | 130 |
| total_accepted_audit_versions | 301 |
| total_role_audit_units_with_repeats | 4816 |
| total_new_role_fit_units_with_repeats | 2730 |
| unique_registry_artifacts | 297 |
| unique_solution_artifacts | 126 |
| accepted_role_audit_units | 4752 |
| new_role_fit_units | 2694 |
| reused_role_audit_units | 2058 |
| mathematical_duplicate_release_count | None |

accepted receipt declarations; active nominal units and preserved numerical versions are separate; repeated-role counts include superseded completed audits; operational scheduler incidents without completed fit receipts are not scientific failures or extra fitted versions; new role-fit units are not individual learner counts; artifact hashes do not prove mathematical channel equality. Exact channel equivalence remains separately adjudicated.

## Incomplete configurations

| Configuration | Missing anchors |
| --- | --- |

## Primary sensitive recovery

| Configuration | Role | Weighting | CE(H) minus CE(arm) | CE(J) minus CE(arm) |
| --- | --- | --- | --- | --- |
| H | A/SEX | unweighted | -0 | -0.0157474765 |
| H | A/SEX | PWGTP | -0 | -0.00823625321 |
| H | A/RAC1P | unweighted | -0 | -0.00652554709 |
| H | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| H | AB/SEX | unweighted | -0 | -0.0164478536 |
| H | AB/SEX | PWGTP | -0 | -0.0105880371 |
| H | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| H | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| J | A/SEX | unweighted | 0.0157474765 | -0 |
| J | A/SEX | PWGTP | 0.00823625321 | -0 |
| J | A/RAC1P | unweighted | 0.00652554709 | -0 |
| J | A/RAC1P | PWGTP | 0.00628527494 | -0 |
| J | AB/SEX | unweighted | 0.0164478536 | -0 |
| J | AB/SEX | PWGTP | 0.0105880371 | -0 |
| J | AB/RAC1P | unweighted | 0.00550439428 | -0 |
| J | AB/RAC1P | PWGTP | 0.00642400314 | -0 |
| T0_C_0.0005_a17 | A/SEX | unweighted | 0.000426194659 | -0.0153212818 |
| T0_C_0.0005_a17 | A/SEX | PWGTP | 0.000407506853 | -0.00782874636 |
| T0_C_0.0005_a17 | A/RAC1P | unweighted | -8.97140296e-05 | -0.00661526112 |
| T0_C_0.0005_a17 | A/RAC1P | PWGTP | -0.00050126112 | -0.00678653606 |
| T0_C_0.0005_a17 | AB/SEX | unweighted | 0.00041124265 | -0.0160366109 |
| T0_C_0.0005_a17 | AB/SEX | PWGTP | 0.000340607205 | -0.0102474299 |
| T0_C_0.0005_a17 | AB/RAC1P | unweighted | -0.000319362902 | -0.00582375718 |
| T0_C_0.0005_a17 | AB/RAC1P | PWGTP | -0.000742397957 | -0.00716640109 |
| T0_C_0.002_a17 | A/SEX | unweighted | 0.00154534554 | -0.014202131 |
| T0_C_0.002_a17 | A/SEX | PWGTP | 0.00131597141 | -0.0069202818 |
| T0_C_0.002_a17 | A/RAC1P | unweighted | 0.000238006333 | -0.00628754076 |
| T0_C_0.002_a17 | A/RAC1P | PWGTP | 0.000158450831 | -0.00612682411 |
| T0_C_0.002_a17 | AB/SEX | unweighted | 0.00432883602 | -0.0121190175 |
| T0_C_0.002_a17 | AB/SEX | PWGTP | 0.00132145439 | -0.00926658275 |
| T0_C_0.002_a17 | AB/RAC1P | unweighted | 0.0014096711 | -0.00409472317 |
| T0_C_0.002_a17 | AB/RAC1P | PWGTP | 0.000916596249 | -0.00550740689 |
| T0_C_0.002_a33 | A/SEX | unweighted | 0.000774351744 | -0.0149731248 |
| T0_C_0.002_a33 | A/SEX | PWGTP | 0.00108807721 | -0.007148176 |
| T0_C_0.002_a33 | A/RAC1P | unweighted | 0.000140617599 | -0.00638492949 |
| T0_C_0.002_a33 | A/RAC1P | PWGTP | 0.000485182618 | -0.00580009232 |
| T0_C_0.002_a33 | AB/SEX | unweighted | 0.000497368967 | -0.0159504846 |
| T0_C_0.002_a33 | AB/SEX | PWGTP | -0.00144646549 | -0.0120345026 |
| T0_C_0.002_a33 | AB/RAC1P | unweighted | 0.00155523496 | -0.00394915932 |
| T0_C_0.002_a33 | AB/RAC1P | PWGTP | 0.000910297827 | -0.00551370531 |
| T0_C_0.01_a17 | A/SEX | unweighted | 0.00318969123 | -0.0125577853 |
| T0_C_0.01_a17 | A/SEX | PWGTP | 0.00260856627 | -0.00562768695 |
| T0_C_0.01_a17 | A/RAC1P | unweighted | 0.00150818767 | -0.00501735942 |
| T0_C_0.01_a17 | A/RAC1P | PWGTP | 0.00188393457 | -0.00440134037 |
| T0_C_0.01_a17 | AB/SEX | unweighted | 0.00500639275 | -0.0114414608 |
| T0_C_0.01_a17 | AB/SEX | PWGTP | 0.00414009564 | -0.00644794149 |
| T0_C_0.01_a17 | AB/RAC1P | unweighted | 0.00251640774 | -0.00298798654 |
| T0_C_0.01_a17 | AB/RAC1P | PWGTP | 0.00229371272 | -0.00413029042 |
| T0_C_0_a17 | A/SEX | unweighted | 0.00135086398 | -0.0143966125 |
| T0_C_0_a17 | A/SEX | PWGTP | -0.0012292731 | -0.00946552631 |
| T0_C_0_a17 | A/RAC1P | unweighted | 0.000764415667 | -0.00576113142 |
| T0_C_0_a17 | A/RAC1P | PWGTP | -7.03855375e-06 | -0.0062923135 |
| T0_C_0_a17 | AB/SEX | unweighted | 4.5406231e-05 | -0.0164024473 |
| T0_C_0_a17 | AB/SEX | PWGTP | 0.000203158631 | -0.0103848785 |
| T0_C_0_a17 | AB/RAC1P | unweighted | 8.23492769e-05 | -0.005422045 |
| T0_C_0_a17 | AB/RAC1P | PWGTP | 0.000343210775 | -0.00608079236 |
| T0_L_0.0005_a17 | A/SEX | unweighted | 0.000166414763 | -0.0155810617 |
| T0_L_0.0005_a17 | A/SEX | PWGTP | 0.000481872769 | -0.00775438044 |
| T0_L_0.0005_a17 | A/RAC1P | unweighted | -0.00011947922 | -0.00664502631 |
| T0_L_0.0005_a17 | A/RAC1P | PWGTP | -0.000210900317 | -0.00649617526 |
| T0_L_0.0005_a17 | AB/SEX | unweighted | -0.000207489891 | -0.0166553434 |
| T0_L_0.0005_a17 | AB/SEX | PWGTP | -0.000538516986 | -0.0111265541 |
| T0_L_0.0005_a17 | AB/RAC1P | unweighted | -0.000299640447 | -0.00580403472 |
| T0_L_0.0005_a17 | AB/RAC1P | PWGTP | -0.000507462367 | -0.0069314655 |
| T0_L_0.002_a17 | A/SEX | unweighted | 0.00114193744 | -0.0146055391 |
| T0_L_0.002_a17 | A/SEX | PWGTP | 0.000621707575 | -0.00761454564 |
| T0_L_0.002_a17 | A/RAC1P | unweighted | -8.69355124e-05 | -0.0066124826 |
| T0_L_0.002_a17 | A/RAC1P | PWGTP | -0.000130471063 | -0.00641574601 |
| T0_L_0.002_a17 | AB/SEX | unweighted | 0.00485323263 | -0.0115946209 |
| T0_L_0.002_a17 | AB/SEX | PWGTP | 0.00193009978 | -0.00865793735 |
| T0_L_0.002_a17 | AB/RAC1P | unweighted | 0.000636632379 | -0.0048677619 |
| T0_L_0.002_a17 | AB/RAC1P | PWGTP | -0.000522934593 | -0.00694693773 |
| T0_L_0.002_a33 | A/SEX | unweighted | 0.00382060154 | -0.011926875 |
| T0_L_0.002_a33 | A/SEX | PWGTP | 0.00194525909 | -0.00629099412 |
| T0_L_0.002_a33 | A/RAC1P | unweighted | 0.00043262072 | -0.00609292637 |
| T0_L_0.002_a33 | A/RAC1P | PWGTP | 0.000648225099 | -0.00563704984 |
| T0_L_0.002_a33 | AB/SEX | unweighted | 0.00163472274 | -0.0148131308 |
| T0_L_0.002_a33 | AB/SEX | PWGTP | 0.00122081568 | -0.00936722146 |
| T0_L_0.002_a33 | AB/RAC1P | unweighted | 0.000531445473 | -0.0049729488 |
| T0_L_0.002_a33 | AB/RAC1P | PWGTP | -1.66397822e-05 | -0.00644064292 |
| T0_L_0.01_a17 | A/SEX | unweighted | 0.00613566926 | -0.00961180724 |
| T0_L_0.01_a17 | A/SEX | PWGTP | 0.00577569463 | -0.00246055858 |
| T0_L_0.01_a17 | A/RAC1P | unweighted | 0.000858287196 | -0.00566725989 |
| T0_L_0.01_a17 | A/RAC1P | PWGTP | 0.00146757257 | -0.00481770237 |
| T0_L_0.01_a17 | AB/SEX | unweighted | 0.00541821079 | -0.0110296428 |
| T0_L_0.01_a17 | AB/SEX | PWGTP | 0.00438815785 | -0.00619987928 |
| T0_L_0.01_a17 | AB/RAC1P | unweighted | 0.00156831171 | -0.00393608256 |
| T0_L_0.01_a17 | AB/RAC1P | PWGTP | 0.00179759324 | -0.0046264099 |
| T0_L_0_a17 | A/SEX | unweighted | 0.000356881744 | -0.0153905948 |
| T0_L_0_a17 | A/SEX | PWGTP | 0.000245914516 | -0.0079903387 |
| T0_L_0_a17 | A/RAC1P | unweighted | 0.000816936831 | -0.00570861026 |
| T0_L_0_a17 | A/RAC1P | PWGTP | 7.15497269e-06 | -0.00627811997 |
| T0_L_0_a17 | AB/SEX | unweighted | 0.000100399144 | -0.0163474544 |
| T0_L_0_a17 | AB/SEX | PWGTP | 0.000344742461 | -0.0102432947 |
| T0_L_0_a17 | AB/RAC1P | unweighted | 8.09608414e-06 | -0.00549629819 |
| T0_L_0_a17 | AB/RAC1P | PWGTP | 0.000278465977 | -0.00614553716 |
| T0_U_unconstrained_a17 | A/SEX | unweighted | 0.00445142498 | -0.0112960515 |
| T0_U_unconstrained_a17 | A/SEX | PWGTP | 0.00539077652 | -0.00284547669 |
| T0_U_unconstrained_a17 | A/RAC1P | unweighted | -0.000228929708 | -0.0067544768 |
| T0_U_unconstrained_a17 | A/RAC1P | PWGTP | 5.68971449e-05 | -0.0062283778 |
| T0_U_unconstrained_a17 | AB/SEX | unweighted | 0.00539399739 | -0.0110538562 |
| T0_U_unconstrained_a17 | AB/SEX | PWGTP | 0.00495830632 | -0.00562973081 |
| T0_U_unconstrained_a17 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| T0_U_unconstrained_a17 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| T0_U_unconstrained_a33 | A/SEX | unweighted | 0.00687902163 | -0.00886845486 |
| T0_U_unconstrained_a33 | A/SEX | PWGTP | 0.0045980115 | -0.00363824171 |
| T0_U_unconstrained_a33 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| T0_U_unconstrained_a33 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| T0_U_unconstrained_a33 | AB/SEX | unweighted | 0.0117918105 | -0.00465604308 |
| T0_U_unconstrained_a33 | AB/SEX | PWGTP | 0.00499705446 | -0.00559098267 |
| T0_U_unconstrained_a33 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| T0_U_unconstrained_a33 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| T0_code | A/SEX | unweighted | 0.00783842939 | -0.00790904711 |
| T0_code | A/SEX | PWGTP | -7.14723779e-05 | -0.00830772559 |
| T0_code | A/RAC1P | unweighted | -0 | -0.00652554709 |
| T0_code | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| T0_code | AB/SEX | unweighted | 0.00821133878 | -0.00823651477 |
| T0_code | AB/SEX | PWGTP | 0.00272432851 | -0.00786370863 |
| T0_code | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| T0_code | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| T0_rr_0.25 | A/SEX | unweighted | -0.000362796254 | -0.0161102728 |
| T0_rr_0.25 | A/SEX | PWGTP | -0.000674562727 | -0.00891081594 |
| T0_rr_0.25 | A/RAC1P | unweighted | 0.00114378176 | -0.00538176533 |
| T0_rr_0.25 | A/RAC1P | PWGTP | 0.000284996057 | -0.00600027888 |
| T0_rr_0.25 | AB/SEX | unweighted | 0.000429363479 | -0.0160184901 |
| T0_rr_0.25 | AB/SEX | PWGTP | 0.000303665054 | -0.0102843721 |
| T0_rr_0.25 | AB/RAC1P | unweighted | -7.1269714e-05 | -0.00557566399 |
| T0_rr_0.25 | AB/RAC1P | PWGTP | -5.48330144e-05 | -0.00647883615 |
| T0_rr_0.25_a33 | A/SEX | unweighted | 0.00315735794 | -0.0125901186 |
| T0_rr_0.25_a33 | A/SEX | PWGTP | -0.00149971562 | -0.00973596883 |
| T0_rr_0.25_a33 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| T0_rr_0.25_a33 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| T0_rr_0.25_a33 | AB/SEX | unweighted | 0.00333166932 | -0.0131161842 |
| T0_rr_0.25_a33 | AB/SEX | PWGTP | 0.000286721602 | -0.0103013155 |
| T0_rr_0.25_a33 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| T0_rr_0.25_a33 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| T0_rr_0.5 | A/SEX | unweighted | 0.00311245109 | -0.0126350254 |
| T0_rr_0.5 | A/SEX | PWGTP | 0.000766542958 | -0.00746971025 |
| T0_rr_0.5 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| T0_rr_0.5 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| T0_rr_0.5 | AB/SEX | unweighted | 0.0010683897 | -0.0153794638 |
| T0_rr_0.5 | AB/SEX | PWGTP | 0.00055632712 | -0.01003171 |
| T0_rr_0.5 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| T0_rr_0.5 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| T0_rr_0.5_a33 | A/SEX | unweighted | 0.00503983533 | -0.0107076412 |
| T0_rr_0.5_a33 | A/SEX | PWGTP | 0.000326107079 | -0.00791014613 |
| T0_rr_0.5_a33 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| T0_rr_0.5_a33 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| T0_rr_0.5_a33 | AB/SEX | unweighted | 0.00738612158 | -0.00906173197 |
| T0_rr_0.5_a33 | AB/SEX | PWGTP | -0.000460535731 | -0.0110485729 |
| T0_rr_0.5_a33 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| T0_rr_0.5_a33 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| T0_rr_0.75 | A/SEX | unweighted | 0.00182055728 | -0.0139269192 |
| T0_rr_0.75 | A/SEX | PWGTP | 0.000273303031 | -0.00796295018 |
| T0_rr_0.75 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| T0_rr_0.75 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| T0_rr_0.75 | AB/SEX | unweighted | 0.00549918057 | -0.010948673 |
| T0_rr_0.75 | AB/SEX | PWGTP | 0.000413228163 | -0.010174809 |
| T0_rr_0.75 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| T0_rr_0.75 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| T0_rr_0.75_a33 | A/SEX | unweighted | 0.00820812067 | -0.00753935583 |
| T0_rr_0.75_a33 | A/SEX | PWGTP | 0.00132442457 | -0.00691182864 |
| T0_rr_0.75_a33 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| T0_rr_0.75_a33 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| T0_rr_0.75_a33 | AB/SEX | unweighted | 0.0102561757 | -0.0061916779 |
| T0_rr_0.75_a33 | AB/SEX | PWGTP | 0.0032136318 | -0.00737440533 |
| T0_rr_0.75_a33 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| T0_rr_0.75_a33 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| T0_withhold_0.25 | A/SEX | unweighted | 0.00304231294 | -0.0127051636 |
| T0_withhold_0.25 | A/SEX | PWGTP | 0.00103079375 | -0.00720545946 |
| T0_withhold_0.25 | A/RAC1P | unweighted | 0.000815238911 | -0.00571030818 |
| T0_withhold_0.25 | A/RAC1P | PWGTP | 0.00050739769 | -0.00577787725 |
| T0_withhold_0.25 | AB/SEX | unweighted | 0.00341935833 | -0.0130284952 |
| T0_withhold_0.25 | AB/SEX | PWGTP | 0.000332781486 | -0.0102552556 |
| T0_withhold_0.25 | AB/RAC1P | unweighted | 0.00107149086 | -0.00443290342 |
| T0_withhold_0.25 | AB/RAC1P | PWGTP | 0.000630189066 | -0.00579381407 |
| T0_withhold_0.25_a33 | A/SEX | unweighted | 0.00334566309 | -0.0124018134 |
| T0_withhold_0.25_a33 | A/SEX | PWGTP | 0.00154587861 | -0.00669037461 |
| T0_withhold_0.25_a33 | A/RAC1P | unweighted | 0.000459440615 | -0.00606610647 |
| T0_withhold_0.25_a33 | A/RAC1P | PWGTP | 0.000335466384 | -0.00594980856 |
| T0_withhold_0.25_a33 | AB/SEX | unweighted | 0.000802577655 | -0.0156452759 |
| T0_withhold_0.25_a33 | AB/SEX | PWGTP | 0.000761827497 | -0.00982620964 |
| T0_withhold_0.25_a33 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| T0_withhold_0.25_a33 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| T0_withhold_0.5 | A/SEX | unweighted | 0.00186998826 | -0.0138774882 |
| T0_withhold_0.5 | A/SEX | PWGTP | 8.57235325e-05 | -0.00815052968 |
| T0_withhold_0.5 | A/RAC1P | unweighted | -0.000438315282 | -0.00696386237 |
| T0_withhold_0.5 | A/RAC1P | PWGTP | -0.000373113035 | -0.00665838798 |
| T0_withhold_0.5 | AB/SEX | unweighted | 0.00254729672 | -0.0139005568 |
| T0_withhold_0.5 | AB/SEX | PWGTP | 0.00257731449 | -0.00801072264 |
| T0_withhold_0.5 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| T0_withhold_0.5 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| T0_withhold_0.5_a33 | A/SEX | unweighted | 0.00409646899 | -0.0116510075 |
| T0_withhold_0.5_a33 | A/SEX | PWGTP | -0.000910227467 | -0.00914648068 |
| T0_withhold_0.5_a33 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| T0_withhold_0.5_a33 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| T0_withhold_0.5_a33 | AB/SEX | unweighted | 0.0079401217 | -0.00850773185 |
| T0_withhold_0.5_a33 | AB/SEX | PWGTP | 0.00236229619 | -0.00822574094 |
| T0_withhold_0.5_a33 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| T0_withhold_0.5_a33 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| T0_withhold_0.75 | A/SEX | unweighted | 0.00212610122 | -0.0136213753 |
| T0_withhold_0.75 | A/SEX | PWGTP | 0.00286557063 | -0.00537068259 |
| T0_withhold_0.75 | A/RAC1P | unweighted | -0.000363980158 | -0.00688952725 |
| T0_withhold_0.75 | A/RAC1P | PWGTP | -0.000228155299 | -0.00651343024 |
| T0_withhold_0.75 | AB/SEX | unweighted | 0.00792324813 | -0.00852460542 |
| T0_withhold_0.75 | AB/SEX | PWGTP | 0.0049582887 | -0.00562974844 |
| T0_withhold_0.75 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| T0_withhold_0.75 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| T0_withhold_0.75_a33 | A/SEX | unweighted | 0.00492900102 | -0.0108184755 |
| T0_withhold_0.75_a33 | A/SEX | PWGTP | -2.56909552e-05 | -0.00826194417 |
| T0_withhold_0.75_a33 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| T0_withhold_0.75_a33 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| T0_withhold_0.75_a33 | AB/SEX | unweighted | 0.0102849181 | -0.00616293549 |
| T0_withhold_0.75_a33 | AB/SEX | PWGTP | 0.00284767029 | -0.00774036684 |
| T0_withhold_0.75_a33 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| T0_withhold_0.75_a33 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| Trisk_C_0.0005_a17 | A/SEX | unweighted | 0.000128409662 | -0.0156190668 |
| Trisk_C_0.0005_a17 | A/SEX | PWGTP | 0.000388044915 | -0.0078482083 |
| Trisk_C_0.0005_a17 | A/RAC1P | unweighted | 0.000125860941 | -0.00639968615 |
| Trisk_C_0.0005_a17 | A/RAC1P | PWGTP | -0.000137318276 | -0.00642259322 |
| Trisk_C_0.0005_a17 | AB/SEX | unweighted | 0.00213816133 | -0.0143096922 |
| Trisk_C_0.0005_a17 | AB/SEX | PWGTP | -0.00070353264 | -0.0112915698 |
| Trisk_C_0.0005_a17 | AB/RAC1P | unweighted | 0.000184742105 | -0.00531965217 |
| Trisk_C_0.0005_a17 | AB/RAC1P | PWGTP | -0.000117904553 | -0.00654190769 |
| Trisk_C_0.0005_a17_fineC | A/SEX | unweighted | 0.000517896012 | -0.0152295805 |
| Trisk_C_0.0005_a17_fineC | A/SEX | PWGTP | 0.000620761373 | -0.00761549184 |
| Trisk_C_0.0005_a17_fineC | A/RAC1P | unweighted | 5.17726811e-06 | -0.00652036982 |
| Trisk_C_0.0005_a17_fineC | A/RAC1P | PWGTP | -0.000185715796 | -0.00647099074 |
| Trisk_C_0.0005_a17_fineC | AB/SEX | unweighted | 0.00233847112 | -0.0141093824 |
| Trisk_C_0.0005_a17_fineC | AB/SEX | PWGTP | 7.22234369e-06 | -0.0105808148 |
| Trisk_C_0.0005_a17_fineC | AB/RAC1P | unweighted | 0.00024014477 | -0.00526424951 |
| Trisk_C_0.0005_a17_fineC | AB/RAC1P | PWGTP | 0.000282297936 | -0.0061417052 |
| Trisk_C_0.002_a17 | A/SEX | unweighted | 0.000804228865 | -0.0149432476 |
| Trisk_C_0.002_a17 | A/SEX | PWGTP | 0.000997561381 | -0.00723869183 |
| Trisk_C_0.002_a17 | A/RAC1P | unweighted | 0.0012875946 | -0.00523795248 |
| Trisk_C_0.002_a17 | A/RAC1P | PWGTP | 0.000620069916 | -0.00566520503 |
| Trisk_C_0.002_a17 | AB/SEX | unweighted | 0.00201338181 | -0.0144344717 |
| Trisk_C_0.002_a17 | AB/SEX | PWGTP | -0.000614633803 | -0.0112026709 |
| Trisk_C_0.002_a17 | AB/RAC1P | unweighted | 0.000604982694 | -0.00489941158 |
| Trisk_C_0.002_a17 | AB/RAC1P | PWGTP | -0.000423342942 | -0.00684734608 |
| Trisk_C_0.002_a17_fineC | A/SEX | unweighted | 0.00136757179 | -0.0143799047 |
| Trisk_C_0.002_a17_fineC | A/SEX | PWGTP | 0.0013224231 | -0.00691383012 |
| Trisk_C_0.002_a17_fineC | A/RAC1P | unweighted | -0.0001006524 | -0.00662619949 |
| Trisk_C_0.002_a17_fineC | A/RAC1P | PWGTP | 7.14791108e-05 | -0.00621379583 |
| Trisk_C_0.002_a17_fineC | AB/SEX | unweighted | 9.6708511e-05 | -0.016351145 |
| Trisk_C_0.002_a17_fineC | AB/SEX | PWGTP | 0.000214263621 | -0.0103737735 |
| Trisk_C_0.002_a17_fineC | AB/RAC1P | unweighted | -6.84462332e-05 | -0.00557284051 |
| Trisk_C_0.002_a17_fineC | AB/RAC1P | PWGTP | -0.000203166779 | -0.00662716991 |
| Trisk_C_0.002_a33 | A/SEX | unweighted | 0.00336967833 | -0.0123777982 |
| Trisk_C_0.002_a33 | A/SEX | PWGTP | 0.000793696605 | -0.00744255661 |
| Trisk_C_0.002_a33 | A/RAC1P | unweighted | 0.000759563622 | -0.00576598347 |
| Trisk_C_0.002_a33 | A/RAC1P | PWGTP | 0.000778713875 | -0.00550656107 |
| Trisk_C_0.002_a33 | AB/SEX | unweighted | 0.000254486083 | -0.0161933675 |
| Trisk_C_0.002_a33 | AB/SEX | PWGTP | -6.8363409e-05 | -0.0106564005 |
| Trisk_C_0.002_a33 | AB/RAC1P | unweighted | 0.000905022951 | -0.00459937133 |
| Trisk_C_0.002_a33 | AB/RAC1P | PWGTP | 7.08374427e-06 | -0.00641691939 |
| Trisk_C_0.01_a17 | A/SEX | unweighted | 0.00596042352 | -0.00978705298 |
| Trisk_C_0.01_a17 | A/SEX | PWGTP | 0.00522191525 | -0.00301433796 |
| Trisk_C_0.01_a17 | A/RAC1P | unweighted | 0.000822372475 | -0.00570317461 |
| Trisk_C_0.01_a17 | A/RAC1P | PWGTP | 0.00185796177 | -0.00442731318 |
| Trisk_C_0.01_a17 | AB/SEX | unweighted | 0.00745720613 | -0.00899064742 |
| Trisk_C_0.01_a17 | AB/SEX | PWGTP | 0.00524175239 | -0.00534628474 |
| Trisk_C_0.01_a17 | AB/RAC1P | unweighted | 0.00188664422 | -0.00361775006 |
| Trisk_C_0.01_a17 | AB/RAC1P | PWGTP | 0.00157890697 | -0.00484509617 |
| Trisk_C_0.01_a17_fineC | A/SEX | unweighted | 0.00421205183 | -0.0115354247 |
| Trisk_C_0.01_a17_fineC | A/SEX | PWGTP | 0.00245348735 | -0.00578276586 |
| Trisk_C_0.01_a17_fineC | A/RAC1P | unweighted | 0.00133199265 | -0.00519355443 |
| Trisk_C_0.01_a17_fineC | A/RAC1P | PWGTP | 0.00147280395 | -0.004812471 |
| Trisk_C_0.01_a17_fineC | AB/SEX | unweighted | 0.00492895648 | -0.0115188971 |
| Trisk_C_0.01_a17_fineC | AB/SEX | PWGTP | 0.0039900251 | -0.00659801204 |
| Trisk_C_0.01_a17_fineC | AB/RAC1P | unweighted | 0.000356566252 | -0.00514782802 |
| Trisk_C_0.01_a17_fineC | AB/RAC1P | PWGTP | -0.000306538833 | -0.00673054197 |
| Trisk_C_0_a17 | A/SEX | unweighted | 0.000751589556 | -0.0149958869 |
| Trisk_C_0_a17 | A/SEX | PWGTP | 0.0006984105 | -0.00753784271 |
| Trisk_C_0_a17 | A/RAC1P | unweighted | 0.00107496686 | -0.00545058023 |
| Trisk_C_0_a17 | A/RAC1P | PWGTP | 0.00023231783 | -0.00605295711 |
| Trisk_C_0_a17 | AB/SEX | unweighted | 0.000369793555 | -0.01607806 |
| Trisk_C_0_a17 | AB/SEX | PWGTP | 0.000457894121 | -0.010130143 |
| Trisk_C_0_a17 | AB/RAC1P | unweighted | 9.85884652e-05 | -0.00540580581 |
| Trisk_C_0_a17 | AB/RAC1P | PWGTP | 0.000384102874 | -0.00603990026 |
| Trisk_L_0.0005_a17 | A/SEX | unweighted | 0.00290531303 | -0.0128421635 |
| Trisk_L_0.0005_a17 | A/SEX | PWGTP | 0.000986439753 | -0.00724981346 |
| Trisk_L_0.0005_a17 | A/RAC1P | unweighted | -9.38392402e-05 | -0.00661938633 |
| Trisk_L_0.0005_a17 | A/RAC1P | PWGTP | -0.000163130032 | -0.00644840497 |
| Trisk_L_0.0005_a17 | AB/SEX | unweighted | 0.000109917342 | -0.0163379362 |
| Trisk_L_0.0005_a17 | AB/SEX | PWGTP | 0.000340517211 | -0.0102475199 |
| Trisk_L_0.0005_a17 | AB/RAC1P | unweighted | 0.000475030742 | -0.00502936353 |
| Trisk_L_0.0005_a17 | AB/RAC1P | PWGTP | 0.000140183622 | -0.00628381951 |
| Trisk_L_0.0005_a17_fineC | A/SEX | unweighted | 0.000707549818 | -0.0150399267 |
| Trisk_L_0.0005_a17_fineC | A/SEX | PWGTP | 0.000646462776 | -0.00758979044 |
| Trisk_L_0.0005_a17_fineC | A/RAC1P | unweighted | 7.6143413e-05 | -0.00644940367 |
| Trisk_L_0.0005_a17_fineC | A/RAC1P | PWGTP | -0.000308748434 | -0.00659402338 |
| Trisk_L_0.0005_a17_fineC | AB/SEX | unweighted | -0.000233691041 | -0.0166815446 |
| Trisk_L_0.0005_a17_fineC | AB/SEX | PWGTP | 0.000342269413 | -0.0102457677 |
| Trisk_L_0.0005_a17_fineC | AB/RAC1P | unweighted | -0.00015930722 | -0.0056637015 |
| Trisk_L_0.0005_a17_fineC | AB/RAC1P | PWGTP | -0.000695002348 | -0.00711900548 |
| Trisk_L_0.002_a17 | A/SEX | unweighted | 0.00118601874 | -0.0145614578 |
| Trisk_L_0.002_a17 | A/SEX | PWGTP | -0.000758442104 | -0.00899469532 |
| Trisk_L_0.002_a17 | A/RAC1P | unweighted | 0.00130947411 | -0.00521607298 |
| Trisk_L_0.002_a17 | A/RAC1P | PWGTP | 0.000340447765 | -0.00594482718 |
| Trisk_L_0.002_a17 | AB/SEX | unweighted | 0.0019276869 | -0.0145201667 |
| Trisk_L_0.002_a17 | AB/SEX | PWGTP | -0.000426072097 | -0.0110141092 |
| Trisk_L_0.002_a17 | AB/RAC1P | unweighted | 0.00129232579 | -0.00421206849 |
| Trisk_L_0.002_a17 | AB/RAC1P | PWGTP | 0.000334299577 | -0.00608970356 |
| Trisk_L_0.002_a17_fineC | A/SEX | unweighted | 0.00114968829 | -0.0145977882 |
| Trisk_L_0.002_a17_fineC | A/SEX | PWGTP | 0.00117351169 | -0.00706274152 |
| Trisk_L_0.002_a17_fineC | A/RAC1P | unweighted | 0.000913301215 | -0.00561224587 |
| Trisk_L_0.002_a17_fineC | A/RAC1P | PWGTP | 8.41600859e-05 | -0.00620111486 |
| Trisk_L_0.002_a17_fineC | AB/SEX | unweighted | 0.00324957433 | -0.0131982792 |
| Trisk_L_0.002_a17_fineC | AB/SEX | PWGTP | 0.000992771355 | -0.00959526578 |
| Trisk_L_0.002_a17_fineC | AB/RAC1P | unweighted | -0.000110326752 | -0.00561472103 |
| Trisk_L_0.002_a17_fineC | AB/RAC1P | PWGTP | -0.000550755156 | -0.00697475829 |
| Trisk_L_0.002_a33 | A/SEX | unweighted | 0.00277697602 | -0.0129705005 |
| Trisk_L_0.002_a33 | A/SEX | PWGTP | 0.00124018707 | -0.00699606615 |
| Trisk_L_0.002_a33 | A/RAC1P | unweighted | -0.000413201336 | -0.00693874842 |
| Trisk_L_0.002_a33 | A/RAC1P | PWGTP | -0.000537624277 | -0.00682289922 |
| Trisk_L_0.002_a33 | AB/SEX | unweighted | 0.00665725396 | -0.00979059959 |
| Trisk_L_0.002_a33 | AB/SEX | PWGTP | 0.00208770415 | -0.00850033298 |
| Trisk_L_0.002_a33 | AB/RAC1P | unweighted | 0.00146356052 | -0.00404083376 |
| Trisk_L_0.002_a33 | AB/RAC1P | PWGTP | 0.000570252799 | -0.00585375034 |
| Trisk_L_0.01_a17 | A/SEX | unweighted | 0.00444212233 | -0.0113053542 |
| Trisk_L_0.01_a17 | A/SEX | PWGTP | 0.00587575209 | -0.00236050112 |
| Trisk_L_0.01_a17 | A/RAC1P | unweighted | 0.000557018414 | -0.00596852867 |
| Trisk_L_0.01_a17 | A/RAC1P | PWGTP | 0.00195045593 | -0.00433481901 |
| Trisk_L_0.01_a17 | AB/SEX | unweighted | 0.0048809733 | -0.0115668803 |
| Trisk_L_0.01_a17 | AB/SEX | PWGTP | 0.00473606845 | -0.00585196868 |
| Trisk_L_0.01_a17 | AB/RAC1P | unweighted | 0.00221461776 | -0.00328977652 |
| Trisk_L_0.01_a17 | AB/RAC1P | PWGTP | 0.00194672844 | -0.00447727469 |
| Trisk_L_0.01_a17_fineC | A/SEX | unweighted | 0.00410974691 | -0.0116377296 |
| Trisk_L_0.01_a17_fineC | A/SEX | PWGTP | 0.00478753574 | -0.00344871747 |
| Trisk_L_0.01_a17_fineC | A/RAC1P | unweighted | 0.00082757836 | -0.00569796873 |
| Trisk_L_0.01_a17_fineC | A/RAC1P | PWGTP | 0.00105831539 | -0.00522695956 |
| Trisk_L_0.01_a17_fineC | AB/SEX | unweighted | 0.00609488501 | -0.0103529685 |
| Trisk_L_0.01_a17_fineC | AB/SEX | PWGTP | 0.00583570822 | -0.00475232891 |
| Trisk_L_0.01_a17_fineC | AB/RAC1P | unweighted | 0.00166015176 | -0.00384424251 |
| Trisk_L_0.01_a17_fineC | AB/RAC1P | PWGTP | 0.00130968007 | -0.00511432306 |
| Trisk_L_0_a17 | A/SEX | unweighted | 0.00389479722 | -0.0118526793 |
| Trisk_L_0_a17 | A/SEX | PWGTP | -0.00053069095 | -0.00876694416 |
| Trisk_L_0_a17 | A/RAC1P | unweighted | -0.000432535176 | -0.00695808226 |
| Trisk_L_0_a17 | A/RAC1P | PWGTP | -0.000423065618 | -0.00670834056 |
| Trisk_L_0_a17 | AB/SEX | unweighted | -0.000436046485 | -0.0168839 |
| Trisk_L_0_a17 | AB/SEX | PWGTP | -0.00017467053 | -0.0107627077 |
| Trisk_L_0_a17 | AB/RAC1P | unweighted | -0.000566264991 | -0.00607065927 |
| Trisk_L_0_a17 | AB/RAC1P | PWGTP | -0.000152501646 | -0.00657650478 |
| Trisk_U_unconstrained_a17 | A/SEX | unweighted | 0.00435713978 | -0.0113903367 |
| Trisk_U_unconstrained_a17 | A/SEX | PWGTP | 0.00357835557 | -0.00465789765 |
| Trisk_U_unconstrained_a17 | A/RAC1P | unweighted | 0.00614328703 | -0.000382260061 |
| Trisk_U_unconstrained_a17 | A/RAC1P | PWGTP | 0.00851162094 | 0.002226346 |
| Trisk_U_unconstrained_a17 | AB/SEX | unweighted | 0.00535129194 | -0.0110965616 |
| Trisk_U_unconstrained_a17 | AB/SEX | PWGTP | 0.00304626605 | -0.00754177109 |
| Trisk_U_unconstrained_a17 | AB/RAC1P | unweighted | 0.00405447137 | -0.0014499229 |
| Trisk_U_unconstrained_a17 | AB/RAC1P | PWGTP | 0.00592578627 | -0.000498216867 |
| Trisk_U_unconstrained_a33 | A/SEX | unweighted | 0.00527191898 | -0.0104755575 |
| Trisk_U_unconstrained_a33 | A/SEX | PWGTP | 0.00370664804 | -0.00452960518 |
| Trisk_U_unconstrained_a33 | A/RAC1P | unweighted | 0.00468851237 | -0.00183703472 |
| Trisk_U_unconstrained_a33 | A/RAC1P | PWGTP | 0.0070377238 | 0.000752448862 |
| Trisk_U_unconstrained_a33 | AB/SEX | unweighted | 0.00545514194 | -0.0109927116 |
| Trisk_U_unconstrained_a33 | AB/SEX | PWGTP | 0.00354077908 | -0.00704725806 |
| Trisk_U_unconstrained_a33 | AB/RAC1P | unweighted | 0.00458439504 | -0.000919999235 |
| Trisk_U_unconstrained_a33 | AB/RAC1P | PWGTP | 0.00516694504 | -0.0012570581 |
| Trisk_code | A/SEX | unweighted | 0.0129051791 | -0.0028422974 |
| Trisk_code | A/SEX | PWGTP | 0.00888289849 | 0.000646645275 |
| Trisk_code | A/RAC1P | unweighted | 0.0133765482 | 0.00685100107 |
| Trisk_code | A/RAC1P | PWGTP | 0.0103095046 | 0.00402422968 |
| Trisk_code | AB/SEX | unweighted | 0.0100023637 | -0.00644548981 |
| Trisk_code | AB/SEX | PWGTP | 0.00797251828 | -0.00261551885 |
| Trisk_code | AB/RAC1P | unweighted | 0.00611542829 | 0.000611034012 |
| Trisk_code | AB/RAC1P | PWGTP | 0.00555731814 | -0.000866684998 |
| Trisk_rr_0.25 | A/SEX | unweighted | -5.33012756e-05 | -0.0158007778 |
| Trisk_rr_0.25 | A/SEX | PWGTP | -9.07050964e-05 | -0.00832695831 |
| Trisk_rr_0.25 | A/RAC1P | unweighted | 0.00113849998 | -0.00538704711 |
| Trisk_rr_0.25 | A/RAC1P | PWGTP | 0.000402526339 | -0.0058827486 |
| Trisk_rr_0.25 | AB/SEX | unweighted | 0.000350183454 | -0.0160976701 |
| Trisk_rr_0.25 | AB/SEX | PWGTP | 0.00022517288 | -0.0103628643 |
| Trisk_rr_0.25 | AB/RAC1P | unweighted | -0.000878477615 | -0.00638287189 |
| Trisk_rr_0.25 | AB/RAC1P | PWGTP | -0.000407576741 | -0.00683157988 |
| Trisk_rr_0.25_a33 | A/SEX | unweighted | 0.00305084671 | -0.0126966298 |
| Trisk_rr_0.25_a33 | A/SEX | PWGTP | 0.00103309509 | -0.00720315812 |
| Trisk_rr_0.25_a33 | A/RAC1P | unweighted | -0.000630590663 | -0.00715613775 |
| Trisk_rr_0.25_a33 | A/RAC1P | PWGTP | -0.00037567233 | -0.00666094727 |
| Trisk_rr_0.25_a33 | AB/SEX | unweighted | -0.000157750157 | -0.0166056037 |
| Trisk_rr_0.25_a33 | AB/SEX | PWGTP | -9.60801138e-05 | -0.0106841172 |
| Trisk_rr_0.25_a33 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| Trisk_rr_0.25_a33 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| Trisk_rr_0.5 | A/SEX | unweighted | 0.00318827243 | -0.0125592041 |
| Trisk_rr_0.5 | A/SEX | PWGTP | 0.000944918615 | -0.0072913346 |
| Trisk_rr_0.5 | A/RAC1P | unweighted | 0.000865811638 | -0.00565973545 |
| Trisk_rr_0.5 | A/RAC1P | PWGTP | 0.000211013276 | -0.00607426167 |
| Trisk_rr_0.5 | AB/SEX | unweighted | 0.00332294782 | -0.0131249057 |
| Trisk_rr_0.5 | AB/SEX | PWGTP | 0.000403665892 | -0.0101843712 |
| Trisk_rr_0.5 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| Trisk_rr_0.5 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| Trisk_rr_0.5_a33 | A/SEX | unweighted | 0.00714856786 | -0.00859890864 |
| Trisk_rr_0.5_a33 | A/SEX | PWGTP | 0.000789798807 | -0.00744645441 |
| Trisk_rr_0.5_a33 | A/RAC1P | unweighted | -0.000368494625 | -0.00689404171 |
| Trisk_rr_0.5_a33 | A/RAC1P | PWGTP | 0.000379784485 | -0.00590549046 |
| Trisk_rr_0.5_a33 | AB/SEX | unweighted | 0.00637684546 | -0.0100710081 |
| Trisk_rr_0.5_a33 | AB/SEX | PWGTP | 0.00153426053 | -0.0090537766 |
| Trisk_rr_0.5_a33 | AB/RAC1P | unweighted | 0.00137108118 | -0.0041333131 |
| Trisk_rr_0.5_a33 | AB/RAC1P | PWGTP | 0.00117832653 | -0.0052456766 |
| Trisk_rr_0.75 | A/SEX | unweighted | 0.00128719014 | -0.0144602864 |
| Trisk_rr_0.75 | A/SEX | PWGTP | 1.05127322e-05 | -0.00822574048 |
| Trisk_rr_0.75 | A/RAC1P | unweighted | 0.00214781637 | -0.00437773072 |
| Trisk_rr_0.75 | A/RAC1P | PWGTP | 0.00321701198 | -0.00306826296 |
| Trisk_rr_0.75 | AB/SEX | unweighted | 0.00623845754 | -0.010209396 |
| Trisk_rr_0.75 | AB/SEX | PWGTP | 0.001380298 | -0.00920773913 |
| Trisk_rr_0.75 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| Trisk_rr_0.75 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| Trisk_rr_0.75_a33 | A/SEX | unweighted | 0.00706921267 | -0.00867826383 |
| Trisk_rr_0.75_a33 | A/SEX | PWGTP | 0.00117922129 | -0.00705703192 |
| Trisk_rr_0.75_a33 | A/RAC1P | unweighted | 0.000563992608 | -0.00596155448 |
| Trisk_rr_0.75_a33 | A/RAC1P | PWGTP | 0.00272638628 | -0.00355888866 |
| Trisk_rr_0.75_a33 | AB/SEX | unweighted | 0.00664837082 | -0.00979948273 |
| Trisk_rr_0.75_a33 | AB/SEX | PWGTP | 0.00163957598 | -0.00894846115 |
| Trisk_rr_0.75_a33 | AB/RAC1P | unweighted | 0.002260685 | -0.00324370928 |
| Trisk_rr_0.75_a33 | AB/RAC1P | PWGTP | 0.00236908029 | -0.00405492284 |
| Trisk_withhold_0.25 | A/SEX | unweighted | 0.00343281787 | -0.0123146586 |
| Trisk_withhold_0.25 | A/SEX | PWGTP | 0.00159159469 | -0.00664465852 |
| Trisk_withhold_0.25 | A/RAC1P | unweighted | 0.00106641171 | -0.00545913538 |
| Trisk_withhold_0.25 | A/RAC1P | PWGTP | 0.00148293235 | -0.00480234259 |
| Trisk_withhold_0.25 | AB/SEX | unweighted | 0.00496846016 | -0.0114793934 |
| Trisk_withhold_0.25 | AB/SEX | PWGTP | 0.00189593829 | -0.00869209884 |
| Trisk_withhold_0.25 | AB/RAC1P | unweighted | 0.0027136665 | -0.00279072777 |
| Trisk_withhold_0.25 | AB/RAC1P | PWGTP | 0.00319065502 | -0.00323334811 |
| Trisk_withhold_0.25_a33 | A/SEX | unweighted | 0.00234003316 | -0.0134074433 |
| Trisk_withhold_0.25_a33 | A/SEX | PWGTP | 7.15314103e-05 | -0.0081647218 |
| Trisk_withhold_0.25_a33 | A/RAC1P | unweighted | 0.000409654986 | -0.0061158921 |
| Trisk_withhold_0.25_a33 | A/RAC1P | PWGTP | 0.00090243316 | -0.00538284178 |
| Trisk_withhold_0.25_a33 | AB/SEX | unweighted | 0.000243476635 | -0.0162043769 |
| Trisk_withhold_0.25_a33 | AB/SEX | PWGTP | 8.18887759e-05 | -0.0105061484 |
| Trisk_withhold_0.25_a33 | AB/RAC1P | unweighted | 0.00154719516 | -0.00395719912 |
| Trisk_withhold_0.25_a33 | AB/RAC1P | PWGTP | 0.00199558535 | -0.00442841779 |
| Trisk_withhold_0.5 | A/SEX | unweighted | 0.00570725688 | -0.0100402196 |
| Trisk_withhold_0.5 | A/SEX | PWGTP | 0.00188889753 | -0.00634735568 |
| Trisk_withhold_0.5 | A/RAC1P | unweighted | 0.00151103138 | -0.00501451571 |
| Trisk_withhold_0.5 | A/RAC1P | PWGTP | 0.00301056827 | -0.00327470667 |
| Trisk_withhold_0.5 | AB/SEX | unweighted | 0.0054100046 | -0.0110378489 |
| Trisk_withhold_0.5 | AB/SEX | PWGTP | 0.000822794511 | -0.00976524262 |
| Trisk_withhold_0.5 | AB/RAC1P | unweighted | 0.00285705603 | -0.00264733824 |
| Trisk_withhold_0.5 | AB/RAC1P | PWGTP | 0.00260508738 | -0.00381891575 |
| Trisk_withhold_0.5_a33 | A/SEX | unweighted | 0.00537368643 | -0.0103737901 |
| Trisk_withhold_0.5_a33 | A/SEX | PWGTP | 0.00109297058 | -0.00714328263 |
| Trisk_withhold_0.5_a33 | A/RAC1P | unweighted | 0.00095753233 | -0.00556801476 |
| Trisk_withhold_0.5_a33 | A/RAC1P | PWGTP | 0.00231248496 | -0.00397278998 |
| Trisk_withhold_0.5_a33 | AB/SEX | unweighted | 0.00547890669 | -0.0109689469 |
| Trisk_withhold_0.5_a33 | AB/SEX | PWGTP | -2.28845673e-05 | -0.0106109217 |
| Trisk_withhold_0.5_a33 | AB/RAC1P | unweighted | 0.00248828291 | -0.00301611137 |
| Trisk_withhold_0.5_a33 | AB/RAC1P | PWGTP | 0.00278775498 | -0.00363624816 |
| Trisk_withhold_0.75 | A/SEX | unweighted | 0.00135166567 | -0.0143958108 |
| Trisk_withhold_0.75 | A/SEX | PWGTP | 0.00215145788 | -0.00608479534 |
| Trisk_withhold_0.75 | A/RAC1P | unweighted | 0.00352698869 | -0.00299855839 |
| Trisk_withhold_0.75 | A/RAC1P | PWGTP | 0.00538350948 | -0.000901765461 |
| Trisk_withhold_0.75 | AB/SEX | unweighted | 0.00590226436 | -0.0105455892 |
| Trisk_withhold_0.75 | AB/SEX | PWGTP | 0.00373523194 | -0.00685280519 |
| Trisk_withhold_0.75 | AB/RAC1P | unweighted | 0.00382678529 | -0.00167760898 |
| Trisk_withhold_0.75 | AB/RAC1P | PWGTP | 0.00368765947 | -0.00273634366 |
| Trisk_withhold_0.75_a33 | A/SEX | unweighted | 0.00290650352 | -0.012840973 |
| Trisk_withhold_0.75_a33 | A/SEX | PWGTP | 0.00108994219 | -0.00714631102 |
| Trisk_withhold_0.75_a33 | A/RAC1P | unweighted | 0.0031886291 | -0.00333691798 |
| Trisk_withhold_0.75_a33 | A/RAC1P | PWGTP | 0.00490887884 | -0.00137639611 |
| Trisk_withhold_0.75_a33 | AB/SEX | unweighted | 0.00678749889 | -0.00966035467 |
| Trisk_withhold_0.75_a33 | AB/SEX | PWGTP | 0.00260095928 | -0.00798707785 |
| Trisk_withhold_0.75_a33 | AB/RAC1P | unweighted | 0.00298152628 | -0.002522868 |
| Trisk_withhold_0.75_a33 | AB/RAC1P | PWGTP | 0.00416530824 | -0.00225869489 |
| Ttask_C_0.0005_a17 | A/SEX | unweighted | 0.000544348825 | -0.0152031277 |
| Ttask_C_0.0005_a17 | A/SEX | PWGTP | 8.70295271e-05 | -0.00814922369 |
| Ttask_C_0.0005_a17 | A/RAC1P | unweighted | 0.000473121213 | -0.00605242587 |
| Ttask_C_0.0005_a17 | A/RAC1P | PWGTP | -0.000397343819 | -0.00668261876 |
| Ttask_C_0.0005_a17 | AB/SEX | unweighted | -8.25126582e-05 | -0.0165303662 |
| Ttask_C_0.0005_a17 | AB/SEX | PWGTP | -3.64196213e-05 | -0.0106244568 |
| Ttask_C_0.0005_a17 | AB/RAC1P | unweighted | 9.1861456e-05 | -0.00541253282 |
| Ttask_C_0.0005_a17 | AB/RAC1P | PWGTP | -0.000369351347 | -0.00679335448 |
| Ttask_C_0.002_a17 | A/SEX | unweighted | 0.00247055181 | -0.0132769247 |
| Ttask_C_0.002_a17 | A/SEX | PWGTP | -0.000437495988 | -0.0086737492 |
| Ttask_C_0.002_a17 | A/RAC1P | unweighted | 0.000176071012 | -0.00634947608 |
| Ttask_C_0.002_a17 | A/RAC1P | PWGTP | -0.000102781123 | -0.00638805607 |
| Ttask_C_0.002_a17 | AB/SEX | unweighted | 0.000538581667 | -0.0159092719 |
| Ttask_C_0.002_a17 | AB/SEX | PWGTP | 0.000504217875 | -0.0100838193 |
| Ttask_C_0.002_a17 | AB/RAC1P | unweighted | 0.0013685285 | -0.00413586577 |
| Ttask_C_0.002_a17 | AB/RAC1P | PWGTP | 0.00106192835 | -0.00536207478 |
| Ttask_C_0.002_a33 | A/SEX | unweighted | 0.00381936007 | -0.0119281164 |
| Ttask_C_0.002_a33 | A/SEX | PWGTP | -0.00105521601 | -0.00929146922 |
| Ttask_C_0.002_a33 | A/RAC1P | unweighted | -0.000195867748 | -0.00672141484 |
| Ttask_C_0.002_a33 | A/RAC1P | PWGTP | 3.45532867e-05 | -0.00625072166 |
| Ttask_C_0.002_a33 | AB/SEX | unweighted | 0.000967163112 | -0.0154806904 |
| Ttask_C_0.002_a33 | AB/SEX | PWGTP | 0.000527998969 | -0.0100600382 |
| Ttask_C_0.002_a33 | AB/RAC1P | unweighted | 0.00223569466 | -0.00326869962 |
| Ttask_C_0.002_a33 | AB/RAC1P | PWGTP | 0.00106608691 | -0.00535791622 |
| Ttask_C_0.01_a17 | A/SEX | unweighted | 0.00531445743 | -0.0104330191 |
| Ttask_C_0.01_a17 | A/SEX | PWGTP | 0.00270129902 | -0.00553495419 |
| Ttask_C_0.01_a17 | A/RAC1P | unweighted | -0.000722771413 | -0.0072483185 |
| Ttask_C_0.01_a17 | A/RAC1P | PWGTP | 0.000920441337 | -0.00536483361 |
| Ttask_C_0.01_a17 | AB/SEX | unweighted | 0.00608801469 | -0.0103598389 |
| Ttask_C_0.01_a17 | AB/SEX | PWGTP | 0.00302963697 | -0.00755840016 |
| Ttask_C_0.01_a17 | AB/RAC1P | unweighted | 0.0015990908 | -0.00390530348 |
| Ttask_C_0.01_a17 | AB/RAC1P | PWGTP | 0.0016446874 | -0.00477931573 |
| Ttask_C_0_a17 | A/SEX | unweighted | 0.000699387926 | -0.0150480886 |
| Ttask_C_0_a17 | A/SEX | PWGTP | 0.000718963597 | -0.00751728962 |
| Ttask_C_0_a17 | A/RAC1P | unweighted | 0.00107590233 | -0.00544964476 |
| Ttask_C_0_a17 | A/RAC1P | PWGTP | 0.000227675841 | -0.0060575991 |
| Ttask_C_0_a17 | AB/SEX | unweighted | 0.00033220186 | -0.0161156517 |
| Ttask_C_0_a17 | AB/SEX | PWGTP | 0.000476855607 | -0.0101111815 |
| Ttask_C_0_a17 | AB/RAC1P | unweighted | -6.92938184e-05 | -0.00557368809 |
| Ttask_C_0_a17 | AB/RAC1P | PWGTP | 0.000290090576 | -0.00613391256 |
| Ttask_L_0.0005_a17 | A/SEX | unweighted | 6.6013349e-07 | -0.0157468164 |
| Ttask_L_0.0005_a17 | A/SEX | PWGTP | -0.000713837571 | -0.00895009078 |
| Ttask_L_0.0005_a17 | A/RAC1P | unweighted | -0.000215392392 | -0.00674093948 |
| Ttask_L_0.0005_a17 | A/RAC1P | PWGTP | -0.000233407342 | -0.00651868228 |
| Ttask_L_0.0005_a17 | AB/SEX | unweighted | 0.00319943682 | -0.0132484167 |
| Ttask_L_0.0005_a17 | AB/SEX | PWGTP | 0.000762192557 | -0.00982584457 |
| Ttask_L_0.0005_a17 | AB/RAC1P | unweighted | 0.000582050756 | -0.00492234352 |
| Ttask_L_0.0005_a17 | AB/RAC1P | PWGTP | 0.000399051078 | -0.00602495206 |
| Ttask_L_0.002_a17 | A/SEX | unweighted | 0.00157193025 | -0.0141755463 |
| Ttask_L_0.002_a17 | A/SEX | PWGTP | 0.00137412882 | -0.00686212439 |
| Ttask_L_0.002_a17 | A/RAC1P | unweighted | -0.000333358691 | -0.00685890578 |
| Ttask_L_0.002_a17 | A/RAC1P | PWGTP | 0.000450343974 | -0.00583493097 |
| Ttask_L_0.002_a17 | AB/SEX | unweighted | 0.000401129542 | -0.016046724 |
| Ttask_L_0.002_a17 | AB/SEX | PWGTP | -0.00019730188 | -0.010785339 |
| Ttask_L_0.002_a17 | AB/RAC1P | unweighted | 0.00104005734 | -0.00446433694 |
| Ttask_L_0.002_a17 | AB/RAC1P | PWGTP | 0.000922910909 | -0.00550109223 |
| Ttask_L_0.002_a33 | A/SEX | unweighted | 0.0060295416 | -0.0097179349 |
| Ttask_L_0.002_a33 | A/SEX | PWGTP | 0.000533058679 | -0.00770319453 |
| Ttask_L_0.002_a33 | A/RAC1P | unweighted | -0.000287347557 | -0.00681289464 |
| Ttask_L_0.002_a33 | A/RAC1P | PWGTP | 0.000169468276 | -0.00611580667 |
| Ttask_L_0.002_a33 | AB/SEX | unweighted | 0.00216989201 | -0.0142779615 |
| Ttask_L_0.002_a33 | AB/SEX | PWGTP | 0.000517533246 | -0.0100705039 |
| Ttask_L_0.002_a33 | AB/RAC1P | unweighted | 0.00146497837 | -0.00403941591 |
| Ttask_L_0.002_a33 | AB/RAC1P | PWGTP | 0.000847638275 | -0.00557636486 |
| Ttask_L_0.01_a17 | A/SEX | unweighted | 0.00245608047 | -0.013291396 |
| Ttask_L_0.01_a17 | A/SEX | PWGTP | 0.00201461065 | -0.00622164256 |
| Ttask_L_0.01_a17 | A/RAC1P | unweighted | -0.000129827663 | -0.00665537475 |
| Ttask_L_0.01_a17 | A/RAC1P | PWGTP | 0.000357482726 | -0.00592779222 |
| Ttask_L_0.01_a17 | AB/SEX | unweighted | 0.00455242353 | -0.01189543 |
| Ttask_L_0.01_a17 | AB/SEX | PWGTP | 0.00355876659 | -0.00702927054 |
| Ttask_L_0.01_a17 | AB/RAC1P | unweighted | 0.00141691916 | -0.00408747511 |
| Ttask_L_0.01_a17 | AB/RAC1P | PWGTP | 0.000936961055 | -0.00548704208 |
| Ttask_L_0_a17 | A/SEX | unweighted | 0.000496155924 | -0.0152513206 |
| Ttask_L_0_a17 | A/SEX | PWGTP | 0.000225463282 | -0.00801078993 |
| Ttask_L_0_a17 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| Ttask_L_0_a17 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| Ttask_L_0_a17 | AB/SEX | unweighted | 0.000465872617 | -0.0159819809 |
| Ttask_L_0_a17 | AB/SEX | PWGTP | 0.000331225646 | -0.0102568115 |
| Ttask_L_0_a17 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| Ttask_L_0_a17 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| Ttask_U_unconstrained_a17 | A/SEX | unweighted | 0.00610323812 | -0.00964423837 |
| Ttask_U_unconstrained_a17 | A/SEX | PWGTP | 0.00278982518 | -0.00544642803 |
| Ttask_U_unconstrained_a17 | A/RAC1P | unweighted | -0.000402146798 | -0.00692769389 |
| Ttask_U_unconstrained_a17 | A/RAC1P | PWGTP | 0.000925358251 | -0.00535991669 |
| Ttask_U_unconstrained_a17 | AB/SEX | unweighted | 0.00510313969 | -0.0113447139 |
| Ttask_U_unconstrained_a17 | AB/SEX | PWGTP | 0.00169200364 | -0.0088960335 |
| Ttask_U_unconstrained_a17 | AB/RAC1P | unweighted | 0.00134305875 | -0.00416133552 |
| Ttask_U_unconstrained_a17 | AB/RAC1P | PWGTP | 0.00107674985 | -0.00534725328 |
| Ttask_U_unconstrained_a33 | A/SEX | unweighted | 0.00179743777 | -0.0139500387 |
| Ttask_U_unconstrained_a33 | A/SEX | PWGTP | 0.00046319161 | -0.0077730616 |
| Ttask_U_unconstrained_a33 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| Ttask_U_unconstrained_a33 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| Ttask_U_unconstrained_a33 | AB/SEX | unweighted | 0.00250541846 | -0.0139424351 |
| Ttask_U_unconstrained_a33 | AB/SEX | PWGTP | 0.00177338783 | -0.0088146493 |
| Ttask_U_unconstrained_a33 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| Ttask_U_unconstrained_a33 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| Ttask_code | A/SEX | unweighted | 0.00173224122 | -0.0140152353 |
| Ttask_code | A/SEX | PWGTP | -0.000224960873 | -0.00846121409 |
| Ttask_code | A/RAC1P | unweighted | -0 | -0.00652554709 |
| Ttask_code | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| Ttask_code | AB/SEX | unweighted | 0.00127369728 | -0.0151741563 |
| Ttask_code | AB/SEX | PWGTP | -0.000734676268 | -0.0113227134 |
| Ttask_code | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| Ttask_code | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| Ttask_rr_0.25 | A/SEX | unweighted | 0.000533850893 | -0.0152136256 |
| Ttask_rr_0.25 | A/SEX | PWGTP | 0.000593391762 | -0.00764286145 |
| Ttask_rr_0.25 | A/RAC1P | unweighted | 0.000107112126 | -0.00641843496 |
| Ttask_rr_0.25 | A/RAC1P | PWGTP | -0.000239047711 | -0.00652432265 |
| Ttask_rr_0.25 | AB/SEX | unweighted | 0.000218084997 | -0.0162297686 |
| Ttask_rr_0.25 | AB/SEX | PWGTP | 0.000349052695 | -0.0102389844 |
| Ttask_rr_0.25 | AB/RAC1P | unweighted | -0.000273764745 | -0.00577815902 |
| Ttask_rr_0.25 | AB/RAC1P | PWGTP | -0.000334760027 | -0.00675876316 |
| Ttask_rr_0.25_a33 | A/SEX | unweighted | 0.000746828606 | -0.0150006479 |
| Ttask_rr_0.25_a33 | A/SEX | PWGTP | -0.00166792749 | -0.0099041807 |
| Ttask_rr_0.25_a33 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| Ttask_rr_0.25_a33 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| Ttask_rr_0.25_a33 | AB/SEX | unweighted | 0.00339494702 | -0.0130529065 |
| Ttask_rr_0.25_a33 | AB/SEX | PWGTP | 0.000662509003 | -0.00992552813 |
| Ttask_rr_0.25_a33 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| Ttask_rr_0.25_a33 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| Ttask_rr_0.5 | A/SEX | unweighted | 0.00291849744 | -0.0128289791 |
| Ttask_rr_0.5 | A/SEX | PWGTP | 0.000469567617 | -0.0077666856 |
| Ttask_rr_0.5 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| Ttask_rr_0.5 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| Ttask_rr_0.5 | AB/SEX | unweighted | -1.5254188e-06 | -0.016449379 |
| Ttask_rr_0.5 | AB/SEX | PWGTP | -0.000205892155 | -0.0107939293 |
| Ttask_rr_0.5 | AB/RAC1P | unweighted | 0.000513161407 | -0.00499123287 |
| Ttask_rr_0.5 | AB/RAC1P | PWGTP | 4.45996206e-05 | -0.00637940351 |
| Ttask_rr_0.5_a33 | A/SEX | unweighted | 0.00504727451 | -0.010700202 |
| Ttask_rr_0.5_a33 | A/SEX | PWGTP | 0.000984886315 | -0.0072513669 |
| Ttask_rr_0.5_a33 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| Ttask_rr_0.5_a33 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| Ttask_rr_0.5_a33 | AB/SEX | unweighted | 0.00429159919 | -0.0121562544 |
| Ttask_rr_0.5_a33 | AB/SEX | PWGTP | 0.00149500256 | -0.00909303457 |
| Ttask_rr_0.5_a33 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| Ttask_rr_0.5_a33 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| Ttask_rr_0.75 | A/SEX | unweighted | 0.000134564106 | -0.0156129124 |
| Ttask_rr_0.75 | A/SEX | PWGTP | 0.000389809102 | -0.00784644411 |
| Ttask_rr_0.75 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| Ttask_rr_0.75 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| Ttask_rr_0.75 | AB/SEX | unweighted | 0.0014006656 | -0.0150471879 |
| Ttask_rr_0.75 | AB/SEX | PWGTP | 0.000561246286 | -0.0100267908 |
| Ttask_rr_0.75 | AB/RAC1P | unweighted | 0.000724533639 | -0.00477986064 |
| Ttask_rr_0.75 | AB/RAC1P | PWGTP | 0.000364572186 | -0.00605943095 |
| Ttask_rr_0.75_a33 | A/SEX | unweighted | 0.00312030998 | -0.0126271665 |
| Ttask_rr_0.75_a33 | A/SEX | PWGTP | 0.00103055104 | -0.00720570217 |
| Ttask_rr_0.75_a33 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| Ttask_rr_0.75_a33 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| Ttask_rr_0.75_a33 | AB/SEX | unweighted | 0.00141629824 | -0.0150315553 |
| Ttask_rr_0.75_a33 | AB/SEX | PWGTP | -0.000976366204 | -0.0115644033 |
| Ttask_rr_0.75_a33 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| Ttask_rr_0.75_a33 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| Ttask_withhold_0.25 | A/SEX | unweighted | 0.0024060757 | -0.0133414008 |
| Ttask_withhold_0.25 | A/SEX | PWGTP | 0.000406535036 | -0.00782971818 |
| Ttask_withhold_0.25 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| Ttask_withhold_0.25 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| Ttask_withhold_0.25 | AB/SEX | unweighted | 0.00334947758 | -0.013098376 |
| Ttask_withhold_0.25 | AB/SEX | PWGTP | 0.00104073535 | -0.00954730178 |
| Ttask_withhold_0.25 | AB/RAC1P | unweighted | 0.000331499255 | -0.00517289502 |
| Ttask_withhold_0.25 | AB/RAC1P | PWGTP | 0.000174475518 | -0.00624952762 |
| Ttask_withhold_0.25_a33 | A/SEX | unweighted | 0.00293315906 | -0.0128143174 |
| Ttask_withhold_0.25_a33 | A/SEX | PWGTP | 0.000495825991 | -0.00774042722 |
| Ttask_withhold_0.25_a33 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| Ttask_withhold_0.25_a33 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| Ttask_withhold_0.25_a33 | AB/SEX | unweighted | 0.000501375072 | -0.0159464785 |
| Ttask_withhold_0.25_a33 | AB/SEX | PWGTP | 0.00042848634 | -0.0101595508 |
| Ttask_withhold_0.25_a33 | AB/RAC1P | unweighted | -0.000114435656 | -0.00561882993 |
| Ttask_withhold_0.25_a33 | AB/RAC1P | PWGTP | 9.21301612e-05 | -0.00633187297 |
| Ttask_withhold_0.5 | A/SEX | unweighted | 0.00320518263 | -0.0125422939 |
| Ttask_withhold_0.5 | A/SEX | PWGTP | 0.00107840042 | -0.0071578528 |
| Ttask_withhold_0.5 | A/RAC1P | unweighted | -0.000420106201 | -0.00694565329 |
| Ttask_withhold_0.5 | A/RAC1P | PWGTP | -0.00011362136 | -0.0063988963 |
| Ttask_withhold_0.5 | AB/SEX | unweighted | 0.00130757461 | -0.0151402789 |
| Ttask_withhold_0.5 | AB/SEX | PWGTP | 0.000865546789 | -0.00972249034 |
| Ttask_withhold_0.5 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| Ttask_withhold_0.5 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| Ttask_withhold_0.5_a33 | A/SEX | unweighted | 0.000118287877 | -0.0156291886 |
| Ttask_withhold_0.5_a33 | A/SEX | PWGTP | -0.000756316616 | -0.00899256983 |
| Ttask_withhold_0.5_a33 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| Ttask_withhold_0.5_a33 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| Ttask_withhold_0.5_a33 | AB/SEX | unweighted | 0.00459714848 | -0.0118507051 |
| Ttask_withhold_0.5_a33 | AB/SEX | PWGTP | 0.00171332486 | -0.00887471227 |
| Ttask_withhold_0.5_a33 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| Ttask_withhold_0.5_a33 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| Ttask_withhold_0.75 | A/SEX | unweighted | 0.00110201925 | -0.0146454573 |
| Ttask_withhold_0.75 | A/SEX | PWGTP | 0.00118483719 | -0.00705141602 |
| Ttask_withhold_0.75 | A/RAC1P | unweighted | -0.00059936605 | -0.00712491314 |
| Ttask_withhold_0.75 | A/RAC1P | PWGTP | -3.90910705e-05 | -0.00632436601 |
| Ttask_withhold_0.75 | AB/SEX | unweighted | 0.00473274181 | -0.0117151117 |
| Ttask_withhold_0.75 | AB/SEX | PWGTP | 0.00199279196 | -0.00859524517 |
| Ttask_withhold_0.75 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| Ttask_withhold_0.75 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| Ttask_withhold_0.75_a33 | A/SEX | unweighted | 0.00379863994 | -0.0119488366 |
| Ttask_withhold_0.75_a33 | A/SEX | PWGTP | 0.00119201061 | -0.0070442426 |
| Ttask_withhold_0.75_a33 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| Ttask_withhold_0.75_a33 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| Ttask_withhold_0.75_a33 | AB/SEX | unweighted | 0.000234233128 | -0.0162136204 |
| Ttask_withhold_0.75_a33 | AB/SEX | PWGTP | 6.2083776e-06 | -0.0105818288 |
| Ttask_withhold_0.75_a33 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| Ttask_withhold_0.75_a33 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| constant_best | A/SEX | unweighted | -0 | -0.0157474765 |
| constant_best | A/SEX | PWGTP | -0 | -0.00823625321 |
| constant_best | A/RAC1P | unweighted | -0 | -0.00652554709 |
| constant_best | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| constant_best | AB/SEX | unweighted | -0 | -0.0164478536 |
| constant_best | AB/SEX | PWGTP | -0 | -0.0105880371 |
| constant_best | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| constant_best | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| constant_best_a33 | A/SEX | unweighted | -0 | -0.0157474765 |
| constant_best_a33 | A/SEX | PWGTP | -0 | -0.00823625321 |
| constant_best_a33 | A/RAC1P | unweighted | -0 | -0.00652554709 |
| constant_best_a33 | A/RAC1P | PWGTP | -0 | -0.00628527494 |
| constant_best_a33 | AB/SEX | unweighted | -0 | -0.0164478536 |
| constant_best_a33 | AB/SEX | PWGTP | -0 | -0.0105880371 |
| constant_best_a33 | AB/RAC1P | unweighted | -0 | -0.00550439428 |
| constant_best_a33 | AB/RAC1P | PWGTP | -0 | -0.00642400314 |
| continuous_task | A/SEX | unweighted | 0.00870480554 | -0.00704267096 |
| continuous_task | A/SEX | PWGTP | 0.00946588195 | 0.00122962874 |
| continuous_task | A/RAC1P | unweighted | 0.0045018093 | -0.00202373779 |
| continuous_task | A/RAC1P | PWGTP | 0.00612324602 | -0.000162028918 |
| continuous_task | AB/SEX | unweighted | 0.0108474001 | -0.00560045341 |
| continuous_task | AB/SEX | PWGTP | 0.00962020972 | -0.000967827411 |
| continuous_task | AB/RAC1P | unweighted | 0.0055855006 | 8.11063288e-05 |
| continuous_task | AB/RAC1P | PWGTP | 0.00392911137 | -0.00249489177 |
| independent_token | A/SEX | unweighted | 0.00076558537 | -0.0149818911 |
| independent_token | A/SEX | PWGTP | 0.000662007352 | -0.00757424586 |
| independent_token | A/RAC1P | unweighted | 0.000598482508 | -0.00592706458 |
| independent_token | A/RAC1P | PWGTP | -0.000360469116 | -0.00664574406 |
| independent_token | AB/SEX | unweighted | 0.000326047499 | -0.0161218061 |
| independent_token | AB/SEX | PWGTP | 0.000471337066 | -0.0101167001 |
| independent_token | AB/RAC1P | unweighted | -0.000215519743 | -0.00571991402 |
| independent_token | AB/RAC1P | PWGTP | -0.000115877014 | -0.00653988015 |
| independent_token_a33 | A/SEX | unweighted | 0.00330247762 | -0.0124449989 |
| independent_token_a33 | A/SEX | PWGTP | -0.00112061799 | -0.0093568712 |
| independent_token_a33 | A/RAC1P | unweighted | -0.000444456099 | -0.00697000319 |
| independent_token_a33 | A/RAC1P | PWGTP | -0.000639018738 | -0.00692429368 |
| independent_token_a33 | AB/SEX | unweighted | 0.00396519884 | -0.0124826547 |
| independent_token_a33 | AB/SEX | PWGTP | 0.00110545308 | -0.00948258405 |
| independent_token_a33 | AB/RAC1P | unweighted | -0.000137494954 | -0.00564188923 |
| independent_token_a33 | AB/RAC1P | PWGTP | 0.00039032658 | -0.00603367655 |
| leace_A0 | A/SEX | unweighted | 0.00679743353 | -0.00895004297 |
| leace_A0 | A/SEX | PWGTP | 0.0068397881 | -0.00139646511 |
| leace_A0 | A/RAC1P | unweighted | 0.0190871163 | 0.0125615692 |
| leace_A0 | A/RAC1P | PWGTP | 0.0189047318 | 0.0126194569 |
| leace_A0 | AB/SEX | unweighted | 0.00744748424 | -0.00900036931 |
| leace_A0 | AB/SEX | PWGTP | 0.00707685213 | -0.003511185 |
| leace_A0 | AB/RAC1P | unweighted | 0.015453527 | 0.00994913277 |
| leace_A0 | AB/RAC1P | PWGTP | 0.0154602951 | 0.00903629195 |
| leace_supervised | A/SEX | unweighted | 0.0338886445 | 0.018141168 |
| leace_supervised | A/SEX | PWGTP | 0.02792357 | 0.0196873168 |
| leace_supervised | A/RAC1P | unweighted | 0.0571599413 | 0.0506343942 |
| leace_supervised | A/RAC1P | PWGTP | 0.0501372945 | 0.0438520196 |
| leace_supervised | AB/SEX | unweighted | 0.0315861128 | 0.0151382592 |
| leace_supervised | AB/SEX | PWGTP | 0.026318477 | 0.0157304399 |
| leace_supervised | AB/RAC1P | unweighted | 0.0482777366 | 0.0427733423 |
| leace_supervised | AB/RAC1P | PWGTP | 0.0420755788 | 0.0356515756 |
| leace_supervised_mechanism40 | A/SEX | unweighted | 0.0282906677 | 0.0125431912 |
| leace_supervised_mechanism40 | A/SEX | PWGTP | 0.0183964687 | 0.0101602155 |
| leace_supervised_mechanism40 | A/RAC1P | unweighted | 0.0568519768 | 0.0503264297 |
| leace_supervised_mechanism40 | A/RAC1P | PWGTP | 0.0496908079 | 0.0434055329 |
| leace_supervised_mechanism40 | AB/SEX | unweighted | 0.0283131622 | 0.0118653087 |
| leace_supervised_mechanism40 | AB/SEX | PWGTP | 0.0221963263 | 0.0116082891 |
| leace_supervised_mechanism40 | AB/RAC1P | unweighted | 0.0480278645 | 0.0425234703 |
| leace_supervised_mechanism40 | AB/RAC1P | PWGTP | 0.0426075262 | 0.0361835231 |
| leace_supervised_union88 | A/SEX | unweighted | 0.029322973 | 0.0135754965 |
| leace_supervised_union88 | A/SEX | PWGTP | 0.0208780632 | 0.01264181 |
| leace_supervised_union88 | A/RAC1P | unweighted | 0.0541661758 | 0.0476406287 |
| leace_supervised_union88 | A/RAC1P | PWGTP | 0.0508539624 | 0.0445686875 |
| leace_supervised_union88 | AB/SEX | unweighted | 0.0274023625 | 0.010954509 |
| leace_supervised_union88 | AB/SEX | PWGTP | 0.022092705 | 0.0115046679 |
| leace_supervised_union88 | AB/RAC1P | unweighted | 0.0453968349 | 0.0398924406 |
| leace_supervised_union88 | AB/RAC1P | PWGTP | 0.0429695738 | 0.0365455707 |
| optnet16_C1 | A/SEX | unweighted | 0.0133670166 | -0.00238045991 |
| optnet16_C1 | A/SEX | PWGTP | 0.0109978505 | 0.00276159733 |
| optnet16_C1 | A/RAC1P | unweighted | 0.028611726 | 0.022086179 |
| optnet16_C1 | A/RAC1P | PWGTP | 0.0256551641 | 0.0193698891 |
| optnet16_C1 | AB/SEX | unweighted | 0.0116232521 | -0.00482460142 |
| optnet16_C1 | AB/SEX | PWGTP | 0.00950422713 | -0.00108381 |
| optnet16_C1 | AB/RAC1P | unweighted | 0.0279168896 | 0.0224124954 |
| optnet16_C1 | AB/RAC1P | PWGTP | 0.024750793 | 0.0183267899 |
| optnet16_L1 | A/SEX | unweighted | 0.0226959329 | 0.00694845645 |
| optnet16_L1 | A/SEX | PWGTP | 0.0176815768 | 0.00944532361 |
| optnet16_L1 | A/RAC1P | unweighted | 0.0344429694 | 0.0279174223 |
| optnet16_L1 | A/RAC1P | PWGTP | 0.0340152377 | 0.0277299627 |
| optnet16_L1 | AB/SEX | unweighted | 0.0183544147 | 0.00190656113 |
| optnet16_L1 | AB/SEX | PWGTP | 0.0158723711 | 0.00528433396 |
| optnet16_L1 | AB/RAC1P | unweighted | 0.0363097221 | 0.0308053278 |
| optnet16_L1 | AB/RAC1P | PWGTP | 0.0337925753 | 0.0273685721 |
| optnet16_L2 | A/SEX | unweighted | 0.0145518688 | -0.00119560769 |
| optnet16_L2 | A/SEX | PWGTP | 0.00948030454 | 0.00124405133 |
| optnet16_L2 | A/RAC1P | unweighted | 0.0269237471 | 0.0203982 |
| optnet16_L2 | A/RAC1P | PWGTP | 0.0227131222 | 0.0164278473 |
| optnet16_L2 | AB/SEX | unweighted | 0.0130767719 | -0.00337108165 |
| optnet16_L2 | AB/SEX | PWGTP | 0.00933543163 | -0.0012526055 |
| optnet16_L2 | AB/RAC1P | unweighted | 0.0236010037 | 0.0180966095 |
| optnet16_L2 | AB/RAC1P | PWGTP | 0.0194634668 | 0.0130394637 |
| splince_A0 | A/SEX | unweighted | 0.00397834248 | -0.011769134 |
| splince_A0 | A/SEX | PWGTP | 0.00303082776 | -0.00520542546 |
| splince_A0 | A/RAC1P | unweighted | 0.00542623207 | -0.00109931502 |
| splince_A0 | A/RAC1P | PWGTP | 0.00309288957 | -0.00319238537 |
| splince_A0 | AB/SEX | unweighted | 0.00371121595 | -0.0127366376 |
| splince_A0 | AB/SEX | PWGTP | 0.00334349048 | -0.00724454665 |
| splince_A0 | AB/RAC1P | unweighted | 0.0117325064 | 0.00622811214 |
| splince_A0 | AB/RAC1P | PWGTP | 0.0104233547 | 0.00399935152 |
| splince_supervised | A/SEX | unweighted | 0.0341566928 | 0.0184092163 |
| splince_supervised | A/SEX | PWGTP | 0.0256170338 | 0.0173807806 |
| splince_supervised | A/RAC1P | unweighted | 0.0568680394 | 0.0503424923 |
| splince_supervised | A/RAC1P | PWGTP | 0.0527121083 | 0.0464268333 |
| splince_supervised | AB/SEX | unweighted | 0.0326126649 | 0.0161648113 |
| splince_supervised | AB/SEX | PWGTP | 0.02580108 | 0.0152130429 |
| splince_supervised | AB/RAC1P | unweighted | 0.0500285793 | 0.044524185 |
| splince_supervised | AB/RAC1P | PWGTP | 0.0456262007 | 0.0392021976 |
| splince_supervised_mechanism40 | A/SEX | unweighted | 0.0272827974 | 0.0115353209 |
| splince_supervised_mechanism40 | A/SEX | PWGTP | 0.0186465049 | 0.0104102516 |
| splince_supervised_mechanism40 | A/RAC1P | unweighted | 0.0569451753 | 0.0504196282 |
| splince_supervised_mechanism40 | A/RAC1P | PWGTP | 0.0490240481 | 0.0427387731 |
| splince_supervised_mechanism40 | AB/SEX | unweighted | 0.0269062548 | 0.0104584013 |
| splince_supervised_mechanism40 | AB/SEX | PWGTP | 0.019937812 | 0.00934977483 |
| splince_supervised_mechanism40 | AB/RAC1P | unweighted | 0.0476781453 | 0.042173751 |
| splince_supervised_mechanism40 | AB/RAC1P | PWGTP | 0.041997564 | 0.0355735609 |
| splince_supervised_union88 | A/SEX | unweighted | 0.0277406623 | 0.0119931858 |
| splince_supervised_union88 | A/SEX | PWGTP | 0.0198225189 | 0.0115862657 |
| splince_supervised_union88 | A/RAC1P | unweighted | 0.0541628406 | 0.0476372935 |
| splince_supervised_union88 | A/RAC1P | PWGTP | 0.0499776675 | 0.0436923926 |
| splince_supervised_union88 | AB/SEX | unweighted | 0.0274050365 | 0.0109571829 |
| splince_supervised_union88 | AB/SEX | PWGTP | 0.0202324725 | 0.00964443536 |
| splince_supervised_union88 | AB/RAC1P | unweighted | 0.0497233049 | 0.0442189106 |
| splince_supervised_union88 | AB/RAC1P | PWGTP | 0.0446906028 | 0.0382665996 |

## All registered roles

| Configuration | Role | Weighting | Selected CE | CE(arm) minus CE(H) | CE(arm) minus CE(J) |
| --- | --- | --- | --- | --- | --- |
| H | attack:A/SEX | unweighted | 0.687893661 | 0 | 0.0157474765 |
| H | attack:A/SEX | PWGTP | 0.686484452 | 0 | 0.00823625321 |
| H | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| H | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| H | attack:AB/SEX | unweighted | 0.685216935 | 0 | 0.0164478536 |
| H | attack:AB/SEX | PWGTP | 0.684361126 | 0 | 0.0105880371 |
| H | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| H | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| H | attack:A/public_coverage | unweighted | 0.519319923 | 0 | 0.00769877971 |
| H | attack:A/public_coverage | PWGTP | 0.521535326 | 0 | 0.00650044367 |
| H | attack:A/commute_over20 | unweighted | 0.683148944 | 0 | 0.00200881942 |
| H | attack:A/commute_over20 | PWGTP | 0.686542873 | 0 | 0.00357197194 |
| H | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| H | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| H | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| H | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| H | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| H | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| H | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| H | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| H | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| H | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| H | utility:A/same_residence | unweighted | 0.525151658 | 0 | 0.0257031045 |
| H | utility:A/same_residence | PWGTP | 0.499069095 | 0 | 0.0222148947 |
| H | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| H | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| H | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| H | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| H | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| H | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| H | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| H | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| J | attack:A/SEX | unweighted | 0.672146184 | -0.0157474765 | 0 |
| J | attack:A/SEX | PWGTP | 0.678248199 | -0.00823625321 | 0 |
| J | attack:A/RAC1P | unweighted | 1.26421381 | -0.00652554709 | 0 |
| J | attack:A/RAC1P | PWGTP | 1.25845437 | -0.00628527494 | 0 |
| J | attack:AB/SEX | unweighted | 0.668769081 | -0.0164478536 | 0 |
| J | attack:AB/SEX | PWGTP | 0.673773089 | -0.0105880371 | 0 |
| J | attack:AB/RAC1P | unweighted | 1.25210061 | -0.00550439428 | 0 |
| J | attack:AB/RAC1P | PWGTP | 1.24592823 | -0.00642400314 | 0 |
| J | attack:A/public_coverage | unweighted | 0.511621143 | -0.00769877971 | 0 |
| J | attack:A/public_coverage | PWGTP | 0.515034882 | -0.00650044367 | 0 |
| J | attack:A/commute_over20 | unweighted | 0.681140125 | -0.00200881942 | 0 |
| J | attack:A/commute_over20 | PWGTP | 0.682970901 | -0.00357197194 | 0 |
| J | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| J | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| J | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| J | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| J | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| J | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| J | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| J | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| J | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| J | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| J | utility:A/same_residence | unweighted | 0.499448553 | -0.0257031045 | 0 |
| J | utility:A/same_residence | PWGTP | 0.4768542 | -0.0222148947 | 0 |
| J | utility:A/income_binary | unweighted | 0.284517176 | -0.0111812687 | 0 |
| J | utility:A/income_binary | PWGTP | 0.296407648 | -0.0111719389 | 0 |
| J | utility:A/civilian_at_work | unweighted | 0.27143724 | -0.0078699532 | 0 |
| J | utility:A/civilian_at_work | PWGTP | 0.262470631 | -0.00534441626 | 0 |
| J | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| J | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| J | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| J | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_C_0.0005_a17 | attack:A/SEX | unweighted | 0.687467466 | -0.000426194659 | 0.0153212818 |
| T0_C_0.0005_a17 | attack:A/SEX | PWGTP | 0.686076946 | -0.000407506853 | 0.00782874636 |
| T0_C_0.0005_a17 | attack:A/RAC1P | unweighted | 1.27082907 | 8.97140296e-05 | 0.00661526112 |
| T0_C_0.0005_a17 | attack:A/RAC1P | PWGTP | 1.2652409 | 0.00050126112 | 0.00678653606 |
| T0_C_0.0005_a17 | attack:AB/SEX | unweighted | 0.684805692 | -0.00041124265 | 0.0160366109 |
| T0_C_0.0005_a17 | attack:AB/SEX | PWGTP | 0.684020519 | -0.000340607205 | 0.0102474299 |
| T0_C_0.0005_a17 | attack:AB/RAC1P | unweighted | 1.25792437 | 0.000319362902 | 0.00582375718 |
| T0_C_0.0005_a17 | attack:AB/RAC1P | PWGTP | 1.25309463 | 0.000742397957 | 0.00716640109 |
| T0_C_0.0005_a17 | attack:A/public_coverage | unweighted | 0.518867789 | -0.000452133123 | 0.00724664659 |
| T0_C_0.0005_a17 | attack:A/public_coverage | PWGTP | 0.520942189 | -0.000593137303 | 0.00590730637 |
| T0_C_0.0005_a17 | attack:A/commute_over20 | unweighted | 0.68317549 | 2.65461148e-05 | 0.00203536554 |
| T0_C_0.0005_a17 | attack:A/commute_over20 | PWGTP | 0.686436572 | -0.000106301091 | 0.00346567085 |
| T0_C_0.0005_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_C_0.0005_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_C_0.0005_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_C_0.0005_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_C_0.0005_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_C_0.0005_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_C_0.0005_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_C_0.0005_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_C_0.0005_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_C_0.0005_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_C_0.0005_a17 | utility:A/same_residence | unweighted | 0.519685223 | -0.00546643437 | 0.0202366702 |
| T0_C_0.0005_a17 | utility:A/same_residence | PWGTP | 0.493884981 | -0.00518411388 | 0.0170307808 |
| T0_C_0.0005_a17 | utility:A/income_binary | unweighted | 0.295721471 | 2.30262004e-05 | 0.0112042949 |
| T0_C_0.0005_a17 | utility:A/income_binary | PWGTP | 0.30755389 | -2.56961329e-05 | 0.0111462428 |
| T0_C_0.0005_a17 | utility:A/civilian_at_work | unweighted | 0.279167324 | -0.000139869089 | 0.00773008411 |
| T0_C_0.0005_a17 | utility:A/civilian_at_work | PWGTP | 0.267574121 | -0.0002409263 | 0.00510348996 |
| T0_C_0.0005_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_C_0.0005_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_C_0.0005_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_C_0.0005_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_C_0.002_a17 | attack:A/SEX | unweighted | 0.686348315 | -0.00154534554 | 0.014202131 |
| T0_C_0.002_a17 | attack:A/SEX | PWGTP | 0.685168481 | -0.00131597141 | 0.0069202818 |
| T0_C_0.002_a17 | attack:A/RAC1P | unweighted | 1.27050135 | -0.000238006333 | 0.00628754076 |
| T0_C_0.002_a17 | attack:A/RAC1P | PWGTP | 1.26458119 | -0.000158450831 | 0.00612682411 |
| T0_C_0.002_a17 | attack:AB/SEX | unweighted | 0.680888099 | -0.00432883602 | 0.0121190175 |
| T0_C_0.002_a17 | attack:AB/SEX | PWGTP | 0.683039671 | -0.00132145439 | 0.00926658275 |
| T0_C_0.002_a17 | attack:AB/RAC1P | unweighted | 1.25619534 | -0.0014096711 | 0.00409472317 |
| T0_C_0.002_a17 | attack:AB/RAC1P | PWGTP | 1.25143563 | -0.000916596249 | 0.00550740689 |
| T0_C_0.002_a17 | attack:A/public_coverage | unweighted | 0.518334657 | -0.000985265977 | 0.00671351374 |
| T0_C_0.002_a17 | attack:A/public_coverage | PWGTP | 0.520610258 | -0.000925068282 | 0.00557537539 |
| T0_C_0.002_a17 | attack:A/commute_over20 | unweighted | 0.683098963 | -4.998098e-05 | 0.00195883844 |
| T0_C_0.002_a17 | attack:A/commute_over20 | PWGTP | 0.686286425 | -0.000256447232 | 0.00331552471 |
| T0_C_0.002_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_C_0.002_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_C_0.002_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_C_0.002_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_C_0.002_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_C_0.002_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_C_0.002_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_C_0.002_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_C_0.002_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_C_0.002_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_C_0.002_a17 | utility:A/same_residence | unweighted | 0.504017388 | -0.0211342701 | 0.00456883444 |
| T0_C_0.002_a17 | utility:A/same_residence | PWGTP | 0.480066642 | -0.0190024534 | 0.00321244127 |
| T0_C_0.002_a17 | utility:A/income_binary | unweighted | 0.295560592 | -0.000137852836 | 0.0110434158 |
| T0_C_0.002_a17 | utility:A/income_binary | PWGTP | 0.307374508 | -0.000205078488 | 0.0109668605 |
| T0_C_0.002_a17 | utility:A/civilian_at_work | unweighted | 0.27916542 | -0.000141772437 | 0.00772818076 |
| T0_C_0.002_a17 | utility:A/civilian_at_work | PWGTP | 0.267622829 | -0.000192218505 | 0.00515219775 |
| T0_C_0.002_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_C_0.002_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_C_0.002_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_C_0.002_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_C_0.002_a33 | attack:A/SEX | unweighted | 0.687119309 | -0.000774351744 | 0.0149731248 |
| T0_C_0.002_a33 | attack:A/SEX | PWGTP | 0.685396375 | -0.00108807721 | 0.007148176 |
| T0_C_0.002_a33 | attack:A/RAC1P | unweighted | 1.27059874 | -0.000140617599 | 0.00638492949 |
| T0_C_0.002_a33 | attack:A/RAC1P | PWGTP | 1.26425446 | -0.000485182618 | 0.00580009232 |
| T0_C_0.002_a33 | attack:AB/SEX | unweighted | 0.684719566 | -0.000497368967 | 0.0159504846 |
| T0_C_0.002_a33 | attack:AB/SEX | PWGTP | 0.685807591 | 0.00144646549 | 0.0120345026 |
| T0_C_0.002_a33 | attack:AB/RAC1P | unweighted | 1.25604977 | -0.00155523496 | 0.00394915932 |
| T0_C_0.002_a33 | attack:AB/RAC1P | PWGTP | 1.25144193 | -0.000910297827 | 0.00551370531 |
| T0_C_0.002_a33 | attack:A/public_coverage | unweighted | 0.518238033 | -0.00108188944 | 0.00661689028 |
| T0_C_0.002_a33 | attack:A/public_coverage | PWGTP | 0.520414446 | -0.00112088012 | 0.00537956355 |
| T0_C_0.002_a33 | attack:A/commute_over20 | unweighted | 0.682127078 | -0.00102186657 | 0.000986952854 |
| T0_C_0.002_a33 | attack:A/commute_over20 | PWGTP | 0.685296356 | -0.00124651653 | 0.00232545541 |
| T0_C_0.002_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_C_0.002_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_C_0.002_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_C_0.002_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_C_0.002_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_C_0.002_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_C_0.002_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_C_0.002_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_C_0.002_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_C_0.002_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_C_0.002_a33 | utility:A/same_residence | unweighted | 0.50393538 | -0.0212162778 | 0.0044868268 |
| T0_C_0.002_a33 | utility:A/same_residence | PWGTP | 0.479853606 | -0.0192154891 | 0.00299940556 |
| T0_C_0.002_a33 | utility:A/income_binary | unweighted | 0.295503628 | -0.000194816773 | 0.0109864519 |
| T0_C_0.002_a33 | utility:A/income_binary | PWGTP | 0.307355129 | -0.000224457365 | 0.0109474816 |
| T0_C_0.002_a33 | utility:A/civilian_at_work | unweighted | 0.279159765 | -0.000147428202 | 0.007722525 |
| T0_C_0.002_a33 | utility:A/civilian_at_work | PWGTP | 0.267595936 | -0.000219111655 | 0.0051253046 |
| T0_C_0.002_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_C_0.002_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_C_0.002_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_C_0.002_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_C_0.01_a17 | attack:A/SEX | unweighted | 0.684703969 | -0.00318969123 | 0.0125577853 |
| T0_C_0.01_a17 | attack:A/SEX | PWGTP | 0.683875886 | -0.00260856627 | 0.00562768695 |
| T0_C_0.01_a17 | attack:A/RAC1P | unweighted | 1.26923117 | -0.00150818767 | 0.00501735942 |
| T0_C_0.01_a17 | attack:A/RAC1P | PWGTP | 1.26285571 | -0.00188393457 | 0.00440134037 |
| T0_C_0.01_a17 | attack:AB/SEX | unweighted | 0.680210542 | -0.00500639275 | 0.0114414608 |
| T0_C_0.01_a17 | attack:AB/SEX | PWGTP | 0.68022103 | -0.00414009564 | 0.00644794149 |
| T0_C_0.01_a17 | attack:AB/RAC1P | unweighted | 1.2550886 | -0.00251640774 | 0.00298798654 |
| T0_C_0.01_a17 | attack:AB/RAC1P | PWGTP | 1.25005852 | -0.00229371272 | 0.00413029042 |
| T0_C_0.01_a17 | attack:A/public_coverage | unweighted | 0.520514553 | 0.00119463051 | 0.00889341022 |
| T0_C_0.01_a17 | attack:A/public_coverage | PWGTP | 0.522794733 | 0.00125940719 | 0.00775985086 |
| T0_C_0.01_a17 | attack:A/commute_over20 | unweighted | 0.681377215 | -0.00177172961 | 0.00023708981 |
| T0_C_0.01_a17 | attack:A/commute_over20 | PWGTP | 0.684514247 | -0.00202862537 | 0.00154334657 |
| T0_C_0.01_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_C_0.01_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_C_0.01_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_C_0.01_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_C_0.01_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_C_0.01_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_C_0.01_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_C_0.01_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_C_0.01_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_C_0.01_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_C_0.01_a17 | utility:A/same_residence | unweighted | 0.486286213 | -0.0388654452 | -0.0131623407 |
| T0_C_0.01_a17 | utility:A/same_residence | PWGTP | 0.463443594 | -0.0356255011 | -0.0134106064 |
| T0_C_0.01_a17 | utility:A/income_binary | unweighted | 0.295948447 | 0.000250003044 | 0.0114312717 |
| T0_C_0.01_a17 | utility:A/income_binary | PWGTP | 0.307921045 | 0.00034145814 | 0.0115133971 |
| T0_C_0.01_a17 | utility:A/civilian_at_work | unweighted | 0.27900408 | -0.000303112863 | 0.00756684033 |
| T0_C_0.01_a17 | utility:A/civilian_at_work | PWGTP | 0.267336476 | -0.000478570814 | 0.00486584545 |
| T0_C_0.01_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_C_0.01_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_C_0.01_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_C_0.01_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_C_0_a17 | attack:A/SEX | unweighted | 0.686542797 | -0.00135086398 | 0.0143966125 |
| T0_C_0_a17 | attack:A/SEX | PWGTP | 0.687713725 | 0.0012292731 | 0.00946552631 |
| T0_C_0_a17 | attack:A/RAC1P | unweighted | 1.26997494 | -0.000764415667 | 0.00576113142 |
| T0_C_0_a17 | attack:A/RAC1P | PWGTP | 1.26474668 | 7.03855375e-06 | 0.0062923135 |
| T0_C_0_a17 | attack:AB/SEX | unweighted | 0.685171529 | -4.5406231e-05 | 0.0164024473 |
| T0_C_0_a17 | attack:AB/SEX | PWGTP | 0.684157967 | -0.000203158631 | 0.0103848785 |
| T0_C_0_a17 | attack:AB/RAC1P | unweighted | 1.25752266 | -8.23492769e-05 | 0.005422045 |
| T0_C_0_a17 | attack:AB/RAC1P | PWGTP | 1.25200902 | -0.000343210775 | 0.00608079236 |
| T0_C_0_a17 | attack:A/public_coverage | unweighted | 0.518970898 | -0.000349024999 | 0.00734975472 |
| T0_C_0_a17 | attack:A/public_coverage | PWGTP | 0.521231677 | -0.000303649163 | 0.00619679451 |
| T0_C_0_a17 | attack:A/commute_over20 | unweighted | 0.682651795 | -0.000497148971 | 0.00151167045 |
| T0_C_0_a17 | attack:A/commute_over20 | PWGTP | 0.685894218 | -0.000648654305 | 0.00292331763 |
| T0_C_0_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_C_0_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_C_0_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_C_0_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_C_0_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_C_0_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_C_0_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_C_0_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_C_0_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_C_0_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_C_0_a17 | utility:A/same_residence | unweighted | 0.524482928 | -0.000668729478 | 0.0250343751 |
| T0_C_0_a17 | utility:A/same_residence | PWGTP | 0.498503555 | -0.000565540175 | 0.0216493545 |
| T0_C_0_a17 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| T0_C_0_a17 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| T0_C_0_a17 | utility:A/civilian_at_work | unweighted | 0.278925664 | -0.000381528378 | 0.00748842482 |
| T0_C_0_a17 | utility:A/civilian_at_work | PWGTP | 0.267420519 | -0.000394528133 | 0.00494988813 |
| T0_C_0_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_C_0_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_C_0_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_C_0_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_L_0.0005_a17 | attack:A/SEX | unweighted | 0.687727246 | -0.000166414763 | 0.0155810617 |
| T0_L_0.0005_a17 | attack:A/SEX | PWGTP | 0.68600258 | -0.000481872769 | 0.00775438044 |
| T0_L_0.0005_a17 | attack:A/RAC1P | unweighted | 1.27085884 | 0.00011947922 | 0.00664502631 |
| T0_L_0.0005_a17 | attack:A/RAC1P | PWGTP | 1.26495054 | 0.000210900317 | 0.00649617526 |
| T0_L_0.0005_a17 | attack:AB/SEX | unweighted | 0.685424425 | 0.000207489891 | 0.0166553434 |
| T0_L_0.0005_a17 | attack:AB/SEX | PWGTP | 0.684899643 | 0.000538516986 | 0.0111265541 |
| T0_L_0.0005_a17 | attack:AB/RAC1P | unweighted | 1.25790465 | 0.000299640447 | 0.00580403472 |
| T0_L_0.0005_a17 | attack:AB/RAC1P | PWGTP | 1.25285969 | 0.000507462367 | 0.0069314655 |
| T0_L_0.0005_a17 | attack:A/public_coverage | unweighted | 0.518373371 | -0.000946551124 | 0.00675222859 |
| T0_L_0.0005_a17 | attack:A/public_coverage | PWGTP | 0.520461244 | -0.00107408234 | 0.00542636133 |
| T0_L_0.0005_a17 | attack:A/commute_over20 | unweighted | 0.683275911 | 0.00012696728 | 0.0021357867 |
| T0_L_0.0005_a17 | attack:A/commute_over20 | PWGTP | 0.686540446 | -2.42709575e-06 | 0.00356954484 |
| T0_L_0.0005_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_L_0.0005_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_L_0.0005_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_L_0.0005_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_L_0.0005_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_L_0.0005_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_L_0.0005_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_L_0.0005_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_L_0.0005_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_L_0.0005_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_L_0.0005_a17 | utility:A/same_residence | unweighted | 0.51639375 | -0.00875790748 | 0.0169451971 |
| T0_L_0.0005_a17 | utility:A/same_residence | PWGTP | 0.490903101 | -0.00816599421 | 0.0140489005 |
| T0_L_0.0005_a17 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| T0_L_0.0005_a17 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| T0_L_0.0005_a17 | utility:A/civilian_at_work | unweighted | 0.279659762 | 0.000352569744 | 0.00822252294 |
| T0_L_0.0005_a17 | utility:A/civilian_at_work | PWGTP | 0.267923241 | 0.000108193417 | 0.00545260968 |
| T0_L_0.0005_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_L_0.0005_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_L_0.0005_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_L_0.0005_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_L_0.002_a17 | attack:A/SEX | unweighted | 0.686751723 | -0.00114193744 | 0.0146055391 |
| T0_L_0.002_a17 | attack:A/SEX | PWGTP | 0.685862745 | -0.000621707575 | 0.00761454564 |
| T0_L_0.002_a17 | attack:A/RAC1P | unweighted | 1.2708263 | 8.69355124e-05 | 0.0066124826 |
| T0_L_0.002_a17 | attack:A/RAC1P | PWGTP | 1.26487011 | 0.000130471063 | 0.00641574601 |
| T0_L_0.002_a17 | attack:AB/SEX | unweighted | 0.680363702 | -0.00485323263 | 0.0115946209 |
| T0_L_0.002_a17 | attack:AB/SEX | PWGTP | 0.682431026 | -0.00193009978 | 0.00865793735 |
| T0_L_0.002_a17 | attack:AB/RAC1P | unweighted | 1.25696837 | -0.000636632379 | 0.0048677619 |
| T0_L_0.002_a17 | attack:AB/RAC1P | PWGTP | 1.25287516 | 0.000522934593 | 0.00694693773 |
| T0_L_0.002_a17 | attack:A/public_coverage | unweighted | 0.5190635 | -0.000256422107 | 0.00744235761 |
| T0_L_0.002_a17 | attack:A/public_coverage | PWGTP | 0.521196671 | -0.000338655218 | 0.00616178845 |
| T0_L_0.002_a17 | attack:A/commute_over20 | unweighted | 0.68234985 | -0.000799094401 | 0.00120972502 |
| T0_L_0.002_a17 | attack:A/commute_over20 | PWGTP | 0.68539009 | -0.00115278307 | 0.00241918887 |
| T0_L_0.002_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_L_0.002_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_L_0.002_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_L_0.002_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_L_0.002_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_L_0.002_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_L_0.002_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_L_0.002_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_L_0.002_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_L_0.002_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_L_0.002_a17 | utility:A/same_residence | unweighted | 0.50244696 | -0.0227046973 | 0.00299840723 |
| T0_L_0.002_a17 | utility:A/same_residence | PWGTP | 0.478752686 | -0.0203164092 | 0.0018984855 |
| T0_L_0.002_a17 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| T0_L_0.002_a17 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| T0_L_0.002_a17 | utility:A/civilian_at_work | unweighted | 0.279115007 | -0.000192185778 | 0.00767776742 |
| T0_L_0.002_a17 | utility:A/civilian_at_work | PWGTP | 0.267530154 | -0.00028489355 | 0.00505952271 |
| T0_L_0.002_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_L_0.002_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_L_0.002_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_L_0.002_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_L_0.002_a33 | attack:A/SEX | unweighted | 0.684073059 | -0.00382060154 | 0.011926875 |
| T0_L_0.002_a33 | attack:A/SEX | PWGTP | 0.684539193 | -0.00194525909 | 0.00629099412 |
| T0_L_0.002_a33 | attack:A/RAC1P | unweighted | 1.27030674 | -0.00043262072 | 0.00609292637 |
| T0_L_0.002_a33 | attack:A/RAC1P | PWGTP | 1.26409142 | -0.000648225099 | 0.00563704984 |
| T0_L_0.002_a33 | attack:AB/SEX | unweighted | 0.683582212 | -0.00163472274 | 0.0148131308 |
| T0_L_0.002_a33 | attack:AB/SEX | PWGTP | 0.68314031 | -0.00122081568 | 0.00936722146 |
| T0_L_0.002_a33 | attack:AB/RAC1P | unweighted | 1.25707356 | -0.000531445473 | 0.0049729488 |
| T0_L_0.002_a33 | attack:AB/RAC1P | PWGTP | 1.25236887 | 1.66397822e-05 | 0.00644064292 |
| T0_L_0.002_a33 | attack:A/public_coverage | unweighted | 0.519099595 | -0.000220327374 | 0.00747845234 |
| T0_L_0.002_a33 | attack:A/public_coverage | PWGTP | 0.521181566 | -0.000353760175 | 0.0061466835 |
| T0_L_0.002_a33 | attack:A/commute_over20 | unweighted | 0.682429123 | -0.000719820746 | 0.00128899868 |
| T0_L_0.002_a33 | attack:A/commute_over20 | PWGTP | 0.685510491 | -0.00103238186 | 0.00253959008 |
| T0_L_0.002_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_L_0.002_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_L_0.002_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_L_0.002_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_L_0.002_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_L_0.002_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_L_0.002_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_L_0.002_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_L_0.002_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_L_0.002_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_L_0.002_a33 | utility:A/same_residence | unweighted | 0.503050849 | -0.0221008084 | 0.00360229615 |
| T0_L_0.002_a33 | utility:A/same_residence | PWGTP | 0.479162157 | -0.0199069378 | 0.00230795687 |
| T0_L_0.002_a33 | utility:A/income_binary | unweighted | 0.296071954 | 0.00037350928 | 0.0115547779 |
| T0_L_0.002_a33 | utility:A/income_binary | PWGTP | 0.307853808 | 0.000274221269 | 0.0114461602 |
| T0_L_0.002_a33 | utility:A/civilian_at_work | unweighted | 0.279081557 | -0.00022563533 | 0.00764431787 |
| T0_L_0.002_a33 | utility:A/civilian_at_work | PWGTP | 0.267550021 | -0.000265025756 | 0.0050793905 |
| T0_L_0.002_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_L_0.002_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_L_0.002_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_L_0.002_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_L_0.01_a17 | attack:A/SEX | unweighted | 0.681757991 | -0.00613566926 | 0.00961180724 |
| T0_L_0.01_a17 | attack:A/SEX | PWGTP | 0.680708758 | -0.00577569463 | 0.00246055858 |
| T0_L_0.01_a17 | attack:A/RAC1P | unweighted | 1.26988107 | -0.000858287196 | 0.00566725989 |
| T0_L_0.01_a17 | attack:A/RAC1P | PWGTP | 1.26327207 | -0.00146757257 | 0.00481770237 |
| T0_L_0.01_a17 | attack:AB/SEX | unweighted | 0.679798724 | -0.00541821079 | 0.0110296428 |
| T0_L_0.01_a17 | attack:AB/SEX | PWGTP | 0.679972968 | -0.00438815785 | 0.00619987928 |
| T0_L_0.01_a17 | attack:AB/RAC1P | unweighted | 1.25603669 | -0.00156831171 | 0.00393608256 |
| T0_L_0.01_a17 | attack:AB/RAC1P | PWGTP | 1.25055464 | -0.00179759324 | 0.0046264099 |
| T0_L_0.01_a17 | attack:A/public_coverage | unweighted | 0.519769376 | 0.000449453105 | 0.00814823282 |
| T0_L_0.01_a17 | attack:A/public_coverage | PWGTP | 0.522658745 | 0.00112341927 | 0.00762386294 |
| T0_L_0.01_a17 | attack:A/commute_over20 | unweighted | 0.68167225 | -0.00147669458 | 0.000532124846 |
| T0_L_0.01_a17 | attack:A/commute_over20 | PWGTP | 0.684311046 | -0.00223182624 | 0.0013401457 |
| T0_L_0.01_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_L_0.01_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_L_0.01_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_L_0.01_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_L_0.01_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_L_0.01_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_L_0.01_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_L_0.01_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_L_0.01_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_L_0.01_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_L_0.01_a17 | utility:A/same_residence | unweighted | 0.483197663 | -0.0419539949 | -0.0162508904 |
| T0_L_0.01_a17 | utility:A/same_residence | PWGTP | 0.460460631 | -0.0386084644 | -0.0163935697 |
| T0_L_0.01_a17 | utility:A/income_binary | unweighted | 0.295904463 | 0.000206018902 | 0.0113872876 |
| T0_L_0.01_a17 | utility:A/income_binary | PWGTP | 0.308312439 | 0.000732852332 | 0.0119047913 |
| T0_L_0.01_a17 | utility:A/civilian_at_work | unweighted | 0.278812934 | -0.000494258762 | 0.00737569444 |
| T0_L_0.01_a17 | utility:A/civilian_at_work | PWGTP | 0.266970114 | -0.000844933217 | 0.00449948304 |
| T0_L_0.01_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_L_0.01_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_L_0.01_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_L_0.01_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_L_0_a17 | attack:A/SEX | unweighted | 0.687536779 | -0.000356881744 | 0.0153905948 |
| T0_L_0_a17 | attack:A/SEX | PWGTP | 0.686238538 | -0.000245914516 | 0.0079903387 |
| T0_L_0_a17 | attack:A/RAC1P | unweighted | 1.26992242 | -0.000816936831 | 0.00570861026 |
| T0_L_0_a17 | attack:A/RAC1P | PWGTP | 1.26473249 | -7.15497269e-06 | 0.00627811997 |
| T0_L_0_a17 | attack:AB/SEX | unweighted | 0.685116536 | -0.000100399144 | 0.0163474544 |
| T0_L_0_a17 | attack:AB/SEX | PWGTP | 0.684016383 | -0.000344742461 | 0.0102432947 |
| T0_L_0_a17 | attack:AB/RAC1P | unweighted | 1.25759691 | -8.09608414e-06 | 0.00549629819 |
| T0_L_0_a17 | attack:AB/RAC1P | PWGTP | 1.25207376 | -0.000278465977 | 0.00614553716 |
| T0_L_0_a17 | attack:A/public_coverage | unweighted | 0.519166329 | -0.00015359381 | 0.0075451859 |
| T0_L_0_a17 | attack:A/public_coverage | PWGTP | 0.521363085 | -0.000172241137 | 0.00632820253 |
| T0_L_0_a17 | attack:A/commute_over20 | unweighted | 0.682885356 | -0.000263587682 | 0.00174523174 |
| T0_L_0_a17 | attack:A/commute_over20 | PWGTP | 0.686293268 | -0.000249604224 | 0.00332236771 |
| T0_L_0_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_L_0_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_L_0_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_L_0_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_L_0_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_L_0_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_L_0_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_L_0_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_L_0_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_L_0_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_L_0_a17 | utility:A/same_residence | unweighted | 0.52456007 | -0.000591588244 | 0.0251115163 |
| T0_L_0_a17 | utility:A/same_residence | PWGTP | 0.498427683 | -0.000641412433 | 0.0215734823 |
| T0_L_0_a17 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| T0_L_0_a17 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| T0_L_0_a17 | utility:A/civilian_at_work | unweighted | 0.279001019 | -0.000306173585 | 0.00756377961 |
| T0_L_0_a17 | utility:A/civilian_at_work | PWGTP | 0.26746354 | -0.000351507228 | 0.00499290903 |
| T0_L_0_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_L_0_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_L_0_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_L_0_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_U_unconstrained_a17 | attack:A/SEX | unweighted | 0.683442236 | -0.00445142498 | 0.0112960515 |
| T0_U_unconstrained_a17 | attack:A/SEX | PWGTP | 0.681093676 | -0.00539077652 | 0.00284547669 |
| T0_U_unconstrained_a17 | attack:A/RAC1P | unweighted | 1.27096829 | 0.000228929708 | 0.0067544768 |
| T0_U_unconstrained_a17 | attack:A/RAC1P | PWGTP | 1.26468274 | -5.68971449e-05 | 0.0062283778 |
| T0_U_unconstrained_a17 | attack:AB/SEX | unweighted | 0.679822937 | -0.00539399739 | 0.0110538562 |
| T0_U_unconstrained_a17 | attack:AB/SEX | PWGTP | 0.67940282 | -0.00495830632 | 0.00562973081 |
| T0_U_unconstrained_a17 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| T0_U_unconstrained_a17 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| T0_U_unconstrained_a17 | attack:A/public_coverage | unweighted | 0.520315549 | 0.000995625986 | 0.0086944057 |
| T0_U_unconstrained_a17 | attack:A/public_coverage | PWGTP | 0.523064602 | 0.00152927595 | 0.00802971962 |
| T0_U_unconstrained_a17 | attack:A/commute_over20 | unweighted | 0.683172328 | 2.33842691e-05 | 0.00203220369 |
| T0_U_unconstrained_a17 | attack:A/commute_over20 | PWGTP | 0.686118688 | -0.000424185109 | 0.00314778683 |
| T0_U_unconstrained_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_U_unconstrained_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_U_unconstrained_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_U_unconstrained_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_U_unconstrained_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_U_unconstrained_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_U_unconstrained_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_U_unconstrained_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_U_unconstrained_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_U_unconstrained_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_U_unconstrained_a17 | utility:A/same_residence | unweighted | 0.482243676 | -0.0429079814 | -0.0172048769 |
| T0_U_unconstrained_a17 | utility:A/same_residence | PWGTP | 0.459199814 | -0.039869281 | -0.0176543863 |
| T0_U_unconstrained_a17 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| T0_U_unconstrained_a17 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| T0_U_unconstrained_a17 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| T0_U_unconstrained_a17 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| T0_U_unconstrained_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_U_unconstrained_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_U_unconstrained_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_U_unconstrained_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_U_unconstrained_a33 | attack:A/SEX | unweighted | 0.681014639 | -0.00687902163 | 0.00886845486 |
| T0_U_unconstrained_a33 | attack:A/SEX | PWGTP | 0.681886441 | -0.0045980115 | 0.00363824171 |
| T0_U_unconstrained_a33 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| T0_U_unconstrained_a33 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| T0_U_unconstrained_a33 | attack:AB/SEX | unweighted | 0.673425124 | -0.0117918105 | 0.00465604308 |
| T0_U_unconstrained_a33 | attack:AB/SEX | PWGTP | 0.679364071 | -0.00499705446 | 0.00559098267 |
| T0_U_unconstrained_a33 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| T0_U_unconstrained_a33 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| T0_U_unconstrained_a33 | attack:A/public_coverage | unweighted | 0.520863585 | 0.00154366253 | 0.00924244224 |
| T0_U_unconstrained_a33 | attack:A/public_coverage | PWGTP | 0.52407307 | 0.00253774434 | 0.00903818802 |
| T0_U_unconstrained_a33 | attack:A/commute_over20 | unweighted | 0.68309756 | -5.13844286e-05 | 0.00195743499 |
| T0_U_unconstrained_a33 | attack:A/commute_over20 | PWGTP | 0.685789303 | -0.000753569182 | 0.00281840276 |
| T0_U_unconstrained_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_U_unconstrained_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_U_unconstrained_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_U_unconstrained_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_U_unconstrained_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_U_unconstrained_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_U_unconstrained_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_U_unconstrained_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_U_unconstrained_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_U_unconstrained_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_U_unconstrained_a33 | utility:A/same_residence | unweighted | 0.481512008 | -0.0436396499 | -0.0179365453 |
| T0_U_unconstrained_a33 | utility:A/same_residence | PWGTP | 0.458575793 | -0.0404933021 | -0.0182784074 |
| T0_U_unconstrained_a33 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| T0_U_unconstrained_a33 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| T0_U_unconstrained_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| T0_U_unconstrained_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| T0_U_unconstrained_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_U_unconstrained_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_U_unconstrained_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_U_unconstrained_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_code | attack:A/SEX | unweighted | 0.680055231 | -0.00783842939 | 0.00790904711 |
| T0_code | attack:A/SEX | PWGTP | 0.686555925 | 7.14723779e-05 | 0.00830772559 |
| T0_code | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| T0_code | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| T0_code | attack:AB/SEX | unweighted | 0.677005596 | -0.00821133878 | 0.00823651477 |
| T0_code | attack:AB/SEX | PWGTP | 0.681636797 | -0.00272432851 | 0.00786370863 |
| T0_code | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| T0_code | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| T0_code | attack:A/public_coverage | unweighted | 0.520363848 | 0.00104392538 | 0.00874270509 |
| T0_code | attack:A/public_coverage | PWGTP | 0.522849443 | 0.00131411741 | 0.00781456108 |
| T0_code | attack:A/commute_over20 | unweighted | 0.683049526 | -9.94180546e-05 | 0.00190940137 |
| T0_code | attack:A/commute_over20 | PWGTP | 0.685673996 | -0.000868876459 | 0.00270309548 |
| T0_code | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_code | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_code | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_code | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_code | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_code | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_code | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_code | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_code | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_code | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_code | utility:A/same_residence | unweighted | 0.485794694 | -0.0393569634 | -0.0136538589 |
| T0_code | utility:A/same_residence | PWGTP | 0.463912802 | -0.035156293 | -0.0129413983 |
| T0_code | utility:A/income_binary | unweighted | 0.296436928 | 0.000738483367 | 0.011919752 |
| T0_code | utility:A/income_binary | PWGTP | 0.307891001 | 0.00031141459 | 0.0114833535 |
| T0_code | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| T0_code | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| T0_code | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_code | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_code | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_code | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_rr_0.25 | attack:A/SEX | unweighted | 0.688256457 | 0.000362796254 | 0.0161102728 |
| T0_rr_0.25 | attack:A/SEX | PWGTP | 0.687159015 | 0.000674562727 | 0.00891081594 |
| T0_rr_0.25 | attack:A/RAC1P | unweighted | 1.26959558 | -0.00114378176 | 0.00538176533 |
| T0_rr_0.25 | attack:A/RAC1P | PWGTP | 1.26445464 | -0.000284996057 | 0.00600027888 |
| T0_rr_0.25 | attack:AB/SEX | unweighted | 0.684787571 | -0.000429363479 | 0.0160184901 |
| T0_rr_0.25 | attack:AB/SEX | PWGTP | 0.684057461 | -0.000303665054 | 0.0102843721 |
| T0_rr_0.25 | attack:AB/RAC1P | unweighted | 1.25767628 | 7.1269714e-05 | 0.00557566399 |
| T0_rr_0.25 | attack:AB/RAC1P | PWGTP | 1.25240706 | 5.48330144e-05 | 0.00647883615 |
| T0_rr_0.25 | attack:A/public_coverage | unweighted | 0.519255835 | -6.40872331e-05 | 0.00763469248 |
| T0_rr_0.25 | attack:A/public_coverage | PWGTP | 0.521444171 | -9.11552381e-05 | 0.00640928843 |
| T0_rr_0.25 | attack:A/commute_over20 | unweighted | 0.683236952 | 8.80077344e-05 | 0.00209682716 |
| T0_rr_0.25 | attack:A/commute_over20 | PWGTP | 0.68655772 | 1.48471476e-05 | 0.00358681909 |
| T0_rr_0.25 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_rr_0.25 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_rr_0.25 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_rr_0.25 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_rr_0.25 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_rr_0.25 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_rr_0.25 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_rr_0.25 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_rr_0.25 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_rr_0.25 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_rr_0.25 | utility:A/same_residence | unweighted | 0.521040032 | -0.00411162551 | 0.021591479 |
| T0_rr_0.25 | utility:A/same_residence | PWGTP | 0.495413843 | -0.00365525161 | 0.0185596431 |
| T0_rr_0.25 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| T0_rr_0.25 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| T0_rr_0.25 | utility:A/civilian_at_work | unweighted | 0.279296892 | -1.03005503e-05 | 0.00785965265 |
| T0_rr_0.25 | utility:A/civilian_at_work | PWGTP | 0.267663575 | -0.00015147178 | 0.00519294448 |
| T0_rr_0.25 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_rr_0.25 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_rr_0.25 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_rr_0.25 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_rr_0.25_a33 | attack:A/SEX | unweighted | 0.684736303 | -0.00315735794 | 0.0125901186 |
| T0_rr_0.25_a33 | attack:A/SEX | PWGTP | 0.687984168 | 0.00149971562 | 0.00973596883 |
| T0_rr_0.25_a33 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| T0_rr_0.25_a33 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| T0_rr_0.25_a33 | attack:AB/SEX | unweighted | 0.681885265 | -0.00333166932 | 0.0131161842 |
| T0_rr_0.25_a33 | attack:AB/SEX | PWGTP | 0.684074404 | -0.000286721602 | 0.0103013155 |
| T0_rr_0.25_a33 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| T0_rr_0.25_a33 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| T0_rr_0.25_a33 | attack:A/public_coverage | unweighted | 0.519448582 | 0.000128659519 | 0.00782743923 |
| T0_rr_0.25_a33 | attack:A/public_coverage | PWGTP | 0.521581809 | 4.64833899e-05 | 0.00654692706 |
| T0_rr_0.25_a33 | attack:A/commute_over20 | unweighted | 0.68316939 | 2.04454975e-05 | 0.00202926492 |
| T0_rr_0.25_a33 | attack:A/commute_over20 | PWGTP | 0.686549216 | 6.34334912e-06 | 0.00357831529 |
| T0_rr_0.25_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_rr_0.25_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_rr_0.25_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_rr_0.25_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_rr_0.25_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_rr_0.25_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_rr_0.25_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_rr_0.25_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_rr_0.25_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_rr_0.25_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_rr_0.25_a33 | utility:A/same_residence | unweighted | 0.521753825 | -0.00339783292 | 0.0223052716 |
| T0_rr_0.25_a33 | utility:A/same_residence | PWGTP | 0.496131018 | -0.00293807692 | 0.0192768178 |
| T0_rr_0.25_a33 | utility:A/income_binary | unweighted | 0.295775674 | 7.7229104e-05 | 0.0112584978 |
| T0_rr_0.25_a33 | utility:A/income_binary | PWGTP | 0.307681773 | 0.000102186487 | 0.0112741254 |
| T0_rr_0.25_a33 | utility:A/civilian_at_work | unweighted | 0.279435224 | 0.000128031584 | 0.00799798478 |
| T0_rr_0.25_a33 | utility:A/civilian_at_work | PWGTP | 0.267739926 | -7.51208966e-05 | 0.00526929536 |
| T0_rr_0.25_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_rr_0.25_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_rr_0.25_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_rr_0.25_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_rr_0.5 | attack:A/SEX | unweighted | 0.684781209 | -0.00311245109 | 0.0126350254 |
| T0_rr_0.5 | attack:A/SEX | PWGTP | 0.685717909 | -0.000766542958 | 0.00746971025 |
| T0_rr_0.5 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| T0_rr_0.5 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| T0_rr_0.5 | attack:AB/SEX | unweighted | 0.684148545 | -0.0010683897 | 0.0153794638 |
| T0_rr_0.5 | attack:AB/SEX | PWGTP | 0.683804799 | -0.00055632712 | 0.01003171 |
| T0_rr_0.5 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| T0_rr_0.5 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| T0_rr_0.5 | attack:A/public_coverage | unweighted | 0.519707993 | 0.000388070243 | 0.00808684996 |
| T0_rr_0.5 | attack:A/public_coverage | PWGTP | 0.522229734 | 0.000694408033 | 0.0071948517 |
| T0_rr_0.5 | attack:A/commute_over20 | unweighted | 0.68318904 | 4.00961834e-05 | 0.00204891561 |
| T0_rr_0.5 | attack:A/commute_over20 | PWGTP | 0.686430599 | -0.000112273793 | 0.00345969815 |
| T0_rr_0.5 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_rr_0.5 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_rr_0.5 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_rr_0.5 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_rr_0.5 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_rr_0.5 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_rr_0.5 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_rr_0.5 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_rr_0.5 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_rr_0.5 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_rr_0.5 | utility:A/same_residence | unweighted | 0.513153481 | -0.0119981764 | 0.0137049281 |
| T0_rr_0.5 | utility:A/same_residence | PWGTP | 0.487926177 | -0.011142918 | 0.0110719767 |
| T0_rr_0.5 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| T0_rr_0.5 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| T0_rr_0.5 | utility:A/civilian_at_work | unweighted | 0.279206758 | -0.000100434305 | 0.00776951889 |
| T0_rr_0.5 | utility:A/civilian_at_work | PWGTP | 0.267373389 | -0.000441657737 | 0.00490275852 |
| T0_rr_0.5 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_rr_0.5 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_rr_0.5 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_rr_0.5 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_rr_0.5_a33 | attack:A/SEX | unweighted | 0.682853825 | -0.00503983533 | 0.0107076412 |
| T0_rr_0.5_a33 | attack:A/SEX | PWGTP | 0.686158345 | -0.000326107079 | 0.00791014613 |
| T0_rr_0.5_a33 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| T0_rr_0.5_a33 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| T0_rr_0.5_a33 | attack:AB/SEX | unweighted | 0.677830813 | -0.00738612158 | 0.00906173197 |
| T0_rr_0.5_a33 | attack:AB/SEX | PWGTP | 0.684821662 | 0.000460535731 | 0.0110485729 |
| T0_rr_0.5_a33 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| T0_rr_0.5_a33 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| T0_rr_0.5_a33 | attack:A/public_coverage | unweighted | 0.519772384 | 0.000452461249 | 0.00815124096 |
| T0_rr_0.5_a33 | attack:A/public_coverage | PWGTP | 0.522176794 | 0.000641468433 | 0.0071419121 |
| T0_rr_0.5_a33 | attack:A/commute_over20 | unweighted | 0.683180338 | 3.13939169e-05 | 0.00204021334 |
| T0_rr_0.5_a33 | attack:A/commute_over20 | PWGTP | 0.686366451 | -0.000176421353 | 0.00339555059 |
| T0_rr_0.5_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_rr_0.5_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_rr_0.5_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_rr_0.5_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_rr_0.5_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_rr_0.5_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_rr_0.5_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_rr_0.5_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_rr_0.5_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_rr_0.5_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_rr_0.5_a33 | utility:A/same_residence | unweighted | 0.513291135 | -0.0118605229 | 0.0138425817 |
| T0_rr_0.5_a33 | utility:A/same_residence | PWGTP | 0.487788217 | -0.0112808779 | 0.0109340168 |
| T0_rr_0.5_a33 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| T0_rr_0.5_a33 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| T0_rr_0.5_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| T0_rr_0.5_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| T0_rr_0.5_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_rr_0.5_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_rr_0.5_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_rr_0.5_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_rr_0.75 | attack:A/SEX | unweighted | 0.686073103 | -0.00182055728 | 0.0139269192 |
| T0_rr_0.75 | attack:A/SEX | PWGTP | 0.686211149 | -0.000273303031 | 0.00796295018 |
| T0_rr_0.75 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| T0_rr_0.75 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| T0_rr_0.75 | attack:AB/SEX | unweighted | 0.679717754 | -0.00549918057 | 0.010948673 |
| T0_rr_0.75 | attack:AB/SEX | PWGTP | 0.683947898 | -0.000413228163 | 0.010174809 |
| T0_rr_0.75 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| T0_rr_0.75 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| T0_rr_0.75 | attack:A/public_coverage | unweighted | 0.520502591 | 0.00118266849 | 0.0088814482 |
| T0_rr_0.75 | attack:A/public_coverage | PWGTP | 0.523273392 | 0.00173806624 | 0.00823850991 |
| T0_rr_0.75 | attack:A/commute_over20 | unweighted | 0.683143407 | -5.53692933e-06 | 0.00200328249 |
| T0_rr_0.75 | attack:A/commute_over20 | PWGTP | 0.686246473 | -0.000296399246 | 0.00327557269 |
| T0_rr_0.75 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_rr_0.75 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_rr_0.75 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_rr_0.75 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_rr_0.75 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_rr_0.75 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_rr_0.75 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_rr_0.75 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_rr_0.75 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_rr_0.75 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_rr_0.75 | utility:A/same_residence | unweighted | 0.501993741 | -0.0231579172 | 0.0025451873 |
| T0_rr_0.75 | utility:A/same_residence | PWGTP | 0.4774424 | -0.0216266953 | 0.000588199435 |
| T0_rr_0.75 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| T0_rr_0.75 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| T0_rr_0.75 | utility:A/civilian_at_work | unweighted | 0.278835918 | -0.000471274792 | 0.00739867841 |
| T0_rr_0.75 | utility:A/civilian_at_work | PWGTP | 0.267273235 | -0.000541812527 | 0.00480260373 |
| T0_rr_0.75 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_rr_0.75 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_rr_0.75 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_rr_0.75 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_rr_0.75_a33 | attack:A/SEX | unweighted | 0.67968554 | -0.00820812067 | 0.00753935583 |
| T0_rr_0.75_a33 | attack:A/SEX | PWGTP | 0.685160028 | -0.00132442457 | 0.00691182864 |
| T0_rr_0.75_a33 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| T0_rr_0.75_a33 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| T0_rr_0.75_a33 | attack:AB/SEX | unweighted | 0.674960759 | -0.0102561757 | 0.0061916779 |
| T0_rr_0.75_a33 | attack:AB/SEX | PWGTP | 0.681147494 | -0.0032136318 | 0.00737440533 |
| T0_rr_0.75_a33 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| T0_rr_0.75_a33 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| T0_rr_0.75_a33 | attack:A/public_coverage | unweighted | 0.519811889 | 0.00049196664 | 0.00819074635 |
| T0_rr_0.75_a33 | attack:A/public_coverage | PWGTP | 0.522239617 | 0.000704291 | 0.00720473467 |
| T0_rr_0.75_a33 | attack:A/commute_over20 | unweighted | 0.683163963 | 1.50193284e-05 | 0.00202383875 |
| T0_rr_0.75_a33 | attack:A/commute_over20 | PWGTP | 0.686133147 | -0.000409726007 | 0.00316224593 |
| T0_rr_0.75_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_rr_0.75_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_rr_0.75_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_rr_0.75_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_rr_0.75_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_rr_0.75_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_rr_0.75_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_rr_0.75_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_rr_0.75_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_rr_0.75_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_rr_0.75_a33 | utility:A/same_residence | unweighted | 0.500802137 | -0.0243495209 | 0.00135358366 |
| T0_rr_0.75_a33 | utility:A/same_residence | PWGTP | 0.476839219 | -0.0222298761 | -1.49814277e-05 |
| T0_rr_0.75_a33 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| T0_rr_0.75_a33 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| T0_rr_0.75_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| T0_rr_0.75_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| T0_rr_0.75_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_rr_0.75_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_rr_0.75_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_rr_0.75_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_withhold_0.25 | attack:A/SEX | unweighted | 0.684851348 | -0.00304231294 | 0.0127051636 |
| T0_withhold_0.25 | attack:A/SEX | PWGTP | 0.685453659 | -0.00103079375 | 0.00720545946 |
| T0_withhold_0.25 | attack:A/RAC1P | unweighted | 1.26992412 | -0.000815238911 | 0.00571030818 |
| T0_withhold_0.25 | attack:A/RAC1P | PWGTP | 1.26423224 | -0.00050739769 | 0.00577787725 |
| T0_withhold_0.25 | attack:AB/SEX | unweighted | 0.681797576 | -0.00341935833 | 0.0130284952 |
| T0_withhold_0.25 | attack:AB/SEX | PWGTP | 0.684028344 | -0.000332781486 | 0.0102552556 |
| T0_withhold_0.25 | attack:AB/RAC1P | unweighted | 1.25653352 | -0.00107149086 | 0.00443290342 |
| T0_withhold_0.25 | attack:AB/RAC1P | PWGTP | 1.25172204 | -0.000630189066 | 0.00579381407 |
| T0_withhold_0.25 | attack:A/public_coverage | unweighted | 0.519545197 | 0.000225274522 | 0.00792405424 |
| T0_withhold_0.25 | attack:A/public_coverage | PWGTP | 0.521905398 | 0.000370072402 | 0.00687051607 |
| T0_withhold_0.25 | attack:A/commute_over20 | unweighted | 0.683233443 | 8.44987588e-05 | 0.00209331818 |
| T0_withhold_0.25 | attack:A/commute_over20 | PWGTP | 0.686509567 | -3.33057892e-05 | 0.00353866615 |
| T0_withhold_0.25 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_withhold_0.25 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_withhold_0.25 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_withhold_0.25 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_withhold_0.25 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_withhold_0.25 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_withhold_0.25 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_withhold_0.25 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_withhold_0.25 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_withhold_0.25 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_withhold_0.25 | utility:A/same_residence | unweighted | 0.51478065 | -0.0103710078 | 0.0153320967 |
| T0_withhold_0.25 | utility:A/same_residence | PWGTP | 0.489185432 | -0.009883663 | 0.0123312317 |
| T0_withhold_0.25 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| T0_withhold_0.25 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| T0_withhold_0.25 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| T0_withhold_0.25 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| T0_withhold_0.25 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_withhold_0.25 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_withhold_0.25 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_withhold_0.25 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_withhold_0.25_a33 | attack:A/SEX | unweighted | 0.684547997 | -0.00334566309 | 0.0124018134 |
| T0_withhold_0.25_a33 | attack:A/SEX | PWGTP | 0.684938574 | -0.00154587861 | 0.00669037461 |
| T0_withhold_0.25_a33 | attack:A/RAC1P | unweighted | 1.27027992 | -0.000459440615 | 0.00606610647 |
| T0_withhold_0.25_a33 | attack:A/RAC1P | PWGTP | 1.26440417 | -0.000335466384 | 0.00594980856 |
| T0_withhold_0.25_a33 | attack:AB/SEX | unweighted | 0.684414357 | -0.000802577655 | 0.0156452759 |
| T0_withhold_0.25_a33 | attack:AB/SEX | PWGTP | 0.683599298 | -0.000761827497 | 0.00982620964 |
| T0_withhold_0.25_a33 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| T0_withhold_0.25_a33 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| T0_withhold_0.25_a33 | attack:A/public_coverage | unweighted | 0.519784868 | 0.000464945874 | 0.00816372559 |
| T0_withhold_0.25_a33 | attack:A/public_coverage | PWGTP | 0.522101836 | 0.000566510224 | 0.0070669539 |
| T0_withhold_0.25_a33 | attack:A/commute_over20 | unweighted | 0.683179188 | 3.02439334e-05 | 0.00203906336 |
| T0_withhold_0.25_a33 | attack:A/commute_over20 | PWGTP | 0.68637223 | -0.000170642316 | 0.00340132962 |
| T0_withhold_0.25_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_withhold_0.25_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_withhold_0.25_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_withhold_0.25_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_withhold_0.25_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_withhold_0.25_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_withhold_0.25_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_withhold_0.25_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_withhold_0.25_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_withhold_0.25_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_withhold_0.25_a33 | utility:A/same_residence | unweighted | 0.515105397 | -0.0100462604 | 0.0156568441 |
| T0_withhold_0.25_a33 | utility:A/same_residence | PWGTP | 0.489047464 | -0.0100216312 | 0.0121932635 |
| T0_withhold_0.25_a33 | utility:A/income_binary | unweighted | 0.295921354 | 0.000222909772 | 0.0114041784 |
| T0_withhold_0.25_a33 | utility:A/income_binary | PWGTP | 0.30773423 | 0.000154643695 | 0.0113265826 |
| T0_withhold_0.25_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| T0_withhold_0.25_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| T0_withhold_0.25_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_withhold_0.25_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_withhold_0.25_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_withhold_0.25_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_withhold_0.5 | attack:A/SEX | unweighted | 0.686023672 | -0.00186998826 | 0.0138774882 |
| T0_withhold_0.5 | attack:A/SEX | PWGTP | 0.686398729 | -8.57235325e-05 | 0.00815052968 |
| T0_withhold_0.5 | attack:A/RAC1P | unweighted | 1.27117767 | 0.000438315282 | 0.00696386237 |
| T0_withhold_0.5 | attack:A/RAC1P | PWGTP | 1.26511275 | 0.000373113035 | 0.00665838798 |
| T0_withhold_0.5 | attack:AB/SEX | unweighted | 0.682669638 | -0.00254729672 | 0.0139005568 |
| T0_withhold_0.5 | attack:AB/SEX | PWGTP | 0.681783811 | -0.00257731449 | 0.00801072264 |
| T0_withhold_0.5 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| T0_withhold_0.5 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| T0_withhold_0.5 | attack:A/public_coverage | unweighted | 0.520114525 | 0.000794601958 | 0.00849338167 |
| T0_withhold_0.5 | attack:A/public_coverage | PWGTP | 0.522678046 | 0.00114271997 | 0.00764316364 |
| T0_withhold_0.5 | attack:A/commute_over20 | unweighted | 0.683219301 | 7.03570568e-05 | 0.00207917648 |
| T0_withhold_0.5 | attack:A/commute_over20 | PWGTP | 0.686430294 | -0.000112578928 | 0.00345939301 |
| T0_withhold_0.5 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_withhold_0.5 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_withhold_0.5 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_withhold_0.5 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_withhold_0.5 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_withhold_0.5 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_withhold_0.5 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_withhold_0.5 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_withhold_0.5 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_withhold_0.5 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_withhold_0.5 | utility:A/same_residence | unweighted | 0.503837196 | -0.0213144615 | 0.00438864307 |
| T0_withhold_0.5 | utility:A/same_residence | PWGTP | 0.478646579 | -0.0204225159 | 0.00179237882 |
| T0_withhold_0.5 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| T0_withhold_0.5 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| T0_withhold_0.5 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| T0_withhold_0.5 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| T0_withhold_0.5 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_withhold_0.5 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_withhold_0.5 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_withhold_0.5 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_withhold_0.5_a33 | attack:A/SEX | unweighted | 0.683797192 | -0.00409646899 | 0.0116510075 |
| T0_withhold_0.5_a33 | attack:A/SEX | PWGTP | 0.68739468 | 0.000910227467 | 0.00914648068 |
| T0_withhold_0.5_a33 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| T0_withhold_0.5_a33 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| T0_withhold_0.5_a33 | attack:AB/SEX | unweighted | 0.677276813 | -0.0079401217 | 0.00850773185 |
| T0_withhold_0.5_a33 | attack:AB/SEX | PWGTP | 0.68199883 | -0.00236229619 | 0.00822574094 |
| T0_withhold_0.5_a33 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| T0_withhold_0.5_a33 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| T0_withhold_0.5_a33 | attack:A/public_coverage | unweighted | 0.520909907 | 0.00158998393 | 0.00928876365 |
| T0_withhold_0.5_a33 | attack:A/public_coverage | PWGTP | 0.523256062 | 0.00172073552 | 0.00822117919 |
| T0_withhold_0.5_a33 | attack:A/commute_over20 | unweighted | 0.683202948 | 5.40034019e-05 | 0.00206282283 |
| T0_withhold_0.5_a33 | attack:A/commute_over20 | PWGTP | 0.68622491 | -0.000317962497 | 0.00325400944 |
| T0_withhold_0.5_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_withhold_0.5_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_withhold_0.5_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_withhold_0.5_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_withhold_0.5_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_withhold_0.5_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_withhold_0.5_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_withhold_0.5_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_withhold_0.5_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_withhold_0.5_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_withhold_0.5_a33 | utility:A/same_residence | unweighted | 0.503537974 | -0.0216136841 | 0.00408942047 |
| T0_withhold_0.5_a33 | utility:A/same_residence | PWGTP | 0.477868452 | -0.0212006433 | 0.00101425143 |
| T0_withhold_0.5_a33 | utility:A/income_binary | unweighted | 0.296236962 | 0.000538517847 | 0.0117197865 |
| T0_withhold_0.5_a33 | utility:A/income_binary | PWGTP | 0.307939829 | 0.000360242322 | 0.0115321813 |
| T0_withhold_0.5_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| T0_withhold_0.5_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| T0_withhold_0.5_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_withhold_0.5_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_withhold_0.5_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_withhold_0.5_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_withhold_0.75 | attack:A/SEX | unweighted | 0.685767559 | -0.00212610122 | 0.0136213753 |
| T0_withhold_0.75 | attack:A/SEX | PWGTP | 0.683618882 | -0.00286557063 | 0.00537068259 |
| T0_withhold_0.75 | attack:A/RAC1P | unweighted | 1.27110334 | 0.000363980158 | 0.00688952725 |
| T0_withhold_0.75 | attack:A/RAC1P | PWGTP | 1.2649678 | 0.000228155299 | 0.00651343024 |
| T0_withhold_0.75 | attack:AB/SEX | unweighted | 0.677293687 | -0.00792324813 | 0.00852460542 |
| T0_withhold_0.75 | attack:AB/SEX | PWGTP | 0.679402837 | -0.0049582887 | 0.00562974844 |
| T0_withhold_0.75 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| T0_withhold_0.75 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| T0_withhold_0.75 | attack:A/public_coverage | unweighted | 0.521086194 | 0.00176627128 | 0.00946505099 |
| T0_withhold_0.75 | attack:A/public_coverage | PWGTP | 0.523859621 | 0.00232429499 | 0.00882473866 |
| T0_withhold_0.75 | attack:A/commute_over20 | unweighted | 0.683238345 | 8.9400827e-05 | 0.00209822025 |
| T0_withhold_0.75 | attack:A/commute_over20 | PWGTP | 0.686384241 | -0.000158632009 | 0.00341333993 |
| T0_withhold_0.75 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_withhold_0.75 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_withhold_0.75 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_withhold_0.75 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_withhold_0.75 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_withhold_0.75 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_withhold_0.75 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_withhold_0.75 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_withhold_0.75 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_withhold_0.75 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_withhold_0.75 | utility:A/same_residence | unweighted | 0.493073742 | -0.0320779156 | -0.00637481109 |
| T0_withhold_0.75 | utility:A/same_residence | PWGTP | 0.468690138 | -0.0303789568 | -0.00816406214 |
| T0_withhold_0.75 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| T0_withhold_0.75 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| T0_withhold_0.75 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| T0_withhold_0.75 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| T0_withhold_0.75 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_withhold_0.75 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_withhold_0.75 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_withhold_0.75 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| T0_withhold_0.75_a33 | attack:A/SEX | unweighted | 0.68296466 | -0.00492900102 | 0.0108184755 |
| T0_withhold_0.75_a33 | attack:A/SEX | PWGTP | 0.686510143 | 2.56909552e-05 | 0.00826194417 |
| T0_withhold_0.75_a33 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| T0_withhold_0.75_a33 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| T0_withhold_0.75_a33 | attack:AB/SEX | unweighted | 0.674932017 | -0.0102849181 | 0.00616293549 |
| T0_withhold_0.75_a33 | attack:AB/SEX | PWGTP | 0.681513456 | -0.00284767029 | 0.00774036684 |
| T0_withhold_0.75_a33 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| T0_withhold_0.75_a33 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| T0_withhold_0.75_a33 | attack:A/public_coverage | unweighted | 0.521284466 | 0.00196454328 | 0.00966332299 |
| T0_withhold_0.75_a33 | attack:A/public_coverage | PWGTP | 0.523742987 | 0.00220766084 | 0.00870810451 |
| T0_withhold_0.75_a33 | attack:A/commute_over20 | unweighted | 0.683187136 | 3.81920869e-05 | 0.00204701151 |
| T0_withhold_0.75_a33 | attack:A/commute_over20 | PWGTP | 0.686035478 | -0.000507395004 | 0.00306457693 |
| T0_withhold_0.75_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| T0_withhold_0.75_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| T0_withhold_0.75_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| T0_withhold_0.75_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| T0_withhold_0.75_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| T0_withhold_0.75_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| T0_withhold_0.75_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| T0_withhold_0.75_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| T0_withhold_0.75_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| T0_withhold_0.75_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| T0_withhold_0.75_a33 | utility:A/same_residence | unweighted | 0.492524991 | -0.032626667 | -0.00692356243 |
| T0_withhold_0.75_a33 | utility:A/same_residence | PWGTP | 0.468222122 | -0.0308469727 | -0.00863207798 |
| T0_withhold_0.75_a33 | utility:A/income_binary | unweighted | 0.296660591 | 0.00096214696 | 0.0121434156 |
| T0_withhold_0.75_a33 | utility:A/income_binary | PWGTP | 0.308228778 | 0.000649191163 | 0.0118211301 |
| T0_withhold_0.75_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| T0_withhold_0.75_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| T0_withhold_0.75_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| T0_withhold_0.75_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| T0_withhold_0.75_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| T0_withhold_0.75_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_C_0.0005_a17 | attack:A/SEX | unweighted | 0.687765251 | -0.000128409662 | 0.0156190668 |
| Trisk_C_0.0005_a17 | attack:A/SEX | PWGTP | 0.686096407 | -0.000388044915 | 0.0078482083 |
| Trisk_C_0.0005_a17 | attack:A/RAC1P | unweighted | 1.2706135 | -0.000125860941 | 0.00639968615 |
| Trisk_C_0.0005_a17 | attack:A/RAC1P | PWGTP | 1.26487696 | 0.000137318276 | 0.00642259322 |
| Trisk_C_0.0005_a17 | attack:AB/SEX | unweighted | 0.683078773 | -0.00213816133 | 0.0143096922 |
| Trisk_C_0.0005_a17 | attack:AB/SEX | PWGTP | 0.685064658 | 0.00070353264 | 0.0112915698 |
| Trisk_C_0.0005_a17 | attack:AB/RAC1P | unweighted | 1.25742026 | -0.000184742105 | 0.00531965217 |
| Trisk_C_0.0005_a17 | attack:AB/RAC1P | PWGTP | 1.25247013 | 0.000117904553 | 0.00654190769 |
| Trisk_C_0.0005_a17 | attack:A/public_coverage | unweighted | 0.518403859 | -0.000916063815 | 0.0067827159 |
| Trisk_C_0.0005_a17 | attack:A/public_coverage | PWGTP | 0.520548261 | -0.000987064648 | 0.00551337902 |
| Trisk_C_0.0005_a17 | attack:A/commute_over20 | unweighted | 0.683185504 | 3.65603034e-05 | 0.00204537973 |
| Trisk_C_0.0005_a17 | attack:A/commute_over20 | PWGTP | 0.68644107 | -0.000101802738 | 0.0034701692 |
| Trisk_C_0.0005_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_C_0.0005_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_C_0.0005_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_C_0.0005_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_C_0.0005_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_C_0.0005_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_C_0.0005_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_C_0.0005_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_C_0.0005_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_C_0.0005_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_C_0.0005_a17 | utility:A/same_residence | unweighted | 0.512689901 | -0.0124617569 | 0.0132413477 |
| Trisk_C_0.0005_a17 | utility:A/same_residence | PWGTP | 0.487482392 | -0.011586703 | 0.0106281917 |
| Trisk_C_0.0005_a17 | utility:A/income_binary | unweighted | 0.295432023 | -0.000266421643 | 0.010914847 |
| Trisk_C_0.0005_a17 | utility:A/income_binary | PWGTP | 0.307302905 | -0.000276681069 | 0.0108952579 |
| Trisk_C_0.0005_a17 | utility:A/civilian_at_work | unweighted | 0.279339775 | 3.25823731e-05 | 0.00790253557 |
| Trisk_C_0.0005_a17 | utility:A/civilian_at_work | PWGTP | 0.267804456 | -1.0591583e-05 | 0.00533382468 |
| Trisk_C_0.0005_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_C_0.0005_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_C_0.0005_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_C_0.0005_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_C_0.0005_a17_fineC | attack:A/SEX | unweighted | 0.687375765 | -0.000517896012 | 0.0152295805 |
| Trisk_C_0.0005_a17_fineC | attack:A/SEX | PWGTP | 0.685863691 | -0.000620761373 | 0.00761549184 |
| Trisk_C_0.0005_a17_fineC | attack:A/RAC1P | unweighted | 1.27073418 | -5.17726811e-06 | 0.00652036982 |
| Trisk_C_0.0005_a17_fineC | attack:A/RAC1P | PWGTP | 1.26492536 | 0.000185715796 | 0.00647099074 |
| Trisk_C_0.0005_a17_fineC | attack:AB/SEX | unweighted | 0.682878464 | -0.00233847112 | 0.0141093824 |
| Trisk_C_0.0005_a17_fineC | attack:AB/SEX | PWGTP | 0.684353904 | -7.22234369e-06 | 0.0105808148 |
| Trisk_C_0.0005_a17_fineC | attack:AB/RAC1P | unweighted | 1.25736486 | -0.00024014477 | 0.00526424951 |
| Trisk_C_0.0005_a17_fineC | attack:AB/RAC1P | PWGTP | 1.25206993 | -0.000282297936 | 0.0061417052 |
| Trisk_C_0.0005_a17_fineC | attack:A/public_coverage | unweighted | 0.518588612 | -0.000731310761 | 0.00696746895 |
| Trisk_C_0.0005_a17_fineC | attack:A/public_coverage | PWGTP | 0.520744202 | -0.000791123888 | 0.00570931978 |
| Trisk_C_0.0005_a17_fineC | attack:A/commute_over20 | unweighted | 0.683132631 | -1.63129791e-05 | 0.00199250644 |
| Trisk_C_0.0005_a17_fineC | attack:A/commute_over20 | PWGTP | 0.686437284 | -0.000105588221 | 0.00346638372 |
| Trisk_C_0.0005_a17_fineC | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_C_0.0005_a17_fineC | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_C_0.0005_a17_fineC | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_C_0.0005_a17_fineC | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_C_0.0005_a17_fineC | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_C_0.0005_a17_fineC | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_C_0.0005_a17_fineC | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_C_0.0005_a17_fineC | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_C_0.0005_a17_fineC | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_C_0.0005_a17_fineC | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_C_0.0005_a17_fineC | utility:A/same_residence | unweighted | 0.516962587 | -0.00818907056 | 0.017514034 |
| Trisk_C_0.0005_a17_fineC | utility:A/same_residence | PWGTP | 0.490828328 | -0.00824076683 | 0.0139741279 |
| Trisk_C_0.0005_a17_fineC | utility:A/income_binary | unweighted | 0.29565314 | -4.53048617e-05 | 0.0111359638 |
| Trisk_C_0.0005_a17_fineC | utility:A/income_binary | PWGTP | 0.307561474 | -1.81120693e-05 | 0.0111538269 |
| Trisk_C_0.0005_a17_fineC | utility:A/civilian_at_work | unweighted | 0.279291491 | -1.57013539e-05 | 0.00785425184 |
| Trisk_C_0.0005_a17_fineC | utility:A/civilian_at_work | PWGTP | 0.267749584 | -6.54632133e-05 | 0.00527895305 |
| Trisk_C_0.0005_a17_fineC | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_C_0.0005_a17_fineC | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_C_0.0005_a17_fineC | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_C_0.0005_a17_fineC | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_C_0.002_a17 | attack:A/SEX | unweighted | 0.687089432 | -0.000804228865 | 0.0149432476 |
| Trisk_C_0.002_a17 | attack:A/SEX | PWGTP | 0.685486891 | -0.000997561381 | 0.00723869183 |
| Trisk_C_0.002_a17 | attack:A/RAC1P | unweighted | 1.26945177 | -0.0012875946 | 0.00523795248 |
| Trisk_C_0.002_a17 | attack:A/RAC1P | PWGTP | 1.26411957 | -0.000620069916 | 0.00566520503 |
| Trisk_C_0.002_a17 | attack:AB/SEX | unweighted | 0.683203553 | -0.00201338181 | 0.0144344717 |
| Trisk_C_0.002_a17 | attack:AB/SEX | PWGTP | 0.68497576 | 0.000614633803 | 0.0112026709 |
| Trisk_C_0.002_a17 | attack:AB/RAC1P | unweighted | 1.25700002 | -0.000604982694 | 0.00489941158 |
| Trisk_C_0.002_a17 | attack:AB/RAC1P | PWGTP | 1.25277557 | 0.000423342942 | 0.00684734608 |
| Trisk_C_0.002_a17 | attack:A/public_coverage | unweighted | 0.518422082 | -0.00089784102 | 0.00680093869 |
| Trisk_C_0.002_a17 | attack:A/public_coverage | PWGTP | 0.520557461 | -0.00097786491 | 0.00552257876 |
| Trisk_C_0.002_a17 | attack:A/commute_over20 | unweighted | 0.682505216 | -0.000643727763 | 0.00136509166 |
| Trisk_C_0.002_a17 | attack:A/commute_over20 | PWGTP | 0.685926524 | -0.000616348749 | 0.00295562319 |
| Trisk_C_0.002_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_C_0.002_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_C_0.002_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_C_0.002_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_C_0.002_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_C_0.002_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_C_0.002_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_C_0.002_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_C_0.002_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_C_0.002_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_C_0.002_a17 | utility:A/same_residence | unweighted | 0.499118879 | -0.0260327784 | -0.000329673877 |
| Trisk_C_0.002_a17 | utility:A/same_residence | PWGTP | 0.474893771 | -0.0241753238 | -0.00196042907 |
| Trisk_C_0.002_a17 | utility:A/income_binary | unweighted | 0.295563657 | -0.000134787023 | 0.0110464816 |
| Trisk_C_0.002_a17 | utility:A/income_binary | PWGTP | 0.307049757 | -0.000529829121 | 0.0106421098 |
| Trisk_C_0.002_a17 | utility:A/civilian_at_work | unweighted | 0.279007961 | -0.000299231362 | 0.00757072184 |
| Trisk_C_0.002_a17 | utility:A/civilian_at_work | PWGTP | 0.26766454 | -0.000150507295 | 0.00519390896 |
| Trisk_C_0.002_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_C_0.002_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_C_0.002_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_C_0.002_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_C_0.002_a17_fineC | attack:A/SEX | unweighted | 0.686526089 | -0.00136757179 | 0.0143799047 |
| Trisk_C_0.002_a17_fineC | attack:A/SEX | PWGTP | 0.685162029 | -0.0013224231 | 0.00691383012 |
| Trisk_C_0.002_a17_fineC | attack:A/RAC1P | unweighted | 1.27084001 | 0.0001006524 | 0.00662619949 |
| Trisk_C_0.002_a17_fineC | attack:A/RAC1P | PWGTP | 1.26466816 | -7.14791108e-05 | 0.00621379583 |
| Trisk_C_0.002_a17_fineC | attack:AB/SEX | unweighted | 0.685120226 | -9.6708511e-05 | 0.016351145 |
| Trisk_C_0.002_a17_fineC | attack:AB/SEX | PWGTP | 0.684146862 | -0.000214263621 | 0.0103737735 |
| Trisk_C_0.002_a17_fineC | attack:AB/RAC1P | unweighted | 1.25767345 | 6.84462332e-05 | 0.00557284051 |
| Trisk_C_0.002_a17_fineC | attack:AB/RAC1P | PWGTP | 1.2525554 | 0.000203166779 | 0.00662716991 |
| Trisk_C_0.002_a17_fineC | attack:A/public_coverage | unweighted | 0.518599381 | -0.000720542037 | 0.00697823768 |
| Trisk_C_0.002_a17_fineC | attack:A/public_coverage | PWGTP | 0.520600757 | -0.000934569414 | 0.00556587426 |
| Trisk_C_0.002_a17_fineC | attack:A/commute_over20 | unweighted | 0.682549477 | -0.000599466942 | 0.00140935248 |
| Trisk_C_0.002_a17_fineC | attack:A/commute_over20 | PWGTP | 0.685682567 | -0.000860305198 | 0.00271166674 |
| Trisk_C_0.002_a17_fineC | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_C_0.002_a17_fineC | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_C_0.002_a17_fineC | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_C_0.002_a17_fineC | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_C_0.002_a17_fineC | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_C_0.002_a17_fineC | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_C_0.002_a17_fineC | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_C_0.002_a17_fineC | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_C_0.002_a17_fineC | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_C_0.002_a17_fineC | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_C_0.002_a17_fineC | utility:A/same_residence | unweighted | 0.506535812 | -0.0186158462 | 0.00708725839 |
| Trisk_C_0.002_a17_fineC | utility:A/same_residence | PWGTP | 0.481877826 | -0.0171912691 | 0.00502362562 |
| Trisk_C_0.002_a17_fineC | utility:A/income_binary | unweighted | 0.295450254 | -0.000248190659 | 0.010933078 |
| Trisk_C_0.002_a17_fineC | utility:A/income_binary | PWGTP | 0.30733186 | -0.000247726819 | 0.0109242121 |
| Trisk_C_0.002_a17_fineC | utility:A/civilian_at_work | unweighted | 0.279183556 | -0.00012363622 | 0.00774631698 |
| Trisk_C_0.002_a17_fineC | utility:A/civilian_at_work | PWGTP | 0.267780142 | -3.4905287e-05 | 0.00530951097 |
| Trisk_C_0.002_a17_fineC | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_C_0.002_a17_fineC | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_C_0.002_a17_fineC | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_C_0.002_a17_fineC | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_C_0.002_a33 | attack:A/SEX | unweighted | 0.684523982 | -0.00336967833 | 0.0123777982 |
| Trisk_C_0.002_a33 | attack:A/SEX | PWGTP | 0.685690756 | -0.000793696605 | 0.00744255661 |
| Trisk_C_0.002_a33 | attack:A/RAC1P | unweighted | 1.2699798 | -0.000759563622 | 0.00576598347 |
| Trisk_C_0.002_a33 | attack:A/RAC1P | PWGTP | 1.26396093 | -0.000778713875 | 0.00550656107 |
| Trisk_C_0.002_a33 | attack:AB/SEX | unweighted | 0.684962449 | -0.000254486083 | 0.0161933675 |
| Trisk_C_0.002_a33 | attack:AB/SEX | PWGTP | 0.684429489 | 6.8363409e-05 | 0.0106564005 |
| Trisk_C_0.002_a33 | attack:AB/RAC1P | unweighted | 1.25669998 | -0.000905022951 | 0.00459937133 |
| Trisk_C_0.002_a33 | attack:AB/RAC1P | PWGTP | 1.25234515 | -7.08374427e-06 | 0.00641691939 |
| Trisk_C_0.002_a33 | attack:A/public_coverage | unweighted | 0.518614299 | -0.000705623362 | 0.00699315635 |
| Trisk_C_0.002_a33 | attack:A/public_coverage | PWGTP | 0.520706738 | -0.000828588081 | 0.00567185559 |
| Trisk_C_0.002_a33 | attack:A/commute_over20 | unweighted | 0.682283651 | -0.000865293237 | 0.00114352619 |
| Trisk_C_0.002_a33 | attack:A/commute_over20 | PWGTP | 0.685773657 | -0.000769215962 | 0.00280275598 |
| Trisk_C_0.002_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_C_0.002_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_C_0.002_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_C_0.002_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_C_0.002_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_C_0.002_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_C_0.002_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_C_0.002_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_C_0.002_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_C_0.002_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_C_0.002_a33 | utility:A/same_residence | unweighted | 0.498950111 | -0.0262015468 | -0.000498442288 |
| Trisk_C_0.002_a33 | utility:A/same_residence | PWGTP | 0.474875322 | -0.0241937728 | -0.00197887807 |
| Trisk_C_0.002_a33 | utility:A/income_binary | unweighted | 0.295737275 | 3.88303017e-05 | 0.011220099 |
| Trisk_C_0.002_a33 | utility:A/income_binary | PWGTP | 0.307685704 | 0.000106117504 | 0.0112780565 |
| Trisk_C_0.002_a33 | utility:A/civilian_at_work | unweighted | 0.279104605 | -0.00020258763 | 0.00766736557 |
| Trisk_C_0.002_a33 | utility:A/civilian_at_work | PWGTP | 0.267717094 | -9.79528113e-05 | 0.00524646345 |
| Trisk_C_0.002_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_C_0.002_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_C_0.002_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_C_0.002_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_C_0.01_a17 | attack:A/SEX | unweighted | 0.681933237 | -0.00596042352 | 0.00978705298 |
| Trisk_C_0.01_a17 | attack:A/SEX | PWGTP | 0.681262537 | -0.00522191525 | 0.00301433796 |
| Trisk_C_0.01_a17 | attack:A/RAC1P | unweighted | 1.26991699 | -0.000822372475 | 0.00570317461 |
| Trisk_C_0.01_a17 | attack:A/RAC1P | PWGTP | 1.26288168 | -0.00185796177 | 0.00442731318 |
| Trisk_C_0.01_a17 | attack:AB/SEX | unweighted | 0.677759729 | -0.00745720613 | 0.00899064742 |
| Trisk_C_0.01_a17 | attack:AB/SEX | PWGTP | 0.679119373 | -0.00524175239 | 0.00534628474 |
| Trisk_C_0.01_a17 | attack:AB/RAC1P | unweighted | 1.25571836 | -0.00188664422 | 0.00361775006 |
| Trisk_C_0.01_a17 | attack:AB/RAC1P | PWGTP | 1.25077332 | -0.00157890697 | 0.00484509617 |
| Trisk_C_0.01_a17 | attack:A/public_coverage | unweighted | 0.517958883 | -0.00136103995 | 0.00633773977 |
| Trisk_C_0.01_a17 | attack:A/public_coverage | PWGTP | 0.519696396 | -0.00183893034 | 0.00466151333 |
| Trisk_C_0.01_a17 | attack:A/commute_over20 | unweighted | 0.682659036 | -0.000489907765 | 0.00151891166 |
| Trisk_C_0.01_a17 | attack:A/commute_over20 | PWGTP | 0.685345055 | -0.00119781718 | 0.00237415475 |
| Trisk_C_0.01_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_C_0.01_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_C_0.01_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_C_0.01_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_C_0.01_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_C_0.01_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_C_0.01_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_C_0.01_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_C_0.01_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_C_0.01_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_C_0.01_a17 | utility:A/same_residence | unweighted | 0.488200611 | -0.0369510464 | -0.0112479418 |
| Trisk_C_0.01_a17 | utility:A/same_residence | PWGTP | 0.465577633 | -0.0334914625 | -0.0112765678 |
| Trisk_C_0.01_a17 | utility:A/income_binary | unweighted | 0.295646984 | -5.14606682e-05 | 0.011129808 |
| Trisk_C_0.01_a17 | utility:A/income_binary | PWGTP | 0.308201324 | 0.000621737089 | 0.011793676 |
| Trisk_C_0.01_a17 | utility:A/civilian_at_work | unweighted | 0.278829101 | -0.000478091257 | 0.00739186194 |
| Trisk_C_0.01_a17 | utility:A/civilian_at_work | PWGTP | 0.267440649 | -0.000374398257 | 0.004970018 |
| Trisk_C_0.01_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_C_0.01_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_C_0.01_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_C_0.01_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_C_0.01_a17_fineC | attack:A/SEX | unweighted | 0.683681609 | -0.00421205183 | 0.0115354247 |
| Trisk_C_0.01_a17_fineC | attack:A/SEX | PWGTP | 0.684030965 | -0.00245348735 | 0.00578276586 |
| Trisk_C_0.01_a17_fineC | attack:A/RAC1P | unweighted | 1.26940737 | -0.00133199265 | 0.00519355443 |
| Trisk_C_0.01_a17_fineC | attack:A/RAC1P | PWGTP | 1.26326684 | -0.00147280395 | 0.004812471 |
| Trisk_C_0.01_a17_fineC | attack:AB/SEX | unweighted | 0.680287978 | -0.00492895648 | 0.0115188971 |
| Trisk_C_0.01_a17_fineC | attack:AB/SEX | PWGTP | 0.680371101 | -0.0039900251 | 0.00659801204 |
| Trisk_C_0.01_a17_fineC | attack:AB/RAC1P | unweighted | 1.25724844 | -0.000356566252 | 0.00514782802 |
| Trisk_C_0.01_a17_fineC | attack:AB/RAC1P | PWGTP | 1.25265877 | 0.000306538833 | 0.00673054197 |
| Trisk_C_0.01_a17_fineC | attack:A/public_coverage | unweighted | 0.517600678 | -0.00171924466 | 0.00597953505 |
| Trisk_C_0.01_a17_fineC | attack:A/public_coverage | PWGTP | 0.520021763 | -0.0015135632 | 0.00498688048 |
| Trisk_C_0.01_a17_fineC | attack:A/commute_over20 | unweighted | 0.682373219 | -0.000775724757 | 0.00123309467 |
| Trisk_C_0.01_a17_fineC | attack:A/commute_over20 | PWGTP | 0.685237836 | -0.00130503647 | 0.00226693547 |
| Trisk_C_0.01_a17_fineC | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_C_0.01_a17_fineC | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_C_0.01_a17_fineC | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_C_0.01_a17_fineC | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_C_0.01_a17_fineC | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_C_0.01_a17_fineC | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_C_0.01_a17_fineC | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_C_0.01_a17_fineC | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_C_0.01_a17_fineC | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_C_0.01_a17_fineC | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_C_0.01_a17_fineC | utility:A/same_residence | unweighted | 0.490284828 | -0.0348668302 | -0.00916372567 |
| Trisk_C_0.01_a17_fineC | utility:A/same_residence | PWGTP | 0.467215886 | -0.0318532087 | -0.00963831398 |
| Trisk_C_0.01_a17_fineC | utility:A/income_binary | unweighted | 0.295475496 | -0.00022294883 | 0.0109583198 |
| Trisk_C_0.01_a17_fineC | utility:A/income_binary | PWGTP | 0.307131152 | -0.000448434215 | 0.0107235047 |
| Trisk_C_0.01_a17_fineC | utility:A/civilian_at_work | unweighted | 0.278967627 | -0.000339565771 | 0.00753038743 |
| Trisk_C_0.01_a17_fineC | utility:A/civilian_at_work | PWGTP | 0.267542533 | -0.000272513918 | 0.00507190234 |
| Trisk_C_0.01_a17_fineC | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_C_0.01_a17_fineC | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_C_0.01_a17_fineC | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_C_0.01_a17_fineC | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_C_0_a17 | attack:A/SEX | unweighted | 0.687142071 | -0.000751589556 | 0.0149958869 |
| Trisk_C_0_a17 | attack:A/SEX | PWGTP | 0.685786042 | -0.0006984105 | 0.00753784271 |
| Trisk_C_0_a17 | attack:A/RAC1P | unweighted | 1.26966439 | -0.00107496686 | 0.00545058023 |
| Trisk_C_0_a17 | attack:A/RAC1P | PWGTP | 1.26450732 | -0.00023231783 | 0.00605295711 |
| Trisk_C_0_a17 | attack:AB/SEX | unweighted | 0.684847141 | -0.000369793555 | 0.01607806 |
| Trisk_C_0_a17 | attack:AB/SEX | PWGTP | 0.683903232 | -0.000457894121 | 0.010130143 |
| Trisk_C_0_a17 | attack:AB/RAC1P | unweighted | 1.25750642 | -9.85884652e-05 | 0.00540580581 |
| Trisk_C_0_a17 | attack:AB/RAC1P | PWGTP | 1.25196813 | -0.000384102874 | 0.00603990026 |
| Trisk_C_0_a17 | attack:A/public_coverage | unweighted | 0.518879675 | -0.000440247216 | 0.0072585325 |
| Trisk_C_0_a17 | attack:A/public_coverage | PWGTP | 0.521169682 | -0.000365644454 | 0.00613479922 |
| Trisk_C_0_a17 | attack:A/commute_over20 | unweighted | 0.682885357 | -0.000263587228 | 0.0017452322 |
| Trisk_C_0_a17 | attack:A/commute_over20 | PWGTP | 0.686293269 | -0.000249603913 | 0.00332236803 |
| Trisk_C_0_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_C_0_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_C_0_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_C_0_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_C_0_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_C_0_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_C_0_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_C_0_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_C_0_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_C_0_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_C_0_a17 | utility:A/same_residence | unweighted | 0.524581741 | -0.00056991727 | 0.0251331873 |
| Trisk_C_0_a17 | utility:A/same_residence | PWGTP | 0.498524433 | -0.000544661789 | 0.0216702329 |
| Trisk_C_0_a17 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Trisk_C_0_a17 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Trisk_C_0_a17 | utility:A/civilian_at_work | unweighted | 0.279045002 | -0.000262190502 | 0.0076077627 |
| Trisk_C_0_a17 | utility:A/civilian_at_work | PWGTP | 0.267515342 | -0.000299705281 | 0.00504471098 |
| Trisk_C_0_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_C_0_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_C_0_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_C_0_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_L_0.0005_a17 | attack:A/SEX | unweighted | 0.684988348 | -0.00290531303 | 0.0128421635 |
| Trisk_L_0.0005_a17 | attack:A/SEX | PWGTP | 0.685498013 | -0.000986439753 | 0.00724981346 |
| Trisk_L_0.0005_a17 | attack:A/RAC1P | unweighted | 1.2708332 | 9.38392402e-05 | 0.00661938633 |
| Trisk_L_0.0005_a17 | attack:A/RAC1P | PWGTP | 1.26490277 | 0.000163130032 | 0.00644840497 |
| Trisk_L_0.0005_a17 | attack:AB/SEX | unweighted | 0.685107017 | -0.000109917342 | 0.0163379362 |
| Trisk_L_0.0005_a17 | attack:AB/SEX | PWGTP | 0.684020609 | -0.000340517211 | 0.0102475199 |
| Trisk_L_0.0005_a17 | attack:AB/RAC1P | unweighted | 1.25712998 | -0.000475030742 | 0.00502936353 |
| Trisk_L_0.0005_a17 | attack:AB/RAC1P | PWGTP | 1.25221205 | -0.000140183622 | 0.00628381951 |
| Trisk_L_0.0005_a17 | attack:A/public_coverage | unweighted | 0.51890178 | -0.000418142458 | 0.00728063726 |
| Trisk_L_0.0005_a17 | attack:A/public_coverage | PWGTP | 0.521057498 | -0.000477827918 | 0.00602261575 |
| Trisk_L_0.0005_a17 | attack:A/commute_over20 | unweighted | 0.683292457 | 0.000143512526 | 0.00215233195 |
| Trisk_L_0.0005_a17 | attack:A/commute_over20 | PWGTP | 0.686550438 | 7.56529351e-06 | 0.00357953723 |
| Trisk_L_0.0005_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_L_0.0005_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_L_0.0005_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_L_0.0005_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_L_0.0005_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_L_0.0005_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_L_0.0005_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_L_0.0005_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_L_0.0005_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_L_0.0005_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_L_0.0005_a17 | utility:A/same_residence | unweighted | 0.510190075 | -0.0149615827 | 0.0107415218 |
| Trisk_L_0.0005_a17 | utility:A/same_residence | PWGTP | 0.48493034 | -0.0141387547 | 0.00807614 |
| Trisk_L_0.0005_a17 | utility:A/income_binary | unweighted | 0.295770904 | 7.24591716e-05 | 0.0112537278 |
| Trisk_L_0.0005_a17 | utility:A/income_binary | PWGTP | 0.307700563 | 0.000120976029 | 0.011292915 |
| Trisk_L_0.0005_a17 | utility:A/civilian_at_work | unweighted | 0.279735797 | 0.000428604212 | 0.00829855741 |
| Trisk_L_0.0005_a17 | utility:A/civilian_at_work | PWGTP | 0.268235114 | 0.000420067268 | 0.00576448353 |
| Trisk_L_0.0005_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_L_0.0005_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_L_0.0005_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_L_0.0005_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_L_0.0005_a17_fineC | attack:A/SEX | unweighted | 0.687186111 | -0.000707549818 | 0.0150399267 |
| Trisk_L_0.0005_a17_fineC | attack:A/SEX | PWGTP | 0.68583799 | -0.000646462776 | 0.00758979044 |
| Trisk_L_0.0005_a17_fineC | attack:A/RAC1P | unweighted | 1.27066322 | -7.6143413e-05 | 0.00644940367 |
| Trisk_L_0.0005_a17_fineC | attack:A/RAC1P | PWGTP | 1.26504839 | 0.000308748434 | 0.00659402338 |
| Trisk_L_0.0005_a17_fineC | attack:AB/SEX | unweighted | 0.685450626 | 0.000233691041 | 0.0166815446 |
| Trisk_L_0.0005_a17_fineC | attack:AB/SEX | PWGTP | 0.684018856 | -0.000342269413 | 0.0102457677 |
| Trisk_L_0.0005_a17_fineC | attack:AB/RAC1P | unweighted | 1.25776431 | 0.00015930722 | 0.0056637015 |
| Trisk_L_0.0005_a17_fineC | attack:AB/RAC1P | PWGTP | 1.25304723 | 0.000695002348 | 0.00711900548 |
| Trisk_L_0.0005_a17_fineC | attack:A/public_coverage | unweighted | 0.518958872 | -0.000361050112 | 0.0073377296 |
| Trisk_L_0.0005_a17_fineC | attack:A/public_coverage | PWGTP | 0.520904475 | -0.000630851281 | 0.00586959239 |
| Trisk_L_0.0005_a17_fineC | attack:A/commute_over20 | unweighted | 0.683184714 | 3.57696329e-05 | 0.00204458906 |
| Trisk_L_0.0005_a17_fineC | attack:A/commute_over20 | PWGTP | 0.686558158 | 1.52855467e-05 | 0.00358725749 |
| Trisk_L_0.0005_a17_fineC | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_L_0.0005_a17_fineC | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_L_0.0005_a17_fineC | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_L_0.0005_a17_fineC | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_L_0.0005_a17_fineC | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_L_0.0005_a17_fineC | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_L_0.0005_a17_fineC | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_L_0.0005_a17_fineC | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_L_0.0005_a17_fineC | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_L_0.0005_a17_fineC | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_L_0.0005_a17_fineC | utility:A/same_residence | unweighted | 0.514170208 | -0.0109814494 | 0.0147216552 |
| Trisk_L_0.0005_a17_fineC | utility:A/same_residence | PWGTP | 0.489232203 | -0.00983689208 | 0.0123780026 |
| Trisk_L_0.0005_a17_fineC | utility:A/income_binary | unweighted | 0.295568555 | -0.000129889269 | 0.0110513794 |
| Trisk_L_0.0005_a17_fineC | utility:A/income_binary | PWGTP | 0.307374789 | -0.000204797351 | 0.0109671416 |
| Trisk_L_0.0005_a17_fineC | utility:A/civilian_at_work | unweighted | 0.279357111 | 4.99180658e-05 | 0.00791987126 |
| Trisk_L_0.0005_a17_fineC | utility:A/civilian_at_work | PWGTP | 0.267913787 | 9.8739853e-05 | 0.00544315611 |
| Trisk_L_0.0005_a17_fineC | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_L_0.0005_a17_fineC | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_L_0.0005_a17_fineC | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_L_0.0005_a17_fineC | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_L_0.002_a17 | attack:A/SEX | unweighted | 0.686707642 | -0.00118601874 | 0.0145614578 |
| Trisk_L_0.002_a17 | attack:A/SEX | PWGTP | 0.687242894 | 0.000758442104 | 0.00899469532 |
| Trisk_L_0.002_a17 | attack:A/RAC1P | unweighted | 1.26942989 | -0.00130947411 | 0.00521607298 |
| Trisk_L_0.002_a17 | attack:A/RAC1P | PWGTP | 1.26439919 | -0.000340447765 | 0.00594482718 |
| Trisk_L_0.002_a17 | attack:AB/SEX | unweighted | 0.683289248 | -0.0019276869 | 0.0145201667 |
| Trisk_L_0.002_a17 | attack:AB/SEX | PWGTP | 0.684787198 | 0.000426072097 | 0.0110141092 |
| Trisk_L_0.002_a17 | attack:AB/RAC1P | unweighted | 1.25631268 | -0.00129232579 | 0.00421206849 |
| Trisk_L_0.002_a17 | attack:AB/RAC1P | PWGTP | 1.25201793 | -0.000334299577 | 0.00608970356 |
| Trisk_L_0.002_a17 | attack:A/public_coverage | unweighted | 0.518437571 | -0.00088235121 | 0.0068164285 |
| Trisk_L_0.002_a17 | attack:A/public_coverage | PWGTP | 0.520389444 | -0.001145882 | 0.00535456167 |
| Trisk_L_0.002_a17 | attack:A/commute_over20 | unweighted | 0.682668832 | -0.000480111771 | 0.00152870765 |
| Trisk_L_0.002_a17 | attack:A/commute_over20 | PWGTP | 0.685774224 | -0.00076864885 | 0.00280332309 |
| Trisk_L_0.002_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_L_0.002_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_L_0.002_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_L_0.002_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_L_0.002_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_L_0.002_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_L_0.002_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_L_0.002_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_L_0.002_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_L_0.002_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_L_0.002_a17 | utility:A/same_residence | unweighted | 0.496842132 | -0.0283095254 | -0.00260642085 |
| Trisk_L_0.002_a17 | utility:A/same_residence | PWGTP | 0.473795252 | -0.0252738427 | -0.00305894794 |
| Trisk_L_0.002_a17 | utility:A/income_binary | unweighted | 0.295751438 | 5.29934049e-05 | 0.0112342621 |
| Trisk_L_0.002_a17 | utility:A/income_binary | PWGTP | 0.307485402 | -9.41844596e-05 | 0.0110777545 |
| Trisk_L_0.002_a17 | utility:A/civilian_at_work | unweighted | 0.27947845 | 0.000171257717 | 0.00804121091 |
| Trisk_L_0.002_a17 | utility:A/civilian_at_work | PWGTP | 0.268085393 | 0.000270345418 | 0.00561476168 |
| Trisk_L_0.002_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_L_0.002_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_L_0.002_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_L_0.002_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_L_0.002_a17_fineC | attack:A/SEX | unweighted | 0.686743972 | -0.00114968829 | 0.0145977882 |
| Trisk_L_0.002_a17_fineC | attack:A/SEX | PWGTP | 0.685310941 | -0.00117351169 | 0.00706274152 |
| Trisk_L_0.002_a17_fineC | attack:A/RAC1P | unweighted | 1.26982606 | -0.000913301215 | 0.00561224587 |
| Trisk_L_0.002_a17_fineC | attack:A/RAC1P | PWGTP | 1.26465548 | -8.41600859e-05 | 0.00620111486 |
| Trisk_L_0.002_a17_fineC | attack:AB/SEX | unweighted | 0.68196736 | -0.00324957433 | 0.0131982792 |
| Trisk_L_0.002_a17_fineC | attack:AB/SEX | PWGTP | 0.683368354 | -0.000992771355 | 0.00959526578 |
| Trisk_L_0.002_a17_fineC | attack:AB/RAC1P | unweighted | 1.25771533 | 0.000110326752 | 0.00561472103 |
| Trisk_L_0.002_a17_fineC | attack:AB/RAC1P | PWGTP | 1.25290299 | 0.000550755156 | 0.00697475829 |
| Trisk_L_0.002_a17_fineC | attack:A/public_coverage | unweighted | 0.518735596 | -0.000584326171 | 0.00711445354 |
| Trisk_L_0.002_a17_fineC | attack:A/public_coverage | PWGTP | 0.520848149 | -0.000687176883 | 0.00581326679 |
| Trisk_L_0.002_a17_fineC | attack:A/commute_over20 | unweighted | 0.682803974 | -0.000344970531 | 0.00166384889 |
| Trisk_L_0.002_a17_fineC | attack:A/commute_over20 | PWGTP | 0.685944205 | -0.000598668055 | 0.00297330388 |
| Trisk_L_0.002_a17_fineC | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_L_0.002_a17_fineC | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_L_0.002_a17_fineC | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_L_0.002_a17_fineC | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_L_0.002_a17_fineC | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_L_0.002_a17_fineC | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_L_0.002_a17_fineC | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_L_0.002_a17_fineC | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_L_0.002_a17_fineC | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_L_0.002_a17_fineC | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_L_0.002_a17_fineC | utility:A/same_residence | unweighted | 0.503111322 | -0.0220403356 | 0.00366276896 |
| Trisk_L_0.002_a17_fineC | utility:A/same_residence | PWGTP | 0.47853231 | -0.0205367851 | 0.00167810962 |
| Trisk_L_0.002_a17_fineC | utility:A/income_binary | unweighted | 0.29562148 | -7.6964229e-05 | 0.0111043044 |
| Trisk_L_0.002_a17_fineC | utility:A/income_binary | PWGTP | 0.307548879 | -3.07074496e-05 | 0.0111412315 |
| Trisk_L_0.002_a17_fineC | utility:A/civilian_at_work | unweighted | 0.279700097 | 0.000392904052 | 0.00826285725 |
| Trisk_L_0.002_a17_fineC | utility:A/civilian_at_work | PWGTP | 0.26827408 | 0.000459033122 | 0.00580344938 |
| Trisk_L_0.002_a17_fineC | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_L_0.002_a17_fineC | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_L_0.002_a17_fineC | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_L_0.002_a17_fineC | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_L_0.002_a33 | attack:A/SEX | unweighted | 0.685116685 | -0.00277697602 | 0.0129705005 |
| Trisk_L_0.002_a33 | attack:A/SEX | PWGTP | 0.685244265 | -0.00124018707 | 0.00699606615 |
| Trisk_L_0.002_a33 | attack:A/RAC1P | unweighted | 1.27115256 | 0.000413201336 | 0.00693874842 |
| Trisk_L_0.002_a33 | attack:A/RAC1P | PWGTP | 1.26527726 | 0.000537624277 | 0.00682289922 |
| Trisk_L_0.002_a33 | attack:AB/SEX | unweighted | 0.678559681 | -0.00665725396 | 0.00979059959 |
| Trisk_L_0.002_a33 | attack:AB/SEX | PWGTP | 0.682273422 | -0.00208770415 | 0.00850033298 |
| Trisk_L_0.002_a33 | attack:AB/RAC1P | unweighted | 1.25614145 | -0.00146356052 | 0.00404083376 |
| Trisk_L_0.002_a33 | attack:AB/RAC1P | PWGTP | 1.25178198 | -0.000570252799 | 0.00585375034 |
| Trisk_L_0.002_a33 | attack:A/public_coverage | unweighted | 0.518530998 | -0.000788924457 | 0.00690985526 |
| Trisk_L_0.002_a33 | attack:A/public_coverage | PWGTP | 0.520375824 | -0.00115950167 | 0.005340942 |
| Trisk_L_0.002_a33 | attack:A/commute_over20 | unweighted | 0.682616756 | -0.000532188428 | 0.001476631 |
| Trisk_L_0.002_a33 | attack:A/commute_over20 | PWGTP | 0.685671783 | -0.000871089358 | 0.00270088258 |
| Trisk_L_0.002_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_L_0.002_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_L_0.002_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_L_0.002_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_L_0.002_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_L_0.002_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_L_0.002_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_L_0.002_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_L_0.002_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_L_0.002_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_L_0.002_a33 | utility:A/same_residence | unweighted | 0.496820073 | -0.0283315848 | -0.00262848023 |
| Trisk_L_0.002_a33 | utility:A/same_residence | PWGTP | 0.473213222 | -0.0258558731 | -0.00364097837 |
| Trisk_L_0.002_a33 | utility:A/income_binary | unweighted | 0.295582059 | -0.000116385557 | 0.0110648831 |
| Trisk_L_0.002_a33 | utility:A/income_binary | PWGTP | 0.307384668 | -0.00019491902 | 0.0109770199 |
| Trisk_L_0.002_a33 | utility:A/civilian_at_work | unweighted | 0.279631144 | 0.000323950972 | 0.00819390417 |
| Trisk_L_0.002_a33 | utility:A/civilian_at_work | PWGTP | 0.268135643 | 0.000320595436 | 0.0056650117 |
| Trisk_L_0.002_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_L_0.002_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_L_0.002_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_L_0.002_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_L_0.01_a17 | attack:A/SEX | unweighted | 0.683451538 | -0.00444212233 | 0.0113053542 |
| Trisk_L_0.01_a17 | attack:A/SEX | PWGTP | 0.6806087 | -0.00587575209 | 0.00236050112 |
| Trisk_L_0.01_a17 | attack:A/RAC1P | unweighted | 1.27018234 | -0.000557018414 | 0.00596852867 |
| Trisk_L_0.01_a17 | attack:A/RAC1P | PWGTP | 1.26278918 | -0.00195045593 | 0.00433481901 |
| Trisk_L_0.01_a17 | attack:AB/SEX | unweighted | 0.680335961 | -0.0048809733 | 0.0115668803 |
| Trisk_L_0.01_a17 | attack:AB/SEX | PWGTP | 0.679625057 | -0.00473606845 | 0.00585196868 |
| Trisk_L_0.01_a17 | attack:AB/RAC1P | unweighted | 1.25539039 | -0.00221461776 | 0.00328977652 |
| Trisk_L_0.01_a17 | attack:AB/RAC1P | PWGTP | 1.2504055 | -0.00194672844 | 0.00447727469 |
| Trisk_L_0.01_a17 | attack:A/public_coverage | unweighted | 0.518535508 | -0.000784414724 | 0.00691436499 |
| Trisk_L_0.01_a17 | attack:A/public_coverage | PWGTP | 0.520491565 | -0.00104376087 | 0.0054566828 |
| Trisk_L_0.01_a17 | attack:A/commute_over20 | unweighted | 0.683133326 | -1.56177685e-05 | 0.00199320165 |
| Trisk_L_0.01_a17 | attack:A/commute_over20 | PWGTP | 0.686297839 | -0.000245033711 | 0.00332693823 |
| Trisk_L_0.01_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_L_0.01_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_L_0.01_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_L_0.01_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_L_0.01_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_L_0.01_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_L_0.01_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_L_0.01_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_L_0.01_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_L_0.01_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_L_0.01_a17 | utility:A/same_residence | unweighted | 0.486483593 | -0.0386680647 | -0.0129649601 |
| Trisk_L_0.01_a17 | utility:A/same_residence | PWGTP | 0.463210704 | -0.0358583915 | -0.0136434968 |
| Trisk_L_0.01_a17 | utility:A/income_binary | unweighted | 0.295842985 | 0.000144540334 | 0.011325809 |
| Trisk_L_0.01_a17 | utility:A/income_binary | PWGTP | 0.308665592 | 0.00108600505 | 0.012257944 |
| Trisk_L_0.01_a17 | utility:A/civilian_at_work | unweighted | 0.278833786 | -0.000473406374 | 0.00739654682 |
| Trisk_L_0.01_a17 | utility:A/civilian_at_work | PWGTP | 0.267536057 | -0.000278990639 | 0.00506542562 |
| Trisk_L_0.01_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_L_0.01_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_L_0.01_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_L_0.01_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_L_0.01_a17_fineC | attack:A/SEX | unweighted | 0.683783914 | -0.00410974691 | 0.0116377296 |
| Trisk_L_0.01_a17_fineC | attack:A/SEX | PWGTP | 0.681696917 | -0.00478753574 | 0.00344871747 |
| Trisk_L_0.01_a17_fineC | attack:A/RAC1P | unweighted | 1.26991178 | -0.00082757836 | 0.00569796873 |
| Trisk_L_0.01_a17_fineC | attack:A/RAC1P | PWGTP | 1.26368132 | -0.00105831539 | 0.00522695956 |
| Trisk_L_0.01_a17_fineC | attack:AB/SEX | unweighted | 0.67912205 | -0.00609488501 | 0.0103529685 |
| Trisk_L_0.01_a17_fineC | attack:AB/SEX | PWGTP | 0.678525418 | -0.00583570822 | 0.00475232891 |
| Trisk_L_0.01_a17_fineC | attack:AB/RAC1P | unweighted | 1.25594485 | -0.00166015176 | 0.00384424251 |
| Trisk_L_0.01_a17_fineC | attack:AB/RAC1P | PWGTP | 1.25104255 | -0.00130968007 | 0.00511432306 |
| Trisk_L_0.01_a17_fineC | attack:A/public_coverage | unweighted | 0.517698546 | -0.0016213765 | 0.00607740322 |
| Trisk_L_0.01_a17_fineC | attack:A/public_coverage | PWGTP | 0.51997731 | -0.00155801587 | 0.00494242781 |
| Trisk_L_0.01_a17_fineC | attack:A/commute_over20 | unweighted | 0.682579299 | -0.000569644637 | 0.00143917479 |
| Trisk_L_0.01_a17_fineC | attack:A/commute_over20 | PWGTP | 0.68523321 | -0.00130966298 | 0.00226230896 |
| Trisk_L_0.01_a17_fineC | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_L_0.01_a17_fineC | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_L_0.01_a17_fineC | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_L_0.01_a17_fineC | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_L_0.01_a17_fineC | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_L_0.01_a17_fineC | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_L_0.01_a17_fineC | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_L_0.01_a17_fineC | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_L_0.01_a17_fineC | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_L_0.01_a17_fineC | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_L_0.01_a17_fineC | utility:A/same_residence | unweighted | 0.489089177 | -0.0360624804 | -0.0103593758 |
| Trisk_L_0.01_a17_fineC | utility:A/same_residence | PWGTP | 0.466183757 | -0.0328853383 | -0.0106704436 |
| Trisk_L_0.01_a17_fineC | utility:A/income_binary | unweighted | 0.295901968 | 0.000203523098 | 0.0113847918 |
| Trisk_L_0.01_a17_fineC | utility:A/income_binary | PWGTP | 0.308777365 | 0.00119777807 | 0.012369717 |
| Trisk_L_0.01_a17_fineC | utility:A/civilian_at_work | unweighted | 0.279003043 | -0.000304149472 | 0.00756580373 |
| Trisk_L_0.01_a17_fineC | utility:A/civilian_at_work | PWGTP | 0.267567459 | -0.000247588246 | 0.00509682801 |
| Trisk_L_0.01_a17_fineC | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_L_0.01_a17_fineC | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_L_0.01_a17_fineC | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_L_0.01_a17_fineC | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_L_0_a17 | attack:A/SEX | unweighted | 0.683998863 | -0.00389479722 | 0.0118526793 |
| Trisk_L_0_a17 | attack:A/SEX | PWGTP | 0.687015143 | 0.00053069095 | 0.00876694416 |
| Trisk_L_0_a17 | attack:A/RAC1P | unweighted | 1.27117189 | 0.000432535176 | 0.00695808226 |
| Trisk_L_0_a17 | attack:A/RAC1P | PWGTP | 1.26516271 | 0.000423065618 | 0.00670834056 |
| Trisk_L_0_a17 | attack:AB/SEX | unweighted | 0.685652981 | 0.000436046485 | 0.0168839 |
| Trisk_L_0_a17 | attack:AB/SEX | PWGTP | 0.684535796 | 0.00017467053 | 0.0107627077 |
| Trisk_L_0_a17 | attack:AB/RAC1P | unweighted | 1.25817127 | 0.000566264991 | 0.00607065927 |
| Trisk_L_0_a17 | attack:AB/RAC1P | PWGTP | 1.25250473 | 0.000152501646 | 0.00657650478 |
| Trisk_L_0_a17 | attack:A/public_coverage | unweighted | 0.51915249 | -0.000167432426 | 0.00753134729 |
| Trisk_L_0_a17 | attack:A/public_coverage | PWGTP | 0.521503065 | -3.22612102e-05 | 0.00646818246 |
| Trisk_L_0_a17 | attack:A/commute_over20 | unweighted | 0.683303751 | 0.000154806453 | 0.00216362588 |
| Trisk_L_0_a17 | attack:A/commute_over20 | PWGTP | 0.686676341 | 0.000133468863 | 0.0037054408 |
| Trisk_L_0_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_L_0_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_L_0_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_L_0_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_L_0_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_L_0_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_L_0_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_L_0_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_L_0_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_L_0_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_L_0_a17 | utility:A/same_residence | unweighted | 0.522634578 | -0.00251707947 | 0.0231860251 |
| Trisk_L_0_a17 | utility:A/same_residence | PWGTP | 0.496757851 | -0.00231124422 | 0.0199036505 |
| Trisk_L_0_a17 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Trisk_L_0_a17 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Trisk_L_0_a17 | utility:A/civilian_at_work | unweighted | 0.279370878 | 6.36856476e-05 | 0.00793363884 |
| Trisk_L_0_a17 | utility:A/civilian_at_work | PWGTP | 0.267835511 | 2.04641889e-05 | 0.00536488045 |
| Trisk_L_0_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_L_0_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_L_0_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_L_0_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_U_unconstrained_a17 | attack:A/SEX | unweighted | 0.683536521 | -0.00435713978 | 0.0113903367 |
| Trisk_U_unconstrained_a17 | attack:A/SEX | PWGTP | 0.682906097 | -0.00357835557 | 0.00465789765 |
| Trisk_U_unconstrained_a17 | attack:A/RAC1P | unweighted | 1.26459607 | -0.00614328703 | 0.000382260061 |
| Trisk_U_unconstrained_a17 | attack:A/RAC1P | PWGTP | 1.25622802 | -0.00851162094 | -0.002226346 |
| Trisk_U_unconstrained_a17 | attack:AB/SEX | unweighted | 0.679865643 | -0.00535129194 | 0.0110965616 |
| Trisk_U_unconstrained_a17 | attack:AB/SEX | PWGTP | 0.68131486 | -0.00304626605 | 0.00754177109 |
| Trisk_U_unconstrained_a17 | attack:AB/RAC1P | unweighted | 1.25355053 | -0.00405447137 | 0.0014499229 |
| Trisk_U_unconstrained_a17 | attack:AB/RAC1P | PWGTP | 1.24642644 | -0.00592578627 | 0.000498216867 |
| Trisk_U_unconstrained_a17 | attack:A/public_coverage | unweighted | 0.519785009 | 0.000465086667 | 0.00816386638 |
| Trisk_U_unconstrained_a17 | attack:A/public_coverage | PWGTP | 0.523122288 | 0.00158696211 | 0.00808740579 |
| Trisk_U_unconstrained_a17 | attack:A/commute_over20 | unweighted | 0.683710803 | 0.000561858792 | 0.00257067822 |
| Trisk_U_unconstrained_a17 | attack:A/commute_over20 | PWGTP | 0.686372314 | -0.000170558626 | 0.00340141331 |
| Trisk_U_unconstrained_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_U_unconstrained_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_U_unconstrained_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_U_unconstrained_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_U_unconstrained_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_U_unconstrained_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_U_unconstrained_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_U_unconstrained_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_U_unconstrained_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_U_unconstrained_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_U_unconstrained_a17 | utility:A/same_residence | unweighted | 0.488296244 | -0.036855414 | -0.0111523094 |
| Trisk_U_unconstrained_a17 | utility:A/same_residence | PWGTP | 0.464819066 | -0.0342500291 | -0.0120351344 |
| Trisk_U_unconstrained_a17 | utility:A/income_binary | unweighted | 0.296448862 | 0.00075041793 | 0.0119316866 |
| Trisk_U_unconstrained_a17 | utility:A/income_binary | PWGTP | 0.309177429 | 0.00159784222 | 0.0127697812 |
| Trisk_U_unconstrained_a17 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Trisk_U_unconstrained_a17 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Trisk_U_unconstrained_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_U_unconstrained_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_U_unconstrained_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_U_unconstrained_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_U_unconstrained_a33 | attack:A/SEX | unweighted | 0.682621742 | -0.00527191898 | 0.0104755575 |
| Trisk_U_unconstrained_a33 | attack:A/SEX | PWGTP | 0.682777804 | -0.00370664804 | 0.00452960518 |
| Trisk_U_unconstrained_a33 | attack:A/RAC1P | unweighted | 1.26605085 | -0.00468851237 | 0.00183703472 |
| Trisk_U_unconstrained_a33 | attack:A/RAC1P | PWGTP | 1.25770192 | -0.0070377238 | -0.000752448862 |
| Trisk_U_unconstrained_a33 | attack:AB/SEX | unweighted | 0.679761793 | -0.00545514194 | 0.0109927116 |
| Trisk_U_unconstrained_a33 | attack:AB/SEX | PWGTP | 0.680820347 | -0.00354077908 | 0.00704725806 |
| Trisk_U_unconstrained_a33 | attack:AB/RAC1P | unweighted | 1.25302061 | -0.00458439504 | 0.000919999235 |
| Trisk_U_unconstrained_a33 | attack:AB/RAC1P | PWGTP | 1.24718529 | -0.00516694504 | 0.0012570581 |
| Trisk_U_unconstrained_a33 | attack:A/public_coverage | unweighted | 0.521244637 | 0.00192471454 | 0.00962349425 |
| Trisk_U_unconstrained_a33 | attack:A/public_coverage | PWGTP | 0.524226561 | 0.00269123524 | 0.00919167891 |
| Trisk_U_unconstrained_a33 | attack:A/commute_over20 | unweighted | 0.683261183 | 0.000112238776 | 0.0021210582 |
| Trisk_U_unconstrained_a33 | attack:A/commute_over20 | PWGTP | 0.685637638 | -0.000905234511 | 0.00266673743 |
| Trisk_U_unconstrained_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_U_unconstrained_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_U_unconstrained_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_U_unconstrained_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_U_unconstrained_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_U_unconstrained_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_U_unconstrained_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_U_unconstrained_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_U_unconstrained_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_U_unconstrained_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_U_unconstrained_a33 | utility:A/same_residence | unweighted | 0.48627394 | -0.0388777179 | -0.0131746134 |
| Trisk_U_unconstrained_a33 | utility:A/same_residence | PWGTP | 0.463022559 | -0.036046536 | -0.0138316413 |
| Trisk_U_unconstrained_a33 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Trisk_U_unconstrained_a33 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Trisk_U_unconstrained_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Trisk_U_unconstrained_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Trisk_U_unconstrained_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_U_unconstrained_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_U_unconstrained_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_U_unconstrained_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_code | attack:A/SEX | unweighted | 0.674988481 | -0.0129051791 | 0.0028422974 |
| Trisk_code | attack:A/SEX | PWGTP | 0.677601554 | -0.00888289849 | -0.000646645275 |
| Trisk_code | attack:A/RAC1P | unweighted | 1.25736281 | -0.0133765482 | -0.00685100107 |
| Trisk_code | attack:A/RAC1P | PWGTP | 1.25443014 | -0.0103095046 | -0.00402422968 |
| Trisk_code | attack:AB/SEX | unweighted | 0.675214571 | -0.0100023637 | 0.00644548981 |
| Trisk_code | attack:AB/SEX | PWGTP | 0.676388608 | -0.00797251828 | 0.00261551885 |
| Trisk_code | attack:AB/RAC1P | unweighted | 1.25148958 | -0.00611542829 | -0.000611034012 |
| Trisk_code | attack:AB/RAC1P | PWGTP | 1.24679491 | -0.00555731814 | 0.000866684998 |
| Trisk_code | attack:A/public_coverage | unweighted | 0.518653937 | -0.00066598592 | 0.00703279379 |
| Trisk_code | attack:A/public_coverage | PWGTP | 0.520297309 | -0.00123801667 | 0.005262427 |
| Trisk_code | attack:A/commute_over20 | unweighted | 0.683118346 | -3.05982422e-05 | 0.00197822118 |
| Trisk_code | attack:A/commute_over20 | PWGTP | 0.685589925 | -0.000952948081 | 0.00261902386 |
| Trisk_code | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_code | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_code | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_code | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_code | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_code | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_code | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_code | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_code | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_code | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_code | utility:A/same_residence | unweighted | 0.49093995 | -0.0342117078 | -0.00850860326 |
| Trisk_code | utility:A/same_residence | PWGTP | 0.466895001 | -0.0321740942 | -0.00995919953 |
| Trisk_code | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Trisk_code | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Trisk_code | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Trisk_code | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Trisk_code | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_code | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_code | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_code | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_rr_0.25 | attack:A/SEX | unweighted | 0.687946962 | 5.33012756e-05 | 0.0158007778 |
| Trisk_rr_0.25 | attack:A/SEX | PWGTP | 0.686575157 | 9.07050964e-05 | 0.00832695831 |
| Trisk_rr_0.25 | attack:A/RAC1P | unweighted | 1.26960086 | -0.00113849998 | 0.00538704711 |
| Trisk_rr_0.25 | attack:A/RAC1P | PWGTP | 1.26433711 | -0.000402526339 | 0.0058827486 |
| Trisk_rr_0.25 | attack:AB/SEX | unweighted | 0.684866751 | -0.000350183454 | 0.0160976701 |
| Trisk_rr_0.25 | attack:AB/SEX | PWGTP | 0.684135953 | -0.00022517288 | 0.0103628643 |
| Trisk_rr_0.25 | attack:AB/RAC1P | unweighted | 1.25848348 | 0.000878477615 | 0.00638287189 |
| Trisk_rr_0.25 | attack:AB/RAC1P | PWGTP | 1.25275981 | 0.000407576741 | 0.00683157988 |
| Trisk_rr_0.25 | attack:A/public_coverage | unweighted | 0.519234632 | -8.52906609e-05 | 0.00761348905 |
| Trisk_rr_0.25 | attack:A/public_coverage | PWGTP | 0.521227214 | -0.000308111991 | 0.00619233168 |
| Trisk_rr_0.25 | attack:A/commute_over20 | unweighted | 0.68324237 | 9.34261443e-05 | 0.00210224557 |
| Trisk_rr_0.25 | attack:A/commute_over20 | PWGTP | 0.686587373 | 4.45004624e-05 | 0.0036164724 |
| Trisk_rr_0.25 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_rr_0.25 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_rr_0.25 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_rr_0.25 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_rr_0.25 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_rr_0.25 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_rr_0.25 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_rr_0.25 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_rr_0.25 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_rr_0.25 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_rr_0.25 | utility:A/same_residence | unweighted | 0.521749864 | -0.00340179412 | 0.0223013104 |
| Trisk_rr_0.25 | utility:A/same_residence | PWGTP | 0.495834946 | -0.00323414871 | 0.018980746 |
| Trisk_rr_0.25 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Trisk_rr_0.25 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Trisk_rr_0.25 | utility:A/civilian_at_work | unweighted | 0.279138895 | -0.000168297236 | 0.00770165596 |
| Trisk_rr_0.25 | utility:A/civilian_at_work | PWGTP | 0.267625499 | -0.000189548094 | 0.00515486817 |
| Trisk_rr_0.25 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_rr_0.25 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_rr_0.25 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_rr_0.25 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_rr_0.25_a33 | attack:A/SEX | unweighted | 0.684842814 | -0.00305084671 | 0.0126966298 |
| Trisk_rr_0.25_a33 | attack:A/SEX | PWGTP | 0.685451357 | -0.00103309509 | 0.00720315812 |
| Trisk_rr_0.25_a33 | attack:A/RAC1P | unweighted | 1.27136995 | 0.000630590663 | 0.00715613775 |
| Trisk_rr_0.25_a33 | attack:A/RAC1P | PWGTP | 1.26511531 | 0.00037567233 | 0.00666094727 |
| Trisk_rr_0.25_a33 | attack:AB/SEX | unweighted | 0.685374685 | 0.000157750157 | 0.0166056037 |
| Trisk_rr_0.25_a33 | attack:AB/SEX | PWGTP | 0.684457206 | 9.60801138e-05 | 0.0106841172 |
| Trisk_rr_0.25_a33 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| Trisk_rr_0.25_a33 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| Trisk_rr_0.25_a33 | attack:A/public_coverage | unweighted | 0.519301112 | -1.88104805e-05 | 0.00767996923 |
| Trisk_rr_0.25_a33 | attack:A/public_coverage | PWGTP | 0.521514493 | -2.08327724e-05 | 0.0064796109 |
| Trisk_rr_0.25_a33 | attack:A/commute_over20 | unweighted | 0.683233615 | 8.46709839e-05 | 0.00209349041 |
| Trisk_rr_0.25_a33 | attack:A/commute_over20 | PWGTP | 0.686533827 | -9.04576029e-06 | 0.00356292618 |
| Trisk_rr_0.25_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_rr_0.25_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_rr_0.25_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_rr_0.25_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_rr_0.25_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_rr_0.25_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_rr_0.25_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_rr_0.25_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_rr_0.25_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_rr_0.25_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_rr_0.25_a33 | utility:A/same_residence | unweighted | 0.52236138 | -0.00279027816 | 0.0229128264 |
| Trisk_rr_0.25_a33 | utility:A/same_residence | PWGTP | 0.496717927 | -0.00235116846 | 0.0198637263 |
| Trisk_rr_0.25_a33 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Trisk_rr_0.25_a33 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Trisk_rr_0.25_a33 | utility:A/civilian_at_work | unweighted | 0.279332533 | 2.53407535e-05 | 0.00789529395 |
| Trisk_rr_0.25_a33 | utility:A/civilian_at_work | PWGTP | 0.267826115 | 1.10679236e-05 | 0.00535548418 |
| Trisk_rr_0.25_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_rr_0.25_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_rr_0.25_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_rr_0.25_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_rr_0.5 | attack:A/SEX | unweighted | 0.684705388 | -0.00318827243 | 0.0125592041 |
| Trisk_rr_0.5 | attack:A/SEX | PWGTP | 0.685539534 | -0.000944918615 | 0.0072913346 |
| Trisk_rr_0.5 | attack:A/RAC1P | unweighted | 1.26987355 | -0.000865811638 | 0.00565973545 |
| Trisk_rr_0.5 | attack:A/RAC1P | PWGTP | 1.26452863 | -0.000211013276 | 0.00607426167 |
| Trisk_rr_0.5 | attack:AB/SEX | unweighted | 0.681893987 | -0.00332294782 | 0.0131249057 |
| Trisk_rr_0.5 | attack:AB/SEX | PWGTP | 0.68395746 | -0.000403665892 | 0.0101843712 |
| Trisk_rr_0.5 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| Trisk_rr_0.5 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| Trisk_rr_0.5 | attack:A/public_coverage | unweighted | 0.519309313 | -1.06096417e-05 | 0.00768817007 |
| Trisk_rr_0.5 | attack:A/public_coverage | PWGTP | 0.521703069 | 0.000167743182 | 0.00666818685 |
| Trisk_rr_0.5 | attack:A/commute_over20 | unweighted | 0.683281798 | 0.000132854309 | 0.00214167373 |
| Trisk_rr_0.5 | attack:A/commute_over20 | PWGTP | 0.686502626 | -4.02465327e-05 | 0.00353172541 |
| Trisk_rr_0.5 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_rr_0.5 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_rr_0.5 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_rr_0.5 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_rr_0.5 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_rr_0.5 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_rr_0.5 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_rr_0.5 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_rr_0.5 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_rr_0.5 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_rr_0.5 | utility:A/same_residence | unweighted | 0.515442819 | -0.00970883913 | 0.0159942654 |
| Trisk_rr_0.5 | utility:A/same_residence | PWGTP | 0.490777842 | -0.00829125265 | 0.0139236421 |
| Trisk_rr_0.5 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Trisk_rr_0.5 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Trisk_rr_0.5 | utility:A/civilian_at_work | unweighted | 0.278981167 | -0.000326025338 | 0.00754392786 |
| Trisk_rr_0.5 | utility:A/civilian_at_work | PWGTP | 0.267732523 | -8.25246538e-05 | 0.00526189161 |
| Trisk_rr_0.5 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_rr_0.5 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_rr_0.5 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_rr_0.5 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_rr_0.5_a33 | attack:A/SEX | unweighted | 0.680745093 | -0.00714856786 | 0.00859890864 |
| Trisk_rr_0.5_a33 | attack:A/SEX | PWGTP | 0.685694654 | -0.000789798807 | 0.00744645441 |
| Trisk_rr_0.5_a33 | attack:A/RAC1P | unweighted | 1.27110785 | 0.000368494625 | 0.00689404171 |
| Trisk_rr_0.5_a33 | attack:A/RAC1P | PWGTP | 1.26435986 | -0.000379784485 | 0.00590549046 |
| Trisk_rr_0.5_a33 | attack:AB/SEX | unweighted | 0.678840089 | -0.00637684546 | 0.0100710081 |
| Trisk_rr_0.5_a33 | attack:AB/SEX | PWGTP | 0.682826865 | -0.00153426053 | 0.0090537766 |
| Trisk_rr_0.5_a33 | attack:AB/RAC1P | unweighted | 1.25623393 | -0.00137108118 | 0.0041333131 |
| Trisk_rr_0.5_a33 | attack:AB/RAC1P | PWGTP | 1.2511739 | -0.00117832653 | 0.0052456766 |
| Trisk_rr_0.5_a33 | attack:A/public_coverage | unweighted | 0.519785006 | 0.00046508324 | 0.00816386295 |
| Trisk_rr_0.5_a33 | attack:A/public_coverage | PWGTP | 0.521853938 | 0.000318611812 | 0.00681905548 |
| Trisk_rr_0.5_a33 | attack:A/commute_over20 | unweighted | 0.683228832 | 7.98877541e-05 | 0.00208870718 |
| Trisk_rr_0.5_a33 | attack:A/commute_over20 | PWGTP | 0.686302521 | -0.000240351568 | 0.00333162037 |
| Trisk_rr_0.5_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_rr_0.5_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_rr_0.5_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_rr_0.5_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_rr_0.5_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_rr_0.5_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_rr_0.5_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_rr_0.5_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_rr_0.5_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_rr_0.5_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_rr_0.5_a33 | utility:A/same_residence | unweighted | 0.514214843 | -0.0109368147 | 0.0147662898 |
| Trisk_rr_0.5_a33 | utility:A/same_residence | PWGTP | 0.488498081 | -0.0105710144 | 0.0116438804 |
| Trisk_rr_0.5_a33 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Trisk_rr_0.5_a33 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Trisk_rr_0.5_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Trisk_rr_0.5_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Trisk_rr_0.5_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_rr_0.5_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_rr_0.5_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_rr_0.5_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_rr_0.75 | attack:A/SEX | unweighted | 0.68660647 | -0.00128719014 | 0.0144602864 |
| Trisk_rr_0.75 | attack:A/SEX | PWGTP | 0.68647394 | -1.05127322e-05 | 0.00822574048 |
| Trisk_rr_0.75 | attack:A/RAC1P | unweighted | 1.26859154 | -0.00214781637 | 0.00437773072 |
| Trisk_rr_0.75 | attack:A/RAC1P | PWGTP | 1.26152263 | -0.00321701198 | 0.00306826296 |
| Trisk_rr_0.75 | attack:AB/SEX | unweighted | 0.678978477 | -0.00623845754 | 0.010209396 |
| Trisk_rr_0.75 | attack:AB/SEX | PWGTP | 0.682980828 | -0.001380298 | 0.00920773913 |
| Trisk_rr_0.75 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| Trisk_rr_0.75 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| Trisk_rr_0.75 | attack:A/public_coverage | unweighted | 0.519824531 | 0.000504608054 | 0.00820338777 |
| Trisk_rr_0.75 | attack:A/public_coverage | PWGTP | 0.522372868 | 0.000837542422 | 0.00733798609 |
| Trisk_rr_0.75 | attack:A/commute_over20 | unweighted | 0.683423005 | 0.000274061255 | 0.00228288068 |
| Trisk_rr_0.75 | attack:A/commute_over20 | PWGTP | 0.686402594 | -0.000140278728 | 0.00343169321 |
| Trisk_rr_0.75 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_rr_0.75 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_rr_0.75 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_rr_0.75 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_rr_0.75 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_rr_0.75 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_rr_0.75 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_rr_0.75 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_rr_0.75 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_rr_0.75 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_rr_0.75 | utility:A/same_residence | unweighted | 0.503921648 | -0.0212300102 | 0.0044730943 |
| Trisk_rr_0.75 | utility:A/same_residence | PWGTP | 0.47875726 | -0.0203118347 | 0.00190306005 |
| Trisk_rr_0.75 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Trisk_rr_0.75 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Trisk_rr_0.75 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Trisk_rr_0.75 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Trisk_rr_0.75 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_rr_0.75 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_rr_0.75 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_rr_0.75 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_rr_0.75_a33 | attack:A/SEX | unweighted | 0.680824448 | -0.00706921267 | 0.00867826383 |
| Trisk_rr_0.75_a33 | attack:A/SEX | PWGTP | 0.685305231 | -0.00117922129 | 0.00705703192 |
| Trisk_rr_0.75_a33 | attack:A/RAC1P | unweighted | 1.27017537 | -0.000563992608 | 0.00596155448 |
| Trisk_rr_0.75_a33 | attack:A/RAC1P | PWGTP | 1.26201325 | -0.00272638628 | 0.00355888866 |
| Trisk_rr_0.75_a33 | attack:AB/SEX | unweighted | 0.678568564 | -0.00664837082 | 0.00979948273 |
| Trisk_rr_0.75_a33 | attack:AB/SEX | PWGTP | 0.68272155 | -0.00163957598 | 0.00894846115 |
| Trisk_rr_0.75_a33 | attack:AB/RAC1P | unweighted | 1.25534432 | -0.002260685 | 0.00324370928 |
| Trisk_rr_0.75_a33 | attack:AB/RAC1P | PWGTP | 1.24998315 | -0.00236908029 | 0.00405492284 |
| Trisk_rr_0.75_a33 | attack:A/public_coverage | unweighted | 0.519682843 | 0.000362920797 | 0.00806170051 |
| Trisk_rr_0.75_a33 | attack:A/public_coverage | PWGTP | 0.522222347 | 0.000687020789 | 0.00718746446 |
| Trisk_rr_0.75_a33 | attack:A/commute_over20 | unweighted | 0.683235362 | 8.64175902e-05 | 0.00209523701 |
| Trisk_rr_0.75_a33 | attack:A/commute_over20 | PWGTP | 0.686023222 | -0.000519650789 | 0.00305232115 |
| Trisk_rr_0.75_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_rr_0.75_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_rr_0.75_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_rr_0.75_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_rr_0.75_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_rr_0.75_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_rr_0.75_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_rr_0.75_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_rr_0.75_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_rr_0.75_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_rr_0.75_a33 | utility:A/same_residence | unweighted | 0.504770076 | -0.0203815814 | 0.00532152319 |
| Trisk_rr_0.75_a33 | utility:A/same_residence | PWGTP | 0.481243249 | -0.0178258459 | 0.00438904884 |
| Trisk_rr_0.75_a33 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Trisk_rr_0.75_a33 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Trisk_rr_0.75_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Trisk_rr_0.75_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Trisk_rr_0.75_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_rr_0.75_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_rr_0.75_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_rr_0.75_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_withhold_0.25 | attack:A/SEX | unweighted | 0.684460843 | -0.00343281787 | 0.0123146586 |
| Trisk_withhold_0.25 | attack:A/SEX | PWGTP | 0.684892858 | -0.00159159469 | 0.00664465852 |
| Trisk_withhold_0.25 | attack:A/RAC1P | unweighted | 1.26967295 | -0.00106641171 | 0.00545913538 |
| Trisk_withhold_0.25 | attack:A/RAC1P | PWGTP | 1.26325671 | -0.00148293235 | 0.00480234259 |
| Trisk_withhold_0.25 | attack:AB/SEX | unweighted | 0.680248475 | -0.00496846016 | 0.0114793934 |
| Trisk_withhold_0.25 | attack:AB/SEX | PWGTP | 0.682465188 | -0.00189593829 | 0.00869209884 |
| Trisk_withhold_0.25 | attack:AB/RAC1P | unweighted | 1.25489134 | -0.0027136665 | 0.00279072777 |
| Trisk_withhold_0.25 | attack:AB/RAC1P | PWGTP | 1.24916158 | -0.00319065502 | 0.00323334811 |
| Trisk_withhold_0.25 | attack:A/public_coverage | unweighted | 0.519413354 | 9.34314375e-05 | 0.00779221115 |
| Trisk_withhold_0.25 | attack:A/public_coverage | PWGTP | 0.521527442 | -7.88422808e-06 | 0.00649255944 |
| Trisk_withhold_0.25 | attack:A/commute_over20 | unweighted | 0.683271374 | 0.000122429785 | 0.00213124921 |
| Trisk_withhold_0.25 | attack:A/commute_over20 | PWGTP | 0.686476659 | -6.62134299e-05 | 0.00350575851 |
| Trisk_withhold_0.25 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_withhold_0.25 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_withhold_0.25 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_withhold_0.25 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_withhold_0.25 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_withhold_0.25 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_withhold_0.25 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_withhold_0.25 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_withhold_0.25 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_withhold_0.25 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_withhold_0.25 | utility:A/same_residence | unweighted | 0.516266344 | -0.00888531345 | 0.0168177911 |
| Trisk_withhold_0.25 | utility:A/same_residence | PWGTP | 0.491302431 | -0.00776666389 | 0.0144482308 |
| Trisk_withhold_0.25 | utility:A/income_binary | unweighted | 0.295871122 | 0.000172677248 | 0.0113539459 |
| Trisk_withhold_0.25 | utility:A/income_binary | PWGTP | 0.308529683 | 0.000950096451 | 0.0121220354 |
| Trisk_withhold_0.25 | utility:A/civilian_at_work | unweighted | 0.279099522 | -0.000207671046 | 0.00766228215 |
| Trisk_withhold_0.25 | utility:A/civilian_at_work | PWGTP | 0.267695 | -0.000120047053 | 0.00522436921 |
| Trisk_withhold_0.25 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_withhold_0.25 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_withhold_0.25 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_withhold_0.25 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_withhold_0.25_a33 | attack:A/SEX | unweighted | 0.685553627 | -0.00234003316 | 0.0134074433 |
| Trisk_withhold_0.25_a33 | attack:A/SEX | PWGTP | 0.686412921 | -7.15314103e-05 | 0.0081647218 |
| Trisk_withhold_0.25_a33 | attack:A/RAC1P | unweighted | 1.2703297 | -0.000409654986 | 0.0061158921 |
| Trisk_withhold_0.25_a33 | attack:A/RAC1P | PWGTP | 1.26383721 | -0.00090243316 | 0.00538284178 |
| Trisk_withhold_0.25_a33 | attack:AB/SEX | unweighted | 0.684973458 | -0.000243476635 | 0.0162043769 |
| Trisk_withhold_0.25_a33 | attack:AB/SEX | PWGTP | 0.684279237 | -8.18887759e-05 | 0.0105061484 |
| Trisk_withhold_0.25_a33 | attack:AB/RAC1P | unweighted | 1.25605781 | -0.00154719516 | 0.00395719912 |
| Trisk_withhold_0.25_a33 | attack:AB/RAC1P | PWGTP | 1.25035664 | -0.00199558535 | 0.00442841779 |
| Trisk_withhold_0.25_a33 | attack:A/public_coverage | unweighted | 0.519751093 | 0.000431170336 | 0.00812995005 |
| Trisk_withhold_0.25_a33 | attack:A/public_coverage | PWGTP | 0.522078031 | 0.000542704709 | 0.00704314838 |
| Trisk_withhold_0.25_a33 | attack:A/commute_over20 | unweighted | 0.683226964 | 7.80194159e-05 | 0.00208683884 |
| Trisk_withhold_0.25_a33 | attack:A/commute_over20 | PWGTP | 0.686352431 | -0.000190441444 | 0.00338153049 |
| Trisk_withhold_0.25_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_withhold_0.25_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_withhold_0.25_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_withhold_0.25_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_withhold_0.25_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_withhold_0.25_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_withhold_0.25_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_withhold_0.25_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_withhold_0.25_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_withhold_0.25_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_withhold_0.25_a33 | utility:A/same_residence | unweighted | 0.516345123 | -0.00880653493 | 0.0168965696 |
| Trisk_withhold_0.25_a33 | utility:A/same_residence | PWGTP | 0.490854592 | -0.00821450277 | 0.0140003919 |
| Trisk_withhold_0.25_a33 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Trisk_withhold_0.25_a33 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Trisk_withhold_0.25_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Trisk_withhold_0.25_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Trisk_withhold_0.25_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_withhold_0.25_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_withhold_0.25_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_withhold_0.25_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_withhold_0.5 | attack:A/SEX | unweighted | 0.682186404 | -0.00570725688 | 0.0100402196 |
| Trisk_withhold_0.5 | attack:A/SEX | PWGTP | 0.684595555 | -0.00188889753 | 0.00634735568 |
| Trisk_withhold_0.5 | attack:A/RAC1P | unweighted | 1.26922833 | -0.00151103138 | 0.00501451571 |
| Trisk_withhold_0.5 | attack:A/RAC1P | PWGTP | 1.26172907 | -0.00301056827 | 0.00327470667 |
| Trisk_withhold_0.5 | attack:AB/SEX | unweighted | 0.67980693 | -0.0054100046 | 0.0110378489 |
| Trisk_withhold_0.5 | attack:AB/SEX | PWGTP | 0.683538331 | -0.000822794511 | 0.00976524262 |
| Trisk_withhold_0.5 | attack:AB/RAC1P | unweighted | 1.25474795 | -0.00285705603 | 0.00264733824 |
| Trisk_withhold_0.5 | attack:AB/RAC1P | PWGTP | 1.24974714 | -0.00260508738 | 0.00381891575 |
| Trisk_withhold_0.5 | attack:A/public_coverage | unweighted | 0.519319923 | 0 | 0.00769877971 |
| Trisk_withhold_0.5 | attack:A/public_coverage | PWGTP | 0.521535326 | 0 | 0.00650044367 |
| Trisk_withhold_0.5 | attack:A/commute_over20 | unweighted | 0.683533748 | 0.000384804112 | 0.00239362354 |
| Trisk_withhold_0.5 | attack:A/commute_over20 | PWGTP | 0.686591305 | 4.84326022e-05 | 0.00362040454 |
| Trisk_withhold_0.5 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_withhold_0.5 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_withhold_0.5 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_withhold_0.5 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_withhold_0.5 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_withhold_0.5 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_withhold_0.5 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_withhold_0.5 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_withhold_0.5 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_withhold_0.5 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_withhold_0.5 | utility:A/same_residence | unweighted | 0.508937959 | -0.0162136983 | 0.00948940625 |
| Trisk_withhold_0.5 | utility:A/same_residence | PWGTP | 0.484414432 | -0.0146546631 | 0.00756023163 |
| Trisk_withhold_0.5 | utility:A/income_binary | unweighted | 0.296083787 | 0.000385342444 | 0.0115666111 |
| Trisk_withhold_0.5 | utility:A/income_binary | PWGTP | 0.308707151 | 0.00112756481 | 0.0122995038 |
| Trisk_withhold_0.5 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Trisk_withhold_0.5 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Trisk_withhold_0.5 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_withhold_0.5 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_withhold_0.5 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_withhold_0.5 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_withhold_0.5_a33 | attack:A/SEX | unweighted | 0.682519974 | -0.00537368643 | 0.0103737901 |
| Trisk_withhold_0.5_a33 | attack:A/SEX | PWGTP | 0.685391482 | -0.00109297058 | 0.00714328263 |
| Trisk_withhold_0.5_a33 | attack:A/RAC1P | unweighted | 1.26978183 | -0.00095753233 | 0.00556801476 |
| Trisk_withhold_0.5_a33 | attack:A/RAC1P | PWGTP | 1.26242716 | -0.00231248496 | 0.00397278998 |
| Trisk_withhold_0.5_a33 | attack:AB/SEX | unweighted | 0.679738028 | -0.00547890669 | 0.0109689469 |
| Trisk_withhold_0.5_a33 | attack:AB/SEX | PWGTP | 0.68438401 | 2.28845673e-05 | 0.0106109217 |
| Trisk_withhold_0.5_a33 | attack:AB/RAC1P | unweighted | 1.25511672 | -0.00248828291 | 0.00301611137 |
| Trisk_withhold_0.5_a33 | attack:AB/RAC1P | PWGTP | 1.24956448 | -0.00278775498 | 0.00363624816 |
| Trisk_withhold_0.5_a33 | attack:A/public_coverage | unweighted | 0.519412409 | 9.24867306e-05 | 0.00779126644 |
| Trisk_withhold_0.5_a33 | attack:A/public_coverage | PWGTP | 0.522286683 | 0.000751357476 | 0.00725180115 |
| Trisk_withhold_0.5_a33 | attack:A/commute_over20 | unweighted | 0.683258343 | 0.00010939911 | 0.00211821853 |
| Trisk_withhold_0.5_a33 | attack:A/commute_over20 | PWGTP | 0.686097786 | -0.000445087011 | 0.00312688493 |
| Trisk_withhold_0.5_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_withhold_0.5_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_withhold_0.5_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_withhold_0.5_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_withhold_0.5_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_withhold_0.5_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_withhold_0.5_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_withhold_0.5_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_withhold_0.5_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_withhold_0.5_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_withhold_0.5_a33 | utility:A/same_residence | unweighted | 0.507073228 | -0.0180784296 | 0.0076246749 |
| Trisk_withhold_0.5_a33 | utility:A/same_residence | PWGTP | 0.4825018 | -0.0165672951 | 0.00564759966 |
| Trisk_withhold_0.5_a33 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Trisk_withhold_0.5_a33 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Trisk_withhold_0.5_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Trisk_withhold_0.5_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Trisk_withhold_0.5_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_withhold_0.5_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_withhold_0.5_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_withhold_0.5_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_withhold_0.75 | attack:A/SEX | unweighted | 0.686541995 | -0.00135166567 | 0.0143958108 |
| Trisk_withhold_0.75 | attack:A/SEX | PWGTP | 0.684332995 | -0.00215145788 | 0.00608479534 |
| Trisk_withhold_0.75 | attack:A/RAC1P | unweighted | 1.26721237 | -0.00352698869 | 0.00299855839 |
| Trisk_withhold_0.75 | attack:A/RAC1P | PWGTP | 1.25935613 | -0.00538350948 | 0.000901765461 |
| Trisk_withhold_0.75 | attack:AB/SEX | unweighted | 0.67931467 | -0.00590226436 | 0.0105455892 |
| Trisk_withhold_0.75 | attack:AB/SEX | PWGTP | 0.680625894 | -0.00373523194 | 0.00685280519 |
| Trisk_withhold_0.75 | attack:AB/RAC1P | unweighted | 1.25377822 | -0.00382678529 | 0.00167760898 |
| Trisk_withhold_0.75 | attack:AB/RAC1P | PWGTP | 1.24866457 | -0.00368765947 | 0.00273634366 |
| Trisk_withhold_0.75 | attack:A/public_coverage | unweighted | 0.519349751 | 2.98279983e-05 | 0.00772860771 |
| Trisk_withhold_0.75 | attack:A/public_coverage | PWGTP | 0.521717523 | 0.000182197099 | 0.00668264077 |
| Trisk_withhold_0.75 | attack:A/commute_over20 | unweighted | 0.683932224 | 0.000783280182 | 0.00279209961 |
| Trisk_withhold_0.75 | attack:A/commute_over20 | PWGTP | 0.686641042 | 9.81692641e-05 | 0.0036701412 |
| Trisk_withhold_0.75 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_withhold_0.75 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_withhold_0.75 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_withhold_0.75 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_withhold_0.75 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_withhold_0.75 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_withhold_0.75 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_withhold_0.75 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_withhold_0.75 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_withhold_0.75 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_withhold_0.75 | utility:A/same_residence | unweighted | 0.497833671 | -0.0273179864 | -0.00161488188 |
| Trisk_withhold_0.75 | utility:A/same_residence | PWGTP | 0.473510283 | -0.0255588121 | -0.00334391736 |
| Trisk_withhold_0.75 | utility:A/income_binary | unweighted | 0.296282124 | 0.000583680032 | 0.0117649487 |
| Trisk_withhold_0.75 | utility:A/income_binary | PWGTP | 0.308877068 | 0.0012974811 | 0.01246942 |
| Trisk_withhold_0.75 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Trisk_withhold_0.75 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Trisk_withhold_0.75 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_withhold_0.75 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_withhold_0.75 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_withhold_0.75 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Trisk_withhold_0.75_a33 | attack:A/SEX | unweighted | 0.684987157 | -0.00290650352 | 0.012840973 |
| Trisk_withhold_0.75_a33 | attack:A/SEX | PWGTP | 0.68539451 | -0.00108994219 | 0.00714631102 |
| Trisk_withhold_0.75_a33 | attack:A/RAC1P | unweighted | 1.26755073 | -0.0031886291 | 0.00333691798 |
| Trisk_withhold_0.75_a33 | attack:A/RAC1P | PWGTP | 1.25983076 | -0.00490887884 | 0.00137639611 |
| Trisk_withhold_0.75_a33 | attack:AB/SEX | unweighted | 0.678429436 | -0.00678749889 | 0.00966035467 |
| Trisk_withhold_0.75_a33 | attack:AB/SEX | PWGTP | 0.681760167 | -0.00260095928 | 0.00798707785 |
| Trisk_withhold_0.75_a33 | attack:AB/RAC1P | unweighted | 1.25462348 | -0.00298152628 | 0.002522868 |
| Trisk_withhold_0.75_a33 | attack:AB/RAC1P | PWGTP | 1.24818692 | -0.00416530824 | 0.00225869489 |
| Trisk_withhold_0.75_a33 | attack:A/public_coverage | unweighted | 0.519564448 | 0.000244525743 | 0.00794330546 |
| Trisk_withhold_0.75_a33 | attack:A/public_coverage | PWGTP | 0.522153628 | 0.000618302257 | 0.00711874593 |
| Trisk_withhold_0.75_a33 | attack:A/commute_over20 | unweighted | 0.683221987 | 7.3043321e-05 | 0.00208186274 |
| Trisk_withhold_0.75_a33 | attack:A/commute_over20 | PWGTP | 0.68578153 | -0.000761342858 | 0.00281062908 |
| Trisk_withhold_0.75_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Trisk_withhold_0.75_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Trisk_withhold_0.75_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Trisk_withhold_0.75_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Trisk_withhold_0.75_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Trisk_withhold_0.75_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Trisk_withhold_0.75_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Trisk_withhold_0.75_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Trisk_withhold_0.75_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Trisk_withhold_0.75_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Trisk_withhold_0.75_a33 | utility:A/same_residence | unweighted | 0.495882304 | -0.0292693539 | -0.00356624937 |
| Trisk_withhold_0.75_a33 | utility:A/same_residence | PWGTP | 0.471380491 | -0.0276886045 | -0.00547370976 |
| Trisk_withhold_0.75_a33 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Trisk_withhold_0.75_a33 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Trisk_withhold_0.75_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Trisk_withhold_0.75_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Trisk_withhold_0.75_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Trisk_withhold_0.75_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Trisk_withhold_0.75_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Trisk_withhold_0.75_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_C_0.0005_a17 | attack:A/SEX | unweighted | 0.687349312 | -0.000544348825 | 0.0152031277 |
| Ttask_C_0.0005_a17 | attack:A/SEX | PWGTP | 0.686397423 | -8.70295271e-05 | 0.00814922369 |
| Ttask_C_0.0005_a17 | attack:A/RAC1P | unweighted | 1.27026624 | -0.000473121213 | 0.00605242587 |
| Ttask_C_0.0005_a17 | attack:A/RAC1P | PWGTP | 1.26513698 | 0.000397343819 | 0.00668261876 |
| Ttask_C_0.0005_a17 | attack:AB/SEX | unweighted | 0.685299447 | 8.25126582e-05 | 0.0165303662 |
| Ttask_C_0.0005_a17 | attack:AB/SEX | PWGTP | 0.684397545 | 3.64196213e-05 | 0.0106244568 |
| Ttask_C_0.0005_a17 | attack:AB/RAC1P | unweighted | 1.25751314 | -9.1861456e-05 | 0.00541253282 |
| Ttask_C_0.0005_a17 | attack:AB/RAC1P | PWGTP | 1.25272158 | 0.000369351347 | 0.00679335448 |
| Ttask_C_0.0005_a17 | attack:A/public_coverage | unweighted | 0.51910834 | -0.000211582616 | 0.0074871971 |
| Ttask_C_0.0005_a17 | attack:A/public_coverage | PWGTP | 0.521468165 | -6.71610387e-05 | 0.00643328263 |
| Ttask_C_0.0005_a17 | attack:A/commute_over20 | unweighted | 0.683047656 | -0.000101287737 | 0.00190753169 |
| Ttask_C_0.0005_a17 | attack:A/commute_over20 | PWGTP | 0.68625652 | -0.000286352308 | 0.00328561963 |
| Ttask_C_0.0005_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_C_0.0005_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_C_0.0005_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_C_0.0005_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_C_0.0005_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_C_0.0005_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_C_0.0005_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_C_0.0005_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_C_0.0005_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_C_0.0005_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_C_0.0005_a17 | utility:A/same_residence | unweighted | 0.517182225 | -0.00796943322 | 0.0177336713 |
| Ttask_C_0.0005_a17 | utility:A/same_residence | PWGTP | 0.491943951 | -0.00712514448 | 0.0150897502 |
| Ttask_C_0.0005_a17 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Ttask_C_0.0005_a17 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Ttask_C_0.0005_a17 | utility:A/civilian_at_work | unweighted | 0.279151346 | -0.000155846999 | 0.0077141062 |
| Ttask_C_0.0005_a17 | utility:A/civilian_at_work | PWGTP | 0.267625756 | -0.000189291521 | 0.00515512474 |
| Ttask_C_0.0005_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_C_0.0005_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_C_0.0005_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_C_0.0005_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_C_0.002_a17 | attack:A/SEX | unweighted | 0.685423109 | -0.00247055181 | 0.0132769247 |
| Ttask_C_0.002_a17 | attack:A/SEX | PWGTP | 0.686921948 | 0.000437495988 | 0.0086737492 |
| Ttask_C_0.002_a17 | attack:A/RAC1P | unweighted | 1.27056329 | -0.000176071012 | 0.00634947608 |
| Ttask_C_0.002_a17 | attack:A/RAC1P | PWGTP | 1.26484242 | 0.000102781123 | 0.00638805607 |
| Ttask_C_0.002_a17 | attack:AB/SEX | unweighted | 0.684678353 | -0.000538581667 | 0.0159092719 |
| Ttask_C_0.002_a17 | attack:AB/SEX | PWGTP | 0.683856908 | -0.000504217875 | 0.0100838193 |
| Ttask_C_0.002_a17 | attack:AB/RAC1P | unweighted | 1.25623648 | -0.0013685285 | 0.00413586577 |
| Ttask_C_0.002_a17 | attack:AB/RAC1P | PWGTP | 1.2512903 | -0.00106192835 | 0.00536207478 |
| Ttask_C_0.002_a17 | attack:A/public_coverage | unweighted | 0.518714336 | -0.000605587045 | 0.00709319267 |
| Ttask_C_0.002_a17 | attack:A/public_coverage | PWGTP | 0.521087877 | -0.000447449279 | 0.00605299439 |
| Ttask_C_0.002_a17 | attack:A/commute_over20 | unweighted | 0.682431089 | -0.000717855154 | 0.00129096427 |
| Ttask_C_0.002_a17 | attack:A/commute_over20 | PWGTP | 0.68569377 | -0.000849102759 | 0.00272286918 |
| Ttask_C_0.002_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_C_0.002_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_C_0.002_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_C_0.002_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_C_0.002_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_C_0.002_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_C_0.002_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_C_0.002_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_C_0.002_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_C_0.002_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_C_0.002_a17 | utility:A/same_residence | unweighted | 0.50290909 | -0.0222425682 | 0.00346053639 |
| Ttask_C_0.002_a17 | utility:A/same_residence | PWGTP | 0.478709803 | -0.0203592919 | 0.00185560277 |
| Ttask_C_0.002_a17 | utility:A/income_binary | unweighted | 0.295788472 | 9.00271238e-05 | 0.0112712958 |
| Ttask_C_0.002_a17 | utility:A/income_binary | PWGTP | 0.30793454 | 0.000354953219 | 0.0115268922 |
| Ttask_C_0.002_a17 | utility:A/civilian_at_work | unweighted | 0.279339207 | 3.20147426e-05 | 0.00790196794 |
| Ttask_C_0.002_a17 | utility:A/civilian_at_work | PWGTP | 0.267688889 | -0.000126157678 | 0.00521825858 |
| Ttask_C_0.002_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_C_0.002_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_C_0.002_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_C_0.002_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_C_0.002_a33 | attack:A/SEX | unweighted | 0.684074301 | -0.00381936007 | 0.0119281164 |
| Ttask_C_0.002_a33 | attack:A/SEX | PWGTP | 0.687539668 | 0.00105521601 | 0.00929146922 |
| Ttask_C_0.002_a33 | attack:A/RAC1P | unweighted | 1.27093523 | 0.000195867748 | 0.00672141484 |
| Ttask_C_0.002_a33 | attack:A/RAC1P | PWGTP | 1.26470509 | -3.45532867e-05 | 0.00625072166 |
| Ttask_C_0.002_a33 | attack:AB/SEX | unweighted | 0.684249772 | -0.000967163112 | 0.0154806904 |
| Ttask_C_0.002_a33 | attack:AB/SEX | PWGTP | 0.683833127 | -0.000527998969 | 0.0100600382 |
| Ttask_C_0.002_a33 | attack:AB/RAC1P | unweighted | 1.25536931 | -0.00223569466 | 0.00326869962 |
| Ttask_C_0.002_a33 | attack:AB/RAC1P | PWGTP | 1.25128614 | -0.00106608691 | 0.00535791622 |
| Ttask_C_0.002_a33 | attack:A/public_coverage | unweighted | 0.518641202 | -0.000678720202 | 0.00702005951 |
| Ttask_C_0.002_a33 | attack:A/public_coverage | PWGTP | 0.520968791 | -0.000566535348 | 0.00593390832 |
| Ttask_C_0.002_a33 | attack:A/commute_over20 | unweighted | 0.682465924 | -0.000683019845 | 0.00132579958 |
| Ttask_C_0.002_a33 | attack:A/commute_over20 | PWGTP | 0.685652387 | -0.000890485427 | 0.00268148651 |
| Ttask_C_0.002_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_C_0.002_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_C_0.002_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_C_0.002_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_C_0.002_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_C_0.002_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_C_0.002_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_C_0.002_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_C_0.002_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_C_0.002_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_C_0.002_a33 | utility:A/same_residence | unweighted | 0.503923927 | -0.0212277309 | 0.00447537362 |
| Ttask_C_0.002_a33 | utility:A/same_residence | PWGTP | 0.479469111 | -0.0195999838 | 0.00261491089 |
| Ttask_C_0.002_a33 | utility:A/income_binary | unweighted | 0.295599998 | -9.84459996e-05 | 0.0110828227 |
| Ttask_C_0.002_a33 | utility:A/income_binary | PWGTP | 0.307749931 | 0.000170344167 | 0.0113422831 |
| Ttask_C_0.002_a33 | utility:A/civilian_at_work | unweighted | 0.279241435 | -6.57574745e-05 | 0.00780419572 |
| Ttask_C_0.002_a33 | utility:A/civilian_at_work | PWGTP | 0.267642456 | -0.000172591446 | 0.00517182481 |
| Ttask_C_0.002_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_C_0.002_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_C_0.002_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_C_0.002_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_C_0.01_a17 | attack:A/SEX | unweighted | 0.682579203 | -0.00531445743 | 0.0104330191 |
| Ttask_C_0.01_a17 | attack:A/SEX | PWGTP | 0.683783153 | -0.00270129902 | 0.00553495419 |
| Ttask_C_0.01_a17 | attack:A/RAC1P | unweighted | 1.27146213 | 0.000722771413 | 0.0072483185 |
| Ttask_C_0.01_a17 | attack:A/RAC1P | PWGTP | 1.2638192 | -0.000920441337 | 0.00536483361 |
| Ttask_C_0.01_a17 | attack:AB/SEX | unweighted | 0.67912892 | -0.00608801469 | 0.0103598389 |
| Ttask_C_0.01_a17 | attack:AB/SEX | PWGTP | 0.681331489 | -0.00302963697 | 0.00755840016 |
| Ttask_C_0.01_a17 | attack:AB/RAC1P | unweighted | 1.25600592 | -0.0015990908 | 0.00390530348 |
| Ttask_C_0.01_a17 | attack:AB/RAC1P | PWGTP | 1.25070754 | -0.0016446874 | 0.00477931573 |
| Ttask_C_0.01_a17 | attack:A/public_coverage | unweighted | 0.517878348 | -0.00144157459 | 0.00625720513 |
| Ttask_C_0.01_a17 | attack:A/public_coverage | PWGTP | 0.520414621 | -0.00112070475 | 0.00537973892 |
| Ttask_C_0.01_a17 | attack:A/commute_over20 | unweighted | 0.682357752 | -0.000791191996 | 0.00121762743 |
| Ttask_C_0.01_a17 | attack:A/commute_over20 | PWGTP | 0.685253569 | -0.00128930343 | 0.00228266851 |
| Ttask_C_0.01_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_C_0.01_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_C_0.01_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_C_0.01_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_C_0.01_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_C_0.01_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_C_0.01_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_C_0.01_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_C_0.01_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_C_0.01_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_C_0.01_a17 | utility:A/same_residence | unweighted | 0.489180088 | -0.03597157 | -0.0102684655 |
| Ttask_C_0.01_a17 | utility:A/same_residence | PWGTP | 0.466222528 | -0.0328465665 | -0.0106316718 |
| Ttask_C_0.01_a17 | utility:A/income_binary | unweighted | 0.296251149 | 0.000552704571 | 0.0117339732 |
| Ttask_C_0.01_a17 | utility:A/income_binary | PWGTP | 0.308352628 | 0.000773041741 | 0.0119449807 |
| Ttask_C_0.01_a17 | utility:A/civilian_at_work | unweighted | 0.279367952 | 6.07597598e-05 | 0.00793071296 |
| Ttask_C_0.01_a17 | utility:A/civilian_at_work | PWGTP | 0.2677433 | -7.1747276e-05 | 0.00527266898 |
| Ttask_C_0.01_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_C_0.01_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_C_0.01_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_C_0.01_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_C_0_a17 | attack:A/SEX | unweighted | 0.687194273 | -0.000699387926 | 0.0150480886 |
| Ttask_C_0_a17 | attack:A/SEX | PWGTP | 0.685765489 | -0.000718963597 | 0.00751728962 |
| Ttask_C_0_a17 | attack:A/RAC1P | unweighted | 1.26966346 | -0.00107590233 | 0.00544964476 |
| Ttask_C_0_a17 | attack:A/RAC1P | PWGTP | 1.26451196 | -0.000227675841 | 0.0060575991 |
| Ttask_C_0_a17 | attack:AB/SEX | unweighted | 0.684884733 | -0.00033220186 | 0.0161156517 |
| Ttask_C_0_a17 | attack:AB/SEX | PWGTP | 0.68388427 | -0.000476855607 | 0.0101111815 |
| Ttask_C_0_a17 | attack:AB/RAC1P | unweighted | 1.2576743 | 6.92938184e-05 | 0.00557368809 |
| Ttask_C_0_a17 | attack:AB/RAC1P | PWGTP | 1.25206214 | -0.000290090576 | 0.00613391256 |
| Ttask_C_0_a17 | attack:A/public_coverage | unweighted | 0.518893355 | -0.000426567182 | 0.00727221253 |
| Ttask_C_0_a17 | attack:A/public_coverage | PWGTP | 0.521180529 | -0.000354796783 | 0.00614564689 |
| Ttask_C_0_a17 | attack:A/commute_over20 | unweighted | 0.682885357 | -0.000263587203 | 0.00174523222 |
| Ttask_C_0_a17 | attack:A/commute_over20 | PWGTP | 0.686293269 | -0.000249603634 | 0.0033223683 |
| Ttask_C_0_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_C_0_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_C_0_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_C_0_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_C_0_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_C_0_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_C_0_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_C_0_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_C_0_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_C_0_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_C_0_a17 | utility:A/same_residence | unweighted | 0.524403653 | -0.000748004696 | 0.0249550999 |
| Ttask_C_0_a17 | utility:A/same_residence | PWGTP | 0.498455712 | -0.000613383241 | 0.0216015115 |
| Ttask_C_0_a17 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Ttask_C_0_a17 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Ttask_C_0_a17 | utility:A/civilian_at_work | unweighted | 0.278910646 | -0.000396546568 | 0.00747340663 |
| Ttask_C_0_a17 | utility:A/civilian_at_work | PWGTP | 0.267396309 | -0.000418738339 | 0.00492567792 |
| Ttask_C_0_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_C_0_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_C_0_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_C_0_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_L_0.0005_a17 | attack:A/SEX | unweighted | 0.687893 | -6.6013349e-07 | 0.0157468164 |
| Ttask_L_0.0005_a17 | attack:A/SEX | PWGTP | 0.68719829 | 0.000713837571 | 0.00895009078 |
| Ttask_L_0.0005_a17 | attack:A/RAC1P | unweighted | 1.27095475 | 0.000215392392 | 0.00674093948 |
| Ttask_L_0.0005_a17 | attack:A/RAC1P | PWGTP | 1.26497305 | 0.000233407342 | 0.00651868228 |
| Ttask_L_0.0005_a17 | attack:AB/SEX | unweighted | 0.682017498 | -0.00319943682 | 0.0132484167 |
| Ttask_L_0.0005_a17 | attack:AB/SEX | PWGTP | 0.683598933 | -0.000762192557 | 0.00982584457 |
| Ttask_L_0.0005_a17 | attack:AB/RAC1P | unweighted | 1.25702296 | -0.000582050756 | 0.00492234352 |
| Ttask_L_0.0005_a17 | attack:AB/RAC1P | PWGTP | 1.25195318 | -0.000399051078 | 0.00602495206 |
| Ttask_L_0.0005_a17 | attack:A/public_coverage | unweighted | 0.51900213 | -0.00031779247 | 0.00738098724 |
| Ttask_L_0.0005_a17 | attack:A/public_coverage | PWGTP | 0.521239471 | -0.000295854912 | 0.00620458876 |
| Ttask_L_0.0005_a17 | attack:A/commute_over20 | unweighted | 0.682952855 | -0.000196089353 | 0.00181273007 |
| Ttask_L_0.0005_a17 | attack:A/commute_over20 | PWGTP | 0.686263736 | -0.000279137117 | 0.00329283482 |
| Ttask_L_0.0005_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_L_0.0005_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_L_0.0005_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_L_0.0005_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_L_0.0005_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_L_0.0005_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_L_0.0005_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_L_0.0005_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_L_0.0005_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_L_0.0005_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_L_0.0005_a17 | utility:A/same_residence | unweighted | 0.511826812 | -0.0133248455 | 0.012378259 |
| Ttask_L_0.0005_a17 | utility:A/same_residence | PWGTP | 0.485930355 | -0.0131387404 | 0.00907615436 |
| Ttask_L_0.0005_a17 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Ttask_L_0.0005_a17 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Ttask_L_0.0005_a17 | utility:A/civilian_at_work | unweighted | 0.27949984 | 0.00019264754 | 0.00806260074 |
| Ttask_L_0.0005_a17 | utility:A/civilian_at_work | PWGTP | 0.267596506 | -0.000218540746 | 0.00512587551 |
| Ttask_L_0.0005_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_L_0.0005_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_L_0.0005_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_L_0.0005_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_L_0.002_a17 | attack:A/SEX | unweighted | 0.68632173 | -0.00157193025 | 0.0141755463 |
| Ttask_L_0.002_a17 | attack:A/SEX | PWGTP | 0.685110324 | -0.00137412882 | 0.00686212439 |
| Ttask_L_0.002_a17 | attack:A/RAC1P | unweighted | 1.27107272 | 0.000333358691 | 0.00685890578 |
| Ttask_L_0.002_a17 | attack:A/RAC1P | PWGTP | 1.2642893 | -0.000450343974 | 0.00583493097 |
| Ttask_L_0.002_a17 | attack:AB/SEX | unweighted | 0.684815805 | -0.000401129542 | 0.016046724 |
| Ttask_L_0.002_a17 | attack:AB/SEX | PWGTP | 0.684558428 | 0.00019730188 | 0.010785339 |
| Ttask_L_0.002_a17 | attack:AB/RAC1P | unweighted | 1.25656495 | -0.00104005734 | 0.00446433694 |
| Ttask_L_0.002_a17 | attack:AB/RAC1P | PWGTP | 1.25142932 | -0.000922910909 | 0.00550109223 |
| Ttask_L_0.002_a17 | attack:A/public_coverage | unweighted | 0.519942349 | 0.000622426778 | 0.00832120649 |
| Ttask_L_0.002_a17 | attack:A/public_coverage | PWGTP | 0.522096568 | 0.000561242109 | 0.00706168578 |
| Ttask_L_0.002_a17 | attack:A/commute_over20 | unweighted | 0.682043862 | -0.00110508202 | 0.000903737403 |
| Ttask_L_0.002_a17 | attack:A/commute_over20 | PWGTP | 0.685045694 | -0.00149717846 | 0.00207479348 |
| Ttask_L_0.002_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_L_0.002_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_L_0.002_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_L_0.002_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_L_0.002_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_L_0.002_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_L_0.002_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_L_0.002_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_L_0.002_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_L_0.002_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_L_0.002_a17 | utility:A/same_residence | unweighted | 0.502724068 | -0.0224275893 | 0.00327551522 |
| Ttask_L_0.002_a17 | utility:A/same_residence | PWGTP | 0.477907128 | -0.0211619665 | 0.00105292819 |
| Ttask_L_0.002_a17 | utility:A/income_binary | unweighted | 0.295837218 | 0.000138773219 | 0.0113200419 |
| Ttask_L_0.002_a17 | utility:A/income_binary | PWGTP | 0.307719318 | 0.000139731684 | 0.0113116706 |
| Ttask_L_0.002_a17 | utility:A/civilian_at_work | unweighted | 0.27971716 | 0.000409967052 | 0.00827992025 |
| Ttask_L_0.002_a17 | utility:A/civilian_at_work | PWGTP | 0.267713079 | -0.000101967941 | 0.00524244832 |
| Ttask_L_0.002_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_L_0.002_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_L_0.002_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_L_0.002_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_L_0.002_a33 | attack:A/SEX | unweighted | 0.681864119 | -0.0060295416 | 0.0097179349 |
| Ttask_L_0.002_a33 | attack:A/SEX | PWGTP | 0.685951394 | -0.000533058679 | 0.00770319453 |
| Ttask_L_0.002_a33 | attack:A/RAC1P | unweighted | 1.27102671 | 0.000287347557 | 0.00681289464 |
| Ttask_L_0.002_a33 | attack:A/RAC1P | PWGTP | 1.26457017 | -0.000169468276 | 0.00611580667 |
| Ttask_L_0.002_a33 | attack:AB/SEX | unweighted | 0.683047043 | -0.00216989201 | 0.0142779615 |
| Ttask_L_0.002_a33 | attack:AB/SEX | PWGTP | 0.683843593 | -0.000517533246 | 0.0100705039 |
| Ttask_L_0.002_a33 | attack:AB/RAC1P | unweighted | 1.25614003 | -0.00146497837 | 0.00403941591 |
| Ttask_L_0.002_a33 | attack:AB/RAC1P | PWGTP | 1.25150459 | -0.000847638275 | 0.00557636486 |
| Ttask_L_0.002_a33 | attack:A/public_coverage | unweighted | 0.51856352 | -0.000756402681 | 0.00694237703 |
| Ttask_L_0.002_a33 | attack:A/public_coverage | PWGTP | 0.520808834 | -0.000726491793 | 0.00577395188 |
| Ttask_L_0.002_a33 | attack:A/commute_over20 | unweighted | 0.682793914 | -0.000355030611 | 0.00165378881 |
| Ttask_L_0.002_a33 | attack:A/commute_over20 | PWGTP | 0.686078162 | -0.000464710212 | 0.00310726173 |
| Ttask_L_0.002_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_L_0.002_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_L_0.002_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_L_0.002_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_L_0.002_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_L_0.002_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_L_0.002_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_L_0.002_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_L_0.002_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_L_0.002_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_L_0.002_a33 | utility:A/same_residence | unweighted | 0.501208979 | -0.0239426784 | 0.00176042613 |
| Ttask_L_0.002_a33 | utility:A/same_residence | PWGTP | 0.476036233 | -0.0230328625 | -0.000817967778 |
| Ttask_L_0.002_a33 | utility:A/income_binary | unweighted | 0.29562204 | -7.64039534e-05 | 0.0111048647 |
| Ttask_L_0.002_a33 | utility:A/income_binary | PWGTP | 0.308030663 | 0.000451076379 | 0.0116230153 |
| Ttask_L_0.002_a33 | utility:A/civilian_at_work | unweighted | 0.279717937 | 0.00041074396 | 0.00828069716 |
| Ttask_L_0.002_a33 | utility:A/civilian_at_work | PWGTP | 0.267839894 | 2.48463464e-05 | 0.00536926261 |
| Ttask_L_0.002_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_L_0.002_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_L_0.002_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_L_0.002_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_L_0.01_a17 | attack:A/SEX | unweighted | 0.68543758 | -0.00245608047 | 0.013291396 |
| Ttask_L_0.01_a17 | attack:A/SEX | PWGTP | 0.684469842 | -0.00201461065 | 0.00622164256 |
| Ttask_L_0.01_a17 | attack:A/RAC1P | unweighted | 1.27086919 | 0.000129827663 | 0.00665537475 |
| Ttask_L_0.01_a17 | attack:A/RAC1P | PWGTP | 1.26438216 | -0.000357482726 | 0.00592779222 |
| Ttask_L_0.01_a17 | attack:AB/SEX | unweighted | 0.680664511 | -0.00455242353 | 0.01189543 |
| Ttask_L_0.01_a17 | attack:AB/SEX | PWGTP | 0.680802359 | -0.00355876659 | 0.00702927054 |
| Ttask_L_0.01_a17 | attack:AB/RAC1P | unweighted | 1.25618809 | -0.00141691916 | 0.00408747511 |
| Ttask_L_0.01_a17 | attack:AB/RAC1P | PWGTP | 1.25141527 | -0.000936961055 | 0.00548704208 |
| Ttask_L_0.01_a17 | attack:A/public_coverage | unweighted | 0.519335165 | 1.52419283e-05 | 0.00771402164 |
| Ttask_L_0.01_a17 | attack:A/public_coverage | PWGTP | 0.521756174 | 0.000220848109 | 0.00672129178 |
| Ttask_L_0.01_a17 | attack:A/commute_over20 | unweighted | 0.683018293 | -0.000130651497 | 0.00187816793 |
| Ttask_L_0.01_a17 | attack:A/commute_over20 | PWGTP | 0.68591924 | -0.000623632854 | 0.00294833908 |
| Ttask_L_0.01_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_L_0.01_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_L_0.01_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_L_0.01_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_L_0.01_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_L_0.01_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_L_0.01_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_L_0.01_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_L_0.01_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_L_0.01_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_L_0.01_a17 | utility:A/same_residence | unweighted | 0.487227122 | -0.0379245361 | -0.0122214316 |
| Ttask_L_0.01_a17 | utility:A/same_residence | PWGTP | 0.464031259 | -0.0350378364 | -0.0128229417 |
| Ttask_L_0.01_a17 | utility:A/income_binary | unweighted | 0.296129052 | 0.000430607102 | 0.0116118758 |
| Ttask_L_0.01_a17 | utility:A/income_binary | PWGTP | 0.308615201 | 0.00103561418 | 0.0122075531 |
| Ttask_L_0.01_a17 | utility:A/civilian_at_work | unweighted | 0.28037454 | 0.00106734704 | 0.00893730024 |
| Ttask_L_0.01_a17 | utility:A/civilian_at_work | PWGTP | 0.268168757 | 0.000353710066 | 0.00569812633 |
| Ttask_L_0.01_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_L_0.01_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_L_0.01_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_L_0.01_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_L_0_a17 | attack:A/SEX | unweighted | 0.687397505 | -0.000496155924 | 0.0152513206 |
| Ttask_L_0_a17 | attack:A/SEX | PWGTP | 0.686258989 | -0.000225463282 | 0.00801078993 |
| Ttask_L_0_a17 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| Ttask_L_0_a17 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| Ttask_L_0_a17 | attack:AB/SEX | unweighted | 0.684751062 | -0.000465872617 | 0.0159819809 |
| Ttask_L_0_a17 | attack:AB/SEX | PWGTP | 0.6840299 | -0.000331225646 | 0.0102568115 |
| Ttask_L_0_a17 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| Ttask_L_0_a17 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| Ttask_L_0_a17 | attack:A/public_coverage | unweighted | 0.519149527 | -0.000170395693 | 0.00752838402 |
| Ttask_L_0_a17 | attack:A/public_coverage | PWGTP | 0.52142167 | -0.000113655897 | 0.00638678777 |
| Ttask_L_0_a17 | attack:A/commute_over20 | unweighted | 0.682816953 | -0.000331991146 | 0.00167682828 |
| Ttask_L_0_a17 | attack:A/commute_over20 | PWGTP | 0.686254404 | -0.000288469023 | 0.00328350292 |
| Ttask_L_0_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_L_0_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_L_0_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_L_0_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_L_0_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_L_0_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_L_0_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_L_0_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_L_0_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_L_0_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_L_0_a17 | utility:A/same_residence | unweighted | 0.523982121 | -0.00116953693 | 0.0245335676 |
| Ttask_L_0_a17 | utility:A/same_residence | PWGTP | 0.497314855 | -0.00175423964 | 0.0204606551 |
| Ttask_L_0_a17 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Ttask_L_0_a17 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Ttask_L_0_a17 | utility:A/civilian_at_work | unweighted | 0.279796345 | 0.000489152346 | 0.00835910554 |
| Ttask_L_0_a17 | utility:A/civilian_at_work | PWGTP | 0.26797382 | 0.000158772717 | 0.00550318898 |
| Ttask_L_0_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_L_0_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_L_0_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_L_0_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_U_unconstrained_a17 | attack:A/SEX | unweighted | 0.681790422 | -0.00610323812 | 0.00964423837 |
| Ttask_U_unconstrained_a17 | attack:A/SEX | PWGTP | 0.683694627 | -0.00278982518 | 0.00544642803 |
| Ttask_U_unconstrained_a17 | attack:A/RAC1P | unweighted | 1.27114151 | 0.000402146798 | 0.00692769389 |
| Ttask_U_unconstrained_a17 | attack:A/RAC1P | PWGTP | 1.26381428 | -0.000925358251 | 0.00535991669 |
| Ttask_U_unconstrained_a17 | attack:AB/SEX | unweighted | 0.680113795 | -0.00510313969 | 0.0113447139 |
| Ttask_U_unconstrained_a17 | attack:AB/SEX | PWGTP | 0.682669122 | -0.00169200364 | 0.0088960335 |
| Ttask_U_unconstrained_a17 | attack:AB/RAC1P | unweighted | 1.25626195 | -0.00134305875 | 0.00416133552 |
| Ttask_U_unconstrained_a17 | attack:AB/RAC1P | PWGTP | 1.25127548 | -0.00107674985 | 0.00534725328 |
| Ttask_U_unconstrained_a17 | attack:A/public_coverage | unweighted | 0.519113171 | -0.000206751794 | 0.00749202792 |
| Ttask_U_unconstrained_a17 | attack:A/public_coverage | PWGTP | 0.520916652 | -0.00061867418 | 0.00588176949 |
| Ttask_U_unconstrained_a17 | attack:A/commute_over20 | unweighted | 0.682580845 | -0.000568099372 | 0.00144072005 |
| Ttask_U_unconstrained_a17 | attack:A/commute_over20 | PWGTP | 0.684670858 | -0.00187201475 | 0.00169995719 |
| Ttask_U_unconstrained_a17 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_U_unconstrained_a17 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_U_unconstrained_a17 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_U_unconstrained_a17 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_U_unconstrained_a17 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_U_unconstrained_a17 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_U_unconstrained_a17 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_U_unconstrained_a17 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_U_unconstrained_a17 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_U_unconstrained_a17 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_U_unconstrained_a17 | utility:A/same_residence | unweighted | 0.48661734 | -0.0385343174 | -0.0128312128 |
| Ttask_U_unconstrained_a17 | utility:A/same_residence | PWGTP | 0.464792951 | -0.0342761439 | -0.0120612492 |
| Ttask_U_unconstrained_a17 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Ttask_U_unconstrained_a17 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Ttask_U_unconstrained_a17 | utility:A/civilian_at_work | unweighted | 0.279317425 | 1.02319222e-05 | 0.00788018512 |
| Ttask_U_unconstrained_a17 | utility:A/civilian_at_work | PWGTP | 0.2677423 | -7.27475876e-05 | 0.00527166867 |
| Ttask_U_unconstrained_a17 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_U_unconstrained_a17 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_U_unconstrained_a17 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_U_unconstrained_a17 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_U_unconstrained_a33 | attack:A/SEX | unweighted | 0.686096223 | -0.00179743777 | 0.0139500387 |
| Ttask_U_unconstrained_a33 | attack:A/SEX | PWGTP | 0.686021261 | -0.00046319161 | 0.0077730616 |
| Ttask_U_unconstrained_a33 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| Ttask_U_unconstrained_a33 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| Ttask_U_unconstrained_a33 | attack:AB/SEX | unweighted | 0.682711516 | -0.00250541846 | 0.0139424351 |
| Ttask_U_unconstrained_a33 | attack:AB/SEX | PWGTP | 0.682587738 | -0.00177338783 | 0.0088146493 |
| Ttask_U_unconstrained_a33 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| Ttask_U_unconstrained_a33 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| Ttask_U_unconstrained_a33 | attack:A/public_coverage | unweighted | 0.519500336 | 0.000180413523 | 0.00787919324 |
| Ttask_U_unconstrained_a33 | attack:A/public_coverage | PWGTP | 0.52305925 | 0.00152392445 | 0.00802436812 |
| Ttask_U_unconstrained_a33 | attack:A/commute_over20 | unweighted | 0.682772371 | -0.000376572809 | 0.00163224661 |
| Ttask_U_unconstrained_a33 | attack:A/commute_over20 | PWGTP | 0.685947352 | -0.000595520819 | 0.00297645112 |
| Ttask_U_unconstrained_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_U_unconstrained_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_U_unconstrained_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_U_unconstrained_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_U_unconstrained_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_U_unconstrained_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_U_unconstrained_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_U_unconstrained_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_U_unconstrained_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_U_unconstrained_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_U_unconstrained_a33 | utility:A/same_residence | unweighted | 0.48617062 | -0.0389810381 | -0.0132779335 |
| Ttask_U_unconstrained_a33 | utility:A/same_residence | PWGTP | 0.464168465 | -0.0349006296 | -0.0126857349 |
| Ttask_U_unconstrained_a33 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Ttask_U_unconstrained_a33 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Ttask_U_unconstrained_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Ttask_U_unconstrained_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Ttask_U_unconstrained_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_U_unconstrained_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_U_unconstrained_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_U_unconstrained_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_code | attack:A/SEX | unweighted | 0.686161419 | -0.00173224122 | 0.0140152353 |
| Ttask_code | attack:A/SEX | PWGTP | 0.686709413 | 0.000224960873 | 0.00846121409 |
| Ttask_code | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| Ttask_code | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| Ttask_code | attack:AB/SEX | unweighted | 0.683943238 | -0.00127369728 | 0.0151741563 |
| Ttask_code | attack:AB/SEX | PWGTP | 0.685095802 | 0.000734676268 | 0.0113227134 |
| Ttask_code | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| Ttask_code | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| Ttask_code | attack:A/public_coverage | unweighted | 0.519447814 | 0.000127891315 | 0.00782667103 |
| Ttask_code | attack:A/public_coverage | PWGTP | 0.52339377 | 0.00185844403 | 0.0083588877 |
| Ttask_code | attack:A/commute_over20 | unweighted | 0.683148944 | 0 | 0.00200881942 |
| Ttask_code | attack:A/commute_over20 | PWGTP | 0.686542873 | 0 | 0.00357197194 |
| Ttask_code | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_code | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_code | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_code | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_code | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_code | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_code | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_code | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_code | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_code | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_code | utility:A/same_residence | unweighted | 0.491722359 | -0.0334292989 | -0.00772619437 |
| Ttask_code | utility:A/same_residence | PWGTP | 0.468271459 | -0.0307976365 | -0.00858274174 |
| Ttask_code | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Ttask_code | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Ttask_code | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Ttask_code | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Ttask_code | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_code | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_code | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_code | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_rr_0.25 | attack:A/SEX | unweighted | 0.68735981 | -0.000533850893 | 0.0152136256 |
| Ttask_rr_0.25 | attack:A/SEX | PWGTP | 0.685891061 | -0.000593391762 | 0.00764286145 |
| Ttask_rr_0.25 | attack:A/RAC1P | unweighted | 1.27063225 | -0.000107112126 | 0.00641843496 |
| Ttask_rr_0.25 | attack:A/RAC1P | PWGTP | 1.26497869 | 0.000239047711 | 0.00652432265 |
| Ttask_rr_0.25 | attack:AB/SEX | unweighted | 0.68499885 | -0.000218084997 | 0.0162297686 |
| Ttask_rr_0.25 | attack:AB/SEX | PWGTP | 0.684012073 | -0.000349052695 | 0.0102389844 |
| Ttask_rr_0.25 | attack:AB/RAC1P | unweighted | 1.25787877 | 0.000273764745 | 0.00577815902 |
| Ttask_rr_0.25 | attack:AB/RAC1P | PWGTP | 1.25268699 | 0.000334760027 | 0.00675876316 |
| Ttask_rr_0.25 | attack:A/public_coverage | unweighted | 0.518996352 | -0.000323570877 | 0.00737520884 |
| Ttask_rr_0.25 | attack:A/public_coverage | PWGTP | 0.521208685 | -0.000326640557 | 0.00617380311 |
| Ttask_rr_0.25 | attack:A/commute_over20 | unweighted | 0.683215626 | 6.66815171e-05 | 0.00207550094 |
| Ttask_rr_0.25 | attack:A/commute_over20 | PWGTP | 0.686547051 | 4.17859417e-06 | 0.00357615053 |
| Ttask_rr_0.25 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_rr_0.25 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_rr_0.25 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_rr_0.25 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_rr_0.25 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_rr_0.25 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_rr_0.25 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_rr_0.25 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_rr_0.25 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_rr_0.25 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_rr_0.25 | utility:A/same_residence | unweighted | 0.52235825 | -0.00279340736 | 0.0229096972 |
| Ttask_rr_0.25 | utility:A/same_residence | PWGTP | 0.496692441 | -0.00237665394 | 0.0198382408 |
| Ttask_rr_0.25 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Ttask_rr_0.25 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Ttask_rr_0.25 | utility:A/civilian_at_work | unweighted | 0.279317931 | 1.07379122e-05 | 0.00788069111 |
| Ttask_rr_0.25 | utility:A/civilian_at_work | PWGTP | 0.267729642 | -8.54053341e-05 | 0.00525901093 |
| Ttask_rr_0.25 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_rr_0.25 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_rr_0.25 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_rr_0.25 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_rr_0.25_a33 | attack:A/SEX | unweighted | 0.687146832 | -0.000746828606 | 0.0150006479 |
| Ttask_rr_0.25_a33 | attack:A/SEX | PWGTP | 0.68815238 | 0.00166792749 | 0.0099041807 |
| Ttask_rr_0.25_a33 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| Ttask_rr_0.25_a33 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| Ttask_rr_0.25_a33 | attack:AB/SEX | unweighted | 0.681821988 | -0.00339494702 | 0.0130529065 |
| Ttask_rr_0.25_a33 | attack:AB/SEX | PWGTP | 0.683698617 | -0.000662509003 | 0.00992552813 |
| Ttask_rr_0.25_a33 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| Ttask_rr_0.25_a33 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| Ttask_rr_0.25_a33 | attack:A/public_coverage | unweighted | 0.519156487 | -0.000163435703 | 0.00753534401 |
| Ttask_rr_0.25_a33 | attack:A/public_coverage | PWGTP | 0.521440609 | -9.47169615e-05 | 0.00640572671 |
| Ttask_rr_0.25_a33 | attack:A/commute_over20 | unweighted | 0.683189933 | 4.09891476e-05 | 0.00204980857 |
| Ttask_rr_0.25_a33 | attack:A/commute_over20 | PWGTP | 0.686560495 | 1.76225393e-05 | 0.00358959448 |
| Ttask_rr_0.25_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_rr_0.25_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_rr_0.25_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_rr_0.25_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_rr_0.25_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_rr_0.25_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_rr_0.25_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_rr_0.25_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_rr_0.25_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_rr_0.25_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_rr_0.25_a33 | utility:A/same_residence | unweighted | 0.522546823 | -0.00260483515 | 0.0230982694 |
| Ttask_rr_0.25_a33 | utility:A/same_residence | PWGTP | 0.496453117 | -0.00261597774 | 0.019598917 |
| Ttask_rr_0.25_a33 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Ttask_rr_0.25_a33 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Ttask_rr_0.25_a33 | utility:A/civilian_at_work | unweighted | 0.279444261 | 0.00013706804 | 0.00800702124 |
| Ttask_rr_0.25_a33 | utility:A/civilian_at_work | PWGTP | 0.267780777 | -3.42701293e-05 | 0.00531014613 |
| Ttask_rr_0.25_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_rr_0.25_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_rr_0.25_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_rr_0.25_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_rr_0.5 | attack:A/SEX | unweighted | 0.684975163 | -0.00291849744 | 0.0128289791 |
| Ttask_rr_0.5 | attack:A/SEX | PWGTP | 0.686014885 | -0.000469567617 | 0.0077666856 |
| Ttask_rr_0.5 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| Ttask_rr_0.5 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| Ttask_rr_0.5 | attack:AB/SEX | unweighted | 0.68521846 | 1.5254188e-06 | 0.016449379 |
| Ttask_rr_0.5 | attack:AB/SEX | PWGTP | 0.684567018 | 0.000205892155 | 0.0107939293 |
| Ttask_rr_0.5 | attack:AB/RAC1P | unweighted | 1.25709184 | -0.000513161407 | 0.00499123287 |
| Ttask_rr_0.5 | attack:AB/RAC1P | PWGTP | 1.25230763 | -4.45996206e-05 | 0.00637940351 |
| Ttask_rr_0.5 | attack:A/public_coverage | unweighted | 0.519368214 | 4.82914214e-05 | 0.00774707114 |
| Ttask_rr_0.5 | attack:A/public_coverage | PWGTP | 0.521674311 | 0.000138985308 | 0.00663942898 |
| Ttask_rr_0.5 | attack:A/commute_over20 | unweighted | 0.682748599 | -0.000400345392 | 0.00160847403 |
| Ttask_rr_0.5 | attack:A/commute_over20 | PWGTP | 0.685797964 | -0.000744908295 | 0.00282706364 |
| Ttask_rr_0.5 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_rr_0.5 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_rr_0.5 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_rr_0.5 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_rr_0.5 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_rr_0.5 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_rr_0.5 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_rr_0.5 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_rr_0.5 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_rr_0.5 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_rr_0.5 | utility:A/same_residence | unweighted | 0.5151534 | -0.00999825777 | 0.0157048468 |
| Ttask_rr_0.5 | utility:A/same_residence | PWGTP | 0.489957295 | -0.00911180026 | 0.0131030944 |
| Ttask_rr_0.5 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Ttask_rr_0.5 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Ttask_rr_0.5 | utility:A/civilian_at_work | unweighted | 0.279210709 | -9.64835425e-05 | 0.00777346965 |
| Ttask_rr_0.5 | utility:A/civilian_at_work | PWGTP | 0.267629602 | -0.000185444956 | 0.0051589713 |
| Ttask_rr_0.5 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_rr_0.5 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_rr_0.5 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_rr_0.5 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_rr_0.5_a33 | attack:A/SEX | unweighted | 0.682846386 | -0.00504727451 | 0.010700202 |
| Ttask_rr_0.5_a33 | attack:A/SEX | PWGTP | 0.685499566 | -0.000984886315 | 0.0072513669 |
| Ttask_rr_0.5_a33 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| Ttask_rr_0.5_a33 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| Ttask_rr_0.5_a33 | attack:AB/SEX | unweighted | 0.680925336 | -0.00429159919 | 0.0121562544 |
| Ttask_rr_0.5_a33 | attack:AB/SEX | PWGTP | 0.682866123 | -0.00149500256 | 0.00909303457 |
| Ttask_rr_0.5_a33 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| Ttask_rr_0.5_a33 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| Ttask_rr_0.5_a33 | attack:A/public_coverage | unweighted | 0.519531411 | 0.000211488819 | 0.00791026853 |
| Ttask_rr_0.5_a33 | attack:A/public_coverage | PWGTP | 0.521677518 | 0.00014219172 | 0.00664263539 |
| Ttask_rr_0.5_a33 | attack:A/commute_over20 | unweighted | 0.683323661 | 0.000174716589 | 0.00218353601 |
| Ttask_rr_0.5_a33 | attack:A/commute_over20 | PWGTP | 0.686443653 | -9.92191627e-05 | 0.00347275278 |
| Ttask_rr_0.5_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_rr_0.5_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_rr_0.5_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_rr_0.5_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_rr_0.5_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_rr_0.5_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_rr_0.5_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_rr_0.5_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_rr_0.5_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_rr_0.5_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_rr_0.5_a33 | utility:A/same_residence | unweighted | 0.513032641 | -0.0121190166 | 0.0135840879 |
| Ttask_rr_0.5_a33 | utility:A/same_residence | PWGTP | 0.488463798 | -0.0106052966 | 0.0116095981 |
| Ttask_rr_0.5_a33 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Ttask_rr_0.5_a33 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Ttask_rr_0.5_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Ttask_rr_0.5_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Ttask_rr_0.5_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_rr_0.5_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_rr_0.5_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_rr_0.5_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_rr_0.75 | attack:A/SEX | unweighted | 0.687759096 | -0.000134564106 | 0.0156129124 |
| Ttask_rr_0.75 | attack:A/SEX | PWGTP | 0.686094643 | -0.000389809102 | 0.00784644411 |
| Ttask_rr_0.75 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| Ttask_rr_0.75 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| Ttask_rr_0.75 | attack:AB/SEX | unweighted | 0.683816269 | -0.0014006656 | 0.0150471879 |
| Ttask_rr_0.75 | attack:AB/SEX | PWGTP | 0.68379988 | -0.000561246286 | 0.0100267908 |
| Ttask_rr_0.75 | attack:AB/RAC1P | unweighted | 1.25688047 | -0.000724533639 | 0.00477986064 |
| Ttask_rr_0.75 | attack:AB/RAC1P | PWGTP | 1.25198766 | -0.000364572186 | 0.00605943095 |
| Ttask_rr_0.75 | attack:A/public_coverage | unweighted | 0.519319923 | 0 | 0.00769877971 |
| Ttask_rr_0.75 | attack:A/public_coverage | PWGTP | 0.521535326 | 0 | 0.00650044367 |
| Ttask_rr_0.75 | attack:A/commute_over20 | unweighted | 0.68250271 | -0.000646233887 | 0.00136258554 |
| Ttask_rr_0.75 | attack:A/commute_over20 | PWGTP | 0.685188553 | -0.00135432003 | 0.0022176519 |
| Ttask_rr_0.75 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_rr_0.75 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_rr_0.75 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_rr_0.75 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_rr_0.75 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_rr_0.75 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_rr_0.75 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_rr_0.75 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_rr_0.75 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_rr_0.75 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_rr_0.75 | utility:A/same_residence | unweighted | 0.505323658 | -0.0198280001 | 0.00587510448 |
| Ttask_rr_0.75 | utility:A/same_residence | PWGTP | 0.480978708 | -0.0180903872 | 0.0041245075 |
| Ttask_rr_0.75 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Ttask_rr_0.75 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Ttask_rr_0.75 | utility:A/civilian_at_work | unweighted | 0.28005379 | 0.000746597048 | 0.00861655025 |
| Ttask_rr_0.75 | utility:A/civilian_at_work | PWGTP | 0.268245074 | 0.000430026611 | 0.00577444287 |
| Ttask_rr_0.75 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_rr_0.75 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_rr_0.75 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_rr_0.75 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_rr_0.75_a33 | attack:A/SEX | unweighted | 0.684773351 | -0.00312030998 | 0.0126271665 |
| Ttask_rr_0.75_a33 | attack:A/SEX | PWGTP | 0.685453901 | -0.00103055104 | 0.00720570217 |
| Ttask_rr_0.75_a33 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| Ttask_rr_0.75_a33 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| Ttask_rr_0.75_a33 | attack:AB/SEX | unweighted | 0.683800637 | -0.00141629824 | 0.0150315553 |
| Ttask_rr_0.75_a33 | attack:AB/SEX | PWGTP | 0.685337492 | 0.000976366204 | 0.0115644033 |
| Ttask_rr_0.75_a33 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| Ttask_rr_0.75_a33 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| Ttask_rr_0.75_a33 | attack:A/public_coverage | unweighted | 0.519498107 | 0.000178184194 | 0.00787696391 |
| Ttask_rr_0.75_a33 | attack:A/public_coverage | PWGTP | 0.521615175 | 7.98490963e-05 | 0.00658029277 |
| Ttask_rr_0.75_a33 | attack:A/commute_over20 | unweighted | 0.68342095 | 0.000272005474 | 0.0022808249 |
| Ttask_rr_0.75_a33 | attack:A/commute_over20 | PWGTP | 0.686285606 | -0.000257266424 | 0.00331470551 |
| Ttask_rr_0.75_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_rr_0.75_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_rr_0.75_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_rr_0.75_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_rr_0.75_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_rr_0.75_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_rr_0.75_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_rr_0.75_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_rr_0.75_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_rr_0.75_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_rr_0.75_a33 | utility:A/same_residence | unweighted | 0.504123137 | -0.0210285207 | 0.00467458385 |
| Ttask_rr_0.75_a33 | utility:A/same_residence | PWGTP | 0.479970675 | -0.01909842 | 0.00311647466 |
| Ttask_rr_0.75_a33 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Ttask_rr_0.75_a33 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Ttask_rr_0.75_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Ttask_rr_0.75_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Ttask_rr_0.75_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_rr_0.75_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_rr_0.75_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_rr_0.75_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_withhold_0.25 | attack:A/SEX | unweighted | 0.685487585 | -0.0024060757 | 0.0133414008 |
| Ttask_withhold_0.25 | attack:A/SEX | PWGTP | 0.686077917 | -0.000406535036 | 0.00782971818 |
| Ttask_withhold_0.25 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| Ttask_withhold_0.25 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| Ttask_withhold_0.25 | attack:AB/SEX | unweighted | 0.681867457 | -0.00334947758 | 0.013098376 |
| Ttask_withhold_0.25 | attack:AB/SEX | PWGTP | 0.68332039 | -0.00104073535 | 0.00954730178 |
| Ttask_withhold_0.25 | attack:AB/RAC1P | unweighted | 1.25727351 | -0.000331499255 | 0.00517289502 |
| Ttask_withhold_0.25 | attack:AB/RAC1P | PWGTP | 1.25217775 | -0.000174475518 | 0.00624952762 |
| Ttask_withhold_0.25 | attack:A/public_coverage | unweighted | 0.519419592 | 9.96690224e-05 | 0.00779844874 |
| Ttask_withhold_0.25 | attack:A/public_coverage | PWGTP | 0.52152537 | -9.95564158e-06 | 0.00649048803 |
| Ttask_withhold_0.25 | attack:A/commute_over20 | unweighted | 0.683245137 | 9.6193089e-05 | 0.00210501251 |
| Ttask_withhold_0.25 | attack:A/commute_over20 | PWGTP | 0.686481443 | -6.1429236e-05 | 0.0035105427 |
| Ttask_withhold_0.25 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_withhold_0.25 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_withhold_0.25 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_withhold_0.25 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_withhold_0.25 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_withhold_0.25 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_withhold_0.25 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_withhold_0.25 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_withhold_0.25 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_withhold_0.25 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_withhold_0.25 | utility:A/same_residence | unweighted | 0.515776299 | -0.00937535862 | 0.0163277459 |
| Ttask_withhold_0.25 | utility:A/same_residence | PWGTP | 0.490358622 | -0.00871047345 | 0.0135044213 |
| Ttask_withhold_0.25 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Ttask_withhold_0.25 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Ttask_withhold_0.25 | utility:A/civilian_at_work | unweighted | 0.279830415 | 0.000523221814 | 0.00839317501 |
| Ttask_withhold_0.25 | utility:A/civilian_at_work | PWGTP | 0.268067673 | 0.000252625403 | 0.00559704166 |
| Ttask_withhold_0.25 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_withhold_0.25 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_withhold_0.25 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_withhold_0.25 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_withhold_0.25_a33 | attack:A/SEX | unweighted | 0.684960502 | -0.00293315906 | 0.0128143174 |
| Ttask_withhold_0.25_a33 | attack:A/SEX | PWGTP | 0.685988626 | -0.000495825991 | 0.00774042722 |
| Ttask_withhold_0.25_a33 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| Ttask_withhold_0.25_a33 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| Ttask_withhold_0.25_a33 | attack:AB/SEX | unweighted | 0.68471556 | -0.000501375072 | 0.0159464785 |
| Ttask_withhold_0.25_a33 | attack:AB/SEX | PWGTP | 0.68393264 | -0.00042848634 | 0.0101595508 |
| Ttask_withhold_0.25_a33 | attack:AB/RAC1P | unweighted | 1.25771944 | 0.000114435656 | 0.00561882993 |
| Ttask_withhold_0.25_a33 | attack:AB/RAC1P | PWGTP | 1.2522601 | -9.21301612e-05 | 0.00633187297 |
| Ttask_withhold_0.25_a33 | attack:A/public_coverage | unweighted | 0.519790353 | 0.000470430719 | 0.00816921043 |
| Ttask_withhold_0.25_a33 | attack:A/public_coverage | PWGTP | 0.522032992 | 0.00049766628 | 0.00699810995 |
| Ttask_withhold_0.25_a33 | attack:A/commute_over20 | unweighted | 0.683199872 | 5.09276507e-05 | 0.00205974707 |
| Ttask_withhold_0.25_a33 | attack:A/commute_over20 | PWGTP | 0.686368391 | -0.000174481991 | 0.00339748995 |
| Ttask_withhold_0.25_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_withhold_0.25_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_withhold_0.25_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_withhold_0.25_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_withhold_0.25_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_withhold_0.25_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_withhold_0.25_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_withhold_0.25_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_withhold_0.25_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_withhold_0.25_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_withhold_0.25_a33 | utility:A/same_residence | unweighted | 0.516016587 | -0.00913507095 | 0.0165680336 |
| Ttask_withhold_0.25_a33 | utility:A/same_residence | PWGTP | 0.490500047 | -0.00856904799 | 0.0136458467 |
| Ttask_withhold_0.25_a33 | utility:A/income_binary | unweighted | 0.295776714 | 7.82693773e-05 | 0.011259538 |
| Ttask_withhold_0.25_a33 | utility:A/income_binary | PWGTP | 0.307605676 | 2.60895101e-05 | 0.0111980285 |
| Ttask_withhold_0.25_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Ttask_withhold_0.25_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Ttask_withhold_0.25_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_withhold_0.25_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_withhold_0.25_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_withhold_0.25_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_withhold_0.5 | attack:A/SEX | unweighted | 0.684688478 | -0.00320518263 | 0.0125422939 |
| Ttask_withhold_0.5 | attack:A/SEX | PWGTP | 0.685406052 | -0.00107840042 | 0.0071578528 |
| Ttask_withhold_0.5 | attack:A/RAC1P | unweighted | 1.27115947 | 0.000420106201 | 0.00694565329 |
| Ttask_withhold_0.5 | attack:A/RAC1P | PWGTP | 1.26485326 | 0.00011362136 | 0.0063988963 |
| Ttask_withhold_0.5 | attack:AB/SEX | unweighted | 0.68390936 | -0.00130757461 | 0.0151402789 |
| Ttask_withhold_0.5 | attack:AB/SEX | PWGTP | 0.683495579 | -0.000865546789 | 0.00972249034 |
| Ttask_withhold_0.5 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| Ttask_withhold_0.5 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| Ttask_withhold_0.5 | attack:A/public_coverage | unweighted | 0.519658533 | 0.00033861082 | 0.00803739053 |
| Ttask_withhold_0.5 | attack:A/public_coverage | PWGTP | 0.521692571 | 0.00015724468 | 0.00665768835 |
| Ttask_withhold_0.5 | attack:A/commute_over20 | unweighted | 0.682819234 | -0.000329710146 | 0.00167910928 |
| Ttask_withhold_0.5 | attack:A/commute_over20 | PWGTP | 0.685945042 | -0.000597830743 | 0.0029741412 |
| Ttask_withhold_0.5 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_withhold_0.5 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_withhold_0.5 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_withhold_0.5 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_withhold_0.5 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_withhold_0.5 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_withhold_0.5 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_withhold_0.5 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_withhold_0.5 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_withhold_0.5 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_withhold_0.5 | utility:A/same_residence | unweighted | 0.506212248 | -0.0189394093 | 0.00676369525 |
| Ttask_withhold_0.5 | utility:A/same_residence | PWGTP | 0.481455031 | -0.0176140643 | 0.00460083041 |
| Ttask_withhold_0.5 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Ttask_withhold_0.5 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Ttask_withhold_0.5 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Ttask_withhold_0.5 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Ttask_withhold_0.5 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_withhold_0.5 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_withhold_0.5 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_withhold_0.5 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_withhold_0.5_a33 | attack:A/SEX | unweighted | 0.687775373 | -0.000118287877 | 0.0156291886 |
| Ttask_withhold_0.5_a33 | attack:A/SEX | PWGTP | 0.687240769 | 0.000756316616 | 0.00899256983 |
| Ttask_withhold_0.5_a33 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| Ttask_withhold_0.5_a33 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| Ttask_withhold_0.5_a33 | attack:AB/SEX | unweighted | 0.680619786 | -0.00459714848 | 0.0118507051 |
| Ttask_withhold_0.5_a33 | attack:AB/SEX | PWGTP | 0.682647801 | -0.00171332486 | 0.00887471227 |
| Ttask_withhold_0.5_a33 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| Ttask_withhold_0.5_a33 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| Ttask_withhold_0.5_a33 | attack:A/public_coverage | unweighted | 0.519319923 | 0 | 0.00769877971 |
| Ttask_withhold_0.5_a33 | attack:A/public_coverage | PWGTP | 0.521535326 | 0 | 0.00650044367 |
| Ttask_withhold_0.5_a33 | attack:A/commute_over20 | unweighted | 0.683099146 | -4.97984337e-05 | 0.00195902099 |
| Ttask_withhold_0.5_a33 | attack:A/commute_over20 | PWGTP | 0.686055811 | -0.000487061912 | 0.00308491003 |
| Ttask_withhold_0.5_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_withhold_0.5_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_withhold_0.5_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_withhold_0.5_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_withhold_0.5_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_withhold_0.5_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_withhold_0.5_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_withhold_0.5_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_withhold_0.5_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_withhold_0.5_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_withhold_0.5_a33 | utility:A/same_residence | unweighted | 0.507240427 | -0.0179112306 | 0.00779187395 |
| Ttask_withhold_0.5_a33 | utility:A/same_residence | PWGTP | 0.48289603 | -0.0161730646 | 0.00604183015 |
| Ttask_withhold_0.5_a33 | utility:A/income_binary | unweighted | 0.295959883 | 0.000261438828 | 0.0114427075 |
| Ttask_withhold_0.5_a33 | utility:A/income_binary | PWGTP | 0.30766774 | 8.81537376e-05 | 0.0112600927 |
| Ttask_withhold_0.5_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Ttask_withhold_0.5_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Ttask_withhold_0.5_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_withhold_0.5_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_withhold_0.5_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_withhold_0.5_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_withhold_0.75 | attack:A/SEX | unweighted | 0.686791641 | -0.00110201925 | 0.0146454573 |
| Ttask_withhold_0.75 | attack:A/SEX | PWGTP | 0.685299615 | -0.00118483719 | 0.00705141602 |
| Ttask_withhold_0.75 | attack:A/RAC1P | unweighted | 1.27133873 | 0.00059936605 | 0.00712491314 |
| Ttask_withhold_0.75 | attack:A/RAC1P | PWGTP | 1.26477873 | 3.90910705e-05 | 0.00632436601 |
| Ttask_withhold_0.75 | attack:AB/SEX | unweighted | 0.680484193 | -0.00473274181 | 0.0117151117 |
| Ttask_withhold_0.75 | attack:AB/SEX | PWGTP | 0.682368334 | -0.00199279196 | 0.00859524517 |
| Ttask_withhold_0.75 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| Ttask_withhold_0.75 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| Ttask_withhold_0.75 | attack:A/public_coverage | unweighted | 0.519363365 | 4.3442273e-05 | 0.00774222199 |
| Ttask_withhold_0.75 | attack:A/public_coverage | PWGTP | 0.521002004 | -0.000533321696 | 0.00596712198 |
| Ttask_withhold_0.75 | attack:A/commute_over20 | unweighted | 0.682791343 | -0.000357601422 | 0.001651218 |
| Ttask_withhold_0.75 | attack:A/commute_over20 | PWGTP | 0.685630219 | -0.000912654032 | 0.00265931791 |
| Ttask_withhold_0.75 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_withhold_0.75 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_withhold_0.75 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_withhold_0.75 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_withhold_0.75 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_withhold_0.75 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_withhold_0.75 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_withhold_0.75 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_withhold_0.75 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_withhold_0.75 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_withhold_0.75 | utility:A/same_residence | unweighted | 0.496543271 | -0.0286083865 | -0.00290528196 |
| Ttask_withhold_0.75 | utility:A/same_residence | PWGTP | 0.473250864 | -0.0258182309 | -0.00360333623 |
| Ttask_withhold_0.75 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Ttask_withhold_0.75 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Ttask_withhold_0.75 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Ttask_withhold_0.75 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Ttask_withhold_0.75 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_withhold_0.75 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_withhold_0.75 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_withhold_0.75 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| Ttask_withhold_0.75_a33 | attack:A/SEX | unweighted | 0.684095021 | -0.00379863994 | 0.0119488366 |
| Ttask_withhold_0.75_a33 | attack:A/SEX | PWGTP | 0.685292442 | -0.00119201061 | 0.0070442426 |
| Ttask_withhold_0.75_a33 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| Ttask_withhold_0.75_a33 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| Ttask_withhold_0.75_a33 | attack:AB/SEX | unweighted | 0.684982702 | -0.000234233128 | 0.0162136204 |
| Ttask_withhold_0.75_a33 | attack:AB/SEX | PWGTP | 0.684354917 | -6.2083776e-06 | 0.0105818288 |
| Ttask_withhold_0.75_a33 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| Ttask_withhold_0.75_a33 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| Ttask_withhold_0.75_a33 | attack:A/public_coverage | unweighted | 0.520850121 | 0.00153019805 | 0.00922897776 |
| Ttask_withhold_0.75_a33 | attack:A/public_coverage | PWGTP | 0.524117802 | 0.00258247644 | 0.00908292011 |
| Ttask_withhold_0.75_a33 | attack:A/commute_over20 | unweighted | 0.683042206 | -0.00010673844 | 0.00190208098 |
| Ttask_withhold_0.75_a33 | attack:A/commute_over20 | PWGTP | 0.685762012 | -0.000780860455 | 0.00279111148 |
| Ttask_withhold_0.75_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| Ttask_withhold_0.75_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| Ttask_withhold_0.75_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| Ttask_withhold_0.75_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| Ttask_withhold_0.75_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| Ttask_withhold_0.75_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| Ttask_withhold_0.75_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| Ttask_withhold_0.75_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| Ttask_withhold_0.75_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| Ttask_withhold_0.75_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| Ttask_withhold_0.75_a33 | utility:A/same_residence | unweighted | 0.495941459 | -0.0292101992 | -0.00350709469 |
| Ttask_withhold_0.75_a33 | utility:A/same_residence | PWGTP | 0.472710935 | -0.0263581603 | -0.0041432656 |
| Ttask_withhold_0.75_a33 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| Ttask_withhold_0.75_a33 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| Ttask_withhold_0.75_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| Ttask_withhold_0.75_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| Ttask_withhold_0.75_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| Ttask_withhold_0.75_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| Ttask_withhold_0.75_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| Ttask_withhold_0.75_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| constant_best | attack:A/SEX | unweighted | 0.687893661 | 0 | 0.0157474765 |
| constant_best | attack:A/SEX | PWGTP | 0.686484452 | 0 | 0.00823625321 |
| constant_best | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| constant_best | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| constant_best | attack:AB/SEX | unweighted | 0.685216935 | 0 | 0.0164478536 |
| constant_best | attack:AB/SEX | PWGTP | 0.684361126 | 0 | 0.0105880371 |
| constant_best | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| constant_best | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| constant_best | attack:A/public_coverage | unweighted | 0.519319923 | 0 | 0.00769877971 |
| constant_best | attack:A/public_coverage | PWGTP | 0.521535326 | 0 | 0.00650044367 |
| constant_best | attack:A/commute_over20 | unweighted | 0.683148944 | 0 | 0.00200881942 |
| constant_best | attack:A/commute_over20 | PWGTP | 0.686542873 | 0 | 0.00357197194 |
| constant_best | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| constant_best | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| constant_best | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| constant_best | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| constant_best | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| constant_best | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| constant_best | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| constant_best | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| constant_best | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| constant_best | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| constant_best | utility:A/same_residence | unweighted | 0.525151658 | 0 | 0.0257031045 |
| constant_best | utility:A/same_residence | PWGTP | 0.499069095 | 0 | 0.0222148947 |
| constant_best | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| constant_best | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| constant_best | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| constant_best | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| constant_best | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| constant_best | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| constant_best | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| constant_best | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| constant_best_a33 | attack:A/SEX | unweighted | 0.687893661 | 0 | 0.0157474765 |
| constant_best_a33 | attack:A/SEX | PWGTP | 0.686484452 | 0 | 0.00823625321 |
| constant_best_a33 | attack:A/RAC1P | unweighted | 1.27073936 | 0 | 0.00652554709 |
| constant_best_a33 | attack:A/RAC1P | PWGTP | 1.26473964 | 0 | 0.00628527494 |
| constant_best_a33 | attack:AB/SEX | unweighted | 0.685216935 | 0 | 0.0164478536 |
| constant_best_a33 | attack:AB/SEX | PWGTP | 0.684361126 | 0 | 0.0105880371 |
| constant_best_a33 | attack:AB/RAC1P | unweighted | 1.25760501 | 0 | 0.00550439428 |
| constant_best_a33 | attack:AB/RAC1P | PWGTP | 1.25235223 | 0 | 0.00642400314 |
| constant_best_a33 | attack:A/public_coverage | unweighted | 0.519319923 | 0 | 0.00769877971 |
| constant_best_a33 | attack:A/public_coverage | PWGTP | 0.521535326 | 0 | 0.00650044367 |
| constant_best_a33 | attack:A/commute_over20 | unweighted | 0.683148944 | 0 | 0.00200881942 |
| constant_best_a33 | attack:A/commute_over20 | PWGTP | 0.686542873 | 0 | 0.00357197194 |
| constant_best_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| constant_best_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| constant_best_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| constant_best_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| constant_best_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| constant_best_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| constant_best_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| constant_best_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| constant_best_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| constant_best_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| constant_best_a33 | utility:A/same_residence | unweighted | 0.525151658 | 0 | 0.0257031045 |
| constant_best_a33 | utility:A/same_residence | PWGTP | 0.499069095 | 0 | 0.0222148947 |
| constant_best_a33 | utility:A/income_binary | unweighted | 0.295698444 | 0 | 0.0111812687 |
| constant_best_a33 | utility:A/income_binary | PWGTP | 0.307579587 | 0 | 0.0111719389 |
| constant_best_a33 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| constant_best_a33 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| constant_best_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| constant_best_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| constant_best_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| constant_best_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| continuous_task | attack:A/SEX | unweighted | 0.679188855 | -0.00870480554 | 0.00704267096 |
| continuous_task | attack:A/SEX | PWGTP | 0.67701857 | -0.00946588195 | -0.00122962874 |
| continuous_task | attack:A/RAC1P | unweighted | 1.26623755 | -0.0045018093 | 0.00202373779 |
| continuous_task | attack:A/RAC1P | PWGTP | 1.25861639 | -0.00612324602 | 0.000162028918 |
| continuous_task | attack:AB/SEX | unweighted | 0.674369535 | -0.0108474001 | 0.00560045341 |
| continuous_task | attack:AB/SEX | PWGTP | 0.674740916 | -0.00962020972 | 0.000967827411 |
| continuous_task | attack:AB/RAC1P | unweighted | 1.25201951 | -0.0055855006 | -8.11063288e-05 |
| continuous_task | attack:AB/RAC1P | PWGTP | 1.24842312 | -0.00392911137 | 0.00249489177 |
| continuous_task | attack:A/public_coverage | unweighted | 0.513588352 | -0.00573157066 | 0.00196720905 |
| continuous_task | attack:A/public_coverage | PWGTP | 0.515568305 | -0.00596702128 | 0.000533422388 |
| continuous_task | attack:A/commute_over20 | unweighted | 0.681282146 | -0.00186679842 | 0.000142021004 |
| continuous_task | attack:A/commute_over20 | PWGTP | 0.683687812 | -0.00285506094 | 0.000716911002 |
| continuous_task | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| continuous_task | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| continuous_task | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| continuous_task | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| continuous_task | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| continuous_task | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| continuous_task | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| continuous_task | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| continuous_task | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| continuous_task | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| continuous_task | utility:A/same_residence | unweighted | 0.479827367 | -0.0453242908 | -0.0196211863 |
| continuous_task | utility:A/same_residence | PWGTP | 0.456206495 | -0.0428626004 | -0.0206477057 |
| continuous_task | utility:A/income_binary | unweighted | 0.294084155 | -0.00161428931 | 0.00956697935 |
| continuous_task | utility:A/income_binary | PWGTP | 0.307548908 | -3.06780332e-05 | 0.0111412609 |
| continuous_task | utility:A/civilian_at_work | unweighted | 0.278385231 | -0.00092196193 | 0.00694799127 |
| continuous_task | utility:A/civilian_at_work | PWGTP | 0.267012921 | -0.000802126522 | 0.00454228974 |
| continuous_task | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| continuous_task | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| continuous_task | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| continuous_task | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| independent_token | attack:A/SEX | unweighted | 0.687128075 | -0.00076558537 | 0.0149818911 |
| independent_token | attack:A/SEX | PWGTP | 0.685822445 | -0.000662007352 | 0.00757424586 |
| independent_token | attack:A/RAC1P | unweighted | 1.27014088 | -0.000598482508 | 0.00592706458 |
| independent_token | attack:A/RAC1P | PWGTP | 1.26510011 | 0.000360469116 | 0.00664574406 |
| independent_token | attack:AB/SEX | unweighted | 0.684890887 | -0.000326047499 | 0.0161218061 |
| independent_token | attack:AB/SEX | PWGTP | 0.683889789 | -0.000471337066 | 0.0101167001 |
| independent_token | attack:AB/RAC1P | unweighted | 1.25782053 | 0.000215519743 | 0.00571991402 |
| independent_token | attack:AB/RAC1P | PWGTP | 1.25246811 | 0.000115877014 | 0.00653988015 |
| independent_token | attack:A/public_coverage | unweighted | 0.518765491 | -0.000554431851 | 0.00714434786 |
| independent_token | attack:A/public_coverage | PWGTP | 0.521085371 | -0.000449955364 | 0.00605048831 |
| independent_token | attack:A/commute_over20 | unweighted | 0.68326802 | 0.000119075582 | 0.00212789501 |
| independent_token | attack:A/commute_over20 | PWGTP | 0.686625164 | 8.22909227e-05 | 0.00365426286 |
| independent_token | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| independent_token | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| independent_token | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| independent_token | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| independent_token | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| independent_token | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| independent_token | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| independent_token | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| independent_token | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| independent_token | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| independent_token | utility:A/same_residence | unweighted | 0.524789087 | -0.000362570516 | 0.025340534 |
| independent_token | utility:A/same_residence | PWGTP | 0.498668347 | -0.0004007483 | 0.0218141464 |
| independent_token | utility:A/income_binary | unweighted | 0.295829594 | 0.000131149328 | 0.011312418 |
| independent_token | utility:A/income_binary | PWGTP | 0.307864411 | 0.000284824042 | 0.011456763 |
| independent_token | utility:A/civilian_at_work | unweighted | 0.279267559 | -3.9633664e-05 | 0.00783031953 |
| independent_token | utility:A/civilian_at_work | PWGTP | 0.267679276 | -0.000135770726 | 0.00520864553 |
| independent_token | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| independent_token | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| independent_token | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| independent_token | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| independent_token_a33 | attack:A/SEX | unweighted | 0.684591183 | -0.00330247762 | 0.0124449989 |
| independent_token_a33 | attack:A/SEX | PWGTP | 0.68760507 | 0.00112061799 | 0.0093568712 |
| independent_token_a33 | attack:A/RAC1P | unweighted | 1.27118382 | 0.000444456099 | 0.00697000319 |
| independent_token_a33 | attack:A/RAC1P | PWGTP | 1.26537866 | 0.000639018738 | 0.00692429368 |
| independent_token_a33 | attack:AB/SEX | unweighted | 0.681251736 | -0.00396519884 | 0.0124826547 |
| independent_token_a33 | attack:AB/SEX | PWGTP | 0.683255673 | -0.00110545308 | 0.00948258405 |
| independent_token_a33 | attack:AB/RAC1P | unweighted | 1.2577425 | 0.000137494954 | 0.00564188923 |
| independent_token_a33 | attack:AB/RAC1P | PWGTP | 1.2519619 | -0.00039032658 | 0.00603367655 |
| independent_token_a33 | attack:A/public_coverage | unweighted | 0.519088484 | -0.0002314383 | 0.00746734141 |
| independent_token_a33 | attack:A/public_coverage | PWGTP | 0.521273501 | -0.000261824571 | 0.0062386191 |
| independent_token_a33 | attack:A/commute_over20 | unweighted | 0.683362848 | 0.000213904316 | 0.00222272374 |
| independent_token_a33 | attack:A/commute_over20 | PWGTP | 0.686683126 | 0.000140253802 | 0.00371222574 |
| independent_token_a33 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| independent_token_a33 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| independent_token_a33 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| independent_token_a33 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| independent_token_a33 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| independent_token_a33 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| independent_token_a33 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| independent_token_a33 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| independent_token_a33 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| independent_token_a33 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| independent_token_a33 | utility:A/same_residence | unweighted | 0.52449474 | -0.000656917765 | 0.0250461868 |
| independent_token_a33 | utility:A/same_residence | PWGTP | 0.498439344 | -0.000629750633 | 0.0215851441 |
| independent_token_a33 | utility:A/income_binary | unweighted | 0.295824004 | 0.000125560015 | 0.0113068287 |
| independent_token_a33 | utility:A/income_binary | PWGTP | 0.308113462 | 0.000533875678 | 0.0117058146 |
| independent_token_a33 | utility:A/civilian_at_work | unweighted | 0.279404664 | 9.74711466e-05 | 0.00796742434 |
| independent_token_a33 | utility:A/civilian_at_work | PWGTP | 0.26776206 | -5.29874854e-05 | 0.00529142877 |
| independent_token_a33 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| independent_token_a33 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| independent_token_a33 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| independent_token_a33 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| leace_A0 | attack:A/SEX | unweighted | 0.681096227 | -0.00679743353 | 0.00895004297 |
| leace_A0 | attack:A/SEX | PWGTP | 0.679644664 | -0.0068397881 | 0.00139646511 |
| leace_A0 | attack:A/RAC1P | unweighted | 1.25165224 | -0.0190871163 | -0.0125615692 |
| leace_A0 | attack:A/RAC1P | PWGTP | 1.24583491 | -0.0189047318 | -0.0126194569 |
| leace_A0 | attack:AB/SEX | unweighted | 0.677769451 | -0.00744748424 | 0.00900036931 |
| leace_A0 | attack:AB/SEX | PWGTP | 0.677284274 | -0.00707685213 | 0.003511185 |
| leace_A0 | attack:AB/RAC1P | unweighted | 1.24215148 | -0.015453527 | -0.00994913277 |
| leace_A0 | attack:AB/RAC1P | PWGTP | 1.23689194 | -0.0154602951 | -0.00903629195 |
| leace_A0 | attack:A/public_coverage | unweighted | 0.507233807 | -0.0120861156 | -0.00438733589 |
| leace_A0 | attack:A/public_coverage | PWGTP | 0.513404315 | -0.00813101067 | -0.00163056699 |
| leace_A0 | attack:A/commute_over20 | unweighted | 0.681703391 | -0.00144555316 | 0.000563266266 |
| leace_A0 | attack:A/commute_over20 | PWGTP | 0.684137082 | -0.00240579023 | 0.00116618171 |
| leace_A0 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| leace_A0 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| leace_A0 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| leace_A0 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| leace_A0 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| leace_A0 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| leace_A0 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| leace_A0 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| leace_A0 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| leace_A0 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| leace_A0 | utility:A/same_residence | unweighted | 0.498948124 | -0.0262035339 | -0.00050042931 |
| leace_A0 | utility:A/same_residence | PWGTP | 0.474571037 | -0.0244980578 | -0.00228316305 |
| leace_A0 | utility:A/income_binary | unweighted | 0.294126852 | -0.0015715928 | 0.00960967586 |
| leace_A0 | utility:A/income_binary | PWGTP | 0.305590679 | -0.00198890765 | 0.0091830313 |
| leace_A0 | utility:A/civilian_at_work | unweighted | 0.27789265 | -0.00141454275 | 0.00645541045 |
| leace_A0 | utility:A/civilian_at_work | PWGTP | 0.267834269 | 1.92222695e-05 | 0.00536363853 |
| leace_A0 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| leace_A0 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| leace_A0 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| leace_A0 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| leace_supervised | attack:A/SEX | unweighted | 0.654005016 | -0.0338886445 | -0.018141168 |
| leace_supervised | attack:A/SEX | PWGTP | 0.658560882 | -0.02792357 | -0.0196873168 |
| leace_supervised | attack:A/RAC1P | unweighted | 1.21357942 | -0.0571599413 | -0.0506343942 |
| leace_supervised | attack:A/RAC1P | PWGTP | 1.21460235 | -0.0501372945 | -0.0438520196 |
| leace_supervised | attack:AB/SEX | unweighted | 0.653630822 | -0.0315861128 | -0.0151382592 |
| leace_supervised | attack:AB/SEX | PWGTP | 0.658042649 | -0.026318477 | -0.0157304399 |
| leace_supervised | attack:AB/RAC1P | unweighted | 1.20932727 | -0.0482777366 | -0.0427733423 |
| leace_supervised | attack:AB/RAC1P | PWGTP | 1.21027665 | -0.0420755788 | -0.0356515756 |
| leace_supervised | attack:A/public_coverage | unweighted | 0.494879273 | -0.0244406492 | -0.0167418695 |
| leace_supervised | attack:A/public_coverage | PWGTP | 0.501246369 | -0.0202889573 | -0.0137885137 |
| leace_supervised | attack:A/commute_over20 | unweighted | 0.682008484 | -0.00114046059 | 0.000868358834 |
| leace_supervised | attack:A/commute_over20 | PWGTP | 0.684542732 | -0.0020001411 | 0.00157183084 |
| leace_supervised | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| leace_supervised | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| leace_supervised | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| leace_supervised | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| leace_supervised | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| leace_supervised | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| leace_supervised | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| leace_supervised | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| leace_supervised | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| leace_supervised | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| leace_supervised | utility:A/same_residence | unweighted | 0.485081702 | -0.0400699559 | -0.0143668514 |
| leace_supervised | utility:A/same_residence | PWGTP | 0.462458118 | -0.0366109766 | -0.0143960819 |
| leace_supervised | utility:A/income_binary | unweighted | 0.285891971 | -0.00980647346 | 0.0013747952 |
| leace_supervised | utility:A/income_binary | PWGTP | 0.297891608 | -0.00968797838 | 0.00148396057 |
| leace_supervised | utility:A/civilian_at_work | unweighted | 0.279344322 | 3.71288967e-05 | 0.00790708209 |
| leace_supervised | utility:A/civilian_at_work | PWGTP | 0.267921082 | 0.000106034964 | 0.00545045122 |
| leace_supervised | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| leace_supervised | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| leace_supervised | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| leace_supervised | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| leace_supervised_mechanism40 | attack:A/SEX | unweighted | 0.659602993 | -0.0282906677 | -0.0125431912 |
| leace_supervised_mechanism40 | attack:A/SEX | PWGTP | 0.668087984 | -0.0183964687 | -0.0101602155 |
| leace_supervised_mechanism40 | attack:A/RAC1P | unweighted | 1.21388738 | -0.0568519768 | -0.0503264297 |
| leace_supervised_mechanism40 | attack:A/RAC1P | PWGTP | 1.21504883 | -0.0496908079 | -0.0434055329 |
| leace_supervised_mechanism40 | attack:AB/SEX | unweighted | 0.656903773 | -0.0283131622 | -0.0118653087 |
| leace_supervised_mechanism40 | attack:AB/SEX | PWGTP | 0.6621648 | -0.0221963263 | -0.0116082891 |
| leace_supervised_mechanism40 | attack:AB/RAC1P | unweighted | 1.20957714 | -0.0480278645 | -0.0425234703 |
| leace_supervised_mechanism40 | attack:AB/RAC1P | PWGTP | 1.2097447 | -0.0426075262 | -0.0361835231 |
| leace_supervised_mechanism40 | attack:A/public_coverage | unweighted | 0.496249531 | -0.0230703911 | -0.0153716114 |
| leace_supervised_mechanism40 | attack:A/public_coverage | PWGTP | 0.503771615 | -0.0177637107 | -0.011263267 |
| leace_supervised_mechanism40 | attack:A/commute_over20 | unweighted | 0.6813694 | -0.00177954456 | 0.00022927486 |
| leace_supervised_mechanism40 | attack:A/commute_over20 | PWGTP | 0.684291659 | -0.00225121413 | 0.00132075781 |
| leace_supervised_mechanism40 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| leace_supervised_mechanism40 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| leace_supervised_mechanism40 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| leace_supervised_mechanism40 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| leace_supervised_mechanism40 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| leace_supervised_mechanism40 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| leace_supervised_mechanism40 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| leace_supervised_mechanism40 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| leace_supervised_mechanism40 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| leace_supervised_mechanism40 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| leace_supervised_mechanism40 | utility:A/same_residence | unweighted | 0.486889924 | -0.0382617336 | -0.0125586291 |
| leace_supervised_mechanism40 | utility:A/same_residence | PWGTP | 0.463497609 | -0.035571486 | -0.0133565913 |
| leace_supervised_mechanism40 | utility:A/income_binary | unweighted | 0.286011213 | -0.00968723153 | 0.00149403713 |
| leace_supervised_mechanism40 | utility:A/income_binary | PWGTP | 0.299132205 | -0.00844738116 | 0.00272455779 |
| leace_supervised_mechanism40 | utility:A/civilian_at_work | unweighted | 0.278800892 | -0.000506300391 | 0.00736365281 |
| leace_supervised_mechanism40 | utility:A/civilian_at_work | PWGTP | 0.267447793 | -0.000367253952 | 0.00497716231 |
| leace_supervised_mechanism40 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| leace_supervised_mechanism40 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| leace_supervised_mechanism40 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| leace_supervised_mechanism40 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| leace_supervised_union88 | attack:A/SEX | unweighted | 0.658570688 | -0.029322973 | -0.0135754965 |
| leace_supervised_union88 | attack:A/SEX | PWGTP | 0.665606389 | -0.0208780632 | -0.01264181 |
| leace_supervised_union88 | attack:A/RAC1P | unweighted | 1.21657318 | -0.0541661758 | -0.0476406287 |
| leace_supervised_union88 | attack:A/RAC1P | PWGTP | 1.21388568 | -0.0508539624 | -0.0445686875 |
| leace_supervised_union88 | attack:AB/SEX | unweighted | 0.657814572 | -0.0274023625 | -0.010954509 |
| leace_supervised_union88 | attack:AB/SEX | PWGTP | 0.662268421 | -0.022092705 | -0.0115046679 |
| leace_supervised_union88 | attack:AB/RAC1P | unweighted | 1.21220817 | -0.0453968349 | -0.0398924406 |
| leace_supervised_union88 | attack:AB/RAC1P | PWGTP | 1.20938266 | -0.0429695738 | -0.0365455707 |
| leace_supervised_union88 | attack:A/public_coverage | unweighted | 0.496285087 | -0.0230348352 | -0.0153360555 |
| leace_supervised_union88 | attack:A/public_coverage | PWGTP | 0.502573373 | -0.0189619529 | -0.0124615092 |
| leace_supervised_union88 | attack:A/commute_over20 | unweighted | 0.681325865 | -0.00182307941 | 0.000185740014 |
| leace_supervised_union88 | attack:A/commute_over20 | PWGTP | 0.684076152 | -0.00246672079 | 0.00110525115 |
| leace_supervised_union88 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| leace_supervised_union88 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| leace_supervised_union88 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| leace_supervised_union88 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| leace_supervised_union88 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| leace_supervised_union88 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| leace_supervised_union88 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| leace_supervised_union88 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| leace_supervised_union88 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| leace_supervised_union88 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| leace_supervised_union88 | utility:A/same_residence | unweighted | 0.485777924 | -0.0393737333 | -0.0136706288 |
| leace_supervised_union88 | utility:A/same_residence | PWGTP | 0.461660868 | -0.0374082265 | -0.0151933318 |
| leace_supervised_union88 | utility:A/income_binary | unweighted | 0.287011697 | -0.00868674704 | 0.00249452162 |
| leace_supervised_union88 | utility:A/income_binary | PWGTP | 0.300321042 | -0.00725854415 | 0.0039133948 |
| leace_supervised_union88 | utility:A/civilian_at_work | unweighted | 0.279040313 | -0.000266879779 | 0.00760307342 |
| leace_supervised_union88 | utility:A/civilian_at_work | PWGTP | 0.267773608 | -4.14393779e-05 | 0.00530297688 |
| leace_supervised_union88 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| leace_supervised_union88 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| leace_supervised_union88 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| leace_supervised_union88 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| optnet16_C1 | attack:A/SEX | unweighted | 0.674526644 | -0.0133670166 | 0.00238045991 |
| optnet16_C1 | attack:A/SEX | PWGTP | 0.675486602 | -0.0109978505 | -0.00276159733 |
| optnet16_C1 | attack:A/RAC1P | unweighted | 1.24212763 | -0.028611726 | -0.022086179 |
| optnet16_C1 | attack:A/RAC1P | PWGTP | 1.23908448 | -0.0256551641 | -0.0193698891 |
| optnet16_C1 | attack:AB/SEX | unweighted | 0.673593683 | -0.0116232521 | 0.00482460142 |
| optnet16_C1 | attack:AB/SEX | PWGTP | 0.674856899 | -0.00950422713 | 0.00108381 |
| optnet16_C1 | attack:AB/RAC1P | unweighted | 1.22968812 | -0.0279168896 | -0.0224124954 |
| optnet16_C1 | attack:AB/RAC1P | PWGTP | 1.22760144 | -0.024750793 | -0.0183267899 |
| optnet16_C1 | attack:A/public_coverage | unweighted | 0.512542983 | -0.00677693995 | 0.000921839762 |
| optnet16_C1 | attack:A/public_coverage | PWGTP | 0.519164514 | -0.00237081208 | 0.00412963159 |
| optnet16_C1 | attack:A/commute_over20 | unweighted | 0.681655686 | -0.00149325818 | 0.000515561246 |
| optnet16_C1 | attack:A/commute_over20 | PWGTP | 0.685269671 | -0.0012732021 | 0.00229876984 |
| optnet16_C1 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| optnet16_C1 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| optnet16_C1 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| optnet16_C1 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| optnet16_C1 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| optnet16_C1 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| optnet16_C1 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| optnet16_C1 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| optnet16_C1 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| optnet16_C1 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| optnet16_C1 | utility:A/same_residence | unweighted | 0.500216783 | -0.0249348748 | 0.000768229776 |
| optnet16_C1 | utility:A/same_residence | PWGTP | 0.478715222 | -0.0203538731 | 0.00186102157 |
| optnet16_C1 | utility:A/income_binary | unweighted | 0.292423043 | -0.00327540191 | 0.00790586675 |
| optnet16_C1 | utility:A/income_binary | PWGTP | 0.304090811 | -0.00348877575 | 0.0076831632 |
| optnet16_C1 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| optnet16_C1 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| optnet16_C1 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| optnet16_C1 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| optnet16_C1 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| optnet16_C1 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| optnet16_L1 | attack:A/SEX | unweighted | 0.665197728 | -0.0226959329 | -0.00694845645 |
| optnet16_L1 | attack:A/SEX | PWGTP | 0.668802876 | -0.0176815768 | -0.00944532361 |
| optnet16_L1 | attack:A/RAC1P | unweighted | 1.23629639 | -0.0344429694 | -0.0279174223 |
| optnet16_L1 | attack:A/RAC1P | PWGTP | 1.2307244 | -0.0340152377 | -0.0277299627 |
| optnet16_L1 | attack:AB/SEX | unweighted | 0.66686252 | -0.0183544147 | -0.00190656113 |
| optnet16_L1 | attack:AB/SEX | PWGTP | 0.668488755 | -0.0158723711 | -0.00528433396 |
| optnet16_L1 | attack:AB/RAC1P | unweighted | 1.22129528 | -0.0363097221 | -0.0308053278 |
| optnet16_L1 | attack:AB/RAC1P | PWGTP | 1.21855965 | -0.0337925753 | -0.0273685721 |
| optnet16_L1 | attack:A/public_coverage | unweighted | 0.512079327 | -0.00724059601 | 0.000458183708 |
| optnet16_L1 | attack:A/public_coverage | PWGTP | 0.51798909 | -0.00354623581 | 0.00295420786 |
| optnet16_L1 | attack:A/commute_over20 | unweighted | 0.682598617 | -0.000550327128 | 0.00145849229 |
| optnet16_L1 | attack:A/commute_over20 | PWGTP | 0.686573364 | 3.04914216e-05 | 0.00360246336 |
| optnet16_L1 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| optnet16_L1 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| optnet16_L1 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| optnet16_L1 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| optnet16_L1 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| optnet16_L1 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| optnet16_L1 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| optnet16_L1 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| optnet16_L1 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| optnet16_L1 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| optnet16_L1 | utility:A/same_residence | unweighted | 0.49668118 | -0.0284704777 | -0.00276737312 |
| optnet16_L1 | utility:A/same_residence | PWGTP | 0.473858894 | -0.0252102007 | -0.00299530595 |
| optnet16_L1 | utility:A/income_binary | unweighted | 0.295533621 | -0.000164823113 | 0.0110164455 |
| optnet16_L1 | utility:A/income_binary | PWGTP | 0.307226754 | -0.00035283207 | 0.0108191069 |
| optnet16_L1 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| optnet16_L1 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| optnet16_L1 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| optnet16_L1 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| optnet16_L1 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| optnet16_L1 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| optnet16_L2 | attack:A/SEX | unweighted | 0.673341792 | -0.0145518688 | 0.00119560769 |
| optnet16_L2 | attack:A/SEX | PWGTP | 0.677004148 | -0.00948030454 | -0.00124405133 |
| optnet16_L2 | attack:A/RAC1P | unweighted | 1.24381561 | -0.0269237471 | -0.0203982 |
| optnet16_L2 | attack:A/RAC1P | PWGTP | 1.24202652 | -0.0227131222 | -0.0164278473 |
| optnet16_L2 | attack:AB/SEX | unweighted | 0.672140163 | -0.0130767719 | 0.00337108165 |
| optnet16_L2 | attack:AB/SEX | PWGTP | 0.675025694 | -0.00933543163 | 0.0012526055 |
| optnet16_L2 | attack:AB/RAC1P | unweighted | 1.234004 | -0.0236010037 | -0.0180966095 |
| optnet16_L2 | attack:AB/RAC1P | PWGTP | 1.23288876 | -0.0194634668 | -0.0130394637 |
| optnet16_L2 | attack:A/public_coverage | unweighted | 0.511587504 | -0.0077324182 | -3.36384867e-05 |
| optnet16_L2 | attack:A/public_coverage | PWGTP | 0.518877446 | -0.00265788038 | 0.00384256329 |
| optnet16_L2 | attack:A/commute_over20 | unweighted | 0.68274222 | -0.000406724342 | 0.00160209508 |
| optnet16_L2 | attack:A/commute_over20 | PWGTP | 0.68529097 | -0.00125190223 | 0.00232006971 |
| optnet16_L2 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| optnet16_L2 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| optnet16_L2 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| optnet16_L2 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| optnet16_L2 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| optnet16_L2 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| optnet16_L2 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| optnet16_L2 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| optnet16_L2 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| optnet16_L2 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| optnet16_L2 | utility:A/same_residence | unweighted | 0.503301014 | -0.0218506436 | 0.00385246093 |
| optnet16_L2 | utility:A/same_residence | PWGTP | 0.480055222 | -0.0190138725 | 0.00320102221 |
| optnet16_L2 | utility:A/income_binary | unweighted | 0.294452266 | -0.00124617857 | 0.00993509009 |
| optnet16_L2 | utility:A/income_binary | PWGTP | 0.306308274 | -0.00127131226 | 0.00990062669 |
| optnet16_L2 | utility:A/civilian_at_work | unweighted | 0.279307193 | 0 | 0.0078699532 |
| optnet16_L2 | utility:A/civilian_at_work | PWGTP | 0.267815047 | 0 | 0.00534441626 |
| optnet16_L2 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| optnet16_L2 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| optnet16_L2 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| optnet16_L2 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| splince_A0 | attack:A/SEX | unweighted | 0.683915318 | -0.00397834248 | 0.011769134 |
| splince_A0 | attack:A/SEX | PWGTP | 0.683453625 | -0.00303082776 | 0.00520542546 |
| splince_A0 | attack:A/RAC1P | unweighted | 1.26531313 | -0.00542623207 | 0.00109931502 |
| splince_A0 | attack:A/RAC1P | PWGTP | 1.26164675 | -0.00309288957 | 0.00319238537 |
| splince_A0 | attack:AB/SEX | unweighted | 0.681505719 | -0.00371121595 | 0.0127366376 |
| splince_A0 | attack:AB/SEX | PWGTP | 0.681017635 | -0.00334349048 | 0.00724454665 |
| splince_A0 | attack:AB/RAC1P | unweighted | 1.2458725 | -0.0117325064 | -0.00622811214 |
| splince_A0 | attack:AB/RAC1P | PWGTP | 1.24192888 | -0.0104233547 | -0.00399935152 |
| splince_A0 | attack:A/public_coverage | unweighted | 0.507917839 | -0.0114020837 | -0.00370330402 |
| splince_A0 | attack:A/public_coverage | PWGTP | 0.513625315 | -0.00791001146 | -0.00140956779 |
| splince_A0 | attack:A/commute_over20 | unweighted | 0.681916585 | -0.0012323594 | 0.000776460024 |
| splince_A0 | attack:A/commute_over20 | PWGTP | 0.684040943 | -0.00250192943 | 0.00107004251 |
| splince_A0 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| splince_A0 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| splince_A0 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| splince_A0 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| splince_A0 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| splince_A0 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| splince_A0 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| splince_A0 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| splince_A0 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| splince_A0 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| splince_A0 | utility:A/same_residence | unweighted | 0.504102009 | -0.0210496489 | 0.00465345568 |
| splince_A0 | utility:A/same_residence | PWGTP | 0.478383693 | -0.0206854018 | 0.0015294929 |
| splince_A0 | utility:A/income_binary | unweighted | 0.295214022 | -0.000484422815 | 0.0106968458 |
| splince_A0 | utility:A/income_binary | PWGTP | 0.307088502 | -0.000491084958 | 0.010680854 |
| splince_A0 | utility:A/civilian_at_work | unweighted | 0.277221388 | -0.00208580437 | 0.00578414882 |
| splince_A0 | utility:A/civilian_at_work | PWGTP | 0.26678274 | -0.00103230686 | 0.0043121094 |
| splince_A0 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| splince_A0 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| splince_A0 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| splince_A0 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| splince_supervised | attack:A/SEX | unweighted | 0.653736968 | -0.0341566928 | -0.0184092163 |
| splince_supervised | attack:A/SEX | PWGTP | 0.660867419 | -0.0256170338 | -0.0173807806 |
| splince_supervised | attack:A/RAC1P | unweighted | 1.21387132 | -0.0568680394 | -0.0503424923 |
| splince_supervised | attack:A/RAC1P | PWGTP | 1.21202753 | -0.0527121083 | -0.0464268333 |
| splince_supervised | attack:AB/SEX | unweighted | 0.65260427 | -0.0326126649 | -0.0161648113 |
| splince_supervised | attack:AB/SEX | PWGTP | 0.658560046 | -0.02580108 | -0.0152130429 |
| splince_supervised | attack:AB/RAC1P | unweighted | 1.20757643 | -0.0500285793 | -0.044524185 |
| splince_supervised | attack:AB/RAC1P | PWGTP | 1.20672603 | -0.0456262007 | -0.0392021976 |
| splince_supervised | attack:A/public_coverage | unweighted | 0.497900636 | -0.021419287 | -0.0137205073 |
| splince_supervised | attack:A/public_coverage | PWGTP | 0.506621997 | -0.0149133285 | -0.00841288488 |
| splince_supervised | attack:A/commute_over20 | unweighted | 0.682055392 | -0.0010935517 | 0.000915267724 |
| splince_supervised | attack:A/commute_over20 | PWGTP | 0.684514016 | -0.00202885659 | 0.00154311535 |
| splince_supervised | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| splince_supervised | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| splince_supervised | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| splince_supervised | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| splince_supervised | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| splince_supervised | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| splince_supervised | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| splince_supervised | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| splince_supervised | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| splince_supervised | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| splince_supervised | utility:A/same_residence | unweighted | 0.48614699 | -0.0390046682 | -0.0133015636 |
| splince_supervised | utility:A/same_residence | PWGTP | 0.464244014 | -0.034825081 | -0.0126101863 |
| splince_supervised | utility:A/income_binary | unweighted | 0.285696425 | -0.0100020193 | 0.00117924931 |
| splince_supervised | utility:A/income_binary | PWGTP | 0.297906154 | -0.00967343271 | 0.00149850624 |
| splince_supervised | utility:A/civilian_at_work | unweighted | 0.279278307 | -2.88855714e-05 | 0.00784106763 |
| splince_supervised | utility:A/civilian_at_work | PWGTP | 0.26741991 | -0.000395136875 | 0.00494927938 |
| splince_supervised | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| splince_supervised | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| splince_supervised | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| splince_supervised | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| splince_supervised_mechanism40 | attack:A/SEX | unweighted | 0.660610863 | -0.0272827974 | -0.0115353209 |
| splince_supervised_mechanism40 | attack:A/SEX | PWGTP | 0.667837948 | -0.0186465049 | -0.0104102516 |
| splince_supervised_mechanism40 | attack:A/RAC1P | unweighted | 1.21379418 | -0.0569451753 | -0.0504196282 |
| splince_supervised_mechanism40 | attack:A/RAC1P | PWGTP | 1.21571559 | -0.0490240481 | -0.0427387731 |
| splince_supervised_mechanism40 | attack:AB/SEX | unweighted | 0.65831068 | -0.0269062548 | -0.0104584013 |
| splince_supervised_mechanism40 | attack:AB/SEX | PWGTP | 0.664423314 | -0.019937812 | -0.00934977483 |
| splince_supervised_mechanism40 | attack:AB/RAC1P | unweighted | 1.20992686 | -0.0476781453 | -0.042173751 |
| splince_supervised_mechanism40 | attack:AB/RAC1P | PWGTP | 1.21035467 | -0.041997564 | -0.0355735609 |
| splince_supervised_mechanism40 | attack:A/public_coverage | unweighted | 0.495157743 | -0.02416218 | -0.0164634003 |
| splince_supervised_mechanism40 | attack:A/public_coverage | PWGTP | 0.501996791 | -0.019538535 | -0.0130380913 |
| splince_supervised_mechanism40 | attack:A/commute_over20 | unweighted | 0.681565864 | -0.00158308026 | 0.000425739164 |
| splince_supervised_mechanism40 | attack:A/commute_over20 | PWGTP | 0.683935842 | -0.0026070309 | 0.000964941035 |
| splince_supervised_mechanism40 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| splince_supervised_mechanism40 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| splince_supervised_mechanism40 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| splince_supervised_mechanism40 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| splince_supervised_mechanism40 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| splince_supervised_mechanism40 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| splince_supervised_mechanism40 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| splince_supervised_mechanism40 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| splince_supervised_mechanism40 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| splince_supervised_mechanism40 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| splince_supervised_mechanism40 | utility:A/same_residence | unweighted | 0.487687676 | -0.0374639822 | -0.0117608777 |
| splince_supervised_mechanism40 | utility:A/same_residence | PWGTP | 0.463979507 | -0.0350895881 | -0.0128746934 |
| splince_supervised_mechanism40 | utility:A/income_binary | unweighted | 0.287487367 | -0.00821107759 | 0.00297019107 |
| splince_supervised_mechanism40 | utility:A/income_binary | PWGTP | 0.301313279 | -0.00626630764 | 0.00490563131 |
| splince_supervised_mechanism40 | utility:A/civilian_at_work | unweighted | 0.279364202 | 5.70096983e-05 | 0.0079269629 |
| splince_supervised_mechanism40 | utility:A/civilian_at_work | PWGTP | 0.268237487 | 0.000422440244 | 0.0057668565 |
| splince_supervised_mechanism40 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| splince_supervised_mechanism40 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| splince_supervised_mechanism40 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| splince_supervised_mechanism40 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
| splince_supervised_union88 | attack:A/SEX | unweighted | 0.660152998 | -0.0277406623 | -0.0119931858 |
| splince_supervised_union88 | attack:A/SEX | PWGTP | 0.666661934 | -0.0198225189 | -0.0115862657 |
| splince_supervised_union88 | attack:A/RAC1P | unweighted | 1.21657652 | -0.0541628406 | -0.0476372935 |
| splince_supervised_union88 | attack:A/RAC1P | PWGTP | 1.21476197 | -0.0499776675 | -0.0436923926 |
| splince_supervised_union88 | attack:AB/SEX | unweighted | 0.657811898 | -0.0274050365 | -0.0109571829 |
| splince_supervised_union88 | attack:AB/SEX | PWGTP | 0.664128653 | -0.0202324725 | -0.00964443536 |
| splince_supervised_union88 | attack:AB/RAC1P | unweighted | 1.2078817 | -0.0497233049 | -0.0442189106 |
| splince_supervised_union88 | attack:AB/RAC1P | PWGTP | 1.20766163 | -0.0446906028 | -0.0382665996 |
| splince_supervised_union88 | attack:A/public_coverage | unweighted | 0.496475268 | -0.0228446545 | -0.0151458748 |
| splince_supervised_union88 | attack:A/public_coverage | PWGTP | 0.503081723 | -0.0184536029 | -0.0119531593 |
| splince_supervised_union88 | attack:A/commute_over20 | unweighted | 0.681479394 | -0.00166955016 | 0.000339269261 |
| splince_supervised_union88 | attack:A/commute_over20 | PWGTP | 0.684198555 | -0.00234431737 | 0.00122765457 |
| splince_supervised_union88 | attack:B/income_binary | unweighted | 0.370400587 | 0 | 0 |
| splince_supervised_union88 | attack:B/income_binary | PWGTP | 0.373836262 | 0 | 0 |
| splince_supervised_union88 | attack:B/civilian_at_work | unweighted | 0.507125227 | 0 | 0 |
| splince_supervised_union88 | attack:B/civilian_at_work | PWGTP | 0.480980872 | 0 | 0 |
| splince_supervised_union88 | attack:B/same_residence | unweighted | 0.528822142 | 0 | 0 |
| splince_supervised_union88 | attack:B/same_residence | PWGTP | 0.498137528 | 0 | 0 |
| splince_supervised_union88 | attack:B/SEX | unweighted | 0.691569787 | 0 | 0 |
| splince_supervised_union88 | attack:B/SEX | PWGTP | 0.690533516 | 0 | 0 |
| splince_supervised_union88 | attack:B/RAC1P | unweighted | 1.26484015 | 0 | 0 |
| splince_supervised_union88 | attack:B/RAC1P | PWGTP | 1.25943457 | 0 | 0 |
| splince_supervised_union88 | utility:A/same_residence | unweighted | 0.486252886 | -0.038898772 | -0.0131956674 |
| splince_supervised_union88 | utility:A/same_residence | PWGTP | 0.462492356 | -0.036576739 | -0.0143618443 |
| splince_supervised_union88 | utility:A/income_binary | unweighted | 0.286864592 | -0.00883385228 | 0.00234741638 |
| splince_supervised_union88 | utility:A/income_binary | PWGTP | 0.300227599 | -0.00735198711 | 0.00381995184 |
| splince_supervised_union88 | utility:A/civilian_at_work | unweighted | 0.277448047 | -0.0018591462 | 0.00601080699 |
| splince_supervised_union88 | utility:A/civilian_at_work | PWGTP | 0.265734752 | -0.00208029525 | 0.00326412101 |
| splince_supervised_union88 | utility:B/public_coverage | unweighted | 0.502145995 | 0 | 0 |
| splince_supervised_union88 | utility:B/public_coverage | PWGTP | 0.511427538 | 0 | 0 |
| splince_supervised_union88 | utility:B/commute_over20 | unweighted | 0.691243071 | 0 | 0 |
| splince_supervised_union88 | utility:B/commute_over20 | PWGTP | 0.690960925 | 0 | 0 |
