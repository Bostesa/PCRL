# R5 corrected reporting closeout

This is an exploratory method search on repeatedly used Adult people. The V2 replay preserves exact original U0/Ucal references in separate REF_* keys. All V1 artifacts and the reference-collision diagnosis remain preserved. No models, readers, orientations, selections, caps, bootstrap draws or endpoint families changed.

Paired means average the two fixed fitted source runs on the same people. Absolute AUC brackets are descriptive pointwise 95% intervals (K=1), computed from the shared 2048 duplicate-group resamples. They are not the simultaneous competitive conjunction. Task metrics are displayed means; point eligibility and every interval requirement remain separate for each task/seed/reference.

## Primary full feature + own-output service

| Arm | Income acc | Income NLL | Income Brier | Occupation acc | Occupation NLL | Occupation Brier | Income AUC [95%] | Occupation AUC [95%] | Pair AUC [95%] | Original point eligibility | Utility K240 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U0 | 0.851454 | 0.333608 | 0.205087 | 0.491275 | 1.266441 | 0.647436 | 0.821195 [0.806777, 0.835613] | 0.858140 [0.844714, 0.871567] | 0.870143 [0.857737, 0.882550] | eligible 24/24 | P21/U3/V0/M0 |
| U_NEW | 0.851454 | 0.333608 | 0.205087 | 0.491275 | 1.266441 | 0.647436 | 0.821195 [0.806777, 0.835613] | 0.858140 [0.844714, 0.871567] | 0.870143 [0.857737, 0.882550] | eligible 24/24 | P21/U3/V0/M0 |
| U_TASKPOWER | 0.851454 | 0.333608 | 0.205087 | 0.491275 | 1.266441 | 0.647436 | 0.821195 [0.806777, 0.835613] | 0.858140 [0.844714, 0.871567] | 0.870143 [0.857737, 0.882550] | eligible 24/24 | P21/U3/V0/M0 |
| LOCAL_TWIN | 0.850783 | 0.335571 | 0.205821 | 0.494183 | 1.268198 | 0.648175 | 0.819384 [0.804994, 0.833773] | 0.849895 [0.836031, 0.863759] | 0.866718 [0.853964, 0.879472] | eligible 24/24 | P13/U11/V0/M0 |
| LOCAL_CONDITIONAL | 0.850336 | 0.335654 | 0.205851 | 0.493736 | 1.268437 | 0.648278 | 0.814814 [0.800146, 0.829483] | 0.849089 [0.835248, 0.862931] | 0.864902 [0.852130, 0.877673] | eligible 24/24 | P11/U13/V0/M0 |
| JOINT_CONDITIONAL | 0.850112 | 0.336150 | 0.206168 | 0.493512 | 1.268751 | 0.648444 | 0.815163 [0.800213, 0.830112] | 0.846895 [0.832739, 0.861051] | 0.861891 [0.848985, 0.874798] | eligible 24/24 | P7/U17/V0/M0 |
| JOINT_UNCONDITIONAL | 0.850112 | 0.336546 | 0.206032 | 0.491946 | 1.268278 | 0.648419 | 0.827341 [0.813017, 0.841666] | 0.851898 [0.837871, 0.865925] | 0.861256 [0.848289, 0.874223] | eligible 24/24 | P10/U14/V0/M0 |
| RAW_JOINT | 0.848770 | 0.339593 | 0.207811 | 0.495302 | 1.269066 | 0.648705 | 0.751910 [0.734385, 0.769436] | 0.838102 [0.823184, 0.853021] | 0.856824 [0.843452, 0.870197] | eligible 24/24 | P6/U18/V0/M0 |
| SEQ12 | 0.850783 | 0.335569 | 0.205819 | 0.493512 | 1.268768 | 0.648456 | 0.819490 [0.804850, 0.834130] | 0.847268 [0.833122, 0.861414] | 0.863005 [0.850320, 0.875690] | eligible 24/24 | P10/U14/V0/M0 |
| SEQ21 | 0.849888 | 0.336136 | 0.206157 | 0.494183 | 1.268198 | 0.648174 | 0.817458 [0.802871, 0.832044] | 0.849119 [0.835253, 0.862984] | 0.867285 [0.854488, 0.880083] | eligible 24/24 | P10/U14/V0/M0 |
| H_RAW_CSTAR | 0.853691 | 0.331446 | 0.204336 | 0.488591 | 1.279670 | 0.654896 | 0.734850 [0.715149, 0.754550] | 0.751460 [0.732394, 0.770525] | 0.781964 [0.763541, 0.800388] | diagnostic: 16 pass/8 fail | outside K240 |

Brier is the sum across classes. P/U/V/M counts mean interval PASS/UNRESOLVED/VIOLATION/MISSING. Each primary-pool arm has 24 utility intervals in the separate K240 family. H_RAW_CSTAR has different historical training/search exposure and passes 16 clauses while failing 8: occupation NLL and Brier against both references in both seeds. It fails the original point gate and is descriptive outside the primary pool and K240. Crossing intervals remain unresolved; they did not prevent the point-qualified fresh attack comparison.

## Secondary score contracts

| Arm | Actual contract | Income acc | Income NLL | Income Brier | Occupation acc | Occupation NLL | Occupation Brier | Income AUC [95%] | Occupation AUC [95%] | Pair AUC [95%] | Original point eligibility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U0 | new probabilities + decision | 0.851454 | 0.333608 | 0.205087 | 0.491275 | 1.266441 | 0.647436 | 0.705219 [0.683985, 0.726454] | 0.838903 [0.823631, 0.854176] | 0.852402 [0.838309, 0.866496] | eligible 24/24 |
| U_NEW | new probabilities + decision | 0.851454 | 0.333608 | 0.205087 | 0.491275 | 1.266441 | 0.647436 | 0.705219 [0.683985, 0.726454] | 0.838903 [0.823631, 0.854176] | 0.852402 [0.838309, 0.866496] | eligible 24/24 |
| U_TASKPOWER | new probabilities + decision | 0.851454 | 0.333608 | 0.205087 | 0.491275 | 1.266441 | 0.647436 | 0.705219 [0.683985, 0.726454] | 0.838903 [0.823631, 0.854176] | 0.852402 [0.838309, 0.866496] | eligible 24/24 |
| LOCAL_TWIN | new probabilities + decision | 0.850783 | 0.335571 | 0.205821 | 0.494183 | 1.268198 | 0.648175 | 0.698160 [0.676271, 0.720048] | 0.823084 [0.806933, 0.839235] | 0.840648 [0.825693, 0.855603] | eligible 24/24 |
| LOCAL_CONDITIONAL | new probabilities + decision | 0.850336 | 0.335654 | 0.205851 | 0.493736 | 1.268437 | 0.648278 | 0.697898 [0.675850, 0.719947] | 0.820231 [0.804020, 0.836443] | 0.840075 [0.825091, 0.855059] | eligible 24/24 |
| JOINT_CONDITIONAL | new probabilities + decision | 0.850112 | 0.336150 | 0.206168 | 0.493512 | 1.268751 | 0.648444 | 0.695908 [0.673732, 0.718083] | 0.815991 [0.799469, 0.832513] | 0.834821 [0.819577, 0.850065] | eligible 24/24 |
| JOINT_UNCONDITIONAL | new probabilities + decision | 0.850112 | 0.336546 | 0.206032 | 0.491946 | 1.268278 | 0.648419 | 0.696691 [0.674086, 0.719295] | 0.821940 [0.805741, 0.838138] | 0.840419 [0.825508, 0.855331] | eligible 24/24 |
| RAW_JOINT | new probabilities + decision | 0.848770 | 0.339593 | 0.207811 | 0.495302 | 1.269066 | 0.648705 | 0.686353 [0.663887, 0.708819] | 0.817047 [0.800648, 0.833446] | 0.838567 [0.823606, 0.853528] | eligible 24/24 |
| SEQ12 | new probabilities + decision | 0.850783 | 0.335569 | 0.205819 | 0.493512 | 1.268768 | 0.648456 | 0.698164 [0.676310, 0.720017] | 0.816311 [0.799796, 0.832826] | 0.836866 [0.821719, 0.852012] | eligible 24/24 |
| SEQ21 | new probabilities + decision | 0.849888 | 0.336136 | 0.206157 | 0.494183 | 1.268198 | 0.648174 | 0.698165 [0.676292, 0.720039] | 0.822686 [0.806513, 0.838860] | 0.840941 [0.826044, 0.855837] | eligible 24/24 |
| H_RAW_CSTAR | new probabilities + decision | 0.853691 | 0.331446 | 0.204336 | 0.488591 | 1.279670 | 0.654896 | 0.686652 [0.663738, 0.709566] | 0.733430 [0.713390, 0.753471] | 0.755977 [0.736305, 0.775649] | diagnostic: 16 pass/8 fail |
| P0 | legacy complete codes + outputs + metadata | 0.851454 | 0.327600 | 0.206646 | 0.491275 | 1.273578 | 0.651569 | 0.703099 [0.681653, 0.724545] | 0.761953 [0.744443, 0.779463] | 0.802763 [0.786527, 0.818998] | eligible 24/24 |
| S12 | legacy complete codes + outputs + metadata | 0.851454 | 0.331724 | 0.207756 | 0.491275 | 1.274156 | 0.651652 | 0.698628 [0.676779, 0.720478] | 0.759333 [0.741925, 0.776740] | 0.798701 [0.782236, 0.815166] | eligible 24/24 |

Every new arm uses only its own probabilities and decision in this table. P0/S12 retain their literal complete legacy codes, probabilities, decisions and metadata, with freshly fitted legacy readers; these are different-capacity services and cannot establish a matched full-feature mechanism effect. Historical task metrics are exact finite-law expected per-action metrics, not metrics of averaged probabilities. Score comparisons use their own frozen K33 family.

## Registered primary privacy contrasts

Pair benefit is comparator AUC minus JOINT AUC. Each local worsening is JOINT minus comparator. All brackets below are the original K51 simultaneous intervals with df2233; competitive pair target is lower ≥0.020 and local guard is upper ≤0.010. The full conjunction also requires all 24 JOINT original utility bounds, valid controls, independent verification and deployment.

| Comparator | Pair benefit [K51] | ≥0.020 | Income worsening [K51] | ≤0.010 | Occupation worsening [K51] | ≤0.010 |
| --- | --- | --- | --- | --- | --- | --- |
| U0 | 0.008252 [0.000111, 0.016393] | SHORTFALL | -0.006032 [-0.012706, 0.000641] | PASS | -0.011245 [-0.019585, -0.002906] | PASS |
| U_NEW | 0.008252 [0.000111, 0.016393] | SHORTFALL | -0.006032 [-0.012706, 0.000641] | PASS | -0.011245 [-0.019585, -0.002906] | PASS |
| U_TASKPOWER | 0.008252 [0.000111, 0.016393] | SHORTFALL | -0.006032 [-0.012706, 0.000641] | PASS | -0.011245 [-0.019585, -0.002906] | PASS |
| LOCAL_TWIN | 0.004826 [-0.002662, 0.012315] | SHORTFALL | -0.004221 [-0.009009, 0.000566] | PASS | -0.003000 [-0.009618, 0.003618] | PASS |
| LOCAL_CONDITIONAL | 0.003010 [-0.003153, 0.009173] | SHORTFALL | 0.000348 [-0.004540, 0.005236] | PASS | -0.002194 [-0.007627, 0.003238] | PASS |
| JOINT_UNCONDITIONAL | -0.000635 [-0.006684, 0.005414] | SHORTFALL | -0.012179 [-0.019856, -0.004501] | PASS | -0.005003 [-0.011138, 0.001132] | PASS |
| RAW_JOINT | -0.005067 [-0.013430, 0.003295] | SHORTFALL | 0.063252 [0.043198, 0.083306] | VIOLATION | 0.008793 [-0.001240, 0.018825] | UNRESOLVED |
| SEQ12 | 0.001114 [-0.004744, 0.006971] | SHORTFALL | -0.004328 [-0.008661, 0.000006] | PASS | -0.000373 [-0.007322, 0.006576] | PASS |
| SEQ21 | 0.005394 [-0.001500, 0.012288] | SHORTFALL | -0.002295 [-0.006665, 0.002074] | PASS | -0.002224 [-0.008833, 0.004385] | PASS |

JOINT primary utility interval statuses: PASS 13/24, UNRESOLVED 11/24, VIOLATION 0/24. Its point gate passes 24/24. These are separate conclusions.

## Component quantities without extra multiplicity

| Claim | Fixed control | Pair benefit [K51] | Direction | Complete component conjunction |
| --- | --- | --- | --- | --- |
| coalition coupling | LOCAL_TWIN | 0.004826 [-0.002662, 0.012315] | unresolved direction | component conjunction not established |
| task conditioning | JOINT_UNCONDITIONAL | -0.000635 [-0.006684, 0.005414] | unresolved direction | component conjunction not established |
| kernel versus raw adversary | RAW_JOINT | -0.005067 [-0.013430, 0.003295] | unresolved direction | component conjunction not established |
| joint versus freeze-first1→2 | SEQ12 | 0.001114 [-0.004744, 0.006971] | unresolved direction | component conjunction not established |
| joint versus freeze-first2→1 | SEQ21 | 0.005394 [-0.001500, 0.012288] | unresolved direction | component conjunction not established |

Component direction reuses the identical registered privacy quantity and requires a lower bound strictly above zero. A favorable pair interval alone cannot establish a component claim when its required utility or local conjunction remains unresolved. Sequential statements concern these bounded optimization procedures.

## Verdict suggestions and limits

The useful baseline and all ten primary-pool arms pass all 24 original point clauses. U_NEW and U_TASKPOWER selected the zero-update source alias; their selected service adds no empirical mechanism change over U0. JOINT stays point useful, with paired-mean income/occupation accuracies 0.850112/0.493512, but its complete utility interval conjunction remains unresolved (13 PASS, 11 UNRESOLVED in K51). Occupation performance is uneven across classes; each seed has one reporting person in class 5 and zero recall there. The per-seed CSV preserves balanced accuracy, every recall and confidence, so overall utility is not mistaken for uniformly strong class performance.

The strongest favorable primary pair comparison is versus the tied U0/U_NEW/U_TASKPOWER baselines: benefit 0.008252 [0.000111,0.016393]. It is positive on this approximate conditional-development interval and remains below the required 0.020 target. Every registered comparator is a SHORTFALL for that pair target. RAW_JOINT is the strongest matched feature-pair reader control (AUC 0.856824 versus JOINT 0.861891). The strongest adverse local comparison is income versus RAW: JOINT worsens AUC by 0.063252 [0.043198,0.083306], violating the 0.010 guard.

Conditioning contributes no established pair gain over JOINT_UNCONDITIONAL, coalition coupling contributes no established gain over its exact LOCAL_TWIN, and joint training contributes no established gain over either sequential order: each component interval crosses zero and its complete conjunction is unestablished. The competitive method verdict is negative. RAW should remain the strongest eligible feature-pair control for the next decision; H_RAW_CSTAR has substantially lower observed recovery but fails the original utility gate and cannot be promoted.

For the new output-only service, JOINT has the lowest observed pair AUC among the new arms (0.834821), while freshly scored literal P0/S12 have lower pair AUCs 0.802763/0.798701 under their separate legacy contract. These narrower/different interfaces do not establish reusable-feature privacy or a matched full-feature mechanism effect. Conditioning, MMD/block testing and end-to-end training are established ideas; this run supplies no empirical competitive or component result that warrants a new-method novelty claim. All required new families and available historical references were actually scored; an unavailable-reference or incomplete-control explanation is unnecessary.

All 26 fresh banks passed their declared null, independently shuffled-label and planted-leak checks. The pair envelopes include every local reader as an ignore-other alias. R6 independently refitted four predeclared strongest ATT-validation reader types for the secondary sensitivity check; this does not mean every reader was independently refitted.

Oracle label-only and true-class/pair conditional diagnostics are interpretation-only and were not independently numerically verified in this run. They never supply primary inputs or success criteria. Source initialization/training history is shared, no new confirmation population exists, and passed fixtures or low kernel losses do not establish unconditional privacy.

Standalone aggregate files retain all per-seed task/balanced-accuracy/recall/confidence values, complete primary/utility/secondary endpoint intervals, per-seed absolute AUCs and reader-level recovery. Root owns the final scientific decisions, independent comparison, deployment/custody receipts and publication closeout.

## Actual V1/V2 correction comparison

Original scalar utility gate changes: 0; legacy scalar gate changes: 0; endpoint status changes: 0. Maximum original scalar clause change: 0; maximum legacy clause change: 2.3203217125455922e-11. Absolute recovery records are exactly identical: True. Original source, nomination and checkpoint associations remain identical. The archive SHA changed because references now have separate keys and preserve the original reference values. R6 supplies the independent array-byte and final interval comparison receipts.

| Family | Max point change | Max lower change | Max upper change | Max SE change | Status changes |
| --- | --- | --- | --- | --- | --- |
| primary | 2.3203195007731603e-11 | 1.9705381857498816e-11 | 2.6701007724283521e-11 | 1.0597902763698208e-12 | 0 |
| secondary | 0 | 0 | 0 | 0 | 0 |
| utility | 2.3203195441412472e-11 | 1.1095663312728812e-10 | 6.4550242442510399e-11 | 2.3622486380405467e-11 | 0 |
