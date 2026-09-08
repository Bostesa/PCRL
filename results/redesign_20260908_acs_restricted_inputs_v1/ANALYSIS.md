# Fixed comparisons and interpretation

The experiment answers a question about access and practical prediction under declared training budgets. F and K start from identical teacher-output functions and trainable tensors, but only F can learn from the additional raw PCA32 block. E and S are the exact previously frozen real- and shuffled-label teachers. C trains real-label observers without a protection gradient; D adds that gradient. E-C is therefore not attribute-unaware. Historical selective models use a different parameterization and initial released function and remain contextual references, never the causal F control.

All numbers refer to DEVELOPMENT EVALUATION. Lower log loss is better utility; lower signed prior-relative attack gain is less measured recovery. Negative gains are retained. Unweighted validation chooses the prediction once, including for PWGTP scoring. Three seeds share a cohort, so their sample SDs are not population or survey-design uncertainty. [All seed results](PER_SEED.csv), [paired losses](PAIRED.csv), and [paired means/SDs](PAIRED_AGGREGATE.csv) include every prescribed comparison, task, scope, weighting, and defined/undefined result. Paired CSV values are left-minus-right **losses**; for the same attribute/prior, the attack-gain contrast has the opposite sign.

## Does the raw-input restriction help?

K reduces mean independent SEX and race gain versus F in all four teacher/arm comparisons under both weights. The improvement is neither universal across individual seeds nor robust to all observer evidence. For E-C, independent gains fall from .016629 to .010100 (SEX) and .056080 to .051189 (race), while pooled race gain rises from .074529 to .077666. The weighted pooled race increase is .001871. E-D's pooled race comparison is nearly tied unweighted (.075159 full, .075270 restricted) and favors restriction under PWGTP (.068705 versus .067273). S restriction lowers mean pooled race recovery for both C and D under both weights; its unweighted race improvement occurs in all three seeds.

The source cost is substantial. E-C civilian-at-work loss rises by .064223 under restriction, with seed costs .102506, .060222, and .029939. Its income and public-coverage mean losses also rise. All four restricted conditions fail the joint original source margins in every seed under both weights. In contrast, full conditions meet those source margins in all weighted seeds and all but one unweighted E-F-C seed. Thus a finite independent-audit improvement accompanies a meaningful loss of useful performance.

The residence cost is smaller and depends on weighting: E-C worsens by .001583 unweighted / .001620 PWGTP, E-D by .001230 / .001127. S-C and S-D worsen slightly unweighted but improve slightly under PWGTP. Commute generally worsens with restriction. Reporting only residence, only one weighting, or only independent recovery would omit a material part of the comparison.

## Can teacher-only training recover practical source performance?

Yes, relative to the original 2,048-label downstream probes. Direct E has mean income/civilian/public losses .356662/.414513/.529162; E-K-C reaches .324197/.356223/.512902 under the same downstream-probe evaluation. S shows analogous mean improvements. This establishes better prediction using a larger trained function and its fitting exposure, not information created by postprocessing or proof that erased features lacked usable information.

The fairer source control is the source-only bank, trained on the same representation-fitting source examples and masks for the same 140 mapper epochs. E bank downstream source losses are .320373/.350527/.504200, better than E-K-C/D on all three means. S bank likewise improves all three mean downstream source losses over S-K-C/D. Native heads give the same mean ordering under both weights: E bank civilian loss .358638 versus .368793 for E-K-C, for example. Individual seeds and tasks remain visible in [native contrasts](NATIVE_PAIRED.csv); these are not universal per-example or per-seed dominance claims.

Source-only training therefore suffices for the useful source gains without preservation or an observer. The comparison does not separately identify an effect of label count: it also changes objective and effective prediction function relative to the small direct-teacher probes. Native and downstream scores are not interchangeable, and neither is selected post hoc as the better result. [SOURCE_BUDGET_CONTROL](SOURCE_BUDGET_CONTROL.md) gives the exposure and native-score accounting.

## Does a representation add useful capability beyond source probabilities?

Residence provides a positive answer within this bounded comparison. E-K-C improves over E bank by .016469 unweighted and .014731 PWGTP; S-K-C by .016070 and .014712. All seeds improve under both weights. E-K-C's unweighted seed gains are .012770, .016212, and .020425. This is useful evidence for an interface richer than the three source probabilities.

There is no comparable commute advantage: E-K-C has .687176 versus E bank .686863; S-K-C .686701 versus S bank .684814. The banks also have much smaller race gains: .025224 for E bank versus .051189 independent / .077666 pooled for E-K-C, and .019934 for S bank versus .056403 / .064855 for S-K-C. The representation-versus-bank .005 gain-excess condition therefore limits the favorable residential interpretation. Detailed original-margin checks, including the .01 feature/bank residence advantage, remain in [bank comparisons](bank_comparisons.json).

## Is real-label erasure useful beyond the shuffled control?

E yields lower mean residence loss than S within every input/arm condition under both weights. For restricted C/D, the unweighted residence advantage occurs in all seeds. E also has lower mean independent attribute recovery. However, E has worse restricted income and civilian source losses, and pooled race recovery reverses the apparent attribute advantage: E-K-C .077666 versus S-K-C .064855, E-K-D .075270 versus S-K-D .064744. E's pooled race gain is larger in every unweighted seed for both restricted arms, and the mean reversal survives PWGTP. E has lower mean pooled SEX gain, with seed qualifications.

The teacher maps are not refitted or reselected. Their earlier matched ranks and normalized distortions are retained as historical facts, with exact map/permutation hashes. This controls an important compression difference but does not make every geometric property or nonlinear downstream trajectory identical. Real-attribute teacher selection has a task-specific benefit here, not a robust overall protection advantage.

## Does D help when the raw route is closed?

For E-K, D lowers pooled race gain from .077666 to .075270, with a decrease in every unweighted seed and a similar weighted mean benefit (.069562 to .067273). But pooled SEX gain increases from .018457 to .018576 unweighted and .014958 to .015490 weighted. Independent mean SEX and race gains both increase slightly. Residence also worsens slightly, and source margins still fail.

For S-K, unweighted pooled SEX and race gains decrease only slightly; both comparisons reverse under PWGTP. Source and reserved-task differences are small and mixed. This does not establish that D is useless in general, but the fixed coefficient and schedule do not produce a clear joint benefit after restriction in this matrix. All D-minus-C contrasts and stage changes are retained rather than choosing the favorable attribute or weighting.

## Are recovery increases reproducible as teacher-only compositions?

Yes. Every recorded restricted/bank witness has matching release-then-auditor and teacher-then-student-then-auditor predictions. The direct serialized K interface accepts only teacher values. This includes the independent candidates and learned-release observer catch-up, with its inherited real-label fitting exposure. Global teacher witness choices were frozen from attacker validation before development scoring; no full-input model is admitted to this witness family.

For public fixed parameters and a new example, H=g(T), so a(H)=a(g(T)). The expanded family has extra depth, parameters, source exposure and sometimes protected-label exposure. It is not the original matched independent teacher audit. Affine recovery of the removed residual can improve after nonlinear postprocessing even without raw access; this is consistent with structure already present in T. It neither proves a raw bypass nor makes the residual a pure sensitive variable. [COMPOSED_ATTACKS](COMPOSED_ATTACKS.md) and [MECHANISM](MECHANISM.md) provide the exact evidence.

## Strongest counterexample and remaining criteria

The strongest counterexample to a broad restriction benefit is E-K-C: worse source utility and greater pooled race gain than its matched F control despite lower independent race gain. The strongest counterexample to a representation benefit over source-only predictions is commute, accompanied by much lower bank attribute recovery. These remain true design limitations even though the residential improvement is useful.

The parent for source allowance and residence denominators remains original PCA32. All restricted models and banks fail joint source preservation in all seeds under both weights; none of the new models meets unweighted half-headroom residence retention. Only S-K-C/D meet that residence criterion for one weighted seed each. Attribute halving and feature/bank margins retain their signed denominators and original thresholds. [criteria.json](criteria.json) records all cases; these descriptive margins are neither privacy budgets nor statistical noninferiority tests. RAC1P code 4 lacks fitting/validation support, so a complete race assessment remains unassessable regardless of aggregate gains.

The 360-epoch extension leaves all main selections unchanged but changes three seed-0 bank/attribute predictions. It is therefore inappropriate to transfer saturation conclusions from the previous study to every new interface. [AUDIT_BUDGET](AUDIT_BUDGET.md) separates independent, saved, catch-up, and epoch-0 exposure effects. No further restart or attack family was fitted.

The separate exact study shows why coordination remains worth testing even without an ACS protection win, and why repeated access must be explicit. [PURPOSE_COORDINATION_EXACT](PURPOSE_COORDINATION_EXACT.md) proves the finite single-pair benefit and fresh-repeat reversal. [PURPOSE_COORDINATION_DESIGN](PURPOSE_COORDINATION_DESIGN.md) is the one proposed next experiment, with prediction-only controls and utility conditions that may fail. It has not been launched.
