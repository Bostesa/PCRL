# Slide 1: Prior work and our recipient

The recipient is the party allowed to receive model information for a stated purpose. The comparison table separates what each study exposes from the class of protection or attack it considers. LEACE limits linear recovery from erased features. Nonlinear recovery does not refute that guarantee. FARE's certificate covers any downstream classifier using only its finite cell, and measures demographic-parity distance rather than our AUC. Song and Shmatikov already study sensitive-attribute inference. Their inference attacker has labelled auxiliary inputs and a black-box encoder oracle. Our main bank instead receives the corresponding labelled releases. These access sets differ, so we do not call ours globally stronger.

Our recommended primary receives exactly its agreed view for a person, once, and learns from releases and protected labels of other records. It has no raw target input or label, model weights, fitted defense map, gradients or chosen/repeated queries. Public algorithm knowledge is permitted, although the main bank simply refits to the defended releases. The table is a summary. The full access table records the separate experiments and exceptions.

Sources: SOURCE_MAP.md M1 sections2–5; L1/L2 LEACE definitions2.1–2.3/section3, official version0.2.4; F1/F2 FARE sections1,3–5 and official tree/evaluator; S1 Song section3.1/Figure1 and4.2. No claim of a new adversary.

# Slide 2: What the attacker receives

This is an illustrative service. A provider gives an income prediction to a recipient for an allowed task. We ask whether the recipient can use that delivered information, with labelled examples, to infer sex. This is not a claim about an observed bank, Census service or legal violation. The person being scored contributes only the agreed output. The labelled examples contain other people's outputs and known sensitive labels. The benchmark grants these pairs. Public sensitive labels alone would not supply the matching model outputs, and we have not shown how a real recipient acquires the declared sample.

Dr. Yus's first decision is whether this is an actor and data budget worth studying. The question remains meaningful if every defense resists, since resistance would be evidence under that stated access. Next agree what the permitted service needs to deliver: features, scores or decisions. A two-recipient extension is separate. It requires both parties to receive their outputs and be able to match the same person. Our one evaluated coalition assumes perfect linkage by aligned rows. A shared account identifier is only an illustrative route, not something we observed.

After agreeing the adversary, discuss methodology. There is no successful new algorithm established in the record. Paper structure comes afterward. Sources: M1 sections1–3/Table1; F1 producer/consumer setting; C2–C7 implementation; O1 original coalition; R1 corrected motivation. Main slides1–2 are about3–5minutes together.

# Slide 3: Appendix: how the attack is evaluated

The owner first fits the task and defense on its owner-side data. The attacker then fits classifiers on attacker_fit, chooses recipes by attacker_val log loss and measures recovery on assessment. The scoring evaluator uses assessment labels to compute AUC. Those labels are not inputs to the attacker's target prediction. Logistic regression has5 settings, boosted trees20 and the MLP18. The nonlinear recipe selected at seed0 is refitted at seeds1 and2. Tree/neural early stopping uses a10% internal fitting holdout.

The dashed feedback arrow is important: FARE configurations and noise scale were also chosen using attacker-validation performance. FARE minimizes validation nonlinear AUC among configurations satisfying the task-probe accuracy rule. The easier task additionally requires80% of probe gain. This is development feedback rather than independent defense tuning. Utility sometimes measures a refitted probe, whereas Experiment4 measures the released useful head. All assessment records had been used in earlier analyses, and intervals condition on fitted models and readers.

E1 fitting/selection/scoring: Adult7571/2239/5250, HMDA6794/2089/4778. E2–E3: Adult6065/2235/5243, HMDA5388/2082/4764. E4 uses Adult roles only. Certification carve-out is1500/1385, affecting all arms. Owner defense_fit has24127/63704 records with20% head holdout in E2–E4. E1 retained overlaps before a later sensitivity removal.

Sources: M1 section5; C2 bench.py648–694/recipes.py71–90; C3 oar/study.py56–85,134–149,325–341 and fare_run.py25–47; C4–C7 protocols. Full source pins and budgets remain in ACCESS_EVIDENCE.csv.

# Slide 4: Appendix: recovery and task usefulness

Existing paper Figure2, copied without changing any plotted value. This is the useful logistic-head comparison on Adult income/sex. The x-axis is accuracy gain over a constant predictor. The y-axis is sensitive-attribute recovery AUC, with higher values meaning the tested attacker can distinguish the attribute more easily. Left: output scores. Right: hard decisions. Large markers are means over three stored encoder seeds. Small pale markers are per-seed values. Displayed bars are marginal90% intervals, not the simultaneous intervals used for the registered endpoints.

FARE lowered recovery from useful-head scores, but its one-point accuracy non-inferiority condition was not established. The zero-fairness tree accounts for part of that reduction. This is a utility-qualified comparison against a finite bank of readers, not an optimal privacy claim. Releasing a useful head's scores is different from adding the untreated clean output as a bypass. The records, defenses and heads were reused, so this is development evidence.

Source: papers/combined_empirical_v2/figures/fig_useful_head.pdf at9e6d01f28. Caption main.tex551–559. C7 useful-head protocol and committed primary/utility outputs. No plot data or analysis script was changed.

# Slide 5: Appendix: output format and pooling

Existing paper Figure3, copied without changing any plotted value. Left: frozen-head output formats for one recipient. Full logits, probabilities, centred logits and hard decisions are different visible objects. The low recovery from these frozen hard decisions coincided with weak task predictions. It is not evidence of useful protection by itself. Right: one Adult coalition, income and occupation-group outputs pooled to infer race. The label employment refers to occupation-group prediction. Both outputs are matched to the same record, with perfect linkage granted. No linkage procedure is tested.

Bars are means over three stored encoder seeds without intervals. The original axes begin near0.45, not zero. Registered secondary pair contrasts were reported separately, with full-logit pair0.917 versus stronger single0.845, centred0.883 versus0.792, and hard decision0.611 versus0.576. The joint bank can choose a single recipient, and some single comparators use format banks, so the searches are not identical. These are existing results, not new calculations. The gain is not a mutual-information synergy or composition theorem.

Source: papers/combined_empirical_v2/figures/fig_formats_pairs.pdf at9e6d01f28, main.tex629–652; C6 odx/run.py193–214. Full qualifications remain in SOURCE_MAP.md and the prior access table.
