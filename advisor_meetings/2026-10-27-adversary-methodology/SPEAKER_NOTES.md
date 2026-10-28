# Slide 1: Prior adversaries and our proposed recipient

MAIN SLIDE 1. About two minutes.

Dr. Yus is asking why a recipient with exactly this access is worth studying before we decide how to attack it. We recommend an authorised recipient, not an insider. It receives an agreed output for a permitted task and attempts an additional, prohibited attribute inference. It learns from sensitive-labelled releases of other people. The target contributes only the assigned release.

The prior-work comparison is deliberately scoped. LEACE protects against linear recovery from its erased representation. A nonlinear classifier changes the attack class, so its success does not refute LEACE. FARE explicitly anticipates malicious consumers and arbitrary downstream classifiers of its finite representation. Its certificate is about demographic-parity distance, not our AUC. If an own head reads only the cell, its output remains a function of the cell. A separate clean output is different access.

Song and Shmatikov already study supervised attribute inference from a representation. Their inference attacker receives auxiliary inputs with sensitive labels and an encoder oracle, which our main bank does not receive. Their separate repurposing attack should not be imported as weights access here. Our recipient is not a new adversary, and these access sets are not globally ordered. The original PCRL paper also studied concatenated purpose representations, while the original Outputs Leak What They Use paper exposed representations and outputs.

The proposed core is the release-only single recipient. The access table treats clean auxiliary features, public noise-law knowledge and linked recipients separately. Public algorithm knowledge is compatible with the core, although our main bank only refits to defended releases.

Sources: SOURCE_MAP IDs M1 sections2–5, O1 section5.5, O2 paper.tex192–269, L1 definitions2.1–2.3/section3 and footnote5 p8, F1 sections1 and3–5, S1 section3.1/Figure1 and4.2. Official code pins L2 and F2. Full dimensions are in ACCESS_EVIDENCE.csv.

# Slide 2: Why this access matters, and what needs agreement

MAIN SLIDE 2. About two minutes.

Start from the legitimate service. FARE explicitly separates a data producer and downstream consumer, so the recipient-side question has a published setting. Our particular income/occupation prediction service is an illustration rather than a documented bank or Census deployment. A recipient could be authorised to use features for downstream prediction or receive a score or decision for a permitted task. Which object the service genuinely needs is the first substantive choice. The evaluation asks about additional inference from that delivered object, not from raw inputs the recipient is assumed to possess. Holding target inputs back isolates the release, but recipients with inputs are also plausible and are simply not evaluated here.

The supervised attack needs sensitive labels AND the corresponding auxiliary releases. Public labels alone are insufficient. The benchmark supplies those pairs from other records of the same dataset. No source establishes that a real recipient can acquire this amount of matched information. That is an explicit conditional data grant. If Dr. Yus rejects it as a deployment assumption, the measured result remains a conditional audit, and the paper must narrow its practical claim rather than invent another source of data.

For the optional coalition, the two recipients must actually hold both outputs. A common provider might identify the same account, or both parties might know whom they serve. Those are illustrative linkage mechanisms, not observations in this experiment. The code aligns two outputs by row_id and grants perfect linkage. It does not evaluate how parties obtain identifiers or permission to pool. Rejecting that assumption leaves the single-recipient findings intact and removes coalition claims about practical exposure.

After access is agreed, discuss methodology: owner fitting, separate reader fitting and selection, final scoring, utility and feedback. All four experiments are development evidence on reused assessment records. There is no successful new algorithm to present next. Paper structure follows these choices.

Sources: F1 section1 for producer/consumer setting; M1 sections1–3, Table1 access, section5 protocol; C2–C7 for implementation. A1 and S1 establish auxiliary-inference precedents, not our service availability. R1 records corrected access and withdrawn injectivity claims.

# Slide 3: Appendix A · Access is a set of capabilities

APPENDIX. The noise-aware variants were exploratory in Experiment1 and only attacked noisy features. A2 uses algorithm/noise-law knowledge while retaining release-only auxiliary inputs. A4 additionally receives pre-noise features for auxiliary fitting records. Neither receives a clean scored target vector or encoder/head parameters, and neither is an averaging attack. Three noise release seeds are separately attacked experimental realizations, not three queries pooled for a target. The original durable-guarantees LRT likewise does not imply model weights merely because it has clean auxiliary examples. Cell-conditional readers use released-cell counts and are not fitted-tree readers. The true-task-label reference reads the scored person's actual task label and must not be described as this recipient.

Coalition: one Adult income+occupation pair targeting race, no defended pairs or imperfect linkage. Canonical record keys assign roles/group duplicates. Pair construction actually asserts matching row_id before concatenation. These are different functions.

Sources: M1 section3 and access Table1; C4 RELEASE_ACCESS_TABLE.csv rows8–9 and access.py29–39/59–77; D1 battery.py53–106; C6 odx/run.py199–210; C3 oar/study.py372–382. Source map pins all code.

# Slide 4: Appendix B · What the empirical protocol does

APPENDIX. Native historical check is separate from these attacker roles: one-hot ridge is fitted AND scored in sample on the test representation, with the scored sensitive labels. Provider training uses source train data. For Experiments2–4, defense_fit excludes train/test overlaps and supplies owner head_fit/head_val (20% holdout). The reader roles are hash-assigned groups of source test records. A certification carve-out of1,500 Adult/1,385 HMDA rows reduces every arm's reader budget. Experiment1 initially retained source-overlap records, with removal sensitivity reported later.

The common reader slate is LR5 regularization settings, GBT20 configurations, MLP18. Selection at seed0 uses validation log loss, followed by refits of the selected nonlinear recipe at seeds1 and2. Trees and MLP hold out10% of fitting rows for early stopping. Assessment labels do not select readers. For concatenations, validation candidates may ignore a component.

FARE is also selected on validation nonlinear AUC among admissible probe-accuracy configurations. Noise scale is selected there. Thus defense and attack selection share validation feedback. The easier-task FARE comparison adds80% probe-gain retention. Utility probe labels live in attacker roles, whereas released own heads fit/select on owner roles. In Experiment4, heads and defenses are reused.

Macro one-vs-rest AUC covers supported classes, not arbitrary rare groups. Three stored encoder seeds are averaged. Inference differs across experiments: E1/E2 one-sided Bonferroni percentile bounds (24/12 endpoints,20,000 bootstrap draws), E3/E4 simultaneous two-sided Bonferroni normal intervals from1,999 group bootstrap draws. E1 level bars differ from E2–E4 contrasts. No new statistic was calculated for this deck.

Sources: M1 section5; C2 bench.py648–694 and recipes.py71–90; C3 oar/study.py56–85/134–149/325–341 and fare_run.py25–47; C4–C7 protocols and historical pins.

# Slide 5: Appendix C · What the findings can establish

APPENDIX. The empirical contribution is a common audit of native checks, actual released surfaces and usefulness, including matched zero-fairness compression controls and one linked-output pair. It does not need a novel adversary. Nonlinear recovery after linear erasure, output leakage and composition are already known. FARE covers nonlinear downstream functions of the cell, so do not call nonlinear recovery simply out of FARE scope. Its protected quantity is demographic-parity distance, while this audit measures AUC. The unavailable/weak certificate does not establish violation. A head using only the cell does not add a new surface beyond that cell; the clean bypass does.

Native11/42 count is corrected in the current source, superseding older12/42 text. Model/head quality matters: low recovery from poor frozen decisions is not useful protection. Useful-head FARE reduction did not establish the one-point accuracy non-inferiority requirement. Coalition results are one untreated Adult pair, not a general composition or information-theoretic synergy result. High or low finite-reader AUC does not identify population information removal. Determinism does not imply injectivity, and sample uniqueness does not prove population sensitive-posterior preservation. Later negative method studies remain negative.

The sequence for discussion is adversary/motivation, empirical methodology, then whether any proposed algorithm has a defensible goal, and only afterward manuscript structure. No new training, reader fitting, statistics or data-role opening occurred during this preparation.

Sources: M1 sections5–7, C1/C4 native check, F1 and C5 certificate scope, C7 utility-qualified results, R1 CORRECTIONS sections1/6/7.
