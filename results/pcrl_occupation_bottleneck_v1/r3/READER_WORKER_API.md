# Occupation bottleneck reader and task diagnosis interfaces

This is an exploratory diagnosis and possible repair pilot on repeatedly used Adult development people. The motivating comparisons and source nominees are already known. New locks, reader fits, model seeds, or internal folds do not make this fresh confirmation.

Code is separate from independent verification. These workers produce scientist receipts. The coordinator alone executes numerical jobs in the new durable occupation queue. Import or AST checks do not establish numerical correctness.

## Reader contracts

The actual full wire remains `z16, centered own logits K, own probability K, own numeric decision`, with income before occupation in the aligned pair. Income width21, occupation width29, pair width50. The nine predecessor readers per local or pair view retain their names and transformations. Pair predictions always contain exact aliases of every local reader ignoring the other recipient.

Separate extended readers are `LOGITS_OUTPUT_{LR_C1,MLP_64x32,HGB_15x200}`, `DECISION_{LR_C1,MLP_64x32,HGB_15x200}`, and `CONFIDENCE_{LR_C1,MLP_64x32,HGB_15x200}`. LOGITS_OUTPUT receives centered own logits, probabilities and decision. CONFIDENCE receives max own probability only. DECISION receives only the actual numeric own decision. Pair views concatenate aligned own contracts. The legacy OUTPUT reader receives probabilities and decision only; its narrower contract cannot be relabeled as centered-logit output recovery.

Primary envelopes use the nine predecessor base readers and their pair local-ignore aliases. The three extended contract envelopes are reported separately. A finite-reader AUC difference does not decompose information and does not establish an information floor.

## Python API

`occupation_bottleneck.readers.fit_development_bank(att, inner=None, *, seed, private_output, queue_guard, bindings, include_base=True, include_extended=False, controls=False, view_aliases=None, diagnostic=None)` requires an authenticated new queue guard before any fit/replay. ATT keys must be exactly `row_id,unit,r1,r2,sex`; INNER and FIT-selection diagnostic releases exactly `row_id,unit,r1,r2`. True task labels cannot enter those matrices.

The inherited SHA-ranked group split and model seed `2026101205+source_seed` are immutable. Readers fit only on ATT training groups. Orientation is selected only on ATT validation. No prediction-row SEX labels influence orientation or model fitting.

When supplied, FIT-selection release and reader scores are preserved for C2/C3. Label-free INNER predictions are frozen before all INNER truth access. Fresh outputs are exclusive-create, owner-only private artifacts. `FROZEN_COMPLETE.json` binds model files, ATT development predictions, optional FIT-selection/INNER predictions, exact bindings, ATT validation AUCs, aliases and fit accounting.

`view_aliases[view]` has `directory,names,expected_bindings,sources`. The coordinator registers and verifies every source pin. Reuse compares the exact literal reader-contract matrices, person ordering, inherited ATT train/validation split, source/model seeds and receipt bindings. Merely matching a checkpoint string or approximate output tolerance is insufficient. Identical frozen income wires in pilot arms must point to the same canonical bank. Projected outputs may use source output readers only if their contract values are bitwise equal; decision aliases can pass when decisions are exactly preserved.

`occupation_bottleneck.reader_specs.reuse_alias(bank)` constructs a private queue alias from the R1 metadata inventory. `reader_bindings(...)` records the exact checkpoint, role feature hashes, release variant, split, preprocessing and source/model seed. `fit_count(...)` calculates the registered numerical model count after aliases. `freeze_refit_targets(...)` freezes strongest ATT-validation targets for U0 and RAW on both source seeds, with additional mandatory pilot/control arms supplied by the coordinator. It performs no refits.

## Queue workers

Common private spec fields are `unit_id,arm_id,seed,manifest,manifest_sha256,science_lock,science_lock_sha256,diagnostic_lock,diagnostic_lock_sha256,checkpoint,checkpoint_sha256,private_output,public_output`. Every input is an exact hash source in the new queue. Existing predecessor queues remain untouched.

`python -m occupation_bottleneck.evaluation_worker --phase trajectory --spec PRIVATE_SPEC` uses stage `admission`, loads authorized FIT features/tasks/references, and evaluates FIT fold2 actual own heads. It reports accuracy, inherited NLL floor1e-12, summed Brier, fixed-list balanced accuracy, every class recall/support, confidence by class, and gain over the inherited all-FIT constant prior. The source saw all FIT. Thresholds are descriptive; the old unequal-exposure direct FIT confidence gate is not applied. Reference arrays use `REF_U0/REF_Ucal` keys distinct from candidate U0.

`--phase fit` uses stage `attacks` and the reader API above. Spec options include `include_base,include_extended,controls,view_aliases,forward_inner,forward_fit_selection,bindings,release_variant`. `release_variant=head_row_space` builds the public geometry from the exact current own head, checks real-row AP/logit/probability/decision parity, releases factorized Pz16, and recomputes every output from actual Pz16. It refuses geometry audit failures.

`--phase report` and `--phase utility-report` require a separately supplied root reporting seal. The canonical reporting lock must be committed at HEAD, declare all candidates/readers frozen, and bind every reader freeze. The workers verify all those receipts before source loading can hash/read any INNER truth. Unit counts and interval families are supplied by the coordinator; no predecessor K51/K240 constants are inherited.

## C3 diagnostics

`python -m occupation_bottleneck.diagnostic_worker --spec PRIVATE_SPEC` uses stage `admission`. The spec binds the new manifest and locks, private/public outputs, and optional frozen bank pins (`arm_id,seed,freeze,release,scores`). FIT folds0/1 train Laplace1 SEX probability tables for true income, occupation and their pair; separate FIT fold2 rows score those tables. Source training saw all FIT, so the score is an exploratory shared-training diagnostic.

Fresh ATT-trained readers are evaluated on the saved FIT-fold2 releases and within true-task strata. The predeclared support rule is at least32 distinct groups of each SEX per stratum. Sparse strata are MISSING, never pooled after seeing results. Predicted-decision and confidence recovery are separate extended reader contracts. True labels remain interpretation-only inputs and never enter deployed primary attacks.

Safe ATT task-label exports are absent. The implementation uses separate authorized FIT diagnostic folds, and does not open/create all-role exports. Final INNER task-association replay must use the same frozen tables after the global reporting seal; it must not refit or reorient them.

## Metadata inventory and model counts

R1 found84 nominal model packages:72 core family/strength/epoch/seed entries and12 fixed anchors, with zero missing checkpoints. Epoch-zero source parity is pending numerical admission. If the18 core epoch-zero packages all match the two U0 anchors,66 unique packages remain.22 stored bank candidates include U_NEW/U_TASKPOWER aliases;18 canonical banks remain after exact U0 parity admission. Each has27 scientific models and6 real-data control models. No reader files, arrays or checkpoint tensors were loaded to produce this metadata summary.

The complete nominal fresh base count after those aliases is48 unmeasured packages×27=1296 scientific fits. Six U0/RAW/JOINT seed anchors require27 extended fits each=162. Six projected anchors require up to54 scientific fits each, reduced only by proven exact contract aliases; exact decisions alone can reduce nine fits per package. The coordinator freezes any budget-limited subset before scores and calculates actual units from the admitted bank aliases. Metadata availability is not a claim that all numerical work has run.
