# One conditional occupation repair

Status at design freeze: not yet triggered. The coordinator must decide from FIT/ATT evidence, source reproduction, both source seeds, matched runnable controls and budget. Newly opened INNER results cannot determine this decision. If extra-feature leakage is the supported explanation and ordinary head compression captures the useful gain, choose the tested interface baseline and do not run this training pilot.

## Prior evidence and interpretation

Already-public predecessor outcomes influenced the focus: useful source heads existed; JOINT had an approximately 0.00825 pair AUC benefit against U0, below 0.020, and worsened income recovery against RAW by approximately 0.06325, above 0.010. RAW occupation recovery was the large remaining recipient component. These are already-exposed motivating aggregates. They are not a forecast, new discovery or a fresh confirmation.

The proposed pilot isolates occupation updates while preserving the exact RAW income package. It does not establish superiority over all joint/sequential training designs and introduces no new privacy theorem. The two source seeds are paired fitted models on the same people.

## Common initialization and disclosure

Use the reproduced admitted RAW incumbent in each source seed, including its own useful current heads and source class order. All arms begin with identical occupation weights, critic states and fixed normalization. Freeze the RAW income encoder/head completely, and deduplicate the identical income wire/readers across arms. The runtime wire remains actual z16, actual own centered logits, actual own probabilities and actual decisions for both recipients; every own output is recomputed from its own actual features. No source teacher outputs, labels, group IDs or hidden flags enter the deployed wire.

## One common recipe

Default maximum 20 epochs with checkpoints 0/5/10/20, batch size 256, bounded SGD learning rate 1e-4 and maximum Euclidean displacement 5e-4 (equivalent to clipping an unbounded direction at norm five before the step). Freeze one common sensitivity coefficient from the admitted RAW configuration before pilot fits; do not recalibrate one arm after seeing results. If engineering forces a fixed step ladder, at most three common recipes may replace this one only through a prefit frozen amendment; no appended recipes.

Use inherited FIT folds 0/1 only for source continuation, task updates, critics, normalization references and finite-step checks. FIT fold 2 remains nomination utility. Match batch order and model/critic seeds across arms. Match five source RAW critic-update rounds per model-update attempt, actual full occupation and pair wires, and source fixed linear/MLP32/MLP64 critic banks. Pair and local banks are trained in every continuation arm so compute and exposure match, even when a term is omitted from that arm's model direction. Source optimizer conventions for the critics remain unchanged.

Let theta include only occupation encoder and current occupation head. Let CE be occupation task CE; let Rocc and Rpair be the source RAW entropy-normalized sensitive risk envelope. The proposed displacement is

`p=−eta*(grad_theta CE + lambda*grad_theta(Rocc+Rpair))`.

The sensitive term is minimized risk, so the proposed sensitivity component increases the winning critic's CE locally. No income-local term is optimized because its parameters are fixed. Current own outputs remain in both critics' actual wires.

## Candidate and controls

The substantive candidate projects the entire proposed displacement, including the task update, onto `grad NLL·d≤0`, `grad Brier·d≤0`, then the declared step ball. It checks the actual rounded displacement, and real finite-step FIT-batch NLL/Brier at fractions 1/.5/.25/.125 with 1e-10 metric tolerance. A zero step is feasible. Log every attempted fraction, constraints, active set, acceptance, rejection, conservative solver fallback and rounding stall.

The unchanged RAW incumbent is an exact zero-update alias. Ordinary adversarial occupation continuation uses the same full local+pair proposed update, optimizer, norm bound, critics, attempt count and checkpoint budget, with no NLL/Brier safeguard. The exact local twin keeps the candidate projection/checks and removes only Rpair from the model direction. Task-only continuation keeps the same critic training and exposure, but removes all sensitive terms and applies the bounded occupation task update. No controller, distillation term or attacker expansion is added.

Nominate one common seed-paired configuration per arm using FIT-fold-2 utility and frozen ATT-development recovery. First require all point utility proxies against both source references; among proxy-feasible candidates, minimize mean pair ATT validation envelope, then worst normalized utility violation, summed occupation NLL, epoch and recipe order. If none are proxy-feasible, retain the minimum worst normalized violation as an explicitly infeasible nominee. Match the pool and selection opportunities across arms. Freeze that ranking before fits. FIT proxies are not assessment feasibility.

## Trigger, falsification and outcome

Evidence for a local task conflict consists of independently regenerated, materially nonzero sensitive/task gradients with opposing directions at useful RAW/JOINT states, recurring signs in both seeds, plus actual loss/recovery checkpoint patterns. A cosine alone is insufficient. Prediction leakage can motivate the same pilot only if a credible useful-prediction improvement direction remains; unchanged decision-preserving score recoding cannot remove information.

The intervention explanation is contradicted if constrained updates offer no usable occupation/pair recovery advantage over the matched ordinary continuation, or if they stall/loss-feasible but fail real task usefulness. If an ordinary compression baseline explains the measured gain, credit compression. A mixed or unresolved diagnosis earns an implementation-ready next experiment, not a gratuitous defense fit.

The frozen INNER reporting family must include all 24 original utility clauses, pair benefit≥0.020 over the strongest point-eligible matched control, both local worsening≤0.010 guards, candidate versus ordinary, and candidate versus local twin. Compute the actual multiplicity family; do not inherit K51/K240 blindly. Use paired duplicate-group resampling across seeds and arms with fixed bootstrap seeds, and keep fixed-reader exploratory intervals separate from population guarantees. There is no selection after INNER exposure.

## Existing method relation and exact interfaces

Halfspace projection is established machinery; [Gradient Episodic Memory](https://papers.neurips.cc/paper_files/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html) used a gradient QP to retain prior-task performance. PCRL's own `jcv/project.py` and `pnx/train.py` already contain task-gradient projection and ordinary adversarial controls. This pilot changes the diagnosed occupation-only target, constrains the entire realized displacement, adds both actual NLL and Brier finite-step checks, and compares matched controls. Those engineering changes do not by themselves establish novelty.

The implementation is `occupation_bottleneck.gradient_worker.occupation_refinement_step(model,x,occupation,raw_terms,arm,coefficient,learning_rate,max_norm)`. The coordinator prepares queue-admitted model/FIT batches and calls the source RAW objective to train critics and compute actual local/pair terms before the step. `raw_terms` must have been computed on this very batch before any model update. The helper verifies frozen income tensors. Checkpoint packaging and selection remain the coordinator's responsibility; scientific fits require the campaign's separate pilot science/nomination locks. No pilot numerical run or positive outcome is claimed by this design artifact.
