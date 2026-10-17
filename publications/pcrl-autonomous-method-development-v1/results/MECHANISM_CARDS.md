# Mechanism cards (exploratory; appended as the campaign proceeds)

Each card was written before its family's first fit. Revisions are appended under the family with their reason and date. These are exploratory decisions, not pre-registered trials.

## Family A: Constrained coalition-aware leaf correction (CCLC)

Written 2026-10-17 ~17:05Z, before any fit.

- **Bottleneck addressed.** JOINT_TREE lowered DEV pair recovery to 0.604 / 0.607 (NEURAL_RAW 0.640 / 0.638) but failed the income Brier and accuracy clauses. Its fixed privacy coefficient has no per-task cost control, and its costs are measured on the people it trains on (role drift).
- **Operation.** Leaf-score corrections on the frozen T_TREE (CatBoost) routing for both recipients, trained against online SEX critics on the income, occupation and linked-pair wires (the JOINT_TREE objective). The new part is a utility controller. Each recipient's privacy pressure is throttled once per epoch by explicit budgets measured on a household-separated 20% internal holdout of MODEL_FIT, on which the corrections never train:
  - income NLL, Brier and accuracy;
  - occupation NLL and Brier;
  - each SEX group's occupation Brier.

  Each budget is relative to the teacher on the holdout, set from the published teacher-versus-U0/Ucal headroom on MODEL_SELECT (`STAGE_A_HEADROOM.json`, no new read) plus the cap, minus a safety margin.
- **Why utility and subgroups might survive.** The pair term's protection is kept where it is cheap. The controller backs off whichever recipient is about to exceed its budget, and the per-SEX budgets directly limit the men's burden that broke SCM.
- **Strongest simple control.** TASK_REFIT_80, the same loop on the same 80% with privacy off (matched task-only learner). Also the stored JOINT_TREE checkpoints (fixed coefficient; stopping time only) and the same controller with the local objective (joint-component control, only if a joint claim is intended).
- **Revise or abandon if.** Controlled runs that meet the MODEL_SELECT caps read no lower than the references' MODEL_SELECT pair proxies, or the controller cannot hold MODEL_SELECT utility within the caps.
- **Closest prior.**
  - JOINT_TREE and LOCAL_TREE in this project: the transported L_TREE recipe with a pair term.
  - FAGTB (adversarial gradient tree boosting).
  - Adversarial debiasing (Zhang et al. 2018).
  - Constrained (proxy-)Lagrangian fairness training (Cotter et al. 2019).

## Family B: Coupled shared-statistic release (CSR)

Written 2026-10-17 ~17:05Z, before any fit.

- **Bottleneck addressed.** Coalition synergy. LOCAL_TREE's outputs each read low on DEV (income 0.558 / 0.549, occupation 0.582 / 0.584), yet the linked pair reads 0.657 / 0.649. The two outputs carry complementary SEX evidence.
- **Operation.** The income recipient's probability vector is computed as a fitted, calibrated function of the same person's released occupation probability vector, h(p_occ(x)). The linked pair is then a deterministic function of the occupation output alone. No reader of the pair can exceed the best possible reader of the occupation output; this is an information identity, not a reader property, although fitted pair and occupation-only readers may still differ. h is fitted on MODEL_FIT using household-cross-fitted occupation outputs.
- **Why utility might survive.** The occupation probability vector summarizes the education, hours and class-of-worker evidence that also predicts income. The teacher has income headroom on MODEL_SELECT (NLL 0.0066, Brier 0.0041 against U0).
- **Strongest simple control.** The same occupation output paired with an uncoupled income output (the teacher or LOCAL income): what coupling buys at the pair level. The occupation-only reader gives the information bound.
- **Revise or abandon if.** Income from the occupation vector exceeds the caps on MODEL_SELECT. Then revise to shared-statistic training: the occupation corrections are trained so that h can predict income within its NLL slack. Abandon if that revision cannot hold both tasks within the caps.
- **Closest prior.**
  - Shared-cell merging (SCM: both outputs from one partition, so the pair is redundant with the occupation output).
  - The privacy-funnel single statistic.
  - Multi-task bottleneck representations.

## Family C: SEX-mixture marginalization (MIX)

Written 2026-10-17 ~17:05Z, before any fit.

- **Bottleneck addressed.** The released task probabilities encode the inputs' SEX propensity, because the input-only prediction mixes SEX-specific task patterns with person-specific SEX weights. Coarsening removed this signal expensively and mostly at men's cost.
- **Operation.** Fit SEX-aware task models P(y | x, s) on MODEL_FIT, with SEX as an input at training time only. Release Σ_s P(y | x, s) w_s with fixed population SEX shares w, or with shares pooled to a few levels of the inputs' SEX propensity, never the person's SEX. This removes the pathway by which the inputs act as a SEX proxy inside the task prediction, while keeping the SEX-specific task structure.
- **Why utility and subgroups might survive.** The cost is the KL between the input-only mixture and the marginalized mixture. It is small wherever the two SEX-specific patterns agree, and it is shared by both SEX groups rather than concentrated on one.
- **Strongest simple control.** A SEX-blind model of the same class, capacity and training data (the identical CatBoost fit without SEX), released without marginalization. A non-SEX shrinkage of matched cost isolates "less confident" from "less SEX-structured".
- **Revise or abandon if.** Full marginalization costs more than the headroom plus the caps: then pool the propensity to a few levels. Abandon if the readers are not reduced at a feasible cost, or the reduction is no larger than matched-cost non-SEX shrinkage.
- **Closest prior.**
  - Pope & Sydnor (2011): estimate with the protected characteristic included, then predict with it averaged out.
  - Marginalization of protected attributes in fair prediction.
