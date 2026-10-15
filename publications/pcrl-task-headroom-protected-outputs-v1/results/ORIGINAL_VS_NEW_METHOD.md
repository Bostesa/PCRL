# Original PCRL release vs this campaign's protected tree outputs

| aspect | original / predecessor line | this campaign |
|---|---|---|
| Task learner | neural encoders (83→64→64→16 per task) with logistic heads (U0/Ucal); privacy via adversarial encoder training (RAW_JOINT), erasure, codes or small adapters over frozen features | CatBoost oblivious trees on the permitted raw83 inputs (depth 4, lr 0.05, ≤800 trees, l2 3; tree count by internal early stopping, then a whole-pool refit) |
| What is released | full wires: representation z16 + logits + probabilities + decision (adapters); or tokens/codes (score-code studies) | only each recipient's own probability vector and decision; the pair adversary gets the aligned concatenation |
| Family containing the useful model | adapters start at RAW_PROJ; codes distil into a finite latent; neither contains a stronger task model | the leaf-score family contains the fitted teacher exactly (θ = 0); its corrections can move any leaf's score |
| Privacy mechanism | adversarial encoder or adapter training with KL trust regions and budgets (adapters) | f_r = teacher_r + Σ_t θ[r, t, leaf_t]/T with fixed routing; AdamW on θ; task CE + λ × entropy-normalized online-critic risk on probability-only wires; no trust region, no decision preservation |
| Utility reference for selection | U0/Ucal on FIT2, which those references had trained on | an equally exposed neural reference N_EQ on FIT2; U0/Ucal kept for the final INNER requirement only |
| Adversary | fresh complete-view readers, envelope maximum | fresh output-only readers (linear, MLP, boosted trees, decision-only, confidence-only); the per-view winner is chosen on attacker validation, never on INNER; the pair bank includes readers that ignore either recipient |
| Result vs RAW_PROJ | two-recipient adapter: 0.0036 [−0.0025, 0.0097] under the full-wire contract and its own reader bank | L_TREE: **0.0505 [0.0257, 0.0754] (established)** under the probability+decision contract. The two numbers use different release contracts and reader banks and are **not comparable**. |
| Result vs historical codes | — | 0.014–0.018, unresolved |
| Joint vs local | tied | joint not established: J vs L coalition unresolved, and J leaks more occupation (established) |

**What is not new.** Gradient-boosted trees, adversarial training against attribute critics, purpose-specific outputs, sequential training and leaf-value refitting are all existing ideas. A positive margin here is evidence that an existing training approach, applied to a stronger task learner, can meet these service requirements against the incumbent. It is not evidence of a new algorithm. Novelty of joint design was not supported.

**What the result does not cover.** Reusable feature services, arbitrary sensitive attributes, adversaries with model or leaf access, repeated queries, imperfect linkage and other populations.
