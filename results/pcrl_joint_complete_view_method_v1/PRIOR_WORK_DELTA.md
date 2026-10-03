# Prior-work delta (bounded review, before any real fit)

Full review with section and equation citations: `review/PRIOR_WORK_DELTA_REVIEW.md`. Bibliography: `review/references_jcv.bib`. The review took about 75 minutes against a 90-minute cap, read primary sources, and searched within stated bounds.

## Established ingredients vs what this integration tests

| Ingredient | Established source | Specific to this study |
|---|---|---|
| Adversarial critic surrogate `H(S) − min CE` | Song et al. 2019, constraint C2 (Eq. 6–10); adversarial representation learning (Edwards & Storkey; LAFTR) | A small bank maximised per view, with a constant prior; normalised by one common `H_fit(S)` |
| Weighted penalty update (arm JP) | The fixed-multiplier Lagrangian: LAFTR Eq. 2; Song et al. Eq. 11 | Used as the single matched ablation |
| Projection `min ‖d−p‖²` s.t. `a_j·d ≤ 0` | GEM (Lopez-Paz & Ranzato 2017, Eq. 8); A-GEM (Chaudhry et al. 2019, Eq. 11); PCGrad pairwise step; Rosen active sets | The privacy gradient is the proposal; ε-active task-loss *budget* guards are the constraints. With separate encoders this reduces exactly to A-GEM per encoder. **An adaptation, not a new optimiser.** |
| Linear erasure inside training | LEACE §7 names it as future work; PCRL's erase-layer pilot ran it, with structural linear compliance but higher nonlinear recovery on 48/60 cells | Per-purpose official LEACE, refitted at epoch boundaries and refitted finally |
| Coalition term on concatenated recipient releases | PCRL's original cross-purpose constraint (linear R² on `h_concat`) | Nonlinear critics on complete views (features plus centred logits) |
| Collusion-aware sequential release | Taylor, Vippathalla & Coon 2026: online, finite alphabets, known pmf, protects the whole record X, per-step optimal | Not implemented. S12/S21 are neural sequential heuristics with both future tasks disclosed to every arm. They are **not** Taylor's algorithm and inherit none of its optimality. |
| Fair restricted encoder + certificate | FARE (Jovanović et al. 2023) | Applied natively to the permitted raw inputs with the admitted grid. Two trees jointly form a restricted encoder with at most k1·k2 cells, so FARE's own certificate applies to the coalition (looser). |

## What an empirical win would and would not mean

- **J beats L.** Coalition coupling helped under fixed information and compute. That is a measured property of this training on reused rows; it is not novelty.
- **J beats C\*.** It was competitive with the strongest feasible control chosen on validation. It is still not a frontier-dominance claim, and still not fresh confirmation.
- **The advantage over S12/S21** partly reflects foreknowledge of both purposes. That is an information advantage, not an algorithm-only advantage. **Only J vs L holds information fixed.**
- **Novelty is low in any case.** No learned-representation method was found that jointly trains several recipient encoders against a coalition critic. That is a bounded-search gap, not evidence of priority.
