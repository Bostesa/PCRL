# Review concerns → response → verified support

**Status of the review record.** No genuine venue review text is in any repository ref (BRANCH_DISCOVERY
§7). The concerns below come from two sources:
- (V) concerns attributed to the actual reviewers and area chair in the committed registration
  `results/rebuttal/ablations_facct/PREDICTIONS.md` @ ecaeba46fcb921802087e792c5dbac2073e25347. These are
  project paraphrases, not review text.
- (P) concerns from project planning files, whose origin (internal or simulated critique) is not
  established.

Nothing here reconstructs review text. The reviewer-handle mapping is in the private handoff. Terminal 2
owns the venue-specific appendix.

Status vocabulary: **addressed**, **partly addressed**, **superseded by a narrower claim**, **unresolved**.

| # | Concern | Source | Response and evidence (verified by T3 unless marked) | Status | Paper action |
|---|---|---|---|---|---|
| R1 | Is LoRA needed, or would a plain per-purpose linear adapter do? | V | Only a parameter-count/latency comparison against full per-purpose encoders exists (NeurIPS compute-cost appendix). The linear-adapter ablation was registered (prediction: they match, ~70%) with code and a 5-epoch smoke, but **never run** | unresolved | Do not claim LoRA necessity or advantage. State the missing comparison. Not a new training assignment here |
| R2 | Is the frozen LEACE erase layer necessary? | V | Ablation 2 registered (prediction: ≤30/60 strict without it), **never run**. The confounded cross-purpose comparison (the erase layer and the constraint added together; 22/33 → 8/33 flags, recounted by the branch-discovery pass) cannot separate the two | unresolved | Say the necessity is untested |
| R3 | Hyperparameters tuned on the evaluation grid | V | Disclosed: λ_min=5 and skip-warmup were developed after examining an earlier evaluation on the same 60-cell grid (PAPER_PASTE.md:96-126 @ 0176f149; C34). Held-out selection (ablation 3) registered, never run | partly addressed (disclosed, not repaired) | Keep the "same-grid post hoc development evidence" wording (main.tex App. A) |
| R4 | Efficacy: does the method meet its constraint? | V/P | Final iterate: 56/60 strict R²≤0.05, 7/60 also pass the health check (NeurIPS headline; the 56 recounted from `dominant_axis_audit.json` final.pt). The best-checkpoint per-seed JSON gives 54/60. Erase-layer rebuttal arm: 60/60 strict, **0/60** clean (provisional: raw per-seed results uncommitted; baseline row uses the best-checkpoint rule, not the paper's final-iterate rule) | partly addressed | Report strict and clean separately, at one checkpoint rule |
| R5 | Compliance via collapse versus task utility | P | VICReg sweep (39c5a84): strict pass holds, per_dim_std unchanged, 0/18 cross 0.5 (recount by the branch-discovery pass). Backbone diagnostic: backbone std 0.18–0.29 in every cell. **Correction:** the backbone is a seeded random MLP (E5), so the "architectural floor" is a property of a random frozen feature map, not of a pretrained encoder | superseded by a narrower claim | Describe the backbone correctly. Do not call the floor a property of the method |
| R6 | Per-purpose bounds may not compose (combined access) | P | Exact-zero composition holds; nonlinear recovery can still be exact (F4). Approximate composition is bounded by Σ R²/λ_min(R) and the bound is tight (F11; NeurIPS P6 proof checked). Empirically, 26/33 (absolute) and 22/33 (incremental) triples flag concatenation recovery (NeurIPS; not recomputed by T3) | addressed (as a bound plus a negative empirical fact) | Composition paragraph with the lemma (patch hunk `compose`) |
| R7 | A shared eraser for conflicting purposes | P | The shared union destroys both tasks; per-purpose LEACE retains them; the restricted PCRL arm ties it (TABLE.md @ 0176f149) | superseded by a narrower claim | Correction C2 already in the paper |
| R8 | LEACE/INLP/LAFTR/SPLINCE baselines and prior work | P | Per-purpose LEACE ties the restricted arm; SPLINCE 60/60 strict on the same backbone (NeurIPS appendix); hard-R² LAFTR results uncommitted (provisional) | partly addressed | Cite per-purpose LEACE as the comparator that ties |
| R9 | Zero R² was claimed to bound classification accuracy | P | Refuted exactly (F1). Retired on research branches; **still active on public main** (patch 0001) | addressed in the paper; code fix pending | Patch 0001 or qualify main.tex:149 |
| R10 | Rank floor / why rank 24 on Diabetes | P | P4 is misapplied to multi-layer LoRA (F12). The rank-8 erase-layer ablation passes 18/18 strict but loses 24.7 pp on medication_change (recount by the branch-discovery pass; 98c4d3c = 2787c83) | superseded by a narrower claim | Corrections item (patch hunk `corr4`) |

## Items from the current (successor) paper's own review, for completeness

- Supervision asymmetry (Q used residence supervision; J did not): stated in main.tex:293-297 ✔.
- Development-data reuse: stated in main.tex:314-316 ✔.
- D17 control visible: main.tex:261-268 ✔.
- Coalition scope of the selected release: **was implicit**; patch hunk `convex` makes it explicit (E2).
- Method-object description ("label-free"): **wrong**; patch hunks `construct` and `baselines` (E1).
