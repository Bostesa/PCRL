# CORRECTIONS_V4 — additive

Nothing historical is edited. Study 4 bytes at `69e790af36c5ca53203dab17b757a8e3415ee934`,
v3 bytes at `ce5ba24a0a0ed1848c4029a922ac66fa629d71fc`, and the predecessor studies
(`73903b7f…`, `c37807e4…`, `349efa45…`) are preserved. v3's corrections A1–A17 and B1–B14 remain in
force (`results/pcrl_manuscript_review_v3/CORRECTIONS.md`). Every number below was recomputed from
machine-readable evidence (`STUDY4_VERIFICATION.json`, `SELECTION_AUDIT.json`,
`SELECTION_RESCORING_CHECK.json`).

Terminal 1's own `CORRECTIONS.md` at `4f09e63c6` independently reaches C3, C4, C5, C6 and part of C1.
Where we agree, we say so; C1(b), C2 and C7 are corrections only this review makes.

| id | as written (source) | measured | correction |
|---|---|---|---|
| C1 | "both residence intervals lie entirely above zero and above the .001 reference" (RESEARCH_DECISION §2 Q2; T1 v3 STATUS notes[1]) | C1-vs-L2: +0.00346, family-adjusted [+0.0000764, +0.00684]; candidate-wide [−0.00184, +0.00875] | (a) crosses .001 — noninferiority at .001 not established in either direction (T1 agrees). (b) Under the candidate-wide correction the interval contains zero; "significant" holds only at the family-adjusted level (T1's §6 still says "significant"). |
| C2 | "coalition conditioning genuinely beats strong local controls" with a three-row table (RESEARCH_DECISION §2 Q2) | Q2 given at family-adjusted level while Q3 uses candidate-wide; omitted row dax8_C1_b100 vs dax8_L2_b100 is 0/4 at both levels, residence +0.00002 | Robust vs weak control L1. Vs strength-matched L2: width 16 1/4 adjusted, 0/4 candidate-wide; width 8 0/4. Report both levels and all four rows. |
| C3 | "a directly adversarial refinement of that same channel [J]" (METHOD §0) | 114/114 trajectories initialised from A0 (75 exact, 39 A0+PCA) | Initialised from A0, not J; same general family, not a fine-tune of J. J-initialised fine-tuning untested. |
| C4 | "the joint schema needs 11 dimensions of rank and the channel has 8, so erasure annihilates the channel" (RESEARCH_DECISION §3) | nominal loss 10; width-16 measured loss 10,10,9; width-8 retained rank 0, cross-cov ~1e-35; one RAC1P class with 1 complete case | Result stands as a measurement; the counting argument is withdrawn. |
| C5 | "A0 is very nearly the minimiser of U by construction … every unit of penalty reduction must be paid for" (§4) | distortion 0 at teacher; penalty at step 0 is the trajectory max; all 63 step-0 trajectories moved (median displacement 0.099) | Only the distortion term is minimised at its reference. Unchanged released channels are a selection outcome, not a causal diagnosis. |
| C6 | "The incumbents … were genuinely strong, so refreshing had almost nothing to recover" (§2 Q1) | 84/1350 kept; catch-up 2.1e-5; stress 0/72 | Bounds the declared finite family in its budget. Not that all recoverable information was found; not a universal result about ensembles. |
| C7 | "At beta 0.1 and 0.3 … 39 of 72 main arms are bitwise identical" (§4) | 18 + 18 + 3 at beta 1.0 | 39/72 correct; three are at beta 1.0. |
| C8 | Amendment 4 scope change | identical-to-A0 arm: 0.0155 artefact under old scope, exact equality under new | Evaluation correction; never a protection gain. Endorsed. |
| C9 | "Coalition conditioning is buying protection with utility" / any dominance reading | lower recovery with lower residence | A trade-off, not dominance. No equivalence to J or baselines follows from nonsignificance. |
| C10 | 2017 "no intervals" | EXPLORATORY_2017.md states this correctly | Endorsed: conditional intervals on a spent partition are permitted and descriptive; they do not restore confirmatory status or cover adaptive selection. The manuscript never says they are forbidden. |
| C11 | "66 of the 69 declared interfaces" | JSON panel lists 26 conditions incl. H/A0/J | Correct under the definition 23 newly transported x 3; stated explicitly. |
| C12 | v3 manuscript: Study 4 "still fitting", Appendix G empty | Study 4 complete | Superseded; v3 text preserved on its branch. |
