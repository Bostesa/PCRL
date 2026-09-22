# Corrections record (Terminal 3)

Additive only. No historical report was edited. Each item states whether a **number** changed, only an
**interpretation** changed, or a **claim was withdrawn**. Terminal 2's C1–C9
(`results/pcrl_submission_review_v1/CORRECTIONS_FINAL.md` @ 30a6fd19e) are verified, not renumbered. The
items below are new, or they sharpen an existing item, which is cited.

| # | Item | Relation to T2's list | Kind | Evidence |
|---|---|---|---|---|
| T3-1 | The selected release's code is residence-supervised, not "label-free" | new | claim withdrawn (method description) | encoding.py @ f4bdf4cd |
| T3-2 | The selected release imposes no coalition constraint; its fitted AB CMI exceeds δ | new (sharpens main.tex:300) | interpretation | MATH_REPLAY_FINAL_anchor{0,1,2} |
| T3-3 | The conditioning partition has 2 cells (A) and 4 cells (AB) | new | interpretation (scope) | ServicePartitions.fit |
| T3-4 | "floor on what is recoverable" | new | claim withdrawn; replaced | fixture F9 |
| T3-5 | "API now refuses to run" holds on research branches only; public main still ships the invalid theorem | sharpens C1 | interpretation; code patch 0001 offered | origin/main 55e4cb1 |
| T3-6 | Valid replacement statement: any loss convex in the affine prediction, not only squared loss | sharpens C1 | interpretation (broadened, correctly) | LEACE Thm 3.1 |
| T3-7 | The convex-combination identity is the definition of variance-weighted R², not an original result | new | claim withdrawn (novelty) | THEORY_REPAIRS T2 |
| T3-8 | DA is max one-vs-rest (in-sample), a lower bound on the worst linear direction | new | interpretation | fixture F3; certificates.py |
| T3-9 | Identity "validated, max residual 0.0021": two Adult cells carry a float32 artefact of the historical code; Adult mean aggregate 0.0080 → 0.0082 under float64 reconstruction | new | number (interpretive reconstruction; stored numbers untouched) | DOMINANT_AXIS_REPLAY.json; HISTORICAL_PRECISION_DIAGNOSTIC.json |
| T3-10 | The encoder line's backbone was a frozen random MLP, not pretrained as the NeurIPS text says | new | claim withdrawn (original paper's method description) | run_v2_dataset.py @ dbe0fdc; lora.py:182 |
| T3-11 | NeurIPS P4 (LoRA erasure floor) is misapplied to multi-layer adapters | new | claim narrowed | fixture F12 |
| T3-12 | NeurIPS P6 (approximate composition) is correct and tight, but elementary; its ridge form needs an unregularised per-purpose premise | extends C4 | interpretation; the lemma is recommended for the paper | F11; THEORY_REPAIRS T3 |
| T3-13 | Rassouli–Gündüz's LP is for MI utility; the convexity is Calmon & Fawaz 2012 or standard; U-FaTE conditions on the label; Sankar and Liao are not the nearest multi-recipient prior art (that is Taylor et al. 2026) | new | attribution corrected | PRIOR_ART_MATRIX M1–M5 |
| T3-14 | The coalition constraint structure has a close antecedent (Taylor et al. 2026 collusion constraints); the novelty claim narrows to the setting | new | claim narrowed | PRIOR_ART_MATRIX |
| T3-15 | Rebuttal "54/60 → 60/60" versus the paper's "56/60": different checkpoint rules | new | interpretation | BRANCH_DISCOVERY amendment |
| T3-16 | REVIEW_TO_EVIDENCE misattributes `results/rebuttal/**` to the other paper and lists the other paper's runs as Appendix A | new (review-file error, not paper) | interpretation | git ls-tree @ 30a6fd19e |
| T3-17 | Correction 4 should name the skipped certificate (Sadeghi–Boddeti). The original P1 (Zhao–Gordon) is valid and prior | sharpens C5 | interpretation | THEORY_REPAIRS T5 |

Verified by T3 as stated (no change): C2 (shared union; TABLE.md source-read), C4 (exact-zero
composition; fixture F4), C6 (tuning disclosure; source-read), C7.

Not re-examined by T3 (taken from T2): C3 (subsumption withdrawn), C8, C9.
