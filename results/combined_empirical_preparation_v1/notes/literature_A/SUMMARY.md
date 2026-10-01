# Literature A — summary (saved by the owner from the role's final report; the role's own write of this file was refused by the harness)

Reading status: all eight primary texts read in full (incl. appendices/supplements where present).
Gaps: Song & Shmatikov read as arXiv v3 (no appendix; OpenReview camera-ready not compared);
Taylor et al. v2 read in full, v1 not read; FairNVT v2 full, v1 skimmed, TMLR/OpenReview record
returned 403 — status rests on the v2 PDF header, arXiv comment and author page.

## EARLY FLAGS

1. **FairNVT** — TMLR 08/2026 per v2 PDF header (v2 title "Fair Classification via Noise Injection
   in ViTs"); v1 (Apr 2026, different title) at the ICLR 2026 AFAA workshop; no evidence of ICLR main
   conference; no code found. Not cited by either manuscript. Design is closely related to the AAAI
   isolate-then-noise channel (split sensitive adapter + clipped Gaussian noise); its v2 Table 3
   (17 Aug 2026) already reports that averaging 5/50 noise draws recovers the attribute (92.5%).
   Uses CelebA cells (Smiling/Male; Young) on a frozen ViT with adapters + HSIC — overlaps PCRL CelebA.
2. **Gitiaux & Rangwala (AGWN)** — a missing certified baseline for the AAAI paper: bounded encoder +
   Gaussian noise (same family as AAAI App. Prop. 3 channel); their Thm 2.1 (deterministic injective
   encoders cannot be certified from finite samples) is close to AAAI Prop. 2. AAAI's certified-methods
   list (paper.tex:709) omits it. Certificate: 1−2·BER* for binary S against every test on a single
   noise draw; not repeated draws or side information. Role's arithmetic: constant vacuous at the
   AAAI paper's own σ (role inference, not independently checked).
3. **Stadler et al. 2024** — AAAI "We contribute its measurement" (paper.tex:171–175) overclaims:
   Stadler §4.1/Fig. 3 already measures label-only "fundamental leakage", i.e. what the AAAI
   label-coupling predictor computes. "Leakage floor established by Stadler" (paper.tex:474) is
   imprecise: Thm 2 quantifies over all attributes (incl. a constructed S*), assumes finite discrete
   spaces and strictly positive posterior, conditions on true Y. It implies nothing for a finite
   declared S — neither prohibits restricted policies nor is "evaded" by them. PCRL does not cite it.
4. **PCRL bibliography errors** — `ravfogel2022rlace` has wrong authors (adds Belrose, Gonen) and
   duplicates `ravfogel2022linear`; S2-related-work.tex:4 calls R-LACE a subsequent oblique-projection
   refinement, but R-LACE predates LEACE and uses orthogonal projections.
5. **Elazar et al. 2021 cited for the wrong point** in AAAI (paper.tex:156–158); its point is
   recoverability ≠ causal use. AAAI title suggests "use" but the paper measures recoverability.
6. **"Ravfogel et al. 2022" is ambiguous** — ICML R-LACE is the literal match for "nonlinear attackers
   recover what linear erasure removes" (§5.1 RBF-SVM/MLP > 90%; App. B.1; Table 2 fn 9 fresh
   adversaries 99%); EMNLP kernel paper matches "nonlinear erasure does not transfer" (MLP 0.97).
   AAAI cites only EMNLP; PCRL only ICML. Cite both. R-LACE arXiv v8 corrects the game to convex–convex.
7. **Taylor, Vippathalla, Coon** — prior PCRL notes mostly agree; corrections: v2 §VII has a
   real-data experiment (discretised Adult, |X|=32); §IX proposes subset-coalition constraints.
   Finite alphabet, known pmf, protects whole X via MI; no declared S.
8. **FairNVT attacker under-powered (role inference)** — attacker on e_f=[e_t, e_s^noised] reads
   52.1% but on e_t alone 68.3%: a superset view reading lower than its subset is a nested-predictor
   failure — exactly the slate defect our protocol must prevent.

## What these papers already establish
Linear certificates do not bound nonlinear recovery; nonlinear erasure does not transfer across
attacker families; training-time adversaries are fooled while fresh attackers succeed;
recoverability ≠ use; universal least privilege is incompatible with utility and label-only leakage
is measurable pre-training; certification needs smoothing/noise; isolated-subspace noise is undone by
repeated draws; coalition budgets are formalised for finite alphabets.

## Open relative to these eight papers only
Cross-family comparison of each method's own stated test vs defense-aware recipients under one
held-out utility protocol; S-recovery from trained-model outputs measured against the label-only
baseline; coalition / combined-release evaluation for learned representations with a declared S;
statistics beyond 3–5 seeds.
