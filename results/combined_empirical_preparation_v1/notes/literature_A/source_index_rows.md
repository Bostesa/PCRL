# Source index: literature statements relied on (literature_A)

**Reading status codes:**
- **F** — full text read, including the appendix.
- **M** — main text read; no appendix exists, or the appendix was not available.
- **S** — skimmed only.
- **X** — not read.

All PDFs were fetched on 2026-10-01.

| # | Statement | Primary source | Version | Status | Exact location |
|---|---|---|---|---|---|
| 1 | After rank-1 linear erasure, RBF-SVM and a ReLU MLP (128 units) still predict gender above 90% accuracy. | Ravfogel et al., ICML 2022 (R-LACE) | PMLR v162 PDF | F | p.6, §5.1, paragraph "Importantly, as expected…" |
| 2 | R-LACE "is not expected to be robust to nonlinear adversaries". | R-LACE | PMLR PDF | F | p.13, App. B.1 |
| 3 | Gradient-reversal adversaries reach near-random accuracy during training, but fresh test-time adversaries recover gender: 99.57 (MLP adversary) and 99.23 (linear adversary). | R-LACE | PMLR PDF | F | p.7, Table 2 and footnote 9 |
| 4 | The game is convex–convex, not convex–concave as an earlier version claimed; the formulation was reordered to max–min. | R-LACE | arXiv 2201.12091v8 | S (correction passage read) | §2.3, footnote 4 |
| 5 | The R-LACE adversary checkpoint is chosen by the highest classification loss of a freshly trained linear classifier on dev. | R-LACE | PMLR PDF | F | App. B.4 (p.15), B.6, B.7 (p.16) |
| 6 | Kernel erasure protects partly against the same kernel (0.49–0.75 accuracy) but does not transfer: an MLP recovers 0.97 for every neutralizing kernel. | Ravfogel, Vargas, Goldberg, Cotterell, EMNLP 2022 | Anthology PDF | F | p.6040, Table 1; p.6041, Table 2, §5.2; App. A.7, Table 8 |
| 7 | Exhaustive erasure against diverse non-linear adversaries "remains an open problem". | EMNLP 2022 kernel | Anthology PDF | F | p.6042, §8 |
| 8 | The Anthology title is "Adversarial Concept Erasure in Kernel Space"; the PDF title is "Kernelized Concept Erasure". | EMNLP 2022 kernel | Anthology page + PDF | F | Landing page `<title>`; PDF p.6034 |
| 9 | Probing shows extractability, not use. Amnesic probing measures the behavioural effect of INLP removal, with Rand and Selectivity controls. | Elazar et al., TACL 2021 | TACL PDF | F | p.160–162, §§1, 2.2–2.3 |
| 10 | Probe accuracy does not correlate with task importance. Phrase markers are probed well yet unused (+0.21 / +0.32 LM points). | Elazar et al. | TACL PDF | F | p.164, Table 1 and text |
| 11 | A property removed at layer i is partly re-recoverable at later layers. | Elazar et al. | TACL PDF | F | p.167–168, §7.1, Fig. 3 |
| 12 | Only linear information is removed; causal interpretations need caution. | Elazar et al. | TACL PDF | F | p.170, §9 |
| 13 | Censored last-layer representations still leak, e.g., UTKFace race: 62.18 (base), 53.28 (adversarial), 53.30 (information-theoretic), vs 42.52 random. | Song & Shmatikov, ICLR 2020 | arXiv v3 | M | p.5, Table 2 |
| 14 | Stronger censoring does not help and can raise inference ("censoring defeats itself"). | Song & Shmatikov | arXiv v3 | M | p.5–6, Fig. 2 and text |
| 15 | Task–attribute pairs have low Cramér's V (0.033–0.149). | Song & Shmatikov | arXiv v3 | M | p.4, Table 1 |
| 16 | Censoring one layer leaves other layers re-purposable. | Song & Shmatikov | arXiv v3 | M | p.7–8, Table 5 and CKA text |
| 17 | LPP (Def. 2): sup over all attributes S with S−(X,Y)−Z of I_∞(S;Z\|Y) ≤ γ. The adversary sees the true label Y. | Stadler et al., ICML 2024 | PMLR v235 PDF | F | p.4–5, Def. 2, eqs 7–9 |
| 18 | Thm 2: under Assumption A (strictly positive posterior; finite discrete spaces), γ-LPP and I_α(Y;Z) > γ cannot both hold. | Stadler et al. | PMLR PDF | F | p.2 (Assumption A); p.5 (Thm 2); p.13–14 (proof) |
| 19 | The supremum is attained by a constructed, maximally revealing attribute S\*. | Stadler et al. | PMLR PDF | F | p.12, Def. 4, Prop. 3 |
| 20 | Restricting leakage for a single attribute is sometimes possible, but the representation does not satisfy LPP. Restricting the attribute class might ease the trade-off. | Stadler et al. | PMLR PDF | F | p.9, §4.2 "Takeaways" and §5 Limitations |
| 21 | Fundamental leakage is measured empirically with a label-only adversary across 12 × 12 LFWA+ pairs. | Stadler et al. | PMLR PDF | F | p.7, §4.1, Fig. 3 |
| 22 | Every task has at least one attribute with ΔAdv > 0; censoring has a "whack-a-mole" effect. | Stadler et al. | PMLR PDF | F | p.8, Fig. 4 and text |
| 23 | Collusion model: one actor may access all releases up to the most recent one. Constraints: individual I(R̂_k;X) ≤ ε_k and collusion I(R̂_k,R̂^{k−1};X) ≤ δ_k. | Taylor, Vippathalla, Coon | arXiv 2601.21859v2 | F | p.2 (§I attack model); p.3, eq. 1 |
| 24 | Finite alphabets with a known joint pmf; Blahut–Arimoto optimisation. Optimal for the distortion utility (convex), locally optimal for MI. | Taylor et al. | arXiv v2 | F | §§III-B, IV-D (p.4–7); Alg. 3 (p.9) |
| 25 | Real-data experiment on discretised Adult (\|X\| = 32). The adaptive release reduces cumulative leakage without utility loss. | Taylor et al. | arXiv v2 | F | §VII, Figs 6–8 (p.10–11) |
| 26 | Subset-collusion constraints are proposed as an extension. | Taylor et al. | arXiv v2 | F | p.13, §IX |
| 27 | The certified quantity Δ\* = sup over all binary tests = 1 − 2·BER\*. | Gitiaux & Rangwala, AISTATS 2021 | PMLR v130 main + supplement | F | p.3, Def. 2.2; supplement Lemma 1.1 |
| 28 | No finite-sample certificate exists for encoders with infinite χ²-MI, e.g., injective deterministic encoders. | Gitiaux & Rangwala | PMLR | F | p.4, Thm 2.1, Cor. 2.1; supplement §§1.1–1.3 |
| 29 | AGWN bound: E[Δ\* − Δ_n] ≤ 2·exp(‖t‖²_∞/2σ²)·(n₀^{−1/2} + n₁^{−1/2}). | Gitiaux & Rangwala | PMLR | F | p.5, Thm 3.1; supplement §1.6 |
| 30 | On the Swiss roll, 18.2% of near-zero adversarial certificates (≤ 0.1) still allow processor disparity > 0.3. | Gitiaux & Rangwala | PMLR | F | p.7, §5 |
| 31 | σ values are 0.02–0.05 and the encoder output is tanh-bounded, so the bound constant is vacuous. | Gitiaux & Rangwala (values); our arithmetic | supplement Tables 1–2 | F | supplement p.9; the inference is ours |
| 32 | FairNVT is published in TMLR (08/2026) and was reviewed on OpenReview (forum rzm6gZrYgl). | Tang et al. | arXiv 2604.16780v2 PDF | F | p.1 header and OpenReview line (every page header) |
| 33 | v1 was presented at the ICLR 2026 AFAA Workshop. | arXiv v1 comment; author page | v1 abstract page; qiaoyuet.github.io (updated 2026-08-14) | S | arXiv comments field; "Selected Projects" |
| 34 | Attacker accuracy (Smiling/Male) on: e_t 68.3, e_f 52.1, Att10 52.4, Avg5 63.6, Avg50 92.5. | Tang et al. | arXiv v2 | F | p.11, Table 3, §4.2 |
| 35 | No guarantee is given; Props B.1–B.2 are idealised intuitions only. | Tang et al. | arXiv v2 | F | p.17–18, App. B |
| 36 | No FairNVT code was found (no paper link, no author-page link, GitHub search returned 0). | Paper + web | 2026-10-01 | — | Negative search result |

**Manuscript-side statements checked (our own sources):**

| # | Statement | Source | Location |
|---|---|---|---|
| M1 | The AAAI paper cites only the EMNLP kernel paper (`ravfogel2022kernelized`). | AAAI27 zip (Aug 30 = Sep 17, identical) references.bib | references.bib lines 100–106; paper.tex:152 |
| M2 | The AAAI paper cites Elazar for "what a probe finds depends on the probe". | AAAI paper.tex | :156–158 |
| M3 | The AAAI paper says of Stadler: "We contribute its measurement" and "approach the leakage floor established by". | AAAI paper.tex | :171–175; :474 |
| M4 | The AAAI references.bib has no entry for Gitiaux, FairNVT, R-LACE or Taylor. | AAAI references.bib | full file grep |
| M5 | The PCRL bib entry `ravfogel2022rlace` lists the wrong authors (Belrose and Gonen added); `ravfogel2022linear` is correct. | NeurIPS (21) bibliography.bib | lines 25–30; lines 303–308 |
| M6 | The PCRL paper describes R-LACE as a "subsequent refinement … oblique projections". | NeurIPS paper-body/S2-related-work.tex | :4 |
| M7 | The PCRL paper does not cite Stadler, Elazar, the kernel paper, Gitiaux, FairNVT or Taylor. | NeurIPS bibliography.bib | full file grep |
| M8 | The PCRL CelebA cell is the Smiling task with Male and Young disallowed. | NeurIPS S5-experiments.tex | :191 |
