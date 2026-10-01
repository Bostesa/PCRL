# Source index

Compiled 2026-10-01 (execution date). Every important literature statement in this package maps to a row
below. Rows give the primary source, the version read, the reading status, and the exact supporting
location. Repository and manuscript sources are pinned in the last section.

What this index does not cover:
- Statements known only from the meeting summary are flagged as such in REVIEWER_RESPONSE_MATRIX.csv.
- Official review text was not available.
- Search method, inclusion rules and coverage gaps for the newer literature are in
  `notes/literature_B/SEARCH_LOG.md`.
- Per-paper structured records for the reviewer-cited and mandatory papers are in `notes/literature_A/*.md`.

Publication-status cautions:
- **FairNVT (Tang et al.)** = TMLR 2026 per the v2 PDF header. Its v1 appeared at the ICLR 2026 AFAA
  workshop. There is **no evidence of ICLR main conference**. The OpenReview/TMLR record returned 403.
- **Taylor, Vippathalla & Coon** = arXiv 2601.21859 v2, with no venue.
- **Tian et al.** = arXiv page states accepted to IEEE TDSC (not independently confirmed).
- Venues for Aalmoes (WISE), Johansson (LREC-COLING) and the colluding-adversaries SoK (USENIX 2026) come
  from Semantic Scholar or PDF headers only.

## A. Reviewer-cited and mandatory primary sources (literature_A)

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

## B. Newer empirical analyses (literature_B)

| Statement | Source | Version | Reading status | Location |
|---|---|---|---|---|
| Dataset leakage = attacker accuracy predicting gender from ground-truth task labels; model leakage = attacker accuracy from the model's task outputs; bias amplification = model leakage minus dataset leakage at the same F1 (labels randomly flipped to match F1). | Wang et al., "Balanced Datasets Are Not Enough" | ICCV 2019 / arXiv 1811.08489 | Read §3–5 | §3 "Leakage and Amplification"; "Computing Leakage" |
| Leakage estimates vary by <2 points across attacker architectures/training sizes except a 1-layer attacker. | Wang et al. 2019 | same | Read | §4 "Attacker Learning is Robust", Table 2 |
| Adversarial removal at conv5 lowers COCO model leakage 70.46→64.92 (Δ 9.93→4.57) with F1 53.75→52.54. | Wang et al. 2019 | same | Read | Table 4 |
| After RLACE guarding (linear probes ≤2% above majority), an adversarially constructed multiclass softmax classifier with 4–8 entries "perfectly recover[s]" gender; honestly trained binary profession classifiers leak much less. | Ravfogel, Goldberg, Cotterell, Log-linear Guardedness | ACL 2023 / arXiv 2210.10012 | Read §5, App. A.3 | §5.2–5.4, Figs. 2–3 |
| LEACE paper: "The softmax probabilities of a multiclass logistic regression classifier can leak the removed information if another classifier is stacked on top of it"; nonlinear erasure conjectured intractable without the data-generating process. | Belrose et al., LEACE | NeurIPS 2023 / arXiv 2306.03819 | Read §5, §7 | footnote 5; §7 |
| LEACE Bios evaluation used logistic-regression probes; profession accuracy 77.3% vs 79.3%; TPR gap 0.198→0.084. | Belrose et al. 2023 | same | Read | §5.1–5.2 |
| INLP drops linear-SVM gender accuracy 100%→49.3% while a 1-layer ReLU MLP recovers 85.0%; RBF-SVM is at chance. | Ravfogel et al., INLP | ACL 2020 / arXiv 2004.07667 | Read §6.1 | §6.1.1 and footnote 3 |
| Out-of-family adversaries recover attributes from adversarial FRL: e.g., LAFTR 72.05% (in-family) vs 84.58% (out-of-family); MaxEnt-ARL 50.00% vs 85.18%. | Balunović, Ruoss, Vechev, FNF | ICLR 2022 / arXiv 2106.05937 | Read §3, §6 | Table 1 |
| Standard-scaling the representation lets a model of similar or lower complexity than the training adversary recover the sensitive attribute from adversarial FRL; common protocol uses max parity over 5 downstream runs. | Gupta et al., FCRL | AAAI 2021 / arXiv 2101.04108 | Read experiments | Experiments setup and discussion paragraphs; Fig. 2 |
| AutoML attackers recover S well above chance from deterministic FRL methods (DebiasClassifier, NVP, DDC, LFR) at many/all γ, even where allocations are fair; stochastic/quantized methods approach chance at high γ. | Cerrato et al., 10 Years of Fair Representations | arXiv 2407.03834 v1 (preprint, under review) | Read §4 | §4.4, Figs. 3–4 |
| In a common benchmark, bias-mitigation models often had MORE sensitive information in latents than baselines despite fairer outputs; results reported as best value per metric over hyperparameters on the test set. | Reddy et al. | NeurIPS 2021 D&B | Read §4–5, App. D | §4 reporting paragraph; §5.1 |
| Representation leakage (fixed MLP attacker) does not track empirical (TPR-gap) fairness across debiasing methods. | Shen et al. | Findings AACL-IJCNLP 2022 | Read §4 | §4.3.3, §4.4.3; Tables 1–2 |
| Evaluating FRL only on the proxy task overestimates fairness; on weakly correlated transfer tasks DP can exceed the unfair baseline. | Pouget et al., Back to the Drawing Board | arXiv 2405.18161 v1 | Read §5 | §5.1, Fig. 2 |
| FRL baselines' training-estimated DP bounds are violated on test (incl. an adversarial task predicting S) with high probability; FRG and FARE keep ΔDP low; 20 resampled-training trials. | Luo et al., FRG | NeurIPS 2025 / arXiv 2510.21017 | Read §6 | §6.1–6.2, Figs. 2–4 |
| FARE certificate (95% confidence DP upper bound 39.1%) lies above the DP of 24 diverse downstream classifiers for a representative point (empirical DP 26.2%). | Jovanović et al., FARE | ICML 2023 / arXiv 2210.07213 | Read §6–7 | "Certificate validity", Fig. 6 |
| Projection-based removal fitted on a dataset makes nearest neighbours tend to have the opposite label; anti-clustering recovers the original grouping, even for random data when d≫n; avoided if the projection is fit on held-aside data. | Johansson, "What Happens to a Dataset Transformed by a Projection-based Concept Removal Method?" | arXiv 2403.16142 v1 (S2: LREC-COLING 2024) | Read §4–6 | §4.4–4.5, Figs. 3–4; §6 |
| Diagnostic-classifier leakage after adversarial removal is sample-specific and does not generalize to new samples/domains. | Barrett et al. | EMNLP-IJCNLP 2019 | Read | §2–4, Table 2 |
| Debiased embeddings: k-means clusters align with gender at 92.5% (Hard-Debiased) / 85.6% (GN-GloVe); RBF-SVM generalizes to held-out biased words at 88.88% / 96.53%. | Gonen & Goldberg | NAACL 2019 | Read | §4 |
| Probe-based removal (INLP, adversarial) can fail to remove the concept and destroy task features even when the concept is fully separable. | Kumar, Tan, Sharma | NeurIPS 2022 / arXiv 2207.04153 | Read §4 | §4.2–4.3 |
| Group-fair classifiers (EGD, AdvDebias) reduce output-based AIA (AdaptAIA) toward chance; ten runs. | Aalmoes, Duddu, Boutet | arXiv 2211.10209 v3 (v1 title "Leveraging Algorithmic Fairness to Mitigate Blackbox Attribute Inference Attacks"); S2: WISE | Read §3–5 | Table 2 |
| Adversary knowing (X,Y) of the training set (A) vs additionally the model's predictions (A0); published fairness constraints used to correct reconstructions; 100 repetitions. | Ferry et al. | SaTML 2023 / arXiv 2209.01215 | Read §III–IV | §IV-A, §IV-B |
| Fairness interventions reduce standard MIA/AIA, but combining predictions of a biased and a fair model (FD-AIA, A(x)=f(T_b(x),T_f(x))) improves attribute inference. | Tian et al. | arXiv 2503.06150 v2 (accepted IEEE TDSC per arXiv) | Read §6–7.1 | §6.3 Eq. 12; §7.1; Table 5 |
| Black-box AIAs rarely learn more than a model-free imputation adversary with the same distributional knowledge; white-box attacks can. | Jayaraman & Evans | CCS 2022 / arXiv 2209.01292 | Read abstract, Table 1 | Table 1 |
| Reducing recoverability from one exported LLM vector (final-token) does not reduce it from another (mean-pooled) or from generated summaries; INLP/linear-removal baselines; cluster-bootstrap CIs. | Liu et al., Vectors Are Not Neutral | arXiv 2605.26433 v1 (preprint) | Read §3–4 + baseline appendix | Abstract; §3; Table 6; App. baselines |
| A noise-aware adaptive attacker beats a non-adaptive one at every noise level on EEG embeddings; single-endpoint audits clear releases that still leak. | Tai, Pretrained, Frozen, Still Leaking | arXiv 2606.09189 v1 (preprint) | Read abstract, intro, Table 9 | Table 9 |
| Decodability measured as max AUROC over LR, GBT and 3 MLP probes, marginal (DA) and within class (DA|Y); invariance enforcement can create new bias. | Parikh, Petersen et al. | arXiv 2609.32004 v1 (preprint) | Read §4, abstract | §4 Evaluation; Table 1 |
| MANCE measures leakage with a freshly retrained 2-layer MLP probe at matched control-accuracy budgets (1/3/5/10 pp) and states the result is "not a guarantee". | Avitan, Goldberg, Elazar, MANCE | arXiv 2607.03973 v1 (preprint) | Read §4, Limitations | §4.1 fn. 3; "Evaluation protocol"; Limitations |
| Obliviator reports leakage as the highest accuracy among several nonlinear adversaries, over five runs. | Akbari, Afshari, Boddeti | NeurIPS 2025 / arXiv 2603.07529 v2 | Read §5 setup | §5 "Utility-Erasure Trade-off" |
| SoK "collusion" = adversaries pooling attack capabilities (e.g., data reconstruction → attribute inference), not recipients pooling permitted releases. | Duddu, He, Waheed, Asokan | arXiv 2606.10091 v1 (USENIX Sec 2026 per abstract) | Read abstract, §5 excerpts | §5.4 |
| Inconsistent model-selection criteria make debiasing comparisons incomparable; DTO-based selection proposed. | Han, Baldwin, Cohn, Fair Enough | EACL 2023 | Read §4.3 | §4.3 |
| Obliviator's authors are Akbari, Afshari, Boddeti (not "Kim et al." as in the brief). | Obliviator PDF / NeurIPS virtual page | arXiv 2603.07529 v2 | Read front matter | title block |
| LinEAS is by Rodriguez et al. (NeurIPS 2025), an activation-steering method (not Guerreiro; not an erasure evaluation). | arXiv 2503.10679 | v3 | Abstract (search result) | abstract |
| "What Happens to a Dataset…" v1 lists a single author, Richard Johansson. | arXiv 2403.16142 v1 PDF | v1 | Read title block | p.1 |
| Wang 2019, log-linear guardedness, Ferry 2023, Aalmoes 2022/24, Shen 2022 and Tian 2025 were not found by string search in the AAAI-27 references.bib (Sep 17 zip) or in durable-guarantees@956f5c8; Jayaraman & Evans, Barrett et al., Obliviator, FNF, FARE are in the AAAI bib. | local grep | AAAI27 zip "(1)"; dg@956f5c8 | grep only | `references.bib`; `git grep` HEAD |

## C. Repository and manuscript sources (pinned)

| Source | Pin | Notes |
|---|---|---|
| Bostesa/durable-guarantees | 956f5c883f515646aa457db55ecbd74b913768b2 | read-only; equals relocated local checkout HEAD |
| Bostesa/PCRL origin/main | 55e4cb1d1b603a52e5ae16fd5c71d41ebb9b3827 | shipped code |
| PCRL research/pcrl-submission-finish-v1 | 55c0c5a358eea193e6d2293a2c814a5435487d14 | ACS manuscript tex; fixed scorers |
| PCRL fix/retire-accuracy-guarantee | 5d4eda04639aae10733e4d72c2ceaae0e849ede5 | unmerged |
| AAAI "Outputs Leak" source zip | sha256 fd251508f99fef7445370cbf40c3efa01e0c6d9325776af4b7342ad5923da124 | main text = Jul 29 submission build |
| AAAI submission build PDF (58) | sha256 2d86afadd0d1b75194499b35098187124d822851f27a298b6fd174376ff3b077 | |
| NeurIPS PCRL final PDF (23) | sha256 4a705d5e77ea83e13e9bb0c9d95a868230cb4b63337c62e7f58a9d363b6d8172 | final .tex unavailable |
| SaTML ACS PDF (Downloads) | sha256 7c8e80e2e77f71f37102e127341cce6a58efb45e68c9d49bae68bb14f550dfd3 | no git source |
| Other PCRL refs | see BOTH_REPOS_INDEX.json `relevant_refs` | |
