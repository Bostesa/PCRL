# Tang, Hosseini, Zhai, Durand, Mori (2026): FairNVT

## Exact citation, versions and publication status (as evidenced)
- **Authors:** Qiaoyue Tang (UBC; work done during an internship at RBC Borealis), Sepidehsadat Hosseini, Mengyao Zhai, Thibaut Durand, Greg Mori (RBC Borealis; v1 also lists Simon Fraser University for Mori).
- **Titles differ by version:**
  - **v2 / TMLR:** "FairNVT: Fair Classification via Noise Injection in Vision Transformers" (arXiv:2604.16780**v2** [cs.CV], 17 Aug 2026, 28 pp.).
  - **v1:** "FairNVT: Improving Fairness via Noise Injection in Vision Transformers" (arXiv:2604.16780v1, 18 Apr 2026, 24 pp.).
  - The author page (https://qiaoyuet.github.io/, "Site last updated 2026-08-14") still uses the **v1 title**.
- **Publication status, evidence only:**
  - **TMLR 2026.** The v2 PDF header on every page reads "Published in Transactions on Machine Learning Research (08/2026)". p.1 reads "Reviewed on OpenReview: https://openreview.net/forum?id=rzm6gZrYgl". The arXiv v2 comments field reads "TMLR". The author page reads "Transactions on Machine Learning Research (TMLR) 2026".
  - **ICLR 2026 workshop.** The arXiv v1 comments read "ICLR 2026 Algorithmic Fairness Across Alignment Procedures and Agentic Systems (AFAA) Workshop". The author page reads "Also presented at … (AFAA) Workshop @ ICLR 2026".
  - **Not an ICLR main-conference paper.** No evidence supports that.
  - **Not independently verified:** I could not open the OpenReview forum or the TMLR decision record (the OpenReview API returned a 403 challenge). The TMLR status rests on the PDF header, the arXiv comment and the author page.
- **Versions read:**
  - **v2 read in full, including Appendices A–E** (Tables 1–14). Fetched 2026-10-01.
  - v1 skimmed for the title, venue and whether Table 3's attacker-strength and averaging columns are present. **They are absent in v1.** The averaging and deeper-attacker analysis first appears in v2 (17 Aug 2026).
- **Code:** I found none.
  - No repository link in v1 or v2.
  - The author page lists no code link for FairNVT, though it does for other projects.
  - github.com/qiaoyuet repositories (API listing): none related.
  - A GitHub search for "FairNVT" returned 0 results.
  - **Code availability: not found as of 2026-10-01.**

## Actual question (authors' scope)
- A lightweight debiasing framework for **frozen pretrained transformer encoders**.
- It "improves prediction fairness while preserving task performance".
- It is motivated by the intuition that reducing sensitive information in the classifier's input representation facilitates fairer predictions (abstract; §1).

## Objective
- **Architecture:**
  - A frozen ViT-B/16 (or BERT-base) with bottleneck **task** and **sensitive** adapters in each block. Reduction factors are 8 and 16 (App. C).
  - Embeddings e_t and e_s are the [CLS] token with the respective adapter active (§3.1).
- **Sensitive branch:**
  - Clip e_s to L2 norm C, then add z ~ N(0, C²σ²I_d), giving e_s^noised (§3.1).
  - Fuse by concatenation: e_f = [e_t, e_s^noised].
  - Gradients from the task loss are **stopped** through e_s^noised.
- **Loss** (§3.2): L = L_ce^t + β₁L_ce^s + β₂(CosSim(e_t, e_s) + HSIC(e_t, s)) + β₃L_dp.
  - L_dp is the batch demographic-parity gap of the predicted positive probability.
  - A multi-class DP loss is used for BIOS (App. D.2).
- **Noise at inference:** fresh noise is drawn at both training and **inference**, so predictions are stochastic (§3.3; Limitations §5).

## Formal guarantee
- **None.**
- **Prop. B.1:** Z ⊥ S implies DP = 0, via TV.
- **Prop. B.2:** if Z_t ⊥ S and the sensitive part is replaced by independent noise, then Z̃ ⊥ S.
- Both are idealised intuitions (App. B, p.17–18). The authors write: "FairNVT does not replace the sensitive component with an independent random variable".
- **No DP accounting:** despite clipping plus calibrated Gaussian noise, no ε or δ is claimed.
- **Broader-impact statement:** "lower benchmark disparity … does not certify fairness in real deployments".

## Recoverability, causal use, fairness or privacy?
- **Primary:** prediction-level fairness (DP, EO, EOpp).
- **Secondary:** representation-level recoverability, measured with a post-hoc attacker on embeddings. Footnote 5 (p.7): "Attacker performance is used in ablation studies only".
- No causal use.
- "Privacy" appears only loosely (App. E, Table 8 text).

## Representation and outputs exposed
- **Fairness metrics:** computed from predictions of the head on e_f.
- **Attacker inputs:** e_t, e_s, e_s^noised and e_f, each attacked separately (Table 3).
- No attack is run on task outputs as an attribute-recovery surface. DP is reported, which is related.

## Datasets and task–attribute relationships
- **CelebA** (official splits; Young is 77/23, Male is 57/43):
  - Smiling / Male;
  - Big Nose / Young;
  - Wavy Hair / Male;
  - more pairs in Table 13.
- **UTKFace:** task Gender, sensitive Age (< 35 vs ≥ 35).
- **BIOS** (BERT-base): Profession (multi-class) / Gender (App. D.2).
- All sensitive attributes are binary.

## Removal methods (baselines) and attacker families
- **Vision baselines:**
  - Vanilla ViT with a task adapter;
  - ViT-FSCL (re-implemented);
  - FairViT;
  - FairVPT (re-implemented; no official code);
  - FairNet.
- **Text baselines:** Vanilla-BERT, FT-Debias, INLP, SUP, ConGater and DAM.
- **Attacker:**
  - An MLP with the same architecture as the task head, following Kumar et al. 2023 (App. C). It is trained on train-split embeddings with S labels until training accuracy plateaus, and tested on the test split.
  - **Att3 / Att10:** MLPs with 3 and 10 hidden layers.
  - **Avg5 / Avg50:** the sensitive embedding is averaged over 5 or 50 independent noise draws (Table 3, v2 only).

## Adaptive knowledge and query access
- **Defense awareness:** none. The attacker is not told the split structure, C or σ.
- **Query access:** single release by default. **Avg5/Avg50 model repeated release.**

## Utility protocol and model selection
- **Hyperparameter search:** a grid over reduction factor, head depth, learning rate, clipping, noise {1, 5, 10} and β ∈ {0.1, 0.3, 0.5, 1.0}. They "report the best validation-selected results" (App. C).
- **Table 1:** the configuration with the **highest validation task accuracy**, mean (sd) over 3 runs. For FairNVT the sd includes inference noise.
- **Pareto curves (Figs 2, 4):** plot test performance of every hyperparameter configuration. The frontier is therefore drawn on test results; this is a reading, not a stated test-selection.
- **Baselines** are tuned on validation over smaller grids.
- **Inconsistency:** Table 8 lists noise level 100, while App. C's grid is {1, 5, 10}.

## Individual vs multiple recipients
- Single downstream classifier and a single attacker.
- Avg5/Avg50 is repeated observation by **one** recipient, not a coalition.

## Strongest results
- **Table 1(a), CelebA Smiling/Male:** Acc 93.1, DP 9.1, EOpp 0.7, EO 2.5. Vanilla: Acc 89.6, DP 16.4.
- **UTKFace:** Acc 97.7, DP 18.2.
- **BIOS:** Acc 80.6 and DP 1.6, against the best baseline ConGater at 82.4 / 1.9 (Table 5).
- **Table 3 (v2), attacker accuracy / balanced accuracy, Smiling/Male:**

  | Input | Acc | BAcc |
  |---|---|---|
  | e_t | 68.3 | 68.6 |
  | e_s | 99.0 | 99.0 |
  | e_s^noised | 51.7 | 50.2 |
  | **e_f** | **52.1** | **51.7** |
  | Att10 | 52.4 | — |
  | **Avg5** | **63.6** | — |
  | **Avg50** | **92.5** | — |

  Big Nose/Young has the same pattern: e_t 74.6 / 60.4, e_f 67.6 / 53.2, Avg50 85.9 / 76.3.
- The authors acknowledge that averaging k draws reduces the noise variance to σ²/k "and makes the sensitive information easier to recover" (§4.2, p.11).

## Limitations relevant to our analysis (including my inferences)
1. **Attacker monotonicity anomaly (my inference, important):**
   - e_t is a sub-vector of e_f = [e_t, e_s^noised].
   - A Bayes-optimal attacker on e_f is at least as accurate as one on e_t. Yet the reported e_f attacker reads 52.1% (BAcc 51.7) against 68.3% (BAcc 68.6) on e_t, for Smiling/Male. Att10 does not close the gap (52.4).
   - So the e_f readings reflect **attacker optimisation failure**, plausibly the large-variance noise coordinates swamping training, not absence of information.
   - The authors interpret the drop as reduced leakage ("worse than on the task embedding e_t"; §4.2).
   - This is a direct example of a method's stated test disagreeing with recoverability.
2. **Text inconsistency:** the text says e_t attacker accuracy is "≈ 80% in Expression/Gender", but Table 3 shows 68.3.
3. **Single-release dependence:** protection relies on one noise draw per input. Avg50 recovers 92.5% (authors' own result).
4. **No defense-aware attacker** (e.g., one that knows the coordinate split and drops e_s^noised).
5. **Scope:**
   - No guarantee.
   - Three runs, with no CIs beyond sd.
   - Binary S only.

## What our work repeats, extends or contradicts
**Durable-guarantees (AAAI)**
- **Not cited.** FairNVT is absent from references.bib.
- **Close prior or concurrent overlap with "isolate-then-noise":** FairNVT isolates sensitive information in a dedicated learned branch (a split adapter plus HSIC/orthogonality), then adds clipped Gaussian noise to that branch only. This is the same design idea as AAAI's **subspace-confined channel**, which learns a basis Q, noises within it, and releases the complement clean.
- **Averaging attack:**
  - FairNVT v2 Table 3 (Avg5/Avg50) already shows that repeated releases with fresh noise undo protection.
  - AAAI's averaging attack (σ/√N; crossing 0.55 at 16 queries) **repeats** this.
  - Dates: FairNVT v2 was posted 17 Aug 2026 and the AAAI source is dated 30 Aug 2026, so this is concurrent at best. v1 (Apr 2026) lacked the averaging result but had the isolate-then-noise design.
- **AAAI extends:**
  - a defense-aware Tier 2: knowing Q, projecting onto the complement, a Gaussian LRT;
  - tabular data;
  - a common utility bar;
  - output-surface floors;
  - a DP-certified full-rank variant.
- **Contradicts:** nothing in FairNVT's stated results. **(Inference)** AAAI's Tier-2 finding (basis knowledge gives 0.92–0.99) predicts FairNVT's e_f is likewise breakable by an attacker that ignores the noised coordinates. FairNVT's own e_t row (68.3%) already shows this.
- **Vision cells:** AAAI uses CelebA Attractive/Young (paper.tex:469). FairNVT uses CelebA Smiling/Male and Big Nose/Young.

**PCRL**
- **Not cited.**
- **Architectural overlap:** a frozen pretrained backbone plus per-branch adapters, with HSIC dependence penalties and a DP or constraint loss. This is close to PCRL's frozen backbone + per-purpose LoRA + HSIC/VICReg auxiliaries.
- **Benchmark overlap:** PCRL's CelebA extension uses task Smiling with disallowed Male and Young (NeurIPS S5-experiments.tex:191). This **is** FairNVT's Smiling/Male cell, and its Young attribute.
- **PCRL differs:** it has multiple purposes with conflicting permissions, a linear-R² constraint via a proxy-Lagrangian, a LEACE warm start, and no noise.
- **PCRL should compare to or cite FairNVT** for the CelebA cells, and for adapter-based debiasing on frozen ViTs.
- **Contradicts:** nothing.
