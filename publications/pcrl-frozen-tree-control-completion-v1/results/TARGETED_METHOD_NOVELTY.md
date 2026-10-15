# Targeted novelty check: frozen-routing leaf-score correction against attribute critics

Role: CLAIMS/LITERATURE. Date: 2026-10-15. No repository file was modified.

**Method sources:** `results/pcrl_task_headroom_protected_outputs_v1/` (ORIGINAL_VS_NEW_METHOD.md, RESEARCH_DECISION.md, ADVERSARY_AND_MOTIVATION.md, SCIENCE_LOCK.json `stage_b`) and `task_headroom/leafcorr.py`.

**Search scope:** targeted, not systematic. I opened 12 primary sources plus official library documentation (section 6). "Not found" below means not found in these sources, not that it does not exist.

## 1. Recipe assessed

- **Teachers.** CatBoost depth-4 oblivious trees (income binary, occupation 6-class), trained on raw inputs without SEX.
- **Correction.** Routing is frozen. Each score is teacher + Σ_t θ[r,t,leaf]/T, so θ = 0 reproduces the teacher exactly.
- **Training.** AdamW, with weight decay pulling toward θ = 0, minimizes task cross-entropy + λ·risk.
- **Risk.** risk(view) = 1 − min(critic CEs, H(SEX)) / H(SEX). The minimum is taken over online linear and MLP critics.
- **Critic inputs.** Critics read each recipient's (probabilities, decision). In the J and S arms they also read the aligned pair.
- **Release.** Only probabilities and decisions.
- **Evaluation.** Fresh linear, MLP and boosted-tree readers, trained on auxiliary releases with SEX labels.
- **Selected service.** The FIT2-selected service, L_TREE, used local critics only.

## 2. Closest prior work

| Work | Optimizes | Fixed vs learned | Adversary | Released | Guarantee | Closeness |
|---|---|---|---|---|---|---|
| FAGTB (Grari et al., ICDM 2019) | boosting loss − λ·adversary CE (min-max) | new tree per round, fit to u = r − λt; earlier trees untouched | logistic or NN adversary on σ(F(x)) (DP) or (σ(F(x)), y) (EO); warm-started on a biased GTB; several adversary steps per round | binary score; scored by p-rule, DI, DFPR/DFNR | only that a perfectly fair, accurate solution is a global optimum | **High**: same game and critic input; but it grows tree structure under the adversary |
| ZestFinance patent US 11,720,962 B2 | GBM loss + P·adversary log-loss | new trees; "warm start" adds trees to an existing ensemble | sigmoid(a·Y_pred+b) → sensitive attribute | credit score | none | **High** (objective); I found no leaf-only mode in the text I read |
| FairGBM (Cruz et al., ICLR 2023) | proxy-Lagrangian with group FPR/FNR/DP constraints | new tree per round; multipliers updated by projected ascent | none (constraints, not critics) | classifier | approximate feasibility and optimality for the randomized classifier only | Medium |
| Global refinement (Ren et al., CVPR 2015) | task loss + ½‖W‖² on leaf vectors (linear SVM/regression) | structure fixed; all leaves relearned jointly | none | forest predictions | convex, global optimum | **High** for the parametrization (≈ our T_REFIT) |
| LightGBM `refit`, XGBoost `refresh`, CatBoost `calc_leaf_indexes` | re-estimate leaf outputs | structure fixed; LightGBM uses decay·old + (1−decay)·new | none | model | none | **High** for the mechanism |
| EiFFFeL (Abebe et al., SAC 2022), extending Kamiran et al. 2010 | statistical-parity target under an accuracy-drop limit | post hoc; greedy flips of binary leaf labels in a random forest | none (group metric) | decisions | none | **High** for post-hoc leaf edits; but discrete, random-forest only |
| Adversarial debiasing (Zhang et al., AIES 2018) | ∇L_P − proj ∇L_P − α∇L_A | all predictor weights | adversary on Ŷ (softmax output) for DP; on (Ŷ, Y) for EO | predictions | if training converges and the adversary is optimal at loss H(Z), then Ŷ ⊥ Z | **High**: our risk normalizes their H(Z) fooling target |
| Adversarial regularization (Nasr et al., CCS 2018) | task loss + λ·membership gain (min-max) | model weights | inference model h(x, y, f(x)) | prediction vector | equilibrium argument | Medium (membership) |
| MemGuard (Jia et al., CCS 2019) | minimize \|E g(s+n) − 0.5\| | model fixed; random noise added per query | defense classifier on confidence vectors; attacker knows the defense | noised vector; label unchanged; E‖n‖₁ ≤ ε | utility loss only | Medium-high; but membership, randomized, decision-preserving |
| AttriGuard (Jia & Gong, USENIX Sec 2018) | minimum-noise evasion + randomized mechanism minimizing KL within a utility budget | user's public data perturbed | attribute classifier trained on users who disclose the attribute; adaptive attacks tested | noised public data, not model outputs | no formal privacy guarantee | Medium |
| Overlearning (Song & Shmatikov, ICLR 2020) | attack paper | — | labeled auxiliary (x, s) + black-box encoder; censored models "still leak" | representations | — | threat-model anchor |
| FARE (Jovanović et al., ICML 2023) | FairGini tree encoder into k cells | partition chosen for fairness | any downstream classifier | cell representation | finite-sample high-confidence bound on DP distance | consumer anchor; a DP bound is not an AUC certificate |

## 3. Component → prior art → what differs here

| Component | Prior art | Difference |
|---|---|---|
| Stronger GBDT teacher creates headroom | FAGTB, FairGBM, ZestFinance: all start from GBDT's strength on tabular data | Headroom measured against an equally exposed neural reference, then spent. This is evidence, not method |
| Frozen routing, learned leaf values | Ren 2015; LightGBM refit; XGBoost refresh | AdamW on additive per-leaf, per-class offsets scaled by 1/T. Not conceptually new |
| Zero start = exact teacher; shrinkage toward the teacher | LightGBM refit decay blend | Decoupled weight decay instead of a blend. Trivial |
| Post-hoc leaf edits for a sensitive-attribute objective | Kamiran 2010 (not opened); EiFFFeL 2022 | Continuous scores on boosted trees, gradient-trained against critics rather than greedy flips toward a parity metric. **Narrowest gap found** |
| Critic predicts SEX from released probabilities (+ decision) | Zhang 2018; FAGTB; ZestFinance; membership analogues Nasr 2018 and MemGuard | Bank of critics, worst case (minimum CE), decision appended. Minor |
| Entropy-normalized risk | Zhang 2018, Proposition 2 | Normalization only |
| Two recipients; local vs joint vs sequential critics | Not found in these sources | Setting difference only; the joint term is not supported (section 4) |
| Deterministic release with no decision constraint | MemGuard (randomized, label-preserving) | Model-level change; decisions may move |
| Fresh auxiliary-data readers, chosen on validation | Song & Shmatikov; AttriGuard; MemGuard; FARE's consumer | Disjoint groups and a coalition view. A finite reader slate gives lower bounds only |
| Guarantee | FARE (DP), MemGuard (utility), FairGBM and Zhang (conditional) | None |

**Structural limit (my reasoning, not from a source).** With routing frozen, the release is a function of the teacher's leaf tuple, so I(S; release) ≤ I(S; leaf tuple). Corrections can only re-value or merge the teacher's cells. FARE instead chooses the partition itself for fairness. Boosted-tree readers, which can relearn the partition, are therefore the right stress test.

## 4. Conclusion

1. **Headroom.** The CatBoost teachers beat an equally exposed neural reference, which shows the earlier caps left room. That is useful evidence, not an algorithm. The GBDT fairness literature above starts from the same premise.
2. **Leaf-score correction.** The recipe is Zhang-style adversarial debiasing applied to Ren/refit leaf parameters. Zhang et al. state that their method applies to any gradient-trained model. The recipe sits in the post-hoc leaf-editing line of Kamiran and EiFFFeL. FAGTB and the ZestFinance patent are the in-processing GBDT versions.

   The only gap I found is narrow. No source I read fine-tunes continuous leaf scores of a fitted boosted ensemble, with routing frozen, against online attribute critics on released outputs. A reviewer would likely call this a natural combination. It supports a "we did not find" sentence, not a novelty claim.
3. **Joint design: not established.** L_TREE was the FIT2-selected primary. On the coalition view, J_TREE vs L_TREE is unresolved: −0.0155 [−0.0311, 0.0002]. J_TREE's excess occupation leakage is established.

## 5. Claims

**Defensible**, with the campaign's caveats (exploratory, exposed Adult roles, perfect linkage assumed, conditional intervals):

- CatBoost teachers created task headroom over an equally exposed neural reference (FIT2) and over U0 (INNER, descriptive).
- Under the probability-plus-decision contract, L_TREE lowered fresh-reader coalition SEX AUC by 0.0505 [0.0257, 0.0754] relative to RAW_PROJ. All 24 point utility clauses passed. Under the same narrowed contract the historical codes also read lower recovery than RAW_PROJ (descriptive); how much of L_TREE's margin is attributable to narrowing is NOT_IDENTIFIED (no single service was measured under both contracts).
- The method may be described as adversarial debiasing applied to the frozen-routing leaf scores of a boosted ensemble, provided the prior work above is cited.
- "In a targeted search we did not find a prior recipe that fine-tunes continuous leaf scores of a fitted boosted ensemble, with frozen routing, against online attribute critics." The claim must name FAGTB, EiFFFeL/Kamiran and Ren/refit as closest.
- The threat model is Song & Shmatikov's auxiliary-data adversary applied to outputs, and FARE's downstream consumer. The evaluation readers are fresh and are never the training critics.

**Not defensible:**

- A new algorithm. Also "the first adversarial tree method": Grari et al. claim "the first adversarial learning method for generic classifiers, including decision trees".
- The first post-hoc leaf modification for fairness or privacy (Kamiran 2010; EiFFFeL 2022).
- The first fixed-structure leaf re-optimization (Ren 2015; LightGBM; XGBoost).
- Any joint or coalition design contribution.
- Competitive or state-of-the-art protection. Against the historical codes the margin is 0.014–0.018 and unresolved.
- That the margin over RAW_PROJ is specific to trees. Under the narrowed contract the historical codes are already about 0.033 below RAW_PROJ (descriptive); the contract's causal share is NOT_IDENTIFIED, so neither a tree-specific nor a contract-specific attribution is supported.
- Any guarantee, or "beats FARE". FARE fails utility, and its DP certificate is not an AUC certificate.
- A novel two-recipient setting. That would need a search of the composition and linkage literature, which was not done here.

## 6. Sources

[F] = full text read from the PDF. [S] = read through a summarizing web-fetch tool.

- [F] V. Grari, B. Ruf, S. Lamprier, M. Detyniecki. Fair Adversarial Gradient Tree Boosting. ICDM 2019; the venue is taken from the official repository README. https://arxiv.org/abs/1911.05369 · [S] https://github.com/vincent-grari/FAGTB
- [S] S. J. Kamkar, G. M. Ward, J. L. Budzik (ZestFinance). Systems and methods for generating gradient-boosted models with improved fairness. US 11,720,962 B2, 2023. I read the description; the claims only in part. https://patents.google.com/patent/US11720962B2/en
- [F] A. F. Cruz, C. Belém, S. Jesus, J. Bravo, P. Saleiro, P. Bizarro. FairGBM: Gradient Boosting with Fairness Constraints. ICLR 2023. https://arxiv.org/abs/2209.07850
- [F] S. Ren, X. Cao, Y. Wei, J. Sun. Global Refinement of Random Forest. CVPR 2015. https://openaccess.thecvf.com/content_cvpr_2015/papers/Ren_Global_Refinement_of_2015_CVPR_paper.pdf
- [S] Library documentation:
  - LightGBM `Booster.refit` and `set_leaf_output`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html
  - LightGBM `refit_decay_rate`: https://lightgbm.readthedocs.io/en/latest/Parameters.html
  - XGBoost `process_type`/`refresh_leaf`: https://xgboost.readthedocs.io/en/stable/parameter.html
  - CatBoost `calc_leaf_indexes`: https://catboost.ai/docs/en/concepts/python-reference_catboost_calc_leaf_indexes
- [F] S. A. Abebe, C. Lucchese, S. Orlando. EiFFFeL: Enforcing Fairness in Forests by Flipping Leaves. ACM SAC 2022. https://arxiv.org/abs/2112.14435
- **NOT OPENED** (the download returned HTTP 403): F. Kamiran, T. Calders, M. Pechenizkiy. Discrimination Aware Decision Tree Learning. ICDM 2010. It is described here only as EiFFFeL and FARE report it.
- [F] B. H. Zhang, B. Lemoine, M. Mitchell. Mitigating Unwanted Biases with Adversarial Learning. AIES 2018; the venue is taken from Google Research's listing, and I read the arXiv text. https://arxiv.org/abs/1801.07593
- [F] M. Nasr, R. Shokri, A. Houmansadr. Machine Learning with Membership Privacy using Adversarial Regularization. CCS 2018. https://arxiv.org/abs/1807.05852
- [F] J. Jia, A. Salem, M. Backes, Y. Zhang, N. Z. Gong. MemGuard: Defending against Black-Box Membership Inference Attacks via Adversarial Examples. CCS 2019. https://arxiv.org/abs/1909.10594
- [F] J. Jia, N. Z. Gong. AttriGuard: A Practical Defense Against Attribute Inference Attacks via Adversarial Machine Learning. USENIX Security 2018. https://arxiv.org/abs/1805.04810
- [F] C. Song, V. Shmatikov. Overlearning Reveals Sensitive Attributes. ICLR 2020. https://arxiv.org/abs/1905.11742
- [F] N. Jovanović, M. Balunović, D. I. Dimitrov, M. Vechev. FARE: Provably Fair Representation Learning with Practical Certificates. ICML 2023, PMLR 202. https://arxiv.org/abs/2210.07213
