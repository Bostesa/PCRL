# Elazar, Ravfogel, Jacovi, Goldberg (2021). Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals

## Citation and version read
- **Citation:** Y. Elazar, S. Ravfogel, A. Jacovi, Y. Goldberg. *Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals.* TACL 9:160–175, 2021. doi 10.1162/tacl_a_00359. Landing page: https://aclanthology.org/2021.tacl-1.10/.
- **Read:** the full text, including Appendices A–B (Figs 5–6). Source PDF: https://aclanthology.org/2021.tacl-1.10.pdf (16 PDF pages, journal pp. 160–175), fetched 2026-10-01. Below, "p." is the journal page.
- **Code:** https://github.com/yanaiela/amnesic_probing (p.160, fn 1). Not inspected.

## Actual question (authors' scope)
- **Question:** is a property *Z* (e.g., POS) **used** by a model for a task (BERT masked-word prediction)?
- **Method:** remove *Z* from the representation and measure the behavioural change (p.160–161).
- **Explicit contrast with probing:** probing "may indicate that the information can be extracted from the representation, [but] provides no evidence for or against the actual use of this information by the model" (p.160).
- **Answers:** a **causal-use / behavioural** question. It does **not** answer a recoverability question; probing does that.

## Formal guarantee
None. This is an interventional methodology with empirical controls.

## Method and controls
- **Removal method:** INLP with linear SVM probes, run until a probe is within one point above majority on dev (p.162 fn 4). This guards against linear recovery only.
- **Rand control:** remove the same number of random directions (p.162 §2.3).
- **Selectivity control:** concatenate gold property embeddings (32-d) to the amnesic representation, fine-tune the subsequent layers, and test whether performance is restored (p.162–163).
- **Hewitt–Liang control task:** argued to be inappropriate for amnesic probing (Appendix B, pp.174–175).

## Recoverability, causal use, fairness or privacy?
- **Causal use** (behavioural influence) of linearly present information.
- Recoverability appears only as a precondition (p.162: amnesic probing "is only relevant in cases where the property of interest can be predicted"). It also appears as §7.1's re-probing of later layers after an intervention.
- Neither fairness nor privacy.

## What is exposed
Intermediate BERT representations. The model's own word-prediction outputs (LM accuracy, D_KL) are the measured behaviour.

## Datasets and properties
- **Model:** BERT-base-uncased.
- **Data:** UD English Treebank (c-pos, f-pos, dep) and OntoNotes (ner, phrase start, phrase end), with 100,000 random tokens for training (p.163 §3.2).
- **Task:** masked and non-masked word prediction.
- There are no protected attributes.

## Removal methods and attacker families
- **Removal:** INLP only. The discussion says non-linear removal methods could be swapped in (p.170).
- **Probes:** linear (SVM). §7.1 re-probes later layers after removal at layer i.

## Adaptive knowledge and query access
Not applicable. There is no adversary.

## Utility protocol and model selection
- **Utility:** "utility" is the main task itself (LM accuracy, D_KL).
- **INLP stopping rule:** one point above majority on dev.
- **Runs:** no repeated seeds or confidence intervals are reported. A Spearman correlation between probe accuracy and task importance is reported with p = 0.871 (p.164).

## Individual vs multiple recipients
Not applicable.

## Strongest results (Table 1, non-masked, p.164)
**Removal effects:**

| Property | Directions removed | LM accuracy: vanilla → amnesic | Rand control | Selectivity |
|---|---|---|---|---|
| dep | 738 | 94.12 → 7.05 | 12.31 | 73.78 |
| f-pos | 585 | → 12.31 | 56.47 | 92.68 |
| c-pos | 264 | → 61.92 | 89.65 | 97.26 |
| ner | 133 | → 83.14 | 92.56 | not listed |
| phrase start/end | — | +0.21 / +0.32 points (improved) | — | — |

**Findings:**
- Phrase markers are well probed (85.12 and 83.09 probing accuracy) but show no influence on LM accuracy.
- "the probe accuracy does not correlate with task importance" (p.164).
- §7.1 (Fig. 3, p.167–168): a property removed at layer i is partly **re-recoverable by probes at later layers**, e.g., c-pos in the non-masked setting. Recoverability after linear removal can re-emerge downstream.
- Discussion (p.170): only linearly present information is removed. Classifiers can rely on correlated features, so "one has to be cautious of causal interpretations".

## Limitations relevant to our analysis
- The intervention is linear only, so residual non-linear information may still be used.
- Selectivity is imperfect for dep, f-pos and the large removals.
- There are no seeds or statistics.
- This is NLP/BERT only, with no protected attributes.

## What our work repeats, extends or contradicts (keep recoverability and use separate)
**Durable-guarantees (AAAI)**
- **Distinction:** the AAAI paper states it measures "whether the attribute can be recovered from these surfaces, not whether it influences the model's decisions" (paper.tex, Setup). That is a **recoverability** design and is consistent with Elazar's distinction.
- **Citation mismatch (flag):** paper.tex:156–158 cites Elazar et al. for "what a probe finds depends on the probe". Elazar's point is different: probe recoverability does not imply **use**. The probe-dependence point belongs to Voita & Titov and Hewitt & Liang (both discussed by Elazar, p.170).
- **Inference:** the title "Outputs Leak What They Use" and the sentence "when the attribute helps with the task, the predictions themselves reveal it" carry a *use* reading. The paper's output-surface measurement (attacking ŷ) and its label-only predictor are recoverability measurements. No amnesic-style intervention on the representation is run. Any causal-use claim would need a separate design, e.g., an amnesic intervention measuring the change in ŷ with Rand/Selectivity controls.
- **Repeats / extends / contradicts:** none of these. This paper answers a different question.

**PCRL**
- PCRL's per-purpose R² constraint concerns **recoverability** of A from h_p by a linear reader. The motivating scenario ("underwriting where access to applicant race is prohibited") reads as a use/access prohibition. A recoverability bound does not show that the purpose head does not *use* A-correlated information. Equally, low use would not imply low recoverability.
- **Inference:** §7.1's layer result suggests that removal applied before later non-linear layers can be partly undone downstream. Here the "erase layer" means LEACE placed before repr_proj. PCRL measures at the final h_p, which is the right place, but the result is a caution for any intermediate-layer claim.
