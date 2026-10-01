# Song & Shmatikov (2020). Overlearning Reveals Sensitive Attributes

## Citation and version read
- **Citation:** C. Song, V. Shmatikov. *Overlearning Reveals Sensitive Attributes.* ICLR 2020.
- **Version read:** arXiv:1905.11742 **v3** (8 Feb 2020; PDF header "Published as a conference paper at ICLR 2020"), 12 pp., read in full.
  - Earlier versions: v1 28 May 2019 and v2 27 Sep 2019 (arXiv abstract page).
  - URL: https://arxiv.org/abs/1905.11742, fetched 2026-10-01.
- **Not read:** the OpenReview camera-ready file. Checking it against v3 is not done.
- **Appendix:** the v3 PDF contains no appendix, so all 12 pages, including references, were read.
- **Naming note:** Stadler et al. 2024 cite this work as "Song & Shmatikov (2019). In NIPS". That is a citation error in Stadler's bibliography; the paper itself is ICLR 2020.

## Actual question (authors' scope)
Do models trained for simple objectives implicitly learn sensitive attributes that are not part of the objective and are (reportedly) statistically uncorrelated with the labels? Can this be exploited at inference time and via model re-purposing? Can "censoring" prevent it (p.1 §1)?

## Formal guarantee
None. The study is empirical, with a conjecture about data complexity (§4.4).

## Recoverability, causal use, fairness or privacy?
- **Recoverability**, framed as privacy: inference-time attribute inference and model re-purposing. Fairness is mentioned as a motivation.
- No causal-use analysis.

## What is exposed
- **Inference attack:** the last-layer (FC) representation z = E(x) of a test input. The adversary has black-box oracle access to E and an auxiliary labelled dataset D_aux (§3.1, p.3).
- **Re-purposing:** the model's layers themselves (white-box weights), which are fine-tuned on a small D_transfer (§3.2).
- Task outputs are not attacked as a leakage surface. Table 2 reports target-task accuracy only as utility.

## Datasets and task–attribute relationships (Table 1, p.4; Cramér's V between y and s)

| Dataset | Task y | Sensitive s | Cramér's V |
|---|---|---|---|
| Health (Heritage) | Charlson index > 0 | age (9 bins) | 0.149 |
| UTKFace | gender | race | 0.035 |
| FaceScrub | gender | identity (500) | 0.044 |
| Places365 | in/outdoor | scene | 0.052 |
| Twitter (PAN16) | age | author | 0.134 |
| Yelp | review score | author | 0.033 |
| PIPA | identity | two faces from the same photo | n/a |

The authors deliberately chose **weakly coupled** task–attribute pairs ("nor statistically correlated with them", p.1).

## Removal ("censoring") methods and attacker families
**Censoring methods:**
- adversarial training with γ = 1.0 and a 2-layer FC adversary (256 and 128 units), following Xie et al. 2017;
- information-theoretic censoring via a VAE with β = 0.01 and λ = 1e-4, following Moyer et al. 2018 (§4.2, p.5).

**Attacks:**
- **Inference attack:** "we use the same architecture as the censoring adversary", 50 epochs.
- **De-censoring (Alg. 1, p.3):** learn a transformer T that maps censored z to an auxiliary uncensored model's features, then attack T(z).
- **Re-purposing:** fine-tune lower layers on 2–10% of the data (Table 4).
- **Inner-layer censoring with CKA analysis** (Table 5).

## Adaptive knowledge and query access
- **Knowledge:** the adversary has D_aux (50% of training-set size) with labels s, and black-box E. It does not know the censoring hyperparameters.
- **Defense awareness:** de-censoring is a partly defense-aware attack. It exploits the existence of an uncensored auxiliary model, not the specific defense.
- **Query access:** a single representation per input. PIPA uses pairs of representations.

## Utility protocol and model selection
- **Utility:** target accuracy on a held-out 20%.
- **Hyperparameters:** fixed censoring values, plus sweeps over γ, β and λ (Fig. 2).
- **No CIs or seeds** are reported.

## Individual vs multiple recipients
- Individual.
- Re-purposing assumes the attacker obtains model weights.
- No coalition analysis.

## Strongest results
- **Table 2 (p.5): last-layer inference accuracy for s.**

  | Dataset | RAND | BASE | ADV | IT |
  |---|---|---|---|---|
  | UTKFace race | 42.52 | 62.18 | 53.28 | 53.30 |
  | FaceScrub identity | 1.42 | 33.65 | 30.23 | 10.61 |
  | Places365 | 1.37 | 31.03 | 12.56 | 2.29 |

  ADV and IT also cost target accuracy, e.g., Twitter 76.22 → 57.97 under ADV.

- **Censoring strength does not help (Fig. 2, p.6).** Increasing γ can *increase* inference accuracy for UTKFace and Yelp ("censoring defeats itself").
- **De-censoring raises inference** against ADV representations, by up to +12.24 on FaceScrub (Table 3).
- **Unseen-race inference:** a gender model trained on one race still reveals race at about 61–62% vs 42.52% random (p.5).
- **Re-purposing beats training from scratch** (Table 4). Censoring lower layers blocks re-purposing only at the cost of task accuracy. Censoring one layer leaves other layers re-purposable (Table 5 + CKA, p.8).
- **Conclusion (p.9–10):** overlearning "may be intrinsic"; regulators should focus on how models are applied.

## Limitations relevant to our analysis
- Single runs, with no statistical treatment.
- The post-hoc attacker has the same architecture as the training adversary. Its strength comes from fresh training, not added capacity.
- **(My reading)** The claim that attributes are "not correlated" rests on Cramér's V between labels, not on conditional dependence given x.
- No output-side analysis, and no multiple releases.
- Censoring baselines date from 2017–2018.

## What our work repeats, extends or contradicts
**Durable-guarantees (AAAI)**
- **Already cited** (paper.tex:155–156) for "models leak attributes they were never trained to predict".
- **Repeats:**
  - Censored or "approved" representations still leak to freshly trained attackers.
  - Stronger censoring weight does not monotonically help. This mirrors AAAI's LAFTR result across adversary weights and the VFAE β result.
- **Extends:**
  - The approval criterion is a linear-R² certificate rather than the training adversary.
  - Attacker families are stronger and varied.
  - A defense-aware tier is added.
  - Utility is measured against the same bar.
  - The output surface is added.
- **Complements:**
  - Song shows representation leakage even when label–attribute coupling is weak (Cramér's V ≤ 0.15).
  - AAAI shows that **output** leakage tracks label–attribute coupling.
  - **(Inference)** Together these support separating representation leakage, which is not governed by coupling, from output leakage, which is.
- **Possible overlap to state:** AAAI's "LoRA reader" attacker resembles Song's re-purposing (fine-tuning model layers). If it needs model weights, it is outside AAAI's Tier-1 "parameters not exposed" model. Clarify which (the AAAI text is ambiguous; inference).

**PCRL**
- **Already cited** (S1-introduction.tex:4) for incidental learning of unrelated attributes. That use is accurate.
- **Relevant caution:** Table 5 shows that censoring at one layer leaves other layers re-purposable. PCRL censors only via per-purpose adapters on a frozen backbone. **(Inference)** If any recipient obtains backbone features or adapter weights rather than only h_p, the per-purpose bound says nothing. PCRL's threat model must state that only h_p is released.
- **Contradicts:** nothing.
