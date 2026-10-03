# Closest prior analyses

**Role:** claims and literature reviewer, branch `research/combined-analysis-paper-v1` (cut from dc6a2e96).
**Date:** 2026-10-03. Read-only; nothing was run except text extraction from public PDFs.

**Starting point.** The reconciled literature assessment:
- `results/combined_empirical_preparation_v1/PRIOR_ANALYSIS_MATRIX.csv` and `notes/literature_{A,B}/`;
- `results/combined_evidence_reconciliation_v1/CORRECTIONS_TO_PREVIOUS_ASSESSMENT.md` (items 3, 11, 12).

This file re-reads only the closest primary papers. It does not restart the survey. Reviewer requests are
referred to by the anonymised labels used in `REVIEW_RESPONSE_MATRIX.csv`:
- AAAI-R1 asks how this differs from work showing linear probes are insufficient;
- AAAI-R2 names Song & Shmatikov 2020, Ravfogel et al. 2022, Stadler et al. 2024 and Elazar et al. 2021;
- NeurIPS-R2 asks for broader privacy-representation literature.

**Verification markers:**
- **[V]** venue and year checked today on the official page, the official PDF, or the publisher's Crossref
  record. The content locations were read today in the paper text.
- **[V-prep]** venue checked today. The content location comes from the 2026-10-01 preparation reading
  (`PRIOR_ANALYSIS_MATRIX.csv`) and was not re-read today.

Bib keys refer to `references.bib` in this directory.

## 1. Verified bibliographic record

| Key | Work | Venue, year | Official page | Status |
|---|---|---|---|---|
| `belrose2023leace` | LEACE | NeurIPS 2023 (Adv. NeurIPS 36) | proceedings.neurips.cc/…/d066d21c…-Abstract-Conference.html | [V]; section numbers from arXiv v4 |
| `jovanovic2023fare` | FARE | ICML 2023, PMLR 202:15401–15420 | proceedings.mlr.press/v202/jovanovic23a.html | [V] |
| `ravfogel2020inlp` | INLP | ACL 2020, pp. 7237–7256 | aclanthology.org/2020.acl-main.647 | [V] |
| `ravfogel2022rlace` | R-LACE | ICML 2022, PMLR 162:18400–18421 | proceedings.mlr.press/v162/ravfogel22a.html | [V-prep] |
| `ravfogel2022kernel` | Kernelized concept erasure | EMNLP 2022, pp. 6034–6055 | aclanthology.org/2022.emnlp-main.405 | [V-prep] |
| `ravfogel2023loglinear` | Log-linear guardedness (**added**) | ACL 2023, pp. 9413–9431 | aclanthology.org/2023.acl-long.523 | [V-prep] |
| `elazar2018adversarial` | Adversarial removal of demographic attributes | EMNLP 2018, pp. 11–21 | aclanthology.org/D18-1002 | [V] |
| `song2020overlearning` | Overlearning | ICLR 2020 (poster) | iclr.cc/virtual_2020/poster_SJeNz04tDS.html; openreview.net/forum?id=SJeNz04tDS | [V] (text from arXiv v3, ICLR header) |
| `zhao2019inherent`, `zhao2022inherent` | Inherent tradeoffs | NeurIPS 2019; JMLR 23(57):1–26, 2022 | proceedings.neurips.cc/…/b4189d9d…; jmlr.org/papers/v23/21-1427.html | [V] (abstract level; experiment sections not re-read) |
| `madras2018laftr` | LAFTR | ICML 2018, PMLR 80:3384–3393 | proceedings.mlr.press/v80/madras18a.html | [V] |
| `moyer2018invariant` | Invariant representations without adversarial training | NeurIPS 2018 (Adv. NeurIPS 31) | proceedings.neurips.cc/…/415185ea…-Abstract.html | [V] |
| `wang2019balanced` | Balanced datasets are not enough (**added**) | ICCV 2019, pp. 5309–5318 | doi 10.1109/ICCV.2019.00541 | [V-prep] |
| `stadler2024lpl` | Least-privilege learning (**added**, AAAI-R2) | ICML 2024, PMLR 235:46393–46411 | proceedings.mlr.press/v235/stadler24a.html | [V-prep] |
| `fredrikson2014pharmacogenetics` | Warfarin model inversion | USENIX Security 2014, pp. 17–32 | usenix.org/conference/usenixsecurity14/…/fredrikson_matthew | [V] |
| `fredrikson2015model` | Model inversion with confidence information | ACM CCS 2015, pp. 1322–1333 | doi 10.1145/2810103.2813677 | [V] |
| `mehnaz2022sensitive` | Model-inversion attribute inference | USENIX Security 2022, pp. 4579–4596 | usenix.org/conference/usenixsecurity22/presentation/mehnaz | [V] |
| `jayaraman2022imputation` | Attribute inference vs imputation | ACM CCS 2022, pp. 1569–1582 | doi 10.1145/3548606.3560663 | [V] venue; content [V-prep] |
| `choquettechoo2021label` | Label-only membership inference | ICML 2021, PMLR 139:1964–1974 | proceedings.mlr.press/v139/choquette-choo21a.html | [V] (abstract level) |
| `ganta2008composition` | Composition attacks | KDD 2008, pp. 265–273 | doi 10.1145/1401890.1401926 | [V] (abstract level) |
| `tian2025fairnessprivacy` | Fairness vs privacy, two-model attack (**added**) | arXiv 2503.06150 v2; "accepted to IEEE TDSC" per arXiv | arxiv.org/abs/2503.06150 | [V-prep]; journal details **UNVERIFIED** |
| `taylor2026collusion` | Sequential releases under collusion (**added**) | arXiv 2601.21859 v2 (preprint) | arxiv.org/abs/2601.21859 | [V-prep] |
| `gupta2021fcrl` | FCRL (the "Gupta et al." candidate) | AAAI 2021, 35(9):7610–7619 | doi 10.1609/aaai.v35i9.16931 | [V-prep] |
| `goodfellow2016deep` | Softmax shift invariance (textbook) | MIT Press 2016, §4.1 | deeplearningbook.org | [V] |
| `guo2017calibration` | Calibration / temperature scaling | ICML 2017, PMLR 70:1321–1330 | proceedings.mlr.press/v70/guo17a.html | [V] |

**Added, with justification:**
- **Log-linear guardedness:** the closest precedent for "a linearly guarded representation still leaks through a
  downstream classifier's output". LEACE's footnote 5 cites it for exactly that point.
- **Wang et al. 2019:** the closest precedent for measuring attribute leakage from task outputs against a label-only
  reference ("dataset leakage") at matched task performance.
- **Stadler et al. 2024:** named by AAAI-R2. It gives the fundamental label-dependence limit.
- **Tian et al. 2025:** the closest located precedent for combining two prediction releases to raise attribute
  inference.
- **Taylor et al. 2026:** the only located collusion-aware release mechanism (CORRECTIONS item 11).

**Removed or demoted:**
- **"Belghazi"** (MINE, a mutual-information estimator) is not relevant. The study reports attacker AUC, not
  estimated mutual information, and makes no MI claim. It is not cited.
- **Gupta et al. (FCRL)** is kept only as a minor precedent: a common downstream protocol and a
  preprocessing-aware attacker.
- **Elazar et al. 2021 (amnesic probing)** is in the bib because a reviewer cited it. It studies the causal use of
  concepts, not recipient recovery, so it is not in the comparison table.

## 2. Six-dimension comparison

The six dimensions:
1. **D1:** the method's native protection criterion tested against nonlinear or adaptive recovery.
2. **D2:** matched held-out task utility.
3. **D3:** the representation **and** the complete prediction output, attacked as one release.
4. **D4:** unnecessary logit components, such as a softmax-invariant offset.
5. **D5:** a compression control: a no-fairness model of matched capacity.
6. **D6:** multiple recipients combining releases.

"Yes", "partly" and "no" refer to what the paper tests, not to what it mentions.

| Work | D1 | D2 | D3 | D4 | D5 | D6 |
|---|---|---|---|---|---|---|
| LEACE | **No.** Evaluation uses linear probes. §7 limits the scope to linear erasure and conjectures that general nonlinear erasure is intractable without the data-generating process. [V] | **Partly.** §5.2 Bios: profession accuracy is 77.3 % on erased embeddings vs 79.3 % originally, and the TPR-gap falls from 0.198 to 0.084; a refit main-task head reaches 78.1 % but raises the TPR-gap to 0.158. There is no cross-method utility matching. [V] | **No.** Footnote 5 notes that the softmax probabilities of a multiclass logistic-regression head can leak the erased concept if a classifier is stacked on top (citing log-linear guardedness), but this is not tested. [V] | No | **Partly, different purpose.** §5.3 uses a random orthogonal projection of equal rank as a control for the size of the intervention on language-model loss, not on concept recovery. [V] | No |
| FARE | **Partly.** The certificate bounds the demographic-parity (DP) distance of *any* classifier on the cells, so nonlinear classifiers are in scope. §6 "Certificate validity" (Fig. 6) compares one point with 24 diverse downstream classifiers, half trained to maximise unfairness, and notes that evaluating with a single model class can underestimate unfairness. The measure is DP, not attribute-recovery AUC; there is no adaptive attacker. [V] | **Partly.** Test-set Pareto fronts with a common downstream 1-hidden-layer network (§6; other classifiers in App. H.1). No validation-only selection at a fixed utility cap. [V] | **Partly.** DP of downstream predictions computed from the cells. A separately released task head or output channel is outside the certificate (this study: the view-2 bypass). [V] | No | **Partly.** App. C compares fairness-unaware k-means restricted encoders certified by the same procedure. App. H.2 ablates the number of cells k̄ (Table 5). §6 states that restricted encoders "lose some information" (≤ 1.5 % accuracy gap). There is no same-budget γ = 0 tree twin, and recovery is not measured. [V] | No. Transfer tasks from one representation (Table 1, Health). [V] |
| INLP | **Yes, partly.** §6.1.1: after INLP a 1-layer ReLU MLP recovers gender at 85.0 %. [V] | **Partly.** Task accuracy and TPR-gap after projection; not matched. [V-prep] | **Partly.** TPR-gap of task predictions (fairness of outputs), not attribute recovery from outputs. [V-prep] | No | No | No |
| R-LACE | **Partly.** §5.1 (GloVe): RBF-SVM and MLP recover gender above 90 %. Table 2, footnote 9: fresh adversaries recover after gradient-reversal training. [V-prep] | **Partly.** Profession accuracy with a refit linear head. [V-prep] | No | No | No | No |
| Kernelized erasure | **Yes.** Protection against the matched kernel is partial and does not transfer to other kernels or an MLP (§5.2, Table 2). [V-prep] | **Partly.** SimLex and profession accuracy. [V-prep] | No | No | No | No |
| Log-linear guardedness | **Partly.** A linearly guarded representation vs downstream log-linear classifiers: binary ones cannot recover the concept, while a constructed multiclass softmax head can (§5). [V-prep] | No | **Partly.** The leakage channel is the predictions of a head on the guarded representation. Representation plus head output is not attacked jointly. [V-prep] | **No.** It shows a binary/multiclass asymmetry for log-linear heads, a different mechanism from a softmax-invariant offset. | No | No |
| Elazar & Goldberg 2018 | **Yes.** The adversary sits near chance during training, while a post-hoc attacker trained on the encoded representations recovers the attribute (Table 3, Δ = attacker − adversary; Table 4). [V] | **Partly.** Main-task accuracy is reported per configuration (Tables 3–4); not matched. [V] | No (representation only) | No | No | No |
| Song & Shmatikov 2020 | **Yes.** Representations censored by adversarial or information-theoretic training remain inferable post hoc (Table 2); de-censoring raises recovery (Table 3). [V] | **Partly.** Fig. 2 plots main-task accuracy loss against inference accuracy under censoring. [V] | **Partly.** Representations at several layers and adversarial re-purposing (Tables 2, 4, 5). The prediction output is not attacked; logit-layer leakage is noted only as an anecdote in related work (§5). [V] | No | No | **No.** The PIPA "pairs" are two images fed to one inference model, not combined recipients. [V-prep] |
| Zhao & Gordon 2019/2022 | No | **Partly.** Theory: if base rates differ, any classifier satisfying DP has a lower bound on group-wise error. Real-data experiments are reported (abstract). [V] | No | No | No | No |
| LAFTR (Madras 2018) | **No.** Evaluated by the fairness gaps of downstream classifiers (Fig. 2), not by fresh-attacker recovery. [V] | **Partly.** Accuracy–fairness trade-off curves and transfer (§6.2). [V] | **Partly.** Fairness of downstream predictions only. | No | No | No (transfer to new tasks from one representation) |
| Moyer et al. 2018 | **Yes, partly.** Post-hoc adversaries with 0–3 hidden layers, trained independently on hold-out sets (§5, Fig. 1). [V] | **Partly.** Held-out prediction accuracy is reported beside adversary accuracy; not matched. [V] | No | No | No | No |
| Wang et al. 2019 | **Partly.** Attacker-architecture robustness (§4, Table 2). Adversarial removal at an intermediate layer, evaluated through output leakage (Table 4). [V-prep] | **Yes, in a specific sense.** Model leakage is compared with label ("dataset") leakage at matched F1, by randomly perturbing labels (§3). [V-prep] | **Partly.** Attribute recovery from task outputs, with a label-only reference. The intermediate representation is modified but not released. [V-prep] | No | No | No |
| Stadler et al. 2024 | No | **Partly.** Utility as normalised accuracy, 5 repetitions. [V-prep] | **Partly.** Leakage fundamentally implied by the true label (Fig. 3); predictions are not attacked as a separate surface. [V-prep] | No | No | No |
| Fredrikson et al. 2014 | No | **Partly.** Differentially private dosing models, with utility judged by simulated clinical outcomes. | **Partly.** Model predictions plus demographics, used to infer genotype. [V] | No | No | No |
| Fredrikson et al. 2015 | No | **Partly.** Countermeasures are claimed to cost "negligible degradation to utility" (abstract). [V] | **Partly.** Confidence values. [V] | **Partly.** §6: rounding reported confidence scores defeats the black-box reconstruction. That removes output *precision* not needed for the decision; it is not a softmax-invariant component, and AUC is not measured. [V] | No | No |
| Mehnaz et al. 2022 | No | No | **Partly.** Confidence-score and label-only black-box attribute inference on training records. [V] | **Partly.** The label-only attack performs on par with the confidence-score attack (§1), so reducing outputs to labels did not remove the leakage. [V] | No | No |
| Jayaraman & Evans 2022 | No | No | **Partly.** Black-box outputs and white-box neurons, against a model-free imputation baseline (Table 1). [V-prep] | No | **No,** but the imputation baseline is the analogue of this study's label-only and constant references. | No |
| Choquette-Choo et al. 2021 | No | No | **Partly.** Hard labels only (membership, not attributes). [V] | **Partly.** Masking confidence scores is shown to be insufficient against label-only attacks. [V] | No | No |
| Ganta et al. 2008 | No | No | No (anonymised tables) | No | No | **Yes.** Independent anonymised releases are combined to breach privacy. Not learned predictions. [V] |
| Tian et al. 2025 | No | **Partly.** Accuracy and fairness reported per intervention. [V-prep] | **Partly.** Attribute inference from task outputs. [V-prep] | No | No | **Partly.** Combining the predictions of a biased and a fair version of the *same* task model raises attribute inference (FD-AIA, §6.3 Eq. 12; Table 5). [V-prep] |
| Taylor et al. 2026 | n/a (mutual-information mechanism) | Distortion utility, exact | n/a | No | No | **Yes, theory.** Sequential releases with a collusion budget; finite alphabet, known pmf, protects all of X. [V-prep] |
| FCRL (Gupta 2021) | **Partly.** A preprocessing-aware attack and several downstream families. [V-prep] | **Partly.** A common downstream protocol. [V-prep] | **Partly.** Parity of downstream predictions. | No | No | No |

**Also checked and not closer** (preparation package): Barrett et al. 2019, Kumar et al. 2022, Cerrato et al. 2024,
Pouget et al. 2024, the FNF/FRG certificate papers, and the 2026 preprints (Liu et al., Tai, Parikh et al., and the
Duddu et al. SoK, whose "collusion" means adversaries pooling attack capabilities, not recipients pooling
releases). They are recorded in `PRIOR_ANALYSIS_MATRIX.csv`.

## 3. What is not new

The manuscript must attribute each of these and must not present any of them as our discovery.

1. **Linear protection does not stop nonlinear recovery.**
   - Shown for adversarially trained representations (`elazar2018adversarial`, `song2020overlearning`,
     `moyer2018invariant`) and for linear erasure (`ravfogel2020inlp` §6.1.1, `ravfogel2022rlace` §5.1).
   - LEACE states it as an explicit limitation (`belrose2023leace` §7).
   - Kernel erasure does not transfer across kernels (`ravfogel2022kernel`).
   - Our LEACE C3 results (recovery around 0.81–0.87 under official LEACE) are a further instance. A nonlinear
     attack does not refute LEACE's theorem.
2. **Task outputs leak sensitive attributes, including beyond the label.**
   - Model-inversion and attribute-inference attacks use confidence scores and labels (`fredrikson2014pharmacogenetics`,
     `fredrikson2015model`, `mehnaz2022sensitive`).
   - Output leakage beyond a label-only reference at matched task performance: `wang2019balanced`.
   - Leakage through a downstream head on a linearly guarded representation: `ravfogel2023loglinear`, LEACE
     footnote 5.
   - Many black-box attribute-inference results do not beat imputation (`jayaraman2022imputation`). Our label-only
     and constant references play that role and must be reported with every output result.
3. **Hard decisions are not a privacy defense in general.**
   - Label-only attacks match confidence-based ones (`mehnaz2022sensitive`; `choquettechoo2021label` for membership).
   - Our low decision recovery is a property of weak heads, not a protection mechanism (see `CLAIMS_SCOPE_REVIEW.md`).
4. **Dependence between the task label and the protected attribute forces a trade-off.**
   - If base rates differ, demographic parity costs accuracy (`zhao2019inherent`, `zhao2022inherent`).
   - A useful release leaks what its label leaks (`stadler2024lpl`).
   - Our label-coupling association is a measurement in this setting, not a new law.
5. **Softmax shift invariance is standard algebra.**
   - softmax(l + c·1) = softmax(l) (`goodfellow2016deep` §4.1).
   - Hence probabilities, argmax decisions and proper task losses do not depend on the common offset.
   - Our contribution, if any, is the *measurement* that particular frozen heads carry attribute signal in that
     offset. The identity is not ours. A refitted logistic-regression head whose logits are log-probabilities has
     an offset determined by its margin, so the effect is head-specific by construction.
6. **Coarsening outputs as a countermeasure is old.** Rounding confidences (`fredrikson2015model` §6) and masking
   confidences (`choquettechoo2021label`, which shows this is insufficient) precede the offset ablation.
7. **Attacker coalitions and composition of releases leak more.**
   - Combining independent releases (`ganta2008composition`).
   - Combining two versions of a model's predictions (`tian2025fairnessprivacy`).
   - Collusion-aware release theory (`taylor2026collusion`).
   - Our recipient-pair result is a measured instance under a purpose-permission table. It is not a composition
     theorem, and the absence of an effect elsewhere would not prove one.
8. **Restricted (compressed) encoders lose task and sensitive information.** FARE itself reports the accuracy gap and
   compares fairness-unaware k-means restricted encoders (`jovanovic2023fare` §6, App. C). That compression accounts
   for part of a fair encoder's effect is therefore anticipated. Our same-budget γ = 0 tree twin is a sharper control
   on recovery AUC, not a new phenomenon.
9. **Fresh attackers beat in-training adversaries; native checks are narrower than recipient recovery.**
   `elazar2018adversarial` (Table 3), `song2020overlearning` (Tables 2–3), `ravfogel2022rlace` (Table 2, footnote 9).

## 4. Unresolved novelty

None of the closest papers we re-read evaluates, in one protocol on the same stored models, all of the following:
- each method's native check;
- nonlinear recovery from the representation, from the complete prediction output and from both;
- held-out utility of the output actually released;
- a capacity-matched zero-fairness control;
- recipient pairs under purpose permissions.

FARE comes closest on D2 and D5. Wang et al. 2019 comes closest on D2 and D3. Tian et al. 2025 comes closest on D6.
None of them tests a softmax-invariant logit component (D4).

**This is a gap in a bounded reading, not evidence that we are first:**
- the search covered the reconciled list plus the papers above;
- privacy-attack venues (USENIX, CCS, S&P, PETS) were sampled, not exhausted;
- newer 2025–2026 preprints were only skimmed.

**What the evidence can support** is narrower than a novelty claim. The paper reports a controlled, utility-qualified
decomposition, on two development datasets, that combines established observations, plus three measured findings:
- the logit-offset ablation, which is specific to these frozen heads;
- the compression share of FARE's effect;
- decision-level coalition gains.

Each finding comes with an adverse qualification. Before any statement such as "to our knowledge, not previously
reported", a targeted search should be run for:
- logit-offset or logit-sum leakage in attribute or membership inference;
- capacity-matched "unfair twin" controls for fair representations;
- multi-recipient attribute inference under purpose permissions.

Neither the manuscript nor the brief should say "no existing method can solve this": no theorem with that scope
exists here.

## 5. Addendum after the useful-head comparison (runner, 2026-10-03)

The matched useful-head comparison (`USEFUL_HEAD_COMPARISON.md`) adds one measurement relevant to D2, D3 and D5. It
does not change the novelty assessment above.

**The study.** On Adult income/sex, every arm releases a deployable logistic-regression head of one common family,
fitted on its own features.

**Findings.**
- **D5 (compression control).** The capacity-matched zero-fairness tree recovers 0.648 from its head's score, and
  FARE's recovers 0.539. The fairness term removes 0.109 [0.092, 0.126] beyond compression, on every seed.
  - This is the opposite share from the easier useful-task cell, where compression reproduced all but 0.0074.
  - FARE's own k-means comparison (App. C) anticipates that compression matters. It does not measure how the share
    varies by cell on recovery AUC.
- **D2 + D3 (utility-qualified output leakage).** LEACE keeps the head's accuracy, but its score still recovers 0.759
  (untreated 0.774).
  - Post hoc, a linear reader of the untreated score recovers only 0.515. This matches `ravfogel2023loglinear` and
    LEACE footnote 5: the residual is nonlinear signal in a linearly guarded score.
  - Every arm's hard decision recovers 0.536–0.546.

**Positioning.** The positioning sentence stays "not reported together in one protocol in a bounded reading". No
priority claim is made.
