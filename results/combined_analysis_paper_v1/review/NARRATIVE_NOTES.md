# Narrative notes for the combined empirical paper

**Role:** claims and literature reviewer, 2026-10-03.

These are suggestions for the manuscript owner. Citation keys refer to `references.bib` in this directory. Wording
constraints come from `CLAIMS_SCOPE_REVIEW.md`, and prior-work positioning from `CLOSEST_PRIOR_ANALYSES.md`.

## Framing in one paragraph

A recipient's complete view is whatever is released to it:
- protected features;
- a task output (full logits, probabilities or a decision);
- or both;
- possibly pooled with another recipient's release.

The paper measures, on stored Adult and HMDA models:
- what a recipient recovers about declared protected attributes from that view;
- how useful the same release is for its task.

It is an empirical analysis on repeatedly used development data. It offers:
- no new algorithm;
- no population privacy certificate;
- no composition theorem.

Say all three in the abstract or introduction, not only in the limitations.

## Established findings the paper builds on (cite, do not claim)

- **Linear protection is not nonlinear protection.** `elazar2018adversarial`, `song2020overlearning`,
  `moyer2018invariant`, `ravfogel2020inlp`, `ravfogel2022rlace`, `ravfogel2022kernel`; LEACE states the limitation
  itself (`belrose2023leace`).
- **Outputs leak attributes, also beyond the label.** `fredrikson2014pharmacogenetics`, `fredrikson2015model`,
  `mehnaz2022sensitive`, `wang2019balanced`, `ravfogel2023loglinear`. The baseline discipline comes from
  `jayaraman2022imputation`.
- **Hard labels are not a defense in general.** `mehnaz2022sensitive`; `choquettechoo2021label` for membership.
- **Label dependence forces trade-offs.** `zhao2019inherent`, `zhao2022inherent`, `stadler2024lpl`.
- **Fair-representation certificates** bound demographic parity for classifiers on the representation only
  (`jovanovic2023fare`). FARE itself notes the information loss of restricted encoders.
- **Composition and coalitions leak more.** `ganta2008composition`, `tian2025fairnessprivacy`, `taylor2026collusion`.
- **Softmax shift invariance is textbook algebra** (`goodfellow2016deep` §4.1).

## The five core questions and what the evidence says

| # | Question | Supported answer (with its adverse qualification) |
|---|---|---|
| 1 | **Native check.** What does each method's own protection check establish? | PCRL's own stored linear check fails on 12 of 42 untreated pair × seed combinations. Official LEACE meets its implementation bound on every map, and its held-out ρ₁² is near 0 on one already-used split. FARE's certificate is unavailable or vacuous at these certification sizes. A nonlinear attack does not refute a linear theorem; keep (i) failing one's own test, (ii) recovery outside the test's scope and (iii) an output bypass apart. |
| 2 | **Recovery.** What can a recipient recover from the representation, the output, or both? | Under official LEACE, nonlinear recovery stays at about 0.81–0.87 AUC. Clean outputs restore recovery to about 0.77–0.79 whatever is done to the features (the output bypass). Outputs recover more than the label-only reference. |
| 3 | **Usefulness.** How useful is the released information? | Adult income decisions beat a constant by 3–5 points but recall only 14–22 % of high earners. HMDA underwriting usefulness is not established. The HMDA fair_lending decision is constant, although its scores leak. Low decision-level recovery partly means decisions that say little. |
| 4 | **Compression.** How much of a defense's improvement is ordinary compression? | On the useful-task cell, most of FARE's reduction is reproduced by a same-budget tree without the fairness term. FARE had a smaller additional improvement, 0.0074 AUC [0.0019, 0.0129], below the registered 0.02 target. |
| 5 | **Combination.** What changes when recipients combine releases? | Two purpose recipients pooling outputs recover race better than either alone under full, centred and decision-only contracts (decisions: 0.611 vs 0.576, lower bound of the difference 0.023). This is a measured property of these fitted attackers, not a theorem. |

**The logit-offset ablation** belongs under questions 2–3 as a utility-preserving output-contract change:
- it lowered mean recovery by about 0.07 AUC on both primary cells, with probabilities exactly unchanged;
- it did not appear on the collapsed HMDA seed 1, on 9 of 14 stored pairs, or for multiclass heads;
- a more useful refitted head leaks about as much through its margin alone.

Present it as a property of these trained heads, not as a method.

## Suggested related-work paragraphs

**Concept erasure and its evaluation.**
Linear concept erasure guarantees that no linear classifier can recover a concept: INLP
\citep{ravfogel2020inlp}, R-LACE \citep{ravfogel2022rlace} and LEACE \citep{belrose2023leace}, which achieves this
in closed form with minimal change to the representation. It is well documented that such guarantees do not extend
to nonlinear recipients:
- INLP and R-LACE report nonlinear recovery after linear erasure;
- kernelized erasure protects against the kernel it was fitted for but does not transfer
  \citep{ravfogel2022kernel};
- LEACE restricts its claim to linear adversaries and conjectures that general nonlinear erasure is intractable.

Adversarially trained representations show the same pattern. Post-hoc attackers recover attributes that the
in-training adversary could not \citep{elazar2018adversarial, song2020overlearning}, and independent post-hoc
adversaries are the standard check for invariance methods \citep{moyer2018invariant}. Our measurements of nonlinear
recovery under official LEACE are another instance of this observation, not a refutation of LEACE's theorem. What we
add is the protocol: the method's native check, held-out recovery and held-out utility, measured on the same stored
models.

**Leakage through task outputs.**
Model outputs reveal sensitive attributes. Model-inversion and attribute-inference attacks exploit confidence scores
\citep{fredrikson2014pharmacogenetics, fredrikson2015model}, and label-only attacks can match confidence-based ones
\citep{mehnaz2022sensitive}. Masking confidences does not stop label-only membership inference either
\citep{choquettechoo2021label}. \citet{wang2019balanced} compare attribute leakage from a model's predictions with
the leakage implied by the true labels at matched task performance. That label-only reference is the one we report
beside every output result. After linear guarding, a downstream classifier's predictions can still reveal the erased
concept \citep{ravfogel2023loglinear}. \citet{jayaraman2022imputation} show that many black-box attribute-inference
results do not exceed a model-free imputation baseline, so we report constant and label-only references throughout.

**What a released output needs.**
Softmax probabilities and argmax decisions are invariant to adding a common constant to all logits
\citep[§4.1]{goodfellow2016deep}. Releasing probabilities rather than raw logits is therefore standard and leaves
every task quantity, including calibration \citep{guo2017calibration}, unchanged. Coarsening outputs has also been
studied as a countermeasure, for example by rounding confidences \citep{fredrikson2015model}. Our contribution here
is only a measurement:
- some frozen task heads carried attribute signal in this task-irrelevant offset;
- withholding it lowered measured recovery without changing probabilities;
- the effect is specific to those heads and absent for a refitted logistic-regression head, whose offset is a
  function of its margin.

**Fair representations, certificates and trade-offs.**
Fair representation learning trades task accuracy against protection when the label depends on the protected
attribute \citep{zhao2019inherent, zhao2022inherent}. More generally, any release useful for a task reveals what that
task's label reveals \citep{stadler2024lpl}. Adversarial methods such as LAFTR \citep{madras2018laftr} lack
guarantees against unseen classifiers. FARE \citep{jovanovic2023fare} restricts the encoder to a finite partition and
certifies the demographic-parity distance of any classifier that sees only the partition cell. Its authors note
that restricted encoders lose some information and compare fairness-unaware k-means encoders. We add a same-budget
zero-fairness tree as a matched compression control and measure attribute recovery, not demographic parity, from the
complete release.

**Combining releases.**
Combining independent releases is a classical route to privacy breaches \citep{ganta2008composition}. For fair
models, combining the predictions of a biased and a debiased version of the same model raises attribute inference
\citep{tian2025fairnessprivacy}. Collusion-aware release mechanisms have been proposed for sequential releases over
finite alphabets \citep{taylor2026collusion}. We measure the purpose-permission version: two recipients, each
permitted its own task output, pool their releases. We report it as a measured property of our attackers, not a
composition bound.

## Phrases to avoid, and replacements

| Avoid | Use instead |
|---|---|
| "no significant difference" / "FARE adds nothing beyond compression" | "most of the reduction was reproduced by compression; FARE had a smaller additional improvement, below the registered 0.02 target" |
| "constant head" | "a head whose decision is constant" (and say whether the scores vary) |
| "the offset effect holds on every seed" | "on 5 of 6 encoder seeds; none on the collapsed HMDA seed" |
| "pretrained backbone" | "frozen, seeded, randomly initialised backbone with trained per-purpose adapters and heads" |
| "hard labels protect" / "offset removal is a privacy method" | "a measured output-contract effect for these heads" |
| "first to show" / "no existing method can solve this" | "to our knowledge, not reported together in one protocol" (only after the targeted search in CLOSEST_PRIOR_ANALYSES §4) |
| any R²-to-accuracy bound; "fixed on main" | omit; "a repair is on an unmerged branch" |
| AUC differences called "points" | "AUC" in AUC units; accuracy in accuracy points, never on one unlabelled axis |
