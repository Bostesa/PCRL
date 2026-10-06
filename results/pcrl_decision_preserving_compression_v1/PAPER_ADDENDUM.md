# Paper addendum: decision-preserving compression of released confidence (exploratory development benchmark)

*Status.* This is a locked development study on previously used Adult rows. Label: **EXPERIMENTAL_NO_ADVANTAGE**. Read it as a negative result under a fixed contract. It supports no novelty claim and no population guarantee.

## Setting

Two recipients each receive a release for one fixed purpose:
- income (2 classes);
- occupation group (6 classes).

Both purposes come from frozen teachers with deployed heads:
- U, the task-only network;
- RAW-J β 0.3, the earlier joint penalty network.

**The release.** Instead of the teacher's probability vector p_i, recipient i receives a public code: a token C_i = g_i(p_i), the token's public decoded probability vector q(C_i), and the decision. Two properties hold:
- g_i is deterministic and never merges vectors with different predicted classes;
- q(C_i) has its argmax at the teacher's class.

So every decision equals the teacher's, and accuracy, confusion matrices and class recalls are unchanged. This holds on all 258 fitted codes and every row of every role, 20.2 million recipient-row checks, verified independently.

## How the codes are built

1. Fine cells: per-class KL k-means (at most 16 cells per class).
2. Coarse maps with at most m ∈ {2, 4, 8} tokens per predicted class, fitted on OSF_DEFENSE_FIT.
3. The fitting objectives are mean KL distortion D_i, plus plug-in mutual information with SEX:
   - task-only: D_1 + D_2;
   - local: D_1 + D_2 + λ(I_1 + I_2)/2;
   - joint: D_1 + D_2 + λ((I_1 + I_2)/2 + I_12);
   - two sequential orders: the second map is fitted against the first.

   Here λ ∈ {0.1, 1, 10}.

Plug-in MI is a training criterion only. All claims rest on held-out attacks:
- the fixed FINAL attack slate (logistic regression, MLP and gradient boosting, plus domain-adaptation members);
- prior-weighted cell readers on finite views;
- composed readers that attack a continuous source through every code fitted from it.

## Contract and family

**Inner-selection gates** (relative to U, on every seed):

| Quantity | Gate |
|---|---|
| Accuracy | ≥ U − 0.01 |
| True-label log loss | ≤ U + 0.01 nats |
| Brier | ≤ U + 0.005 |
| Gain retention | ≥ 0.8 |
| Gain | ≥ 0.03 |

**Primary family:** three 11-clause conjunctions, joint versus matched task-only, joint versus strongest non-joint, and privacy-trained versus strongest privacy-untrained.

| Component | Setting |
|---|---|
| Size | 33 slots |
| Critical value | z = 3.17 |
| Bootstrap | 1,999 exact-record-group resamples |

## Results (assessment, 13,936 rows; means over three seeds)

| Release | Pair SEX AUC | Occupation recipient AUC | Occupation log loss | Occupation Brier |
|---|---:|---:|---:|---:|
| U continuous (features + scores view) | 0.883 | 0.878 | — | — |
| U continuous (released interface) | 0.858 | 0.856 | 1.269 | 0.652 |
| U task-only code, m 8 | 0.825 | 0.786 | 1.289 | 0.660 |
| U joint code, m 8, λ 0.1 | 0.807 | 0.764 | 1.290 | 0.660 |
| U sequential codes, m 8, λ 0.1 | 0.807–0.809 | 0.752–0.762 | 1.291–1.292 | 0.660 |
| U class-only (decisions) | 0.739 | 0.687 | 1.379 | 0.693 |
| FARE (official) | 0.704 | 0.636 | 1.315 | 0.676 |

**Eligibility.** No finite code met the score contract on every inner seed. Occupation confidence is the binding constraint: the joint code costs +0.021 nats [0.015, 0.028] and +0.008 Brier [0.006, 0.010] against U. Income stays within +0.002 nats.

The selected comparator is therefore U's continuous output. The joint, privacy-trained and matched-control roles had only descriptive fallbacks, so every primary clause is descriptive. All three claims are NOT_ESTABLISHED.

**Descriptive comparisons:**
- Joint versus matched task-only coding: 0.018 [0.014, 0.021] pair AUC, short of the registered 0.02 margin.
- Joint versus sequential: within 0.002.
- Joint versus the continuous scores: 0.051 [0.045, 0.058] lower pair AUC.

## Interpretation (bounded)

- On this benchmark, the joint decision vector alone gives SEX AUC about 0.74. A decision-preserving release cannot go below that, whatever it does to confidence.
- The confidence-borne leakage that coding removes (about 0.05 pair AUC) is paid for in occupation calibration at roughly twice the registered allowance.
- Coordinated fitting (joint or sequential) moves the trade-off by about 0.018 pair AUC at matched rate. That difference is not established and is not separable from sequential fitting.

## Relation to prior work (no novelty claim)

The components are established:
- agglomerative information-bottleneck merging (Slonim and Tishby 1999);
- multivariate bottleneck merging (Slonim, Friedman and Tishby 2001);
- privacy-funnel merging against an unobserved attribute (Makhdoumi et al. 2014);
- sequential releases under collusion (Taylor, Vippathalla and Coon 2026);
- confidence-score transformation as a defence (PURIFIER; Yang et al. 2023);
- limits of least-privilege learning (Stadler et al. 2024).

Neither PURIFIER nor the Taylor et al. solver could be compared as an official method:
- The PURIFIER repository is empty.
- No code for the Taylor et al. solver was found.
- Faithful mappings would change either method.

Our sequential arms are matched, class-preserving adaptations, not the Taylor et al. algorithm. The study therefore makes no claim relative to either method.

## Limitations

- **Reused data.** Adult was used before, and the intervals do not correct for adaptive research history.
- **Fixed teachers.** The two teachers are fixed, and both have near-zero recall on two occupation classes.
- **No guarantee.** The attack slate is finite, and plug-in MI is not a privacy guarantee.
- **Custody.** The backup is same-device only (the drive was absent), and predecessor custody repair is pending.
