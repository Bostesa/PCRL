# Paper addendum: confidence capacity of decision-preserving codes (exploratory development benchmark)

*Status.* A locked development comparison on previously used Adult rows. Label: **CONFIDENCE_FEASIBILITY_ESTABLISHED**, with no method advantage established. It supports no novelty claim and no population guarantee. The exposure statement is in `EXPOSURE_LEDGER.md`.

## Setting

Two recipients receive releases for income (2 classes) and occupation group (6 classes), computed from the frozen task-only teacher U.

**The release.** Each recipient receives a categorical token, its public decoded probability vector and the teacher's decision. Tokens never cross predicted classes, and every prototype's argmax is the teacher's class. So every decision, accuracy and class recall equals U's: 66 policy units, 5,170,440 recipient-row checks.

**Two stages.**
- **Stage A** fitted task-only KL k-means codes directly on the defence-fit rows, at income caps {4, 8} × occupation caps {8, 16, 32, 64} per predicted class. It used three fixed starts and a ≤ 200-round convergence rule. A1 reproduced the earlier 20-round code exactly, then let it converge.
- **Stage B** ran at the single rate that passed the inner confidence gate (8, 64). It compared task-only FINE-TASK compression with LOCAL, two SEQUENTIAL orders and JOINT privacy objectives:
  - F_local = D1 + D2 + λ(I1 + I2)/2;
  - F_joint = D1 + D2 + λ((I1 + I2)/2 + I12);
  - λ ∈ {0.01, 0.1, 1}, using plug-in MI with SEX on the fitting rows.

  Sequential stage one is fitted against the other recipient's class-only release under F_joint.

**Recovery** is held-out SEX AUC from a fixed attack slate:
- logistic, MLP and gradient-boosting models, defence-aware readers, and token-conditional readers;
- fitted and selected on separate roles;
- continuous scores attacked through every fitted public code.

## Results (assessment, 13,936 rows; means over three seeds)

| Release | Pair AUC | Occupation recipient AUC | Occupation LL excess vs U | Occupation Brier excess vs U |
|---|---:|---:|---:|---:|
| U continuous | 0.858 | 0.856 | 0 | 0 |
| Task-only, 8 occupation states | 0.820 | 0.787 | +0.0210 | +0.0077 |
| Task-only, 16 | 0.834 | 0.807 | +0.0107 | +0.0044 |
| Task-only, 32 | 0.842 | 0.826 | +0.0057 | +0.0026 |
| **Task-only, 64 (Q\*)** | **0.849** | 0.834 | **+0.0030** | **+0.0017** |
| FINE-TASK, 64 (T\*) | 0.846 | 0.831 | +0.0031 | +0.0014 |
| LOCAL, 64, λ 0.1 | 0.834 | 0.800 | +0.0048 | +0.0019 |
| SEQ-12 / SEQ-21, 64, λ 0.1 | 0.816 / 0.816 | 0.782 / 0.771 | +0.0082 / +0.0084 | +0.0032 / +0.0033 |
| **JOINT, 64, λ 0.1 (P\*)** | **0.813** | 0.782 | **+0.0081** | +0.0029 |
| Decisions alone | 0.739 | 0.687 | +0.1102 | +0.0411 |
| FARE (official) | 0.704 | 0.636 | +0.0460 | +0.0233 |

Income stays within +0.0018 nats for every code at 8 income states.

## Registered outcomes

37-slot family; z = 3.2048; 1,999 exact-record-group bootstrap replicates.

- **Q\* confidence feasibility: PASS** on all four bounds. Occupation log loss +0.00295 [−0.0002, 0.0061]; Brier +0.0017 [0.0007, 0.0027].
- **Privacy versus task-only at equal capacity (claim C): NOT_ESTABLISHED, 10/11 clauses.**
  - Pair recovery is 0.0336 [0.0286, 0.0387] lower, which passes.
  - The occupation log-loss upper bound is 0.0121 against 0.01, which fails.
- **Joint versus the strongest non-joint control (claims A and B):** no eligible joint nominee. Descriptively, joint is within 0.003 of sequential.

## Interpretation (bounded)

**On this benchmark:**
- The occupation confidence loss of earlier decision-preserving codes was a capacity effect, not an optimisation artefact. At 64 states per class the codes keep confidence within the registered allowances, but recovery is then close to that of the continuous scores.
- Privacy objectives trade about 0.03 pair AUC for about 0.005 nats. That is a better trade than lowering capacity, but it stays outside the registered bound.
- Coordinated (joint) optimisation adds nothing measurable beyond sequential optimisation.
- The decision vector alone leaves pair AUC about 0.74, measured by the declared attackers.

**What none of this is:** fitted MI decreases are training criteria, not protection; there is no population or differential-privacy guarantee; and the larger codebook is not an algorithmic contribution.

## Relation to prior work

The components are established:
- agglomerative and multivariate information bottleneck;
- privacy funnel;
- output/confidence transformation (PURIFIER);
- sequential releases under collusion (Taylor, Vippathalla and Coon 2026).

PURIFIER's official repository remains empty, and no Taylor et al. solver code exists (checked 2026-10-06). Our sequential arms are matched adaptations, not that solver. No superiority over either is claimed.

## Limitations

- **Data:** reused rows; intervals condition on fitted artifacts and do not correct for the adaptive history.
- **Search:** JOINT is a local search with documented small optimiser gaps.
- **Rates:** one selected rate only.
- **Bias:** composed source recovery is a maximum over 22 codes and is selection-optimistic.
- **Custody:** the backup is same-device only (drive absent), and predecessor custody repair is pending.
