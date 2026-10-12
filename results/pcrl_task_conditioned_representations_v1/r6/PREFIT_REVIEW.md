# Independent scientific audit scope

R6 read the complete execution handoff and independently inspected the pinned
official MMD-B-Fair source at commit
`280a178abc86802cef58f99ab3eff4d2365ddacb`, including `utils/mmd.py`,
`trainer/frep_eq.py`, and `config/eq/adult/lambda_1.yml`.
The independent module imports no candidate scientific models, kernels,
objectives, trainers, metrics, selectors or deployment code. Its only shared
runtime import is the authenticated mature queue guard.

The literal official Adult variance branch is
`4*(mean(rowmean(H)^2)-mean(H)^2)`. It ignores hypothetical block size. The
paper's variance formula includes division by block size; the registered
adaptation preserves the code proxy and explicitly disclaims calibrated
power-probability interpretation. The off-diagonal MMD removes the cross
diagonal; the variance H includes its diagonal. Target power has negative
coefficient one, cross entropy positive coefficient one. Six-class pairwise
frequency weighting, actual wire kernels, coordinate normalization and aligned
coalition product kernels are named adaptations.

R6 separately derived the scaler/logistic fold: W=A/s, b=c-W*mean. Binary
logits are (-d/2,+d/2); multiclass logits center the actual fitted coefficient
rows. Float32 encoder arithmetic and float64 folded heads preserve the source
service. Historical source encoder/head fitting exposed every FIT row, and
source head C selection exposed HEAD aggregates. New fold 2 selection is
internal development selection and inherits source fitting exposure. Ucal
comes from the historical global inverse-temperature calibrator, with its
historical calibration/family selection exposure disclosed; no old person
arrays are needed.

The proposed training strata need at least 32 distinct groups in each SEX arm,
use fixed fold 0/1 frequencies without renormalizing supported mass, and use a
separate population task minibatch. Unsupported pair mass remains uncovered.
Fixed detached FIT scaling references refresh at epoch start and retain every
coordinate with predefined floors. Absolute final SEX recovery remains the
empirical requirement.

The required common two-seed configuration uses FIT fold 2 only. Among task
surrogate feasible configurations (worst normalized task deviation <=1), a
protected family minimizes its fitting sensitive diagnostic, with the frozen
task/tie order. Its diagnostic fallback minimizes worst task deviation alone
before NLL/ties. Utility families minimize task NLL among feasible candidates.
The exact LOCAL_TWIN uses JOINT's strength and epoch. Every family, both
sequential orders, source references and fallback hashes must freeze before
any INNER task labels. All reader banks must freeze before INNER SEX.

Sequential second branches consume 5+10+20 epochs after one first-stage
20-epoch path; searched total is 55 encoder epochs versus simultaneous 40.
This gives the controls 37.5% extra search computation. Frozen first encoder
and its head must remain exactly unchanged within each second branch.

R6 raised two stale R3 plan statements before science: protected nomination
was described as minimizing task deviation even when feasible, and sequential
search compute was described as 40 aggregate epochs. The coordinator and R3
confirmed the corrected selector and branch accounting. Their effective lock
must contain the corrected definitions before any protected fit.

Queued independent prefit numerical evidence is pending. The worker checks
source/role/schema hashes, useful U0 and Ucal forward parity on all three
admitted roles, FIT metrics/strata, authentic official-core parity, reciprocal
scale handling, conditional no-extra-signal/planted/XOR/alignment fixtures,
metric conventions, and AUC tie handling. It refuses direct unqueued launch
and never opens INNER task labels or INNER SEX during prefit admission.

Independent final replay will cover actual new releases, point gates,
FIT selections, all fixed-reader AUC envelopes, paired group bootstraps,
separate score-only endpoints, and restore outputs. Prediction replay is not
independent reader refitting. A small independent reader refit, if admitted by
the queue reservation, will be identified by exact arm/seed/view/model scope.
