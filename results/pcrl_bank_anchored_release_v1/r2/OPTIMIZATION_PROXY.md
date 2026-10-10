# Bank-anchored channel fitting proxy (R2 draft)

Status: **prefit design input for R0's effective lock**. No new FIT row, SEX
label, checkpoint or INNER outcome was opened to prepare this document. The
selected source bank is W_TSTAR unless R1 finds a mechanical incompatibility.
This is exploratory development on repeatedly used Adult people.

## Distinct fitting and reporting comparisons

The inherited sampled-channel optimizer compared each candidate's FIT mean
NLL and summed multiclass Brier with the lower (more demanding) of the U0 and
Ucal FIT references plus 0.010 and 0.005. It had **four** fitted utility caps;
accuracy was structural because every action kept the source decision. Under
that old rule, B0's occupation U0 NLL excess was 0.014312011 (seed 0) and
0.015420556 (seed 1), versus 0.010. Seed 1 occupation Brier excess was
0.005348077 versus 0.005. This was a real rejection under the old fitting
reference, not an arithmetic error. The exact old margins for both tasks,
references and roles remain in the pinned predecessor
`pcrl_code_utility_witness_v1/CHANNEL_IDENTITY_AUDIT.csv`.

This new fit instead uses **the immutable B0 release on identical ordered FIT
people** as its only optimization utility reference. For each recipient,
source seed and task, an eligible checkpoint must satisfy inclusive

* `accuracy(B0)-expected_accuracy(q) <= 0.002`;
* `expected_NLL(q)-NLL(B0) <= 0.001` natural-log nats;
* `expected_summed_multiclass_Brier(q)-Brier(B0) <= 0.0005`.

Expected cost is `mean_i sum_(z1,z2) q_i(z1,z2) cost(fixed_bank_z,y_i)`, **not**
loss of a mean probability. B0 has exactly zero excess. The same source bank,
strict first-argmax decision and class-preserving action sets make expected
accuracy identical to B0 when implemented correctly. The implementation
checks this identity, rather than omitting the .002 clause. NLL clips only
the true-class probability at 1e-12 for scoring; Brier sums all class terms.
Weights are uniform per FIT person, with duplicate-group IDs retained for
later uncertainty. Every public action pair with the same global code pair
has its probability aggregated before costs and recovery are computed.

This is a **new prospective fitting formulation**. It does not assert that
.001/.0005 FIT tolerances imply any held-out or population utility guarantee.
For every resulting policy we also report its old-style FIT excess versus U0
and Ucal, solely as diagnostics; an old-style failure does not exclude a
policy from this study's new proxy.

Original common-INNER evaluation remains unchanged: every task/seed must
meet accuracy loss <=0.010, NLL excess <=0.010 and summed Brier excess
<=0.005 against **both** U0 and Ucal on the same INNER people. Point
eligibility is separate from simultaneous confidence intervals. R3 owns the
original-cap endpoint calculation, exact-law primary estimand and cached
single-release secondary results. INNER outcomes are never used to select
among FIT checkpoints.

## Source objective and feasible identity

The source fixed bank has 16 income and 321 occupation global public states
per seed. Input fine cells number 64 and 641. Each input cell's fixed support
is its exact bank action, two nearest same-class outputs by symmetric KL,
then the class medoid, deduplicated in that order. Every recipient has at
most four actions, so each joint row has at most 16. The central J gate uses
both frozen U0 probability vectors, clipped entropy and top margin for each:
12 inputs, 16 tanh hidden units, 16 padded pair logits (`4*a1+a2`), 480
parameters. R1 verifies the bank and routing; no score, cell or local action
position is transmitted. The complete transmitted state is one ten-bit
global code, its fixed decoded probability and decision per recipient. There
is no emitted fallback or per-person metadata field; public weights, bank and
source identities remain side information for defense-aware readers.

**Public ID repair before FIT:** the inherited channel source sorted decoded
bank vectors by class and probability bytes, whereas W_TSTAR's existing ten-bit
wire names original QPC token rows. The source action construction is first
run in its inherited sorted namespace to retain neighbor/medoid tie rules.
Its unique old-token→sorted-ID map is then inverted and all bank rows/actions
are converted to the original W_TSTAR token namespace before fitting or
release. The required identity check is `action[cell][0] == original
cell_token[cell]` for every fine cell, and the original table row must match
the source decoded vector exactly. Duplicate bank rows that prevent this
bijection block admission. R1/R5 separately check actual raw-wire parity.

The inherited protected objective is exact finite empirical-law SEX Bayes
accuracy `G_pair + 0.5*(G_r1+G_r2)`, with each local `G_r` constrained to be
at most its **B0 fitted-law value**. This is a fitting objective and local
guard, neither an AUC bound nor a fresh attack result. There is no hard
required pair-privacy improvement. T is privacy-label-blind and minimizes
empirical `I(X;(Z1,Z2))` under the same bank-relative utility caps; its FIT
worker must not receive SEX. D uses the inherited deterministic central
objective/action support. P factorizes conditional on both central scores
with **independent recipient coins**; L uses recipient-local scores/cells;
S12/S21 freeze the first fitted release, then condition the second on the
actual first public code. J uses the joint categorical law. B0 is an
immutable identity control, not a fitted protected policy.

All stochastic families get an **executable exact-identity boundary**.
For J and T it is `q_eta=(1-eta)*delta_B0 + eta*q_theta`, with `eta=0`
allowed exactly. For P and L each recipient uses its own independent
`(1-eta_r)*delta_B0_r+eta_r*q_theta_r`, then the two marginals multiply;
a shared branch coin is prohibited. For sequential designs the first-stage
identity is a first-stage checkpoint and remains frozen; the second-stage
identity completion is an exact conditional action. D has a separate hard
identity checkpoint, not a stochastic mixture. R2's data-free tests must
verify exact B0 codes, probabilities and decisions, law normalization,
factorization, duplicate aggregation, both costs, and zero proxy excess.

The inherited numeric recipe is a full-FIT float64 Adam augmented-Lagrangian
heuristic, two starts (`identity_biased`, `uniform_logit`), learning rate .01,
rho 100, gradient clip 5, zero weight decay, one numerical thread,
checkpoint/dual interval 25 updates, 300 updates per start for central J/T/D/P,
150 per recipient or stage for L/S12/S21, and a matched whole-arm 300 CPU
second ceiling from the effective old producer amendment. R0 must seal any
necessary timing-scale change **before** FIT, not after outcomes. The new
boundary `eta` and exact-checkpoint rule are engineering changes to represent
B0 and search adjacent feasible laws, not a new coefficient grid. Each arm
gets one selected policy per seed, using only FIT/internal admitted quantities:
feasible checkpoints first, then the arm's objective, then start index, then
checkpoint update, then exact identity on an exact tie. Preserve all
checkpoints, infeasible starts and stopped attempts.

This finite-law objective is distinct from assessed fresh recovery. Only
original-INNER point-eligible fixed arms proceed to R4's ATTACK_FIT_NEW to
INNER complete-view linear, nonlinear, categorical and defense-aware attacks
for each recipient and aligned pair. Benefit is measured against the
strongest required eligible comparator, with local worsening separately
guarded. Identity is a feasible **fit starting point**, not a privacy win.

## Source anchors for R0 seal

* Source design `sampled_channel_v1/design/DESIGN_LOCK_CANDIDATE.json` and
  effective `AMENDMENT_20261010_A_PRODUCER_STAGE_FIT_PATH_CORRECTION.json`.
* Source methods `sampled_channel_v1/impl/{bank,channel,fit}.py` and
  `sampled_channel_v1/controls/{deterministic_trainer,factor_controls,fit_local,fit_product,seq_train,task_rate}.py`.
* Witness audit `pcrl_code_utility_witness_v1/r2/R2_UTILITY_AND_CHANNEL_AUDIT.md`,
  `CHANNEL_IDENTITY_AUDIT.csv` and `CHANNEL_IDENTITY_RECEIPT.json`.

R0's source index binds exact content hashes and the final numeric plan. This
draft is inert until its source, package interface, checkpoint cadence and
budget are included in the effective campaign seal.
