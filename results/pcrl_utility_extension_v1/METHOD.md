# METHOD — `pcrl_utility_extension_v1`

Implementation: `experiments/pcrl_utility_extension_v1/` (`extension.py`, `program.py`,
`calibrate.py`, `tier0.py`, `tier1_reanalysis.py`, `scheduler.py`). Infrastructure:
`infra/pcrl_utility_extension_v1/`.

## 1. Release and the inclusion fact
Release: `wire/A = [H_A, Z_J, R]`, `wire/B = H_B`, `wire/AB = [H_A, Z_J, R, H_B]`, float64.
`H_A`, `H_B`, `Z_J` are the stored release arrays, asserted byte-identical to the untouched J
release on every pool. `R = g(x)` uses only A-side inference inputs (PCA_32 through J's frozen
standardiser); no protected label, no row identifier and no `H_B` at inference.

**Inclusion fact.** Let V0 = (H_A, Z_J) and V1 = (V0, R). For any target Y and any loss, the Bayes
risk under V1 is at most that under V0, because every predictor q(V0) is also a predictor of V1
that ignores R: inf_{q(V1)} E l(Y, q) <= inf_{q(V0)} E l(Y, q). Taking Y = residence gives:
appending R cannot reduce optimal population utility. Taking Y = S (sensitive) and log loss gives
H(S | V1) <= H(S | V0), i.e. I(S; V1) >= I(S; V0): appending R cannot reduce optimal sensitive
recovery. A deterministic append is therefore never a privacy amplification of J; the target is
extra useful capability with a bounded **empirical** disclosure increase. Finite fitted probes can
move either way (a larger input can make a finite learner worse), which is why untouched-J
predictor candidates are part of every comparison (ref_J is audited under the same slate).

## 2. Utility proxy without residential labels
D0: (H_A, Z_J) -> Z_A0 (frozen A0 feature vector), fitted on representation-fitting households
(p0_fit U mapper_fit), monitor held out. Training rows get **out-of-fold** residuals
T = Z_A0 - D0_oof(H_A, Z_J) from a 5-fold household cross-fit; every other use (monitor,
validation, test, deployed controls) uses the single **final** D0. The out-of-fold residuals are
not available at inference and are never used as a lookup.
D1 = D0 + E(H_A, Z_J, R) with E zero-initialised, so D1 starts exactly at D0 (the no-extension
baseline is explicit). Loss: mean ||T - E||^2 / var_T with var_T fixed from the training
residuals. Reconstruction is a capability proxy only; a better teacher reconstruction is not
automatically better residence transfer.

## 3. Protection objective
For trainable role j, the baseline view includes J: local (H_A, Z_J), coalition (H_A, H_B, Z_J).
p0J_j is fitted on the p0_fit household fold and frozen. Differentiable attackers are corrections
to p0J_j's logits on [view, R], so the incremental gain is zero at initialisation by construction.
Objective: recon + beta * policy_penalty(G) with the predecessor's L1 / L2 / C1 definitions and
slot schedules. Subtracting the fixed baseline cross-entropy adds no gradient; no benefit is
claimed from it. Low finite-family gain is not conditional independence or an MI bound.

## 4. Controls
* **Unprotected learned extension** `U_r`: beta = 0 (C1 attackers train alongside, weight zero).
* **Deployable PCA residual** `P_r`: R = (Z_A0(x) - D0_final(H_A, Z_J(x)) - mu) V_r with mu, V_r
  fitted on mapper_fit rows. Both Z_A0(.) and D0_final(.) are functions of A-side inputs, so this
  is a deployable map; the audited function is exactly this map.
* **LEACE on R** `LEACE_r` (T3): predecessor rank-stabilised LEACE fitted on U_r's R only (joint
  one-hot SEX / RAC1P / public_coverage, complete cases of representation_fit); realised
  projection rank and class support recorded; rank zero is reported, not replaced.
* **Untouched J** (`ref_J`) and **LEACE on A0** (`ref_leace_A0`) re-audited under the same slate.

## 5. Non-claims
No confidence guarantee from screening thresholds; no privacy theorem; no claim that a J-relative
increment near zero means no information remains; negative empirical increments do not undo
disclosure; 2018/2017 are repeatedly used development pools.
