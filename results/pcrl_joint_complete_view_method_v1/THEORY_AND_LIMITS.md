# Theory and limits (before the real-data lock)

The independent mathematical review is in `review/THEORY_REVIEW.md`, with proofs, counterexamples and numerical checks. Every statement below is an **established identity or a standard fact**. None is a new PCRL theorem, and none certifies the measured privacy.

## Statements

1. **Concatenation.**
   - **Statement.** If `Cov(r_i, S_onehot) = 0` for each i *under the same distribution*, then `Cov([r_1, r_2], S_onehot) = 0`. The cross-covariance of a concatenation is the stack of the blocks.
   - **What it means here.** LEACE enforces this on the defense_train rows at refit time only. Held-out moments are measured separately (`NATIVE_VS_AUDIT.csv`). Fitted training moments do not establish population moments.
2. **Post-processing.**
   - **Affine.** Affine post-processing preserves (1): `Cov(Ar + b, S) = A Cov(r, S)`. Centred logits of an affine head are affine in `r`, so the primary release is linearly guarded on the fitting rows.
   - **Nonlinear.** Softmax probabilities and hard argmax decisions do not inherit it. In the reviewer's counterexample, a guarded `r` with `S`-dependent variance yields `softmax(r)` correlated with `S`. These formats get their own empirical audit.
3. **Adding a view.**
   - A Bayes-optimal attacker on `[v1, v2]` can ignore `v2`, so ideal recovery cannot decrease when a view is added.
   - Finite fitted AUC can decrease. The coalition bank therefore contains the selected v1-only and v2-only attackers (ignore-other-view), selected on attacker_val.
4. **Known head.** A fixed, known head computed from `r` adds no information beyond `r`. It can still help a finite attacker, so the primary view includes it explicitly.
5. **Joint vs sequential.**
   - Joint optimisation weakly dominates sequential optimisation only when its feasible family contains the sequential candidates under the same constraints and information.
   - That says nothing about a nonconvex local optimiser, and any advantage over S12/S21 partly reflects foreknowledge of both purposes.
6. **Local caps bind.**
   - Removing information from recipient 2 cannot reduce what recipient 1 reveals alone: `R_v1` depends only on `g1`.
   - There is no cost-free reallocation, so a coalition gain must not come with a local loss beyond the registered allowance (0.01 AUC).
7. **Projection.**
   - `min ½‖d−p‖²` s.t. `a_j·d ≤ 0` has a unique solution, found by exact active-set enumeration with certified residuals (`jcv/project.py`).
   - Checks: it agrees with scipy SLSQP to ≤ 2.5e-13 on 5,900 instances (reviewer), and with our test suite on 400 random cases plus zero, parallel and anti-parallel guards.
   - With separate encoders `a_1·a_2 = 0`, so the exact projection equals A-GEM applied to each encoder.
   - **Scope of the guarantee.** It covers first-order non-increase of each active guard loss, for the *projected privacy component*, on the *guard-minibatch gradient estimate*. It does not cover:
     - the unprojected task component, which is computed on a different minibatch;
     - finite steps (backtracking handles these empirically);
     - changing critics;
     - LEACE refits;
     - the final head refit.
   - **Final held-out utility tests decide.**
8. **Surrogate `R_v`.**
   - It equals Song et al.'s C2 with an empirical marginal, maximised over a finite bank.
   - It is nonnegative on the fitting rows, where the constant prior is in the bank, but can be negative on held-out rows.
   - It is not mutual information, not an AUC bound, and weak critics do not establish privacy.
   - `H_fit(S)` is one constant across views, so dividing by it does not change the relative weights of the views.
9. **XOR fixture** (`jcv/fixtures.py`). Take A, B independent Bernoulli(½) bits and S = A xor B.
   - Each bit alone carries no information about S; the pair determines it.
   - Releasing (Y1, A) to recipient 1 and (Y2, B) to recipient 2 leaks S only to the coalition. Dropping A or B keeps both tasks and removes the clue.
10. **FARE coalition.** Two FARE trees form a restricted encoder with at most k1·k2 cells. FARE's own certificate procedure therefore bounds `AUC_coalition ≤ ½ + T` for any attacker, with probability 1−ε, when run on rows unused by either tree. The neural arms have no analogue.

## Executable fixture results (before the lock; `jcv/tests`, 9 pass)

**XOR attacker positive control.**
- Each bit alone: AUC ≤ 0.515 for linear, MLP and tree attackers.
- The pair: linear 0.516, MLP 1.000, HGB 1.000.

**XOR training positive control** (recipient 1 sees `(Y1, A, noise)`, recipient 2 sees `(Y2, B, noise)`; β = 10). Fresh inner-slate attackers:

| Arm | Coalition recovery | Local recovery | Tasks |
|---|---|---|---|
| L | 1.000 | ≈ 0.50 | — |
| J | **0.519** | ≈ 0.50 | 1.000 / 0.989 |

**Task = S.** J's task-1 accuracy falls to 0.509. LEACE removes the task signal, so preserving the task and hiding S is impossible. The runner reports the conflict; a utility gate fails, and no threshold is repaired.

**Constant release.** Recovery is 0.500 and the gain is 0: withholding, which fails the usefulness gates.

**Other tests.** Gradient signs, seed aliasing, score centring, affine centred logits and projection residuals all pass.

## Technical repairs made on fixtures (logged; before any real fit)

1. **The XOR training fixture's input masks.** My first version gave both encoders all bits, which is not the registered fixture. Per-encoder masks are used in fixtures only; the real study gives both encoders the same permitted X.
2. **Unwhitened critics could not detect low-variance encodings.**
   - The coalition critic's R stayed near 0 for 20 epochs, while an audit HGB found AUC 0.92.
   - Cause: A was encoded along a direction whose linear read-out needed weights of about 200.
   - Fix: critic inputs are whitened.
3. **Per-epoch PCA whitening amplified encoder drift.** Fix: ZCA whitening (continuous in the covariance), refreshed before every step from a fixed reference subset.
4. **Critics too slow against the moving encoder.**
   - With 2 critic steps per encoder step, the encoder kept fooling the online critic, and fresh attackers still recovered 0.97–0.98.
   - Registered fix: 5 critic steps per encoder step, critic learning rate 3e-3. Coalition recovery then fell from 1.00 to 0.52 on the fixture.

**This study also illustrates the limitation of adversarial training.** An encoder can satisfy weak online critics without removing information (repair 4). Only fresh attackers decide.

## What no statement here certifies

- No population, mutual-information or differential-privacy statement is made for the neural arms.
- LEACE's guarantee is linear, holds on the fitting rows, and covers each released feature vector (and, by (1)–(2), the affine primary view) at the time of the final refit.
- Nonlinear recovery of SEX through features or outputs is outside LEACE's theorem. Measuring it is not a refutation of LEACE.
