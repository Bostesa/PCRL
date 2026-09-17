# Method and mathematical scope

This document states what the residual spectral mechanism computes, which results are theorems (with their assumptions), and which claims are only empirical. The full frozen numerical specification, fixtures and pre-fit alignment notes are in the development study's [`MATHEMATICS.md`](../redesign_20260910_acs_residual_spectral_v1/MATHEMATICS.md). Nothing here was changed after 2017 outcomes were opened.

## 1. Interface and threat model

A person has covariates X and sensitive attributes S (SEX, 2 classes; RAC1P, 9 classes). Two fixed service predictors already exist:

* H_A(X) in [0,1]^4: complete probability vectors for income > $50,000 and civilian employment, released to recipient A;
* H_B(X) in [0,1]^2: the public-coverage probability vector, released to recipient B.

These vectors are released **bitwise unchanged** in every condition. The mechanism adds a channel Z = g(X, H_A) in R^16 **only for A**. Recipient views: A sees (H_A, Z); B sees H_B; the coalition AB sees (H_A, Z, H_B). Every published function (preprocessing, spectral map, public auxiliary heads of historical neural channels) is in audit scope. The policy authorizes A to predict income, employment and residence; B to predict coverage and commute; it forbids A from coverage, commute, SEX and race; B from income, employment, residence, SEX and race; and AB from SEX and race. The attacker has labeled data from the same release year, chooses among fixed model families by validation loss, and sees only its legal view.

Exact output preservation is a structural property: the channel is appended, never mixed into H. It does **not** mean the service predictions stay accurate after population shift; 2017 service quality is measured separately.

## 2. Construction (per seed; all fitting on the 2018 representation-fit pool)

1. T = frozen PCA32 coordinates of the encoded covariates (the "teacher").
2. phi(T) = [standardized T, sqrt(2/96) cos(T_std Omega + b)] in R^128, with Omega ~ N(0, sigma^-2), sigma the median pairwise distance on a seeded 1,024-row subset, b ~ U[0, 2pi). This approximates a Gaussian kernel with bandwidth sigma (Rahimi-Recht random Fourier features). It is a fixed finite feature map, not a learned encoder.
3. q_A(H_A): intercept plus all monomials of degree <= 2 in the two positive-class probabilities, standardized on training rows, constant/duplicate columns removed at 1e-12. q_AB adds the coverage probability.
4. Residualize by SVD least squares (relative cutoff 1e-10): V0 = phi - q_A B_phi, R = T - q_A B_T; center; eigen-whiten V0 on eigenvalues > max(1e-12, 1e-10 lambda_max) to V with V'V/n = I.
5. U = C_VR C_RV with C_VR = V'R/n; U_norm = U / tr(U).
6. Protection moments. For target j with one-hot Y_j (all K classes), nuisance m_j(H) estimated by household-grouped 3-fold cross-fitted multinomial logistic regression on the basis (C = 1, lbfgs, converged), and each basis column b_k:
   G_jk = (1/n_j) sum_i V_i b_k(H_i) (Y_ji - m_j(H_i))'
   P_j = sum_k G_jk G_jk' / tr(...) (zero and flagged if the trace <= 1e-12).
   Local roles (basis q_A): SEX, race, coverage; coalition roles (basis q_AB): SEX, race; marginal roles: constant basis and training prior. P_L, P_AB, P_M are the equal averages over their roles (denominators 3, 2, 3 fixed even when a role is zero).
7. W = top-16 eigenvectors of A_arm (table below); Z = W'V with canonical sign.

| Arm | A_arm |
|---|---|
| S0 | U_norm |
| M025 / M1 | U_norm - 0.25 P_M / U_norm - P_M |
| L025 / L1 / L2 | U_norm - {0.25, 1, 2} P_L |
| C025 / C1 | U_norm - {0.25, 1} (P_L + P_AB) |

Inference uses only T and H_A; H_B enters fitting (coalition moments) and audits, never the released map.

## 3. What is proved, and under which assumptions

**Proposition 1 (fixed-matrix optimum; Ky Fan).** For symmetric A in R^{d x d} with eigenvalues a_1 >= ... >= a_d and r <= d, max over W'W = I_r of tr(W'AW) equals a_1 + ... + a_r, attained by the top-r eigenvectors. *Proof sketch:* with D = E'W, tr(W'AW) = sum_i a_i ||D_i||^2 where each row weight lies in [0,1] and the weights sum to r. The result holds even if some selected a_i are negative. *Scope:* global optimality for **this fixed empirical matrix only**. It says nothing about the choice of phi, basis, nuisances, lambda, rank, or about any attacker. With a fixed r = 16, directions whose penalty exceeds their utility can be selected; the original spectral ARL analysis instead restricts to non-negative eigenvalues.

**Proposition 2 (reconstruction identity).** If V'V/n = I and W'W = I, then Z = VW has Z'Z/n = I, the least-squares decoder is D* = Z'R/n, and
min_D ||R - Z D||_F^2 / n = ||R||_F^2 / n - tr(W' U W),
with the raw (unnormalized) U, row-summed squared error, and centered R. The code checks this against a direct least-squares fit for every arm (development `NUMERICAL_SUMMARY.json`). *Scope:* this is linear reconstruction of the residual teacher on training rows. It is not a guarantee of usefulness for arbitrary future tasks.

**Proposition 3 (what the penalty measures).** tr(W' P_j W) is proportional to sum_{c,k} || E_n[ Z b_k(H) (1{S_j = c} - m_jc(H)) ] ||^2, a finite set of empirical first moments between Z and the nuisance residual, weighted by basis functions of H. In population, E[g(Z,H)(1{S_j = c} - P(S_j = c | H))] = 0 for **every** square-integrable g and every class c holds exactly when P(S_j = c | Z, H) = P(S_j = c | H), that is, S_j and Z are conditionally independent given H (the L2 characterization, attributed to Daudin, that underlies kernel conditional-independence tests). The mechanism tests only g(Z,H) = Z_l b_k(H): linear in Z, a degree-2 polynomial in H, with estimated nuisances and finite samples. Hence:

* vanishing fitted moments do **not** establish conditional independence, marginal independence from H, or small conditional mutual information;
* moments can be nonzero because the nuisance is misspecified rather than because Z leaks;
* cross-fitting reduces overfitting artifacts but does not certify the nuisance.

**Counterexample A (conditioning matters; XOR).** S, H independent fair signs, Z = SH. Then E[ZS] = 0 and Z is marginally independent of S, but E[ZSH] = 1 and S = ZH exactly. A marginal penalty sees nothing; an interaction basis term detects it. Verified numerically in the development fixtures.

**Counterexample B (first moments are insufficient).** (Z, S) uniform on {(-1,0), (1,0), (-2,1), (2,1)}: E[Z(S - 1/2)] = 0, yet |Z| > 1.5 recovers S perfectly. Zero first moments, even exactly and in population, cannot certify privacy against nonlinear attackers.

**Proposition 4 (log loss and conditional mutual information).** For population-optimal probabilistic predictors under log loss,
inf_q E[-log q(S | H)] - inf_q E[-log q(S | H, Z)] = H(S | H) - H(S | H, Z) = I(S; Z | H).
*Why fitted score differences are not bounds:* our "additional recovery" is L(f_H) - L(f_HZ) for two finite, validation-selected, finite-sample predictors. L(f_H) >= H(S|H) and L(f_HZ) >= H(S|H,Z), so their difference can exceed or fall short of I(S;Z|H) by the difference of two unknown excess risks. It can even be negative although I(S;Z|H) >= 0 (observed for historical interfaces in development). It is neither an upper nor a lower bound, and a low value is evidence only about the tested attack families. Estimating mutual information from samples has fundamental limits (McAllester and Stratos), which is why no information quantity is reported as certified.

**Proposition 5 (a chain-rule lower bound).** For any variable Y (for example a proxy computable by the attacker),
I(S; Z | H) >= I(S; Y | H) - H(Y | H, Z).
*Proof:* by the chain rule, I(S; Y | H) <= I(S; Y, Z | H) = I(S; Z | H) + I(S; Y | H, Z), and I(S; Y | H, Z) <= H(Y | H, Z) for discrete Y. Replacing H(Y | H, Z) by the log loss L_Y of any population decoder of Y from (H, Z) gives the weaker bound I(S; Z | H) >= I(S; Y | H) - L_Y, since L_Y >= H(Y | H, Z). Using it empirically requires estimating I(S; Y | H) with valid uncertainty; this study does not do so and reports no certified lower bound.

**Proposition 6 (withholding).** Let B ~ Bernoulli(p) be independent of (X, S), visible to the recipient, fixed per person, with the channel released only when B = 1. Then I(S; B, Z_B | H) = p I(S; Z | H), and for a routed family that uses f_H when B = 0 and f_HZ when B = 1, the expected log loss is (1-p) L(f_H) + p L(f_HZ). This justifies mixing per-person losses (never probabilities or features). The branch must be persistent: independent re-draws on repeated requests would eventually reveal Z. Utility is an average over people who receive different information.

**Finite-source LP (diagnostic only).** For a known finite joint distribution of (U, S, H) and fixed decoder U_hat = z, minimizing classification error over channels Q(z | x) subject to Q >= 0, rows summing to one and P_Q(z | s, h) = P_Q(z | h) for supported (s, h) is a linear program. With independent fair useful and sensitive bits it attains error 0; with U = S and constant H it attains 1/2 (solver status and primal residuals recorded in the development `NUMERICAL_SUMMARY.json`). This illustrates exact conditional privacy on a known finite source. It is not an ACS guarantee, and binning ACS anchor values would not transfer it to their exact published values.

## 4. Transport-specific scope

* 2017 transport is between **public survey years** of California ACS. It is not an untouched sample from the 2018 distribution, and public identifiers cannot establish that 2017 respondents are different people from 2018 respondents.
* In Mode A nothing is refitted; weaker frozen attacks after shift do not show that fresh attackers fail.
* In Mode B only probes, priors and attackers are refitted; representations, protection matrices and nuisances are frozen. The residual-moment diagnostic applies the frozen 2018 nuisance ensemble to 2017 rows.
* Inference statements are about people in the 2017 final partition for three fixed fitted systems (household bootstrap), not about retraining variability or Census design variance.
* Only recipient A receives a new channel. Nothing here demonstrates joint optimization of channels for an arbitrary number of recipients.
* Subtracting a fixed H anchor score in reporting changes no gradient and is not a new algorithm.
* Lower additional coalition recovery does not imply lower individual-recipient recovery; both are reported.
