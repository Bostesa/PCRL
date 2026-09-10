# Residual spectral baseline: finite objective and scope

This is an adaptation of spectral representation learning and conditional residual moments, not a novelty claim. Its theorem concerns the exact optimizer of a fixed empirical matrix objective. It establishes no unrestricted marginal privacy, conditional privacy, generalization guarantee, or guarantees against a nonlinear downstream attacker.

## Frozen construction

Inputs are original, unstandardized frozen PCA32 teacher coordinates `T` (n by 32), complete income/employment probability vectors `hA` (n by 4), and coverage probabilities `hB` (n by 2). Only `SEX` (0/1), `RAC1P` (0..8), and `public_coverage` (0/1) enter fitting. Each target uses its own mask `y >= 0`; missing target rows are excluded from that target's nuisance fits and moments, including its denominator. No reserved label is accepted by the API.

All transforms are fit on representation-training rows. Standardize each T column by its population standard deviation when greater than 1e-12; otherwise use scale 1 so negligible variation is not amplified. Choose min(1024,n) rows without replacement using NumPy default_rng(20260910+100*seed). Let sigma be the median strictly positive pairwise Euclidean distance between their standardized coordinates. If none is positive, abort fitting with a bandwidth diagnostic error. Draw omega with shape 32 by 96 from N(0,1/sigma²), then phases uniformly in [0,2pi), using default_rng(20261910+100*seed). Set

`phi(T) = [T_standardized, sqrt(2/96) cos(T_standardized omega + phase)]`.

The A basis contains an intercept, the two positive-class probabilities `hA[:,[1,3]]`, and all degree-two monomials (including squares); the AB basis analogously uses these two variables and `hB[:,1]`. Order is intercept, linear terms in input order, then lexicographic combinations with replacement. Center and population-scale each nonconstant column. Retain the intercept; discard columns whose population standard deviation is <=1e-12 and later columns equal to a retained standardized column within maximum absolute tolerance 1e-12. This removes exact binary squares and duplicate inputs without an unspecified rank-selection heuristic. Remaining dependencies are handled by least squares.

Fit `B_phi = Q_A^+ phi` with SVD least squares relative cutoff 1e-10. Set `V0 = phi - Q_A B_phi`, subtracting its training mean explicitly for roundoff. Diagonalize `C0 = V0' V0/n`; retain eigenvalues strictly above `max(1e-12, 1e-10 * largest_eigenvalue)`. For retained eigensystem `(E,D)`, set `V=V0 E D^(-1/2)`. Thus V has training mean zero and training covariance identity on its retained subspace, to numerical tolerance. No ridge whitening is used. Record the full spectrum, threshold, rank, mean, covariance error, and cross moment with Q_A. Reject covariance/mean errors above 1e-6.

Independently fit `R=T-Q_A Q_A^+ T` using **original unstandardized T**, then explicitly recenter. Every arm uses `r=min(16,rank(V))`; a rank below 16 is flagged as a dimension mismatch with 16-dimensional baselines. All saved matrices use float64. Eigenvector signs are canonicalized by making the largest-absolute component nonnegative; signs do not resolve repeated-eigenvalue rotations, so exact cross-platform coordinate identity is not promised.

## Nuisances and finite moments

Sort unique household IDs using NumPy unique, shuffle their indices using default_rng(20262910+100*seed), and assign shuffled positions cyclically to three folds. All rows of one household stay in one fold. At least three households are required. For each of the three local targets fit three out-of-fold models using Q_A; for SEX and race fit three using Q_AB. The unsupervised basis scaling above is shared across the three folds and uses only representation-training rows. Each model is logistic regression with C=1, lbfgs, max_iter=10000, tol=1e-8 and fit_intercept=False because Q includes an intercept. This is 15 possible nuisance fits per seed, reduced when a training fold has fewer than two supported classes. Record each fold's class support on train/holdout, unsupported classes, iteration count, convergence warnings and status. Any nonconverged logistic solver aborts fitting with these diagnostic details. One supported training class yields its constant predictor. No supported training class yields a zero vector and an explicit unsupported flag. Fixed-schema predictions have zeros for classes absent from the fold's training data.

For marginal penalties use a constant empirical prior from valid representation-training rows and the constant basis 1. Conditional roles use their corresponding OOF predictions and every retained polynomial column. For target j and class c, on its n_j valid rows define

`G_jc = V_valid' [b(H) * (1{y_j=c} - p_hat_jc(H))] / n_j`,

where multiplication by the scalar residual is rowwise, so G is rank(V) by basis-width. Use **all K one-hot columns**, no reference-class deletion, weights, class balancing, or separate class normalization. Set `P_j_raw=sum_c G_jc G_jc'`. Normalize P_j by its trace only when that trace exceeds 1e-12; otherwise set it to zero and flag it. Store every class's support, G, squared Frobenius norm, and every output map's squared projected norm. Unsupported classes are explicitly flagged even when their moment is zero.

`P_M` is the mean of the three normalized marginal attributes; `P_L` the mean of three normalized local attributes; `P_AB` the mean of two normalized coalition attributes (SEX, race). Denominators remain exactly 3, 3, 2 even when an attribute is zero or unsupported. Set `U_raw=(V'R/n)(R'V/n)` and normalize by its trace only when that trace exceeds 1e-12; otherwise abort with a utility diagnostic error.

## Spectral optimizer and reconstruction identity

For fixed symmetric A and rank r, maximize `tr(W' A W)` over `W'W=I_r`. If `A=E diag(a_i) E'` with eigenvalues decreasing, write `D=E'W`; the objective is `sum_i a_i ||D_i||²`, with row weights in [0,1] summing to r. It is at most the sum of the largest r eigenvalues. Their eigenvectors attain this bound. Thus the saved top-r eigenspace is globally optimal **for this fixed finite objective**, including when selected eigenvalues are negative. It does not optimize the choice of phi, nuisances, bandwidth, basis, rank, conditional distribution or attacker class. The code records direct objective, top-eigenvalue sum, their gap, and W orthogonality.

The eight A matrices are:

| Arm | Matrix |
|---|---|
| spectral_S0 | U |
| spectral_M025 | U - 0.25 P_M |
| spectral_M1 | U - P_M |
| spectral_L025 | U - 0.25 P_L |
| spectral_L1 | U - P_L |
| spectral_C025 | U - 0.25 (P_L + P_AB) |
| spectral_C1 | U - (P_L + P_AB) |
| spectral_L2 | U - 2 P_L |

For `Z=VW`, whitening and W orthogonality imply `Z'Z/n=I_r`. Therefore the least-squares decoder from Z to R is `D*=Z'R/n`, and

`min_D ||R-ZD||_F²/n = ||R||_F²/n - tr(W' U_raw W)`.

This identity uses **raw** U, not trace-normalized U. If `tr(U_raw)>1e-12`, the objective's utility term is the explained reconstruction energy divided by `tr(U_raw)`. Each penalty term is similarly `sum_c ||W'G_jc||_F² / tr(P_j_raw)` before fixed attribute averaging. Code compares the identity against an independent direct SVD least-squares reconstruction for every arm. A zero-trace utility aborts fitting rather than producing an uninformative matrix objective.

## Fitted inference and heldout diagnostics

The joblib-serializable model stores training standardization, omega/phases, both basis transforms, residual coefficients and mean, whitening, all eight W maps, priors, and fitted OOF nuisance models. `transform(T,hA,arm)` computes the complete release without hB. No test-set centering, refitting, labels or hB are required for inference.

Optional `moment_diagnostics(T,hA,hB,labels)` uses an equal ensemble of the three saved training-fold nuisance models on new rows, and the saved training priors for marginal diagnostics. It performs no fits and labels its prediction source. These heldout diagnostics are not OOF estimates on the heldout rows and are not a formal conditional-independence test.

## Failure modes and finite-source oracle

For independent fair signs S,H, let Z=S*H. Then Z and S are marginally independent and `E[ZS]=0`, yet `E[ZSH]=1` and S=ZH. An interaction basis detects leakage that a constant basis misses. Separately, let (Z,S) be equally likely (-1,0),(1,0),(-2,1),(2,1). The first moment `E[Z(S-1/2)]` is zero, yet thresholding |Z| recovers S perfectly. Thus even exact zero finite first moments cannot guarantee unrestricted privacy. A finite polynomial/RFF basis and estimated logistic residuals further limit the claim; observed moments can reflect nuisance misspecification as well as leakage.

The diagnostic linear program uses a supplied finite joint source distribution over (U,S,H), a binary channel Q(z|source), and fixed decoder U_hat=z. Minimize `sum_x,z p(x) Q(z|x) 1{z != U(x)}` with Q>=0, row sums one, and for every supported (s,h),z:

`sum_x p(x|s,h) Q(z|x) - sum_x p(x|h) Q(z|x) = 0`.

These are exact finite conditional-privacy constraints, not the spectral moment constraints. scipy.optimize.linprog (HiGHS) records status, channel, objective and maximum primal residual. Independent fair useful and sensitive bits with constant H permit error 0. If U=S is a fair bit and H is constant, any private binary output is independent of U and the minimum error is 1/2. This oracle permits a channel to observe the entire finite source; it is only a feasibility diagnostic for the two specified distributions and fixed decoder.

## Attribution and boundaries

Closed-form spectral adversarial representation objectives predate this baseline: [Sadeghi, Yu and Boddeti (2019)](https://arxiv.org/abs/1910.07423) study linear/global spectral solutions and kernel extensions; [Sadeghi, Wang and Boddeti (2021)](https://arxiv.org/abs/2109.05535) study closed-form adversarial/target solvers. These motivate the computational approach but do not establish a new theorem for this specific residualized ACS procedure.

[Zhang et al., KCI (2012)](https://arxiv.org/abs/1202.3775) formulate a kernel conditional-independence test with a null distribution. [Strobl et al. (2017)](https://arxiv.org/abs/1702.03877) develop random-Fourier-feature approximations for conditional tests. This baseline uses restricted finite residual moments for optimization; it does not implement KCI/RCIT or inherit their testing calibration.

The [privacy funnel](https://arxiv.org/abs/1402.1774) and [perfect privacy](https://arxiv.org/abs/1712.08500) literature concern information disclosure and privacy/utility feasibility. The tiny source LP above is an explicitly specified finite feasibility diagnostic, not an application-wide perfect-privacy certificate. The reference abstracts were checked at the primary arXiv records on 2026-09-10; the finite algebra and fixtures here are derived explicitly rather than attributed to an unverified paper theorem.

## Pre-fit implementation alignment

Before any ACS fitting, review against the frozen protocol corrected four implementation behaviors: negligible T standard deviations use scale 1; no-positive-distance bandwidth aborts; zero utility trace aborts; and nuisance nonconvergence aborts. Each was covered by a regression that failed before the repair. This was implementation alignment before fitting, not a change to the theorem or a response to ACS outcomes. The rank-deficiency fixture now has nonzero utility and 12 rows, so it validly exercises a shared output rank below 16.

## Information-theoretic interpretation remains separate

With population-optimal probabilistic decoders under logarithmic loss, the difference in optimal risks on H and (H,Z) equals I(S;Z|H), because the two infima equal the corresponding conditional entropies. Two finite validation-selected attack families do not attain these infima in general. Their signed test loss difference is neither automatically an upper nor a lower bound on that conditional mutual information. An H attack remains executable from an augmented wire even if validation chooses another attack that generalizes worse. Subtracting a fixed H score changes reporting but has zero derivative with respect to mapper parameters.

The elementary chain-rule bound I(S;Z|H) >= I(S;Y|H)-H(Y|H,Z) also implies I(S;Z|H) >= I(S;Y|H)-L_Y for a population decoder log loss L_Y. Estimating the first term and its uncertainty is a different problem; no ACS lower-bound certificate is claimed here.

For an independent visible branch B~Bernoulli(p), with Z released only when B=1, conditioning on B yields I(S;B,Z_B|H)=p I(S;Z|H). For fixed branch-routed predictors the same conditioning gives expected log loss (1-p)L_H+pL_augmented. This justifies the explicit withholding mechanism, not averaging probability vectors. Its assignment must persist for a person's release, because repeated independent requests would eventually disclose the auxiliary channel.

The literature grounds components rather than certifies this study: [spectral ARL](https://arxiv.org/pdf/1910.07423) analyzes restricted spectral solutions; [closed-form ARL](https://arxiv.org/pdf/2109.05535) uses kernel ridge best responses; [KCI](https://arxiv.org/pdf/1202.3775) and [random-feature CI tests](https://arxiv.org/abs/1702.03877) motivate richer conditional moments; [privacy funnels](https://arxiv.org/abs/1402.1774) connect optimal log loss to information; [On Perfect Privacy](https://arxiv.org/pdf/1712.08500) analyzes finite-source privacy feasibility. Their assumptions and guarantees do not transfer to the finite ACS attacks here. The fixed-decoder LP is a small separate feasibility diagnostic.
