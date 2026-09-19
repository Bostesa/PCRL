#!/usr/bin/env python3
"""Algebraic fixtures for the independent mathematical review of the proposed
coalition-conditioned rank-reducing projection and its convex-correction claim.

These are FIXTURES, not a benchmark: each one is a small closed-form construction
whose answer is known analytically, used to decide whether a stated claim is true,
false, or true only under conditions the specification does not yet impose.

No ACS data is read, no model is fitted, nothing here touches 2016.
"""
import numpy as np, json
rng = np.random.default_rng(20260918)
R = {}

# =====================================================================
# F1  Row/column conventions, pseudoinverse support and idempotence
#     v = (z - mu) Sigma^(-1/2);  z_out = mu + v P Sigma^(1/2)
#     Composite:  z_out = mu + (z - mu) M,  M = Sigma^(-1/2) P Sigma^(1/2)
# =====================================================================
def sqrt_and_pinv_sqrt(S, tol=1e-10):
    w, V = np.linalg.eigh(S)
    keep = w > tol * max(w.max(), 1.0)
    wc = np.where(keep, w, 0.0)
    half = V @ np.diag(np.sqrt(wc)) @ V.T
    inv = V @ np.diag(np.where(keep, 1.0 / np.sqrt(np.where(keep, w, 1.0)), 0.0)) @ V.T
    return half, inv, V[:, keep], int(keep.sum())

d = 6
A = rng.normal(size=(d, d)); S_full = A @ A.T + 0.5 * np.eye(d)
H, Hi, supp, r = sqrt_and_pinv_sqrt(S_full)
# P projects onto the top-k eigendirections of a symmetric "local matrix" in whitened space
def topk_projector(Mloc, k):
    w, V = np.linalg.eigh(Mloc)
    idx = np.argsort(w)[::-1][:k]
    U = V[:, idx]
    return U @ U.T, w[np.argsort(w)[::-1]]
Mloc = rng.normal(size=(d, d)); Mloc = Mloc @ Mloc.T
P, evals = topk_projector(Mloc, 3)
M = Hi @ P @ H
R["F1_conventions"] = {
    "verdict": "CONSISTENT, with two conditions the specification must state",
    "M_idempotent_max_abs_error": float(np.abs(M @ M - M).max()),
    "M_symmetric_max_abs_error": float(np.abs(M - M.T).max()),
    "rank_M": int(np.linalg.matrix_rank(M, tol=1e-9)),
    "rank_P": int(np.linalg.matrix_rank(P, tol=1e-9)),
    "finding": (
      "With v a ROW vector, z_out = mu + (z-mu) Sigma^(-1/2) P Sigma^(1/2) is affine in z and the "
      "composite M is idempotent of the same rank as P, so the construction does define a projection "
      "of the declared rank. M is NOT symmetric: it is an OBLIQUE projector in the original "
      "coordinates, exactly as LEACE's P* is. Any implementation that assumes a symmetric projector, "
      "or that uses M.T where M is meant, silently changes the method.")}

# Rank-deficient Sigma: pseudoinverse restricted to the support
B = rng.normal(size=(d, 4)); S_def = B @ B.T                      # rank 4 < d = 6
Hd, Hid, suppd, rd_ = sqrt_and_pinv_sqrt(S_def)
Pi_supp = Hd @ Hid                                                 # support projector
Pd, _ = topk_projector(Mloc, 3)
Md = Hid @ Pd @ Hd
in_support = np.abs(Pi_supp @ Pd @ Pi_supp - Pd).max()
R["F1b_pseudoinverse_support"] = {
    "verdict": "CONDITION REQUIRED",
    "ambient_dim": d, "rank_Sigma": rd_,
    "Sigma_half_times_pinv_is_identity": bool(np.allclose(Pi_supp, np.eye(d))),
    "support_projector_rank": int(np.linalg.matrix_rank(Pi_supp, tol=1e-9)),
    "Md_idempotent_max_abs_error": float(np.abs(Md @ Md - Md).max()),
    "P_confined_to_support_max_abs_error": float(in_support),
    "finding": (
      "When Sigma is rank deficient the pseudoinverse gives Sigma^(1/2) Sigma^(-1/2) = Pi_supp, not I. "
      "Idempotence of the composite then requires range(P) to lie inside the support; if P is built "
      "from a local matrix with mass off the support this fails and the map is no longer a projection. "
      "Separately, the map annihilates any off-support component of z. Every training row lies in the "
      "empirical support by construction, so this is invisible at fit time and appears only on new "
      "rows - which is precisely the transport setting. The rank tolerance that defines the support "
      "must be declared, reported, and applied to a RELATIVE eigenvalue scale.")}

# =====================================================================
# F2  Fixed-rank alias: c > 0 leaves eigenspaces unchanged
# =====================================================================
c = 2.0
Pc, evals_c = topk_projector(c * Mloc, 3)
# degenerate case: force a tie at the k / k+1 boundary
w, V = np.linalg.eigh(Mloc)
w_tie = w.copy(); order = np.argsort(w)[::-1]
w_tie[order[2]] = w_tie[order[3]]                                  # tie at the k=3 boundary
M_tie = V @ np.diag(w_tie) @ V.T
Pt, _ = topk_projector(M_tie, 3); Ptc, _ = topk_projector(c * M_tie, 3)
# absolute-tolerance rank rule, where the alias BREAKS
def abs_tol_rank(Mm, tol): return int((np.linalg.eigvalsh(Mm) > tol).sum())
tol_abs = float(np.sort(np.linalg.eigvalsh(Mloc))[::-1][2]) * 1.0001   # between lambda_3 and lambda_4... by construction above lambda_3
R["F2_fixed_rank_alias"] = {
    "verdict": "PROVED for a fixed-rank hard projection; FALSE for an absolute-tolerance rank rule",
    "projector_max_abs_difference_under_scaling": float(np.abs(P - Pc).max()),
    "eigenvalue_ratio_max_abs_error_from_c": float(np.abs(evals_c / evals - c).max()),
    "degenerate_boundary_projector_difference": float(np.abs(Pt - Ptc).max()),
    "abs_tol_rank_at_c_1": abs_tol_rank(Mloc, tol_abs),
    "abs_tol_rank_at_c_2": abs_tol_rank(c * Mloc, tol_abs),
    "proof": (
      "If Mloc v = lambda v then (c Mloc) v = (c lambda) v for c > 0, so eigenvectors are identical and "
      "eigenvalues are scaled by a positive constant, which preserves their order AND their ties: "
      "c*lambda_k = c*lambda_{k+1} iff lambda_k = lambda_{k+1}. The top-k eigenspace, and hence the "
      "hard projector, is therefore literally the same object. A mass-doubled local hard projection is "
      "NOT a distinct method and must not be registered or counted as a separate comparator."),
    "the_exception_that_must_be_stated": (
      "The alias holds for a FIXED-RANK rule. It fails for a rank rule of the form #{lambda > tol} with "
      "an ABSOLUTE tolerance, where scaling by c changes the selected rank (shown above: the same "
      "tolerance selects different ranks at c=1 and c=2), and it fails for any invertible SHRINKAGE "
      "form such as (Mloc + tau I)^(-1), where c does not commute with the ridge. If the projection "
      "track uses an absolute tolerance anywhere, the two local arms are not aliases and the registry "
      "must say which rule is in force."),
    "the_control_that_is_genuinely_different": (
      "The local-EXPANDED control changes the FEATURE FAMILY, not the mass, so it is not an alias. It "
      "is a useful but imperfect control for the richer coalition basis: it matches capacity without "
      "matching which conditioning set generated the directions.")}

# =====================================================================
# F3  Rank reduction and information limits
# =====================================================================
Ginv = rng.normal(size=(d, d)); Ginv = Ginv + d * np.eye(d)        # invertible shrinkage
Phard, _ = topk_projector(Mloc, 2)
R["F3_information"] = {
    "verdict": "STATED CORRECTLY ONLY IF SPLIT IN TWO",
    "invertible_map_rank": int(np.linalg.matrix_rank(Ginv, tol=1e-9)),
    "hard_projection_rank": int(np.linalg.matrix_rank(Phard, tol=1e-9)),
    "ambient": d,
    "part_1_true": (
      "An INVERTIBLE shrinkage cannot reduce unrestricted information: z -> zG with G invertible is a "
      "bijection, so any attacker composing G^(-1) with its own predictor recovers exactly what it "
      "could before. Sup over ALL measurable attackers is unchanged."),
    "part_2_true": (
      "A HARD projection can discard information: it is not injective, and the discarded component is "
      "unrecoverable from the output alone."),
    "part_3_the_limit": (
      "NEITHER fact guarantees improvement in a finite attacker benchmark. An invertible shrinkage can "
      "still lower MEASURED recovery by making the retained signal harder for the declared finite "
      "family to extract inside its budget, and a hard projection can discard capacity without "
      "discarding what the attacker was actually using. The benchmark measures the declared family, "
      "not the supremum."),
    "trace_criterion_scope": (
      "SUPPORTS: among rank-k projections, the retained subspace maximises the trace objective on the "
      "FITTED empirical moments. A FULL-SPAN projection annihilates the specified empirical moments "
      "exactly. DOES NOT SUPPORT: a partial top-k projection generally leaves those moments non-zero; "
      "it is not minimal leakage, not optimal against the attacker slate, and not a population statement.")}
# demonstrate: top-k does not annihilate the moment, full span does
C = rng.normal(size=(d, 3))                                        # cross-moment Z x S
U, sv, Vt = np.linalg.svd(C, full_matrices=False)
P_full = np.eye(d) - U @ U.T
P_topk = np.eye(d) - U[:, :1] @ U[:, :1].T
R["F3_information"]["moment_after_full_span_projection"] = float(np.abs(P_full @ C).max())
R["F3_information"]["moment_after_top1_projection"] = float(np.abs(P_topk @ C).max())

# =====================================================================
# F4  The narrow convex-correction claim, and the smallest counterexample
# =====================================================================
def softmax(L):
    L = L - L.max(1, keepdims=True); E = np.exp(L); return E / E.sum(1, keepdims=True)
def ce(L, y):
    p = np.clip(softmax(L), 1e-12, None); p = p / p.sum(1, keepdims=True)
    return -np.log(p[np.arange(len(y)), y]).mean()
def grad0(Phi, L0, y):
    """d/dtheta of mean CE(L0 + Phi theta) at theta = 0  =  Phi.T (p0 - Y) / n."""
    p0 = softmax(L0); Y = np.zeros_like(p0); Y[np.arange(len(y)), y] = 1.0
    return Phi.T @ (p0 - Y) / len(y)

n, K = 4000, 2
Hf = rng.normal(size=(n, 3)); Zf = rng.normal(size=(n, 4))
beta_true = rng.normal(size=(3, K)) * 0.8
L0 = Hf @ beta_true
y = np.array([rng.choice(K, p=p) for p in softmax(L0)])            # p0 is the TRUE model here
# (a) basis that is genuinely orthogonal to the residual in expectation: centred channel, no intercept
Zc = Zf - Zf.mean(0, keepdims=True)
Phi_ok = Zc
g_ok = grad0(Phi_ok, L0, y)
# project the basis so the EMPIRICAL gradient is exactly zero (this is what the erasure enforces)
p0 = softmax(L0); Y = np.zeros_like(p0); Y[np.arange(n), y] = 1.0; Rres = p0 - Y
Q, _ = np.linalg.qr(Rres)                                          # span of the residual columns
Phi_exact = Phi_ok - Q @ (Q.T @ Phi_ok)                            # empirical cross-moment exactly 0
g_exact = grad0(Phi_exact, L0, y)
# verify theta = 0 is the global minimiser over that family, by direct search
def sweep(Phi, scale=np.linspace(-2, 2, 81)):
    base = ce(L0, y); worst = base
    dirs = [rng.normal(size=(Phi.shape[1], K)) for _ in range(40)] + \
           [np.eye(Phi.shape[1])[:, [i]] @ np.eye(K)[[j], :] for i in range(Phi.shape[1]) for j in range(K)]
    best = base
    for D in dirs:
        D = D / np.linalg.norm(D)
        for s in scale:
            v = ce(L0 + Phi @ (s * D), y)
            best = min(best, v)
    return base, best
base_e, best_e = sweep(Phi_exact)
# (b) THE COUNTEREXAMPLE: add an intercept column, and fit p0 on a DISJOINT fold so it is not calibrated
half = n // 2
beta_fit = np.linalg.lstsq(np.c_[Hf[:half], np.ones(half)],
                           np.eye(K)[y[:half]] - 0.5, rcond=None)[0]
L0_mis = np.c_[Hf[half:], np.ones(n - half)] @ beta_fit * 3.0      # a deliberately miscalibrated offset
y2 = y[half:]
Phi_int = np.c_[np.ones(n - half), Zf[half:] - Zf[half:].mean(0, keepdims=True)]
g_int = grad0(Phi_int, L0_mis, y2)
# the intercept direction is swept directly against the miscalibrated offset
def sweep_intercept(L0_, y_, scale=np.linspace(-3, 3, 241)):
    base = ce(L0_, y_); best = base; arg = 0.0
    for s in scale:
        v = ce(L0_ + np.c_[np.ones(len(y_)), np.zeros(len(y_))] * s, y_)
        if v < best: best, arg = v, s
    return base, best, arg
b0, b1, barg = sweep_intercept(L0_mis, y2)
R["F4_convex_correction"] = {
  "verdict": "TRUE AS A NARROW STATEMENT; FALSE IF AN INTERCEPT OR AN UNCALIBRATED OFFSET IS IN THE FAMILY",
  "convexity": ("For a FIXED offset L0 and a correction that is LINEAR in a fixed basis Phi, the map "
                "theta -> mean CE(L0 + Phi theta) is convex: it is a sum of log-sum-exp of affine "
                "functions minus affine functions. Hence a stationary point is a global minimum."),
  "gradient_identity": ("grad at theta = 0 equals Phi.T (p0 - Y) / n, i.e. exactly the EMPIRICAL "
                        "CROSS-MOMENT between the basis and the residual of the service-only "
                        "predictor. 'The projection annihilates the specified empirical moments' and "
                        "'zero empirical gradient at zero correction' are the SAME statement, and "
                        "together with convexity they give empirical optimality of the zero correction "
                        "FOR THAT FAMILY."),
  "supported_case": {"max_abs_gradient_at_zero": float(np.abs(g_exact).max()),
                     "ce_at_zero": float(base_e), "best_ce_found_over_family": float(best_e),
                     "improvement_found": float(base_e - best_e)},
  "raw_centred_basis_gradient": float(np.abs(g_ok).max()),
  "COUNTEREXAMPLE": {
     "construction": ("Put an INTERCEPT column in the correction basis and let the service-only offset "
                      "be fitted on a disjoint fold, so it is not calibrated on the scoring pool - "
                      "which is exactly the p0_fit / mapper_fit / monitor split in use."),
     "max_abs_gradient_at_zero": float(np.abs(g_int).max()),
     "intercept_gradient": float(np.abs(g_int[0]).max()),
     "ce_at_zero_correction": float(b0), "best_ce_over_intercept_alone": float(b1),
     "argmin_shift": float(barg), "improvement": float(b0 - b1),
     "why": ("The gradient in the intercept direction is mean(p0) - mean(Y), the in-sample calibration "
             "error. Erasing a cross-moment between the CHANNEL and the labels does nothing to it. So "
             "the zero correction is NOT optimal in a family containing an intercept unless p0 is "
             "calibrated on the very pool being scored.")},
  "corrected_statement": (
     "Let the offset be FIXED, let the correction family be exactly {Phi theta} for a basis Phi whose "
     "EMPIRICAL cross-moment with the residual (p0 - Y) is zero ON THE MASKED ROWS ACTUALLY SCORED, and "
     "let Phi contain no intercept and no column outside that annihilated span. Then theta = 0 is a "
     "global minimiser of the empirical cross-entropy over that family. The conditions are not "
     "decorative: each of the intercept, the role masks, the choice of centring, and any parameter "
     "direction left out of the moment being erased can break it independently."),
  "does_NOT_carry_over_to": [
     "a freely retrained H baseline - retraining moves L0, so the residual changes and the gradient is no longer zero",
     "an arbitrary nonlinear correction - the family is no longer spanned by Phi and convexity in theta says nothing about it",
     "population leakage - the statement is about an EMPIRICAL moment on a finite sample, not about the population conditional",
     "the augmented release - H_A is still present, and the claim concerns the transformed channel only"],
  "centring_warning": (
     "The moment must be computed with the declared valid-role masks and a GLOBALLY centred channel. "
     "Re-centring per role changes which object is annihilated: per-role centring erases the WITHIN-ROLE "
     "covariance and leaves the between-role mean shift, and the difference is precisely a rank-one "
     "term per role. An implementation must not switch between a covariance and a raw cross-moment "
     "silently - the two coincide only when the channel is centred on the same mask the moment uses.")}

# demonstrate the centring warning numerically
S_lab = rng.integers(0, 3, size=n)
mask = rng.random(n) < 0.7
Zg = Zf - Zf[mask].mean(0, keepdims=True)                          # globally centred on the mask
onehot = np.eye(3)[S_lab]
raw = Zf[mask].T @ onehot[mask] / mask.sum()
cov = Zg[mask].T @ (onehot[mask] - onehot[mask].mean(0, keepdims=True)) / mask.sum()
# global centring keeps the BETWEEN-ROLE mean shift; per-role centring destroys it exactly
glob = np.zeros_like(cov); perrole = np.zeros_like(cov)
for k in range(3):
    mk = mask & (S_lab == k)
    glob[:, k] = (Zf[mk] - Zf[mask].mean(0, keepdims=True)).mean(0)
    perrole[:, k] = (Zf[mk] - Zf[mk].mean(0, keepdims=True)).mean(0)
R["F4_convex_correction"]["centring_numbers"] = {
    "raw_cross_moment_max_abs": float(np.abs(raw).max()),
    "centred_covariance_max_abs": float(np.abs(cov).max()),
    "globally_centred_between_role_shift_max_abs": float(np.abs(glob).max()),
    "per_role_recentred_max_abs": float(np.abs(perrole).max()),
    "raw_minus_covariance_max_abs": float(np.abs(raw - cov).max()),
    "note": ("the raw cross-moment and the centred covariance are different matrices, and per-role "
             "re-centring drives the between-role mean shift to zero BY CONSTRUCTION - so a moment "
             "computed that way is already zero and erasing it erases nothing")}

# =====================================================================
# F5  Does the coalition basis detect information unlocked by the second view?
# =====================================================================
m = 20000
a = rng.integers(0, 2, m)      # a service-view bit visible in H_A
bbit = rng.integers(0, 2, m)   # a channel bit carried by Z
s_xor = a ^ bbit               # sensitive attribute: XOR
Za = bbit.astype(float)[:, None]
HA = a.astype(float)[:, None]
HB = rng.integers(0, 2, m).astype(float)[:, None]
def cm(X, s):
    Xc = X - X.mean(0, keepdims=True); sc = s - s.mean()
    return float(np.abs(Xc.T @ sc / len(s)).max())
marg = cm(Za, s_xor)
inter_A = cm(np.c_[Za, HA, Za * HA], s_xor)
inter_AB = cm(np.c_[Za, HA, HB, Za * HA, Za * HB], s_xor)
dup = cm(np.c_[Za, HA, HA, Za * HA, Za * HA], s_xor)               # duplicated labels, same conditioning set
# a SECOND sensitive attribute whose partner bit lives only in H_B: the A-view basis cannot see it,
# the AB-view basis can. This is the case that separates H_A from H_AB.
bB = HB[:, 0].astype(int)
s_xorB = bB ^ bbit
seenA = cm(np.c_[Za, HA, Za * HA], s_xorB)
seenAB = cm(np.c_[Za, HA, HB, Za * HA, Za * HB], s_xorB)
R["F5_interaction_fixture"] = {
  "verdict": "THE INTERACTING BASIS DOES DETECT IT; THE MARGINAL BASIS DOES NOT",
  "construction": "S = A XOR B, where A is a service-view bit in H_A and B is a channel bit in Z",
  "max_abs_cross_moment_channel_only": marg,
  "max_abs_cross_moment_with_H_A_interactions": inter_A,
  "max_abs_cross_moment_with_H_AB_interactions": inter_AB,
  "max_abs_cross_moment_with_duplicated_H_A_columns": dup,
  "second_attribute_partner_bit_in_H_B": {
     "construction": "S2 = B_HB XOR B_Z, where the partner bit is carried only by H_B",
     "max_abs_cross_moment_A_view_basis": seenA,
     "max_abs_cross_moment_AB_view_basis": seenAB},
  "finding": (
     "Z alone and H_A alone each carry zero marginal cross-moment with S, yet the pair determines S. A "
     "basis of service-view x channel PRODUCTS recovers a cross-moment of order 0.25 where the marginal "
     "basis sees ~0. This is why the proposed targets combine a service-view basis with the channel: "
     "the coalition view can unlock information neither view carries alone."),
  "why_H_A_versus_H_AB_changes_the_target": (
     "Adding H_B ENLARGES the conditioning set and therefore changes the residual being erased, and it "
     "enlarges the interacting basis with new product columns. DUPLICATING a label changes neither: the "
     "duplicated design has the same column span, so the same residual and the same annihilated moment "
     "(shown above - the duplicated basis returns exactly the same value as the single one). Both halves "
     "are demonstrated: for S2 = B_HB XOR B_Z the A-view interacting basis sees ~0 while the AB-view "
     "basis sees ~0.125. The distinction is the SPAN of the conditioning set, not the number of columns."),
  "cross_fitting_caveat": (
     "Cross-fitting the sensitive residuals controls one bias - the residual is not fitted on the rows "
     "it is evaluated on. It is NOT evidence that the nuisance model is correct, and it is NOT evidence "
     "of conditional independence. A misspecified nuisance model cross-fits just as happily.")}

# =====================================================================
# F6  Counterexample to completeness: zero finite moments, recoverable information
# =====================================================================
m = 200000
s = rng.integers(0, 2, m)
z = np.where(s == 1, rng.choice([-1.0, 1.0], m), rng.normal(size=m))
# S=1 -> Z in {-1,+1};  S=0 -> Z ~ N(0,1).  Both mean 0, variance 1, all odd moments 0.
mom = {f"E[Z^{k} (S - E S)]": float(np.mean((z ** k) * (s - s.mean()))) for k in (1, 2, 3)}
acc = float(np.mean((np.abs(np.abs(z) - 1.0) < 1e-12).astype(int) == s))
R["F6_completeness_counterexample"] = {
  "verdict": "ZERO FINITE MOMENTS DO NOT IMPLY NO RECOVERABLE INFORMATION",
  "construction": "S ~ Bernoulli(1/2); Z | S=1 is +-1 with probability 1/2; Z | S=0 is N(0,1)",
  "moments_that_vanish": mom,
  "population_first_two_moments": "E[Z|S=0] = E[Z|S=1] = 0 and E[Z^2|S=0] = E[Z^2|S=1] = 1, exactly",
  "accuracy_of_the_rule_S_hat_=_1{|Z| = 1}": acc,
  "finding": (
     "The first and second cross-moments of Z with S vanish identically, so ANY method whose criterion "
     "is the annihilation of those moments - including the proposed projection and including LEACE - "
     "leaves this channel untouched, while the nonlinear rule 1{|Z| = 1} recovers S essentially "
     "perfectly. Moment annihilation is not sufficiency, and a low measured recovery under a finite "
     "attacker family is not independence. This is the standing counterexample the manuscript carries "
     "against any completeness reading of the construction.")}

# =====================================================================
# F7  Proposed form versus LEACE's subtractive form (Belrose et al. 2023, Thm 4.2/4.3 Eq. 1,
#     as transcribed from the PDF body in BASELINE_ADAPTATIONS.md s3.1 at 73903b7f)
#     LEACE (row form):  r(z) = z - (z - mu) W P_rm W^+ ,  W = (Sigma^+)^{1/2}
#     proposed:          z_out = mu + (z - mu) Sigma^{-1/2} P_keep Sigma^{1/2},  P_keep = I - P_rm
# =====================================================================
def both(S, Crm, zrows, mu):
    Hh, Hin, _, _ = sqrt_and_pinv_sqrt(S)
    W = Hin; Wp = Hh
    A = W @ Crm
    Prm = A @ np.linalg.pinv(A)
    leace = zrows - (zrows - mu) @ W.T @ Prm.T @ Wp.T   # row form of x - W^+ P W (x - mu)
    Pkeep = np.eye(S.shape[0]) - Prm
    prop = mu + (zrows - mu) @ Hin @ Pkeep @ Hh
    return leace, prop
mu = np.zeros(d)
Crm = rng.normal(size=(d, 2))
zr = rng.normal(size=(50, d))
L_full, P_full_ = both(S_full, Crm, zr, mu)
L_def, P_def = both(S_def, Crm, zr, mu)
# off-support component of test rows
_, Hid2, _, _ = sqrt_and_pinv_sqrt(S_def)
Pi = sqrt_and_pinv_sqrt(S_def)[0] @ Hid2
off = zr - zr @ Pi
R["F7_versus_LEACE"] = {
  "verdict": "IDENTICAL TO LEACE'S FORM WHEN SIGMA IS FULL RANK; DIFFERENT WHEN IT IS NOT",
  "full_rank_max_abs_difference": float(np.abs(L_full - P_full_).max()),
  "rank_deficient_max_abs_difference": float(np.abs(L_def - P_def).max()),
  "off_support_norm_in_test_rows": float(np.linalg.norm(off, axis=1).mean()),
  "leace_retains_off_support": float(np.abs((L_def - L_def @ Pi) - off).max()),
  "proposed_retains_off_support": float(np.linalg.norm(P_def - P_def @ Pi, axis=1).mean()),
  "finding": (
    "With full-rank Sigma and P_keep = I - P_rm, the proposed map IS LEACE's closed form with a "
    "different cross-moment substituted (difference at machine precision). With rank-deficient "
    "Sigma they differ: LEACE's subtractive form passes off-support components of new rows through "
    "unchanged, while the project-in-whitened-space form deletes them. Neither is wrong a priori, "
    "but the specification must choose and say so; the subtractive form is the one whose "
    "guarantees Belrose et al. prove. Algebraically the construction is an adaptation of LEACE: the "
    "potentially useful difference is WHICH cross-moment is erased (coalition-conditioned "
    "residual targets) and the explicit rank control, not the projection.")}

print(json.dumps(R, indent=1))
