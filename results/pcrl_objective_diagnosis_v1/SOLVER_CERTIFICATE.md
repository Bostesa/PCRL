# Frozen finite-programme solver certificate

This is a verification calculation for the **historical Q**, not a new release fit. The original programme minimizes `⟨C,Q⟩` over 32 stochastic rows and four separate empirical constraints `g_j(Q)=I_j(S;Z|C_A) ≤ b_j=0.01`. Every T0 row is supported, so the historical unsupported-state equations are inactive. The stored 32×17 cost and all four joint laws are identical to D17's; only Q's constraints differ. The calculation uses no 2016 rows.

For any multipliers `λ_j ≥ 0`, let `L(Q)=⟨C,Q⟩+Σ_j λ_j(g_j(Q)−b_j)`. Conditional information is convex in a channel with fixed input law, so for any strictly positive stochastic `Q₀`,

`L(Q) ≥ L(Q₀)+⟨∇L(Q₀), Q−Q₀⟩`.

Minimizing the right side over all row simplices gives a valid lower bound on the constrained optimum:

`B(λ,Q₀)=Σ_t min_z [C_tz+Σ_j λ_j ∂g_j(Q₀)/∂Q_tz] + Σ_j λ_j[g_j(Q₀)−b_j−⟨∇g_j(Q₀),Q₀⟩]`.

For `p_j(s,c,t)` and `A_j(s,c,z)=Σ_t p_j(s,c,t)Q₀(t,z)`, the gradient used is

`∂g_j/∂Q_tz = Σ_sc p_j(s,c,t) log{ A_j(s,c,z) / [p_j(s|c) Σ_s A_j(s,c,z)] }`.

Zero input cells contribute zero. The evaluation point is `(1−ε)Q_historical + ε·uniform`, so every channel probability is positive. A small HiGHS **dual-certificate LP** maximizes this tangent lower bound over nonnegative multipliers and one epigraph variable per T0 row. It does not solve a new primal channel. Four predeclared interior mixtures (`10⁻⁶`, `10⁻⁸`, `10⁻¹⁰`, `10⁻¹²`) are checked and the largest valid bound is retained. Tests finite-difference the gradient on a separate analytic fixture. [SOLVER_CERTIFICATE.json](SOLVER_CERTIFICATE.json) contains all multiplier vectors, LP residuals, alternate-mixture bounds, and the unconstrained lower bounds.

| Anchor | Historical Q objective | Tangent dual lower bound | Raw gap | Gap with 10⁻⁷ numerical guard | Historical status |
|---:|---:|---:|---:|---:|---|
| 0 | 0.464320942814 | 0.464320927570 | 1.5244×10⁻⁸ | 1.1524×10⁻⁷ | optimal |
| 1 | 0.475931947131 | 0.475931920925 | 2.6206×10⁻⁸ | 1.2621×10⁻⁷ | optimal_inaccurate |
| 2 | 0.478907410948 | 0.478907410494 | 4.5412×10⁻¹⁰ | 1.0045×10⁻⁷ | optimal |

The largest reported LP inequality residual is 6.1×10⁻¹⁰; the certificate is recomputed directly from multipliers and row minima rather than trusting epigraph values. The 10⁻⁷ subtraction is a conservative numerical allowance, not a validated interval-arithmetic error bound. It leaves the fitted objective gap below 1.27×10⁻⁷ at every anchor. This supports near optimality for the **fitted finite programme** and rules out a material optimizer shortfall at the scale of Q's 0.00139–0.00553 fixed-objective premium. It does not certify the empirical privacy model against a full-H recipient or the independently trained audit probe.

For comparison, setting `λ=0` gives exactly the unconstrained lower bound `Σ_t min_z C_tz`; D17 attains it in all 32 supported rows at every anchor. Since D17 violates all four Q budgets, this simpler lower bound alone cannot certify the constrained optimum. The multipliers above supply that missing check.
