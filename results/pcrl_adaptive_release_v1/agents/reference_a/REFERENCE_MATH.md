# Reference and alternating-update mathematics

For one recipient/attribute/weighting group `r`, let `A_a[t,z]` be the
coefficient matrix of fixed fitted attack `a`, with normalized original-person
mass **already inside** the entries. On the exact coefficient people used by
the channel solver, set `ell_a(Q)=sum_t,z A_a[t,z] Q[t,z]`, and for the unchanged
D17 release `Q_ref`, set `rho_r(B)=min_{a in B_r} ell_a(Q_ref)`. The constraints
are `ell_a(Q) >= rho_r(B)-delta_r` for every retained attack. If
`delta_r >= 0`, then `ell_a(Q_ref) >= rho_r(B) >= rho_r(B)-delta_r` for each cut.
This is a constructive primal witness. It holds even for an H-only attack
whose coefficient row is constant across actions. The old selected-on-one-pool
reference did not have that property when its selected predictor lost rank on
the coefficient pool.

Adding an attack can lower `rho_r`; **all** old floors must be rebuilt together.
This is a moving empirical fitted-risk target. A larger bank need not make its
feasible set smaller after rebasing, and no monotone privacy claim follows.
An optional paired-cut rule `ell_a(Q) >= ell_a(Q_ref)-delta` would also keep the
witness but is stricter for non-minimizing attacks; it is a separate ablation,
not an unannounced replacement.

For frozen task decoder `d`, the coefficient matrices `C_U,C_W` encode exact
per-token CE, and the objective is `sum 0.5*(C_U+C_W)*Q`. The fixed-bank,
fixed-decoder update is an LP. Refit a decoder using exactly the current
channel's token distribution (with registered uniform/reference coverage),
select only on inner development rows, rebuild its coefficients on the
coefficient rows, then resolve. Retained decoders remain available for
validation selection. For a fixed decoder family, `min_d L_Y(d,Q)` is a
minimum of affine functions and is concave in `Q`; jointly minimizing over
decoder and channel is generally nonconvex. No LP gap proves global joint
optimality. Validation-selected decoder changes and bank rebasing also break
any simple training-objective monotonicity statement across rounds.

The read-only [historical replay](HISTORICAL_REPLAY.json) hash-checks the old
312-cut banks and D17 maps for anchors 1 and 2, without accessing person rows
or fitting models. The old maximum cut violations at D17 were 0.0095631768
and 0.0074400880 nats, respectively. The published single H-only cuts alone
proved unavoidable gaps 0.0080164577 and 0.0056375454. Recomputing the min
bank reference on the same coefficient rows gives zero maximum D17 violation
for both full banks at delta zero. These new floors belong only to the new
development protocol; they do not revise the closed result.

No part of the construction establishes a Bayes-risk bound, population MI,
full-H conditional privacy, or protection against an arbitrary attacker.
