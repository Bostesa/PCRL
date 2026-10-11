# Affine finite-law LP, R2

Each row denotes an admitted person and each action an actual original emitted
global-code tuple. Let `q = base + delta @ theta`; `delta` is CSR with one row
per person/action. Shared parameters are context coefficients or shared action
template weights, never per-person fitted coefficients. Unchanged metadata and
decoded probabilities must be deterministic functions of emitted codes.

The caller must guarantee all supported laws are normalized and nonnegative,
or supply explicit simplex equalities/inequalities. Allocation bounds [0,1]
and a frozen donor guarantee this for G/C. Free templates need explicit
simplex equalities. The kernel checker validates the selected solution.

For each view v and binary SEX b, aggregate base and derivative through the
actual emitted IDs to obtain `m_bv = base_bv + D_bv theta`. Epigraph t_vz obeys
`D_bv theta - t_vz <= -base_bv`. The minimization objective is
`sum t_pair + .5 sum t_r1 + .5 sum t_r2`. The empirical joint distribution
uses row weights summing to one; the default is equal people. No conditional
SEX normalization enters the Bayes objective. Equal-class AUC is a separate
reporting estimand.

All expected loss/CE constraints are affine: `E(loss) = intercept + c theta`.
For each named cap caller provides `c theta <= bound - intercept`. ORIGINAL
uses direct U0/Ucal reference means on the identical role/fold; negative
remaining headroom is retained. Accuracy loss is negative accuracy change.
Action loss is averaged, not loss evaluated at averaged probability.

Utility-only maximum change uses the indicator that either emitted code
differs from the input identity tuple. This excludes donor identity and any
alias action positions. Maximum change and minimum loss do not consume SEX.

## Serialization and independent certificate

`LPProgram` supplies objective c, CSR A_ub, b_ub, CSR A_eq, b_eq, bounds,
inequality names, parameter count and constant objective offset. Save each
array in the private artifact tree. Optimization results supply x (full
primal), theta, inequality duals y<=0, equality duals free, lower-bound duals
l>=0 and upper-bound duals u<=0. These follow SciPy/HiGHS minimization signs.

Independent replay checks `c - A_ub.T@y - A_eq.T@ye - l - u = 0`, primal
feasibility, dual signs and finite-bound complementarity. Dual objective is
`b_ub@y + b_eq@ye + finite_lower@l + finite_upper@u + constant`; primal
objective is `c@x + constant`. Success requires all absolute violations,
complementarity errors and gap <=1e-7 (or max(1e-7,10solver_tolerance)).
This certificate concerns only the supplied finite empirical law, support
and constraints. It makes no population or held-out AUC guarantee.

G is nested in C: assigning the same scalar to every supported context
reconstructs G exactly, including identity fallback. Identical constraints
and objective therefore imply optimal C objective <= optimal G objective
in exact arithmetic. This inclusion is not evidence of a strict advantage.
A fitted strict improvement must exceed the independently recomputed
numerical error and solver gap; its held-out effect requires fresh readers.

A solver infeasible status without a supplied dual ray is explicitly
uncertified infeasibility. It blocks that solve, but does not certify global
impossibility. A restricted master is certified only within its columns.
`certify_infeasibility(program)` is a separate explicitly counted Phase-I
LP minimizing one nonnegative common slack for all inequalities and both
signs of each equality, retaining parameter bounds. A certified positive
dual objective exceeding tolerance proves infeasibility for the original
finite constraints within those bounds. No extra solve runs implicitly.
Every omitted allowed column must be evaluated before full-support
optimality is claimed. Pricing is `new_c - new_A.T@y - new_Aeq.T@ye` for
nonnegative omitted variables. Negative reduced cost invalidates a
full-support optimality claim.

## Public API

`build_mixture_problem(emitted1, emitted2, identity1, identity2, donor,
contexts, coefficient_mode='scalar'|'conditional', weights=None)`.
`AffineLaw(..., base, delta, bounds, weights=None, equality_matrix=None,
equality_rhs=None)` for other affine kernels. `ConstraintRows(names,matrix,rhs)`.
`law.expected_coefficients(action_loss,mask)` returns intercept and slope.
`law.change_coefficients()` returns actual-change intercept and slope.
`protection_program(law,sensitive,rows)` exposes epigraph matrices for MIP.
`solve_protection(...)`, `solve_linear(...,maximize=True|False)` return all
certifying arrays and numerical metadata. These modules have no loaders.
