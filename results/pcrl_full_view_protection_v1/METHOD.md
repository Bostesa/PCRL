# Method investigated and implementation scope

## Interface and optimization

The prospective interface is unchanged `H_A` plus one categorical token Z for A. The coalition holds unchanged `(H_A,H_B)` plus Z. J is an alternative comparator, not an assumed previous disclosure. T is the existing residence-supervised T0 encoder code from permitted PCA32 and H_A; the local mechanism sees T, never H_B, S, Y or held-out labels at deployment. The private coin is independent and sampled once per person. The 2018 Q and D17 kernels were read unchanged for radius diagnostics.

The finite candidate is `min_Q E_{P0}[c(T,Z,H_A)]` subject to `I_P(S;Z|H_A)<=δ_A` and `I_P(S;Z|H_A,H_B)<=δ_AB` for *each* entire joint law P in a fixed finite list U. The output alphabet, encoder access and cost are identical for all synthetic comparisons. The implemented action cost is binary Hamming loss; the 2018 fixed residence decoder and 17 actions would be inherited only if a pilot were triggered. Finite-law CMI is convex in Q for fixed preprocessing. The implementation uses CLARABEL via CVXPY, re-evaluates every full-view CMI independently, and records a Lagrangian supporting-hyperplane lower bound and a feasible-channel upper cost. The numerical guard is `1e-7`. `optimal_inaccurate` and wide brackets are retained, not hidden.

For the registered suite, U is either a single exact law or three entire laws formed by `P(T|S,H_A)` shifts. Amendment 2 separately adds nine coherent laws varying both `P(T|S,H_A)` and protected priors. This is **scenario robustness**, exact only for the enumerated list. There is no exact separation oracle for a continuum of ACS conditional laws. The matched adaptation of the published uniform robust-information approach to the same conditional Shannon-MI target gives the **identical programme**; it is not a second algorithm or fit.

## Controls and fairness

The `0,.001,.01,.05` budgets are explicit CMI or radius constraints, not the historical empirical `+.001` attacker cap. The controls hold T, binary action space, task cost and access equal: all deterministic maps by enumeration, zero/constant and unprotected endpoints, randomized response and withholding at the three fixed levels, nominal full-view CMI, finite-list robust full-view CMI, and all-input radius. The best deterministic/RR/withholding comparisons are selected only among channels feasible at the *same actual full-view* target. In the list-uncertainty case, nominal channels at `.01` and `.05` violate the list target and are reported as diagnostics, not feasible winners.

The information-radius control solves expected task distortion with a per-row KL bound to one reference output law. For the unchanged 2018 kernels, a Blahut–Arimoto input prior supplies a lower capacity bound and its induced reference r supplies the maximum-row-KL upper certificate. All 32 allowed input rows are included even if a deployment row lacks training support; all-zero output columns are removed only for the computation. A state-dependent Q would require a separate row-radius bound in every publicly known state.

## Implementation files and reproducibility

- `experiments/pcrl_full_view_protection_v1/finite.py`: exact finite-law CMI evaluator, chain-rule terms, information-radius bracket.
- `synthetic.py`: rational law generator, fixed Hamming cost and task information.
- `optimize.py`: small convex programmes, independent feasibility checks, affine objective lower bounds, exhaustive deterministic controls.
- `run_synthetic.py`: atomic per-fixture checkpoints and resource measurement; skips completed checkpoints on resume.
- `summarize.py`: complete table and numerical audit.
- `radius_existing.py`: unchanged 2018 kernel radius computation with SHA-256 input verification.
- `tests/pcrl_full_view_protection_v1/`: XOR, coalition conditioning, row-radius orientation/zero support, law normalization, safe task, solver bounds and CMI-gradient finite differences.

Run on Python 3.14 with NumPy 2.4.4, SciPy 1.17.1, CVXPY 1.9.3 and CLARABEL 0.11.1. CVXPY and solver wheels were installed only under `/private/tmp/pcrl_fullview_py` for this run; no main-environment mutation or cloud fit occurred. `REPRODUCE.md` gives commands. The public synthetic checkpoints contain no person rows. The 2018 archived Q and D17 files remain in their owning worktree; this branch stores only hashes and aggregate bounds.

## Decision boundary

The exact finite-law programme protects the full finite H view and can preserve useful task signal. The present candidate does not add a distinct optimization algorithm to prior robust privacy work. Its continuous-H extension requires uniform conditional-law assumptions at roughly `3e-4` TV precision to support a `.01`-nat 17-action bridge from zero reference leakage; no such confidence envelope is available for the repeatedly used 2018 ACS rows. The registered ACS gate is therefore not met, and 0 of the maximum 27 new main fits were started. This is a decision about this candidate and evidence, not an impossibility theorem for every attribute-specific release.
