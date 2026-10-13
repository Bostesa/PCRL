# Frozen C1 head geometry specification

This is a diagnostic compression baseline for the current head. It does not promise usefulness for arbitrary future tasks, and it is not a privacy training contribution. This specification is written before real feature projection, reader fitting or INNER labels are opened.

## Inputs and precise arithmetic

Use the admitted current affine head with the predecessor's folded `StandardScaler + LogisticRegression` useful heads for U0. Never use the older historical heads in `model.pt`. Encoders retain the admitted float32 arithmetic; cast the resulting z to float64 before compression and head evaluation. Preserve the admitted head class order, first-class tie convention and binary folding convention.

Let W have K rows, and let A=W−mean_classes(W), c=b−mean_classes(b). Compute a float64 thin SVD A=U diag(s) Vt. Freeze cutoff

`tau=max(1e-12,32*max(K,D)*eps64)*s_max`.

Retain s>tau. Any positive singular value in [0.5 tau,2 tau] is a rank-boundary ambiguity and fails package admission. An all-zero A has rank zero. Do not use an absolute singular-value floor, which could erase a uniformly small but useful head. Measure every rank: class centering bounds income rank by one and occupation rank by five; neither nonzero rank nor full theoretical rank is assumed. Canonicalize each retained right singular vector by making its largest-magnitude coordinate positive; serialize the actual B matrix to avoid future SVD basis ambiguity.

Define B as the retained orthonormal rows, P=BᵀB=A⁺A under the frozen rank rule. Verify `A@P=A` with atol=rtol=1e-10, and report the maximum absolute residual. Reject a failing package.

## Recipient contract and rounding

Compute internal q=z Bᵀ, then transmit protected_z=q B (D=16 source-coordinate columns). Do not evaluate raw z against rounded P to construct a released coordinate: raw residual-dependent rounding can be amplified by a reader that retains arbitrarily small positive variance. Do not add the discarded residual or raw z. q is internal, never an additional per-row channel. All transmitted coordinates are functions of the same q and fixed public matrices.

Compute actual logits as `protected_z @ W.T + b`, then center across classes. This is the current head evaluated on the actual protected representation, including the original affine arithmetic order. Use softmax probabilities and the first class in source order at an argmax tie. The wire is `Pz16, centered_own_logitsK, own_probabilitiesK, own_decision1`. No clean teacher outputs, source z or labels enter it. Fixed W,b,B,P and the SVD rule are public reader knowledge.

In exact arithmetic, Pz is reconstructible from centered logits and fixed head parameters. Re-expression of those scores is information-equivalent. Numeric release comparisons must still use the actual rounded transmitted wire. Retaining 16 storage columns does not confer 16 effective degrees of freedom; a lower rank disclosure must not be advertised as an equal-capacity representation improvement.

## Real-row admission checks

On authorized FIT, ATT and INNER feature exports only, compare unprojected actual head outputs to protected outputs. No labels are needed. Freeze logit atol=rtol=1e-10, probability atol=1e-10/rtol=0. Require exact decision equality on every checked row. A top-two original logit gap ≤1e-10 is a near tie; count and report these rows and mismatches without granting an exception. If any decision changes, the package is not decision preserving even if probability tolerance passes.

Attack actual Pz with its own outputs on both recipients and their aligned pair for the frozen U0/RAW/JOINT anchors. Pair banks inherit ignore-other aliases. Do not use original outputs as a second channel. Do not conceal unavailable rank-boundary states or parity failures.

## Interfaces and replay

`occupation_bottleneck.geometry.build_geometry(W,b,class_order)` returns the fixed head/basis artifact. `projected_release(z,g)` produces the exact five-field representation/logit/probability/decision/full dictionary. `audit_geometry(z,g)` produces label-free parity and near-tie aggregates. `public_geometry(g)` serializes the public fixed metadata.

Data-free fixtures in `occupation_bottleneck/test_geometry_gradient.py` cover binary rank, nullspace channel removal, six-class rank, actual protected head arithmetic, zero-rank ties and an ambiguous singular value. They are queue-owned tests, not real-data admission evidence. An independent verifier must reconstruct its own SVD and compute probability/decision parity from pinned model/features rather than importing these routines.
