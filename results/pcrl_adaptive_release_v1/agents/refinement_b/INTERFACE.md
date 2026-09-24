# Branch B refinement interface

`experiments.pcrl_adaptive_release_v1.refinement` supplies a private nested
state router. Start with `NestedPartition.base(32)` and add a split with
`partition.split(leaf, feature_name, threshold)`. The split keeps the old leaf
for values at or below the threshold and appends one child ID. `route_runtime`
accepts the historical `RuntimeInputs(X_A,H_A)`, a frozen historical encoder,
and a separately fitted, frozen full-schema nuisance predictor with
`probabilities(RuntimeInputs) -> {"SEX": (n,2), "RAC1P": (n,9)}`. It computes
the T0 code and all legal split features without a person's task/protected
label, H_B, row loss, household ID, or assessment role. The caller must verify
that nuisance fitting used the registered dedicated household resource.

`partition.parent_of_leaf` maps every child to its T32 parent.
`copy_parent_kernel(Q32, parent_of_leaf)` copies the 17-token parent rows
exactly and is the constructive feasibility witness while data rows, decoder
and attack bank are fixed. The router has JSON `to_record`/`from_record`
methods, validates feature names, and stops at 128 total cells.

`priced_contributions(task_rows, attack_rows, multipliers)` implements
`u_i(z) - sum_j lambda_j a_ij(z)`, with nonnegative dual prices and all U/PWGTP
normalization **already included** in each original-person row. For a binary
split, `fixed_price_gain` compares the parent action minimum with the two
child minima. This nonnegative score is fixed-price freedom, not feasible
primal gain or held-out privacy. `rank_splits` searches a bounded set of
training-only quantiles and enforces both unique-household and Kish effective
household support per child; optional checking scores require disjoint
households. The caller fixes features, support, candidate limits, dual prices,
tie rule, and checking resource before reading validation outcomes.

`fixture_b1` uses exact fractions for sensitive independence, enumerates all
15 deterministic partitions, and yields 0.08972125227425051 nats task
information for the stated stochastic X-channel. `fixture_b2` checks exact
conditional sensitive independence and yields log(2)/2 nats task information
for a deterministic context-dependent channel. Both are synthetic proofs of
possible gains from richer internal distinctions, not ACS findings or a new
population guarantee.

`nuisance.fit_frozen_nuisance` trains and saves one full-schema SEX/RAC1P
probability model using only the global `nuisance_train` role and local
`RuntimeInputs(X_A,H_A)`. Absent RAC1P classes remain in the nine-class schema
at the declared probability floor; this does not impute learned support.
`fit_b.run_center(anchor, delta, index_path, output_dir,
a_center_dir=..., max_states=64, max_rounds=6)` is the central-queue callable.
It verifies the completed A bank and decoder, registered pre-B amendment,
source bytes, and role hashes; prices splits on `coefficient_split`; checks the
fixed 25% gain ratio on `inner_selection`; enforces 100 unique and effective
households per child; and never reads `inner_check` labels. It reaggregates
the frozen person-token losses under child IDs, solves the calibrated bank,
and retains bounded fresh best responses. Earlier Q rounds must pass the
final rebased union bank before selection. The selected Q is loaded from the
exact saved round. A no-split result is an explicit A alias with a reason.

On the pinned Linux x86 worker, invoke:

```sh
python -m experiments.pcrl_adaptive_release_v1.fit_b fit \
  --anchor 0 --delta 0.001 --index-path results/pcrl_task_aligned_cuts_v1/REUSABLE_INPUTS_PINNED.json \
  --a-center-dir PRIVATE_A_CENTER --output-dir PRIVATE_B_CENTER \
  --max-states 64 --max-rounds 6
python -m experiments.pcrl_adaptive_release_v1.fit_b parity \
  --anchor 0 --index-path results/pcrl_task_aligned_cuts_v1/REUSABLE_INPUTS_PINNED.json \
  --center-dir PRIVATE_B_CENTER --receipt-path PRIVATE_PARITY_RECEIPT.json
```

`parity` verifies the selected channel, complete inventory, original
coefficient role, pinned frozen encoder, nuisance weights, historical T32
codes, child IDs, task posterior/residual, and H_A bytes. It writes its
separate private receipt only after Linux x86 parity passes; putting that
receipt inside the immutable fit directory is refused. The fit receipt says
`REQUIRED_BEFORE_PRODUCTION` until this check is complete. The Mac can test
stored predictions and synthetic fixtures, but cannot substitute a fresh
float32 T0 encoding for the archived Linux codes.

`RefinedReleaseSession` takes selected Q, partition, verified encoder/nuisance,
the matching parity receipt and private 32-byte-or-longer replay key. It
routes only X_A/H_A to a child, samples exactly one of the historical 17
tokens, and emits only unchanged H_A and that token. Keyed HMAC replay gives
the same draw across process restarts for the same private record ID and
immutable input snapshot. The private key, child ID, Q row and random draw
are not on the wire. This is computational replay, not an information
theorem about repeated independent releases.

The new B fit has no ACS scientific result from this agent. Exact synthetic
B1/B2 fixtures and a supported-child two-round saved-fit/resume fixture
verify the implementation, not advantage over matched controls. The
coordinator alone dispatches ACS fits and independent audits.
