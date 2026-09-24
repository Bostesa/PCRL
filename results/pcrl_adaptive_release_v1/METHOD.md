# Adaptive release method and contract

This is a supervised residence-utility **2018 development** mechanism. The implemented candidate has a private 32-state historical T0 encoder, a 32×17 stochastic channel, and one token beside the unchanged service vector. The original T0 code was trained with residence labels; neither the encoder nor this study supports a label-free or unseen-task claim. The exact selected channel, hashes, and validation-selected decoders are recorded in the fitted private center receipts and later selection lock. A learned child-state partition was attempted under the registered support rule; its result is reported separately in `STATE_REFINEMENT.md`.

## Deployment inputs and output

The local encoder receives only the allowed `X_A,H_A` and the frozen, hash-pinned encoder/preprocessing objects. For a child partition it may also compute frozen task-posterior/residual and SEX/RAC1P *predictions* from those same local inputs. It may not read the current person's observed residence or protected label, `H_B`, loss, household role, or private cache key as a predictive feature. The private code/cell and channel probability row are never released. Each record emits one sampled integer `Z∈{0,…,16}`; A sees `(H_A,Z)`, AB sees `(H_A,H_B,Z)`, and B keeps its unchanged historical service. `H_A` and `H_B` are byte-preserved. The wire does not contain a missingness flag, full kernel row, random seed, or unused draws.

`fit_b.RefinedReleaseSession` implements the one-token interface with an immutable-input cache and an optional private HMAC replay key. The latter provides a stable computational draw across restarts, not a mathematical independent redraw. Exact token expectation is used privately for evaluation; it is not a wire representation. Repeated independent releases or composition with previously available tokens have not been analyzed here.

## Coefficients and reference calibration

Let `c(i)` be the private state, `q_cz` a row-stochastic 17-token law, `w_i^v` separately normalized original-person weights for `v∈{U,PWGTP}`, and `d(y_i|H_Ai,z)` a frozen task decoder. For a frozen attack `a(s_i|H_ri,z)` with legal role `r∈{A/SEX,A/RAC1P,AB/SEX,AB/RAC1P}`, define

```
C[d,v,c,z] = Σ_{i:c(i)=c} w_i^v [-log d(y_i|H_Ai,z)]
A[r,a,v,c,z] = Σ_{i:c(i)=c} w_i^v [-log a(s_i|H_ri,z)]
C[d,v](Q) = Σ_c,z C[d,v,c,z] q_cz
L[r,a,v](Q) = Σ_c,z A[r,a,v,c,z] q_cz.
```

Every coefficient already contains state mass and is formed on original people; multiplying by state mass again or counting 17 expanded token rows as new people is invalid. The exact token loss is the expectation of `-log p(label|H,z)`, not `-log Σ_z q(z|x)p(label|H,z)`. Protected classes retain the original full schemas, including classes absent from a fitting role, with a common probability floor.

Historical D17 is the fixed `Q_ref`. On the **same coefficient people, weights, class order, and clipping** for each retained attack bank `B_k`, calibration is

```
rho[r,v,k] = min_{a∈B_k[r,v]} L[r,a,v](Q_ref).
```

Every retained cut then requires `L[r,a,v](Q) ≥ rho[r,v,k]−delta`; `delta∈{0,.001,.003}` is an optimization allowance, with `.001` the three-anchor center. After adding any attack, all old cuts are rebased together. For nonnegative delta, D17 is a constructive witness: every `L_a(D17)≥min_b L_b(D17)≥rho−delta`. The code independently replays that fact and phase I before solving. This is a relative **fitted-risk** target. It is not conditional mutual information, arbitrary-attacker protection, or a population privacy guarantee.

For fixed decoder and attacks, the channel step minimizes `0.5 C[d,U](Q)+0.5 C[d,PWGTP](Q)` subject to those cuts and row simplices. It is a finite LP in `number_of_cells×17` variables, plus one inequality per retained attack/role/weighting. Both task and attack losses are affine only while the predictors are frozen. Valid LP/dual and deterministic MILP bounds concern the *same* fixed bank, decoder, rows, and tolerances; they do not bound the adaptive attacker oracle or outer performance.

## Bounded updates and selection

Branch A starts from the historical T32 code. Its initial task decoder is fitted under an equal current/D17/uniform-token coverage law on allowed training households. It freezes fitted decoder choices before coefficient construction. Each update solves the calibrated channel LP, refits a task decoder on the current channel's exact expected-token training law, retains prior decoders, fits fresh sensitive best responses on the audit-fitting households, scores their cut coefficients on the separate coefficient households, rebases the bank, and re-solves. The algorithm retains all previous attacks. At most six update rounds were registered. Inner selection chooses a saved channel/decoder without using outer assessment labels. A fresh, common independent audit slate is fitted for all compatible candidate/control releases; validation chooses its probes and attacks, then inner checking scores them.

Branch B starts from T32 and can split a parent using deployable `H_A`, frozen task residual/posterior, and frozen nuisance sensitive-risk predictions. Nuisance fits use only globally assigned nuisance households. At fixed decoder/bank dual prices, the split freedom score is `min_z Σ_parent g_i(z) − Σ_child min_z Σ_child g_i(z)`, with `g_i(z)=u_i(z)−Σ_j λ_j a_ij(z)` and already normalized per-person contributions. It is nonnegative for an exact split at fixed prices, but is neither a feasible primal gain nor a held-out gain. Children must meet the preregistered 100-unique-household and 100-effective-weighted-support floors, and copying parent rows into children must preserve every old release. No extra token value or `H_B` encoder access is permitted. Simple task-only, nonadaptive joint-risk, random eligible partitions, and same-partition deterministic optimization are its attribution controls when a child partition exists.

```text
verify Linux x86 frozen objects, service bytes, household roles and class support
fit common task decoder; load D17 as Q_ref
for each registered branch/allowance/anchor within the queue:
    fit or restore frozen attack bank and coefficient matrices
    repeat up to six bounded rounds:
        rebase every cut to same-row min-bank D17 risk
        replay D17 witness and phase I
        solve the fixed-decoder channel LP; save primal/dual/residuals
        fit equal-budget decoder and best-response attacks on allowed inner roles
        retain all valid attacks and previous decoders; save the checkpoint
    fit and archive matched deterministic, gradient and simple controls
fit one common validation-selected independent audit slate per distinct release
choose route slots and family representatives from inner_selection only
commit and remotely verify the lock; then score outer development households once
```

The decoder/attack updates and any partition choice make the complete procedure adaptive and generally nonconvex. A minimum over fixed-decoder affine task losses is concave in `Q`; minimizing it jointly with `Q` is not solved globally by the individual LP steps. A failure of heuristic best responses to find another cut is an oracle stopping observation, not a certified oracle gap. The fixed-bank same-partition MILP/search and exact-gradient controls receive the same state/token contract and permitted supervision; continuous J is a contextual reference with a different interface.

For `n` people, `k=17` tokens, `m` retained attack cuts, and `t` states, constructing exact coefficients needs `O(nk)` frozen-predictor evaluations plus aggregation; a channel LP has `tk` probabilities and `m` attack inequalities. The repeated decoder/attack fits dominate elapsed time. Empirical three-anchor comparisons share households, so anchors and token expansions are not independent sampling units. Households are paired in uncertainty calculations; intervals condition on chosen fitted objects and omit full adaptive selection and survey-design uncertainty.

The ingredients of stochastic adversarial release, randomized constrained prediction, privacy-guided quantization, and side-information/collusion analysis are established. The study tests whether this particular task-aligned, reference-feasible finite release construction and legal internal distinctions earn a measurable development advantage. See `agents/prior_art/PRIOR_ART_SCOPE.md` for primary-source equation and section pointers. A numerical bank gap or preserved service bytes alone do not establish the user-facing tradeoff, algorithmic novelty, or a population information guarantee.
