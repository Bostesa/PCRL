# Full-view information accounting

All statements concern one fixed encoder and public kernel, one sampled token, and the distribution named with the statement. They do not protect the training records from inference through learned parameters. Notation follows `PROTOCOL.md`. The code T may depend arbitrarily on permitted PCA32 and H_A; it does not see H_B. The random draw of Z is independent of `(S,Y,H_A,H_B)` given T, unless a stated public-state kernel is used.

## 1. Coarse versus full conditioning

Because `B=b(H)`, `I(S;Z|H)=I(S;Z|H,B)`. Expand `I(S,H;Z|B)` in two orders:

`I(S,H;Z|B) = I(S;Z|B)+I(H;Z|S,B) = I(H;Z|B)+I(S;Z|H,B)`.

Therefore exactly

`I(S;Z|H) = I(S;Z|B) + I(H;Z|S,B) − I(H;Z|B)`.

This is a chain-rule consequence, not a new theorem. The correction has either sign. With independent fair bits S,H, B constant and `T=Z=S xor H`, the coarse term is zero, `I(H;Z|S,B)=log 2`, `I(H;Z|B)=0`, and the true full-view leakage is `log 2=0.69314718056`. If instead S=H=Z with B constant, coarse leakage is `log 2` and full-view leakage is zero. An executable finite calculation is in `tests/pcrl_full_view_protection_v1/test_finite.py`.

For standard Borel continuous H and finite S,Z, regular conditional distributions exist and each term is well-defined; all displayed MI terms are finite, bounded by `log|Z|` where Z is involved. The identity still holds for the exact law. Conditioning on the 2-cell A or 4-cell AB historical partitions supplies neither the correction nor its sign. These examples do not show that the coarsening *caused* Q's measured loss to D17.

## 2. All-input information radius

For a finite channel Q with all *deployment-allowed* input rows T, set

`R(Q)=inf_r max_t KL(Q(.|t)||r)`.

The infimum ranges over output distributions whose support contains every positive output mass of an allowed row. If `r_z=0` for some `Q(z|t)>0`, that row's KL is infinite. Columns zero in every row may be dropped. A particular reference r gives the explicit **upper** certificate `R(Q) <= max_t KL(Q_t||r)`. For finite alphabets the matching channel-capacity expression `max_pi I_pi(T;Z)` is a **lower** bound for any proposed r, with equality at the optimum information center [van Erven–Harremoës §VI-B](https://arxiv.org/pdf/1206.2459). The computation records both sides, including all 32 rows and zero-output handling; it never calls capacity maximization an upper certificate.

Let W denote any recipient's existing view and suppose `(S,Y,W) -> T -> Z` under independent release randomness. Conditional data processing gives `I(S;Z|W) <= I(T;Z|W)`. For each w and any r,

`I(T;Z|W=w) = Σ_t p(t|w) KL(Q_t || q_w) = Σ_t p(t|w) KL(Q_t || r) − KL(q_w || r) <= max_t KL(Q_t||r)`, where `q_w=Σ_t p(t|w)Q_t`. Averaging and minimizing r proves `I(S;Z|W)<=R(Q)`. The same argument with Y proves `I(Y;Z|W)<=R(Q)`. It holds for A, AB, and even additional side information so long as the Markov premise remains true. A public-state kernel `Q(.|t,u)` needs a separate reference and radius for *every* u; the view W must already contain u. The local encoder cannot use H_B.

For Bayes-optimal log-loss prediction, the expected improvement from W to `(W,Z)` equals `I(Y;Z|W)` and is thus bounded by R. No equality transfers that bound to a finite or imperfect J predictor's measured gain. In particular, the historical +.001-nat cap on `CE_J−CE_M` for a specified attacker slate is neither R nor a CMI budget.

The unchanged archived 2018 Q and D17 kernels were read by SHA-256 from Terminal 3's reusable-input manifest, with no 2016 rows and no refit. `EXISTING_RADIUS.json` contains the reference vector, maximizing input prior, row count and numerical bracket. The upper includes a `5e-13` floating guard. Q's Blahut–Arimoto iteration hit the registered 100,000-iteration limit before a `1e-10` gap, but its reference remains a valid upper certificate; the lower/upper gaps below remain tiny. D17 has 16, 14 and 15 distinct used actions respectively, so its exact deterministic radius is the corresponding logarithm.

| Anchor | Kernel | Capacity lower (nats) | Reference upper (nats) | Interpretation |
|---:|---|---:|---:|---|
| 0 | Q | 1.762167 | 1.762168 | exceeds `log 2`; below `log 9` but far above `.01` |
| 1 | Q | 1.936616 | 1.936616 | same |
| 2 | Q | 1.967602 | 1.967602 | same |
| 0 | D17 | 2.772589 | 2.772589 | `log 16` |
| 1 | D17 | 2.639057 | 2.639057 | `log 14` |
| 2 | D17 | 2.708050 | 2.708050 | `log 15` |

The lower value is not itself a sensitive leakage claim; it shows how expensive the distribution-free all-input ceiling is for these unchanged kernels. A `.01` radius would cap any Bayes incremental task log-loss gain at `.01` nats. The completed 2016 study's descriptive Q gain over H was about `.026` nats (Terminal 2 review M3); that is a scale comparison, not a new 2016 evaluation or a causal attribution.

## 3. Attribute-specific finite and conditional guarantees

For a known finite joint law P of `(S,Y,T,H_A,H_B)`, evaluate `I_P(S;Z|H_r)` by summing the exact joint table after applying Q. Fix the law, encoder and cost. This CMI is convex in Q: each conditional law `P(S,Z|H_r=h)` is affine in Q, and mutual information is convex in the channel with its input law fixed. The supremum over a *fixed* set U is convex. This does not produce an exact worst-case oracle for a large or continuous U. A finite list of entire laws can be enumerated exactly, but certifies that list only. Local and coalition constraints must come from the same full joint law, not unrelated fitted conditional tables.

If the true law is known and finite, recomputation of every full-view CMI gives an exact mathematical guarantee (up to numerical evaluation tolerance). If the true law belongs to U, a robust feasible Q guarantees the declared target under that assumption. A claim that the true law lies in U with a chosen confidence level additionally needs a coverage proof under explicit sampling assumptions. Neither calibration, bootstrap variation, a finer partition nor a validation average provides uniform coverage of `P(T|S,H=h)` for sparse continuous H.

## 4. A quantitative conditional-law bridge and its cost

Here is a valid but demanding route for continuous H. Suppose for **every** H value in deployment support a specified reference conditional joint law `P0(S,T|H=h)` satisfies `TV(P(S,T|h),P0(S,T|h))<=ε`, with a fixed Q. Data processing gives the same TV bound for the induced `(S,Z)|h` and its S and Z marginals. Write `f_d(ε)=ε log(d−1)+h_2(ε)` for `0<=ε<=1−1/d`, and `f_d(ε)=log d` otherwise, where `h_2` is binary entropy. The finite-alphabet entropy-continuity inequality applied to `I=H(S)+H(Z)−H(S,Z)` yields

`I_P(S;Z|H=h) <= I_P0(S;Z|H=h) + f_|S|(ε)+f_|Z|(ε)+f_(|S||Z|)(ε)`.

Thus if the *per-h* reference leakage is at most δ0 everywhere, the actual full-view CMI is at most δ0 plus the displayed continuity term under any H marginal. If only an average reference CMI is known, a shift in `P(H)` needs a separate bound; the displayed average cannot silently be reused. A known prior error `TV(P(S|h),P0(S|h))<=ε_p` and class-conditional input errors `sup_s TV(P(T|s,h),P0(T|s,h))<=ε_t` imply the required joint error at most `ε_p+ε_t` by a triangle coupling bound. Zero-class support or an unbounded/unsupported H region cannot be assigned ε=0; without an envelope its conditional class law is unrestricted.

For 17 token actions and δ0=0, the TV radius required to keep the continuity term at or below `.01` is at most `2.9735e-4` for binary SEX and `2.6617e-4` for nine-class RAC1P. At `.001` it is `2.4305e-5` and `2.2212e-5` respectively. These are sensitivity thresholds, **not estimated ACS radii**. Even before adding estimation and numerical errors, the available records provide no uniform continuous-H conditional-law envelope at such precision. Consequently this theorem does not certify ACS full-view CMI. The trivial ceilings are `log 2=.693147` and `log 9=2.197225`; a proposed bound must be compared to these and the declared δ.

## 5. Scope of any solution

An exact finite-law selective channel can preserve task information independent of S while having zero `I(S;Z|H)`, unlike a `.01` all-input radius channel that cannot carry more than `.01` task nats. This is the established privacy-funnel phenomenon, not yet an algorithmic advance. If U permits *arbitrary* `P(T|S,H)` and the channel has two different reachable rows, one may assign those rows to two S values after conditioning on H; a positive binary channel information results. Nontrivial attribute-specific guarantees require genuine conditional-law restrictions or a nearly constant channel. Any certificate must name that restriction, the one-token scope, and the fixed training/model scope.
