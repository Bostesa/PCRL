# R4 matched-control specification, before scientific fitting

All new controls use original P0 global IDs, fixed probability bank, class-preserving NARROW neighbor templates, full expected action costs and the identical R3 utility/reference/fixed-reader clauses. No head, teacher, temperature, partition or reader guard is refitted here. Exact public output aliases may reuse attacks only after law and wire identity verification. Final controls are triggered only for the single FIT-selected nominee budget/support. Wider-support adapters require a dated pre-fit parameterization amendment; an unfinished wide comparison is not a narrow win.

## Families

* **Local**: one recipient marginal distribution per own frozen P0 code. Both distributions are fitted under the shared pair-plus-half-local empirical Bayes objective by bounded alternating LPs; deployed marginal routing reads only that recipient's own P0 code. Per-purpose own-code groups can have more parameters than joint context groups. This inherent difference is reported. Two independent cached coins generate the pair.
* **Central product**: one marginal distribution per purpose and pooled input context, both routing from the same permitted central P0 inputs. Two independent cached coins. Joint probabilities are products. Coordinate subproblems are LPs; the entire optimization is bilinear, with unresolved global gap.
* **Deterministic**: one binary action choice per pooled input context, with a 16-way exact simplex and the same full public-tuple epigraph objective. Fit a genuine mixed-integer problem. A separately counted continuous relaxation supplies a bound. No rounded stochastic fit is presented as training.
* **Sequential 1→2 and 2→1**: optimize first recipient's context-shared marginal while the other remains KEEP, then freeze it. Optimize the second distribution conditionally on the input context and actual emitted-first global code. The second stage is an LP. The first stage is not reoptimized after stage two. This evaluates a specified freeze-first optimization procedure, not limits of all sequential factorizations.

KEEP and near1/near2/near3 are private executable templates; actual original global tuples are emitted. Padding is forbidden. Where one shared group has different validity sets, only the intersection is feasible; KEEP always exists. This restriction and its counts are disclosed. Action aliases are aggregated by the common evaluation law, never exposed as action positions. Unsupported primary contexts form an exact-KEEP group, not an accidental context merge.

## Fixed optimization ceilings

Local and product each use exactly two declared starts, KEEP and uniform valid templates, at most four alternating rounds and the fixed recipient update order 1 then 2. One round has two coordinate solves. Total cap is 600 seconds including all starts, matrix construction and solves. A failed frozen-coordinate subproblem stops that start; it does not certify the whole bilinear family infeasible. Every subproblem and stop is retained. The minimum fitted objective among feasible completed coordinates is selected, ties within 1e-12 retain the earlier coordinate. There is no new restart after outcomes.

Deterministic cap is 600 seconds total: at most 150 for the continuous relaxation and 450 for the binary solve, relative MIP gap 1e-6. A feasible incumbent at a time limit is eligible subject to the common utility gate, with remaining MIP gap and unresolved optimality disclosed. A solver status alone is insufficient. Independently recompute primal inequalities/equalities, normalization, integrality, support and objective; preserve program and primal values privately.

Sequential cap is 600 seconds per order, split equally between first and second stages. Second-stage grouping is fixed before protected optimization from potential same-class source support: child key (input context, emitted-first global code); child counts use uniform valid first-template mass, a label-free count. Retain children with count at least 64, largest count first then lexicographic key, while reserving one fallback parent per context and enforcing at most 512 total groups (2048 second weights). Other children fall back to their input-context parent. This registered restriction has a different information/capacity ceiling from a joint context policy and is reported. An infeasible KEEP-other first stage does not certify every sequential procedure infeasible.

Primal tolerance 1e-8, exact simplex equalities; same source numerical safety policy and identical frozen reader constraints. One numerical thread. Sparse map matrices are O(rows×16) nonzeros; reserve 1.5 GiB per control worker. No dense all-state kernel allocation. Root may reduce breadth before science lock, but never silently remove an eligible mandatory comparator after outcomes.

## Unit accounting

Conditional on one narrow nominee, two seeds require ten new control policy units: local, product, deterministic, sequential12, sequential21. Upper bound on actual solver calls is 74 LPs and two binary MIPs: local/product four policy units ×16 LP coordinates =64; deterministic two relaxations; sequential four policy units ×two LP stages =8. Earlier terminations remain explicit skipped/failed subsolves. Utility-only and optimized scalar comparisons reuse only exact registered primary aliases and are charged to their original units. Fresh attacks and utility scoring are distinct R5/R3 units.

Every call must run inside root's durable queue/semaphore. This file is a proposal until root seals the campaign lock. No Adult fit or forbidden role read has run from R4.

## Data-free correctness coverage

Six tests cover product factorization with independent coins, correct recipient coordinate ordering, first-stage immutability, emitted-first-code alias grouping, categorical one-hot normalization and shared support validity/rejected malformed factors. No test demands an impossible privacy gain. LP/MIP fixture tests are gated numerical engineering work and will be queued under the locked synthetic law specification.
