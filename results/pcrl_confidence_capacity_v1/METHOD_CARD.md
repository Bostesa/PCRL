# Method card: confidence capacity and privacy (`qpc/kmeans.py`, `stagea.py`, `partition.py`, `compress.py`, `release.py`, `deploy.py`)

Owner: role B (compressor implementation and efficient search). Tests: `qpc/tests/test_method.py` (synthetic data
only). Every rule below is fixed in code before any real-data fit. Nothing here reads Adult rows, task labels or
SEX; the lead runs real data after the governing lock is pushed.

Provenance. The arithmetic is imported unchanged from the pinned dpc package (source SHA
`0a7b05a52746544213742f50efd0a48167efffb1`, `dpc/partition.py`, `dpc/compress.py`, `dpc/release.py`,
`dpc/deploy.py`, never edited): input validation, smoothing, masked KL, sufficient statistics, deployment
(`assign_fine`), token tables, canonical token IDs, the content fingerprint, the plug-in MI identities and the teacher
forward application. qpc adds the new k-means starts and convergence rule, asymmetric caps, the corrected sequential
baseline, objective-improving extra merges and its own record kinds. Old dpc outputs and new qpc outputs are
distinguishable by record kind (`dpc.*` vs `qpc.*`); `qpc` loaders and `qpc.deploy` refuse `dpc.*` records.

Recipients: 1 = income (K = 2), 2 = occupation_group (K = 6). float64 throughout, natural logs.

## 1. Inputs, smoothing, KL, statistics (unchanged from dpc)

- `P` (n, K): 2-D float, finite, nonnegative, rows summing to 1 within 1e-9 (`dpc.partition.check_probs`; `-0.0` is
  mapped to `+0.0`). The decision array must equal `argmax(P, 1)` (numpy first-index ties) exactly
  (`check_decisions`); any other label array (true task label, SEX, loss bin) is refused with `ValueError`. No qpc
  fitting function has a label argument.
- `smooth(mean, c) = (mean + eps*1 + eps*e_c) / (1 + (K+1) eps)`, `eps = 1e-12`; checked finite, nonnegative,
  `|sum - 1| <= 1e-12`, strict argmax c.
- `KL(p || q) = sum_{k: p_k > 0} p_k (log p_k - log q_k)`, increasing-k summation (`dpc.partition.kl_matrix`), q
  smoothed. Used for every assignment, every deployment and every objective.
- Cell statistics: `n` (int64), `S` (float64 `bincount` sums in row order), `A = sum over members of
  sum_k p_k log p_k`. `sum_rows KL(p || smooth(S/n, c)) = A - S . log smooth(S/n, c)` exactly.

## 2. Release interface (`qpc/release.py`)

- A `Policy` = assignment partition `fine` (routing centroids, class per cell, statistics) + `cell_token` map.
  Stage A (DIRECT-TASK) policies use the identity map on a direct k-means partition; Stage B policies map fine cells
  to coarse tokens. Tokens never mix predicted classes (refused).
- Token IDs are canonical: first appearance scanning cells in increasing index, i.e. (class, lowest member cell)
  order; non-canonical IDs are refused at construction. A class without fitting rows has exactly one fallback token.
- Decoded prototype of token t = `smooth(S_t / n_t, class_t)`, `S_t, n_t` accumulated over member cells in increasing
  cell index; a fallback token decodes to `smooth(uniform, class)`. Decision of a token = its class.
- Deployment of a row (p, d): the cells of class d; nearest by KL to the stored smoothed routing centroid, ties to
  the lowest index; a class with only a fallback cell maps to it; then the token, its decoded vector, its class.
  Nothing is updated at deployment.
- `encode(policy, P)` returns `(tokens int64, decoded float64 (n, K), decisions int64)` and raises unless, for every
  row: decision == argmax p (first-index ties), `|sum q - 1| <= 1e-12`, `q[d] > q[k]` for every k != d, and for every
  emitted token the decoded vector is bitwise equal to the registered smoothing formula recomputed from its
  statistics.
- Release arrays per mapping-pair unit (dpc format, ALL rows): `row_id`, `tok1`, `q1`, `hard1`, `alpha1`, `tok2`,
  `q2`, `hard2`, `alpha2` (`alpha_i` = full alphabet incl. fallback tokens). The full token identity is released:
  two tokens with identical decoded vectors are distinct disclosures (tested).
- Pair config: `family`, `m1`, `m2` (per-recipient caps), `lam` (None for task-only), `teacher`, `seed`, `config`
  (configuration ID), `teacher_model_sha256`, `feature_names_sha256`. Both bindings are required for a unit
  (`check_bound`) and copied into each recipient policy's meta. A pair is refused if a recipient has more tokens in
  any predicted class than its cap.
- Fingerprint: dpc's content hash (SHA-256 of K, assignment classes, routing centroids, map, token classes, decoded
  prototypes; little-endian). Metadata does not enter, so identical maps alias. A qpc reproduction of a dpc policy
  has the same fingerprint as the dpc policy.
- Serialisation: JSON (`qpc.PolicyPair` / `qpc.Policy` / `qpc.FinePartition`, Python float repr = exact float64 round
  trip). Loading recomputes prototypes from stored sums and refuses any prototype, centroid or fingerprint mismatch
  and any non-qpc kind. Save/restore gives identical assignments, decoded vectors and decisions (tested).
- `token_parity(tokA, qA, tokB, qB)`: exact token IDs, token bijection (canonical relabelling) and decoded-vector
  equality (bitwise, and max-abs difference; parity requires a bijection and max-abs <= 1e-15).

## 3. KL/Bregman k-means within one predicted class (`qpc/kmeans.py`)

Rows: the fitting (OSF_DEFENSE_FIT) rows of one teacher-predicted class c. `k = min(m, #distinct vectors, #rows)`
(distinct = exact float equality, `np.unique(axis=0)`). So `m = 64` creates up to 64 cells per class on the rows
themselves, not on an old fine partition.

**Starts** (fixed order, which is also the tie order):

1. `source`: dpc deterministic quantile initialisation. Distinct vectors in ascending lexicographic order, stably
   re-sorted by `p_c` descending; initial centroids = the order statistics at positions `floor((2j+1) U / (2k))`,
   j = 0..k-1 (U = #distinct). Identical to `dpc.partition._init_centroids` (tested).
2. `kpp:20261006` and 3. `kpp:20261007`: KL k-means++.
   - Index space: distinct vectors `U` (np.unique axis 0, ascending lexicographic) with multiplicities `w`
     (`return_counts`, float64).
   - Generator: `rng = np.random.default_rng([seed, K, c])` (SeedSequence of the three integers; PCG64); seed is
     20261006 or 20261007, K the recipient's class count, c the predicted class. The stream does not depend on m.
   - Exactly one `rng.random()` per centre. Draw from weights v: `cum = np.cumsum(v)`, `t = rng.random() * cum[-1]`,
     `i = np.searchsorted(cum, t, side="right")`; if `i == |U|`, i = the last index with v > 0. Zero-weight vectors
     can never be drawn.
   - Centre 1: `v = w` (multiplicity weighted = uniform over rows). Centre j > 1: `v_u = w_u * min over chosen
     centres x of KL(u || smooth(x, c))`, already-chosen distinct vectors set to exactly 0. If every unchosen weight
     is 0, `v = w` on the unchosen vectors (counted as `degenerate_draws`).
   - Initial centroid = the chosen distinct vector itself (unsmoothed), in selection order. Receipts: chosen
     distinct indices, their multiplicities, the generator expression, clip and degenerate counts.
   - **Roundoff guard (initialisation probabilities only):** KL values in [-1e-12, 0) are clipped to 0 and counted
     (`kl_clipped`); a value < -1e-12 raises (not roundoff). Assignments and objectives never clip: they use the raw
     masked KL argmin and the exact sufficient-statistic objective, so the guard can never change the optimised
     objective or an assignment.

**Iterations (rule `qpc`).** Assignment pass r: every row to `argmin_j KL(p || Q_j)` (ties to the lowest j). The pair
(Q, a) is a *coherent iterate*: deploying routing centroids Q reproduces a on these rows, and the decoded prototype
of cell j is `smooth(S_j/n_j, c)` from the statistics of a. Its objective is
`J(a) = A_c + sum_{j: n_j > 0} -S_j . log smooth(S_j / n_j, c)` (= the class total of KL(p || decoded prototype);
`A_c = sum over rows of sum_k p_k log p_k` computed once in row order). Stop rules, checked in this order after pass r:

1. assignment fixed point: r > 1 and `a_r == a_{r-1}` -> converged, `stop_reason = "assignment_fixed_point"`;
2. relative tolerance: `rel_r = |J_{r-1} - J_r| / max(|J_{r-1}|, 1e-300) < RTOL = 1e-9` on PATIENCE = 3 successive
   passes -> one final update and assignment pass (the final consistent assignment/update), then converged,
   `stop_reason = "relative_tolerance"`;
3. cap: after pass r = `rounds` (200) -> one final update and assignment pass; converged only if that final assignment
   equals pass r's (`"assignment_fixed_point"`), otherwise `stop_reason = "cap"`, `converged = False` (labelled
   non-convergence).

Otherwise update `centroid_j = S_j / n_j` (arithmetic mean = the KL Bregman centroid); an **empty cell keeps its
previous centroid** (dpc rule; the number of empty cells after every pass is recorded, `empty_cells_per_pass`,
`empty_cell_events`). Identical rules at every rate and start.

**Best coherent iterate.** The returned iterate is the one with the lowest J over every assignment pass (including the
final pass); later passes win exact ties, so a fixed point returns the routing centroids that equal the decoded
prototypes. Lloyd steps are monotone up to smoothing roundoff (tested: trajectory non-increasing within 1e-10
relative), so this is normally the last pass; the rule only guards against roundoff. Cells empty in the returned
assignment are removed (order kept); removing a centroid that no training row chose cannot change the deployment of
the training rows. The stored partition holds the returned routing centroids and the statistics `n, S, A, mean = S/n`
of exactly that assignment; `fit_recipient` re-deploys the stored partition on the fitting rows and asserts the
statistics are bitwise equal (training statistics = deployed assignment on training rows).

**Rule `dpc` (A1 reproduction only).** `dpc.partition.kmeans_class` semantics: the same passes and update, stop rule 1
only, at most `rounds` (20) updates followed by the final assignment, and the LAST iterate is returned. With the single
`source` start it reproduces `dpc.partition.fit_fine(P, d, K, max_cells=m, rounds=rounds)` bit for bit (fingerprint
and means; tested at (m, rounds) in {(4,5), (8,20), (16,20), (8,3)} on both recipients, and asserted at runtime in
every A1 unit).

**Start selection per class.** Lowest J (the DEFENSE_FIT KL distortion of the deployed cells) wins; a later start
replaces the incumbent only if `J_new < J_inc - 1e-12 * max(|J_inc|, 1)`, so ties go to the earlier start in the
order source, kpp:20261006, kpp:20261007. Per-class winners are assembled into the recipient's partition (classes are
independent: a class's cells only affect its own rows). No task label or SEX enters any start, iteration or selection.

**Absent class.** A predicted class with no fitting row gets one reserved fallback cell exactly as dpc: `n = 0`,
mean uniform 1/K, routing centroid `smooth(uniform, c)` (strict argmax c via eps). Real data: occupation class 5 is
never predicted on fitting rows.

**Receipts per class and start** (`fit_recipient(...).receipt["per_class"][c]["starts"]`): start, rule, rows,
distinct vectors, initial cells k, init detail (positions or kpp draw record), rounds cap, rounds used (assignment
passes before stopping, excluding the final pass), converged, stop reason, rtol, patience, objective trajectory (J per
assignment pass), rows changed per pass, empty cells per pass, empty-cell events, returned pass, objective J and mean
KL, final cell counts, empty cells in the returned assignment, removed empty cells, cell counts after removal, sparse
cells (n < 5), work (assignment passes, centroid updates, KL evaluations = rows x cells summed over passes, plus
k-means++ KL evaluations). Per recipient: winners, objective total, mean fitting KL, all/winner convergence,
fingerprints of the winner partition and of each start's partition, total work, wall and process CPU seconds.

## 4. Stage A runner functions (`qpc/stagea.py`)

T = teacher dict over ALL rows (`row_id`, `p1`, `p2`, `d1`, `d2`); `tr` = DEFENSE_FIT row indices; only `T[*][tr]`
enters a fit.

- `a1_unit(T, tr, src_release, meta)`: per recipient, `fit_recipient(P, d, K, 8, starts=("source",), rounds=20,
  rule="dpc")` (asserted bitwise equal to `dpc.partition.fit_fine(..., max_cells=8, rounds=20)`), then
  `fit_recipient(P, d, K, 8, starts=("source",), rounds=200, rule="qpc")`. Both pairs (`U|DIRECT-TASK|i8o8` with
  `a1_variant` `src20` / `r200`) are encoded on ALL rows. Parity of `src20` against the admitted dpc
  `pol__s{k}__U_DIRECT-TASK_m8/release.npz`: `row_id` equal; per recipient `token_parity` (reported `ids_rule`
  `exact_ids` or `bijection`, `q_rule` `bitwise` or `max_abs_diff<=1e-15`), `hard` equal, `alpha` equal. A failed
  parity sets `parity_with_admitted_release.ok = False` and `ENGINEERING_BLOCKER` in the record (an engineering
  blocker, not evidence). The record also holds per-recipient/per-class objective, rounds, convergence, work and the
  fixed-rate fitting-distortion contrast `D_i(r200) - D_i(src20)` (teacher KL, no labels). Files:
  `release_src20.npz`, `release_r200.npz`, `policy_src20.json`, `policy_r200.json`.
  The A1 r200 refit is the same computation as the `source` start of the A2 m = 8 fit (identical partition
  fingerprint; tested), so it is an A2 restart receipt, not an extra nominal mapping-pair fit.
- `recipient_fit(P_fit, d_fit, K, m, recipient)`: three starts, at most 200 rounds, per-class winner; returns the
  `qpc.Policy` dict (identity map; `meta.m = m`) and the receipts. Cached by the runner per (seed, recipient, m):
  income m1 in {4, 8}, occupation m2 in {8, 16, 32, 64}.
- `pair_unit(pol1_dict, pol2_dict, T, meta, tr=None)`: assembles `U|DIRECT-TASK|i{m1}o{m2}` (m from the cached
  policies; a conflicting `meta.config` is refused), binds the metadata, encodes ALL rows (raises on any class
  preservation failure) and records fingerprint, `alpha1`, `alpha2`, effective states emitted on fitting rows per
  recipient and in total, token populations per class, emitted entropy and singleton tokens on fitting rows (when
  `tr` is passed; fitting-row token counts are asserted equal to the stored statistics), and the fitting distortion
  `D_i`. Files: `policy.json`, `release.npz`.
- `config_id(teacher, family, m1, m2, lam)`: `U|DIRECT-TASK|i{m1}o{m2}`, `U|FINE-TASK|i{m1}o{m2}`,
  `U|{LOCAL,SEQ-12,SEQ-21,JOINT}|i{m1}o{m2}|l{lam:g}`, `U|CLASS|i1o1`.

## 5. Deployment (`qpc/deploy.py`)

```
OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m qpc.deploy --unit <teacher unit dir> \
    --policy <policy.json> --X <input.npz> --schema <pinned schema npz|json> --out <release.npz> \
    [--schema-sha256 <hex>] [--seed k]
```

- Input npz: exactly the arrays `X` (n, 83) finite float and `feature_names` (83,), equal to the pinned schema in the
  same order; missing, extra, renamed or reordered columns and any extra array are refused (dpc checks, imported).
- Teacher: the hash-complete unit's `jcv.train.Model(83, [2, 6], seed)` + deployed heads; float32 input, float64
  encoder output, `p_i = head_i.predict_proba`, `d_i = argmax p_i` (dpc/osf convention, imported).
- Policy: only `qpc.PolicyPair` records (a dpc record is refused); integrity checks on load. Both bindings are
  mandatory and must match: `teacher_model_sha256` = sha256 of the unit's `model.pt`, `feature_names_sha256` = sha256
  of the newline-joined pinned names. There is no unbound override.
- Output: ONLY `tokens_1, probs_1, decision_1, tokens_2, probs_2, decision_2` (`write_release` refuses any other key).
  Every flag outside `--unit --policy --X --schema --schema-sha256 --out --seed --help` is refused, with an explicit
  message for export or bypass attempts (fine/cell IDs, raw/teacher scores, logits, features, latent, distances,
  debug, export, allow/unbound, continuous). Exit code 2 on every refusal.
- Deployment equals the stored release arrays on the same rows, also from a restored copy of the policy (tested).

## 6. Stage B (`qpc/partition.py`, `qpc/compress.py`)

To follow with the Stage B deliverable (06:15Z target).

## 7. Synthetic timing (Stage A)

Real-shaped synthetic teacher outputs (15,434 fitting rows of 45,000; income K = 2 with 195 exact 0/1 rows and class
imbalance; occupation K = 6 with class 5 never predicted), one thread, under the shared semaphore, three seeds:

| Item | Process CPU (s) |
|---|---|
| A1 unit (src20 + parity + r200 + both releases) | 0.47-0.55 |
| recipient_fit income m1 = 4 / 8 | 0.13-0.15 / 0.22-0.25 |
| recipient_fit occupation m2 = 8 / 16 / 32 / 64 | 0.39-0.55 / 0.62-0.73 / 1.29-1.47 / 2.03-2.24 |
| 8 pair units (assembly + encode 45,000 rows) | 0.30-0.32 |
| Stage A fitting, one seed | about 6 |

Every synthetic winner converged by assignment fixed point (max 45-100 rounds used); the A1 src20 reproduction had
exact token IDs and bitwise decoded vectors against a dpc-produced DIRECT-TASK m8 release on all three seeds.
Stage A fitting cost is negligible (about 20 CPU-seconds for three seeds).
