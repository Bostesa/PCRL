# Method card: decision-preserving joint compression (`dpc/partition.py`, `dpc/compress.py`, `dpc/release.py`, `dpc/deploy.py`)

Owner: method owner (role 2). Tests: `dpc/tests/test_method.py` (34 tests, synthetic data only). Every rule below is
fixed in code before any real-data fit. Nothing here reads Adult rows; the lead runs real data after the locks.

## 1. What is released

For recipient i (1 = income, K = 2; 2 = occupation_group, K = 6) a policy g_i maps the teacher's probability vector
p_i and its decision d_i = argmax p_i (numpy first-index tie rule) to

- a categorical token ID (integer; audited as a one-hot over the FULL alphabet including fallback tokens),
- the token's public decoded probability vector (prototype), and
- the decision = the token's class.

g_i reads only its own p_i and d_i. It never reads SEX, a true label, the other recipient's score or the role. The
deployed release writes nothing else: no fine-cell ID, teacher probability, logit, latent feature or assignment
distance (section 9).

## 2. Inputs and validation (`partition.check_probs`, `partition.check_decisions`)

- P must be 2-D floating, finite, nonnegative, with every row summing to 1 within `INPUT_SUM_TOL = 1e-9`. It is used
  in float64 (`-0.0` mapped to `+0.0`). Rows failing the check are refused, not encoded.
- The decision array must equal `argmax(P, axis=1)` exactly (first-index ties); `d = None` means "compute it".
  **Any other label array is refused** (`ValueError`). This is the gate that makes a partition keyed by the true task
  label, SEX, a loss bin or an assessment label impossible: no fitting function accepts a group/label argument, and a
  partition is a nearest-centroid partition of the teacher scores only.
- `S_fit` (fitting rows only) must be binary {0, 1} and aligned with the fitting rows.

## 3. Smoothing, KL and sufficient statistics

- `smooth(mean, c) = (mean + eps*1 + eps*e_c) / (1 + (K+1)*eps)`, `eps = 1e-12`, float64, for every prototype and
  every assignment centroid of every method. Checked: finite, nonnegative, `|sum - 1| <= 1e-12`, and **strict**
  argmax c (`q_c > q_k` for every k != c). A mean whose argmax is not c raises.
- `KL(p || q) = sum_{k: p_k > 0} p_k (log p_k - log q_k)`, summed in increasing k (0 log 0 = 0); q is smoothed so
  every term is finite. The same formula is used for assignment and for D.
- Per cell: `n` (int64), `S = sum of member p` (float64 `bincount`, i.e. sequential in row order) and
  `A = sum over members of sum_k p_k log p_k` (0 log 0 = 0). Exactly: `sum_rows KL(p || q) = A - S . log q`.
- The unsmoothed-vs-smoothed prototype difference is recorded per policy (`max_abs_unsmoothed_vs_smoothed`, order
  1e-12). Smoothed prototypes are not bit-identical to the teacher barycentres.

## 4. Fine partitions (`partition.fit_fine`, `fit_fine_pair`)

Per teacher, seed and recipient, separately within each teacher-predicted class c, on OSF_DEFENSE_FIT rows only:

1. `n_cells = min(max_cells = 16, #distinct vectors, #rows)` (distinct = exact float equality, `np.unique(axis=0)`).
2. **Initialisation (no randomness):** distinct vectors in ascending lexicographic order are stably re-sorted by
   `p_c` DESCENDING (ties keep the lexicographic order); the initial centroids are the order statistics at positions
   `floor((2j+1) U / (2 n_cells))`, j = 0..n_cells-1 (midpoints of n_cells equal blocks; U = #distinct).
3. **Rounds:** round r assigns every row of the class to `argmin_j KL(p || smooth(centroid_j, c))`, ties to the
   lowest j (`np.argmin`). If r > 1 and the assignment equals round r-1's, stop: `converged = True`,
   `rounds_used = r` (counts assignment passes). Otherwise update `centroid_j = arithmetic mean of members` (the KL
   Bregman centroid); an **empty cell keeps its previous centroid**. At most 20 rounds; if the cap is reached a final
   assignment with the last centroids is made and `converged` records whether it equals round 20's.
4. **Stored state:** the smoothed centroids used in the last assignment (deployment centroids) and the statistics
   `n, S, A, mean = S/n` of the rows that this deployment rule maps to each cell. Fitting-row statistics therefore
   always equal deployment-on-fitting-rows (asserted). If not converged, the deployment centroid and `smooth(mean)`
   differ slightly (recorded as `max_abs_mean_vs_centroid`).
5. **Empty cells after the final assignment are removed** (order kept). Deployment can therefore never route a row
   to a non-fallback cell without fitting rows.
6. **Fallback:** a predicted class with zero fitting rows gets exactly one reserved fallback cell, `n = 0`, mean =
   uniform 1/K, centroid `smooth(uniform, c)` (argmax c via eps). Real data: occupation class 5 is never predicted on
   fitting rows and uses this cell.
7. Receipts: per class rows, distinct vectors, initial cells, init positions, rounds used, converged, rows changed per
   round, final cells, effective cells, removed empty cells, **sparse cells (n < 5)** and cell counts.

Fine cells are enumerated globally in (class, within-class index) order. Sensitive labels and true labels never
enter. On synthetic real-shaped data most classes use all 20 rounds without exact convergence (continuous scores);
this is recorded, not repaired.

**Deployment** (`assign_fine`): the row's predicted class selects that class's cells; nearest by KL to the stored
smoothed centroids, ties to the lowest index; a class with only a fallback cell maps to it. Selection and assessment
rows never update anything. `validate_fine_against_rows` refuses a partition whose stored statistics are not exactly
the deployment statistics of the given fitting rows (e.g. a planted label grouping); every `fit_policy_pair` call runs
it.

## 5. Policies, tokens, prototypes (`release.Policy`, `release.PolicyPair`)

- A policy = an assignment partition (`fine`) + `cell_token` map. Tokens never mix predicted classes (refused).
- **Token IDs are canonical:** numbered by first appearance scanning fine cells in increasing index, i.e. in
  (class, lowest member fine index) order, one fallback token per class without fitting rows. Non-canonical IDs are
  refused at construction.
- Decoded prototype per token = `smooth(S_t / n_t, class_t)` with `S_t, n_t` accumulated sequentially over member
  fine cells in increasing fine index; fallback token: `smooth(uniform, class)`. Decision per token = its class,
  checked to be the strict argmax of its prototype.
- `encode(policy, P, d)` returns `(tokens int64, probs float64 (n, K), decisions int64)` and raises if any released
  decision differs from `argmax(P)` or any prototype's argmax differs from the decision.
- Fingerprint (`Policy.fingerprint`): SHA-256 of K, assignment classes, assignment centroids, map, token classes and
  prototypes (little-endian bytes). `PolicyPair.fingerprint` hashes both. Family, metadata and fitting counts do not
  enter, so identical maps from different configurations give identical fingerprints (exact aliases).

**Structural statement 1 (pointwise class preservation).** For any valid input row, `assign_fine` chooses a cell of
class d = argmax p (fallback if the class had no fitting rows), the map keeps the class, and the token's prototype has
strict argmax d; hence the released decision equals the teacher decision for every row, including ties (first-index
rule), exact-zero underflow components and classes unseen in fitting. It is a property of the code, independent of the
data distribution; tests cover binary and 6-class ties at every pair of positions, one-hot rows, 1e-300 components,
uniform rows, the unseen class and serialization round trips.

**Structural statement 2 (post-processing).** With the public fitted state M fixed, C_i = g_i(p_i) is a deterministic
function of p_i, so `I(S; C_i | M) <= I(S; p_i | M)` and `I(S; C_1, C_2 | M) <= I(S; p_1, p_2 | M)` (standard data
processing; not a new theorem, not a secrecy or training-data privacy bound). Because d_i is a function of C_i,
`I(S; C_i) = I(S; d_i) + I(S; C_i | d_i)`: confidence compression cannot remove the information carried by the
decisions themselves.

## 6. Objectives (`compress`)

On the N fitting rows, natural logs:

- `D_i = (1/N) sum_c [A_c - S_c . log q_c]`, q_c the smoothed decoded prototype (n_c = 0 contributes 0). A
  teacher-distortion surrogate, not a bound on true-label log loss.
- `I_i = I(S; C_i)`, `I_12 = I(S; C_1, C_2)`: plug-in MI of the exact fitting contingency tables n(s, c_i) and
  n(s, c_1, c_2), 0 log 0 = 0, empty cells contribute 0, no smoothing of any table. Computed as
  `(1/N)[sum_cols phi(col) - sum_s n_s log n_s + N log N]`, `phi(col) = sum_s n_sc log n_sc - n_c log n_c`.
- The fine table n(s, f1, f2) (int64) is built once per unit from the deployed fine cells of the fitting rows; every
  coarse table is an exact sum of it.
- `F_task = D1 + D2`; `F_local = D1 + D2 + lam (I1 + I2)/2`; `F_joint = D1 + D2 + lam ((I1 + I2)/2 + I12)`.
  Task-only receipts report F_task only (lam = None); F_local/F_joint for any lam follow from the recorded terms.
- Every reported objective is recomputed from scratch from the final state AND independently from the released
  tokens/prototypes of the fitting rows (`evaluate_rows`: row-wise KL mean and direct plug-in MI); the receipt records
  the maximum difference (asserted <= 1e-9; observed ~1e-15).

## 7. Search engine

State: per recipient a label per fine cell (label = a member fine index, unique per coarse cell), per-label `n, S, h
= -S . log q`, per-label count tables, and the label-pair table.

**Greedy agglomeration.** Each step evaluates ALL eligible merges: same recipient, same predicted class, class
currently above the cap m, for the recipients being optimised. The increment is exact (from summed statistics and
summed count columns, including the full pair table when the I12 weight is nonzero). The smallest increment wins;
**candidates within `TIE_TOL = 1e-12` (objective units) of the smallest are tied, and the first in lexicographic order
(recipient, class, label a, label b) wins.** Labels are the lowest member fine index; the merged cell keeps the lower
label. Greedy stops when every class has <= m coarse cells; positive increments are allowed and recorded (each merge
records recipient, class, a, b, increment, dD, dI_own, dI12, number of tied candidates). Classes already at or below
the cap are never merged, so the solver's reachable set has **exactly min(m, #fine cells of the class)** coarse cells
per class; maps with fewer cells (e.g. class-only) are separate configurations.

**Refinement.** Sweeps over the recipients being optimised (recipient 1 then 2), fine cells in increasing index. For a
fine cell whose coarse cell has another member (never empty a cell), evaluate moving it to every other coarse cell of
its class with the exact objective change; take the best (ties to the lowest coarse label) and accept only if
`Delta F < -TOL`, `TOL = 1e-12`. Coarse statistics are re-accumulated from members (increasing fine index) after each
accepted change. At most `SWEEPS = 5` full sweeps; stop after a sweep with no accepted move. A JOINT sweep covers both
recipients, so each recipient receives at most five passes, the same allowance as LOCAL and SEQ stages. Coarse IDs are
fixed during refinement; final tokens are re-canonicalised. Work counts: greedy candidate evaluations and steps,
refinement candidate evaluations and accepted moves, sweeps and convergence per stage/start.

The procedure is greedy and smoothed; no global optimality is claimed. The exhaustive fixture in the tests (128 maps)
reports the bracket: there JOINT equals the optimum over maps with exactly min(m, F_c) cells (gap 0), while the
optimum over maps with <= m cells is lower (fewer cells) — a scope fact, not a solver certificate.

## 8. Families (`compress.fit_policy_pair`)

All deterministic, on the same fine partitions, cap m per predicted class:

| Family | Exact rule |
|---|---|
| CLASS-ONLY (m = 1) | All fine cells of a class -> one token (fallback classes keep their fallback token). |
| FINE-TASK | Greedy on D_1 (recipient 1) and D_2 (recipient 2), then refinement of each on its own D_i. lam = None. |
| DIRECT-TASK | `fit_fine(P_i, d_i, K_i, max_cells=m)` on the original teacher vectors: identical init/round/tie/empty/fallback rules with k = min(m, distinct, rows); identity map; deployment = nearest smoothed centroid within the predicted class; prototype = smoothed mean of deployed fitting members. No sensitive labels (S_fit only feeds the receipt objective). Not contained in the fine-state family. |
| LOCAL | Recipient 1: greedy + refinement on D_1 + lam I_1/2; recipient 2 likewise, independently. |
| SEQ-12 | Stage 1: recipient 1 alone, greedy + refinement on D_1 + 1.5 lam I_1 (= F_joint with C_2 constant, minus D_2; the identity is asserted numerically in every unit and recorded). Freeze recipient 1. Stage 2: recipient 2 greedy + refinement under the full F_joint given the frozen map. Recipient 1 is never revised (asserted). |
| SEQ-21 | Mirror: recipient 2 on D_2 + 1.5 lam I_2, freeze, recipient 1 under F_joint. |
| JOINT | (i) Greedy F_joint where each step may merge on either recipient, then joint refinement. (ii) Joint refinement from the FINE-TASK(m), LOCAL(m, lam), SEQ-12(m, lam) and SEQ-21(m, lam) solutions (passed in via `witnesses`, validated to share the fine partitions, family, m and lam; any missing slot is recomputed and recorded). (iii) Candidates = the five refined starts and the four unchanged witnesses; the lowest from-scratch F_joint wins by exact comparison, ties to the fixed order JOINT-GREEDY, FINE-TASK, LOCAL, SEQ-12, SEQ-21 (refined before unchanged). (iv) Asserted: final F_joint <= every witness's F_joint. All start objectives (initial, refined), moves, sweeps and the winner are recorded. |

For FINE-TASK and LOCAL (separable objectives) both recipients are first agglomerated, the `after_greedy` snapshot is
taken, then each is refined; this equals optimising each recipient alone. For SEQ the `after_greedy` snapshot is the
stage-2 greedy state (stage 1 already refined); each stage also records its own greedy/refined stage objective.

JOINT's dominance holds only on the fine-state family and the fitting objective. It does not imply better assessment
recovery, global optimality or containment of DIRECT-TASK. JOINT's CPU includes its five refinement starts but not
the witnesses' own fits when they are passed in (`starts[*].source` = passed_in / recomputed).

**Aliases.** `find_aliases({config_id: PolicyPair})` maps each configuration to the first (sorted) configuration with
an identical pair fingerprint, plus per-recipient alias maps. Example (test): LOCAL with lam = 0 aliases FINE-TASK.

## 9. Release views, serialization, deployment

- `release_views(policy1, policy2, P1, d1, P2, d2)`: per recipient `tokens`, `token_onehot` over the full alphabet
  (fallbacks included), `probs`, `decision`, `decision_onehot`, `alphabet`; and the aligned `pair` (`tokens` (n, 2),
  `pair_index = t1 * T2 + t2`, `pair_alphabet`, hstacked one-hots and probabilities). Two tokens with identical
  prototypes remain distinct columns (decoder collisions never merge IDs).
- Any invertible renumbering of token IDs leaves the one-hot view equal up to the column permutation
  (`renumbered_view`; tested).
- Serialization: JSON (`save_policy` / `load_policy`, Python float repr: exact float64 round trip) and npz
  (`save_policy_npz` / `load_policy_npz`, `allow_pickle=False`). Loading recomputes prototypes from the stored sums and
  refuses any prototype, centroid or fingerprint mismatch.
- `python -m dpc.deploy --unit U --policy pair.json --X input.npz --schema pinned.npz|.json --out release.npz
  [--seed k] [--schema-sha256 h] [--allow-unbound-policy]`:
  - the input npz must contain exactly `X` (n, 83) float and `feature_names` (83,); names must equal the pinned
    schema (from the caller or the source input's `feature_names`) in the same order; missing, extra, renamed or
    reordered columns and any extra array are refused; an optional schema hash is checked; a policy carrying
    `feature_names_sha256` must match;
  - the unit must be hash-complete (`jcv.finalize.unit_complete`); teacher = `jcv.train.Model(83, [2, 6], seed)` with
    `model.pt`, input cast to float32, encoder outputs cast to float64, `p_i = head_{i-1}.predict_proba(r_i)`,
    `d_i = argmax p_i` (the rgj/finalize.py and osf/deploy.py convention);
  - the policy must be bound to that teacher (`teacher_model_sha256` in the pair config or policy meta) unless
    `--allow-unbound-policy` is given; a mismatched binding is refused;
  - only `tokens_1, probs_1, decision_1, tokens_2, probs_2, decision_2` are written (`write_release` refuses any
    other key); every flag outside the fixed allow-list is refused, with an explicit message for export attempts
    (fine/cell IDs, raw/teacher scores, logits, features, latent/hidden/repr, distances, debug, export, ...).

## 10. API

```python
# dpc.partition
smooth(mean_p, c, eps=1e-12, check=True)
kl_matrix(P, Q); kl_rows(P, Q); cell_stats(P, cell, F) -> (n, S, A)
fit_fine(P, d, K, *, max_cells=16, rounds=20, tag="") -> FinePartition
assign_fine(P, d, fine) -> int64 global fine-cell ids
fit_fine_pair(P1, d1, P2, d2, *, max_cells=16, rounds=20, tag="") -> (fine1, fine2, receipts)
validate_fine_against_rows(fine, P, d) -> fitting fine ids (raises if not deployment-consistent)
to_dict(fine); from_dict(z)          # also FinePartition.to_dict / from_dict / fingerprint

# dpc.compress
fit_policy_pair(family, fine1, fine2, P1, d1, P2, d2, S_fit, m, lam, witnesses=None, meta=None) -> (PolicyPair, receipts)
    # family in CLASS-ONLY, FINE-TASK, DIRECT-TASK (lam None), LOCAL, SEQ-12, SEQ-21, JOINT (lam >= 0)
fit_class_only(fine1, fine2, P1, d1, P2, d2, S_fit, lam=None, meta=None) -> (PolicyPair, receipts)
fit_bank(fine1, fine2, P1, d1, P2, d2, S_fit, ms=(2, 4, 8), lams=(0.1, 1.0, 10.0), class_only=True, meta=None)
    -> {config_id: (PolicyPair, receipts)}   # JOINT receives the bank's own witnesses
config_id(family, m, lam); find_aliases(pairs) -> {"pair": {...}, "r1": {...}, "r2": {...}}
evaluate_rows(pair, P1, d1, P2, d2, S_fit, lam=None) -> {D1, D2, I1, I2, I12, F_*}   # brute force from releases
check_joint_dominance(joint_pair, witnesses, P1, d1, P2, d2, S_fit, lam) -> list of violations
mi_plugin(s, c); mi_from_table(T); mi_terms_from_labels(Tfine, lab1, lab2); fine_table(f1, f2, s, F1, F2)
State, greedy, refine, Weights, W_task, W_local, W_joint, W_seq_stage1   # engine (exposed for reviewer tests)

# dpc.release
Policy(recipient, fine, cell_token, family="", meta={}); PolicyPair(p1, p2, family="", config={})  # pair[0], pair[1]
encode(policy, P, d=None) -> (tokens, probs, decisions)
recipient_view(policy, P, d=None); release_views(policy1, policy2, P1, d1, P2, d2); renumbered_view(policy, perm, P)
canonical_tokens(fine, labels); alphabet_size(policy); fingerprint(obj)
save_policy / load_policy (JSON); save_policy_npz / load_policy_npz; policy_pair_to_dict / policy_pair_from_dict

# dpc.deploy
main(argv); release(unit_dir, pair, X, seed=None, allow_unbound=False) -> ({6 arrays}, info)
write_release(path, out); load_permitted_input(path, pinned); schema_names(path); schema_sha256(names)
load_teacher(unit_dir, seed=None) -> (model, heads, model_sha256); teacher_probs(model, heads, X) -> [P1, P2]
```

Receipts (JSON-safe, strict `allow_nan=False` tested): `family, m, lam, eps, tol, tie_tol, sweeps_cap`, `final`
(D1, D2, I1, I2, I12, F_task[, F_local, F_joint]), `after_greedy` (same terms; None for CLASS-ONLY/DIRECT-TASK),
`row_level_check`, `row_level_max_abs_diff`, `r1`/`r2` (tokens, tokens and effective states per class, fallback
tokens, unsmoothed-vs-smoothed difference, fingerprints), `pair_fingerprint`, `work`, `summary` (merges,
positive-increment merges, accepted moves, sweeps, convergence, JOINT starts and winner), `stages` (FINE-TASK, LOCAL,
SEQ: merge and move lists, sweeps, convergence, stage objectives, weights), SEQ `order`, `stage1_coefficient`,
`stage1_identity`; JOINT `starts`, `candidates`, `winner`, `witness_dominance`; DIRECT-TASK `direct_partitions`;
`wall_seconds`, `cpu_seconds` (process time).

## 11. Tests and synthetic timing

`OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q dpc/tests/test_method.py` — 34 tests, ~8 s:
smoothing rule; KL sufficient-statistic identity; input validation; init order statistics; fine-partition
determinism, deployment consistency, nearest-cell brute force, fallback, fewer-cells-than-cap; assignment ties;
refusal of true-label/SEX keys and of a planted label grouping; class preservation for all families on adversarial
rows (ties, one-hot, 1e-300 underflow, uniform, unseen class); exact JSON/npz round trips and tamper refusal;
canonical IDs and renumbering; decoder collisions; no cross-class tokens; runner shims; independent brute-force D/MI
for every family; merge and move deltas on both recipients against from-scratch recomputation with pair-table
marginal checks; SEQ stage-one identity, frozen first map and stage-2 weights; JOINT witness dominance, detection of a
weak "joint", witness validation; recomputed vs passed witnesses; caps, effective states, sweeps and strict-decrease
moves; JSON-safe receipts; task families invariant to permuting S; aliases; greedy ties and positive increments;
exhaustive-fixture bracket; coalition-positive XOR fixture (I1, I2 ~ 0, I12 > 0.6 at fine level; LOCAL keeps
I12 > 0.6, JOINT drives it below 0.05 with lower F_joint); null fixture; deploy end to end and refusals (extra,
missing, renamed, reordered, bundled arrays; export flags; extra output keys; unbound or mismatched policy); timing.

Synthetic real-shaped timing (15,434 rows; income K = 2 with 195 exact 0/1 rows; occupation K = 6 with class 5 never
predicted; fine cells 32 + 81; one thread; two runs, seeds 0 and 1):

| Item | Wall (s) |
|---|---|
| Fine partitions, one teacher/seed (both recipients) | 0.14-0.15 |
| Full bank, one teacher/seed: 1 CLASS + 6 task-only + 36 privacy = 43 units | 4.09-4.23 (CPU 4.09-4.23) |
| Mean per unit | 0.095-0.098 |
| JOINT unit (witnesses passed in), mean / max | 0.224-0.231 / 0.28 |
| LOCAL / SEQ-12 / SEQ-21 / FINE-TASK / DIRECT-TASK / CLASS mean | 0.053 / 0.068-0.072 / 0.057-0.060 / 0.053 / 0.10 / 0.017 |
| Projected 2 teachers x 3 seeds (fine + full bank) | about 26 |

JOINT without passed-in witnesses recomputes the four witnesses (+ about 0.25 s). The synthetic timing supports the
full bank (m in {2, 4, 8}, lam in {0.1, 1, 10}); the fitting cost is negligible next to the attack banks.

## 12. Design choices fixed by the method owner (beyond the study text)

1. Init sort key: `p_c` descending, ties by ascending lexicographic order of the full vector; positions
   `floor((2j+1) U / (2k))`.
2. `rounds_used` counts assignment passes; at most 20 centroid updates; a final assignment follows a non-converged
   20th update; stored statistics are those of the final (deployment) assignment.
3. Cells empty after the final assignment are deleted (order kept).
4. Assignment uses the full masked KL (not only the cross-entropy term), in increasing-k summation.
5. Input row-sum tolerance 1e-9 (refuse beyond); prototype check 1e-12.
6. Greedy tie tolerance 1e-12 in objective units with the lexicographic rule; coarse labels = lowest member fine
   index; merged cell keeps the lower label.
7. Refinement ties to the lowest coarse label; coarse IDs fixed during refinement; final tokens canonicalised.
8. FINE-TASK/LOCAL: agglomerate both recipients, snapshot, then refine each (equivalent for separable objectives).
9. JOINT winner: exact comparison of from-scratch F_joint; tie order JOINT-GREEDY, FINE-TASK, LOCAL, SEQ-12, SEQ-21,
   refined before unchanged; dominance asserted exactly.
10. DIRECT-TASK reuses `fit_fine` with `max_cells = m`; its deployment centroid is the last-round centroid (equals the
    decoded prototype when converged; difference recorded).
11. CLASS-ONLY is built on the fine partition (one token per class); it is not bitwise identical to DIRECT-TASK with
    m = 1 (different summation order), and only CLASS-ONLY is the registered m = 1 control.
12. Deployment refuses unbound policies unless explicitly overridden, and refuses all non-allow-listed flags.

## 13. Limits

The objectives are fitting-row training criteria; plug-in MI is optimistic on sparse cells and is not a population
or nonlinear privacy guarantee; independent attack AUC governs claims. Smoothing changes probabilities by about
1e-12. Greedy plus five refinement sweeps is a local search. Class preservation preserves the teacher's decisions,
including its errors and weak classes; it cannot improve recall or confidence quality, which are measured separately.
