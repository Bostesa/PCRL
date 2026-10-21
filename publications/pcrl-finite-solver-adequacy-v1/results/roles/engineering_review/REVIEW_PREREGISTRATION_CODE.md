# Engineering and claims review of the campaign code before the first real-data unit

Campaign `pcrl_finite_solver_adequacy_v1`, role engineering_review.
- Review started 02:43:22Z (`date -u`). Findings checked against HEAD at 03:05:15Z; this file was written after that.
- No label, no real-data array and no private model file was read. Every numerical check below used invented data.
- No code file was edited.

**Versions.**
- `fsa/fa/solver.py` was reviewed at `91ba9a27e`, the revision with the current-book line (card FA_FSA_v2). It is unchanged at HEAD
  `9d62b4c3e`. I first read it at `c20dde9ca`, which had the refreshed line only. Every solver result below is from `91ba9a27e`.
- `fsa/fa/solve.py` was reviewed at `91ba9a27e` and again at HEAD. The HEAD blob `2ef3368f2` (commit `6e8594bf0`) is the
  working-tree edit I read at 03:01Z. It fixes B1 and adds fields to the replay, audit and source stages. Line numbers are HEAD
  line numbers unless marked `@91ba9a27e`.
- Registration (`f301965a7`, receipt verified 03:03:08Z) happened before this review was finished. A SHOULD FIX below that changes
  stage behaviour therefore needs the campaign's dated amendment route.
- `fsa/common.py`, `fsa/jobqueue.py`, `fsa/devreaders.py`, `fsa/admit.py`, `fsa/receipt.py`, and a grep of every other `fsa/*.py`
  for constants left over from the previous campaign. All are unchanged between `91ba9a27e` and HEAD, except `fsa/fa/route.py`
  (card names restored) and `fsa/evaluate.py` (FRC: comparator added).
- **Not reviewed:** `fsa/rules.py`, `fsa/test_rules.py` and `fsa/fa/advance.py`. The latter is modified again in the working tree at
  03:04Z.

## Summary

| Class | Count | Items |
|---|---|---|
| BLOCKER | 1, resolved | B1: stage `source` crashed on the new trace format at `91ba9a27e`. Fixed in `6e8594bf0`, and the fix is verified on invented traces. No blocker is open at HEAD. |
| SHOULD FIX | 9 | S1 ulp-narrow intervals and duplicated people; S2 write-once checks happen after the computation; S3 source private outputs are not write-once and not hashed; S4 the nested-vs-replay equality is optional and partial; S5 the cached-statistics stages run outside the queue; S6 the selftest does not cover what the protocol says it checks; S7 the closeout process filter; S8 the `frc:` prefix means this campaign; S9 the devreaders member-type docstring |
| NOTE | 14 | N1-N14 below |

The replay stage reads labels and is presumably the 03:40Z unit. It has no blocker: its call of `frc.fa.route.nested_c` matches the
inputs of `frc.fa.route.stage_paths` exactly (section 2). Before it runs I would fix S2 (check the npz before computing) and add the
row hash (N7).

## 1. What I ran (invented data only; scripts in my scratchpad, not committed)

| Check | Result |
|---|---|
| `python3 -m fsa.fa.solver` (selftest), at `c20dde9ca` and again at `91ba9a27e` | passed 9/9 both times. The SV1 objective values were identical in both runs, so the selftest never exercises the current line (S6). |
| cost_terms / envelope / line_search against `nrm.fa.prc.route` at **every** interval midpoint. 12 configurations: n = 900 / 3000 / 4500 (past the 2048 chunk), PRC and HQ (restricted embedding), strengths 2 and 8, refreshed books from SV1 chains | 4,039 intervals; **0** midpoint partition mismatches; \|incremental - exact objective\| <= 7.1e-15 |
| Brute force of the same lines: an 801-point tilt grid plus the 7 original tilts | the best candidate is <= the best grid value in all 12 (equal to 1e-15 when the grid hits the best interval) |
| Capacity 16 x 64, n = 5000, strength 8 (150 sampled intervals of 8,256) | 0 mismatches; incremental minus exact 8.9e-15. The line beats a 201-point grid (2.52940 against 2.53252). Line search 0.26 CPU-s. |
| Memory at the MODEL_FIT pool size: n = 20,000, 16 x 64, 3 SV1 steps (both lines) | peak RSS 875 MiB; 1.1-2.0 CPU-s per step; 34,400 and 67,401 intervals per step |
| Constant propensity; duplicated codeword with identical tables (identical lines) | constant π: 0 mismatches. Identical cells: the tie-break agrees with route (lowest index). |
| **Duplicated people** (identical qI, qO, π) | 28-32 of 339-357 intervals are 1e-17 to 1e-15 wide. At their midpoints, route differs from the envelope partition by one person. Incremental minus exact is up to 1.07e-3 (S1). |
| SV1 / SV1_first chains on duplicated people (3 seeds x PRC/HQ) | SV1_first **accepted** 1-3 intervals narrower than 1e-12 per PRC chain, with \|exact - incremental\| up to 3.2e-4 at acceptance. SV1 accepted none in these seeds. |
| Current-book line (3 seeds x PRC/HQ x SV1/SV1_first, full grid) | the current state's interval on the current line has jv - J = -8.9e-16, so it is present. Current-line acceptances: SV1 0-3 per chain, SV1_first 4-22 per chain. |
| SV1 final objective minus the original, 36 strength cells | lower in most cells (down to -0.0095); **higher** in 2 (+1.9e-5 for seed 12, PRC, λ=0.5; +5e-6 for seed 13, PRC, λ=0.25) |
| `fsa.fa.solve.nested_s` (original entries, with the SV1 entries in the same plan) against `frc.fa.route.nested_c` | invented population n = 900, capacity 3 x 4, T_fixed 8, 5 outer folds: 117 per-person arrays, max \|diff\| = **0.0** |
| The stage_source effort sum `@91ba9a27e`:327 on SV1 and SV1_first traces | **TypeError: unsupported operand type(s) for +: 'int' and 'dict'** (B1) |
| The same sums at HEAD (`solve.py:360-363`) | SV1 310 events, stop "no improving proposal" after 2 steps. SV1_first 1,162 events, stop "path_steps reached" after 8 steps (cap 8 in this invented run; see N1). original_traced 0 events. |

Measured CPU of these checks is about 41 CPU-s. I charged an upper estimate of 120 CPU-s
(`python3 -m fsa.jobqueue charge --label engineering_reviewer`).

## 2. Verified as correct (with the evidence)

- **cost_terms equals route.**
  - `fsa/fa/solver.py:43-59` builds A = AI + AO and D = w_loc(πa_I+b_I) + w_loc(πa_O+b_O) + κ·guard + w_pair(πa_P+b_P). These are
    `nrm/fa/prc.py:318-350` with the tilt factored out, on `restrict_embedding` for HQ, with the guard on the restricted R as in route.
  - In floating point, route adds tilt·w·X term by term, a different association, so the two agree only to rounding. This matters
    only at ulp-narrow intervals (S1).
- **envelope** (`solver.py:62-105`).
  - Each event strictly lowers the current slope D, so there are at most C-1 events per person. `max_iter` = 10000 >= 8191, so every
    capacity up to 32 x 256 fits.
  - The clamp at t_cur and the "smallest slope among near-simultaneous crossings" rule are right.
  - Events are processed per chunk and stable-sorted by (t, person). Same-t events are applied before the next interval
    (`line_search`, `solver.py:192-209`).
- **IncrementalObjective equals objective.** It uses the same plug-in clipping (1e-9) and the same base rate (mean π over all
  people), and counts only occupied cells. The guard weights (1-π)/p0 and π/p1 come from the book. The guard applies only when λ > 0
  and κ > 0. For HQ it uses the person-level E['pi'], as `objective` does (which never restricts).
- **Acceptance** (`solver.py:256-266`). Every accepted state is the book with the midpoint tilt and `P.route` of it. Its value is
  `P.objective` of that partition, with a strict 1e-12 decrease. Steps are therefore monotone and rule-consistent, within one process.
- **Ordering** (`solver.py:242-245`).
  - 'best' sorts by (incremental value, tilt, refreshed line first). This matches card FA_FSA_v2: "ties: smallest tilt, refreshed
    line first".
  - 'first' scans the refreshed line from the largest tilt down, then the current line from the largest tilt down.
- **Current-book tables** (the coordinator's question).
  - The current line is used only when the book holds rI, rO and rP (`solver.py:235-236`).
  - `_protected_start` copies the twin, and twin books have no tables (`twin_book` runs lloyd at λ = 0, so `update_book` sets none).
    The current line is therefore skipped on the first step of a chain and after every twin reset.
  - Every accepted book either comes from `update_book` with λ > 0 or is a current book that already had tables. When the line is
    used, its tables belong to the partition the book was refreshed on. That is the deployed rule's own state, not a stale copy.
  - Not checked: whether the saved TWB_C64 book (a private file) carries table keys. If it does, the first step searches a current
    line built on those tables. This is harmless, because acceptance is exact.
- **chain() equals protected_path.**
  - It uses the same `_prepare` (SHUF donor copy), the same grid continuation `{g < max} ∪ targets`, the same twin reset and the same
    path_steps.
  - Evidence: the selftest (original_traced = protected_path), `nested_s` = `nested_c` above, and a side-by-side reading of `_chain`
    (`nrm/fa/prc.py:483-496`).
- **Termination** is bounded by path_steps, and each accepted step lowers J by more than 1e-12.
- **Replay inputs equal the P1 inputs.**
  - P1 ran `stage_paths(v, data, 'B1', 'C64', stats=...)` (`frc/fa/route.py:1594`). So hp was None, and outer, budgets, ladder and
    capacity took their defaults.
  - The replay passes `FR._plan(capacity_id, recorded ladder)` (the same names and order), B1, (16, 64), 5 folds, the recorded
    T_fixed, hp None (P.HGB), and the same budgets per family to `select_path`, `brackets_for` and `nested_lookup`.
  - `fit_teachers` casts `int(T_fixed[nm])`, so the JSON round trip is exact.
- **The per-person file is enough for recomputation.** For every plan entry the replay saves inc/occ NLL, Brier, accuracy and the
  look_pair, look_income and look_occupation scores, plus unit and SEX. The verifier can therefore recompute:
  - `spend` and `cluster_se`;
  - `select_path` (the candidate order comes from the names and the recorded ladder);
  - `nested_lookup` (B = 400, seed [2026102100, v, 4], fixed in code).
- **Pins.**
  - `frc_model`, `frc_sexside` and `frc_record` check against `SOURCE_ADMISSION.json['frc_private_artefacts']`. The keys exist for
    TWB_C64, PRC_S2 and HQ_S2 (V0 and V1), SEXSIDE V0/V1, SELECTION V0/V1 and QUAL_DECISION.
  - The record pins equal the current files: SELECTION_V0 e6863555…, SELECTION_V1 77787598…, QUAL_DECISION 2b67fd49….
  - `statistics()` reads TASK.pkl through `fsa.fa.route.pinned('task')`, against `source_private_artefacts['fa_cache/V<v>/TASK.pkl']`.
  - The replay's `estimator_identity` and the `fsa/rules.py` pins equal the files: frc/fa/route.py 933d0d6f…, nrm/fa/prc.py
    1192f34d…, and the FRC1 RULES.json a0dc0fc7….
- **Label access.** replay and nested read labels only through `C.load('MODEL_FIT', …, operation='fit', card=CARD)`. That path
  requires the registration receipt, the queue unit, the `fsa.fa` entry point and the committed `FA_` card, and it is logged.
- **The audit's strength 8.** The hard-coded strength 8 (`solve.py:280`) is right for the recorded selections: HQ_S2 = HQ@8.0 and
  PRC_S2 = PRC@8.0 in V0 and V1.
- **Constants.**
  - `fsa/common.py`:
    - CAMPAIGN, and CAMPAIGN_START_UTC 2026-10-21T02:22:58+00:00 (as in the agent context);
    - CONFIG_LIMIT 32;
    - the FRC paths (`.private/pcrl_finite_routing_comparison_v1/{runs,fa_cache}` and results);
    - REGISTRATION_FILES (8, including RULES.md and RULES.json);
    - POLICY: fit is MODEL_FIT only and needs a card; MODEL_SELECT is used only for benchmark and readers, never inside a fitting
      entry point;
    - ENTRY_POINTS.
  - `fsa/jobqueue.py`: the same start time, 12 CPU-h, an 8 h window plus 30 min of closeout, 2 slots, 8 GiB, one thread.
  - `fsa/devreaders.py`: SEED_BASE_OFFSET 800, BOOT_SEED 2026102200, FSA1 tags, and the 'frc' member type, which loads `frc.release`.
  - `fsa/custody.py`: BASE 278cb1ed6, and CODE and PRIVATE lists that include frc and nrm.
  - `fsa/receipt.py`: uses `C.REGISTRATION_FILES`, `C.PUBLICATION_BRANCH` and `C.REMOTE_URL`.

## 3. Findings

### BLOCKER

**B1 (resolved in `6e8594bf0`). Stage `source` crashed after its first SV1 chain** (`fsa/fa/solve.py@91ba9a27e:327`).
- Cause: `91ba9a27e` made the SV1 trace field `n_events` a dict per line (`solver.py:237-247`). The source stage still computed
  `sum(s.get('n_events', 0) for s in st)`, which raises `TypeError: unsupported operand type(s) for +: 'int' and 'dict'`.
  Reproduced on invented data for SV1 and SV1_first.
- Failure scenario: `FSA_SOURCE_*` runs the PRC original chain and the full PRC SV1 chain (minutes of CPU at n = 20,000), then dies.
  It writes no record, the CPU is charged, and a rerun needs a new unit id.
- Fix: HEAD (`solve.py:360-361`) sums the dict values. I checked it on invented SV1, SV1_first and original traces. Run no source
  unit from a commit older than `6e8594bf0`.

### SHOULD FIX

**S1. Ulp-narrow intervals from rounding-separated breakpoints** (`solver.py:192-209`; the envelope itself is right).
- Cause:
  - `line_search` merges breakpoints only when they are exactly equal (`np.unique`, then `ev_t[pos] == b`).
  - People with identical statistics get crossing times about 1 ulp apart. Such people are common on real data: identical six
    fields in the same cross-fit fold give identical out-of-fold posteriors. I believe the matrix product in `_xent` rounds identical
    rows differently depending on their position, but I did not isolate this.
  - The result is intervals 1e-17 to 1e-15 wide (28-32 of about 350 on invented duplicates). At their midpoints, `route` uses a
    different association and does not reproduce the envelope partition.
- Failure scenarios:
  - (a) An interval's incremental value is off by about one person's contribution (up to 1.07e-3 here). 'best' may try it first and
    accept a partition that is not the best of the line.
  - (b) If the error is upward, a truly improving interval can be ranked above the stop line, giving a false stall.
  - (c) An accepted tilt within 1e-16 of a breakpoint (seen for SV1_first, 1-3 per chain) gives a deployed rule that separates
    identical inputs only through rounding. The verifier's own code, with a different summation order (VERIFY_TOL 1e-9), or another
    row order can then route those people differently. The "rule-consistent" claim then holds only within one process.
- Fix:
  - merge breakpoints within a relative tolerance, for example \|Δt\| <= 1e-12·max(1, t), before forming intervals, or skip intervals
    narrower than that;
  - record how many were merged;
  - add a duplicated-people case to the selftest.
- This changes which proposals are tried, never the acceptance rule. Do it before the source and nested units, as a dated amendment,
  because registration has already happened.

**S2. Write-once checks run after the computation** (`solve.py:176-178`, replay npz; `solve.py:482-484`, nested npz).
- Failure scenario: a first attempt dies between `np.savez_compressed(npz)` and `_write_once(rec_path)`, for example on a
  serialization error or a kill at the cap. The next unit passes the JSON check at the start and recomputes the whole nested
  estimator: 170.9 CPU-s for P1 V0, plus the new solver chains for nested. It then refuses on the npz.
- Fix: check both paths before any computation, as the previous campaign did (frc engineering finding 3, `frc/fa/route.py:935`).

**S3. The source stage's private outputs are not write-once and not linked to the record** (`solve.py:384-389`).
- `np.savez_compressed(_priv('SOURCE_PARTITIONS_V<v>.npz'))` silently overwrites an existing file. The books pickle is opened with
  'xb', so it is write-once. Both files are written after the JSON, and neither hash is in the record.
- Failure scenario: a second source unit for the same version, for example after a crash between the writes, replaces the partitions
  that the first record describes, and nothing records the replacement.
- Fix: open both files with 'xb' before writing the record, and put their SHA-256 in the record.

**S4. The nested-vs-replay equality check is optional and partial** (`solve.py:491-503`).
- It runs only if `P1_REPLAY_V<v>.npz` already exists, and otherwise records None silently. It compares 3 of the 11 per-person keys:
  occ_nll, inc_nll and look_pair.
- Failure scenario: the nested unit runs before the replay, or for a version whose replay is missing. The record says None, so card
  FA_FSA_v2 §3 ("its original-solver entries must equal the replay's per-person outputs") is not tested, and nothing flags it.
- Fix: refuse to run without the replay file, or record an explicit `replay_missing` failure. Compare every per-person key of the
  original entries.
- The check is worth doing: on invented data, `nested_s` with original entries equals `nested_c` exactly.

**S5. The cached-statistics stages are not tied to a queue unit** (`solve.py:513-518`, `_features()`).
- `_features()` checks the registration and the card, but not `FSA_UNIT_ID`. The audit and source stages read label-derived caches
  (TASK.pkl and SEXSIDE.pkl, both derived from MODEL_FIT labels) and write records, but nothing records these reads in the access
  log.
- Failure scenario: someone runs `python3 -m fsa.fa.solve audit --version 0` from a shell. It works and writes the write-once
  record, with no ledger entry and no CPU charge. A later queued unit is then refused because the record exists.
- Fix: require `os.environ['FSA_UNIT_ID']` in `_features()`. Append a "cached statistics read" line to the access log, or to a
  separate log, so the exposure map can cite it.

**S6. The protocol claims more than the selftest checks.**
- PROTOCOL §6 says SV1 "is checked for objective arithmetic, finite termination, empty cells, tie handling and rule consistency
  (`fsa.fa.solver.selftest`)". The selftest (`solver.py:346-427`) has:
  - no empty-cell case;
  - no tie or duplicate-person case;
  - no case where the current line is used or accepted (its SV1 values did not change between `c20dde9ca` and `91ba9a27e`);
  - only 25 sampled intervals.
- The comment at `solver.py:403` says SV1 "is never worse than the original". The selftest does not assert this, and it is false in
  general: 2 of 36 invented strength cells are worse, by +1.9e-5 and +5e-6.
- Fix: add cases for duplicated people, an empty cell (a codeword nobody routes to) and a current-line acceptance, and check every
  interval. Delete the "never worse" sentence, or narrow the protocol's list of claims.

**S7. The closeout process filter matches the previous campaign** (`fsa/closeout.py:60`).
- `owned = re.compile(r'(-m frc\.|/fsa/|…verifier/code)')` should match `-m fsa\.`.
- Failure scenario: at closeout, a `python3 -m fsa.fa.solve nested …` or `python3 -m fsa.jobqueue run …` is still running. Its
  command line contains neither `-m frc.` nor `/fsa/`, so the resource accounting reports that no campaign process is running.

**S8. The `frc:` member prefix means this campaign** (`fsa/plan.py:63`, `fsa/claims.py:120`).
- `members = {m: {'eval': f'frc:FA/{c}'}}`, and `costs_between` strips `frc:` as the own-campaign key. At the same time,
  `fsa/evaluate.py:68` (as of `6e8594bf0`) uses `FRC:` for the source campaign's releases.
- Failure scenario: a plan author means the source PRC_S2 and writes `frc:FA/PRC_S2`. It resolves to this campaign's FA/PRC_S2,
  which does not exist (or does after a pilot), so the cost is missing or comes from the wrong release.
- Fix: use `fsa:` for this campaign's members, and map the source explicitly to `FRC:`.

**S9. The devreaders docstring gives the wrong member type** (`fsa/devreaders.py:36`, and lines 6-7).
- The docstring documents `{"type": "frc", …}` as "a release of this campaign". In the code (`devreaders.py:471-475`), 'fsa' is this
  campaign and 'frc' loads `frc.release`, the source campaign's saved releases.
- Failure scenario: a bank written from the docstring scores the source FA/PRC_S2 under this campaign's member name. Only the
  provenance key `frc_member` shows it.

### NOTE

- **N1. SV1_first uses more Lloyd steps than SV1.**
  - It keeps accepting the first improving interval of the current line, creeping along the same line. On invented data it took
    24-39 steps per 6-strength chain, against 16-24 for SV1. With the cap at 8 in one invented run, SV1_first reached the cap while
    SV1 stopped after 2 steps.
  - So it is an equal-effort control per step (same candidate set, same incremental evaluation), not per chain. The 30-step cap can
    bind for one variant and not the other. The HEAD fields `stopping_reason` and `accepted_steps` record this: report them with CPU
    per chain.
  - With 'best', the current line can help only on the first step after a strength change. Within one strength, the current book's
    line is the line searched in the previous step, so its best interval is the current state. The extra search costs CPU but is
    harmless.
- **N2. The card's candidate wording is too strong.** Card FA_FSA_v2 §2 and the solver docstring say "zero, the previous tilt and
  every boundary are candidates". In the code:
  - boundaries are never evaluated, only interval midpoints;
  - t = 0 is evaluated only through the first interval's midpoint (A-ties at 0 are broken by D, while route at tilt 0 breaks them by
    index);
  - the previous tilt is a candidate only when the current book has tables, so not at a chain start or after a twin reset.
  - "Exact for one proposal step" holds up to S1.
- **N3. PROTOCOL §4 item 4 does not match the audit.** The protocol promises "the task losses and per-SEX occupation losses of those
  emitted vectors". The label-free audit computes cross-entropy to the teacher posteriors and the π-weighted guard
  (`solve.py:122-147`). These are proxies. Label them as proxies in the protocol or the record, or add a labelled MODEL_FIT computation.
- **N4. The card's effort claim covers only the source stage.** Card §2 says "CPU time and route evaluations are recorded for every
  chain". `fit_pipeline_s` calls `S.chain` without a trace (`solve.py:412`), so the nested stage records no per-chain effort or
  stopping reason for either solver.
- **N5. `fit_pipeline_s` drops part of the plan entry.** It ignores the plan entry's override `p[3]` (it passes `{}`, `solve.py:411`)
  and always uses `P.HGB` (`solve.py:397`). Both are correct for `nested_plan()`, but an override in a plan would be dropped silently.
  `nested_s` also lacks `nested_c`'s household-overlap assertion (`frc/fa/route.py:431-432`). The folds are household folds, so this
  is defensive only.
- **N6. The audit hard-codes strength 8 and reuses the saved tables.**
  - It uses `path[8.0]` (`solve.py:280`) rather than the saved model's selected λ (the record field `selected`).
  - Its emitted-against-prototype comparison pairs `saved['dec']` with the recomputed (b8, k8, l8) (`solve.py:300`). That is valid
    only when `reproduces_saved_partition` is True. Gate it on that, or use `FR.tables('B1', b8, k8, l8, E)`.
- **N7. The records do not tie the per-person files to their rows.** Add the row-set hash (`row_sha`) and the outer-fold ids to the
  replay and nested records and npz files, so each per-person file is tied to the admitted rows. The replay npz also stores the income
  and occupation labels, which no recomputation needs. That is a new private copy of labels: list it in the exposure map, or drop it.
- **N8. The brackets equality is too strict** (`solve.py:212`). `br == sel['brackets']` compares floats such as the target total-NLL
  spend exactly. Compare the `brackets` sub-dict, and the floats against REPLAY_TOL, separately.
- **N9. `chain()` does not check grid membership** (`solver.py:318-342`). For non-original solvers it accepts strengths outside the
  grid, which `protected_path` would refuse. Add the same check.
- **N10. `cost_terms` is not chunked**, unlike `route`: A and D are full n x C float64 arrays plus temporaries. At 16 x 64 the
  measured peak is fine (875 MiB at n = 20,000). At 24 x 128 it would be about 3x larger, so re-measure before any C128 use. The
  machine is known to corrupt numbers under memory pressure.
- **N11. Compare library versions before reading a replay mismatch.** Before reading a mismatch against REPLAY_TOL 1e-12, compare the
  interpreter and libraries (numpy, scikit-learn) with the P1 unit's CODE_AT_START. A bitwise replay assumes the same libraries and one
  thread.
- **N12. Leftover comments and docstrings.** None changes behaviour.
  - `fsa/fa/route.py:160`: `UNT_TAG` became `FSA_UNT_S{v}`. It is unused here, but it would not reproduce frc's UNT_BR. The card-name
    renames in that file were restored in `6e8594bf0`.
  - `fsa/devreaders.py:11` names `frc.common.load`; the code uses `fsa.common.load`.
  - `fsa/devreaders.py:710`: the selftest's list of earlier seed bases (300, 400, 500, 600) omits frc's 700, which
    `rules.FORBIDDEN_OFFSETS` includes. There is no actual collision (800+ against 700+0..19).
  - `fsa/deploy.py:108-135`: this campaign's runs are keyed `FRC::`; they should be `FSA::`.
  - `fsa/restore_check.py:10` says "relative imports resolve to frc"; they resolve to fsa.
  - `fsa/readercustody.py:5` names `frc.restore_check`.
  - `fsa/decide.py:2` says "FSA1 rules of `frc.rules`"; the code uses `fsa.rules`.
  - `fsa/refitcheck.py:5` says "frc.claims.stability … decided 00:31Z", the previous campaign's decision time.
  - `fsa/common.py:60` and `fsa/admit.py:14-16` call fd933a824 "the base of this branch"; the base is 278cb1ed6.
  - `fsa/custody.py:61` uses the temp prefix `frc_restore_`.
  - `fsa/jobqueue.py:5-7`: the docstring's package list omits frc and the exact_reference code, which CODE_PACKAGES includes.
- **N13. A stray lock file in the source campaign.** `results/pcrl_finite_routing_comparison_v1/roles/verifier/records/.checks.lock`
  is untracked and dated 2026-10-20 23:54Z, before this campaign started. The previous campaign's verifier left it, not this code.
  Leave it untouched, since the source is read-only, and do not commit it.
- **N14. Repeated routing in the source stage.** `rule_consistent` calls `FR.route_any(E, b)` twice per strength (`solve.py:366`).
  Compute it once.

## 4. Not checked

- The private frc and nrm artefacts' bytes against their pins. I did not hash or open MODEL.pkl, RELEASE.npz, SEXSIDE.pkl or
  TASK.pkl; the code does that at run time.
- Whether the saved TWB_C64 book carries table keys.
- Real-data runtime and memory of the replay. I relied on P1's recorded 170.9 CPU-s for V0.
- The audit's new expressivity block (`6e8594bf0`). I read it but did not execute it.
- `fsa/rules.py`, `fsa/test_rules.py` and `fsa/fa/advance.py`.
- The remote and publication comparisons in `fsa/admit.py`. They need network calls, so I read the logic and did not rerun them.
- The verifier's code and the exact-reference code.
