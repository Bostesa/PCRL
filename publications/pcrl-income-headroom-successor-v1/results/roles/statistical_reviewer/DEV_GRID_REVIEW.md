# Review of the development reader-proxy grid (`ihs/evaluate.py`, version `IHS1-dev-proxy-v1`)

**Verdict: APPROVED for development selection.** I request two small engineering safeguards before the first new fit. They
change no number and need no re-run of the reference units (S1, S2). The other items are documentation and reporting notes.
Nothing in the slate, splits, seeds or definitions needs to change.

Context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (MODEL_SELECT, 8,001 people / 5,306
households). Readers estimate the dataset's recorded SEX column from the released probability vectors and argmax decisions
only.

## What was checked

| Item | Finding |
|---|---|
| Repetitions | A: the source campaign's household halves (`trees.seed_for('SCM1proxysplit')` permutation of household ids). B: new halves from the SHA-256 order of household ids (`household_folds(unit, 2, 'IHS1_dev_proxy_rep_B')`). Both are label-free, deterministic and household-separated. Each half is used once for training and once for scoring, and the cell is the mean of the two directions, so every person is scored once per repetition. Correct |
| Reader seeds | random_state = MODEL_SEED_BASE + 100 + k, k = 0, 1. I checked the constructors. LOG_MLP is an MLP with random_state (`fit_reader`) and ET25 an ExtraTrees with random_state (`ffv.readers.make_model`), so both depend on the seed. OUT_HGB is HGB with early_stopping=False (`fit_reader`), HGB31 has early_stopping=False and no subsampling at these sizes, and OUT_LR, CATI1 and LOOKUP are deterministic. So `SEED_DEPENDENT = ('LOG_MLP', 'ET25')` is right, and fitting the seed-invariant readers once per repetition loses nothing |
| Slate | OUT_LR, OUT_HGB, LOG_MLP (source constructors on the source `out` / `log` features); ET25, HGB31, CATI1 (validation recipes via `ffv.readers.fit_new`, FiniteScaler fitted on the training half; CATI design fitted label-free on the training half); LOOKUP (exact vectors). This covers every reader family of the final enhanced bank: linear on outputs, categorical interactions, boosting, extra-trees, MLP and lookup. The omitted final recipes (OUT_MLP, LOG_LR, DEC_LR, CONF_HGB, ET5, HGB15, MLP64, MLP6432, CATI01) are lower-capacity or same-family variants on features that the included readers already see. CONF_HGB uses only the maximum probability, a function of the probabilities that OUT_HGB and HGB31 already receive. Adequate for development |
| Training size | About 4,000 people (about 2,650 households) per fit, half the final readers' 8,001-person ATTACK_FIT. Flexible readers on continuous releases (NEURAL_RAW, H_LJ) are likely further from their asymptote than readers on coarse codes, which could understate pair benefits against those references. The direction is not guaranteed. Observed for the retired candidate: development grid +0.0244 / +0.0301 vs NEURAL_RAW and +0.0190 / +0.0183 vs H_LJ, against final ENHANCED +0.027 / +0.022 and +0.028 / +0.026. Acceptable for ranking; the final readers stay the standard |
| Pair maximum | `max_incl_single` = max(pair view max, income view max, occupation view max). It includes single-view alternatives, as the final bank does. Correct, and covered by the selftest (`planted_pair_max_includes_single`) |
| Benefit signs | `benefits()`: pair = reference - candidate (positive favours the candidate); single views = candidate - reference (non-worsening iff <= 0.010). Matches PROTOCOL.md section 4 and the nomination rule |
| Caching | Key = SHA-256 of version tag, view hash (view name + float64 bytes of the released arrays), reader, repetition, effective seed, SEX and unit hashes. Identical released views are fitted once. This is valid because readers see only the released vectors and their argmax. Selftest `cache_identical` passes |
| Selftest | `checks/EVALUATOR_SELFTEST.json`: all 6 checks PASS (clustered SE equals the direct formula, identical release gives zero, planted SEX reads high, independent outputs read about 0.5, cache identical, pair max includes single views) |

## Requested before the first new fit (no number changes; no reference re-run)

- **S1. Atomic cache writes.** `proxy_cell` writes with `p.write_text(...)`. With two queue slots, two units can score the same
  view at once: the inherited occupation view and the eligible references' views are shared by many configurations. A reader
  can then see a half-written file and fail on `json.loads`. Write to a temporary file in `CACHE` and `os.replace` it.
  Equivalently, treat a JSON decode error as a cache miss.
- **S2. Frozen fingerprint check, not a key change.** The cache key does not include the reader code or library versions, so a
  later edit of `ffv/readers.py`, `ffv/features.py`, `scm_bench/readers.py`, `task_conditioned_representation/attacks.py` or
  `ihs/evaluate.py` would silently reuse stale AUCs. Record the SHA-256 of those files and the numpy / scikit-learn / scipy
  versions once, in the registration, and have `proxies()` refuse to run if they differ. Do not add them to the key: that
  would invalidate the cached reference cells without need.

## Documentation and reporting notes (no code change required)

1. **Versions.** PROTOCOL.md section 2 calls the `python3` stack (numpy 2.4.4, scikit-learn 1.8.0, scipy 1.17.1; I confirmed
   3.14.3 / 2.4.4 / 1.8.0 / 1.17.1) "the versions of the validation's READER_BANK_LOCK". That lock records numpy **2.4.2**.
   Correct the sentence. Before FINALIST_LOCK, also state how the enhanced bank will be run (under 2.4.4, with the bitwise
   reproduction of stored validation scores as the check, or under the locked 2.4.2).
2. **What the four cells measure.** The cells share the same people, so their range is reader-fitting variability (split and
   initialization), not sampling uncertainty. No development benefit should be described with a confidence statement. The
   nomination rule therefore requires positivity in every cell and reports ranges, near ties and winner changes.
3. **Maximum over readers on the scoring half.** Each view's value is the maximum of 7 (pair: 21) AUCs on the half where they
   are scored. That is optimistic for every release alike. Reference and candidate face the same maximum, so benefits are
   roughly fair, but absolute proxy AUCs are not comparable with the final AV-selected, DEV-scored values.
4. **No orientation step.** A reader below 0.5 on the scoring half simply loses the maximum. That is harmless for a maximum
   and differs from the final bank's AV orientation. Report it if any view maximum is below 0.5.
5. **LOOKUP** reads about 0.5 on mostly unique continuous vectors and is near Bayes-optimal on coarse codes (for example a
   6-level income view). That asymmetry is intended, and it is the reason the pair maximum includes it.
6. **Fit counts.** `n_reader_fits_requested` (168 per release and seed) counts cache hits. With the seed-invariant readers
   fitted once per repetition, at most 108 fits per release and seed are distinct, and fewer when views repeat (for example
   the inherited occupation view). CPU accounting should use the queue's measured CPU, not this count.
7. **Repetition A reuses the source campaign's halves.** It was used for selection there. That is acceptable for development
   ranking and is one more reason MODEL_SELECT results are not fresh.

## Not requested (and why)

- **Adding final-bank recipes (CONF_HGB, DEC_LR, ET5, MLP6432).** They are cheap, but each extra reader adds to the
  optimistic maximum, and none adds a feature the slate cannot see. Changing the slate now would also require re-running the
  reference grid. The final enhanced bank is the standard.
- **A household-bootstrap SE for development benefits.** It is not needed for ranking under the every-cell rule, and it would
  not cover reader refits.
