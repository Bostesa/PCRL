# Data independence: no unused same-population households

**The authorized independent evaluation cannot run on this file.** The original population contains **80,329 eligible persons in 53,907 households**, and all are excluded by verified historical research use. The locked new sample therefore contains **0 persons and 0 households**. This is a provenance blocker, not an observed null result or an independent confirmation of the parent findings.

The audit completed at 2026-09-10T15:14:50.787332+00:00; reconstruction took 10.438 seconds. No new model fit, optimizer update, candidate selection, or scientific outcome scoring occurred. There is no fresh support table: no fresh cohort exists. Raw-file fields were read solely to reconcile documented historical population/cache membership and identifier integrity; no new task performance or class-support analysis was computed.

## Why the 30,000-person exclusion is insufficient

The original transfer cohort has 30,000 persons in 20,147 households. Every regenerated pool in all three seeds exactly matches its saved raw-row array and schema hash. Excluding that cohort alone would leave 50,329 persons in 33,760 households. Those people already appeared in earlier v2 Folktables research.

The earlier v2 loader uses California 2018 ACS, retains ages 16 and above with nonmissing AGEP/SEX/RAC1P, and splits **all** retained rows into 80/10/10 train/validation/test. Its preserved runs explicitly report 248,720 training rows, 31,090 validation rows, and 31,091 test rows. The source and completed-run evidence establish model fitting, validation selection, held-out scoring, and representation diagnostics; mere loading of a public CSV is not the basis for exclusion.

- Population, cache and split construction: [pcrl/data/folktables.py](../../pcrl/data/folktables.py#L191), lines 191–252.
- Uncapped three-split dataset construction: [experiments/run_v2_dataset.py](../../experiments/run_v2_dataset.py#L146), lines 146–159 and 239–250.
- Eraser calibration/model fitting and validation: [run_v2_dataset.py](../../experiments/run_v2_dataset.py#L392), lines 392–402. Audits, task scores and test-representation extraction: lines 456–487.
- Executed population and completed training/test evidence: [Round 1 verdict](../V2_FOLKTABLES_ROUND1_VERDICT.md#L7), lines 7–25; [Round 2 verdict](../V2_FOLKTABLES_ROUND2_VERDICT.md#L3), lines 3–6 and 23–26. The [fixed probe records](../v2_folktables_FOLKTABLES_FIXED_PROBE/per_seed_results.json) independently retain `[31091,64]` test-representation shapes, selected epoch 47, and last epoch 49.

## Independent reconstruction

Raw CSV SHA256: `dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0`. It contains 378,817 rows. The original same-population filter is integer AGEP 19–34 inclusive and positive PWGTP; identifiers SERIALNO and SPORDER were explicitly parsed as strings. No target duplicate person keys or fractional ages occurred. Identical duplicates are deduplicated and conflicting duplicate person records raise an integrity error. SERIALNO remains the household/group key, including group quarters; no new group-quarter filtering is introduced.

Historical v2 cache SHA256: `b1d08f98f9a95ac163d0b8fd56edec2218c40f4995de47aa55650b5dd3476f04`. I reconstructed all 310,901 historical rows from the verified raw CSV and compared every cached column and row position using `assert_frame_equal(check_dtype=False, check_exact=True)`. This passed. The cache serves only to recover historical exposure and was not substituted as evaluation data. These rows cover 150,826 households, including older adults outside the authorized evaluation ages. Their household exclusion union contains every eligible person and household.

The verifier reconstructs the original transfer cohort and all seven pools for seeds 0, 1, and 2 independently of the historical loader. It compares each raw-row array against both `schema_support.json` and `seed_s/split_rows.npz`; all 21 comparisons pass. The complete cohort raw-row hash is `b869e863fb9f002732758d10c7f4b4cc8fa6bcb64b23f34d36b2fbe507250307`.

| Historical use reason | Historical persons | Historical households | Direct eligible persons | Eligible persons excluded by household | Eligible households excluded |
|---|---:|---:|---:|---:|---:|
| v2_train | 248,720 | 137,731 | 64,058 | 76,678 | 50,713 |
| v2_validation | 31,090 | 28,626 | 8,077 | 20,842 | 12,655 |
| v2_test | 31,091 | 28,621 | 8,194 | 20,876 | 12,710 |
| complete_transfer_cohort | 30,000 | 20,147 | 30,000 | 30,000 | 20,147 |
| legacy_acsincome_population | 195,665 | 111,426 | 63,716 | 75,298 | 49,334 |

Reason-specific counts overlap; they must not be added. Legacy ACSIncome rows are included using their exact source predicate and are dominated by the v2 union. The explicit whole-cohort exclusion covers all transfer pools and downstream dependencies, not merely representation-fit or selected head fitting rows.

## Dependency and aggregate-exposure coverage

The retained transfer preprocessor and PCA fit only each seed's representation-fit pool: [run_acs_transfer.py](../../experiments/run_acs_transfer.py#L119), lines 119–153. The current fitted systems are **not** being described as having fitted on the whole eligible file. The earlier v2 numeric preprocessor used its entire historical 248,720-row training split, and those historical models used all three splits for fitting, selection, and evaluation. The blocker is broad historical research exposure.

Protection, preservation, bottleneck, PCA16/PCA16 initialization, selective preservation, restricted inputs, coalition, coalition-strength, source-guard and fixed-prediction configs route to the same transfer parent. Their configuration hashes are recorded in `DATA_USE_INVENTORY.json`; the full parent cohort therefore covers their representation/preprocessing/PCA/eraser/teacher/anchor/auxiliary/head/observer/auditor/selection/development/test uses. This exhaustive v2 exclusion already dominates any additional target-row use by these dependencies.

The application screen records ACS as inventory-only and says no ACS outcomes/correlations were analyzed: [saved application selection](../redesign_20260907_application_screen_v1/source_snapshot/docs/PCRL_APPLICATION_SELECTION.md#L246), lines 246–253. Original-file loading, eligibility checks and household sizes are prior aggregate exposure rather than model fitting. Transfer schema preparation computed supports only for the entire selected transfer cohort, which is fully excluded.

Some historical jobs lack raw-identifier manifests. Exact cache-to-raw reconciliation recovers their population membership, while preserved executed-run records establish its scientific use. The older ACSIncome processed cache is absent, so its exact source predicate provides conservative coverage supported by retained results; it cannot enlarge the eligible remainder because its relevant rows are already covered by v2. Unrecorded exploratory uses or outside-repository activity cannot be exhaustively reconstructed; any such extra use only enlarges the already exhaustive exclusion. Missing historical tables are never treated as evidence of non-use.

## Locked sample and locally retained evidence

Canonical household IDs use compact ASCII JSON `["ACS/2018/1-Year/CA/person",SERIALNO]`; person IDs append original string SPORDER. Eligible households would be ordered by SHA256 of compact JSON `["PCRL_LOCKED_EVAL_20260910_V1",canonical_household_ID]`, breaking digest ties by canonical ID. Selection takes the longest initial sequence of complete households totaling at most 30,000 persons, and never skips a group to fill a gap. This selection has an empty remainder and produces an empty sample once. No task outcomes, sensitive labels, predictions or weights beyond the original positive-weight eligibility filter influence the order.

The independent verifier implements this ordering separately from the evaluator. Synthetic tests verify older household members exclude otherwise eligible people, original string identifiers survive, conflicting duplicates fail, and a too-large middle household terminates the prefix rather than allowing later groups to fill the gap. The complete local union includes older household members, while the target exclusion manifest preserves only authorized-population rows.

All identifiers are retained only in ignored `local/` files. The empty person and ordered-household sample manifests are zero-byte JSONL files. Reproduction: run `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/verify_acs_locked_independence.py` from the repository root. Compatible replays compare completed artifact bytes and preserve the original completion timestamp. Any differing completed artifact causes an error rather than an overwrite.

| Local artifact (omitted from publication) | SHA256 |
|---|---|
| `exclusion_households.jsonl` | `c1a909d9db84cf2c979fd9312b93c839deff6ebb43a960fbcb8a5fc1faff9305` |
| `excluded_target_people.jsonl` | `5d6f4ceebc8d9f1b4135e014d284f7ebdf7a5463a569ec12bb4b3e0f2ab46648` |
| `historical_used_people.jsonl` | `a6ec3fbcfe2b473a44f629a1d50a876ab5b3fe2181af39ebaf12aaee9595f227` |
| `sample_people.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `sample_households_ordered.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `v2_train_people.jsonl` | `a96a62159897a5766731641a5a562bddff7da516f6606d45dbf9cef1c3d4eb39` |
| `v2_train_households.jsonl` | `4afb64884010405082c5158cd6d2018bd2d24daeacfde8e332ec35cb0ae179ff` |
| `v2_validation_people.jsonl` | `d966bd50491dd2593625401f643ccbdd77ef11c0c64c4afe6cd77e3cf1ad639f` |
| `v2_validation_households.jsonl` | `23c2cf9dd9ef1d44df171090fab9442f62ff2104c8f069917057e3e3918aa13c` |
| `v2_test_people.jsonl` | `4cc86b22dbabeca49b4d69db1628f2d1fbb81c9259a09ea94721ab4fe813b82d` |
| `v2_test_households.jsonl` | `94add20a6019b53317662be532e2de5fd7a3d567145e03904e38f8eaeea8f2a0` |
| `complete_transfer_cohort_people.jsonl` | `c9f89489307a03efdadd3b3787551c68a895ce01e28b5bfd9fa7e3a44ce4a2fb` |
| `complete_transfer_cohort_households.jsonl` | `33c93dd73673ffa4170182d2ffb2adaf7209deb6567a4d86c1412de63fb5df55` |
| `legacy_acsincome_population_people.jsonl` | `e08858a45ae63c55e781b46ef0edde3679ab99b608e76222cdcc374be4e13497` |
| `legacy_acsincome_population_households.jsonl` | `a9bbe2539bed878bd4f6f77ddc557dfd2967daecb48905071f28e7731c9867bf` |

Independent verifier source SHA256: `fae38e8bfc68bde133c70d48d114e14eb05ecf7cd9a7333f3509db04cd3dbab5`. The aggregate [DATA_USE_INVENTORY.json](DATA_USE_INVENTORY.json) binds source files, local union/sample artifacts, all per-reason counts, and all 21 transfer-pool comparisons. [DATA_INDEPENDENCE_COMPLETION.json](DATA_INDEPENDENCE_COMPLETION.json) binds the completed inventory with its actual timestamps.

All new scientific inference, source allowances, residence/disclosure comparisons, fresh support metrics, six bootstrap intervals, and observed-point tradeoff evidence remain unavailable. The evaluator must reject those phases for this zero-household sample. No alternative cohort or population was consumed.
