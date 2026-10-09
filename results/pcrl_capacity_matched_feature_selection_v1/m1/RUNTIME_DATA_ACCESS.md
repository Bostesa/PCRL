# Runtime data access: focused FIT seeds 0 and 1

This is an aggregate, source-and-receipt audit of the first two focused scientific units. It does not expose row-level data or claim an operating-system syscall trace. The effective lock SHA-256 is `93ed6271f30da39e6010784895d105f77a96a17fe573dcaf5209fb8bc932a6cb`; the focused role manifest SHA-256 is `46a8c609304a67a7d1031b1ae929b43b09a630624db39139b4ed45affad6196e`.

| Unit | Queue status | FIT receipt SHA-256 | Task-only receipt SHA-256 | Result |
| --- | --- | --- | --- | --- |
| `M2-FIT-S0` | DONE, exit 0 | `be387533d68828ebc45d12132ba5f9c4d1a90092a68160d8af45d78434e70534` | `759bafad69e3b97a2e0c16e3c12bf4cc058fcc10996eaa44125e235e4ec3e3ae` | `NO_FEASIBLE_J` |
| `M2-FIT-S1` | DONE, exit 0 | `0984398e414f9fe3c96e5b988483d67cbb2df56f2762e23b21e7828e807f7ccb` | `f5afaa731909fae32abdfde78f3548f3b707fb96bea3e693f93d2c6b03e16f7f` | `NO_FEASIBLE_J` |

The worker's guarded loader requested exactly these role-only records in each unit:

- `OSF_DEFENSE_FIT`: `features`, `income`, `occupation`, and `references_s{seed}`. The FIT role has 15,434 rows in 15,428 groups.
- `CALIBRATION_HELDOUT`: `features`, `income`, and `occupation`. The calibration role has 2,000 rows in 2,000 groups.

Those requests supported the task-only internal split, 12 validation head packages per seed (six per task, including full input), and task utility comparison. The two task-only receipts record `NO_UTILITY_FEASIBLE_T` with no selected pair. Each control plan marks all ten registered rule slots `INVALID_NO_UTILITY_FEASIBLE_PAIR`; no pair table, joint selection, final release package, or privacy-search critic was produced. The FIT receipts record `search_receipt.status=NOT_RUN_TASK_NO_GO` and `critic_spec=null`.

**FIT SEX was not deserialized by the scientific worker.** In governed `m2/fit_pipeline.py` SHA-256 `cae3b281b43ee3941e6b71e20037749cf835879b54df647f2a5940a0ecb786f1`, the no-go branch returns immediately after the task-only receipt callback and before `read_sex()`. Governed `m2/fit_worker.py` SHA-256 `c79d079681c9c7e6159aa95a7b1ff33ce1c3dafe1f03a4b7b2e8a18f7eb56a50` contains the sole `store.read(OSF_DEFENSE_FIT, "sex", "defense_fit")` inside that callback. Both FIT receipts bind the no-go branch, and their private output directories contain the task table, task-only receipt, control plan, and FIT receipt only. The conclusion follows from the locked call chain and receipts; it is not a separate syscall measurement.

The sealed queue specs declared `OSF_DEFENSE_FIT:sex` as a *potential* FIT input. M0's queue opened and SHA-256-hashed all eight declared role NPZ archives per unit, including the FIT SEX archive, as opaque compressed bytes during admission/input checks. That custody hashing did not deserialize NPZ members or supply protected labels to the task search. This distinction applies to the earlier failed startup attempts as well.

At this audit, `M2-FIT-S0` and `M2-FIT-S1` were the only real focused queue units marked DONE. Their call chain contains no `ATTACK_FIT_NEW`, `INNER_SELECTION`, `HEAD_VALIDATION`, or old assessment loader request. The first two roles were therefore not read by these scientific workers; the latter two remain forbidden. The role-only exports used here came from the prior portfolio's admitted custody; its documented transient mixed-source deserialization occurred during the earlier mechanical export, not during either focused FIT unit.

This no-go is a task-utility result on reused Adult development roles. It does not establish a privacy gain, confidence-preserving release, or independent confirmation.
