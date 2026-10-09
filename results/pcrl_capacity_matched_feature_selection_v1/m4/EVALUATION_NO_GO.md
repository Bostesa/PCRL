# M4 evaluation no-go after the task-only screen

The two locked screen FIT units, `M2-FIT-S0` and `M2-FIT-S1`, completed with `NO_FEASIBLE_J` and reason `NO_UTILITY_FEASIBLE_TASK_MASK`. In each seed, zero of five fixed masks for income and zero of five for occupation met the predeclared task-only utility test. Each immutable `CONTROL_PLAN.json` marks all ten rule slots `INVALID_NO_UTILITY_FEASIBLE_PAIR`. No J or T pair was selected.

This is an exploratory task-utility no-go for this fixed five-mask bank, training design, references, and seeds. It is not a privacy result. No candidate release, fresh SEX attacker, complete-view AUC, V2 real-role probe, utility endpoint on INNER_SELECTION, or J-versus-T comparison was produced. It does not establish that purpose-specific feature selection in general fails.

## Access and execution boundary

The M2 pipeline writes the task-only table and returns before its `read_sex()` callback when T has no utility-feasible pair. The no-go FIT receipts contain no pair table or joint selection, no final head packages, and `search_receipt.status=NOT_RUN_TASK_NO_GO`. The executed FIT units read allowed FIT task features/labels/references and held-out calibration features/task labels. They did not read FIT SEX or any `ATTACK_FIT_NEW` or `INNER_SELECTION` role arrays. The queue records only the two FIT units as DONE.

Every M4 `release`, `raw_union`, `attack_view`, and `v2_probe` worker requires a hash-bound M2 parent receipt with `status=FIT_COMPLETE` before constructing the role store or reading ATTACK_FIT_NEW/INNER_SELECTION arrays. The M4 endpoint worker likewise checks FIT_COMPLETE and selected-child receipts before INNER_SELECTION access. A queue status of DONE for a no-go FIT is therefore insufficient to admit downstream units. M5 pair replay requires releases, attacks and V2 probes that do not exist; none ran.

## Frozen unit accounting

The sealed `UNIT_SPECS.json` registers **628 potential units**, including all 25 mask pairs per seed. Registration is an upper-bound roster, not 628 promised fits. The screen-seed descendants below were never enqueued after the two FIT no-go outcomes.

| Unit class | Seeds 0–1 registered | Disposition |
| --- | ---: | --- |
| M2 FIT | 2 | DONE, both NO_FEASIBLE_J |
| M4 raw-union reader | 12 | Skipped: no selected pair |
| M4 release | 50 | Skipped: no selected pair |
| M4 complete-view attack | 150 | Skipped: no release |
| M4 real-role V2 probe | 150 | Skipped: no release |
| M5 independent pair replay | 50 | Skipped: no pair artifacts |
| M4 FIRST and SCREEN endpoint | 2 | Skipped: no selected rule |
| M3 SCREEN inference | 1 | Skipped: no endpoint index |
| M5 SCREEN aggregate replay | 1 | Skipped: no endpoint or inference |

Thus **416 screen-path potential units were skipped**, including 50 registered M5 pair replay identities. Had both seeds selected the maximum ten rule pairs, at most 20 distinct pair replays would have been scheduled; the 50 figure is the sealed 25-pair-per-seed superset. The conditional finalist path was not activated: its one seed-2 FIT, 206 seed-2 potential descendants, and three FINALIST endpoint/inference/aggregate units total **210 additional unrun potential units**. The ledger is 2 executed + 416 skipped screen-path + 210 untriggered conditional finalist = 628.

## Hash-bound evidence and limit

- Campaign lock SHA-256: `93ed6271f30da39e6010784895d105f77a96a17fe573dcaf5209fb8bc932a6cb`.
- Sealed private unit registry SHA-256: `0eaae64a41ac7a279db3048596ce4b0d607f2523ffaacc958fd5b7876f7ae9f2`.
- Seed-0 FIT receipt SHA-256: `be387533d68828ebc45d12132ba5f9c4d1a90092a68160d8af45d78434e70534`.
- Seed-1 FIT receipt SHA-256: `0984398e414f9fe3c96e5b988483d67cbb2df56f2762e23b21e7828e807f7ccb`.

These are execution and custody observations from aggregate receipts and the sealed registry. M5's independent replay of the task-only no-go is a separate pending check; this note does not claim that verification has passed.
