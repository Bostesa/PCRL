# R19 proposal: D point endpoint rows

Status: `POINT_AND_FIXED_K567_INTERVALS_INDEPENDENTLY_REPLAYED`. These are read-only candidate exports for R00's shared `PRIMARY_ENDPOINTS` work. They are not the shared endpoint file or a selection decision.

The endpoint CSVs contain exactly 56 task rows (seven arms × two paired seeds × two tasks × U0/Ucal) and 42 recovery rows (seven arms × two seeds × recipient 1, recipient 2, aligned pair). A third CSV preserves the full 567-clause simultaneous family:

- `D_PRIMARY_TASK_ENDPOINTS_PROPOSAL.csv`, SHA-256 `41029367fe51f0eb1672cd4d49a51446417d14be6b3886f08f4896d06f912b7e`.
- `D_PRIMARY_RECOVERY_ENDPOINTS_PROPOSAL.csv`, SHA-256 `27d4be956660a273116a029cd78bd1ffe9b45b36bfcb379a76fa8813a184c5b4`.
- `D_K567_CLAUSES_PROPOSAL.csv`, SHA-256 `3501b7f1fa268bad28bb55f87cd9bf92e773f602b3147a4866e0ee54a4c35877`.

Every endpoint row records its independent R04 point receipt hash and the interval receipt SHA-256 `3c8ad07fbedcc62044c538451f2ffd4292f0b95088120fc9c26206c02c7d5efa`. Point values come from `all_seven_point` in R04's seed-0 and seed-1 `FOLLOWUP_S*_ATTACK_INDEPENDENT_REPLAY.json` receipts, SHA-256 `e7b2a5ca7a28466008e9a26dab8441437462b23299f9ffd8faee2f08a3b1024f` and `7654b780f1db676c173b1d625f4da7ca18f18a77e619fd340b0cf31cb1eae086`. Both report `FOLLOWUP_COMPLETE_VIEW_REPLAY_PASS`. R04's `D_567_INDEPENDENT_INTERVALS.json` reports `FIXED_567_INTERVALS_COMPUTED`, point parity for both seeds, 2,048 paired group resamples, and one simultaneous critical value for all 567 registered raw contrasts. The exports contain no person-level rows or private paths. The public `derive_d_k567_proposal.py` regenerates the interval fields from that hash-pinned receipt and the R19 point rows.

| Arm | Seed 0 pair SEX AUC | Seed 1 pair SEX AUC | Both-seed task point gate |
| --- | ---: | ---: | --- |
| T | 0.832582 | 0.835960 | Fail |
| D-K | 0.805905 | 0.793778 | Fail |
| L | 0.817431 | 0.796456 | Fail |
| S12 | 0.802288 | 0.797881 | Fail |
| S21 | 0.811717 | 0.797484 | Fail |
| C | 0.743901 | 0.764677 | Fail |
| F, continuous | 0.869335 | 0.862196 | Pass at point level |

All six ten-bit hard-code arms fail the both-task, both-reference point utility gate in each seed. F alone meets all 24 point clauses, while disclosing 16 continuous latent values per recipient and task outputs, a different wire capacity. The hard-code matched eligible set is empty. D-K lowers pair AUC versus T by 0.026676/0.042181 across seeds, but also fails utility. C has the lowest hard-code pair AUC and fails utility; it cannot serve as an eligible competitive control. F has higher pair AUC than T by 0.036754/0.026237 and higher than D-K by 0.063430/0.068418, but those comparisons span different capacities. Neither a competitive D-K result nor a kernel or joint-design component gain follows.

The computed raw K567 family contains 252 utility, 189 arm-pair recovery and 126 arm-versus-U0/Ucal recovery clauses, for seed 0, seed 1 and their equal mean. The task CSV adds each signed accuracy difference and NLL/Brier excess interval plus a three-metric status: 3 task/reference rows `PASS`, 16 `VIOLATION`, 37 `UNRESOLVED`. Among all 252 utility clauses, 42 pass, 43 violate and 167 remain unresolved. The recovery CSV still has **no absolute AUC confidence interval**: K567 gives signed *comparisons*, preserved without assuming every pair's control/candidate direction in the 567-row CSV.

D-K has a confirmed mean occupation NLL excess over U0: lower bound **0.010368** above the 0.010 cap. Its mean occupation Brier excess over Ucal has lower bound **0.006994** above the 0.005 cap. The mean T-minus-D-K pair AUC reduction is **0.034429**, simultaneous interval **[0.013719, 0.055138]**; it crosses the 0.020 benefit target and is `UNRESOLVED`, even before accounting for both arms' utility failures. F meets all 24 task *point* clauses; across its 36 seed/mean utility intervals, 27 `PASS` and 9 are `UNRESOLVED`, so it is not a complete simultaneous utility pass. The two model seeds share the same INNER people. No competitive, kernel or joint-design claim is supported. R00 will decide whether and how to incorporate these rows after source and eligibility checks.
