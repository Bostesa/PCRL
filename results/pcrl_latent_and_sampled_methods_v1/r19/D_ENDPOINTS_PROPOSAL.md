# R19 proposal: D point endpoint rows

Status: `POINT_ONLY_INDEPENDENT_REPLAY; K567_INTERVALS_PENDING`. These are read-only candidate exports for R00's shared `PRIMARY_ENDPOINTS` work. They are not the shared endpoint file or a selection decision.

The two CSVs contain exactly 56 task rows (seven arms × two paired seeds × two tasks × U0/Ucal) and 42 recovery rows (seven arms × two seeds × recipient 1, recipient 2, aligned pair):

- `D_PRIMARY_TASK_ENDPOINTS_PROPOSAL.csv`, SHA-256 `6b155e930b09ce57e9738a459c49aeac665bc1fa17d7df33231acdc5eff8bd97`.
- `D_PRIMARY_RECOVERY_ENDPOINTS_PROPOSAL.csv`, SHA-256 `cf0eee23354b4daae72045fa6cea3d6dada7c3d1c2ec111f5e145a1d15f5afd0`.

Every row records its independent R04 receipt hash and `simultaneous_interval_status=PENDING_K567`. Values come from `all_seven_point` in R04's seed-0 and seed-1 `FOLLOWUP_S*_ATTACK_INDEPENDENT_REPLAY.json` receipts, SHA-256 `e7b2a5ca7a28466008e9a26dab8441437462b23299f9ffd8faee2f08a3b1024f` and `7654b780f1db676c173b1d625f4da7ca18f18a77e619fd340b0cf31cb1eae086`. Both report `FOLLOWUP_COMPLETE_VIEW_REPLAY_PASS`. The exports contain no person-level rows or private paths. R04's [all-arm point and interval plan](../r04/D_ALL_ARM_POINT_AND_INTERVAL_PLAN.md) defines the remaining independent inference work.

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

The sealed raw K567 interval family, including 126 arm-versus-U0/Ucal recovery contrasts, is still pending prospective anchor reader banks and R04 simultaneous replay. Keep all 567 clauses. A point pass is not an interval pass; the two model seeds share the same INNER people. R00 will decide whether and how to incorporate these rows after source, interval and eligibility checks.
