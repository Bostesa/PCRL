# Validation and provenance

Both complete-matrix independent replays passed. The [score/model replay](SCORE_REPLAY.json) checked 5,877 candidate records and 1,980,377 numerical score fields with maximum discrepancy `6.66e-16`, below the declared `2e-12` tolerance; all replayed probabilities matched bitwise. The [training replay](TRAINING_REPLAY.json) checked 54 fixed gradient points with maximum discrepancy zero. Scientific execution source remains frozen; no replay refitted a scientific model or changed a selection.

| Independent check | Completed scope |
|---|---:|
| Raw-label score dictionaries, including both weighting rules | 23,508 |
| Candidate probability arrays | 11,754 |
| Literal released arrays / public-head composition arrays | 819 / 189 |
| Frozen audit selection pools | 1,386 |
| Inherited singleton probability arrays | 2,976 |
| Actual 120/360 terminal Adam/RNG checkpoints | 1,728 |
| Nested trajectory-prefix checks | 864 |
| Genuine saved-observer fidelity checks | 162 |
| Selected-auditor repeated-access prediction arrays | 1,011 |
| Native source probability arrays / score dictionaries | 144 / 288 |
| Local artifact hashes | 15,087 |

The 864 nested-prefix checks include the 840 new MLP/catch-up trajectories and 24 reused historical MLP trajectories. The full score/model replay took 235.263 seconds; full training replay took 2.498 seconds. Scientific fitting-process wall time, including the failed attempt and bounded recovery, was 1,883.536 seconds (31.392 minutes). These timings describe different phases; the replay durations are not additional fits.

The final [report replay](FINAL_REPORT_REPLAY.json) independently recomputes all 3,908 main aggregate rows and 4,368 fixed paired aggregate rows from the canonical selected scores: all 20,512 mean/SD/sign comparisons match exactly. It also verifies 174 report-input hashes. The separate [publication check](REPORT_PUBLICATION_CHECK.json) verifies all four gzip roundtrips, report links and 6,120 contextual paired aggregates. [Final source identities](FINAL_SOURCE_IDENTITIES.json) distinguish publication/reporting code from the original fitting freeze and its recorded amendment. [Runtime](runtime.json) separates scientific process wall, measured read-only phases and total elapsed work at report freeze.

## Focused implementation checks

The prefit suite passed 20 parameterized checks: eight training, nine audit, and three runner checks. The nine audit tests cover all declared observer dimensions, direct saved-start prediction fidelity, Adam reset and actual nested terminal states, observer immutability, the complete toy P audit matrix, exact inherited predictions, three pool memberships, P deduplication, held-out-input rejection, public-derived projection, and all role/seed/restart/catch-up seed collisions. A memory-layout discrepancy discovered in the toy inherited-path check was fixed before freezing by canonical contiguous auditor inputs; the exact equality assertion remains in place.

Three additional read-only replay tests check literal paired inference and 6,355 parameter accounting, assigned public source-head composition, exact same-person concatenation, canonical singleton projection, and independent membership of the three attack pools. The recovery/replay bundle passed seven checks after the operational amendment: four runner and three replay checks; three runner checks repeat the prefit suite. The reporter suite passed 13 checks, including lineage and compressed-CSV fixtures. These are phase counts with overlap, not a sum of distinct experiments. The first-seed training replay passed 18 fixed gradient points with maximum numerical discrepancy zero in 1.3928 seconds. The first-seed score replay also passed: 1,959 candidate records, 7,836 score dictionaries, 3,918 probability arrays, and 576 actual terminal checkpoints, with maximum score error `4.44e-16`, in 79.898 seconds. This is a completed first-seed plumbing check, not the final full-matrix replay.

## Documented operational recovery

The first scientific process completed all six learned conditions and their utility/audit fits, then stopped while loading direct-E utility metadata. The historical fitted-artifact manifest lists binary artifacts but omits `metadata.json`. This was a provenance loader mismatch, not a numerical objective or attacker failure.

[EXECUTION_AMENDMENTS.json](EXECUTION_AMENDMENTS.json) preserves the original runner hash and a copy of its exact source, the failure record, and hashes of every pre-recovery artifact. The loader now binds omitted historical metadata to the original published selection records. Six selection anchors are checked against commit `dcab4e16f5b9a094c67a5505a22dc2bbd6d6d286`, and selective-study completion hashes supply an additional binding. Binary artifacts retain their original manifest identities.

The single bounded retry loads every completed model, observer, candidate, and validation selection without optimization. It verifies the original release hashes and preserves all 4,512 pre-recovery files, as independently checked again during the final replay. Only previously unstarted direct-control additions are fitted. Seed 0’s six learned-condition fits were executed by the original frozen runner and are only loaded by the amended runner; subsequent unstarted fits use the amended loader. Original protocol/source hashes were not overwritten, and the amendment records the actual later runner source. Failure and recovery process times count toward the scientific process ceiling.

## Independent score and model replay

[scripts/verify_acs_coalition.py](../../scripts/verify_acs_coalition.py) reads raw ACS columns and reproduces masks, target schemas, fitting row permutations, and person weights independently. It uses the established independent scalar/sklearn score formulas and literal saved-network inference, with an absolute metric tolerance of `2e-12`; predictions and projection paths must match bitwise. It imports no experiment fitter or scorer.

The verifier checks source/configuration freezes and amendment lineage, original raw-data/split identities, local artifact hashes, both literal forward branches, exact F-to-public-head compositions, P's published dimensions, all eleven role memberships, every candidate score and probability, utility and three audit selections, the full singleton inheritance sets, saved observer state/exposure and direct-coordinate fidelity, MLP initializations and schedules, and actual last-training Adam/RNG checkpoints at epochs 120 and 360. Nested best-checkpoint curves must share the same first-120 prefix. Native source probabilities and both scoring weights are checked separately. Selected 360-epoch auditors are evaluated through each first-copy projection for repeated k = 1, 2, and 4 access.

The separate training verifier inspects original initialization, shared source warmup, exact interface/regime forks, schedules and Adam histories, objective coefficients, source/protection gradient routing, parameter accounting, and snapshot immutability. It does not retrain the model to claim a second scientific replication.

Original test households remain **DEVELOPMENT EVALUATION**. Three seeds share one sampled cohort; their dispersion is descriptive. No fresh household pool, dataset, state/year, attacker family, or follow-up sweep was consumed. Full race support flags and exposed-control failures remain limitations of the finite audit.
