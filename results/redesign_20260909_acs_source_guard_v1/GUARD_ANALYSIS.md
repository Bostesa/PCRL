# What the guard changed locally

The two Adam proposals start from the same current parameter and optimizer state. The source-only current proposal retains past protection moments; it is not the trajectory T would have taken. The guard projects the current full-minus-source proposal increment on the original F/P protection support. Full-objective moments are retained once. Each guard transition performs two disposable candidate Adam calculations and one live parameter transition; T performs one ordinary live Adam step. Candidate calculations are not additional live training steps. Shared T forward transitions are counted once per seed, while both interface observer sets receive their own updates.

These are local fitting diagnostics. They do not establish monotone native loss, independent probe utility, transfer or privacy. Native losses at pre/source/full/accepted states use only the predeclared first minibatches of epochs 1/20/40/60/80; they never choose or alter an optimizer update. T uses its literal source-only path and has no guard projection. Its original step stream records source losses, support and schedules but not source-displacement norms/dots; any bounded source-only reconstruction supplement is separate evidence and must pass its own replay before being treated as complete.

Guard-support dot fields exclude the ordinary source-head displacement for F; explicitly named full-displacement dot fields include it. The source/full/accepted native losses evaluate the complete forward model on the same fitting minibatch. Ideal float64 feasibility and stored float32 task dots are separate. Positive stored dots can be explained by the declared cast and summation bound; no corrective float32 step is taken. The bound is checked against every real guard update, with full vectors saved at the diagnostic points for independent replay.

| Condition | Present seeds | Guard steps | Nonempty active subset | Max ideal dot | Max stored dot | Max stored dot minus bound | Outside-support identity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F_G_J | [0, 1, 2] | 10000 | 9702 | 4.69e-18 | 2.89e-08 | -1.29e-11 | True |
| F_G_L025 | [0, 1, 2] | 10000 | 9661 | 3.47e-18 | 2.6e-08 | -2.59e-12 | True |
| F_G_L20 | [0, 1, 2] | 10000 | 9704 | 1.02e-17 | 4.19e-08 | -3.39e-10 | True |
| P_G_J | [0, 1, 2] | 10000 | 9754 | 6.51e-18 | 3.22e-08 | -1.06e-10 | True |
| P_G_L025 | [0, 1, 2] | 10000 | 9706 | 3.21e-18 | 2.39e-08 | -3.39e-11 | True |
| P_G_L20 | [0, 1, 2] | 10000 | 9780 | 7.7e-18 | 3.2e-08 | -3.45e-10 | True |

[Per-epoch summaries](GUARD_EPOCHS.csv.gz) include active-set/rank counts, raw/projected/cast norms and per-task before/after dots. [Fixed-point diagnostics](GUARD_DIAGNOSTICS.json.gz) retain all compact state hashes, losses, bounds and support. [Native finite-step changes](GUARD_NATIVE_CHANGES.csv) and [guard summary](GUARD_SUMMARY.json) report signs and magnitudes without turning a fitting calculation into a downstream success criterion.

## Separate exact source-only diagnostic reconstruction

The bounded supplement passed for all 3 shared T trajectories / 10,000 replay updates. It reproduces every original source-loss record, all five saved pre/post model and Adam states, and both final F/P model and Adam states exactly. These are verification updates, not additional fitted systems or live training transitions; the full process is separately charged to the scientific ceiling. No observer/auditor was refitted, no reserved label was read and no original evidence or release was replaced.

The original T stream omitted displacement norms/dots. Their values now come from the separately frozen reconstruction, with a distinct provenance; they are not retroactively attributed to the original fitting logs. T has zero protection increment and no active projection. Its source/full/accepted diagnostic states are identical aliases of one ordinary Adam proposal. Individual finite-step native losses can still increase under the weighted source objective.

[Exact replay certificate and five-point losses](t_source_reconstruction/REPLAY.json) · [Per-seed/task diagnostic rows](T_SOURCE_DIAGNOSTICS.csv) · [Prospective supplement plan](T_DIAGNOSTIC_RECONSTRUCTION_PLAN.json)
