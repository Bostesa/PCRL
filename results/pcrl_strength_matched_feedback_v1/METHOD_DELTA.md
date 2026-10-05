# Method delta versus the refreshed guarded joint study (research/pcrl-refreshed-guarded-joint-v1 @ ccdcc5a)

| Item | Refreshed study (rgj) | This study (smf) |
|---|---|---|
| Fitting rows | DEFENSE_FIT (old defense_train, 19,230) | NEW_DEFENSE_FIT (80%, 15,434); the other 20% becomes the new assessment |
| Assessment | 70% of old defense_val (3,397) | NEW_DEVELOPMENT_ASSESSMENT (20% of old defense_train groups, 3,796), labels sealed at load |
| Numeric scaling | fitted on old defense_train | refitted on NEW_DEFENSE_FIT (raw integers recovered exactly) |
| Warm starts / U / LEACE / FARE | reused or derived from rows now in the assessment | all refit from scratch on NEW_DEFENSE_FIT |
| Penalty scale | beta × proxy gradient (scale set by the transform; online arms ~3× larger) | per-encoder norm control: ‖q_i‖ = rho s_i ‖t_i‖ |
| Schedules | refreshed vs online (unmatched effort) | ONLINE, REFRESHED, ONLINE_MATCHED (extra updates = refit receipt) at equal strength |
| Local feedback | CE-surrogate budgets (ref + 0.005), λ updates at 5 refits, η = β — never engaged | AUC targets (ref − 0.01), 20 updates, gain 1/0.01, weights 0.25–8, RMS-preserving allocation |
| Controller rows | CALIB (CE surrogate) | CONTROLLER_CALIB (AUC of fitted probes) |
| Arms | J-G, L-G, J-R, J-O | Phase A schedule frontier; Phase B J-F, L-F, J-N, L-N; raw RAW-J/RAW-L (= rgj J-O/L-O) at 40 epochs |
| Nomination buffer | zero buffer vs min(L-R, L-G, C*) | +0.005 vs L-F and C*; final +0.01 clause unchanged |
| C* local guard | local ≤ L-R | none (strongest task-feasible control) |
| Audit reader | DA canonical PCA dropped < 1e-9 relative variance (blind to rotated low-scale clues) | float64 eps-justified rank tolerance; rotated 1e-6 clue control; selection-aware nulls on a separate split |

Unchanged: architecture, fixed critic head, surrogate, critic kinds, bounded-refit procedure, SGD 0.05, clip 5,
batch 256, 18-slot nine-clause primary family, z = 2.991316, 1,999 grouped bootstrap replicates.
