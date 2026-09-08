# Independent completed-run review

Read-only review of the frozen protocol, executed source, raw metrics, actual optimizer/checkpoint files, fitting schedules, cached releases, and split IDs. No additional model fitting or sweep was performed. This new report is the only result-directory addition from the review.

## Checks completed

- All 50 MLP fits contain two complete trajectories: 100 final Adam states each report 1,800 actual updates. Each saved schedule has shape (1800,256), totaling 460,800 fitting-row presentations per trajectory and 921,600 per fit. Total across task and attacker probes: 180,000 updates and 46,080,000 row presentations. Attacker fitting rows appear exactly 225 times per trajectory; task fitting rows appear 112 or 113 times. Initialization optimizer states are empty, and initial/final weight hashes match metadata.
- Every trajectory logs validation at update 0 and every 40 updates through 1,800. All 70 selected target checkpoints match the minimum raw validation MSE among 92 opportunities, with the declared restart-then-update tie order. All selected checkpoint weight hashes and target names/indices match metadata. Saved schedules are identical across release arms for each seed/view/restart.
- All 50 independently fitted histogram models contain exactly 200 iterations and the fixed 15-leaf, learning-rate .05, l2=1, no-early-stopping configuration.
- Every probe's saved preprocessing arrays and fitting/validation hashes match its designated cached fitting/validation arrays. Attackers use 2,048 attacker-fitting rows; task probes use 4,096 representation-training rows. No validation or test moments appear in preprocessing.
- Reconstructed saved MLP, OLS, and histogram models reproduce all 380 validation/test target score records exactly: maximum absolute MSE/R²/variance difference is zero. Every evaluated target has positive variance and the correct sample count: validation 2,048, test 4,096.
- All 30 within-seed split pairs are disjoint, including calibration and fresh test. IDs have the prescribed RNG-seed prefixes and row counts; fitting manifests exactly match historical manifests. The runner saves all selection metadata before generating final test, and each current selection-file hash matches both recorded pre-test and post-test hashes. Family selections match validation-only R² maxima.
- All three scalar source checkpoint hashes and all 12 frozen source hashes, plus the frozen protocol hash, match. Re-loading the saved encoder/heads and applying them in evaluation mode exactly reproduces every cached scalar release on all four evaluated splits. Parameter/buffer snapshots before and after fitting/evaluation match; historical scalar validation R² replay differences are exactly zero for both tasks and all seeds.
- The saved original and continued adversary checkpoints for the full representation also reproduce all their validation/test score dictionaries exactly. No final eraser was refitted.

## Verified scalar results

MLP utility and descriptive maxima over all three tested attacker families on the fresh test. Negative predictive R² remains unmodified; it means worse-than-evaluation-mean prediction, not negative information. All 15 individual/family checks pass separately on validation and test for each seed; both independent utility scores and native scalar task scores exceed .99 for each purpose.

| Seed | U utility R² | V utility R² | Native U R² | Native V R² | Worst individual R² | Combined-S maximum R² | Test checks passed |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.998005 | 0.996804 | 0.997879 | 0.996539 | 0.000228 | -0.000593 | 15/15 |
| 1 | 0.998146 | 0.997708 | 0.997939 | 0.997704 | 0.000075 | -0.002529 | 15/15 |
| 2 | 0.997953 | 0.996046 | 0.997743 | 0.995993 | -0.000473 | -0.003616 | 15/15 |

The largest scalar individual leakage estimate over validation, test, all seeds and families is 0.000916; the individual threshold is .05. The largest combined-S estimate is -0.000593; its threshold is .10. Every underlying relationship/family is checked separately in the saved metrics.

## Controls

The exposed-target control verifies target alignment and learning competence. Minimum fresh-test R² across all exposed forbidden targets and seeds: OLS 1.000000, MLP 0.999811, histogram boosting 0.996258. Restricting this calculation to S gives minima 1.000000, 0.999811, and 0.996635. The oracle control passes all 15 leakage checks in all three fresh-test seeds; its largest fresh-test leakage estimate is 0.000325.

The known-leaky full seed-2 E_dual_0.01 representation fails the audit (7/15 fresh-test checks pass). Fresh MLP R² is 0.349013 for P1→V, 0.149281 for P1→S, 0.296702 for P2→U, 0.185202 for P2→S, and 0.352950 for concatenated S. Trees independently detect P1→V (0.155012), P2→U (0.195354), and concatenated S (0.125327). Thus the stronger audit detects nonlinear leakage in the fixed positive control despite low linear leakage there.

## Limits

These are empirical results for the declared two tasks and three attack families. Reused validation makes this exploratory; checkpoint and family selection are nevertheless isolated from the fresh final test. Equal update counts do not imply equal attacker strength: fresh attackers use 2,048 distinct fitting examples, while inherited continued weights also saw prior representation-training data, and scalar/full input dimensions differ. The exposed and known-leaky controls demonstrate useful competence without establishing universal privacy or robustness to every attack. Scalar prediction releases do not establish utility for future unannounced tasks. No PCRL novelty claim follows.

No correctness discrepancy was found in the completed runs. The pre-run minor undefined-R² family-selection concern did not occur: all evaluated target scores were defined.
