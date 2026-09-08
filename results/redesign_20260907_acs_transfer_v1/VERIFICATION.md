# Independent ACS artifact verification

All three completed seeds passed artifact, split, source-selection, and schedule consistency checks. No optimizer update or model fit was executed. Source-validation predictions were replayed from saved objects; final-test predictions were not used to select anything.

| Seed | Source-fit / source-val rows | Selected LR | Selected epoch / step | Total source steps per LR | Selected tree leaves |
| --- | --- | --- | --- | --- | --- |
| 0 | 10513 / 3004 | 0.003 | 22 / 924 | 2520 | 15 |
| 1 | 10428 / 3010 | 0.001 | 35 / 1435 | 2460 | 15 |
| 2 | 10551 / 2979 | 0.001 | 47 / 1974 | 2520 | 15 |

- Both source learning-rate configurations start from identical saved tensors and use the independently regenerated same minibatch schedule. Initial checkpoints contain epoch 0 / step 0 state; selected and final tensors match their recorded hashes.
- Source fitting uses only the five permitted label arrays on the recorded rows. Reconstructed masks, class supports, income edges, feature allowlist, and fitting-only preprocessing agree with saved artifacts in every pool. The two source tree settings use those same labels/examples, L2=1, and 150 iterations per nonconstant head.
- All seven releases replay on source validation within the reported numerical tolerance. Every saved development-array hash and local artifact-manifest hash verifies. Household/person pools are disjoint; whole-household sampling and the common cohort across seeds verify.
- Downstream and audit heads use identical selected rows, labels, validation rows, seeds, full 40-epoch MLP schedules, and exposure counts across interfaces. Source configurations and downstream families/checkpoints minimize the recorded validation criteria; no final-test choice was made.
- Each saved source/head selection precedes the recorded test-evaluation timestamp; its hash remains unchanged. Recorded neural/tree/PCA/preprocessing before/after hashes agree. JSON records whether reloaded map object hashes also agree, separately from prediction replay.

Coverage caveat: the fixed ESR probability schema retains all six categories even when a source-fitting category is absent. Explicit per-seed supports and tree absent-column checks are in the JSON; absence is not a learned-class or privacy-success claim.

Serialization caveat: reloaded HistGB/tree-map joblib object hashes differ from their in-memory hashes; even a pure pickle roundtrip changes these object hashes. The saved file SHA256 checks and source-validation prediction replay pass. Recorded in-process before/after map hashes agree; this verifier does not claim to reconstruct a pre-serialization object hash.

Standalone prior metadata contains its validation scores, consistent with the frozen source and selection records.

These checks validate internal evidence consistency and inference replay, not an independent retraining. Actual optimizer execution counts are checked against saved logs/checkpoints and regenerated schedules. The source/protocol/data hashes remain frozen.

Verification runtime: 6.528 seconds, CPU, one numerical thread. INDEPENDENT_VERIFICATION.json includes the verification source and command so these inference-only checks can be replayed without modifying frozen experiment files.
