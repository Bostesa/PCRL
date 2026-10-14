# Separate-pool utility guard: read this first

**COMPLETE_NEGATIVE_FEASIBILITY.** The privacy comparison is **NOT_TESTED**.

The fixed trial asked whether an absolute safeguard that checks a whole separate fitting pool (GUARD, 2,043 people) could make real occupation-privacy updates while keeping occupation usefulness on a slice excluded from this continuation's gradients and acceptance decisions (FIT fold 2, 5,211 rows).

What happened:
- **The guard trained.** SEPARATE_GUARD accepted 80/160 and 102/160 nonzero updates. In both seeds, all 8 counterfactual-scanned updates were driven by the privacy term beyond arithmetic noise and changed the released occupation wire.
- **It held its own pool.** GUARD Brier ended exactly at its +0.005 budget, and GUARD NLL rose +0.008.
- **It did not generalize at the required tolerance.** On VALIDATION, occupation NLL rose +0.012 to +0.016 and summed Brier +0.006 to +0.008 versus the unchanged start, at every saved epoch in both seeds. The caps are +0.010 and +0.005.

So the frozen rule selected epoch 0, the unchanged start, and Stage B (fresh attacks, INNER) did not run.

None of the matched privacy-applying arms (GLOBAL_UPDATE_GUARD, OLD_BATCH_GUARD, ORDINARY) passed in both seeds either. TASK_ONLY passed at epoch 5, but it applies no privacy gradient.

This is trainability evidence, not a privacy result. The frozen-income uncertainty from the corrected predecessor trial (12 unresolved income interval clauses) is unchanged.

An independent numpy replay, sharing no metric code with the runner, reproduced the role split, frozen income, every whole-pool budget state, the counterfactual records and all 40 VALIDATION scores. It found no discrepancy and the same FAIL classification.

## Where to go
- FEASIBILITY_DECISION.md: gate table, generalization comparison and matched controls.
- VALIDATION_UTILITY.csv: every seed × arm × epoch × task metric, incremental deltas and U0/Ucal diagnostics.
- GUARD_BUDGET_TRACE.csv / PRIVACY_MOVEMENT_DIAGNOSTICS.csv: per-step slacks and budgets; wire movement and counterfactual scans.
- ROLE_SPLIT_AND_EXPOSURE.md / RECIPE_AND_CONTROLS.md: frozen design.
- SCIENCE_LOCK.json / TRAINING_COMPLETE_LOCK.json: locks committed before fitting and before VALIDATION.
- INDEPENDENT_VERIFICATION.json / VALIDATION.md: independent replay coverage.
- MODEL_MANIFEST.json / QUICKSTART.md / deployment/: packages and raw83 deployment.
- BACKUP_VERIFICATION.json: snapshot, backup-alone restore and pending off-device custody.
- HANDOFF.json, owner_handoffs/: completion dimensions; manuscript and claims notes.

Public artifacts are code, hashes, aggregates and decisions. People, predictions, models and checkpoints stay private. All people were already exposed: this is exploratory development, not independent confirmation.
