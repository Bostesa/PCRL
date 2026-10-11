# Conditional budgeted releases: start here

## Result

The complete eligible study ran. **No competitive method or joint-design advantage was established.** The selected wide kernel reduced coalition SEX AUC by 0.00196 versus the incumbent; the target was 0.020. Sequential 1→2 was the strongest eligible control and had lower recovery than the selected kernel. Absolute coalition recovery remains about 0.80.

Read [RESEARCH_DECISION.md](RESEARCH_DECISION.md) for the conclusion and [COMPARISON_TABLE.md](COMPARISON_TABLE.md) for task/confidence/privacy numbers. [PRIMARY_ENDPOINTS.csv](PRIMARY_ENDPOINTS.csv) preserves all 552 registered slots. [RESULT_SUMMARY.json](RESULT_SUMMARY.json) and [COMPLETE_VIEW_RECOVERY.csv](COMPLETE_VIEW_RECOVERY.csv) include both seeds and their equal mean.

## Different ideas tested

- G: one optimized shared change rate for the exact frozen donor.
- C: shared rates per label-free context, with the SAME donor and fitting budget.
- N: a context-shared free kernel over KEEP plus narrow neighbor templates.
- W: a context-shared free kernel over the complete same-class frozen bank.
- Matched utility-objective, local, independent-product, trained deterministic and both freeze-first sequential controls; exact P0 and historical joint/sequential references.

C did improve the fitting objective over G, but one seed failed the original utility point gate. C has no fresh held-out privacy result. N and local controls also missed the paired gate. W preserved useful point predictions/confidence but did not establish the full bound conjunction. Its utility bounds are 9 passed, 15 unresolved. Four ORIGINAL-budget families were infeasible on the registered FIT law; their missing recovery results are dependencies, not privacy failures.

## What ran

Two paired model seeds on the same Adult development people: 15,434 fitting rows, 4,064 attack rows and 2,235 reporting rows. No new people or reserved assessment were opened. The workflow used 48 candidate/witness/feasibility LP calls, 59 matched-control calls including preserved failed attempts, 600 fresh reader/probe fits, 20 actual-law reports and 2,048 paired group bootstrap replicates. No teacher/head/calibration/partition was trained.

Read [PROTOCOL.md](PROTOCOL.md), [SCIENCE_LOCK.json](SCIENCE_LOCK.json), [ATTACKS_LOCK.json](ATTACKS_LOCK.json), [REPORTING_LOCK_V2.json](REPORTING_LOCK_V2.json) and dated amendments for the frozen design. [UNIT_MANIFEST.json](UNIT_MANIFEST.json) records completed, failed, superseded and untriggered work. The original failed attempts remain preserved.

## Runnable evidence

[QUICKSTART.md](QUICKSTART.md) gives the executed fixture and raw83 deployment commands. Public source contains code, fixed synthetic fixtures and exact environment versions. Private model/data artifacts are supplied through the authorized versioned restore index, not published person arrays. P0/G/C/W and controls have actual raw83 deployment checks; a runnable diagnostic does not mean its utility or privacy criteria pass.

[INDEPENDENT_VERIFICATION.json](INDEPENDENT_VERIFICATION.json) and [VALIDATION.md](VALIDATION.md) state actual replay scope, including zero independent reader refits. [MODEL_MANIFEST.json](MODEL_MANIFEST.json), [RESTORE_INDEX.json](RESTORE_INDEX.json) and [BACKUP_VERIFICATION.json](BACKUP_VERIFICATION.json) bind models and verified custody. Final [HANDOFF.json](HANDOFF.json) states completion and any remaining dependencies.

## Reading the claim

This is exploratory development on repeatedly used Adult people. A fresh fit or lock does not make them confirmation. Exact integration averages actual sampled outputs, not probability vectors; its intervals condition on fitted readers/kernels. C's point utility failure is not an established confidence-bound violation. The local control has one demonstrated occupation-loss violation. No new novelty, population privacy, repeated fresh-query protection or feature-sharing claim is made.

The next decision is to close this frozen-bank search and retain its verified sequential control/feasibility evidence for a separately authorized representation study.
