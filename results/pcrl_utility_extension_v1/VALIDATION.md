# VALIDATION — `pcrl_utility_extension_v1`

Every check below is machine-recorded; the file that carries it is named.

| check | result | evidence |
|---|---|---|
| Frozen A0 / J mappers re-execute from restored bytes | max abs difference 0.0 on this host's own arithmetic; tolerance 1e-5 | `_scheduler/tier0_smoke.json` |
| Stored probe reproduces its stored predictions and log loss | max abs probability difference 0.0, log-loss difference 5.6e-17; tolerance 1e-6 | `_scheduler/tier0_smoke.json` |
| Execution bundle restored and re-hashed on AWS | 2,848 files, 0 bad, stream hashes equal | `verification/exec_*.verify.json` |
| Archive chunks read back and re-hashed on AWS | 33 chunks, 164,927 files, 0 bad | `verification/*.verify.json` |
| H_A, H_B and Z_J byte-identical to the untouched J release | asserted per unit, every pool, before each release was written | `extension.assert_parity`, `seed_*/fits/*/fit_record.json` |
| R depends only on A-side inference inputs | by construction (PCA_32 through J's frozen standardiser); no label, id or H_B | `METHOD.md` section 1 |
| Release identity of the references | `ref_J`/`ref_A0` rebuilt from stored arrays and asserted equal wire-by-wire | `program.references` |
| Subset-index hashes match the historical study | asserted inside every audit | `run_dev_2018.evaluate_seed` |
| Probability matrices finite, in range, normalised | validated before any unit completed; 0 failures, 0 quarantines | `run_dev_2018.validate_predictions` |
| Routed-ancestor score check (Amendment 1) | 1,848 comparisons, **0 rejected**; realised max deviation log_loss 4.8e-9, auroc 1.5e-5 | `PORTABILITY_AMENDMENT_1.json` |
| Positive control: the slate finds A0's known signal | +0.032 A/SEX, +0.051 A/RAC1P over H (seed means, both weightings) | `_scheduler/gates/T1.json` |
| Calibration: is any new recipe stronger? | no (0 of 8); standard slate retained, added stress strength NOT established | `CALIBRATION.json` |
| Cross-platform reproduction of the HISTORICAL audits | **exceeded its 0.003-nat tolerance**: max 0.0031 (`ref_J`, seeds 1-2, both weightings); `ref_A0` below it | `CALIBRATION.json` |
| Fixture tests | 6 passed (zero-extension start, D1 = D0, inclusion fact, counts, selection tie-break, Amendment 1 acceptance/rejection) | `tests/pcrl_utility_extension_v1/` |

## The one check that did not pass, and what it means

Re-auditing the *same* channel on x86-64 does not reproduce the arm64 historical audit to better than
0.0031 nats, because the audit **refits** its attacker slate and that optimisation is platform
dependent. This is refit noise, not a defect in the release: the deterministic parts (channel arrays,
stored-probe predictions, routed-ancestor log losses) reproduce to 5e-9 or exactly.

It is declared as a limitation rather than absorbed: every comparison in this study is made against
references re-audited **on the same host under the identical slate** (`ref_J`, `ref_A0`,
`ref_leace_A0`), so no conclusion here depends on comparing an x86 audit to an arm64 one. The
practical consequence is that this study's numbers should not be differenced against the previous
study's published audit values at a precision finer than ~0.003 nats.
