# Verification

Independent verification of campaign `pcrl_new_release_mechanisms_v1` by a separate verifier with its own numerical, metric, selection
and reader code (`roles/verifier/code/`), run through the shared queue. Full report, saved verbatim: `roles/verifier/FINAL_REPORT.md`;
interim reports: `roles/verifier/INTERIM_REPORT_1624.md`; every check: `roles/verifier/CHECKS.json` (253 append-only checks) and
`roles/verifier/records/`.

## Result

All decision-relevant records agree with the verifier's independent replay; there is no procedural FAIL and no blocking discrepancy.

| What was replayed | Outcome |
|---|---|
| Source admission, registration receipt, access policy | PASS (pin verified on the remote; 2,618 private copies and 201 mirror hashes) |
| Label-access audit (97 reads) | PASS: no read before the receipt, no closed role, MODEL_SELECT labels only through their entry points, cards committed before every fitting read; the reader-custody amendment committed before its single use |
| Utility of every evaluated release and reference | PASS within 3.6e-16 (values) and 8.7e-19 (SEs); three records carry stale draw-0 derived fields (see below) |
| Training and calibration exposure | PASS; twin units' MODEL_FIT SEX reads shown unused (SEX-permutation invariance, exact base wires) |
| Released outputs | PASS: valid float64 vectors, hashes equal the records |
| Registered grid NRM1_MAIN, CTL grid NRM1_CTL, robustness grids | Replay exact (maximum difference 0.0); locked grid code |
| Claims MAIN_V2, FINAL and FINAL_V2, and selection | 0 mismatches; NO_DEVELOPMENT_CANDIDATE |
| Decisive records (FB2P rule, F1 three times, MEAN rule twice) | All agree: FB2P NOT_RUN, F1_HOLDS, MEAN NOT_FITTED |
| Reserve-qualified frontier (TABLES.md section 4) | Agrees: FB2_SEX_S3 NO_RESERVE_FRONTIER_IMPROVEMENT |
| Deployment from the backup alone (own extraction, own probes) | 6 packages bitwise against the restored and the live releases through three entry points; all 21 refusals each (warm, same device) |
| Backup manifest | All four recorded file hashes re-hash equal |

## Non-PASS checks (all explained)

- **FAIL, records only (2):** FB_ISO_S2 and FB_POSTHOC_S2 evaluation records written before commit 705966524 carried point_pass and
  related fields from draw 0, flipping four per-clause flags; the decision fields replay exactly. Both are family-B v1 configurations
  that fail utility anyway.
- **FAIL, expected (1):** the superseded first claims run (`claims/MAIN/`, deviation 10) gives 91 mismatches; MAIN_V2 agrees exactly.
- **ATTENTION:** stale draw-0 `normalized` fields in old randomized records (fixed for new records in 999e216b0; no rule reads them);
  under the verifier's weaker own readers, FB2_SEX_S3 against H_LJ is a near tie in one cell (not a grid failure); twin units' unused
  MODEL_FIT SEX reads; a duplicated configuration number from concurrent saves (count unaffected).

## Not covered by the verifier

The verifier did not re-score the reader-custody pickles itself; the custody restore check did, under both interpreters
(`custody/BACKUP_VERIFICATION.json`). The off-device copy is pending (the drive was not attached).
