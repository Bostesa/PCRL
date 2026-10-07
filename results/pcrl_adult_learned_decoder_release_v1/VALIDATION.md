# Validation — lra

**Label:** CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION
[A=NOT_ESTABLISHED; B=NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE; C=NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE; Q=PASS].

## 1. Tests (synthetic data and toy laws only; no Adult labels)

- 284 tests pass under the semaphore before CORRECTNESS_LOCK (`lra.sema A:test-all-prelock`, about 149 s). Suites:
  - admit, run_gate, decoder, fixtures, deploy, mapper (26), audit, select, truth_table, late, closeout (41).
- 38 registered wiring and finding nodes (81 cases) are bound into ENGINEERING_GATE_RULE.json. They were run by the
  correctness stage (check E11/E12) and again externally by role E.

## 2. Correctness-only launch gate (prompt §9)

- **Rule.** ENGINEERING_GATE_RULE.json (fc074992…) holds the 12 checks. The fixture laws are the pinned lcr laws
  (24f70745… / 5c5e3bda…).
- **Run.** Under CORRECTNESS_LOCK (de4495f, pushed 05:07:48Z) at 05:07:54–05:08:58Z: ENGINEERING_READY
  (ENGINEERING_GATE_RESULT.json).
  - 2,043 accepted states were replayed from the persisted traces, with a worst difference of 1.5e-15.
  - The oracle tables are byte-identical to lcr's.
  - No paired move was accepted on the fixtures. The paired path is covered by registered synthetic tests, and later by
    99 real paired moves on Adult (§3).
- **Independent reproduction** (role E, phase 1; INDEPENDENT_VERIFICATION.json): all 12 checks reproduced.
  - 120 traces and 2,043 states, from E's own engine.
  - 1,921 certified vectors, from E's own solver.
  - Exhaustive enumeration.
  - The challenge harness on selection, fit feasibility, labels and assessment refusal: 12/12, 11/11, 10/10 and 5/5.
- **Registered numerical behaviour** (PROTOCOL.md, before SCIENCE_LOCK). The local cap is compared on exact float
  bits. On F3 this conservatively excluded some starts and witnesses at a true MI tie of 0. No violating state is ever
  accepted.

## 3. Science and inner selection (role E, phase 2; all PASS)

- **Exposure.** Every science hold and unit post-dates SCIENCE_LOCK (pushed 05:22:30Z). Unsealing exists only in
  lra.data / lra.assess. Every assessment unit post-dates EVALUATION_LOCK (pushed 07:47:30Z).
- **Decoder certificates.** 46,598 released vectors in 171 units, independently certified: FW gap ≤ 5.3e-16,
  stationarity ≤ 8.1e-16.
- **Releases.** Source parity and decision preservation hold, and dec__ tokens are bitwise equal to their D0 maps.
- **Budgets.** The deployed fitting budgets and local caps of all 15 constrained units hold at margin 0.
- **Trace replay.** Registered sample: all 15 K, 3 C-TASK and 24 weighted units (λ 0.025, 0.08). 185,057 accepted
  states, including 99 atomic paired moves; terms within 3.6e-15 and deltas within 5.1e-15.
- **Inner audits.** All 267 units rescored, max difference 3.3e-16. Utility matches exactly, and the composition
  winners and freeze lists are reproduced.
- **Selection.** Zero differences in status, config, fallback, reason, alias or representative.
- **EVALUATION_LOCK.** The bound selection, the 21-label scored list and the 37 endpoints are reproduced. The 3 d0s__
  units are token-identical to C-TASK D1.

## 4. Real-data controls (AUDIT_PRELOCK_CHECKS.json)

- all_ok.
- Null calibration has 0 exceedances; the null threshold matches the source.
- Every positive control is detected on every planted release: CONF, COLL and XOR on D0/D1/constrained codes, and ROT
  on the interfaces.

## 5. Assessment and inference

- **Assessment.** 63 outer units, run once under EVALUATION_LOCK 186afb6, 07:47:40–08:31:03Z.
- **Inference.** 37 primary slots on 1,999 draws (seed 20261010), z = 3.2048452050105634. All statistics are finite
  (finiteness receipt).
- **Independent check.** Role E's phase 3 recomputation of every endpoint, bound, conjunction, table and figure is
  recorded in INDEPENDENT_VERIFICATION.json (§7).

## 6. Deployment and custody

- **Deployment** (DEPLOYMENT_RECEIPT.json).
  - On seed 1, 5 releases deploy bitwise, including 2 learned-decoder releases: Q, P\*, C-TASK D1, K-SEQ-21 D1 and the
    D1 JOINT λ0.01 control.
  - 9 refusals exit with code 2 and write nothing: 84 columns, reordered columns, an extra sex array, a D1 map without
    its decoder, a mismatched decoder, a mismatched teacher, raw-score export, fine-ID export, and an unknown flag.
  - Both D1 releases also deploy from the same-device copy, bitwise.
- **Custody** (BACKUP_VERIFICATION.json, RESTORE_INDEX.json).
  - Status: LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING. The drive was absent, detected by content.
  - The copy holds 2,985 files, all re-read uncached and matching.
  - From the copy alone, these restore bitwise: the U and RAW-J teachers, 3 learned decoders (re-solved bitwise), P\*,
    Q, the constrained fallback, the decoder-only baseline, the sequential winner, and P\*'s selected attacker (refit,
    max difference 0).
  - The predecessor custody (cbp → qpc/dpc/osf/smf) stays PENDING because the drive is absent. The opening it waited for
    has happened.

## 7. Independent phase 3

The results are recorded in INDEPENDENT_VERIFICATION.json and summarized in COST_AND_CLOSEOUT.md / HANDOFF.json.

## 8. Disclosed deviations and corrections

1. **Port bug.** The lcr→lra rename broke the pinned laws' schema string. Role B caught it before any run and fixed it.
   The other schema comparisons were checked (PORT_LOG.md).
2. **CLASS|D1.** It was added as a fixed-map code (role F, R-1; prompt §10). That gives 81 D1 units, 84 codes and 267
   inner units, against the prompt's 78 / 83 / 264. It was registered before SCIENCE_LOCK.
3. **CLASS joint witness.** It was added as a feasible-only joint witness (prompt §8), giving 30 witnesses. Registered
   before the locks.
4. **R-4.** The mapper's pair-step screening overshot its evaluation share. This was fixed before the locks, and the
   fix is replayed by the verifier.
5. **Same-map D0 for P\*.** P\* is itself a D0 map, so only C-TASK needed same-map D0 units (3).
6. **Custody receipt field.** The attacker receipt's "rows_compared: 3" counts attacker seeds, not rows; the
   comparison covers the full prediction array.
7. **Evaluation-lock documents.** F's PREDECESSOR_GATE_DIAGNOSIS.md and PRIOR_ART_AND_CLAIM_SCOPE.md were committed at
   8b7e823, before CORRECTNESS_LOCK and SCIENCE_LOCK, and are unchanged since (verifiable in git). EVALUATION_LOCK binds
   the locks and the selection by hash, not those two documents.
