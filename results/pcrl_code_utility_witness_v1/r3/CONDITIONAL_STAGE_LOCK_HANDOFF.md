# R3 protection stage lock handoff

R0 is the only writer of the effective stage lock. The worker source is committed at `3d37bdeb8`; its SHA-256 is `53327d84523de5a798eae1ea95b143ea0c9fa3be7cd1c69d749f524ac909dc7c`. No real protection fit has run. Use this handoff only after the learned T0 point gate passes at one sealed capacity on both seeds/tasks and R5 checks the gate.

The worker accepts a public committed lock with schema `pcrl-code-utility-protection-lock-v1`, status `LOCALLY_SEALED_PROTECTION`, and these exact fields:

| Field | Required value/source |
| --- | --- |
| `source_lock_sha256` | active R0 queue source seal byte hash |
| `gate_receipt_sha256` | committed R0 capacity-gate receipt byte hash |
| `selected_bits` | 10 or 16, identical to gate |
| `seeds` | `[0,1]` |
| `arms` | ordered `TPLUS_Bb,L_Bb,J_Bb,S12_Bb,S21_Bb,C_Bb` for selected `b` |
| `lambda_rule` | `FIRST_SHARED_FIT_BATCH_GRADIENT_RATIO_V1` |
| `continuation_epochs_by_seed` | positive selected additional FIT epochs, keyed by string seed |
| `continuation_batch_seed_offset` | one fixed integer shared by all arms |
| `role_manifest_sha256` | admitted role-only manifest bytes |
| `source_model_sha256_by_seed` | pinned frozen source model.pt bytes |
| `t0_manifest_sha256_by_seed` | immutable T0 service manifest bytes |
| `t0_scaler_sha256_by_seed` | same scaler identity recorded in R0 gate |
| `source_files_sha256` | exact relative source-file hash mapping, including worker, R3 kernel/protection, R1 model/learner/service, role reader, frozen encoder, queue, `common_eval.py`, and its sibling `safe_data_guard.py` |

The gate receipt must have schema `pcrl-code-witness-capacity-gate-v1`, status `POINT_GATE_PASS`, `explicit_r0_go:true`, `all_two_seed_task_reference_point_caps_pass:true`, matching `source_lock_sha256`/`selected_bits`, and `t0_by_seed` entries containing each service `manifest_sha256`, model-state `model_sha256` and `scaler_sha256`. The worker loads T0 only after checking the active queue and local-HEAD committed gate/lock bytes. Its real FIT guard requires those identities, the selected bit count, both seed IDs and R0 go. An uncommitted lock, mismatched source or direct CLI launch fails before role access.

Register one heavy queue unit per seed at stage `protection`. Each unit spec must bind `stage_lock_sha256`, `gate_receipt_sha256` and integer `seed`; the queue wrapper supplies its existing slot environment. The worker's CLI also requires `--unit-id`, `--stage-lock`, `--stage-lock-sha256`, `--gate-receipt`, `--gate-receipt-sha256`, `--role-manifest`, `--source-model`, `--t0-manifest`, `--private-output`, `--public-receipt`, and `--seed`. Private output must be the fresh package `.private/protection_s{seed}`; public receipt must be `r3/PROTECTION_S{seed}_RECEIPT.json`. A failed attempt is preserved. Any replacement must use a dated amendment and a changed output identity before a rerun.

The worker fits T+, L, J, S12, S21 and C from the same cloned full-FIT T0 state. It checks the restored scaler's canonical SHA against both R0 seals and the T0 package's 83-column source schema SHA against the admitted role manifest. New packages carry the same source schema pin. It computes the seed-specific FIT-only gradient ratio on the first shared 256-person full-FIT batch, records its row-order hash and gradient norms, and makes no update during that diagnostic. It hashes the entire ordered batch plan, including epoch and batch boundaries, in the public fit receipt. It checks the queue again before each fit. Each sequential first stage is calibrated on admitted CAL and frozen before the second starts. It writes six owner-only service packages and ATT/INNER NPZ releases with exact R4 fields: `row_id,unit,code0,code1,prob0,prob1,decision0,decision1`. The public receipt has only aggregate fit history, hashes, occupancy, package identities and resource linkage; private paths stay in an owner-only index.

The planned two-seed continuation is **16 mapper/head stage fits**: per seed one each T+, L, J, C and two each S12/S21. C adds two critic steps per mapper step. There are **24 new task-global calibrations** across six arm IDs, two tasks and two seeds, with sequential first-stage calibrations performed before their second-stage fits. R4 attacker and R5 replay units must be separately counted. One stage unit per seed includes all six arm fits and releases so R0 must obtain a queue-charged role-sized smoke CPU/wall forecast before admitting both units. The 5 CPU-hour protection reservation and global 20 CPU-hour ceiling apply. These counts are not runtime measurements.
