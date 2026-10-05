# Online strength frontier and fixed recipient allocation (osf)

**Verdict: EXPERIMENTAL_NO_ADVANTAGE (complete).**
- Matching the incumbent's measured update strength with normalised training reproduced the raw trade-off rather than improving it.
- Fixed income/occupation allocations only moved leakage between recipients or traded it for occupation accuracy.
- Neither the normalised joint nominee nor the ordinary raw joint nominee had a feasible configuration on inner selection.
- The best development model is the incumbent, ordinary online raw joint training at β 0.3: pair SEX AUC 0.802 vs 0.883 for U, at 0.9 fewer occupation-accuracy points. It still permits substantial recovery.

**Reading order:**
1. `RESEARCH_DECISION.md` (answers and table)
2. `ADVISOR_BRIEF.md`
3. `figures/fig_frontier.pdf`
4. `VALIDATION.md`
5. `PROTOCOL.md`

**Evidence files:**

| File | Contents |
|---|---|
| `PRIMARY_ENDPOINTS.csv` | 27 slots; every clause DESCRIPTIVE_ONLY because no nominee was valid |
| `SECONDARY_ENDPOINTS.csv` | 61 declared contrasts |
| `ALL_LEVELS.csv` | Every grid point, seed, view and format |
| `ACTUAL_TASK_UTILITY.csv` | Accuracy, gain, gain retention, balanced accuracy, minority recall, Brier, log loss, ECE |
| `LINEAR_DIAGNOSTICS.csv` | Linear diagnostics |
| `STRENGTH_PROFILES.csv`, `STRENGTH_COMPARISON.json` | Per-step strength receipts summarised per run and epoch |
| `RAW_FIDELITY.json` | Parity, fidelity and bitwise replays |
| `CRITIC_TRACKING.csv` | Critic tracking |
| `INNER_FRONTIERS.csv`, `SELECTION_TABLE.csv`, `SELECTION.json` | Inner-only selection |
| `MODEL_MANIFEST.json` | Packaged units |
| `INDEPENDENT_VERIFICATION.json` | Independent replay |
| `BACKUP_VERIFICATION.json` | Backup and restore checks |
| `PREDECESSOR_CUSTODY_REPAIR.json` | smf custody repair (pending) |

**Locks:** `DATA_AND_ENGINEERING_LOCK.json`, `AMENDMENT_A1.json`, `AMENDMENT_A2.json`, `TRAINING_PROTOCOL_LOCK.json`, `SELECTION_AND_AUDIT_LOCK.json`, `EVALUATION_LOCK.json`.

**Data roles:**
- `ROLE_MANIFEST.json`, `ADMISSION.json`, `EXPOSURE_LEDGER.md` and `PRECISION_PLANNING.json`.
- The consolidated assessment reuses four previously used pools. It is not fresh data.

**Review and audit:** `MATH_REVIEW.md` (two REQUIRED defects found and repaired before their lock) and `AUDIT_PRELOCK_CHECKS.json`.

**Code:**
- `osf/` holds data, training, runner, locks, selection, families, inference, assessment, audit, baselines, deployment, reports and closeout.
- It pins `smf`, `rgj` and `jcv` helpers.
- How to run everything: `QUICKSTART.md`.

**Private material.** Models, labels, row keys and per-person predictions stay in `<PRIVATE_CACHE>/osf_v1` (not in the repository). A same-device copy is verified; the off-device backup is pending, because the external drive was not mounted.
