# Team plan: confidence capacity and privacy (qpc)

| Item | Value |
|---|---|
| Branch | `research/pcrl-confidence-capacity-v1`, created at source evidence SHA `0a7b05a52746544213742f50efd0a48167efffb1` (dpc) |
| Teacher provenance | `925e0fddfcb666116c6179575339728a324ed78e` (osf) |
| Package | `qpc/` |
| Public evidence | `results/pcrl_confidence_capacity_v1/` |
| Private store | `<PRIVATE_CACHE>/qpc_v1/`, never committed. Public files use the placeholders `<PRIVATE_CACHE>`, `<WORKTREE>` and `<DRIVE_ROOT>` |
| Start | 2026-10-06T03:48:42Z (`<PRIVATE_CACHE>/qpc_v1/START.txt`) |
| Ceilings | 10 h elapsed (hard stop 13:48Z); 20 CPU-h; 8 GiB; ≥ 5 GiB free; $0 cloud |
| Reserve | 11:48Z–13:48Z and 4 CPU-h, for verification, packaging, reports and custody |

## Hard process rule: two heavy processes in total

The lead owns the shared semaphore `qpc/sema.py`.

**Every heavy process runs under it.** "Heavy" means:
- any real-data stage, attacker fit, verification replay or restore check;
- any synthetic timing or test run expected to exceed about 1 CPU-minute.

Use:

```
OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m qpc.sema --label <role:what> -- <command ...>
```

When invoking the wrapper by file path, pass `-P`: `~/PCRL/.venv/bin/python -P <WORKTREE>/qpc/sema.py --label ... -- ...`. Without `-P`, `qpc/select.py` shadows the stdlib `select` module and the wrapper fails before taking a slot. The `-m qpc.sema` form with `PYTHONPATH=.` is unaffected.

The wrapper blocks until one of the two slots is free. **Two study workers plus a verifier is three, and is forbidden.** The semaphore makes that impossible, provided everyone uses it.

Use one BLAS/OpenMP thread per process (`OMP_NUM_THREADS=1`; torch threads 1). Light unit tests and document work need no slot.

## Who may touch real data

- **Only the lead (A)** starts real-data stages: admit, stagea, gate, partition, fit, inner, controls, select and assess. It does so through `qpc.run` or `qpc.assess` after the governing lock is pushed.
- **Agents** develop and test on synthetic fixtures. Nobody reads assessment labels.
- **The data/custody owner (E)** may load the sealed roles to compute role manifests and source hashes, but must not run any fit or attacker.
- **Review agents** must not run reference audits, browse masked labels or launch fits.

## Roles and exclusive file ownership

| Role | Owns |
|---|---|
| A. Lead | `qpc/sema.py`, `lock.py`, `run.py`, `gate.py`, `select.py`, `family.py`, `infer.py`, `eval_lock.py`, `report.py`, `qpc/tests/test_pipeline.py`, `test_late.py`, `test_truth_table.py`; `PROTOCOL.md`, `PREDICTIONS.json`, `FIT_MANIFEST.json`, `PRIMARY_FAMILY.json`, `LABEL_TRUTH_TABLE.json`, locks/amendments, decision documents |
| B. Compressor | `qpc/kmeans.py`, `stagea.py`, `partition.py`, `compress.py`, `release.py`, `deploy.py`, `qpc/tests/test_method.py`; `METHOD_CARD.md`, `TIMING.json` (fitting part) |
| C. Math/protocol review | `MATH_REVIEW.md`, `PRIOR_ART_AND_BASELINE_GAPS.md`, `qpc/tests/test_math_review.py` |
| D. Attackers/metrics | `qpc/audit.py`, `utility.py`, `baselines.py`, `assess.py`, `qpc/tests/test_audit.py`; `AUDIT_COMPUTE.json` |
| E. Data/custody | `qpc/data.py`, `admit.py`, `closeout.py`, `qpc/tests/test_admit.py`, `test_closeout.py`; `SOURCE_ADMISSION.json`, `ROLE_MANIFEST.json`, `SOURCE_INDEX.json`, `EXPOSURE_LEDGER.md`, `provenance/` |
| F. Independent verifier | `results/pcrl_confidence_capacity_v1/verification/replay_qpc.py`, `INDEPENDENT_VERIFICATION.json` |

**Rules for files and git:**
- Edit only files you own. Ask the lead, by message, for changes elsewhere.
- Nobody but the lead commits or pushes.
- Reuse dpc code by importing it, since dpc is pinned at the source SHA in this tree, or copy it into qpc with a provenance note. Never edit `dpc/`.

## Fixed definitions and interfaces

**Data.** `qpc.data.load()` returns the pinned osf/dpc D unchanged, with assessment labels masked to −1.
- Roles: OSF_DEFENSE_FIT 15,434; HEAD_VALIDATION 1,500; AUDIT_FIT 6,065; INNER_SELECTION 2,235; OSF_DEVELOPMENT_ASSESSMENT 13,936 rows / 13,929 groups.
- Index: `D["idx"][role]`. The dpc key for OSF_DEFENSE_FIT is `"DEFENSE_FIT"`; see dpc/run.py.
- Unsealing: `qpc.data.load(unseal=True)` succeeds only when called from `qpc.assess` and the committed EVALUATION_LOCK is on origin.

**Teacher units.** `<PRIVATE_CACHE>/qpc_v1/run/units/tea__s{k}__U` and `tea__s{k}__RAW-J_b0.3`.
- File `teacher.npz` holds `row_id`, `p1` (n, 2), `p2` (n, 6), `d1`, `d2`, plus `c1`, `c2`, `r1`, `r2` as in dpc.
- They are rebuilt by qpc's own forward application from the hash-pinned admitted artifacts, with bitwise parity against the dpc teacher units.

**Reference units.** `ref__s{k}__{E,F,F0}` with `reference.npz`, admitted by verified copy and parity.

**Release format** (dpc-compatible, per mapping-pair unit):

| File | Contents |
|---|---|
| `policy.json` | The deployable policy pair |
| `release.npz` | `row_id`, `tok1`, `q1`, `hard1`, `alpha1`, `tok2`, `q2`, `hard2`, `alpha2` for ALL rows |

The token is categorical: one token per coarse cell, with canonical IDs. `q` is the decoded smoothed prototype, and `hard` is the teacher decision.

**Configuration IDs:**
- `U|DIRECT-TASK|i{m1}o{m2}` (Stage A)
- `U|FINE-TASK|i{m1}o{m2}`
- `U|{LOCAL,SEQ-12,SEQ-21,JOINT}|i{m1}o{m2}|l{lam}`
- `U|CLASS|i1o1`
- `SRC|U`, `SRC|RAW-J_b0.3`
- `REF|E`, `REF|F`, `REF|F0`

**Smoothing, logs, ties and the clip** follow dpc exactly: eps 1e-12; float64; natural logs; first-index ties; log-loss clip 1e-12.

**Stage A** (k-means per teacher-predicted class, directly on DEFENSE_FIT rows):
- **A1:** reproduce `dpc.partition.fit_fine(..., max_cells=8, rounds=20)` per recipient, with parity against the admitted dpc `pol__s{k}__U_DIRECT-TASK_m8` release. Then refit the same source initialization with at most 200 rounds and the convergence rule.
- **A2:** income m1 ∈ {4, 8}; occupation m2 ∈ {8, 16, 32, 64}; seeds 0–2. Three starts (source deterministic; KL k-means++ seeds 20261006 and 20261007); at most 200 rounds. The lowest DEFENSE_FIT KL distortion per class wins.
- **Fit caching:** per-recipient fits are cached (income at 2 rates, occupation at 4) and assembled into 8 rate pairs.
- **A3:** utility on INNER_SELECTION versus U, using D's `qpc.utility`.
- **Gate:** the lead's `qpc.gate`.

**Stage B** (only if the gate passes):
- **Fine partitions:** income ≤ 32 and occupation ≤ 128 per predicted class, with the same starts and convergence rules.
- **Families:** FINE-TASK, LOCAL, SEQ-12, SEQ-21 and JOINT at the selected rates, with λ ∈ {0.01, 0.1, 1}.
- **Search:** at most the cap, including objective-improving extra merges.
- **Sequential correction:** stage one is fitted against the other recipient's CLASS-ONLY map under the actual F_joint.
- **JOINT:** starts from the five witnesses, with 5 sweeps each.

**Attacks.**
- **Slate:** the FINAL slate from smf.audit, plus defence-aware readers and finite-code cell readers at α ∈ {0.5, 1, 5}.
- **Roles:** fitted on AUDIT_FIT, selected on INNER_SELECTION. AUC and cross-entropy selection are kept separate.
- **Composed source bank:** U continuous-source banks include composition with EVERY fitted code of this study, via the cached code-reader candidates, assembled before selection.

## Timeline targets (UTC; the lead may move them)

| Target | Lock and what it covers | Stages it opens |
|---|---|---|
| 04:45 | **SOURCE_ADMISSION_LOCK** (E: data/admit; lead: run.py) | admit |
| 05:15 | **STAGE_A_LOCK** (B: kmeans/stagea; D: utility; lead: gate, predictions; C: Stage A review) | stagea → gate |
| 06:30 | **STAGE_B_LOCK** (B: partition/compress, synthetic-tested and timed) | Stage B fits |
| 07:30 | **AUDIT_AND_SELECTION_LOCK** (D: audit; lead: select/family/infer; C: review) | inner audits, controls, select |
| about 09:00 | **EVALUATION_LOCK** | the single assessment → inference → reports |
| 10:00–11:30 | Verification (F), using semaphore slots | |
| 11:30–12:30 | Packaging, backup, closeout | |

If the Stage A gate fails, Stage B and the assessment do not run, and the label is CAPACITY_GATE_NOT_MET.
