# Amendment items: original versus executed settings

Role 1 input to AMENDMENT_1.md. Written 2026-10-02, before any real-data attacker or probe fit.

**Abbreviations:**
- "Original" means the preparation record at 031860f: `PILOT_LOCK.json` (lock), `PILOT_PROTOCOL.md` (PP),
  `protocol_config.json` (CFG), `ATTACKER_ACCESS_TABLE.csv` (AT), `run_pilot_cell_a.sh` (SH), and the code.
- "Executed" means FROZEN_DESIGN.md (FD).

## Has any assessment outcome been seen?

**Answer, from the evidence:** no new assessment-role outcome has been seen for any of the 26 units. The
preparation ran only:
- frozen forward passes (`forward_smoke_result.json`, then the full test split into `cache/adult_s0_test.npz`);
- admission of the 152 manifests;
- `fit-attackers --dry-run` (`pilot_dryrun_counts.json`: `fits_performed: 0`);
- synthetic fixtures and tests;
- recounts of stored historical values.

`HANDOFF.json:90` records "No scientific fit has been performed". The admission outputs are row and class
counts only. Role 1 has also not opened `run_v1/` or any predictions.

**Prior exposure (not blinding, and to be stated as such).** These artifacts are development data already
seen:

| Source | What was seen |
|---|---|
| `origin/main:results/v2_adult_ROUND4/dominant_axis_audit.json` `per_seed["0"].rows[*].r2_onehot` | Historical N0-type values for all 8 pairs on this checkpoint and these 15,060 test rows |
| `final_vs_best.md` | The same values as a table (for example s0 final income_prediction/race = 0.058, above τ) |
| `per_seed_results.json` | EmpiricalAudit held-out max-suite accuracy, train→test, on this encoder (HL-03) |
| AAAI 67-config recount (`recount_aaai67_*.json`) | durable-guarantees XGB/MLP attack AUCs on this same encoder, including the noise σ arms on income/sex, on dg rows (the PCRL train partition) |

The AAAI recount is the closest prior look at P2-type and noise-arm recovery, though on other rows and with
other attackers.

**Consequences:**
- The noise grid, the cell and the bars were all chosen knowing those results.
- The pilot is development evidence and is **not blinded**.
- The 16-endpoint family is fixed to all 8 untreated pairs regardless of the known N0, so it is not
  conditioned on native pass.

## A. Executable-contract changes introduced by FD

For every row, "results seen?" means "any assessment outcome seen?", and the answer is **No** (prior
exposure only, as above).

| # | Item | Original (preparation) | Executed (FD) |
|---|---|---|---|
| A1 | **Attacker slate wiring** | SH calls `fit-attackers` without `--attackers`. The CLI defaults to linear, gbt, mlp (`cli.py:177`); R09–R12 are defined but unwired. PP §5 slate: R00–R13 incl. R06N/R07N/R08/R11. | L, CAND-GBT, CAND-MLP and NL-selected per surface; A2 LRT (R10) and A4 LRT (R09) on noise rep; R12 staged; **R11 deferred** (absent from FD and must be stated); R13/R14/R15 NA. |
| A2 | L grid | CFG/AT R01: C ∈ {0.01, 0.1, 1, 10, 100}, `max_iter` 5000. Effective at pin: 5 values, `max_iter` 2000. | FD: C ∈ {0.01, 0.1, 1, 10}. **Conflict: resolve and record.** |
| A3 | NL grids | CFG R03: 20 random configs from lr × leaves × min_samples_leaf × l2, `max_iter` 500, early stopping, 3 seeds. R04: 18 configs incl. `learning_rate_init`, 3 seeds. Pin: 9 + 9, GBT without early stopping. | "As frozen in EFFECTIVE_PROTOCOL.json" (not yet written). Every dropped key must be listed. |
| A4 | GBT vs MLP selection | Not implemented. PP §13 P2 = "best of R03/R04" (and a selection in report would use assessment). | One NL-selected predictor per unit and surface, chosen on attacker_val log-loss. Assessment never selects. |
| A5 | Output and combined recipes | R06N = GBT only (10 configs). R07N = nested 38-config slate including R03/R04/R06N. | NL-selected (GBT vs MLP) on outputs and on rep+outputs. No nesting rule stated. |
| A6 | Noise contract | Global `release_contract.noise="none"` (`config.py:45`) stamped on every access record. Manifest `release` block unread. | Per-unit contract from the manifest: Gaussian, σ_abs, one persistent draw per row per seed. The FD's literal `ReleaseContract(noise="gaussian", persistent=True)` does not fit the pinned API (`access.py:26`); map it to `persistent_token` or amend the class. |
| A7 | A4 LRT form | `NoiseLRTAttacker` = point-mass mixture over clean vectors (`attackers.py:175-186`). | Class-conditional Gaussians from clean attacker_fit reps + σ²I (matches AT R09). New implementation or amendment. |
| A8 | A2 LRT | Absent. | Gaussian class-conditional LRT on released rows, covariance − σ²I, eigen floor 1e-6·tr/d. |
| A9 | **Native naming and the R02/G1/G2 split** | One R² family: "native" = fit = score on assessment rows (`pipeline.py:74`); R02 (rho = 0) = the P1 endpoint (PP §13, CFG multiplicity P1). | N0 = historical check (all 15,060 test rows; float64 + mixed float32). N1 = within-assessment descriptive. **G1** = held-out fixed ridge 1e-6, fit-mean SS_tot, unclamped = **primary P1**. G2 = R02 secondary. Plus held-out ρ₁² and the pure-metric row. |
| A10 | R02 rho grid | CFG/AT: rho ∈ {0, 1e-4, 1e-2}, selected on attacker_val. Pin: rho = 0. | FD silent. Freeze either the grid with val selection or rho = 0. |
| A11 | **Label-only, U1, U2** | PP §8/§14: R05 = multinomial LR on one-hot y + frequency table. U2 = LR + MLP. Utility metrics incl. macro-F1, normalised-lift and frontier rules. None implemented. | LO = frequency table P(s\|y_task) on attacker_fit (smoothing to freeze). LO contrast = outputs NL-selected − LO, paired. U1 = frozen head (noise: head(h+ε)). U2 = LR only, all 26 units. MLP probe, macro-F1, lift and frontier rules are not in FD: list them as dropped or kept. |
| A12 | **Task labels: new inputs, manifests and lock** | `labels.npz` has no occupation_group or education_level (only income as a sensitive key). The lock pins 152 manifests built on it. | Add `income`, `occupation_group` and `education_level` from PCRL b96c412 `task_labels`. New manifest array `task_labels` → new manifest hashes → new lock. **Recommendation:** write a separate `task_labels.npz` instead of rewriting `labels.npz`. `np.savez` zip entries carry write timestamps, so a rewrite changes the `labels.npz` sha even for identical content and breaks the provenance chain to the admitted arrays. |
| A13 | Recovery metrics | CFG `logloss_reduction` = LL_prior − LL_model (nats; `metrics.py:519-525`). | FD: `1 − LL/LL_prior`. **Definition conflict:** choose one (or report both, named). |
| A14 | **Multiplicity** | PP §13 / CFG: ≤ 3 endpoints (P1 R02 > τ; P2 max(R03, R04) > 0.55; P3 R06N − R05 > 0) over all native-pass (cell, arm) combinations, including noise arms. Holm with bootstrap p-values (share of replicates beyond the bar). | 16 frozen endpoints P1-/P2-<pair> on the 8 untreated pairs only, **not** conditioned on native pass. Bonferroni simultaneous one-sided bounds, α = 0.003125 each, percentile B = 20,000, seed 20261003, resolution 5e-5. **P3 demoted** to secondary. Noise arms secondary. Asymptotic validity note. **Quantile interpolation rule must be frozen** (tail count 62.5). |
| A15 | Exploratory intervals | Two-sided 90 %, B = 2,000, seed 20261002. | Unchanged. Noise arms are seed-averaged within each replicate, with spread reported separately. |
| A16 | **Support freezing across roles** | PP §10/§18: 100/100 + 30 in val. Code: assessment-only (`metrics.py:292`, `:344-353`); admission applies 100 in every role. | Frozen `supported.json` per unit from all 3 roles before inference. Pairs need both classes supported. NE when fewer than 2 classes. Coverage and reason codes. NE kept distinct from UNRESOLVED. |
| A17 | Decomposition | CFG/PP §7: F0 native → F1 R02 R² → F2 R01 AUC → F3 NL Brier skill → F4 NL AUC → F5 out → F6 rep+out → F7 adaptive. | F0 N0 → F1 G1 → F2 pure metric (G1 predictor as AUC) → F3 L AUC → F4 NL-selected AUC rep → F5 outputs NL → F6 rep+outputs NL. F7 dropped. Brier-skill step dropped from the path (still reported as a metric). |
| A18 | Refit and permutation controls | PP §12: 3 attacker seeds per selected config (refit SD); 200-permutation null with the same max rule. PP §16: per-cell positive control (untreated beats R05) and null control (permuted s); the cell stops if a control fails. | **Absent from FD.** The prompt replaces the circular positive control with a synthetic direct signal. List each as executed, dropped or staged. |
| A19 | τ grid | {0.01, 0.02, 0.05, 0.10}, with 0.05 primary. | FD names only 0.05. State whether the sensitivity τ decisions are reported. |
| A20 | **Lock contents** | Lock v1: code_commit fbdb35a; sha of PP.md, CFG and AT; 152 manifest shas; panel; "decisions" string. CFG `exposure.lock_contents` also asks for per-role row manifests, declared output objects, permission table and primary family; these are missing. | New lock (the original is preserved byte-identical). Covers: effective protocol, consumed code file shas at the new commit, AT, the exact 26 unit IDs (from 152), data / label / task-label / fwd / release / checkpoint shas, supported-class freeze, primary family (16 IDs, α, B, seed, quantile rule), declared output object (logit vector per purpose), permission table (b96c412 disallowed lists). |
| A21 | **Runner path** | `SH:7` hard-codes the preparation worktree. The lock check covers only CFG and manifests, and only in the shell. | Root derived (or validated `--root`), and the branch asserted. The lock is checked inside the CLI before any `--execute-scientific-fits` work. Expected-vs-actual unit IDs compared. Consumed-code and manifest changes block execution. |
| A22 | Output locations | `$C/scores/<manifest>/scores.npz`. | `~/PCRL_eval_cache_private/pilot_adult_s0/run_v1/units/<unit_id>/{preds.npz, fit_records.json, models/, supported.json}`. |
| A23 | Role-split description | CFG `roles.stratify ["s","y"]`, `split_seed 20261002`. | Actual (§18 and lock): unstratified record-key hash 'pilot-roles-v1'. CFG text is stale; supersede it in EFFECTIVE_PROTOCOL. |
| A24 | Noise-unit N0 | n/a. | NA ("unverified on these rows"). The dg noise approval (card 5) was on dg rows. |

## B. Stale statements to correct in the new branch's copy only

The historical branch `research/combined-evaluation-preparation-v1` must not be touched.

**Caution: editing hashed preparation files changes their sha256.** The lock pins `protocol_md_sha256 095aca43…`
and `protocol_config_sha256 6dae304f…`.
- Record the original shas in AMENDMENT_1, verifiable at 031860f.
- Pin the corrected copies in the new lock.
- Keep `PILOT_LOCK.json` itself unedited.

| File:line (at 031860f) | Stale statement | Correction |
|---|---|---|
| `notes/methodology/protocol_config.json:2-3` | `"version": "pilot-protocol-draft-v1…"`, `"status": "DRAFT - not locked; no real-data … fitting until approved and locked"` | "Locked v1 (PILOT_LOCK.json, 2026-10-02); superseded for execution by EFFECTIVE_PROTOCOL.json + AMENDMENT_1" |
| `PILOT_PROTOCOL.md:3` | "Version. Draft v1…" | Locked v1 plus pointer to the amendment |
| `PILOT_PROTOCOL.md:8-9` | "Nothing in this protocol has been run." / "needs approval before it happens." | Execution authorised under AMENDMENT_1; results in `results/combined_stored_model_pilot_v1/` |
| `PILOT_PROTOCOL.md:63` | CELL-B "**Eligible by rule**, inputs not yet prepared" | CELL-B does **not** meet rule 4 (it has no projection arm; PP:63 itself lists "no projection"). It is not part of this run. |
| `ADVISOR_BRIEF.md:57` | "CELL-B … It is eligible but its inputs are not yet prepared." | Same correction |
| `ADVISOR_BRIEF.md:3-4` | "Nothing scientific was fitted in this preparation." | True of the preparation. Add: "the pilot run is in results/combined_stored_model_pilot_v1/" |
| `ADVISOR_BRIEF.md:94-95` | "probe … prepared, but not run" | Pointer to the pilot results |
| `ADVISOR_BRIEF.md:74`, `:82` | "one-sided 95 % bound … B = 2,000" as *the* decision rule; "Held-out linear R² (R02)" as the check | Primary = Bonferroni bounds on G1/NL-selected (A14). R02 = G2, secondary (A9). |
| `ADVISOR_BRIEF.md:78` | "Support … 100 rows in attacker-fit and in assessment" | Add "and 30 in attacker_val" |
| `ADVISOR_BRIEF.md:129-131` | Run command `EXECUTE=1 sh …run_pilot_cell_a.sh`; "About 1 CPU-hour" | Replace with the repaired runner and the measured or staged budget |
| `QUICKSTART.md:4`, `:16` | `cd …/combined-evaluation-preparation-v1`; "26 passed" | New worktree root; 31 (+ new) tests |
| `QUICKSTART.md:29` | "Prepared scientific pilot (NOT executed; needs approval and the protocol lock)" | Executed under AMENDMENT_1 |
| `QUICKSTART.md:126` | E1 CELL-B as the next extension | Keep as an extension, not in this run, with no eligibility claim |
| `HANDOFF.json:3`, `:81`, `:90` | "no scientific … fits"; "CELL-B HMDA input preparation"; "No scientific fit has been performed." | Historical record of the preparation. Leave it, or annotate "superseded by pilot HANDOFF". It becomes stale once fitting starts. |
| `notes/evaluator/run_pilot_cell_a.sh:3`, `:7`, `:15-25` | "no scientific fits" default; hard-coded old WT; partial lock check | Replace with the repaired runner. Keep the old script as a record, or mark it superseded. |
| `GUARANTEE_CARDS.md` (card 2 "Numerical tolerance") | "On origin/main the Gram is computed in float32" | Precise form: float32 centring and Gram, float64 ridge, solve and one-hot (`verification.py:89-103`). Matters for the N0 replica. |
| `stored_model_eval/access.py:3` | Cites `EVALUATION_PROTOCOL_DRAFT.md` | Cite PP / EFFECTIVE_PROTOCOL |
| `cli.py:106`, `plan.py:85` | `"fits_performed": 0` | Correct for dry-run and plan only. Make sure the real run reports actual counts. |

Any statement that CELL-B is in this run: none found in the package. PP:58-64 lists it as a slot only.
FD and the prompt both exclude it. State the exclusion in EXECUTED_PROTOCOL.
