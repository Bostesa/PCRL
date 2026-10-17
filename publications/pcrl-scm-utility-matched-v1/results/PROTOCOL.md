# Protocol: quality-matched comparison of grouped prediction tables (SCM utility-matched follow-up)

**Status.** Registered before any new data-dependent work of this follow-up (no trajectory beyond the historical ones has been built, no MODEL_SELECT label has been read, no reader has been fitted). Machine-readable twin: `UNIT_INVENTORY.json`; both are hashed in `ANALYSIS_LOCK.json`.

**Nature.** An exploratory engineering comparison of existing methods on already-used Florida ACS 2019 development records. Shared-Cell Merging (SCM) applies established merging methods (privacy funnel, agglomerative information bottleneck, IBSI). No novelty, new-algorithm or joint-design claim is made. A favourable outcome would be an engineering result on previously used development records.

**History disclosure.** This design was informed by the published results of the SCM engineering benchmark (research branch `research/pcrl-scm-engineering-benchmark-v1`, recorded tip `2738cc30b`; public package `6fc06021d`). Those results showed that the historical nominee (SCM β = 1.0, 261 / 266 cells) read lower than its same-cell-count ordinary twin (pair AUC 0.618 / 0.622 against 0.658 / 0.660 on DEV_REPORT) but spent far more occupation log-loss (plug-in 0.0100 nats against 0.0027 / 0.0026), missed one DEV_REPORT utility point clause (occupation NLL against Ucal, seed 0, 0.0106 > 0.010) and exceeded the men's occupation Brier guard at the point. Its MODEL_SELECT values (including men's occupation Brier excess 0.0044 / 0.0059 and the tightest occupation NLL clause, 0.0067) are public and are known to the registrant. This registration fixes the execution order of the new work; it does not remove the history of development on these records. Those earlier results and labels are unchanged.

**Questions (kept separate).**
1. **Similar occupation NLL (primary).** At similar held-out occupation prediction loss, does SCM reduce demographic classification accuracy (fresh-reader recovery of the recorded SEX column) relative to ordinary task-only grouping?
2. **Same capacity.** Does SCM differ from controls with the same cell count and income-group structure?
3. **Overall benchmark.** How does the selected release compare with the eligible frozen neural (NEURAL_RAW) and tree (H_LJ) references?

Matching occupation NLL and matching cell count are different conditions, and matching one task loss is not matching the entire utility vector: every comparison displays the complete utility vector.

## 1. Source admission (completed before this registration; no label read)

`SOURCE_ADMISSION.json` (unit `UM_ADMIT_V1`) records:
- the source research tip (`2738cc30b`, matches) and the GitHub publication ref (`6fc06021d`, matches; `main` at `55e4cb1d1`, untouched);
- the 45 public results files compared with the sanitised local files: 41 identical; 3 report CSVs identical after CRLF→LF normalisation (the publication workflow's `read_text`/`write_text` round trip; content unchanged); `BACKUP_VERIFICATION.json` gained its `final_refresh` block on the research branch after the publication build;
- 12 private artifacts checked against their own manifests (TRAINING_COMPLETE_LOCK, SELECTION_LOCK, SERVICE_BANK_MANIFEST, REPORTING_LOCK), all matching;
- 51 hash-checked copies into `.private/pcrl_scm_utility_matched_v1/source/`: role exports (admitted fields only, no DEV_REPORT truth), the frozen comparator bank, and per seed the ten fold teachers, `PARTITION.npz` (out-of-fold task predictions, out-of-fold SEX propensity, the 512-cell fine partition and centroids, folds), `ARM_TABLES.npz`, `ARM_OUTPUTS.npz` and `FIT_SEED.json`, plus the frozen 145-column encoder record;
- **routing by hash:** the fold-average of the admitted teachers on each output role, KL-nearest admitted centroid, reproduces **all 31 historical arm outputs bitwise on all four output roles for both seeds**; the centroids equal the pooled out-of-fold cell means exactly. The routing arrays are stored and hashed (`routing/ROUTING_S{s}.npz`).

The fold SEX-propensity models were never persisted by the source; their out-of-fold MODEL_FIT outputs, which are all the merge engine uses, are admitted by hash. Nothing is retrained: teachers, the fine partition (512 occupied cells per seed), κ = 20 shrinkage, the 32-group income rule, the occupation tables, float64 representation and the out-of-fold (fitting) versus fold-ensemble (runtime) distinction are all inherited unchanged.

## 2. Roles, access and assumptions

| Role | Use |
|---|---|
| MODEL_FIT | Only through the admitted source statistics (teacher out-of-fold law, out-of-fold SEX propensity, SEX and occupation labels inside the within-SEX guard's fixed fine-cell law). No refit |
| CALIBRATION | Unused (U0 / Ucal are frozen references) |
| MODEL_SELECT | Utility clauses, subgroup guards, occupation-NLL expenditure, cross-fitted proxies, nomination, matching, comparator selection (features, task labels, SEX) |
| ATTACK_FIT (8,001) / ATTACK_VALIDATION (3,000) | Final evaluation readers only (fit; orient and select) |
| DEV_REPORT | Features and raw inputs for releases; truth re-read once by the coordinator and once by the verifier, logged, after the committed REPORTING_LOCK |
| SEALED_ASSESSMENT and other projects' reserved records | Closed |

**Service (controlled benchmark).** Two recipients. R1 receives the income probability vector (from one of at most 32 stored group vectors) and its argmax decision; R2 receives the occupation probability vector (one stored vector per final cell) and its argmax decision; nothing else is transmitted. The coalition view is the record-aligned pair. **Explicit experimental assumptions:** the readers' demographic-labelled auxiliary records (ATTACK_FIT) are available, linkage of the two outputs is accurate, and the auxiliary and target records are distributionally similar (same Florida ACS 2019 population). Deployment takes exactly the six permitted raw fields AGEP, WKHP, COW, SCHL, MAR, POBP through the frozen 145-column encoder.

**SEX access, described accurately.** SCM's merge priority uses the out-of-fold SEX propensity (a demographic-inference reduction term). The ordinary merge objective contains no demographic-inference reduction term, but its construction uses SEX through the common within-SEX occupation Brier guard (guard-only access, not SEX-blind; the guard enters both its admissibility and its merge order, because the priority normalises every budgeted cost including the two guard terms), and comparator selection (§7) uses the disclosed MODEL_SELECT demographic reader proxy. PUB_IBSI's objective uses the propensity. Shuffled controls keep the source procedure: initial-cell SEX rates permuted across initial cells, true-label guard (disclosed). Evaluation labels are never changed.

## 3. Fixed trajectories (per model seed s ∈ {0, 1}; cached once per configuration)

All trajectories start from the admitted C0 (the 512 unmerged occupied fine cells plus the 32-group income rule) and use the source engine `scm_bench/engine.py` unchanged (exact all-pairs greedy; ties to the lowest (a, b); grouped income costs in every capped admissibility check; 1e-15 admissibility slack and the other conventions of the source's post-run clarification 2).

| Trajectory | Rule | Stop |
|---|---|---|
| `SCM` | Source SCM, β = 1.0 budgets and priority | First step with no admissible pair (source rule) |
| `ORD` | Source task-only twin rule at β = 1.0: priority −Σ_k max(Δ_k, 0)/R_k (cheapest normalised merge), same β = 1.0 budgets including the within-SEX guard. **No target count**: it may continue past SCM's cell count while the source constraints permit | First step with no admissible pair |
| `IBSI_g0.5`, `IBSI_g1.0`, `IBSI_g2.0` | PUB_IBSI: an adaptation of the published agglomerative IBSI objective (Chechik and Tishby 2002), AIB-style greedy, minimise ΔNLL_income + ΔNLL_occupation − γ ΔI(T; SEX) (plug-in), no per-clause caps (source §5 and §12.6, unchanged) | 32 cells |
| `SHUF_r0` … `SHUF_r4` | Source shuffled-signal SCM (initial-cell SEX rates permuted with seed `seed_for('SCM1shuffle', s, r)`, true-label guard), β = 1.0 budgets | First step with no admissible pair |

**Recording.** A recorder re-implements the source `run_path` loop line for line and additionally records, after every merge m, the merged pair, the live cell count (512 − m) and the engine's cumulative plug-in spending (`spent`), in particular the fitted occupation-NLL increase over C0, **F_m** = the engine's running sum of the per-merge occupation-NLL changes (F_0 = 0). Prefix tables are then rebuilt by replaying the recorded merges once, in order.

SCM's prefixes are truncations of its single β = 1.0 path, not SCM runs at smaller budgets. An arm need not reach 32 cells to be eligible. No artificial prediction errors, unrelated decoders or extra noise are added to any control to make its loss resemble SCM's.

**Historical reproduction gate (before any additional prefix is admitted).** Per seed: (a) the recorder's final merge map, step count, stop reason, final cell count and all eight running costs equal (bitwise) the source `engine.run_path` run with the same arguments, the replayed terminal map equals the recorded one, and the replayed terminal tables equal the engine's own final-state tables bitwise; (b) the prefix at the historical cell count K_hist (261 / 266) reproduces the historical `ARM_TABLES.npz` tables bitwise (v1, v2, cell_of_initial, group_of_initial) for SCM_b1.0 (= `SCM`'s terminal), TWIN_b1.0 (`ORD`), IBSI_b1.0_g{0.5,1.0,2.0} (`IBSI_g*`) and SHUF_b1.0_r{0..4} (`SHUF_r*`); (c) their outputs on the four output roles equal the historical `ARM_OUTPUTS.npz` bitwise. Any failure writes a failed-gate record and halts the unit; no checkpoint is admitted until an engineering repair passes the gate.

**Trajectory lock.** After both seeds pass, `TRAJECTORY_COMPLETE_LOCK.json` is committed **before any MODEL_SELECT label is read**. It hashes each seed's `TRAJ_SEED.json`, which records every checkpoint label → (merge count m, F_m, effective-release hash), and the release tables. A later engineering repair must reproduce the locked label → release hashes; otherwise the affected role is recorded `REPAIR_CHANGED_RELEASE` and is not re-selected.

## 4. Checkpoints (registered families; nothing is added after MODEL_SELECT outcomes are seen)

For a trajectory with terminal step M:
- **NLL grid** `N{t}` for t = k/1000, k = 0, …, 10: the last prefix before F first exceeds the threshold, m*(t) = min{m : F_m > t + 1e-15} − 1 (or M if F never exceeds it); this first-crossing form makes float-level dips in F irrelevant. These fitted thresholds locate checkpoints; they are not guarantees about held-out loss. **Reached** iff the trajectory crossed t or ended at it (F_M ≥ t − 1e-15); otherwise the label aliases the terminal prefix and is recorded as **unreached (missing coverage of expenditure t)** together with the shortfall t − F_M (for example, SCM's terminal plug-in spend can sit a hair below 0.010 because another budget stopped it first).
- **Cell-count grid** `C{c}` for c ∈ {512, 448, 384, 320, 256, 224, 192, 160, 128, 96, 64, 32}: the prefix with exactly c live cells (m = 512 − c), when reached; unreached counts are recorded as missing coverage (no release).
- **Terminal** `TERM`: m = M.
- **SCM-count companions** `S{t}` (all comparator trajectories: ORD, IBSI_g*, SHUF_r*): the prefix with exactly SCM's cell count at SCM's `N{t}` checkpoint in the same seed, when reached. These pre-register the same-cell-count endpoints for every possible SCM nominee.
- **Historical** `HIST`: the prefix at K_hist (verification checkpoint; for SCM it is `TERM`).

**Effective releases and aliases.** A checkpoint's release is the function initial fine cell → (income vector, occupation vector). Two checkpoints with bit-identical per-initial-cell tables (v1[group_of_initial], v2[cell_of_initial], float64) are one **effective release**; every alias label is kept. Interpolating probability outputs from different partitions is never done (it could identify distinctions present in neither partition); lines joining plotted endpoints are illustrations only.

**Capacity, reported for every effective release:** nominal cells; occupied cells per role; distinct transmitted income, occupation and pair vectors per role; income groups in use.

## 5. MODEL_SELECT measurements (every effective release, both seeds)

- **Original utility clauses (24 per release-pair over both seeds; 12 per seed):** both tasks × {U0, Ucal} × {accuracy loss ≤ 0.010, NLL excess ≤ 0.010 nats, summed Brier excess ≤ 0.005}, inclusive (`scm_bench.selection.clauses`, unchanged).
- **Within-SEX occupation Brier guard:** occupation Brier excess against U0 within each recorded SEX group, ≤ 0.005 (point).
- **Occupation-NLL expenditure:** e_s(R) = mean occupation NLL of R − mean occupation NLL of the frozen C0 release, on MODEL_SELECT, seed s (per-person NLL −log max(p_y, 1e-12), the source convention). Because C0 is shared within a seed, e_s(R) − e_s(R′) is also the difference in absolute occupation NLL.
- **Cross-fitted pair-reader proxy:** the source rule unchanged (`scm_bench.select_scm.proxy`): MODEL_SELECT split into two household halves (seed `seed_for('SCM1proxysplit')`, same split for every release); OUT_LR, OUT_HGB and LOOKUP on the pair wire; fit on one half, score the other, both directions; orientation +1 (every reader estimates P(female)); reader proxy = mean of the two directions; release proxy = max over the three readers. Known bias (source §12.10): the max over readers biases upward similarly for every release, and LOOKUP under-reads fine partitions.
- **Worst normalised violation** of a condition set: max over its conditions of value / limit.

## 6. Conservative SCM endpoint (nomination; MODEL_SELECT only)

Candidates: SCM's NLL-grid checkpoints `N{t}`, t ∈ T = (0, 0.001, …, 0.010), **one common threshold for both seeds**. Every `N{t}` label is a candidate, including an unreached label (whose release is SCM's terminal prefix; its shortfall is reported).

**Nomination policy (prospective; does not revise earlier results).** t is eligible iff, on both seeds:
1. all 12 original point clauses pass (24 in total);
2. occupation NLL excess ≤ **0.008** against U0 and against Ucal (selection slack; the scientific acceptance limit remains 0.010);
3. both within-SEX occupation Brier point guards (SEX 0 and SEX 1) ≤ 0.005 against U0.

**Nominee:** among eligible t, minimise the mean over seeds of the release proxy; ties go to the smaller t (lower expenditure).

**Fallback:** if no t is eligible, the single t minimising the worst normalised violation over all 32 nomination conditions on both seeds (24 clauses at their caps, 4 occupation-NLL conditions at 0.008, 4 subgroup guards at 0.005), ties to the smaller t, is the **ineligible diagnostic fallback**. All specified diagnostic work continues with it; it can never be competitively eligible.

The nominee's cell count in seed s is K*_s; its expenditure is e*_s = e_s(SCM*). If the nominee's release equals C0 in a seed (no merge), claims M, K and B are labelled `NOMINEE_IS_C0` and no comparison is credited to SCM.

## 7. Explicit occupation-NLL matching and comparator selection (MODEL_SELECT only)

**Cost match.** A release R matches in seed s iff |e_s(R) − e*_s| ≤ **0.001** nats (inclusive). A MODEL_SELECT match does not guarantee a DEV_REPORT match (§10).

**Candidate set of a comparator trajectory in seed s:** the effective releases of all its registered checkpoints (`N{t}` including unreached aliases of `TERM`, reached `C{c}`, `TERM`, reached `S{t}`, `HIST`).

**Ordinary, occupation-NLL-matched (`ORD_NLLM`), per seed.** Eligible: candidates of `ORD` that match in seed s and pass that seed's 12 original point clauses. Choose the minimum of (proxy_s, |e_s − e*_s|, merge count m) — the strongest (most protective on the proxy) eligible ordinary control; ties toward the smaller mismatch, then the earlier prefix. If a seed has no eligible candidate, the matched ordinary comparison is **unavailable** in that seed.

**PUB_IBSI, occupation-NLL-matched (`IBSI_NLLM`), one common γ.** For each γ ∈ (0.5, 1.0, 2.0), per seed choose as for ORD within `IBSI_g{γ}`; γ is admissible iff both seeds have an eligible candidate. Choose the admissible γ minimising the mean over seeds of the chosen candidates' proxies; ties in grid order.

**Shuffled, occupation-NLL-matched (`SHUF_NLLM`), one common r.** As for IBSI over r ∈ (0, …, 4); ties in grid order.

**Unavailable matches.** A role with no valid cost match is recorded **unavailable**, and the fixed final roster holds, in that role, the **nearest eligible diagnostic**, labelled `NEAREST_DIAGNOSTIC_NOT_MATCHED`; its comparisons are descriptive and never called matched. If no candidate passes the clauses, the role is empty (MISSING).
- **ORD (per seed).** A seed without a valid match gets, in that seed only, the candidate passing its 12 point clauses with the smallest |e_s − e*_s|, ties by proxy then earlier prefix; the other seed keeps its matched pick. Claim M then reads `MATCH_UNAVAILABLE` (it requires both seeds).
- **IBSI and SHUF (common γ or r).** If no γ (r) is admissible in both seeds, then for each γ (r) each seed takes its nearest candidate as above, and the γ (r) with the smallest worst-seed gap max_s |e_s − e*_s| is chosen, ties by mean proxy then grid order.

**Same cell count (separately).**
- `ORD_CNT`: the `ORD` prefix with exactly K*_s cells (its `S{t*}` companion), included whatever its eligibility (recorded). If `ORD` never reaches K*_s, the same-capacity comparison is **unavailable** (`UNMATCHED_CAPACITY`).
- `IBSI_CNT`: the `IBSI_g{γ}` prefixes with exactly K*_s cells; γ by the source rule 6: among γ whose same-count releases pass all 24 original point clauses, the minimum mean proxy, ties in grid order; if none passes, the smallest worst normalised violation over the 24 clauses, labelled ineligible and shown descriptively.

**Selection effort** (number of distinct candidate releases examined per role and method) is reported for every role.

**Not matched by construction.** Income loss, occupation Brier and accuracy, and subgroup burdens can differ even when occupation NLL matches; the complete utility vector is displayed for every bank release. Beside every matched and same-count comparison the SCM − comparator differences in income and occupation NLL, Brier and accuracy are reported on MODEL_SELECT and DEV_REPORT, with the qualifier `INCOME_EXPENDITURE_NOT_MATCHED` when SCM's income NLL exceeds the comparator's by more than 0.001 (descriptive; labels unchanged).

**Matching coverage** is reported per comparator trajectory and seed: the sorted held-out expenditures of its candidates and the largest gap bracketing e*_s. No finer comparator grid is added (the checkpoint grid is fixed by the commission).

## 8. Final evaluation bank (per seed; at most ten releases)

Roles, in order: `SCM` (nominee or fallback), `ORD_NLLM`, `ORD_CNT`, `IBSI_NLLM`, `IBSI_CNT`, `SHUF_NLLM`, `NEURAL_RAW` (frozen ρ1 E2, eligible), `H_LJ` (frozen, eligible), `C0`, `HIST_SCM` (the earlier SCM β = 1.0 nominee, labelled *historically selected, diagnostic*). Roles sharing an effective release are one service (aliases recorded). Unavailable roles are recorded as such. Frozen comparators are admitted bitwise from the hash-checked frozen bank; U0 / Ucal are utility references, never attacked.

**Readers (source slate, fixed budget).** Per view (income recipient, occupation recipient, linked pair): OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP, DEC_LR, CONF_HGB and the corrected exact-vector LOOKUP reader (key = exact float64 bytes of the released vector(s); score (n_female + 2π)/(n + 2); unseen key → π; float64; orientation by ATTACK_VALIDATION like every reader). Pair candidates: 8 pair readers plus the 16 ignore-one alternatives that use only one output. Fit on ATTACK_FIT (8,001), orient and select on ATTACK_VALIDATION (3,000), ties to the earlier candidate, model seed `MODEL_SEED_BASE + s`, equal hyperparameter effort for every release. Readers are fitted freshly for every effective release (once per unique wire; aliases share). No extra reader family, no learning-curve search (the reader budget is fixed by the commission). **Disclosed bias:** at matched occupation NLL, SCM keeps more cells than an ordinary control, and a fixed 8,001-person reader budget can under-read finer releases (the source recorded 18 against 24 people per pair vector at the same count); this favours SCM and is reported beside every recovery comparison.

**Controls (source).** On the SCM release's wire, per view: shuffled-SEX null (fixed permutation of ATTACK_FIT SEX, seed 2026101602 + s; OUT_HGB) and planted SEX bit (OUT_HGB). Valid iff on all six views both planted AUCs (DEV_REPORT and ATTACK_VALIDATION) ≥ 0.99 and both null AUCs lie in [0.45, 0.55].

**Label-free replay.** Reader scores of releases also attacked by the source (historical SCM, C0, NEURAL_RAW, H_LJ) are compared with the source's stored DEV_REPORT scores (recorded only).

**Predefined independent reader refits (verifier).** The pair winners of SCM, ORD_NLLM, IBSI_NLLM and NEURAL_RAW on both seeds.

## 9. Reporting lock

Before any DEV_REPORT label is read, the coordinator commits `REPORTING_LOCK.json` (hashes of the selection lock, bank manifest, reader outputs and the frozen report code) and publishes a sanitised reporting receipt through the existing publication workflow (orphan publication branch, push to the GitHub remote, remote SHA verified). If public publication is unavailable a verifiable local receipt is kept and the limitation disclosed (a local-mirror push is not public registration). The analysis code that evaluates DEV_REPORT is frozen in that lock; it implements the rosters and claim rules registered here and in `UNIT_INVENTORY.json`.

## 10. DEV_REPORT evaluation (one coordinator evaluation, one logged verifier replay)

The coordinator's report unit re-reads DEV_REPORT truth once through the logged procedure and computes every endpoint, label, utility vector and diagnostic in that one run. The verifier replays once under a separately logged event. Neither result is used to choose another checkpoint, widen a tolerance, switch an endpoint or repair a result. **Crash contingency:** the report and verifier pipelines are first dry-run end to end on invented labels. If the coordinator's run still fails after its logged re-read and before writing any result, the failed attempt is preserved and one repair re-read is allowed under the separately logged event `DEV_REPORT_TRUTH_REREAD_BY_FOLLOWUP_CRASH_REPAIR` (enforced in code); no result of the failed attempt exists to inform it.

**Matching on DEV_REPORT.** For every intended cost-matched comparison the report gives the occupation-NLL difference (SCM − comparator) with its paired household-bootstrap interval, whether the point difference is within 0.001, the reader AUC difference and interval, and all other task and subgroup results. If the point difference drifts outside 0.001 the selection-role match **did not carry over**; this is stated, not repaired. A point match with a wide interval does not establish population equivalence.

## 11. Inference, endpoints and labels

**Inference (source conventions).** One household bootstrap over DEV_REPORT households (`task_headroom.inference.group_weights`), B = 2048, seed 2026101705 (the source's), shared across releases, endpoints and seeds; AUC replicates by `task_conditioned_representation.inference.bootstrap_auc`. Every replicate applies one household-weight vector to every release and endpoint; an AUC difference is the difference of the two weighted AUCs within a replicate; loss and subgroup differences are weighted means of person-level differences within a replicate. SE = replicate standard deviation (ddof 1). The report halts unless G = 3955 households and 5,943 persons (the source's DEV_REPORT). Wald bounds: point ± t(G − 1) × bootstrap SE. Claim clauses one-sided at **α/4** (α = 0.025; four claim slices M, K, B, C). Primary family: two-sided 95 % Bonferroni-t over **K_primary = 58**. Secondary: two-sided 95 % Bonferroni-t over **K_secondary = 24**. Critical values at G = 3955: clause 2.4988491; primary 3.3346075; secondary 3.0801279; marginal descriptive 1.9605641 (computed by the frozen code from the observed G). The frozen design of record is the block `FROZEN DESIGN` in `roles/statistical_reviewer/PREFIT_STATISTICAL_REVIEW.md`; it is authoritative where this text is ambiguous. Conditional on the fitted releases and readers; the two seeds evaluate the same people and are not separate populations; the intervals do not include the history of development and selection on these records; not ACS survey-design variance.

**Signs.** Benefit = comparator AUC − SCM AUC (positive favours SCM). Worsening = SCM AUC − comparator AUC per recipient (≤ 0.010 is non-worsening, the binding source definition). NLL match difference = SCM occupation NLL − comparator occupation NLL.

**Primary roster (per seed; 29 × 2 = 58).**

| Family | Endpoints per seed | Kind, margin |
|---|---|---|
| MATCH (claim M) | coalition benefit vs ORD_NLLM; income and occupation worsening vs ORD_NLLM | benefit > 0 (0.020 nested); worsening ≤ 0.010 |
| MATCH_COMPONENT (inside M) | coalition benefit vs IBSI_NLLM and vs SHUF_NLLM | benefit > 0 |
| CAPACITY (claim K) | coalition benefit vs ORD_CNT; income and occupation worsening vs ORD_CNT; coalition benefit vs IBSI_CNT | benefit > 0 (0.020 nested); worsening ≤ 0.010; benefit > 0 |
| COMPETITIVE (claim B) | coalition benefit vs NEURAL_RAW and vs H_LJ; income and occupation worsening vs each | benefit > 0 (0.020 nested); worsening ≤ 0.010 |
| UTILITY | SCM's 12 original clauses | ≤ caps |
| SUBGROUP (claim C) | SCM's within-SEX occupation Brier excess vs U0, SEX 0 and SEX 1 | ≤ 0.005 |

Every one of the 58 primary and 24 secondary keys is always reported and stays in its K. Each endpoint has one availability, fixed before DEV_REPORT from SELECTION_LOCK and the bank manifest: **AVAILABLE**; **DIAGNOSTIC_NOT_MATCHED** (the role holds its nearest diagnostic; per seed for ORD_NLLM, both seeds together for the common-γ / common-r roles); **COMPARATOR_INELIGIBLE** (ORD_CNT, NEURAL_RAW or H_LJ fails that seed's 12 MODEL_SELECT point clauses, or IBSI_CNT's γ is the ineligible fallback); **MISSING** (the role is empty: no candidate passes its clauses, or ORD never reaches K*_s). An endpoint takes the least available of its two roles. Endpoints that are not AVAILABLE are still computed whenever a release exists; they carry their availability as a status prefix, have null clause fields and never satisfy a claim condition. When both releases of a difference are one effective release in a seed, the endpoint is flagged `IDENTICAL_RELEASE` (point and SE exactly 0).

**Endpoint statuses** (family interval [L, U]): benefit — ESTABLISHED (L > m), ESTABLISHED_SHORTFALL (U < m), UNRESOLVED; no-excess — PASS (U ≤ m), ESTABLISHED_VIOLATION (L > m), UNRESOLVED; difference — ABOVE_ZERO, BELOW_ZERO, UNRESOLVED. Clause bounds (one-sided α/4) exist only for AVAILABLE benefit and no-excess endpoints.

**Occupation-NLL match endpoints (kind `match`; secondary family; their points are gates that spend no α).** Per person d_i = NLL_i(SCM) − NLL_i(comparator) with NLL_i = −log max(p_{i,y_i}, 1e-12) on the released occupation vector, over all DEV_REPORT persons; point = mean d_i; replicate b = Σ_i W_bi d_i / Σ_i W_bi. Point status: POINT_MATCH iff |point| ≤ 0.001 (inclusive, float64, no slack), else POINT_MISMATCH. Interval status: EQUIVALENCE_ESTABLISHED iff L ≥ −0.001 and U ≤ 0.001; MISMATCH_ESTABLISHED iff L > 0.001 or U < −0.001; else EQUIVALENCE_UNRESOLVED. No clause bound. Beside each endpoint the MODEL_SELECT mismatch actually achieved (SCM − comparator) is printed. "Matched" in any label means the DEV_REPORT point match only; POINT_MATCH with EQUIVALENCE_UNRESOLVED is written "point-matched; equivalence not established". Every seed's match also gets a direction sub-status, reported in every case: CARRIED_OVER, SCM_SPENT_MORE (> +0.001) or COMPARATOR_SPENT_MORE (< −0.001); it states only who spent more occupation NLL. A drift in either direction is MATCH_DID_NOT_CARRY_OVER (the tolerance is never widened); the benefit and worsening clauses are then still shown as a comparison-only status, labelled not cost-matched on DEV_REPORT, and never called matched.

**Failure categories.** Each claim reports every condition of its conjunction with its truth value; a false condition gets exactly one category: UNAVAILABLE (MISSING, DIAGNOSTIC_NOT_MATCHED, COMPARATOR_INELIGIBLE, UNMATCHED_CAPACITY, MATCH_UNAVAILABLE, NOMINEE_INELIGIBLE, NOMINEE_IS_C0 or NOT_TESTED_*); CONTROL_FAILURE; POINT_FAILURE (UG or SG at the point, or the DEV_REPORT match point); for a failed benefit clause (L1 ≤ m) ESTABLISHED_SHORTFALL if the family U < m, else POINT_SHORTFALL if point ≤ m, else UNRESOLVED; for a failed no-excess clause (U1 > m) ESTABLISHED_VIOLATION if the family L > m, else POINT_EXCESS if point > m, else UNRESOLVED. Text never merges these categories: an UNRESOLVED worsening clause is not evidence of worsening and an UNRESOLVED benefit clause is not evidence of no benefit.

**Common conditions.** CV: controls valid. NOM: the SCM release is a nominee (not the fallback). UG: SCM's 24 DEV_REPORT point clauses pass. SG: SCM's four DEV_REPORT within-SEX occupation Brier point guards pass (≤ 0.005).

**Claim M (matched ordinary; primary), first failing condition in this order:** CONTROL_FAILURE (¬CV) → NOMINEE_INELIGIBLE (¬NOM) → NOMINEE_IS_C0 (the nominee equals C0 in some seed) → MATCH_UNAVAILABLE (no MODEL_SELECT match in some seed) → MATCH_DID_NOT_CARRY_OVER (some seed's DEV point \|difference\| > 0.001) → NOMINEE_FAILS_DEV_UTILITY_GATE (¬UG) → NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD (¬SG) → **MATCHED_ADVANTAGE_ESTABLISHED** iff the benefit clause bound L1 > 0 on both seeds **and** all four worsening clause bounds U1 ≤ 0.010; otherwise MATCHED_ADVANTAGE_NOT_ESTABLISHED. Nested: MATERIAL_ADVANTAGE_ESTABLISHED iff established and L1 > 0.020 on both seeds (the original per-seed target). The comparison-only status (benefit and worsening clauses alone) is always reported beside the label.

**Matched components (inside M's α/4 slice; strict fixed sequence M → IBSI → SHUF).** IBSI is tested only if M is MATCHED_ADVANTAGE_ESTABLISHED; otherwise both are NOT_TESTED_M_NOT_ESTABLISHED and descriptive. IBSI: UNAVAILABLE (not MATCHED on both seeds), MATCH_DID_NOT_CARRY_OVER (some seed's DEV point |SCM − IBSI_NLLM| > 0.001), PASS (benefit L1 > 0 on both seeds) or FAIL. SHUF is tested only if IBSI is PASS (otherwise NOT_TESTED_SEQUENCE_STOPPED), with the same four statuses. COMPONENTS_ESTABLISHED iff both PASS; otherwise COMPONENTS_NOT_ESTABLISHED naming the first non-PASS. Shuffled controls are component comparisons, not permutation tests; a component PASS is about the linked pair at the achieved match, never a recipient-level or competitive result.

**Claim K (same capacity):** CONTROL_FAILURE → NOMINEE_INELIGIBLE → NOMINEE_IS_C0 → UNMATCHED_CAPACITY (ORD never reaches K*_s) → COMPARATOR_INELIGIBLE (ORD_CNT fails its seed's 12 MODEL_SELECT clauses) → NOMINEE_FAILS_DEV_UTILITY_GATE → NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD → **SAME_CAPACITY_ADVANTAGE_ESTABLISHED** iff benefit L1 > 0 both seeds and the four worsening U1 ≤ 0.010; else NOT_ESTABLISHED; nested material 0.020. IBSI_CNT is tested in sequence inside K only if K is established (COMPARATOR_INELIGIBLE if its γ was the ineligible fallback).

**Claim B (competitive):** CONTROL_FAILURE → NOMINEE_INELIGIBLE → NOMINEE_IS_C0 → NOMINEE_FAILS_DEV_UTILITY_GATE → NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD → **SUPERIOR_TO_ALL_ELIGIBLE** iff benefit L1 > 0 against NEURAL_RAW and H_LJ on both seeds **and** all eight recipient-worsening U1 ≤ 0.010; else NOT_SUPERIOR_TO_ALL_ELIGIBLE; nested MATERIAL_SUPERIORITY (L1 > 0.020, all four). Unfavourable utility, recipient or subgroup conditions are never disregarded.

**Claim C (subgroup):** SUBGROUP_QUALIFIED iff all four guard clause bounds U1 ≤ 0.005; SUBGROUP_HARM iff any family lower bound > 0.005; else SUBGROUP_UNRESOLVED. The point status is reported separately.

**Utility status (separate fields, always all reported):** (1) MODEL_SELECT original point eligibility of the bank's SCM release (ORIGINAL_POINT_FEASIBLE_24 or the failed keys; the 24 original clauses); (2) conservative nomination eligibility (NOMINATION_ELIGIBLE or INELIGIBLE_FALLBACK with the failed conditions); (3) DEV_REPORT point (POINT_FEASIBLE_24 or the failed keys); (4) DEV_REPORT simultaneous confidence on the K = 58 family (CONFIDENCE_FEASIBLE_24 iff every family upper bound ≤ cap; ESTABLISHED_UTILITY_VIOLATION iff any family lower bound > cap; else CONFIDENCE_UNRESOLVED; with counts); (5) subgroup point status and each guard's family status beside claim C. For every bank role the MODEL_SELECT and DEV_REPORT 12-clause point statuses per seed are also given; a comparator failing a DEV point clause is named as a qualifier beside each comparison, which does not change its eligibility (MODEL_SELECT-based, source convention). Claim C is reported as being about the ineligible diagnostic fallback when that is the SCM release.

**Per-seed 0.020 reporting.** Every benefit endpoint of M, K and B reports the point, the clause bound L1, the family interval, and whether L1 > 0 and L1 > 0.020 (the original per-seed target).

**Research decision (fixed mapping).** MATERIAL_ADVANTAGE_ESTABLISHED → ENGINEERING_REFERENCE_FROZEN_AT_ORIGINAL_TARGET; MATCHED_ADVANTAGE_ESTABLISHED but not material → ENGINEERING_REFERENCE_FROZEN_POSITIVE_BELOW_ORIGINAL_TARGET (the text states that the 0.020 target was not established); MATCH_UNAVAILABLE → FRONTIER_DELIVERED_MATCH_UNAVAILABLE (missing coverage is neither a demographic-inference result nor a method win); MATCH_DID_NOT_CARRY_OVER → COMPARISON_CLOSED_INCONCLUSIVE_AT_REGISTERED_TOLERANCE; CONTROL_FAILURE → COMPARISON_CLOSED_CONTROL_FAILURE; otherwise COMPARISON_CLOSED_<label>, with the measured limitation identified and no generalisation to every possible shared partition. Every decision is printed with four mandatory qualifiers: the DEV_REPORT utility confidence status, the claim C label, the DEV_REPORT equivalence status of the ORD_NLLM match per seed, and claim B's label. A release is called competitive only under SUPERIOR_TO_ALL_ELIGIBLE. A frozen engineering reference never authorises opening the sealed assessment.

**Descriptive companions (outside K, no status, no claim).** Beside every comparison (ORD_NLLM, ORD_CNT, IBSI_NLLM, IBSI_CNT, SHUF_NLLM, NEURAL_RAW, H_LJ, and C0 / HIST_SCM): paired SCM − comparator differences in income and occupation accuracy, NLL and Brier; within-SEX occupation Brier and NLL; and per-recipient and pair AUC differences, each with point, SE and a marginal 95 % interval labelled MARGINAL_DESCRIPTIVE. When M fails at MATCH_UNAVAILABLE or MATCH_DID_NOT_CARRY_OVER its comparison-only status is labelled "not cost-matched on DEV_REPORT".

**Secondary (K = 24; 12 per seed; no claim α):** pair-AUC differences C0 − SCM, C0 − ORD_NLLM, HIST_SCM − SCM and ORD_CNT − ORD_NLLM on the full 8-reader bank (24 pair candidates); the pair-AUC difference ORD_NLLM − SCM on the inherited 7-reader slate (7 pair readers and 14 ignore-one candidates); occupation-NLL differences SCM − ORD_CNT and SCM − IBSI_CNT; SCM pair AUC − SCM occupation-recipient AUC (full bank); the three occupation-NLL match endpoints SCM − ORD_NLLM, SCM − IBSI_NLLM and SCM − SHUF_NLLM (kind `match`; their points gate claim M and its components); and the income-NLL difference SCM − ORD_NLLM. Kind `difference` endpoints carry ABOVE_ZERO / BELOW_ZERO / UNRESOLVED on the K = 24 family interval and no clause fields.

**Levels.** P(label established | any null) ≤ α/4 unconditionally; it is not a guarantee conditional on the DEV_REPORT match having carried over. "Both seeds" is robustness to the model seed on the same people with the same bootstrap draws, not replication. The intervals exclude the selection of t, γ, r, the comparator and the readers, and the development history on these records.

## 12. Bounded diagnostics (computed inside the one coordinator evaluation; DEV_REPORT and MODEL_SELECT)

For every bank release: occupation NLL and Brier by recorded SEX group; classwise occupation NLL contributions and predicted-class frequencies against true class frequencies; fitted (plug-in, MODEL_FIT) merge costs against held-out costs on MODEL_SELECT and DEV_REPORT; lookup coverage and occupancy on the attack roles; the source's routing and teacher-prediction differences (out-of-fold against runtime). **Common-cell check:** per release and role, whether the occupation output identifies the shared cell (distinct occupation vectors = occupied final cells) and whether income is a function of it; if so, the information redundancy of the linked pair is explained separately from differences between fitted readers. The lookup reader is never described as population-optimal. These tables locate where prediction quality changes; they authorise no calibration, class-weight, capacity, teacher or merge-objective search.

## 13. Registered predictions (bets; scored after the report, never used for any decision)

| Id | Prediction | Probability |
|---|---|---|
| P1 | The historical nominee's threshold (t = 0.010) is not nominated (it fails the men's guard on MODEL_SELECT seed 1, 0.0059) | 0.95 |
| P2 | ORD_NLLM is available (MODEL_SELECT match) on both seeds | 0.70 |
| P3 | On MODEL_SELECT, proxy(ORD_NLLM) − proxy(SCM) ≥ 0.020 on both seeds (if available) | 0.70 |
| P4 | The DEV_REPORT point match vs ORD_NLLM carries over on both seeds (\|Δ\| ≤ 0.001) | 0.35 |
| P5 | DEV_REPORT coalition benefit vs ORD_NLLM point ≥ 0.020 on both seeds | 0.55 |
| P6 | SCM passes the 24 DEV_REPORT point clauses | 0.75 |
| P7 | SCM passes the four DEV_REPORT subgroup point guards | 0.50 |
| P8 | Claim M is MATCHED_ADVANTAGE_ESTABLISHED | 0.10 |

Scoring: each bet is HELD, NOT_HELD or VOID; P3 is VOID if P2 fails; P4 and P5 are NOT_HELD unless ORD_NLLM is MATCHED on both seeds; the mean Brier score over non-void bets is reported. P8 was revised from a draft 0.20 to 0.10 before the lock, after the statistical reviewer noted that its necessary conditions P4, P6 and P7 alone multiply to about 0.13 under independence.

## 14. Deployment, custody and publication

Raw-field packages for SCM and, when selected and available, ORD_NLLM and IBSI_NLLM (both seeds), plus the frozen NEURAL_RAW (seed 0 only; the source never built seed 1) and H_LJ (both seeds) packages because their actual source packages exist. Each accepts exactly the six permitted raw fields, reproduces the evaluated DEV_REPORT probabilities and decisions bitwise through both the package API and the CSV entry point, and refuses extra or reordered inputs, SEX as an input and requests for internal outputs. Backup restore-and-deploy from the backup alone; off-device copy on the external drive if writable (warm read-back distinguished from a cold read after remounting; no forced unmount; a same-host copy is never called off-device). Person-level arrays and fitted artifacts stay private. Public package: `publication/pcrl-scm-utility-matched-v1` via the existing sanitised workflow; `main`, manuscript and earlier branches are not changed and no history is rewritten.

## 15. Units, queue and budget

One shared queue (`scm_um/jobqueue.py`): 6 elapsed hours from 2026-10-17T04:38:00Z, 10 CPU-hours, zero paid or cloud compute, at most two heavy numerical jobs at once, one numerical thread per job. Verification, reader refits, failed attempts and stopped work are all charged; agent analysis is charged with upper estimates.

| Unit | Content | Estimate (CPU-h) |
|---|---|---|
| UM_ADMIT_V1 | admission and routing by hash (done, 0.003) | 0.01 |
| UM_TRAJ_S0, UM_TRAJ_S1 | 10 trajectories, reproduction gate, checkpoint tables and outputs | 0.25 each |
| UM_SELECT | MODEL_SELECT measurements for every effective release; nomination; matching | 0.3 |
| UM_BANK | final bank | 0.02 |
| UM_READERS_S0, UM_READERS_S1 | final readers and controls | 0.15 each |
| UM_REPORT | the one coordinator DEV_REPORT evaluation | 0.05 |
| UM_DEPLOY | packages, API and CSV reproduction, refusals | 0.05 |
| Verifier (recomputation, refits, replay) | | 0.5 |
| Backup and off-device check | | 0.05 |
| Agents and light scripts (charged) | | 1.5 |
| Failed-attempt margin | | 1.5 |
| **Total** | | **about 4.8** |

Planned elapsed: about 45 min admission and registration, 90 min trajectory bank, 90 min evaluation, the remainder verification and closeout (estimates, not waiting periods). The run finishes when the specified work is done; unused time is not filled.

## 16. Repairs and stopping

No early stop: one unfavourable starting release or checkpoint does not stop the other registered trajectories. Engineering repairs (code defects only) get a dated amendment, a new versioned unit id, retained failed attempts and reruns of affected checks; they never change thresholds, selected endpoints or comparator rules after results are seen. Locked files are never edited; post-lock clarifications go to a separate dated file.

## 17. What is new relative to the source (declared before fitting)

1. Single β = 1.0 trajectories with recorded prefixes replace the three-β bank; ORD, SHUF continue to their own source stopping rule and PUB_IBSI to 32 cells (the source stopped them at SCM's count).
2. Checkpoint families of §4 and effective-release deduplication.
3. The conservative nomination policy (occupation NLL ≤ 0.008 and both subgroup point guards) and the common-threshold rule.
4. Explicit occupation-NLL matching (0.001 nats per seed) with strongest-proxy comparator selection, nearest diagnostics and separate same-count roles.
5. A ten-role bank including the historical nominee; readers fitted once per unique wire.
6. Four claim slices (α/4) with K_primary = 58 and K_secondary = 24 (frozen by the statistical reviewer), explicit label conjunctions including recipient and subgroup conditions for every claim; the occupation-NLL match points are no-α gates.
7. A registered limit of one coordinator and one verifier DEV_REPORT re-read (plus the crash contingency), enforced in code.
8. A trajectory lock of every checkpoint label → release hash before any MODEL_SELECT label is read.
