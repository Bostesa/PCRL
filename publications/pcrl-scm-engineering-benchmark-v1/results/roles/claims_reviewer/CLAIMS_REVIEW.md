# Claims and closeout review: SCM engineering benchmark

**Reviewer.** Claims and closeout reviewer, 2026-10-17 (UTC). Brief: `roles/coordinator/CLAIMS_REVIEWER_BRIEF.md`.

**Documents reviewed.**
- `DECISION.md`, `RESULTS_TABLES.md`, `PROTOCOL.md` §13 and `ENGINEERING_REPAIRS.md`, as of the coordinator's follow-up commit `11d54c649`.
- `START_HERE.md`, unchanged since `04facb7fe`.
- `HANDOFF.json` does not exist yet and was not reviewed.

**Checked against.**
- `report/DEV_REPORT_RESULTS.json` and the three `report/*.csv` files.
- `SELECTION_LOCK.json`, `FIT_SUMMARY.json`, `READERS_SUMMARY.json` and `ANALYSIS_LOCK.json`.
- `PROTOCOL.md` and `ENGINEERING_REPAIRS.md`.
- `DEPLOYMENT_RESULTS.json` and `DEPLOYMENT_RECHECK.json`.
- `roles/verifier/VERIFICATION.md` and its check files.
- The frozen code (`scm_bench/report.py`, `acs_replication/schema.py`) and git history, read-only.

**Method.** Every quoted number was recomputed with small Python scripts over the JSON and CSV records. No private array was opened, and no git command changed state.

## What reproduces

- **`RESULTS_TABLES.md`.** All rows match the records at the displayed precision:
  - §1: 17 rows against SELECTION_LOCK, including recomputed worst-normalised values, feasibility, proxies (max over three readers) and the F1 check 0.0356 / 0.0372;
  - §2: 14 rows;
  - §3: 42 rows, including clause bounds (L1 for benefit endpoints, U1 for the rest) and family intervals;
  - §5: 14 rows against READERS_SUMMARY.
- **CSVs.** The three CSVs equal the JSON exactly.
- **Endpoints.** All 52 endpoint points (42 primary, 10 secondary) were recomputed from the per-seed AUCs, utilities and subgroup values, and all 52 sets of bounds from SE × the recorded critical values. The 42 primary statuses and clause flags were recomputed from the margins. There were 0 mismatches.
- **Claim labels.** All match the frozen `report.py` rules (A, attribution, B, C and the separate status fields).
- **Selection.** Reproduces: SCM β = 1.0 nominee (mean proxy 0.6095 against 0.6253 for β 0.5), IBSI γ = 1.0, SHUF r2. β = 1.5 fails only occupation NLL (0.0106–0.0118).
- **SEX coding.** sex0 = male and sex1 = female (`acs_replication/schema.py` SEX_CODE), so the "men" figures are labelled correctly.
- **Lock hashes.** ANALYSIS_LOCK, SELECTION_LOCK, REPORTING_LOCK and SERVICE_BANK_MANIFEST match the hashes recorded downstream. PROTOCOL.md does not (MF-7).
- **Receipts.** `f0a3b94e9` was pushed at 03:08:29Z, before the first fit attempt (03:08:3x). `bd31bcde0` was pushed at 03:20:10Z, before the DEV_REPORT re-read (03:20:19Z). Both times come from the remote-tracking reflog.
- **Math review.** The five math-review must-fix items (M1–M5) are all reflected in PROTOCOL §7, §9 and §12 and in `report.py`.
- **Scope language is right where it appears:**
  - existing-method engineering benchmark, no novelty claim;
  - the lookup reader is "not a population Bayes oracle";
  - finite-cell merging "does not certify protection against arbitrary readers";
  - failure scoped to this recipe and capacity.

---

## MUST-FIX

### MF-1. `DECISION.md` opening: "Every recovery comparison favoured the nominee" is false

Per recipient, on DEV_REPORT:
- SCM's income recipient reads higher than NEURAL_RAW's (+0.018 / +0.015) and higher than H_LJ's (+0.006 / +0.017).
- SCM's occupation recipient reads higher than H_LJ's (+0.041 / +0.030).

The statement is true only for the linked pair, and only at the point. The opening also attributes all formal failures to the utility clause, but claim C is unresolved for its own reason.

**Find (line 17):**
> Every recovery comparison favoured the nominee. The formal claims fail because the nominee **misses one DEV_REPORT utility point clause**: occupation NLL excess against Ucal on seed 0 is 0.0106, against a 0.010 cap.

**Replace with:**
> Every linked-pair recovery comparison favoured the nominee at the point. These comparisons are descriptive, because the nominee is not DEV-utility-eligible. Per recipient they did not all favour it:
> - SCM's income recipient read higher than NEURAL_RAW's (+0.018 / +0.015) and H_LJ's (+0.006 / +0.017);
> - its occupation recipient read higher than H_LJ's (+0.041 / +0.030).
>
> Claims A and B fail, and attribution is not tested, because the nominee **misses one DEV_REPORT utility point clause**: occupation NLL excess against Ucal on seed 0 is 0.0106, against a 0.010 cap. Claim C is unresolved for its own reason: men's guard clause bounds (0.0111 / 0.0127) exceed 0.005.

**Find (§4, "Per recipient" bullet):**
> - **Per recipient.** SCM's occupation recipient reads lower than NEURAL_RAW's (−0.021 / −0.010). Its income recipient reads higher (+0.018 / +0.015, unresolved).

**Replace with:**
> - **Per recipient, against NEURAL_RAW.** SCM's occupation recipient reads lower (−0.021 / −0.010). The secondary family interval is below 0 on seed 0 and unresolved on seed 1. Its income recipient reads higher (+0.018 / +0.015, unresolved).
> - **Per recipient, against H_LJ.** SCM reads higher on both recipients. Income is 0.564 / 0.565 against 0.558 / 0.549, and occupation is 0.601 / 0.613 against 0.560 / 0.583. These are descriptive and not endpoints.

### MF-2. `DECISION.md` line 3: "DEV_REPORT truth was re-read once" is no longer true

VERIFICATION.md check 8 records two logged events in both access logs:
- the coordinator's `DEV_REPORT_TRUTH_REREAD_BY_FOLLOWUP`, at 03:20:19Z;
- the verifier's `DEV_REPORT_TRUTH_REREAD_BY_INDEPENDENT_VERIFIER`, at 03:29:39Z.

The verifier also kept a temporary crash copy outside the repository and deleted it at 03:30Z.

**Find:**
> The sealed assessment stayed closed, and DEV_REPORT truth was re-read once, logged, after the committed reporting lock.

**Replace with:**
> The sealed assessment stayed closed. DEV_REPORT truth was re-read twice, each time after the committed reporting lock and logged in both access logs:
> - once for this report (`DEV_REPORT_TRUTH_REREAD_BY_FOLLOWUP`, 03:20:19Z);
> - once by the independent verifier (`DEV_REPORT_TRUTH_REREAD_BY_INDEPENDENT_VERIFIER`, 03:29:39Z; `roles/verifier/VERIFICATION.md`, check 8).
>
> The verifier's temporary crash copy of the re-read arrays, kept outside the repository, was deleted at 03:30Z.

### MF-3. `DECISION.md` §2: "no clause has an upper bound below its cap" is false in the document's own terms

DECISION uses "clause bound" for the one-sided α/3 bound (see §3 and §4). At that level, six of the 24 utility clauses have upper bounds below their caps:
- income NLL against U0, 0.0081 / 0.0087;
- occupation accuracy loss against U0 and against Ucal, 0.0085 / 0.0093.

`report.py` defines confidence status on the family-wise (K = 42) interval. That is why all 24 are UNRESOLVED.

**Find:**
> - Confidence status for all 24 clauses is `UNRESOLVED`: no clause has an upper bound below its cap, and none is an established violation.

**Replace with:**
> - Confidence status for all 24 clauses is `UNRESOLVED`: no clause's family-wise (K = 42) upper bound is below its cap, and no family-wise lower bound is above it. Confidence status is defined on that family interval. At the one-sided α/3 clause level, six upper bounds are below their caps: income NLL against U0 (0.0081 / 0.0087), and occupation accuracy loss against U0 and against Ucal (0.0085 / 0.0093).

### MF-4. `DECISION.md` §3 overstates the margins and states untested attribution as findings

**The opener is false for one of the section's own comparisons.** It says the margins "would meet the 0.020 recipe target". Against the published IBSI objective the margin is 0.014 / 0.014, with family lower bounds 0.003 / 0.001.

**The bullets assert conclusions that the frozen labels do not allow** (`RECIPE_EFFECT_NOT_ESTABLISHED`, `NOT_TESTED_RECIPE_NOT_ESTABLISHED`):
- "The reduction comes from which cells the SEX signal chose to merge, not from coarsening";
- "so it is specific to the SEX signal rather than a by-product…";
- "beat".

**They also omit that the comparisons are matched on cell count, not on utility.** SCM spent 0.0100 nats of plug-in occupation NLL. The twin spent 0.0027 / 0.0026, IBSI 0.0064 / 0.0066 and SHUF 0.0039 / 0.0038. SCM is also the only release in the bank that fails a DEV point clause.

**Find (opener):**
> Descriptively, yes, and by margins that would meet the 0.020 recipe target. Formally it is not established, because the DEV utility gate failed. Pair AUC on DEV_REPORT, seed 0 / seed 1:

**Replace with:**
> Descriptively, SCM read lower than every comparator in this section. Only against the matched twin is the margin beyond the 0.020 recipe target at the clause bound. Against the published IBSI objective it is 0.014 / 0.014, with family lower bounds of 0.003 / 0.001.
>
> Formally nothing is established: claim A failed the DEV utility gate, so attribution was not tested. The comparisons are matched on cell count, not on utility. SCM spent more plug-in occupation NLL than every comparator, and it is the only release here that fails a DEV point clause. Pair AUC on DEV_REPORT, seed 0 / seed 1:

**Find (the four bullets after the table):**
> - **Ordinary compression alone did almost nothing.** The task-only twin at the same cell count is 0.0035 / 0.0030 below C0, unresolved. The reduction comes from which cells the SEX signal chose to merge, not from coarsening.
> - **The SEX-guided choice beat merging with permuted SEX rates** by 0.038 / 0.036, so it is specific to the SEX signal rather than a by-product of departing from task-only merging.
> - **SCM's constrained merge beat the published IBSI objective** by 0.014 / 0.014 at the same capacity.
> - All of these are descriptive here. Attribution is tested only when claim A is established, and A failed the utility gate.

**Replace with:**
> - **Task-only merging alone changed little.** The matched twin, which uses SEX only through the within-SEX guard, is 0.0035 / 0.0030 below C0 (unresolved).
> - **SCM against the twin.** At the same cell count, SCM is 0.041 / 0.038 below the twin, but it also spent far more utility. Its plug-in occupation NLL was 0.0100 nats against the twin's 0.0027 / 0.0026, and it fails a DEV point clause that the twin passes. The record therefore does not separate the effect of the SEX-guided merge order from that of the extra utility spent.
> - **Against merging with permuted SEX rates** (the shuffled control, which keeps the true-label guard and spent 0.0039 / 0.0038 nats), SCM read lower by 0.038 / 0.036.
> - **Against the published IBSI objective** at the same capacity (0.0064 / 0.0066 nats), SCM read lower by 0.014 / 0.014.
> - All of these are descriptive. Attribution is tested only when claim A is established. A failed the utility gate, so none of these is an attribution result.

### MF-5. `DECISION.md` §6: a misquoted number and two conclusions the records contradict

**The DEV seed-1 number is wrong.** The held-out occupation-NLL increase over C0 on DEV_REPORT seed 1 is 1.2392579 − 1.2300300 = 0.0092, not 0.0093.

**"Did not overstate" is false.** The plug-in spend was 0.0100 nats (0.009999 / 0.010000), and DEV seed 1 came in below it, so the plug-in overstated there.

**"Absorbed most" is false except on MODEL_SELECT seed 0.** There, C0's headroom against U0 / Ucal was 0.0071 / 0.0062 nats of the 0.0106 increase, 67% / 59%. The shares elsewhere are:

| Role and seed | Share absorbed (U0 / Ucal) |
|---|---|
| MODEL_SELECT seed 1 | 40% / 40% |
| DEV_REPORT seed 0 | 34% / 21% |
| DEV_REPORT seed 1 | 14% / 11% |

**Find:**
> - On held-out people the occupation-NLL increase relative to C0 was 0.0106 / 0.0111 on MODEL_SELECT and 0.0134 / 0.0093 on DEV_REPORT. So the fitted plug-in costs did not overstate, and on DEV seed 0 understated, the held-out cost.
> - C0's NLL headroom over the references absorbed most of the cost, not all of it.

**Replace with:**
> - On held-out people the occupation-NLL increase relative to C0 was 0.0106 / 0.0111 on MODEL_SELECT and 0.0134 / 0.0092 on DEV_REPORT, against the plug-in 0.0100. The plug-in cost understated the held-out cost on MODEL_SELECT and on DEV_REPORT seed 0, and overstated it on DEV_REPORT seed 1. It is not a bound on the held-out cost.
> - C0's occupation-NLL headroom below the references absorbed most of the increase only on MODEL_SELECT seed 0 (0.0071 / 0.0062 nats against U0 / Ucal; 67% / 59%). Elsewhere it absorbed a minority: 40% on MODEL_SELECT seed 1, 34% / 21% on DEV_REPORT seed 0 and 14% / 11% on DEV_REPORT seed 1.

### MF-6. `START_HERE.md`, "What the evidence supports": three errors in four sentences

- **Rounding.** "Changed recovery by only 0.003" misrounds seed 0: C0 − TWIN is 0.0035 / 0.0030. It also does not say which direction.
- **Untested attribution.** "The reduction came from SEX-guided merges" asserts the content of claim A, labelled `RECIPE_EFFECT_NOT_ESTABLISHED`. It sits directly under the line saying attribution was not tested.
- **A false held-out statement.** "With held-out costs equal or larger" is false for DEV seed 1 (0.0092 < 0.0100).
- **An incomplete bottleneck.** The bottleneck sentence leaves out that the grouped income-Brier budget was also exhausted.

**Find:**
> **What the evidence supports.** Ordinary compression to the same cell count changed recovery by only 0.003. The reduction came from SEX-guided merges, and it was paid for in occupation log-loss: the full plug-in budget, with held-out costs equal or larger. That is the bottleneck at these caps.

**Replace with:**
> **What the evidence supports (descriptive).** Task-only merging to the same cell count lowered pair AUC by only 0.0035 / 0.0030 relative to C0 (unresolved). This is the matched twin, which uses SEX only through the within-SEX guard. SCM read 0.041 / 0.038 below the twin.
>
> That comparison is matched on cell count, not on utility. SCM spent its whole plug-in occupation-NLL budget (0.0100 nats, against the twin's 0.0027 / 0.0026) and its whole grouped income-Brier budget (0.005). On held-out people its occupation-NLL increase over C0 was 0.0106 / 0.0111 on MODEL_SELECT and 0.0134 / 0.0092 on DEV_REPORT.
>
> Occupation NLL is the only clause any SCM setting breached: β = 1.5 on MODEL_SELECT and β = 1.0 on DEV_REPORT seed 0. It is therefore the measured held-out bottleneck at these caps. Whether the reduction is due to the SEX-guided merge order was not tested.

### MF-7. `PROTOCOL.md` was edited after the run, so the locked protocol hash no longer matches

Commit `11d54c649` appended §13 to `PROTOCOL.md`. Its sha256 is now `70f537d8…`. ANALYSIS_LOCK records `protocol_sha256` = `d19d364d…`.

The locked bytes still exist, both with that hash:
- at `e17b9e999` (the ANALYSIS_LOCK commit);
- in the public pre-fit receipt `f0a3b94e9`.

Nothing above §13 changed, but no document discloses the hash change. ENGINEERING_REPAIRS lists the code-lock deltas but not this one. START_HERE item 3 still says the protocol was "frozen before any real fit". Anyone who re-checks the lock will now find a mismatch on the frozen protocol.

**Fix (preferred).**
1. Restore the locked bytes: `git show e17b9e999:results/pcrl_scm_engineering_benchmark_v1/PROTOCOL.md > results/pcrl_scm_engineering_benchmark_v1/PROTOCOL.md`, then check that sha256 = `d19d364d…`.
2. Move the §13 text verbatim into a new file, `PROTOCOL_POSTRUN_CLARIFICATIONS.md`, with this heading and first line:
   > # Post-run clarifications to PROTOCOL.md (2026-10-17, from the independent verifier; no number or decision changes)
   >
   > `PROTOCOL.md` is unchanged and matches ANALYSIS_LOCK `protocol_sha256` (`d19d364d…`). These clarifications are recorded here so the frozen protocol keeps its locked bytes.
3. In `DECISION.md` §4, find `(verifier D2; \`PROTOCOL.md\` §13)` and replace it with `(verifier D2; \`PROTOCOL_POSTRUN_CLARIFICATIONS.md\` item 1)`.
4. In `START_HERE.md`, find:
   > 3. `PROTOCOL.md` and `ANALYSIS_LOCK.json`: frozen before any real fit. The public pre-fit receipt is `f0a3b94e9` and the reporting-lock receipt is `bd31bcde0`.

   Replace with:
   > 3. `PROTOCOL.md` and `ANALYSIS_LOCK.json`: frozen before any real fit (PROTOCOL sha256 `d19d364d…`, as locked). The public pre-fit receipt is `f0a3b94e9` and the reporting-lock receipt is `bd31bcde0`. `PROTOCOL_POSTRUN_CLARIFICATIONS.md` holds the verifier's post-run clarifications; no number or decision changed.

**Fallback (if §13 stays in place).** Add this row to `ENGINEERING_REPAIRS.md`:
> | 2026-10-17 03:3x | No failure; post-run documentation | Verifier D2, D3, D4 and D6 | §13 appended to `PROTOCOL.md`. Its sha256 (`70f537d8…`) therefore differs from ANALYSIS_LOCK (`d19d364d…`). The locked bytes are at `e17b9e999` and in the public pre-fit receipt `f0a3b94e9`. Nothing above §13 changed | None |

Then replace START_HERE item 3 with:
> 3. `PROTOCOL.md` and `ANALYSIS_LOCK.json`: frozen before any real fit. §13 was appended after the run, so the current file's hash differs from the lock; the locked version is at `e17b9e999` and in the public pre-fit receipt `f0a3b94e9`. The reporting-lock receipt is `bd31bcde0`.

### MF-8. `START_HERE.md` items 6–8 are out of date after `11d54c649`

ENGINEERING_REPAIRS now has five rows: a post-report deployment schema repair, a docstring change and lock-coverage notes, besides the original two. Three locked modules now differ from their ANALYSIS_LOCK hashes, as I recomputed: `common.py`, `selection.py` and `deploy_scm.py`. `DEPLOYMENT_RECHECK.json` exists, and the verifier finished with 8/8 PASS.

**Find:**
> 6. `DEPLOYMENT_RESULTS.json` and `QUICKSTART.md`: raw-field packages for SCM and TWIN (both seeds) and the seed-0 neural reference, each reproducing the attacked outputs bitwise.
> 7. `ENGINEERING_REPAIRS.md`: one environment failure, preserved and rerun, and one logged-access addition.

**Replace with:**
> 6. `DEPLOYMENT_RESULTS.json`, `DEPLOYMENT_RECHECK.json` and `QUICKSTART.md`: raw-field packages for SCM and TWIN (both seeds) and the seed-0 neural reference, each reproducing the attacked DEV_REPORT outputs bitwise.
>    - After the verifier's D1, the SCM package path gained its own schema refusals.
>    - The re-check refused all 12 cases on all four packages, and the CSV path reproduces their wires bitwise. The packages were not rebuilt.
>    - No seed-1 neural package exists, a coverage gap.
> 7. `ENGINEERING_REPAIRS.md`:
>    - one environment failure before any fit, preserved and rerun;
>    - one post-lock logged-access addition for the verifier (`common.py`);
>    - one post-report deployment schema repair (`deploy_scm.py`, re-checked);
>    - a docstring note in `selection.py`;
>    - the verifier's coverage notes.
>
>    Three locked modules therefore differ from their ANALYSIS_LOCK hashes: `common.py`, changed before the report, and `deploy_scm.py` and `selection.py`, changed after it. None changed a reported number.

**Find:**
> - `roles/verifier/`: independent recomputation;

**Replace with:**
> - `roles/verifier/`: independent recomputation, 8/8 checks PASS. Discrepancies D1–D8 are documentation and protocol-text gaps; no reported number changed.

### MF-9. `RESULTS_TABLES.md` §3 shows "ESTABLISHED" with no caption

Several endpoints read `ESTABLISHED` even though their claim is not established:
- all four ATTRIBUTION endpoints;
- three of the four COMPETITIVE endpoints;
- both RECIPE coalition endpoints.

These are endpoint interval statuses. The claim labels are `NOT_TESTED_RECIPE_NOT_ESTABLISHED`, `NOMINEE_FAILS_DEV_UTILITY_GATE` and `RECIPE_EFFECT_NOT_ESTABLISHED`. Without a caption, someone skimming the table can cite a claim the frozen labels do not allow. The header also does not say which bound "Clause bound" is.

**Fix.** Edit `scm_bench/tables.py` and regenerate. After the line

> ## 3. Primary endpoints (benefit = comparator AUC - SCM AUC; worsening = SCM - TWIN; clause one-sided alpha/3; family K = 42)

insert:

> Status is each endpoint's own family-interval status (K = 42) against its margin. It is not a claim status. The nominee fails the DEV utility gate, so every RECIPE, ATTRIBUTION and COMPETITIVE endpoint is descriptive; the claim labels are in section 4. Clause bound is the one-sided α/3 lower bound (L1) for benefit endpoints, and the upper bound (U1) for worsening, utility and subgroup endpoints.

---

## SHOULD-FIX

**S1. `START_HERE.md` bottom line: mark it descriptive and add the single-recipient view.** The scope paragraph poses the question for "one recipient or two linked recipients", but the bottom line answers only the pair. The seed-1 NEURAL_RAW comparison is UNRESOLVED.

Find:
> **Recovery.** The selected SCM setting (β = 1.0, 261 / 266 cells) lowered fresh-reader recovery of SEX from the linked pair against every comparator, on DEV_REPORT, seed 0 / seed 1:

Replace with:
> **Recovery (descriptive).** At the point, the selected SCM setting (β = 1.0, 261 / 266 cells) lowered fresh-reader recovery of SEX from the linked pair against every comparator, on DEV_REPORT, seed 0 / seed 1. These are not claims: the nominee fails one DEV utility point clause (below), and the seed-1 family interval against NEURAL_RAW includes 0.

Then, after "SCM's own pair AUC was 0.618 / 0.622.", add:
> Per recipient the picture is mixed. SCM's income recipient read higher than NEURAL_RAW's and H_LJ's on both seeds, and its occupation recipient read higher than H_LJ's (`DECISION.md` §4).

**S2. Comparator row labels.** "Best" and "strongest" read as results on DEV, but they are selection labels.
- In `START_HERE.md`, replace "Best shuffled-signal control" with "Selected shuffled-signal control (r2; most protective eligible on the MODEL_SELECT proxy)".
- In `START_HERE.md`, replace "Strongest frozen tree hybrid" with "Most protective eligible frozen tree hybrid (H_LJ)".
- In `DECISION.md` §3, replace "Best shuffled control" with "Selected shuffled control (r2)".

**S3. `DECISION.md` §5: scope the linked-pair statement to what was counted.** FIT_SUMMARY and verifier check 6 record capacity only for MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT. On those four roles and both seeds, distinct pair vectors = distinct occupation vectors = occupied cells, and every row is a stored table row. Ties among stored cells that are unoccupied on these roles were not checked.

Find:
> On every role, SCM's occupation output takes as many distinct float64 vectors as there are occupied cells (260 on ATTACK_FIT and 253 on DEV_REPORT for seed 0), and its income output, with 32 values, is a function of the cell. So the occupation output identifies the shared cell. **For an optimal reader the linked pair adds nothing beyond the occupation output.**

Replace with:
> On each of the four recorded roles (MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT) and both seeds, SCM's occupation output takes as many distinct float64 vectors as there are occupied cells: 260 / 265 on ATTACK_FIT and 253 / 259 on DEV_REPORT. Its income output, with 32 values, is a function of the cell. So for these people the occupation output identifies the shared cell. **For a Bayes-optimal reader, the linked pair adds nothing beyond the occupation output for these people.** This rests on the recorded counts (verifier check 6). Ties among stored cells never occupied on these roles were not checked.

Also extend "on seed 0" in the fitted-readers sentence to: "(0.6175 against 0.6010 on seed 0; 0.6219 against 0.6131 on seed 1)".

**S4. `DECISION.md` §6 heading: the bottleneck statement goes beyond what the fit record shows.** At stopping, every SCM setting (β 0.5, 1.0, 1.5; both seeds) had exhausted two plug-in budgets: occupation NLL and grouped income Brier. The record does not say which one blocked further merges. What singles out occupation NLL is the held-out evidence.

Find:
> **Occupation log-loss limits SEX-guided merging at these caps.**

Replace with:
> **Occupation log-loss is the binding held-out utility clause at these caps.** In the fit, every SCM setting stopped with two plug-in budgets exhausted, occupation NLL and grouped income Brier, so the fit record alone does not say which one blocked further merges. On held-out people, only occupation NLL breached a cap.

**S5. `DECISION.md` §6, third bullet: the mechanism is mis-stated.** A merge's plug-in NLL cost is a loss of resolution under the fitted law, not of calibration.

Find:
> and that costs occupation calibration.

Replace with:
> and that costs occupation log-loss. The plug-in cost of a merge is the mass-weighted KL divergence of the pooled cells' occupation vectors from their merged mean: lost resolution, not lost calibration under the fitted law.

**S6. `DECISION.md` §6, second bullet: give the per-seed values and the other comparators' spend.**

Find:
> - The twin reached the same cell count for 0.0027 nats.

Replace with:
> - The twin reached the same cell count for 0.0027 / 0.0026 nats. At the same count, IBSI spent 0.0064 / 0.0066 and the shuffled control 0.0039 / 0.0038.

**S7. `DECISION.md` §6, "Where the cost lands": the subgroup evidence is Brier, not NLL.**

Find:
> **Where the cost lands.** It concentrates on men's occupation predictions: their within-SEX Brier excess is the largest in the bank.

Replace with:
> **Where the cost lands.** The measured within-SEX cost concentrates on men. That cost is occupation Brier; NLL by SEX was not measured. Men's excess against U0 (0.0062 / 0.0078) is the largest of the seven final-bank releases on DEV_REPORT, on both seeds. Women's is negative (−0.0015 / −0.0048).

**S8. `RESULTS_TABLES.md` §5 header (via `tables.py`): the share column is mislabelled.** The record field is `share_ATTACK_VALIDATION_people_whose_vector_occurs_in_ATTACK_FIT`, a share of people, not of distinct vectors. Replace "Share of validation vectors seen in ATTACK_FIT" with "Share of ATTACK_VALIDATION people whose pair vector occurs in ATTACK_FIT".

**S9. State budgets, custody and receipts in the closeout (brief item 6).** START_HERE and DECISION do not report compute use or custody. Add this to START_HERE (or to HANDOFF.json, and point to it), filling the bracketed values from the job-queue ledger and the custody run:
> **Compute, custody and receipts.**
> - Compute: [x.xx] CPU-h of the 8 CPU-h cap and [h:mm] of the 6 h elapsed limit, from 02:38:58Z (job-queue ledger). The two fit units used 856 CPU-s (`FIT_SUMMARY.json`). The verifier's scripts used about 2.5 CPU-minutes.
> - Receipts: the pre-fit receipt `f0a3b94e9` was pushed at 03:08:29Z, before the first fit attempt. The reporting-lock receipt `bd31bcde0` was pushed at 03:20:10Z, before the DEV_REPORT re-read at 03:20:19Z. Final publication: [sha].
> - Custody: on-device backup [status], off-device backup [status]. Restore-and-deploy check from the backup alone (`scm_bench/restore_check.py`): [result].
>
> Do not claim off-device custody until that restore check has passed.

**S10. `ENGINEERING_REPAIRS.md` row 4: name every locked module that changed after the report.**

Find:
> `deploy_scm.py` changed after the report (D1 above)

Replace with:
> `deploy_scm.py` (D1 above) and `selection.py` (docstring only, D5) changed after the report. Both are locked modules, so their ANALYSIS_LOCK hashes now differ.

If the MF-7 fallback is used, add `PROTOCOL.md` to the same sentence.

**S11. `DECISION.md` §1: make the clause count explicit.** Replace "pass all 24 held-out point clauses on both seeds" with "pass all 24 held-out point clauses (12 per seed)".

---

## NOTES

- **N1. DECISION §4's new paragraph checks out.**
  - All four competitive clause lower bounds are reproduced: 0.0084 / 0.0024 against NEURAL_RAW, 0.0102 / 0.0074 against H_LJ.
  - The source convention it cites holds: `acs_hybrids/report.py` lines 283–284 apply `fails_gate` when `not (ms_ok and gate24)`.
  - Stating both readings without naming the alternative label is appropriately conservative.
  - For balance, consider adding the failing clause's own confidence status: "The gate fails on a single point clause (0.0106 against 0.010), whose confidence status is UNRESOLVED (family interval [−0.0000, 0.0212])."
- **N2. DECISION §3 table: inconsistent detail in the status column.** The IBSI row gives family bounds, but SHUF and C0 show only "descriptive". Their family lower bounds are 0.022 / 0.020 (SHUF, primary) and 0.029 / 0.026 (C0, secondary).
- **N3. RESULTS_TABLES §1: "n/a" hides recorded values for the frozen comparators.** SELECTION_LOCK records their MODEL_SELECT clauses: all 24 pass, worst normalised 0.340 / 0.570 (NEURAL_RAW) and 0.341 / 0.295 (H_LJ). These could replace "n/a".
- **N4. ATTACK_FIT (8,001) comes from the protocol.** It is not printed in any public record checked here. ATTACK_VALIDATION (3,000) is consistent with READERS_SUMMARY's shares (310 / 3,000).
- **N5. Seed 0's k-means stopped at the 50-iteration cap without converging; seed 1 converged at 33** (verifier observation). This is within PROTOCOL §3, and DECISION §1 could mention it.
- **N6. "Neural adversarial comparator" matches the frozen-hybrids study's naming** of NEURAL_RAW ρ1.0 E2. Every final-bank comparator passes all 24 DEV point clauses, so no comparator is counted as beaten while ineligible.
- **N7. The linked-pair checks are confirmed against serialised outputs** (brief item 4). Verifier check 6 confirms that bank wires are byte-equal to ARM_OUTPUTS, every released row is a stored table row, used occupation table rows are distinct, and the income group is a function of the final cell. LOOKUP won exactly one view (S1 TWIN income). Every pair winner is an inherited reader, and the 7-reader pair AUCs are identical.
- **N8. When HANDOFF.json is written:**
  - carry the corrected statements: two logged DEV re-reads; the per-recipient adverse comparisons; DEV seed-1 NLL increase 0.0092; the state of the PROTOCOL hash;
  - do not repeat commit `04facb7fe`'s phrase "recovery lower than every comparator" unless it is qualified as linked-pair, at the point and descriptive.
