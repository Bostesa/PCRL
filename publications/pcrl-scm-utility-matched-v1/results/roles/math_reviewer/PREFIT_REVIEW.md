# Pre-fit review: baseline and mathematics (SCM utility-matched follow-up)

Reviewer role: baseline and mathematical reviewer. Date: 2026-10-17.

Read: draft `PROTOCOL.md` (32,220 bytes, 00:55), `scm_bench/{engine,fit_scm,select_scm,selection,readers,common}.py`, source `PROTOCOL.md` and `PROTOCOL_POSTRUN_CLARIFICATIONS.md`, public `FIT_SUMMARY.json`, `SELECTION_LOCK.json` and `RESULTS_TABLES.md`, and new code `scm_um/{admit,common,trajectory,select}.py`, plus the claim-label block of `scm_um/report.py`.

Not done: no numerical job was run, no `.npz` was loaded, and nothing under `.private/` was read. The only arithmetic was a check of how the float grid literals are represented.

**Verdict:** the design is sound and faithful to the source. Register it after the five MUST-FIX text and label changes below; the trajectory code needs no change.

---

## MUST-FIX (blocking)

### M1. The DEV carry-over gate is two-sided and is defeated by the comparator-selection rule

**Where:** §10, §11 claim M and its components; `report.py` (`carry`, `component()`).

**Problem.**
- The rule picks ORD_NLLM as the lowest-proxy candidate inside |Δ| ≤ 0.001.
- Along a merge path, more merging generally lowers the proxy. The rule therefore tends to choose the high-expenditure edge of the window, where Δ = SCM − ORD ≈ −0.001 on MODEL_SELECT.
- The DEV gate `abs(point) <= 0.001` then fails about half the time per seed on noise alone. The registrant's own P4 (0.35) expects the carry-over to fail more often than not.
- The gate also mislabels the conservative case. When ORD spent more occupation NLL than SCM on DEV and still reads higher, the SCM advantage holds even more strongly. The label nonetheless closes the comparison as "inconclusive".
- Only SCM spending more than the comparator recreates the original confound.

**Proposed rule (labels only; MODEL_SELECT matching stays two-sided as commissioned).**
- `MATCH_DID_NOT_CARRY_OVER` applies iff, in some seed, the DEV point (SCM − comparator occupation NLL) is > +0.001.
- If the point is < −0.001, record `COMPARATOR_SPENT_MORE (conservative for SCM)` and continue down the label sequence.
- Apply the same rule to the IBSI and SHUF components.
- The two-sided POINT_MATCH / POINT_MISMATCH and EQUIVALENCE_* statuses stay reported unchanged, and P4 is scored as registered.
- Code change: in `report.py`, compute `carry` from `point <= 0.001` (signed) instead of `point_pass`, and do the same for `component()`.

### M2. Whether unreached N{t} labels are nomination candidates

**Where:** §4 and §6.

**Problem.**
- In the real data, SCM's terminal fitted NLL is F_M = 0.0099989 (seed 0) and 0.00999997 (seed 1), from public FIT_SUMMARY `nll2`. Both are below 0.010 − 1e-15.
- `checkpoints()` therefore flags SCM's `N0.010` as "unreached (missing coverage)", even though it is the budget-exhausted terminal and the historical nominee that P1 refers to.
- §6 does not say whether unreached labels are candidates. `select.py` includes them.

**Proposed text for §4/§6.**
> "The reached flag is descriptive coverage metadata. Every N{t} label, reached or unreached (an unreached label is TERM), is a candidate in §6 and in every §7 candidate set. For each unreached label the shortfall t − F_M is reported."

### M3. A role that is available in one seed only, and the nearest-diagnostic rule

**Where:** §7 "Unavailable matches".

**Problem.**
- The text speaks of a "role" and a "worst-seed mismatch max_s". For ORD the selection is per seed, so the text is ambiguous.
- The code does something specific and reasonable that the protocol must register:
  - ORD is per seed: a seed with a match is `MATCHED`, and a seed without one is `NEAREST_DIAGNOSTIC_NOT_MATCHED`.
  - For IBSI/SHUF, the code takes each seed's clause-passing candidate minimising (|Δ|, proxy, m), then picks the γ or r minimising (max_s |Δ|, mean proxy, grid index).
  - The text instead describes a joint minimisation, which differs from the code on ties.

**Proposed text.**
> "ORD_NLLM is selected and, if unmatched, replaced by its nearest diagnostic per seed: the clause-passing ORD candidate minimising (|e_s − e*_s|, proxy_s, m). A seed holding a matched ORD candidate keeps it, and the other seed holds the diagnostic; claim M is then MATCH_UNAVAILABLE. For IBSI/SHUF the role is matched only if one γ or r matches in both seeds. Otherwise, per γ or r, take each seed's nearest candidate as above, and choose the γ or r minimising (max_s |Δ_s|, mean proxy, grid order); both seeds are then DIAGNOSTIC_NOT_MATCHED."

### M4. A nominee equal to C0

**Where:** §6 and §11.

**Problem.**
- t = 0 is a candidate, and `N0.000` is C0 on every trajectory (m*(0) = 0 unless a first merge costs ≤ 1e-15).
- If t* = 0, the bank's SCM, C0 and ORD_CNT (S0.000) roles are one service.
- ORD_NLLM's candidates include C0 itself with Δ = 0, so M and K cannot be established and no false advantage arises. Claim B, however, would label C0 itself as "SCM superior".

**Proposed text.**
> "If the nominee's effective release equals C0, claims M, K and B are labelled NOMINEE_IS_C0 (no SCM merge applied; not an SCM result). The bank is evaluated descriptively."

### M5. The trajectory lock and the repair path for checkpoints

**Where:** §3, §4, §15 and §16.

**Problem.**
- `select.py` refuses to run without a committed `TRAJECTORY_COMPLETE_LOCK.json`, which is good, but the protocol never names this lock.
- Under §16 ("reruns of affected checks"), an engineering repair of a trajectory after MODEL_SELECT has been seen would silently create new checkpoint releases. This is the only path to a post-hoc checkpoint.

**Proposed text.**
> "UM_TRAJ outputs (TRAJ_SEED.json with every label→(m, F_m, release hash) map, TRAJECTORIES.npz with merge sequences and the full spending matrix, RELEASE_TABLES.npz) are hashed in TRAJECTORY_COMPLETE_LOCK.json, committed before any MODEL_SELECT label is read. UM_SELECT evaluates exactly the locked effective releases. A later repair must reproduce every locked release hash. Any role whose locked release changes is recorded REPAIR_CHANGED_RELEASE and is not re-selected."

---

## SHOULD-FIX

### S1. Checkpoint index under a non-monotone F (§4)

**Issue.**
- Analytically each per-merge Δ_occ-NLL is a sum of KL divergences and is ≥ 0. Negative values (about 1e-17) are rounding only.
- `checkpoints()` uses `max{m : F_m ≤ t + 1e-15}`. If a decrease straddled the threshold, this could choose a prefix after a crossing.

**Proposal.**
- Define m*(t) as the last prefix before the first crossing: m*(t) = min{m : F_m > t + 1e-15} − 1, or M if F never crosses. This equals the max definition on a running maximum of F.
- Record min_m (F_{m+1} − F_m) per trajectory.
- State the grid as t_k = k/1000 in float64. The code's `round(0.001*i, 3)` does this; note that 9*0.001 = 0.009000000000000001 ≠ 0.009.
- The +1e-15 slack in m*(t) matches the engine's admissibility. With B = 0.01, every capped trajectory therefore has N0.010 = TERM even if F_M exceeds 0.010 by up to 1e-15. This is consistent and should be stated.

### S2. Update the gate text in §3 to match the code

**Issue.** §3(a) says "final merge map and step count". The code also compares the stop reason, the final cell count and all eight `spent` entries bitwise. Those entries are order-dependent running sums, which makes the comparison a near-conclusive fingerprint of the merge sequence.

**Proposal.**
- Write the stronger check into the text.
- Add one cheap check: the replayed TERM tables equal `E.tables(...)` on `engine.run_path`'s returned final state, bitwise. Without it, the extension beyond K_hist is tied to the engine only through the final map.
- Robustness: if SCM's M ≠ k − K_hist, `snaps[hm]` raises a KeyError before the failed-gate record is written. The code should write `TRAJ_SEED_FAILED_GATE.json` instead.
- Faithfulness checked:
  - Every ORD, IBSI and SHUF prefix at count c is exactly the source rule run with `target = c`, because the target only breaks the loop. Continuing past the old count is therefore consistent with the source objectives.
  - `run_path_recorded` matches `engine.run_path` line for line: priorities, R floor, prefilter, grouped-income trial check, 1e-15 slack, spent updates and stopping.

### S3. Income expenditure is not matched, so the confound can move channels

**Issue.** At 261 cells the historical SCM spent plug-in grouped income NLL 0.0068, against 0.0012 for TWIN. At matched occupation NLL, SCM may still spend materially more income NLL than ORD_NLLM.

**Proposal (a descriptive qualifier; the label is unchanged).** Beside claim M, report the SCM − ORD_NLLM income NLL, Brier and accuracy differences on MODEL_SELECT and DEV. Attach `INCOME_EXPENDITURE_NOT_MATCHED` if SCM's MODEL_SELECT income NLL exceeds ORD_NLLM's by more than 0.001 in a seed.

### S4. Finite reader budget favours finer releases

**Issue.**
- At matched occupation NLL, SCM keeps many more cells than ORD. ORD spent only 0.0027 fitted NLL at 261 cells, so to match about 0.008–0.010 it must merge much further.
- With 8,001 ATTACK_FIT people, LOOKUP and the flexible readers under-read finer partitions. Historically, at the same count, SCM had a median of 18 people per pair vector against 24 for TWIN.
- This biases claim M in SCM's favour.

**Proposal (pre-registered, descriptive, never primary).**
- Beside claim M, show distinct pair vectors and median people per vector for SCM and ORD_NLLM.
- Run a two-point learning curve for SCM and ORD_NLLM only: the pair slate refitted on 25 % and 50 % household subsets of ATTACK_FIT, using the source `LC_SEED` logic in `readers.py`. Its sole use is to report the direction of the dependence on reader budget.

### S5. Matching coverage

**Issue.**
- In the source bank, the ratio of MODEL_SELECT expenditure to fitted expenditure is 1.04–1.44. Examples: SCM_b1.0 0.0106/0.0100 and 0.0111/0.0100; TWIN_b1.0 seed 1 0.0032/0.0026; SCM_b0.5 seed 1 0.0072/0.0050. Each is computed from the public C0 and arm NLL clauses.
- A fitted spacing of 0.001 therefore maps to about 0.0010–0.0014 held-out. Usually that is within the 0.002 window, but it is not guaranteed locally.

**Proposal.** Report, around e*_s, the largest held-out gap between consecutive candidates of each comparator. Optionally, register now a comparator-only fitted grid at 0.00025 steps. This is allowed before MODEL_SELECT is read, and it only strengthens the baselines.

### S6. Commissioned sentences missing from the protocol text

Add to §3/§4:
> "An arm need not reach 32 cells to be eligible."

> "No artificial errors, decoders or noise are added to any release to make a control resemble SCM."

---

## NOTES

- **SEX access in ORD.** ORD's priority includes the normalised guard costs, so SEX labels affect ORD's merge order as well as admissibility. In §2, write "through the guard, in both priority and admissibility". "Guard-only, not SEX-blind" remains accurate.
- **What an SCM prefix is.** SCM prefixes N{t} are truncations of the β = 1.0 path. They are not SCM run at budget t: the historical SCM_b0.5 is a different release. Describe the nominee as "the β = 1.0 trajectory truncated at fitted occupation NLL t".
- **The nomination slack does not bind.** On MODEL_SELECT, C0's occupation NLL excess is −0.0071 / −0.0062 (seed 0, against U0 / Ucal) and −0.0045 / −0.0044 (seed 1). Even SCM's terminal sits at 0.0035–0.0067.
  - The 0.008 slack will not bind anywhere on SCM's path. The binding condition will be the men's guard (0.0044 / 0.0059 at the terminal).
  - The seed-0 Ucal clause of the same release moved from 0.0043 (MODEL_SELECT) to 0.0106 (DEV). The 0.002 slack therefore offers little protection against that role shift.
  - Disclose this; do not re-tune using DEV knowledge.
- **Comparator identical to the nominee.** Aliasing across trajectories (C0 everywhere; possibly shared early prefixes) can make a comparator role the same effective release as the nominee. The benefit is then exactly 0, which is conservative. Keep this, and record the alias.
- **Asymmetric freedom is intended.** ORD is chosen per seed, while SCM uses one common t. ORD is not required to pass the 0.008 slack or the subgroup guards. Both asymmetries favour the baseline, as commissioned.
- **SHUF_NLLM has the most selection effort** (5 r × candidates), so its MODEL_SELECT proxy is the most optimistic. Fresh DEV readers remove the bias, but expect it to regress.
- **The "match" equivalence endpoints add little to the primary family.** In the primary family (6 slots), EQUIVALENCE_ESTABLISHED needs a family half-width ≤ 0.001, roughly an SE of 0.0003. That is unlikely: the source DEV SE of an NLL clause against Ucal was about 0.003. Consider making them secondary, which would shrink K_primary.
- **Proxy bias.** The proxy's max over readers and LOOKUP's under-reading of fine partitions (source §12.10) also apply within the ORD window. "Strongest by proxy" is therefore proxy-strongest, not necessarily the strongest on DEV.
- **IBSI adaptation unchanged.** PUB_IBSI keeps the source adaptation: cell-level income NLL in its score, while the release groups income. The protocol states this.
- **Code checked and found consistent with §6–§7.** In `select.py`: the 32-condition fallback, ties to smaller t, the (proxy, |Δ|, m) key, the common-γ/r rule, the S{t*} same-count roles, the IBSI_CNT source rule 6 with its ineligible fallback, and deduplication of services.
