# Disposition of the pre-fit reviews (coordinator, 2026-10-17, before ANALYSIS_LOCK; nothing fitted, no MODEL_SELECT label read)

Reviews: `roles/math_reviewer/PREFIT_REVIEW.md` (baseline and mathematical reviewer) and `roles/statistical_reviewer/PREFIT_STATISTICAL_REVIEW.md` (statistical-design reviewer, whose `FROZEN DESIGN` block is authoritative). Both reviewers ran no numerical job, loaded no array and read nothing under `.private/`.

## Baseline and mathematical review

| Item | Disposition |
|---|---|
| M1 one-sided DEV carry-over gate | **Not adopted for labels**, by the statistical reviewer's decision (a): the commission says to state that the match did not carry over when the difference drifts outside the tolerance and never to widen it. Claim M keeps the two-sided gate; a per-seed direction sub-status (CARRIED_OVER / SCM_SPENT_MORE / COMPARATOR_SPENT_MORE) is reported in every case and states only who spent more |
| M2 unreached N{t} labels as candidates | Applied (§4, §6): every N{t} is a candidate; unreached labels alias TERM and report the shortfall t − F_M |
| M3 one-seed availability and nearest diagnostic | Applied (§7): the coded per-seed ORD rule and the common-γ / common-r worst-seed-gap rule are registered |
| M4 nominee equal to C0 | Applied: `NOMINEE_IS_C0` for M, K and B (statistical decision (c)) |
| M5 trajectory lock | Applied (§3): TRAJECTORY_COMPLETE_LOCK of every label → (m, F_m, release hash) before any MODEL_SELECT read; repairs must reproduce the hashes or the role is `REPAIR_CHANGED_RELEASE` |
| S1 first-crossing threshold rule, t = k/1000 | Applied (code and §4) |
| S2 gate text, terminal-table check, crash-safe gate record | Applied (code and §3) |
| S3 income expenditure not matched | Applied: SCM − comparator income and occupation NLL / Brier / accuracy differences beside every comparison, with `INCOME_EXPENDITURE_NOT_MATCHED`; income-NLL difference vs ORD_NLLM added as a secondary endpoint |
| S4 learning-curve check | **Declined**: the commission fixes the reader budget and excludes learning-curve searches. The bias (a fixed budget can under-read SCM's finer partition) is disclosed in §8 and reported beside the recovery comparisons |
| S5 matching coverage; optional finer comparator grid | Coverage reporting applied (§7). The finer grid is **declined**: the commission fixes the checkpoint grid |
| Commissioned sentences (32 cells not required; no artificial errors or noise) | Applied (§3) |
| Notes (ORD's merge order uses the guard; SCM prefixes are truncations) | Applied (§2, §3) |

## Statistical-design review

| Item | Disposition |
|---|---|
| MF-1 strict fixed sequence M → IBSI → SHUF | Applied (code and §11) |
| MF-2 match endpoint definition | Applied (code and §11) |
| MF-3 availability vocabulary and IDENTICAL_RELEASE | Applied (code and §11) |
| MF-4 failure categories | Applied (code and §11) |
| MF-5 separate eligibility and status fields | Applied (code and §11) |
| MF-6 0.020 status and four qualifiers in the research decision | Applied (code and §11) |
| MF-7 secondary-roster wording | Applied (§11) |
| Decision (b): match endpoints to the secondary family; income-NLL difference added | Applied: K_primary = 58, K_secondary = 24 (`make_lock.py` asserts them) |
| S1 pairing wording; S2 bootstrap identity (G = 3955, 5,943 persons asserted) | Applied |
| S3 identical-release flag; S4 comparator DEV qualifier; S5 descriptive companions; S6 table naming | Applied |
| S7 bets: scoring rules; P8 coherence | Applied: P8 revised from 0.20 to 0.10 before the lock |
| S8 claim C wording for a fallback; S9 comparison-only wording | Applied |
