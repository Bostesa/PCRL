# Label-blind support census: shared-context release v1

Machine-readable output: `SUPPORT_CENSUS.json`. Generator: `support_census.py` in this folder. The run takes about 4 s, peaks at about 530 MB RSS, uses `OMP_NUM_THREADS=1` and loads one anchor at a time. It was run on the macOS host (numpy from the repo `.venv`). The census computes nothing from floating-point re-encoding, so the host does not affect it.

## Label-blind statement

- **Arrays accessed:** `ctx.anchor`, `ctx.pools.<pool>.{households, ids, weights, x, ha}`, `encoded.<pool>.codes.T0`, `prepared.roles` (the task-directed household index lists), and `Q` from the D17, D33 and Q `Q.npz` maps.
- **Never accessed:**
  - `ctx.pools.<pool>.labels` (any key, any pool, any role, including outer);
  - `prepared.tables` (label-derived support and cost tables);
  - `prepared.erasers`;
  - `encoded.{p,b,r,risk,actions,global_offsets}`;
  - `hb`, `J`, `A0`, `raw_rows`;
  - the historical fineC tables, which were hash-checked only.
- **Why the label arrays were in memory:** `joblib.load` unpickles the whole `prepared.joblib`, so the label arrays were resident. No code path reads them.
- **Class-support fields:** the `class_support` fields of `DATA_ROLE_COUNTS.json` were neither read nor recomputed.
- **Outer role:** only people counts, household counts and PWGTP masses were used.
- **Output:** aggregate only. A grep for ACS household-ID patterns returns 0 hits.

## 1–2. Roles, overlap, ESS

The role rule is the predecessor's: SHA256(`pcrl_adaptive_release_v1|household_id`), first 64 bits / 2^64. Role membership follows the predecessor semantics:

- **Non-outer roles** draw on `representation_fit`, `downstream_fit`, `downstream_validation` and `attacker_fit`. `attacker_validation` is excluded, as in `roles.pooled_role` and `roles.summarize_roles`.
- **Outer** draws on all five loaded pools.

Cells show people / households / household-mass PWGTP ESS.

| role | anchor 0 | anchor 1 | anchor 2 | global union people / hh / hh-ESS / person-ESS | hh in 1 / 2 / 3 anchors | legacy teacher hh (fit / fit+int.val) |
|---|---|---|---|---|---|---|
| nuisance_train | 4,138 / 2,795 / 1422 | 4,113 / 2,777 / 1418 | 4,124 / 2,809 / 1430 | 5,758 / 3,884 / 1961 / 3755 | 732/1807/1345 | 1389 / 1749 |
| audit_fit | 4,158 / 2,765 / 1388 | 4,176 / 2,807 / 1424 | 4,144 / 2,776 / 1418 | 5,816 / 3,894 / 1987 / 3736 | 773/1788/1333 | 1371 / 1705 |
| coefficient_split | 5,257 / 3,506 / 1848 | 5,317 / 3,556 / 1851 | 5,334 / 3,569 / 1870 | 7,326 / 4,907 / 2586 / 4787 | 952/2186/1769 | 1783 / 2244 |
| inner_selection | 2,215 / 1,468 / 744 | 2,187 / 1,453 / 754 | 2,144 / 1,420 / 748 | 3,021 / 2,003 / 1042 / 1984 | 379/910/714 | 749 / 930 |
| inner_check | 2,131 / 1,454 / 740 | 2,142 / 1,442 / 740 | 2,119 / 1,447 / 736 | 2,956 / 2,018 / 1047 / 1965 | 372/967/679 | 697 / 879 |
| outer_assessment | 3,555 / 2,409 / 1252 | 3,535 / 2,384 / 1238 | 3,549 / 2,371 / 1207 | 4,403 / 2,968 / 1524 / 2865 | 304/1132/1532 | 1021 / 1289 |

**Cross-check against the predecessor.** Every per-anchor count of people, households and weight_sum in `DATA_ROLE_COUNTS.json` is reproduced exactly (18/18). All six global household counts match. The outer historical-teacher overlap of 1289 is also reproduced: it counts outer households that sit in task-directed `teacher_fit` ∪ `teacher_internal_validation` in any anchor.

**Overlap across roles.** Household overlap is 0 within every anchor and 0 in the global union.

**People and households across anchors.** Most people and households appear in two or three anchors: 81% of coefficient people are in at least two. Person weights and household IDs are consistent across anchors, with 0 mismatches. The three anchors are therefore not independent samples. The global union is the honest support figure: 4,907 coefficient households, with household ESS of about 2,586.

**Effect of PWGTP heterogeneity.** It roughly halves the effective support. Household-mass ESS is about 0.53 × the household count; person ESS is about 0.65 × the person count.

**Other structural checks.**

- `attacker_validation` rows fall into each role and are excluded from the non-outer roles. The coefficient role loses 744, 702 and 757 such people in anchors 0, 1 and 2.
- Per-person IDs are unique within each anchor.
- No household spans two pools.

## 3. T32 parents and D17 map

**Where the T32 codes come from.** They are stored in `prepared.joblib` as `encoded[pool]['codes']['T0']`, with values 0–31, for every pool. The files were created on 2026-09-21 and 2026-09-22 on Linux x86 (c7i.8xlarge). Their sha256 matches the pin and the `PREPARED.json` receipts. No re-encoding was done on the Mac, and none is needed for this census.

**Coefficient role, unique households per T32 state:**

| anchor | min | median | max | ESS range |
|---|---|---|---|---|
| 0 | 131 | 161 | 198 | 67–130 |
| 1 | 127 | 162.5 | 195 | 79–137 |
| 2 | 142 | 161 | 192 | 61–127 |

This reproduces the handoff's "largest parent 198" for anchor 0.

**Binary splits at a floor of 100 unique households per child (coefficient role).** With disjoint households, which requires a parent with at least 200 households, 0/32 states are count-feasible in any anchor. The "upper-bound" column instead lets multi-person households appear in both children. Under that allowance, anchor 0 has one state that could reach 200, and anchors 1 and 2 have none. At floors of 50 and 25, all 32/32 states are count-feasible.

**Nuisance+coefficient union.** Parents hold 237–331 households, with household ESS of 93–232. All 32/32 states can count-feasibly split at 100 with disjoint households. However, the household-mass ESS reaches 200 in only 7–8 states per anchor (anchor 0: 7, anchor 1: 8, anchor 2: 7), so a 100-per-child ESS floor stays tight.

**D17 map.**

- `T0_U_unconstrained_a17/Q.npz` is deterministic in every anchor (32/32 one-hot rows), but it is fitted per anchor.
- It uses 16, 14 and 15 distinct tokens of the 17 in anchors 0, 1 and 2.
- Merge histograms, written as {states per token: number of tokens}:
  - anchor 0: {1:8, 2:4, 3:1, 4:2, 5:1};
  - anchor 1: {1:4, 2:4, 3:4, 4:2};
  - anchor 2: {1:5, 2:5, 3:4, 5:1}.
- The full state→token lists are in the JSON.
- For comparison, D33 is also deterministic (23, 22 and 21 tokens used). `Q` (T0_L_0.01) is stochastic.
- **Coefficient households per D17 token** range over 131–724, 127–665 and 152–764. Every used token has at least 100 households.

## 4. Pooled-context support (indicative only)

Cut points are unweighted quantiles of PC1, fitted on nuisance_train rows of each anchor. The counts are unique coefficient households per context, with household ESS in parentheses.

| design | a0 | a1 | a2 | ctx×D17 cells (a0/a1/a2) | cells ≥50 hh | cells ≥100 hh |
|---|---|---|---|---|---|---|
| K=1 | 3506 (1848) | 3556 (1851) | 3569 (1870) | 16/14/15 | all | all |
| K=2 X_A-PC1 | 2056 (1109), 2112 (1199) | 1948, 2186 | 2188, 2022 | 32/28/30 | 32/26/28 | 18/20/22 |
| K=2 H_A-PC1 | 2114 (1137), 1901 (1090) | 2130, 1980 | 2177, 1953 | 32/28/30 | 30/28/30 | 20/22/23 |
| K=4 X_A-PC1 | ~1030–1290 each (ESS 548–775) | | | 64/56/60 | 38/38/43 | 18/18/18 |
| K=4 H_A-PC1 | ~1060–1210 each (ESS 589–694) | | | 64/56/60 | 37/42/44 | 16/21/20 |
| K=4 X_A×H_A medians | ~990–1300 each (ESS 563–807) | | | 64/56/60 | 38/40/43 | 18/20/22 |

**Reading.**

- **K=2** keeps every context at about 1,900–2,200 households (ESS above 1,000). Between 56% and 79% of context×token cells reach 100 households, and 93–100% reach 50.
- **K=4** keeps contexts at about 1,000–1,300 households (ESS about 550–800). Only 25–38% of context×token cells reach 100 households, and 58–75% reach 50. The smallest cells hold 6–22 households.

A fully saturated context×token parameterization is therefore support-limited at K=4 under a 100-household floor. A pooled or regularized model is needed; at K ≤ 2 the design is borderline workable.

**Caveats.**

- Contexts are defined per person, so households can span contexts. For K=2, 509–662 households have an extra context membership; per-context household sums exceed the total.
- X_A is whitened: PC1 carries only 3.7% of the nuisance variance, so it is an essentially arbitrary direction.
- The real contexts will come from frozen task and risk predictions, so treat these figures only as a sizing check.

## 5. Feature rank (coefficient role, standardized)

**X_A.** Full rank, 32. The correlation eigenvalues span 0.79–1.24, with a condition number of about 1.2: X_A is PCA-whitened, so it contains no redundancy.

**H_A.** Four columns but effective rank 2. Columns 0+1 and columns 2+3 each sum to 1 to within 9e-8 (two binary probability pairs). The eigenvalues are {≈2.93–3.01, ≈0.99–1.08, 0, 0}. Rank 3–4 at the float64 default tolerance is float32 round-off only.

**[X_A, H_A].** Effective rank 34 of 36, with two exact dependencies. The next-smallest eigenvalues are about 0.16 and 0.03–0.04. The H_A pairs are largely linear in X_A: R² is 0.79–0.81 for pair 1 and 0.94–0.95 for pair 2.

**Implication.** Use at most 2 H_A coordinates, for example one per pair, or a logit per pair. The joint design is near-collinear along the second H_A pair.

## 6. Local inputs and hash verification

- **REUSABLE_INPUTS_PINNED:** all 36 local files exist and their sha256 matches. For each anchor these are `prepared.joblib`, `encoder.joblib`, the Q.npz, solution and tables files for D17, D33 and Q, and the `/tmp` fineC tables.
- **PREPARED.json receipts:** all 202/202 artifact hashes verify in each anchor, 606 in total. These cover the encoder, the task and baseline teacher slates, MLP trajectories, split_rows, erasers, audits and evaluation files.
- **INPUT_RESTORE.json:** all 22/22 raw and preprocessing inputs verify under `/Users/nathansamson/PCRL`. These are `data/folktables/2018/1-Year/psam_p06.csv` (267 MB) plus, for each of seeds 0–2, `anchors.npz`, `pca.npz`, `split_rows.npz` and the A0/J `final.pt` and `releases.npz`. They are absent from the sparse worktree and from the prospective worktree.
- **Pinned source files:** 7/7 match. `acs_transfer_data.py` and `acs_transfer_heads.py` are excluded by the sparse checkout and were verified against the HEAD git blobs.
- **Environment record:** `results/pcrl_task_directed_release_v1/ENVIRONMENT.json` (Linux x86, numpy 2.4.2, sklearn 1.8.0) is present.
