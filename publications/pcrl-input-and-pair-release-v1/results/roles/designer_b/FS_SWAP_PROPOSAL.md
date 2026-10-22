# FS_SWAP: cross-recipient keyed swap (replacement family `FS_SWAP`; the challenger's finding 20)

Designer B, written 2026-10-22 from 02:43Z (`date -u`), on the coordinator's 02:36Z request.

Label-free, under the same rules as `PROPOSAL.md`:
- not opened: any label file, anything of MODEL_SELECT, any reader cache, any `.private` runs or access content;
- inputs: TASK.pkl, SEXSIDE.pkl, and the admitted MODEL_FIT features;
- one thread, about 80 CPU-s (section 5).

Sketch: `fs_swap_sketch.py` (`selftest()` passes). Plug-in records: `fs_swap_feasibility.py` writes `FS_SWAP_FEASIBILITY.json`
(2 reader draws); `fs_swap_point.py` writes `FS_SWAP_POINT.json` (lambda 0.05, 4 draws).

**Bottom line.** The swap is fully specifiable, deployable from the six fields and a fixed bank, and SEX-blind. In my plug-in world
it does not do what finding 20 hoped:
- **It costs about twice as much per person.** A swapped person costs 2.1-2.2x the occupation decision-cell collapse.
- **It does not isolate the pair link.** It lowers the occupation own view at least as much as the pair, and the link through the
  kept occupation decision survives.
- **Its effect at the S3 point is too small to resolve.** At lambda about 0.05 the pair falls 0.003-0.004 below the twin, only
  0.0005-0.0007 below DQ at equal KL, and it is no better than occupation-only flattening (OCC_FLAT) or CSF at equal KL.
- **Prediction:** D2 `NOT_SEPARATED` (p 0.74). The falsification is cheap (4-6 configurations).

## 1. The exact operation (mode `swap`)

- **Twin.** f = (f_I, f_O), CTL_TWIN's cached HGB task teachers. SEX is never read, so `operation_sex_blind: true`.
- **Donor bank at deployment.**
  - The bank is all 20,000 MODEL_FIT people's six-field records b_j (compact coding), each with the deployed twin's vector f_O(b_j).
  - It is grouped by the cell k_j = argmax f_O(b_j) and kept people-weighted (duplicates stay in), so a donor is a draw from the
    cell's population of wires.
  - Within a cell, rows are ordered by SHA-256(`IPR_FS_SWAP|V<v>|bank|` + record bytes), with a stable sort. The release therefore
    does not depend on storage order; ties are identical records, which carry identical deployed vectors.
- **Bank inside the nested MODEL_FIT selection:** the other four outer folds' people, with their out-of-fold twin vectors.
- **Keys:** u(x) = SHA-256(`tag|sel` \| six fields) / 16^13 and v(x) = SHA-256(`tag|donor` \| six fields) / 16^13, with tag
  `IPR_FS_SWAP|V<v>`.
- **Who is swapped:** person x is swapped iff u(x) < lambda.
- **Which donor:** row floor(v(x) |B_k|) of cell k = argmax f_O(x), stepping cyclically past any bank row whose record equals x's
  (no self-donation). The release is a deterministic function of the six fields.
- **Release:**
  - the income recipient gets f_I(x), bitwise;
  - the occupation recipient gets f_O(donor) if x is swapped, else f_O(x).
  - Both decisions, and therefore both accuracies, are kept exactly.
- **Disclosed.** A swapped wire is another (MODEL_FIT) person's twin vector, released under someone else's identity. The privacy of
  the bank members is outside this benchmark.

## 2. Strength path and selection

- **Grid:** lambda in {0.01, 0.02, ..., 0.10, 0.125, 0.15, 0.20}.
- **Budget S3.** The swap spends only occupation NLL and the occupation guards. S3 roughly doubles the affordable lambda against S2,
  as for CSF (finding 13).
- **Rule:** the PROTOCOL 4 default. Take the largest lambda such that it and every smaller lambda meet S3 (estimate + 2
  household-clustered SE) on MODEL_FIT nested 5-fold household folds. Per-person spend uses task labels; the men's and women's
  guards use SEX labels.
- **Plug-in S3 ratios:** 0.42 / 0.38 at 0.025, 0.82 / 0.84 at 0.05, 1.31 / 1.24 at 0.075. The binding component is the men's occupation
  Brier guard at 0.05 (the women's from 0.075 in V1), so I expect lambda_S3 = 0.05 (range 0.04-0.06).
- **Reported beside:** the summed occupation Brier excess is +0.0037 / +0.0038 at 0.05. That is a G1 clause (cap 0.005) that is not
  an S3 component (finding 17).

## 3. Matched controls and D5 (SEX-blind route, `RULES.json` D2 `sex_blind_operation`)

- **Gating paths: TWIN and SPEND.**
  - SPEND is the shared DQ path, with the family-independent tag `IPR_DQ_S<v>` (finding 16).
  - It has two bracket points per version at the swap's MODEL_FIT nested total-NLL spend.
- **SBC and the substituted RO path** are degenerate by construction and not built. GENERIC_ZTILT is reported beside, never gating,
  if the shared RO path exists.
- **Ablation ABL_OCC_FLAT** is the decisive one for finding 20's mechanism.
  - It uses the same selection key; the selected people get the occupation decision-cell mean instead of a donor.
  - lambda_f is matched on MODEL_FIT nested total-NLL spend.
  - It asks whether hiding *who* lost the link buys anything over a recognizable collapse.
- **CSF at matched spend:** descriptive only, from IB's configurations, if their spends bracket. No new configuration.
- **D5: `NOT_APPLICABLE_ONE_OUTPUT_CHANGED`.**
  - Construction argument: the income wire is bitwise the twin's, so only one output changes.
  - The separately designed occupation map is ABL_OCC_FLAT-like and is reported as the ablation, not as a D5 comparator.

## 4. Plug-in estimate (the existing feasibility world; V0 / V1)

| Release | Pair | Income | Occupation | KL occ (inc) | S3 ratio |
|---|---|---|---|---|---|
| TWIN (4-draw splits) | .6410 / .6419 | .5566 / .5538 | .6276 / .6280 | 0 | 0 |
| **SWAP lambda .05** | .6379 / .6381 | same (bitwise) | .6232 / .6229 | .0090 / .0094 | .82 / .84 |
| DQ at equal total KL | .6384 / .6388 | .5558 / .5550 | .6261 / .6256 | .0050 (.0041) / .0051 (.0043) | .68 / .72 |
| OCC_FLAT at equal KL (lambda_f .10 / .11) | .6367 / .6374 | same | .6221 / .6216 | .0090 / .0094 | .89 / .91 |
| CSF at equal KL (pi .065 / .067) | .6372 / .6382 | .5557 / .5530 | .6232 / .6242 | .0058 (.0033) / .0059 (.0035) | .58 / .60 |
| SWAP .20 (2-draw splits; twin .6394 / .6413) | .6244 / .6277 | same | .6052 / .6079 | .0383 / .0368 | 3.40 / 3.33 |
| ... DQ / OCC_FLAT / CSF at equal KL | .6311 / .6316; .6202 / .6220; .6221 / .6253 | | | | |
| SWAP for everyone vs occupation flattened for everyone | .5925 / .5951 vs .5960 / .5970 | | .531 vs .532 | .198 vs .092 | |

- **Spend per treated person.**
  - The swap costs 0.198 / 0.193 nats of occupation KL, against 0.092 / 0.090 for the occupation decision-cell collapse (2.2x).
  - CSF's two-recipient collapse costs 0.142 / 0.141 (the swap is 1.4x that).
  - Occupation Brier is 0.0805 against 0.0402, and the men's guard 0.080 against 0.040. This is exactly 2x: an independent
    same-cell donor gives E\|\|p' - p\|\|^2 = 2 Var_k.
  - Joint-cell donors (`swap_joint`) are 13% cheaper (0.171 / 0.168), with the same pair (.6259 / .6272 at lambda .20).
- **Pair AUC drop per nat.**
  - SWAP 0.37 / 0.40;
  - OCC_FLAT 0.47 / 0.51;
  - CSF 0.45 / 0.43;
  - DQ 0.19 / 0.26.
- **Mechanism.** Because the occupation decision must be kept, the swap removes the same information as collapsing the occupation
  wire; it only hides who lost it.
  - At lambda = 1 the hiding is worth 0.0035 / 0.0019 on the pair, while the donor noise doubles the cost.
  - The pair link through the kept decision survives: both swapped and flattened for everyone give a pair of about .592-.597,
    against a .577 floor.
  - A donor's wire carries the donor's SEX association, so the occupation own view falls as much as the pair (-0.0044 / -0.0051
    against -0.0031 / -0.0038). Finding 20's "own-view leakage almost unchanged" does not hold.

## 5. Registered predictions, configurations, CPU

Probabilities for the D2 label (SEX-blind route; they sum to 1):

| Label | p |
|---|---|
| NOT_SEPARATED | 0.74 |
| NOT_ASSESSED_TASK_INFEASIBLE (the occupation Brier or a guard fails G1 at the point) | 0.10 |
| NOT_ASSESSED_NO_FEASIBLE_STRENGTH | 0.05 |
| PARTIAL_TWIN_ONLY | 0.03 |
| READER_TARGETED | 0.03 |
| PARTIAL_SPEND_FRONTIER_ONLY | 0.02 |
| INCOMPLETE | 0.02 |
| SEX_BLIND_IMPROVEMENT_BEYOND_TWIN_AND_SPEND | 0.01 |

Point predictions (registered grid; benefit = comparator - SWAP):

| Prediction | p |
|---|---|
| lambda_S3 = 0.05 in both versions | 0.6 |
| lambda_S3 within 0.04-0.06 | 0.8 |
| Pair benefit vs TWIN +0.001 to +0.007 | 0.75 |
| vs DQ -0.003 to +0.004 | 0.8 |
| vs ABL_OCC_FLAT <= +0.002, i.e. the swap no better than flattening | 0.7 |
| Occupation view below the twin's by 0.002-0.008 | 0.7 |
| Income view equal to the twin's (an exact alias; the recipient gate passes) | 1.0 |
| D4 NOT_MET | 0.99 |

Falsifier: the swap beating ABL_OCC_FLAT by at least 0.010 in both versions would mean that hiding who is affected matters for the
benchmark readers. That is the one outcome worth confirming.

**Configurations.**
- CAND 2 + ABL_OCC_FLAT 2 = 4 new.
- The shared SPEND points are reused. At most 2 more if they fail the 25% bracket-fineness rule at the swap's spend.
- At most 6 in total.
- Revision: one, if the reserve allows. R1 `swap_joint` (joint decision-cell donors, 13% cheaper per person), triggered when
  lambda_S3 <= 0.05 in both versions with a guard binding.

**CPU.**
- Mechanism side: the bank hash takes under 1 s, donors under 1 s per 20,000 people, and the nested selection about 30 s per
  version, for under 2 CPU-minutes in all.
- The reader grid is about 200 CPU-s per configuration, so 800-1,200 CPU-s.
- This task used about 80 CPU-s: in-script 3.0 + 3.1 + 34.9 + 29.6 = 70.6 s, plus imports.

**Novelty: NOT_CLAIMED.** The components are data swapping (Dalenius and Reiss 1982), rank swapping (Moore 1996) and PRAM
(Gouweleeuw et al. 1998). Swapping to break record linkage is classical statistical disclosure control. I ran no literature search.
