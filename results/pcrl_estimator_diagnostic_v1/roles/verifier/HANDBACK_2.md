# EDG1 independent verifier: round 2 handback

Campaign `pcrl_estimator_diagnostic_v1` (EDG1). Role: independent verifier; no agents spawned.

Clock, read with `date -u`:
- round 2 started 2026-10-25T17:56:43Z;
- this file written 2026-10-25T18:04:26Z.

Inputs:
- the saved outputs `out/sim_all.jsonl.gz` and `out/bank_all.jsonl.gz` (read only; nothing written to `out/`);
- `SUMMARY.json`, `POST_HOC.json`, and `ESTIMATOR_DIAGNOSTIC.md` at HEAD 523f0bec6.

## Bottom line

- **8 PASS.** Every SUMMARY.json and POST_HOC.json number I reproduced equals my own computation from the saved rows, to 1e-12.
  That covers 14 view entries, 10 comparisons, 4 bank entries and 2 POST_HOC entries.
- **9 PASS.** Three repetitions re-run from their seeds give JSON lines byte-identical to the saved ones. My own C_same and
  B_indep for one of them match the saved values.
- **10.** I checked 64 report items (159 numbers) against the current report. 62 items match. Two items have last-digit rounding
  mismatches:
  - lines 48 and 124;
  - line 100.

  Neither changes a conclusion. Three further mismatches that I found in the commit 6ceaa0b40 text had already been revised in
  the current text.

## 8. Reproduced from the saved outputs with my own code (not `edg.analyze` or `posthoc`)

**(a) View entries.** R, mean, SD (ddof 1), MCSE = SD / sqrt(R), bias against my exact A (brute force over wire pairs) and RMSE all
equal SUMMARY.json:

| cell | view | estimator | mean | bias vs exact A | SD | MCSE |
|---|---|---|---|---|---|---|
| 15 | L1024 | C_same | 0.82264 | +0.14123 | 0.00847 | 0.00038 |
| 15 | L1024 | C_nested | 0.92799 | +0.24658 | 0.00510 | 0.00023 |
| 15 | L1024 | D_nested | 0.60624 | −0.07517 | 0.01778 | 0.00080 |
| 17 | L1024 | C_same | 0.71484 | +0.03343 | 0.00348 | 0.00016 |
| 17 | L1024 | D_bank | 0.65363 | −0.02779 | 0.00717 | 0.00032 |
| 17 | L1024 | B_indep | 0.68124 | −0.00017 | 0.00383 | 0.00017 |
| 5 | pair | C_same | 0.60673 | +0.10673 | 0.00283 | 0.00013 |
| 5 | pair | B_indep | 0.5 | 0 | 0 | 0 |
| 29 | pair | C_same | 0.65559 | +0.01369 | 0.00364 | 0.00016 |
| 29 | pair | D_nested | 0.63108 | −0.01082 | 0.00451 | 0.00020 |
| 41 | pair | C_nested | 0.63712 | +0.07449 | 0.00316 | 0.00014 |
| 41 | pair | D_bank | 0.54234 | −0.02029 | 0.00772 | 0.00035 |
| 48 (clustered) | L1024 | C_same | 0.71472 | +0.03331 | 0.00322 | 0.00014 |
| 48 (clustered) | L1024 | B_indep | 0.68147 | +0.00005 | 0.00363 | 0.00016 |

All rows have R = 500.

**(b) Comparisons**, computed from the saved [d, se_unit, se_person] triples with my own R2SE rule (null iff the exact dA is 0). All
equal SUMMARY.json:
- cell 17, L1024 − L16 (null):
  - C_same: mean d +0.0326, false positives 100%;
  - C_nested: +0.0959, 100%;
  - D_bank: −0.0272, 100%.
- cell 29, pair − income (null):
  - C_same: +0.0126, 100%;
  - D_nested: −0.0084, 98.8%;
  - B_indep: −0.0001, 6.4%.
- cell 28, income − occupation (dA > 0):
  - C_nested: wrong direction 100%;
  - C_same: wrong direction 6.6%;
  - B_indep: power 89.4%, never wrong.
- cell 48, L64 − L16, B_indep false positives: 0.040 with the household SE, 0.036 with the person SE.

**(c) Bank entries.** Winner SCORE mean, winner population mean, selection loss and A − best population AUC all equal SUMMARY.json:

| bank cell | view | winner SCORE mean | winner pop mean | selection loss | A − best pop |
|---|---|---|---|---|---|
| REF\|0.5\|skewed | L16 | 0.6802 | 0.6788 | 0.0012 | 0.0015 |
| REF\|0.5\|skewed | L256 | 0.6728 | 0.6705 | 0.0032 | 0.0077 |
| RED\|0.5\|skewed | pair | 0.6349 | 0.6389 | 0.0018 | 0.0012 |
| COMP\|0.5\|uniform | pair | 0.5656 | 0.5683 | 0.0025 | 0.0121 |

The recorded A also equals my exact A.

**POST_HOC `dbank_target`**, re-implemented over the 13 REF cells (12 independent cells plus cell 48, R = 6,500), equals POST_HOC.json:

| entry | coverage | resolved | mean dpop |
|---|---|---|---|
| REF\|L1024-L16 | 0.9554 | 0.9654 | −0.0701 |
| REF\|L64-L16 | 0.9585 | 0.4005 | −0.0122 |

## 9. Repetitions re-run from their seeds (`edg.simulate.one_rep`, `laws.build`, `scenarios()`)

- **Byte-identical.** `json.dumps(row)` equals the saved line, byte for byte, for:
  - (cell 17, rep 0), with the source check on (keys passed, as in the study);
  - (cell 48, rep 7), clustered;
  - (cell 41, rep 499).
- **Own recomputation for (17, 0)** from `sample_rep`'s sample, using my own merging and exact brute force:
  - C_same equals the saved value: difference 0 at L16 and 1.8e-15 at L1024.
  - B_indep, computed on the float-posterior keys, equals the saved value (1.1e-16).
  - With exact ties, B_indep would differ by +9.9e-6 at L16 and −4.0e-4 at L1024 in this repetition. That is D1, which is
    accepted and disclosed.

## 10. Report numbers against SUMMARY.json / POST_HOC.json

This was 64 items and 159 numbers, prioritising the five answers. A value matches if it rounds to the printed digits; ranges are
checked at both ends.

**Mismatches in the current report** (SHA-256 d2c8967694557457..., HEAD 523f0bec6):

| location | report says | data | should read |
|---|---|---|---|
| line 48 (answer 1) and line 124 (answer 4 table): coverage of D_bank's own target | 94.8–95.9% | POST_HOC dbank_target coverage 94.78–95.85% (maximum REF\|L64-L16 = 0.95846) | 94.8–95.8% |
| line 100 (answer 3 table), COMP pair (240) row, gap and optimism | 0.092 \| 0.075 | SUMMARY cell 41: gap 0.09149, optimism 0.07449 | 0.091 \| 0.074 |

In the COMP pair row, the 81% / 18% shares, the loss of 0.016 and the term of −0.001 are correct.

**Already revised.** I found these in the 6ceaa0b40 text (my first CHECK10 run read that version), and the current text already
has them right:
- RED pair (256) test variation "−0.000" is now −0.001 (the data give −0.00053).
- The cross-fit range "−0.0006 to −0.0027" now starts at −0.0005 (the data give −0.00053).
- "In RED the pair winner was an income-only ignore-one candidate in 10 of 10" is now scoped "at pi = 0.5". For RED|0.2|skewed the
  count is 4 of 10.

**Matched.** Among the items that match:

*Answer 1.*
- 100% P(d > 0) for C_same and C_nested in every REF comparison.
- R2SE 99.6–100%, 99.2% in the clustered cell, and 100% for C_nested.
- RED pair − income: +0.011 to +0.111 (C_same) and +0.045 to +0.229 (C_nested), with 100% false positives.
- IND pair − income: +0.093 to +0.355, with 100% false positives.
- COMP equal-support null: 0–1.0% false positives.
- D_nested / D_bank: P(d > 0) at most 0.29 / 0.11; false positives 14–100% / 26–100%.
- At most 0.8 pp in the other direction (0.6% in the independent cells, 0.8% with the clustered cell).
- Exact-null D_bank 4.2–5.2%; D_nested 13–20%.
- B_indep bias within ±0.0014; B coverage 95.2–95.8%; B null false positives 0–7.0%.

*Answer 2.*
- All 21 bias-table entries, and MCSE at most 0.0005.
- REF L1024 − L16 ranges at n = 20,000, 8,000 and 2,000 for C_same, and at n = 20,000 for C_nested.
- The ceiling values 0.91 / 0.99.

*Answer 3.*
- The IND, REF L1024, REF L256 and RED rows of the table.
- 160 of 160 cell-views negative (−0.00004 to −0.011).
- D_bank −0.0029 to +0.0027; beyond 2 MCSE in 7 of 160.
- The cross-fit term: 12 of 13 views; IND pair −0.0002 ± 0.0003.
- Pooled and per-fold AUCs within 0.0001.

*Answer 4.*
- A − best population AUC 0.0012–0.0121; selection loss 0.0004–0.0034; winner's curse up to +0.029.
- Paired REF L256 − L16: −0.0083 ± 0.0013 and −0.0074 ± 0.0023.
- COMP class limits 0.4999–0.5049, class loss 0.078–0.083; LOOKUP / CATI01 0.573 / 0.577.
- 140 of 140 LOOKUP equalities; 10 of 10 ignore-one winners at pi = 0.5.

*Answer 5.*
- M20 shares 20 / 61 / 100% (C_same) and 60 / 100 / 100% (C_nested); never at or below −0.020 on a nested partition.
- Support-optimism differences 0.03–0.05 and 0.06–0.12.
- D_bank L256 − L16 −0.021; agreement −0.007 to +0.005 over 14 views.

*Further results.*
- IND income +0.044 / +0.022 / +0.014; pi = 0.2 gives +0.049 against +0.033.
- Unseen 37.7%.
- The RED wrong-direction table and the RED pair − occupation ranges.
- COMP D_bank power 5–9% / 26–71%.
- Clustered 0.040 / 0.036.
- 0 undefined values in 24,500 repetitions.
- 495 occupied wires.

## Defect (mine)

The first CHECK10 run (`out/CHECK10_report_numbers.json`, kept) flagged four items because of my own criterion, not the report:
- A1.7, A3.neg160 and A4.class were compared with more decimals than the report prints;
- A1.8 ("at most 0.8") was judged as an equality.

The corrected run `out/CHECK10_report_numbers_r2.json` judges each component at its printed precision and treats A1.8 as an upper
bound. It also judges the current report text. No claimed value was changed.

## CPU and rules

- 1.76 CPU-s recorded:
  - CHECK8 and CHECK9 ran in one process (1.61 s);
  - CHECK10 took 0.07 + 0.08 s.
- About 0.3 s more went to inline inspection commands. One process at a time, one numerical thread.
- The re-runs used `one_rep` in memory only; nothing was written to `out/`.
- `CHECKS.json` was appended with four round-2 entries plus a `round2` block. The round-1 entries were asserted unchanged before
  writing.

## Files (round 2, under `roles/verifier/`)

- `HANDBACK_2.md`.
- `code/check8_9_reproduce.py`, `code/check10_report_numbers.py` (original), `code/check10_report_numbers_r2.py`,
  `code/make_checks_r2.py`.
- `out/CHECK8_reproduce_summaries.json`, `out/CHECK9_rerun_repetitions.json`, `out/CHECK10_report_numbers.json` (kept, FAIL),
  `out/CHECK10_report_numbers_r2.json`.
- `CHECKS.json`, with entries appended.
