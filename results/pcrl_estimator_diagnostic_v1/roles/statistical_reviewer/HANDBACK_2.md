# Statistical reviewer, hand-back 2 (verbatim)

The reviewer's final message of round 2 was received at about 18:00Z. Its introductory paragraphs follow verbatim. The "Full text of
REVIEW_REPORT.md" that followed them in the message is identical to the final file `REVIEW_REPORT.md` in this folder. The
coordinator checked the distinctive final wording ("one merge per step from 80") and recorded its SHA-256 at 18:02Z:
0a03cfd4502c3fcf336882454db64449180e2658fb972f65515bb52e1591df87.

> The draft has resolved all five of my round-1 REQUIRED items in substance (a2, a3, a4, e2, f1), but I found three new REQUIRED errors.
>
> 1. **Answer 5's main new DRF1 reading (c1).** The draft says the matched coarsening point's support is unknown. DRF1's published records give it: 40 / 35 pair wires, against OPT_L's 374 / 398, roughly a tenfold difference. So the confound runs against OPT_L. The nested "not useful" reading is not evidence against OPT_L, and "in either direction" has to go.
> 2. **The bottom line on LOOKUP and sample size (b1).** "More often ... the smaller the sample" is contradicted: D_bank's false-positive rate rises with n, and D_nested's falls only for L64 − L16.
> 3. **Bank ignore-one winners (d1).** "10 of 10" holds only at pi = 0.5; at pi = 0.2 it was 4 of 10.
>
> None of the three changes DRF1's no-method conclusion or requires re-running anything. I checked more than 45 report numbers: all match except the 9 in section 3 (3 of them the errors above). There are also 14 RECOMMENDED fixes, mostly scoping, wording and post hoc labels.
>
> I wrote only `D/roles/statistical_reviewer/REVIEW_REPORT.md`, made no git changes, and ran three read-only scripts on one thread (under 10 CPU-s in total).
>
> Full text of REVIEW_REPORT.md: [identical to the file; see above]

## Coordinator's disposition (18:02Z)

All three new REQUIRED items and all 14 RECOMMENDED items were applied in `ESTIMATOR_DIAGNOSTIC.md`:
- **c1.** The coordinator re-checked the supports from DRF1's published FIT and OPTIMISTIC records: step 67 / 68 has 40 / 35 pair
  cells; OPT_L has 374 / 398.
- **b1, d1, b2-b9, c2-c5, d2-d4.** Applied as stated.
- **NOTE rows.** The cross-fit range, the RED pair third term, B_indep's MCSE count and D_bank's per-cell exact-null range were
  corrected; h1-h4 were applied.

No result was recomputed.
