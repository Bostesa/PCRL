# REVIEW_INDEX — manuscript v5

Branch `research/pcrl-manuscript-integrated-v5` (base `72a63833`, manuscript v4). PDF:
`papers/pcrl_manuscript_v5/main.pdf` (25 pages; main text pp. 1–11, appendices pp. 12–25).
**Zero new ACS model fits.** Table arithmetic ran with one CPU worker and one BLAS thread.

## Read in this order
| file | what it is |
|---|---|
| `CONTRIBUTION_ASSESSMENT.md` | strongest supported claim; method status after five studies; remaining gap |
| `CORRECTIONS_V5.md` | C1–C10, additive; includes a v4 table transcription error (C6) and Study 5 wording narrowed (C3–C5, C7) |
| `CLAIM_LEDGER.csv` | 61 rows: v4 rows carried (V25/W20 updated), S5-01…S5-13 new; commit, file, key, pool, family, value, caveat |
| `STUDY5_VERIFICATION.json` | 113 independent checks of Study 5 numbers, all agreeing; recomputed conjunction |
| `MATHEMATICAL_REVIEW.md` | A: how the rank-deficiency and fixed-offset issues were resolved; B: review of the utility-first extension before outcomes |
| `PENDING_INTEGRATION.md` | Study 6 state (tier 0, AWS blocked), checks, commands, four alternative texts kept out of the PDF |
| `SOURCE_MANIFEST.json`, `ASSET_HASHES_STUDY5.json`, `../../papers/pcrl_manuscript_v5/MANIFEST.json` | provenance hashes |
| `REPRODUCE_PAPER.md` | how to regenerate |
| `MANUSCRIPT_STORAGE_LEDGER.md` | what Terminal 2 deleted, what survives, what is pending on Terminal 1 |
| `checks/verify_study5.py`, `checks/write_manifests.py` | generators |

## What changed from v4
* Study 5 integrated (§8): negative verdict; J initialisation tested; projection nominees trade residence
  like LEACE on J; no coalition advantage over matched local projections on J; A0 k=2 race effect exploratory
  (family X); stronger attack uninformative; 2017 exploratory; amendment-4 timing printed.
* The v4 "pending" table and alternative-conclusion paragraphs removed from the PDF; one conclusion.
* New §9: two forms of competitiveness; utility-first criterion stated prospectively; Study 6 carries no outcome.
* Compressed Studies 1–4 into §§6–7; full tables and Study 2–4 detail moved to appendices E–H.
* New figures: chronology strip (Fig. 2), Study 5 per-endpoint frontier (Fig. 4, person-weighted Fig. 8),
  Study 5 contrasts with candidate-wide intervals, both weightings (Fig. 5). Tables 12–17 for Study 5.
* Main text now 11 pages (v4: ~16) against the ~10–11 page **editorial** target (not a verified venue limit).

## Verification actually run
* `verify_study5.py`: 113 checks (denominators, seed means vs addendum, P/X interval verdicts, conjunction,
  2017 means, stress, supported rank, family-X counts). All pass.
* PDF compiled with latexmk: 0 errors, 0 undefined references, 0 overfull boxes; every page rendered once at
  low resolution and inspected; defects fixed (axis-label overlap, float drift past references, wide tables).

## Not verified here
* Study 5 per-person bootstrap replicates (not re-run; intervals read from committed CSVs).
* Stress per-person losses (not stored by Study 5; seed means only).
* Anything of Study 6.
