# REVIEW_INDEX — manuscript v6

Branch `research/pcrl-manuscript-integrated-v6` (base v5 `1d18e898`). PDF:
`papers/pcrl_manuscript_v6/main.pdf` (28 pages; main text pp. 1–12, appendices pp. 13–28).
**Zero new ACS model fits.** Table arithmetic ran with one CPU worker and one BLAS thread.

## Read in this order
| file | what it is |
|---|---|
| `CONTRIBUTION_ASSESSMENT.md` | strongest supported claim; what the sixth study adds; remaining gap |
| `CORRECTIONS_V6.md` | D1–D6, additive (C1–C10 of v5 carried) |
| `CLAIM_LEDGER.csv` | 71 rows: v5 rows carried (W20 updated), S6-01…S6-10 new |
| `STUDY6_VERIFICATION.json` | 118 independent checks; the pilot's pass/fail legs re-derived from stored per-anchor increments, not read from its flags |
| `../pcrl_manuscript_review_v5/` | Study 5 verification (113 checks), math review, v5 corrections — all still current |
| `MATHEMATICAL_REVIEW.md` | how the v5 review items were answered by the executed study |
| `NEXT_STUDY_NOTES.md` | what a follow-on prospective protocol would need; kept out of the PDF |
| `SOURCE_MANIFEST.json`, `ASSET_HASHES_STUDY6.json`, `../../papers/pcrl_manuscript_v6/MANIFEST.json` | provenance hashes |
| `MANUSCRIPT_STORAGE_LEDGER.md` | Terminal 2 storage actions and what remains pending |
| `checks/verify_study6.py`, `checks/write_manifests.py` | generators |

## What changed from v5
* Study 6 integrated as §9: tier ladder (T0/T1 pass, T2 pilot fail, T3/T4 **not triggered**), the inclusion
  fact that bounds what an appended channel could ever show, the utility-first reanalysis of the 372-slot
  evidence (0 of 124), the pilot screen table and the capability-versus-disclosure figure, no coalition
  advantage, and a third failed search for a stronger attacker.
* Title, abstract, contribution list, chronology figure/table and conclusion updated from five negatives to six.
* New limitations: screening thresholds are not confidence statements; audits do not transport across
  architectures below ~0.003 nats; an unrun tier is not evidence; a deterministic append cannot protect.
* Compression to hold length while adding a study: related work and the Study 5 findings condensed, the
  Study 5 contrast figure, the Study 6 tier table and the claim-support table moved to appendices.

## Verification actually run
* `verify_study6.py`: 118 checks — counts and cumulative slot accounting, per-configuration re-derivation of
  all three screen legs, monotonicity in β, the closest configuration under both weightings, the tier-1
  reanalysis recounted from `per_config`, calibration and positive control, portability tolerance. All pass.
* PDF: 0 errors, 0 undefined references, 0 overfull boxes; changed pages rendered and inspected.

## Not verified here
* Study 6 per-person losses and any interval (none exist: nothing was nominated).
* Terminal 1's AWS archive bytes (verified by Terminal 1 on AWS: 33 chunks, 164,927 files, 0 bad).

## Editorial length
Main text is 12 pages against the ~10–11 page **editorial** target (not a verified venue limit). v5 was 11
pages with five studies; v6 adds a sixth study and its evidence while removing about 1.5 pages of prose.
Flagged, not hidden: further compression would have to cut scientific qualifications or the strongest
negative comparisons, which the brief protects.
