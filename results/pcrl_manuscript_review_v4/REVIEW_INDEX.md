# REVIEW_INDEX — v4 manuscript and review package

Branch `research/pcrl-manuscript-integrated-v4`, from `ce5ba24a0a0ed1848c4029a922ac66fa629d71fc` (v3).
Study 4 integrated from `research/pcrl-direct-adversarial-v1` evidence `69e790af36c5ca53203dab17b757a8e3415ee934`
(published head `106de9af…`). Terminal 1's diagnosis at `4f09e63c6` on `research/pcrl-competitive-method-v1`
read and verified. All source branches, v3 bytes and historical study directories preserved. Zero ACS model fits.

## Read in this order
1. `CORRECTIONS_V4.md` — C1/C2 (the Study 4 positive is narrower than reported), C3 (A0 not J).
2. `main.pdf` §9 (Study 4), §10 (pending programme and what a competitive claim needs).
3. `MATHEMATICAL_REVIEW.md` — M1, M2, M6, M8 are the items to fix before projection fits.
4. `CLAIM_LEDGER.csv` — W01–W20 are new.

## Deliverables
| File | What |
|---|---|
| `papers/pcrl_manuscript_v4/main.tex`, `main.pdf` | 25 pp.: main text 1–16, appendices A–J 17–24, refs 24–25 |
| `papers/pcrl_manuscript_v4/tables/`, `figures/` | 20 new generated assets (14 tables incl. notes, 3 figures × pdf/png); 48 carried byte-identical from v3 |
| `papers/pcrl_manuscript_v4/MANIFEST.json` | sha256 of every asset, generator, v3 identity |
| `CLAIM_LEDGER.csv` | 48 claims (28 carried, 20 new) with provenance |
| `SOURCE_MANIFEST.json` | pinned commits, sha256 of every evidence file read |
| `CORRECTIONS_V4.md` | C1–C12, additive |
| `MATHEMATICAL_REVIEW.md` | M1–M10, blocking vs disclose |
| `COMPARATOR_NOVELTY.md` | positioning, four claim levels, private-overlap status |
| `VALIDATION_SUMMARY.md` | every check and what was not checked |
| `PENDING_INTEGRATION.md` | Study 5 preconditions and commands |
| `STUDY4_VERIFICATION.json`, `SELECTION_AUDIT.json`, `SELECTION_RESCORING_CHECK.json`, `DISPLACEMENT_CHECK.json`, `MATH_FIXTURES.json` | machine-readable check outputs |
| `checks/verify_study4.py`, `checks/build_assets.py`, `checks/math_fixtures.py` | generators |
| `HANDOFF.json` | public handoff record |

## What changed from v3
- Appendix G "pending" replaced by a main-text Study 4 section (§9), full tables in App. G.
- Title/abstract: four negative method results; pending fifth programme reviewed before outcomes.
- Chronology: five studies; new chronology figure.
- New figures: per-endpoint frontiers (Fig. 4); selection-vs-training movement with an explicit pending panel (Fig. 5).
- Coalition result restated at both correction levels, including the omitted strength-matched width-8 row.
- New §10: pending programme, mathematical review summary, four claim levels; detail in App. H.
- Two alternative conclusion paragraphs prepared; neither outcome observed.
- v3's 31 corrections moved to App. I; v4 corrections in §12.

## Editorial length
Main text is ~16 pages against a ~10–11 page editorial target (not a verified venue limit). Reaching it
would mean compressing Studies 1–3 (§§6–8) to a page with details in appendices; not done in this revision
because it trades against keeping outcomes visible. Flagged, not hidden.
