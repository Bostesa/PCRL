# SUBMISSION_READY_CHECK

Live call verified 2026-09-22: <https://satml.org/call-for-papers/>, <https://satml27.hotcrp.com/>.
**Nothing has been submitted, registered, certified or uploaded.** Those are the author's actions.

## Abstract registration — due TODAY, 22 Sep 11:59 PM AoE = 23 Sep 11:59 UTC

| item | status |
|---|---|
| Title | **READY** — `REGISTRATION_READY.md` |
| Abstract (228 words, outcome-neutral) | **READY** |
| Topics recommendation | **READY** |
| Non-blank title and abstract in HotCRP | **AUTHOR ACTION** |
| Author list and order | **MISSING — author must supply**; fixed after registration |
| Affiliations | **MISSING — author must supply**; fixed after registration |
| ORCID for every author | **MISSING — author must supply** in each author's HotCRP profile |
| Author certification for every author | **MISSING — author must complete** in HotCRP |
| Conflicts of interest | **MISSING — author must declare** |

If registration does not happen by the deadline, the paper is not eligible and no workaround exists.

## Paper submission — due 29 Sep 11:59 PM AoE

| item | status |
|---|---|
| `\documentclass[conference]{IEEEtran}`, 10pt, two columns, unmodified margins | **PASS** |
| Body within 12 pages | **PASS** — body ~5 pages; references and appendices are outside the limit |
| Central evidence in the body, not relegated to appendices | **PASS** — the primary comparison, the release table and the audit result are all in the body |
| Anonymous (no names, affiliations, acknowledgements) | **PASS** |
| Own prior work cited in third person | **AUTHOR DECISION** — see `ANONYMIZATION_CHECK.md` risk 1 |
| Open Science section immediately before references | **PASS** |
| LLM usage considerations section after Open Science | **PASS**, and it discloses substantial model assistance in design, code, drafting and auditing |
| Compiles clean, no undefined references or citations | **PASS** |
| Figures legible, tables not crowded, captions match generating evidence | **PASS** — every page inspected |
| **Prior reviews appended after all appendices** | **MISSING — BLOCKING IF THE LINEAGE ANSWER IS "same paper"** (below) |

## The prior-review item, precisely

The venue requires the complete, unedited (anonymised) reviews of the **latest previous submission of the
same paper**, plus how they were addressed.

**Established:** the earlier paper exists and was submitted — "One Encoder, Many Purposes: Purpose-Conditioned
Representation Learning with Per-Purpose Linear Leakage Bounds", NeurIPS 2026, PDF created at the submission
deadline, local file `32955_One_Encoder_Many_Purpose.pdf`.

**Not found:** its venue reviews. Searched the repository at the pinned commit, `~/Downloads` (1,395 files)
and `~/Documents`. Everything review-shaped that exists is either agent-simulated or belongs to a different
paper ("Outputs Leak What They Use"), as itemised in `REVIEW_TO_EVIDENCE.md`. **No agent-written review may
be substituted, and none has been.**

**Author must supply, if the reviews exist:** the full review text for submission 32955 (all reviewers, plus
any meta-review), exported from the venue system. The response mapping is already drafted in
`REVIEW_TO_EVIDENCE.md` and can be completed within an hour of receiving them.

**Author must also answer:** is this submission *the same paper*? The contract, mechanism, data and results
all differ; two contributions (the dominant-axis audit and the composition/guarantee corrections) are reused.
A defensible answer is "a successor paper reusing two contributions", in which case no prior-review appendix
is required. That answer must be the author's honest judgement of content, not a consequence of retitling.

## Overlap with the concurrent AAAI manuscript

Checked against the actual supplied draft and supplement. Different problem (auditing attribute-removal
certificates), different data (Adult/HMDA/CelebA), different claims; shared vocabulary only. **No
substantive overlap and no dual-submission conflict is asserted** — current submission status of that paper
is unknown to this terminal and is flagged in the private handoff rather than assumed.

## Artifact

| item | status |
|---|---|
| Anonymous artifact directory and archive built | **PASS** — `artifact/pcrl_satml_anon/` and `.tar.gz` |
| Scanned for identifying strings | **PASS** — 0 hits |
| Public data recipe included | **PASS** |
| Uploaded to an anonymous host | **AUTHOR ACTION** — due within 3 days of submission |
