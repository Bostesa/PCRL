# SaTML 2027 call — verified requirements

**Source:** https://satml.org/call-for-papers/ (IEEE SaTML 2027, Reykjavik, May 2027; program chairs
Fabio Pierazzi and Florian Tramèr). Cross-checked against https://satml.org/ and the HotCRP instance
https://satml27.hotcrp.com/ referenced from the site.
**Verified:** 2026-09-21 by Terminal 2. Re-verify before any submission action; deadlines are AoE.

| item | requirement as published |
|---|---|
| Abstract registration | Tue, 22 Sep 2026 (mandatory) |
| Full paper | Tue, 29 Sep 2026 |
| Artifacts (anonymized) | Fri, 2 Oct 2026 |
| Template | `\documentclass[conference]{IEEEtran}`, default 10pt. "Using a different template, or modifying font size, margins, or spacing to fit more content, is grounds for desk rejection." |
| Body limit | Research/SoK: up to **12 pages of body text**. Position: 5–12. |
| References / appendices | **No limit.** |
| Anonymity | Double-blind; omit author/institutional references, cite own work in third person, avoid identity-revealing artifact statements. |
| Authors / ORCID | All authors provide ORCIDs; Author Certification confirmed in HotCRP **by the abstract registration deadline**. Affiliations fixed at abstract registration; authors cannot be added or removed afterwards (grounds for desk rejection). |
| Title/abstract after registration | "no substantial changes are allowed to abstract or title". |
| Open science | Anonymized artifacts shared within 3 days of submission; accepted papers deposit final artifacts on Zenodo; violation risks desk rejection. |
| LLM usage | An **"LLM usage considerations" section is mandatory if any LLM was used**; authors remain "ultimately responsible for all submitted content and results"; must address accountability, transparency, responsibility. |
| Prior reviews | If previously submitted elsewhere, append the previous venue's reviews at the end, anonymized but otherwise unedited, describing how the feedback was addressed. |

## Consequences for this package
1. The title and abstract must be final-quality **before** 22 Sep, because substantial changes are barred
   afterwards. Neither may promise a result this project has not measured.
2. The 12-page limit binds the **body only**; the full matrices, per-study tables and the correction log
   move to appendices without penalty.
3. The LLM section cannot describe this work as lightly edited by a model: the experiments, analysis,
   audit and drafting were substantially model-assisted, and the section says so.
4. Prior-review appendix: applies only if this manuscript was actually submitted somewhere before. See
   `PRIOR_SUBMISSION_CHECK.md`; do not invent a prior submission.
