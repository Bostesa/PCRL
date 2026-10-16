# Which manuscript source this revision starts from

## Search

- **Repository instructions.** The repository has no `AGENTS.md` (and no `CLAUDE.md`) at its root.
- **Git provenance.** Across every local ref of both local clones (the primary clone, whose remote is GitHub, and a
  second working clone used by later campaigns, whose `origin` is the primary clone), only two commits ever touched
  `papers/combined_empirical_v1/`:
  - `90253ed39` (2026-10-03), which added the useful-head results, corrections, prior-work review and figures;
  - `5150f42e8` (2026-10-03), which completed the combined analysis paper.
  The branch `research/combined-analysis-paper-v1` ends at `3b5065805`, which records the backup and handoff only and
  leaves the manuscript unchanged. GitHub's copy of that branch also points to `3b5065805`. The study handoff
  (`results/combined_analysis_paper_v1/HANDOFF.json`, `QUICKSTART.md`) names this manuscript and its build command
  (`latexmk -pdf main.tex`). No later handoff names another manuscript source.
- **The later revision.** A later revision is reported to exist as `PCRL_Combined_Empirical_Revised.pdf` and
  `PCRL_Combined_Empirical_Revised_Source.zip`. No file with either name, and no file whose name contains
  `Combined_Empirical`, was found on this host (Spotlight and a file-system search of the user's folders). The only
  similarly named local files are May 2026 drafts of the earlier single-venue paper in NeurIPS format (`PCRL_revised`),
  which are a different manuscript. External accounts were not searched.

## Decision

The newer revision is unavailable here, so this revision starts from the latest verifiable source:
`papers/combined_empirical_v1/` at commit `3b5065805`. It does **not** update, and does not claim to update, the
unavailable `PCRL_Combined_Empirical_Revised` version.

**Outstanding version differences:** unknown. Whoever holds the newer files should diff them against both
`3b5065805` and this revision, then port any improvements made there that this revision lacks.

## Admitted source (unchanged in `papers/combined_empirical_v1/`)

| File | SHA-256 |
|---|---|
| `main.tex` | `76206d1b3b3b48dd87ab1663cf7694aa320bad9183101970f6be62417ee91d65` |
| `references.bib` | `86944628be53c27c3c11001406e4b20a294baf3e33142c68e668b570fd818c02` |
| `fig_recipient.tex` | `2474740e27850c58d024cf28ae75332a3bafcedcbfee42480fbfe7f006aa6254` |
| `main.pdf` (10 pages) | `7e4b7ac4b20ce13db25b5ddd3a73a0a57d07b66f21406628725bdefe5021b744` |
| `figures/fig_formats_pairs.pdf` | `9be104dfcca6b0af40590781ecf800824d39c703792ea6db18b27f330628d5a2` |
| `figures/fig_useful_head.pdf` | `b7827b7750bb12597b951ee0747377d009820ef996863ca464d515aa8a3bccdd` |

## Adversary patch v2

- **Source:** `publications/pcrl-mechanism-design-v1/results/MANUSCRIPT_ADVERSARY_PATCH.diff`, public commit
  `c9709a2`, SHA-256 `3d296d2be026c381d8c5b3fa7bd6b3c6dc1d7ecd51d0f3efc32526f2b9d4e068`.
- **Verified base:** `3b5065805`, which is the admitted source, so the patch was applied directly (paths retargeted to
  `papers/combined_empirical_v2/`).
- **Result:** `main.tex` `cf0b25f2323b8b78c23af915505d405473a0b3df330f443726e7d44e28918c86` and `references.bib`
  `b7043344021b75388abf43e5292305eb6ae13c2c9b309ee793830c616881fea1`. Both equal the outputs recorded when the patch was
  verified.
- **Commits:** `bf8f768af` (verbatim copy into `papers/combined_empirical_v2/`) and `133d2c740` (patch applied).
  Every later edit builds on `133d2c740`. The final hashes are in `HANDOFF.json`.

## Evidence cutoff

- **Numbers.** The numeric record is that of the admitted manuscript: four stored-model experiments, the last of which
  (the useful-head comparison) was locked at `a0de44953` with results at `90253ed39` and `5150f42e8`, all 2026-10-03.
  No number was added or changed.
- **Later work.** The later census (ACS) studies, the rejected mechanism proposals and the other negative method results
  are not added to the paper.
- **Later corrections.** These are used only where they repair a claim the paper makes:
  - the adversary and access text (patch v2 and the 2026-10-16 correction);
  - the scope of Stadler et al.'s theorem;
  - two bibliography entries.
