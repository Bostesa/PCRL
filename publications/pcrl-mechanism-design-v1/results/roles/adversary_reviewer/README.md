# Adversary and methodology reviewer: deliverables

Written on 2026-10-16 by the adversary and methodology reviewer agent. The only files I wrote in the repository are in
this directory. I made no commits, changed no other file and ran no numerical job. I opened no person-level array and
read nothing under any `.private/` directory.

## Files

| File | Content |
|---|---|
| `ACCESS_TABLE.md` | One block per source: NeurIPS paper, AAAI paper, combined manuscript and ACS service. For each, what the paper says, what the code gives the attacker (`file:line`), plausibility, unverified assumptions, the claims that apply, and every paper/code mismatch found. |
| `ACS_SERVICE_ADVERSARY_SPEC.md` | A formal adversary specification for the current ACS output service, with each of the nine coordinator bullets checked against code. |
| `MOTIVATION_DRAFTS.md` | (a) a motivation paragraph for the manuscript; (b) the explanation for the advisor; (c) the pair-versus-single numbers and why they are not a synergy result; (d) prior adversary models with checked links. |
| `MANUSCRIPT_PATCH.md` | Proposed replacement LaTeX for `main.tex` at `3b5065805`: a location table, a unified diff that applies cleanly, and the result of a build check. |
| `PROPOSAL_THREAT_ASSESSMENT.md` | Phase 2: threat-model assessment of designer 1's MMP (sha `2b7273ae…`) and designer 2's RCCS (sha `83653f2d…`), after the reveal. |
| `PRIVATE_NOTES.md` | **Private and git-ignored** (see `.gitignore` here): where the access question arises in the reviews and the transcript. Pseudonyms and line pointers only. |
| `.gitignore` | Keeps `PRIVATE_NOTES.md` out of `git add`. |

## Sources I actually opened

**PDFs** (Read tool page images, plus `pdftotext` for search):

| File | SHA-256 |
|---|---|
| `<NeurIPS submission PDF>` (NeurIPS venue download; **the version used**) | `397d85d5ef4275a9219da54a69a4cbc3a681a94cbdf1e91d50ae987557ed894f` |
| `<NeurIPS submission PDF>` (earlier download) | `4a705d5e77ea83e13e9bb0c9d95a868230cb4b63337c62e7f58a9d363b6d8172` |
| `<AAAI submission PDF>` (AAAI) | `2d86afadd0d1b75194499b35098187124d822851f27a298b6fd174376ff3b077` |

The two NeurIPS files differ in bytes but not in paper text. Both have 26 pages. After normalising whitespace and
removing the venue-inserted reviewer-copy stamping, the body text is identical character for character. The venue copy
was re-stamped with pypdf; the earlier copy is the author's pdfTeX build of 2026-05-07 06:48 EDT. The NeurIPS and AAAI
hashes match those recorded in the private follow-up file.

**Private sources:**
- `<private meeting notes>`, SHA-256 `2bef3bba…`, which matches `SHA256SUMS`.
- `README_PRIVATE.md`, SHA-256 `5abd52f6…`.
- `SHA256SUMS`.

I did **not** open `reviews/<private review matrix>` or anything in `backup_manifest_private/`.

**Code, read only:**
- NeurIPS submission-era code in the PCRL main checkout, read with `git show` at `17ef7d449`, plus commit `5d4eda046`
  on `fix/retire-accuracy-guarantee`.
- The combined manuscript and its code with `git show` at `3b5065805`.
- The campaign worktree at `54f089d3e`: `acs_replication/`, `acs_hybrids/`,
  `task_conditioned_representation/attacks.py`, `task_headroom/reader_worker.py`.
- `Bostesa/durable-guarantees`, cloned read-only into
  `<session scratch>`
  at HEAD `956f5c8` and compared with the submission-era commit `6021b4c`. None of its code was run.

**Published aggregates:**
- `results/pcrl_acs_frozen_hybrids_v1/` — `SERVICE_TABLE.csv`, `ANALYSIS_LOCK.json`, `MANUSCRIPT_CLAIMS_HANDOFF.md`.
- `results/pcrl_acs_protected_tree_replication_v1/` — `SCIENCE_LOCK.json`, `EXTRACTION_SUMMARY.json`,
  `RESEARCH_DECISION.md`.
- `results/v2_adult_ROUND5/per_seed_results.json` at `17ef7d449`, which holds per-pair summaries only.
- Prior-art notes, searched with grep for "collusion" and "sequential".

**Web pages:**
- arXiv abstract pages: 0803.0032, 1905.11742, 2601.21859, 2209.01292, 1709.01604, 1104.0459, 1805.04810, 1606.05893,
  2503.06150, 2210.07213, cs/0610105.
- PDFs, text extracted with `pdftotext`: 1905.11742v3, 1805.04810, and the USENIX Security 2014 paper.
- Crossref records for `10.1145/1401890.1401926` and `10.1109/tkde.2011.124`.
- ACL Anthology D18-1002.

**Build check:** a scratch copy of the manuscript, patched and compiled with `pdflatex` and `bibtex`. Shell use was
limited to `grep`, `sed`, small text-processing scripts and LaTeX.

## Could not access

- The anonymous NeurIPS code snapshot (`anonymous.4open.science/r/PCRL-29E1`) returned HTTP 403. That it equals
  `17ef7d449` is therefore UNVERIFIED.
- The AAAI supplement, which holds Propositions 1–3, is not available locally. The statement of the certified channel
  is therefore UNVERIFIED beyond the main text.
- Some landing pages were unavailable: the ACM DL page returned 403 (I used Crossref instead), the USENIX 2014 landing
  page returned 403 (I used the PDF), dblp was blocked by a bot check, and Semantic Scholar returned 429.
- The incomplete citations in two reviews ("Zhang et al. 2024" in N-R2; "[1,2]" in A-R1) remain unidentified and were
  not guessed.
- CelebA attackers (NeurIPS App. N) were not checked in code. Which post-hoc auditors actually ran in the NeurIPS
  60-cell runs is not recorded in the stored results.
