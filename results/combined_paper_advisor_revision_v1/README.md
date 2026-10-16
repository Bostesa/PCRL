# Advisor-ready revision of the combined empirical paper

This revision improves the paper and states a proposed scope for a future method. It establishes no new algorithm, and
nothing scientific was run: no defense training, attacker fitting, new statistics, real-person array reads, population
admission or sealed-assessment access.

## Read in this order

1. `../../papers/combined_empirical_v2/main.pdf`: the revised manuscript (17 pages). Its source is `main.tex`,
   `references.bib` and `fig_recipient.tex` in the same folder.
2. `ADVISOR_BRIEF.pdf` (source `ADVISOR_BRIEF.tex`): two pages for the advisor covering:
   - the question;
   - the adversary and access;
   - what the results establish and what remains unproved;
   - the proposed method objective;
   - the decisions needed.
3. `METHOD_SCOPE_PROPOSAL.md`: the proposed method objective and its distinctions. Benchmark definitions are unchanged,
   and no experiment is authorised.
4. `CHANGE_NOTE.md`: what changed in the story, in the adversary definition and in the numbers, round by round.
5. `CLAIM_SOURCE_CHECK.md`: the claim-to-source check for the revised claims (51 rows), with the claims reviewer's
   verdict.
6. `SOURCE_RESOLUTION.md`: which manuscript was revised, the hashes, the evidence cutoff, and the newer version that was
   unavailable.
7. `MANUSCRIPT_DIFF.diff` (unified) and `MANUSCRIPT_WORD_DIFF.txt` (word level): the revision against the admitted
   manuscript in `../../papers/combined_empirical_v1/` at `3b5065805`.
8. `EVIDENCE_EXTENSIONS.md`: possible extensions, listed separately; none was run or authorised.
9. `HANDOFF.json`: commits, hashes, commands, backups and unresolved dependencies.

## Reviews (suggestions only; the owner applied them)

- `review/ACCESS_FACTS.md`: the adversary reviewer's per-experiment access facts, with file and line citations.
- `review/ADVERSARY_REVIEW_DRAFT.md`: the adversary reviewer on the revised access and motivation.
- `review/CLAIM_SOURCE_INDEX.md` and `review/CITATION_PRECHECK.md`: the claims reviewer's source index and its
  primary-source citation checks.
- `review/CLAIMS_REVIEW_DRAFT.md`: the claims reviewer on the revised claims.
- `review/INDEPENDENT_READER.md`: a reader who saw only the PDF and none of the project history.

## Rebuild

```
cd papers/combined_empirical_v2 && SOURCE_DATE_EPOCH=1792108800 FORCE_SOURCE_DATE=1 latexmk -pdf main.tex
cd results/combined_paper_advisor_revision_v1 && SOURCE_DATE_EPOCH=1792108800 FORCE_SOURCE_DATE=1 pdflatex ADVISOR_BRIEF.tex  # run twice
bash results/combined_paper_advisor_revision_v1/backup_and_restore.sh <backup-dir>   # backup plus byte-identical rebuild check
```
