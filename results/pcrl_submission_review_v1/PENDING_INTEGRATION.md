# PENDING_INTEGRATION — nothing remains pending integration

**The prospective ACS 2016 study is complete and integrated.** Evidence commit
`5e154e5c4fdaeb23d327a0ebefe838525f1a19cb` on `research/pcrl-final-prospective-v1`, remote-verified;
handoff `.git/pcrl_finish_handoff_v1/terminal_1/FINAL_HANDOFF.json`; lock commit `0b65a1cd0` with
evaluation-lock SHA-256 `e17b8167…` created before any final label was read.

Integrated into §VII of the paper, Table IV, Figure 3, the claim ledger (rows `P16-1`…`P16-8`) and
`PROSPECTIVE_2016_VERIFICATION.json` (21 checks, all re-derived from estimates and bootstrap standard
errors rather than read from status strings).

Outcome, in the fixed vocabulary of the earlier outcome map: **neither candidate passes.** Q and D17 each
pass all eight registered sensitive-recovery clauses against J and fail both task clauses, so each is
8/10 and neither establishes the full conjunction. Per that map, we report the useful task result and the
exact unresolved bounds, and we do not turn an unmet margin into demonstrated harm.

## Superseded statements

* "Awaiting Terminal 1" — superseded; the study is done.
* "The protocol review is outstanding" — superseded by `PROTOCOL_REVIEW_RETROSPECTIVE.md`, which is
  **retrospective** and labelled as such. It was written after outcomes and is not backdated.

## What is genuinely still open (not integration items)

| item | owner | note |
|---|---|---|
| Genuine prior reviews for NeurIPS submission 32955 | author | reviews are confirmed to exist (handles AkJK, NY7k, AC); their text is in no local record. Exact request in `SUBMISSION_READY_CHECK.md` |
| Lineage decision: same paper or successor? | author | decides whether the prior-review appendix is required at all |
| Registration facts: authors, affiliations, ORCIDs, certification, topics, conflicts | author | fixed at abstract registration |
| Abstract wording now that 2016 has landed | author + venue rules | proposed text in `REGISTRATION_WORDING.md`; **not** applied to any registration |
| `fix/retire-accuracy-guarantee` | author | pushed at `5d4eda046`, deliberately **unmerged**; `origin/main` still ships the retired API |
