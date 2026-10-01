# Run status: combined empirical preparation v1

**State: COMPLETE (preparation phase).**
Last updated 2026-10-01. Owner/integration session, branch `research/combined-empirical-preparation-v1`.

## Steps

| Step | Status |
|---|---|
| Discovery and indexing of both repositories | done (BOTH_REPOS_INDEX.json) |
| Literature A: reviewer-cited and mandatory papers, 8 read in full | done |
| Literature B: systematic search, 31 included | done |
| Methods catalog (51 methods) | done |
| Methodology audit: durable-guarantees | done |
| Methodology audit: PCRL encoder lineage | done |
| Methodology audit: PCRL ACS lineage | done |
| Independent verification fixtures F1–F10 | done; rerun byte-identical on 2026-10-01 (88 s) |
| Integration into the 14 required files plus CLAIM_LEDGER.csv | done |
| Commit and push of the preparation branch | done; pushed, remote SHA verified (HANDOFF.json `git`) |

## Agent roles

Seven native subagents ran in parallel: literature A, literature B, methods, methodology ×3 and
verification. Each wrote only to its own `notes/<role>/` (verification also wrote `fixtures/`).

The harness refused subagent writes of `SUMMARY.md` and `FINDINGS.md` for several roles. The owner saved
that text verbatim from the roles' final reports and marked each file accordingly.

## To resume or re-verify

1. Recreate the session-scratch inputs that `fixtures/run_all.sh` expects (paths are hard-coded to the
   2026-10-01 scratchpad; edit `SCR=` in run_all.sh for a new location):
   - `git clone https://github.com/Bostesa/durable-guarantees $SCR/dg && git -C $SCR/dg checkout 956f5c883f515646aa457db55ecbd74b913768b2`
   - `git -C /Users/nathansamson/PCRL archive origin/main pcrl tests | tar -x -C $SCR/verify/origin_main`
     (create the target directory first)
   - `git -C /Users/nathansamson/PCRL archive research/pcrl-submission-finish-v1 pcrl tests | tar -x -C $SCR/verify/research_pcrl-submission-finish-v1`
   - `$SCR/verify/pcrl_results`: stored PCRL results JSON used by F08. See
     `fixtures/F08_recount_headlines.py` for the files read.
2. Run `sh fixtures/run_all.sh` and compare `fixtures/outputs/*.json` with the committed copies.

## Not done (deliberately out of scope for this phase)

- No model fits, attacks on stored representations, cloud jobs or new data.
- No merge to main.
- No changes to any other branch or worktree.
