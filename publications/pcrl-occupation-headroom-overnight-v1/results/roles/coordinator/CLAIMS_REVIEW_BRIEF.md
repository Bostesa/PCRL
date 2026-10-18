# Brief: claims reviewer (overnight occupation-headroom campaign; issued about 07:55Z)

(Correction: the first committed version of this header said "issued 08:12Z"; the brief was written and the agent launched at
about 07:55Z, before the 07:56:04Z commit 2698a58e7.)

You are the claims reviewer of an exploratory development campaign (PCRL, public-use ACS 2019 Florida PUMS development roles;
records already used by earlier campaigns). The research is finished: the frozen nomination returned NO_QUALIFYING_CONFIGURATION,
DEV_REPORT stayed closed and no ATTACK or DEV_REPORT label was read. Your job is to check the decision documents against the records
before they are published.

Worktree: `<repository>/.claude/worktrees/pcrl-occupation-headroom-overnight-v1`. Campaign folder:
`results/pcrl_occupation_headroom_overnight_v1/`.

## Documents under review

`RESEARCH_DECISION.md`, `ADVISOR_BRIEF.md`, `MANUSCRIPT_CLAIMS_HANDOFF.md`, `START_HERE.md`, `MECHANISM_CARDS.md`.

## Records to check them against (read-only)

- `NOMINATION.json` and `NOMINATION_TABLE.csv`;
- `READER_STABILITY.csv`, `UTILITY_FRONTIER.csv`, `GRID_RECOVERY.csv`, `EXPERIMENT_LEDGER.csv`, `WITNESS_SELECTION.json`;
- `decomposition/*.json`;
- `roles/*/REPORT.md` and `roles/*/CARD.md`;
- `roles/reader_robustness/RR_FINAL__*.csv`;
- `roles/verifier/CHECKS.json`;
- `roles/statistical_reviewer/RULES.json`, `SELECTION_RULE.md` and `CLAIMS_DRAFT.md`;
- `PROTOCOL.md`, `PROTOCOL_DEVIATIONS.md`, `roles/coordinator/FINDINGS_NOTES.md`.

You may also read aggregate private grid summaries in `.private/pcrl_occupation_headroom_overnight_v1/runs/GRID/*/SUMMARY.json` and
evaluation files `.private/.../runs/*/*/S*/EVAL_MS_*.json`. Do NOT open person-level arrays (`*.npz`), label files, ATTACK or
DEV_REPORT material.

## What to check

1. Every number in the documents equals its record: same seed, same comparator, same sign convention. Benefit is comparator AUC
   minus candidate AUC, so positive favours the candidate. Report each mismatch with the correct value and its source.
2. Every claim respects the campaign's interpretation rules:
   - deterministic does not imply one-to-one;
   - an invertible recalibration may change usefulness or fitted-reader behaviour without removing information;
   - lower recovery for the declared readers is not protection against every possible reader;
   - a point pass is not a population guarantee;
   - an unresolved condition is neither an established violation nor equivalence;
   - the venue target (0.020) does not change scientific criteria;
   - with only occupation optimized, the pair term is "a coalition-aware occupation correction", never a two-output joint design;
   - rotated splits of previously used people are not fresh confirmation;
   - no method win is claimed.
3. The claim ladder in CLAIMS_DRAFT.md: no registered final claim (W, E, R, P, N, C, M, T, O) may be stated as evaluated. J is
   NOT_CLAIMABLE and V is NOT_CLAIMED.
4. Both seeds are given wherever a seed-specific number is quoted. The strongest favourable and adverse comparisons are both present.
   The distance to each governing requirement is stated correctly:
   - 24 utility clauses (accuracy ≤ 0.010, NLL ≤ 0.010, summed Brier ≤ 0.005);
   - SEX-group guards ≤ 0.005;
   - 2-SE reserve;
   - recipient gate ≤ 0.010 per cell;
   - the 0.020 target, reported separately.
5. Omissions that would mislead a reader: a missing caveat, deviation or failed check that should be cited.

## Rules

- Do not edit the documents. Return your findings as the text of your final response, ordered by severity, each with the document,
  the exact current text, the correction, and the source record. Also list what you checked and found correct.
- No queue units are needed; if you run a small read-only script, keep it under a few CPU-seconds and use only `python3`.
- Platform requirements and tool permissions apply. If something is refused, say so and continue.
- Hand back by about 09:00Z.
