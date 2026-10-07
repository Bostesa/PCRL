# Port log — lcr → lra

`lra/` is a port of `lcr/` at the lcr final tip 091afc2 (lcr evidence 418e529). The port did three things:
- renamed `lcr` to `lra` in module names, schemas and private-store names;
- moved the results path to `results/pcrl_adult_learned_decoder_release_v1`;
- set the branch to `research/pcrl-adult-learned-decoder-release-v1`.

The `lcr/` package and the lcr results stay unchanged.

The draft registration documents were copied from the lcr package and are then edited for this study:
LABEL_TRUTH_TABLE, SELECTION_RULES, PRIMARY_FAMILY, SEARCH_RULES, MATH_REVIEW, the METHOD_CARD parts, the prior-art and
fairness reviews, AUDIT_COMPUTE, TIMING, SOURCE_INDEX and FIT_MANIFEST. The source review of the selection stack is at
source_reviews/SELECTION_STACK_REVIEW.json.

Pinned unchanged copies:

| File | Source | sha256 |
|---|---|---|
| FIXTURE_LAWS.json | lcr's file | 24f70745… |
| SOURCE_FIXTURE_GATE.json | lcr's FIXTURE_GATE.json, the historical GATE_NOT_MET | 772f17e3… |

## Edits after the port (owner, file, reason)

- A, lra/lock.py: new pins (lcr evidence/tip, cbp tip); ORDER = SOURCE_ADMISSION_LOCK, CORRECTNESS_LOCK, SCIENCE_LOCK;
  the correctness stage; the new DOCS list; the verbatim exposure statement; lcr added to GLOBS and TOP, so an
  accidental lcr import must be locked.
- A, lra/run.py: the `fixture` stage became `correctness` (lra.fixtures.stage_correctness).
