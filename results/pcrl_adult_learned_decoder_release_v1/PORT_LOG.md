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
- A, lra/admit.py: lcr_manifest_check (each planned unit and admitted dir must equal lcr SOURCE_ADMISSION.json at
  091afc2, read via git show); the result is recorded in the receipt.
- A, lra/run.py: SCIENCE_STAGES require engineering_ready(), meaning a lock-bound, pushed ENGINEERING_GATE_RESULT.json
  with verdict ENGINEERING_READY (tests: lra/tests/test_run_gate.py). Also the CLASS|D1 diagnostic decode as
  diag__s{k}__U_CLASS_i1o1_D1 (not a candidate; not in code_ids).
- A, lra/run.py and lra/lock.py: the same-map D0 diagnostic ids `<D1 cid>|D0SAME` (arm d0_same, unit prefix d0s__,
  diagnostic only) and the `d0same` science stage (lra.select.stage_d0same), as agreed with role D.
- A, lra/run.py (role F R-1): CLASS|D1 is a registered fixed-map D1 code. That gives 27 per seed, 84 codes and 89
  configurations. It replaces the earlier diag__ diagnostic idea, which was never run.
- A, lra/lock.py and lra/run.py (role F R-3): CBP_LOCAL_ONLY can no longer bypass the push check for any stage, and
  engineering_ready() always requires the result on origin.
