# Team plan — lra (Adult learned-decoder and constrained-release test)

Started 2026-10-07T04:03:43Z (`<PRIVATE_CACHE>/lra_v1/START.txt`). The worktree branch is
`research/pcrl-adult-learned-decoder-release-v1`, from the lcr tip `091afc2` (lcr evidence `418e529`). The package is
`lra/`, a port of `lcr/` at 091afc2 with lcr→lra renames (PORT_LOG.md). The public results directory is
`results/pcrl_adult_learned_decoder_release_v1/` (PKG) and the private store is `<PRIVATE_CACHE>/lra_v1/`. The closed
lcr/cbp/qpc code and results stay unchanged.

## Roles (native agents)

| Role | Owns (disjoint files) | Deliverables |
|---|---|---|
| A, lead | lra/lock.py, lra/run.py, lra/data.py, lra/admit.py, lra/sema.py, lra/report.py, lra/eval_lock.py (refusal wiring only) | provenance, locks, ledger, orchestration, integration, commits; PROTOCOL, EXPOSURE_LEDGER, FIT_MANIFEST, PREDICTIONS, decision documents |
| B, decoder/math | lra/decoder.py, lra/fixtures.py, lra/deploy.py, tests test_decoder/test_fixtures/test_deploy | ENGINEERING_GATE_RULE.json; the correctness stage (checks 1–10, plus 11–12 wiring); MATH_REVIEW.md; METHOD_CARD decoder part |
| C, mapper | lra/mapper.py, test_mapper | persisted starts/moves/stats-hash/termination receipts and an independent-style replay of incremental terms (check 8); temporary sequential partner rule (check 7); SEARCH_RULES.json; TIMING fitting |
| D, attacker/statistics | lra/audit.py, lra/assess.py, lra/baselines.py, lra/select.py, lra/family.py, lra/infer.py, tests test_audit/test_select/test_truth_table/test_late | the 14 inherited findings (REVIEW_FINDINGS_DISPOSITION.json); controls; composition closure; selection; primary family; TIMING audit |
| E, independent verifier | `PKG/verification/replay_lra.py` (imports no lra/lcr runner modules) | phases 0, 1 (gate, only after CORRECTNESS_LOCK), 2 (Adult fits, selection), 3 (endpoints, tables, deploy, restore) |
| F, claims/custody | lra/closeout.py, test_closeout; PKG docs PRIOR_ART_AND_CLAIM_SCOPE.md, BASELINE_FAIRNESS_REVIEW.md, PREDECESSOR_GATE_DIAGNOSIS.md | baseline fairness, prior art, public/private separation, backup, claims review |

## Shared rules

- **Locks and commits.** Only A writes locks, commits or pushes.
- **Semaphore.** Every numerical process (tests, timing, oracles, stages) runs through
  `python -m lra.sema --label <ROLE>:<what> -- …`. That means at most 2 slots, one thread, ≤ 8 GiB aggregate and ≥ 5
  GiB free disk.
- **Fixture laws.** No agent runs a fixture algorithm or oracle on the pinned laws before CORRECTNESS_LOCK is pushed.
- **Adult labels.** No Adult label-based decoder solve or mapping fit runs before SCIENCE_LOCK.
- **Assessment labels.** No assessment label is read before the pushed EVALUATION_LOCK; only `lra.assess` unseals.
- **Reviews.** Reviewers send findings; they do not edit locked files.

## Stage plan

1. **Admission.** SOURCE_ADMISSION_LOCK, then `admit`. This reuses the lcr/cbp verified copy and cross-checks it
   against lcr SOURCE_ADMISSION.json.
2. **Correctness.** CORRECTNESS_LOCK, then `correctness` (ENGINEERING_GATE_RESULT.json), then E's phase 1.
3. **Science.** SCIENCE_LOCK, then `d1` (78 fixed-map D1 units plus the CLASS|D1 diagnostic records), `ctask` (3),
   `fit` (72 weighted plus 15 constrained), `inner` (264), `inner_src`, `controls`, `select`, then E's phase 2.
4. **Assessment.** EVALUATION_LOCK, then `assess` and `infer`.
5. **Closeout.** Reports, E's phase 3, deploy, backup and handoffs.
