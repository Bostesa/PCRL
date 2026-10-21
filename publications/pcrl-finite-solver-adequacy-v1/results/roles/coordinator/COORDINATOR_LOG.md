# Coordinator log (campaign `pcrl_finite_solver_adequacy_v1`)

All times UTC on 2026-10-21, from `date -u`. Append-only.

- **02:22:58Z** Campaign start (first `date -u`). No AGENTS.md at the worktree root or below (checked to depth 3) or at the main
  checkout. Branch `research/pcrl-finite-solver-adequacy-v1` created from the finite-routing research tip 278cb1ed6.
- **02:23-02:25Z** Evidence read and checked: every named file of 04ac9161f (START_HERE, RESEARCH_DECISION, ADVISOR_BRIEF, TABLES,
  cards, strength-probe records, coordinator log, ER1, verification) is blob-identical to the local research copy; `nrm/fa/prc.py`
  is blob-identical to 682068d52's published copy; frc/common.py differs only by the publication sanitizer (one path). Card FA_PRC_v1
  section 1 documents the shrunk routing tables, cell-mean codeword refresh, seven-tilt first-improvement search and exact-objective
  acceptance as the specification; its R2 remedy was "a stronger search under the same objective".
- **02:26:55Z** Queue opened (`fsa.jobqueue`, 12 CPU-h, 8.5 h). Package `fsa/` copied from `frc/` with this campaign's constants
  (commit ac0626aa4).
- **02:28-02:30Z** Agents launched: statistical and challenger reviewer (FSA1 rules, advancement condition, reader validity, pilot
  rules) and exact-reference investigator (laws committed before experiments, narrow and broader references). Live: coordinator + 2.
- **02:30Z** Source admission ok (frc 04ac9161f and nrm 682068d52 on the remote; 31/31 frc code files identical, results identical
  except the post-publication PUBLICATION_RECEIPT.json; every frc private artefact re-hashes).
- **02:31-02:40Z** Solver code: `fsa.fa.solver` (SV1 exact tilt line search with best improvement; SV1_first; traced original;
  selftest 9/9 on invented data: cost terms equal `route`, interval partitions equal `route` at midpoints, incremental objective
  equals `objective`, traced original equals `lloyd` and `protected_path`, SV1 monotone and rule-consistent, HQ books handled) and
  `fsa.fa.solve` (stages replay / audit / source / nested). Card FA_FSA_v1 (no release configuration). Reader splits (+800 seeds,
  FSA1 tags; FIT ~4,000 / VALIDATION ~1,590 / SCORE ~2,400 people per repetition). Committed 250a78950 at 02:40:24Z.
- **02:40-02:42Z** PROTOCOL.md, EXPOSURE_MAP.md, FAILURE_MAP.md drafted (committed 0f03bc975 at 02:42:35Z; the first version of this line and of the two maps carried times ahead of the clock, corrected at 02:43Z). Registration waits for the statistical reviewer's RULES.
- **02:42:16Z** The investigator committed LAWS.md / laws.json (c27a91477) before any numerical experiment on them.
- **02:44-02:47Z** Invented-data timing and trace (not a registered law; 6,000-20,000 random rows): both solvers stalled after the
  first strengths; every partition on the refreshed book's tilt line was worse than the current state. The SV1 proposal class was
  extended to the current book's tilt line (card FA_FSA_v2 supersedes FA_FSA_v1, which governed no run; commit 91ba9a27e at
  02:46:39Z), before any real-data or registered-law run.
- **02:44Z** Engineering reviewer launched; **02:50Z** independent verifier launched. Live: coordinator + 4.
- **02:50Z** The investigator's first result files (narrow reference, enumeration) exist. The coordinator does not read them and
  relays nothing about them to the statistical reviewer until the reviewer's rules, including the advancement condition, are
  committed.
