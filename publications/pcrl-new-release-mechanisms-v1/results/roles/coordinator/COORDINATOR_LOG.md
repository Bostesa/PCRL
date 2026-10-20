# Coordinator log (UTC, real clock; append-only)

- **14:29:43Z** Campaign start (`date -u`, the first action).
  - AGENTS.md was inspected first: it does not exist at the worktree root or in the main checkout.
  - Branch `research/pcrl-new-release-mechanisms-v1` was created from 0f27e5937, the controlled training-size research tip and the base of
    publication ed7bc1ccd.
- **14:30-14:40Z** Read the latest source (START_HERE, RESEARCH_DECISION, EXPOSURE_MAP, conditional/EQUAL_RESERVE.csv) and the reusable
  infrastructure (tsz.common, tsz.jobqueue, tsz.devreaders, tsz.evaluate, ohn.evaluate, ohn.devrun).
- **~14:38Z** Wrote `roles/coordinator/AGENT_CONTEXT.md`, the shared brief for every agent. Launched wave 1, four assignments, five live agents
  including the coordinator:
  - Designer A: learned finite release;
  - Designer B: joint task / randomized channel;
  - challenger: failure map, controls catalogue, generic SEX-blind control code;
  - statistical reviewer: generic claims, selection rules, screen checks.
- **14:38-14:47Z** Built and selftested `nrm.common`, `nrm.jobqueue`, `nrm.lane`, `nrm.release`, `nrm.evaluate`, `nrm.screen`, `nrm.devreaders`,
  `nrm.admit`, `nrm.receipt` and `nrm.publish_build`. Deviation 1: the grid selftest's CPU was charged out of the queue.
- **14:45Z** Source admission `ok`:
  - pin ed7bc1ccd equals the remote branch tip;
  - 222 published results files and 28 code files are identical to the local copies (some after the source sanitiser);
  - all 2,618 private source copies are bitwise identical to the source worktree.
- **14:46Z** READER_SPLITS.json written (label-free): NRM1_devgrid_R1..R4 and the NRM1_screen halves. ACCESS_POLICY.json written.
- **14:48Z** Commit febfba43a: infrastructure and registration drafts. Waiting for the failure map and rules before the receipt.
- **14:49-14:52Z** Checked the earlier MMP proposal (mechanism-design campaign, data-free):
  - it was declined because an injective deterministic release leaves P(S | X) learnable;
  - so a SEX-standardization family would add only a learnability effect;
  - it is not proposed as a third family. Many-to-one cells or randomization are the information-removing routes.
- **14:51-14:52Z** Grid timing (one thread): about 16 CPU-s per release per cell. Deviation 2: the first timing unit failed on an odd
  invented size. Reference evaluation and screening lane written (runs after the receipt).
- **14:53Z** Publication dry run: sanitizer and leak scan report 0 problems. Agents' uncommitted files were skipped.
- **14:54Z** `nrm.deploy`: a generic raw-field packager over the family interface `predict` / `model_paths`.
- **~14:56Z** Launched the independent verifier, an early wave-2 assignment, to build separate numerical code. That makes six live
  agents including the coordinator, the limit.
- **14:57Z** Challenger returned. FAILURE_MAP.md (10 families), the controls catalogue and nrm/ctl (selftests 29/29 and 17/17) are
  committed. The closeout tools (custody, restore check, ledgers, receipts) are adapted; the lineage DEV_REPORT log snapshot is
  c7829e99 at 14:55:55Z.
- **15:00-15:02Z** An extraction sub-agent of the challenger reported directly to the coordinator. ListAgents showed one more unknown
  agent: the statistical reviewer's code-review sub-agent. So sub-agents had pushed the live count above six (deviation 3).
  About 15:02Z every running agent was told not to spawn sub-agents; the rule was added to AGENT_CONTEXT.md.
- **15:00-15:05Z** Designer B returned FSC-CA (channel-aware joint training with a SEX-targeted, Fisher-shaped, coupled Gaussian logit
  channel; code ef0bc1ddf) and a second hand-back with its sub-agent disclosure. Designer A returned FA-PRC (propensity-routed finite
  cells; code 47916952c), disclosing 3 sub-agents and a multi-threaded selftest (deviation 4). All saved verbatim.
- **15:10Z** Decisions, provisional pending the challenger's equivalence review and the receipt:
  - FA-PRC: 18 configurations (+2 contingent for R0). No joint-design claim unless PRC beats LOC clearly; sequential orders would
    then be the decisive phase.
  - FSC-CA: 16 configurations (S1 dropped from both paths). No coupling claim, since the SEX and PCA paths are both coupled.
  - Configuration plan: A 18 + B 16 = 34. About 8-12 go to generic controls, about 8 to ablations and perturbations, and the rest are
    kept for revisions.
  - Both designers are wiring access, release and deployment; the deploy contract now carries out-of-band draw keys for randomized
    packages.
  - The challenger is reviewing both proposals, due about 15:30Z.
