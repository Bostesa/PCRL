# Brief given identically to both mechanism designers

You are mechanism designer {k} in a bounded PCRL mechanism-design campaign. Two designers work independently. Do not open the other designer's directory (`roles/designer_{other}/`) until the coordinator reveals it. You write only inside `results/pcrl_mechanism_design_v1/roles/designer_{k}/` in the worktree `<worktrees>/pcrl-mechanism-design-v1`.

Read first:
- `roles/ROLES_AND_OWNERSHIP.md` (binding data rules);
- `PROBLEM_SPECIFICATION.md`;
- `EVIDENCE_PACK.md`;
- `CLOSED_LANES.md`.

You may read code in the worktree and published aggregate tables. Useful code: `acs_replication/leafcorr.py`, `neural_raw.py`, `trees.py` and `data.py`. You must never open person-level arrays or anything under `.private/` other than `.private/mechanism_design/`.

## Task

Propose one substantive mechanism for the ACS two-recipient output service, or conclude that you cannot find a credible one. `NO_CREDIBLE_PROPOSAL`, with reasons, is an acceptable and useful outcome. Do not invent a candidate to satisfy the brief.

A new name for a standard penalty, another coefficient, a larger adapter or a different checkpoint is insufficient justification. If you revisit a closed family, you must name the substantive missing mathematical or statistical ingredient and explain how a test isolates it.

Write `PROPOSAL.md` (about 4 pages at most) with these sections.

0. **Header.** Designer id; decision (`PROPOSAL` or `NO_CREDIBLE_PROPOSAL`); the UTC time you finished.
1. **Plain-language summary.** One paragraph.
2. **Release contract.** Is it the primary contract (each recipient gets its own probabilities and decision from the six fields; deterministic)? If not, state the change exactly: feature service, metadata, seeds, shared state, interaction, randomisation or abstention. Comparators must be evaluated under the same contract.
3. **The eight questions,** answered one by one:
   1. Exactly what information would be removed or changed?
   2. Why should that information be unnecessary at the required task utility?
   3. Why does the existing neural benchmark retain it?
   4. What computation or statistical principle achieves the removal?
   5. What prevents the method from merely fooling its own training critic?
   6. Why should the mechanism move the utility/recovery frontier rather than select a different point on the old frontier?
   7. What is different from the closed lanes (row by row where relevant)?
   8. What observation would falsify the proposed explanation?
4. **Formal objective or constrained optimisation,** in the notation of `PROBLEM_SPECIFICATION.md`.
5. **Training procedure,** with access by role. SEX is used only in MODEL_FIT; selection uses MODEL_SELECT under the frozen rule.
6. **Inference procedure.** The six raw fields map to the releases; no SEX.
7. **Complexity** of training and inference.
8. **Expected failure modes.**
9. **Claims, kept separate:**
   - the competitive-method claim (vs NEURAL_RAW, including the 0.020 margin where applicable);
   - the joint-component claim, if any (vs the strongest local twin and both sequential orders);
   - what you do **not** claim.
10. **Feasibility.** Why the target improvement is mathematically possible under the utility caps (see `PROBLEM_SPECIFICATION.md` §6 planning arithmetic and `EVIDENCE_PACK.md` §C). What quantitative evidence supports it, and what would make it impossible. Do not lower thresholds or redefine comparators.
11. **Data-free prototype plan.** Finite laws or analytically tractable models with exact checks, specified before any optimisation:
    - a negative case where required task disclosure already reveals S;
    - a feasible case where genuinely unnecessary information can be removed while preserving useful predictions or confidence;
    - if joint design is claimed, a complementary-clue case with nontrivial local controls and both sequential-order controls.

    Give the laws, access, budgets, comparators (including the strongest applicable baseline analogue, not only a weak one), metrics (exact Bayes AUC from the released view, exact NLL / summed Brier / accuracy) and the expected outcomes.
12. **Real-data pilot sketch** (not to be run). Arms; equal exposure and budgets; nominee rule; unit count; runtime estimate; stopping rule.

## Rules

- Do not claim novelty. The prior-work reviewer judges that.
- Do not claim a fitted reader's AUC is a population privacy floor.
- Respect the facts in `PROBLEM_SPECIFICATION.md` §9:
  - joint vs sequential with full look-ahead;
  - decision-preserving releases;
  - true labels are not a floor.
- Small numerical checks are welcome. Run them through the campaign queue so CPU is ledgered, with unit ids `D{k}_<NAME>_V<n>` and scripts in your directory:
  `cd <worktrees>/pcrl-mechanism-design-v1 && PYTHONPATH=<worktrees>/pcrl-mechanism-design-v1 ~/PCRL/.venv/bin/python -m mechanism_design.jobqueue run --unit D{k}_<NAME>_V1 --stage designer_{k} --rss-gib 1.0 -- ~/PCRL/.venv/bin/python <script path>`; stdout/stderr land in `.private/mechanism_design/compute/units/<UNIT>/` (your own unit logs may be read)
  Start every script with `from mechanism_design import no_real_data; no_real_data.install()`.
- Aim to finish within about 75 minutes. Reply with a ≤250-word summary when done.
