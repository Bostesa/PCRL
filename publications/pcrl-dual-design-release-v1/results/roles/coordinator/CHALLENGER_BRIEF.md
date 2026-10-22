# Challenger brief (coordinator, campaign `pcrl_dual_design_release_v1`)

You are the challenger. Two designers worked independently (Designer A: what the task models are trained on or how; Designer B: how
released vectors are formed from fitted statistics). Read, in this order:

1. `results/pcrl_dual_design_release_v1/roles/coordinator/DESIGN_BRIEF.md` (setting, evidence, closed recipes).
2. `results/pcrl_dual_design_release_v1/PROTOCOL.md` (draft) and `FAILURE_MAP.md`.
3. `results/pcrl_dual_design_release_v1/roles/designer_a/PROPOSAL.md` and `mech_sketch.py`.
4. `results/pcrl_dual_design_release_v1/roles/designer_b/PROPOSAL.md` and `mech_sketch.py`.
5. Whatever earlier records you need (`results/pcrl_*`, `nrm/`, `frc/`, `fsa/` code) to check claims of difference.

Read no label file and nothing of MODEL_SELECT; run nothing on real data. You may run the sketches' selftests (invented data).

Write `results/pcrl_dual_design_release_v1/roles/challenger/REVIEW_DESIGNS.md` with numbered findings, each marked REQUIRED (must
change before the card is committed) or ADVISORY, covering for each proposal:

1. **Equivalence.** Are the two proposals the same operation in different words (same released function class, same SEX
   information used the same way)? If so, say which designer should be redirected and to what. Is either one a re-labelling of a
   closed recipe (propensity-routed cells, Gaussian logit channels, learned decoders, controllers, leaf corrections)?
2. **The simplest alternative explanation.** Is the proposed matched control really the same operation with only the SEX-specific
   information removed, at matched task spend, with the same tuning effort? Name the control you would require, and the SEX-blind
   spend path that must bracket the candidate's spend (an invertible transform such as temperature removes no information and is
   not a spend control).
3. **Utility.** Is the feasibility estimate credible? Which of the 24 clauses or 4 guards is most likely to bind (the men's
   occupation Brier guard has bound before)? Is the strength selection MODEL_FIT-only and predeclared?
4. **Recovery.** Which readers (LOOKUP, smooth, pair vs per-view) could defeat it, and where does the leakage likely move (pair
   super-additivity, the other recipient)?
5. **Revisions.** Are the two revisions each triggered by a named, observable failure, and bounded?
6. **Prior art.** The closest published method; whether any novelty claim is supported (the default is NOT_CLAIMED).
7. **Budget.** Configurations and CPU against the campaign limits (48 configurations in all, 12 CPU-hours, two jobs x one thread).

End with a short verdict per proposal: PROCEED, PROCEED_WITH_REQUIRED_CHANGES, or REDIRECT (with the redirect).
