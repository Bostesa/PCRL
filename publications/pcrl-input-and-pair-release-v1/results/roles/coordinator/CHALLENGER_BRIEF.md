# Challenger brief (coordinator, campaign `pcrl_input_and_pair_release_v1`)

You are the challenger. Two designers worked independently (Designer A, family IA: before the release, what the task models see
and learn from; Designer B, family IB: after the task models, the joint release map for the linked pair). Read, in this order:

1. `results/pcrl_input_and_pair_release_v1/roles/coordinator/DESIGN_BRIEF.md` (setting, evidence, closed recipes, the correction
   note at its end).
2. `results/pcrl_input_and_pair_release_v1/PROTOCOL.md` (draft) and `FAILURE_MAP.md`.
3. `results/pcrl_input_and_pair_release_v1/roles/designer_a/PROPOSAL.md` and `mech_sketch.py`.
4. `results/pcrl_input_and_pair_release_v1/roles/designer_b/PROPOSAL.md` and `mech_sketch.py`.
5. Whatever earlier records you need (`results/pcrl_*`, `ddr/`, `nrm/`, `frc/`, `fsa/` code, `ipr/ctlkit.py` for the shared SBC /
   SPEND / RO constructions) to check claims of difference.

Read no label file and nothing of MODEL_SELECT; run nothing on real data except the label-free cached MODEL_FIT out-of-fold
predictions the designers used (TASK.pkl, SEXSIDE.pkl) if you need to check a feasibility number (one thread, at most 200
CPU-seconds, report what you used). You may run the sketches' selftests (invented data).

Write `results/pcrl_input_and_pair_release_v1/roles/challenger/REVIEW_DESIGNS.md` with numbered findings, each marked REQUIRED
(must change before the card is committed) or ADVISORY, covering for each proposal:

1. **Equivalence.** Are the two proposals the same operation in different words (same released function class, same SEX
   information used the same way)? If so, say which designer should be redirected and to what. Is either one a re-labelling of a
   closed recipe (propensity-routed cells, Gaussian or coupled logit channels, learned decoders and calibrators, controllers, leaf
   corrections) or of the dual-design idea family (SEX-standardized posteriors, inverse-propensity reweighting, linear propensity
   tilts, release-only tilts)?
2. **The simplest alternative explanation.** For each of the three gating control paths (PROTOCOL section 4): is the SBC path the
   same operation with only the SEX-specific information replaced, at matched task spend, with the same tuning effort? Does the DQ
   spend path bracket the candidate's spend? Is the RO path (the operation driven only by what can be estimated from the twin's
   released vectors, or the generic release-only tilt) correctly specified for this operation, and fine enough near the candidate's
   spend? Would you require any further control? (An invertible transform such as temperature removes no information and is not a
   spend control.)
3. **Utility.** Is the feasibility estimate credible? Which of the 24 clauses or 4 guards is most likely to bind (the men's
   occupation Brier guard has bound before)? Is the strength selection MODEL_FIT-only and predeclared? For a retrained task model:
   is its twin equally trained (same learner, procedure, folds, seeds and effort)?
4. **Recovery.** Which readers (LOOKUP, smooth, pair vs per-view) could defeat it, and where does the leakage likely move (pair
   super-additivity, the other recipient)?
5. **Joint design (D5).** For a jointly designed operation: are the separately designed recipients and both sequential orders
   well defined at matched spend, and affordable? If they are not built, say what label D5 gets.
6. **Revisions.** Are the two revisions each triggered by a named, observable failure, and bounded?
7. **Prior art.** The closest published method; whether any novelty claim is supported (the default is NOT_CLAIMED).
8. **Budget.** Configurations and CPU against the campaign limits (48 configurations in all, of which 2 are the shared twin; 12
   CPU-hours; two jobs x one thread).

End with a short verdict per proposal: PROCEED, PROCEED_WITH_REQUIRED_CHANGES, or REDIRECT (with the redirect).
