# Coordinator log (campaign `pcrl_input_and_pair_release_v1`)

Times are UTC, each copied from a `date -u` read at or before the event.

- **01:19:06Z** Campaign start (first `date -u`). The task brief repeats the previous campaign's brief; the previous campaign
  (`pcrl_dual_design_release_v1`) was complete and published (e30cf1c6b). This is a new campaign: memory notes, the finite-solver-
  adequacy and finite-routing records, the inherited failure map and the dual-design records read again. YOTUO not attached.
- **01:27:15Z** Branch `research/pcrl-input-and-pair-release-v1` created from c577424c0 (the dual-design research tip).
- **01:30:24Z** Design brief written (`DESIGN_BRIEF.md`): Designer A's space = before the release (inputs, training rows, targets,
  objective), Designer B's space = the joint release map for the linked pair; the dual-design idea family (STD, IPW-STD, PTC, ZTILT)
  added to the closed list; the release-only explanation stated as an alternative every design must answer. Both designers launched
  just after (independent agents, label-free, 400 CPU-s cap each).
- **01:33-01:37Z** `ipr/` derived from `ddr/` by a mechanical rename (scratchpad script; family modules not copied; new seed offset
  900, bootstrap base 2026102300, split tags IPR1); `ipr.admit run` `ok` (dual-design publication e30cf1c6b pinned, parent = its
  receipt, two post-publication result differences explained; ddr private saved releases hash-checked); access policy; queue opened
  01:35:28Z; release, famkit, ctlkit, devreaders and evaluate selftests pass; READER_SPLITS.json written (DDR1 overlaps added).
- **01:38:33Z** Protocol draft written (three gating control paths SBC / SPEND / RO; D5 planned for a jointly designed family;
  planned resolution about 0.02-0.025). Statistical reviewer launched just after (pre-registration review and the IPR1 rules).
- **01:39-01:41Z** The generic release-only path RO (the dual-design ZTILT law on a 1.1-ratio grid) and a 1.1-ratio DQ grid added to
  `ipr/ctlkit.py`; selftest passes. A correction note added to the design brief: TWJ's income is HC_STRONG, not DA_TWIN's.
- **01:42-01:50Z** `ipr/ctl/twin.py` (the shared twin CTL_TWIN, card CTL_TWIN_v1 drafted), `ipr/plan.py` and `ipr/claims.py`
  generalized to three control paths and joint-design controls; `ipr/paths.py` (shared DQ / RO bracket points with a fineness
  record); read-only support for the dual-design releases as beside references (`ipr.evaluate source_ddr`, bank prefix `DDR:`);
  reference bank IPR1_REFS and lane REF_EVAL pre-written. Infrastructure commits 091d0c88f (01:47:39Z) and a6f3bb1dc.
- **01:53Z** Designer A's proposal (SFG, shared SEX-aware field generalization) and hand-back received; the hand-back is saved
  verbatim (`roles/designer_a/HANDBACK.md`). The designer predicts no separation from its matched controls (release-only tilt about as
  good at equal spend) and recommends a lean run as the input-side closure test.
- **02:02Z** Designer B's proposal (CSF-T, coordinated cell-targeted selective flattening) and hand-back received and saved verbatim.
  The designer predicts no separation and no joint-design advantage, and recommends a reduced allocation. A note was sent to the
  statistical reviewer asking that the rules give a defined label for a SEX-blind release-only operation and for D5 controls that
  collapse by construction.
- **02:03Z** The designers' label-free CPU charged (160 + 150 CPU-s). Challenger launched on both proposals (brief
  `CHALLENGER_BRIEF.md`).
- **02:03-02:07Z** Draft family modules written with invented-data selftests passing: `ipr/ia/sfg.py` (menu checked label-free on
  the admitted MODEL_FIT features; full stage selftest on invented labels) and `ipr/ib/csf.py` (operation algebra). Not yet cards.
