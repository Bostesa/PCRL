# Card CTL_TWIN_v1: the shared task-only twin (family CTL)

Coordinator, campaign `pcrl_input_and_pair_release_v1`. Governs the configuration CTL_TWIN (one per version, 2 configurations). Code:
`ipr/ctl/twin.py` (`python3 -m ipr.ctl.twin fit --version V --card <this card>`), committed with this card before the fit.

1. **Operation.** P_inc = HGB_inc(Z), P_occ = HGB_occ(Z), Z = `ohn.fam_a.compact_po(X)` from the six permitted fields; decisions are
   the argmax. HGB_inc / HGB_occ are the cached task teachers of the new-release-mechanisms campaign (`TASK.pkl` V<v>: sklearn
   HistGradientBoosting with the A1_PO_16 settings, inner household-CV T*, full refit on all 20,000 MODEL_FIT people with their income
   and occupation labels), hash-pinned in `SOURCE_ADMISSION.json`. Deterministic. This unit reads no label.
2. **Closest previous implementation.** The dual-design campaign's DA_TWIN (`ddr/da/std.py`, op 'twin'): the same release law. Its saved
   release is kept as the beside reference DDR_TWIN; equal wires are expected and are counted once by the grid (exact alias).
3. **Role.** The equally trained task-only twin of every family whose task models are these teachers (PROTOCOL section 4), the base
   of the SPEND (DQ) and RO control paths, and the deployment twin of any packaged release built on it. A family that retrains task
   models states its own twin in its card (same learner, procedure and effort with protection off) and may declare CTL_TWIN identical
   to it only if the procedure is identical.
4. **Strength, controls, revisions.** None (strength 0 of every path). No revision.
5. **Configuration count and CPU.** 2 configurations; under 60 CPU-s per version (releases for all six roles).
