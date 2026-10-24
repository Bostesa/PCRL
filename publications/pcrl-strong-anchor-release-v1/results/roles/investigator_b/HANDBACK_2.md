# Investigator B second hand-back, after the coordinator's adjudication (saved verbatim by the coordinator, about 05:09Z on 2026-10-24)

## FB_PFD revision: all seven required changes are in, and both selftests pass

Everything stayed label-free. I changed only `sar/fb/` and my role directory, and no shared code. Work ran 04:58Z–05:07Z. Design CPU is now about 80 CPU-s in total, against the 600 cap.

### Changes in `sar/fb/pfd.py`
1. **CTRL path (FB_PFD_DQ_LO / _HI): SEX-blind quantization of NEURAL_RAW.**
   - It uses `sar.ctlkit.dq_fit` / `dq_release` over the steps `DQ_GRID_FINE`. The keyed dither has tag `SAR_DQ_FB_S<v>` and uses the six compact fields (`famkit.compact`).
   - The basis is fitted on the cross-fitted RAWJ-E2 MODEL_FIT outputs.
   - The deployed point is DQ applied to the pinned NEURAL_RAW E2 weights, run through this family's float64 numpy forward pass, with the same basis, step and dither.
   - `predict` gained an `op: 'dq'` payload type.
2. **Bracket rule (`bracket`, used by both CTRL and ALT).** Origin is START = the cross-fitted RAWJ E2, and the selection data are MODEL_FIT out-of-fold.
   - CAND's spend over START is the out-of-fold paired total-NLL difference against RAWJ E2, recorded as value and SE.
   - LO = the point with the largest spend that is ≤ CAND's.
   - HI = the point with the smallest spend ≥ CAND's + 2 SE, where SE is that point's paired SE against CAND.
   - Points are chosen by spend value, not grid order, so a path whose spend is not monotone is still handled. Ties go to the first point in grid order.
   - Flags: CONTROL_PATH_MAY_NOT_BRACKET (HI falls back to the largest-spend point) and CTRL_BRACKET_IS_ORIGIN (nothing saved).
   - I added two flags you didn't list:
     - LO_IS_ORIGIN: no point is ≤ CAND's, so LO is the origin and is not saved.
     - LO_EQUALS_HI: only HI is saved. This mirrors the lo == hi rule in `sar.paths.bracket_points`.
3. **ALT path (FB_PFD_RAWJ_LO / _HI): NEURAL_RAW's own recipe continued.** The grid is E3..E10 under the same rule; E2 is the origin and is not saved again. Both bitwise checks are kept: the full-run RAWJ E2 equals `NEURAL_RAW__rho1.0__E2.pt`, and the U0 recipe reproduces U0.
4. **CONT (FB_PFD_TASK)** is unchanged and declared gating.
5. **Fit record** `results/pcrl_strong_anchor_release_v1/fb/FIT_PFD_V<v>.json` now holds:
   - `cand`: config, strength k*, checkpoint, status FITTED / NO_FEASIBLE_STRENGTH, and spend over START;
   - `paths.CTRL` and `paths.ALT`: origin, `selection_data = MODEL_FIT_OOF`, every grid point's spend over START with its SE and its paired difference and SE against CAND, `lo_config` / `hi_config` with their spends, flags and fineness;
   - `cont`;
   - `pins_sha256`.
   - Income-share diagnostics are recorded for CAND and for each saved bracket point.
6. **Pins:** SHA-256 of `U0_MODEL.pt`, `STAGE_A_SEED.json` and `NEURAL_RAW__rho1.0__E2.pt` (keyed by absolute path) is written into the fit record, every MODEL.pkl and every release record.

### Changes in `CARD_DRAFT.md`
- The specification count is stated as at most 12: CAND 2 + CONT 2 + DQ 4 + RAWJ 4.
- Item 7 is rewritten with CTRL, ALT and CONT, the bracket rule, the flags, the checks and the fit-record contents.
- **SBC / RO wording — I changed my own reasons.** You asked to keep them, but my original wording was inaccurate. Both are constructible: PFD with the PSEUDO2 attribute in place of SEX, or with a SEX surrogate computed from the release only. They are left out for the specification budget, and the not-built paragraph now says so. SPEND is no longer listed as not built. Joint design is NOT_APPLICABLE_NOT_DECLARED.
- Revisions are now at most 8 specifications each: their own CONT and CTRL, with the ALT bracket re-selected from the already trained RAWJ path.
- CPU estimate is now 4–6 CPU-min per version (upper estimate 12).
- The registered predictions are unchanged. A dated note at 05:04Z reads "matched control" as the ALT bracket and adds predictions for the new controls:

| New prediction | P |
|---|---|
| CTRL brackets CAND in both versions | 0.80 |
| ALT brackets CAND within E3..E10 | 0.75 |
| CTRL_BRACKET_IS_ORIGIN in either version | 0.10 |
| CAND's pair below DQ_HI at the point | 0.75 |
| CAND's pair below DQ_HI, resolved | 0.30 |
| CAND separated from every gating control | 0.12 |

`PROPOSAL.md` has a dated addendum saying its section 5 is superseded.

### Test results
- **`selftest`: 25 of 25 checks pass.** New ones cover the bracket rule with all four flag branches, and the DQ deployment law matching `ctlkit.dq_release` bitwise.
- **`selftest_stage` (5 folds, epochs 4, rho 20 on invented labels): all pass.** CAND's spend over START was +0.0092, so the bracket branch ran. Saved configs were FB_PFD, FB_PFD_TASK, FB_PFD_DQ_HI and FB_PFD_RAWJ_HI; both paths flagged CONTROL_PATH_MAY_NOT_BRACKET and LO_EQUALS_HI. The saved-config set matched the brackets, every release equals `predict`'s output bitwise, and the fit-record fields are present. 26.6 CPU-s.
- **`check_deploy_python3`: all pass.** Under python3 with no torch, both the plain network and DQ payloads match bitwise and are deterministic.

### Notes for you
- If the RAWJ E2 bitwise check fails at fit time, the ALT points come from my replica while the DQ base stays the pinned NEURAL_RAW. Both facts are recorded.
- The deployed DQ wire differs from NEURAL_RAW's stored torch wire by about 1e-7 before quantization.
- The MODEL_FIT descriptive readers now run on up to 8 views: fold start, START, and the 6 saved configs.
- Training runs under `<repository>/.venv/bin/python`; deployment and `predict` run under python3.

Outputs are saved in `results/pcrl_strong_anchor_release_v1/roles/investigator_b/`:
- `selftest_output.json`
- `selftest_stage_output.json`
- `check_deploy_python3.json`
