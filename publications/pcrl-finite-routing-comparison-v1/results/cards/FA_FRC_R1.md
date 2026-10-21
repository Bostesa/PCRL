# FA_FRC_R1: revision 1 of family FA (a spend-matched untargeted re-routing control)

Campaign `pcrl_finite_routing_comparison_v1`. Written from 2026-10-21T00:05:52Z (`date -u`), after P1 and P2 (MODEL_FIT only) and
**before any MODEL_SELECT read of a protected release** (no protected release has been evaluated or read by the grid). Revision 1 of the at most two allowed (PROTOCOL section 11). Governs units `FRC_UNT_V<v>_*`.
Everything not stated here is card FA_FRC_v1 (with addendum A1).

## Failure (identified on MODEL_FIT, recorded before this fit)

- The registered untargeted control SHUF (donor-copy propensity, own-qO guard) does not reach the nested total-NLL spend of the PRC
  configuration selected at S2 at any grid strength, in either version (P1 records: SHUF spend about -0.0002 nats at every lambda,
  identical states for lambda 0.25 to 8; PRC_S2 target 0.00214 / 0.00194; brackets `CONTROL_DID_NOT_REACH_SPEND`). Its chain moves
  only 5-6% of people because a propensity that carries no person-level information gives the objective almost nothing to reduce.
- Consequence: the D2 conjunct "beyond the SHUF path" can never be credited at matched spend (the frontier rule gives
  `CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT`), so the registered design cannot separate targeted from untargeted re-routing.
  This is a property of the control, not an outcome of PRC.

## Change

Add one untargeted re-routing control, **UNT**, the "justified untargeted-routing control" the brief allows:
- rule: the twin's nearest-codeword rule, except that a person whose keyed hash u_O = SHA-256('FRC_UNT_S<v>|O' | compact features)
  / 2^64 is below f_O takes the second-nearest occupation codeword, and one whose u_I (tag '...|I') is below f_I takes the
  second-nearest income codeword. A deterministic function of the six fields; SEX-blind; no propensity, no SEX-side teacher;
- f_I = f_O x r, with r = PRC_S2's own income / occupation re-routing ratio on the MODEL_FIT out-of-fold statistics (so UNT moves
  the two outputs in PRC's proportion);
- f_O grid (predeclared): 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30;
- bracket (MODEL_FIT only): the first f_O in the grid whose nested total-NLL spend over the twin (the same outer folds, teachers and
  twin books as P1; the twin's nested means must equal P1's within 1e-12) is >= PRC_S2's P1 target; none -> `CONTROL_DID_NOT_REACH_
  SPEND` and nothing is saved;
- release `UNT_BR` (one configuration per version): the construction of the qualification decision (B1) recomputed on UNT's own
  out-of-fold partition; stores the task teachers only; nested held-out pair LOOKUP AUC and its bootstrap difference from the twin
  recorded as for P1 (SEX only as that evaluation label).

## Control and reading

- The untargeted path of D2 becomes **[SHUF_S2, UNT_BR]** (both SEX-blind untargeted re-routing controls). The conjunct is
  `nrm.rules.frontier_comparison` of the PRC configuration selected at S2 against that path on MODEL_SELECT costs, as registered;
  with UNT_BR present the bracket is normally UNT_BR.
- **Both readings are reported:** the registered D2 (untargeted path [SHUF_S2]) and the R1 D2 (path [SHUF_S2, UNT_BR]); the
  registered label is never overwritten. Every other D2 conjunct, D3-D6, the reader-targeting rule and F4 are unchanged.
- UNT is compared on the same grid cells, readers and costs as every other control and enters the reader bank as a member.

## Stopping rule

- If PRC_S2 is not `DOMINATES_CONTROL_FRONTIER` against [SHUF_S2, UNT_BR] on MODEL_SELECT, the targeted-versus-untargeted question
  is closed for this family: no further revision addresses it.
- If UNT does not reach PRC_S2's spend at f_O <= 0.30, the conjunct stays uncredited (`CONTROL_DID_NOT_REACH_SPEND`) and the
  family's D2 cannot be met.
- This revision uses one of the two allowed revisions. Configurations: 2 (UNT_BR in both versions).

## Plan

| Unit | Command (`python3 -m frc.fa.route ...`; card `results/pcrl_finite_routing_comparison_v1/cards/FA_FRC_R1.md`) |
|---|---|
| FRC_UNT_V0_V1, FRC_UNT_V1_V1 | `unt --version v` |

Code: `frc/fa/route.py` (`unt_route`, `unt_book`, `unt_ratio`, `stage_unt`, `route_any`); selftest 50/50 at 00:05Z (share 0 equals
the twin, moves about f, ignores SEX statistics, the nested twin equals P1's, predict reproduces the saved release, task teachers
only).
