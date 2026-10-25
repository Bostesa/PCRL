> Saved verbatim by the coordinator (2026-10-25, after 03:09:53Z by `date -u`): the statistical reviewer's hand-back message. Its
> final section (the full text of REPORT_PREREGISTRATION.md) is saved separately in `REPORT_PREREGISTRATION.md` and is not repeated
> here; everything else below is unchanged.

## DRF1 statistical-reviewer hand-back (decision rules written before any label read)

Writing started at 2026-10-25T02:38:01Z and finished at 03:08:00Z (both from `date -u`). First freeze was at 03:04:59Z. I revised it before registration to make the comparator ids configurable and to add a note on aligning with the protocol.

I read no label, nothing under `.private/`, no card, no investigator file and no MODEL_SELECT data, and I fitted nothing. Every number comes from invented data. I ran no git command that changes state, and I did not touch `nrm/`.

The harness refused my write of `REPORT_PREREGISTRATION.md` ("subagents return reports as text"), as it did for SAR1. Its full text is at the end of this message; please save it verbatim with attribution in `results/pcrl_direct_recovery_frontier_v1/roles/statistical_reviewer/`.

### Files (SHA-256 at completion)
- `results/pcrl_direct_recovery_frontier_v1/roles/statistical_reviewer/RULES.md`: c411a14acb316e754952543a9a5be031d6ea6878aa3ec191be07c3c1a46d59b1
- `results/pcrl_direct_recovery_frontier_v1/roles/statistical_reviewer/RULES.json`: 5277b21de6c8cc506c910ce6f91dbf4ef19c4fbb41b42259d915edcf61cfeb7d
- `drf/rules.py`: aa552f02df63ede5981a8f85fb766ac863aabbe6808f4f695de8225ecbe9ef4a
- `drf/test_rules.py`: d7390bade4ba96ed67dba4960565f1e70eae33ac46d867d694c076c6a3173893
- Inherited files, checked unchanged by `load_rules()`:
  - `nrm/rules.py`: d5df868678d483a3452e251d8f24575932d23f0e5d22e3ef1ca3627a35030df1
  - NRM1 `RULES.json`: 84f187768817e79ef9f2bb36151bea81f00b8f62d9179c8cd523069550674652
- `REPORT_PREREGISTRATION.md`: not written by me (text below).

### Tests
- `python3 -m drf.test_rules`: **24 of 24 PASS, exit 0**, about 0.29 CPU-s per run.
- In-memory mutation check (scratch script outside the worktree; no repository file changed): **37 of 37** seeded single-point bugs caught.
- `load_rules()` refuses 17 altered constants, including an absolute 0.58 AUC margin, the SAR1 offset +1000 and a MODEL_SELECT bracket source.

### The rules in brief
- **D1 (usefulness).** G1 means all 28 original conditions hold at the point: per version, 12 task clauses plus 2 within-SEX guards. It is the only usefulness gate.
  - The confidence conjunction and the two-SE reserve are reported for DR, SUCC, TWIN and OBJ_MI and never gate.
  - If DR fails usefulness, the reader stage may stop (`reader_stage = STOPPED_USEFULNESS_FAILURE`); the fitted release is kept and the class is `TASK_INFEASIBLE`.
- **D2 (component gain).** It gates only on the two family controls; the met label is `COMPONENT_GAIN_BEYOND_FAMILY_CONTROLS`.
  - (c) **CTRL frontier.** The path starts at TWIN, followed by the two saved CTRL points. It is read with the inherited `frontier_comparison`, plus the recipient gate at Q\* and "no cheaper point resolved less recoverable than DR". If no point brackets DR's spend, it reads NOT_SEPARATED (NO_BRACKET). The bracket data must be `MODEL_FIT_NESTED`.
  - (o) **OBJ_MI (one point).** It needs a resolved pair benefit, the recipient gates and no secondary metric resolved higher. On spend, a single control point cannot be chosen to match DR, so the point-bracket rule becomes a resolution rule:
    - OBJ_MI spends at least as much as DR: credited.
    - OBJ_MI spends less but not resolved less: credited, flagged and stated.
    - OBJ_MI spends resolved less: `NOT_SEPARATED_FROM_SPEND`; the objective is not credited for a spend change and the result is reported as a budget-use difference.
    - Always reported: the spend gap with its SE, the 8 secondary metrics, and the strict point-rule reading.
  - (d) **Reader targeting**, inherited.
  - Partial results are labelled `PARTIAL_CTRL_FRONTIER_ONLY` or `PARTIAL_OBJ_ONLY` and are never claims.
- **FC and TWIN.** The family change (DR vs SUCC) is reported separately from the objective comparison and is never part of D2. TWIN is reported beside D2 and is never sufficient on its own (flag `TWIN_ONLY_ATTRIBUTION_INSUFFICIENT`).
- **D3 / D4 / D5 / D6.**
  - D3: the inherited reference test (NRM1 G4) against the nine references, with lists of those passed and failed and the reader-budget credit.
  - D4: a resolved 0.020 benefit against both NEURAL_RAW and H_LJ with the recipient gates; never weakened.
  - D5: not applicable unless joint design is claimed.
  - D6: NOT_CLAIMED.
  - No absolute AUC threshold appears anywhere.
- **Useful optimistic point** (MODEL_FIT only, comparator-based, reported, never a gate). OPT_S or OPT_L counts as useful when, in both versions:
  - it stays within all 8 DR budgets;
  - it is at least 0.020 below, in pair AUC\*, the cheapest SEX-blind CTRL point (same law and mode) that spends at least as much;
  - neither single view is more than 0.010 above that point.
  - It is judged in-sample; whether it survives on nested held-out data is reported beside. It is unchanged by a common shift of all AUCs.
- **Failure stage** (reported). The legal stage is L2 (DR fails G1 or DR = TWIN), L3 (ties its controls), L4 (D2 only) or L5 (D3 met).
  - L2 or L3 with no useful optimistic point gives stage 1.
  - L2 with a useful point gives stage 2; L3 with a useful point gives stage 3.
  - L4 gives stage 4, called a scoped component result only if D2R is confirmed with no sign change and no robustness recipient failure.
  - L5 gives stage 5, flagged `D3_WITHOUT_D2` if D2 is not met.
  - If the optimistic diagnostic is incomplete or unbracketed, the stage is UNDETERMINED.
- **Outcome class.** In order: INCOMPLETE, NO_OUTPUT_CHANGE, TASK_INFEASIBLE, INCOMPLETE, PROTECTION_NOT_ASSESSED, INCOMPLETE, METHOD_CANDIDATE, COMPONENT_GAIN_NOT_COMPETITIVE, REFERENCE_ADVANTAGE_NOT_ATTRIBUTED, FEASIBLE_NO_RELIABLE_GAIN.
  - The three outcomes (component gain, strong references, material target) are reported separately, with FC, TWIN and the stage beside them.
- **Registration.**
  - Family is DR; revisions 0–2 only (revision 3 or more is refused).
  - A reading is registered if its card was committed before any MODEL_SELECT outcome. A revision also needs a prospective note naming the failure, the changed operation and a MODEL_FIT or SYNTHETIC trigger.
  - Otherwise the tag is ADAPTIVE_R0_LATE_CARD, ADAPTIVE_R<n> or ADAPTIVE_R<n>_NO_NOTE, and every label carries that prefix.
  - The headline is the latest pre-outcome card; the multiplicity statement is inherited.
- **D2R (robustness).** Uses R3/R4 with seeds k2/k3 and no retuning. The decisive comparators are the registered Q\* and OBJ_MI.
  - Confirmation also needs both single-view recipient gates to hold on the robustness cells.
  - Sign changes and failed robustness recipient gates on any comparator, decisive or reported beside, go into the headline.
- **Grid validity.**
  - Seeds and tags:
    - Reader seeds are MODEL_SEED_BASE + 1100 + 10v + k (2026102305–08 and 2026102315–18); offsets 300–1000 are forbidden (160 seeds).
    - Bootstrap seeds are 2026102500 + 100v + repetition index; the v0 seeds equal SAR1's v1 seeds, which is disclosed and harmless.
    - Split tag `DRF1_devgrid_R{r}`.
    - `READER_SPLITS.json` agrees with all of this.
  - Few-valued explanation: SAR1's bound of 320 is unchanged for every view.
  - New finite-pair extension: a pair-view seed equality with 320 < n distinct values is explained only if both of that wire's single-view counts are at most 320, n is at most their product (and at most 1024), and the same recipe differs on another wire's pair view. Such equalities are counted once (n_eff); the strict NRM1 and SAR1 readings are reported beside.
- **Estimator section (RULES.md 10).** It keeps three sources of uncertainty apart:
  - household sampling;
  - fitting variation: the two versions are procedure repeats, not replications; the nested estimator measures the procedure's expected behaviour at 80% training size, not the single deployed fit, and its fold spread is never used as an SE;
  - numerical randomness: the optimizer is deterministic, so no Monte Carlo term; reader seeds enter through block means.
  - All intervals are conditional and do not correct the adaptive search.
- **Power.** D2 resolves true benefits of 0.010 / 0.015 / 0.020 / 0.025 / 0.030 about 12% / 33% / 61% / 84% / 95% of the time. If OBJ_MI captures half the gain, 0.25 / 0.40 / 0.55 at 0.020 / 0.025 / 0.030.

### REQUIRED findings (summary; full text below)
1. **R1.** Freeze the 16 budgets before any fit; use identical budgets for DR, OBJ_MI and OPT\_\*; disclose that they were tuned on the twin's MODEL_SELECT aggregates.
2. **R2.** OBJ_MI must be equally exposed (record the settings side by side for the verifier) and fitted before any MODEL_SELECT read; report the binding constraints and the nested spends.
3. **R3.** Save the CTRL points by the registered rule, with DR's nested spend from the same estimator. Record the full path; extend it if it may not bracket DR.
4. **R4.** Record distinct-input counts for all three views of every minimum-set wire, on the MODEL_SELECT pool, plus full seed checks. The minimum set is DR, TWIN, the CTRL points, OBJ_MI and SUCC.
5. **R5.** Record the reader-stage stop and preserve the release; compute the optimistic diagnostic anyway.
6. **R6.** Provide the optimistic inputs on the same law and mode, with the CTRL path covering their spends. The plan must state that no diagnostic gates any fit.
7. **R7.** Compute exact-alias records on the wires.
8. **R8.** Record the time of the first MODEL_SELECT read and the commit of each card; revision notes go in the card.
9. **R9.** Determinism: no se_mc; record the seeds and the restart rule.
10. **R10.** D2R grid entries need all three views.
11. **R11.** Keep FC and TWIN out of D2 in every sentence.
12. **R12 (alignment).** The draft `PROTOCOL.md` section 8 says D2 is "beyond START, the CTRL frontier and OBJ_MI". The rules keep SUCC out of D2 (it is reported as FC and sits in D3). Align PROTOCOL section 8, or ask me to re-freeze with START gating, before registration.

### Notes for the coordinator (code)
- `drf/rules.py` API changes from the mechanical copy:
  - `um` and `strongest_nonzero` are removed.
  - `grid_fineness(spends)` now takes one argument: `drf/claims.py` line 235 passes a second argument and must drop it.
  - The decision record format is described in the module docstring and RULES.md 11.
  - The protocol's names work through `comparator_ids` (test `t_protocol_ids`): candidate `cand.config_id` = DR_AUC, twin `twin.id` = TWB_C64, `obj_mi.config_id` = DR_MI, CTRL points DR_CTRL_LO / DR_CTRL_HI.
  - The `drf/` modules that call the API (claims, plan, refitcheck) were not changed by me.
- Total CPU about 45 CPU-seconds (about 8 test runs, 3 mutation runs at about 11 s each, one normal-approximation power calculation), one numerical thread, no real data.

---

### Full text of `REPORT_PREREGISTRATION.md` (please save verbatim)

(saved verbatim in `REPORT_PREREGISTRATION.md`, see the note at the top of this file)
