# Coalition / multi-recipient attack-slate audit — PCRL ACS release-study lineage


Lineage status (coordinator, 2026-10-01): the SaTML '27 manuscript "Sharing Data for Specific Predictions with
Privacy Constraints" was registered 2026-09-23 and **withdrawn by an author on 2026-09-26**; it is not under review.

Role: METHODOLOGY (pcrl_acs). Date 2026-10-01. Read-only source inspection via `git show <ref>:<path>`;
no fits, no reruns. Line numbers refer to the file at the named ref.
Status vocabulary: **source inspected** (code read at the ref), **stored results checked** (committed JSON/MD
read), **not independently rerun** (nothing in this audit refits or rescored a model).

## What a correct slate needs (the audit criteria)

* **C1 nested singletons.** For a coalition view AB = (view_A, view_B), the eligible AB candidate set must
  contain every single-recipient predictor (A-view and B-view predictors applied to their own columns of
  the AB wire, i.e. "extra features ignored"). Then the validation-selected AB attacker is never worse on
  validation than the best singleton, and coalition-minus-singleton contrasts are not artefacts of a
  finite learner doing worse on a larger input.
* **C2 complete ancestor for appended channels.** If the release is *appended* to an already-available
  channel (wire = [H, Z_J, R]), the slate must contain a predictor that reads exactly [H, Z_J] and ignores
  R. An H-only predictor does **not** substitute: it lets the measured increment of R include the part of
  Z_J that the finite learner fails to exploit. For a *replacement* release (wire = [H, Z], J not on the
  wire) the H-only predictor *is* the complete ancestor.
* **C3 equal eligibility (catch-up routing).** Identical released views should get identical eligible
  candidate sets; extra candidates that exist only because a system had a saved training adversary
  ("catch-up") make slates unequal across systems.
* **C4 selection ≠ evaluation.** Candidate/route selection must use a pool disjoint from the pool on
  which the reported contrast is computed (no maximum over the evaluation set).

## Per-study table

| Study (ref) | Release shape | Views audited | C1 nested singletons | C2 complete ancestor | C3 equal eligibility | C4 selection vs evaluation | Verdict |
|---|---|---|---|---|---|---|---|
| redesign coalition v1 (`ablations-facct-2026-07-24` ad2c08872, `experiments/acs_coalition_audits.py`) | learned F/P interfaces vs static E | A, B, AB (+derived public heads for F) | **Yes**: `inherit_singletons` adds *every* A/B sensitive candidate to AB by column projection (`acs_coalition_audits.py:180-196`), parity asserted bitwise on validation (`:310-327`) | n/a (no appended channel) | **No (disclosed)**: `expanded_catchup` adds saved-start catch-up candidates only where a training observer exists; "E has no saved observer; its expanded-catchup selector contains the available independent candidates only" (`results/redesign_20260908_acs_coalition_v1/TABLE.md:61`); catch-up "wins 52 of 54 eligible race endpoints" (`COALITION_AUDIT.md:5`) | Yes: selection on `attacker_validation` (`:334` "minimum unweighted attacker-validation log loss"); `development_received: False` | OK on C1/C4; **DEFECT-disclosed** on C3 |
| coalition-strength / source-guard / fixed-predictions (Sept 9, same ref) | learned vs static/H controls | A, B, AB | Yes (same `inherit_singletons`; "AB includes every legal singleton sensitive candidate", coalition_strength md) | n/a | **No (disclosed)**: source-guard "Independent auditors favor the stronger local control on coalition SEX and race; the catch-up-inclusive selections reverse that ordering" (`redesign_20260909_acs_source_guard_v1` analysis); H/E "have stationary A inputs" without learned observers (fixed_predictions md) | Yes | **DEFECT-disclosed** (C3): headline ordering is scope-dependent |
| residual spectral v1 (Sept 10) and 2017 spectral transport (`origin/ablations-facct` 349efa454) | spectral R appended/replacing on [H_A, …] | A, B, AB | Yes (`acs_spectral_audits.py:187` calls `inherit_singletons`) | H-only anchors routed into A/AB (`acs_spectral_audits.py:172-183`, `source_condition='H'`) — adequate only where J is not on the wire | catch-up "historical_catchup_retained" (`:201`) — same C3 caveat | Yes (lock; final partition opened once) | OK on C1/C4; C3 caveat inherited |
| Study 6 utility extension (`research/pcrl-utility-extension-aws-v1` 0517c06a7) | **appended**: `wire/A = [H_A, Z_J, R]` (`results/pcrl_utility_extension_v1/METHOD.md:8`) | A, B, AB | Yes (inherited) | **No**: METHOD promised "untouched-J predictor candidates are part of every comparison (ref_J is audited under the same slate)" (`METHOD.md:20-21`), but the slate routed H-only ancestors (the `acs_spectral_audits.build_audits(..., ancestor=H)` path) and audited `ref_J` as a *separate condition*; "no in-slate predictor read [H_A, Z_J] while ignoring R" (`research/pcrl-manuscript-integrated-v6` eb4aa9668 `results/pcrl_manuscript_review_v6/CORRECTIONS_V6.md` D8; v6 `main.tex` §9 paragraph "Two defects in the pilot's own machinery") | n/a | Yes | **DEFECT-disclosed (the integrated-v6 "audit-slate defect")**; direction: the omission can only make measured increments of R *smaller* (flatters the release); not repaired; not used by the SaTML paper |
| Stochastic channel (RCSR) Stage B (`research/pcrl-stochastic-channel-v1` cd895e429) | code T vs J, utility only | utility probes | n/a | `best_code = min(unconstrained, ref_J)` on the **same validation rows** (`experiments/pcrl_stochastic_channel_v1/stage_b.py:201-205`) | n/a | **No** (min over the reporting pool) and the B1 partition used per-row *loss* deciles that encode the label (`stage_b.py:185-186, 271-273`; fixture `artifact/pcrl_satml_anon/fixtures/b1_label_leak_fixture.py`) | **DEFECT-disclosed/withdrawn** ("subsumption" claim withdrawn: CORRECTIONS_FINAL C3; decile numbers withdrawn: replacement `PAPER_ADDENDUM.md:31`) |
| Replacement Stage R (`research/pcrl-stochastic-replacement-overnight-v1` e3415b94d) | views H, H+J, H+T, H+raw (utility probes) | utility only (no sensitive audit ran) | n/a | `ancestor_inclusive(candidate, H)` (H is a genuine ancestor of every view; `replacement.py:160-180`) | Yes (identical probe schedule for every view, `replacement.py:121-125`) | **No**: `fit_view` selects the probe by `downstream_validation` loss and returns that same pool's per-row losses (`replacement.py:122-158`); `ancestor_inclusive` takes the min **on the same rows**, separately per weighting (`:160-180`); PROTOCOL lists `downstream_validation` as "screen and selection" (`PROTOCOL.md:41`) and the study itself says "Selected-validation bootstrap intervals are not independent evidence for the selected maximum" (`STATISTICAL_PLAN.md:107`) | **C4 violation, partly disclosed**. The SaTML correction C3/R1 quotes these as "adjusted bounds below zero" without the selection-pool caveat (see FINDINGS F8). Effect size (≈5.5 SE) makes reversal unlikely, but the bound is not post-selection valid |
| **Task-directed release (SaTML development evidence)** (`research/pcrl-task-directed-release-v1` f4bdf4cd5) | **replacement**: wire/A = [H_A, token] (`evaluation.py:160-175`); J is a comparator, never an ancestor (`evaluation.py:270`) | A, B, AB; H baseline; utility A/residence | **Yes**: for AB/SEX and AB/RAC1P every candidate of the same release's A role and of the B role is copied in (`evaluation.py:328-333`); B uses the H_B wire only | **Yes (replacement design)**: every role also receives the H-release's H-only model candidates (`evaluation.py:316-327`); no J ancestor by design since J is not on the wire | **Yes**: per-role seed depends only on anchor and role (`evaluation.py:306`), fitted on identical inputs → identical releases get identical candidates; constant releases are routed as H releases (`is_h`, `evaluation.py:272,297`) and score exactly 0 (stored `constant_best` recovery_over_H = −0.0 on 8/8; recounts.json) | **Yes**: selection on `attacker_validation`/`downstream_validation` (`evaluation.py:345-347`); evaluation replays the frozen choice, "no evaluation selection" (`evaluation.py:397-435`) | **OK** |
| **Final prospective 2016 (SaTML §VII)** (`research/pcrl-final-prospective-v1` 5e154e5c4) | replacement, frozen releases (`releases.py:19-20`) | same roles | **Yes**: "AB roles: all candidates of the same release's A role plus H's B role" (`audit_panel.py:7-12, 157-165`) | **Yes**: H-only candidates for every non-H release (`audit_panel.py:148-156`); "J is never an ancestor of a replacement release" (`:12`) | **Yes**: `role_seed(anchor, role)` independent of release (`audit_panel.py:51-53`) | **Yes**: fit pool `fit`, selection on `attack_val`/`task_val`, frozen replay on `final` (`audit_panel.py:225-246`); lock before final labels (`EVALUATION_LOCK.json` `final_labels_read_before_lock: false`) | **OK** |
| Task-aligned cuts / adaptive release / shared-context / privacy-first selector (Sept 23–24) | replacement, 2018 development | A, B, AB | **Yes**: "Complete coalition ancestors require A same-release and B H-only registries" (`research/pcrl-task-aligned-cuts-v1` `experiments/pcrl_task_aligned_cuts_v1/audit.py:452-476`); shared-context `audit_panel.py:17,102-105` reuses these legal routes | **Yes** (replacement; "J is a comparator, never an appended ancestor", `audit.py:456`) | Yes (inherited seeds/slates) | Yes: inner_selection → outer (privacy-first `PROTOCOL.md:97,213`) | **OK, with a resolution floor**: the "H-only-route floor" (below) |

## The privacy-first "H-only-route floor" (not a nesting defect; a resolution limit)

`research/pcrl-privacy-first-selector-v1` 8fdc61e39. Because the ignore-channel H-only ancestor is always
eligible, whenever a release's own token-reading AB/SEX attacker loses to H on `inner_selection`, the
frozen route is H-only and the release's measured AB/SEX recovery **equals H's exactly** on the outer pool
(`results/pcrl_privacy_first_selector_v1/PRELOCK_NOTES.md:21-36`; `RESEARCH_DECISION.md:25,30,36-37,105`).
Consequences: (i) the prior "DET_SEL4 −.0036" lead was D17's own measured gap to H, produced by a route
switch, not a graded reduction; (ii) R4 vs D4 on AB/SEX is an exact zero by construction; (iii) under this
slate no release can score below the H route on that endpoint. This is the correct nested behaviour (C1/C2
hold), but it acts as a **validation-level floor** on increments: the contract's "negative increments
retained unclipped" is true numerically (no clipping anywhere: `uncertainty.py:6`, negative stored values
such as T0_code A/SEX PWGTP −7e-05 are retained) yet route selection makes "increment = 0" an absorbing
outcome. The same mechanism operates in the SaTML development audit (e.g. `T0_code` A/RAC1P recovery over H
is exactly −0.0 in both weightings, `NARRATIVE_MANIFEST.json` `primary_sensitive_points`), although Q's own
increments over H are all positive (0.00086–0.00614), so Q is not at the floor.

## Catch-up routing answer

* Redesign studies (Sept 8–10): **no** — eligibility differs by whether a system had a saved training
  observer; disclosed in each study; source-guard's coalition ordering reverses between independent and
  catch-up scopes.
* Task-directed / prospective / Sept 23–24 studies: **yes** — the `catchup` slate there means the nested
  120/360-epoch MLP trajectory fitted fresh for every release (`audits.py` docstring; `audit_panel.py:3-6`),
  seeds depend only on anchor and role, and identical inputs yield identical candidates (privacy-first
  labels such endpoints `exact_zero_same_route`, `PROTOCOL.md:153`).

## Selection vs evaluation maximum answer

Separate in every study that produced SaTML evidence (task-directed, prospective) and in the redesign and
Sept 23–24 studies. **Not separate** in the replacement Stage-R screen and the stochastic-channel Stage-B
subsumption check; the first is cited by the SaTML paper (correction C3 / ledger R1) — see FINDINGS F8.

## What was not verified this phase

* Bit-level replay of any slate (requires private archives; `S3` archive bucket / relocated worktrees).
* Whether Study 6's omission changed any decision — the v6 text says the fix "can only raise measured
  increments", which is correct by inclusion for the validation-selected attacker but not guaranteed for
  every held-out estimate; magnitude unmeasured.
