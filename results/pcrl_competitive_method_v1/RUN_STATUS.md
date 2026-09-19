# RUN_STATUS — pcrl_competitive_method_v1

Session start `2026-09-18T21:47Z` (worktree created); ceiling `2026-09-19T09:47Z`; final
hour reserved for analysis, verification and publication. A session interruption
occurred between ~22:10Z and ~02:50Z (no computation ran; the machine was restarted in
that interval, which is why the resource picture below differs from the start).

## Resources

| time (UTC) | swap used | free / inactive | compressor | load | decision |
|---|---|---|---|---|---|
| 2026-09-18 21:51 | 11.1 of 12.3 GB | 0.06 / 6.4 GB | 7.6 GB | 4.6 | one worker |
| 2026-09-19 02:54 | 0 | 0.39 / 8.4 GB | 2.8 GB | 3.5 | 3 workers (one per anchor), 1 thread each |

The unrelated heavy workloads present at 21:51Z were no longer running at 02:54Z; none
was stopped by this study. Concurrency was raised only after this measurement, and is
re-measured between stages (drop to one worker if swap use exceeds 2 GB).

## Pre-lock implementation corrections (no outcome seen)

### P1 — basis redundancy tolerance (outcome-free invariant)

The first Track E fitting pilot (seed 0, training rows only, nothing scored) produced
basis dimensions 10/21 instead of the specified 6/10. Cause, established: the service
probability pairs sum to 1 only to float32 rounding (max deviation 8.9e-8), and the
redundancy tolerance `1e-8` was below that rounding. Corrected to `1e-6`; a float32
fixture was added; the pilot was deleted and refit (dimensions now 6/10/16). Affected
units: all Track E maps; refit required and done before lock.

### P2 — audit compaction (disk)

A full audit stores ~106 MB; ~300 audits would approach the 33 GB free. Added the
verified compaction of PROTOCOL §3 before any new outcome. Applied to `ref_A0` seed 0
(1020 -> 60 entries, bitwise-equal kept arrays, original sha256 recorded).

## Pre-lock measurements that touched no new outcome

* Track N pilot (scratch directory, deleted): `N_A0_g100_C1_b100` reproduced historical
  `dax16_C1_b100` seed 0 bit-exactly on all 7 pools with identical monitor scores and
  selected step 300; `N_J_g000_C1_b300` ran (selected step 0). 10.9 s per unit.
* `ref_A0` seed 0 audit: reproduced the historical audit of the bitwise-A0 channel
  (`dax16_C1_b010`) exactly — 144,778 numeric metrics, 0 differences, predictions
  bitwise equal. 41 s, 0.99 GB peak RSS.

## Amendments

### Amendment 1 — concurrency reduced to one worker; single global audit queue

**Declared `2026-09-19T03:06Z`, during Track N fitting, before any new audit outcome.**
No 2018 or 2017 outcome of any new release had been opened.

At `03:04Z` swap use was 5.6 GB (0 at `02:54Z`), above the 2 GB threshold PROTOCOL §9
registered for dropping to one worker. The three per-anchor workers were then ~130 MB
resident each, and several unrelated sessions had started in the same interval; the
cause of the swap growth is **not established** and is not attributed to this study.
The rule was applied as registered:

* the three per-anchor workers were stopped between units (every fit directory had its
  completion marker; no partial unit existed: 25/22/21 completed Track N units);
* a single global orchestrator (`run_all.py`) resumes the remaining fits and runs **one
  audit queue across all anchors**, ordered by (priority, anchor, track, unit), so the
  registered cutoffs remove a reduction slice uniformly from every anchor;
* units are claimed atomically and the identity table is file-locked, so concurrency can
  be raised again **only if swap use is below 1 GB at a stage boundary** (a rule declared
  here, before any audit outcome).

Affected units: none refit; scheduling only. The prospective cutoffs of PROTOCOL §9
(`06:45Z` for priority 3, `07:00Z` for priority 2) are unchanged.

### Amendment 2 — full-span projector made exactly zero

**Declared `~03:10Z`, before any audit of any new release had started (0 audit lines in
the orchestrator log).** Outcome-free invariant: for `k = r` the projector
`I - U U'` is analytically zero, but its floating residue (~1e-16) made the pilot
report "realised rank 16" for what METHOD §2.6 describes as a constant map. The residue
is a deterministic function of `z`; a standardising attacker could amplify it into a
spurious signal. `projection.projector` now returns exact zeros when `U` spans the
space (new fixture, 16/16 tests pass). The 18 full-span releases were rebuilt as the
exact constant `mu` (realised rank 0; `maps.joblib`, `track_e_fit.json` and markers
updated with new hashes). Within each channel and anchor the L/C/LX full-span
releases are now bitwise identical and are audited once. No other unit is affected.

## Incidents

None yet.

## Response to Terminal 2 review (handoff v4, `2026-09-19T02:54Z`), recorded `~03:40Z`

Read during the 2018 audit stage, after fitting, before any panel or new test-split
table was produced. No decision rule changes; these are declarations and additional
reporting.

* **B1 (one correction level).** Already registered (PROTOCOL §5-6): decisions use the
  candidate-wide bound within family P only; both that level and the within-contrast
  level are written for every contrast. `CORRECTIONS.md` addendum corrects this study's
  own §6 wording about the predecessor's C1-vs-L2 residence contrast.
* **B2 (measured rank, not cardinality).** Track E reports the measured cross-moment
  spectra, the relative tolerance, realised release rank and per-class support per role
  and anchor (`TRACK_E_FITS.json`). No rank is set from schema cardinality.
* **B4 (counts).** Planned / released / distinct / duplicate-of-untouched-start /
  audited are counted separately in `COUNTS.json`.
* **B5 (projection form).** Declared form: whitened projection
  `z_out = mu + (z-mu) R P S`, support defined at relative tolerance `1e-10`. **Measured:
  the supported rank is 16 of 16 for A0 and J on every anchor at every tolerance from
  1e-14 to 1e-6**, so `R S = I` exactly, no direction is off-support, and the whitened
  and LEACE-subtractive forms coincide on training AND transported rows (off-support
  norm identically zero). `P` is `r x r` in whitened coordinates, so its range lies in
  the support by construction. Were a channel rank-deficient, this form would delete
  off-support components of new rows; that case does not arise here.
