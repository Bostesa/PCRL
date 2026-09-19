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

None yet.

## Incidents

None yet.
