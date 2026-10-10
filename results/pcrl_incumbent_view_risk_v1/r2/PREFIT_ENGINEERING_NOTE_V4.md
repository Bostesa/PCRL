# R2 prefit fixture correction V4

The queued V3 correctness unit failed one of 15 tests. The failing synthetic
fallback fixture used reader names ending in `/constant`, while the source
reader fitter names the constant-prior family `/prior`. The fallback's
nomination function requires a pair `/prior` reader. This was a fixture
construction error: the actual training fitter emits the required family.

V3 made no real-data reads. Its original failed receipt is preserved in
owner-only custody. The public failed receipt records the same error after
replacing only the personal worktree prefix in its traceback. V4 changes
only the synthetic reader names in that fixture to `/prior`; policy,
objective, reader fitting, utility, thresholds, and selection code stay
unchanged. V4 must be a new queued admission-stage unit after a dated R0
amendment. No real fit is admitted by this note.
