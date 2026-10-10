# R2 prefit correctness amendment request, 2026-10-10

R2 correctness V1 passed 10/10 queued data-free tests. No scientific FIT was
launched. Static review then found two untested implementation defects:

1. `np.unique(..., return_inverse=True)` returns an inverse array with the
   same two-dimensional shape as the prepared action IDs. O-J's Torch scatter
   used a flat source with this two-dimensional index. The index now flattens
   before scatter. A new fixture compares O-J Torch G exactly with NumPy G
   on two-dimensional finite actions, including all three actual-code views.
2. The S21 sampler drew recipient 2 first using recipient 1's HMAC domain
   coin and then drew recipient 1 using recipient 2's coin. It now draws the
   first emitted recipient 2 from recipient 2's domain and the conditional
   recipient 1 from recipient 1's domain. The exact law remains unchanged.
   A new known-CDF fixture checks this mapping.

These are prospective engineering corrections, not changes to the family,
support, task/privacy caps, reader objective or fitting schedule. V1 fixture
receipt and source pins remain intact. R0 must register a dated amendment to
the fixture test definition and queue before V2 runs; all real fits remain
unadmitted. V2 tests and code hashes bind both repairs.

V1 source hashes before correction:

- `r2_train.py`: `e7e50ee6809ee817874ce1e73f44e9bea9fa8248b830a8d745b24d316148cd50`.
- `r2_policy.py`: `11ad6cfc19436036616f8041c1952ed8f59c9bffb42f7cea9bd3656eb1d620c4`.

New exact hashes appear in `R2_CORRECTNESS_V2_SPEC_INDEX.json` after sealing.
