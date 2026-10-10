# D controls follow-up handoff

The first seed-0 T/D-K fit completed before the controls follow-up. Its public
receipt is `D_FIRST_S0_R1_RECEIPT.json`, SHA-256
`c3abf8a3f3287b53a551e2e26c854cd22387da53fffd680cd1a4546db67879b5`.
It selects warm length 15, continuation length 15, and privacy coefficient
1.2998533524563551. The owner-only first package holds the exact warm state.

`run_followup.py` is seed generic. For each seed it requires a completed first
T/D-K receipt and its hash-pinned warm state, then fits L, S12, S21, ordinary
adversarial C, and unchanged-latent F. It calibrates each arm on the admitted
CAL role and writes owner-only ATT/INNER releases. It checks the active R03
`d_followup` queue slot before any role read. A dated effective seal amendment
must be registered separately from the immutable original seal and be the
latest R03 amendment affecting `d_followup`. A2 must bind
`controls_worker_sha256`, `first_receipt_sha256_by_seed[str(seed)]`,
`role_manifest_sha256`, and `source_seed_sha256_by_seed[str(seed)]`; the queue
unit spec must bind A2's byte hash. Seed-1 T/D-K used a separate seed-1 worker
and A1 amendment before A2. Both controls fits completed under R03. Public
receipts `D_FOLLOWUP_CONTROLS_S0_RECEIPT.json` and
`D_FOLLOWUP_CONTROLS_S1_RECEIPT.json` have SHA-256 values
`a1901a87cd808215fddd0943befc0e5add4cc0c987013f6b0dd54c5ea24697ac`
and `b3f8153f0e8b2ecac6d11a47c20c4077f134b8c15c13ed6e6cd64a52c396e2e4`,
respectively. Each contains C/F/L/S12/S21 packages and ATT/INNER releases.
Fresh attacks and independent all-arm replay remain separate queue work.

`f_adapter.py` passes F's full scaled 16-dimensional latent, probabilities,
decisions, and all public metadata into R02's complete view. F uses
`reader_plan(family="D", code_bits=False)`; no code-ID or bit specialist applies.
The controls suite passes 19 tests, including queue rejection before reads,
warm-state hash validation, F private package roundtrip, and F attack view.

Commit `a51d1d847` contains the two D follow-up worker files and five A
controls files that another agent had already staged in the shared Git index:
`CONTROL_SPEC.md`, `deterministic_policy.py`, `deterministic_trainer.py`,
`test_deterministic_policy.py`, and `test_deterministic_trainer.py` under
`sampled_channel_v1/controls/`. Their content was not modified by R08. The
commit was left intact; the A implementer and coordinator were notified.
