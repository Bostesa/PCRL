# Restored runtime dependency correction, 2026-10-10

The first snapshot-only numerical replay, `R1_RESTORE_B0_J_READER`, failed
after 1.403 seconds at the first restored B0 CLI call. The failed queue
receipt, private stderr and original versioned snapshot/restore remain
preserved. No new defense fit, attack fit or release draw occurred in this
attempt.

A data-free import from the restored code tree reproduced the first failure:
`code_witness.model` imports
`results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.model.FixedScaler`,
but the pinned runtime source tar omitted `latent_repair_v1`. The earlier
source archive was therefore incomplete for standalone deployment. The
logical ordered-schema digest passed to the CLI is the newline-joined
83-name digest; it correctly differs from the JSON file's byte hash.

The successor source archive adds the tracked `latent_repair_v1` package.
An isolated import of `bank_anchored.r1_deploy` from its extracted bytes
passes with the same scientific virtual environment. The next snapshot uses
a new private custody plan and a new versioned target. All 50 deployment
data entries, four reader entries, selected source models, public tables,
policies, 32-row samples and J draws retain their prior hashes. The V2
replay worker captures any nested CLI traceback in an owner-only diagnostic
while leaving the public receipt aggregate-only. Two data-free diagnostic
tests pass.

This is an engineering dependency correction. It does not alter utility,
privacy, comparison, selection or uncertainty results. The corrected
snapshot-only numerical replay remains pending a dated queue amendment and
independent pre-execution audit.
