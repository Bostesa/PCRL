# T* role-only source-cell contract

Status: R01 source-custody check for the later A stage. The read-only
verification is `TSTAR_SOURCE_LOADER_VERIFICATION.json`; its exact file SHA-256,
role-pair SHA-256, receipt SHA-256, and source-binding SHA-256 values are the
machine-readable path-to-identity map. R04 must replay source/decoder parity
independently; this loader is not that oracle.

## Existing private files

In the predecessor portfolio worktree, use
`results/pcrl_method_portfolio_v1/shared/.private/tstar_score_bank_s{k}_v1/`
for `k=0,1`. The only admitted files there are `BANK_ADMISSION_RECEIPT.json`,
`bank_r1.npz`, `bank_r2.npz`, and
`tokens_{OSF_DEFENSE_FIT,ATTACK_FIT_NEW,INNER_SELECTION}.npz`. The receipt
binds each NPZ byte hash, fixed source JSONs, U0 checkpoint, 83-column schema,
ordered role identity and role-specific U0/Ucal reference hash. R01's
`SOURCE_AUDIT.json` independently checked those source-file bytes.

The exact token archive member set is `row_id`, `unit`, `cell_1`, `cell_2`,
`token_1`, `token_2`, all int64 vectors in the admitted role's source order.
`cell_r` is the private qpc fine partition input for recipient `r`;
`token_r == fine_to_hcal_token_r[cell_r]` is the inherited deterministic
coarse action. Never pass the coarse token as a channel input. There is no
CAL, HEAD or assessment token file in this contract.

The exact bank member set is `probability` float64 `[Z,K]`, `token_class`
int64 `[Z]`, `token_id` int64 `[Z]`, `input_class` int64 `[F]`, and
`fine_to_hcal_token` int64 `[F]`. For each seed, income has `F=64`, `Z=16`,
`K=2`; occupation has `F=641`, `Z=321`, `K=6`. Token IDs are `0..Z-1` in
bank order. Every bank vector has its `token_class` as argmax; the fine-to-token
map preserves that class. Exact probability-vector-plus-decision dedup leaves
all original IDs distinct, so `canonical_action_id == token_id` for this
source. The bank is already H-GLOBAL-TEMP calibrated and fixed.

## Runner API

Import `TStarSourceStore` from `r01.tstar_source` within the repository's
`results.pcrl_latent_and_sampled_methods_v1` package. Construct it with the
predecessor portfolio package directory, predecessor repair
`m1/DATA_ACCESS_MANIFEST.json`, current R01 `SOURCE_AUDIT.json`, and model
seed. `read_bank(1|2)` returns fixed actions; `read_role_cells(role)` accepts
only FIT, ATT or INNER and returns private role-specific cells/tokens. The
constructor and each read verify hashes and source binding before returning
arrays. The caller must keep returned cells and IDs private, use R00's A-stage
seal/queue, and join by ordered `row_id,unit` rather than positional guesswork.

The audit read no task or SEX labels, raw `X`, mixed-role source archive, or
assessment arrays; it performed no Adult fit. The existing predecessor bank
receipt says `ROLE_ONLY_BANK_ADMITTED`, while independent numerical
source/decoder parity remains pending for R04.
