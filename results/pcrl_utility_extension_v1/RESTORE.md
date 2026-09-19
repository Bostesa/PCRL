# RESTORE — archived PCRL result bytes

Archive: private S3 bucket (name in the private ledger under the git common dir,
`pcrl_parallel_handoff_v5/terminal_1/private/`), prefix `pcrl_utility_extension_v1/`:
`chunks/<chunk_id>.tar.zst` (zstd-compressed tar; first path component = group label),
`manifests/<chunk_id>.json` (relative paths, byte counts, per-file SHA-256, archive SHA-256,
object key and version ID, source commit, required readers), `verification/<chunk_id>.verify.json`
(independent read-back on AWS: stream hash + extraction + per-file hash).

Do not refit something that is archived. Restore it:

    python infra/pcrl_utility_extension_v1/restore.py --bucket <bucket> --prefix pcrl_utility_extension_v1 \
        --chunks <chunk_id> [...] --roots roots.json --verify-dir /tmp/verify

`roots.json` maps each group label to the worktree root it came from (see DELETION_LEDGER for the
group of every removed file). The script streams the exact object version, checks the archive
hash, extracts in place and re-hashes every file against its manifest; it reports FAILED on any
mismatch. Groups: rs_spectral_transport, rs_residual_spectral (residual-spectral worktree),
competitive, direct_adversarial, invariant, nonlinear_rank (their Terminal-1 worktrees), exec_*
(execution inputs, still present locally), results_run (cloud run outputs).
