# Operational recovery

The first optional-audit attempt stopped during reference loading with `KeyError: PCA16`, before fitting any attacker. The original PCA16 freeze uses a flat pool-to-hash mapping; multi-release freezes use a release-to-pool mapping. A strict adapter now checks the original PCA16 release name, dimension16, components0–15 and no-representation-fit flag before reading its flat hashes.

[Amendment01](EXECUTION_AMENDMENT_01.json) preserves original/new source hashes, failed-attempt time, the unchanged original protocol freeze and the passing pre-amendment core replay. [The exact patch](EXECUTION_AMENDMENT_01.patch) also adds explicit amendment-chain verification; historical hashes were not replaced. Extended release and selection records bind the amendment hash before fitting. All core units remain valid and were not retrained. The failed log and hash-matching original source bytes remain local under `operational_attempts/attempt_01/`; the failed log is also published as [EXTENSION_PREFLIGHT_FAILURE.txt](EXTENSION_PREFLIGHT_FAILURE.txt).

A focused artificial schema test verifies both historical forms and rejects incorrect PCA16 metadata. Only the affected optional preflight is retried; the1.149346-second failed attempt remains charged to scientific wall time.
