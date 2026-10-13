# Supplementary applicable-reader audit

The locked K108 base9 diagnostic family remains unchanged. A separate K18 family assesses six fixed U0/RAW.3epoch5/JOINT.3epoch5 full and projected releases against full RAW.3epoch5. Each local envelope includes all18 frozen base9 and extended9 readers. Each pair envelope includes its own18 plus36 exact local-ignore aliases. Existing reader fits and ATT orientations are reused.

Metadata-only registry construction:

`python -m occupation_bottleneck.complete_view_audit --phase registry --output PUBLIC_REGISTRY.json`

Root must freeze and commit the registry, code and supplementary ROOT lock before the first successful INNER score. Supplementary lock fields are `owner:"ROOT"`, `status:"FROZEN"`, `registry_sha256`, `code_sha256`, `reporting_lock_sha256`. This binds the supplementary definitions to the canonical reporting freeze without changing the original registry definitions.

Root-owned `attacks` queue execution:

`python -m occupation_bottleneck.complete_view_audit --phase audit --spec PRIVATE_CONFIG.json`

Config contains `unit_id`, the operational `reporting_seal`, `registry` and `supplementary_lock` pins, `unit_input`, `arrays` from REPORT_INPUT_INDEX, fresh `private_output` and R8 `public_output` directories. Register all pins and code as queue sources. The worker verifies commitment and exact frozen definitions before numerical imports or person reads.

The supplemental registry follows the independent assessment API. It uses the same B2048 paired duplicate-group draws, seed2026101301, maximizes every applicable-reader envelope within each draw, averages both source seeds within each privacy contrast, and uses K18 Bonferroni Student-t bounds with df2233. Missing/failed required control gates retain MISSING endpoint slots at K18. Numerical code reuses runner-owned R8 arithmetic and never imports independent scientific code.

Outputs are COMPLETE_VIEW_AUDIT.json (18 endpoints and36 arm/seed/view envelope audits), COMPLETE_VIEW_ENVELOPE_AUDIT.csv, COMPLETE_VIEW_GAP_TO_TARGET.csv, and private COMPLETE_VIEW_ENDPOINT_BOOTSTRAP_DRAWS.npz. Expanded-versus-base9 AUC differences disclose whether an applicable score/decision/confidence reader strengthens recovery. Neither family carries a confirmation claim.
