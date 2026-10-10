# R15 review: independent T* S0 fixed pair replay

Status: **SCOPED STATIC GO** for an R03 verification unit. This is a source review, not a numerical replay result.

- Worker: `r04/replay_a_fixed_tstar_pair.py`, SHA-256 `5b0e806ce0c0698fccd2cfcca05b7d9508a584d7fa2fd93cee9d2bcaeddce60e`.
- Proposal: `r04/A_TSTAR_S0_FIXED_PAIR_VERIFICATION_PROPOSAL.json`, SHA-256 `2abeefd60be1981a09569c1334baeff98799ab933d1c80489311d102a1c4a021`.
- Both files were committed at `b9bd13754` when reviewed.

The worker imports only the R04 independent source helper and NumPy. It opens role-only T* banks, tokens, ordered reference records, and saved ATT/INNER releases; those contracts contain no SEX or task labels. It does not import the release writer, candidate runner, attack scorer, or endpoint implementation. It independently derives both recipient codes from the fine-cell map, retrieves the exact emitted probability and decision, and hashes the full public finite wire for each recipient. It checks ordered people/groups, duplicate-group consistency, source and release hashes, and the aligned pair of recipient codes.

The proposal names both completed release dependencies and lists 21 source files before the new verification amendment, which is the 22nd source in the generated spec. It sets a bounded R03 verification unit. The worker does not independently authenticate the command-line amendment SHA or active queue slot; R03 must seal and enforce the exact immutable unit spec before role reads. This review does not establish a numerical PASS, attack result, or endpoint result.
