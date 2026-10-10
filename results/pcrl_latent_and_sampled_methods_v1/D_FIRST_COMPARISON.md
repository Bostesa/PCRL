# First real latent-repair comparison (paired seed 0)

Status: **independently replayed exploratory diagnostic**, not a competitive method result. This comparison uses the admitted Adult development rows. The old assessment remained embargoed.

Both releases transmitted a 10-bit hard code per recipient, a probability vector and a decision computed solely from that code. T was a task-only hard-code compressor. D-K started from the same task-trained initialization and applied a fixed-kernel distribution-matching objective. Fresh complete-view attackers were fitted on ATTACK_FIT_NEW, then scored on INNER_SELECTION (2,235 people). The independent replay reproduced the raw-input-to-wire output, all task endpoints, attack selections, complete views, controls and AUCs exactly. Its public receipt is `r04/FIRST_PAIR_INDEPENDENT_REPLAY.json` (SHA-256 `6ebf1a518ce23ef32835c3f39cdac4c5663e56a14df51f4c8b38866a9e68bfad`).

| Arm | Income accuracy | Income NLL | Income Brier | Occupation accuracy | Occupation NLL | Occupation Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| U0 | .851902 | .330550 | .203973 | .490828 | 1.268392 | .648344 |
| Ucal | .851902 | .328190 | .204783 | .490828 | 1.267029 | .648991 |
| T | .854139 | .323721 | .203957 | .489933 | 1.293619 | .658673 |
| D-K | .849664 | .334466 | .210943 | .474273 | 1.299836 | .662251 |

| Arm | Recipient 1 SEX AUC | Recipient 2 SEX AUC | Coalition SEX AUC |
| --- | ---: | ---: | ---: |
| T | .710718 | .795267 | .832582 |
| D-K | .671178 | .753706 | .805905 |

D-K reduced coalition recovery relative to T by **.026676 AUC** in this seed, while both individual recoveries also fell. It failed the prespecified point utility caps against both U0 and Ucal: income Brier excess was .006970/.006160 (cap .005); occupation accuracy loss was .016555 (cap .010), with NLL and Brier excesses also above their caps. T itself failed occupation NLL and Brier caps. Thus D-K's privacy difference is a descriptive tradeoff, not an eligible win. The prespecified two-seed matched-control comparison and simultaneous intervals are still pending.

First source attempt `D_FIRST_T_DK_S0` failed before a role read because it pointed to the tracked copy of a manifest with no adjacent private exports. Dated amendment `AMENDMENT_20261010_D_FIRST_MANIFEST_PATH.json` preserved that attempt and bound the corrected predecessor worktree manifest. Replacement `D_FIRST_T_DK_S0_R1` completed. No method, data role, or endpoint changed.
