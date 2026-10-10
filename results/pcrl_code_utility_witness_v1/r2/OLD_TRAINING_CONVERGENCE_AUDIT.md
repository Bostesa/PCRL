# Old ten-bit task training: scoped pre-refit audit

This is a source/aggregate-receipt audit of the prior latent-repair study.
The new R2 queued `LOSS_DECOMPOSITION.csv` must supply matched current-role
losses before any conclusion about the new witness. The prior study reused
the same INNER people in both seeds.

| Seed | T warm selected / continuation selected | Full-FIT warm / continuation updates | F selected epochs (income / occupation) | F full-FIT updates | INNER occupied T states (income / occupation) | INNER occupied D-K states |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 15 / 15 | 915 / 915 | 30 / 35 | 1,830 / 2,135 | 11 / 168 | 14 / 166 |
| 1 | 20 / 20 | 1,220 / 1,220 | 35 / 35 | 2,135 / 2,135 | 11 / 176 | 28 / 165 |

Source: pinned prior `D_FIRST_S0_R1_RECEIPT.json`,
`D_FIRST_S1_RECEIPT.json`, and `D_FOLLOWUP_CONTROLS_S0/S1_RECEIPT.json`.
These are reported fitting and release aggregate counts. `F` discloses 16
continuous features per purpose and does not have a finite 1,024-state code.
The hard-code receipts report at most 1,024 possible states per purpose; the
occupied counts above refer only to INNER, not a population codebook size.

The old T recipe used a non-amplifying scaler fitted on defense FIT latents,
Adam at learning rate 0.001, batch size 256, true-label cross-entropy plus
0.5 teacher KL, and hard forward bits with a straight-through gradient
surrogate. The initial split fit and continuation each had a fixed 20-epoch
horizon and task-NLL checkpoints at 5, 10, 15 and 20. Source code selects
the lowest checkpoint NLL with a strict `<` comparison; there is no
patience-based stopping or training beyond 20 epochs in either phase. The
selected stage lengths were then refitted on all 15,434 defense FIT rows.
The old D-K arm started from the same full-FIT warm state, had the same
continuation update count and added fixed FIT-gradient-ratio kernel pressure.

The continuous F comparator was fitted differently: unchanged scaled
latents, separate continuous task heads, independent per-purpose checkpoint
selection and longer full-FIT schedules. F-versus-T and T-versus-D-K losses
are therefore descriptive matched-row differences, not isolated causal
estimates of code discretization or privacy penalty.

All prior arms used the inherited held-out global temperature family on
their actual CALIBRATION_HELDOUT outputs. Public receipts bind package
manifests, source model, role manifest, release hashes and calibration code.
The saved prior `FitResult.history` has per-epoch training loss and selected
validation NLL in memory, but the first/second-seed public fit receipts and
package fit receipts do not serialize that curve. The endpoint and state
counts show the result of the declared recipe; they cannot prove optimizer
convergence. The seed-one T stage selects the last available checkpoint,
which supports testing a longer predeclared task-only recipe. It does not
prove that extra epochs will meet the unchanged utility caps.

The queued R2 loss audit will independently restore the deployed hard-code
services, recompute true-label metrics on identical INNER people, and separate
uncalibrated teacher KL from calibrated transmitted-score NLL. No protected
labels enter this task-only utility scoring.
