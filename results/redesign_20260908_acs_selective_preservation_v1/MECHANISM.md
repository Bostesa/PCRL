# Teacher structure and local optimization diagnostics

Separate affine decoders predict raw PCA16, retained E and removed qE=t−E from each frozen release; S conditions additionally predict retained S and qS. The removed residual is not pure sensitive information. Targets use original raw-PCA16 scales, with fitting-only centers/intercepts and fixed rcond=1e-12. Evaluation target variance is descriptive and never refits a decoder. Near-zero prior denominators (≤1e-12 in original-standardized units) give undefined ratios.

[All mean/prior/variance/rank results](MECHANISM.csv), [every coordinate](MECHANISM_COORDINATES.csv), [direct matching](DIRECT_MATCHING.csv), [gradient norms/dots/cosines](GRADIENT_DIAGNOSTICS.csv), [objective curves and coefficients](TRAINING_CURVES.csv), [retained/removed structure plot](retained_removed_structure.png), [raw/applied gradient directions](gradient_directions.png).

Low affine error supports recoverability; high error does not exclude nonlinear recovery. Real-attribute audits, task utility and these reconstruction diagnostics answer different questions.

## Development affine errors

Cells are three-seed mean MSE / mean per-seed MSE-to-prior ratio. Ratios are dimensionless; each target keeps its own fitting prior. Source-validation, every seed/coordinate and full spectra remain in the linked CSVs. Raw R/rho.1 geometry toward E/qE is unmeasured.

| Snapshot | Raw t: MSE / ratio | Retained E: MSE / ratio | Removed qE: MSE / ratio | Retained S: MSE / ratio | Removed qS: MSE / ratio |
| --- | --- | --- | --- | --- | --- |
| I: raw-PCA16 initialization | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 / 0.000000 |
| R, rho=0, W | 0.005847 / 0.005846 | 0.001400 / 0.003051 | 0.004447 / 0.008273 | undefined / undefined | undefined / undefined |
| E, rho=.1, W | 0.344233 / 0.344257 | 0.001279 / 0.002799 | 0.342954 / 0.635908 | undefined / undefined | undefined / undefined |
| E, rho=0, W | 0.404104 / 0.404238 | 0.000962 / 0.002092 | 0.403142 / 0.746718 | undefined / undefined | undefined / undefined |
| S, rho=.1, W | 0.357151 / 0.357132 | 0.193360 / 0.426204 | 0.163791 / 0.303782 | 0.002202 / 0.004752 | 0.354949 / 0.653575 |
| S, rho=0, W | 0.403198 / 0.402942 | 0.222456 / 0.488286 | 0.180743 / 0.335403 | 0.001379 / 0.002990 | 0.401819 / 0.740700 |
| R, rho=0, C | 0.007325 / 0.007321 | 0.001549 / 0.003367 | 0.005776 / 0.010726 | undefined / undefined | undefined / undefined |
| R, rho=0, D | 0.007489 / 0.007485 | 0.001647 / 0.003581 | 0.005842 / 0.010846 | undefined / undefined | undefined / undefined |
| E, rho=.1, C | 0.364142 / 0.364285 | 0.001734 / 0.003773 | 0.362408 / 0.671101 | undefined / undefined | undefined / undefined |
| E, rho=.1, D | 0.374065 / 0.374312 | 0.001890 / 0.004116 | 0.372175 / 0.688938 | undefined / undefined | undefined / undefined |
| E, rho=0, C | 0.378176 / 0.378187 | 0.001278 / 0.002784 | 0.376898 / 0.698162 | undefined / undefined | undefined / undefined |
| E, rho=0, D | 0.366461 / 0.366397 | 0.001565 / 0.003407 | 0.364896 / 0.675632 | undefined / undefined | undefined / undefined |
| S, rho=.1, C | 0.379525 / 0.379487 | 0.211203 / 0.465145 | 0.168321 / 0.312354 | 0.002583 / 0.005564 | 0.376941 / 0.694282 |
| S, rho=.1, D | 0.381661 / 0.381709 | 0.210749 / 0.464661 | 0.170912 / 0.317111 | 0.002817 / 0.006064 | 0.378843 / 0.697666 |
| S, rho=0, C | 0.382085 / 0.382029 | 0.214578 / 0.471959 | 0.167507 / 0.310701 | 0.001549 / 0.003365 | 0.380536 / 0.701151 |
| S, rho=0, D | 0.374788 / 0.374609 | 0.206997 / 0.454447 | 0.167791 / 0.311971 | 0.001907 / 0.004131 | 0.372881 / 0.688234 |
| E: real-attribute teacher | 0.539772 / 0.539961 | 0.000000 / 0.000000 | 0.539772 / 0.999196 | undefined / undefined | undefined / undefined |
| S: permuted-label teacher | 0.542718 / 0.542809 | 0.278577 / 0.612057 | 0.264141 / 0.489204 | 0.000000 / 0.000000 | 0.542718 / 1.000165 |

## Applied versus hypothetical gradients

Fixed first common-base minibatch diagnostics occur at I, W, common fork and final C/D. Raw protection is positive entropy-normalized CE; D applies coefficient−.1, C zero. rho0 has zero applied reconstruction and no decoder gradients/Adam updates even when a hypothetical reconstruction gradient is reported. Teacher/reconstruction, teacher/protection and source/protection dots/cosines keep raw and applied versions distinct; zero-vector cosine is undefined. Pre-observer diagnostics use explicitly labeled disposable initialized observers.

The common shared-fork diagnostic evaluates the prospective D objective hypothetically. No protection gradient is applied during common warmup. The arm-specific fork diagnostics record the actual C coefficient0 and D coefficient−.1 separately.

A projection retained rank is a map-construction property. Numerical release/design ranks at rcond1e-12 may include tiny directions left by the fixed float32 release convention; full spectra are preserved. Rank alone does not measure useful or sensitive recoverability. Reused R/rho.1 releases have no new E/qE decoder fit; missing geometry remains unmeasured.

No diagnostic takes an optimizer step or changes RNG/state. Local gradient opposition is not causal proof that an objective caused recoverability. Controlled rho contrasts provide stronger intervention evidence; reserved-task outcomes never select a diagnostic point.
