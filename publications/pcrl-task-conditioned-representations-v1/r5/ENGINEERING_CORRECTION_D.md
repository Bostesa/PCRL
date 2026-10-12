# Reader scale correction D

Dated 2026-10-12 UTC. The original science lock and its commit `71262c7b1d72f3ba02659593da4ab4c6fc7856eb` remain preserved. R00 owns the effective amendment and stage registration. This correction changes reader engineering before any Adult reader is fitted; it uses no recovery outcome.

R6 identified that the previous float64 machine-epsilon denominator floor suppressed informative coordinates at scale 1e-25. Keeping a coordinate in the matrix did not ensure that an attacker could use its signal. The registered attack was intended to standardize every finite positive-variance coordinate on ATT training people.

`FiniteScaler` now uses every strictly positive finite empirical population standard deviation unchanged. Exactly zero standard deviation gets scale 1. Mean and standard deviation must remain finite; no coordinate is removed. Weighted legacy readers use the same rule with their exact emitted-law weights. The legacy adapter substitutes this scaler for the inherited sklearn scaler, while retaining the validated classifier recipes, finite laws and seed rules. Receipt fields explicitly report the absence of a positive-variance floor and count exactly constant coordinates. New release contents, task heads, reader slates, endpoint IDs and multiplicity do not change.

The unit builder's maximum wall time and CPU reservation now both equal 1200 seconds. Its previous 1200-second wall cap with 900-second reservation failed the queue's resource contract. This changes a prospective accounting bound, not model fitting opportunities.

Root executed RED/GREEN checks in the shared queue. The 1e-25 population-variance and actual fresh-LR planted-bit fixtures exposed the previous suppression and passed after correction. Weighted tiny-coordinate and legacy-adapter fixtures also passed in the combined D/E suite. Public receipts preserve these failures and repairs. No Adult arrays or numeric jobs were opened/launched by R5 during this correction.
