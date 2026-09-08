# Measured first seed, before expansion

Seed0 completed the frozen full protocol, including all controls, five new
training runs, independent task/attack probes and fresh final evaluation.
`/usr/bin/time -p`: **25.45 seconds wall**,24.84 user,.42 system.
Inside-seed time: see `seed_0/metrics.json` (about24.70 seconds).

Estimate for the remaining two identical seeds: **2 ×25.45 =50.90 seconds**.
This is comfortably within the30-minute experiment budget. Continue unchanged
seeds1 and2: no generator, checkpoint, threshold, grid, training, audit or
selection changes. This estimate was saved before launching those seeds.

First-seed validation observation: oracle/prediction-only controls passed;
selected utility-eligible fixed/dual variants still failed protection. These
observations do not change the predeclared experiment.
