# Measurement before expanding beyond seed 0

The complete frozen seed-0 pilot finished successfully:

- Runner complete-seed elapsed: 10.2539945830 seconds.
- Includes shared pretraining (0.6538295830 s), all five arms, final erasers,
  all linear and MLP task/attacker fitting, validation selection, final test.
- `/usr/bin/time -p`: real 10.87 s, user 10.59 s, sys 0.18 s.
- Estimate for remaining two seeds at measured process rate: 2×10.87=21.74 s.
  Allow one minute. This is well below the 30-minute experiment-runtime budget.
- Continue predeclared seeds 1 and 2 with the identical frozen generator,
  configuration, protection grid, probe budgets, source hashes and selection.
- No budget reduction, generator redesign, additional hyperparameters or
  final-test-based method selection. Seed 0 remains valid evidence.

Hardware: existing Apple M4 Pro, 24 GiB RAM, CPU only, one numerical-library
thread. No process interrupted, downloads performed, or paid resources used.
