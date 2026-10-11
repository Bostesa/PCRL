# Primary development comparison

| Release | Accuracy I/O (%) | Worst ΔNLL I/O | Worst ΔBrier I/O | AUC R1/R2/pair | Status |
|---|---:|---:|---:|---:|---|
| P0 incumbent | 85.15/49.13 | -0.00187/0.00881 | 0.00239/0.00443 | 0.702878/0.761953/0.802763 | point eligible; 10/24 utility bounds pass |
| G shared rate | 85.15/49.13 | -0.00114/0.00921 | 0.00272/0.00469 | 0.702069/0.760298/0.801201 | point eligible; 10/24 utility bounds pass |
| C conditional rates | 85.15/49.13 | -0.00078/0.01070 | 0.00287/0.00518 | not run | point ineligible; attacks untriggered |
| W selected kernel | 85.15/49.13 | 0.00482/0.00977 | 0.00493/0.00465 | 0.698759/0.761808/0.800805 | point eligible; 9/24 utility bounds pass |
| Sequential 1→2 control | 85.15/49.13 | 0.00456/0.00931 | 0.00468/0.00475 | 0.698367/0.759333/0.798701 | point eligible; 9/24 utility bounds pass |

I/O means income/occupation. Accuracy and AUC are equal means over two paired source seeds. Loss excess is the worst of both seeds and both frozen references, U0 and Ucal; caps are 0.010 nats and 0.005 summed-class Brier. Confidence losses integrate the actual random output law, never the mean probability. All decisions remain those of P0. Point eligibility is not a passed uncertainty conjunction. C has no fresh recovery result because its paired utility gate failed. W is the FIT-selected nominee; S12 is the frozen-pool descriptive strongest control, not a replacement nominee.
