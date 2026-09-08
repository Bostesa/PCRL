# Input boundary and matched initialization

**The restricted students and source banks use only their fixed teacher on each new example.** The completed checks found no raw-input route in K. This establishes the implemented access boundary; it does not make the teacher private or turn improved finite-audit prediction into information creation.

F receives `[u,v]`, where `u=(T−m16)/s16` and `v=(PCA32−m32)/s32`. K accepts a 16-coordinate teacher argument and constructs `[u,zero32]` internally. Its inference/training interfaces reject non-None raw arguments; the source-bank trainer has no raw-input or protected-label argument. Original representation-fitting means/scales are immutable. No erased-coordinate whitening, raw target, skip connection or auxiliary reconstruction enters K optimization. Frozen raw/E/qE targets belong only to the separate post-training geometry analysis. E is already informed by real fitting attributes; S uses the previously frozen paired shuffle.

Within each seed, E/S, F/K and the source banks share **identical full initial tensors**, including the genuine original source heads and unused decoder. The wider mapper does not change those heads through constructor RNG consumption. The dedicated sign-canonicalized Gaussian QR matrix Q uses seed `20260909+seed`; its saved value is reused, with deterministic identity validation rather than a new choice. For `D=diag(s16)`, initialization is

```
W1[:, :16] = [ I ; -I ; Q ; -Q ]     W1[:, 16:] = 0
b1 = [ 1 ; -1 ; 1 ; -1 ]
W2 = [ D, -D, 0, 0 ]                 b2 = m16 - s16
```

The shifted ReLU pairs return T in real arithmetic. Float64 preprocessing followed by float32 inference passed the frozen `atol=rtol=1e-5` teacher-parity rule on fitting and held-out inputs. Direct-teacher utility/audit evidence is therefore a named starting-function reference; it is not relabeled as newly fitted I evidence. Historical raw-input models have different initialization and remain contextual references, while the new F conditions provide the matched controls.

Observed initialization errors, in raw teacher-coordinate units, are:

| Parity checks | Maximum absolute error range | RMS error range |
| --- | ---: | ---: |
| 12 prefit checks, fitting rows | 2.38419e-7–4.76837e-7 | 2.46190e-8–2.99816e-8 |
| 18 completed-unit development checks | 2.38419e-7–4.76837e-7 | 2.53250e-8–3.01569e-8 |
| All 126 completed-unit/pool checks | 2.38419e-7–4.76837e-7 | 2.44997e-8–3.03467e-8 |

The overall maximum absolute error is **4.76837e-7** and the largest pool RMS is **3.03467e-8**. [Prefit evidence](PREFIT_IDENTITY.json), each unit's `release_freeze.json` and `initial_evaluation_parity.json`, and [independent replay](SCORE_REPLAY.json) retain the underlying checks. The 126 checks cover six fitting/validation pools plus development evaluation in each of 18 units. Repeated F/K/bank starting-function checks are identity checks, not independent estimates; checking a bank's initial mapper does not expose its hidden 16 coordinates as a bank release.

| Parameter block | Stored parameters | F can update | K/bank can update |
| --- | ---: | ---: | ---: |
| Mapper, including 2,048 raw-input weights | 4,176 | 4,176 | 2,128 |
| Three native source heads | 51 | 51 | 51 |
| Unused reconstruction decoder | 3,168 | 0 | 0 |
| Total model | 7,395 | 4,227 | 2,179 |

“Can update” describes the objective's usable parameter blocks, not a count of nonzero gradients on every batch. In particular, the initially zero auxiliary readout weights remain trainable and are included. F/K match stored architecture and initialization; K's disabled connections reduce effective usage as part of the input restriction. The two observers add 6,699 parameters for main C/D training (SEX 3,234; RAC1P 3,465); bank fitting constructs neither observer. Fixed teacher maps and preprocessing are separate from these learned parameter counts.

All K/bank raw weights and their Adam first/second moments remain exactly zero at every checked checkpoint. Decoder weights remain initial, decoder gradients are absent, and no decoder Adam state exists. F's raw route is usable: initial source-gradient norms on its raw-weight block range **0.040952–0.065773** across the six teacher/seed units. The separate disposable fixture gives nonzero gradients for all 32 raw columns even at exactly `u=0`, without updating the real model. Final F raw-block weight norms range **3.523874–4.005593**. These establish connectivity and use, not a causal decomposition of final task losses.

Within every main unit, C/D forks copy model, observers and complete Adam state exactly and share all subsequent batches. No mapper/Adam update occurs during observer warmup. [Training replay](TRAINING_REPLAY.json) passed all 18 units and independently reproduced all 114 fixed-batch gradient diagnostics with maximum scalar discrepancy **0**. [Score replay](SCORE_REPLAY.json) independently reconstructs serialized teacher-only releases/native heads, checks unchanged outputs when an external raw array is perturbed while T stays fixed, and replays composed attacks. [Focused tests](focused_tests.txt), [all gradient/block measurements](GRADIENT_DIAGNOSTICS.csv), and [protocol](PROTOCOL.md) give the detailed contracts.

For fixed public model parameters on a new example, K is `H=g(T)` and its bank is three fixed source probabilities computed from T. Every attack on H therefore composes into an attack on T. The [composition evidence](COMPOSED_ATTACKS.md) concerns that larger fitted function family and its source/observer exposure; it does not replace the direct teacher's matched independent audit, establish training-data privacy, or solve combined-access coordination.
