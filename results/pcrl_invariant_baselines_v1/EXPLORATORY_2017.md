# EXPLORATORY_2017 — cross-year development, not confirmation

**Every number on this page is EXPLORATORY DEVELOPMENT.** The 2017 locked
evaluation is finished and its `final_evaluation` partition is spent. This is not a
second confirmation of anything. The original frozen 2017 transport result keeps its
historical status and is neither restated nor overwritten here. **2017 is not a new
test merely because these methods had not been scored there.**

There are **no intervals**: the transport study's uncertainty machinery is not
re-applied to a spent partition. Years are reported separately and rows are never
pooled across years.

Probes were fitted on the 2017 fitting partitions and selected on the 2017
validation partitions, exactly as the transport study did, so selection never saw
the evaluation rows. `ev.load_final` is deliberately not called.

## Scope of this pass

Transported: the **6 repaired `spectral_riv*` arms**, which answers the registered
cross-year question. Not transported, each with its reason:

* `leace_A0` — acts on the frozen A0 neural auxiliary channel; its 2017 construction belongs to a different pipeline.
* `splince_A0` — same as leace_A0.
* `optnet16_*` — encoders were not persisted by this run's fit stage; persistence was added afterwards and the stage is deterministic, so the resume command reproduces them.

These are recorded as scoped-out components, not silently omitted. The resume
commands are in `REPRODUCE.md`.

## Seed means, scope `common_fresh`, budget 360

Additional recovery is relative to `H` on the same year, so **lower is less
disclosure**. Residence is a gain over `H`, so **higher is more capability**.

### Unweighted

| condition | A/SEX | AB/SEX | A/RAC1P | AB/RAC1P | residence gain |
|---|---|---|---|---|---|
| `H` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| `J` | +0.0013 | +0.0069 | +0.0039 | +0.0015 | +0.0189 |
| `spectral_C1` | +0.0170 | +0.0148 | +0.0471 | +0.0436 | +0.0282 |
| `spectral_L1` | +0.0198 | +0.0207 | +0.0608 | +0.0515 | +0.0285 |
| `spectral_L2` | +0.0196 | +0.0195 | +0.0545 | +0.0466 | +0.0288 |
| `spectral_S0` | +0.0346 | +0.0317 | +0.0758 | +0.0654 | +0.0304 |
| `E` | +0.0184 | +0.0161 | +0.0473 | +0.0371 | +0.0263 |
| `A0` | +0.0325 | +0.0312 | +0.0632 | +0.0564 | +0.0289 |
| `spectral_riv16_C1` | +0.0127 | +0.0103 | +0.0402 | +0.0359 | +0.0273 |
| `spectral_riv16_L1` | +0.0162 | +0.0190 | +0.0560 | +0.0498 | +0.0289 |
| `spectral_riv16_L2` | +0.0111 | +0.0127 | +0.0402 | +0.0361 | +0.0273 |
| `spectral_riv8_C1` | +0.0096 | +0.0100 | +0.0203 | +0.0223 | +0.0260 |
| `spectral_riv8_L1` | +0.0157 | +0.0170 | +0.0462 | +0.0444 | +0.0284 |
| `spectral_riv8_L2` | +0.0116 | +0.0118 | +0.0249 | +0.0237 | +0.0267 |

### Person-weighted

| condition | A/SEX | AB/SEX | A/RAC1P | AB/RAC1P | residence gain |
|---|---|---|---|---|---|
| `H` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| `J` | -0.0014 | +0.0038 | +0.0019 | -0.0001 | +0.0171 |
| `spectral_C1` | +0.0145 | +0.0136 | +0.0412 | +0.0380 | +0.0253 |
| `spectral_L1` | +0.0195 | +0.0205 | +0.0571 | +0.0457 | +0.0254 |
| `spectral_L2` | +0.0185 | +0.0183 | +0.0502 | +0.0417 | +0.0254 |
| `spectral_S0` | +0.0327 | +0.0306 | +0.0705 | +0.0593 | +0.0263 |
| `E` | +0.0162 | +0.0146 | +0.0421 | +0.0336 | +0.0219 |
| `A0` | +0.0284 | +0.0284 | +0.0606 | +0.0527 | +0.0266 |
| `spectral_riv16_C1` | +0.0110 | +0.0093 | +0.0380 | +0.0322 | +0.0256 |
| `spectral_riv16_L1` | +0.0150 | +0.0187 | +0.0535 | +0.0456 | +0.0258 |
| `spectral_riv16_L2` | +0.0073 | +0.0125 | +0.0397 | +0.0327 | +0.0255 |
| `spectral_riv8_C1` | +0.0078 | +0.0079 | +0.0194 | +0.0204 | +0.0251 |
| `spectral_riv8_L1` | +0.0144 | +0.0162 | +0.0434 | +0.0420 | +0.0270 |
| `spectral_riv8_L2` | +0.0076 | +0.0107 | +0.0228 | +0.0209 | +0.0252 |

## What this shows

**The 2017 look reproduces the direction of the 2018 verdict and widens the gap.**

* **The frozen neural channel `J` dominates every spectral arm by a wide margin.**
  Its additional race recovery is 0.0039 (`A/RAC1P`, unweighted) against 0.0203 for
  the best repaired arm — roughly a factor of five — while its residence gain,
  0.0189, is only modestly below the repaired arms' 0.026-0.029. Under
  person weighting `J` is at or below zero on two of the four sensitive endpoints.
* **The repaired arms do beat their own linear-moment controls here**, which they
  did not consistently do on 2018: `riv16_C1` reaches 0.0402 on `A/RAC1P` against
  `spectral_C1`'s 0.0471, and `riv8_C1` reaches 0.0203. Rank-8 compression is again
  the larger of the two effects on race recovery.
* **That does not rescue the verdict.** Beating a local control while losing to `J`
  by a factor of five is the same qualitative outcome the 2018 intervals reached,
  and it is the outcome the predecessor recorded on 2017 as well.

## What this cannot show

* No intervals, so **no significance claim of any kind** is made here.
* The partition is spent. These rows have been exposed before, and repeated use
  cannot be undone by any computation on them.
* Three seeds and no adjustment; seed-to-seed variability on race recovery is large
  relative to the differences between neighbouring arms.
* The erasure and OptNet baselines are **absent** from this table, so nothing here
  speaks to the external-baseline comparison. That comparison rests entirely on the
  2018 development evaluation.

## Data integrity for this stage

This stage is where the sporadic in-memory prediction corruption described in
`RUN_STATUS.md` failures item 4 occurred. Every occurrence was detected by the
scorer's full-schema validation and aborted its unit **before** any metrics file
was written, so **no corrupted array reached this table**. Units computed under the
diagnostic observer were quarantined and recomputed by the unpatched pipeline. The
scoring phase completed on its first attempt inside the bounded retry loop.

