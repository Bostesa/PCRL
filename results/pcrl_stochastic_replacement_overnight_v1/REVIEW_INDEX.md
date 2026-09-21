# REVIEW_INDEX — read in this order

| # | file | why |
|---|---|---|
| 1 | `RESEARCH_DECISION.md` | the verdict, the numbers, the refuted prediction |
| 2 | `PRECURSOR_CORRECTIONS.md` | seven corrections to the closed precursor, with fixtures |
| 3 | `RELEASE_CONTRACT.md` | S1 vs S2 and the threat-model evidence search |
| 4 | `PROTOCOL.md` · `METHOD.md` · `STATISTICAL_PLAN.md` · `DATA_USE.md` · `RUN_MATRIX.json` | the registration, locked before outcomes |
| 5 | `gates/G_R.json` | the gate that closed the branch |
| 6 | `UTILITY_PANEL.png` + `UTILITY_PANEL.json` | the figure and its machine-readable twin |
| 7 | `stage_R/` | per-family/resolution records, intervals, adjusted intervals |
| 8 | `synthetic/DETERMINISTIC_COMPARISON.json` | certified stochastic-vs-deterministic comparison |
| 9 | `VALIDATION.md` + `REPLAY.json` | independent replay, 14/14 |
| 10 | `PAPER_ADDENDUM.md` | what may and may not be written |
| 11 | `RUN_STATUS.md` · `COST_AND_SHUTDOWN.md` · `REPRODUCTION.md` · `HANDOFF.json` | operations |

## The three things a reviewer should push on

1. **The headline positive is utility-only.** `H+raw` beating `J` by 0.0153/0.0185 nats says nothing
   about disclosure. No privacy constraint was ever applied on ACS.
2. **Only two code families at two resolutions were tested.** The failure is attributed to
   quantization on the evidence that the unquantized view has no deficit — that attribution is a
   measurement, but "a better coder would fix it" is an **unregistered hypothesis**, not a finding.
3. **The synthetic coalition advantage is a different model.** It is certified within that model and
   carries no ACS implication.

## Known weaknesses, stated by the author

* `k=256` is confounded: it is both a finer code and a 256-column one-hot on 4,539 fitting rows, so
  its worse result mixes coder and estimator. `k=64` is the cleaner reading.
* The screen uses one probe family. A different family could move the deficit.
* Three anchors share people; the aggregate resamples the union household set once, which is the
  right treatment but leaves the anchors far from independent replicates.
* `LEACE-on-A0`, `SPLINCE-on-A0` and OptNet were never compared — their stage never ran.
