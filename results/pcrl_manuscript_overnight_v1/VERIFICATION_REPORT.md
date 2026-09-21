# VERIFICATION_REPORT — what was independently checked, and what was not

Session of 2026-09-21. All work is arithmetic on committed evidence, small synthetic fixtures, and
document compilation. **No ACS model was fitted, no protected evaluation data was transformed, no GPU or
cloud resource was used, and 2016 was never loaded, transformed, inspected, scored or bundled.** One BLAS
thread throughout.

## Recomputed from the smallest machine-readable source

| quantity | source | result |
|---|---|---|
| Stage B anchor-mean advantage, `|T|=64` and `256` | `STAGE_B_RESULTS.json` | reproduces the gate file exactly (`−0.00268`, `−0.01136` unweighted) |
| per-anchor advantage, **both** weightings | same | person-weighted anchor 0 is positive at both resolutions; the gate's `seeds_with_positive_advantage: 0` is unweighted-only |
| `|estimate|/SE` per anchor | same | `0.16, 0.98, 0.41` at `|T|=64`; `0.72, 3.07, 2.64` at `|T|=256` |
| subsumption `J`-only and `J+T` columns | `gates/G2.json` vs `STAGE_B_RESULTS.json` | exact match, all three anchors |
| B1 decile-partition baseline loss | `STAGE_B_RESULTS.json` | `0.0936 / 0.0647 / 0.0706` nats on 10 cells, against a real residence loss near `0.52` |
| B1 label path | `stage_b.py` source | `decile_cells ← per_row_loss ← −log p[i, y_i]`, a function of the person's own label |
| nominal constrained-fit cap | `RUN_MATRIX.json` | `2×2×2×2×3×3 = 144`, `K=9` tier first (72) |
| G0 arithmetic (from the prior session, re-used) | `PAIRED_INTERVALS.csv` | 544 sensitive rows, min adjusted half-width `0.0021988`, fraction `>.001` equal to `1.00` |

## Fixtures written and run this session

| fixture | claim it settles | result |
|---|---|---|
| `checks/b1_label_leak_fixture.py` | conditioning on per-row loss deciles encodes the label when the class is skewed | pure-noise label "predicted" at `0.057` nats against entropy `0.590`; label-free control does not leak; balanced toy does not leak |
| `checks/replacement_vs_append_fixtures.py` §1 | append-redundancy cannot reject a replacement channel | appending adds exactly `0`; replacing preserves all task utility and removes `0.693` nats of disclosure |
| `checks/replacement_vs_append_fixtures.py` §2 | a null from one probe family is a capacity statement | additive probe gains `2.2e-16` over the prior where the joint view determines the label |

All three assert their own conclusions and fail loudly if the effect does not appear.

## Not verified, and stated as such

* **`prior` and `T`-only probe values.** No machine-readable backing and no committed generator at the
  pinned commit. Reported in the paper as reported-not-verified, on the table itself.
* **The Stage B bootstrap.** Per-row losses were not serialised, so the standard errors are read from the
  gate file rather than recomputed. The ratios above inherit that.
* **Any replacement-study outcome.** None exists: zero units planned/fitted/verified at review time, and
  its branch was not pushed when read.
* **Cloud and instance state.** Terminal 1's status file reports no owned running resources and a dead
  run-lock holder; that is its report, not something this terminal verified. The file itself warns that a
  status record is not proof of liveness.

## Build verification

* SaTML paper: `latexmk` clean, 0 errors, 0 undefined references, 0 undefined citations, 11 pages total
  with a body of 8 pages against the venue's 12-page body limit. Every page rendered and inspected; two
  defects found and fixed (wide floats unreadable in a single column; a mismatched float environment).
* All 21 tables and 6 figures resolve; the two new tables and the new figure are generated from the
  pinned diagnostic commit with hashes recorded in `SATML_ASSET_HASHES.json`.
