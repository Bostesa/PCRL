# Exact original source-function preservation

All nine pinned selected models reproduce their original complete class0/class1 vectors exactly on downstream validation and development. Their original unweighted/PWGTP source scores replay exactly. The wire uses a lossless float64 container; it does not replace a separately rounded softmax column by a complement. Prediction metrics apply the historical scorer to the authoritative vectors.

| Seed | Task | Selected original model | Original fit labels | Checkpoint epoch | Validation U / PWGTP | Development U / PWGTP |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | income_binary | mlp | 2048 | 20 | 0.306755417 / 0.290351813 | 0.306898065 / 0.318726324 |
| 0 | civilian_at_work | logistic | 2048 | fixed | 0.314096556 / 0.305677881 | 0.284754961 / 0.265905801 |
| 0 | public_coverage | mlp | 2048 | 20 | 0.488816090 / 0.501322227 | 0.517921591 / 0.527690152 |
| 1 | income_binary | mlp | 2048 | 20 | 0.305972367 / 0.305041661 | 0.289399681 / 0.301258469 |
| 1 | civilian_at_work | mlp | 2048 | 15 | 0.299587589 / 0.303298455 | 0.286353846 / 0.275281382 |
| 1 | public_coverage | mlp | 2048 | 20 | 0.486613481 / 0.496265771 | 0.497103751 / 0.501596119 |
| 2 | income_binary | mlp | 2048 | 20 | 0.308208466 / 0.312819984 | 0.299437410 / 0.311718715 |
| 2 | civilian_at_work | mlp | 2048 | 20 | 0.300419567 / 0.300072780 | 0.294112614 / 0.290580613 |
| 2 | public_coverage | mlp | 2048 | 20 | 0.484928770 / 0.500774225 | 0.498844546 / 0.513046178 |

[Exact identities, models, checkpoints and complete metric dictionaries](ANCHOR_PARITY.json), [prefit inputs](PREFIT_IDENTITY.json), [reuse manifest](REUSE_MANIFEST.json), and [schema](INPUT_SCHEMA.json) bind provenance. H and every augmented system expose those same values. Public source-anchor parameters do not provide raw person PCA inputs. Auxiliary native heads are separate diagnostics and are public compositions when their features are available.
