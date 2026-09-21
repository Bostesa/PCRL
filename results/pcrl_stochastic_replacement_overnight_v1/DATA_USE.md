# DATA_USE — what sees which labels

Locked 2026-09-21. 2018 development pools only. **2016 sealed. 2017 not opened.**

## Artifacts and their label access

| artifact | pool | labels it sees | labels it must never see |
|---|---|---|---|
| Codebook `g` (k-means, both families) | `representation_fit` | **none** | all |
| Constraint partitions `C_A`, `C_AB` | `representation_fit` | **none** — permitted predictions/features only | sensitive, residence, per-row loss, correctness |
| Constraint distributions `p_r(s,c,t)` | `representation_fit` | **protected** `S_r` (required to constrain it) | residence |
| SUP baseline decoder `b(H_A)` | `representation_fit` **decoder-fit** half | **residence** | protected |
| SUP cost matrix `D[t,z]` | `representation_fit` **cost-fit** half (disjoint) | **residence** | protected |
| LF prototypes and `D` | `representation_fit` **cost-fit** half | **none** | all |
| `Q` (the convex fit) | — | none directly; inherits via `D` and `p_r` | — |
| Utility probes | `downstream_fit` (all eligible households) | **residence** | protected |
| Screen / selection | `downstream_validation` | **residence** | — |
| Audit attackers | `attacker_fit` / `attacker_validation` | **protected** + source | residence |
| Frozen finalist evaluation | `test` | all designated endpoints | — |
| Descriptive commute | `test` | commute, **after** nomination only | — |

Household separation between pools is inherited unchanged. The decoder-fit / cost-fit split inside
`representation_fit` is by household, by the frozen hash rule in `METHOD.md` §4.

## Rules

* Codebooks and LF prototypes are **label-free**. Calling them label-free does **not** make the
  overall selected method a reserved-task result: residence guides the screen and the nomination, so
  the whole programme is **task-informed development** (`PROTOCOL.md` §0).
* Conditioning coordinates never depend on a label, a per-row loss or correctness. This is the rule
  that the precursor's withdrawn `probability_decile_partition` violated
  (`PRECURSOR_CORRECTIONS.md` C4).
* Commute is **descriptive only**, evaluated after nomination, and may never rescue a selection.
* `Z_J` is computed internally for the `zj` code family and is **never released** in an S1 candidate.

## 2016 containment

Loaders are constrained to the 2018 path. A test asserts that no path, manifest entry, archive member
or cloud bundle file in this study references `2016` or `ss16`. Any such reference is a hard failure,
not a warning.
