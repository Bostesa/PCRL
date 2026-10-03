# Purpose permissions and recipient views

## Permission table (declared, not "protect everything except Y")

| Recipient | Purpose (permitted task) | Classes | Forbidden (primary) | Secondary stress audit |
|---|---|---|---|---|
| R1 | `income` (>50K) | 2 | SEX | race (original 5 labels; supported classes only) |
| R2 | `occupation_group` (pinned loader's 6-group map of `occupation`) | 6 | SEX | race |

**What the labels themselves reveal.**
- Both permitted tasks are associated with SEX in Adult (occupation group strongly so).
- A useful release cannot have zero disclosure: the task label reveals what it reveals.
- The label-only reference recovery is reported as a diagnostic, not as a deployable release.

**No coalition member is legitimately released SEX**, so "the coalition cannot learn SEX" is a meaningful target here.

**Race.** Race is audited separately. No race-protection claim is made: the pilot trains against SEX only, and the method generalises to several attributes only in code (concatenated one-hot concepts).

## Views audited

| View | Contents | Role |
|---|---|---|
| `v1` | Recipient 1 features `r1` (16-d, after the final official LEACE) + centred logits of its deployed affine head (2) | **Primary**, local |
| `v2` | `r2` (16-d) + centred logits (6) | **Primary**, local |
| `pair` | `[v1, v2]` | **Primary**, coalition. Its attacker bank also contains the selected `v1`-only and `v2`-only attackers. |
| `p1`, `p2`, `ppair` | Probabilities only | Secondary (output-only) |
| `h1`, `h2`, `hpair` | One-hot hard decisions only | Secondary (output-only) |
| FARE `v_i` | One-hot tree cell + centred logits of an affine head on cells | Primary, for F/F0 |

## Routing invariants

- Every released logit is computed by an affine head from the released (post-LEACE) features only.
- No pre-erasure head and no untreated "clean" output is ever released.
- Logits are centred for every arm, so a common offset cannot explain a difference.
- Deployment (`jcv.deploy`) accepts only the 83 permitted input columns. Labels, row keys and protected fields are not inputs.
