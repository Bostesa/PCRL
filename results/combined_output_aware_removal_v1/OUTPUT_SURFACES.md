# Output surfaces: what the released prediction adds

**Recipient.** This is the purpose-specific recipient (Adult income_prediction, HMDA underwriting). The earlier
attacker received only this purpose's 2-column logit vector, and so does this one. Other purposes' heads are not part
of this release (`RELEASE_CONTRACTS.md`).

**Statistic.** All AUCs are the supported-class macro AUC of the validation-selected nonlinear attacker, averaged over
3 encoder seeds × 3 attacker seeds (and 3 noise draws where noise applies), on the common, exposure-cleaned
assessment rows. Bracketed values are 90 % intervals. Sources: `EXPLORATORY.csv`; P1 is from `PRIMARY_ENDPOINTS.csv`.

## Outputs alone

| Surface | Adult (sex) | HMDA (race) |
|---|---|---|
| Historical full logits `O_full` | 0.776 [0.767, 0.784] | 0.768 [0.763, 0.773] |
| Probabilities `O_prob` (softmax) | 0.705 [0.697, 0.714] | 0.702 [0.698, 0.707] |
| Hard prediction `O_hard` (frozen head argmax) | **0.513** [0.511, 0.515] | **0.505** [0.503, 0.508] |
| True task label (label-only reference; diagnostic, not a release) | 0.601 [0.593, 0.611] | 0.522 [0.514, 0.531] |
| Constant | 0.500 | 0.500 |

**Primary P1 (full minus hard ≥ 0.02):**

| Cell | Δ | Simultaneous lower bound | Decision |
|---|---|---|---|
| Adult | +0.263 | 0.249 | **PASS** |
| HMDA | +0.263 | 0.254 | **PASS** |

**Reading these results:**
- **Full logits leak more than probabilities.** The logit vector also carries a per-row offset, the sum of the two
  logits (sd 0.55 on Adult s0), that the probability vector discards. That offset adds 0.07 AUC of measured recovery
  without changing any decision. Full and probability are therefore *not* aliases: the probability is a function of
  the logits, but not the reverse. This is checked numerically per seed (`run/aliases/`).
- **Hard predictions sit just above chance.** On Adult they are even below the label-only reference (0.513 vs 0.601).
  The frozen head's decision is less sex-informative than the true income label.
- **Not a privacy claim.** This does not show that hard labels "solve privacy". It shows that, for these heads and
  this attacker slate, the confidence detail carries most of the measured recovery and the decision itself carries
  little.
- **The defense-aware cell-conditional attacker** on the hard surface gives the same AUC as the standard slate (0.513 /
  0.505). Stronger attack performance was not established.

## Utility of the output variants

**Exact check (frozen head, not a hypothesis).** `O_hard` is the head's own argmax, so its decisions and its task
accuracy are *identical* to the full output's.
- This parity says nothing about log loss, calibration, ranking or other uses.
- A hard label has no finite log loss. No calibrated probability was invented for it.
- The full-output contract's own utility is the frozen head's accuracy, unchanged by any defense in this study.

## Outputs together with protected features

| Release | Adult | HMDA |
|---|---|---|
| Noise at σ\* (2.0 / 4.0) alone | 0.524 | 0.508 |
| Noise + full logits | 0.776 | 0.768 |
| Noise + probabilities | 0.705 | 0.702 |
| Noise + hard prediction | **0.527** | **0.510** |
| FARE\* features alone | 0.550 | 0.501 |
| FARE\* + historical clean logits (view 2) | **0.788** | **0.768** |
| FARE\* + head fitted only on FARE features (view 3) | 0.550 | 0.501 |
| Target LEACE + clean logits | 0.814 | 0.864 |
| Target LEACE + own head (view 3) | 0.817 | 0.863 |

**The bypass.** Whatever defense is applied to the features, releasing the *historical clean* task output puts
recovery back at the outputs-only level: about 0.77–0.79 for noise and FARE. This is the "output bypass": changing
the features cannot remove information carried by an output that is released unchanged.

**What fixes it here.** In these cells, replacing the clean output by a head computed only from the protected
features (view 3), or by the hard decision, removes the bypass. Both change the release contract, and their utility
is reported separately:
- view-3 head accuracy is in `HEAD_UTILITY.csv`;
- hard-decision accuracy equals the frozen head's.
