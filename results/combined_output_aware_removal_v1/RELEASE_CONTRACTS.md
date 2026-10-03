# Release contracts

This file names exactly what each recipient receives.

**Common to every contract:**
- Every release is a single, persistent, per-person release. There are no repeated queries and no fresh noise per
  query.
- The recipient never receives sensitive labels, row identifiers or lookup keys.
- Each recipient is the purpose-specific recipient of one purpose: Adult `income_prediction`, HMDA `underwriting`.

**Earlier attacker inventory.** The earlier attacker was given that purpose's own head output: a 2-column logit vector
(`logits_income_prediction`, `logits_underwriting`). The admitted forward caches also contain the other purposes'
outputs (Adult: employment 6 columns and education 4; HMDA: pricing and fair-lending heads). Those belong to other
recipients and were **not** part of this recipient's release, before or now. Any coalition of purposes would be a
different contract, and it is not evaluated here.

## Feature releases ("protected features")

| Tag | Released features | Fitted on | Runtime inputs |
|---|---|---|---|
| A | Untreated frozen PCRL representation (64 columns) | — | encoder features |
| B | Official target-only LEACE (concept-erasure 0.2.4) of A, using the benchmark's pinned maps | defense_fit | A only |
| C | Official LEACE on the purpose's disallowed set, using the benchmark maps (descriptive) | defense_fit | A only |
| D\* | A + N(0, σ\*²I), with σ\* = 2.0 (Adult) / 4.0 (HMDA) frozen from the benchmark validation; 3 persistent draws per person, identical to the benchmark draws | — | A and a fixed seed |
| F\* | Official FARE cell index (one-hot), from the validation-selected configuration | defense_fit | A only |
| F0 | FARE at the nominee's tree/output budget with fairness weight 0 (compression and task-tree control) | defense_fit | A only |
| — | Constant / withheld: nothing released (attacker = attacker_fit prior) | — | — |

## Access views per defense

| View | Recipient receives |
|---|---|
| 1. `rep` | the protected features only |
| 2. `rep+clean` | the protected features **and the historical clean task output**, the frozen PCRL head's logits on the untreated representation. This tests a bypass: changing the features does not change information in an output that is released unchanged. |
| 3. `rep+head` | the protected features and the logits of a **release head fitted only on those protected features** |

**The view-3 head.**
- Family: LR grid, the benchmark U2 family and budget.
- Fitted on defense_fit, minus a hash-held-out 20 %. Selected on that holdout by log loss.
- At runtime it reads only the protected features. It never sees attacker roles, sensitive labels or historical
  outputs.
- The same head family and budget is used for A, B, F\* and F0.
- Its task accuracy and log loss on assessment are reported as the utility of the view-3 contract.

## Output-surface variants (outputs-only recipient and noise contexts)

| Tag | Content |
|---|---|
| `O_full` | the historical full logit vector, exactly as previously released |
| `O_prob` | its softmax class-probability vector |
| `O_hard` | the frozen head's own decision: argmax, given as a one-hot |
| `LO` | the true task label: a diagnostic reference (label-only attacker), not a release |
| `const` | the constant prior |

**Alias check.** This is recorded per seed under `run/aliases/`. The probability vector is a function of the logits,
but not the reverse: the logit vector also carries a per-row offset (the sum of the logits) that the probabilities
discard. `O_full` and `O_prob` are therefore different surfaces, and `O_full` determines `O_prob`. This is verified
numerically, not assumed.

**Accuracy parity.** Accuracy parity between the full output and its argmax is an exact property of the frozen head:
the decision is the same. It does not imply parity in log loss, calibration, ranking or other uses.

**Noise contexts.** At σ\*, `D*+O_full`, `D*+O_prob` and `D*+O_hard` give the noisy features together with each
output variant.
