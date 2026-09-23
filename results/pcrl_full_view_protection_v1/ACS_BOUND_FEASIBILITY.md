# Full-view bound feasibility before ACS fitting

This calculation precedes and prevents new ACS fitting. For a 17-token action alphabet, suppose optimistically that the per-H reference CMI is exactly zero. The three-entropy continuity term in `FULL_VIEW_THEORY.md` would require a **uniform** conditional-joint total-variation radius no greater than:

| Protected alphabet | `log|S|` ceiling | For .001 nat | For .01 nat | For .05 nat |
|---|---:|---:|---:|---:|
| SEX, 2 | .693147 | 0.00002431 | 0.00029735 | 0.00176803 |
| RAC1P, 9 | 2.197225 | 0.00002221 | 0.00026617 | 0.00154873 |

These are **allowable errors**, not estimates of the actual ACS error. A positive reference leakage, uncertainty in `P(S|H)`, solver error and an H-marginal shift consume more budget. The numerical CMI feasibility tolerance in the finite suite is `1e-7` nats; the current ACS conditional-law estimation error and uniform approximation error are **unbounded by available evidence**, because H is continuous, the data have sparse conditional support, the encoder was fitted, and no household-level uniform-coverage theorem or validated smoothness constant is supplied. A zero protected-class count or an unsupported full-H region cannot receive a zero uncertainty radius. Therefore the continuity route yields no nontrivial unconditional ACS bound: after capping at the entropy ceiling, the justified bound is at most the trivial `.693147` / `2.197225` nats. Even a purely assumption-conditional `.01` bound would require the stringent radii above at every relevant H.

The unchanged 2018 Q's all-input radius references produce full-view upper bounds of 1.762168, 1.936616 and 1.967602 nats across anchors. Intersecting with the protected entropy ceiling leaves `.693147` for SEX and 1.762–1.968 for RAC1P. D17's radius references yield 2.639–2.773, leaving only the entropy ceilings. These are valid but do not meet any of the `.001/.01/.05` attribute-specific illustrative budgets. A `.01` all-input radius also caps Bayes task information at `.01`, below the approximate `.026` historical Q-versus-H descriptive task-gain scale. This is a feasibility check, not an ACS result for a new channel.

The only way to improve the continuity calculation is an externally justified envelope or a different mechanism property with a verifiable full-H premise. More local CPU does not supply uniform conditional-law coverage for an unrestricted continuous H. The empirical pilot was stopped by this scientific precondition and by the matched-method novelty gate, so there is no remaining 27-fit resource request to price or launch.
