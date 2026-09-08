# Released values, legal compositions and repeated access

The auditor sees the interface actually available to its modeled recipient. Public model knowledge does not provide missing per-person inputs. The [purpose policy](PURPOSE_POLICY.md) declares authorized and forbidden targets; the [protocol](PROTOCOL.md) fixes fitting and selection.

| Interface | A receives | B receives | AB receives | Publicly computable additional view |
| --- | --- | --- | --- | --- |
| F | A's frozen 16 coordinates | B's frozen 16 coordinates | Their ordered 32-coordinate concatenation for the same person | A's two and B's one fixed source-head probabilities, computed from the available respective features |
| P | Two positive probabilities, `[income_binary,civilian_at_work]` | One positive probability, `[public_coverage]` | Their ordered three-probability concatenation | The native-probability view is already the wire output; reuse and deduplicate its evidence |
| Direct E control | Frozen E | The same frozen E | `(E,E)` | Selecting either copy; no trained source heads or saved observers belong to this control |

Recipients do not receive raw inputs, a separately supplied teacher, other unreleased activations, same-person ground-truth task/attribute labels, or additional model versions. In particular, P's internal 16-dimensional coordinates are not released and cannot be audited as if they were P. F's public source heads are legal deterministic postprocessing of F. A cannot compute B's head without B's feature coordinates; AB can compute both. The direct E pair is duplicated information, not two independent measurements.

Training examples and labels used to fit the map, source heads and observers are an exposure history, not additional ground-truth labels delivered about a development-evaluation person. Report that history for every candidate, especially saved-start catch-up and public compositions. This study does not establish training-data privacy.

## Three audit scopes

Every scope uses minimum **unweighted attacker-validation log loss**, with fixed deterministic ties, before development predictions are scored. Each scope is reported at nested 120 and 360 epochs. Individual rows and AB rows have different legal observations.

| Scope | A/B candidate pool | AB candidate pool |
| --- | --- | --- |
| `standard_independent` | Five fresh candidates fitted to the recipient's wire output | Five fresh concatenation candidates plus every legal A-only and B-only sensitive candidate applied by coordinate selection |
| `expanded_independent` | Standard pool plus the five candidates fitted to the recipient's public native-probability view, where that view differs from the wire | Standard pool plus native-probability concatenation candidates and all legal native A-only/B-only sensitive candidates |
| `expanded_catchup` | Expanded independent pool plus the recipient's own saved-start observer trajectory for targets that had a training observer | Expanded independent pool plus coalition, A-only and B-only saved-start sensitive trajectories |

The five fixed independent candidates are logistic regression, two MLP initializations and the two boosted-tree configurations. All A/B sensitive candidates enter AB, not merely the selected singleton winners. A/B attacks on opposing task labels remain individual-policy evidence; those task labels are authorized to AB and are not invented coalition-prohibition targets. The post-freeze reserved-task attackers have no saved training observer or catch-up trajectory.

For F, the public-probability attacks add fitting work and a composed function family. Their extra compute, source-head lineage and exposure are explicit. For P, native equals wire, so the same fitted candidate is not counted as an independent duplicate. Direct E role-specific seed/recipe additions and compatible reused evidence retain separate provenance; no observer is synthesized to make static and trained releases appear equally exposed.

An inherited singleton candidate keeps its own preprocessing, model, fitting rows, validation predictions and labels. Projection to A or B followed by that candidate must reproduce its singleton predictions. The same applies to an F native head followed by a native-view attacker. A legal composition cannot silently change a standardizer, map, target, row identity or view order.

The AB validation-selected loss cannot exceed any candidate included in its pool on the same rows. This is a selection invariant. Its development loss can be worse than a singleton's development loss because validation and development rankings need not agree. Do not select the development minimum, clip the difference, or claim an impossible information reduction from that ranking reversal. Report absolute signed prior-relative gains and coalition-minus-singleton comparisons separately.

## Saved observers and budget extension

Catch-up starts from that final system's actual saved training observer, on its original coordinates, with the declared Adam reset. Verify identical predictions before its first update. Epoch zero is eligible inside the trajectory. The standalone saved-only row remains diagnostic, not an extra selected candidate. If the epoch-zero checkpoint wins, a pooled improvement over fresh attacks can reflect inherited fitting exposure without beneficial extra optimization.

The 120/360 fresh and catch-up results come from nested checkpoints of the same prescribed trajectory, retaining actual last model/Adam/RNG states separately from validation-best states. Stronger budget is not a new family, data subset, seed search or mapper update. Independent, inherited, public-composition and saved-start origins remain visible even when their candidates enter the same expanded selector.

## Fixed-version repetitions

The deployed object in this benchmark is one deterministic frozen pair for a person. At k=1,2,4 requests, each recipient receives identical copies of its own output. Verify byte identity and replay first-pair attacks on the full repeated observation through the fixed first-copy projection. Label this a **DUPLICATION/INVARIANCE CHECK**, not a fresh-release privacy experiment or independently fitted history audit.

Never concatenate outputs from the three study seeds as model versions: their household roles differ. Changing underlying values, releasing new model versions, introducing fresh noise, or supplying extra side information is outside this access policy. The historical [exact sign calculation](../redesign_20260908_acs_restricted_inputs_v1/PURPOSE_COORDINATION_EXACT.md) shows why this distinction matters: its fresh coordinated noise improves one-call coalition accuracy from 5/8 to 1/2 but reverses the ordering at four calls, 107/128 versus 377/512. Its cached result and this deterministic ACS duplication check do not certify fresh access or general coalition privacy.
