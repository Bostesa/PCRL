# Matched control implementations

These controls use the selected WIDE original bank: explicit KEEP plus every original global code with the recipient's predicted class. KEEP aliases are aggregated with named codes. Recipients receive original code IDs, bank probabilities and decisions. Internal action positions, context IDs and coins are not released.

All arms share the selected STRICT budget, frozen bank task losses, pooled and three-fold constraints, and original fixed-reader safeguards. Their parameter families differ as recorded in the locked specification; these differences are not hidden behind a claim of identical capacity.

| Arm | Information law and fitting procedure | Runtime inputs |
|---|---|---|
| L_SELECTED | Two own-code marginal tables, independently sampled; bounded alternating coordinate LPs optimize the common privacy objective. This is local runtime, not separately trained privacy objectives. | Each marginal reads only its own original code. |
| P_SELECTED | Central-context marginal tables with independent coins; two fixed starts and four alternating rounds. | The provider may read the two original codes to determine the context. |
| D_SELECTED | One binary categorical action per central context; a genuine mixed-integer epigraph program, not a rounded stochastic policy. | The same central inputs. |
| S12_SELECTED | First recipient optimized while the second keeps its original code; frozen first law; second conditional law keyed by context and actual first emitted global code. | The same central inputs plus the actual first output. |
| S21_SELECTED | The same two stages in the reverse recipient order. | The same central inputs plus the actual first output. |

The fitting objective is complete-pair empirical Bayes recovery plus one half of each recipient's empirical Bayes recovery. It is not AUC, population information, differential privacy or a fresh-attack guarantee. Both sequential orders and the product fits are bounded procedures, not globally optimal family searches. MIP solver gap reports are retained separately from independently certified arithmetic bounds.

## Numerical corrections

The source dependency list and preflight output manifest were corrected before affected runs, preserving the rejected job records. No scientific fit was caused by those metadata corrections.

The original deterministic seed-1 result was refused because its raw integer/primal residual exceeded the predeclared 1e-8 executable tolerance. The original product seed-0 run saved nine coordinate solves but failed strict nonnegative probability serialization. Both original artifacts remain. Published amendment Q authorizes one corrected fit each, fixed LP solver accuracy 1e-10 for product/sequential fits and fixed MIP feasibility accuracy 1e-9. The deterministic continuous relaxation still requests 1e-8. The acceptance tolerance, certificate cap, banks, constraints, objectives, starts, rounds and compute ceilings are unchanged. No post-fit projection or normalization is used.

Each full coordinate primal, dual and sparse program is archived immediately. Compact history retains the original earliest-best rule and selected law/weights. Sequential FIRST and SECOND archives retain the frozen-stage boundary. Public receipts contain aggregates and hashes; per-person laws and model tables stay private.

Fresh attacks and independent replay determine competitive claims. A fitting-objective gain with failed screening utility remains a utility shortfall. No control is called Taylor's method merely because it is sequential; the separately restored historical S21 law remains a distinct source control.
