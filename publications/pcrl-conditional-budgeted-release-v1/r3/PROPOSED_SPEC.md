# R3 pre-fit allocation and utility witness specification

R3 owns pure context construction, exact expected task coefficients, task-only witnesses and primary frozen-donor fits. No new teacher, decoder, temperature or task head. Only the admitted V-J pre-identity donor is used for G/C.

## Contexts (input-only proposal)

Predicted income class (source first-index argmax), predicted occupation class, absolute decoded income margin `abs(p[1]-p[0])`, and occupation entropy divided by `log(6)` identify an initial grid. Margin edges are 0,.25,.5,.75,1; entropy edges are 0,.5,1. Left-closed/right-open bins, last bin includes 1. This gives 12 class pairs times 4 times 2 = 96 initial contexts. No data-dependent numeric edges.

FIT count floor is 64 rows. Each class-pair/margin parent whose two entropy children include a child with count below 64 merges both children into its parent, if that parent has at least 64. If any of four margin parents remains below 64, merge the entire class pair, provided its total count is at least 64. Class pairs below 64 have exact-P0 fallback. Merging uses all FIT input rows, before protected labels are opened. Deployment applies the registered mapping to every supported original-code pair, including new raw83 rows. Global code pairs absent from FIT are supported if in the admitted frozen P0 banks and the resulting context is supported. Schema refusal is distinct from fallback. Each coefficient is shared by all mapped people. Class pairs are never merged together, so the same contexts can support later class-compatible action templates.

One coefficient per supported mapped context for C; one scalar for G. No per-person coefficient. Store FIT counts and complete 96-to-context mapping; publish aggregate counts only. Unsupported contexts return exact identity.

## Utility and guards

Task losses use every emitted ORIGINAL code's frozen probability, source floor 1e-12, summed multiclass Brier and structural same-class decisions. Every pooled/f0/f1/f2 fitting view carries all clauses. STRICT has 24 task clauses per seed (two tasks, three metrics, four views): accuracy loss <=.002, NLL excess <=.001, Brier excess <=.0005 versus P0. ORIGINAL has 48 clauses (two U0/Ucal references): loss(q)-loss(P0) <= original cap - (loss(P0)-loss(reference)). Accuracy is encoded as negative accuracy. Negative RHS remains negative. Fixed restored local reader CE safeguards have coefficient sign -CE delta and RHS source numeric tolerance 1e-10, on each four views. All source retained recipient readers are used identically for G/C; pair readers do not become local guards. Read-only restoration is preferred over refitting.

For every seed, budget and G/C setup, run minimum total task NLL witness and maximum actual tuple-change witness before privacy optimization. Identity donor mass and duplicate emitted tuples count as unchanged. Feasible nonidentity witness triggers protection fit even when minimum loss returns identity. A jointly infeasible witness blocks that setup, not merely one clause. Fitting-only quantities never establish held-out service eligibility.

Protection objective is empirical Bayes actual-code pair recovery plus half each recipient recovery. Both objectives and constraints use the same fixed donor. No final reader or INNER SEX can change the policy.

## Units and limits

Two seeds x two budgets x two allocation families = 8 setups, each with two utility-only solves and one eligible protection solve: 24 nominal LP solves. Each source seed has one count/budget/reader preparation unit, no numerical reader fits if all source losses restore. One engineering fixture unit. Original setups that prove infeasible retain both witness receipts and do not run protection. Source reader data estimated 54 reader losses x15434x16x8 =107 MB per seed; local subset72 MB. Donor/action/cost arrays estimated <100 MB; LP sparse epigraph approximately three views × two protected classes × emitted states, at most ~25,000 rows and under 100 coefficients per row, provisionally <500 MB. Reserve 1.5 GiB per preparation/solve worker; one numerical thread. Forecast primary preparation plus24 LP solves under1200 CPU seconds; root may scale before lock from source-only dimensions and synthetic timing. No new real-data solve until root science lock and active scheduler unit.

## Dependencies

R1 must bind source banks, exact V-J weights/law, row-safe FIT/INNER P0 and task/reference inputs, source groups/internal folds, READERS archive/catalog and direct helper paths. R2 supplies affine-law LP/certificate interface. R6 must independently accept contexts, negative RHS, alias-change and fixed-guard sign. Root seals locks and queue. Missing source inputs block the exact corresponding fits; no substitute teacher or reader is silently introduced.
