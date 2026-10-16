# Manuscript / claims handoff (concise; no manuscript branch was edited)

**Allowed claims: development evidence, Florida ACS 2019.**

Scope of every claim below:
- custom two-task benchmark, SEX protected;
- DEV_REPORT opened once for analysis; sealed 20% closed;
- all intervals and decisions are conditional on the two fitted seed pairs and the frozen 7-reader slate, which bounds recovery from below;
- no between-fit (refit) variance and no survey-design variance is included.

1. **Teacher headroom** (point estimates on MODEL_SELECT; no interval). A depth-4 CatBoost teacher had modest headroom over an equally exposed neural reference on both tasks, for both seeds: NLL lower by 0.007 for income and 0.008–0.012 for occupation.
2. **Recipient-level recovery reduction with point-level utility.** The fixed local leaf-score recipe was L_TREE ρ = 1, E20.
   - **Transport:** it was transported with only the registered ACS changes: input width, five occupation classes, critic widths 3/6/9, role names, and an access mapping in which both the task pool and the privacy pool are MODEL_FIT. The source's separate UPDATE-with-SEX / GUARD-without-SEX split does not exist here. Labelling disclosure D1.
   - **Utility:** it passed all 24 overall utility clauses at the point, with no established violation.
   - **Recovery:** against matched continued task fitting, it lowered each recipient's own SEX recovery, by 0.014–0.018 AUC for income and 0.061–0.067 for occupation. The K = 40 simultaneous intervals lie wholly below zero (reporting family; not a pre-registered claim decision).
3. **No coalition protection at the target margin.** Against matched continued task fitting, the linked coalition's recovery fell only 0.007 on seed 0 (an established shortfall against the 0.020 margin) and 0.0175 on seed 1 (unresolved; positive, with simultaneous interval [+0.006, +0.029]). The recipe effect as pre-registered is **not established**.
4. **Not competitive.**
   - The eligible neural adversarial control had lower coalition recovery at feasible utility. LOCAL_TREE minus NEURAL_RAW coalition AUC was +0.017 and +0.011, with seed 0 an established shortfall and seed 1 unresolved.
   - FAGTB (faithful binary port plus documented multiclass adaptation; the seed-1 tree random_state is a registered deviation) and official FARE had no MODEL_SELECT-feasible configuration. They failed every DEV_REPORT utility clause and are descriptive only.
5. **Subgroup: SUBGROUP_UNRESOLVED.**
   - Women: the occupation-Brier guard passes in the K = 40 simultaneous interval on seed 1 and is unresolved on seed 0.
   - Men: unresolved on both seeds. The one-sided bounds, 0.0051 and 0.0071, exceed the 0.005 cap; no harm is established.
   - Any recommendation must state that subgroup suitability is unresolved.

**Do not claim:**
- confirmation of the frozen Adult packages;
- transfer of Adult AUCs or of the 317,000-group requirement;
- algorithmic novelty;
- race protection;
- survey-design variance;
- certified non-use of Florida 2019 people;
- that JOINT_TREE works (diagnostic only: lower coalition recovery, but income utility clauses fail);
- that FAGTB was beaten (it was ineligible);
- independence of the two seeds;
- that the sealed role's labels were never computed. They were computed in memory during the extraction's eligibility pass and discarded unwritten (disclosure D2).

**Suggested sentence:**

> In a Florida ACS 2019 development replication (conditional on two fitted seed pairs and a frozen reader slate), the routing-preserving local recipe passed all 24 utility clauses at the point and lowered each recipient's own SEX recovery relative to matched continued task fitting (by 0.014–0.018 AUC for income and 0.061–0.067 for occupation). The linked coalition's recovery fell only 0.007 and 0.0175 AUC, below the pre-registered 0.020 margin (an established shortfall on one seed, unresolved on the other). An eligible neural adversarial control had lower coalition recovery at the point on both seeds (established on one). Usefulness within the male group was unresolved.

**Pointers:**
- `RESEARCH_DECISION.md` for the full decision;
- `PRIMARY_ENDPOINTS.csv`, `SERVICE_TABLE.csv` and `SUBGROUP_TABLE.csv` for the tables;
- `INDEPENDENT_VERIFICATION.json` and `CLAIMS_REVIEW.md` for verification and wording review;
- `POPULATION_AND_SCHEMA.md` for provenance and schema;
- `SCIENCE_LOCK.json` and the public receipts on GitHub branch `publication/pcrl-acs-protected-tree-replication-v1` (`85b2a41`, `7d23e0e`) for the locks;
- `LOCK_AMENDMENTS.md` for amendment A1 and disclosures D1, D2 and D3.
