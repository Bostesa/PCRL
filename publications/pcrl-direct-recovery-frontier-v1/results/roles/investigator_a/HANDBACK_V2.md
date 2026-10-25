> Saved verbatim by the coordinator (2026-10-25, after 03:12:27Z by `date -u`): investigator A's second hand-back message
> (TINY_EXACT_V2 against the updated solver), unchanged below.

## Investigator A: TINY_EXACT_V2 against the updated solver

The merge polish and seeds cut shortfalls at the campaign settings to 4 of 348 cases (AUC) and 3 of 348 (MI), from 35 and 36 under the V1 default call. There is still no implementation defect, but reproducer 3 is still not solved at the campaign settings.

All times are from `date -u`, 03:05:20Z to 03:11:56Z. Synthetic data only; nothing under `.private/` was read and I ran no state-changing git command. I tested `drf/solver.py` at bc5c006eb8823ee3901a14fe0e5351a8a23f490752d0c746b14f679f8eed13e7 (also recorded inside the JSON).

**Calls used.** Classification uses the campaign call (seeds 0 and 1). The comparison columns repeat the V1 calls; they now include the merge polish. I did not run 40/3, so there is no extra column for it.

| Label | Settings | Role |
|---|---|---|
| campaign | max_sweeps=30, restarts=2, perturb_moves=16, seeds=2, seed=0 | primary |
| s0r0, s1r0, s2r0 | max_sweeps=50, no restarts, seeds 0 / 1 / 2 | comparison |
| s0r3 | seed 0, 3 restarts, default perturb_moves | comparison |

**Files written**
- `drf/tiny.py` (now V2; writes the V2 file by default): 777ff547fcff3ec8be3ebab6d15964293e77d338556b6767d1a1360d16a9abd4
- `results/pcrl_direct_recovery_frontier_v1/roles/investigator_a/TINY_EXACT_V2.json`: e6ea112a48a2b934cf72fa852cd28330976c8c61912d181c069b7069646a79c9. Two runs gave the same hash.
- `TINY_EXACT.json` was not overwritten (still fe568b05b87e6f604c63844b55c43111c4f2040f4381879e423c229d10c2e5ba).

### (b) Shortfalls: the solver stays above the exhaustive optimum of its own objective
Random scan, 348 improvable cases per objective:

| | AUC | MI |
|---|---|---|
| Campaign call | **4** | **3** |
| Campaign: median relative gap | 0.228 | 0.439 |
| Campaign: maximum relative gap | 1.0 | 0.65 |
| New default (s0r0) | 20 | 19 |
| V1 default | 35 | 36 |
| V1 best of 4 calls | 9 | 10 |
| Best over all 5 V2 calls | 2 | 1 |

- One joint two-cell move from the campaign answer closes 3 of the 4 AUC and all 3 MI shortfalls. These are two-cell moves the merge polish cannot make: a cell shifted to a different atom, or a swap.
- Named instances, campaign call:
  - instance A at q25 is unchanged from V1: AUC gap 0.0211 (26% of the 0.082 available) and MI gap 0.00125;
  - instance B at q60: AUC gap 0.0038, MI gap 0.00008;
  - everything else is solved, including the tied-tables instance E and B unconstrained.
- No answer is infeasible or misreported: `best_J` matches the independent J within 1e-9 on every call. No converged answer has a feasible improving single move or a feasible improving merge, so there is no implementation defect.

**V1 reproducers**
1. Seed 1117 (MI, q60): **solved** by the campaign call. The single-seed default still has gap 0.0349.
2. Seed 1101 (AUC, unconstrained): **solved** by every call; the merge polish does it.
3. Seed 1114 (AUC, q25): **not solved** by the campaign call. It returns the identity (J 0.17565 vs optimum 0.15867), missing all of the available improvement.
   - s0r3 solves it.
   - Campaign settings with more seeds or restarts also solve it: (restarts, seeds) = (2,5), (5,2) and (10,4) all reach the optimum.

**New smallest campaign shortfall: seed 1064.** Instance: `tiny.random_instance(seed=1064, N=43, I=2, MI=2, J=4, MO=3, law='pi')` at the q25 budgets. The budgets and full payload are in the JSON under `random_scan.smallest_primary_call_shortfall`.
- The campaign call reaches J 0.37377 (A_occ [0,0,1,2]); the optimum is 0.33399 (A_occ [0,1,2,2]), a gap of 0.0398.
- More restarts or seeds do not help: (2,5), (5,2) and (10,4) give the same gap.
- The optimum cannot be reached by improving single moves.
- One joint two-cell move from the answer reaches the optimum exactly. A pair-move polish would cover it; merges cannot.

### (c) Solver arithmetic against the independent finite_law recomputation
- `solver.evaluate` J and the 8 spends, on every assignment: maximum difference 8.9e-16 on the named instances and 1.03e-15 across the scan.
- The solver's internal candidate values (`_move_candidates` and the new `_merge_candidates`, both objectives plus all 8 spends) at the resulting assignments: maximum difference 1.09e-15. This covered 3,820 single moves and 842 merges in the scan, plus 20 random states per named instance.

The objective misalignment does not depend on the solver and is unchanged: 12 of 360 witnesses, maximum excess 0.0305.

**CPU:** about 90 s for this follow-up (two full runs at 39.3 s each plus about 10 s of checks), single-threaded. About 220 s cumulative for my role.
