# Statistical reviewer, engineering repair ER1 (saved verbatim with attribution)

Author: the statistical reviewer of campaign `pcrl_finite_routing_comparison_v1`. Returned at about 00:31Z on 2026-10-21 (its own
times: 00:28:29Z-00:30:35Z), after the coordinator's request of 00:27Z (deviation 6). Saved verbatim (indentation removed).
Coordinator decision on its closing question: `refit_effectiveness` keeps the strict rule (no further post-hoc change); recipes
that are seed-invariant on finite views count once there.

---

## Statistical reviewer: ER1 grid-validity repair implemented, 18 of 18 tests pass

I worked 00:28:29Z to 00:30:35Z (`date -u`). No labels, real data, grid outputs or comparison values read; no sub-agents, commits or pushes. CPU is charged with `python3 -m frc.jobqueue charge --label statistical_reviewer --cpu-s 10` (exit 0).

### New SHA-256
- `results/pcrl_finite_routing_comparison_v1/roles/statistical_reviewer/RULES.md`: 0121711e2d239090eb20fe7ee1c0531ea3f2e8feda309369dea7e6fbb2207a82
- `.../roles/statistical_reviewer/RULES.json`: e2a6af232c2835413e7da85cf805c00c857c32931c79bcc5afe5b8099032dfec
- `frc/rules.py`: 11d3df7ffac404831167645ae565604af4d140f1b189c8b96f665613e62c01a4
- `frc/test_rules.py`: a2c634edd4a2f2165557e12451a9ee1a421420c4cef88ae0d68112f6dc4a43d6

The inherited files are unchanged (nrm/rules.py d5df8686…, NRM1 RULES.json 84f18776…).

### What ER1 does
- **Rule:** a seed-dependent k0/k1 equality on the minimum set is now judged by the same NRM1 (iii) explanation (a)+(b)+(c) as on other wires. In practice `nrm.rules.grid_validity` is called unchanged with an empty minimum set.
- **Explained equalities** no longer invalidate the grid. They are reported under `explained_minimum_set_equalities`, with these fields:
  - `count`
  - `items`
  - `n_eff_implied`, keyed by wire|view|recipe: blocks checked, blocks with k0 = k1, `n_eff_seed_cells_if_this_recipe_wins` (= 2 × blocks − equal blocks), and the recorded distinct-input counts.
- **Unexplained equalities** still invalidate. The error reads `<n> unexplained seed-dependent equalities on the minimum set`, and the number is also in `unexplained_minimum_set_equalities`.
- **Seed checks are unchanged:** offsets 300/400/500/600 are forbidden and the registered +700 seeds are required.
- **The strict frozen result** is still reported as `pre_ER1`, and `repair` carries the post-hoc label.
- **Consistency check:** a local per-equality classification is asserted against `nrm.rules`' own count, and an `AssertionError` is raised if they disagree.

### How to call it
The signature is unchanged and ER1 is the default:

`frc.rules.grid_validity(seeds, checks, minimum_set, distinct_inputs)`

It also accepts `required_k=` as before, and a new keyword `er1=True`; `er1=False` gives the frozen strict rule. `frc.claims.validity` can pass its arguments as they are.

### Records
- **RULES.md section 15:**
  - labelled POST HOC, dated, written after the FRC1_MAIN validity failure (85 minimum-set equalities) and before any D2–D4 read;
  - gives your facts, the extra-trees frequency-table mechanism, the rule, and why it does not weaken a decision (duplicate cells count once in n_eff, and the household SE bound holds for any cell correlation).
- **RULES.md section 16:** completion record.
- **Section 9:** carries a pointer to ER1.
- **RULES.json:** `reader_seeds.engineering_repair_ER1` and a `change_record` entry.

### Tests
- `python3 -m frc.test_rules`: **18 of 18 PASS, exit 0.** The new test `t_grid_validity_er1` checks:
  - two explained minimum-set equalities pass, with n_eff 7 out of 4 blocks checked, and `pre_ER1` failing;
  - unexplained equalities fail, with (c) missing, with (a) failing (every recipe equal), and with (b) failing (equal on every view);
  - seeds at +600 still fail;
  - behaviour off the minimum set is unchanged;
  - the refit check is unchanged.
- I also seeded 24 single-line bugs into an in-memory copy of `frc/rules.py`, 4 of them in ER1. The suite caught all 24.

### Decision for you
`refit_effectiveness` still uses the strict rule (`er1=False`), as instructed. By the same extra-trees mechanism it will probably report `REFIT_NOT_GENUINELY_EFFECTIVE`, with the robustness reader seeds counted once. Extending ER1 to it would need your separate decision.
