# FRC1 rules, addendum ER1 (post hoc engineering repair of grid validity)

Author of the rule text below: the statistical reviewer of campaign `pcrl_finite_routing_comparison_v1` (written 2026-10-21
00:28:29Z-00:30:09Z, commit 6d6cd0829). Moved here verbatim by the coordinator at 2026-10-21T00:52:31Z (`date -u`); the reviewer's words are
unchanged.

**Why it was moved.** The reviewer wrote ER1 into the two registration files `RULES.md` (a two-line pointer in section 9 and new
sections 15-16) and `RULES.json` (`reader_seeds.engineering_repair_ER1` and one `change_record` entry). The campaign's access gate
(`frc.common.registered`) allows a label read only while the eight registration files equal the published registration receipt
7fed2ca44, so every label read after commit 6d6cd0829 (00:32:13Z) was refused: four queue units failed before reading anything and
no label was read (PROTOCOL_DEVIATIONS.md, deviation 9). Registered files are kept immutable and amendments live in separate dated
files, as for the mechanism cards (deviation 7). `RULES.md` and `RULES.json` are restored to the receipt's bytes (SHA-256
2b7e0be8... and a0dc0fc7...). The post-ER1 versions (SHA-256 0121711e... and e2a6af23...) stay in git at 6d6cd0829 and are exactly
the registered files plus the text below (checked by script before the move).

**What did not change.** No rule. ER1 is implemented in `frc/rules.py` (`grid_validity`, `er1=True` default for claims;
`refit_effectiveness` keeps the frozen strict rule) and tested by `frc/test_rules.py` (`t_grid_validity_er1`). Neither file is
touched by this move. References to "RULES.md section 15" in those files and in the claims records now point here. `frc.rules.load_rules`
checks only constants present in the registered RULES.json.

---

## Pointer the reviewer added at the end of RULES.md section 9 (verbatim)

**Post hoc engineering repair ER1 (2026-10-21, section 15)** changes how minimum-set equalities are judged; the seed part is
unchanged.


## 15. Engineering repair ER1 of grid validity (POST HOC; written 2026-10-21T00:28:29Z to 00:30:09Z `date -u`)

**Status.** POST HOC. Written after `frc.rules.grid_validity` failed on the FRC1_MAIN grid and before any D2-D4 label, comparison
value or AUC was read: the coordinator reports (00:27Z) that the claims run returned only `NOT_COMPUTED_GRID_INVALID`. I read no
label, no grid output and no comparison value; my inputs were the coordinator's description below. This is the "engineering repair"
that section 6 of NRM1 (inherited) requires before a configuration can be selected.

**Failure (coordinator's facts).** 4,032 recorded k0 / k1 checks; the frozen rule reported "85 seed-dependent equalities on the
minimum set". All 85 are ET5 / ET25 on the income view (16 distinct transmitted vectors) or ET5 (once ET25, ORD_S3 with 32 values)
on the occupation view (64 distinct vectors) of the finite releases PRC_S2, TWIN, HQ_S2, SHUF_S2, UNT_BR, ORD_S2 and ORD_S3. For
every one: (a) another seed-dependent recipe differs in the same wire, view, version and repetition (MLP64, MLP6432, OUT_MLP and
LOG_MLP differ in all 28 minimum-set groups of each view); (b) the same recipe differs on the pair view of the same wire; (c) the
view's distinct-input count is recorded label-free.

**Mechanism.** An extra-trees forest on a view with few distinct inputs and many reader-FIT people per value (about 250 per income
value, about 62 per occupation value; minimum leaf 5 / 25) isolates each distinct value in its own leaf in every tree. It then
reduces to a seed-invariant, unshrunk frequency table, so its k0 and k1 SCORE outputs are bitwise equal. The reader seed is still
effective: the MLP recipes differ in every such group, and the same ET recipe differs on the pair view.

**Repair ER1.** For a seed-dependent equality on the minimum set, the NRM1 (iii) explanation (a) + (b) + (c) applies exactly as for
other wires (`nrm.rules.grid_validity` is called unchanged with an empty minimum set). An explained minimum-set equality no longer
invalidates the grid. It is reported separately as `explained_minimum_set_equalities`: count, items, and the n_eff it implies per
wire / view / recipe (`n_eff_seed_cells_if_this_recipe_wins` = 2 x blocks checked - blocks with k0 = k1; the screen already counts
such duplicate cells once). An unexplained minimum-set equality still invalidates ("... unexplained seed-dependent equalities on
the minimum set"). The seed part (offsets 300 / 400 / 500 / 600 forbidden, registered +700 seeds required) is unchanged. The strict
frozen result is reported beside as `pre_ER1`.

**Why it does not weaken a decision.** An explained equality means one recipe is deterministic on one view. Its duplicated k0 / k1
cells are counted once in n_eff by the aggregate screen, and the household SE bound of NRM1 section 5.2 holds for any cell
correlation. "Across reader seeds" stays governed by the reader-seed qualifier of each comparison. A wire where no other
seed-dependent recipe differs (a), where the recipe is equal on every view (b), or whose distinct-input count is not recorded (c)
still fails.

**Scope.** `grid_validity(seeds, checks, minimum_set, distinct_inputs)`: same call, with `er1=True` the default and the rule for
claims. `refit_effectiveness` keeps the frozen strict rule (`er1=False`). By the same mechanism it is likely to report
`REFIT_NOT_GENUINELY_EFFECTIVE`, with the robustness reader seeds counted once, unless ER1 is extended to it by a separate decision.
Nothing else changed. Test: `t_grid_validity_er1` (an explained minimum-set equality passes; unexplained ones fail on (a), (b) or
(c); seeds still checked; off-minimum-set behaviour unchanged; refit unchanged). RULES.json: `reader_seeds.engineering_repair_ER1`
and `change_record`.

## 16. Completion record (ER1)

- Completed 2026-10-21T00:30:09Z (`date -u`, read before this section was written).
- `python3 -m frc.test_rules`: 18 of 18 PASS, exit 0 (invented data only). The scratch in-memory mutation check (24 single-line
  seeded bugs, 4 of them in ER1: ER1 switched off, unexplained minimum-set equalities ignored, seed errors dropped, the refit using
  ER1) was caught 24 of 24 times; no file was changed by it.
- SHA-256 at completion: given in the hand-back report (RULES.json, frc/rules.py and frc/test_rules.py were hashed after this
  line was written, and this file cannot contain its own hash).
- No label was read and no real-data computation was run.
