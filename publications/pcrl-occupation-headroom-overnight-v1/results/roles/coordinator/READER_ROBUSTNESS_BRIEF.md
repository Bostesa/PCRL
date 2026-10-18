# Brief for the reader-robustness investigator (coordinator)

Read `AGENT_BRIEF_COMMON.md`, `DEV_GRID.json`, `DEVGRID_CODE_LOCK.json`, `ohn/devreaders.py`, `ohn/grid_summary.py`, and the frozen
stability rule in `roles/statistical_reviewer/SELECTION_RULE.md` / `RULES.json`. Development evidence only (MODEL_SELECT, adaptively
used); never ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT; final attack-validation outcomes can never guide a release.

The registered grid already gives: two household-split repetitions with household-disjoint FIT / VALIDATION / SCORE reader roles x
two reader seeds per release seed (8 cells per configuration), the enhanced slate with ignore-one alternatives, comparator readers
refitted in every cell, winners on VALIDATION, SCORE AUCs with household-bootstrap SEs, near ties and winner changes, and the frozen
stability label. Your job is to make reader robustness part of the development evidence beyond that:
1. Learning curves: nested FIT fractions 1/4, 1/2 and full for a SMALL frozen bank (the selected witness(es), the best protected
   candidates and their matched continuation / compression controls, SUCC, RETIRED_B3CTL, H_LJ, NEURAL_RAW): does each benefit grow
   or shrink as readers get more data? (`"fractions": [1.0, 0.5, 0.25]` in a bank spec; the 1.0 cells are cached.) Flag
   BENEFIT_SHRINKS_WITH_READER_DATA per the frozen rule. Smaller samples cannot establish full-budget protection; say so.
2. Sign stability: for every candidate-comparator pair, the paired benefit distribution over the 8 cells (mean, min, per-seed means,
   sd_total and its reader-split / sampling components, predicted flip probability), which pairs change sign and in which cells,
   and whether a positive conditional interval in one cell coexists with sign changes elsewhere.
3. Winner and near-tie analysis: how often the VALIDATION winner differs across cells for the same release, how often winners are
   near-tied, and whether candidates and comparators are read by different recipe families (a confound for paired benefits).
4. A DIAGNOSTIC probe beyond the frozen slate (never used for selection): for the best protected candidate, its witness and SUCC,
   refit 2-3 additional reader designs on the same FIT / VALIDATION / SCORE roles with the same access (released probability vectors
   and decisions only; e.g. a larger MLP, a longer-trained gradient boosting, k-nearest neighbours on log p) and report whether the
   ordering of pair recovery changes. Lower recovery for the declared readers is not protection against every possible reader.
Run everything through the queue (stage `readers` or `diagnostic`, the `.venv` interpreter for readers); keep people and reader-fit
counts separate. Return the full report as the text of your final response; write machine-readable results under
`results/pcrl_occupation_headroom_overnight_v1/roles/reader_robustness/`.
