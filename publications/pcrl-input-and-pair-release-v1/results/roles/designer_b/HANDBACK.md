# Designer B's hand-back summary (verbatim)

Saved by the coordinator after 02:02Z on 2026-10-22, verbatim from Designer B's final message (a subagent of this campaign; its
words, not the coordinator's). The proposal itself is `PROPOSAL.md` in this folder. One change: the absolute worktree path in the
last paragraph is shortened to the repository-relative path (a personal path would be removed by the publication sanitizer).

---

Designer B is done (01:31Z to 02:02Z). I proposed CSF-T, but my label-free plug-in evidence says nothing left open in space B plausibly beats its matched controls by 0.02 pair AUC. I registered null predictions and recommend a reduced allocation.

1. **Operation (CSF-T).** The twin is the cached TASK.pkl HGB teachers, unchanged. For a keyed fraction of people (SHA-256 of the six compact fields), both recipients receive their decision-cell mean vector instead of the twin's (the repository's B1 teacher-mean tables); the same people are collapsed in both recipients. The rate per joint decision cell (10 cells) is water-filled using a cross-fitted release-level SEX propensity on the twin's log-ratio pair (MODEL_FIT SEX labels). Each cell's priority is its within-cell propensity variance divided by its S2-normalized flattening KL. Decisions are kept exactly, so accuracy loss is 0, and no runtime SEX is read. Strength grid pi {0.025, 0.05, 0.075, 0.10, 0.15, 0.20}, selected by the PROTOCOL 4 default rule with budget S2 on MODEL_FIT nested household folds.
2. **Closest previous work.** CTL_REM (`nrm/ctl/release.py`: occupation remainder only, everyone, top probability kept), DQ (`ddr/ctlkit.py`: everyone, independent dithers per recipient), HQ/PRC (`nrm/fa/prc.py`: routes every person) and the row-4 bank channels. CSF-T differs by being selective, collapsing completely, coordinating across recipients, and targeting only through the 10 released decision cells.
3. **Failures addressed.** (2)/(3) super-additivity and leakage moving between recipients: the plug-in world reproduces the CTL_REM re-assembly (occupation view down 0.070, pair down only 0.013). Also (4) table readers, (5) income accuracy, and (6) release-only, since CSF-T is release-only by design.
4. **Twin and controls.** The twin is the shared cached teachers. The simplest matched control is uniform SEX-blind CSF at the same spend, which is also the ablation. The SBC path refits the propensity on permuted SEX (PSEUDO2 fallback). The SPEND path is DQ. The release-only (RO) path is ZTILT on a fine grid, because CSF-T uses no person-level SEX statistic beyond the twin's release. D5 controls: separately designed recipients (each recipient allocated on its own view, independent keys) and both sequential orders.
5. **Label-free feasibility (plug-in, V0/V1):**
   - Twin pair .6394/.6413; both recipients reduced to decisions .577/.578.
   - Coordination gain (CSF vs independent keys) is +0.0011/+0.0001 at pi .15 and +0.0028/.0000 at pi .30.
   - CSF-T vs uniform CSF is +0.0023/+0.0007 at pi .10, and its allocation lands on a different single cell in each version. An oracle allocation adds only +0.002/+0.0005.
   - CSF-T vs DQ at equal KL is +0.0045/+0.0030.
   - Efficiency is 0.3-0.5 pair AUC per nat, against about 5.6 for the closed PTC tilt in the same plug-in world.
   - At the S2-feasible pi (about 0.05) I expect about 0.004 below the twin.
6. **Protocol consequence.** Under PROTOCOL 4, a SEX-blind operation's SBC path is an exact alias of the operation, so it cannot meet D2 by construction. That is why the targeted CSF-T is the candidate.
7. **Revisions.** R1 splits the income and occupation rates on nested subsets, triggered when income NLL binds while occupation keeps 40% slack. R2 uses a shallower collapse, triggered when pi_S2 <= 0.025 in both versions.
8. **Predictions.** D2 not separated against SBC, SPEND, RO and the ablation; no D5 advantage; D4 not met (p 0.99); outcome TASK_ONLY_FEASIBLE_NO_PROTECTION (p 0.85).
9. **Configurations and CPU.** 16 configurations plus 6 for D5 and at most 4 for revisions. My recommendation is 8: CAND, ablation, one SPEND and one SBC point per version. Under that option D2 reads PROTECTION_NOT_ASSESSED, because the RO path and the second bracket points would be missing. Mechanism-side CPU is about 5 minutes; the reader grid is about 3,200 CPU-s for 16 configurations.
10. **CPU used:** about 135 s of process CPU, one thread (the in-script totals sum to 124 s). No label file, MODEL_SELECT, reader cache or `.private` runs/access content was opened, and designer A's directory was not read. Novelty is not claimed.

I wrote only inside `results/pcrl_input_and_pair_release_v1/roles/designer_b/`; nothing was committed and no background processes were started. `PROPOSAL.md` is about 240 lines, longer than the two pages asked for.

Files are in `results/pcrl_input_and_pair_release_v1/roles/designer_b/`:
- `PROPOSAL.md`
- `mech_sketch.py` (`selftest()` passes on invented data)
- `feasibility.py`, `feasibility_extra.py`, `feasibility_cell.py`, `spend_grid.py`
- `FEASIBILITY.json`, `FEASIBILITY_V0.json`, `FEASIBILITY_EXTRA.json`, `FEASIBILITY_CELL.json`, `FEASIBILITY_CELL_UNNORMALIZED.json`, `SPEND_GRID.json`
