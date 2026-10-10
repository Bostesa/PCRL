# R3 postfit utility units

The four B0 witness units are registered in `WITNESS_UNIT_INDEX.json` and run in queue stage `witness` before the bank/channel milestones. They evaluate both seeds on FIT and INNER with an exact one-action identity law synthesized from the pinned saved B0 wires. Their FIT bank-relative margins must equal zero and all 24 original INNER point clauses must pass before protection fits are admitted. They do not need any R2 candidate fit.

After R0 marks the new bank and channel identity milestones, register at most **28** R3 `protection` scoring units: seven fitted arms `T,D,P,L,J,S12,S21` × two seeds × roles `OSF_DEFENSE_FIT,INNER_SELECTION`. Keep absent or failed R2 outputs as `NOT_RUN`, not synthetic identity results. One R2 checkpoint/arm/seed is chosen using FIT/internal quantities before its INNER unit. For each arm/seed, the FIT R3 unit depends on R2's saved checkpoint plus FIT-law materialization; its INNER R3 unit depends on the FIT utility receipt and its own INNER-law materialization. These are proposed logical dependencies; R0 binds exact queue unit IDs and bytes when R2 outputs exist.

`python3 -m bank_anchored.r3_make_protection_config` derives one private
config, private queue spec and public hash-only index from the corresponding
corrected B0 witness config/spec, a completed R2 realization-zero law and
`BAR1-role-release-v1` receipt, the selected R2 policy receipt and package,
the seed's public full-FIT source-identity receipt, the authoritative public
R2 F1 selected-unit receipt, and the latest R0
protection-stage amendment. It verifies that release, selected policy and
source identity share exactly one source-binding hash; the release names the
selected policy receipt hash; and the policy receipt names the pinned policy
package hash. The selected-unit receipt must name the same policy and exact
role/realization-zero law and release hashes, and its explicit successor unit
ID must appear in the requested queue dependencies. It also verifies inherited source bytes and R2 arm/seed/role/law
identity without opening role arrays or law members. The spec inherits the
exact interpreter path from its successfully completed corrected B0 witness
unit, avoiding an untested `python3` launcher. R0 still chooses and registers
the dependency IDs in the shared queue. Each output unit ID and path is new.

Each protected private config uses `pcrl-bank-r3-utility-job-v1`, `stage=protection`, exact arm and seed, exact FIT or INNER role, and the same source records as its matching B0 witness unit **plus** `law={path,sha256}` for the R2 private role law. That NPZ must have exactly eight members: `row_id,unit,code_r1,code_r2,mass,valid,b0_code_r1,b0_code_r2`. The R3 worker verifies ordered role rows/groups, law normalization, in-bank global IDs, class preservation and exact B0 codes against the pinned original R1 wire. The queue sources additionally pin the R2 law file; no label or SEX archive is an R2-to-R3 payload.

Each queue spec calls `python3 -m bank_anchored.r3_worker --config <owner-only path>` from the campaign worktree, with `stage=protection`, 60-second wall and CPU reservation, 1 GiB RSS cap, 8 MB output reservation, numerical threads one, and source byte pins for config, seal, role manifest, three role-only archives, selected source table, R1 bank receipt/wire, historical independent bank replay, R3 worker/scorer and R2 law. R0 should revise the reservation after the first witness smoke if it is materially inaccurate, before registering the affected units.

Each unit emits an 18-row aggregate CSV and a public receipt, plus an owner-only paired-person cost-vector NPZ. On FIT, six B0-relative proxy rows are primary; U0/Ucal rows are old-style diagnostics and do not gate selection. On INNER, 12 U0/Ucal rows are primary; B0 rows are descriptive. A two-seed 24/24 point pass is assembled by R0 from the two INNER receipts before an arm enters R4's fresh attack stage. R4's admission JSON needs `arm_id`, `all_two_seed_task_reference_point_caps_pass`, both R3 receipt hashes and the frozen law/decoder identity. The main J nominee is never replaced by a passing control.

R3's `r3_intervals.py` later reads only the owner-only INNER cost vectors from completed R3 units, retains all 360 primary utility clauses in the fixed K=1,575 family, and reports paired duplicate-group conditional development intervals. W_Q/P_STAR family slots remain null unless R0 separately admits their exact current-role cost vectors under the same service/reference contract. The R4 recovery inference includes three whole-release cached draws per stochastic arm; R3 exact-law utility does not select a draw.
