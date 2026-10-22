# Designer B proposal: CSF-T, coordinated, cell-targeted selective flattening (campaign `pcrl_input_and_pair_release_v1`)

Written 2026-10-22, 01:52-02:02Z (`date -u`). Design space B: after the task models, jointly for the linked pair.

Label-free throughout:
- not opened: any label file, anything under `admitted/exports/*/{income,occupation,sex}.npz`, anything of MODEL_SELECT, any reader
  cache, any `.private/*/runs` or `.private/*/access` content;
- read: the brief, the records it names, this campaign's `PROTOCOL.md` (sections 4, 6 and 8), and the previous designer B's
  proposal and feasibility code;
- loaded: `TASK.pkl` and `SEXSIDE.pkl` (V0, V1), and `ddr.common.features('MODEL_FIT')` (six-field keys and household units only);
- not read: designer A's directory.

CPU used: about 135 s of process CPU, one thread. The in-script process CPU sums to 124 s; imports add about 10-15 s.

**Bottom line first. My label-free evidence says nothing left open in space B plausibly beats its matched controls by 0.02 pair
AUC.**
- **The redundancy idea fails in simulation.** The most pair-specific open operation collapses a person's information in both
  recipients at once. In the plug-in world its joint-design (coordination) gain is 0.000-0.003. Targeting it at the coarse
  joint-decision-cell level adds 0.001-0.004, and even an oracle allocation adds 0.0005-0.002.
- **SEX-blind information removal is too expensive.** It buys 0.3-0.5 pair AUC per nat, about ten times less than the closed
  tilts.
- **The protocol rules out a SEX-blind candidate.** Under PROTOCOL section 4, a SEX-blind operation's SBC path is an exact alias of
  the operation, so it cannot meet D2 by construction. The registered candidate is therefore the targeted form CSF-T, and the
  SEX-blind form CSF is its targeting-off ablation.
- **What I am proposing.** CSF-T, as a cheap, decisive close-out of the pair-joint part of B, with null predictions registered.
  I recommend a reduced allocation (section 7).

## 1. The operation (CSF-T; mode `cell` of `mech_sketch.py`)

| Fitted part | Role, labels | Size |
|---|---|---|
| p_I(x), p_O(x): the cached task-only HGB teachers (`TASK.pkl`, A1_PO_16), unchanged | the twin; task labels only | fixed |
| m_I[d], m_O[k]: means of the twin's household out-of-fold vectors over MODEL_FIT people with income decision d, or occupation decision k (B1 teacher-mean tables) | MODEL_FIT, from the twin's vectors | 4 + 25 numbers |
| z(p): release-level SEX propensity, an HGB (15 leaves, 200 iterations) on the twin's centred log-probability pair, cross-fitted on 5 household folds | MODEL_FIT SEX label (training only) | 1 model |
| rates r_c over the 10 joint decision cells c = (income decision, occupation decision), water-filled at strength pi | from z and the task-only costs | 10 numbers |
| u(x) = SHA-256(`IPR_B_CSF\|V<v>` \| six compact fields) / 16^13 | none | none |

**Rates.**
- The per-person flattening cost is kappa(x) = KL(p_I \|\| m_I[d]) / 0.004 + KL(p_O \|\| m_O[k]) / 0.008, normalized by the S2 NLL
  budgets.
- A cell's priority is Var_c(z) / mean_c(kappa).
- Cells are fully flattened in priority order, with the marginal cell partially, until sum kappa x rate = pi x sum kappa. That is
  the cost of uniform flattening of a fraction pi.

**Release.**

    if u(x) < r_c(x):  r_I = m_I[argmax p_I(x)],  r_O = m_O[argmax p_O(x)]      (both recipients, the same people)
    else:              r_I = p_I(x),              r_O = p_O(x)                    (the twin, bitwise)

- Each recipient receives only its own vector and argmax.
- Decisions are kept exactly: a decision-cell mean keeps its decision as argmax, so both accuracy losses are 0.
- The release is deterministic in the six fields and never reads runtime SEX.
- The only SEX-dependent fitted input is z, a function of the twin's release.

**Strength and selection.**
- The grid is pi in {0.025, 0.05, 0.075, 0.10, 0.15, 0.20}.
- The PROTOCOL 4 default rule applies with budget S2: the largest pi such that it and every smaller pi meet S2 over the twin
  (estimate + 2 household-clustered SE). The estimate is nested: 5 outer household folds, with tables, z and rates refit on four
  folds and applied to the fifth.
- MODEL_FIT SEX labels enter only as the guard constraint and through z.
- Reason for S2: the plug-in binding components (income NLL for CSF-T, the men's occupation guard for CSF) reach S2 at
  pi 0.05-0.075.

## 2. Closest previous implementations and what is substantively different

- **CTL_REM** (`nrm/ctl/release.py`, `pcrl_new_release_mechanisms_v1`): flattening of the occupation remainder for everyone,
  SEX-blind, occupation only, keeping the top probability.
- **DQ** (`ddr/ctlkit.py`, `pcrl_dual_design_release_v1`): SEX-blind dithered quantization of everyone, with independent dithers
  per recipient.
- **HQ / PRC** (`nrm/fa/prc.py`, `pcrl_finite_routing_comparison_v1`): routing every person to 16 x 64 cells under a plug-in
  SEX-information objective.
- **Row 4's frozen-bank channels** (Adult): one draw per person between the identity and a class medoid, per recipient.

CSF-T differs in four ways:
- **Selective:** a keyed fraction is collapsed and everyone else gets the continuous twin.
- **Complete collapse:** the top probability is removed too. REM_TOP below shows why keeping it fails.
- **Coordinated:** the same people are collapsed in both recipients.
- **Targeted only through 10 released decision cells:** no person is routed by a propensity, and there is no tilt, logit noise,
  decoder or controller.

## 3. Which measured failure it addresses

- **(2) Super-additivity and (3) leakage moving between recipients.**
  - The plug-in world reproduces the measured CTL_REM re-assembly. REM_TOP_OCC (decision and top probability kept, twin income)
    lowers the occupation view by 0.070 / 0.073 but the pair by only 0.013 / 0.014. On the benchmark, CTL_REM_1p00 gave
    0.5603 / 0.6457 against a 0.6605 base, and TWJ gave 0.560 / 0.648.
  - For a collapsed person, neither recipient carries anything beyond the joint decision cell, so nothing can re-assemble.
- **(4) Table readers.** The only new finite values are the 10 decision-cell rows, which carry exactly the decision floor.
- **(5) Income accuracy.** Unchanged by construction.
- **(6) Release-only explanations.** CSF-T is release-only by design: z is fitted on the twin's release. So its RO comparison asks
  whether information-removing collapse beats a release-only re-encoding (ZTILT) at equal spend.

## 4. Plug-in feasibility, and why it will not reach 0.02

The plug-in world is the previous designer B's construction (`feasibility.py`, `feasibility_extra.py`, `feasibility_cell.py`,
`spend_grid.py` and their JSON records):
- q plays the truth.
- Readers are HGB15 and LR on (p, log p, one-hot), trained on 4,000 pseudo-labelled people; the expected AUC is taken on the other
  16,000; the better reader per view is kept; 2 draws.
- Spend is measured against the twin posterior; the occupation guards use `occ_s`.
- CSF-T's z is fitted on one pseudo-SEX draw from q.
- These are simulations, not benchmark evidence.

| Release (V0 / V1) | Pair | Income | Occupation | KL inc / occ (V0) | Worst S2 ratio |
|---|---|---|---|---|---|
| TWIN | .6394 / .6413 | .5546 / .5521 | .6261 / .6283 | 0 | 0 |
| Both flattened for everyone (decision floor) | .5766 / .5779 | .534 / .535 | .532 / .533 | .050 / .092 | 13.4 / 13.3 |
| Occupation flattened for everyone, twin income | .5960 / .5970 | .555 / .552 | .532 / .533 | 0 / .092 | 13.4 / 13.3 |
| Income flattened for everyone, twin occupation | .6301 / .6325 | .534 / .535 | .626 / .628 | .050 / 0 | 12.5 / 12.8 |
| REM_TOP_OCC (CTL_REM analogue) | .6264 / .6275 | .555 / .552 | .556 / .555 | 0 / .038 | 4.74 / 4.64 |
| CSF pi .10 (uniform, SEX-blind) | .6333 / .6357 | .554 / .551 | .619 / .622 | .0051 / .0091 | 1.39 / 1.34 |
| **CSF-T pi .10** | .6310 / .6350 | .554 / .550 | .620 / .620 | .0039 / .0124 | 1.94 / 1.66 |
| DQ at CSF-T .10's total KL | .6355 / .6380 | .552 / .555 | .624 / .626 | .0073 / .0090 | 1.91 / 1.54 |
| CSF pi .15 | .6304 / .6322 | .554 / .550 | .616 / .618 | .0077 / .0139 | 2.11 / 1.98 |
| IND pi .15 (independent keys) | .6315 / .6323 | .554 / .550 | .617 / .618 | .0077 / .0137 | 2.00 / 2.04 |
| **CSF-T pi .15** | .6264 / .6316 | .552 / .549 | .617 / .616 | .0057 / .0185 | 2.86 / 2.43 |
| DQ at CSF-T .15's total KL | .6330 / .6364 | .551 / .552 | .622 / .625 | .0107 / .0134 | 2.83 / 2.35 |
| CSF pi .30 / IND pi .30 | .6202 / .6229 vs .6230 / .6229 | | | .0156 / .0279 | 4.19 / 3.91 |
| Occupation-only CSF at CSF .30's total KL (pi_O .47) | .6173 / .6206 | .555 / .552 | .588 / .591 | 0 / .043 | 6.49 / 6.14 |
| CSF with an ORACLE (q-based) cell allocation, at CSF .15's KL | .6283 / .6317 | | | | 2.67 / 2.69 |

What the evidence says:
- **Coordination buys almost nothing:** CSF vs IND is +0.0011 / +0.0001 at pi 0.15 and +0.0028 / 0.0000 at 0.30.
  - The redundancy is inside the occupation vector (confidence and remainder), not between the two recipients.
  - Flattening income for everyone moves the pair by only 0.009.
- **Targeting adds 0.001-0.004 and is unstable.**
  - CSF-T vs uniform CSF is +0.0023 / +0.0007 at pi 0.10 and +0.0040 / +0.0006 at 0.15.
  - The allocation lands on one cell, and not the same one in both versions: cell (income class 1, occupation 0) at rate 0.28 in
    V0, cell (income class 0, occupation 2) at rate 0.71 in V1.
  - Concentrating the cost raises the worst S2 ratio, from 1.39 to 1.94 at pi 0.10.
  - Without cost normalization (`FEASIBILITY_CELL_UNNORMALIZED.json`), income NLL binds at 1.84 x S2. With it, V0 moves the
    spend onto occupation (KL .0039 / .0124) and V1 does not (.0066 / .0059).
- **Against DQ at equal total KL:** +0.0045 / +0.0030 at pi 0.10. At equal S2 feasibility the margin is smaller.
- **Efficiency is far below what 0.02 needs.** Pair AUC drop per nat of total spend:
  - CSF: 0.42 / 0.44;
  - occupation-only: 0.51 / 0.50;
  - DQ: 0.29 / 0.31;
  - the closed tilt PTC in this plug-in world: about 5.6;
  - on the benchmark, DB_PTC_R1 about 2.4, DB_ZTILT_R1 about 0.66 and DA_SPD_HI 0.2-0.3.
  - At the S2-feasible pi (about 0.05), I expect about 0.004 below the twin and 0.001-0.003 below DQ.
- **Why the open part of B is exhausted (argument, not proof).** A release in B is h(p(x), a(x)).
  - If a(x) carries SEX information beyond p (the propensity, or SEX-conditional posteriors), h is a propensity tilt, an STD or a
    routing. All are closed.
  - Otherwise h is release-only.
    - On a finite support (17,744 distinct six-field records), a deterministic h removes information only by merging. Merges are
      the closed PRC / HQ / ORD line, or SEX-blind or coarsely targeted ones like CSF.
    - A randomized h adds noise (closed) or random merges (CSF).
    - Anything else is a re-encoding, which lowers declared-reader AUC only by making the signal harder to learn. That is the RO
      control's mechanism.
  - At 0.3-0.5 AUC per nat, a 0.02 effect needs 0.04-0.07 nats, several times the budget.

## 5. Twin and matched controls (PROTOCOL section 4)

| Config | Role | Content |
|---|---|---|
| TWIN | equally trained task-only twin | the shared cached teachers; CSF-T at pi = 0 is bitwise the twin |
| SBC_LO / SBC_HI | same operation, SEX-independent statistic | CSF-T with z fitted on a fixed-seed permutation of the MODEL_FIT SEX column (fallback PSEUDO2 by the protocol rule); grid as CAND's |
| SPEND_LO / SPEND_HI | ordinary spending | DQ on the twin's vectors; step grid fine enough for the 25% bracketing rule |
| RO_LO / RO_HI | release-only explanation | CSF-T uses no person-level SEX statistic beyond the twin's release, so RO is the generic ZTILT (`ipr.ctlkit`) on a fine grid |
| ABL | targeting off | uniform CSF (mode `coord`) at CAND's pi. It is also the simplest matched control: same spend form, no SEX information |
| D5 (declared joint design) | separately designed recipients; both sequential orders | `cell_indep`: each recipient allocated over its own decision cells with its own-view propensity, independent keys. SEQ_OI: occupation allocated first on its own view, then income's rates water-filled over the joint cells given occupation's released vectors. SEQ_IO: the reverse. All at CAND's MODEL_FIT spend |

## 6. Bounded revision plan (at most two)

- **R1, nested split rates.**
  - *Trigger:* in both versions at pi_S2, the binding S2 component is income NLL or income Brier while every occupation component
    has at least 40% slack. This is observed for CSF-T in the plug-in world.
  - *Revision:* in each targeted cell, flatten occupation at r_c and income at rho x r_c on the same keyed people (a nested
    subset, so still coordinated). rho is in {0, 0.25, 0.5}, selected by the same rule.
  - *Controls:* the SBC and SPEND paths refitted at the new spend.
- **R2, shallower collapse.**
  - *Trigger:* pi_S2 <= 0.025 in both versions.
  - *Revision:* collapsed people keep a 3-level income confidence code and a 2-level occupation top-probability code (cell
    means), which raises the affordable pi.
  - *Controls:* the same, and IND at the same depth.

## 7. Registered predictions, configurations, CPU

Benefit = comparator AUC - CSF-T AUC, on the registered grid.

| # | Prediction | Probability |
|---|---|---|
| P1 | pi_S2 in {0.025, 0.05} in both versions; binding component income NLL / Brier or the men's occupation guard | 0.7 |
| P2 | CAND meets the original 24 clauses and 4 guards at the point in both versions; accuracy loss exactly 0 | 0.85 / 1 |
| P3 | vs TWIN: +0.001 to +0.008; not resolved | 0.75 / 0.9 |
| P4 | vs SBC: within +-0.005; not separated | 0.85 |
| P5 | vs SPEND (DQ): -0.002 to +0.005; not separated | 0.85 |
| P6 | vs RO (ZTILT): CAND not resolved below; the point estimate is more likely negative than positive | 0.9 / 0.6 |
| P7 | vs ABL (uniform CSF): within +-0.004 | 0.8 |
| P8 | D5: vs cell_indep and both sequential orders, all within +-0.005; no joint-design advantage | 0.8 |
| P9 | Recipient non-worsening holds in both versions | 0.95 |
| P10 | D4 (0.020 below NEURAL_RAW 0.6377 and H_LJ 0.6209) NOT met; CAND pair 0.640-0.650 | 0.99 |
| P11 | Outcome class TASK_ONLY_FEASIBLE_NO_PROTECTION | 0.85 |

Falsifiers:
- CAND resolved below SBC, SPEND and RO with a pair benefit of at least 0.010 over each: the pair-joint collapse matters for real
  readers.
- CSF resolved below IND by at least 0.010: redundancy between the recipients is real, contradicting the plug-in reading.

**Configurations** (PROTOCOL 8).

| Allocation | Contents | Count |
|---|---|---|
| Full | CAND 2, SBC 4, SPEND 4, RO 4, ABL 2 | 16 |
| D5 | cell_indep, SEQ_OI, SEQ_IO, 3 per version | 6 |
| Revisions | at most | 4 |

- My recommendation, given section 4: if the configurations are better spent on family IA, run CAND, ABL, SPEND_HI and SBC_HI
  per version (8). In that case D2 is PROTECTION_NOT_ASSESSED, because the RO and second bracket points are missing; that loss is
  acceptable given the predictions.

**CPU, mechanism side, both versions.**
- Tables, rates and keyed hashes over 28,001 rows take under 5 s.
- z: 5 cross-fitted HGB fits per version and per SBC refit, about 10 s each.
- Nested selection: 5 outer folds x z refits, about 60 s per version.
- DQ and ZTILT grids: about 1-2 min.
- The total is about 5 CPU-minutes.
- The reader grid dominates, at about 200 CPU-s per configuration last campaign, so about 3,200 CPU-s for 16.

## 8. Novelty status

Not claimed. The components are published statistical disclosure control devices:
- local suppression and microaggregation (Willenborg and de Waal 2001; Hundepool et al. 2012);
- post-randomization (PRAM; Gouweleeuw et al. 1998) and randomized response (Warner 1965);
- coordinating suppressions across linked releases: complementary (secondary) suppression in tabular SDC (Cox 1980);
- for the targeted allocation, cell-level SEX statistics of the release, as in the privacy funnel and IBSI line already in this
  repository.

The tables are the repository's B1 teacher means. I ran no literature search in this session.

**Record.**
- `mech_sketch.py` (modes `coord`, `indep`, `occ_only`, `inc_only`, `cell`, `cell_indep`; `selftest()` passes on invented data).
- `feasibility.py` -> `FEASIBILITY.json` (and the identical V0 pre-run `FEASIBILITY_V0.json`).
- `feasibility_extra.py` -> `FEASIBILITY_EXTRA.json`.
- `feasibility_cell.py` -> `FEASIBILITY_CELL.json`, plus `FEASIBILITY_CELL_UNNORMALIZED.json`, the first run before the cost
  normalization.
- `spend_grid.py` -> `SPEND_GRID.json`, the uniform-CSF spend on the registered grid.
