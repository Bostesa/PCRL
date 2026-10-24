# Anchor selection (coordinator; frozen before any new fit, reader or label read of this campaign)

Written 2026-10-24 from 04:20Z (`date -u`), from existing records only (`ANCHOR_TABLE.md` / `.json`, built label-free by
`python3 -m sar.anchors table` from the input-and-pair campaign's saved grid IPR1_MAIN and its MODEL_SELECT utility records). No
candidate result of this campaign exists. The choice is informed by extensive past development on the same people: every number
below was produced and read by earlier campaigns, so this is an adaptive, informed choice and is described that way.

## The two starting releases

| Role | Release | Why it is a starting point | The condition that limits further progress |
|---|---|---|---|
| **A**: strong protection with a specific usefulness weakness | **SUCC** = H3_STRONG_LJDIR_PB income + JOINT_TREE E20 occupation (frozen; both versions S0, S1) | Second-strongest continuous-output release on IPR1_MAIN (pair 0.6074, se 0.0113; income view 0.5312; occupation view 0.5731). Point benefit +0.0179 (0.0105) over NEURAL_RAW and +0.0112 (0.0074) over H_LJ (both point only). All 28 utility conditions pass at the point in both versions. Its income output has wide margin (income reserve 3.51 / 2.86) and continuous probabilities usable for thresholds, ranking or deferral. | **The men's within-SEX occupation Brier guard** of the JOINT occupation output: +0.0044 (se 0.0018) in v0 and +0.0039 (0.0017) in v1 against the 0.005 cap (0.883 / 0.788 of cap; two-SE reserve 0.32 / 0.62), while the women's guard is -0.0049 / -0.0030. The JOINT correction costs both groups about the same Brier against its own task-only routing (TASK_REFIT: men -0.0012 / -0.0004, women -0.0102 / -0.0082), but the tree base leaves men almost no slack against U0. Second: the pair (0.6074) sits 0.034 above the occupation view: the H3 income vector was never protected jointly with this occupation output. |
| **B**: a useful alternative with enough margin for a different construction | **NEURAL_RAW** = the U0 network trained from U0 with the joint objective, rho 1, checkpoint E2 (frozen; both versions) | A material-target reference with large point margins on every condition (max 0.340 / 0.570 of cap) and continuous outputs; both task models are trainable networks with stored checkpoints for both versions. | **Its protection is mild and sits in the occupation view**: pair 0.6253 = occupation view 0.6253 (income view 0.5336). Against U0 it removes only 0.013 pair AUC (resolved, se 0.0047). The stronger points of its own path (rho 3, 10; later epochs) were not chosen by the replication's rule. Reserve 1.95 / 1.56 (men's guard v0; income Brier vs U0 v1), but point margins of 0.66 / 0.43 of cap remain to be spent. |

## Why not the others (not selected by pair AUC alone)

- **RETIRED_B3CTL** (pair 0.5964, the strongest; +0.0289 (0.0139) over NEURAL_RAW, +0.0222 (0.0111) over H_LJ, both AGG_POSITIVE,
  0.020 at the point only). Two binding conditions with near-zero slack (the same men's guard, 0.32 SE, and the income Brier clause
  against Ucal, 0.35 SE), a 6-level income code (6 distinct income vectors, little use for ranking), READER_SENSITIVE under the
  frozen-finalist validation's enhanced bank, and retired as a competitive candidate. Kept as a strong reference (D3); its retired claim
  is not reopened.
- **B2TWIN** (0.6061): the same occupation output as SUCC with a 6-level income code; a control of the B3CTL search. Reference.
- **H_LJ** (0.6186): the same occupation output and binding guard as SUCC with a weaker income partner; a material-target reference.
- **DDR_PTC_R1** (0.6239): wide margins, but a closed idea family (propensity tilts), an adaptive winner's-curse selection and never
  separated from its release-only control. Beside reference.
- **TWJ, W, TT100, TR100** (0.636-0.648): weaker; references only.
- **The recent campaigns' task-only twins** (CTL_TWIN 0.6480): the weak starting point this campaign replaces.

## Frozen identities

- **SUCC_S<v>** (v = 0, 1): income wire sha256 S0 `067a21bb...1d83`, S1 `1e028f72...5d52`; occupation wire S0 `a7451cc6...0c13`, S1
  `1bc06ccc...c058` (MODEL_SELECT; full hashes in `results/pcrl_input_and_pair_release_v1/SOURCE_ADMISSION.json`
  `reference_releases_MODEL_SELECT_wire_sha256`). Components: H3 = `ihs` H/H3_STRONG_LJDIR_PB S<v> (admitted mirror); JOINT occupation =
  acs_replication STAGE_B_<v> JOINT_TREE E20 on the STAGE_A_<v> T_TREE routing.
- **NEURAL_RAW_S<v>**: income S0 `f4de1c9f...781d`, S1 `44560815...ef37`; occupation S0 `8311ca98...1a1b`, S1 `1c6d0848...91be`.
  Component: acs_replication runs/NEURAL_RAW_<v>/NEURAL_RAW__rho1.0__E2.pt.

## The original benchmark (unchanged)

Florida ACS 2019 development cohort; income class (2) and occupation group (5) from AGEP, WKHP, COW, SCHL, MAR, POBP; each recipient
receives only its own float64 probability vector and argmax decision; one released instance per person. Utility: accuracy loss
<= 0.010, NLL excess <= 0.010 nats, summed Brier excess <= 0.005 against U0_S<v> and Ucal_S<v> for both tasks (12 clauses per
version) and the men's and women's occupation summed-Brier excess against U0 <= 0.005 (2 guards per version): 28 conditions over the
two versions, all at the point. Recipient condition: neither single view more recoverable than the comparator's by more than 0.010
AUC (per version, at the aggregate point). Material target: a resolved 0.020 linked-pair benefit against NEURAL_RAW and H_LJ.
