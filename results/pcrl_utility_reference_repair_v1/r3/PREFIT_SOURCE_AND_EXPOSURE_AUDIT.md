# R3 independent source and exposure audit, before new fits

Status: static review of pinned source and prior immutable receipts. This is not a forward-pass parity test or an outcome on the proposed common comparison role.

## Prior focused utility gate

The prior `m2/fit_pipeline.py` used `task_only_split` to hold out roughly one quarter of `OSF_DEFENSE_FIT` by group. Its saved `FIT_RECEIPT.json` records 11,574 fit and 3,860 validation rows (11,571 and 3,857 groups) for seed 0; 11,575 fit and 3,859 validation rows (11,571 and 3,857 groups) for seed 1. It trained the F and masked validation heads on the fit subset, then scored F, masks and stored U0/Ucal vectors on the validation subset. No final full-role F head was fitted after the no-go.

The source U model is `smf` task-only `tl__s{k}__e40`, admitted by `osf/admit.py` and used by `osf/baselines.py`. `osf/data.py` and `results/pcrl_online_strength_frontier_v1/PROTOCOL.md` state that its warm start, encoder and training heads use all 15,434 `OSF_DEFENSE_FIT` rows. Its deployed logistic task heads also fit all of that role; their C is selected with `HEAD_VALIDATION` labels. The `osf` role manifest binds the source U release to all 15,434 fitting rows and records scaler moments equal to the full fitting role. The hcal provenance report records independent head refits and exact fit-role receipts.

Therefore all 3,860/3,859 prior external validation rows were already in U0 parameter fitting. They were absent from the new F and masked head fitting. Ucal is the same U source with a global temperature fitted on disjoint `CALIBRATION_HELDOUT`; this does not remove U's training overlap. The old comparison was an exposed-reference versus held-out-candidate diagnostic. Its recorded numerical no-go and independent replay remain valid descriptions of that diagnostic. The exposure asymmetry is established; its share of the observed deficit is not isolated.

The prior F learner also differs from U: `teams/c/training.py` fits a standalone selected-column 64×64 ReLU head with Adam and internal early stopping, whereas U uses 83→64→64→16 encoders with an affine training head, a warm start and task-only continuation, followed by separately fitted logistic deployed heads. Thus a full-role refit and learner change would be a combined repair unless a matched contrast isolates causes.

## Proposed INNER_SELECTION comparison exposure

The admitted role manifest reports zero row and group overlap among `OSF_DEFENSE_FIT`, `CALIBRATION_HELDOUT` and `INNER_SELECTION`. Consequently source U parameter fits and its fitted preprocessing, the Ucal temperature fit, and a newly full-role-fitted candidate have no **direct** training or calibration overlap with INNER_SELECTION under the pinned role definitions. This still needs a fresh ID/hash check in the new effective admission receipt.

There is historical **selection** exposure. U's deployed-head C was selected using `HEAD_VALIDATION`, which is barred to the present scientific workers; source U itself is a fixed task-only e40 reference, not selected by INNER_SELECTION for this utility comparison. Hcal selected the Ucal H-GLOBAL-TEMP family by the lowest mean summed NLL on INNER_SELECTION (`results/pcrl_heldout_calibration_v1/SELECTION.json`, `ucal_star.rule`). Thus INNER_SELECTION is development data previously used to choose one reference calibration family. It is not an untouched population. This selection use does not equal direct calibration fitting on INNER_SELECTION; report both facts.

## R3 checks remaining before fit/replay claims

1. Verify new role/row/group hashes and overlap receipt against the admitted role-only exports, including common comparison, reference fit, head selection and calibration roles.
2. Check source checkpoint, preprocessing, deployed-head, class order and hcal calibrator byte identities; independently replay U0/Ucal forwards and permitted-role saved-vector equality to the predeclared tolerance.
3. Check the exact sealed candidate recipes, task-only selection score, full-role refit, calibration, freeze-before-INNER order and strict data allowlist. No R3 scientific worker may run before M0 seal and shared semaphore admission.
4. Independently compute accuracy, natural-log NLL and summed multiclass Brier and the inclusive caps for both seeds/tasks/references; confirm selection of the single candidate recipe and conditional branch. No runner scoring or selection import.
5. If a release stage is admitted, independently reconstruct recipient/raw-union complete views, control eligibility, transported reader identity, fixed reader envelope and paired inference. Separate exact replay from attacker refitting.

No reference-forward parity, new full-input utility, mask utility or privacy outcome is asserted here.
