# Designer B, wiring status at 15:15Z (saved verbatim with attribution)

Author: Designer B (family B) of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at about 15:16Z on 2026-10-20.
Saved verbatim; nothing below was edited.

---

DESIGNER B (15:15Z): coordinator changes 1-5 are done. Code is committed and the selftests pass. As instructed, I have not run any real fit, read any label, or spawned any sub-agent.

1. Allocation: 16 configurations (8 per version). FB_SEX_S1 and FB_PCA_S1 are gone from the code.
   - Per version: FB_TWIN, FB_SEX_S2, FB_SEX_S3, FB_PCA_S2, FB_PCA_S3, FB_SEX_S2_ID, FB_SEX_S2_MEAN, FB_POSTHOC_S2.
   - 5 learners per version (none, sex:S2, sex:S3, pca:S2, pca:S3).
   - The S4 extension (R2) is listed as needing its own recorded revision.

2. No coupling claim. This is stated in the card (section 1, "Noise coupling"), in the code docstring, and as a `coupling_claim` field in every release record. The SEX and PCA paths share the same coupled rank-1 draw, so their comparison only isolates targeting.

3. Deployment interface: new module `nrm/fb/release.py`. `nrm.deploy.family_module('FB')` resolves to it, and `CODE_MODULES` pins the code of `nrm.fb.channel`, `nrm.fb.learner` and `nrm.fb.release`.
   - `predict(config_id, version, X, feature_names, draw=0, keys=None)`:
     - Refuses missing or short keys on randomized configurations, and keys on deterministic ones (PermissionError).
     - Refuses any draw other than 0 on deterministic configurations, reordered names, wrong width, non-finite input and dropped configuration ids (ValueError).
     - Randomized configurations return instance `draw` from the keyed one-draw-per-person emission.
     - Fit-time emission and deployment share one per-instance code path, so they are bitwise identical.
   - `model_paths(config_id, version)` lists the files under `runs/FB/MODELS/V<v>/`: `LEARNER_<learner>.pt`, `CHANNEL_<config>.json` (frozen geometry, temperatures, s, cap, feature names) and, for randomized configurations only, `RELEASE_ROOT.bin` (private, mode 0600).
   - All are under `C.RUNS`, as `deploy.build` requires. They are write-once and are written before the release is saved. Every RELEASE.json records their SHA-256 (`model_files_sha256`).
   - Selftest: `predict` with keys={'role': R, 'row_id': row ids of R} and draw=0 reproduces RELEASE draw 0 bitwise for all 8 configurations × 6 roles. Instance 3 also reproduces. All 8 refusal probes refuse.

4. Card text: prepared at `<repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/results/pcrl_new_release_mechanisms_v1/cards/FB_FSC_v1.md`.
   - It follows `cards/README.md` sections 1-10 and includes changes 1-2.
   - It also covers the frozen seed handling, R = 8, the reader-instance mapping (cell (v, j, k) uses instance r = j; MC replicates on instances j+2 and j+4 for SEX_S2 and PCA_S2), predictions P1-P6, and the revision plan.
   - It is **UNCOMMITTED** until your GO with the reviews' changes.

5. Wiring:
   - The fit reads `nrm.common.load('MODEL_FIT', (features, income, occupation, sex), 'fit', purpose, card=<card>)` and `nrm.common.load('CALIBRATION', (features, income, occupation), 'calibrate', purpose, card=<card>)`, then saves through `nrm.release.save`.
   - It runs on one thread.
   - Queue command: `python3 -m nrm.jobqueue run --unit FB_FIT_V<v> --stage fit --rss-gib 2 -- <repository>/.venv/bin/python -m nrm.fb.learner fit --version <v> --card results/pcrl_new_release_mechanisms_v1/cards/FB_FSC_v1.md`

Commits (my paths only, plain messages):
- `ef0bc1ddf`: wave 1.
- `39414150e`: changes 1, 3 and 5. Files: `nrm/fb/__init__.py`, `nrm/fb/channel.py`, `nrm/fb/learner.py`, `nrm/fb/release.py`.

Selftests re-run at 15:13-15:14Z, invented arrays only:
- `python3 -m nrm.fb.channel selftest`: ok, 22/22 (exit 0; also exit 0 under the torch venv).
- `<repository>/.venv/bin/python -m nrm.fb.learner selftest`: ok, 18/18.
  - New checks: allocation_16, predict_reproduces_draw0_every_config_every_role, predict_reproduces_instance_3, predict_refusals, model_paths_exist_and_key_only_if_randomized, model_file_hashes_in_record.
  - Refusal probes, all true: randomized_without_keys, deterministic_with_keys, deterministic_draw_1, short_keys, reordered_names, non_finite, wrong_width, dropped_config_S1.

CPU charged out of queue under label designer_b: 40 s at wave 1 and 25 s now, both upper estimates. The ledger shows 2 entries.

Sub-agents: still the same 2 read-only Explore agents (about 14:37Z to about 14:44-14:45Z, both finished); nothing new spawned.

Waiting for GO: the reviews' required changes, then the card commit, then the receipt-verified fit.
