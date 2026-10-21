# Engineering reviewer, pre-registration code review (saved verbatim with attribution)

Author: the engineering reviewer of campaign `pcrl_finite_routing_comparison_v1`. Its review ran 23:40-23:51Z on 2026-10-20 and
was returned at about 23:52Z, after the registration receipt (23:48:04Z) and after the T1 units had run (the reviewer notes this
itself). Saved verbatim (the harness's two-space indentation removed). The coordinator's dispositions are in
`roles/coordinator/ADJUDICATION_ENGINEERING.md`.

---

ENGINEERING REVIEW: frc/ for pcrl_finite_routing_comparison_v1 (engineering reviewer; review ran 23:40-23:51Z 2026-10-20)

Context: while I was reviewing, queue unit FRC_NEST_C128_V0_V1 (stage fixed_routing_nested) ended at 23:49:40Z. So T1 labels have already been read, and anything "before the first label read" is now "fix now". I edited no files, made no commits, spawned no helpers and read no labels. All five invented-data selftests PASS: frc.fa.route (17 s wall), frc.evaluate, frc.devreaders selftest_draws, frc.readerbudget, frc.test_receipt. I charged 60 CPU-s ("engineering_reviewer", 23:49:37Z; the ledger row is confirmed).

FINDINGS

1. FIX now (provenance of T1/T2) — frc/fa/route.py:602 source_task, 607 stage_repro, 646 stage_nest, 694 stage_twb.
   - Problem: no code compares the source inputs with this campaign's pins in SOURCE_ADMISSION.json['source_private_artefacts']. The unpinned inputs are the nrm TASK.pkl, the PRC_TWIN and PRC_TWIN_C128 MODEL/RELEASE files, and the GATE_V*.json records.
   - The checks that do run are internal only: the TASK.pkl sidecar, MODEL.json versus MODEL.pkl, and the release record. A consistently replaced source set would pass all of them.
   - The NEST_*.json record also stores no hash of TASK.pkl or of the GATE file it reproduced, so this cannot be checked after the fact.
   - Fix: add `_pinned(relpath)`, which compares sha256 with SOURCE_ADMISSION.json['source_private_artefacts'] and raises on a mismatch. Call it in source_task, in the GATE read inside stage_nest, and in every P._load_model / NRL.load source read. Write `task_cache_sha256` and `gate_record_sha256` into the NEST record.
   - For the unit that already ran: log both hashes now (the files are read-only) next to the pins.

2. FIX before P2 — route.py:956-960 (stage_configs) and 1034-1040 (stage_check).
   - Problem: SEXSIDE.pkl is read with only its own sidecar check. It is never compared with SELECTION_V<v>.json['sexside_cache_sha256'].
   - Scenario: `_cache_put` (line 849) renames any earlier SEXSIDE aside. A second P1 attempt therefore writes a new SEXSIDE before failing on the write-once SELECTION file. P2 then fits on teachers that the selection never saw, silently.
   - Fix: `assert _sha_bytes(SEXSIDE.pkl) == selrec['sexside_cache_sha256']` in stage_configs. In stage_check, compare with the SELECTION record of that version.

3. FIX before P1/T1 retries (CPU budget) — route.py:676-680 (stage_nest) and stage_paths (write at the end).
   - Problem: the write-once outputs are created only after the full nested computation.
   - stage_paths: if SELECTION_V<v>.json already exists, the stage refits the SEX-side teachers, runs every nested chain, and only then fails.
   - stage_nest: it writes the private npz with 'xb' before the JSON. A crash between the two writes makes every retry fail at the npz, after all the CPU is spent.
   - Fix: at stage start, refuse if any of the stage's output files already exist (NEST npz/json, SELECTION json).

4. FIX before E4 — route.py:913 (_choice, ablation branch).
   - Problem: `--ablation-lambda` takes any float. P.protected_book does not check grid membership (only protected_path does), so a typo or the other version's lambda silently fits ABL_LOC/ABL_PAIR at the wrong strength.
   - Fix: require `ablation_lambda in P.LAMBDA_GRID`, and require it to equal the `selected` lambda of a saved PRC_S* release of the same version. Record that config id in the ablation record.

5. FIX before P1/P2 — route.py:756-768 (protection_decision), used at stage_configs:964.
   - Problem: the decision checks construction and capacity_id, but not that decision['capacity'] == CAPACITIES[capacity_id], and not that decision['twin'] belongs to that capacity and construction.
   - stage_configs fits on `tb`'s real shape but writes decision['capacity'] into the payload and record. A malformed decision (e.g. capacity C128 with twin TWB_C64) would give a P2 fit on a different twin than P1 used, with no error.
   - Fix: assert `list(CAPACITIES[cid]) == decision['capacity']` and `(tb['CI'].shape[0], tb['CO'].shape[0]) == CAPACITIES[cid]`. Assert the twin's construction: TWB_* for B1; nrm PRC_TWIN/PRC_TWIN_C128 or TWA_C256 for A.

6. FIX — route.py:531-535 (cells_of).
   - Problem: it calls `P.route(P.embed(...), book)` without entry_embedding. For SHUF (task teachers only, no 'pi') it raises KeyError. If given an E that has pi, it would route by the person's own pi, unlike predict().
   - No current caller in frc.
   - Fix: `P.route(entry_embedding(P.embed(M['teachers'], Zc), M['book']), M['book'])`.

7. FIX before the reader grid — frc/readerextra.py:49 and :54.
   - Problem: SCHEMA is still 'NRM1-readerextra-v1' and CV_TAG is still 'NRM1_tablecv'. TABLE_CV therefore reuses NRM1's household-fold key: not a leak, but the grid is not fully "fresh FRC1", and outputs carry the NRM1 schema. The docstring at line 37 also says `python3 -m nrm.readerextra`.
   - Fix: set 'FRC1-readerextra-v1' / 'FRC1_tablecv', or register the reuse explicitly. This changes cache semantics only if extras are cached.

8. NOTE (vacuous check) — route.py:193-199, recorded at 259 and in selftest 1154.
   - Problem: `A_and_B1_count_the_same_people` compares `bincount(k0*2+y).reshape(KI,2).sum(1)` with `bincount(k0)`. That is an identity for any cells and labels, so it can never fail.
   - It is correct by construction (A and B1 receive the same k0, l0), but it is not evidence.
   - Fix: drop it from the record, or replace it with `array_sha(k0, l0)` recorded for both constructions.

9. NOTE (registration) — route.py:819.
   - The F4/nested-LOOKUP bootstrap seed SeedSequence([2026102100, v, 4]) is not in PROTOCOL (line 201 registers only B = 400 with scores fixed). Put it in the card before P1.
   - State in the F4 record that the SE is conditional on the fitted LOOKUP tables (table refit variance is excluded).

10. NOTE — route.py:1323 and 1327/1337.
   - `run('twb')` silently uses only configs[0]. Fix: refuse unless exactly one config is given.
   - Labels from `_real` are not checked for row alignment against `C.features('MODEL_FIT')`. Fix: assert equal `row_id` and `unit` before tables('A', ...).

11. NOTE (selftest coverage) — route.py:1171-1176.
   - The bitwise nested_c-versus-P.nested check covers only {PRC_TWIN, PRC}. HQ and ORD are compared only through fit_pipeline_c.
   - The real RL.save path is never exercised (fake_save), and SHUF_BR fitting was not exercised (CONTROL_DID_NOT_REACH_SPEND in the run).
   - Fix: add HQ_S2 and ORD entries to the nested_c-versus-source check (cheap).

12. FIX before closeout — frc/restore_check.py (not yet adapted). It needs:
   - (a) a re-hash of SOURCE_ADMISSION.json['source_private_artefacts']. Line 25 only re-hashes the ohn mirror;
   - (b) the frc fa_cache: SEXSIDE.pkl against its sidecar and SELECTION's sexside_cache_sha256, and fa_cache/nested/NEST_*.npz against NEST_*.json per_person_private_file_sha256;
   - (c) a re-hash of every FA MODEL.pkl against MODEL.json, not only the packaged ones re-run via predict;
   - (d) reader custody. frc has no readercustody module, but common.py:169 lists 'frc.readercustody'. So `representative_readers_bitwise` is {} and `all({})` passes vacuously. Port nrm.readercustody, or make all_ok fail when the grid ran with zero custody readers;
   - (e) confirmation that custody.py PRIVATE (line 36) contains every store that refs_of/'W'/'REF:' and devreaders RELEASE_ROOTS read (tsz/cts private stores are not listed).

13. NOTE — common.py:100 (REGISTRATION_FILES).
   - The receipt excludes cards/FA_FRC_v1.md, roles/coordinator/ADJUDICATION_PREREG.md, the two review reports, and any code hash (frc/fa/route.py, frc/rules.py).
   - card_info only checks "committed", not "at or before the receipt commit and on the remote". Record the card and code commit against the receipt in the log.

VERIFIED CORRECT
- tables():
  - A = P.decoders with α = 20 (bitwise source).
  - B0 = a copy of the codeword.
  - B1 = the OOF teacher mean on the same (k, l), clipn'd; unoccupied cells fall back to the prototype and are counted.
  - B0 and B1 read only E['qI'] and E['qO'] (no label path).
- emit/entry_embedding: the partition depends on book and statistics only. Constructions never change the partition (asserted per fold and in twb).
- nested_twin:
  - outer folds are household-disjoint (asserted);
  - tables come only from outer-training rows;
  - every person is held out exactly once;
  - held-out cells are identical across constructions.
- nest_summary:
  - per-person paired a − b, with accuracy treated as a loss;
  - household-clustered SE (P.cluster_se);
  - fold SE = SD(5 fold means)/√5, and se_used = max of the two;
  - spend over CONT follows P.spend's sign convention;
  - gate_reproduction compares means and spends to 1e-12 and stops on failure;
  - the GATE path matches the files.
- fit_pipeline_c / nested_c / apply_c:
  - construction A is bitwise equal to the source for twin/PRC/HQ/ORD;
  - SHUF runs the PRC_S2 parameters on its shuffled copy (pi from the donor bank of training OOF pi, qO0 = qO1 = the person's own qO);
  - at runtime the copy is rebuilt from features and is rule-consistent;
  - P.route reads only R's pi/qO0/qO1, so the person's own SEX statistics are never read for SHUF;
  - in nested folds the bank is the training part only, so no held-out person can be their own donor.
- stage_configs:
  - PRC and the ablations store 4 teachers; HQ, SHUF and ORD store task teachers only and run correctly without the SEX teachers;
  - the saved wire is predict() output;
  - B1 is recomputed on each config's own OOF partition;
  - aliases are never saved twice.
- brackets_for uses MODEL_FIT nested point spends in path order. The F4 bootstrap is paired (same W for twin and entry) and household-level (household multiplicity).
- stage_check uses OOF cells via entry_embedding, with SEX as an evaluation label only.
- Storage: write-once MODEL with hash and identity checks; predict refusals (keys, draw, feature columns); RL.save refuses overwrites and the 49th configuration and checks the card prefix.
- common.py equals nrm.common except for constants. The MODEL_SELECT refusal covers frc.fa/ctl/fs on the call chain; a queue unit, the registration and the card are required.
- jobqueue: 12 CPU-h, start 22:49:23Z, 8.0 h research and 8.5 h total, 2 slots, one thread, CODE_PACKAGES include frc and nrm.
- evaluate: `source` writes only under this campaign's runs/SOURCE and ledger; 'NRM:' is a read-only nrm.release.load; the reserve rule is nrm.rules.reserve_status (frc.rules does not redefine it).
- devreaders: FRC1 version and tags, offset 700, BOOT_SEED 2026102100, own cache, no source cache. readerextra's relative imports resolve to frc.devreaders, and it is in LOCK_FILES.
- deploy: FA resolves to frc.fa.release, which resolves to route.predict; CODE_MODULES include nrm.fa.prc and ohn.fam_a.
- The nrm.fa.prc functions frc uses never write. P.stage_twin runs only in the selftest, under temporary roots.

NOT REVIEWED (time box)
- frc/rules.py and test_rules.py (statistical reviewer's), readerbudget/admit/receipt/publish_build/banks internals, deploy.check refusal probes, devreaders beyond its constants and imports, ledger.py, and PROTOCOL beyond targeted greps.
