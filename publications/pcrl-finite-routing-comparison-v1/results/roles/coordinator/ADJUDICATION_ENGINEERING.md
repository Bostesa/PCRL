# Coordinator dispositions of the engineering review (2026-10-20, written from 23:55Z, `date -u`)

Input: `roles/engineering_review/REVIEW_PREREGISTRATION_CODE.md` (13 findings; returned about 23:52Z, after the T1 units had
run). Fixes are in `frc/fa/route.py` and `frc/readerextra.py`, committed before P1 (the next fit). The family selftest stays
45/45 (one vacuous check removed, one refusal check added); the readerextra selftest passes.

| # | Disposition |
|---|---|
| 1 source pins | **Fixed.** `pinned()` compares every read-only source artefact (TASK.pkl, the twins' MODEL.pkl and RELEASE.npz, the GATE records) with SOURCE_ADMISSION.json and raises on a mismatch; called in `source_task`, `source_model`, `_source_release`, `stage_repro`, `stage_nest`, `stage_twb` and `_twin_book_for`; NEST records now carry `task_cache_sha256` and `gate_record_sha256`. **For the units that already ran (T0, T1, T2):** at 23:55:16Z every pinned artefact's current SHA-256 equals its admission pin (recorded at 23:19Z, before T1 ran at 23:49Z), and the T0 and T2 records' `task_cache_sha256` equal the pins; the hashes are listed in `roles/coordinator/SOURCE_PIN_CHECK.json`. Nothing writes those files. |
| 2 SEXSIDE hash | **Fixed** (before P2): `stage_configs` and `stage_check` require SEXSIDE.pkl to equal the selection record's `sexside_cache_sha256`. |
| 3 refuse before computing | **Fixed**: `stage_nest` and `stage_paths` refuse at start when their write-once outputs exist. |
| 4 ablation lambda | **Fixed** (before E4): the lambda must be on the grid and equal the selected lambda of a saved PRC configuration of the same version; the ablation record names it (`ablation_of`). |
| 5 decision consistency | **Fixed**: `protection_decision` checks the capacity against its id and the twin against the capacity and construction; `stage_configs` checks the twin book's shape. |
| 6 cells_of | **Fixed** (it now reads `entry_embedding`); no caller had used it. |
| 7 readerextra constants | **Fixed** before the grid: `FRC1-readerextra-v1`, `FRC1_tablecv`. |
| 8 vacuous identity | **Accepted.** `A_and_B1_count_the_same_people` is an identity and is not evidence. It stays in the committed T1 records (append-only) with this note; future records carry `oof_partition_sha256` instead, and the selftest no longer counts it. |
| 9 F4 seed not registered | **Fixed**: card addendum A1 (FA_FRC_v1.md, committed before P1) registers B = 400, the seed SeedSequence([2026102100, v, 4]) and that the SE is conditional on the fitted LOOKUP tables; the F4 record states the same. |
| 10 twb with several configs; label alignment | **Fixed**: `twb` refuses anything but exactly one configuration; decoder label reads must match the admitted MODEL_FIT row order. |
| 11 selftest coverage | **Fixed**: the bitwise nested_c-versus-source check now includes HQ and ORD entries. The real `frc.release.save` path is exercised by every real unit (its own checks run there); SHUF_BR is exercised in the selftest when the bracket exists (it did not in that run; disclosed). |
| 12 restore_check | **Accepted, to be done before closeout** (source pins, fa_cache SEXSIDE and nested files, every FA MODEL.pkl re-hash, reader custody: the module `frc/readercustody.py` was ported at 23:44Z, during the review; `all_ok` will fail on zero custody readers). |
| 13 receipt scope | **Accepted as a note.** The receipt gates the eight registration files. The card and the code at research commit 98ede167e are also in the receipt tree 7fed2ca44 (published, not gated); the governing card FA_FRC_v1.md is blob-identical between the two (checked at commit time and logged). The addendum A1 is a later, logged card change. |
