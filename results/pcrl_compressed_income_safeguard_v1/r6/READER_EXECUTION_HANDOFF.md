# Fresh reader execution plan

This prospective plan binds 42 bank jobs: 34 scientific banks and eight epoch-zero continuation banks. It requires 1,332 fresh main reader fits and 24 logistic control fits. Counts are planned work, not completed reader evidence.

| Bank category | Jobs | Main fits per job | Control fits per job | Exact aliases |
| --- | ---: | ---: | ---: | --- |
| RAW_HYBRID | 2 | 54 | 0 | None; canonical bank first |
| Positive continuation checkpoint | 24 | 36 | 0 | Income from new same-seed RAW_HYBRID |
| Epoch-zero continuation checkpoint | 8 | 0 | 0 | All three views from new same-seed RAW_HYBRID |
| RAW_FULL | 2 | 36 | 6 | Occupation from new same-seed RAW_HYBRID |
| RAW_PROJ | 2 | 36 | 0 | Income from new same-seed RAW_HYBRID |
| U0_FULL | 2 | 54 | 6 | None |
| U0_PROJ | 2 | 54 | 0 | None |

Every bank contains all 18 declared readers per direct view. The pair envelope contains its 18 direct readers and 36 exact local ignore-other aliases, giving 54 candidate orientations selected using ATT validation. INNER forwards are label free. FIT selection forwards use a full canonical batch forward followed by the inherited fold-two slice. No old reader bank is reused.

ROOT runs the private R6 metadata helper with the queue's Python and `--phase canonical`. This writes two configurations and queue specifications with no dependencies; it never enqueues work. ROOT admits these two jobs. After both queue units are done and their new freeze receipts exist, ROOT runs `--phase dependent` to write the remaining 40 configurations and specifications. Their aliases require the new canonical receipts, all 54 saved model pins, ATT and INNER release pins, stored score pins, and FIT selection release and score pins. Numerical pins are copied from queued freeze metadata; the helper never opens or hashes numerical artifacts.

The queued worker verifies literal ATT and INNER reader-contract equality, source people and inherited split, replays saved models and frozen orientations, and compares ATT validation recovery before accepting an alias. Thus an epoch-zero checkpoint hash differing from its source hash does not establish reuse by itself. Source model, geometry, role features, protected ATT labels, code, recipe, science lock, wire history, preprocessing, source seed and model seed are bound separately.

`READER_PLAN.json` and `READER_PACKAGE_INDEX.csv` give safe bank identities and planned counts. Canonical generated specifications were checked by metadata inspection: 42 unique planned identities, eight zero-fit epoch-zero jobs, totals 1,332 plus 24, and two dependency-free canonical configurations with both label-free forward flags enabled. Actual fit accounting comes from each queued `FROZEN_COMPLETE` receipt and must be reconciled against this plan after completion.

The active metadata helper now generates V2 unit IDs and filenames. `GENERATED_CANONICAL_V2.json` lists the two current source-bank specifications; dependent generation uses the same V2 identity scheme. Original V1 metadata is preserved. `READER_PLAN_V2.json` and `READER_PACKAGE_INDEX_V2.csv` record the revised executable identities. Amendment D supplies the corrected queue scheduler source hash and active amendment hash. Bindings retain the original science-lock code registry and separately identify the engineering amendment and effective scheduler. This queue guard repair changes no scientific reader recipe or fit count.
