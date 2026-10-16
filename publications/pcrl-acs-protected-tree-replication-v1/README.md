# ACS development replication of the protected-tree recipe (Florida ACS 2019)

**Outcome:** `RECIPE_EFFECT_NOT_ESTABLISHED`, `NOT_SUPERIOR_TO_ALL_ELIGIBLE` (complete), `SUBGROUP_UNRESOLVED`.

Relative to matched continued task fitting, the routing-preserving local leaf-score recipe passed all 24 utility clauses at the point, a gate with no error control. It lowered each recipient's own SEX recovery by 0.014–0.018 AUC for income and 0.061–0.067 for occupation.

The linked coalition of the two recipients' outputs still carried most of the teacher's SEX signal. Its recovery fell only 0.007 and 0.0175 AUC (seed 0 / seed 1), below the pre-registered 0.020 margin: an established shortfall on seed 0, unresolved on seed 1.

An eligible neural adversarial control had lower coalition recovery at the point on both seeds, established on seed 0. Subgroup usefulness is unresolved: for men on both seeds, and for women on seed 0.

This is development evidence for a recipe replication (new models, inputs and tasks). It is not confirmation of the frozen Adult packages. All intervals are conditional on two fitted seed pairs and a frozen reader slate.

The sealed 20% of households (SEALED_ASSESSMENT) stays closed. Nothing about it was written or used beyond counts and a key-set hash. The extraction's eligibility pass did parse its records in memory (disclosure D2).

Start with `results/START_HERE.md`.

## Layout

| Path | Contents |
|---|---|
| `prefit/` | Public receipts pushed **before** outcomes were seen: `85b2a41` (SCIENCE_LOCK, before any ACS label) and `7d23e0e` (training, selection and reporting locks, before DEV_REPORT truth was opened). These files are left unchanged. |
| `results/` | Final results, decision documents, tables, figures, independent verification, claims review, deployment, custody and resource records. Amendment A1 and the post-report disclosures D1–D3 are in `results/LOCK_AMENDMENTS.md`. This is a copy of `results/pcrl_acs_protected_tree_replication_v1/` at research commit `80c8a633e`. |
| `code/acs_replication/` | Campaign code at research commit `80c8a633e`. |
| `code/acs_replication_independent/` | The independent verifier's separate code. |
| `code/{jcv,oar,task_conditioned_representation,task_headroom}/` | The repository modules the campaign imports, copied unchanged. In `jcv/finalize.py`, `finalize_neural` imports `stored_model_eval`, which is not included; this campaign never calls that function. |

## Sanitization

Exactly one change was made relative to the research commit. In `code/acs_replication/provenance.py`, the literal home-directory prefix of the provenance search scope was replaced by `<HOME>` (5 lines). Every other file under `results/` and `code/` is byte-identical to its counterpart at research commit `80c8a633e`. This README exists only in the publication, and `prefit/` is unchanged from receipt `7d23e0e`.

The following are not published anywhere, by design:
- ACS person records and role exports;
- task or SEX labels;
- per-person predictions;
- fitted model weights;
- the official FAGTB and FARE source code (no licence file; run and cited only);
- anything about SEALED_ASSESSMENT beyond its counts and key-set hash.

The ACS microdata themselves are public from the Census Bureau; `results/POPULATION_AND_SCHEMA.md` gives the URL and hashes.

## Running the code

Use Python 3.13 with numpy, scipy, scikit-learn, torch, CatBoost 1.2.8, openpyxl and matplotlib, and set `PYTHONPATH=code`. FARE needs the pinned official environment described in `code/oar/fare_official.py`.

`results/QUICKSTART.md` covers deployment, re-deriving the tables and end-to-end reproduction of the development roles. Fitted packages are private, so deployment needs a fresh fit.

The synthetic tests (`test_pipeline_synthetic`, `test_report_synthetic` and `test_select_readers_synthetic` in `code/acs_replication/`) need no ACS data.
- They passed from an isolated copy of this code tree, with no repository checkout on the path (queue unit `PUB_CODE_STANDALONE_TESTS_V1`, code at research commit `bd398ef6c`).
- The only later code change is `custody.py`, which they do not exercise.
- `test_fagtb.py` additionally needs the private TensorFlow parity environment, which is not published.
