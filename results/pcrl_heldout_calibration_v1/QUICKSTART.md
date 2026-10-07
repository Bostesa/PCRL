# Quickstart — hcal (held-out, shared calibration of frozen releases)

Every command runs from the worktree root with one BLAS/torch thread. The private store is named by
`PCRL_HCAL_PRIVATE_CACHE`, which defaults to `<HOME>/PCRL_eval_cache_private/hcal_v1`. Heavy commands run under the
shared two-slot semaphore. `<PY>` is the project virtual-environment Python.

```sh
export PCRL_HCAL_PRIVATE_CACHE=<PRIVATE_CACHE>/hcal_v1
```

## Tests (synthetic only; no Adult row)

```sh
OMP_NUM_THREADS=1 PYTHONPATH=. <PY> -m hcal.sema --label A:tests -- \
    env OMP_NUM_THREADS=1 PYTHONPATH=. <PY> -m pytest -q tests/pcrl_heldout_calibration_v1
```

## Staged pipeline

Each stage refuses unless its lock (and any later amendment) is committed, pushed and byte-identical on origin, and
every loaded study module is locked.

```sh
<PRIVATE_CACHE>/hcal_v1/run/work.sh SOURCE_ADMISSION_LOCK admit
<PRIVATE_CACHE>/hcal_v1/run/work.sh ENGINEERING_LOCK engineering        # -> ENGINEERING_RESULT.json
<PRIVATE_CACHE>/hcal_v1/run/work.sh SCIENCE_LOCK calibrate 0/2          # and 1/2
<PRIVATE_CACHE>/hcal_v1/run/work.sh SCIENCE_LOCK utility                # Ucal*, gates, audit plan
<PRIVATE_CACHE>/hcal_v1/run/work.sh SCIENCE_LOCK audit 0/2              # and 1/2: fresh banks + common records
<PRIVATE_CACHE>/hcal_v1/run/work.sh SCIENCE_LOCK compose                # continuous-U composed record
<PRIVATE_CACHE>/hcal_v1/run/work.sh SCIENCE_LOCK controls 0/2           # and 1/2
<PRIVATE_CACHE>/hcal_v1/run/work.sh SCIENCE_LOCK select
<PRIVATE_CACHE>/hcal_v1/run/work.sh SCIENCE_LOCK replay
PYTHONPATH=. <PY> -m hcal.eval_lock write                               # after E's independent inner replay
OMP_NUM_THREADS=1 PYTHONPATH=. <PY> -m hcal.sema --label A:assess -- env OMP_NUM_THREADS=1 PYTHONPATH=. \
    PCRL_HCAL_PRIVATE_CACHE=$PCRL_HCAL_PRIVATE_CACHE <PY> -m hcal.assess \
    --evaluation-lock results/pcrl_heldout_calibration_v1/EVALUATION_LOCK.json --shard 0/2   # and 1/2
OMP_NUM_THREADS=1 PYTHONPATH=. <PY> -m hcal.infer \
    --evaluation-lock results/pcrl_heldout_calibration_v1/EVALUATION_LOCK.json
PYTHONPATH=. <PY> -m hcal.report inner && PYTHONPATH=. <PY> -m hcal.report assess
```

## Deployment (research only; not authorisation to deploy externally)

```sh
OMP_NUM_THREADS=1 PYTHONPATH=. <PY> -m hcal.deploy \
    --unit <PRIVATE_CACHE>/hcal_v1/admitted/rel__s<k>__U \
    --policy <PRIVATE_CACHE>/hcal_v1/admitted/units/<pol__ | new__ unit>/policy.json \
    [--decoder <decoder.json: hcal.CalibratedDecoderPair or lra.DecoderPair>] [--decoder-sha256 <hex>] \
    --X <83-column input npz> --schema <PRIVATE_CACHE>/hcal_v1/admitted/inputs/schema.json --out <release.npz>
```

The output holds only `tokens_i`, `probs_i` and `decision_i`. Every binding, schema or export violation exits with
code 2. The packaged release and its evidence status are in MODEL_MANIFEST.json and DEPLOYMENT_RECEIPT.json.

## Independent verification (role E; imports no hcal calculation module)

```sh
<PY> -P hcal/sema.py --label E:inner -- env OMP_NUM_THREADS=1 PCRL_HCAL_PRIVATE_CACHE=$PCRL_HCAL_PRIVATE_CACHE \
    <PY> results/pcrl_heldout_calibration_v1/verification/verify_inner.py
```
