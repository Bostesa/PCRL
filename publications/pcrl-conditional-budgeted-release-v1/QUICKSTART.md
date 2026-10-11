# Runnable campaign

## Exact environment used

Python 3.13.7; numpy 2.4.2, scipy 1.17.1, scikit-learn 1.8.0, torch 2.10.0, joblib 1.5.3, psutil 7.2.2, concept-erasure 0.2.4 and pytest 9.0.2. The export's source/requirements.txt pins dependencies. Numerical workers explicitly set OMP/BLAS/MKL/VECLIB/NUMEXPR/BLIS/RAYON/POLARS/TORCH thread counts to one.

## Public fixture command actually run

From the export's `source` directory:

```sh
python -m pytest -q conditional_release conditional_release_independent
```

The original exported tree passed all 112 tests with no exclusion or skip, under the shared numerical queue. The final export receipt binds its final source bytes. Fixed synthetic fixture JSONs and inherited helper modules are included. These fixtures test correctness; they do not reproduce private Adult fits.

## Exact executed scientific commands

Registered workers ran under this single queue command:

```sh
python -m conditional_release.host_queue --count 1
```

Actual unit specs/config hashes and measured resources are preserved in the private versioned snapshot; public UNIT_MANIFEST.json records their identities/status. Policy workers, controls, reader fitting, reports, inference and independent replays refuse launches without the durable queue slot. The configured admission/science/stage locks must be restored together with the allowed inputs; running a worker directly is not an authorized shortcut.

Public aggregate closeout commands actually executed:

```sh
python -m conditional_release.closeout_summary --root results/pcrl_conditional_budgeted_release_v1
python -m conditional_release.closeout_aggregate --root results/pcrl_conditional_budgeted_release_v1 --queue .private/conditional_release/compute/state.json
```

## Raw83 deployment

See [r1/DEPLOYMENT_QUICKSTART.md](r1/DEPLOYMENT_QUICKSTART.md) for the actual command shape and refusal contract. The deployment module consumes the original ordered 83 columns, frozen source pins, a fitted policy package, private provider keys/root and persistent cache. It receives no task or SEX labels. The actual registered CLI/API checks cover P0, historical joint/sequential references, G/C/N/W, utility-objective, local/product/deterministic and both sequential controls on permitted real fitting inputs. The additional twenty-package label-independence unit ran with all role/label records removed and original nonruntime source paths refused. That is verification of these unchanged Python loaders, not an OS security sandbox.

Each person gets one cached draw; repeats and process restarts reproduce it. Duplicate records share a provider key. Changed raw input under an existing key is refused. Keys, roots, private source states, action positions and fallback flags never appear in recipient wires. This is a score-release service; the original raw inputs stay with the provider.

## Backup-only execution

The private restore index supplies exact versioned snapshot aliases and hash pins. Source models, fitted policies/readers and person arrays are private. `conditional_release.custody` reconstructs the role-safe source manifest; `conditional_release.restore_deployment_worker` and independent restore readers reject original paths outside the restored tree. Actual same-device/off-device and representative restore coverage is recorded in RESTORE_INDEX.json/BACKUP_VERIFICATION.json and the independent verification. Public code/fixtures are runnable without private data; raw deployment/scientific replay additionally require the admitted private snapshot.
