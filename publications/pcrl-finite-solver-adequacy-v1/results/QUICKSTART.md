# Quickstart (campaign `pcrl_finite_solver_adequacy_v1`, code `fsa/`)

Exploratory development code for public-use ACS 2019 Florida records. Private arrays, cached teacher statistics, per-person files
and saved books are never published. The commands below run only inside the owner's research worktree.

## 1. The solver and its checks (invented data; no labels)

```bash
python3 -m fsa.fa.solver           # SV1 selftest: cost terms equal nrm route, interval partitions, incremental objective,
                                   # traced original equals lloyd / protected_path, monotone and rule-consistent acceptance, HQ books
python3 -m fsa.test_rules          # FSA1 rules (statistical reviewer), invented data
```

`fsa.test_rules` also reads the registered RULES.json files of `pcrl_new_release_mechanisms_v1` and
`pcrl_occupation_headroom_overnight_v1` (under `results/<campaign>/roles/statistical_reviewer/`). A copy that holds only `code/fsa`
fails 4 of its 15 tests with FileNotFoundError. The code also imports `nrm`, `frc` and `mechanism_design`, which are published with
their own campaigns.

Use the revised solver on any embedding E = {qI, qO, pi, qO0, qO1} (the objective and routing family of `nrm.fa.prc`, unchanged):

```python
from nrm.fa import prc as P
from fsa.fa import solver as S
rp = P.route_params('PRC_S2')                 # or 'HQ_S2' for the task-score-only control
tb, k0, l0 = P.twin_book(E, 16, 64, 0, rp)
trace = {}
path = S.chain(E, tb, k0, l0, [8.0], rp, solver='SV1', trace=trace)     # 'original', 'original_traced', 'SV1_first'
book, k, l = path[8.0]                        # deployable: P.route(E, book) reproduces (k, l)
```

SV1 improves the stated objective. That is not a usefulness or protection guarantee: on two invented laws the objective's optimum
violates the utility limits.

## 2. The campaign's stages (owner's worktree; every numerical job through the queue; labels only after the registration receipt)

The lane files `lanes/SOLVER_V0.json` and `lanes/SOLVER_V1.json` record what ran: the replay, audit, source and nested stages for
each version. Running them again does nothing. `fsa.lane` skips every unit id that the ledger already marks done (`fsa/lane.py`),
and the four stages write fixed, write-once output paths. A rerun therefore needs a new lane file with new unit ids, plus new output
paths in `fsa/fa/solve.py`, which is a code change.

The decision builder and the tables can be rebuilt into new paths:

```bash
python3 -m fsa.jobqueue run --unit <new ID> --stage decide --rss-gib 3 -- python3 -m fsa.fa.advance route_s --out <new path>
python3 -m fsa.jobqueue run --unit <new ID> --stage decide --rss-gib 1 -- python3 -m fsa.fa.advance decide \
    --e results/pcrl_finite_solver_adequacy_v1/roles/exact_reference/route_e_input_SV1.json --s <route S input> --out <new path>
python3 -m fsa.report              # rewrites tables/ in place from the records (label-free); the committed tables reproduce exactly
```

- The replay reruns the previous campaign's `frc.fa.route.nested_c` unchanged. The exact-reference code is
  `results/pcrl_finite_solver_adequacy_v1/roles/exact_reference/code/fsa_exact.py` (see its LAWS.md).

## 3. Custody

```bash
python3 -m fsa.custody backup <parent dir>      # same-device backup + restore-and-check from the backup alone (fsa.restore_check)
python3 -m fsa.custody offdevice <backup dir> <external drive>   # when attached
```

The restore check re-hashes the sources and this campaign's private outputs, re-routes every saved book bitwise from the backup,
recomputes the fitted objectives and replay means, runs the solver selftest, and re-derives the route S and advancement labels.

## 4. Deployment

No deployment package was built: the conditional pilot did not run, so this campaign has no new release. The previous campaign's
packages (`publications/pcrl-finite-routing-comparison-v1`) are unchanged.
