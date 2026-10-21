# Quickstart (campaign `pcrl_finite_routing_comparison_v1`, code `frc/`)

Exploratory development code for public-use ACS 2019 Florida records. Private arrays, fitted models, reader artifacts and the
admitted data copies are never published; the commands below run only inside the owner's research worktree.

## 1. Use a deployment package (six raw fields in, two recipients' outputs out)

```python
from frc import deploy as DP
pkg = '.private/pcrl_finite_routing_comparison_v1/deploy/<NAME>_V<v>'
rows = [{'AGEP': 42, 'WKHP': 40, 'COW': 1, 'SCHL': 21, 'MAR': 1, 'POBP': 12}]
out = DP.release(pkg, rows)                       # {'income': {'probability', 'decision'}, 'occupation': {...}}
```

- Every row must carry exactly AGEP, WKHP, COW, SCHL, MAR, POBP, in that order. SEX, task targets, identifiers and any other field
  are refused; only `probability` and `decision` may be requested (`DP.release_csv(pkg, path)` is the CSV entry point).
- `DP.underlying(pkg, X, names)` is the deployment function on the admitted 145-column encoding; it refuses any other column list
  and non-finite or mis-shaped input.
- The packages are deterministic: per-request keys and draws other than 0 are refused. No routing index or cell identity is
  exported; internal cells emitting identical vectors are one observable value.
- Which packages exist and what they are: `deploy/DEPLOYMENT_CHECK.json`, START_HERE.md.

## 2. Re-run the checks

```bash
python3 -m frc.fa.route selftest          # family module, invented data (no labels)
python3 -m frc.test_rules                 # FRC1 rules, invented data
python3 -m frc.evaluate selftest
python3 -m frc.readerextra selftest
python3 -m frc.deploy check --packages <pkg dirs> --report <out.json>      # bitwise parity on DEV_REPORT raw inputs + refusals
python3 -m frc.custody backup <parent dir>                                   # same-device backup + restore-and-check from the backup alone
```

## 3. Re-run the study (owner's worktree; every numerical job through the queue)

```bash
python3 -m frc.jobqueue run --unit <ID> --stage <stage> --rss-gib <GiB> -- python3 -m frc.fa.route <stage> --version <v> --card <card>
```

Stages, in order: `repro` (T0), `nest --capacity C64|C128` (T1), `twb --config TWB_C64|TWB_C128` (T2), `frc.evaluate release|source`
(T3), `frc.decide qual`, `paths` (P1), `cfg` (P2), `unt` (revision R1), `frc.evaluate release` (E1), `frc.devreaders run --spec
grid_specs/<bank>.json --seed v` with the torch venv interpreter (E2 / E3), `frc.claims run --plan claims/PLAN_<id>.json --bank
FRC1_MAIN --out claims/<id>`, `frc.claims stability`, `frc.tables`. The lane files under `lanes/` hold the exact unit lists.
Labels are readable only after the registration receipt and only through `frc.common.load`.

Added after the registered grid (all label-free, from saved records):

```bash
python3 -m frc.fa.probe_strength <v>                                    # through the queue: the strength probe (fa/PROBE_STRENGTH_V<v>.json)
python3 -m frc.claims stability --main FRC1_MAIN --rob FRC1_ROB --out results/pcrl_finite_routing_comparison_v1/claims/STABILITY
python3 -m frc.refitcheck --rob FRC1_ROB --out results/pcrl_finite_routing_comparison_v1/claims/STABILITY/REFIT_EQUALITIES.json
<torch venv python> -m frc.figures frontier --bank FRC1_MAIN            # figures/FRONTIER.{png,svg,csv}
python3 -m frc.readercustody run --bank FRC1_MAIN --seed v --rep R1 --k 0 --item PRC_S2:pair:WINNER ...   # through the queue
python3 -m frc.lane results/pcrl_finite_routing_comparison_v1/lanes/DEPLOY_V1.json                      # deployment store / build / check
python3 -m frc.ledger && python3 -m frc.closeout                         # ledgers, resource and access receipts
```

Records are write-once: a rerun into an existing output path is refused; use a new output name.

## 4. Deployment packages built by this campaign

`FA/PRC_S2` (the strongest feasible nonzero release), `FA/TWB_C64` (its task-only counterpart, the repaired twin) and `FA/HQ_S2`
(the decisive matched control), each in versions 0 and 1, under `.private/pcrl_finite_routing_comparison_v1/deploy/<NAME>_V<v>`.
`deploy/DEPLOYMENT_CHECK.json` shows them bitwise equal to the stored, evaluated wires on DEV_REPORT raw inputs, through the API, the CSV
path and the underlying function. All 19 refusal probes refused in each package, and no label was read. DEV_REPORT features are used
label-free for this parity check only, as registered (PROTOCOL section 2).
