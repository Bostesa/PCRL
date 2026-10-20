# Quickstart: re-run the label-free assembly and the raw-field deployment

Everything below runs from the repository root of the research branch `research/pcrl-new-release-mechanisms-v1` (or a restored backup)
and needs the campaign's private tree (`.private/pcrl_new_release_mechanisms_v1/`, plus the admitted source copies). The private tree is
never published. Without it, the published package lets you read every record and the code, but not recompute releases or readers.

Interpreters used by the campaign: `python3` 3.14.3 (numpy 2.4.4, scikit-learn 1.8.0) for the FB2 and CTL fits, evaluation, claims and
deployment; the torch virtual environment (Python 3.13.7, numpy 2.4.2, scikit-learn 1.8.0, matplotlib 3.10.8) for the family-B v1 fits,
the reader grid, reader custody and figures. Each release records its fit interpreter in its RELEASE.json. Every
command that reads a label must run inside the campaign queue (`python3 -m nrm.jobqueue run --unit <ID> --stage <stage> --rss-gib <G>
-- <argv>`); the access layer refuses label reads outside a queue unit.

## 1. Re-run the claims assembly, the decisive rules, the tables and the figure (no label is read)

```bash
python3 -m nrm.claims run --plan results/pcrl_new_release_mechanisms_v1/claims/PLAN_FINAL.json --bank NRM1_CTL \
    --out /tmp/claims_check                      # compare with results/pcrl_new_release_mechanisms_v1/claims/FINAL_V2/
python3 -m nrm.decisive fb2p --claims results/pcrl_new_release_mechanisms_v1/claims/MAIN_V2/CLAIMS.json \
    --plan results/pcrl_new_release_mechanisms_v1/claims/PLAN_MAIN.json --bank NRM1_MAIN --out /tmp/fb2p.json
python3 -m nrm.decisive f1 --claims results/pcrl_new_release_mechanisms_v1/claims/FINAL_V2/CLAIMS.json --out /tmp/f1.json
python3 -m nrm.decisive mean --claims results/pcrl_new_release_mechanisms_v1/claims/FINAL_V2/CLAIMS.json --out /tmp/mean.json
python3 -m nrm.tables                            # rewrites TABLES.md / TABLES.json
<torch venv python> -m nrm.figures frontier --bank NRM1_CTL     # rewrites figures/FRONTIER.{svg,png,csv}
python3 -m nrm.test_rules; python3 -m nrm.decisive selftest; python3 -m nrm.evaluate selftest   # invented data only
# (nrm.test_rules rewrites roles/statistical_reviewer/SCREEN_CHECKS.json with its own CPU time; restore it with git checkout)
```

## 2. Rebuild and check the deployment packages (label-free)

```bash
python3 -m nrm.deploy store --name FB/FB2_SEX_S5 --version 0      # recompute from admitted features on all six roles; must equal
                                                                 # the saved, evaluated release bitwise
python3 -m nrm.deploy build --name FB/FB2_SEX_S5 --version 0 --out <package dir>
python3 -m nrm.deploy check --packages <package dir> ... --report <report.json>
```

`store` and `build` are write-once: on the campaign's own private tree they refuse to overwrite the existing stored release and
package, so rebuild into a fresh tree (for example a restored backup) or run `check` on the existing packages.

The campaign's packages are FB2_SEX_S5 (V0, V1), FB2_BASE (V0, V1), FB2_MINF_S5 (V0) and CTL_REM_0p50 (V0)
(`lanes/DEPLOY_V1.json`). `deploy/DEPLOYMENT_CHECK.json` records, for each package: bitwise parity with the stored release on the
DEV_REPORT raw inputs through the Python wrapper, the CSV wrapper and the encoded-feature function; equal decisions; and 19 refusals
(SEX, income and occupation target fields or columns, extra, missing or reordered fields at both entry points, hidden-output requests,
a randomized package without draw keys, a deterministic package given draw keys). No label is read.

## 3. Use a package

```python
import numpy as np
from nrm import deploy as D

rows = [{'AGEP': '40', 'WKHP': '40', 'COW': '1', 'SCHL': '21', 'MAR': '1', 'POBP': '012'}]  # invented example; PUMS codes as strings
out = D.release('<package dir of FB2_BASE_V0>', rows)            # deterministic package: no keys
# out['income'] = {'probability': (1, 2), 'decision': (1,)}; out['occupation'] = {'probability': (1, 5), 'decision': (1,)}
# A service routes out['income'] only to the income recipient and out['occupation'] only to the occupation recipient.

keys = {'role': 'SERVICE', 'row_id': np.array([123456789])}     # randomized package: per-request keys, supplied out of band
out = D.release('<package dir of FB2_SEX_S5_V0>', rows, keys=keys)
```

Exactly the six fields AGEP, WKHP, COW, SCHL, MAR, POBP, in that order, are accepted. SEX, task targets, identifiers and any other field
are refused. Categorical codes must be written as in the PUMS files (POBP as three digits, for example '012'; SCHL as two digits, for
example '01'): a code the encoder does not know is mapped to its `__OTHER__` column without an error ('12' for POBP is such a code).
This example was run on the campaign's packages (unit NRM_QUICKSTART_EXAMPLE_V1). Only `probability` and `decision` can be requested. A randomized package draws one instance per person from its private
release root and the keys; the root is never published, and repeated-query protection is not claimed.

## 4. Custody

```bash
python3 -m nrm.custody backup <destination parent>       # same-device backup and a restore-and-check from the backup alone
python3 -m nrm.custody offdevice <backup dir> <dest>     # copy one backup to an already attached drive and repeat the check there
```

The custody tool never mounts, unmounts or forces a drive. A check right after copying to a still-mounted drive is warm, not cold.
`custody/BACKUP_VERIFICATION.json` records what was done and what is still pending.
