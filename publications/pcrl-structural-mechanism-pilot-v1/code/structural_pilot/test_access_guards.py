"""Access-guard checks that read no truth content: DEV_REPORT truth is refused without this campaign's committed
REPORTING_LOCK; the source receipt's DEV_REPORT truth paths exist (existence only: no hashing, no loading); sealed and
non-admitted fields are refused by load_role; no access log was written by these checks."""
import json
from pathlib import Path

from structural_pilot import common as C


def main():
    out = {}
    try:
        C.dev_report_truth('guard test: must be refused')
        out['truth_refused_without_lock'] = False
    except PermissionError:
        out['truth_refused_without_lock'] = True
    files = json.loads((C.SRC_PRIV / 'extraction' / 'EXTRACTION_RECEIPT.json').read_text())['files']['DEV_REPORT']
    out['truth_paths_exist'] = {k: (C.SRC_PRIV / Path(files[k]['path']).relative_to('.private/acs_replication')).is_file() for k in ('income', 'occupation', 'sex')}
    out['truth_dir'] = sorted({str(Path(files[k]['path']).parent.name) for k in ('income', 'occupation', 'sex')})
    for role, wanted in (('SEALED_ASSESSMENT', ('features',)), ('DEV_REPORT', ('features', 'sex')), ('DEV_REPORT', ('features', 'income')),
                         ('ATTACK_FIT', ('features', 'income')), ('MODEL_SELECT', ('features', 'weights'))):
        try:
            C.load_role(role, wanted); out[f'refused|{role}|{"+".join(wanted)}'] = False
        except PermissionError:
            out[f'refused|{role}|{"+".join(wanted)}'] = True
    out['campaign_access_log_absent'] = not C.ACCESS_LOG.exists()
    out['all_ok'] = (out['truth_refused_without_lock'] and all(out['truth_paths_exist'].values()) and out['campaign_access_log_absent']
                     and all(v for k, v in out.items() if k.startswith('refused|')))
    print(json.dumps(out, indent=1))
    assert out['all_ok']


if __name__ == '__main__':
    main()
