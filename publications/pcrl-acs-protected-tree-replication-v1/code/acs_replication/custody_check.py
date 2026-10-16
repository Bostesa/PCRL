"""Representative restore from the snapshot ALONE (label-free): redeploy a restored LOCAL_TREE package on restored raw
inputs and compare with the stored released probabilities; reload a restored winning reader and re-score restored
DEV_REPORT release tables, comparing with the stored scores. Uses only files under the restore directory."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np


def main(restore_dir, out_path):
    rd = Path(restore_dir)
    priv = rd / 'private'; code = rd / 'code'
    sys.path.insert(0, str(code)); sys.path.insert(1, str(priv / 'pylib')) if (priv / 'pylib').exists() else None
    from acs_replication import deploy, readers
    import joblib
    rec = {'schema': 'ACSR1-restore-use-check-v1', 'restore_dir_name': rd.name, 'labels_used': False}
    raw_csv = priv / 'deploy_tmp' / 'DEV_REPORT_raw_inputs.csv'
    rows = deploy.read_raw_csv(raw_csv)
    out = deploy.release(priv / 'deploy' / 'LOCAL_TREE_S0', rows)
    with np.load(priv / 'runs' / 'STAGE_B_0' / 'STAGE_B_OUTPUTS_E20.npz') as z:
        stored = {t: z[f'LOCAL_TREE__DEV_REPORT__{t}'] for t in ('income', 'occupation')}
    rec['redeploy_LOCAL_TREE_S0'] = {t: {'max_abs_prob_diff': float(np.max(np.abs(out[t]['probability'] - stored[t]))),
                                         'decisions_identical': bool(np.array_equal(out[t]['decision'], stored[t].argmax(1)))} for t in stored}
    rr = json.loads((priv / 'runs' / 'READERS_0' / 'READERS_SEED.json').read_text())
    win = rr['services']['LOCAL_TREE']['winner']['pair']['reader']
    model = joblib.load(priv / 'runs' / 'READERS_0' / f'MODEL__LOCAL_TREE__pair__{win}.joblib')
    sign = rr['services']['LOCAL_TREE']['validation']['pair'][win]['orientation']
    t = readers.tables({k: stored[k] for k in stored})
    kind = dict(readers.SLATE)[win]
    from task_conditioned_representation.attacks import predict_reader, orient
    s = orient(predict_reader(model, t['pair'][kind]), sign)
    with np.load(priv / 'runs' / 'READERS_0' / 'DEV_REPORT_SCORES.npz') as z:
        ref = z[f'LOCAL_TREE__pair__{win}']
    rec['restored_reader_LOCAL_TREE_pair_winner_s0'] = {'reader': win, 'max_abs_score_diff': float(np.max(np.abs(s - ref)))}
    rec['ok'] = (all(v['max_abs_prob_diff'] <= 1e-12 and v['decisions_identical'] for v in rec['redeploy_LOCAL_TREE_S0'].values())
                 and rec['restored_reader_LOCAL_TREE_pair_winner_s0']['max_abs_score_diff'] <= 1e-12)
    Path(out_path).write_text(json.dumps(rec, indent=2, sort_keys=True))
    return rec


if __name__ == '__main__':
    print(json.dumps(main(sys.argv[1], sys.argv[2])))
