"""Unit FFV_DRYRUN_REPORT: the frozen report code run end to end on INVENTED DEV_REPORT labels (real label-free row and household
ids; random SEX, income and occupation), before REPORTING_LOCK. No DEV_REPORT truth is read and no access event is logged."""
from __future__ import annotations
import json

import numpy as np

from . import common as C
from . import report as RP


def main():
    R = C.load_role('DEV_REPORT', ('features',))
    rng = np.random.default_rng(424242)
    n = len(R['row_id'])
    truth = {'row_id': R['row_id'], 'unit': R['unit'], 'sex': (rng.random(n) < 0.47).astype(np.int64),
             'income': (rng.random(n) < 0.4).astype(np.int64), 'occupation': rng.integers(0, 5, n).astype(np.int64)}
    out = C.PRIV / 'dryrun' / 'REPORT_INVENTED'
    res = RP.run(out, truth=truth, priv_dir=out / 'private')
    summary = {'invented_labels': True, 'claims': {sc: {k: v['label'] for k, v in res['scorecards'][sc]['claims'].items()} for sc in res['scorecards']},
               'CV': res['control_validity']['CV'], 'endpoints': {sc: len(res['scorecards'][sc]['endpoints']) for sc in res['scorecards']},
               'interpretation': {k: v['interpretation'] for k, v in res['interpretation'].items()},
               'note': 'invented labels: values are meaningless; this checks that the frozen code runs end to end'}
    C.write_json_x(C.RESULTS / 'DRYRUN_REPORT_INVENTED.json', summary)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
