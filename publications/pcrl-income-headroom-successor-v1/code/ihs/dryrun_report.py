"""Unit IHS_DRYRUN_REPORT: the frozen report and claims code run end to end on INVENTED DEV_REPORT labels (real label-free row and
household ids; random SEX, income and occupation), after the final readers and before REPORTING_LOCK. No DEV_REPORT truth is read
and no access event is logged; the values are meaningless and only show that the frozen code runs."""
from __future__ import annotations
import json

import numpy as np

from . import claims as CL
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
    lock = json.loads((C.RESULTS / 'ANALYSIS_LOCK.json').read_text())
    labels = {sc: {k: v['label'] for k, v in CL.evaluate(res['scorecards'][sc]['endpoints'], res['CV'], lock['declared']).items()}
              for sc in res['scorecards']}
    summary = {'invented_labels': True, 'claims': labels, 'CV': res['CV'],
               'endpoints': {sc: len(res['scorecards'][sc]['endpoints']) for sc in res['scorecards']},
               'note': 'invented labels: values are meaningless; this checks that the frozen code runs end to end'}
    C.write_json_x(C.RESULTS / 'checks' / 'DRYRUN_REPORT_INVENTED.json', summary)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
