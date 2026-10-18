"""Verifier re-check of ERRATA.md E1 (label-free): the erratum's corrected fold tags (CTS1_cf_P70 / CTS1_inner_P70) for the P70A witness
regenerate the saved P70A outer folds. A separate item: the C2.1 FAIL (the recorded tag CTS1_cf_P70A does not regenerate the saved
folds) stays as it is. Inner folds are not saved, so the inner tag is checked only through the code path (cts.trackf: same_rows_as).

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records).

    python3 .../v_erratum.py --tag V1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from cts import common as C  # noqa: E402   (label-free admitted features only)

TRACKF = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1' / 'runs' / 'TRACKF'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    F = C.features('MODEL_FIT')
    with np.load(TRACKF / 'P70A' / 'WITNESS' / 'OOF.npz') as z:
        fit_rows, fold = z['fit_rows'].astype(bool), z['fold'].copy()
    unit = F['unit'][fit_rows]
    with_erratum = V.deal(unit, 5, 'CTS1_cf_P70')
    with_record = V.deal(unit, 5, 'CTS1_cf_P70A')
    errata = (V.RESULTS / 'ERRATA.md').read_text()
    num = {'erratum_tag_regenerates_saved_outer_folds': bool(np.array_equal(with_erratum, fold)),
           'recorded_tag_regenerates_saved_outer_folds': bool(np.array_equal(with_record, fold)),
           'erratum_E1_present': 'E1. P70A witness record' in errata,
           'inner_folds': 'not saved; cts.trackf.fit_witness deals them with the same_rows_as tag (code path), not re-derivable here'}
    ok = num['erratum_tag_regenerates_saved_outer_folds'] and not num['recorded_tag_regenerates_saved_outer_folds'] and num['erratum_E1_present']
    chk = {'id': f'C2.11.P70A.ERRATUM_E1_FOLD_TAGS__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': 'ERRATA.md E1: the corrected tags (CTS1_cf_P70 / CTS1_inner_P70) regenerate the saved P70A outer folds (separate item; the '
                   'C2.1 FAIL for the recorded tag stays)',
           'method': 'own household dealing of the P70A FIT households with each tag, compared with the saved OOF.npz fold array (label-free)',
           'status': 'PASS' if ok else 'FAIL', 'numbers': num, 'code_sha256': V.code_hashes(__file__, V.__file__)}
    V.write_record(f'V_ERRATUM_E1__{a.tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], **num}))


if __name__ == '__main__':
    main()
