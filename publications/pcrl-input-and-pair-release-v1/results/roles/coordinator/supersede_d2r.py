"""Move the two D2R records that ran before the version-0 robustness grid had finished (label NOT_RUN: incomplete robustness
cells) out of the `claims/*/D2R.json` pattern, to `claims/superseded/<reading>_D2R_EARLY/`, with a note recording their SHA-256,
the unit that wrote them and why; the readings are then rerun into the original folders (coordinator; label-free).

    python3 results/pcrl_input_and_pair_release_v1/roles/coordinator/supersede_d2r.py
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ipr import common as C          # noqa: E402

out = {}
for reading, unit in (('IA_REG', 'IPR_D2R_IA_REG_V1'), ('IB_REG', 'IPR_D2R_IB_REG_V1')):
    src = C.RESULTS / 'claims' / f'{reading}_D2R' / 'D2R.json'
    rec = json.loads(src.read_text())
    if rec['d2r'].get('label') != 'NOT_RUN':
        raise SystemExit(f'{src} is not the early NOT_RUN record')
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    dst = C.RESULTS / 'claims' / 'superseded' / f'{reading}_D2R_EARLY'
    dst.mkdir(parents=True, exist_ok=False)
    shutil.move(str(src), str(dst / 'D2R.json'))
    (C.RESULTS / 'claims' / f'{reading}_D2R').rmdir()
    out[reading] = {'moved_to': str((dst / 'D2R.json').relative_to(C.ROOT)), 'sha256': sha, 'written_by_unit': unit,
                    'label': 'NOT_RUN', 'record_utc': rec['utc']}
note = {'schema': 'IPR1-superseded-v1', 'utc': C.now(), 'records': out,
        'why': 'the D2R lane was queued while IPR_GRID_ROB2_V0 still ran; its units took the slot freed by IPR_GRID_ROB2_V1 at 04:04:45Z '
               'and read IPR1_ROB with version 0 missing, so the rules returned D2R NOT_RUN (incomplete robustness cells); the readings '
               'are rerun into claims/<reading>_D2R after the version-0 grid ended (PROTOCOL_DEVIATIONS item 9); never read as results'}
C.write_json_x(C.RESULTS / 'claims' / 'superseded' / 'SUPERSEDED.json', note)
print(json.dumps(note, indent=1))
