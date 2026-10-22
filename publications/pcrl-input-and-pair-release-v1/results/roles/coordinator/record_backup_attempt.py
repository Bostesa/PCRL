"""Record a custody backup attempt from its queue unit's stdout (coordinator; label-free).

    python3 results/pcrl_input_and_pair_release_v1/roles/coordinator/record_backup_attempt.py <unit id> <output name> [<note>]

The unit's stdout is the JSON returned by `ipr.custody backup`. It is written unchanged under `custody/<output name>`, together with
the unit id and an optional note (for example, why a failed attempt's backup folder was removed).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ipr import common as C          # noqa: E402

unit, name = sys.argv[1], sys.argv[2]
note = sys.argv[3] if len(sys.argv) > 3 else None
rec = json.loads((C.PRIV / 'compute' / 'units' / unit / 'stdout.log').read_text())
out = {'schema': 'IPR1-custody-record-v1', 'unit_id': unit, 'recorded_utc': C.now(), 'note': note, 'backup': rec}
C.write_json_x(C.RESULTS / 'custody' / name, out)
print(json.dumps({'written': f'custody/{name}', 'all_ok': rec.get('all_ok'), 'path': rec.get('path')}))
