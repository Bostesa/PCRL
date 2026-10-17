"""Write FINALIST_LOCK.json from PROMOTION.json and the declared final bank roster (no label read; write-once).

    python -m amd.finalize ROLE=KIND:NAME [ROLE=KIND:NAME ...]
        KIND frozen (frozen-hybrids bank), um (SCM follow-up bank) or release (FAMILY/CONFIG of this campaign)
    The role FINALIST must be the promoted configuration; every 'release' role is frozen by its release hash on both seeds.
"""
from __future__ import annotations
import json
import sys

from . import common as C
from . import devrun as DR


def main(specs):
    prom = json.loads((C.RESULTS / 'PROMOTION.json').read_text())
    if prom['decision'] != 'FINALIST_PROMOTED':
        raise SystemExit('no finalist promoted: nothing to freeze')
    fin = prom['finalist']
    bank, hashes, records = [], {}, {}
    for spec in specs:
        role, src = spec.split('=', 1); kind, name = src.split(':', 1)
        if kind == 'release':
            family, config_id = name.split('/')
            source = {'kind': 'release', 'family': family, 'config_id': config_id}
            hashes[name] = {}
            for s in C.SEEDS:
                _, rec = DR.load_release(family, config_id, s, roles=('MODEL_SELECT',))
                hashes[name][f'S{s}'] = rec['release_sha256']
                records[f'{name}|S{s}'] = {k: rec.get(k) for k in ('mechanism', 'note', 'labels_used_at_training', 'runtime_inputs', 'h_fit_role',
                                                                   'income_statistic_definition', 'base_occupation', 'config')}
        elif kind in ('frozen', 'um'):
            source = {'kind': kind, 'name': name}
        else:
            raise ValueError('unregistered source kind')
        bank.append({'role': role, 'source': source})
    f = next(b for b in bank if b['role'] == 'FINALIST')['source']
    if f.get('kind') != 'release' or (f['family'], f['config_id']) != (fin['family'], fin['config_id']):
        raise SystemExit('FINALIST role must be the promoted configuration')
    rec = {'schema': 'AMD1-finalist-lock-v1', 'utc': C.now(), 'promotion_sha256': C.sha(C.RESULTS / 'PROMOTION.json'),
           'finalist': fin, 'bank': bank, 'release_sha256': hashes, 'release_records': records,
           'output_contract': 'each recipient: its own float64 probability vector and argmax decision; the linked pair is the aligned two outputs; nothing else',
           'note': 'frozen before any final reader fit; the final bank deduplicates exact aliases (at most 8 effective releases per seed)'}
    C.write_json_x(C.RESULTS / 'FINALIST_LOCK.json', rec)
    print(json.dumps({'finalist': fin, 'roles': [b['role'] for b in bank]}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
