"""Assemble INDEPENDENT_VERIFICATION.json from the per-stage independent/check*.json files (no recomputation here)."""
import datetime as dt
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
IND = RES / 'independent'
NAMES = {1: 'SOURCES', 2: 'TASK MAPPING', 3: 'INPUT EXCLUSIONS', 4: 'HOUSEHOLD ROLES', 5: 'CLOSED ASSESSMENT + ACCESS',
         6: 'TEACHER IDENTITY + WIRES', 7: 'NOMINATIONS', 8: 'UTILITY, RECOVERY, SUBGROUP, INFERENCE', 9: 'READER REFITS',
         10: 'CONTROLS', 11: 'DEPLOYMENT'}
KEY = {1: ('fail_reasons', 'evidence'), 2: ('evidence',), 3: ('evidence',), 4: ('problems', 'sealed_hash_status', 'evidence'), 5: ('evidence',),
       6: ('evidence',), 7: ('evidence',), 8: ('evidence',), 9: ('evidence',), 10: ('evidence',), 11: ('evidence',)}


def brief(obj, limit=6000):
    s = json.dumps(obj, sort_keys=True, default=str)
    return obj if len(s) <= limit else {'truncated_json': s[:limit] + '...'}


def main(extra=None):
    checks = []
    found = {}
    addons = {}
    for f in sorted(glob.glob(str(IND / 'check*.json'))):
        d = json.loads(Path(f).read_text())
        if isinstance(d['id'], str) and not d['id'].isdigit():
            addons[int(d['id'][:-1])] = (Path(f).name, d); continue
        found[int(d['id'])] = (Path(f).name, d)
    for i in range(1, 12):
        if i not in found:
            checks.append({'id': i, 'name': NAMES[i], 'status': 'MISSING', 'evidence': {'note': 'stage output not produced'}})
            continue
        name, d = found[i]
        ev = {k: brief(d.get(k)) for k in dict.fromkeys(KEY[i] + ('fail_reasons', 'problems', 'scope')) if k in d}
        ev['file'] = f'independent/{name}'
        status = d['status']
        if i in addons:
            an, ad = addons[i]
            ev['addon'] = {'file': f'independent/{an}', 'name': ad['name'], 'status': ad['status'], 'evidence': brief(ad.get('per_seed_task'))}
            if ad['status'] != 'PASS':
                status = 'FAIL'
        if extra and i in extra:
            ev['verifier_assessment'] = extra[i]
        checks.append({'id': i, 'name': NAMES[i], 'status': status, 'evidence': ev})
    counted = [c for c in checks if c['status'] in ('PASS', 'FAIL')]
    out = {'schema': 'ACSR1-independent-verification-v1', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
           'checks': checks, 'pass_count': sum(c['status'] == 'PASS' for c in checks), 'fail_count': sum(c['status'] == 'FAIL' for c in checks),
           'other_count': len(checks) - len(counted),
           'scope': ('replay vs refit: points, metrics, clauses, nominations, endpoints, bounds and claims are RECOMPUTED by separate code from the stored '
                     'probability outputs, stored reader scores and exported labels (replay of scoring, not refits of the services); teacher (T_TREE) and '
                     'E20 tree wires are RECOMPUTED from the saved teacher JSON and theta files (own routing); the household bootstrap is both re-drawn with an '
                     'independent seed and replayed from the registered seed with own code; reader models are REFIT for the predeclared sample of 5 pair-view '
                     'winners (plus control readers and the validation winner choice) from released ATTACK_FIT outputs; no service model (CatBoost, U0, Stage B '
                     'trainer, NEURAL_RAW, FAGTB, FARE) was refit.'),
           'DEV_REPORT_truth_reread': True,
           'DEV_REPORT_truth_reread_note': 'verification re-read of the already-opened DEV_REPORT role (exports/DEV_REPORT_TRUTH); the access log was not written',
           'sealed_assessment_touched': False,
           'sealed_assessment_note': 'no SEALED export exists; only the raw SERIALNO identifier column was read to re-derive household roles and the sealed key-set hash'}
    (RES / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(out, indent=1, sort_keys=False, default=str))
    print(json.dumps({c['id']: c['status'] for c in checks}), out['pass_count'], out['fail_count'])


if __name__ == '__main__':
    import sys
    extra = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else None
    main({int(k): v for k, v in extra.items()} if extra else None)
