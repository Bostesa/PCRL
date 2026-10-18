"""Independent verifier, check 1: source admission and role integrity (own hashing code; no ihs.admit).

- Re-hash every admitted copy in SOURCE_ADMISSION.json (sha256 and byte count) and the expected source hash where recorded.
- Re-hash every run release (RELEASE.npz vs RELEASE.json), the final service banks (vs SERVICE_BANK_MANIFEST.json), the final
  reader files (vs REPORTING_LOCK.json) and the locks (vs REPORTING_LOCK.json).
- Access logs: MODEL_SELECT use log (a purpose on every line), ATTACK label reads (all after the FINALIST_LOCK commit),
  DEV_REPORT events (this campaign's log and the lineage-wide log).
- Live gate tests that read nothing: sealed refusal, MODEL_SELECT labels without purpose, unknown role field.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from ihs import common as C

RES = ROOT / 'results' / 'pcrl_income_headroom_successor_v1'
PRIV = ROOT / '.private' / 'pcrl_income_headroom_successor_v1'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def commit_time(rev):
    out = subprocess.run(['git', '-C', str(ROOT), 'log', '-1', '--format=%cI', rev], capture_output=True, text=True).stdout.strip()
    return dt.datetime.fromisoformat(out)


def main():
    out = {}
    adm = json.loads((RES / 'SOURCE_ADMISSION.json').read_text())
    bad, n, exp_bad, cats = [], 0, [], {}
    for rel, rec in adm['copies'].items():
        p = PRIV / 'admitted' / rel; n += 1
        cats[rel.split('/')[0]] = cats.get(rel.split('/')[0], 0) + 1
        h = sha(p) if p.is_file() else None
        if h != rec['sha256'] or (rec.get('bytes') is not None and p.stat().st_size != rec['bytes']):
            bad.append(rel)
        if rec.get('expected_sha256') and rec['expected_sha256'] != rec['sha256']:
            exp_bad.append(rel)
    out['admitted'] = {'n_rehashed': n, 'n_recorded': adm['n_copies'], 'mismatch': bad, 'expected_vs_recorded_mismatch': exp_bad,
                       'by_category': cats, 'recorded_problems': adm['problems'], 'sealed_assessment': adm.get('sealed_assessment')}
    # run releases
    rel_bad, rel_n = [], 0
    for rj in sorted((PRIV / 'runs').glob('*/*/S*/RELEASE.json')):
        rec = json.loads(rj.read_text()); rel_n += 1
        if sha(rj.parent / 'RELEASE.npz') != rec['release_sha256']:
            rel_bad.append(str(rj.parent.relative_to(PRIV)))
    out['run_releases'] = {'n_rehashed': rel_n, 'mismatch': rel_bad}
    man = json.loads((RES / 'SERVICE_BANK_MANIFEST.json').read_text()); rl = json.loads((RES / 'REPORTING_LOCK.json').read_text())
    fl = json.loads((RES / 'FINALIST_LOCK.json').read_text())
    out['final_banks'] = {s: sha(PRIV / 'runs' / 'FINAL' / f'SERVICE_BANK_{s}.npz') == h for s, h in man['bank_files_sha256'].items()}
    out['reader_files'] = {s: {'AV': sha(PRIV / 'runs' / f'READERS_{s}' / 'AV_SCORES.npz') == r['av_scores_sha256'],
                               'DEV': sha(PRIV / 'runs' / f'READERS_{s}' / 'DEV_REPORT_SCORES.npz') == r['dev_report_scores_sha256'],
                               'RECORD': sha(PRIV / 'runs' / f'READERS_{s}' / 'READERS_SEED.json') == r['record_sha256']} for s, r in rl['readers'].items()}
    out['locks_vs_reporting_lock'] = {n_: sha(RES / n_) == h for n_, h in rl['locks_sha256'].items()}
    # successor / twin / control release records named in the manifest vs run releases
    rr = {}
    for s in ('S0', 'S1'):
        for svc, h in man['per_seed'][s].get('release_records', {}).items():
            rr[f'{s}|{svc}'] = h
    found = {}
    for rj in sorted((PRIV / 'runs').glob('*/*/S*/RELEASE.json')):
        rec = json.loads(rj.read_text()); found.setdefault(rec['release_sha256'], []).append(str(rj.parent.relative_to(PRIV / 'runs')))
    out['manifest_release_records'] = {k: found.get(h) for k, h in rr.items()}
    # committed status of the locks and registration files
    out['committed'] = {n_: C.committed(RES / n_) for n_ in ('PROTOCOL.md', 'FINALIST_LOCK.json', 'ANALYSIS_LOCK.json', 'READER_PLAN.json',
                                                         'SERVICE_BANK_MANIFEST.json', 'REPORTING_LOCK.json', 'NOMINATION.json',
                                                         'roles/statistical_reviewer/RESERVE_RULE.json', 'roles/statistical_reviewer/NOMINATION_RULE.md')}
    # logs
    ms = [json.loads(l) for l in (PRIV / 'access' / 'MODEL_SELECT_USE.jsonl').read_text().splitlines() if l.strip()]
    out['model_select_log'] = {'lines': len(ms), 'lines_without_purpose': sum(1 for e in ms if not str(e.get('purpose') or '').strip()),
                               'roles': sorted({e.get('role') for e in ms}), 'first_utc': ms[0]['utc'] if ms else None, 'last_utc': ms[-1]['utc'] if ms else None,
                               'lines_without_unit_id': sum(1 for e in ms if not e.get('unit_id'))}
    reg_t = commit_time('74376912c'); fl_t = commit_time('930235e9e'); rl_t = commit_time('8e559efe8')
    out['commit_times'] = {'registration_74376912c': reg_t.isoformat(), 'finalist_lock_930235e9e': fl_t.isoformat(), 'reporting_lock_8e559efe8': rl_t.isoformat()}
    ms_before_reg = [e for e in ms if dt.datetime.fromisoformat(e['utc']) < reg_t]
    out['model_select_log']['lines_before_registration_commit'] = len(ms_before_reg)
    out['model_select_log']['units_before_registration_commit'] = sorted({str(e.get('unit_id')) for e in ms_before_reg})
    al = [json.loads(l) for l in (PRIV / 'access' / 'ATTACK_LABEL_READS.jsonl').read_text().splitlines() if l.strip()]
    out['attack_log'] = {'lines': len(al), 'before_finalist_lock_commit': sum(1 for e in al if dt.datetime.fromisoformat(e['utc']) < fl_t),
                         'without_purpose': sum(1 for e in al if not str(e.get('purpose') or '').strip()), 'first_utc': al[0]['utc'] if al else None,
                         'units': sorted({str(e.get('unit_id')) for e in al})}
    dv = C.dev_access_events()
    out['dev_log'] = [{'event': e['event'], 'utc': e['utc'], 'unit_id': e.get('unit_id'), 'reporting_lock_sha256': e['reporting_lock_sha256'],
                       'after_reporting_lock_commit': dt.datetime.fromisoformat(e['utc']) > rl_t} for e in dv]
    src = [json.loads(l) for l in Path(C.SRC_ACCESS_LOG).read_text().splitlines() if l.strip()]
    mine = [e for e in src if e.get('campaign') == C.CAMPAIGN]
    out['lineage_log'] = {'lines_total': len(src), 'this_campaign': [{'event': e['event'], 'utc': e['utc']} for e in mine],
                          'before_this_campaign': len(src) - len(mine), 'recorded_before_campaign': adm.get('lineage_dev_report_events_before_this_campaign')}
    # live gate tests (each raises before any data or log access)
    tests = {}
    for name, fn in (('sealed', lambda: C.load_role('SEALED_ASSESSMENT', ('features',))),
                     ('model_select_labels_without_purpose', lambda: C.load_role('MODEL_SELECT', ('features', 'sex'))),
                     ('dev_report_label_field', lambda: C.load_role('DEV_REPORT', ('features', 'sex'))),
                     ('calibration_sex_field', lambda: C.load_role('CALIBRATION', ('features', 'sex')))):
        n_before = len((PRIV / 'access' / 'MODEL_SELECT_USE.jsonl').read_text().splitlines())
        try:
            fn(); tests[name] = 'NOT_REFUSED'
        except PermissionError as ex:
            tests[name] = f'REFUSED: {ex}'
        tests[name + '_log_lines_added'] = len((PRIV / 'access' / 'MODEL_SELECT_USE.jsonl').read_text().splitlines()) - n_before
    out['live_gate_tests'] = tests
    (PRIV / 'verifier' / 'ADMISSION_CHECK.json').write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(out, indent=1, sort_keys=True, default=str))


if __name__ == '__main__':
    main()
