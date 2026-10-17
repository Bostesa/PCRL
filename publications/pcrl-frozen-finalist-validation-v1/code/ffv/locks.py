"""Writes REPORTING_LOCK.json after the reader fits and before the one logged DEV_REPORT re-read (PROTOCOL.md section 9; review
B5): fitted readers (model-file hashes inside each seed record), selected winners and candidate order, ATTACK_VALIDATION and
DEV_REPORT score files (no labels), the CATI fitted state (checked against READER_BANK_LOCK), controls on ATTACK_VALIDATION
(band check; any exit would be a logged deviation), the bank, the source reader files, the report and feasibility code, the
invented-label dry run and the frozen interpretation and feasibility specifications."""
from __future__ import annotations
import json

from . import common as C


def main():
    lock = json.loads((C.RESULTS / 'READER_BANK_LOCK.json').read_text())
    src_man = json.loads((C.SRC_RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())
    src_rl = json.loads((C.SRC_RESULTS / 'REPORTING_LOCK.json').read_text())
    out = {'schema': 'FFV1-reporting-lock-v1', 'utc': C.now(), 'enhanced_readers': {}, 'checks': {}}
    problems = []
    for s in C.SEEDS:
        d = C.RUNS / f'READERS_ENH_S{s}'
        rec = json.loads((d / 'READERS_ENH_SEED.json').read_text())
        out['enhanced_readers'][f'S{s}'] = {'record_sha256': C.sha(d / 'READERS_ENH_SEED.json'), 'av_scores_sha256': C.sha(d / 'AV_SCORES.npz'),
                                           'dev_report_scores_sha256': C.sha(d / 'DEV_REPORT_SCORES.npz'), 'n_model_files': len(rec['model_files_sha256']),
                                           'winners': {svc: {k: {v: x['candidate'] + (' [new]' if x['origin'] == 'new' else '') for v, x in w[k].items()} for k in w}
                                                       for svc, w in rec['winners'].items()}}
        if rec['av_scores_sha256'] != out['enhanced_readers'][f'S{s}']['av_scores_sha256'] or rec['dev_report_scores_sha256'] != out['enhanced_readers'][f'S{s}']['dev_report_scores_sha256']:
            problems.append(f'S{s}: score files differ from the reader record')
        # the fitted CATI state of the run equals the label-free state recorded in READER_BANK_LOCK before any fit
        st = lock['cati_fitted_state'][f'S{s}']
        mism = []
        for rid, des in rec['designs'].items():
            if not rid.endswith(('CATI01', 'CATI1')) or rid.startswith(('VAR__', 'AUX__')):
                continue
            kind, rest = rid.split('__', 1)
            wire = rest.rsplit('__', 1)[0]
            key = wire if kind == 'MARG' else f'{wire}|pair'
            ref = st.get(key)
            if ref is None:
                svc, view = key.split('|')
                ref = next((v for k, v in st.items() if k.endswith('|' + view) and v == {kk: vv for kk, vv in des.items() if kk in v}), None)
            cmp = {k: v for k, v in des.items() if k not in ('n_columns', 'unseen_income_vectors')}
            if ref is None or cmp != ref:
                mism.append(rid)
        out['checks'][f'S{s}|cati_state_equals_reader_bank_lock'] = not mism
        if mism:
            problems.append(f'S{s}: CATI fitted state differs from READER_BANK_LOCK for {mism[:4]}')
        bands_ok = {f'{view}|{name}': (c['planted_attack_validation_auc'] >= 0.95 and abs(c['shuffled_attack_validation_auc'] - 0.5) <= 0.05)
                    for view, cs in rec['controls'].items() for name, c in cs.items()}
        out['checks'][f'S{s}|attack_validation_controls_in_band'] = bands_ok
        out['checks'][f'S{s}|out_hgb_controls_bitwise_equal_source'] = rec['controls_out_hgb_vs_source_bitwise']
        out['checks'][f'S{s}|reproduction_subset_bitwise'] = {k: v['bitwise_equal_to_stored_score'] for k, v in rec['reproduction'].items()}
        out['checks'][f'S{s}|not_converged'] = rec['not_converged']
        if not all(bands_ok.values()):
            problems.append(f'S{s}: an ATTACK_VALIDATION control is outside its band (logged deviation required)')
    out['bank_sha256'] = {f'S{s}': C.sha(C.admitted(f'bank/SERVICE_BANK_S{s}.npz')) for s in C.SEEDS}
    out['bank_equals_source_manifest'] = all(out['bank_sha256'][f'S{s}'] == src_man['bank_files_sha256'][f'S{s}'] for s in C.SEEDS)
    out['source_reader_files_sha256'] = {f'S{s}': {'readers_seed': C.sha(C.admitted(f'runs/READERS_{s}/READERS_SEED.json')),
                                                   'dev_report_scores': C.sha(C.admitted(f'runs/READERS_{s}/DEV_REPORT_SCORES.npz'))} for s in C.SEEDS}
    out['source_reader_files_equal_source_reporting_lock'] = all(
        out['source_reader_files_sha256'][f'S{s}']['readers_seed'] == src_rl['readers'][f'S{s}']['readers_seed_sha256'] and
        out['source_reader_files_sha256'][f'S{s}']['dev_report_scores'] == src_rl['readers'][f'S{s}']['dev_report_scores_sha256'] for s in C.SEEDS)
    out['locks_sha256'] = {n: C.sha(C.RESULTS / n) for n in ('PROTOCOL.md', 'READER_BANK_LOCK.json', 'FROZEN_CANDIDATE_MANIFEST.json', 'SOURCE_ADMISSION.json')}
    out['code_sha256'] = {f'ffv/{p.name}': C.sha(p) for p in sorted((C.ROOT / 'ffv').glob('*.py'))}
    out['dryrun_report_invented_sha256'] = C.sha(C.RESULTS / 'DRYRUN_REPORT_INVENTED.json')
    out['frozen_specifications'] = {'interpretation': 'PROTOCOL.md section 8 (review B1, B2); implemented in ffv/report.py',
                                    'feasibility': 'PROTOCOL.md section 10 (review B4); implemented in ffv/feasibility.py',
                                    'dev_report_events': list(C.DEV_EVENTS)}
    out['problems'] = problems
    out['note'] = 'frozen before the one logged coordinator follow-up DEV_REPORT re-read; nothing listed here may change afterwards'
    C.write_json_x(C.RESULTS / 'REPORTING_LOCK.json', out)
    print(json.dumps({'problems': problems, 'checks': out['checks'], 'bank_equals_source_manifest': out['bank_equals_source_manifest'],
                      'source_reader_files_equal_source_reporting_lock': out['source_reader_files_equal_source_reporting_lock']}, indent=1))


if __name__ == '__main__':
    main()
