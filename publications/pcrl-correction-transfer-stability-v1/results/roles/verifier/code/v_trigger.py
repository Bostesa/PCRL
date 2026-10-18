"""Verifier provenance check of its own question-A bootstrap finding (sent to the coordinator as the "20:29Z" ATTENTION), for the
statistical reviewer's condition 1 of adjudication A5 (roles/statistical_reviewer/ADJUDICATION_SECTION8.md). Label-free: reads the queue
ledger, the label-access log, the verifier's own records and file hashes. No data.

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records).

    python3 .../v_trigger.py --tag V1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
CAMP = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1'
TRACKF = CAMP / 'runs' / 'TRACKF'
REC = V.VER_DIR / 'records'
FINDING_UNITS = ('VER_QA_D0_REFIT_PAIR_FULL_OOF_V1', 'VER_QA_D0_ALL_V2', 'VER_QA_D1_ALL_V2', 'VER_QA_BOOT_D0_V1', 'VER_QA_BOOT_D1_V1')
# bounds of the message time, from the verifier's own session: the last record written before sending (the D1 bootstrap-scheme check,
# appended at 20:27:28Z) and the first shell clock reading after sending (date -u: 20:28:04Z); the message itself was headed "20:29Z",
# an estimate written by the verifier
SENT_AFTER, SENT_BEFORE = '2026-10-18T20:27:28+00:00', '2026-10-18T20:28:04+00:00'
RULING_COMMIT = '2026-10-18T20:34:05+00:00'          # eece7dbdb: corrected code and ADJUDICATION_QA_BOOTSTRAP.md


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    ledger = [json.loads(l) for l in (CAMP / 'compute' / 'ledger.jsonl').read_text().splitlines() if l.strip()]
    access = [json.loads(l) for l in (CAMP / 'access' / 'LABEL_ACCESS.jsonl').read_text().splitlines() if l.strip()]
    units = {}
    for e in ledger:
        if e.get('event') in ('unit_start', 'unit_end') and e['unit_id'] in FINDING_UNITS:
            units.setdefault(e['unit_id'], {})[e['event']] = e['utc']
    reads = [{'utc': e['utc'], 'unit': e.get('unit_id'), 'operation': e['operation'], 'role': e['role'], 'partition': e.get('partition'),
              'prole': e.get('prole'), 'people': e['people'], 'rows_sha256': e['rows_sha256'], 'touches_repair_panel': e['touches_repair_panel']}
             for e in access if e.get('unit_id') in FINDING_UNITS]
    # records that carried the finding
    rec_files = sorted([p for p in REC.glob('V_QUESTIONA__D[01]__*__V[12].json') if '__REG__' not in p.name]   # REG replays came later
                       + [REC / 'V_QA_BOOT__D0__V1.json', REC / 'V_QA_BOOT__D1__V1.json'])
    records = {p.name: {'sha256': V.file_sha(p), 'code_sha256_at_run': json.loads(p.read_text()).get('code_sha256'),
                        'utc_in_CHECKS': None} for p in rec_files}
    checks = json.loads((V.VER_DIR / 'CHECKS.json').read_text())['checks']
    for c in checks:
        for name in records:
            stem = name[2:-5]                                          # 'QUESTIONA__D0__...__V1' / 'QA_BOOT__D0__V1'
            if stem.startswith('QA_BOOT__') and c['id'].startswith(f"C3.QA.{stem.split('__')[1]}.BOOTSTRAP_SCHEME__V1"):
                records[name]['utc_in_CHECKS'] = c['utc']
    # campaign input files of those replays (D partitions only), with SHA-256
    inputs = {}
    for P in ('D0', 'D1'):
        for f in sorted((TRACKF / P / 'QUESTION_A').glob('*.json')):
            if not f.name.endswith('__REG.json'):
                inputs[str(f.relative_to(ROOT))] = V.file_sha(f)
        for f in sorted((TRACKF / P / 'SELECT').glob('*.json')):
            if '__REALIZED' not in f.name:
                inputs[str(f.relative_to(ROOT))] = V.file_sha(f)
        for f in ('WITNESS/WITNESS.npz', 'WITNESS/OOF.npz'):
            inputs[str((TRACKF / P / f).relative_to(ROOT))] = V.file_sha(TRACKF / P / f)
        for dd in sorted((TRACKF / P / 'DIR').glob('*')):
            for f in ('TRAIN.npz', 'DELTA.npz'):
                inputs[str((dd / f).relative_to(ROOT))] = V.file_sha(dd / f)
        nd = TRACKF / P / 'NESTED'
        for f in sorted(nd.glob('NESTED_*.npz')) if nd.is_dir() else []:
            inputs[str(f.relative_to(ROOT))] = V.file_sha(f)
    import datetime as dt
    mt = max(dt.datetime.fromtimestamp((ROOT / k).stat().st_mtime, dt.timezone.utc) for k in inputs)
    inputs_unmodified = mt.isoformat(timespec='seconds') < SENT_AFTER
    # every verifier label read of R partitions or panels before the message bound and before the ruling commit
    ver = [e for e in access if str(e.get('entry_point', '')).startswith('results/pcrl_correction_transfer_stability_v1/roles/verifier/code/')]
    r_before_msg = [(e['utc'], e.get('unit_id'), e['partition'], e['prole'], e['touches_repair_panel']) for e in ver
                    if e.get('partition') in ('R0', 'R1', 'C0', 'C1') and e['utc'] < SENT_BEFORE]
    r_before_ruling = [(e['utc'], e.get('unit_id'), e['partition'], e['prole'], e['touches_repair_panel']) for e in ver
                       if e.get('partition') in ('R0', 'R1', 'C0', 'C1') and SENT_BEFORE <= e['utc'] < RULING_COMMIT]
    panel_before_msg = [x for x in r_before_msg if x[4]]
    finding_reads_ok = all(r['partition'] in ('D0', 'D1') and not r['touches_repair_panel'] for r in reads if r['role'] == 'MODEL_FIT')
    units_before = all(v.get('unit_end', '9') <= SENT_AFTER for v in units.values())
    qa_r_checks = sorted((c['utc'], c['id'], c['status']) for c in checks
                         if c['id'].startswith(('C3.QA.R0.', 'C3.QA.R1.')) and c['utc'] < RULING_COMMIT)
    out = [{'id': f'P4_QA_CORRECTION_TRIGGER_INPUTS__{a.tag}', 'kind': 'PROCEDURAL_COMPLIANCE',
            'what': 'adjudication A5 condition 1: the inputs of the verifier\'s question-A bootstrap finding (message headed "20:29Z") were '
                    'D-stage only; no R0 / R1 output and no repair-panel read entered it',
            'method': 'queue ledger (start / end of the five units behind the finding), label-access log (every read of those units), '
                      'SHA-256 of the campaign files those replays read and of the verifier records that carried the finding',
            'status': 'PASS' if (finding_reads_ok and units_before and not panel_before_msg and inputs_unmodified) else 'FAIL',
            'numbers': {'message_sent_between_utc': [SENT_AFTER, SENT_BEFORE], 'message_header_time': '20:29Z (estimate written by '
                        'the verifier; the bounds above are the record)',
                        'units': units, 'label_reads_of_these_units': reads, 'campaign_input_files_sha256': inputs,
                        'latest_input_file_modification_utc': mt.isoformat(timespec='seconds'),
                        'inputs_unmodified_since_before_the_message': inputs_unmodified,
                        'verifier_records': records,
                        'code_basis': 'the per-statistic resampling was first noted by reading cts.rules.prediction_comparison (code, no '
                                      'data) before any question A output existed; the replays quantified it on D0 / D1 only',
                        'verifier_R_or_panel_reads_before_message': r_before_msg,
                        'note_R_reads_before_message': 'R0 / R1 FIT and GUARD rows only (witness / routing refit identity checks, '
                                                       '20:21:04-20:21:50Z); no panel; not used by the question-A finding'}},
           {'id': f'P4b_R_QA_RESULTS_IN_SHARED_RECORDS_BEFORE_RULING__{a.tag}', 'kind': 'PROCEDURAL_COMPLIANCE',
            'what': 'disclosure for A5: verifier results derived from repair panels that existed in the shared CHECKS.json before the '
                    'correction commit eece7dbdb (20:34:05Z), although they were not inputs of the finding and not messaged before it',
            'method': 'CHECKS.json entry times; label-access log of verifier units between the message and the commit',
            'status': 'ATTENTION' if qa_r_checks else 'PASS',
            'numbers': {'verifier_R_or_panel_reads_after_message_before_ruling_commit': r_before_ruling,
                        'R_question_A_checks_written_before_ruling_commit': qa_r_checks,
                        'second_message_on_R1_labels': 'sent after the VER_QA_BOOT_R1_V1 unit ended (20:34:34Z), i.e. after the commit',
                        'reading': 'R0 replays (20:29:04Z) showed no scheme-dependent label; the R1 replays (20:33:36Z) wrote the two '
                                   'R1 scheme-dependent labels into CHECKS.json 29 s before the commit; whether anyone read them before '
                                   'the commit is not known to the verifier'}}]
    code = V.code_hashes(__file__, V.__file__)
    for c in out:
        c['code_sha256'] = code
    V.write_record(f'V_TRIGGER__{a.tag}.json', {'checks': out, 'utc': V.utc()})
    print(json.dumps(V.append_checks(out)))
    print(json.dumps({c['id']: c['status'] for c in out}, indent=1))
    print(json.dumps({'units': units, 'reads': [(r['utc'], r['unit'], r['partition'], r['prole'], r['touches_repair_panel']) for r in reads],
                      'r_before_msg': r_before_msg, 'r_before_ruling': r_before_ruling, 'qa_r_checks': qa_r_checks}, indent=0)[:4000])


if __name__ == '__main__':
    main()
