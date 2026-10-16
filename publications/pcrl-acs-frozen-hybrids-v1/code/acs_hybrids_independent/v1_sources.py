"""V1 Source identities (label-free).

A. Every file of the published source results tree (commit f4ea199, publications/pcrl-acs-protected-tree-replication-v1/
   results) equals the inherited copy in results/pcrl_acs_protected_tree_replication_v1 byte for byte (sha256 of
   `git show`). Extra inherited files are listed.
B. Every restored source file under .private/acs_frozen_hybrids/source matches each published hash that pins it:
   TRAINING_COMPLETE_LOCK.frozen_files_sha256, REPORTING_LOCK.reader_files_sha256, EXTRACTION_SUMMARY.export_file_sha256,
   MODEL_MANIFEST.models and SELECTION_LOCK.output_file_sha256 (hashes read from the inherited documents verified in A).
   The private extraction receipt is unpinned; every export hash it lists must equal EXTRACTION_SUMMARY (this includes the
   DEV_REPORT truth files that acs_hybrids.common.dev_report_truth binds to).
C. Consistency of the campaign's own restore receipt and SOURCE_ADMISSION record with the hashes computed here.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from . import vcommon as V

NAMED = ('TRAINING_COMPLETE_LOCK.json', 'REPORTING_LOCK.json', 'EXTRACTION_SUMMARY.json', 'MODEL_MANIFEST.json',
         'SELECTION_LOCK.json', 'DEV_REPORT_RESULTS.json')


def git(*args, text=True):
    return subprocess.run(['git', '-C', str(V.ROOT), *args], capture_output=True, check=True, text=text).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(V.OUT / 'V1_source_identities.json'))
    a = ap.parse_args()

    # ---- A. inherited documents == published commit
    names = git('ls-tree', '-r', '--name-only', V.PUB_COMMIT, V.PUB_PATH + '/').split()
    rel = sorted(n[len(V.PUB_PATH) + 1:] for n in names)
    docs = {}
    for r in rel:
        pub = git('show', f'{V.PUB_COMMIT}:{V.PUB_PATH}/{r}', text=False)
        p = V.SRC_RES / r
        inh = V.sha256_file(p) if p.is_file() else None
        docs[r] = {'published_sha256': V.sha256_bytes(pub), 'inherited_sha256': inh,
                   'identical': inh == V.sha256_bytes(pub)}
    inherited_all = sorted(str(p.relative_to(V.SRC_RES)) for p in V.SRC_RES.rglob('*') if p.is_file())
    extra = sorted(set(inherited_all) - set(rel))
    missing = sorted(set(rel) - set(inherited_all))
    A = {'published_files': len(rel), 'identical': sum(v['identical'] for v in docs.values()),
         'not_identical': [k for k, v in docs.items() if not v['identical']], 'missing_in_inherited': missing,
         'extra_inherited_files_not_in_published_commit': extra,
         'named_documents': {n: docs.get(n, {}).get('identical') for n in NAMED}}
    A['pass'] = bool(A['identical'] == len(rel) and not missing and all(A['named_documents'].values()))

    # ---- B. restored files vs published hashes (taken from the inherited documents, which A ties to the commit)
    tl = V.jload(V.SRC_RES / 'TRAINING_COMPLETE_LOCK.json')['frozen_files_sha256']
    rl = V.jload(V.SRC_RES / 'REPORTING_LOCK.json')['reader_files_sha256']
    ex = V.jload(V.SRC_RES / 'EXTRACTION_SUMMARY.json')['export_file_sha256']
    mm = V.jload(V.SRC_RES / 'MODEL_MANIFEST.json')['models']
    sel = V.jload(V.SRC_RES / 'SELECTION_LOCK.json')['output_file_sha256']
    restored = sorted(p for p in V.SRCP.rglob('*') if p.is_file())
    files, mine = {}, {}
    for p in restored:
        r = str(p.relative_to(V.SRCP))
        h = V.sha256_file(p)
        mine[r] = h
        pins = []
        if r.startswith('runs/'):
            k = r[len('runs/'):]
            if k in tl:
                pins.append(('TRAINING_COMPLETE_LOCK.frozen_files_sha256', tl[k]))
            if k in rl:
                pins.append(('REPORTING_LOCK.reader_files_sha256', rl[k]))
        if r in sel:
            pins.append(('SELECTION_LOCK.output_file_sha256', sel[r]))
        for name, m in mm.items():
            if m.get('file') == r:
                pins.append((f'MODEL_MANIFEST.models.{name}', m['sha256']))
        parts = Path(r).parts
        if parts[0] == 'exports' and len(parts) == 3:
            role, fld = parts[1], Path(parts[2]).stem
            if isinstance(ex.get(role), dict) and fld in ex[role]:
                pins.append(('EXTRACTION_SUMMARY.export_file_sha256', ex[role][fld]))
        files[r] = {'sha256': h, 'pins': {k: v for k, v in pins}, 'n_pins': len(pins),
                    'all_pins_match': bool(pins) and all(v == h for _, v in pins)}
    receipt_rel = 'extraction/EXTRACTION_RECEIPT.json'
    unpinned = [r for r, v in files.items() if v['n_pins'] == 0 and r != receipt_rel]
    mismatched = [r for r, v in files.items() if v['n_pins'] > 0 and not v['all_pins_match']]
    # private extraction receipt: every export hash it lists must equal EXTRACTION_SUMMARY
    rc = V.jload(V.SRCP / receipt_rel)
    rc_cmp, sealed_listed = {}, False
    for role, flds in rc['files'].items():
        if role == 'SEALED_ASSESSMENT':
            sealed_listed = True
        if not isinstance(flds, dict):
            continue
        for fld, item in flds.items():
            if isinstance(item, dict) and 'sha256' in item:
                pub = ex.get(role, {}).get(fld) if isinstance(ex.get(role), dict) else None
                rc_cmp[f'{role}/{fld}'] = {'receipt': item['sha256'], 'published': pub, 'equal': pub == item['sha256']}
    dev_truth_keys = [f'DEV_REPORT/{f}' for f in ('income', 'occupation', 'sex', 'weights')]
    B = {'restored_files': len(files), 'pinned_files': sum(v['n_pins'] > 0 for v in files.values()),
         'pins_checked': sum(v['n_pins'] for v in files.values()), 'mismatched': mismatched,
         'unpinned_other_than_receipt': unpinned,
         'by_pin_source': {},
         'extraction_receipt': {'entries': len(rc_cmp), 'equal_to_EXTRACTION_SUMMARY': sum(v['equal'] for v in rc_cmp.values()),
                                'unequal': [k for k, v in rc_cmp.items() if not v['equal']],
                                'dev_report_truth_hashes_bound_by_reread_equal_published': all(rc_cmp.get(k, {}).get('equal') for k in dev_truth_keys),
                                'sealed_assessment_listed_with_export': sealed_listed},
         'dev_report_truth_files_restored': [r for r in files if r.startswith('exports/DEV_REPORT/') and Path(r).stem in ('income', 'occupation', 'sex', 'weights')]}
    for v in files.values():
        for src in v['pins']:
            key = src.split('.models.')[0] if src.startswith('MODEL_MANIFEST') else src
            d = B['by_pin_source'].setdefault(key, {'checked': 0, 'match': 0})
            d['checked'] += 1
            d['match'] += int(v['pins'][src] == v['sha256'])
    B['named_restored_groups'] = {k: B['by_pin_source'].get(k) for k in (
        'TRAINING_COMPLETE_LOCK.frozen_files_sha256', 'REPORTING_LOCK.reader_files_sha256',
        'EXTRACTION_SUMMARY.export_file_sha256', 'MODEL_MANIFEST', 'SELECTION_LOCK.output_file_sha256')}
    B['pass'] = bool(not mismatched and not unpinned and B['extraction_receipt']['equal_to_EXTRACTION_SUMMARY'] == len(rc_cmp)
                     and B['extraction_receipt']['dev_report_truth_hashes_bound_by_reread_equal_published']
                     and not B['dev_report_truth_files_restored'] and len(files) > 0)

    # ---- C. campaign records agree with the hashes computed here
    rr = V.jload(V.PRIV / 'RESTORE_RECEIPT.json')['files']
    adm = V.jload(V.RES / 'SOURCE_ADMISSION.json')
    C_ = {'restore_receipt_entries': len(rr), 'restore_receipt_equal': sum(rr[k]['sha256'] == mine.get(k) for k in rr),
          'restore_receipt_unequal_or_missing': [k for k in rr if rr[k]['sha256'] != mine.get(k)],
          'restored_files_not_in_receipt': sorted(set(mine) - set(rr)),
          'source_admission_restored_equal': sum(v['sha256'] == mine.get(k) for k, v in adm['restored_files'].items()),
          'source_admission_restored_entries': len(adm['restored_files']),
          'source_admission_pinned_docs_identical_flags': all(v['identical'] for v in adm['published_source']['documents'].values()),
          'source_admission_pinned_docs_hash_equal_mine': all(docs.get(k, {}).get('published_sha256') == v['published_sha256'] for k, v in adm['published_source']['documents'].items())}
    C_['pass'] = bool(C_['restore_receipt_equal'] == len(rr) and not C_['restored_files_not_in_receipt']
                      and C_['source_admission_restored_equal'] == C_['source_admission_restored_entries']
                      and C_['source_admission_pinned_docs_hash_equal_mine'])

    out = {'check': 'V1 Source identities', 'scope': 'hash replay (label-free); no model or reader computation',
           'published_commit': V.PUB_COMMIT, 'A_inherited_documents_vs_published_commit': A,
           'B_restored_files_vs_published_hashes': B, 'C_campaign_records_consistency': C_,
           'status': 'PASS' if (A['pass'] and B['pass'] and C_['pass']) else 'FAIL'}
    V.write_json(a.out, out)
    print({'status': out['status'], 'A': A['pass'], 'B': B['pass'], 'C': C_['pass'], 'extra': extra, 'mismatched': mismatched})


if __name__ == '__main__':
    main()
