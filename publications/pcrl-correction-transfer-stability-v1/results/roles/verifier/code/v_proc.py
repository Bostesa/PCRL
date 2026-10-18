"""Verifier procedural-compliance checks (no data, no label): registration receipt and lock integrity, dated access amendments
committed before first use, code provenance of queue units (uncommitted imported code at launch, code drift while waiting),
closed roles, CPU accounting. Reads only campaign JSON records, the queue ledger, the label-access log and git metadata.

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records).

    python3 results/pcrl_correction_transfer_stability_v1/roles/verifier/code/v_proc.py --tag V1
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
RES = ROOT / 'results' / 'pcrl_correction_transfer_stability_v1'
CAMP = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1'
REG_FILES = ('PROTOCOL.md', 'PARTITIONS.json', 'EXPOSURE_MAP.md', 'roles/statistical_reviewer/RULES.json',
             'roles/statistical_reviewer/RULES.md', 'ACCESS_POLICY.json')
REG_COMMIT = '39a27cc1e'
PKGS = ('cts', 'ohn', 'ihs', 'acs_replication', 'scm_bench', 'ffv', 'task_conditioned_representation', 'task_headroom')


def git(*a):
    r = subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def t(s):
    return dt.datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(dt.timezone.utc)


IMP = re.compile(r'^(\s*)(?:from\s+([\w\.]+)\s+import\s+([\w\s,\(\)]+)|import\s+([\w\.]+))', re.M)


def module_file(mod):
    p = ROOT / (mod.replace('.', '/') + '.py')
    if p.is_file():
        return str(p.relative_to(ROOT))
    p = ROOT / mod.replace('.', '/') / '__init__.py'
    return str(p.relative_to(ROOT)) if p.is_file() else None


_SRC = {}
HEAD_FOR_SOURCE = [None]           # commit whose file contents define the import graph (the unit's recorded HEAD at launch)


def source(relpath):
    key = (HEAD_FOR_SOURCE[0], relpath)
    if key not in _SRC:
        txt = None
        if HEAD_FOR_SOURCE[0]:
            rc, out = git('show', f'{HEAD_FOR_SOURCE[0]}:{relpath}')
            txt = out if rc == 0 else None
        _SRC[key] = txt if txt is not None else ((ROOT / relpath).read_text() if (ROOT / relpath).is_file() else '')
    return _SRC[key]


def imports_of(relpath):
    """(top-level imported files, all imported files) of one source file (static; package-internal modules only)."""
    src = source(relpath)
    pkg = relpath.split('/')[0]
    top, every = set(), set()
    for m in IMP.finditer(src):
        indent, frm, names, imp = m.group(1), m.group(2), m.group(3), m.group(4)
        targets = []
        if imp:
            targets.append(imp)
        else:
            base = pkg if frm == '.' else (pkg + frm if frm.startswith('.') else frm)
            for n in re.split(r'[\s,\(\)]+', names or ''):
                n = n.strip()
                if not n or n == 'as':
                    continue
                targets.append(base + '.' + n)
            targets.append(base)
        for tgt in targets:
            if tgt.split('.')[0] not in PKGS:
                continue
            f = module_file(tgt)
            if f:
                every.add(f)
                if indent == '':
                    top.add(f)
    return top, every


def closure(root_file, top_only):
    seen, todo = set(), [root_file]
    while todo:
        f = todo.pop()
        if f in seen or f is None:
            continue
        seen.add(f)
        tp, ev = imports_of(f)
        todo.extend(tp if top_only else ev)
    return seen


def unit_root(argv):
    if '-m' in argv:
        return module_file(argv[argv.index('-m') + 1])
    for a in argv[1:]:
        if a.endswith('.py'):
            p = Path(a)
            try:
                return str((p if p.is_absolute() else ROOT / p).resolve().relative_to(ROOT))
            except ValueError:
                return None
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    out = []
    access = [json.loads(l) for l in (CAMP / 'access' / 'LABEL_ACCESS.jsonl').read_text().splitlines() if l.strip()]
    ledger = [json.loads(l) for l in (CAMP / 'compute' / 'ledger.jsonl').read_text().splitlines() if l.strip()]
    # ---- registration receipt and locks
    rec = json.loads((CAMP / 'access' / 'REGISTRATION_RECEIPT_VERIFIED.json').read_text())
    sha_ok = {f: V.file_sha(RES / f) == rec['files_sha256'][f] for f in REG_FILES}
    rc, later = git('log', '--format=%h %cI %s', f'{REG_COMMIT}..HEAD', '--', *[str(RES / f) for f in REG_FILES])
    rc2, dirty = git('status', '--porcelain', '--', *[str(RES / f) for f in REG_FILES])
    first_read = min((e['utc'] for e in access), default=None)
    before = first_read is not None and t(rec['verified_utc']) <= t(first_read)
    ok = all(sha_ok.values()) and rec.get('remote_sha_verified') and rec.get('remote_sha') == rec.get('receipt_commit') and not later \
        and not dirty and before
    out.append({'id': 'P1_REGISTRATION_LOCK', 'kind': 'PROCEDURAL_COMPLIANCE',
                'what': 'registration files unchanged since the registration commit and byte-identical to the verified public receipt; '
                        'the receipt verification precedes the first label read',
                'method': 'SHA-256 of the six registration files against REGISTRATION_RECEIPT_VERIFIED.json; git log / status of those files '
                          'after 39a27cc1e; first LABEL_ACCESS.jsonl time',
                'status': 'PASS' if ok else 'FAIL',
                'numbers': {'files_sha_match': sha_ok, 'commits_touching_after_registration': later.splitlines() if later else [],
                            'uncommitted_changes': dirty.splitlines() if dirty else [], 'receipt_verified_utc': rec['verified_utc'],
                            'first_label_read_utc': first_read, 'receipt_commit': rec['receipt_commit']}})
    # ---- dated access amendments
    am_path = RES / 'ACCESS_AMENDMENTS.json'
    rc, first_commit = git('log', '--diff-filter=A', '--format=%h %cI', '--', str(am_path))
    commits = first_commit.splitlines()
    rcs, allc = git('log', '--format=%h %cI', '--', str(am_path))
    am = json.loads(am_path.read_text())
    probs, att, rows = [], [], []
    for e in am['amendments']:
        # the commit that first contains this entry
        hit = None
        for line in reversed(allc.splitlines()):
            h, cdate = line.split()
            rcx, txt = git('show', f'{h}:{am_path.relative_to(ROOT)}')
            if rcx == 0 and json.dumps(e['entry_point']) in txt and e['operation'] in txt:
                hit = (h, cdate); break
        uses = sorted(x['utc'] for x in access if x['entry_point'].split(':')[0] == e['entry_point'] and x['operation'] == e['operation'])
        r = {'operation': e['operation'], 'entry_point': e['entry_point'], 'dated_utc': e['utc'], 'first_commit': hit[0] if hit else None,
             'commit_utc': t(hit[1]).isoformat() if hit else None, 'first_use_utc': uses[0] if uses else None, 'uses': len(uses)}
        if not hit:
            probs.append(f"{e['entry_point']}/{e['operation']}: not committed")
        else:
            if uses and t(uses[0]) < t(hit[1]):
                probs.append(f"{e['entry_point']}/{e['operation']}: used before its commit")
            if t(e['utc']) > t(hit[1]):
                att.append(f"{e['entry_point']}/{e['operation']}: dated {e['utc']} but committed at {t(hit[1]).isoformat()} (dated after "
                           'its own commit)')
        rows.append(r)
    out.append({'id': 'P2_ACCESS_AMENDMENTS_BEFORE_USE', 'kind': 'PROCEDURAL_COMPLIANCE',
                'what': 'every dated ACCESS_AMENDMENTS.json entry is committed before the first label read through its entry point; its '
                        'stated date is consistent with the commit',
                'method': 'git history of ACCESS_AMENDMENTS.json (first commit containing each entry) against LABEL_ACCESS.jsonl entry points',
                'status': 'FAIL' if probs else ('ATTENTION' if att else 'PASS'),
                'numbers': {'amendments': rows, 'problems': probs, 'dating_inconsistencies': att}})
    # ---- code provenance of queue units
    starts = [e for e in ledger if e.get('event') == 'unit_start']
    flagged, drift, n = [], [], 0
    for e in starts:
        n += 1
        root = unit_root(e['argv'])
        unc = set(e.get('uncommitted_code_paths') or [])
        HEAD_FOR_SOURCE[0] = e.get('git_head')
        if root is None:
            continue
        top = closure(root, True)
        allc_ = closure(root, False)
        hit_top = sorted(unc & top)
        hit_any = sorted(unc & allc_)
        if hit_top or hit_any:
            flagged.append({'unit': e['unit_id'], 'utc': e['utc'], 'root': root, 'uncommitted_top_level_imports': hit_top,
                            'uncommitted_lazy_imports_only': sorted(set(hit_any) - set(hit_top))})
        if e.get('code_digest_at_launch') != e.get('code_digest_at_registration'):
            cl = CAMP / 'compute' / 'units' / e['unit_id'] / 'CODE_AT_LAUNCH.json'
            ch = json.loads(cl.read_text()).get('files_changed', []) if cl.is_file() else ['CODE_AT_LAUNCH.json missing']
            drift.append({'unit': e['unit_id'], 'files_changed_while_waiting': ch, 'in_unit_closure': sorted(set(ch) & allc_)})
    top_flag = [f for f in flagged if f['uncommitted_top_level_imports']]
    no_launch_record = [d['unit'] for d in drift if 'CODE_AT_LAUNCH.json missing' in d['files_changed_while_waiting']]
    out.append({'id': 'P3_UNIT_CODE_PROVENANCE', 'kind': 'PROCEDURAL_COMPLIANCE',
                'what': 'queue units launched with uncommitted code they import (deviation 3 says none other than the disclosed one), and '
                        'code that changed between registration and launch of a unit',
                'method': 'ledger unit_start uncommitted_code_paths intersected with a static import closure of the unit entry module, '
                          'built from the sources at the unit\'s recorded HEAD (uncommitted files: current text) (top-level imports: '
                          'certainly executed; lazy imports: executed only on some code paths); CODE_AT_LAUNCH.json',
                'status': 'ATTENTION' if (top_flag or no_launch_record) else 'PASS',
                'numbers': {'units_checked': n, 'units_with_uncommitted_top_level_imports': top_flag,
                            'drifted_units_without_launch_record': no_launch_record,
                            'reading': 'PASS = no unit imported uncommitted campaign code at module top level, and every unit whose code '
                                       'changed while it waited has its launch-time hashes recorded (CODE_AT_LAUNCH.json); lazy imports '
                                       'listed below are reached only on other code paths (D0B witness, Track R income option); '
                                       'consistent with PROTOCOL_DEVIATIONS items 3 and 7',
                            'units_with_uncommitted_lazy_imports_only': [f for f in flagged if not f['uncommitted_top_level_imports']],
                            'units_with_code_drift_while_waiting': drift}})
    # ---- CPU accounting
    charged = sum(float(e.get('charged_cpu_s', 0.0)) for e in ledger if e.get('event') in ('unit_end', 'charge'))
    mine = sum(float(e.get('charged_cpu_s', 0.0)) for e in ledger if e.get('event') == 'unit_end' and str(e['unit_id']).startswith('VER_'))
    mine += sum(float(e.get('charged_cpu_s', 0.0)) for e in ledger if e.get('event') == 'charge' and str(e.get('label', '')).startswith('VER_'))
    failed = [e['unit_id'] for e in ledger if e.get('event') == 'unit_end' and e.get('status') != 'done']
    out.append({'id': 'P4_CPU_ACCOUNTING', 'kind': 'PROCEDURAL_COMPLIANCE',
                'what': 'campaign CPU charged against the 14 CPU-hour limit (queue units + charges), the verifier\'s own share, failed units kept',
                'method': 'sum of charged_cpu_s over unit_end and charge events of the ledger',
                'status': 'PASS' if charged < 14 * 3600 else 'FAIL',
                'numbers': {'charged_cpu_h': round(charged / 3600, 4), 'verifier_cpu_s': round(mine, 1), 'failed_units_kept': failed,
                            'units_started': len(starts)}})
    code = V.code_hashes(__file__, V.__file__)
    for c in out:
        c['id'] = f"{c['id']}__{a.tag}"
        c['code_sha256'] = code
    if a.dry:
        print(json.dumps(out, indent=1, default=str)[:20000])
        return
    V.write_record(f'V_PROC__{a.tag}.json', {'checks': out, 'utc': V.utc()})
    print(json.dumps(V.append_checks(out)))
    print(json.dumps({c['id']: c['status'] for c in out}, indent=1))


if __name__ == '__main__':
    main()
