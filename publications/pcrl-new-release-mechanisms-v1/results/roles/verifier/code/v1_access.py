"""V1 source admission and role use: audit of the append-only label-access log against the verified registration receipt, the
registered access policy, the mechanism cards (git history), the shared queue ledger (unit intervals, code at launch) and the
admitted row sets. Label-free: the log, the receipt, the policy, git and the label-free features are read; no label is read.

    python3 .../v1_access.py --tag V1            # real audit (writes V1.ACCESS.* checks)
    python3 .../v1_access.py --selftest          # invented logs only: every rule must flag its planted violation

Rules audited (ACCESS_POLICY.json, PROTOCOL.md, AGENT_CONTEXT.md):
 R1  the registration receipt is verified (remote tip = receipt commit at verification, published bytes = committed research bytes,
     research commit an ancestor of HEAD) and the registration files are still committed and unchanged;
 R2  no label read before the receipt's verification time;
 R3  no ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT / SEALED label read;
 R4  every read is a registered operation, role, field set and authorized entry point (committed amendments only, before use);
 R5  MODEL_SELECT labels only through nrm.evaluate / nrm.screen / nrm.devreaders / the verifier's code (never a fitting module);
 R6  every 'fit' / 'calibrate' read logs a card under results/<campaign>/cards/ whose commit precedes the read, whose bytes at that
     commit hash to the logged SHA-256, which is an ancestor of HEAD and is unchanged now;
 R7  every read carries a queue unit id whose interval contains the read (null unit ids are listed as ATTENTION);
 R8  the logged row-set hash, people and households equal the admitted role (label-free rebuild);
 R9  units that read labels ran committed code: the entry module's package files hashed at launch equal the blobs at the launch HEAD.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

CLOSED = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT', 'SEALED_ASSESSMENT')
MS_ENTRY = ('nrm.evaluate', 'nrm.screen', 'nrm.devreaders', f'results/{V.CAMPAIGN}/roles/verifier/code/')
FIT_ENTRY = ('nrm.fa', 'nrm.fb', 'nrm.fc', 'nrm.ctl')
REG_FILES = ('PROTOCOL.md', 'EXPOSURE_MAP.md', 'ACCESS_POLICY.json', 'SOURCE_ADMISSION.json', 'READER_SPLITS.json',
             'FAILURE_MAP.md', 'roles/statistical_reviewer/RULES.md', 'roles/statistical_reviewer/RULES.json')
PUB_NAME = 'pcrl-new-release-mechanisms-v1'


def entry_ok(mod, allowed):
    return any(mod == e or mod.startswith(e if e.endswith('/') else e + '.') or (e.endswith('/') and mod.startswith(e))
               for e in allowed)


def audit(events, receipt_utc, policy, amendments, card_lookup, units, role_rows, code_check):
    """Pure audit over in-memory inputs. card_lookup(card, commit) -> {'commit_utc', 'sha_at_commit', 'ancestor', 'sha_now'};
    units[unit_id] = {'start', 'end', 'stage'}; role_rows[role] = (rows_sha256, people, households);
    code_check(unit_id, entry_module) -> list of files that differed from the launch HEAD (None if not checkable)."""
    f = []

    def flag(sev, rule, i, detail):
        f.append({'severity': sev, 'rule': rule, 'event': i, 'detail': detail})
    for i, e in enumerate(events):
        t = V.parse_utc(e['utc'])
        role, op = e.get('role'), e.get('operation')
        fields = set(e.get('fields') or [])
        mod = str(e.get('entry_point', '?')).split(':')[0]
        if receipt_utc is None or t < receipt_utc:
            flag('FAIL', 'R2_read_before_receipt', i, e['utc'])
        if role in CLOSED:
            flag('FAIL', 'R3_closed_role', i, role)
        pol = policy.get(op)
        if pol is None:
            flag('FAIL', 'R4_unregistered_operation', i, op)
        else:
            if role not in pol['roles']:
                flag('FAIL', 'R4_role_not_allowed', i, f'{op} {role}')
            if fields - set(pol['fields']):
                flag('FAIL', 'R4_field_not_allowed', i, f'{op} {sorted(fields)}')
            allowed = list(pol['entry_points']) + [a['entry_point'] for a in amendments
                                                   if a['operation'] == op and a['commit_utc'] is not None and a['commit_utc'] < t]
            if not entry_ok(mod, allowed):
                flag('FAIL', 'R4_entry_point_not_authorized', i, f'{mod} for {op}')
        if role == 'MODEL_SELECT' and fields:
            if not entry_ok(mod, MS_ENTRY) or entry_ok(mod, FIT_ENTRY):
                flag('FAIL', 'R5_model_select_by_other_entry', i, mod)
        if op in ('fit', 'calibrate'):
            card = e.get('card')
            if not card or not str(card).startswith(f'results/{V.CAMPAIGN}/cards/'):
                flag('FAIL', 'R6_card_missing_or_outside_cards', i, card)
            else:
                info = card_lookup(card, e.get('card_commit'))
                if info is None or info.get('commit_utc') is None:
                    flag('FAIL', 'R6_card_commit_unknown', i, f'{card} {e.get("card_commit")}')
                else:
                    if not info['commit_utc'] < t:
                        flag('FAIL', 'R6_card_committed_after_read', i, f'{card} commit {info["commit_utc"].isoformat()}')
                    if info['sha_at_commit'] != e.get('card_sha256'):
                        flag('FAIL', 'R6_card_bytes_differ_from_logged_sha', i, card)
                    if not info['ancestor']:
                        flag('FAIL', 'R6_card_commit_not_ancestor_of_HEAD', i, e.get('card_commit'))
                    if info['sha_now'] != e.get('card_sha256'):
                        flag('FAIL', 'R6_card_changed_after_use', i, card)
                    if e.get('card_commit_utc') and V.parse_utc(e['card_commit_utc']) != info['commit_utc']:
                        flag('FAIL', 'R6_logged_card_time_differs_from_git', i, card)
        uid = e.get('unit_id')
        if uid is None:
            flag('ATTENTION', 'R7_null_unit_id', i, f'{mod} {op} {role}')
        elif uid not in units:
            flag('FAIL', 'R7_unit_not_in_queue_ledger', i, uid)
        else:
            u = units[uid]
            if not (u['start'] <= t and (u['end'] is None or t <= u['end'])):
                flag('FAIL', 'R7_read_outside_unit_interval', i, uid)
        rr = role_rows.get(role)
        if rr is None or (e.get('rows_sha256'), e.get('people'), e.get('households')) != rr:
            flag('FAIL', 'R8_row_set_differs', i, f'{role} {e.get("rows_sha256")} {e.get("people")} {e.get("households")}')
        if uid is not None and uid in units:
            bad = code_check(uid, mod)
            if bad:
                flag('FAIL', 'R9_uncommitted_code_at_launch', i, {'unit': uid, 'files': bad})
            elif bad is None:
                flag('ATTENTION', 'R9_code_not_checkable', i, uid)
    return f


# ---------------------------------------------------------------- real inputs
def _units():
    out = {}
    for e in V.jsonl(V.QUEUE_LEDGER):
        if e.get('event') == 'unit_start':
            out[e['unit_id']] = {'start': V.parse_utc(e['utc']), 'end': None, 'stage': e['stage'], 'git_head': e.get('git_head'),
                                 'argv': e.get('argv')}
        elif e.get('event') == 'unit_end' and e['unit_id'] in out:
            out[e['unit_id']]['end'] = V.parse_utc(e['utc'])
            out[e['unit_id']]['status'] = e.get('status')
    return out


def _module_files(mod):
    """Repository files of the entry module's package (a verifier script: the script and vlib)."""
    if mod.startswith('results/'):
        return [mod, f'results/{V.CAMPAIGN}/roles/verifier/code/vlib.py']
    parts = mod.split('.')
    pkg = parts[0] + '/' + parts[1] if len(parts) > 1 else parts[0]
    files = ['nrm/common.py', 'nrm/release.py']
    p = V.ROOT / pkg
    if p.is_dir():
        files += [str(q.relative_to(V.ROOT)) for q in sorted(p.rglob('*.py')) if '__pycache__' not in q.parts]
    elif (V.ROOT / (pkg + '.py')).is_file():
        files.append(pkg + '.py')
    return files


def _code_checker(units):
    cache = {}

    def check(uid, mod):
        key = (uid, mod)
        if key in cache:
            return cache[key]
        p = V.PRIV / 'compute' / 'units' / uid / 'CODE_AT_START.json'
        launch = V.PRIV / 'compute' / 'units' / uid / 'CODE_AT_LAUNCH.json'
        if not p.is_file():
            cache[key] = None
            return None
        rec = json.loads((launch if launch.is_file() else p).read_text())
        head = rec.get('git_head') or units[uid].get('git_head')
        bad = []
        for fpath in _module_files(mod):
            h = rec['files'].get(fpath)
            if h is None:
                continue
            if V.git_blob_sha256(head, fpath) != h:
                bad.append(fpath)
        cache[key] = bad
        return bad
    return check


def _card_lookup():
    cache = {}

    def look(card, commit):
        key = (card, commit)
        if key not in cache:
            if not commit:
                cache[key] = None
            else:
                cache[key] = {'commit_utc': V.commit_time(commit), 'sha_at_commit': V.git_blob_sha256(commit, card),
                              'ancestor': V.is_ancestor(commit), 'sha_now': V.file_sha(V.ROOT / card) if (V.ROOT / card).is_file() else None}
        return cache[key]
    return look


def _amendments():
    p = V.RESULTS / 'ACCESS_AMENDMENTS.json'
    if not p.is_file():
        return []
    rc, out, _ = V.git('log', '--format=%H %cI', '--', str(p.relative_to(V.ROOT)))
    first = out.splitlines()[-1].split() if out else None
    amends = json.loads(p.read_text()).get('amendments', [])
    res = []
    for a in amends:
        # an amendment counts from the first commit whose version of the file contains it
        when = None
        for line in reversed(out.splitlines()):
            c, t = line.split()
            blob = V.git('show', f'{c}:{p.relative_to(V.ROOT)}')[1]
            try:
                if any(x.get('operation') == a.get('operation') and x.get('entry_point') == a.get('entry_point')
                       for x in json.loads(blob).get('amendments', [])):
                    when = V.parse_utc(t)
                    break
            except json.JSONDecodeError:
                continue
        res.append({'operation': a.get('operation'), 'entry_point': a.get('entry_point'), 'commit_utc': when})
    _ = first
    return res


def receipt_checks():
    """R1: own re-verification of the receipt record (git; the remote tip is read with ls-remote for information)."""
    out = {'record_exists': V.RECEIPT.is_file()}
    if not out['record_exists']:
        return out, None
    r = json.loads(V.RECEIPT.read_text())
    out['remote_sha_verified'] = bool(r.get('remote_sha_verified'))
    out['remote_sha_equals_receipt_commit'] = r.get('remote_sha') == r.get('receipt_commit')
    out['receipt_commit'] = r.get('receipt_commit'); out['research_commit'] = r.get('research_commit')
    out['verified_utc'] = r.get('verified_utc')
    files_now = {f: V.file_sha(V.RESULTS / f) if (V.RESULTS / f).is_file() else None for f in REG_FILES}
    out['files_sha256_equal_now'] = r.get('files_sha256') == files_now
    out['files_committed_unchanged'] = all(V.committed_unchanged(V.RESULTS / f) for f in REG_FILES)
    out['research_commit_ancestor_of_HEAD'] = V.is_ancestor(r['research_commit'])
    rt = V.commit_time(r['research_commit'])
    out['research_commit_utc'] = rt.isoformat() if rt else None
    out['research_commit_before_verification'] = bool(rt and rt <= V.parse_utc(r['verified_utc']))
    at_research = {f: V.git_blob_sha256(r['research_commit'], f'results/{V.CAMPAIGN}/{f}') for f in REG_FILES}
    out['research_commit_bytes_equal_receipt'] = at_research == r.get('files_sha256')
    pub = {}
    have = V.git('cat-file', '-e', f'{r["receipt_commit"]}^{{commit}}')[0] == 0
    out['receipt_commit_available_locally'] = have
    if have:
        for f in REG_FILES:
            pub[f] = V.git_blob_sha256(r['receipt_commit'], f'publications/{PUB_NAME}/results/{f}')
        # the publication copy is sanitized: equal bytes are expected for these registration files (receipt rule)
        out['published_bytes_equal_research'] = {f: pub[f] == at_research[f] for f in REG_FILES}
        rcv = V.commit_time(r['receipt_commit'])
        out['receipt_commit_utc'] = rcv.isoformat() if rcv else None
    rc, ls, err = V.git('ls-remote', 'https://github.com/Bostesa/PCRL.git', f'refs/heads/publication/{PUB_NAME}')
    out['remote_tip_now'] = ls.split()[0] if rc == 0 and ls else f'unavailable: {err[-200:]}'
    out['attempts'] = [{'outcome': a.get('outcome'), 'verified_utc': a.get('verified_utc'), 'problems': a.get('problems')}
                       for a in V.jsonl(V.ACCESS_DIR / 'RECEIPT_ATTEMPTS.jsonl')]
    return out, V.parse_utc(r['verified_utc']) if r.get('remote_sha_verified') else None


def policy_from_json():
    pj = json.loads((V.RESULTS / 'ACCESS_POLICY.json').read_text())
    return {op: {'roles': v['roles'], 'fields': v['fields'], 'entry_points': v['entry_points'], 'card': bool(v.get('card'))}
            for op, v in pj['operations'].items()}, pj


def policy_matches_code(pol):
    sys.path.insert(0, str(V.ROOT))
    import nrm.common as C
    diffs = []
    for op, p in C.POLICY.items():
        q = pol.get(op)
        if q is None or set(p['roles']) != set(q['roles']) or set(p['fields']) != set(q['fields']) or bool(p.get('card')) != q['card'] \
                or set(C.ENTRY_POINTS[op]) != set(q['entry_points']):
            diffs.append(op)
    if set(pol) != set(C.POLICY):
        diffs.append('operation sets differ')
    return diffs


def role_rows():
    sys.path.insert(0, str(V.ROOT))
    import nrm.common as C
    out = {}
    for r in ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'):
        F = C.features(r)
        out[r] = (V.arr_sha(np.asarray(F['row_id'])), int(len(F['row_id'])), int(len(np.unique(F['unit']))))
    return out


def run(tag):
    rec, rutc = receipt_checks()
    pol, pj = policy_from_json()
    pdiff = policy_matches_code(pol)
    events = V.jsonl(V.ACCESS_LOG)
    units = _units()
    findings = audit(events, rutc, pol, _amendments(), _card_lookup(), units, role_rows(), _code_checker(units))
    fails = [x for x in findings if x['severity'] == 'FAIL']
    atts = [x for x in findings if x['severity'] == 'ATTENTION']
    by_rule = {}
    for x in findings:
        by_rule.setdefault(f"{x['severity']}|{x['rule']}", []).append(x['event'])
    summary = {'n_events': len(events), 'first_utc': events[0]['utc'] if events else None, 'last_utc': events[-1]['utc'] if events else None,
               'by_operation_role': {}, 'units_with_reads': sorted({e.get('unit_id') or 'null' for e in events})}
    for e in events:
        k = f"{e.get('operation')}|{e.get('role')}|{','.join(e.get('fields') or [])}|{str(e.get('entry_point')).split(':')[0]}"
        summary['by_operation_role'][k] = summary['by_operation_role'].get(k, 0) + 1
    rec_ok = bool(rec.get('record_exists') and rec.get('remote_sha_verified') and rec.get('remote_sha_equals_receipt_commit')
                  and rec.get('files_sha256_equal_now') and rec.get('files_committed_unchanged') and rec.get('research_commit_ancestor_of_HEAD')
                  and rec.get('research_commit_bytes_equal_receipt') and rec.get('research_commit_before_verification')
                  and all((rec.get('published_bytes_equal_research') or {'x': False}).values()))
    checks = [
        {'id': f'V1.RECEIPT__{tag}', 'what': 'registration receipt re-verified with own git calls (R1): record, remote tip at verification, '
         'files unchanged and committed, research commit ancestor and earlier, published copies equal to research bytes',
         'inputs': V.inputs_sha([V.RECEIPT, V.ACCESS_DIR / 'RECEIPT_ATTEMPTS.jsonl'] + [V.RESULTS / f for f in REG_FILES]),
         'result': rec, 'numerical_status': 'NOT_APPLICABLE', 'procedural_status': 'PASS' if rec_ok else 'FAIL'},
        {'id': f'V1.ACCESS_POLICY_CODE__{tag}', 'what': 'nrm.common POLICY / ENTRY_POINTS equal the registered ACCESS_POLICY.json',
         'inputs': V.inputs_sha([V.RESULTS / 'ACCESS_POLICY.json', V.ROOT / 'nrm' / 'common.py']), 'result': {'differences': pdiff},
         'numerical_status': 'NOT_APPLICABLE', 'procedural_status': 'PASS' if not pdiff else 'FAIL'},
        {'id': f'V1.ACCESS_LOG__{tag}', 'what': 'label-access log audit R2-R9 (receipt order, closed roles, policy, MODEL_SELECT entry points, '
         'cards committed before fitting reads, unit intervals, row sets, committed code at launch)',
         'inputs': {**V.inputs_sha([V.ACCESS_LOG, V.QUEUE_LEDGER, V.RECEIPT]), 'n_events': len(events)},
         'result': {'summary': summary, 'findings_by_rule': by_rule, 'fail_examples': fails[:20], 'attention_examples': atts[:20],
                    'receipt_verified_utc': rutc.isoformat() if rutc else None},
         'numerical_status': 'NOT_APPLICABLE',
         'procedural_status': 'FAIL' if fails else ('ATTENTION' if atts else 'PASS')},
        {'id': f'V1.ROW_SETS__{tag}', 'what': 'logged row-set hashes, people and households equal the admitted roles (label-free rebuild, R8)',
         'inputs': V.inputs_sha([V.ACCESS_LOG]),
         'result': {'n_events': len(events), 'row_set_failures': [x for x in findings if x['rule'] == 'R8_row_set_differs'][:10]},
         'numerical_status': 'FAIL' if any(x['rule'] == 'R8_row_set_differs' for x in findings) else 'PASS',
         'procedural_status': 'NOT_APPLICABLE'},
    ]
    print(json.dumps(V.append_checks(checks), indent=1))
    print(json.dumps({'receipt_ok': rec_ok, 'policy_diffs': pdiff, 'by_rule': {k: len(v) for k, v in by_rule.items()}, 'summary': summary},
                     indent=1, default=V._jd)[:6000])
    return 0


# ---------------------------------------------------------------- invented-data selftest of the audit rules
def selftest():
    import datetime as dt
    T = lambda m: dt.datetime(2026, 10, 20, 16, m, tzinfo=dt.timezone.utc)  # noqa: E731
    pol = {'fit': {'roles': ['MODEL_FIT'], 'fields': ['income', 'occupation', 'sex'], 'entry_points': ['nrm.fa', 'nrm.fb'], 'card': True},
           'benchmark': {'roles': ['MODEL_SELECT'], 'fields': ['income', 'occupation', 'sex'], 'entry_points': ['nrm.evaluate'], 'card': False},
           'verify': {'roles': ['MODEL_FIT', 'MODEL_SELECT'], 'fields': ['income', 'occupation', 'sex'],
                      'entry_points': [f'results/{V.CAMPAIGN}/roles/verifier/code/'], 'card': False}}
    rows = {'MODEL_FIT': ('h1', 10, 5), 'MODEL_SELECT': ('h2', 8, 4), 'DEV_REPORT': ('h3', 6, 3)}
    card = f'results/{V.CAMPAIGN}/cards/FA_CARD_v1.md'
    cards = {(card, 'c1'): {'commit_utc': T(5), 'sha_at_commit': 's1', 'ancestor': True, 'sha_now': 's1'},
             (card, 'c2'): {'commit_utc': T(30), 'sha_at_commit': 's2', 'ancestor': True, 'sha_now': 's1'}}
    units = {'U1': {'start': T(10), 'end': T(20), 'stage': 'fit'}, 'U2': {'start': T(10), 'end': T(50), 'stage': 'evaluate'}}
    good_fit = {'utc': T(12).isoformat(), 'unit_id': 'U1', 'role': 'MODEL_FIT', 'fields': ['income', 'sex'], 'operation': 'fit',
                'entry_point': 'nrm.fa.fit:main', 'rows_sha256': 'h1', 'people': 10, 'households': 5, 'card': card,
                'card_commit': 'c1', 'card_sha256': 's1', 'card_commit_utc': T(5).isoformat()}
    good_ms = {'utc': T(12).isoformat(), 'unit_id': 'U2', 'role': 'MODEL_SELECT', 'fields': ['income', 'occupation', 'sex'],
               'operation': 'benchmark', 'entry_point': 'nrm.evaluate:ms', 'rows_sha256': 'h2', 'people': 8, 'households': 4}
    plant = {
        'R2_read_before_receipt': {**good_ms, 'utc': T(1).isoformat()},
        'R3_closed_role': {**good_ms, 'role': 'DEV_REPORT', 'rows_sha256': 'h3', 'people': 6, 'households': 3},
        'R4_entry_point_not_authorized': {**good_ms, 'entry_point': 'nrm.release:load'},
        'R5_model_select_by_other_entry': {**good_ms, 'operation': 'verify', 'entry_point': 'nrm.fa.fit:main'},
        'R6_card_committed_after_read': {**good_fit, 'card_commit': 'c2', 'card_sha256': 's2', 'card_commit_utc': T(30).isoformat()},
        'R6_card_missing_or_outside_cards': {**good_fit, 'card': None},
        'R7_read_outside_unit_interval': {**good_fit, 'utc': T(25).isoformat()},
        'R7_null_unit_id': {**good_ms, 'unit_id': None},
        'R8_row_set_differs': {**good_ms, 'people': 7},
        'R9_uncommitted_code_at_launch': {**good_ms, 'unit_id': 'U2', 'entry_point': 'nrm.evaluate:dirty'},
    }

    def code_check(uid, mod):
        return ['nrm/evaluate.py'] if mod == 'nrm.evaluate' and uid == 'U2' and code_check.dirty else []
    ok = {}
    code_check.dirty = False
    base = audit([good_fit, good_ms], T(2), pol, [], lambda c, k: cards.get((c, k)), units, rows, code_check)
    ok['clean_log_no_findings'] = base == []
    for rule, ev in plant.items():
        code_check.dirty = rule.startswith('R9')
        fs = audit([good_fit, ev], T(2), pol, [], lambda c, k: cards.get((c, k)), units, rows, code_check)
        ok[f'flags_{rule}'] = any(x['rule'] == rule for x in fs)
    ok['all_ok'] = all(ok.values())
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag'); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        chk = {'id': f'V1.ACCESS_AUDIT_SELFTEST__{a.tag or "V1"}', 'what': 'access-log audit rules on invented logs: a clean log gives no '
               'finding and every planted violation (R2-R9) is flagged', 'inputs': V.inputs_sha([Path(__file__)]), 'result': ok,
               'numerical_status': 'PASS' if ok['all_ok'] else 'FAIL', 'procedural_status': 'NOT_APPLICABLE'}
        print(json.dumps(V.append_checks([chk]), indent=1)); print(json.dumps(ok, indent=1))
        return 0 if ok['all_ok'] else 1
    return run(a.tag)


if __name__ == '__main__':
    sys.exit(main())
