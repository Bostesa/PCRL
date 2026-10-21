"""V1 access and registration: audit of the append-only label-access log against the verified registration receipt, the
registered access policy, the mechanism cards (git history), the shared queue ledger (unit intervals, main module, code at launch)
and the admitted row sets; own re-verification of the registration receipt (git ls-remote of the publication branch on the GitHub
remote equals the receipt commit; published registration files equal the committed research bytes). Label-free: the log, the
receipt, the policy, git and the label-free admitted features (frc.common.features) are read; no label is read.

    python3 .../v1_access.py --tag V1            # real audit (writes V1.* checks)
    python3 .../v1_access.py --selftest          # invented logs only: every rule must flag its planted violation

Rules audited (ACCESS_POLICY.json, PROTOCOL.md, AGENT_CONTEXT.md):
 R1  receipt: record verified; own ls-remote of publication/pcrl-finite-routing-comparison-v1 equals the receipt commit; the
     registration files are committed, unchanged and equal to the receipt hashes and to the published copies; the research commit
     is an ancestor of HEAD and earlier than the verification;
 R2  no label read before the receipt's verified_utc;
 R3  no ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT / SEALED label read;
 R4  every read is a registered operation, role, field set and authorized entry point (committed amendments only, before use);
 R5  MODEL_SELECT labels only through frc.evaluate / frc.devreaders / frc.readercustody / the verifier's code, and never with a
     fitting module (frc.fa / frc.ctl / frc.fs) on the logged module chain;
 R6  every 'fit' / 'calibrate' read logs a card under results/<campaign>/cards/ whose commit precedes the read, whose bytes at that
     commit hash to the logged SHA-256, which is an ancestor of HEAD and is unchanged now;
 R7  every read carries a queue unit id whose interval contains the read;
 R8  the logged row-set hash, people and households equal the admitted role (label-free rebuild);
 R9  units that read labels ran committed code (package files hashed at launch equal the blobs at the launch HEAD);
 R10 the unit's main module is the operation's own entry point; no 'fit'-stage unit read MODEL_SELECT labels.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

CLOSED = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT', 'SEALED_ASSESSMENT')
VERIFY_EP = f'results/{V.CAMPAIGN}/roles/verifier/code/'
MS_ENTRY = ('frc.evaluate', 'frc.devreaders', 'frc.readercustody', VERIFY_EP)
FIT_ENTRY = ('frc.fa', 'frc.ctl', 'frc.fs')
OP_MAIN = {'benchmark': ('frc.evaluate',), 'readers': ('frc.devreaders', 'frc.readercustody'), 'verify': (VERIFY_EP,),
           'fit': FIT_ENTRY, 'calibrate': FIT_ENTRY}
REG_FILES = ('PROTOCOL.md', 'EXPOSURE_MAP.md', 'ACCESS_POLICY.json', 'SOURCE_ADMISSION.json', 'READER_SPLITS.json',
             'FAILURE_MAP.md', 'roles/statistical_reviewer/RULES.md', 'roles/statistical_reviewer/RULES.json')


def entry_ok(mod, allowed):
    mod = str(mod)
    return any(mod == e or mod.startswith(e if e.endswith('/') else e + '.') or (e.endswith('/') and mod.startswith(e))
               for e in allowed)


def audit(events, receipt_utc, policy, amendments, card_lookup, units, role_rows, code_check):
    """Pure audit over in-memory inputs (see the module docstring for the rules)."""
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
        amended = [a['entry_point'] for a in amendments if a['operation'] == op and a['commit_utc'] is not None
                   and a['commit_utc'] < t]
        if pol is None:
            flag('FAIL', 'R4_unregistered_operation', i, op)
        else:
            if role not in pol['roles']:
                flag('FAIL', 'R4_role_not_allowed', i, f'{op} {role}')
            if fields - set(pol['fields']):
                flag('FAIL', 'R4_field_not_allowed', i, f'{op} {sorted(fields)}')
            if not entry_ok(mod, list(pol['entry_points']) + amended):
                flag('FAIL', 'R4_entry_point_not_authorized', i, f'{mod} for {op}')
        if role == 'MODEL_SELECT' and fields:
            ms_amended = [a['entry_point'] for a in amendments if a['operation'] in ('benchmark', 'readers', 'verify')
                          and a['commit_utc'] is not None and a['commit_utc'] < t]
            if not entry_ok(mod, list(MS_ENTRY) + ms_amended) or entry_ok(mod, FIT_ENTRY):
                flag('FAIL', 'R5_model_select_by_other_entry', i, mod)
            chain = e.get('module_chain') or []
            if any(entry_ok(m, FIT_ENTRY) for m in chain):
                flag('FAIL', 'R5_fitting_module_on_chain', i, chain)
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
            flag('FAIL', 'R7_null_unit_id', i, f'{mod} {op} {role}')
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
            main_mod, stage = units[uid].get('main'), units[uid].get('stage')
            want = tuple(OP_MAIN.get(op, ())) + tuple(amended)
            if main_mod is not None and want and not entry_ok(main_mod, want):
                flag('FAIL', 'R10_unit_main_module_differs_from_operation', i, {'unit': uid, 'main': main_mod, 'operation': op})
            if role == 'MODEL_SELECT' and fields and stage == 'fit':
                flag('FAIL', 'R10_fit_stage_unit_read_model_select', i, uid)
    return f


def main_module(argv):
    if not argv:
        return None
    if '-m' in argv:
        i = argv.index('-m')
        return argv[i + 1] if i + 1 < len(argv) else None
    for x in argv[1:]:
        if x.endswith('.py'):
            p = Path(x)
            p = p if p.is_absolute() else V.ROOT / p
            try:
                return str(p.resolve().relative_to(V.ROOT))
            except ValueError:
                return str(p)
    return None


# ---------------------------------------------------------------- real inputs
def _units():
    out = {}
    for e in V.jsonl(V.QUEUE_LEDGER):
        if e.get('event') == 'unit_start':
            out[e['unit_id']] = {'start': V.parse_utc(e['utc']), 'end': None, 'stage': e['stage'], 'git_head': e.get('git_head'),
                                 'argv': e.get('argv'), 'main': main_module(e.get('argv')),
                                 'uncommitted_code_paths': e.get('uncommitted_code_paths')}
        elif e.get('event') == 'unit_end' and e['unit_id'] in out:
            out[e['unit_id']]['end'] = V.parse_utc(e['utc'])
            out[e['unit_id']]['status'] = e.get('status')
    return out


def _module_files(mod):
    if str(mod).startswith('results/'):
        return [str(mod), f'results/{V.CAMPAIGN}/roles/verifier/code/vlib.py']
    parts = str(mod).split('.')
    pkg = '/'.join(parts[:2]) if len(parts) > 1 else parts[0]
    files = ['frc/common.py', 'frc/release.py']
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
        d = V.PRIV / 'compute' / 'units' / uid
        p, launch = d / 'CODE_AT_START.json', d / 'CODE_AT_LAUNCH.json'
        if not p.is_file():
            cache[key] = None
            return None
        rec = json.loads((launch if launch.is_file() else p).read_text())
        head = units[uid].get('git_head') or rec.get('git_head')
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
                              'ancestor': V.is_ancestor(commit),
                              'sha_now': V.file_sha(V.ROOT / card) if (V.ROOT / card).is_file() else None}
        return cache[key]
    return look


def _amendments():
    p = V.RESULTS / 'ACCESS_AMENDMENTS.json'
    if not p.is_file():
        return []
    relp = str(p.relative_to(V.ROOT))
    rc, out, _ = V.git('log', '--format=%H %cI', '--', relp)
    res = []
    for a in json.loads(p.read_text()).get('amendments', []):
        when = None
        for line in reversed(out.splitlines()):
            c, t = line.split()
            blob = V.git('show', f'{c}:{relp}')[1]
            try:
                if any(x.get('operation') == a.get('operation') and x.get('entry_point') == a.get('entry_point')
                       for x in json.loads(blob).get('amendments', [])):
                    when = V.parse_utc(t)
                    break
            except json.JSONDecodeError:
                continue
        res.append({'operation': a.get('operation'), 'entry_point': a.get('entry_point'), 'commit_utc': when})
    return res


def receipt_checks():
    """R1: own re-verification of the receipt (git calls of this script; ls-remote against the GitHub remote)."""
    out = {'record_exists': V.RECEIPT.is_file()}
    if not out['record_exists']:
        return out, None, False
    r = json.loads(V.RECEIPT.read_text())
    out.update({k: r.get(k) for k in ('kind', 'receipt_commit', 'research_commit', 'remote_sha', 'remote_url', 'verified_utc')})
    out['remote_sha_verified'] = bool(r.get('remote_sha_verified'))
    out['remote_sha_equals_receipt_commit'] = r.get('remote_sha') == r.get('receipt_commit')
    rc, ls, err = V.git('ls-remote', V.REMOTE_URL, f'refs/heads/{V.PUB_BRANCH}')
    out['own_ls_remote'] = ls if rc == 0 else f'unavailable: {err[-200:]}'
    tip = ls.split()[0] if rc == 0 and ls else None
    out['own_remote_tip_equals_receipt_commit'] = tip == r.get('receipt_commit')
    rc, url, _ = V.git('remote', 'get-url', 'origin')
    out['origin_url'] = url
    out['origin_is_github_remote'] = url == V.REMOTE_URL
    files_now = {f: V.file_sha(V.RESULTS / f) if (V.RESULTS / f).is_file() else None for f in REG_FILES}
    out['files_sha256_equal_now'] = r.get('files_sha256') == files_now
    out['files_committed_unchanged'] = {f: V.committed_unchanged(V.RESULTS / f) for f in REG_FILES}
    out['research_commit_ancestor_of_HEAD'] = V.is_ancestor(r['research_commit'])
    rt = V.commit_time(r['research_commit'])
    out['research_commit_utc'] = rt.isoformat() if rt else None
    out['research_commit_before_verification'] = bool(rt and rt <= V.parse_utc(r['verified_utc']))
    at_research = {f: V.git_blob_sha256(r['research_commit'], f'results/{V.CAMPAIGN}/{f}') for f in REG_FILES}
    out['research_commit_bytes_equal_receipt'] = at_research == r.get('files_sha256')
    have = V.git('cat-file', '-e', f'{r["receipt_commit"]}^{{commit}}')[0] == 0
    out['receipt_commit_available_locally'] = have
    pub_eq = {}
    if have:
        for f in REG_FILES:
            pub_eq[f] = V.git_blob_sha256(r['receipt_commit'], f'publications/{V.PUB_NAME}/results/{f}') == at_research[f]
        rcv = V.commit_time(r['receipt_commit'])
        out['receipt_commit_utc'] = rcv.isoformat() if rcv else None
    out['published_bytes_equal_research'] = pub_eq
    # commits that touched a registration file after the registered research commit (the access gate refuses label reads while
    # a registration file differs from its receipt hash; disclosure of any edit and restoration)
    hist = []
    for f in REG_FILES:
        rc_, log, _ = V.git('log', '--format=%H %cI', f'{r["research_commit"]}..HEAD', '--', f'results/{V.CAMPAIGN}/{f}')
        for line in log.splitlines():
            c, t = line.split()
            hist.append({'file': f, 'commit': c, 'commit_utc': V.parse_utc(t).isoformat(),
                         'bytes_equal_receipt': V.git_blob_sha256(c, f'results/{V.CAMPAIGN}/{f}') == r.get('files_sha256', {}).get(f)})
    out['registration_file_commits_after_research_commit'] = hist
    out['attempts'] =[{'outcome': a.get('outcome'), 'verified_utc': a.get('verified_utc'), 'problems': a.get('problems')}
                       for a in V.jsonl(V.ACCESS_DIR / 'RECEIPT_ATTEMPTS.jsonl')]
    pubrec = V.RESULTS / 'RECEIPTS.json'
    if pubrec.is_file():
        pr = json.loads(pubrec.read_text()).get('receipts', [])
        out['public_RECEIPTS_json_matches_private_record'] = any(
            x.get('receipt_commit') == r.get('receipt_commit') and x.get('verified_utc') == r.get('verified_utc') for x in pr)
    ok = bool(out['remote_sha_verified'] and out['remote_sha_equals_receipt_commit'] and out['own_remote_tip_equals_receipt_commit']
              and out['origin_is_github_remote'] and out['files_sha256_equal_now'] and all(out['files_committed_unchanged'].values())
              and out['research_commit_ancestor_of_HEAD'] and out['research_commit_bytes_equal_receipt']
              and out['research_commit_before_verification'] and pub_eq and all(pub_eq.values()))
    return out, (V.parse_utc(r['verified_utc']) if r.get('remote_sha_verified') else None), ok


def policy_from_json():
    pj = json.loads((V.RESULTS / 'ACCESS_POLICY.json').read_text())
    return {op: {'roles': v['roles'], 'fields': v['fields'], 'entry_points': v['entry_points'], 'card': bool(v.get('card'))}
            for op, v in pj['operations'].items()}


def policy_matches_code(pol):
    if str(V.ROOT) not in sys.path:
        sys.path.insert(0, str(V.ROOT))
    import frc.common as C
    diffs = []
    for op, p in C.POLICY.items():
        q = pol.get(op)
        if q is None or set(p['roles']) != set(q['roles']) or set(p['fields']) != set(q['fields']) or \
                bool(p.get('card')) != q['card'] or set(C.ENTRY_POINTS[op]) != set(q['entry_points']):
            diffs.append(op)
    if set(pol) != set(C.POLICY):
        diffs.append('operation sets differ')
    return diffs


def role_rows():
    if str(V.ROOT) not in sys.path:
        sys.path.insert(0, str(V.ROOT))
    import frc.common as C
    out = {}
    for r in V.ROLES6:
        F = C.features(r)
        out[r] = (V.arr_sha(np.asarray(F['row_id'])), int(len(F['row_id'])), int(len(np.unique(F['unit']))))
    return out


def run(tag):
    rec, rutc, rec_ok = receipt_checks()
    pol = policy_from_json()
    pdiff = policy_matches_code(pol)
    events = V.jsonl(V.ACCESS_LOG)
    units = _units()
    amends = _amendments()
    rr = role_rows()
    findings = audit(events, rutc, pol, amends, _card_lookup(), units, rr, _code_checker(units))
    fails = [x for x in findings if x['severity'] == 'FAIL']
    atts = [x for x in findings if x['severity'] == 'ATTENTION']
    by_rule = {}
    for x in findings:
        by_rule.setdefault(f"{x['severity']}|{x['rule']}", []).append(x['event'])
    summary = {'n_events': len(events), 'first_utc': events[0]['utc'] if events else None,
               'last_utc': events[-1]['utc'] if events else None, 'by_operation_role': {},
               'units_with_reads': sorted({e.get('unit_id') or 'null' for e in events})}
    for e in events:
        k = f"{e.get('operation')}|{e.get('role')}|{','.join(e.get('fields') or [])}|{str(e.get('entry_point')).split(':')[0]}"
        summary['by_operation_role'][k] = summary['by_operation_role'].get(k, 0) + 1
    fit_reads = [{k: e.get(k) for k in ('utc', 'unit_id', 'role', 'fields', 'card', 'card_commit', 'card_commit_utc')}
                 for e in events if e.get('operation') in ('fit', 'calibrate')]
    # R11 (every queue unit, label-reading or not, e.g. label-free B1 fits): the entry module's package files hashed at launch
    # equal the blobs at the launch HEAD; units launched while other files were uncommitted are listed (disclosure)
    cc = _code_checker(units)
    r11 = {'differs': {}, 'uncommitted_elsewhere': {}, 'not_checkable': []}
    for uid, u in units.items():
        if u.get('main') is None:
            continue
        bad = cc(uid, u['main'])
        if bad is None:
            r11['not_checkable'].append(uid)
        elif bad:
            r11['differs'][uid] = bad
        elif u.get('uncommitted_code_paths'):
            r11['uncommitted_elsewhere'][uid] = u['uncommitted_code_paths']
    checks = [
        {'id': f'V1.RECEIPT__{tag}', 'what': 'registration receipt re-verified with own git calls (R1): record verified; own '
         'ls-remote of publication/pcrl-finite-routing-comparison-v1 on https://github.com/Bostesa/PCRL.git equals the receipt '
         'commit; registration files committed, unchanged, equal to the receipt hashes, to the research commit and to the '
         'published copies; research commit an ancestor of HEAD and earlier than the verification',
         'inputs': V.inputs_sha([V.RECEIPT, V.ACCESS_DIR / 'RECEIPT_ATTEMPTS.jsonl', V.RESULTS / 'RECEIPTS.json']
                                + [V.RESULTS / f for f in REG_FILES]),
         'result': rec, 'numerical_status': 'NOT_APPLICABLE', 'procedural_status': 'PASS' if rec_ok else 'FAIL'},
        {'id': f'V1.ACCESS_POLICY_CODE__{tag}', 'what': 'frc.common POLICY / ENTRY_POINTS equal the registered ACCESS_POLICY.json',
         'inputs': V.inputs_sha([V.RESULTS / 'ACCESS_POLICY.json', V.ROOT / 'frc' / 'common.py']), 'result': {'differences': pdiff},
         'numerical_status': 'NOT_APPLICABLE', 'procedural_status': 'PASS' if not pdiff else 'FAIL'},
        {'id': f'V1.ACCESS_LOG__{tag}', 'what': 'label-access log audit R2-R10 (receipt order, closed roles, policy, MODEL_SELECT '
         'entry points and module chains, cards committed before fitting reads, unit intervals, row sets, committed code at '
         'launch, unit main modules)',
         'inputs': {**V.inputs_sha([V.ACCESS_LOG, V.QUEUE_LEDGER, V.RECEIPT]), 'n_events': len(events)},
         'result': {'summary': summary, 'findings_by_rule': by_rule, 'fail_examples': fails[:20], 'attention_examples': atts[:20],
                    'fit_reads': fit_reads, 'amendments': [{**a, 'commit_utc': a['commit_utc'].isoformat() if a['commit_utc']
                                                             else None} for a in amends],
                    'receipt_verified_utc': rutc.isoformat() if rutc else None},
         'numerical_status': 'NOT_APPLICABLE',
         'procedural_status': 'FAIL' if fails else ('ATTENTION' if atts else ('PASS' if events else 'NOT_APPLICABLE'))},
        {'id': f'V1.UNIT_CODE_AT_LAUNCH__{tag}', 'what': 'R11: every queue unit ran its entry package as committed at the launch HEAD '
         '(label-free units included); units launched while files outside their entry package were uncommitted are listed',
         'inputs': V.inputs_sha([V.QUEUE_LEDGER]), 'result': {**r11, 'n_units': len(units)},
         'numerical_status': 'NOT_APPLICABLE',
         'procedural_status': 'FAIL' if r11['differs'] else ('ATTENTION' if (r11['uncommitted_elsewhere'] or r11['not_checkable'])
                                                             else 'PASS')},
        {'id': f'V1.ROW_SETS__{tag}', 'what': 'logged row-set hashes, people and households equal the admitted roles (R8)',
         'inputs': V.inputs_sha([V.ACCESS_LOG]),
         'result': {'n_events': len(events), 'admitted': {r: list(v) for r, v in rr.items()},
                    'row_set_failures': [x for x in findings if x['rule'] == 'R8_row_set_differs'][:10]},
         'numerical_status': ('FAIL' if any(x['rule'] == 'R8_row_set_differs' for x in findings)
                              else ('PASS' if events else 'NOT_APPLICABLE')),
         'procedural_status': 'NOT_APPLICABLE'},
    ]
    print(json.dumps(V.append_checks(checks), indent=1))
    print(json.dumps({'receipt_ok': rec_ok, 'policy_diffs': pdiff, 'by_rule': {k: len(v) for k, v in by_rule.items()},
                      'summary': summary}, indent=1, default=V._jd)[:6000])
    return 0


# ---------------------------------------------------------------- invented-data selftest of the audit rules
def selftest():
    T = lambda m: dt.datetime(2026, 10, 20, 23, m, tzinfo=dt.timezone.utc)  # noqa: E731
    pol = {'fit': {'roles': ['MODEL_FIT'], 'fields': ['income', 'occupation', 'sex'], 'entry_points': list(FIT_ENTRY), 'card': True},
           'benchmark': {'roles': ['MODEL_SELECT'], 'fields': ['income', 'occupation', 'sex'], 'entry_points': ['frc.evaluate'],
                         'card': False},
           'readers': {'roles': ['MODEL_SELECT'], 'fields': ['sex'], 'entry_points': ['frc.devreaders', 'frc.readercustody'],
                       'card': False},
           'verify': {'roles': ['MODEL_FIT', 'MODEL_SELECT'], 'fields': ['income', 'occupation', 'sex'],
                      'entry_points': [VERIFY_EP], 'card': False}}
    rows = {'MODEL_FIT': ('h1', 10, 5), 'MODEL_SELECT': ('h2', 8, 4), 'DEV_REPORT': ('h3', 6, 3)}
    card = f'results/{V.CAMPAIGN}/cards/FA_FRC_v1.md'
    cards = {(card, 'c1'): {'commit_utc': T(5), 'sha_at_commit': 's1', 'ancestor': True, 'sha_now': 's1'},
             (card, 'c2'): {'commit_utc': T(30), 'sha_at_commit': 's2', 'ancestor': True, 'sha_now': 's1'}}
    units = {'U1': {'start': T(10), 'end': T(20), 'stage': 'fit', 'main': 'frc.fa.route'},
             'U2': {'start': T(10), 'end': T(50), 'stage': 'evaluate', 'main': 'frc.evaluate'},
             'U3': {'start': T(10), 'end': T(50), 'stage': 'fit', 'main': 'frc.fa.route'},
             'U4': {'start': T(10), 'end': T(50), 'stage': 'verify', 'main': VERIFY_EP + 'v4_utility.py'}}
    good_fit = {'utc': T(12).isoformat(), 'unit_id': 'U1', 'role': 'MODEL_FIT', 'fields': ['income', 'occupation'],
                'operation': 'fit', 'entry_point': 'frc.fa.route:_real', 'rows_sha256': 'h1', 'people': 10, 'households': 5,
                'card': card, 'card_commit': 'c1', 'card_sha256': 's1', 'card_commit_utc': T(5).isoformat(),
                'module_chain': ['frc.fa.route']}
    good_ms = {'utc': T(12).isoformat(), 'unit_id': 'U2', 'role': 'MODEL_SELECT', 'fields': ['income', 'occupation', 'sex'],
               'operation': 'benchmark', 'entry_point': 'frc.evaluate:ms', 'rows_sha256': 'h2', 'people': 8, 'households': 4,
               'module_chain': ['frc.evaluate']}
    good_ver = {**good_ms, 'unit_id': 'U4', 'operation': 'verify', 'entry_point': VERIFY_EP + 'v4_utility.py:load_labels',
                'module_chain': [VERIFY_EP + 'v4_utility.py']}
    plant = {
        'R2_read_before_receipt': {**good_ms, 'utc': T(1).isoformat()},
        'R3_closed_role': {**good_ms, 'role': 'DEV_REPORT', 'rows_sha256': 'h3', 'people': 6, 'households': 3},
        'R4_entry_point_not_authorized': {**good_ms, 'entry_point': 'frc.release:load'},
        'R4_field_not_allowed': {**good_ms, 'operation': 'readers', 'entry_point': 'frc.devreaders:x'},
        'R5_model_select_by_other_entry': {**good_ms, 'operation': 'verify', 'entry_point': 'frc.fa.route:main'},
        'R5_fitting_module_on_chain': {**good_ms, 'module_chain': ['frc.evaluate', 'frc.fa.route']},
        'R6_card_committed_after_read': {**good_fit, 'card_commit': 'c2', 'card_sha256': 's2', 'card_commit_utc': T(30).isoformat()},
        'R6_card_missing_or_outside_cards': {**good_fit, 'card': None},
        'R7_read_outside_unit_interval': {**good_fit, 'utc': T(25).isoformat()},
        'R7_null_unit_id': {**good_ms, 'unit_id': None},
        'R8_row_set_differs': {**good_ms, 'people': 7},
        'R9_uncommitted_code_at_launch': {**good_ms, 'unit_id': 'U2', 'entry_point': 'frc.evaluate:dirty'},
        'R10_unit_main_module_differs_from_operation': {**good_ms, 'unit_id': 'U3'},
        'R10_fit_stage_unit_read_model_select': {**good_ms, 'unit_id': 'U3'},
    }

    def code_check(uid, mod):
        return ['frc/evaluate.py'] if mod == 'frc.evaluate' and uid == 'U2' and code_check.dirty else []
    ok = {}
    code_check.dirty = False
    base = audit([good_fit, good_ms, good_ver], T(2), pol, [], lambda c, k: cards.get((c, k)), units, rows, code_check)
    ok['clean_log_no_findings'] = base == []
    for rule, ev in plant.items():
        code_check.dirty = rule.startswith('R9')
        fs = audit([good_fit, ev], T(2), pol, [], lambda c, k: cards.get((c, k)), units, rows, code_check)
        ok[f'flags_{rule}'] = any(x['rule'] == rule for x in fs)
    # an amendment committed after the read does not authorize it; one committed before does
    am_late = [{'operation': 'benchmark', 'entry_point': 'frc.extra', 'commit_utc': T(40)}]
    am_early = [{'operation': 'benchmark', 'entry_point': 'frc.extra', 'commit_utc': T(3)}]
    ev = {**good_ms, 'entry_point': 'frc.extra:f', 'module_chain': ['frc.extra']}
    units2 = dict(units, U2={**units['U2'], 'main': 'frc.extra'})
    code_check.dirty = False
    ok['late_amendment_flagged'] = any(x['rule'] == 'R4_entry_point_not_authorized' for x in
                                       audit([ev], T(2), pol, am_late, lambda c, k: None, units2, rows, code_check))
    ok['early_amendment_accepted'] = audit([ev], T(2), pol, am_early, lambda c, k: None, units2, rows, code_check) == []
    ok['all_ok'] = all(ok.values())
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        print(json.dumps(ok, indent=1))
        return 0 if ok['all_ok'] else 1
    return run(a.tag)


if __name__ == '__main__':
    sys.exit(main())
