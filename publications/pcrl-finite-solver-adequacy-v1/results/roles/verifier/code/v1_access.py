"""V1: receipt, registration bytes, source admission and the label-access log (campaign `pcrl_finite_solver_adequacy_v1`).

Label-free: it reads the receipt records, git, the registered policy, the queue ledger, the unit code manifests, the access log and
the label-free admitted features (fsa.common.features, for the row-set hashes). No label is read.

Checks (each one CHECKS.json entry with a record):
  V1.RECEIPT     own `git ls-remote` of publication/pcrl-finite-solver-adequacy-v1 on the GitHub remote equals the receipt commit;
                 the receipt record is verified with no problems; every registration file is byte-equal across the receipt commit's
                 published copy, the research commit, HEAD and the working tree, and hashes to the receipt's files_sha256; the
                 research commit is an ancestor of HEAD and older than the verification; no commit after the research commit touches
                 a registration file; the public RECEIPTS.json carries the same receipt.
  V1.ADMISSION   every pinned artefact of SOURCE_ADMISSION.json re-hashed (frc and nrm private artefacts and records); own ls-remote
                 of the two source publications equals the pins; the RULES p1_replay estimator pins equal the working files and the
                 published copies; frc/ and nrm/ code and the source campaigns' results unchanged since the branch base.
  V1.ACCESS_LOG  the audit of every logged label read:
                 A1 at or after the receipt's verified_utc; A2 no closed role; A3 registered operation, role and fields, an authorized
                 entry point (registered policy or a committed amendment older than the read) and a stated purpose; A4 MODEL_SELECT
                 only through benchmark / readers / verify entry points and never with a fitting module on the chain; A5 fit /
                 calibrate reads log a card under cards/ committed before the read, whose bytes at that commit hash to the logged
                 SHA-256, which is an ancestor of HEAD and unchanged now, and which is not a superseded card; A6 a queue unit whose
                 interval contains the read; A7 the logged row-set hash, people and households equal the admitted role; A8 the unit's
                 main module is the operation's entry point; A9 the label-reading unit's import closure ran committed code (FAIL) and
                 other uncommitted paths at launch are reported (ATTENTION); A10 the cached-statistics stages (audit, source) read
                 no label.

    python3 .../v1_access.py --selftest           # invented logs, ledgers and receipts: every rule flags its planted violation
    python3 .../v1_access.py --tag V1             # real (queue unit, stage verify)
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcore as V  # noqa: E402

SCRIPT = f'results/{V.CAMPAIGN}/roles/verifier/code/v1_access.py'
REG_FILES = ('PROTOCOL.md', 'EXPOSURE_MAP.md', 'ACCESS_POLICY.json', 'SOURCE_ADMISSION.json', 'READER_SPLITS.json',
             'FAILURE_MAP.md', 'roles/statistical_reviewer/RULES.md', 'roles/statistical_reviewer/RULES.json')
CLOSED = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT', 'SEALED_ASSESSMENT')
LABELS = ('income', 'occupation', 'sex')
VERIFY_EP = f'results/{V.CAMPAIGN}/roles/verifier/code/'
FIT_MODS = ('fsa.fa', 'fsa.ctl', 'fsa.fs')
MS_OPS = ('benchmark', 'readers', 'verify')
SUPERSEDED_CARDS = {f'results/{V.CAMPAIGN}/cards/FA_FSA_v1.md': 'superseded by FA_FSA_v2 before any use'}
NO_LABEL_STAGES = ('audit', 'source')
BRANCH_BASE = '278cb1ed6'
PKGS = ('fsa', 'frc', 'nrm', 'tsz', 'cts', 'ohn', 'ihs', 'acs_replication', 'scm_bench', 'ffv', 'task_conditioned_representation',
        'task_headroom', 'mechanism_design')


def entry_ok(mod, allowed):
    mod = str(mod)
    return any(mod == e or mod.startswith(e if e.endswith('/') else e + '.') or (e.endswith('/') and mod.startswith(e)) for e in allowed)


# ---------------------------------------------------------------- import closure of a unit's main module (AST, repository files only)
def _mod_file(mod, root):
    p = Path(root, *mod.split('.'))
    if (p.with_suffix('.py')).is_file():
        return p.with_suffix('.py')
    if (p / '__init__.py').is_file():
        return p / '__init__.py'
    return None


def import_closure(main_file, root):
    """Repository .py files reachable from main_file through import statements anywhere in a module (function-level included),
    with the __init__.py of every package on the way; top-level names are also resolved in main_file's own folder (a script run
    by path imports its sibling modules). Paths relative to root."""
    root = Path(root)
    side = Path(main_file).resolve().parent
    seen, stack = set(), [Path(main_file)]
    while stack:
        f = stack.pop()
        try:
            r = str(f.resolve().relative_to(root.resolve()))
        except ValueError:
            continue
        if r in seen or not f.is_file():
            continue
        seen.add(r)
        parts = Path(r).parts
        for i in range(1, len(parts)):                                   # parent packages' __init__.py
            ini = root.joinpath(*parts[:i], '__init__.py')
            if ini.is_file():
                stack.append(ini)
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        pkg = parts[:-1] if parts[-1] != '__init__.py' else parts[:-1]
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = list(pkg[:len(pkg) - (node.level - 1)]) if node.level > 1 else list(pkg)
                    stem = '.'.join(base + ([node.module] if node.module else []))
                else:
                    stem = node.module or ''
                mods = [stem] + [f'{stem}.{a.name}' for a in node.names]
            for m in mods:
                while m:
                    mf = _mod_file(m, root)
                    if mf is not None and m.split('.')[0] in PKGS + ('results',):
                        stack.append(mf)
                    elif '.' not in m and _mod_file(m, side) is not None:
                        stack.append(_mod_file(m, side))
                    m = m.rpartition('.')[0]
    return seen


def main_file_of(argv, root):
    if not argv:
        return None
    a = list(argv)
    if '-m' in a:
        return _mod_file(a[a.index('-m') + 1], root)
    for x in a[1:]:
        if x.endswith('.py'):
            p = Path(root) / x
            return p if p.is_file() else None
    return None


# ---------------------------------------------------------------- the access-log audit (pure; invented inputs in the selftest)
def audit(events, receipt_utc, policy, amendments, cards, units, role_rows, closures):
    """events: logged reads; receipt_utc: datetime or None; policy: ACCESS_POLICY operations; amendments: [{operation,
    entry_point, commit_utc}]; cards: {path: {'commit_ok', 'first_commit_utc', 'now_sha', 'blob_sha_at': {commit: sha}, 'ancestor':
    {commit: bool}}}; units: {unit_id: {'start', 'end', 'argv', 'stage', 'uncommitted', 'git_head'}}; role_rows: {role: (rows_sha,
    people, households)}; closures: {unit_id: {'closure': set, 'diff_files': set}} for label-reading units."""
    f = []

    def flag(sev, rule, i, detail):
        f.append({'severity': sev, 'rule': rule, 'event': i, 'detail': detail})
    for i, e in enumerate(events):
        t = V.parse_utc(e.get('utc'))
        role, op = e.get('role'), e.get('operation')
        fields = set(e.get('fields') or [])
        mod = str(e.get('entry_point', '?')).split(':')[0]
        if receipt_utc is None or t is None or t < receipt_utc:
            flag('FAIL', 'A1_read_before_receipt', i, e.get('utc'))
        if role in CLOSED:
            flag('FAIL', 'A2_closed_role', i, role)
        pol = policy.get(op)
        amended = [a['entry_point'] for a in amendments if a['operation'] == op and a.get('commit_utc') is not None
                   and V.parse_utc(a['commit_utc']) < t]
        if pol is None:
            flag('FAIL', 'A3_unregistered_operation', i, op)
        else:
            if role not in pol['roles']:
                flag('FAIL', 'A3_role_not_allowed', i, f'{op} {role}')
            if fields - set(pol['fields']) or not fields <= set(LABELS):
                flag('FAIL', 'A3_field_not_allowed', i, f'{op} {sorted(fields)}')
            if not entry_ok(mod, list(pol['entry_points']) + amended):
                flag('FAIL', 'A3_entry_point_not_authorized', i, f'{mod} for {op}')
        if not e.get('purpose') or len(str(e.get('purpose'))) < 12:
            flag('FAIL', 'A3_no_purpose', i, e.get('purpose'))
        if role == 'MODEL_SELECT':
            if op not in MS_OPS:
                flag('FAIL', 'A4_model_select_operation', i, op)
            if any(entry_ok(m, FIT_MODS) for m in e.get('module_chain', [])):
                flag('FAIL', 'A4_model_select_with_fitting_module', i, e.get('module_chain'))
        if op in ('fit', 'calibrate'):
            card = e.get('card')
            ci = cards.get(card)
            if not card or not str(card).startswith(f'results/{V.CAMPAIGN}/cards/') or ci is None:
                flag('FAIL', 'A5_no_card', i, card)
            else:
                cc = e.get('card_commit')
                ct = V.parse_utc(e.get('card_commit_utc'))
                if not ci['commit_ok'].get(cc):
                    flag('FAIL', 'A5_card_commit_unknown', i, cc)
                if ct is None or ct >= t:
                    flag('FAIL', 'A5_card_not_committed_before_read', i, f'{e.get("card_commit_utc")} vs {e.get("utc")}')
                if ci['blob_sha_at'].get(cc) != e.get('card_sha256'):
                    flag('FAIL', 'A5_card_bytes_differ_at_commit', i, card)
                if not ci['ancestor'].get(cc):
                    flag('FAIL', 'A5_card_commit_not_ancestor', i, cc)
                if ci['now_sha'] != e.get('card_sha256'):
                    flag('FAIL', 'A5_card_changed_after_use', i, card)
                fc = V.parse_utc(ci.get('first_commit_utc'))
                if fc is None or fc >= t:
                    flag('FAIL', 'A5_card_first_commit_after_read', i, ci.get('first_commit_utc'))
                if card in SUPERSEDED_CARDS:
                    flag('FAIL', 'A5_superseded_card', i, f'{card}: {SUPERSEDED_CARDS[card]}')
        uid = e.get('unit_id')
        u = units.get(uid)
        if u is None:
            flag('FAIL', 'A6_no_queue_unit', i, uid)
        else:
            s, en = V.parse_utc(u['start']), V.parse_utc(u['end']) if u.get('end') else None
            if t < s or (en is not None and t > en):
                flag('FAIL', 'A6_read_outside_unit_interval', i, f'{uid} {u["start"]}..{u.get("end")} read {e.get("utc")}')
            argv = u.get('argv') or []
            main = argv[argv.index('-m') + 1] if '-m' in argv else next((x for x in argv[1:] if x.endswith('.py')), '?')
            if op in ('fit', 'calibrate') and not entry_ok(main, FIT_MODS):
                flag('FAIL', 'A8_unit_main_not_fit_entry', i, main)
            if op == 'verify' and not entry_ok(main, (VERIFY_EP,)):
                flag('FAIL', 'A8_unit_main_not_verifier', i, main)
            if op == 'benchmark' and not entry_ok(main, ('fsa.evaluate',)):
                flag('FAIL', 'A8_unit_main_not_benchmark_entry', i, main)
            if op == 'readers' and not entry_ok(main, ('fsa.devreaders', 'fsa.readercustody')):
                flag('FAIL', 'A8_unit_main_not_reader_entry', i, main)
            if u.get('stage') in NO_LABEL_STAGES:
                flag('FAIL', 'A10_label_read_in_cached_statistics_stage', i, f'{uid} stage {u.get("stage")}')
            cl = closures.get(uid)
            if cl is not None:
                hit = sorted(x for x in cl['diff_files'] if x in cl['closure'])
                if hit:
                    flag('FAIL', 'A9_uncommitted_code_in_import_closure', i, f'{uid}: {hit}')
                elif u.get('uncommitted'):
                    flag('ATTENTION', 'A9_uncommitted_paths_outside_import_closure', i, f'{uid}: {u["uncommitted"]}')
        rr = role_rows.get(role)
        if rr is None or (e.get('rows_sha256'), e.get('people'), e.get('households')) != tuple(rr):
            flag('FAIL', 'A7_row_set_differs', i, f'{role}: logged {(e.get("rows_sha256"), e.get("people"), e.get("households"))}')
    return f


# ---------------------------------------------------------------- real-data gatherers
def receipt_check():
    prob, notes, rec = [], [], {}
    rp = V.ACCESS / 'REGISTRATION_RECEIPT_VERIFIED.json'
    if not rp.is_file():
        return 'FAIL', ['no verified registration receipt'], [], {}
    r = V.load_json(rp)
    rec['receipt'] = r
    if not r.get('remote_sha_verified') or r.get('problems') or r.get('remote_sha') != r.get('receipt_commit'):
        prob.append('receipt record not verified or with problems')
    rc, out = V.git('ls-remote', V.REMOTE, f'refs/heads/{V.PUB_BRANCH}')
    tip = out.split()[0] if rc == 0 and out else None
    rec['own_ls_remote'] = out
    if tip is None:
        prob.append('own ls-remote failed')
    elif tip != r['receipt_commit']:
        if V.git('cat-file', '-e', tip)[0] == 0 and V.is_ancestor(r['receipt_commit'], tip):
            notes.append(f'the publication tip moved to {tip}; the receipt commit is its ancestor')
        else:
            prob.append(f'remote tip {tip} differs from the receipt commit and is not a known descendant')
    vt = V.parse_utc(r['verified_utc'])
    files = {}
    for f in REG_FILES:
        local = (V.RES / f).read_bytes() if (V.RES / f).is_file() else None
        pub = V.git_blob(r['receipt_commit'], f'publications/{V.PUB_NAME}/results/{f}')
        res = V.git_blob(r['research_commit'], f'results/{V.CAMPAIGN}/{f}')
        head = V.git_blob('HEAD', f'results/{V.CAMPAIGN}/{f}')
        sh = V.sha_bytes(local) if local is not None else None
        ok = local is not None and pub == local and res == local and head == local and r['files_sha256'].get(f) == sh
        _, later = V.git('log', '--format=%H', f'{r["research_commit"]}..HEAD', '--', f'results/{V.CAMPAIGN}/{f}')
        files[f] = {'sha256': sh, 'published_equal': pub == local, 'research_equal': res == local, 'head_equal': head == local,
                    'receipt_hash_equal': r['files_sha256'].get(f) == sh, 'commits_after_research_commit': later.split()}
        if not ok:
            prob.append(f'{f}: bytes differ across copies')
        if later.strip():
            prob.append(f'{f}: changed by a commit after the research commit')
    rec['files'] = files
    if not V.is_ancestor(r['research_commit'], 'HEAD'):
        prob.append('research commit is not an ancestor of HEAD')
    for c in (r['research_commit'], r['receipt_commit']):
        ct = V.git_commit_utc(c)
        rec[f'commit_utc_{c[:9]}'] = None if ct is None else ct.isoformat()
        if ct is None or ct > vt:
            prob.append(f'commit {c[:9]} time {ct} not before the verification {vt}')
    pub = V.load_json(V.RES / 'RECEIPTS.json') if (V.RES / 'RECEIPTS.json').is_file() else {'receipts': []}
    same = [x for x in pub.get('receipts', []) if x.get('receipt_commit') == r['receipt_commit'] and x.get('verified_utc') == r['verified_utc']]
    rec['public_receipts'] = len(pub.get('receipts', []))
    if not same or any(x.get('files_sha256') != r['files_sha256'] for x in same):
        prob.append('RECEIPTS.json does not carry the same receipt')
    _, rc_committed = V.git('log', '-1', '--format=%H %cI', '--', f'results/{V.CAMPAIGN}/RECEIPTS.json')
    rec['RECEIPTS_json_commit'] = rc_committed
    att = V.ACCESS / 'RECEIPT_ATTEMPTS.jsonl'
    rec['attempts'] = [{k: a.get(k) for k in ('outcome', 'verified_utc', 'receipt_commit', 'problems')}
                       for a in (json.loads(x) for x in att.read_text().splitlines() if x.strip())] if att.is_file() else []
    return ('FAIL' if prob else ('ATTENTION' if notes else 'PASS')), prob, notes, rec


def _admission_paths(key, group):
    if group == 'frc':
        if key.startswith('frc/fa_cache/'):
            return [V.FRC_PRIV / key[len('frc/'):]]
        if key.startswith('frc/results/'):
            return [V.FRC_RES / key[len('frc/results/'):]]
        if key.startswith('frc/runs/'):
            return [V.FRC_PRIV / key[len('frc/'):] / 'MODEL.pkl', V.FRC_PRIV / key[len('frc/'):] / 'RELEASE.npz']
    else:
        if key.startswith('fa_cache/'):
            return [V.NRM_PRIV / key]
        if key.startswith('results/designer_a/'):            # pinned('gate') keys the role-folder file roles/designer_a/<name>
            return [V.NRM_RES / 'roles' / 'designer_a' / key[len('results/designer_a/'):]]
        if key.startswith('results/'):
            return [V.NRM_RES / key[len('results/'):]]
        if key.startswith('runs/'):
            return [V.NRM_PRIV / key / 'MODEL.pkl', V.NRM_PRIV / key / 'RELEASE.npz']
    return []


def admission_check(rules):
    prob, notes, rec = [], [], {'artefacts': {}}
    adm = V.load_json(V.RES / 'SOURCE_ADMISSION.json')
    n = 0
    for group, key in (('frc', 'frc_private_artefacts'), ('nrm', 'source_private_artefacts')):
        for k, pin in adm[key].items():
            paths = _admission_paths(k, group)
            if not paths:
                prob.append(f'unmapped admission key {k}')
                continue
            for p in paths:
                want = pin['sha256'] if 'sha256' in pin else pin.get(p.name)
                got = V.sha_file(p) if p.is_file() else None
                n += 1
                rec['artefacts'][str(p.relative_to(V.ROOT))] = {'pin': want, 'now': got, 'ok': got == want and got is not None}
                if got != want or got is None:
                    prob.append(f'{p.relative_to(V.ROOT)} differs from its pin')
    rec['n_rehashed'] = n
    for name, sha in (('pcrl-finite-routing-comparison-v1', adm['frc_publication_on_remote']['ls_remote'].split()[0]),
                      ('pcrl-new-release-mechanisms-v1', adm['pinned_publication_on_remote']['ls_remote'].split()[0])):
        rc, out = V.git('ls-remote', V.REMOTE, f'refs/heads/publication/{name}')
        rec[f'ls_remote_{name}'] = out
        if rc != 0 or not out or out.split()[0] != sha:
            prob.append(f'publication/{name} on the remote differs from the admission pin')
    pins = rules['p1_replay']['estimator_sha256']
    pub = {'nrm/fa/prc.py': ('682068d52147bc1bdfc75d9e49199115b1deaaed', 'publications/pcrl-new-release-mechanisms-v1/code/nrm/fa/prc.py'),
           'frc/fa/route.py': ('04ac9161fa1dd32cbf5dea4ce2175a6b8f8cf067', 'publications/pcrl-finite-routing-comparison-v1/code/frc/fa/route.py')}
    for f, h in pins.items():
        now = V.sha_file(V.ROOT / f)
        b = V.git_blob(*pub[f]) if f in pub else None
        rec[f'estimator_{f}'] = {'pin': h, 'now': now, 'published': None if b is None else V.sha_bytes(b)}
        if now != h:
            prob.append(f'{f} differs from the RULES pin')
        if b is None:
            notes.append(f'{f}: no published copy found at {pub.get(f)}')
        elif V.sha_bytes(b) != h:
            notes.append(f'{f}: published copy differs from the pin (sanitiser?)')
    for paths, what in ((['frc', 'nrm'], 'source code'),
                        ([f'results/{V.FRC}', f'results/{V.NRM}'], 'source campaign results')):
        _, d = V.git('diff', '--name-only', BRANCH_BASE, 'HEAD', '--', *paths)
        _, st = V.git('status', '--porcelain', '--', *paths)
        rec[f'{what}_changed_since_base'] = d.split()
        rec[f'{what}_worktree_status'] = st.splitlines()
        if d.strip():
            prob.append(f'{what} changed since the branch base: {d.split()[:5]}')
        known = [x for x in st.splitlines() if x.endswith('roles/verifier/records/.checks.lock')]
        other = [x for x in st.splitlines() if x not in known]
        if other:
            prob.append(f'{what} has working-tree changes: {other[:5]}')
        if known:
            notes.append(f'{what}: the previous verifier\'s untracked empty lock file {known} (pre-existing, disclosed in FRC1)')
    return ('FAIL' if prob else ('ATTENTION' if notes else 'PASS')), prob, notes, rec


def gather_access():
    C = V.admitted_loaders()                            # fsa.common: features label-free; load logged, operation 'verify'
    events = [json.loads(x) for x in (V.ACCESS / 'LABEL_ACCESS.jsonl').read_text().splitlines() if x.strip()] \
        if (V.ACCESS / 'LABEL_ACCESS.jsonl').is_file() else []
    r = V.load_json(V.ACCESS / 'REGISTRATION_RECEIPT_VERIFIED.json')
    pol = V.load_json(V.RES / 'ACCESS_POLICY.json')['operations']
    amend = []
    ap = V.RES / 'ACCESS_AMENDMENTS.json'
    if ap.is_file():
        _, log = V.git('log', '--diff-filter=A', '--format=%cI', '--', str(ap.relative_to(V.ROOT)))
        for a in V.load_json(ap).get('amendments', []):
            amend.append({**a, 'commit_utc': log.split()[-1] if log else None})
    cards = {}
    for e in events:
        c = e.get('card')
        if not c or c in cards:
            continue
        p = V.ROOT / c
        commits = {x.get('card_commit') for x in events if x.get('card') == c}
        _, first = V.git('log', '--diff-filter=A', '--format=%cI', '--', c)
        cards[c] = {'commit_ok': {cc: V.git('cat-file', '-e', f'{cc}^{{commit}}')[0] == 0 for cc in commits if cc},
                    'blob_sha_at': {cc: (lambda b: V.sha_bytes(b) if b is not None else None)(V.git_blob(cc, c)) for cc in commits if cc},
                    'ancestor': {cc: V.is_ancestor(cc, 'HEAD') for cc in commits if cc},
                    'now_sha': V.sha_file(p) if p.is_file() else None,
                    'first_commit_utc': first.split()[-1] if first else None}
    led = [json.loads(x) for x in (V.COMPUTE / 'ledger.jsonl').read_text().splitlines() if x.strip()]
    units = {}
    for e in led:
        if e.get('event') == 'unit_start':
            units[e['unit_id']] = {'start': e['utc'], 'end': None, 'argv': e.get('argv'), 'stage': e.get('stage'),
                                   'uncommitted': e.get('uncommitted_code_paths') or [], 'git_head': e.get('git_head')}
        elif e.get('event') == 'unit_end' and e['unit_id'] in units:
            units[e['unit_id']]['end'] = e['utc']
            units[e['unit_id']]['status'] = e.get('status')
            units[e['unit_id']]['code_changed_during_run'] = e.get('code_changed_during_run')
    role_rows = {}
    for role in sorted({e.get('role') for e in events if e.get('role') not in CLOSED and e.get('role')}):
        ft = C.features(role)
        role_rows[role] = (V.arr_sha(ft['row_id']), int(len(ft['row_id'])), int(len(set(ft['unit'].tolist()))))
    closures = {}
    for uid in sorted({e.get('unit_id') for e in events if e.get('unit_id') in units}):
        u = units[uid]
        mf = main_file_of(u['argv'], V.ROOT)
        cl = import_closure(mf, V.ROOT) if mf else set()
        d = V.COMPUTE / 'units' / uid
        cs = V.load_json(d / 'CODE_AT_LAUNCH.json') if (d / 'CODE_AT_LAUNCH.json').is_file() else V.load_json(d / 'CODE_AT_START.json')
        diff = set()
        for f, h in cs['files'].items():
            b = V.git_blob(u['git_head'], f)
            if b is None or V.sha_bytes(b) != h:
                diff.add(f)
        changed = set(u.get('code_changed_during_run') or [])
        closures[uid] = {'closure': cl, 'diff_files': diff | changed, 'main_file': None if mf is None else str(mf.relative_to(V.ROOT))}
    stage_reads = {u['stage']: 0 for u in units.values()}
    for e in events:
        if e.get('unit_id') in units:
            stage_reads[units[e['unit_id']]['stage']] = stage_reads.get(units[e['unit_id']]['stage'], 0) + 1
    return events, V.parse_utc(r['verified_utc']), pol, amend, cards, units, role_rows, closures, stage_reads


def main_real(tag, only=('receipt', 'admission', 'access'), supersedes=None):
    rules = V.load_json(V.RES / 'roles' / 'statistical_reviewer' / 'RULES.json')
    checks = []
    st, prob, notes, rec = receipt_check() if 'receipt' in only else (None, None, None, None)
    if 'receipt' in only:
      checks.append({'id': f'V1.RECEIPT__{tag}', 'block': 'V1', 'kind': 'procedural', 'status': st,
                   'summary': f'own ls-remote: {rec.get("own_ls_remote")}; receipt {rec.get("receipt", {}).get("receipt_commit")} verified '
                              f'{rec.get("receipt", {}).get("verified_utc")}; {len(REG_FILES)} registration files byte-equal across the '
                              f'published copy, research commit, HEAD and worktree and equal to the receipt hashes'
                              + (f'; problems {prob}' if prob else '') + (f'; notes {notes}' if notes else ''),
                   'detail': {'problems': prob, 'notes': notes}, 'record': rec})
    st, prob, notes, rec = admission_check(rules) if 'admission' in only else (None, None, None, None)
    if 'admission' in only:
      checks.append({'id': f'V1.ADMISSION__{tag}', 'block': 'V1', 'kind': 'procedural', 'status': st,
                   'summary': f'{rec["n_rehashed"]} pinned artefact files re-hashed; source publications on the remote; estimator pins; '
                              f'frc / nrm code and source results unchanged since {BRANCH_BASE}'
                              + (f'; problems {prob}' if prob else '') + (f'; notes {notes}' if notes else ''),
                   'detail': {'problems': prob, 'notes': notes, 'n_rehashed': rec['n_rehashed']}, 'record': rec})
    if 'access' in only:
      events, rt, pol, amend, cards, units, role_rows, closures, stage_reads = gather_access()
      fl = audit(events, rt, pol, amend, cards, units, role_rows, closures)
      sev = {x['severity'] for x in fl}
      st = 'FAIL' if 'FAIL' in sev else ('ATTENTION' if 'ATTENTION' in sev else 'PASS')
      per_unit = {}
      for e in events:
          per_unit.setdefault(e.get('unit_id'), []).append({k: e.get(k) for k in ('utc', 'role', 'fields', 'operation', 'entry_point')})
      checks.append({'id': f'V1.ACCESS_LOG__{tag}', 'block': 'V1', 'kind': 'procedural', 'status': st,
                   'summary': f'{len(events)} logged label reads audited (rules A1-A10); findings '
                              f'{sorted({(x["severity"], x["rule"]) for x in fl})}; reads per unit '
                              f'{ {u: len(v) for u, v in per_unit.items()} }',
                   'detail': {'n_reads': len(events), 'n_findings': len(fl), 'rules_flagged': sorted({x['rule'] for x in fl}),
                              'reads_by_stage': stage_reads},
                   'record': {'findings': fl, 'reads_by_unit': per_unit,
                              'closures': {u: {'main_file': c['main_file'], 'n_files': len(c['closure']),
                                               'differing_from_head': sorted(c['diff_files'])} for u, c in closures.items()},
                              'units': {u: units[u] for u in per_unit if u in units}, 'role_rows': role_rows,
                              'receipt_verified_utc': rt.isoformat()}})
    for c in checks:
        c['script'] = SCRIPT
        if supersedes:
            c['detail']['supersedes'] = supersedes
    V.Checks().append(checks)
    print(json.dumps({c['id']: (c['status'], c['summary'][:400]) for c in checks}, indent=1))
    return 0                                     # the script ran to completion; statuses (FAIL included) live in CHECKS.json


# ---------------------------------------------------------------- selftest (invented logs only)
def selftest():
    import datetime as dt
    import tempfile
    res = []
    T = lambda s: dt.datetime.fromisoformat(f'2026-10-21T{s}+00:00')            # noqa: E731
    pol = {'fit': {'roles': ['MODEL_FIT'], 'fields': ['income', 'occupation', 'sex'], 'entry_points': ['fsa.fa', 'fsa.ctl', 'fsa.fs']},
           'verify': {'roles': ['MODEL_FIT', 'MODEL_SELECT'], 'fields': ['income', 'occupation', 'sex'], 'entry_points': [VERIFY_EP]},
           'benchmark': {'roles': ['MODEL_SELECT'], 'fields': ['income', 'occupation', 'sex'], 'entry_points': ['fsa.evaluate']}}
    card = f'results/{V.CAMPAIGN}/cards/FA_X.md'
    cards = {card: {'commit_ok': {'c1': True}, 'blob_sha_at': {'c1': 'h1'}, 'ancestor': {'c1': True}, 'now_sha': 'h1',
                    'first_commit_utc': '2026-10-21T02:00:00+00:00'}}
    units = {'U1': {'start': '2026-10-21T03:10:00+00:00', 'end': '2026-10-21T03:20:00+00:00', 'argv': ['python3', '-m', 'fsa.fa.solve'],
                    'stage': 'replay', 'uncommitted': [], 'git_head': 'g'},
             'U2': {'start': '2026-10-21T03:10:00+00:00', 'end': '2026-10-21T03:20:00+00:00',
                    'argv': ['python3', f'{VERIFY_EP}v2_replay.py'], 'stage': 'verify', 'uncommitted': [], 'git_head': 'g'}}
    rows = {'MODEL_FIT': ('r', 20000, 13337), 'MODEL_SELECT': ('s', 8001, 5306)}
    good = {'utc': '2026-10-21T03:11:00+00:00', 'role': 'MODEL_FIT', 'fields': ['income', 'occupation', 'sex'], 'operation': 'fit',
            'entry_point': 'fsa.fa.solve:_labelled', 'purpose': 'a stated purpose here', 'module_chain': ['fsa.fa.solve'],
            'card': card, 'card_commit': 'c1', 'card_commit_utc': '2026-10-21T02:00:00+00:00', 'card_sha256': 'h1',
            'unit_id': 'U1', 'rows_sha256': 'r', 'people': 20000, 'households': 13337}
    ver = {'utc': '2026-10-21T03:12:00+00:00', 'role': 'MODEL_SELECT', 'fields': ['sex'], 'operation': 'verify',
           'entry_point': f'{VERIFY_EP}v2_replay.py:main_real', 'purpose': 'verifier recomputation', 'module_chain': ['x'],
           'unit_id': 'U2', 'rows_sha256': 's', 'people': 8001, 'households': 5306}
    clos = {'U1': {'closure': {'fsa/fa/solve.py', 'fsa/common.py'}, 'diff_files': set()}, 'U2': {'closure': set(), 'diff_files': set()}}

    def run(name, evs, want_rules, receipt=T('03:03:08'), cards_=cards, units_=units, clos_=clos, amend=()):
        fl = audit(evs, receipt, pol, list(amend), cards_, units_, rows, clos_)
        got = {x['rule'] for x in fl}
        res.append((name, got == set(want_rules), sorted(got)))
    run('clean', [good, ver], [])
    run('before_receipt', [dict(good, utc='2026-10-21T03:01:00+00:00')], ['A1_read_before_receipt', 'A6_read_outside_unit_interval'])
    run('no_receipt', [good], ['A1_read_before_receipt'], receipt=None)
    run('closed_role', [dict(ver, role='DEV_REPORT', rows_sha256='d')], ['A2_closed_role', 'A3_role_not_allowed', 'A7_row_set_differs'])
    run('bad_entry_point', [dict(good, entry_point='fsa.evaluate:x')], ['A3_entry_point_not_authorized'])
    run('amended_entry_point_ok', [dict(ver, entry_point='fsa.extra:x', role='MODEL_FIT', fields=['sex'], rows_sha256='r', people=20000,
                                        households=13337)], [], amend=[{'operation': 'verify', 'entry_point': 'fsa.extra',
                                                                        'commit_utc': '2026-10-21T03:00:00+00:00'}])
    run('amendment_after_read', [dict(ver, entry_point='fsa.extra:x', role='MODEL_FIT', fields=['sex'], rows_sha256='r', people=20000,
                                      households=13337)], ['A3_entry_point_not_authorized'],
        amend=[{'operation': 'verify', 'entry_point': 'fsa.extra', 'commit_utc': '2026-10-21T03:30:00+00:00'}])
    run('field_not_allowed', [dict(good, fields=['sex', 'raw_inputs'])], ['A3_field_not_allowed'])
    run('no_purpose', [dict(good, purpose='x')], ['A3_no_purpose'])
    run('model_select_fit_chain', [dict(ver, module_chain=['fsa.fa.solve'])], ['A4_model_select_with_fitting_module'])
    run('card_after_read', [dict(good, card_commit_utc='2026-10-21T03:15:00+00:00')], ['A5_card_not_committed_before_read'])
    run('card_changed', [good], ['A5_card_changed_after_use'], cards_={card: dict(cards[card], now_sha='h2')})
    run('card_bytes_differ', [dict(good, card_sha256='h9')], ['A5_card_bytes_differ_at_commit', 'A5_card_changed_after_use'])
    run('superseded_card', [dict(good, card=f'results/{V.CAMPAIGN}/cards/FA_FSA_v1.md')], ['A5_superseded_card'],
        cards_={**cards, f'results/{V.CAMPAIGN}/cards/FA_FSA_v1.md': cards[card]})
    run('no_unit', [dict(good, unit_id=None)], ['A6_no_queue_unit'])
    run('row_set', [dict(good, people=19999)], ['A7_row_set_differs'])
    run('fit_read_from_verifier_unit', [dict(good, unit_id='U2')], ['A8_unit_main_not_fit_entry'])
    run('label_read_in_source_stage', [good], ['A10_label_read_in_cached_statistics_stage'],
        units_={**units, 'U1': dict(units['U1'], stage='source')})
    run('uncommitted_in_closure', [good], ['A9_uncommitted_code_in_import_closure'],
        clos_={**clos, 'U1': {'closure': {'fsa/fa/solve.py'}, 'diff_files': {'fsa/fa/solve.py'}}},
        units_={**units, 'U1': dict(units['U1'], uncommitted=['fsa/fa/solve.py'])})
    run('uncommitted_outside_closure', [good], ['A9_uncommitted_paths_outside_import_closure'],
        clos_={**clos, 'U1': {'closure': {'fsa/fa/solve.py'}, 'diff_files': {'fsa/rules.py'}}},
        units_={**units, 'U1': dict(units['U1'], uncommitted=['fsa/rules.py'])})
    # import closure on an invented package tree
    with tempfile.TemporaryDirectory() as td:
        r = Path(td)
        (r / 'fsa' / 'fa').mkdir(parents=True)
        (r / 'frc' / 'fa').mkdir(parents=True)
        (r / 'fsa' / '__init__.py').write_text('')
        (r / 'fsa' / 'fa' / '__init__.py').write_text('')
        (r / 'frc' / '__init__.py').write_text('')
        (r / 'frc' / 'fa' / '__init__.py').write_text('')
        (r / 'fsa' / 'fa' / 'solve.py').write_text('from frc.fa import route as FR\nfrom . import solver\ndef f():\n    import fsa.common\n')
        (r / 'fsa' / 'fa' / 'solver.py').write_text('import numpy as np\n')
        (r / 'fsa' / 'common.py').write_text('')
        (r / 'fsa' / 'rules.py').write_text('')
        (r / 'frc' / 'fa' / 'route.py').write_text('')
        cl = import_closure(r / 'fsa' / 'fa' / 'solve.py', r)
        want = {'fsa/fa/solve.py', 'fsa/fa/solver.py', 'fsa/common.py', 'frc/fa/route.py', 'fsa/__init__.py', 'fsa/fa/__init__.py',
                'frc/__init__.py', 'frc/fa/__init__.py'}
        res.append(('import_closure', cl == want, sorted(cl ^ want)))
        res.append(('main_file_of', main_file_of(['python3', '-m', 'fsa.fa.solve', 'replay'], r) == r / 'fsa' / 'fa' / 'solve.py', None))
        (r / 'tools').mkdir()
        (r / 'tools' / 'job.py').write_text('import helper\n')
        (r / 'tools' / 'helper.py').write_text('def f():\n    from fsa import common as C\n')
        cl2 = import_closure(r / 'tools' / 'job.py', r)
        res.append(('import_closure_sibling_modules', cl2 == {'tools/job.py', 'tools/helper.py', 'fsa/common.py', 'fsa/__init__.py'},
                    sorted(cl2)))
    n_ok = sum(x[1] for x in res)
    for name, ok, info in res:
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  [{info}]'))
    print(f'v1_access selftest: {n_ok}/{len(res)}')
    return n_ok == len(res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--tag', default='V1')
    ap.add_argument('--only', default='receipt,admission,access')
    ap.add_argument('--supersedes', default=None)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not os.environ.get('FSA_UNIT_ID'):
        sys.exit('real runs go through the queue (fsa.jobqueue run --stage verify)')
    sys.exit(main_real(a.tag, tuple(a.only.split(',')), a.supersedes))
