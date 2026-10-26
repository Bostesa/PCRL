#!/usr/bin/env python3
"""RFA1 verifier, wave 5 task 4: independent counts-only scan of the local publication commit (label-free, read-only on git).

Inputs are produced beforehand by plain read-only git commands into a scratch folder (passed as argv[1]):
  git ls-tree -r --long <commit>   > {pub,rcpt,res}_ls.txt      (publication, registration receipt, research commit)
  git archive --format=tar -o pub.tar <publication commit>; rcpt.tar <receipt>; res.tar <research commit> results/<campaign> rfa
  git cat-file -p <commit> > {pub,rcpt}_commit.txt; git rev-list <publication commit> > pub_revlist.txt
  git config --get user.email > git_email.txt; git config --get user.name > git_name.txt
Every archive member is checked against its tree blob id (sha1 of 'blob <n>\\0' + bytes), so the scan covers the tree blobs exactly.

Counts only: no matched string, identifier or pattern is printed or written (vsafe.safe guards every printed path).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))
import vsafe as V  # noqa: E402

PUB, PUB_TREE = 'a9e535345999f0fac89f7afd8085f13a67087cda', '5276df3a291969245d1622a9b87639bdff67d809'
RCPT, RES = '34dc42b5193d6d19406abcd88dc42d5da2944fd4', 'c4bc191e1d7efee6fb86f6d8b03c91cb0a58b997'
PFX = 'publications/pcrl-reader-fit-aware-release-v1/'
CAMP = 'results/pcrl_reader_fit_aware_release_v1/'
OUT = HERE.parent / 'out' / 'V5_PUBSCAN.json'
MAIL = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}')
ABS = re.compile(r'(?<![\w.~-])/(?:Users|home|Volumes|private|tmp|var|opt|usr|System|Library|Applications|mnt|media|root|etc|srv|run)'
                 r'(?:/[^\s\'"`)<>|,;\]\}]*)?')
WIN = re.compile(r'[A-Za-z]:\\\\?(?:Users|Documents and Settings)\\')
TILDE = re.compile(r'(?<![\w/])~/[^\s\'"`)<>|,;]*')
FILEURL = re.compile(r'file://')
CRED = re.compile(r'(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gho_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|'
                  r'xox[baprs]-[A-Za-z0-9-]{10,}|sk-[A-Za-z0-9]{32,})')
BAD_SUFFIX = {'.npz', '.npy', '.pt', '.pth', '.pkl', '.pickle', '.joblib', '.h5', '.hdf5', '.parquet', '.feather', '.arrow', '.tar', '.gz',
              '.tgz', '.zip', '.bundle', '.jsonl', '.lock', '.db', '.sqlite', '.bin', '.safetensors', '.onnx', '.ckpt', '.sav', '.rds',
              '.dta', '.sas7bdat', '.xlsx', '.xls', '.png', '.jpg', '.pdf'}
CENSUS = re.compile(r'\b(SERIALNO|SPORDER|PWGTP\d*|ST_PUMA|PUMA)\b')


def blob_id(b):
    return hashlib.sha1(b'blob %d\x00' % len(b) + b).hexdigest()


def load(scr, name):
    with tarfile.open(scr / f'{name}.tar') as t:
        files = {m.name: t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
        kinds = Counter('file' if m.isfile() else 'dir' if m.isdir() else 'other' for m in t.getmembers())
    ls = {}
    for line in (scr / f'{name}_ls.txt').read_text().splitlines():
        meta, path = line.split('\t', 1)
        mode, typ, sha, size = meta.split()
        ls[path] = (mode, typ, sha, size)
    return files, ls, kinds


def text_of(b):
    try:
        return b.decode('utf-8'), True
    except UnicodeDecodeError:
        return b.decode('latin-1'), False


def classify_abs(m, text, users, drive_label):
    s = m.group(0)
    parts = s.split('/')
    root = parts[1] if len(parts) > 1 else ''
    nxt = parts[2] if len(parts) > 2 else ''
    if root in ('Users', 'home'):
        if nxt.lower() in {u.lower() for u in users}:
            return 'PERSONAL_HOME'
        if nxt in ('Shared', ''):
            return 'shared_or_bare_home_root'
        return 'OTHER_USER_HOME'
    if root == 'Volumes':
        if text[max(0, m.start() - 7):m.start()] == '/System':
            return 'system_data_volume (/System/Volumes/...)'
        if nxt.startswith('...') or nxt == '':
            return 'prose_placeholder_volume (/Volumes/...)'
        if drive_label and nxt == drive_label:
            return 'drive_mount_point (label = recorded drive name)'
        return 'OTHER_VOLUME'
    if root in ('private', 'tmp', 'var'):
        if s.startswith(('/private/var/folders', '/var/folders', '/private/tmp/claude-', '/tmp/claude-')):
            return 'SESSION_OR_USER_TEMP'
        return 'generic_temp_path (e.g. /tmp/<name> in an example command)'
    if root in ('opt', 'usr', 'System', 'Library', 'Applications', 'etc', 'run') and s.count('/') <= 6:
        return 'system_path'
    return f'OTHER_ROOT_{root}'


def scan_text(text, cids, pb_pats, users, drive_label):
    c = V.count_ids(text, cids)
    spans = set()
    for k, pats in cids.items():
        if k == 'git_handle':
            continue
        for p in pats:
            spans |= {(m.start(), m.end()) for m in p.finditer(text)}
    merged, last = 0, -1
    for a, b in sorted(spans):
        if a >= last:
            merged += 1
        last = max(last, b)
    c['owner_union_distinct_spans'] = merged
    c['pb_owner_patterns'] = sum(len(p.findall(text)) for p, _ in pb_pats)
    c['email_like'] = len(MAIL.findall(text))
    absl = []
    for m in ABS.finditer(text):
        absl.append((text.count('\n', 0, m.start()) + 1, classify_abs(m, text, users, drive_label)))
    c['abs_path'] = len(absl)
    c['windows_user_path'] = len(WIN.findall(text))
    c['tilde_path'] = len(TILDE.findall(text))
    c['file_url'] = len(FILEURL.findall(text))
    c['credential_like'] = len(CRED.findall(text))
    c['census_markers'] = len(CENSUS.findall(text))
    return c, absl


def numeric_lists(obj, path='$', out=None):
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            numeric_lists(v, f'{path}.{k}', out)
    elif isinstance(obj, list):
        if len(obj) >= 500 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj):
            out.append((path, len(obj), sorted({type(x).__name__ for x in obj}), len(set(obj)) if len(obj) < 10 ** 6 else None))
        for i, v in enumerate(obj[:2000]):
            numeric_lists(v, f'{path}[{i}]', out)
    return out


PH = ['<owner host>', '<owner>', '<worktrees>', '<repository>', '<home>', '<session scratch>', '<external drive>', '<author e-mail>',
      '<submission id>', '<private review sources>', '<private meeting notes>', '<private review matrix>', '<private backup>',
      '<NeurIPS submission PDF>', '<AAAI submission PDF>']           # the bracketed replacements of the two sanitizer layers
PH_RE = re.compile('|'.join(re.escape(x) for x in PH))


def align(a, b):
    """Align a research line a with a publication line b around b's placeholders: [(placeholder, replaced research text)] or None."""
    toks = PH_RE.findall(b)
    if not toks:
        return None
    parts = PH_RE.split(b)
    pat = ''.join(re.escape(parts[i]) + ('(.+?)' if i < len(toks) else '') for i in range(len(parts)))
    m = re.fullmatch(pat, a, re.S)
    return None if m is None else list(zip(toks, m.groups()))


def main():
    t0 = time.process_time()
    scr = Path(sys.argv[1])
    ids = V.identifiers(scr / 'git_email.txt', scr / 'git_name.txt')
    cids = V.compile_ids(ids)
    from rfa import publish_build as PB
    from rfa import common as C
    from mechanism_design.publish_build import sanitise
    pb_pats = PB.owner_patterns()
    users = ids['user_name']
    res = {'schema': 'RFA1-verifier-pubscan-v5', 'commits': {'publication': PUB, 'tree': PUB_TREE, 'receipt': RCPT, 'research': RES},
           'identifier_categories': {k: len(v) for k, v in ids.items()}, 'pb_owner_patterns': len(pb_pats),
           'note': 'counts only; no matched string, identifier or pattern is printed or written'}
    chk = {}
    # ---------------------------------------------------------------- integrity of the inputs and of the two-commit history
    pub, pls, pk = load(scr, 'pub')
    rc, rls, rk = load(scr, 'rcpt')
    rs, sls, sk = load(scr, 'res')
    for nm, files, ls, kinds in (('publication', pub, pls, pk), ('receipt', rc, rls, rk)):
        chk[f'{nm}: every tree blob is an archive member with the same blob id'] = (
            set(files) == set(ls) and all(blob_id(files[p]) == ls[p][2] for p in ls) and all(ls[p][1] == 'blob' for p in ls))
        chk[f'{nm}: only regular 100644 files (no symlink, executable or submodule)'] = all(ls[p][0] == '100644' for p in ls) and kinds['other'] == 0
    scope = {p for p in sls if p.startswith(CAMP) or p.startswith('rfa/')}
    chk['research: archived scope equals the tree entries (results/<campaign>, rfa)'] = set(rs) == scope and all(blob_id(rs[p]) == sls[p][2] for p in scope)
    pc = (scr / 'pub_commit.txt').read_text()
    rcc = (scr / 'rcpt_commit.txt').read_text()
    hdr_p, msg_p = pc.split('\n\n', 1)
    hdr_r, msg_r = rcc.split('\n\n', 1)
    revs = (scr / 'pub_revlist.txt').read_text().split()
    chk['publication commit: tree and parent as stated; receipt is a root commit; history has exactly these two commits'] = (
        f'tree {PUB_TREE}' in hdr_p.splitlines() and f'parent {RCPT}' in hdr_p.splitlines()
        and not any(x.startswith('parent ') for x in hdr_r.splitlines()) and revs == [PUB, RCPT])
    drive_label = None
    try:
        drive_label = json.loads(pub[PFX + 'results/custody/OFFDEVICE_COPY.json'])['drive']['name']
    except Exception:
        pass
    # ---------------------------------------------------------------- identifier, path and e-mail scan
    for nm, files in (('publication_tree', pub), ('receipt_tree', rc)):
        tot, per_file, abs_hits, binary = Counter(), {}, [], 0
        path_hits = Counter()
        for p, b in files.items():
            t, is_text = text_of(b)
            if not is_text:
                binary += 1
            c, absl = scan_text(t, cids, pb_pats, users, drive_label)
            if not is_text:                                        # also UTF-16 encodings inside binary blobs
                c2, _ = scan_text(b.decode('utf-16-le', 'ignore'), cids, pb_pats, users, drive_label)
                for k in c2:
                    c[k] += c2[k]
            tot.update(c)
            pc_, _ = scan_text(p, cids, pb_pats, users, drive_label)
            path_hits.update({k: v for k, v in pc_.items() if v})
            hit = {k: v for k, v in c.items() if v and k != 'census_markers'}
            if hit:
                per_file[V.safe(p.replace(PFX, ''), cids)] = hit
            abs_hits += [(V.safe(p.replace(PFX, ''), cids), ln, cat) for ln, cat in absl]
        res[nm] = {'files': len(files), 'binary_files': binary, 'totals': dict(tot), 'files_with_hits': per_file,
                   'path_name_hits': dict(path_hits), 'abs_path_hits': abs_hits,
                   'abs_path_categories': dict(Counter(cat for _, _, cat in abs_hits))}
    hist = {}
    for nm, hdr, msg in (('publication_commit', hdr_p, msg_p), ('receipt_commit', hdr_r, msg_r)):
        ident = '\n'.join(x for x in hdr.splitlines() if x.startswith(('author ', 'committer ')))
        ci, _ = scan_text(ident, cids, pb_pats, users, drive_label)
        cm, cm_abs = scan_text(msg, cids, pb_pats, users, drive_label)
        other = '\n'.join(x for x in hdr.splitlines() if not x.startswith(('author ', 'committer ', 'tree ', 'parent ')))
        hist[nm] = {'identity_lines': {k: v for k, v in ci.items() if v}, 'message': {k: v for k, v in cm.items() if v},
                    'message_abs_categories': [c for _, c in cm_abs], 'other_header_lines': len([x for x in other.splitlines() if x.strip()])}
    res['history'] = hist
    # ---------------------------------------------------------------- private content
    suffixes = Counter(Path(p).suffix.lower() for p in pub)
    res['private_content'] = {'suffix_counts': dict(suffixes), 'bad_suffix_files': [V.safe(p, cids) for p in pub if Path(p).suffix.lower() in BAD_SUFFIX],
                              'private_or_runs_paths': [V.safe(p, cids) for p in pub if '.private' in p or '/runs/' in p],
                              'largest_files': sorted(((len(b), V.safe(p.replace(PFX, ''), cids)) for p, b in pub.items()), reverse=True)[:5]}
    big_lists = {}
    for p, b in pub.items():
        if p.endswith('.json'):
            try:
                found = numeric_lists(json.loads(b))
            except Exception as e:
                found = [('PARSE_ERROR', str(type(e).__name__), [], None)]
            if found:
                big_lists[V.safe(p.replace(PFX, ''), cids)] = [(V.safe(a, cids), n, ty, nd) for a, n, ty, nd in found]
    res['private_content']['json_numeric_lists_ge_500'] = big_lists
    csvs = {}
    for p, b in pub.items():
        if p.endswith(('.csv', '.tsv')):
            rows = b.decode('utf-8', 'replace').splitlines()
            header = rows[0].split(',') if rows else []
            csvs[V.safe(p.replace(PFX, ''), cids)] = {'rows': len(rows) - 1, 'columns': [V.safe(h, cids) for h in header],
                                                     'census_or_person_columns': [V.safe(h, cids) for h in header if re.search(
                                                         r'(?i)^(serialno|sporder|pwgtp|puma|sex|agep|wkhp|cow|schl|mar|pobp|person|row_id|label)$', h.strip())]}
    res['private_content']['csv_files'] = csvs
    # verbatim copies of any .private file (size match, then SHA-256)
    by_size = defaultdict(list)
    for p, b in pub.items():
        by_size[len(b)].append(p)
    pub_sha = {p: hashlib.sha256(b).hexdigest() for p, b in pub.items()}
    n_priv, n_hashed, copies = 0, 0, []
    for dp, _, fns in os.walk(ROOT / '.private'):
        for fn in fns:
            fp = Path(dp) / fn
            try:
                sz = fp.stat().st_size
            except OSError:
                continue
            n_priv += 1
            # only aggregate text records are hashed (never an array, model, reader file or person-level table)
            if sz in by_size and sz > 0 and fp.suffix.lower() in ('.json', '.md', '.txt', '.py'):
                n_hashed += 1
                h = hashlib.sha256(fp.read_bytes()).hexdigest()
                copies += [V.safe(p.replace(PFX, ''), cids) for p in by_size[sz] if pub_sha[p] == h]
    res['private_content']['private_files_seen'] = n_priv
    res['private_content']['private_files_size_matched_and_hashed'] = n_hashed
    res['private_content']['verbatim_private_copies_in_tree'] = copies
    chk['no private array, model, reader artefact, archive, log or image suffix in the tree'] = not res['private_content']['bad_suffix_files']
    chk['no .private/ or runs/ path in the tree'] = not res['private_content']['private_or_runs_paths']
    chk['no verbatim copy of any .private file in the tree'] = not copies
    chk['no binary blob in the tree'] = res['publication_tree']['binary_files'] == 0
    # ---------------------------------------------------------------- gated registration files
    rj = json.loads(pub[PFX + 'results/RECEIPTS.json'])['receipts'][0]
    gated = {}
    for f in C.REGISTRATION_FILES:
        pb_, rb, cb = pub.get(PFX + 'results/' + f), rs.get(CAMP + f), rc.get(PFX + 'results/' + f)
        gated[f] = {'pub_equals_research': pb_ is not None and pb_ == rb, 'pub_equals_receipt_tree': pb_ is not None and pb_ == cb,
                    'sha256_equals_receipt_record': pb_ is not None and hashlib.sha256(pb_).hexdigest() == rj['files_sha256'].get(f),
                    'blob_equals_receipt_published_blob': pb_ is not None and blob_id(pb_) == rj['published_blobs'].get(f)}
    res['gated_registration_files'] = gated
    chk['gated registration files byte-identical: publication = research c4bc191e1 = receipt tree = RECEIPTS.json hashes'] = (
        len(gated) == 9 and all(all(x.values()) for x in gated.values()) and sorted(rj['files_sha256']) == sorted(C.REGISTRATION_FILES))
    # ---------------------------------------------------------------- package equals the research commit apart from the sanitizer
    def res_path(p):
        r = p[len(PFX):]
        return CAMP + r[len('results/'):] if r.startswith('results/') else (r[len('code/'):] if r.startswith('code/') else None)

    identical, explained, unexplained, missing_src, newline_files = [], {}, [], [], []
    span_cats = Counter()
    unexplained_spans = []
    for p, b in sorted(pub.items()):
        q = res_path(p)
        if q is None or q not in rs:
            missing_src.append(V.safe(p.replace(PFX, ''), cids))
            continue
        a = rs[q]
        if a == b:
            identical.append(p)
            continue
        try:
            at, bt = a.decode('utf-8'), b.decode('utf-8')
        except UnicodeDecodeError:
            unexplained.append(V.safe(q, cids) + ' (binary differs)')
            continue
        rec = PB.redact_owner(sanitise(at), pb_pats)
        ok_rule = rec == bt
        newline_only = (not ok_rule) and PB.redact_owner(sanitise(at.replace('\r\n', '\n')), pb_pats) == bt
        if newline_only:                                    # Path.read_text() universal newlines: CRLF -> LF, not a recorded substitution
            explained[V.safe(q, cids)] = {'recorded_sanitizer_reproduces_publication': False, 'newline_normalization_only': True,
                                          'crlf_in_research_copy': at.count('\r\n'), 'spans': 0, 'span_categories': {}}
            newline_files.append(V.safe(q, cids))
            continue
        la, lb = at.split('\n'), bt.split('\n')
        n_sp, cats = 0, Counter()
        if len(la) == len(lb):
            for i, (x, y) in enumerate(zip(la, lb), 1):
                if x == y:
                    continue
                al = align(x, y)
                if al is None:
                    cats['UNALIGNED_LINE'] += 1
                    unexplained_spans.append((V.safe(q, cids), i, 'UNALIGNED_LINE'))
                    continue
                for tok, ra in al:
                    if ra == tok:
                        cats['placeholder already in the research copy'] += 1
                        continue
                    n_sp += 1
                    cat = []
                    if any(pt.search(ra) for k, ps in cids.items() if k not in ('git_handle', 'host') for pt in ps):
                        cat.append('owner_identifier')
                    if any(pt.search(ra) for pt in cids['host']):
                        cat.append('host_name')
                    if ABS.search(ra) or ra.startswith(('/Users/', '/Volumes/', '/private/tmp/', '/home/')):
                        cat.append('abs_path')
                    if MAIL.search(ra):
                        cat.append('email')
                    key = f"{'+'.join(cat) or 'OTHER'} -> {tok}"
                    cats[key] += 1
                    if not cat:
                        unexplained_spans.append((V.safe(q, cids), i, key))
        else:
            cats['LINE_COUNT_DIFFERS'] += 1
            unexplained_spans.append((V.safe(q, cids), 0, 'LINE_COUNT_DIFFERS'))
        span_cats.update(cats)
        explained[V.safe(q, cids)] = {'recorded_sanitizer_reproduces_publication': ok_rule, 'spans': n_sp, 'span_categories': dict(cats)}
        if not ok_rule:
            unexplained.append(V.safe(q, cids))
    pub_src = {res_path(p) for p in pub}
    not_published = sorted(V.safe(q, cids) for q in scope if q not in pub_src)
    res['package_vs_research'] = {'publication_files': len(pub), 'identical': len(identical), 'differing': len(explained),
                                  'differing_files': explained, 'differing_not_reproduced_by_recorded_sanitizer': unexplained,
                                  'differing_by_crlf_to_lf_newline_normalization_only': newline_files,
                                  'span_categories_total': dict(span_cats), 'spans_not_placeholder_or_uncategorized': unexplained_spans,
                                  'publication_files_without_research_source': missing_src,
                                  'research_files_in_scope_not_published': not_published}
    chk['every publication file has its research source at c4bc191e1'] = not missing_src
    chk['every differing file equals the recorded sanitizer applied to the research copy'] = not unexplained and not newline_files
    chk['every differing file is explained (recorded sanitizer, or CRLF -> LF newline normalization only)'] = not unexplained
    chk['every replaced span is an identifying string or absolute path replaced by a placeholder'] = not unexplained_spans
    # ---------------------------------------------------------------- headline counts
    pt = res['publication_tree']['totals']
    res['headline'] = {'owner_identifiers_tree': pt.get('owner_union_distinct_spans', 0), 'pb_owner_patterns_tree': pt.get('pb_owner_patterns', 0),
                       'user_name_tree': pt.get('user_name', 0), 'host_tree': pt.get('host', 0), 'email_full_tree': pt.get('email_full', 0),
                       'email_like_tree': pt.get('email_like', 0), 'abs_path_tree': pt.get('abs_path', 0),
                       'abs_path_categories': res['publication_tree']['abs_path_categories'], 'git_handle_tree': pt.get('git_handle', 0),
                       'credential_like_tree': pt.get('credential_like', 0), 'windows_user_path_tree': pt.get('windows_user_path', 0),
                       'tilde_path_tree': pt.get('tilde_path', 0), 'file_url_tree': pt.get('file_url', 0)}
    chk['tree: zero owner identifiers (run-time patterns, both derivations), user names, host names, owner e-mail'] = (
        pt.get('owner_union_distinct_spans', 0) == 0 and pt.get('pb_owner_patterns', 0) == 0 and res['publication_tree']['path_name_hits'] == {})
    chk['tree: zero e-mail-like strings'] = pt.get('email_like', 0) == 0
    chk['tree: no personal home path, other-user home, session or user temp path, or unknown volume'] = not any(
        cat in ('PERSONAL_HOME', 'OTHER_USER_HOME', 'SESSION_OR_USER_TEMP', 'OTHER_VOLUME') or cat.startswith('OTHER_ROOT')
        for _, _, cat in res['publication_tree']['abs_path_hits'])
    chk['commit messages: zero owner identifiers, e-mail-like strings and absolute paths (both commits)'] = all(
        not any(v for k, v in h['message'].items() if k != 'git_handle') for h in hist.values())
    em = (scr / 'git_email.txt').read_text().strip()
    res['observations'] = {
        'commit_identity_lines_carry_configured_git_identity': {
            nm: {'owner_email_full': h['identity_lines'].get('email_full', 0), 'git_handle': h['identity_lines'].get('git_handle', 0)}
            for nm, h in hist.items()},
        'configured_git_email_is_noreply_alias': ('noreply' in em.split('@')[-1].lower()) if '@' in em else None,
        'receipt_commit_already_on_remote': bool(rj.get('remote_sha_verified')) and rj.get('receipt_commit') == RCPT,
        'note': 'author and committer lines of both commits carry the configured git identity (name = the public repository handle; '
                'e-mail = the owner e-mail, the same string the shared sanitizer replaces in file content); pre-existing and already '
                'public through the pushed receipt commit; not introduced by this build'}
    res['checks'] = chk
    res['all_pass'] = all(chk.values())
    res['cpu_s'] = time.process_time() - t0
    res['code_sha256'] = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ('v5_pubscan.py', 'vsafe.py')}
    text = json.dumps(res, indent=1, sort_keys=True, default=str)
    if V.safe(text, cids) != text:                      # never write an identifying string
        raise SystemExit('refused: output would contain an identifying string')
    OUT.write_text(text + '\n')
    print(json.dumps({'all_pass': res['all_pass'], 'failed': [k for k, x in chk.items() if not x], 'headline': res['headline'],
                      'history': hist, 'abs_hits': res['publication_tree']['abs_path_hits'],
                      'differing': {k: (v['spans'], v['span_categories'], v['recorded_sanitizer_reproduces_publication']) for k, v in explained.items()},
                      'not_published': not_published, 'big_lists': big_lists, 'csv': csvs, 'cpu_s': res['cpu_s']}, indent=1, default=str))


if __name__ == '__main__':
    main()
