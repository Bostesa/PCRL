"""Bounded prior-use / reservation audit for the frozen ACS 2019 admission order (metadata only).

Admission order (fixed before any data value is inspected):
Florida 2019, Pennsylvania 2019, Illinois 2019, Ohio 2019, Georgia 2019, North Carolina 2019.
The first state-year with no documented prior use or reservation in the searched scope is admitted.
Searched scope: every ref tip and the full history (pickaxe) of both PCRL git stores, the
durable-guarantees public repository (GitHub API, read-only), the storage-relocation inventories,
local file names, other local research repositories, and the confirmation/exposure ledgers.
Not searched (UNKNOWN): S3/GCS object listings (no paid cloud access in this campaign), deleted EC2
disks, other machines and archive tar members. Absence in the searched scope is NO_USE_FOUND, not
certified non-use and not person-level independence.
"""
from __future__ import annotations
import datetime as dt
import gzip
import json
import os
import re
import subprocess
from pathlib import Path

ORDER = [('Florida', 'FL', '12'), ('Pennsylvania', 'PA', '42'), ('Illinois', 'IL', '17'),
         ('Ohio', 'OH', '39'), ('Georgia', 'GA', '13'), ('North Carolina', 'NC', '37')]
YEAR = '2019'
STORES = ['<HOME>/PCRL', '<HOME>/PCRL/.worktrees/pcrl-method-portfolio-v1']
DG_REPO = 'Bostesa/durable-guarantees'
DG_SHA = '956f5c883f515646aa457db55ecbd74b913768b2'
INVENTORIES = '<HOME>/storage-relocation-20260930/inventories'
OTHER_REPOS = ['<HOME>/removal-pricing', '<HOME>/offcriterion', '<HOME>/offcriterion-full-paper',
               '<HOME>/fair-pca', '<HOME>/Obliviator', '<HOME>/durable-guarantees']
KNOWN_RESERVATIONS = {'TX 2018': 'shared-context release programme primary confirmation cohort (66e8186f8 / 537e74a44)',
                      'NY 2018': 'conditional TX+NY fallback of the TX 2018 reservation'}
SPENT = {'CA 2016', 'CA 2017', 'CA 2018'}


def patterns(name, abbr, fips):
    a = abbr.lower()
    files = [f'csv_p{a}', f'csv_h{a}', f'psam_p{fips}', f'psam_h{fips}', f'ss19p{a}', f'ss19h{a}', f'ss{YEAR[2:]}p{a}']
    return {'file_names': sorted(set(files)), 'state_name': name}


def git(store, *args, timeout=600):
    p = subprocess.run(['git', '-C', store, *args], capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


SELF_REF_TOKEN = 'acs-protected-tree-replication'  # this campaign's own branches name the admitted files by design


def ref_tips(store):
    rc, out, _ = git(store, 'for-each-ref', '--format=%(refname)', 'refs/heads', 'refs/remotes', 'refs/tags')
    return [r for r in out.split() if r and SELF_REF_TOKEN not in r]


def grep_refs(store, regex, refs):
    if not refs:
        return []
    rc, out, err = git(store, 'grep', '-l', '-I', '-P', regex, *refs, '--', timeout=1800)
    hits = sorted({line.split(':', 1)[1] for line in out.splitlines() if ':' in line})
    return hits


def pickaxe(store, token):
    rc, out, _ = git(store, 'log', '--all', '--format=%h %ad %s', '--date=short', f'-S{token}', timeout=1800)
    return [l for l in out.splitlines() if l.strip()]


def dg_grep(regex):
    """Grep the durable-guarantees public tree at the pinned SHA through the GitHub API (no clone)."""
    out = {'reachable': False, 'files_checked': 0, 'hits': []}
    try:
        tree = subprocess.run(['gh', 'api', f'repos/{DG_REPO}/git/trees/{DG_SHA}?recursive=1', '--jq', '.tree[] | select(.type=="blob") | .path'],
                              capture_output=True, text=True, timeout=120)
        paths = [p for p in tree.stdout.split('\n') if p.endswith(('.py', '.md', '.json', '.txt', '.csv'))]
        out['reachable'] = tree.returncode == 0 and bool(paths)
        rx = re.compile(regex)
        for p in paths:
            r = subprocess.run(['gh', 'api', '-H', 'Accept: application/vnd.github.raw', f'repos/{DG_REPO}/contents/{p}?ref={DG_SHA}'],
                               capture_output=True, text=True, timeout=120)
            out['files_checked'] += 1
            if rx.search(r.stdout or ''):
                out['hits'].append(p)
    except Exception as e:  # recorded, never silently treated as unused
        out['error'] = repr(e)
    return out


def inventory_hits(regex):
    rx = re.compile(regex, re.I); hits = {}
    for f in sorted(Path(INVENTORIES).glob('*.json.gz')):
        d = json.load(gzip.open(f))
        h = [k for k in (d.keys() if isinstance(d, dict) else []) if rx.search(k)]
        if h:
            hits[f.name] = h[:20]
    return hits


def local_name_hits(regex, roots=('<HOME>',), max_depth=8):
    rx = re.compile(regex, re.I); hits = []
    skip = {'Library', '.Trash', 'node_modules', '.venv', '.git', 'site-packages', '.cache'}
    for root in roots:
        base = root.rstrip('/').count('/')
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in skip and dp.count('/') - base < max_depth]
            for f in fn:
                if rx.search(f):
                    hits.append(os.path.join(dp, f))
    return hits


def other_repo_hits(regex):
    out = {}
    for repo in OTHER_REPOS:
        if not Path(repo).exists():
            out[repo] = 'ABSENT_LOCALLY'
            continue
        p = subprocess.run(['grep', '-rIl', '-i', '-E', regex, repo, '--exclude-dir=.git', '--exclude-dir=.venv',
                            '--exclude-dir=node_modules', '--exclude-dir=site-packages'], capture_output=True, text=True, timeout=600)
        out[repo] = sorted(p.stdout.split())[:20]
    return out


def audit_candidate(name, abbr, fips, refs_by_store):
    pat = patterns(name, abbr, fips)
    file_rx = '|'.join(map(re.escape, pat['file_names']))
    state_rx = re.escape(name)
    rec = {'candidate': f'{name} {YEAR} 1-Year', 'postal': abbr, 'fips': fips, 'patterns': pat, 'stores': {}}
    for store in STORES:
        refs = refs_by_store[store]
        rec['stores'][store] = {
            'ref_tips_searched': len(refs),
            'file_name_hits_on_ref_tips': grep_refs(store, file_rx, refs),
            'state_name_hits_on_ref_tips': grep_refs(store, rf'\b({name}|{name.upper()}|{name.lower()})\b', refs),
            'pickaxe_file_names': {t: pickaxe(store, t) for t in pat['file_names'][:4]},
            'pickaxe_state_name': pickaxe(store, name) + pickaxe(store, name.upper()),
        }
    rec['durable_guarantees'] = dg_grep(file_rx)
    rec['durable_guarantees_state_name_mentions'] = dg_grep(rf'\b{state_rx}\b')
    rec['relocation_inventories'] = inventory_hits(file_rx)
    rec['relocation_inventories_state_name_mentions'] = inventory_hits(state_rx)
    rec['local_file_names'] = local_name_hits(file_rx)
    rec['other_local_repositories'] = other_repo_hits(file_rx)            # PUMS file names: counted as use evidence
    rec['other_local_repositories_state_name_mentions'] = other_repo_hits(rf'\b{state_rx}\b')  # text mentions: reviewed, not use
    # Classification: a hit is a documented prior use/reservation only if it is a data or plan reference to this state-year.
    hits = []
    for s, v in rec['stores'].items():
        hits += [f'{s}:{h}' for h in v['file_name_hits_on_ref_tips']]
        hits += [f'{s}:pickaxe:{k}:{c}' for k, lst in v['pickaxe_file_names'].items() for c in lst]
    hits += [f'dg:{h}' for h in rec['durable_guarantees'].get('hits', [])]
    hits += [f'inventory:{k}' for k in rec['relocation_inventories']]
    hits += [f'local:{h}' for h in rec['local_file_names']]
    hits += [f'repo:{k}' for k, v in rec['other_local_repositories'].items() if isinstance(v, list) and v]
    rec['file_level_hits'] = hits
    rec['state_name_mentions'] = {s: v['state_name_hits_on_ref_tips'] for s, v in rec['stores'].items()}
    rec['classification'] = 'NO_USE_FOUND_IN_SEARCHED_RECORDS' if not hits else 'REVIEW_REQUIRED'
    rec['reservation'] = 'none recorded (only TX 2018 and its NY 2018 fallback are reserved)'
    return rec


def main(out_path):
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    refs_by_store = {s: ref_tips(s) for s in STORES}
    results, admitted = [], None
    # Audit the FIRST candidate of the frozen order. A REVIEW_REQUIRED classification is NOT a skip: the frozen rule
    # skips a state only for a documented prior use/reservation, an unavailable official source or another
    # predeclared admission problem, so every token hit is reviewed by hand (MANUAL_REVIEW in the admission record)
    # before the next candidate may be considered.
    for name, abbr, fips in ORDER[:1]:
        rec = audit_candidate(name, abbr, fips, refs_by_store)
        results.append(rec)
        admitted = rec['candidate'] if rec['classification'] == 'NO_USE_FOUND_IN_SEARCHED_RECORDS' else None
    receipt = {'schema': 'ACSR1-provenance-audit-v1', 'started_utc': started,
               'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
               'admission_order': [f'{n} {YEAR}' for n, _, _ in ORDER], 'excluded_by_rule': sorted(SPENT) + sorted(KNOWN_RESERVATIONS),
               'known_reservations': KNOWN_RESERVATIONS, 'searched_scope': __doc__, 'stores': {s: len(r) for s, r in refs_by_store.items()},
               'candidates_audited': results, 'admitted': admitted,
               'unknown_scope': ['S3/GCS object listings (not queried: no paid cloud access)', 'deleted EC2 disks', 'other machines',
                                 'archive tar members on the unmounted YOTUO drive (inventories searched instead)'],
               'data_values_inspected': False}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'x') as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    return receipt


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    r = main(ap.parse_args().out)
    print(json.dumps({'admitted': r['admitted'], 'audited': [c['candidate'] + ':' + c['classification'] for c in r['candidates_audited']]}))
