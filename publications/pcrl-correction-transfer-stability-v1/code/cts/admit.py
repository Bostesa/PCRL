"""Label-free admission of the source artefacts of the PCRL correction-transfer campaign (no label is read).

    python3 -m cts.admit run        writes results/pcrl_correction_transfer_stability_v1/SOURCE_ADMISSION.json

Pins (brief section 1):
- the occupation-headroom publication 1435c395d (publications/pcrl-occupation-headroom-overnight-v1): every published results file
  is compared byte for byte (git blob ids) with the local research copy at the source research tip ca8725e72; files amended after
  publication (the dated post-closeout custody amendment) are listed, not hidden;
- the H3 source publication 89fd2f6e6 (publications/pcrl-income-headroom-successor-v1): admitted only as its manifests require: the
  H3 / HC releases and the source mirror are checked against the occupation-headroom SOURCE_ADMISSION.json (itself byte-identical to
  its published copy), and every release hash used here is searched for in the published records of both publications;
- every private file copied into this worktree (.private/pcrl_occupation_headroom_overnight_v1/{runs,admitted_logs} and the H3 mirror
  .private/pcrl_income_headroom_successor_v1/) is compared bitwise (SHA-256) with the original in the source worktree (read-only).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from . import common as C

SRC_WT = C.ROOT.parent / 'pcrl-occupation-headroom-overnight-v1'          # source worktree (read-only)
OHN_PUB_DIR = 'publications/pcrl-occupation-headroom-overnight-v1/results'
IHS_PUB_DIR = 'publications/pcrl-income-headroom-successor-v1/results'
COPIED = ('.private/pcrl_occupation_headroom_overnight_v1/runs', '.private/pcrl_occupation_headroom_overnight_v1/admitted_logs',
          '.private/pcrl_income_headroom_successor_v1')
USED_RELEASES = {          # (family, config, seeds) of the source campaign used here as witnesses, controls or replay targets
    'A': ('A1_PO_16',), 'B': ('B2_SINGLE',),
    'PROT': ('P2_PAIR_B80', 'P2_PAIR_B50', 'P2_LOC_B80', 'P2_CONT', 'P2A_PAIR_B80', 'P2A_PAIR_B50', 'P2A_LOC_B80', 'P2A_CONT', 'P2_COMP_B80',
             'P2A_COMP_B80'),
}


def _git(*a):
    r = subprocess.run(['git', '-C', str(C.ROOT), *a], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def _sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def published_files(commit, prefix):
    rc, out, err = _git('ls-tree', '-r', commit, '--', prefix + '/')
    if rc != 0:
        raise PermissionError(f'cannot list {commit}:{prefix}: {err}')
    res = {}
    for line in out.splitlines():
        meta, path = line.split('\t', 1)
        res[path[len(prefix) + 1:]] = meta.split()[2]
    return res


def published_text(commit, prefix, files):
    out = {}
    for f in files:
        if f.endswith(('.json', '.csv', '.md')):
            rc, txt, _ = _git('show', f'{commit}:{prefix}/{f}')
            if rc == 0:
                out[f] = txt
    return out


def run():
    rec = {'schema': 'CTS1-source-admission-v1', 'utc': C.now(), 'labels_read': False,
           'pins': {'occupation_headroom_publication': C.OHN_PUBLICATION, 'h3_publication': C.IHS_PUBLICATION,
                    'occupation_headroom_research_tip': C.OHN_TIP}}
    # 1. published results of the occupation-headroom campaign vs the local research copy
    pub = published_files(C.OHN_PUBLICATION, OHN_PUB_DIR)
    same, differ, missing = [], [], []
    for f, blob in sorted(pub.items()):
        p = C.OHN_RESULTS / f
        if not p.is_file():
            missing.append(f); continue
        rc, local, _ = _git('hash-object', str(p))
        (same if local.strip() == blob else differ).append(f)
    from mechanism_design.publish_build import sanitise          # the sanitiser the source publication applied
    after_sanitise, amended = [], []
    for f in differ:
        txt = (C.OHN_RESULTS / f).read_text()
        r = subprocess.run(['git', '-C', str(C.ROOT), 'hash-object', '--stdin'], input=sanitise(txt), capture_output=True, text=True)
        (after_sanitise if r.stdout.strip() == pub[f] else amended).append(f)
    rec['ohn_published_results'] = {'files': len(pub), 'identical': len(same), 'identical_after_source_sanitiser': len(after_sanitise),
                                    'differ_after_sanitiser': amended, 'missing_locally': missing,
                                    'note': 'files that still differ after the source sanitiser were changed on the source research branch after the '
                                            'publication commit: the final closeout re-run recorded with the publication record (commit '
                                            '4b9ccf3e9: ACCESS_RECEIPT, RESOURCE_ACCOUNTING, HANDOFF, COORDINATOR_LOG) and the dated '
                                            'post-closeout custody amendment (commit ca8725e72, deviation 22)'}
    same = same + after_sanitise
    # 2. bitwise copies of every private file
    copies, bad = 0, []
    for rel in COPIED:
        for p in sorted((C.ROOT / rel).rglob('*')):
            if p.is_dir() or '__pycache__' in p.parts:
                continue
            r = p.relative_to(C.ROOT)
            src = SRC_WT / r
            copies += 1
            if not src.is_file() or _sha(src) != _sha(p):
                bad.append(str(r))
    rec['private_copies'] = {'files': copies, 'bitwise_identical_to_source_worktree': copies - len(bad), 'problems': bad}
    # 3. the H3 mirror against the occupation-headroom SOURCE_ADMISSION.json (itself published)
    adm = json.loads((C.OHN_RESULTS / 'SOURCE_ADMISSION.json').read_text())
    mism = [rel for rel, v in adm['copies'].items() if _sha(OC_MIRROR / rel) != v['sha256']]
    rec['h3_mirror'] = {'copies_in_source_admission': len(adm['copies']), 'hash_mismatches': mism,
                        'source_admission_identical_to_published': 'SOURCE_ADMISSION.json' in same}
    # 4. release-level pins searched in the published records of both publications
    ohn_txt = published_text(C.OHN_PUBLICATION, OHN_PUB_DIR, pub)
    ihs_pub = published_files(C.IHS_PUBLICATION, IHS_PUB_DIR)
    ihs_txt = published_text(C.IHS_PUBLICATION, IHS_PUB_DIR, ihs_pub)
    rels = {}
    for fam, cfgs in USED_RELEASES.items():
        for cfg in cfgs:
            for s in C.SEEDS:
                d = C.OHN_RUNS / fam / cfg / f'S{s}'
                if not (d / 'OCC.json').is_file():
                    rels[f'{fam}/{cfg}/S{s}'] = {'present': False}; continue
                o = json.loads((d / 'OCC.json').read_text())
                h = o.get('occ_sha256')
                rels[f'{fam}/{cfg}/S{s}'] = {'present': True, 'occ_sha256': h, 'occ_npz_sha256': _sha(d / 'OCC.npz'),
                                             'published_in': sorted(f for f, t in ohn_txt.items() if h and h in t)}
    for cfg in ('H3_STRONG_LJDIR_PB', 'HC_STRONG'):
        for s in C.SEEDS:
            rj = OC_MIRROR / 'runs' / 'H' / cfg / f'S{s}' / 'RELEASE.json'
            if not rj.is_file():
                rels[f'H/{cfg}/S{s}'] = {'present': False}; continue
            h = json.loads(rj.read_text()).get('release_sha256')
            rels[f'H/{cfg}/S{s}'] = {'present': True, 'release_sha256': h,
                                     'published_in_h3_publication': sorted(f for f, t in ihs_txt.items() if h and h in t),
                                     'published_in_ohn_publication': sorted(f for f, t in ohn_txt.items() if h and h in t)}
    rec['releases_used'] = rels
    unpinned = [k for k, v in rels.items() if v.get('present') and not (v.get('published_in') or v.get('published_in_h3_publication')
                                                                         or v.get('published_in_ohn_publication'))]
    rec['releases_without_a_published_hash'] = unpinned
    rec['ok'] = not bad and not mism and not missing and all(v.get('present') for v in rels.values())
    C.write_json(C.RESULTS / 'SOURCE_ADMISSION.json', rec)
    print(json.dumps({k: v for k, v in rec.items() if k != 'releases_used'}, indent=1))
    return rec


OC_MIRROR = C.ROOT / '.private' / 'pcrl_income_headroom_successor_v1'


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('run',))
    ap.parse_args(argv)
    rec = run()
    return 0 if rec['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
