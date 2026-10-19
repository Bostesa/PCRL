"""Label-free admission of the source artefacts of the PCRL controlled training-size campaign (no label is read).

    python3 -m tsz.admit run        writes results/pcrl_controlled_training_size_v1/SOURCE_ADMISSION.json

Pins:
- the correction-transfer publication e32a185b4 (publications/pcrl-correction-transfer-stability-v1): every published results file
  and every published code file (code/cts/*.py) is compared byte for byte (git blob ids, after the source's own sanitiser) with the
  local research copy at this branch's base (research tip 507024649); files written after that publication (its publication record)
  are listed, not hidden;
- the occupation-headroom publication 1435c395d and the H3 publication 89fd2f6e6, through the chain the correction-transfer campaign
  admitted: its published SOURCE_ADMISSION.json must be byte-identical to the local copy, and every private file of the two source
  copies (.private/pcrl_occupation_headroom_overnight_v1/{runs,admitted_logs}, .private/pcrl_income_headroom_successor_v1) is compared
  bitwise (SHA-256) with the original in the occupation-headroom worktree (read-only);
- the source releases this campaign uses (A1_PO_16 S0 / S1, H3_STRONG_LJDIR_PB S0 / S1, the frozen bank S0 / S1, the admitted T_TREE
  teachers) by SHA-256, each searched for in the published records;
- the interpreters (python3, the CatBoost venv, the torch venv) and their numerical library versions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from . import common as C

CTS_PUB_PREFIX = 'publications/pcrl-correction-transfer-stability-v1'
CTS_RESULTS = C.ROOT / 'results' / 'pcrl_correction_transfer_stability_v1'
OHN_WT = C.ROOT.parent / 'pcrl-occupation-headroom-overnight-v1'           # source worktree (read-only)
OHN_RESULTS = C.ROOT / 'results' / C.OHN_CAMPAIGN
OHN_PUB_DIR = 'publications/pcrl-occupation-headroom-overnight-v1/results'
IHS_PUB_DIR = 'publications/pcrl-income-headroom-successor-v1/results'
COPIED = ('.private/pcrl_occupation_headroom_overnight_v1/runs', '.private/pcrl_occupation_headroom_overnight_v1/admitted_logs',
          '.private/pcrl_income_headroom_successor_v1')
MIRROR = C.ROOT / '.private' / 'pcrl_income_headroom_successor_v1'
POST_PUBLICATION = ('PUBLICATION_RECORD.json',)        # written after the final push (not part of the commit it describes)


def _git(*a, inp=None):
    r = subprocess.run(['git', '-C', str(C.ROOT), *a], input=inp, capture_output=True, text=True)
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


def _compare(pub, local_of):
    from mechanism_design.publish_build import sanitise
    same, sanitised, differ, missing = [], [], [], []
    for f, blob in sorted(pub.items()):
        p = local_of(f)
        if not p.is_file():
            missing.append(f); continue
        rc, local, _ = _git('hash-object', str(p))
        if local.strip() == blob:
            same.append(f); continue
        try:
            txt = p.read_text()
        except UnicodeDecodeError:
            differ.append(f); continue
        rc, h, _ = _git('hash-object', '--stdin', inp=sanitise(txt))
        (sanitised if h.strip() == blob else differ).append(f)
    return {'files': len(pub), 'identical': len(same), 'identical_after_source_sanitiser': len(sanitised), 'differ': differ,
            'missing_locally': missing}


def _versions(exe):
    probe = ('import json, sys, importlib.metadata as m\nout = {"python": sys.version.split()[0]}\n'
             'for p in ("numpy", "scipy", "scikit-learn", "torch", "catboost"):\n'
             '    try:\n        out[p] = m.version(p)\n    except Exception:\n        out[p] = None\nprint(json.dumps(out))')
    r = subprocess.run([str(exe), '-c', probe], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {'error': r.stderr[-300:]}


def run():
    rec = {'schema': 'TSZ1-source-admission-v1', 'utc': C.now(), 'labels_read': False,
           'pins': {'correction_transfer_publication': C.CTS_PUBLICATION, 'correction_transfer_research_tip': C.CTS_TIP,
                    'occupation_headroom_publication': C.OHN_PUBLICATION, 'h3_publication': C.IHS_PUBLICATION}}
    rc, head, _ = _git('merge-base', '--is-ancestor', C.CTS_TIP, 'HEAD')
    rec['branch_base_is_cts_tip'] = rc == 0
    # 1. the correction-transfer publication vs the local research copy (results + code)
    pub_res = published_files(C.CTS_PUBLICATION, CTS_PUB_PREFIX + '/results')
    rec['cts_published_results'] = _compare(pub_res, lambda f: CTS_RESULTS / f)
    pub_code = published_files(C.CTS_PUBLICATION, CTS_PUB_PREFIX + '/code/cts')
    rec['cts_published_code'] = _compare(pub_code, lambda f: C.ROOT / 'cts' / f)
    local_only = sorted(str(p.relative_to(CTS_RESULTS)) for p in CTS_RESULTS.rglob('*') if p.is_file()
                        and str(p.relative_to(CTS_RESULTS)) not in pub_res and p.suffix not in ('.lock', '.jsonl'))
    rec['cts_local_results_not_in_publication'] = local_only
    # 2. the source chain: the correction-transfer SOURCE_ADMISSION (published) and bitwise private copies
    rec['cts_source_admission_published_identical'] = 'SOURCE_ADMISSION.json' in pub_res and \
        _git('hash-object', str(CTS_RESULTS / 'SOURCE_ADMISSION.json'))[1].strip() == pub_res['SOURCE_ADMISSION.json']
    copies, bad = 0, []
    for rel in COPIED:
        for p in sorted((C.ROOT / rel).rglob('*')):
            if p.is_dir() or '__pycache__' in p.parts:
                continue
            r = p.relative_to(C.ROOT)
            copies += 1
            src = OHN_WT / r
            if not src.is_file() or _sha(src) != _sha(p):
                bad.append(str(r))
    rec['private_copies'] = {'files': copies, 'bitwise_identical_to_source_worktree': copies - len(bad), 'problems': bad[:50],
                             'n_problems': len(bad)}
    adm = json.loads((OHN_RESULTS / 'SOURCE_ADMISSION.json').read_text())
    mism = [rel for rel, v in adm['copies'].items() if _sha(MIRROR / rel) != v['sha256']]
    rec['h3_mirror'] = {'copies_in_source_admission': len(adm['copies']), 'hash_mismatches': mism}
    # 3. releases used here, searched for in the published records
    ohn_pub = published_files(C.OHN_PUBLICATION, OHN_PUB_DIR)
    ohn_txt = published_text(C.OHN_PUBLICATION, OHN_PUB_DIR, ohn_pub)
    ihs_pub = published_files(C.IHS_PUBLICATION, IHS_PUB_DIR)
    ihs_txt = published_text(C.IHS_PUBLICATION, IHS_PUB_DIR, ihs_pub)
    from ohn import common as OC
    rels = {}
    for s in C.SEEDS:
        d = OC.RUNS / 'A' / 'A1_PO_16' / f'S{s}'
        o = json.loads((d / 'OCC.json').read_text())
        h = o.get('occ_sha256')
        rels[f'A/A1_PO_16/S{s}'] = {'occ_sha256': h, 'occ_npz_sha256': _sha(d / 'OCC.npz'), 'T_star': o.get('T_star'),
                                    'published_in': sorted(f for f, t in ohn_txt.items() if h and h in t)}
        rj = MIRROR / 'runs' / 'H' / 'H3_STRONG_LJDIR_PB' / f'S{s}' / 'RELEASE.json'
        h = json.loads(rj.read_text()).get('release_sha256')
        rels[f'H/H3_STRONG_LJDIR_PB/S{s}'] = {'release_sha256': h,
                                               'published_in_h3_publication': sorted(f for f, t in ihs_txt.items() if h and h in t)}
        bank = OC.admitted(f'admitted/bank/FROZEN_BANK_S{s}.npz')
        rels[f'FROZEN_BANK/S{s}'] = {'sha256': _sha(bank), 'checked_against': 'FROZEN_BANK_MANIFEST (ohn.common.frozen_bank)'}
        for t in C.TASKS:
            te = OC.teacher(s, t)
            rels[f'T_TREE/{t}/S{s}'] = {'teacher_sha256': te.sha(), 'T': te.T}
    rec['releases_used'] = rels
    rec['interpreters'] = {'python3': _versions('python3'), 'catboost_venv': _versions(C.CATBOOST_PY), 'torch_venv': _versions(C.TORCH_PY)}
    rec['preserved_verdicts'] = {
        'occupation_headroom': 'NO_QUALIFYING_CONFIGURATION (unchanged, never reinterpreted)',
        'correction_transfer': 'a replicated, independently replayed diagnosis and a negative method result (NO_RECIPE; claim ladder rung 0)',
        'h3_income': 'H3_STRONG_LJDIR_PB kept income Brier / NLL headroom at 100% but was not competitive; its income view alone fails the old '
                     'recipient gate; its recipe refitted on about half of MODEL_FIT lost the headroom (training size not separated)'}
    cts_res = rec['cts_published_results']
    unexplained = [f for f in cts_res['differ'] if f not in POST_PUBLICATION]
    rec['ok'] = (rec['branch_base_is_cts_tip'] and not cts_res['missing_locally'] and not unexplained
                 and not rec['cts_published_code']['differ'] and not rec['cts_published_code']['missing_locally']
                 and rec['cts_source_admission_published_identical'] and not bad and not mism)
    rec['cts_results_differ_unexplained'] = unexplained
    C.write_json(C.RESULTS / 'SOURCE_ADMISSION.json', rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ('releases_used', 'interpreters')}, indent=1))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('run',))
    ap.parse_args(argv)
    rec = run()
    return 0 if rec['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
