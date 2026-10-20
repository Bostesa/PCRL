"""Label-free admission of the source artefacts of the PCRL finite-routing comparison campaign (no label is read).

    python3 -m frc.admit run        writes results/pcrl_finite_routing_comparison_v1/SOURCE_ADMISSION.json
    python3 -m frc.admit policy     writes results/pcrl_finite_routing_comparison_v1/ACCESS_POLICY.json

Pins:
- the new-release-mechanisms publication 682068d52 (publications/pcrl-new-release-mechanisms-v1), verified on the GitHub remote with
  `git ls-remote`: every published results file and every published code file (code/nrm/**/*.py) is compared byte for byte (git blob
  ids, after the source's own sanitiser) with the local research copy at this branch's base (research tip fd933a824); files written
  after that publication are listed, not hidden;
- the source chain the new-release-mechanisms campaign admitted: its published SOURCE_ADMISSION.json must be byte-identical to the
  local copy (it pins the controlled training-size publication and, through it, the occupation-headroom and H3 sources and their
  private mirror copies, re-hashed here against the occupation-headroom admission record);
- the private source artefacts this campaign reuses read-only, by SHA-256 against the source's own records: the saved FA twins
  PRC_TWIN and PRC_TWIN_C128 (MODEL.pkl against MODEL.json, RELEASE.npz against RELEASE.json), the stage-1 task caches (TASK.pkl
  against TASK.sha256 and against the twins' recorded task_cache_sha256), and the GATE records;
- the reference releases this campaign compares against (frozen bank, the previous witness W), by SHA-256 of their MODEL_SELECT arrays;
- the interpreters and their numerical library versions; the absence of AGENTS.md at the worktree root and the main checkout; whether
  the off-device drive is attached (never mounted or forced).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from . import common as C

NRM_PUB_PREFIX = 'publications/pcrl-new-release-mechanisms-v1'
POST_PUBLICATION = {'roles/coordinator/COORDINATOR_LOG.md': 'six closing-log lines appended after the publication commit (research '
                    'commit fd933a824: publication pushed, remote verified, queue closed); the published lines are unchanged'}
REFERENCES = ('U0', 'Ucal', 'T_TREE', 'TASK_REFIT', 'NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'H3+TASK_REFIT',
              'H3+T_TREE')
TWINS = ('PRC_TWIN', 'PRC_TWIN_C128')


def _git(*a, inp=None):
    r = subprocess.run(['git', '-C', str(C.ROOT), *a], input=inp, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def _sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run():
    from tsz import admit as TA
    rec = {'schema': 'FRC1-source-admission-v1', 'utc': C.now(), 'labels_read': False,
           'pins': {'new_release_mechanisms_publication': C.NRM_PUBLICATION, 'new_release_mechanisms_research_tip': C.NRM_TIP,
                    'chain': 'controlled training-size ed7bc1ccd and, through it, correction transfer e32a185b4, occupation headroom '
                             '1435c395d and H3 89fd2f6e6 (as admitted by the new-release-mechanisms campaign)'}}
    rc, out, err = _git('ls-remote', 'origin', 'refs/heads/publication/pcrl-new-release-mechanisms-v1')
    rc2, url, _ = _git('remote', 'get-url', 'origin')
    rec['pinned_publication_on_remote'] = {'remote_url': url.strip(), 'ls_remote': out.strip(),
                                           'equal_to_pin': bool(out) and out.split()[0] == C.NRM_PUBLICATION,
                                           'origin_is_github': url.strip() == C.REMOTE_URL}
    rc, _, _ = _git('merge-base', '--is-ancestor', C.NRM_TIP, 'HEAD')
    rec['branch_base_is_nrm_tip'] = rc == 0
    pub_res = TA.published_files(C.NRM_PUBLICATION, NRM_PUB_PREFIX + '/results')
    rec['nrm_published_results'] = TA._compare(pub_res, lambda f: C.NRM_RESULTS / f)
    pub_code = TA.published_files(C.NRM_PUBLICATION, NRM_PUB_PREFIX + '/code/nrm')
    rec['nrm_published_code'] = TA._compare(pub_code, lambda f: C.ROOT / 'nrm' / f)
    local_only = sorted(str(p.relative_to(C.NRM_RESULTS)) for p in C.NRM_RESULTS.rglob('*') if p.is_file()
                        and str(p.relative_to(C.NRM_RESULTS)) not in pub_res and p.suffix not in ('.lock', '.jsonl')
                        and p.name != '.DS_Store' and '__pycache__' not in p.parts)
    rec['nrm_local_results_not_in_publication'] = local_only
    rec['nrm_source_admission_published_identical'] = 'SOURCE_ADMISSION.json' in pub_res and \
        _git('hash-object', str(C.NRM_RESULTS / 'SOURCE_ADMISSION.json'))[1].strip() == pub_res['SOURCE_ADMISSION.json']
    # the occupation-headroom / H3 private mirror copies the chain reads (re-hashed against the source admission record)
    from ohn import common as OC
    adm = OC.admission()['copies']
    bad_adm = [r for r, v in adm.items() if _sha(OC.MIRROR / r) != v['sha256']]
    rec['ohn_mirror_copies'] = {'files': len(adm), 'hash_mismatches': bad_adm[:50], 'n_mismatches': len(bad_adm)}
    # the private source artefacts reused read-only
    arts, problems = {}, []
    for v in C.VERSIONS:
        tdir = C.NRM_FA_CACHE / f'V{v}'
        tsha = _sha(tdir / 'TASK.pkl')
        rec_sha = (tdir / 'TASK.sha256').read_text().strip()
        arts[f'fa_cache/V{v}/TASK.pkl'] = {'sha256': tsha, 'equals_TASK.sha256': tsha == rec_sha}
        if tsha != rec_sha:
            problems.append(f'TASK V{v}')
        for tw in TWINS:
            d = C.NRM_RUNS / 'FA' / tw / f'V{v}'
            meta = json.loads((d / 'MODEL.json').read_text())
            rel = json.loads((d / 'RELEASE.json').read_text())
            msha, rsha = _sha(d / 'MODEL.pkl'), _sha(d / 'RELEASE.npz')
            ok = msha == meta['model_sha256'] and rsha == rel['release_sha256'] and rel.get('task_cache_sha256') == tsha
            arts[f'runs/FA/{tw}/V{v}'] = {'MODEL.pkl': msha, 'RELEASE.npz': rsha, 'model_json_agrees': msha == meta['model_sha256'],
                                          'release_json_agrees': rsha == rel['release_sha256'],
                                          'record_task_cache_sha256_agrees': rel.get('task_cache_sha256') == tsha,
                                          'card': rel.get('mechanism_card'), 'card_sha256': rel.get('card_sha256'),
                                          'saved_utc': rel.get('utc'), 'unit_id': rel.get('unit_id')}
            if not ok:
                problems.append(f'{tw} V{v}')
    for g in sorted((C.NRM_RESULTS / 'roles' / 'designer_a').glob('GATE_V*.json')):
        arts[f'results/designer_a/{g.name}'] = {'sha256': _sha(g), 'published_blob_equal': pub_res.get(
            f'roles/designer_a/{g.name}') == _git('hash-object', str(g))[1].strip()}
    rec['source_private_artefacts'] = arts
    rec['source_private_problems'] = problems
    # reference releases (label-free hashes of their MODEL_SELECT arrays)
    from ohn.devrun import reference_release
    from tsz.evaluate import source_a1, h3_income
    refs = {}
    for v in C.VERSIONS:
        for name in REFERENCES:
            r = reference_release(name, v, 'MODEL_SELECT')
            refs[f'{name}/S{v}'] = {t: C.wire_sha(r[t]) for t in C.TASKS}
        refs[f'W/S{v}'] = {'income': C.wire_sha(h3_income(v)), 'occupation': C.wire_sha(source_a1(v)),
                           'definition': 'A1_PO_16 S<v> occupation + H3_STRONG_LJDIR_PB income S<v>'}
    rec['reference_releases_MODEL_SELECT_wire_sha256'] = refs
    nrm_adm = json.loads((C.NRM_RESULTS / 'SOURCE_ADMISSION.json').read_text())
    nrm_refs = nrm_adm.get('reference_releases_MODEL_SELECT_wire_sha256', {})
    rec['reference_wires_equal_source_admission'] = {k: all(refs[k][t] == nrm_refs[k][t] for t in C.TASKS)
                                                     for k in nrm_refs if k in refs}
    rec['interpreters'] = {'python3': TA._versions('python3'), 'catboost_venv': TA._versions(C.CATBOOST_PY),
                           'torch_venv': TA._versions(C.TORCH_PY)}
    rec['agents_md'] = {'worktree_root': (C.ROOT / 'AGENTS.md').is_file(),
                        'main_checkout': (C.ROOT.parents[2] / 'AGENTS.md').is_file(),
                        'note': 'inspected first (22:52Z): no AGENTS.md exists at either location'}
    rec['offdevice_drive_attached'] = Path('/Volumes/YOTUO').is_dir()
    rec['preserved_verdicts'] = {
        'new_release_mechanisms': 'finite negative result: NO_DEVELOPMENT_CANDIDATE; family A closed at its internal capacity stage '
                                  '(FINITE_CELL_CAPACITY_NOT_ESTABLISHED; only its task-only twins evaluated; no protected fit); '
                                  'family B / FB2 closed by F1 (not reopened here)',
        'controlled_training_size': 'CORRECTION_RECIPE_CLOSED (leaf-score correction; not reopened here)'}
    res = rec['nrm_published_results']
    rec['nrm_results_differ'] = res['differ']
    rec['nrm_results_differ_explained'] = {f: POST_PUBLICATION[f] for f in res['differ'] if f in POST_PUBLICATION}
    unexplained = [f for f in res['differ'] if f not in POST_PUBLICATION]
    rec['nrm_results_differ_unexplained'] = unexplained
    rec['ok'] = (rec['pinned_publication_on_remote']['equal_to_pin'] and rec['pinned_publication_on_remote']['origin_is_github']
                 and rec['branch_base_is_nrm_tip'] and not res['missing_locally'] and not unexplained
                 and not rec['nrm_published_code']['differ'] and not rec['nrm_published_code']['missing_locally']
                 and rec['nrm_source_admission_published_identical'] and not problems and not bad_adm
                 and all(rec['reference_wires_equal_source_admission'].values()))
    C.write_json(C.RESULTS / 'SOURCE_ADMISSION.json', rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ('reference_releases_MODEL_SELECT_wire_sha256', 'interpreters',
                                                                 'source_private_artefacts')}, indent=1))
    return rec


def policy():
    """ACCESS_POLICY.json: the machine-readable copy of frc.common's ROLE x FIELD x OPERATION x PURPOSE policy (label-free)."""
    rec = {'schema': 'FRC1-access-policy-v1', 'utc': C.now(), 'admitted_fields': {r: list(v) for r, v in C.ADMITTED.items()},
           'closed_label_roles': list(C.CLOSED_LABEL_ROLES), 'refused_everywhere': ['SEALED_ASSESSMENT', 'capped-out records'],
           'operations': {k: {**{kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()},
                              'entry_points': list(C.ENTRY_POINTS[k])} for k, v in C.POLICY.items()},
           'gates': ['no label read before the committed registration and its verified public receipt (frc.common.registered)',
                     "'fit' and 'calibrate' reads need a committed, unchanged mechanism card under results/<campaign>/cards/ "
                     '(path, SHA-256 and commit logged with the read)',
                     'MODEL_SELECT labels are never read by a fitting entry point (frc.fa, frc.ctl, frc.fs)',
                     'label reads run only inside a queue unit (FRC_UNIT_ID)',
                     'entry points can be added only by a dated, committed ACCESS_AMENDMENTS.json entry before first use'],
           'source_artefacts': 'the new-release-mechanisms private artefacts are read-only inputs (SOURCE_ADMISSION.json); nothing is '
                               'written under the source campaign\'s paths',
           'log': '.private/<campaign>/access/LABEL_ACCESS.jsonl (append-only; utc, unit id, entry point, role, fields, operation, '
                  'purpose, people, households, row-set hash, card)',
           'configuration_limit': C.CONFIG_LIMIT}
    C.write_json(C.RESULTS / 'ACCESS_POLICY.json', rec)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('run', 'policy'))
    a = ap.parse_args(argv)
    if a.cmd == 'policy':
        print(json.dumps(policy(), indent=1)); return 0
    return 0 if run()['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
