"""Label-free admission of the source artefacts of the PCRL strong-anchor release campaign (no label is read).

Derived from `ipr.admit` (itself derived from `ddr.admit`). Added for this campaign: the input-and-pair release publication
9c2aa4d99 (publications/pcrl-input-and-pair-release-v1; child of its registration receipt e909fb8c0), verified on the remote and
compared file by file with the local research copy at this branch's base (9c9f95385); its private records read by the anchor table
(the IPR1_MAIN grid GRID.json / BOOT.npz, the MODEL_SELECT utility records of the references, the saved releases) by SHA-256; and
the working cache (APFS clones of the source worktree's .private/<campaign> trees, compared file by file with the source trees).
Kept from the input-and-pair admission: the dual-design release publication e30cf1c6b
(publications/pcrl-dual-design-release-v1; child of its registration receipt c5a84c19d), verified on the remote and compared file
by file with the local research copy at this branch's base (c577424c0), and its private saved releases (MODEL.pkl against
MODEL.json, RELEASE.npz against RELEASE.json; the shared twin DA_TWIN and the inherited references DB_PTC_R1 / DB_ZTILT_R1 are read
from them, read-only). The finite-solver-adequacy pins (ddc231a97), the finite-routing comparison pins (publication 04ac9161f, its
private release artefacts and SEX-side teacher caches) and the new-release-mechanisms pins (its task caches and twins) stay, all
read-only.

    python3 -m sar.admit run        writes results/pcrl_strong_anchor_release_v1/SOURCE_ADMISSION.json
    python3 -m sar.admit policy     writes results/pcrl_strong_anchor_release_v1/ACCESS_POLICY.json

Pins:
- the new-release-mechanisms publication 682068d52 (publications/pcrl-new-release-mechanisms-v1), verified on the GitHub remote with
  `git ls-remote`: every published results file and every published code file (code/nrm/**/*.py) is compared byte for byte (git blob
  ids, after the source's own sanitiser) with the local research copy at the new-release-mechanisms research tip fd933a824; files written
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
FRC_PUB_PREFIX = 'publications/pcrl-finite-routing-comparison-v1'
FRC_POST_PUBLICATION = {'PUBLICATION_RECEIPT.json': 'written after the publication commit 04ac9161f (local record of the push and its checks)'}
FSA_PUB_PREFIX = 'publications/pcrl-finite-solver-adequacy-v1'
FSA_POST_PUBLICATION = {'PUBLICATION_RECEIPT.json': 'written after the publication commit ddc231a97 (local record of the push and its checks)'}
DDR_PUB_PREFIX = 'publications/pcrl-dual-design-release-v1'
IPR_PUB_PREFIX = 'publications/pcrl-input-and-pair-release-v1'
IPR_POST_PUBLICATION = {
    'PUBLICATION_RECEIPT.json': 'written after the publication commit 9c2aa4d99 (local record of the push and its remote checks; '
                                'research commit 9c9f95385)',
    'PROTOCOL_DEVIATIONS.md': 'deviation 13 appended after the publication build (research commit 9c9f95385: 8 lines added, 0 removed)',
    'roles/coordinator/COORDINATOR_LOG.md': 'publication lines appended after the publication build (research commit 9c9f95385: 11 '
                                            'lines added, 0 removed)'}
CACHED_TREES = ('pcrl_controlled_training_size_v1', 'pcrl_correction_transfer_stability_v1', 'pcrl_dual_design_release_v1',
                'pcrl_finite_routing_comparison_v1', 'pcrl_finite_solver_adequacy_v1', 'pcrl_income_headroom_successor_v1',
                'pcrl_input_and_pair_release_v1', 'pcrl_new_release_mechanisms_v1', 'pcrl_occupation_headroom_overnight_v1')
DDR_POST_PUBLICATION = {
    'PUBLICATION_RECEIPT.json': 'written after the publication commit e30cf1c6b (local record of the push and its remote checks; '
                                'research commit c577424c0)',
    'roles/coordinator/verify_publication.py': 'changed after the publication build (research commit c577424c0): the verification '
                                               'script re-applies the sanitizer before comparing code files',
    'roles/coordinator/COORDINATOR_LOG.md': 'publication lines appended after the publication build (research commit c577424c0)'}


def _git(*a, inp=None):
    r = subprocess.run(['git', '-C', str(C.ROOT), *a], input=inp, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def _sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def working_cache():
    """Each cached tree (this worktree's .private/<campaign>, an APFS clone made 2026-10-24T04:11Z) against the source worktree's
    .private/<campaign>: every regular file's size and SHA-256 must agree, and no file may exist on one side only (symlinks are
    compared by target). Lock files are reported, not failed."""
    out, ok_all = {}, True
    for t in CACHED_TREES:
        a, b = C.ROOT / '.private' / t, C.LINEAGE_WORKTREE / '.private' / t
        fa = {str(q.relative_to(a)): q for q in a.rglob('*') if q.is_file() or q.is_symlink()}
        fb = {str(q.relative_to(b)): q for q in b.rglob('*') if q.is_file() or q.is_symlink()}
        only_a, only_b = sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))
        diff, nbytes = [], 0
        for r in sorted(set(fa) & set(fb)):
            qa, qb = fa[r], fb[r]
            if qa.is_symlink() or qb.is_symlink():
                if not (qa.is_symlink() and qb.is_symlink() and qa.readlink() == qb.readlink()):
                    diff.append(r)
                continue
            nbytes += qa.stat().st_size
            if qa.stat().st_size != qb.stat().st_size or _sha(qa) != _sha(qb):
                diff.append(r)
        ok = not diff and not [x for x in only_a + only_b if not x.endswith('.lock')]
        ok_all &= ok
        out[t] = {'files': len(fa), 'bytes': nbytes, 'differ': diff[:20], 'n_differ': len(diff), 'only_in_cache': only_a[:20],
                  'only_in_source': only_b[:20], 'ok': ok}
    return {'source_worktree': '<repository>/.claude/worktrees/' + C.LINEAGE_WORKTREE.name, 'method': 'cp -cRp (APFS clone) at 2026-10-24T04:11:26Z-04:11:31Z',
            'trees': out, 'ok': ok_all}


def run():
    from tsz import admit as TA
    rec = {'schema': 'SAR1-source-admission-v1', 'utc': C.now(), 'labels_read': False,
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
    # the finite-routing comparison campaign (this branch's base): publication, results, code and private artefacts (read-only)
    rc, out, err = _git('ls-remote', 'origin', 'refs/heads/publication/pcrl-finite-routing-comparison-v1')
    rec['frc_publication_on_remote'] = {'ls_remote': out.strip(), 'equal_to_pin': bool(out) and out.split()[0] == C.FRC_PUBLICATION}
    rc, _, _ = _git('merge-base', '--is-ancestor', C.FRC_TIP, 'HEAD')
    rec['branch_base_is_frc_tip'] = rc == 0
    frc_res = TA.published_files(C.FRC_PUBLICATION, FRC_PUB_PREFIX + '/results')
    rec['frc_published_results'] = TA._compare(frc_res, lambda f: C.FRC_RESULTS / f)
    frc_code = TA.published_files(C.FRC_PUBLICATION, FRC_PUB_PREFIX + '/code/frc')
    rec['frc_published_code'] = TA._compare(frc_code, lambda f: C.ROOT / 'frc' / f)
    rec['frc_local_results_not_in_publication'] = sorted(
        str(q.relative_to(C.FRC_RESULTS)) for q in C.FRC_RESULTS.rglob('*') if q.is_file()
        and str(q.relative_to(C.FRC_RESULTS)) not in frc_res and q.suffix not in ('.lock', '.jsonl') and q.name != '.DS_Store'
        and '__pycache__' not in q.parts)
    frc_arts, frc_problems = {}, []
    for d in sorted(C.FRC_RUNS.glob('FA/*/V*')):
        if not (d / 'MODEL.pkl').is_file():
            continue
        meta = json.loads((d / 'MODEL.json').read_text())
        rel = json.loads((d / 'RELEASE.json').read_text())
        msha, rsha = _sha(d / 'MODEL.pkl'), _sha(d / 'RELEASE.npz')
        ok = msha == meta['model_sha256'] and rsha == rel['release_sha256']
        frc_arts[f'frc/runs/FA/{d.parent.name}/{d.name}'] = {
            'MODEL.pkl': msha, 'RELEASE.npz': rsha, 'model_json_agrees': msha == meta['model_sha256'],
            'release_json_agrees': rsha == rel['release_sha256'], 'card': rel.get('mechanism_card'), 'saved_utc': rel.get('utc'),
            'unit_id': rel.get('unit_id'), 'wire_sha256': rel.get('wire_sha256')}
        if not ok:
            frc_problems.append(f'{d.parent.name} {d.name}')
    for v in C.VERSIONS:
        q = C.FRC_FA_CACHE / f'V{v}' / 'SEXSIDE.pkl'
        ssha = _sha(q)
        side = (C.FRC_FA_CACHE / f'V{v}' / 'SEXSIDE.sha256').read_text().strip()
        sel = json.loads((C.FRC_RESULTS / 'fa' / f'SELECTION_V{v}.json').read_text())
        agrees = ssha == side and sel.get('sexside_cache_sha256') == ssha
        frc_arts[f'frc/fa_cache/V{v}/SEXSIDE.pkl'] = {'sha256': ssha, 'equals_sidecar': ssha == side,
                                                     'equals_selection_record': sel.get('sexside_cache_sha256') == ssha}
        if not agrees:
            frc_problems.append(f'SEXSIDE V{v}')
    for f in ('decisions/QUAL_DECISION.json', 'fa/SELECTION_V0.json', 'fa/SELECTION_V1.json', 'fa/UNT_SELECTION_V0.json',
              'fa/UNT_SELECTION_V1.json', 'fa/PROBE_STRENGTH_V0.json', 'fa/PROBE_STRENGTH_V1.json', 'fa/NEST_C64_V0.json',
              'fa/NEST_C64_V1.json', 'claims/MAIN_ER1/DECISIONS.json', 'claims/R1_ER1/DECISIONS.json', 'claims/MAIN/DECISIONS.json'):
        frc_arts[f'frc/results/{f}'] = {'sha256': _sha(C.FRC_RESULTS / f),
                                        'published_blob_equal': frc_res.get(f) == _git('hash-object', str(C.FRC_RESULTS / f))[1].strip()}
    rec['frc_private_artefacts'] = frc_arts
    rec['frc_private_problems'] = frc_problems
    # the finite-solver-adequacy campaign (this branch's base): publication, results and code (read-only)
    rc, out, err = _git('ls-remote', 'origin', 'refs/heads/publication/pcrl-finite-solver-adequacy-v1')
    rec['fsa_publication_on_remote'] = {'ls_remote': out.strip(),
                                        'equal_to_pin': bool(out) and out.split()[0].startswith(C.FSA_PUBLICATION)}
    rc, _, _ = _git('merge-base', '--is-ancestor', C.FSA_TIP, 'HEAD')
    rec['branch_base_is_fsa_tip'] = rc == 0
    fsa_full = out.split()[0] if out else C.FSA_PUBLICATION
    fsa_res = TA.published_files(fsa_full, FSA_PUB_PREFIX + '/results')
    rec['fsa_published_results'] = TA._compare(fsa_res, lambda f: C.FSA_RESULTS / f)
    fsa_code = TA.published_files(fsa_full, FSA_PUB_PREFIX + '/code/fsa')
    rec['fsa_published_code'] = TA._compare(fsa_code, lambda f: C.ROOT / 'fsa' / f)
    rec['fsa_local_results_not_in_publication'] = sorted(
        str(q.relative_to(C.FSA_RESULTS)) for q in C.FSA_RESULTS.rglob('*') if q.is_file()
        and str(q.relative_to(C.FSA_RESULTS)) not in fsa_res and q.suffix not in ('.lock', '.jsonl') and q.name != '.DS_Store'
        and '__pycache__' not in q.parts)
    # the dual-design release campaign (this branch's base): publication, results, code and private saved releases (read-only)
    rc, out, err = _git('ls-remote', 'origin', 'refs/heads/publication/pcrl-dual-design-release-v1')
    rec['ddr_publication_on_remote'] = {'ls_remote': out.strip(), 'equal_to_pin': bool(out) and out.split()[0] == C.DDR_PUBLICATION}
    rc, par, _ = _git('rev-parse', f'{C.DDR_PUBLICATION}^')
    rec['ddr_publication_parent_is_receipt'] = par.strip() == C.DDR_RECEIPT
    rc, _, _ = _git('merge-base', '--is-ancestor', C.DDR_TIP, 'HEAD')
    rec['branch_base_is_ddr_tip'] = rc == 0
    ddr_res = TA.published_files(C.DDR_PUBLICATION, DDR_PUB_PREFIX + '/results')
    rec['ddr_published_results'] = TA._compare(ddr_res, lambda f: C.DDR_RESULTS / f)
    ddr_code = TA.published_files(C.DDR_PUBLICATION, DDR_PUB_PREFIX + '/code/ddr')
    rec['ddr_published_code'] = TA._compare(ddr_code, lambda f: C.ROOT / 'ddr' / f)
    rec['ddr_local_results_not_in_publication'] = sorted(
        str(q.relative_to(C.DDR_RESULTS)) for q in C.DDR_RESULTS.rglob('*') if q.is_file()
        and str(q.relative_to(C.DDR_RESULTS)) not in ddr_res and q.suffix not in ('.lock', '.jsonl') and q.name != '.DS_Store'
        and '__pycache__' not in q.parts)
    ddr_arts, ddr_problems = {}, []
    for d in sorted(C.DDR_RUNS.glob('*/*/V*')):
        if not (d / 'RELEASE.json').is_file():
            continue
        rel = json.loads((d / 'RELEASE.json').read_text())
        rsha = _sha(d / 'RELEASE.npz')
        msha = _sha(d / 'MODEL.pkl') if (d / 'MODEL.pkl').is_file() else None
        meta = json.loads((d / 'MODEL.json').read_text()) if (d / 'MODEL.json').is_file() else {}
        ok = rsha == rel['release_sha256'] and (msha is None or msha == meta.get('model_sha256'))
        ddr_arts[f'ddr/runs/{d.parent.parent.name}/{d.parent.name}/{d.name}'] = {
            'RELEASE.npz': rsha, 'MODEL.pkl': msha, 'release_json_agrees': rsha == rel['release_sha256'],
            'model_json_agrees': None if msha is None else msha == meta.get('model_sha256'), 'kind': rel.get('kind'),
            'card': rel.get('mechanism_card'), 'saved_utc': rel.get('utc'), 'wire_sha256': rel.get('wire_sha256')}
        if not ok:
            ddr_problems.append(f'{d.parent.parent.name}/{d.parent.name} {d.name}')
    rec['ddr_private_artefacts'] = ddr_arts
    rec['ddr_private_problems'] = ddr_problems
    # the input-and-pair release campaign (this branch's base): publication, results, code and private records (read-only)
    rc, out, err = _git('ls-remote', 'origin', 'refs/heads/publication/pcrl-input-and-pair-release-v1')
    rec['ipr_publication_on_remote'] = {'ls_remote': out.strip(), 'equal_to_pin': bool(out) and out.split()[0] == C.IPR_PUBLICATION}
    rc, par, _ = _git('rev-parse', f'{C.IPR_PUBLICATION}^')
    rec['ipr_publication_parent_is_receipt'] = par.strip() == C.IPR_RECEIPT
    rc, _, _ = _git('merge-base', '--is-ancestor', C.IPR_TIP, 'HEAD')
    rec['branch_base_is_ipr_tip'] = rc == 0
    ipr_res = TA.published_files(C.IPR_PUBLICATION, IPR_PUB_PREFIX + '/results')
    rec['ipr_published_results'] = TA._compare(ipr_res, lambda f: C.IPR_RESULTS / f)
    ipr_code = TA.published_files(C.IPR_PUBLICATION, IPR_PUB_PREFIX + '/code/ipr')
    rec['ipr_published_code'] = TA._compare(ipr_code, lambda f: C.ROOT / 'ipr' / f)
    rec['ipr_local_results_not_in_publication'] = sorted(
        str(q.relative_to(C.IPR_RESULTS)) for q in C.IPR_RESULTS.rglob('*') if q.is_file()
        and str(q.relative_to(C.IPR_RESULTS)) not in ipr_res and q.suffix not in ('.lock', '.jsonl') and q.name != '.DS_Store'
        and '__pycache__' not in q.parts)
    ipr_arts, ipr_problems = {}, []
    for d in sorted(C.IPR_RUNS.glob('*/*/V*')) + sorted(C.IPR_RUNS.glob('SOURCE_DDR/*/*/V*')):
        if not (d / 'RELEASE.json').is_file():
            continue
        rel = json.loads((d / 'RELEASE.json').read_text())
        rsha = _sha(d / 'RELEASE.npz')
        msha = _sha(d / 'MODEL.pkl') if (d / 'MODEL.pkl').is_file() else None
        meta = json.loads((d / 'MODEL.json').read_text()) if (d / 'MODEL.json').is_file() else {}
        ok = rsha == rel['release_sha256'] and (msha is None or msha == meta.get('model_sha256'))
        ipr_arts[str(d.relative_to(C.IPR_RUNS))] = {'RELEASE.npz': rsha, 'MODEL.pkl': msha, 'release_json_agrees': rsha == rel['release_sha256'],
                                                    'model_json_agrees': None if msha is None else msha == meta.get('model_sha256'),
                                                    'kind': rel.get('kind'), 'wire_sha256': rel.get('wire_sha256')}
        if not ok:
            ipr_problems.append(str(d.relative_to(C.IPR_RUNS)))
    for f in ('GRID/IPR1_MAIN/S0/GRID.json', 'GRID/IPR1_MAIN/S0/BOOT.npz', 'GRID/IPR1_MAIN/S1/GRID.json', 'GRID/IPR1_MAIN/S1/BOOT.npz',
              'REFERENCES/EVAL_MS.json', 'CTL/CTL_TWIN/EVAL_MS.json', 'SOURCE_DDR/DB/DB_PTC_R1/EVAL_MS.json'):
        ipr_arts[f] = {'sha256': _sha(C.IPR_RUNS / f)}
    rec['ipr_private_artefacts'] = ipr_arts
    rec['ipr_private_problems'] = ipr_problems
    # the working cache: APFS clones of the source worktree's private trees (compared file by file; read-only use)
    rec['working_cache'] = working_cache()
    # the protected-tree replication files this campaign's families read directly (read-only; outside the working cache)
    rep = C.ROOT.parents[2] / '.worktrees' / 'pcrl-acs-protected-tree-replication-v1' / '.private' / 'acs_replication' / 'runs'
    rin, rprob = {}, []
    for v in C.VERSIONS:
        seed_a = json.loads((rep / f'STAGE_A_{v}' / 'STAGE_A_SEED.json').read_text())
        u0_rec = None
        stack = [seed_a]
        while stack:                                   # the U0_MODEL.pt hash recorded anywhere in the Stage A seed record
            o = stack.pop()
            if isinstance(o, dict):
                for k, x in o.items():
                    if k == 'U0_MODEL.pt' and isinstance(x, str):
                        u0_rec = x
                    elif isinstance(x, (dict, list)):
                        stack.append(x)
            elif isinstance(o, list):
                stack.extend(o)
        u0 = _sha(rep / f'STAGE_A_{v}' / 'U0_MODEL.pt')
        nr_seed = json.loads((rep / f'NEURAL_RAW_{v}' / 'NEURAL_RAW_SEED.json').read_text())
        ck = {}
        for rho in ('1.0', '3.0', '10.0'):
            for e in ('E0', 'E2', 'E5', 'E10', 'E20'):
                f = rep / f'NEURAL_RAW_{v}' / f'NEURAL_RAW__rho{rho}__{e}.pt'
                rec_sha = ((nr_seed.get('checkpoints') or {}).get(rho) or {}).get(e, {}).get('sha256')
                h = _sha(f)
                ck[f'rho{rho}/{e}'] = {'sha256': h, 'equals_seed_record': h == rec_sha}
                if h != rec_sha:
                    rprob.append(f'NEURAL_RAW_{v} rho{rho} {e}')
        rin[f'S{v}'] = {'STAGE_A_SEED.json': _sha(rep / f'STAGE_A_{v}' / 'STAGE_A_SEED.json'),
                        'U0_MODEL.pt': u0, 'U0_MODEL_equals_seed_record': u0 == u0_rec,
                        'NEURAL_RAW_SEED.json': _sha(rep / f'NEURAL_RAW_{v}' / 'NEURAL_RAW_SEED.json'),
                        'NEURAL_RAW_checkpoints': ck,
                        'STAGE_B_SEED.json': _sha(rep / f'STAGE_B_{v}' / 'STAGE_B_SEED.json')}
        if u0 != u0_rec:
            rprob.append(f'U0_MODEL S{v}')
    rec['acs_replication_inputs'] = {'root': '<worktrees>/pcrl-acs-protected-tree-replication-v1/.private/acs_replication/runs',
                                     'files': rin, 'problems': rprob,
                                     'note': 'read-only; the archived-and-kept replication worktree (present locally); FB starts from '
                                             'U0_MODEL.pt and checks its RAW-J E2 against NEURAL_RAW__rho1.0__E2.pt'}
    # SUCC's frozen component files that family FA's deployment function reads (the admitted income-headroom mirror; each pinned
    # here and checked against the occupation-headroom admission record of that mirror)
    fa_in, fa_prob = {}, []
    for v in C.VERSIONS:
        for rel in (f'runs/H/H3_STRONG_LJDIR_PB/S{v}/STATE.json', f'runs/H/STEP1/S{v}/STEP1.json', f'runs/H/STEP1/S{v}/HGB.pkl',
                    f'admitted/stage_a/S{v}/TREE_TEACHER_income.json', f'admitted/stage_a/S{v}/TREE_TEACHER_occupation.json',
                    f'admitted/stage_b/S{v}/LOCAL_TREE__E20.npz', f'admitted/stage_b/S{v}/JOINT_TREE__E20.npz'):
            h = _sha(OC.MIRROR / rel)
            rec_h = (adm.get(rel) or {}).get('sha256')
            fa_in[f'ihs_mirror/{rel}'] = {'sha256': h, 'equals_admission_record': h == rec_h}
            if h != rec_h:
                fa_prob.append(rel)
    rec['fa_frozen_inputs'] = {'root': '.private/pcrl_income_headroom_successor_v1 (the working cache copy of the admitted mirror)',
                               'files': fa_in, 'problems': fa_prob}
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
                        'note': 'checked by sar.admit at run time (utc above)'}
    rec['offdevice_drive_attached'] = Path('/Volumes/YOTUO').is_dir()
    rec['preserved_verdicts'] = {
        'new_release_mechanisms': 'finite negative result: NO_DEVELOPMENT_CANDIDATE; family A closed at its internal capacity stage '
                                  '(FINITE_CELL_CAPACITY_NOT_ESTABLISHED; only its task-only twins evaluated; no protected fit); '
                                  'family B / FB2 closed by F1 (not reopened here)',
        'controlled_training_size': 'CORRECTION_RECIPE_CLOSED (leaf-score correction; not reopened here)',
        'finite_solver_adequacy': 'OPTIMIZER_REPAIR_CLOSED (D_NO_MATERIAL_DEFECT, E_SUPERSEDED_BY_S, S_PROXY_ONLY); the '
                                  'conditional pilot did not run; SV1 lowers the surrogate objective only (not reopened here)',
        'finite_routing_comparison': 'ENGINEERING_REPAIR under the post hoc ER1 reading and INCOMPLETE under the frozen rule (both '
                                     'readings preserved); D1 REPAIRED_TASK_ONLY_RELEASE; D2 NO_REDUCTION_BEYOND_ANY_CONTROL; D3 '
                                     'NOT_SUPERIOR; D4 NOT_MET; PRC_S2 pair AUC 0.6485 vs HQ_S2 0.6489 vs TWIN 0.6555',
        'dual_design_release': 'no method: D2 met in 0 of 2 registered and 0 of 2 adaptive readings (STD and PTC, one idea family); '
                               'DB_PTC_R1 (adaptive) +0.0239 vs the twin, not separated from its release-only control DB_ZTILT_R1 '
                               '(+0.0128, se 0.0081); D3 NOT_SUPERIOR; D4 NOT_MET; D5 NOT_ASSESSED_MISSING_CONTROLS; D6 NOT_CLAIMED '
                               '(not reopened here)'}
    res = rec['nrm_published_results']
    rec['nrm_results_differ'] = res['differ']
    rec['nrm_results_differ_explained'] = {f: POST_PUBLICATION[f] for f in res['differ'] if f in POST_PUBLICATION}
    unexplained = [f for f in res['differ'] if f not in POST_PUBLICATION]
    rec['nrm_results_differ_unexplained'] = unexplained
    rec['ok'] = (rec['pinned_publication_on_remote']['equal_to_pin'] and rec['pinned_publication_on_remote']['origin_is_github']
                 and rec['branch_base_is_nrm_tip'] and not res['missing_locally'] and not unexplained
                 and not rec['nrm_published_code']['differ'] and not rec['nrm_published_code']['missing_locally']
                 and rec['nrm_source_admission_published_identical'] and not problems and not bad_adm
                 and all(rec['reference_wires_equal_source_admission'].values())
                 and rec['frc_publication_on_remote']['equal_to_pin'] and rec['branch_base_is_frc_tip']
                 and not rec['frc_published_code']['differ'] and not rec['frc_published_code']['missing_locally']
                 and not rec['frc_published_results']['missing_locally']
                 and not [f for f in rec['frc_published_results']['differ'] if f not in FRC_POST_PUBLICATION]
                 and not frc_problems
                 and rec['fsa_publication_on_remote']['equal_to_pin'] and rec['branch_base_is_fsa_tip']
                 and not rec['fsa_published_code']['differ'] and not rec['fsa_published_code']['missing_locally']
                 and not rec['fsa_published_results']['missing_locally']
                 and not [f for f in rec['fsa_published_results']['differ'] if f not in FSA_POST_PUBLICATION]
                 and rec['ddr_publication_on_remote']['equal_to_pin'] and rec['ddr_publication_parent_is_receipt']
                 and rec['branch_base_is_ddr_tip']
                 and not rec['ddr_published_code']['differ'] and not rec['ddr_published_code']['missing_locally']
                 and not rec['ddr_published_results']['missing_locally']
                 and not [f for f in rec['ddr_published_results']['differ'] if f not in DDR_POST_PUBLICATION]
                 and not ddr_problems
                 and rec['ipr_publication_on_remote']['equal_to_pin'] and rec['ipr_publication_parent_is_receipt']
                 and rec['branch_base_is_ipr_tip']
                 and not rec['ipr_published_code']['differ'] and not rec['ipr_published_code']['missing_locally']
                 and not rec['ipr_published_results']['missing_locally']
                 and not [f for f in rec['ipr_published_results']['differ'] if f not in IPR_POST_PUBLICATION]
                 and not ipr_problems and rec['working_cache']['ok'] and not rprob and not fa_prob)
    C.write_json(C.RESULTS / 'SOURCE_ADMISSION.json', rec)
    short = {k: (v if k not in ('nrm_published_results', 'nrm_published_code', 'frc_published_results', 'frc_published_code',
                            'fsa_published_results', 'fsa_published_code', 'ddr_published_results', 'ddr_published_code',
                            'ipr_published_results', 'ipr_published_code')
                 else {kk: (len(vv) if kk in ('same', 'sanitised') else vv) for kk, vv in v.items()})
             for k, v in rec.items() if k not in ('reference_releases_MODEL_SELECT_wire_sha256', 'interpreters',
                                                  'source_private_artefacts', 'frc_private_artefacts', 'ddr_private_artefacts',
                                                  'ipr_private_artefacts',
                                                  'nrm_local_results_not_in_publication')}
    print(json.dumps(short, indent=1))
    return rec


def policy():
    """ACCESS_POLICY.json: the machine-readable copy of sar.common's ROLE x FIELD x OPERATION x PURPOSE policy (label-free)."""
    rec = {'schema': 'SAR1-access-policy-v1', 'utc': C.now(), 'admitted_fields': {r: list(v) for r, v in C.ADMITTED.items()},
           'closed_label_roles': list(C.CLOSED_LABEL_ROLES), 'refused_everywhere': ['SEALED_ASSESSMENT', 'capped-out records'],
           'operations': {k: {**{kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()},
                              'entry_points': list(C.ENTRY_POINTS[k])} for k, v in C.POLICY.items()},
           'gates': ['no label read before the committed registration and its verified public receipt (sar.common.registered)',
                     "'fit' and 'calibrate' reads need a committed, unchanged mechanism card under results/<campaign>/cards/ "
                     '(path, SHA-256 and commit logged with the read)',
                     'MODEL_SELECT labels are never read by a fitting entry point (sar.fa, sar.fb, sar.ctl, sar.fs)',
                     'label reads run only inside a queue unit (SAR_UNIT_ID)',
                     'entry points can be added only by a dated, committed ACCESS_AMENDMENTS.json entry before first use'],
           'source_artefacts': 'the input-and-pair, dual-design release, finite-routing comparison and new-release-mechanisms private artefacts (the working cache) and the acs_replication tree are read-only inputs '
                               '(SOURCE_ADMISSION.json); nothing is written under the source campaigns\' paths',
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
