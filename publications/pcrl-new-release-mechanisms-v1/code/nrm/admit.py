"""Label-free admission of the source artefacts of the PCRL new-release-mechanisms campaign (no label is read).

    python3 -m nrm.admit run        writes results/pcrl_new_release_mechanisms_v1/SOURCE_ADMISSION.json

Pins:
- the controlled training-size publication ed7bc1ccd (publications/pcrl-controlled-training-size-v1), verified on the GitHub remote
  with `git ls-remote`: every published results file and every published code file (code/tsz/*.py) is compared byte for byte (git
  blob ids, after the source's own sanitiser) with the local research copy at this branch's base (research tip 0f27e5937); files
  written after that publication (its publication record) are listed, not hidden;
- the source chain the controlled training-size campaign admitted (correction transfer e32a185b4, occupation headroom 1435c395d,
  H3 89fd2f6e6): its published SOURCE_ADMISSION.json must be byte-identical to the local copy, and every private file of the two
  source copies (.private/pcrl_occupation_headroom_overnight_v1/{runs,admitted_logs}, .private/pcrl_income_headroom_successor_v1)
  is compared bitwise (SHA-256) with the original in the occupation-headroom worktree (read-only), as the source did;
- the reference releases this campaign compares against (frozen bank, lineage and combined releases, the previous witness W), by
  SHA-256 of their MODEL_SELECT arrays (label-free);
- the interpreters and their numerical library versions; the absence of AGENTS.md at the worktree root and the main checkout.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys

from . import common as C

TSZ_PUB_PREFIX = 'publications/pcrl-controlled-training-size-v1'
TSZ_RESULTS = C.ROOT / 'results' / C.TSZ_CAMPAIGN
POST_PUBLICATION = ('PUBLICATION_RECORD.json',)
REFERENCES = ('U0', 'Ucal', 'T_TREE', 'TASK_REFIT', 'NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ')


def _git(*a):
    r = subprocess.run(['git', '-C', str(C.ROOT), *a], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def run():
    from tsz import admit as TA
    rec = {'schema': 'NRM1-source-admission-v1', 'utc': C.now(), 'labels_read': False,
           'pins': {'controlled_training_size_publication': C.TSZ_PUBLICATION, 'controlled_training_size_research_tip': C.TSZ_TIP,
                    'chain': 'correction transfer e32a185b4, occupation headroom 1435c395d, H3 89fd2f6e6 (as admitted by the '
                             'controlled training-size campaign)'}}
    rc, out, err = _git('ls-remote', 'origin', 'refs/heads/publication/pcrl-controlled-training-size-v1')
    rec['pinned_publication_on_remote'] = {'ls_remote': out.strip(), 'equal_to_pin': out.split()[0] == C.TSZ_PUBLICATION if out else False}
    rc, _, _ = _git('merge-base', '--is-ancestor', C.TSZ_TIP, 'HEAD')
    rec['branch_base_is_tsz_tip'] = rc == 0
    pub_res = TA.published_files(C.TSZ_PUBLICATION, TSZ_PUB_PREFIX + '/results')
    rec['tsz_published_results'] = TA._compare(pub_res, lambda f: TSZ_RESULTS / f)
    pub_code = TA.published_files(C.TSZ_PUBLICATION, TSZ_PUB_PREFIX + '/code/tsz')
    rec['tsz_published_code'] = TA._compare(pub_code, lambda f: C.ROOT / 'tsz' / f)
    local_only = sorted(str(p.relative_to(TSZ_RESULTS)) for p in TSZ_RESULTS.rglob('*') if p.is_file()
                        and str(p.relative_to(TSZ_RESULTS)) not in pub_res and p.suffix not in ('.lock', '.jsonl'))
    rec['tsz_local_results_not_in_publication'] = local_only
    rec['tsz_source_admission_published_identical'] = 'SOURCE_ADMISSION.json' in pub_res and \
        _git('hash-object', str(TSZ_RESULTS / 'SOURCE_ADMISSION.json'))[1].strip() == pub_res['SOURCE_ADMISSION.json']
    copies, bad = 0, []
    for rel in TA.COPIED:
        for p in sorted((C.ROOT / rel).rglob('*')):
            if p.is_dir() or '__pycache__' in p.parts:
                continue
            r = p.relative_to(C.ROOT)
            copies += 1
            src = TA.OHN_WT / r
            if not src.is_file() or TA._sha(src) != TA._sha(p):
                bad.append(str(r))
    rec['private_copies'] = {'files': copies, 'bitwise_identical_to_source_worktree': copies - len(bad), 'problems': bad[:50],
                             'n_problems': len(bad), 'source_worktree': 'pcrl-occupation-headroom-overnight-v1 (read-only)'}
    # reference releases (label-free hashes of their MODEL_SELECT arrays)
    from ohn.devrun import reference_release
    from tsz.evaluate import source_a1, h3_income
    refs = {}
    for v in C.VERSIONS:
        for name in REFERENCES:
            rel = reference_release(name, v, 'MODEL_SELECT')
            refs[f'{name}/S{v}'] = {t: C.wire_sha(rel[t]) for t in C.TASKS}
        refs[f'W/S{v}'] = {'income': C.wire_sha(h3_income(v)), 'occupation': C.wire_sha(source_a1(v)),
                           'definition': 'A1_PO_16 S<v> occupation (bitwise = the W100_v refit) + H3_STRONG_LJDIR_PB income S<v>'}
    rec['reference_releases_MODEL_SELECT_wire_sha256'] = refs
    rec['interpreters'] = {'python3': TA._versions('python3'), 'catboost_venv': TA._versions(C.CATBOOST_PY),
                           'torch_venv': TA._versions(C.TORCH_PY)}
    rec['agents_md'] = {'worktree_root': (C.ROOT / 'AGENTS.md').is_file(),
                        'main_checkout': (C.ROOT.parents[2] / 'AGENTS.md').is_file(),
                        'note': 'inspected first at 14:29Z: no AGENTS.md exists at either location'}
    rec['preserved_verdicts'] = {
        'controlled_training_size': 'SUPPORTED_ON_THIS_POOL size explanation (not specific to the recipe; CONTROL_EXPLAINS_GAIN) and '
                                    'CORRECTION_RECIPE_CLOSED (leaf-score correction s = 0.3: ~0.005 pair AUC, not stable)',
        'correction_transfer': 'replicated diagnosis and negative method result (NO_RECIPE; claim ladder rung 0)',
        'occupation_headroom': 'NO_QUALIFYING_CONFIGURATION',
        'h3_income': 'H3_STRONG_LJDIR_PB kept income Brier / NLL headroom at 100% but was not competitive'}
    res = rec['tsz_published_results']
    unexplained = [f for f in res['differ'] if f not in POST_PUBLICATION]
    rec['tsz_results_differ_unexplained'] = unexplained
    rec['ok'] = (rec['pinned_publication_on_remote']['equal_to_pin'] and rec['branch_base_is_tsz_tip'] and not res['missing_locally']
                 and not unexplained and not rec['tsz_published_code']['differ'] and not rec['tsz_published_code']['missing_locally']
                 and rec['tsz_source_admission_published_identical'] and not bad)
    C.write_json(C.RESULTS / 'SOURCE_ADMISSION.json', rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ('reference_releases_MODEL_SELECT_wire_sha256', 'interpreters')}, indent=1))
    return rec


def policy():
    """ACCESS_POLICY.json: the machine-readable copy of nrm.common's ROLE x FIELD x OPERATION x PURPOSE policy (label-free)."""
    rec = {'schema': 'NRM1-access-policy-v1', 'utc': C.now(), 'admitted_fields': {r: list(v) for r, v in C.ADMITTED.items()},
           'closed_label_roles': list(C.CLOSED_LABEL_ROLES), 'refused_everywhere': ['SEALED_ASSESSMENT', 'capped-out records'],
           'operations': {k: {**{kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()},
                              'entry_points': list(C.ENTRY_POINTS[k])} for k, v in C.POLICY.items()},
           'gates': ['no label read before the committed registration and its verified public receipt (nrm.common.registered)',
                     "'fit' and 'calibrate' reads need a committed, unchanged mechanism card under results/<campaign>/cards/ "
                     '(path, SHA-256 and commit logged with the read)',
                     'MODEL_SELECT labels are never read by a fitting entry point (nrm.fa, nrm.fb, nrm.fc, nrm.ctl)',
                     'entry points can be added only by a dated, committed ACCESS_AMENDMENTS.json entry before first use'],
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
