"""V1 source admission replay (label-free): the pinned controlled training-size publication on the GitHub remote, the branch base,
the published results / code against the local copies (exact blob, else after the source sanitiser), the private source copies
against the read-only source worktree and the admitted-copy hash record, the frozen reference wires, and the interpreter versions,
all recomputed with the verifier's own calls and compared with SOURCE_ADMISSION.json.

    python3 .../v1_admission.py --tag V1
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402
import vref  # noqa: E402

PIN = 'ed7bc1ccd550f452c112ba3144a9d64b014c52fd'
TIP = '0f27e59373ee8bf4601ad5e582757d2df2a2c14f'
PUB_PREFIX = 'publications/pcrl-controlled-training-size-v1'
TSZ_RESULTS = V.ROOT / 'results' / 'pcrl_controlled_training_size_v1'
OHN_WT = V.ROOT.parent / 'pcrl-occupation-headroom-overnight-v1'
COPIED = ('.private/pcrl_occupation_headroom_overnight_v1/runs', '.private/pcrl_occupation_headroom_overnight_v1/admitted_logs',
          '.private/pcrl_income_headroom_successor_v1')
REFS = ('W', 'REF:NEURAL_RAW', 'REF:H_LJ', 'REF:B2TWIN', 'REF:RETIRED_B3CTL', 'REF:SUCC', 'REF:TWJ', 'REF:U0', 'REF:Ucal',
        'REF:TASK_REFIT', 'REF:T_TREE')


def blob_of_text(txt):
    r = subprocess.run(['git', '-C', str(V.ROOT), 'hash-object', '--stdin'], input=txt, capture_output=True, text=True)
    return r.stdout.strip()


def published(prefix):
    rc, out, err = V.git('ls-tree', '-r', PIN, '--', prefix + '/')
    res = {}
    for line in out.splitlines():
        meta, path = line.split('\t', 1)
        res[path[len(prefix) + 1:]] = meta.split()[2]
    return res


def compare(pub, local_of):
    sys.path.insert(0, str(V.ROOT))
    from mechanism_design.publish_build import sanitise          # the source publication text transform (definition, not a metric)
    out = {'identical': 0, 'identical_after_source_sanitiser': 0, 'differ': [], 'missing_locally': []}
    for f, blob in sorted(pub.items()):
        p = local_of(f)
        if not p.is_file():
            out['missing_locally'].append(f); continue
        rc, local, _ = V.git('hash-object', str(p))
        if local == blob:
            out['identical'] += 1; continue
        try:
            txt = p.read_text()
        except UnicodeDecodeError:
            out['differ'].append(f); continue
        if blob_of_text(sanitise(txt)) == blob:
            out['identical_after_source_sanitiser'] += 1
        else:
            out['differ'].append(f)
    out['files'] = len(pub)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
    adm_path = V.RESULTS / 'SOURCE_ADMISSION.json'
    adm = json.loads(adm_path.read_text())
    res, probs = {}, []
    rc, ls, err = V.git('ls-remote', 'https://github.com/Bostesa/PCRL.git', 'refs/heads/publication/pcrl-controlled-training-size-v1')
    res['remote_pin'] = ls.split()[0] if rc == 0 and ls else f'unavailable: {err[-200:]}'
    if res['remote_pin'] != PIN:
        probs.append('pinned publication is not the remote branch tip')
    res['tip_ancestor_of_HEAD'] = V.is_ancestor(TIP)
    if not res['tip_ancestor_of_HEAD']:
        probs.append('controlled training-size research tip is not an ancestor of HEAD')
    r_res = compare(published(PUB_PREFIX + '/results'), lambda f: TSZ_RESULTS / f)
    r_code = compare(published(PUB_PREFIX + '/code/tsz'), lambda f: V.ROOT / 'tsz' / f)
    res['published_results'] = r_res; res['published_code'] = r_code
    for k, mine in (('tsz_published_results', r_res), ('tsz_published_code', r_code)):
        rec = adm[k]
        if (mine['files'], mine['identical'], mine['identical_after_source_sanitiser'], sorted(mine['differ']), sorted(mine['missing_locally'])) != \
                (rec['files'], rec['identical'], rec['identical_after_source_sanitiser'], sorted(rec['differ']), sorted(rec['missing_locally'])):
            probs.append(f'{k} differs from SOURCE_ADMISSION.json')
    n = same = 0
    bad = []
    for d in COPIED:
        for p in sorted((V.ROOT / d).rglob('*')):
            if p.is_dir() or '__pycache__' in p.parts:
                continue
            n += 1
            rp = p.relative_to(V.ROOT)
            src = OHN_WT / rp
            if src.is_file() and V.file_sha(src) == V.file_sha(p):
                same += 1
            else:
                bad.append(str(rp))
    res['private_copies'] = {'files': n, 'bitwise_identical_to_source_worktree': same, 'problems': bad[:20]}
    pc = adm['private_copies']
    if (n, same) != (pc['files'], pc['bitwise_identical_to_source_worktree']) or bad:
        probs.append('private copies differ from the record or from the source worktree')
    ohn_adm = json.loads(vref.OHN_ADMISSION.read_text())['copies']
    mirror_bad = [rel for rel, r in ohn_adm.items() if not (vref.IHS_MIRROR / rel).is_file() or V.file_sha(vref.IHS_MIRROR / rel) != r['sha256']]
    res['admitted_mirror_copies'] = {'n': len(ohn_adm), 'hash_mismatch_or_missing': mirror_bad[:20]}
    if mirror_bad:
        probs.append('admitted mirror copies differ from their admission hashes')
    wires = {}
    for name in REFS:
        for v in (0, 1):
            _, prov = vref.reference(name, v)
            wires[f'{name}/S{v}'] = prov['wire_equals_admission']
    res['reference_wires_equal_admission'] = wires
    if not all(wires.values()):
        probs.append('a reference wire differs from SOURCE_ADMISSION.json')
    import numpy, scipy, sklearn  # noqa: E401
    res['python3_versions'] = {'numpy': numpy.__version__, 'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__}
    ip = adm['interpreters']['python3']
    if (ip['numpy'], ip['scipy'], ip['scikit-learn']) != (numpy.__version__, scipy.__version__, sklearn.__version__):
        probs.append('python3 library versions differ from the admission record')
    res['admission_ok_flag'] = adm.get('ok')
    chk = {'id': f'V1.SOURCE_ADMISSION__{a.tag}', 'what': 'source admission replayed label-free with own calls: remote pin, branch base, '
           'published results / code vs local copies (exact or after the source sanitiser), private copies vs the source worktree, '
           'admitted mirror hashes, 22 reference wires, interpreter versions',
           'inputs': V.inputs_sha([adm_path, vref.OHN_ADMISSION, vref.BANK_MANIFEST]), 'result': {**res, 'problems': probs},
           'numerical_status': 'PASS' if all(wires.values()) else 'FAIL',
           'procedural_status': 'PASS' if not probs else 'FAIL'}
    print(json.dumps(V.append_checks([chk]), indent=1))
    print(json.dumps({**{k: x for k, x in res.items() if k != 'reference_wires_equal_admission'}, 'problems': probs}, indent=1, default=V._jd)[:4000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
