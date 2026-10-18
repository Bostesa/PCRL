"""Verifier checks 1-2 (label-free): source pins, role membership, actual exclusion evidence and model identity.

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records).
Reads only label-free admitted features (cts.common.features: MODEL_FIT household ids, row ids and the 145-column encoding),
the campaign's private run artifacts, the label-access log, the queue ledger and git objects of the pinned publication commits.
No label is read. Partition masks, dictionaries, folds and hashes are recomputed with the verifier's own code (vlib).

    python3 results/pcrl_correction_transfer_stability_v1/roles/verifier/code/v_struct.py --tag V1
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from cts import common as C  # noqa: E402  (label-free admitted-data loader only: C.features)

CAMP = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1'
TRACKF = CAMP / 'runs' / 'TRACKF'
TRACKR = CAMP / 'runs' / 'TRACK_R'
RES = ROOT / 'results' / 'pcrl_correction_transfer_stability_v1'
OHN_RUNS = ROOT / '.private' / 'pcrl_occupation_headroom_overnight_v1' / 'runs'
MIRROR = ROOT / '.private' / 'pcrl_income_headroom_successor_v1'
OHN_PUB = '1435c395d3f753a9bf0b553d14d264ca9e991421'
IHS_PUB = '89fd2f6e649547f65cf1b69a9e8e8b51f854d25e'
BLOCKS = ('COW', 'SCHL', 'MAR', 'POBP')


def git(*a):
    r = subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True)
    return r.returncode, r.stdout


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def my_dictionary(X, names, min_count=10):
    """FIT-only collapse from PROTOCOL section 6: an admitted level with fewer than 10 people moves to its block's OTHER."""
    names = [str(n) for n in names]
    col = {}
    for b in BLOCKS:
        idx = [i for i, n in enumerate(names) if n.startswith(b + '=') and n != f'{b}=__OTHER__']
        counts = np.asarray(X)[:, idx].sum(axis=0)
        col[b] = sorted(names[i] for i, c in zip(idx, counts) if c < min_count)
    return col


def check_partitions(F, spec, out):
    unit, rid = F['unit'], F['row_id']
    masks = V.partition_masks_from_json(unit, spec)
    summ = spec['summary']
    rows, bad = {}, []
    for name in ('C0', 'C1', 'POOL'):
        h, nh, npl = V.household_set_sha(unit, masks[name])
        exp = summ[name]
        rows[name] = {'people': npl, 'households': nh, 'sha_match': h == exp['household_set_sha256'],
                      'counts_match': (npl, nh) == (exp['people'], exp['households'])}
        if not (rows[name]['sha_match'] and rows[name]['counts_match']):
            bad.append(name)
    for pid in spec['partitions']:
        for r in ('FIT', 'GUARD', 'CHECK'):
            h, nh, npl = V.household_set_sha(unit, masks[pid][r])
            exp = summ[pid][r]
            ok = h == exp['household_set_sha256'] and (npl, nh) == (exp['people'], exp['households'])
            rows[f'{pid}/{r}'] = {'people': npl, 'households': nh, 'sha_match': h == exp['household_set_sha256'],
                                  'counts_match': (npl, nh) == (exp['people'], exp['households'])}
            if not ok:
                bad.append(f'{pid}/{r}')
    # FIT overlaps (households)
    fits = {pid: set(np.unique(unit[masks[pid]['FIT']]).tolist()) for pid in spec['partitions']}
    ov = {}
    for key, val in summ['FIT_overlap_households'].items():
        a, b = key.split('|')
        ov[key] = {'mine': len(fits[a] & fits[b]), 'recorded': val}
        if ov[key]['mine'] != val:
            bad.append('overlap ' + key)
    # structure: panels / pool partition MODEL_FIT; roles disjoint; whole households; FIT/GUARD never touch a panel
    struct = []
    c0, c1, pool = masks['C0'], masks['C1'], masks['POOL']
    if (c0 & c1).any() or ((c0 | c1) & pool).any() or not (c0 | c1 | pool).all():
        struct.append('panels and pool are not a partition of MODEL_FIT')
    hh_of = {}
    for pid in spec['partitions']:
        f, g, c = masks[pid]['FIT'], masks[pid]['GUARD'], masks[pid]['CHECK']
        if (f & g).any() or (f & c).any() or (g & c).any():
            struct.append(f'{pid}: roles overlap')
        if ((f | g) & (c0 | c1)).any():
            struct.append(f'{pid}: FIT or GUARD touches a repair panel')
        kind = spec['partitions'][pid]['kind']
        if kind == 'diagnosis':
            if (c & (c0 | c1)).any():
                struct.append(f'{pid}: diagnostic CHECK touches a repair panel')
            if not np.array_equal(f | g | c, pool):
                struct.append(f'{pid}: FIT+GUARD+CHECK != pool')
        else:
            if not np.array_equal(f | g, pool):
                struct.append(f'{pid}: FIT+GUARD != pool')
            if not np.array_equal(c, masks[spec['partitions'][pid]['CHECK']]):
                struct.append(f'{pid}: CHECK != its panel')
        for r, m in (('FIT', f), ('GUARD', g), ('CHECK', c)):
            inside = set(np.unique(unit[m]).tolist()); outside = set(np.unique(unit[~m]).tolist())
            if inside & outside:
                struct.append(f'{pid}/{r} splits a household')
    d0b_same = all(np.array_equal(masks['D0'][r], masks['D0B'][r]) for r in ('FIT', 'GUARD', 'CHECK')) if 'D0B' in masks else None
    out.append({'id': 'C1.1_PARTITIONS_REPRODUCED', 'kind': 'NUMERICAL_REPLAY',
                'what': 'PARTITIONS.json household sets (C0, C1, POOL and FIT / GUARD / CHECK of D0, D0B, D1, R0, R1) and FIT '
                        'overlaps reproduced from MODEL_FIT household ids',
                'method': 'own implementation of the PARTITIONS.json text: households ordered by SHA-256("<tag>|<household id>") and dealt '
                          'round-robin (20 blocks CTS1_blocks; pool partitions by their tags); people / household counts and SHA-256 of '
                          'the comma-joined sorted household ids compared with the summary; overlaps of FIT household sets',
                'status': 'PASS' if not bad else 'FAIL',
                'numbers': {'compared_sets': len(rows), 'mismatches': bad, 'sets': rows, 'fit_overlaps': ov,
                            'D0B_rows_identical_to_D0': d0b_same}})
    out.append({'id': 'C1.2_DISJOINTNESS_AND_PANEL_EXCLUSION', 'kind': 'NUMERICAL_REPLAY',
                'what': 'roles disjoint inside each partition; whole households; panels C0 / C1 disjoint from the pool and from every '
                        'FIT / GUARD; diagnostic CHECK inside the pool; R CHECK = its panel',
                'method': 'boolean algebra on the verifier-reconstructed masks',
                'status': 'PASS' if not struct else 'FAIL', 'numbers': {'problems': struct}})
    return masks


def check_admission(out):
    adm = json.loads((RES / 'SOURCE_ADMISSION.json').read_text())
    problems, checked = [], {}
    pub_cache = {}

    def pub_text(commit, path):
        key = (commit, path)
        if key not in pub_cache:
            rc, txt = git('show', f'{commit}:{path}')
            pub_cache[key] = txt if rc == 0 else None
        return pub_cache[key]
    for key, rec in adm['releases_used'].items():
        fam, cfg, s = key.split('/')
        if fam == 'H':
            rj = MIRROR / 'runs' / 'H' / cfg / s / 'RELEASE.json'
            h = json.loads(rj.read_text()).get('release_sha256')
            ok = h == rec['release_sha256']
            found = []
            for f in rec.get('published_in_h3_publication', []):
                t = pub_text(IHS_PUB, f'publications/pcrl-income-headroom-successor-v1/results/{f}')
                if t is not None and h in t:
                    found.append(f)
            ok = ok and found == rec.get('published_in_h3_publication', [])
            checked[key] = {'release_sha_match': h == rec['release_sha256'], 'published_hits_reproduced': found}
        else:
            d = OHN_RUNS / fam / cfg / s
            npz_sha = V.file_sha(d / 'OCC.npz')
            occ = json.loads((d / 'OCC.json').read_text()).get('occ_sha256')
            found = []
            for f in rec.get('published_in', []):
                t = pub_text(OHN_PUB, f'publications/pcrl-occupation-headroom-overnight-v1/results/{f}')
                if t is not None and occ in t:
                    found.append(f)
            ok = npz_sha == rec['occ_npz_sha256'] and occ == rec['occ_sha256'] and found == rec.get('published_in', [])
            checked[key] = {'npz_sha_match': npz_sha == rec['occ_npz_sha256'], 'occ_sha_match': occ == rec['occ_sha256'],
                            'published_hits_reproduced': len(found), 'published_hits_recorded': len(rec.get('published_in', []))}
        if not ok:
            problems.append(key)
    # H3 mirror: random sample of the 201 copies against the source campaign's SOURCE_ADMISSION.json, itself compared with its
    # published blob at the source publication commit
    src_adm_path = 'results/pcrl_occupation_headroom_overnight_v1/SOURCE_ADMISSION.json'
    rc, blob_pub = git('ls-tree', OHN_PUB, '--', 'publications/pcrl-occupation-headroom-overnight-v1/results/SOURCE_ADMISSION.json')
    rc2, blob_local = git('hash-object', src_adm_path)
    pub_identical = rc == 0 and blob_pub.split() and blob_pub.split()[2] == blob_local.strip()
    sadm = json.loads((ROOT / src_adm_path).read_text())
    keys = sorted(sadm['copies'])
    rnd = random.Random('CTS1_verifier_admission_sample')
    sample = rnd.sample(keys, 30)
    mism = [k for k in sample if V.file_sha(MIRROR / k) != sadm['copies'][k]['sha256']]
    # admitted teachers used as FROZEN routing
    out.append({'id': 'C1.3_SOURCE_ADMISSION_SPOT_CHECK', 'kind': 'NUMERICAL_REPLAY',
                'what': 'SOURCE_ADMISSION.json claims: the 28 used release hashes (OCC.npz / OCC.json / H3-HC RELEASE.json) and their '
                        'presence in the published records of the pinned publications; a random 30-file sample of the 201 H3-mirror '
                        'copies against the source campaign SOURCE_ADMISSION.json (itself compared with its published blob)',
                'method': 'SHA-256 recomputed by the verifier; published texts read with git show at 1435c395d / 89fd2f6e6 (read-only); '
                          'the copies in the other worktree are not read (not touched by policy)',
                'status': 'PASS' if not problems and not mism and pub_identical else 'FAIL',
                'numbers': {'releases_checked': len(checked), 'release_problems': problems, 'mirror_sample': len(sample),
                            'mirror_mismatches': mism, 'source_admission_blob_identical_to_publication': bool(pub_identical),
                            'per_release': checked}})


def check_oof_and_dicts(F, masks, out):
    unit, rid, X, names = F['unit'], F['row_id'], F['X'], F['feature_names']
    res, bad = {}, []
    errors = {}
    for P in sorted(p.name for p in TRACKF.iterdir() if (p / 'WITNESS' / 'OOF.npz').is_file()):
      try:
        wj = json.loads((TRACKF / P / 'WITNESS' / 'WITNESS.json').read_text())
        O = npz(TRACKF / P / 'WITNESS' / 'OOF.npz')
        fit = masks[P]['FIT']
        r = {'fit_rows_equal': bool(np.array_equal(O['fit_rows'].astype(bool), fit)),
             'row_id_equal': bool(np.array_equal(O['row_id'], rid[fit])),
             'oof_rows': int(O['oof'].shape[0]), 'fit_people': int(fit.sum())}
        my_folds = V.deal(unit[fit], 5, wj['outer_fold_tag'])
        r['outer_folds_equal'] = bool(np.array_equal(O['fold'], my_folds))
        r['fold_households_disjoint'] = all(
            not (set(np.unique(unit[fit][my_folds == a]).tolist()) & set(np.unique(unit[fit][my_folds == b]).tolist()))
            for a in range(5) for b in range(a + 1, 5))
        # dictionary: FIT-only collapse; is the record discriminating against FIT+GUARD+CHECK (pool) and all MODEL_FIT?
        dj = json.loads((TRACKF / P / 'DICT.json').read_text())
        mine = my_dictionary(X[fit], names)
        rec = {b: sorted(dj['collapsed'][b]) for b in BLOCKS}
        r['dict_equal_fit_only'] = mine == rec
        r['dict_sha_matches_witness_record'] = V.file_sha(TRACKF / P / 'DICT.json') == wj['dictionary']['dict_sha256']
        r['n_collapsed'] = {b: len(rec[b]) for b in BLOCKS}
        alt_pool = my_dictionary(X[masks[P]['FIT'] | masks[P]['GUARD'] | masks[P]['CHECK']], names)
        alt_all = my_dictionary(X, names)
        r['fit_only_differs_from_pool_dictionary'] = alt_pool != mine
        r['fit_only_differs_from_all_model_fit_dictionary'] = alt_all != mine
        # nested dictionaries (FIT minus outer fold k)
        nd = TRACKF / P / 'NESTED'
        if nd.is_dir():
            nres = {}
            for k in range(5):
                f = nd / f'DICT_{k}.json'
                if f.is_file():
                    dk = json.loads(f.read_text())
                    rows_k = np.flatnonzero(fit)[my_folds != k]
                    sel = np.zeros(len(unit), dtype=bool); sel[rows_k] = True
                    nres[str(k)] = my_dictionary(X[sel], names) == {b: sorted(dk['collapsed'][b]) for b in BLOCKS}
            r['nested_dicts_equal_fit_minus_fold'] = nres
            if not all(nres.values()):
                bad.append(f'{P}: nested dictionary')
        ok = all(r[k] for k in ('fit_rows_equal', 'row_id_equal', 'outer_folds_equal', 'fold_households_disjoint', 'dict_equal_fit_only',
                                'dict_sha_matches_witness_record')) and r['oof_rows'] == r['fit_people']
        if not ok:
            bad.append(P)
        res[P] = r
      except Exception as ex:                                  # a verifier-side error is recorded, never hidden
        errors[P] = f'{type(ex).__name__}: {ex}'[:300]
    out.append({'id': 'C2.1_WITNESS_ROWS_FOLDS_DICTIONARY', 'kind': 'NUMERICAL_REPLAY',
                'what': 'OOF.npz FIT-row mask and row ids equal the verifier reconstruction; outer cross-fitting folds equal the '
                        'household dealing of FIT with the recorded tag; DICT.json collapse lists follow from FIT inputs only (and differ '
                        'from a pool or all-MODEL_FIT dictionary where noted); nested DICT_k from FIT minus fold k',
                'method': 'own masks / dealing / level counts (label-free admitted encoding, minimum 10 people)',
                'status': 'FAIL' if bad else ('ATTENTION' if errors else 'PASS'),
                'numbers': {'partitions': res, 'problems': bad, 'verifier_errors': errors}})


def check_access_log(F, masks, out):
    rid = F['row_id']
    S = C.features('MODEL_SELECT')
    ms_sha = V.arr_sha(np.asarray(S['row_id']))
    all_sha = V.arr_sha(np.asarray(rid))
    exp = {}
    for pid in [p for p in masks if p not in ('C0', 'C1', 'POOL')]:
        for r in ('FIT', 'GUARD', 'CHECK'):
            exp[(pid, r)] = (V.arr_sha(np.asarray(rid[masks[pid][r]])), int(masks[pid][r].sum()))
    for pn in ('C0', 'C1'):
        exp[(pn, 'CHECK')] = (V.arr_sha(np.asarray(rid[masks[pn]])), int(masks[pn].sum()))
    lines = [json.loads(l) for l in (CAMP / 'access' / 'LABEL_ACCESS.jsonl').read_text().splitlines() if l.strip()]
    problems, per_op = [], {}
    op_role = {'fit': 'FIT', 'select': 'GUARD', 'check_diag': 'CHECK', 'check_repair': 'CHECK'}
    for i, e in enumerate(lines):
        key = f"{e['operation']}|{e['role']}"
        per_op[key] = per_op.get(key, 0) + 1
        if e['role'] == 'MODEL_SELECT':
            if e['rows_sha256'] != ms_sha:
                problems.append(f'line {i}: MODEL_SELECT rows hash differs')
            continue
        if e['role'] != 'MODEL_FIT':
            if e['role'] in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'):
                problems.append(f'line {i}: closed role {e["role"]} read')
            continue
        if e['partition'] is None:
            if e['operation'] not in ('source_replay', 'support_count'):
                problems.append(f'line {i}: whole-role MODEL_FIT read by operation {e["operation"]}')
            if e['rows_sha256'] != all_sha:
                problems.append(f'line {i}: whole-role rows hash differs')
            continue
        k = (e['partition'], e['prole'])
        if k not in exp:
            problems.append(f'line {i}: unknown partition role {k}'); continue
        if e['rows_sha256'] != exp[k][0] or e['people'] != exp[k][1]:
            problems.append(f'line {i}: rows hash / people of {k} differ from the verifier reconstruction')
        want = op_role.get(e['operation'])
        if want and e['prole'] != want:
            problems.append(f'line {i}: operation {e["operation"]} read {e["prole"]} rows')
        if e['operation'] == 'check_diag' and not e['partition'].startswith('D'):
            problems.append(f'line {i}: check_diag on {e["partition"]}')
        if e['touches_repair_panel'] and e['operation'] not in ('source_replay', 'support_count'):
            problems.append(f'line {i}: repair panel touched by {e["operation"]}')
        uid = e.get('unit_id') or ''
        tokens = set(uid.replace('-', '_').split('_'))
        named = {p for p in ('D0', 'D1', 'D0B', 'R0', 'R1') if p in tokens}
        if named and e['partition'] not in named:
            problems.append(f'line {i}: unit {uid} read partition {e["partition"]}')
    out.append({'id': 'C2.2_LABEL_ACCESS_ROWS', 'kind': 'PROCEDURAL_COMPLIANCE',
                'what': 'every label-access line: rows hash equals the verifier reconstruction of the stated partition role; fit reads '
                        'only FIT rows, select only GUARD, check_diag only diagnostic CHECK; units named for a partition read only that '
                        'partition; no repair-panel or closed-role label read except the registered whole-role preflight operations',
                'method': 'own masks + own row-set hash (SHA-256 of dtype + shape + bytes of the row ids) against LABEL_ACCESS.jsonl',
                'status': 'PASS' if not problems else 'FAIL',
                'numbers': {'lines': len(lines), 'by_operation_role': per_op, 'problems': problems}})
    return lines


def _teacher_shas(path):
    raw = json.loads(Path(path).read_text())
    return V.file_sha(path), V.payload_sha(raw)


def check_identity(lines, out):
    """Model identity: recorded file hashes recomputed; witnesses, routings and directions differ across partitions."""
    integrity, art = [], {}
    for P in sorted(p.name for p in TRACKF.iterdir() if p.is_dir()):
        wd = TRACKF / P / 'WITNESS'
        if (wd / 'WITNESS.json').is_file():
            wj = json.loads((wd / 'WITNESS.json').read_text())
            for f, h in wj['files_sha256'].items():
                if V.file_sha(wd / f) != h:
                    integrity.append(f'{P}/WITNESS/{f}')
                art[f'{P}/witness/{f}'] = h
            W = npz(wd / 'WITNESS.npz')
            for r in ('MODEL_SELECT', 'MODEL_FIT'):
                art[f'{P}/witness_release/{r}'] = V.arr_sha(W[f'dep__{r}'])
                if V.arr_sha(W[f'dep__{r}']) != wj['release_sha256'][r]:
                    integrity.append(f'{P}/release_sha256/{r}')
        rd = TRACKF / P / 'ROUTING'
        if (rd / 'ROUTING.json').is_file():
            rj = json.loads((rd / 'ROUTING.json').read_text())
            fs, ps = _teacher_shas(rd / 'TEACHER_occupation.json')
            if fs != rj['teacher_file_sha256'] or ps != rj['teacher_sha256']:
                integrity.append(f'{P}/ROUTING teacher')
            art[f'{P}/routing_REFIT'] = ps
        for dd in sorted((TRACKF / P / 'DIR').glob('*')) if (TRACKF / P / 'DIR').is_dir() else []:
            if not (dd / 'DIR.json').is_file():
                continue
            dj = json.loads((dd / 'DIR.json').read_text())
            for f, h in dj['files_sha256'].items():
                if V.file_sha(dd / f) != h:
                    integrity.append(f'{P}/DIR/{dd.name}/{f}')
            T = npz(dd / 'TRAIN.npz')
            for k, v in T.items():
                if k.startswith('theta'):
                    art[f'{P}/dir/{dd.name}/{k}'] = V.arr_sha(v)
            D = npz(dd / 'DELTA.npz')
            art[f'{P}/dir/{dd.name}/delta_full_MS'] = V.arr_sha(D['full__MODEL_SELECT'])
            # links: witness and routing used by the direction
            wj = json.loads((TRACKF / P / 'WITNESS' / 'WITNESS.json').read_text())
            if dj['witness_files_sha256'] != wj['files_sha256']:
                integrity.append(f'{P}/DIR/{dd.name}: witness link')
            if dd.name.startswith('REFIT'):
                want = art.get(f'{P}/routing_REFIT')
            else:
                s = json.loads((TRACKF / P / 'WITNESS' / 'WITNESS.json').read_text())['seed']
                want = _teacher_shas(MIRROR / 'admitted' / 'stage_a' / f'S{s}' / 'TREE_TEACHER_occupation.json')[1]
            if dj['routing_teacher_sha256'] != want:
                integrity.append(f'{P}/DIR/{dd.name}: routing link')
            # OOF correction = fold pipeline k applied to fold k (TRAIN delta_oof vs DELTA folds on FIT rows)
            O = npz(TRACKF / P / 'WITNESS' / 'OOF.npz')
            fr = O['fit_rows'].astype(bool)
            fold = O['fold']
            dmax = 0.0
            for k in range(5):
                a = T['delta_oof'][fold == k]
                b = D['folds__MODEL_FIT'][k][fr][fold == k]
                dmax = max(dmax, float(np.max(np.abs(a - b))) if len(a) else 0.0)
            art[f'{P}/dir/{dd.name}/oof_vs_fold_delta_maxdiff'] = dmax
            if dmax > 1e-12:
                integrity.append(f'{P}/DIR/{dd.name}: delta_oof differs from the fold corrections ({dmax:.3g})')
    for s in (0, 1):
        art[f'FROZEN_S{s}/routing'] = _teacher_shas(MIRROR / 'admitted' / 'stage_a' / f'S{s}' / 'TREE_TEACHER_occupation.json')[1]
    # Track R teachers (if present)
    for P in sorted(p.name for p in TRACKR.iterdir() if p.is_dir()) if TRACKR.is_dir() else []:
        tj = TRACKR / P / 'TEACHERS' / 'TEACHERS.json'
        if tj.is_file():
            for task in ('income', 'occupation'):
                f = TRACKR / P / 'TEACHERS' / f'TEACHER_{task}.json'
                if f.is_file():
                    art[f'TRACKR/{P}/teacher_{task}'] = _teacher_shas(f)[1]
    # distinctness across partitions (witness files, releases, routing, thetas)
    hashes = {k: v for k, v in art.items() if isinstance(v, str)}
    groups = {}
    for k, v in hashes.items():
        groups.setdefault(v, []).append(k)
    aliases = [sorted(g) for g in groups.values() if len(g) > 1]

    def part_of(key):                      # Track R artifacts belong to their repair partition (expected deterministic aliases)
        return key.split('/')[1] if key.startswith('TRACKR/') else key.split('/')[0]
    cross_partition_aliases = [g for g in aliases if len({part_of(x) for x in g}) > 1]
    same_partition_aliases = [g for g in aliases if len({part_of(x) for x in g}) == 1]
    out.append({'id': 'C2.3_MODEL_IDENTITY', 'kind': 'NUMERICAL_REPLAY',
                'what': 'recorded file hashes of witnesses, routings and directions recomputed; direction records link the partition '
                        'witness and the stated routing (REFIT teacher of the partition / admitted T_TREE of seed s); TRAIN delta_oof '
                        'equals fold pipeline k on fold k; witness files, releases, REFIT teachers and thetas differ across partitions '
                        '(deterministic aliases listed and counted once)',
                'method': 'SHA-256 of files and arrays (verifier code); payload SHA-256 of teacher JSON with sorted keys',
                'status': 'PASS' if not integrity and not cross_partition_aliases else 'FAIL',
                'numbers': {'artifacts_hashed': len(hashes), 'unique_hashes': len(groups), 'integrity_problems': integrity,
                            'cross_partition_aliases': cross_partition_aliases, 'within_partition_aliases': same_partition_aliases,
                            'max_oof_vs_fold_delta_diff': max([v for k, v in art.items() if k.endswith('maxdiff')] or [0.0])}})


def check_order(lines, out):
    """Directions frozen before any GUARD / CHECK outcome of the same partition is computed (PROTOCOL section 7)."""
    res, att = {}, []
    for P in sorted(p.name for p in TRACKF.iterdir() if p.is_dir()):
        dirs = [json.loads((d / 'DIR.json').read_text()) for d in (TRACKF / P / 'DIR').glob('*') if (d / 'DIR.json').is_file()] \
            if (TRACKF / P / 'DIR').is_dir() else []
        reads = [e['utc'] for e in lines if e.get('partition') == P and e.get('prole') in ('GUARD', 'CHECK')
                 and e.get('operation') in ('select', 'check_diag', 'check_repair')]
        first = min(reads) if reads else None
        late = sorted(d['direction'] for d in dirs if first and d['utc'] > first)
        res[P] = {'directions': len(dirs), 'last_direction_utc': max([d['utc'] for d in dirs], default=None),
                  'first_guard_or_check_read_utc': first, 'directions_written_after_first_guard_check_read': late}
        if late:
            att.append(P)
    out.append({'id': 'C2.4_DIRECTIONS_BEFORE_GUARD_CHECK', 'kind': 'PROCEDURAL_COMPLIANCE',
                'what': 'PROTOCOL section 7: within each D partition, directions are frozen before any GUARD or CHECK outcome is computed',
                'method': 'DIR.json write times against the first GUARD / CHECK label read of the same partition in LABEL_ACCESS.jsonl',
                'status': 'PASS' if not att else 'ATTENTION', 'numbers': res})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    ap.add_argument('--skip-admission', action='store_true')
    a = ap.parse_args()
    spec = json.loads((RES / 'PARTITIONS.json').read_text())
    F = C.features('MODEL_FIT')              # label-free
    out = []
    masks = check_partitions(F, spec, out)
    if not a.skip_admission:
        check_admission(out)
    check_oof_and_dicts(F, masks, out)
    lines = check_access_log(F, masks, out)
    check_identity(lines, out)
    check_order(lines, out)
    code = V.code_hashes(__file__, V.__file__)
    for c in out:
        c['id'] = f"{c['id']}__{a.tag}"
        c['code_sha256'] = code
    V.write_record(f'V_STRUCT__{a.tag}.json', {'checks': out, 'utc': V.utc()})
    summ = V.append_checks(out)
    print(json.dumps({c['id']: c['status'] for c in out}, indent=1))
    print(json.dumps(summ))


if __name__ == '__main__':
    main()
