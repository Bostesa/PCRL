"""Verifier stage V0 (label-free): subsets, orderings, nesting, nested-selection folds, inner folds, reader splits, reader seeds,
source pins and registered constants, each reconstructed from the TEXT of the definitions with the verifier's own code
(vlib) and compared with the registered manifests (SUBSETS.json, READER_SPLITS.json, SOURCE_ADMISSION.json, RULES.json).

PCRL controlled training-size campaign (TSZ1; exploratory development on previously used ACS 2019 Florida PUMS person records).
No label of any role is read: MODEL_FIT / MODEL_SELECT household ids and row ids only (tsz.common.features, label-free).

    python3 results/pcrl_controlled_training_size_v1/roles/verifier/code/v0_struct.py --tag V1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from tsz import common as C  # noqa: E402   (label-free admitted features only)

RES = V.RESULTS
CTS_RES = ROOT / 'results' / 'pcrl_correction_transfer_stability_v1'
PUB = 'e32a185b4fe921cba474f392659f493d0658b771'
PUB_PREFIX = 'publications/pcrl-correction-transfer-stability-v1'
TIP = '507024649560b620ac3f0b6ae096f353344f8d05'
OHN_PUB = '1435c395d3f753a9bf0b553d14d264ca9e991421'
IHS_PUB = '89fd2f6e649547f65cf1b69a9e8e8b51f854d25e'
MIRROR = ROOT / '.private' / 'pcrl_income_headroom_successor_v1'
OHN_RUNS = ROOT / '.private' / 'pcrl_occupation_headroom_overnight_v1' / 'runs'


def git(*a):
    r = subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def chk(cid, kind, what, inputs, expected, observed, status, **extra):
    c = {'id': cid, 'kind': kind, 'what': what, 'inputs': inputs, 'expected': expected, 'observed': observed, 'status': status}
    c.update(extra)
    return c


def subsets_checks(tag):
    out = []
    man_p = RES / 'SUBSETS.json'
    man = json.loads(man_p.read_text())
    F = C.features('MODEL_FIT')
    unit, row_id = np.asarray(F['unit']), np.asarray(F['row_id'])
    masks = V.subset_masks(unit)
    G = len(np.unique(unit))
    obs, mism = {}, []
    for sid, m in masks.items():
        hh = np.unique(unit[m])
        rec = {'people': int(m.sum()), 'households': int(len(hh)), 'household_set_sha256': V.hh_set_sha(hh),
               'row_set_sha256': V.arr_sha(row_id[m])}
        obs[sid] = rec
        exp = dict(man['subsets'][sid]); exp['row_set_sha256'] = man['row_set_sha256'][sid]
        for k in rec:
            if rec[k] != exp[k]:
                mism.append(f'{sid}.{k}: mine {rec[k]} vs manifest {exp[k]}')
    whole = all(not (set(np.unique(unit[m]).tolist()) & set(np.unique(unit[~m]).tolist())) for m in masks.values())
    defs = {'S50_households': V.n_households(G, 50), 'S70_households': V.n_households(G, 70), 'S100_households': G}
    for k, v in defs.items():
        if man['definition'][k] != v:
            mism.append(f'definition.{k}: mine {v} vs manifest {man["definition"][k]}')
    import hashlib
    unit_sha = hashlib.sha256(np.ascontiguousarray(unit).tobytes()).hexdigest()
    if unit_sha != man['model_fit_unit_sha256']:
        mism.append('model_fit_unit_sha256 differs')
    # production function agreement (cross-check only; the numbers above are the verifier's own)
    prod_same = all(np.array_equal(C.subset_mask(unit, sid), m) for sid, m in masks.items())
    status = 'PASS' if (not mism and whole and prod_same and man.get('labels_read') is False and not man.get('problems')) else 'FAIL'
    out.append(chk(f'V0.SUBSETS.RECONSTRUCTION__{tag}', 'PROCEDURAL_COMPLIANCE',
                   'the 9 registered MODEL_FIT subsets (S50_O<o>, S70_O<o>, S100) reconstructed from the definition: households ranked by '
                   'SHA-256("TSZ1_order_<o>|<household>"), first ceil(0.5 G) / ceil(0.7 G) households, all G for S100; people, households, '
                   'household-set SHA-256 and row-set SHA-256 vs SUBSETS.json; whole households; model_fit_unit_sha256',
                   V.inputs_sha([man_p]), {'manifest': {sid: {**man['subsets'][sid], 'row_set_sha256': man['row_set_sha256'][sid]} for sid in masks},
                                           'definition': {k: man['definition'][k] for k in defs}},
                   {'mine': obs, 'definition': defs, 'G': G, 'whole_households': whole, 'mismatches': mism,
                    'equals_production_subset_mask': prod_same}, status,
                   method='own SHA-256 ranking (Python sorted on hex digests), integer ceil, own hashes; tsz.common.subset_mask compared '
                          'only as a cross-check'))
    # nesting and overlaps
    nm, nobs = [], {}
    for o in V.ORDERINGS:
        a, b = masks[f'S50_{o}'], masks[f'S70_{o}']
        rec = {'S50_in_S70': bool(not (a & ~b).any()), 'S70_in_S100': True, 'added_50_to_70_people': int((b & ~a).sum()),
               'added_70_to_100_people': int((~b).sum())}
        nobs[o] = rec
        if rec != man['nesting'][o]:
            nm.append(o)
    ov = {}
    for f in (50, 70):
        for i, a in enumerate(V.ORDERINGS):
            for b in V.ORDERINGS[i + 1:]:
                A = set(np.unique(unit[masks[f'S{f}_{a}']]).tolist()); B = set(np.unique(unit[masks[f'S{f}_{b}']]).tolist())
                key = f'S{f}_{a}|S{f}_{b}'
                ov[key] = {'shared_households': len(A & B), 'jaccard': len(A & B) / len(A | B)}
                e = man['pairwise_overlap'][key]
                if ov[key]['shared_households'] != e['shared_households'] or abs(ov[key]['jaccard'] - e['jaccard']) > 1e-15:
                    nm.append(key)
    share = {f: float(np.mean([ov[k]['shared_households'] for k in ov if k.startswith(f'S{f}_')]) / V.n_households(G, f)) for f in (50, 70)}
    out.append(chk(f'V0.SUBSETS.NESTING_OVERLAP__{tag}', 'PROCEDURAL_COMPLIANCE',
                   'nesting S50(o) inside S70(o) inside S100 for every ordering (people added 50->70 and 70->100) and the 12 pairwise '
                   'household overlaps / Jaccard indices across orderings vs SUBSETS.json',
                   V.inputs_sha([man_p]), {'nesting': man['nesting'], 'pairwise_overlap': man['pairwise_overlap']},
                   {'nesting': nobs, 'pairwise_overlap': ov, 'mean_shared_fraction_of_subset': share, 'mismatches': nm},
                   'PASS' if not nm and all(r['S50_in_S70'] for r in nobs.values()) else 'FAIL'))
    # nested-selection folds and A1 inner folds
    f10 = V.folds(unit, 10, 'TSZ1_ncv10')
    ncv = {'people_per_fold': np.bincount(f10, minlength=10).tolist(),
           'households_per_fold': [int(len(np.unique(unit[f10 == k]))) for k in range(10)],
           'training_people_per_outer_fold': [int((f10 != k).sum()) for k in range(10)]}
    from ohn import common as OC
    same_ohn = bool(np.array_equal(f10, OC.household_folds(unit, 10, 'TSZ1_ncv10')))
    exp_ncv = {k: man['nested_selection_folds'][k] for k in ncv}
    inner_obs, inner_mism, inner_same = {}, [], True
    for v in (0, 1):
        fi = V.folds(unit, 5, f'OHN_A_cv_S{v}')
        inner_same = inner_same and bool(np.array_equal(fi, OC.household_folds(unit, 5, f'OHN_A_cv_S{v}')))
        inner_obs[f'V{v}'] = {sid: np.bincount(fi[m], minlength=5).tolist() for sid, m in masks.items()}
        for sid in masks:
            if inner_obs[f'V{v}'][sid] != man['a1_inner_fold_people_per_subset'][f'V{v}'][sid]:
                inner_mism.append(f'V{v}.{sid}')
    whole10 = all(len(set(f10[unit == u].tolist())) == 1 for u in np.unique(unit)[:2000])
    ok = ncv == exp_ncv and not inner_mism and same_ohn and inner_same and man['nested_selection_folds']['tag'] == 'TSZ1_ncv10' and \
        man['nested_selection_folds']['k'] == 10
    out.append(chk(f'V0.FOLDS.NCV10_AND_INNER__{tag}', 'PROCEDURAL_COMPLIANCE',
                   'nested strength-selection outer folds (households ranked by SHA-256("TSZ1_ncv10|<household>"), dealt round-robin into '
                   '10) and the A1_PO_16 inner folds OHN_A_cv_S<v> (5 folds over all MODEL_FIT units) restricted to each subset: people per '
                   'fold, households per fold, outer training sizes, inner-fold people per subset vs SUBSETS.json',
                   V.inputs_sha([man_p]), {'nested_selection_folds': exp_ncv, 'a1_inner_fold_people_per_subset': man['a1_inner_fold_people_per_subset']},
                   {'nested_selection_folds': ncv, 'a1_inner_fold_people_per_subset': inner_obs, 'mismatches': inner_mism,
                    'equals_ohn_common_household_folds': {'ncv10': same_ohn, 'inner': inner_same}, 'whole_households_sample': whole10},
                   'PASS' if ok else 'FAIL'))
    return out, masks, unit, row_id


def reader_checks(tag):
    out = []
    p = RES / 'READER_SPLITS.json'
    man = json.loads(p.read_text())
    S = C.features('MODEL_SELECT')
    unit = np.asarray(S['unit'])
    obs, mism = {}, []
    roles = {}
    for rep in ('R1', 'R2'):
        r = V.reader_roles(unit, f'TSZ1_devgrid_{rep}')
        roles[rep] = r
        obs[rep] = {}
        for role, m in r.items():
            hh = np.unique(unit[m])
            obs[rep][role] = {'people': int(m.sum()), 'households': int(len(hh)), 'household_set_sha256': V.hh_set_sha(hh)}
            if obs[rep][role] != man['repetitions'][rep][role]:
                mism.append(f'{rep}.{role}')
        sets = {role: set(np.unique(unit[m]).tolist()) for role, m in r.items()}
        obs[rep]['disjoint'] = not (sets['FIT'] & sets['VALIDATION'] or sets['FIT'] & sets['SCORE'] or sets['VALIDATION'] & sets['SCORE'])
        obs[rep]['covers_all'] = bool((r['FIT'] | r['VALIDATION'] | r['SCORE']).all())
        if man['repetitions'][rep]['tag'] != f'TSZ1_devgrid_{rep}' or obs[rep]['disjoint'] is not True:
            mism.append(f'{rep}.tag_or_disjoint')
    hh = lambda m: set(np.unique(unit[m]).tolist())
    ov = {f'R1_vs_R2_{role}': len(hh(roles['R1'][role]) & hh(roles['R2'][role])) for role in ('FIT', 'VALIDATION', 'SCORE')}
    if ov != man['household_overlap_between_repetitions']:
        mism.append('overlap_between_repetitions')
    earlier = {}
    for rep in ('R1', 'R2'):
        for old in ('R1', 'R2'):
            th = V.reader_roles(unit, f'OHN1_devgrid_{old}')
            for role in ('FIT', 'VALIDATION', 'SCORE'):
                earlier[f'{rep}_{role}_vs_earlier_{old}_{role}'] = len(hh(roles[rep][role]) & hh(th[role]))
    if earlier != man['household_overlap_with_earlier_grid_splits']:
        mism.append('overlap_with_earlier')
    base = 2026101205
    seeds = {f'V{v}k{k}': base + 500 + 10 * v + k for v in (0, 1) for k in (0, 1)}
    forbidden = {base + 300 + 10 * v + k for v in (0, 1) for k in range(10)}
    seeds_ok = seeds == man['reader_seeds'] and len(set(seeds.values())) == 4 and not (set(seeds.values()) & forbidden)
    rules = json.loads((RES / 'roles' / 'statistical_reviewer' / 'RULES.json').read_text())
    cells_ok = man['cells'] == [f'S{v}R{r}k{k}' for v in (0, 1) for r in (1, 2) for k in (0, 1)] == rules['readers']['cells']
    boot_ok = man['bootstrap']['B'] == 400 == rules['readers']['bootstrap_B']
    out.append(chk(f'V0.READER_SPLITS__{tag}', 'PROCEDURAL_COMPLIANCE',
                   'development reader splits reconstructed from the definition (MODEL_SELECT households ranked by SHA-256("TSZ1_devgrid_<R>|'
                   '<household>"), dealt round-robin into 10 parts; 0-4 FIT, 5-6 VALIDATION, 7-9 SCORE): people, households, household-set '
                   'SHA-256, disjointness, R1/R2 overlap and overlap with the earlier OHN1 grid splits vs READER_SPLITS.json; reader seeds '
                   '(MODEL_SEED_BASE 2026101205 + 500 + 10 v + k) distinct and outside the forbidden OHN1 / CTS1 range (+300 + 10 v + k, k 0..9); '
                   'cells and B vs RULES.json',
                   V.inputs_sha([p, RES / 'roles' / 'statistical_reviewer' / 'RULES.json']),
                   {'repetitions': man['repetitions'], 'overlap': man['household_overlap_between_repetitions'],
                    'earlier': man['household_overlap_with_earlier_grid_splits'], 'reader_seeds': man['reader_seeds']},
                   {'repetitions': obs, 'overlap': ov, 'earlier': earlier, 'reader_seeds': seeds, 'seeds_distinct_and_new': seeds_ok,
                    'cells_match_rules': cells_ok, 'bootstrap_B_match': boot_ok, 'mismatches': mism},
                   'PASS' if (not mism and seeds_ok and cells_ok and boot_ok) else 'FAIL'))
    return out


def pin_checks(tag, newline_translate=True, artifacts=True):
    out = []
    objs = {}
    for name, sha_ in (('cts_publication', PUB), ('cts_research_tip', TIP), ('ohn_publication', OHN_PUB), ('ihs_publication', IHS_PUB)):
        rc, o, _ = git('cat-file', '-t', sha_)
        objs[name] = o.strip() if rc == 0 else None
    rc, _, _ = git('merge-base', '--is-ancestor', TIP, 'HEAD')
    tip_anc = rc == 0
    rc, o, _ = git('rev-parse', 'HEAD')
    head = o.strip()
    rc, o, _ = git('ls-tree', '-r', PUB, '--', PUB_PREFIX + '/')
    rows = [l.split('\t') for l in o.splitlines() if l.strip()]
    blobs = {path: meta.split()[2] for meta, path in rows}
    from mechanism_design.publish_build import sanitise          # the source publication's own sanitiser (definition)
    same, after_san, differ, missing = [], [], [], []
    for path, blob in sorted(blobs.items()):
        rel = path[len(PUB_PREFIX) + 1:]
        if rel.startswith('code/'):
            local = ROOT / rel[len('code/'):]
        elif rel.startswith('results/'):
            local = CTS_RES / rel[len('results/'):]
        else:
            local = None
        if local is None or not local.is_file():
            missing.append(rel); continue
        data = local.read_bytes()
        if V.git_blob_sha(data) == blob:
            same.append(rel); continue
        try:
            # V2 correction: the source build (mechanism_design.publish_build.build) reads text files with Path.read_text()
            # (universal newlines: CRLF -> LF) before sanitise(); V1 decoded the raw bytes and kept CRLF (verifier bug)
            txt = local.read_text() if newline_translate else data.decode()
        except UnicodeDecodeError:
            differ.append(rel); continue
        (after_san if V.git_blob_sha(sanitise(txt).encode()) == blob else differ).append(rel)
    adm_p = RES / 'SOURCE_ADMISSION.json'
    adm = json.loads(adm_p.read_text())
    n_code = sum(1 for p in blobs if p[len(PUB_PREFIX) + 1:].startswith('code/'))
    n_res = len(blobs) - n_code
    code_same = sum(1 for r in same if r.startswith('code/')); res_same = len(same) - code_same
    code_san = sum(1 for r in after_san if r.startswith('code/')); res_san = len(after_san) - code_san
    expected = {'code': [adm['cts_published_code']['files'], adm['cts_published_code']['identical'], adm['cts_published_code']['identical_after_source_sanitiser']],
                'results': [adm['cts_published_results']['files'], adm['cts_published_results']['identical'], adm['cts_published_results']['identical_after_source_sanitiser']]}
    observed = {'code': [n_code, code_same, code_san], 'results': [n_res, res_same, res_san]}
    ok = (objs['cts_publication'] == 'commit' and objs['cts_research_tip'] == 'commit' and objs['ohn_publication'] == 'commit'
          and objs['ihs_publication'] == 'commit' and tip_anc and not differ and not missing and observed == expected)
    out.append(chk(f'V0.PINS.PUBLICATION_TREE__{tag}', 'PROCEDURAL_COMPLIANCE',
                   'source pins: the four pinned commits exist; the research tip 507024649 is an ancestor of HEAD (branch base); every file of '
                   'the pinned correction-transfer publication tree e32a185b4 (publications/pcrl-correction-transfer-stability-v1/) equals '
                   'the local file by the verifier\'s own git blob id (SHA-1 of "blob <len>\\0" + bytes), or after the source\'s own '
                   'sanitiser; counts vs SOURCE_ADMISSION.json',
                   V.inputs_sha([adm_p]), expected,
                   {**observed, 'objects': objs, 'tip_is_ancestor_of_HEAD': tip_anc, 'HEAD': head, 'differ': differ, 'missing_locally': missing,
                    'identical_after_sanitiser_files': after_san},
                   'PASS' if ok else 'FAIL',
                   method='text files are compared as the source build writes them: Path.read_text() (universal newlines) then '
                          'sanitise()' if newline_translate else 'raw bytes decoded (V1; no newline translation)'))
    if not artifacts:
        return out
    # pinned release / teacher / bank hashes (spot checks)
    spot = {}
    exp_spot = {}
    ru = adm['releases_used']
    for v in (0, 1):
        spot[f'A/A1_PO_16/S{v}'] = V.file_sha(OHN_RUNS / 'A' / 'A1_PO_16' / f'S{v}' / 'OCC.npz'); exp_spot[f'A/A1_PO_16/S{v}'] = ru[f'A/A1_PO_16/S{v}']['occ_sha256']
        spot[f'FROZEN_BANK/S{v}'] = V.file_sha(MIRROR / 'admitted' / 'bank' / f'FROZEN_BANK_S{v}.npz'); exp_spot[f'FROZEN_BANK/S{v}'] = ru[f'FROZEN_BANK/S{v}']['sha256']
        spot[f'H/H3_STRONG_LJDIR_PB/S{v}'] = V.file_sha(MIRROR / 'runs' / 'H' / 'H3_STRONG_LJDIR_PB' / f'S{v}' / 'RELEASE.npz')
        exp_spot[f'H/H3_STRONG_LJDIR_PB/S{v}'] = ru[f'H/H3_STRONG_LJDIR_PB/S{v}']['release_sha256']
        for t in ('income', 'occupation'):
            spot[f'T_TREE/{t}/S{v}'] = V.file_sha(MIRROR / 'admitted' / 'stage_a' / f'S{v}' / f'TREE_TEACHER_{t}.json')
            exp_spot[f'T_TREE/{t}/S{v}'] = ru[f'T_TREE/{t}/S{v}']['teacher_sha256']
    tstar = {}
    for v in (0, 1):
        rec = json.loads((OHN_RUNS / 'A' / 'A1_PO_16' / f'S{v}' / 'OCC.json').read_text())
        tstar[v] = {'OCC.json': rec.get('T_star'), 'SOURCE_ADMISSION': ru[f'A/A1_PO_16/S{v}']['T_star']}
    # the H3 mirror: 20 files of SOURCE_ADMISSION-listed private copies re-hashed
    ok2 = spot == exp_spot and all(x['OCC.json'] == x['SOURCE_ADMISSION'] for x in tstar.values())
    out.append(chk(f'V0.PINS.SOURCE_ARTIFACT_HASHES__{tag}', 'PROCEDURAL_COMPLIANCE',
                   'spot checks of the pinned source artifacts the campaign uses: A1_PO_16 S0/S1 OCC.npz, frozen bank S0/S1, H3 income '
                   'RELEASE.npz S0/S1, T_TREE income / occupation teachers S0/S1 (file SHA-256) and the source T* vs SOURCE_ADMISSION.json',
                   V.inputs_sha([adm_p]), exp_spot, {'sha256': spot, 'T_star': tstar}, 'PASS' if ok2 else 'FAIL'))
    return out


def constants_checks(tag):
    out = []
    rp = RES / 'roles' / 'statistical_reviewer' / 'RULES.json'
    r = json.loads(rp.read_text())
    op = ROOT / V.OHN1_RULES
    ohn = json.loads(op.read_text())
    fl = ohn['reserve']['floor']['se']
    mism = []
    if V.file_sha(op) != V.OHN1_RULES_SHA:
        mism.append('OHN1 RULES.json SHA-256 differs from the pinned value')
    if r['reserve']['rule_sha256'] != V.OHN1_RULES_SHA or r['source_files'].get(V.OHN1_RULES) != V.OHN1_RULES_SHA:
        mism.append('TSZ RULES.json does not pin the OHN1 rule SHA-256')
    for v in ('0', '1'):
        for c in V.OCC_SIDE:
            if r['reserve']['floor_se_occupation_side'][v][c] != fl[v][c]:
                mism.append(f'floor {v} {c}')
    slack = r['selection']['slack_of']
    allow_mine = {v: {j: min(fl[v][c] for c in slack[j]) * float(r['selection']['allowance_lambda']) for j in V.STATS5} for v in ('0', '1')}
    if allow_mine != {v: {j: r['selection']['allowance'][v][j] for j in V.STATS5} for v in ('0', '1')}:
        mism.append('allowance table')
    exp_slack = {'men_brier': ['guard|sex0'], 'women_brier': ['guard|sex1'],
                 'brier': ['occupation|U0|brier_excess', 'occupation|Ucal|brier_excess'],
                 'nll': ['occupation|U0|nll_excess', 'occupation|Ucal|nll_excess'],
                 'acc_loss': ['occupation|U0|accuracy_loss', 'occupation|Ucal|accuracy_loss']}
    if slack != exp_slack:
        mism.append('slack_of mapping')
    fixed = {'k': r['reserve']['multiplier_k'] == 2.0, 'floor_households': r['reserve']['floor_households'] == 5306,
             'caps': r['original_limits']['caps'] == {'accuracy_loss': 0.01, 'nll_excess': 0.01, 'brier_excess': 0.005},
             'guard_cap': r['original_limits']['within_sex_guard_cap'] == 0.005,
             'strengths': tuple(r['correction']['strengths']) == V.STRENGTHS, 'selection_z': r['selection']['z'] == 2.0,
             'size_z': r['size_design']['z'] == 2.0, 'reproduction_tol': r['reproduction']['max_abs_tolerance'] == 1e-12,
             'occ_side_list': tuple(r['reserve']['occupation_side_conditions']) == V.OCC_SIDE,
             'trigger_fields': r['trigger']['deployment_record_fields'] == ['bitwise', 'refusals_all_pass'],
             'primary_endpoint': r['size_design']['primary_endpoint'] == {'contrast': '50->100', 'metric': 'men_brier', 'versions': 'both'},
             'ncv': r['selection']['outer_folds'] == 10 and r['selection']['fold_tag'] == 'TSZ1_ncv10',
             'ohn1_k_and_hh': ohn['reserve']['multiplier_k'] == 2.0 and ohn['reserve']['floor']['households'] == 5306}
    mism += [k for k, ok in fixed.items() if not ok]
    out.append(chk(f'V0.RULES.CONSTANTS__{tag}', 'PROCEDURAL_COMPLIANCE',
                   'registered constants: OHN1 RULES.json SHA-256 (e18f1c68...), the 16 occupation-side floor SEs copied unchanged, the '
                   'allowance table = lambda (1.0) x min mapped floor per statistic and version (recomputed), the slack mapping, k = 2, '
                   '5306 floor households, original caps, strength set, z values, reproduction tolerance, trigger record fields, primary '
                   'endpoint, nested CV folds',
                   V.inputs_sha([rp, op]), {'ohn1_rules_sha256': V.OHN1_RULES_SHA},
                   {'allowance_recomputed': allow_mine, 'fixed_constants': fixed, 'mismatches': mism}, 'PASS' if not mism else 'FAIL'))
    return out


def registration_record(tag):
    files = ('PROTOCOL.md', 'SUBSETS.json', 'EXPOSURE_MAP.md', 'ACCESS_POLICY.json', 'SOURCE_ADMISSION.json', 'READER_SPLITS.json',
             'DEVGRID_CODE_LOCK.json', 'roles/statistical_reviewer/RULES.md', 'roles/statistical_reviewer/RULES.json')
    rc, o, _ = git('ls-files', '--error-unmatch', *[str((RES / f).relative_to(ROOT)) for f in files])
    committed = rc == 0
    return chk(f'V0.REGISTRATION.FILES_AT_V0__{tag}', 'PROCEDURAL_COMPLIANCE',
               'snapshot (no verdict) of the registration files at verifier stage V0: SHA-256 of each and whether they are tracked by git; '
               'the registration receipt is checked later against these hashes',
               V.inputs_sha([RES / f for f in files]), 'informational snapshot', {'tracked_by_git': committed,
                                                                                 'receipt_exists': C.REG_RECEIPT.is_file()},
               'PASS', note='informational: the coordinator commits the registration after V0; any later change of these files is checked '
                            'against the receipt in V4')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--only', default=None, choices=('pins',))
    a = ap.parse_args()
    if a.only == 'pins':
        checks = pin_checks(a.tag, newline_translate=True, artifacts=False)
        for c in checks:
            c['code_sha256'] = V.code_hashes(__file__, V.__file__)
        V.write_record(f'V0_PINS__{a.tag}.json', {'checks': checks, 'utc': V.utc()})
        print(json.dumps(V.append_checks(checks)))
        print(json.dumps({c['id']: [c['status'], c['observed']['code'], c['observed']['results'], c['observed']['differ']] for c in checks}))
        return
    good, detail = V.selftest()
    checks = [chk(f'V0.VLIB.SELFTEST__{a.tag}', 'NUMERICAL_REPLAY', 'verifier library invented-data self-test (folds, subsets, SEs, '
                  'reserve edge conventions, ordering summaries, across-version labels, strength selection, stability, encoder, git blob)',
                  V.inputs_sha([HERE / 'vlib.py']), 'all true', detail, 'PASS' if good else 'FAIL')]
    c1, masks, unit, row_id = subsets_checks(a.tag)
    checks += c1
    checks += reader_checks(a.tag)
    checks += pin_checks(a.tag)
    checks += constants_checks(a.tag)
    checks.append(registration_record(a.tag))
    for c in checks:
        c['code_sha256'] = V.code_hashes(__file__, V.__file__)
    V.write_record(f'V0_STRUCT__{a.tag}.json', {'checks': checks, 'utc': V.utc()})
    print(json.dumps(V.append_checks(checks)))
    print(json.dumps({c['id']: c['status'] for c in checks}, indent=1))


if __name__ == '__main__':
    main()
