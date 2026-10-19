"""Verifier stage V4 (label-free): the training records of every size fit and every control fit, recomputed from the verifier's own
subset reconstruction: people, households, row-set hash, FIT-only dictionary (collapsed levels per block), MODEL_SELECT shares with a
collapsed or an unseen level, inner-fold sizes, aliases; controls: rows, dictionary, Stage B update count and epoch coverage.

PCRL controlled training-size campaign (TSZ1; exploratory development on previously used ACS 2019 Florida PUMS person records).
No label is read (features, row ids and household ids only).

    python3 .../v4_records.py --tag V1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from tsz import common as C  # noqa: E402   (label-free features only)

PRIV = ROOT / '.private' / 'pcrl_controlled_training_size_v1'
SIZE = PRIV / 'runs' / 'SIZE'
CONTROL = PRIV / 'runs' / 'CONTROL'


def diagnostics(Xs, Xms, names, collapsed):
    out, any_c, any_u = {}, np.zeros(len(Xms), bool), np.zeros(len(Xms), bool)
    for b in V.BLOCKS:
        cols = [i for i, nm in enumerate(names) if nm.startswith(b + '=')]
        cnt = np.asarray(Xs)[:, cols].sum(0)
        act = np.asarray(Xms)[:, cols].argmax(1)
        coll = np.array([names[c] in set(collapsed[b]) for c in cols])
        unseen = cnt == 0
        any_c |= coll[act]; any_u |= unseen[act]
        out[b] = {'n_collapsed': int(coll.sum()), 'levels_absent_from_subset': int(unseen.sum()),
                  'MODEL_SELECT_people_collapsed': int(coll[act].sum()), 'MODEL_SELECT_people_unknown': int(unseen[act].sum())}
    out['any_block'] = {'MODEL_SELECT_share_collapsed': float(any_c.mean()), 'MODEL_SELECT_share_unknown': float(any_u.mean())}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    F = C.features('MODEL_FIT'); M = C.features('MODEL_SELECT')
    names = [str(x) for x in F['feature_names']]
    unit, row_id, X = np.asarray(F['unit']), np.asarray(F['row_id']), np.asarray(F['X'])
    masks = V.subset_masks(unit)
    rows_out, problems, inputs = {}, [], []
    for d in sorted(p for p in SIZE.iterdir() if p.is_dir() and (p / 'FIT.json').is_file()):
        rec = json.loads((d / 'FIT.json').read_text()); inputs.append(d / 'FIT.json')
        parts = d.name.split('__'); sid, v = parts[1], int(parts[2][1:]); adm = len(parts) == 4
        m = masks[sid]
        coll = {b: [] for b in V.BLOCKS} if adm else V.fit_only_dictionary(X[m], names, 10)
        dic = json.loads((d / 'DICT.json').read_text())
        diag = diagnostics(X[m], M['X'], names, coll)
        inner = V.folds(unit, 5, f'OHN_A_cv_S{v}')[m]
        fold_sizes = {str(j): {'households': int(len(np.unique(unit[m][inner == j]))), 'people': int((inner == j).sum())} for j in range(5)}
        pd_ = rec['dictionary']['diagnostics']
        obs = {'n_people': int(m.sum()), 'n_households': int(len(np.unique(unit[m]))), 'rows_sha256': V.arr_sha(row_id[m]),
               'dictionary_equal': all(sorted(coll[b]) == sorted(dic['collapsed'][b]) for b in V.BLOCKS),
               'n_collapsed': {b: len(coll[b]) for b in V.BLOCKS}, 'diagnostics_any_block': diag['any_block'],
               'inner_fold_sizes_equal': fold_sizes == rec['inner_cv']['fold_sizes'],
               'aliases': rec['aliases'], 'T_star': rec['T_star'], 'flags': rec['flags']}
        exp_ok = (obs['n_people'] == rec['n_people'] and obs['n_households'] == rec['n_households'] and obs['rows_sha256'] == rec['rows_sha256']
                  and obs['dictionary_equal'] and obs['inner_fold_sizes_equal']
                  and all(diag[b][k] == pd_[b][k] for b in V.BLOCKS for k in ('n_collapsed', 'levels_absent_from_subset',
                                                                               'MODEL_SELECT_people_collapsed', 'MODEL_SELECT_people_unknown'))
                  and abs(diag['any_block']['MODEL_SELECT_share_collapsed'] - pd_['any_block']['MODEL_SELECT_share_collapsed']) < 1e-15
                  and abs(diag['any_block']['MODEL_SELECT_share_unknown'] - pd_['any_block']['MODEL_SELECT_share_unknown']) < 1e-15
                  and rec['aliases'] == (list(V.ORDERINGS) if sid == 'S100' else [sid.split('_')[1]])
                  and rec['dictionary_kind'] == ('admitted' if adm else 'fit_only') and rec['subset'] == sid and rec['version'] == v)
        cfg = rec.get('config') or {}
        same_effort = (rec.get('config_override') is None and cfg.get('lr') == 0.025 and cfg.get('leaves') == 16 and cfg.get('msl') == 100
                       and cfg.get('l2') == 3.0 and cfg.get('max_iter') == 1000 and cfg.get('rep') == 'cmp_po' and cfg.get('calibrator') is None
                       and rec.get('T_max') == 1000 and rec['inner_cv']['tag'] == f'OHN_A_cv_S{v}' and rec['inner_cv']['k'] == 5
                       and 'T_STAR_AT_T_MAX' not in rec.get('flags', []))
        obs['fixed_recipe_and_effort'] = same_effort
        exp_ok = exp_ok and same_effort
        obs['agrees_with_FIT_json'] = bool(exp_ok)
        if not exp_ok:
            problems.append(d.name)
        rows_out[d.name] = obs
    ctl = {}
    for d in sorted(p for p in CONTROL.iterdir() if p.is_dir()):
        tj, sj = d / 'TEACHERS' / 'TEACHERS.json', d / 'STAGE_B' / 'STAGE_B.json'
        if not (tj.is_file() and sj.is_file()):
            ctl[d.name] = {'complete': False}
            continue
        t, s = json.loads(tj.read_text()), json.loads(sj.read_text()); inputs += [tj, sj]
        sid, v = d.name.split('__')[0], int(d.name.split('__')[1][1:])
        m = masks[sid]
        coll = V.fit_only_dictionary(X[m], names, 10)
        dic = json.loads((d / 'TEACHERS' / 'DICT.json').read_text())
        o = {'complete': True, 'rows_equal_teachers': t['rows_sha256'] == V.arr_sha(row_id[m]), 'rows_equal_stage_b': s['rows_sha256'] == V.arr_sha(row_id[m]),
             'n_people': [t['n_people'], int(m.sum())], 'n_households': [t['n_households'], int(len(np.unique(unit[m])))],
             'dictionary_equal': all(sorted(coll[b]) == sorted(dic['collapsed'][b]) for b in V.BLOCKS),
             'teachers_sha_consistent': s['teachers_sha256'] == {k: t['files_sha256'][f'TEACHER_{k}.json'] for k in ('income', 'occupation')},
             'epoch_coverage': s.get('epoch_coverage'), 'recipe_overrides': s.get('recipe_overrides'), 'n': s.get('n'),
             'tree_counts': {k: (t['teachers'][k].get('tree_count') if isinstance(t.get('teachers', {}).get(k), dict) else None) for k in ('income', 'occupation')}}
        import hashlib
        rcp = s.get('recipe') or {}
        run = (s.get('runs') or {}).get('TASK_REFIT') or {}
        teach_ok = True
        for tk in ('income', 'occupation'):
            tp = t['teachers'][tk]['params']
            seed_exp = int(hashlib.sha256(f'ACSR1catboost|{v}|{tk}|4'.encode()).hexdigest()[:8], 16)
            teach_ok = teach_ok and (tp['depth'] == 4 and tp['learning_rate'] == 0.05 and tp['l2_leaf_reg'] == 3.0 and tp['thread_count'] == 1
                                     and tp['random_seed'] == seed_exp and t['teachers'][tk]['max_trees'] == 800
                                     and tp['loss_function'] == ('Logloss' if tk == 'income' else 'MultiClass'))
        stage_ok = (rcp.get('batch_size') == 256 and rcp.get('steps_per_epoch') == 40 and rcp.get('epochs_per_active_recipient') == 20
                    and rcp.get('learning_rate') == 0.01 and rcp.get('weight_decay') == 0.01 and not s.get('recipe_overrides')
                    and s.get('arms') == ['TASK_REFIT'] and run.get('lambda') == [0.0, 0.0] and run.get('updates_per_recipient') == [800, 800])
        o['fixed_teacher_recipe_and_seed'] = teach_ok; o['fixed_stage_b_effort'] = stage_ok
        o['ok'] = bool(o['rows_equal_teachers'] and o['rows_equal_stage_b'] and o['n_people'][0] == o['n_people'][1]
                       and o['n_households'][0] == o['n_households'][1] and o['dictionary_equal'] and o['teachers_sha_consistent']
                       and teach_ok and stage_ok)
        if not o['ok']:
            problems.append(d.name)
        ctl[d.name] = o
    chk = {'id': f'V4.RECORDS.TRAINING_RECORDS__{a.tag}', 'kind': 'PROCEDURAL_COMPLIANCE',
           'what': 'label-free training records of every saved A1_PO_16 size fit (people, households, row-set hash, FIT-only dictionary, '
                   'MODEL_SELECT collapsed / unseen-level counts and shares, inner-fold sizes, aliases, dictionary kind) and every saved '
                   'control fit (rows of both stages, dictionary, teacher hashes consistent between stages), recomputed from the verifier\'s '
                   'own subset reconstruction',
           'inputs': V.inputs_sha(inputs), 'expected': 'every record agrees', 'observed': {'size_fits': rows_out, 'controls': ctl, 'problems': problems},
           'status': 'PASS' if not problems else 'FAIL', 'code_sha256': V.code_hashes(__file__, V.__file__)}
    V.write_record(f'V4_RECORDS__{a.tag}.json', {'checks': [chk], 'utc': V.utc()})
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], 'n_size': len(rows_out), 'n_ctl': len(ctl), 'problems': problems}))


if __name__ == '__main__':
    main()
