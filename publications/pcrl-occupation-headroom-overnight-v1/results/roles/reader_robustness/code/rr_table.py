"""Per-candidate stability table with reader-robustness annotations (label-free: reads this role's JSON outputs only).

For every candidate (task-only witness, protection output, income-branch output, SUCC) and comparator: the frozen stability rule on
the registered grid cells (8 cells when both release seeds exist, else the seed-0 cells descriptively), the recipient maxima, and
the robustness annotations from the diagnostic probe (probe winner, union winner, SCORE-max oracle, fixed recipes: candidate and
comparator read by the same recipe) and from the fixed-recipe learning curves. Development evidence on MODEL_SELECT (adaptively
used); diagnostic; never selection; reader-relative.

    python3 rr_table.py --final-tag RR_FINAL --probe-tag RR_PROBE_A --probe-tag RR_PROBE_P1 ... --lcfix-tag RR_LCFIX
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from ohn import common as C          # noqa: E402

OUT = C.RESULTS / 'roles' / 'reader_robustness'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--final-tag', required=True)
    ap.add_argument('--probe-tag', action='append', default=[])
    ap.add_argument('--lcfix-tag', default=None)
    a = ap.parse_args(argv)
    st = json.loads((OUT / f'{a.final_tag}__STABILITY.json').read_text())['pairs']
    probe = {}
    for t in a.probe_tag:
        p = json.loads((OUT / f'{t}__PROBE_SUMMARY.json').read_text())
        for pk, schemes in p['pairs'].items():
            rec = probe.setdefault(pk, {})
            for key, v in schemes.items():
                if key.startswith('pair|') and (key not in rec or v['n_cells'] > rec[key]['n_cells']):
                    rec[key] = v
        for pk, f in p['fixed_recipe_pair_benefit_sign'].items():
            prev = probe.setdefault(pk, {}).get('__fixed')
            n_cells = max((v['n_cells'] for k, v in p['pairs'][pk].items() if k.startswith('pair|SLATE:')), default=0)
            if prev is None or n_cells > prev['n_cells']:
                probe[pk]['__fixed'] = {**f, 'n_cells': n_cells}
    lcf = json.loads((OUT / f'{a.lcfix_tag}__LC_FIXED_RECIPE.json').read_text())['pairs'] if a.lcfix_tag else {}
    lcw = json.loads((OUT / f'{a.final_tag}__LEARNING_CURVES.json').read_text())['pair_benefit_by_fraction']
    rows = []
    for pk, r in sorted(st.items()):
        cand, comp = pk.split('|')
        row = {'candidate': cand, 'comparator': comp, 'label': r['label']}
        if r['label'] != 'INCOMPLETE':
            row.update({'n_cells': 8, 'mean': r['mean'], 'mean_s0': r['per_seed_mean'].get('0', r['per_seed_mean'].get(0)),
                        'mean_s1': r['per_seed_mean'].get('1', r['per_seed_mean'].get(1)), 'min': r['min'], 'max': r['max'],
                        'sd_total': r['sd_total'], 'sd_reader_split': r['sd_reader_split'], 'rms_cell_se': r['rms_cell_se'],
                        'flip_p_s0': r['predicted_flip_probability_per_seed'].get('0', r['predicted_flip_probability_per_seed'].get(0)),
                        'flip_p_s1': r['predicted_flip_probability_per_seed'].get('1', r['predicted_flip_probability_per_seed'].get(1)),
                        'cells_le0': ';'.join(r['cells_at_or_below_0']), 'cells_pos_interval': len(r['cells_positive_conditional_interval']),
                        'cells_neg_interval': len(r['cells_negative_conditional_interval']),
                        'occ_recipient_max': r['recipient']['occupation']['max'], 'inc_recipient_max': r['recipient']['income']['max'],
                        'target_label': r['target_label']})
        else:
            x = r.get('descriptive_present_cells') or {}
            if not x or x.get('mean') is None:
                continue
            s0 = (x.get('per_seed_descriptive') or {}).get('0') or (x.get('per_seed_descriptive') or {}).get(0) or {}
            row.update({'n_cells': len(x['cells']), 'mean': x['mean'], 'mean_s0': s0.get('mean'), 'min': x['min'], 'max': x['max'],
                        'sd_total': s0.get('sd_total_one_seed'), 'flip_p_s0': s0.get('flip_probability'),
                        'cells_le0': ';'.join(x['cells_at_or_below_0']), 'cells_pos_interval': len(x['cells_positive_conditional_interval']),
                        'cells_neg_interval': len(x['cells_negative_conditional_interval']),
                        'occ_recipient_max': x['occupation_recipient_max'], 'inc_recipient_max': x['income_recipient_max']})
        pr = probe.get(pk, {})
        for key, col in (('pair|PROBE_WINNER', 'probe_winner_mean'), ('pair|UNION_WINNER', 'union_winner_mean'),
                         ('pair|ORACLE_SLATE', 'oracle_slate_mean'), ('pair|PROBE:MLP_BIG', 'mlp_big_mean'),
                         ('pair|PROBE:GB_LONG', 'gb_long_mean'), ('pair|PROBE:KNN_LOGP', 'knn_logp_mean')):
            if key in pr:
                row[col] = pr[key]['mean']; row[col.replace('_mean', '_cells')] = pr[key]['n_cells']
                if pr[key].get('rule_label'):
                    row[col.replace('_mean', '_label')] = pr[key]['rule_label']
        if '__fixed' in pr:
            f = pr['__fixed']
            row.update({'fixed_recipes_positive': f['n_positive'], 'fixed_recipes_n': f['n_recipes_informative'],
                        'fixed_recipes_median': f['median'], 'fixed_recipes_min': f['min'], 'fixed_recipes_max': f['max'],
                        'fixed_recipes_cells': f['n_cells']})
            for s in ('0', '1'):
                vals = [v['per_seed_mean'].get(s, v['per_seed_mean'].get(int(s))) for k, v in pr.items()
                        if (k.startswith('pair|SLATE:') and k != 'pair|SLATE:LOOKUP') or k.startswith('pair|PROBE:')]
                vals = [x for x in vals if x is not None]
                if vals:
                    row[f'fixed_recipes_median_s{s}'] = float(np.median(vals))
                    row[f'fixed_recipes_positive_s{s}'] = int(sum(x > 0 for x in vals))
        if pk in lcw:
            row.update({'lc_winner_flag': lcw[pk]['flag'], 'lc_winner_q_to_full': lcw[pk]['full_minus_quarter'],
                        'lc_cells_per_fraction': lcw[pk]['n_cells']['1.0']})
        if pk in lcf:
            row.update({'lc_fixed_flagged': lcf[pk]['n_recipes_flagged_shrink'], 'lc_fixed_n': lcf[pk]['n_recipes_informative'],
                        'lc_fixed_median_q_to_full': lcf[pk]['median_full_minus_quarter']})
        rows.append(row)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(OUT / f'{a.final_tag}__CANDIDATE_TABLE.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    C.write_json(OUT / f'{a.final_tag}__CANDIDATE_TABLE.json', {'utc': C.now(), 'rows': rows, 'probe_tags': a.probe_tag, 'lcfix_tag': a.lcfix_tag,
                                                               'note': 'frozen-rule labels on the registered grid with reader-robustness annotations; diagnostic; never selection'})
    print(json.dumps({'rows': len(rows)}))


if __name__ == '__main__':
    main()
